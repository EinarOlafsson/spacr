"""AI text keeps spaCR's docs links; reports carry the screen and every problem."""

from __future__ import annotations

import pytest

from spacr import crashreport as cr
from spacr.validate import ERROR, Problem


def test_the_docs_and_repository_links_survive_for_display():
    text = ("See https://einarolafsson.github.io/spacr/mask.html, "
            "https://github.com/EinarOlafsson/spacr/issues and "
            "https://example.com/x or http://10.0.0.1/a.")
    kept = cr.scrub_public_text(text, 2000, keep_docs_urls=True)
    assert "https://einarolafsson.github.io/spacr/mask.html," in kept
    assert "https://github.com/EinarOlafsson/spacr/issues" in kept
    assert "example.com" not in kept and "10.0.0.1" not in kept
    assert kept.count("<URL>") == 2


def test_other_text_still_loses_every_url():
    text = "See https://einarolafsson.github.io/spacr/ now"
    assert cr.scrub_public_text(text) == "See <URL> now"


def test_a_lookalike_address_is_not_kept():
    text = "https://einarolafsson.github.io/spacr-evil/ and https://github.com/EinarOlafsson/spacrx"
    assert cr.scrub_public_text(text, 500, keep_docs_urls=True) == "<URL> and <URL>"


def test_lines_are_kept_and_scrubbed_one_by_one():
    text = ("**What happened**\nThe run stopped at /home/someone/data.\n\n\n"
            "1. Fix it.\n2. See https://einarolafsson.github.io/spacr/")
    assert cr.scrub_public_lines(text, 2000, True) == (
        "**What happened**\nThe run stopped at <PATH>\n\n1. Fix it.\n"
        "2. See https://einarolafsson.github.io/spacr/")


def test_problem_lines_drop_paths():
    lines = cr.public_problem_lines([
        Problem(ERROR, "src", "src does not exist: /data/plate1 (its parent "
                "folder /data does not exist either)", "fix"),
        Problem(ERROR, "cell_channel", "no segmentation channel is set.", ""),
        "Could not read C:\\Users\\a\\b.tif",
    ])
    assert lines == [
        "src does not exist (its parent folder does not exist either)",
        "cell_channel: no segmentation channel is set.",
        "Could not read",
    ]


def test_the_summary_carries_screen_error_text_and_a_long_problem_list():
    message = "The run did not start.\n" + "\n".join(
        f"- problem {i} " + "x" * 60 for i in range(9))
    summary = cr.public_summary(
        traceback_text="spacr.errors.ConfigurationError: no channel",
        module="mask", kind=cr.KIND_USER_ERROR, screen="Mask",
        error_text="✗ The run did not start: /home/someone/x", message=message)
    assert summary["screen"] == "Mask"
    assert summary["error_text"] == "✗ The run did not start: <PATH>"
    assert 300 < len(summary["message"]) <= cr.MAX_USER_ERROR_MESSAGE_CHARS
    assert summary["message"].startswith("The run did not start.\n- problem 0")


def test_a_screen_name_is_cleaned_and_capped():
    summary = cr.public_summary(
        traceback_text="ValueError: x", module="align",
        screen="Align & Stitch/" + "y" * 80)
    assert summary["screen"].startswith("Align & Stitch")
    assert "/" not in summary["screen"]
    assert len(summary["screen"]) <= cr.MAX_SCREEN_CHARS
    fault = cr.public_summary(traceback_text="ValueError: " + "z" * 900,
                              module="mask")
    assert len(fault["message"]) <= cr.MAX_MESSAGE_CHARS
    assert "screen" not in fault


def test_the_refusal_records_its_call_site(caplog):
    from spacr.errors import raise_if_strict

    def refuse_here():
        raise_if_strict("no channel", settings={})

    with caplog.at_level("ERROR", logger="spacr.errors"):
        refuse_here()
    site = "".join(caplog.records[-1].spacr_call_site)
    assert "in refuse_here" in site
    summary = cr.public_summary(
        traceback_text="Traceback (most recent call last):\n" + site
        + "spacr.errors.ConfigurationError: no channel",
        module="mask", kind=cr.KIND_USER_ERROR)
    assert summary["frames"][-1]["function"] == "refuse_here"


def test_the_signature_matches_the_relay():
    refused = {"exception_type": "ConfigurationError", "module": "mask",
               "kind": cr.KIND_USER_ERROR,
               "message": "The run did not start: 2 settings problems.\n"
                          "- src does not exist <PATH> 3\n"
                          "- cell_channel: no channel",
               "frames": [{"file": "core.py",
                           "function": "preprocess_generate_masks", "line": 1},
                          {"file": "errors.py", "function": "raise_if_strict",
                           "line": 2}]}
    assert cr.explanation_signature(refused) == (
        "e3c173dbd2f958c9e2fdf3df84e0d46d6d0a85ac5ed4240f37cb8f6144424b93")
    fault = {"exception_type": "ValueError", "module": "measure",
             "kind": cr.KIND_FAULT, "message": "x",
             "frames": [{"file": "a.py", "function": "f", "line": 1}]}
    assert cr.explanation_signature(fault) == (
        "564195c398e71f9ba2cbcb694d719e71c63ed00c69695be2f393633883a4809f")


TB = ('Traceback (most recent call last):\n'
      '  File "/x/spacr/measure.py", line 9, in measure_crop\n'
      '    run()\n'
      'ValueError: operands could not be broadcast together\n')


def _summary():
    return cr.public_summary(traceback_text=TB, module="measure")


def test_the_cheap_key_equals_the_full_summary_signature():
    assert cr.explanation_key(TB, "measure") == cr.explanation_signature(
        _summary())


def test_a_saved_answer_skips_the_network(monkeypatch):
    from spacr.qt.ai import providers
    from tests.test_700_anonymous_error_reports import FakeRelay

    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    relay = FakeRelay()
    relay.routes["/diagnose"] = [(200, {"advice": "**Cause** shapes differ.",
                                        "model": "big", "tier": "best",
                                        "cached": False})]
    monkeypatch.setattr(cr, "_open_url", relay)
    hosted = providers.hosted_provider()
    first = providers.explain_failure(_summary(), hosted)
    assert first["text"] == "**Cause** shapes differ." and not first["saved"]
    second = providers.explain_failure(_summary(), hosted)
    assert second == {"text": "**Cause** shapes differ.", "tier": "best",
                      "refused": "", "saved": "local", "cause": ""}
    assert len(relay.calls_to("/diagnose")) == 1


def test_a_relay_saved_answer_is_marked(monkeypatch):
    from spacr.qt.ai import providers
    from tests.test_700_anonymous_error_reports import FakeRelay

    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    relay = FakeRelay()
    relay.routes["/diagnose"] = [(200, {"advice": "Reused.", "model": "m",
                                        "tier": "light", "cached": True})]
    monkeypatch.setattr(cr, "_open_url", relay)
    answer = providers.explain_failure(_summary(), providers.hosted_provider())
    assert answer["saved"] == "relay" and answer["tier"] == "light"


def test_saved_answers_expire_and_follow_the_version(monkeypatch):
    import time

    now = time.time()
    cr.save_explanation("sig", "Old.", now=now - 31 * 86400)
    assert cr.saved_explanation("sig", now=now) is None
    cr.save_explanation("sig", "New.", "best", "m", now=now)
    assert cr.saved_explanation("sig", now=now)["text"] == "New."
    assert cr.saved_explanation("sig", now=now + 30 * 86400) is None
    monkeypatch.setattr(cr, "_spacr_version", lambda: "99.1.0")
    assert cr.saved_explanation("sig", now=now) is None


def test_at_most_the_newest_two_hundred_are_kept(monkeypatch):
    monkeypatch.setattr(cr, "MAX_SAVED_EXPLANATIONS", 3)
    for index in range(5):
        cr.save_explanation(f"s{index}", f"t{index}", now=1000.0 + index)
    monkeypatch.setattr(cr.time, "time", lambda: 1010.0)
    assert [cr.saved_explanation(f"s{i}") is not None for i in range(5)] == [
        False, False, True, True, True]


def test_a_cli_provider_reads_the_saved_answer_first(monkeypatch):
    from spacr.qt.ai import providers

    class Cli:
        name = "Claude"

    def never(*_args, **_kwargs):
        raise AssertionError("the CLI must not be asked")

    cr.save_explanation(cr.explanation_signature(_summary()), "Saved.")
    monkeypatch.setattr(providers, "write_report_summary", never)
    assert providers.explain_failure(_summary(), Cli())["saved"] == "local"
    monkeypatch.setattr(providers, "write_report_summary",
                        lambda *a, **k: "Fresh from the CLI.")
    other = cr.public_summary(traceback_text=TB.replace("ValueError",
                                                        "KeyError"),
                              module="measure")
    assert providers.explain_failure(other, Cli())["text"] == (
        "Fresh from the CLI.")
    assert cr.saved_explanation(cr.explanation_signature(other))[
        "text"] == "Fresh from the CLI."


PLANTED = ("annasvensson", "labhost42", "10.20.30.40", "anna@lab.example.org",
           "ghp_0123456789abcdefABCDEF0123456789abcd", "patient_042",
           "/srv/data", "C:\\Users")


@pytest.fixture
def identity(monkeypatch):
    """This computer is annasvensson on labhost42."""
    monkeypatch.setattr(cr, "_home_variants",
                        lambda: ["/home/annasvensson"])
    monkeypatch.setattr(cr, "_identity_scrub", lambda text: str(text).replace(
        "annasvensson", "<USER>").replace("labhost42", "<HOST>"))


def test_planted_personal_data_never_reaches_any_new_field(identity,
                                                            tmp_path):
    import json

    plate = tmp_path / "patient_042"
    plate.mkdir()
    (plate / "patient_042_A01.tif").write_bytes(b"")
    tb = ("Traceback (most recent call last):\n"
          '  File "/home/annasvensson/x.py", line 3, in load\n'
          "OSError: cannot read /srv/data/patient_042 on labhost42\n\n"
          "The above exception was the direct cause of the following "
          "exception:\n\n"
          "Traceback (most recent call last):\n"
          '  File "/home/annasvensson/spacr/spacr/core.py", line 9, in run\n'
          "ValueError: failed for anna@lab.example.org at 10.20.30.40\n")
    settings = {
        "src": str(plate), "dst": "C:\\Users\\anna\\out",
        "experiment": "patient_042", "owner": "annasvensson",
        "host": "labhost42", "notes": "mail anna@lab.example.org",
        "token": "ghp_0123456789abcdefABCDEF0123456789abcd",
        "ip": "10.20.30.40", "cell_channel": 1, "magnification": 20,
        "cellpose_model": "cyto3", "save": True, "channels": [0, 1, 2],
        "plates": ["patient_042", "/srv/data/x"],
    }
    log = ("[INFO] host labhost42 user annasvensson\n"
           "[INFO] reading /srv/data/patient_042/A01.tif from 10.20.30.40\n"
           "[ERROR] notify anna@lab.example.org with "
           "ghp_0123456789abcdefABCDEF0123456789abcd\n")
    summary = cr.public_summary(
        traceback_text=tb, module="mask", screen="Mask", settings=settings,
        log_text=log, error_text=tb, diagnostics=True,
        run={"stage": "preprocess_generate_masks", "elapsed_s": 2.0,
             "dry_run": True})
    text = json.dumps(summary)
    for planted in PLANTED:
        assert planted not in text, planted
    assert summary["settings"]["src"] == "<path: exists=yes>"
    assert summary["settings"]["dst"] == "<path: exists=no>"
    for key in ("experiment", "owner", "host", "notes", "token", "ip"):
        assert summary["settings"][key] == "<text>", key
    assert summary["settings"]["cellpose_model"] == "cyto3"
    assert summary["settings"]["channels"] == [0, 1, 2]
    assert summary["settings"]["plates"] == ["<text>", "<path: exists=no>"]
    assert summary["inputs"] == {"images": 1, "extensions": ["tif"]}
    assert summary["chain"] == [{"type": "OSError",
                                 "message": "cannot read <PATH> on <HOST>"}]
    assert summary["run"] == {"stage": "preprocess_generate_masks",
                              "elapsed_s": 2.0, "dry_run": True}
    assert summary["log_tail"].count("\n") == 2
    env = summary["environment"]
    assert env["cores"] >= 1 and env["install"] in (
        "pip", "conda", "installer", "editable", "unknown")
    assert set(env["versions"]) <= {"torch", "cellpose", "numpy",
                                    "scikit_image", "pyside6", "qt"}


def test_the_example_shows_every_new_field():
    summary = cr.example_public_summary()
    for key in ("screen", "error_text", "chain", "settings", "environment",
                "run", "log_tail"):
        assert key in summary, key
    assert summary["settings"]["src"].startswith("<path: exists=")
    assert summary["settings"]["experiment"] == "<text>"


def test_the_log_tail_keeps_the_last_lines_within_its_cap():
    log = "\n".join(f"line {i} " + "x" * 100 for i in range(100))
    tail = cr.public_log_tail(log)
    assert len(tail) <= cr.MAX_LOG_TAIL_CHARS
    assert tail.endswith("line 99 " + "x" * 100)
    assert "line 59 " not in tail


def test_a_big_report_stays_under_the_body_cap():
    import json

    summary = cr._validated_summary({
        "install_id": cr.install_id(), "exception_type": "E", "message": "m",
        "frames": [{"file": "a.py", "function": "f" * 80, "line": 1,
                    "code": "x" * 160}] * 40,
        "settings": {f"s{i}": "word" for i in range(150)},
        "log_tail": "y" * 5000, "kind": "fault"})
    assert len(json.dumps(summary).encode()) <= cr.MAX_SUMMARY_BYTES


def test_the_cause_line_is_taken_off_and_normalised():
    assert cr.split_cause("Fix it.\n**CAUSE:** No Object Channel Set") == (
        "Fix it.", "no-object-channel-set")
    assert cr.normalise_cause("one two three four five six") == (
        "one-two-three-four-five")
    summary = cr._validated_summary({"install_id": cr.install_id(),
                                     "cause": "CUDA out of memory!"})
    assert summary["cause"] == "cuda-out-of-memory"


def test_a_cli_answer_gives_its_cause_and_shows_without_it(monkeypatch):
    from spacr.qt.ai import providers

    class Cli:
        name = "Claude"

    monkeypatch.setattr(providers, "write_report_summary",
                        lambda *a, **k: "Set the channel.\nCAUSE: no-channel")
    answer = providers.explain_failure(_summary(), Cli())
    assert answer["text"] == "Set the channel."
    assert answer["cause"] == "no-channel"
    assert cr.saved_explanation(cr.explanation_signature(_summary()))[
        "cause"] == "no-channel"
