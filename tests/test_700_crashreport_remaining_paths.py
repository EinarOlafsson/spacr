"""Item 700: the anonymous-report paths the main tests do not reach.

Each test drives one edge of :mod:`spacr.crashreport`'s anonymised error
reports (clock and reset parsing, scrubbing fallbacks, frame and traceback
reading, the queue's damaged-file handling, the relay transport's odd
answers, background sending and the allowance shape) and checks what came
out. No test reaches the network: the relay is a fake, and the real
transport is only ever called with ``urlopen`` replaced.
"""
from __future__ import annotations

import io
import json
import sys
import threading
import types
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spacr import crashreport as cr
from tests.test_700_anonymous_error_reports import FakeRelay, _Response


@pytest.fixture
def relay(monkeypatch):
    """Allow the network for this test and install a fake relay."""
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cr.REPORT_RELAY_ENV, raising=False)
    fake = FakeRelay()
    monkeypatch.setattr(cr, "_open_url", fake)
    return fake


_ORIGINALS = {"_reports_dir": cr._reports_dir, "_open_url": cr._open_url}


def _original(name):
    """Return the module's own ``name``, captured at import.

    ``tests/conftest.py`` replaces both for the length of every test; this
    module is imported at collection, before any test patches them.
    """
    return _ORIGINALS[name]


def _summary():
    """A small valid summary."""
    return {"exception_type": "ValueError", "message": "plain failure",
            "frames": [{"file": "core.py", "function": "run", "line": 3}]}


def test_next_reset_treats_naive_time_as_utc():
    """A naive moment counts as UTC and resets at the next midnight."""
    reset = cr.next_reset_utc(datetime(2026, 10, 14, 23, 59))
    assert reset == datetime(2026, 10, 15, tzinfo=timezone.utc)


def test_parse_reset_reads_epoch_naive_and_garbage():
    """Epoch seconds and naive ISO text are UTC; garbage reads as None."""
    assert cr._parse_reset(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert cr._parse_reset("2026-10-14T05:00:00") == datetime(
        2026, 10, 14, 5, tzinfo=timezone.utc)
    assert cr._parse_reset("not a time") is None
    assert cr._parse_reset(1e300) is None


def test_reports_dir_lives_under_spacr_home(monkeypatch, tmp_path):
    """The state folder is ``anonymous_reports`` in the spaCR home."""
    from spacr import logging_util

    monkeypatch.setattr(logging_util, "_spacr_home", lambda: tmp_path)
    assert _original("_reports_dir")() == tmp_path / "anonymous_reports"


def test_user_error_names_survive_a_diamond_hierarchy():
    """A class reached twice through two parents is named once."""
    from spacr.errors import ConfigurationError

    left = type("DiamondLeftError", (ConfigurationError,), {})
    right = type("DiamondRightError", (ConfigurationError,), {})
    leaf = type("DiamondLeafError", (left, right), {})
    names = cr._user_error_names()
    assert {"DiamondLeftError", "DiamondRightError",
            "DiamondLeafError"} <= names
    assert leaf.__name__ in names


def test_classify_failure_falls_back_to_fault(monkeypatch):
    """A parser that breaks while classifying leaves the failure a fault."""
    text = "Traceback (most recent call last):\nConfigurationError: bad"
    assert cr.classify_failure(None, text) == cr.KIND_USER_ERROR

    def broken(_text):
        """Fail like a damaged parser."""
        raise RuntimeError("parser broke")

    monkeypatch.setattr(cr, "_parse_traceback_text", broken)
    assert cr.classify_failure(None, text) == cr.KIND_FAULT


def test_install_id_still_returned_when_it_cannot_be_saved(monkeypatch):
    """An unwritable profile still yields an id, just a fresh one each time."""
    def refuse(_state):
        """Fail like a read-only disk."""
        raise OSError("read-only")

    monkeypatch.setattr(cr, "_write_state", refuse)
    first, second = cr.install_id(), cr.install_id()
    assert cr._UUID_RE.fullmatch(first) and cr._UUID_RE.fullmatch(second)
    assert first != second


def test_home_variants_empty_without_a_usable_home(monkeypatch):
    """No home folder, or a root-like one, gives nothing to scrub."""
    def no_home():
        """Fail like a profile without a home folder."""
        raise RuntimeError("no home")

    monkeypatch.setattr(cr.Path, "home", no_home)
    assert cr._home_variants() == []
    monkeypatch.setattr(cr.Path, "home", lambda: Path("/"))
    assert cr._home_variants() == []


def test_identity_scrub_fallback_replaces_login_and_host(monkeypatch):
    """Without the issue-report helpers, login and host are still removed."""
    import getpass
    import socket

    monkeypatch.setitem(sys.modules, "spacr.qt.ai.issue_report", None)
    monkeypatch.setattr(getpass, "getuser", lambda: "zorbaxuser")
    monkeypatch.setattr(socket, "gethostname", lambda: "zorbaxhost.lab")
    out = cr._identity_scrub("ZorbaxUser logged in on zorbaxhost today")
    assert out == "<USER> logged in on <HOST> today"


def test_identity_scrub_fallback_survives_unknown_identity(monkeypatch):
    """A login and host that cannot be read leave the text unchanged."""
    import getpass
    import socket

    def fail():
        """Fail like a system without the information."""
        raise OSError("unknown")

    monkeypatch.setitem(sys.modules, "spacr.qt.ai.issue_report", None)
    monkeypatch.setattr(getpass, "getuser", fail)
    monkeypatch.setattr(socket, "gethostname", fail)
    assert cr._identity_scrub("user ab on 12345") == "user ab on 12345"


def test_identity_scrub_fallback_skips_short_and_numeric_names(monkeypatch):
    """Names too short or all digits are not replaced, to spare real words."""
    import getpass
    import socket

    monkeypatch.setitem(sys.modules, "spacr.qt.ai.issue_report", None)
    monkeypatch.setattr(getpass, "getuser", lambda: "ab")
    monkeypatch.setattr(socket, "gethostname", lambda: "12345")
    assert cr._identity_scrub("ab used node 12345") == "ab used node 12345"


def test_scrub_public_text_without_issue_report_helpers(monkeypatch):
    """The stricter rules still remove paths when the helpers are missing."""
    monkeypatch.setitem(sys.modules, "spacr.qt.ai.issue_report", None)
    out = cr.scrub_public_text("failed at /srv/data/plate1 now")
    assert "/srv" not in out and "<PATH>" in out


def test_library_roots_skip_empty_prefixes(monkeypatch):
    """An empty interpreter prefix adds no root such as the current folder."""
    monkeypatch.setattr(sys, "base_prefix", "")
    monkeypatch.setattr(sys, "exec_prefix", "")
    roots = cr._library_roots()
    assert "./" not in roots
    assert all(root.endswith("/") and len(root) > 2 for root in roots)


def test_public_frame_file_keeps_only_tidy_pseudo_files():
    """``<stdin>`` stays; an odd pseudo-file becomes ``<string>``."""
    assert cr._public_frame_file("<stdin>") == "<stdin>"
    assert cr._public_frame_file("<bad$name/../x>") == "<string>"


def test_line_number_unreadable_is_zero():
    """A line number that is not a number reads as 0."""
    assert cr._line_number("line seven") == 0


def test_frame_drops_blank_spacr_code():
    """A spaCR frame whose source line is blank carries no ``code``."""
    frame = cr._frame(cr.__file__, "run", 5, "    ")
    assert "code" not in frame
    assert frame["file"] == "crashreport.py"


def test_parse_traceback_text_empty_and_unparseable():
    """Empty text gives nothing; a tail that is not an exception is a message."""
    assert cr._parse_traceback_text("") == ("", "", [])
    assert cr._parse_traceback_text("   just some words") == (
        "", "just some words", [])


@pytest.mark.parametrize("name, family", [
    ("win32", "windows"), ("cygwin", "windows"), ("darwin", "macos"),
    ("sunos5", "other")])
def test_os_family_maps_platforms(monkeypatch, name, family):
    """Each platform name maps to its published family."""
    monkeypatch.setattr(sys, "platform", name)
    assert cr._os_family() == family


def test_distribution_version_missing_is_empty():
    """A distribution that is not installed has no version."""
    assert cr._distribution_version("spacr-no-such-distribution-xyz") == ""


def _fake_torch(cuda=False, hip=None, mps=None, broken=False):
    """A stand-in torch with just the attributes the GPU check reads."""
    def available():
        """Answer like ``torch.cuda.is_available``."""
        if broken:
            raise RuntimeError("driver crashed")
        return cuda

    backends = types.SimpleNamespace(mps=None if mps is None else
                                     types.SimpleNamespace(
                                         is_available=lambda: mps))
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=available),
        version=types.SimpleNamespace(hip=hip), backends=backends)


@pytest.mark.parametrize("torch, expected", [
    (_fake_torch(cuda=True, hip="6.0"), "amd"),
    (_fake_torch(cuda=True), "nvidia"),
    (_fake_torch(mps=True), "apple"),
    (_fake_torch(mps=False), "none"),
    (_fake_torch(), "none"),
    (_fake_torch(broken=True), "unknown"),
])
def test_gpu_class_from_loaded_torch(monkeypatch, torch, expected):
    """An already-imported torch decides the GPU class."""
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert cr._gpu_class() == expected


def test_gpu_class_apple_silicon_without_torch(monkeypatch):
    """An arm64 Mac without torch loaded is ``apple``."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cr.platform, "machine", lambda: "arm64")
    assert cr._gpu_class() == "apple"


def test_validated_summary_repairs_id_frames_and_kind():
    """A bad id is replaced, non-dict frames dropped, an odd kind left out."""
    out = cr._validated_summary({"install_id": "not-a-uuid",
                                 "frames": ["junk", {"file": "a.py",
                                                     "function": "f",
                                                     "line": 2}],
                                 "kind": "something-else"})
    assert out["install_id"] == cr.install_id()
    assert out["frames"] == [{"file": "a.py", "function": "f", "line": 2}]
    assert "kind" not in out
    assert cr._validated_summary({"frames": "not a list"})["frames"] == []


def test_validated_summary_drops_outer_frames_to_fit():
    """An oversized summary loses its outermost frames until it fits."""
    code = ("lorem ipsum dolor sit amet " * 10)
    frames = [{"file": f"m{index}" + "a" * 70, "function": "b" * 80,
               "line": index, "code": code} for index in range(cr.MAX_FRAMES)]
    out = cr._validated_summary({"frames": frames,
                                 "message": "word " * 100,
                                 "user_note": "note " * 300,
                                 "ai_summary": "text " * 500})
    assert len(json.dumps(out).encode("utf-8")) <= cr.MAX_SUMMARY_BYTES
    assert 0 < len(out["frames"]) < cr.MAX_FRAMES
    assert out["frames"][-1]["line"] == cr.MAX_FRAMES - 1


def test_queue_files_unreadable_folder_is_empty(monkeypatch):
    """A queue folder that cannot be listed counts as empty."""
    class Unlistable:
        """A folder whose listing fails."""

        def glob(self, _pattern):
            """Fail like a vanished network share."""
            raise OSError("gone")

    monkeypatch.setattr(cr, "_queue_dir", lambda: Unlistable())
    assert cr._queue_files() == []


def test_prune_queue_deletes_damaged_entries():
    """Unreadable entries, and entries without a summary, are deleted."""
    good = cr._enqueue(cr._validated_summary(_summary()))
    folder = cr._queue_dir()
    damaged = {"000000000000001-a.json": "{not json",
               "000000000000002-b.json": json.dumps(["a", "list"]),
               "000000000000003-c.json": json.dumps(
                   {"queued_at": 9e12, "summary": ["no", "dict"]})}
    for name, text in damaged.items():
        (folder / name).write_text(text, encoding="utf-8")
    kept = cr._prune_queue()
    assert kept == [good]
    assert sorted(p.name for p in folder.glob("*.json")) == [good.name]


def test_enqueue_returns_none_when_folder_cannot_be_made(monkeypatch,
                                                         tmp_path):
    """A queue folder that cannot be created queues nothing."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    monkeypatch.setattr(cr, "_queue_dir", lambda: blocker / "queue")
    assert cr._enqueue(_summary()) is None


def test_queued_summaries_skip_entries_that_vanish(monkeypatch, tmp_path):
    """An entry gone or broken after pruning is skipped, others returned."""
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"summary": {"message": "kept"}}),
                    encoding="utf-8")
    keyless = tmp_path / "keyless.json"
    keyless.write_text(json.dumps({"queued_at": 1}), encoding="utf-8")
    monkeypatch.setattr(cr, "_prune_queue", lambda now=None: [
        tmp_path / "missing.json", keyless, good])
    assert cr.queued_summaries() == [{"message": "kept"}]


def test_open_url_hands_request_to_urlopen(monkeypatch):
    """The real transport passes the request and timeout to ``urlopen``."""
    import urllib.request

    seen = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda request, timeout: seen.append(
                            (request, timeout)) or "answer")
    assert _original("_open_url")("the request", 4.5) == "answer"
    assert seen == [("the request", 4.5)]


def test_post_json_unreadable_error_body(monkeypatch, relay):
    """An HTTP error whose body cannot be read gives its status and ``{}``."""
    class Unreadable(io.BytesIO):
        """An error body that fails to read."""

        def read(self, *args):
            """Fail like a reset connection."""
            raise ConnectionResetError("reset")

    def refuse(request, timeout):
        """Answer 503 with an unreadable body."""
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {},
                                     Unreadable())

    monkeypatch.setattr(cr, "_open_url", refuse)
    assert cr._post_json("/report", {"a": 1}, 1.0) == (503, {})


def test_post_json_non_json_answer(monkeypatch, relay):
    """An answer that is not JSON reads as an empty mapping."""
    response = _Response(200, {})
    response._body = b"<html>not json</html>"
    monkeypatch.setattr(cr, "_open_url", lambda request, timeout: response)
    assert cr._post_json("/report", {"a": 1}, 1.0) == (200, {})


def test_send_one_unreadable_issue_number(relay):
    """A sent report whose issue number is not a number has number 0."""
    relay.answers = [(200, {"issue_number": "twelve", "issue_url": "u"})]
    outcome = cr._send_one(cr._validated_summary(_summary()), 1.0)
    assert outcome["status"] == cr.REPORT_SENT
    assert outcome["issue_number"] == 0


def test_sending_allowed_false_when_check_breaks(monkeypatch, relay):
    """A switch that cannot be read means nothing is sent."""
    def broken():
        """Fail like a damaged state file reader."""
        raise RuntimeError("state broke")

    monkeypatch.setattr(cr, "anonymous_reports_enabled", broken)
    assert cr._sending_allowed() is False


def test_paused_until_unreadable_is_not_paused():
    """A pause time that is not a number does not pause reports."""
    cr._update_state(reports_paused_until="tomorrow")
    assert cr._reports_paused_until() is None


def test_background_disabled_still_tells_caller():
    """With the network forbidden, ``on_done`` hears ``disabled`` at once."""
    heard = []
    assert cr.submit_public_summary_in_background(
        _summary(), on_done=heard.append) is None
    assert heard == [{"status": cr.REPORT_DISABLED}]
    assert cr._queue_files() == []


def test_background_send_failure_is_reported_as_queued(monkeypatch, relay):
    """A send that raises leaves the report queued and says why."""
    def explode(path, summary, timeout):
        """Fail like an unexpected bug in the sender."""
        raise RuntimeError("boom")

    monkeypatch.setattr(cr, "_send_queued", explode)
    heard = []
    thread = cr.submit_public_summary_in_background(_summary(),
                                                    on_done=heard.append)
    thread.join(10)
    assert heard == [{"status": cr.REPORT_QUEUED,
                      "detail": "RuntimeError: boom"}]
    assert len(cr._queue_files()) == 1


def test_background_callback_failure_is_swallowed(relay):
    """An ``on_done`` that raises does not break the sending thread."""
    heard = []

    def bad_callback(outcome):
        """Record the outcome, then fail."""
        heard.append(outcome)
        raise ValueError("callback broke")

    thread = cr.submit_public_summary_in_background(_summary(),
                                                    on_done=bad_callback)
    thread.join(10)
    assert not thread.is_alive()
    assert heard and heard[0]["status"] == cr.REPORT_SENT
    assert len(relay.calls_to("/report")) == 1


def test_flush_waits_while_another_send_runs(relay):
    """With a send in progress, flushing posts nothing and counts the queue."""
    cr._enqueue(cr._validated_summary(_summary()))
    cr._enqueue(cr._validated_summary(_summary()))
    holder_ready, release = threading.Event(), threading.Event()

    def hold():
        """Hold the send lock like a running send."""
        with cr._SEND_LOCK:
            holder_ready.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        holder_ready.wait(10)
        counts = cr.flush_report_queue(1.0)
    finally:
        release.set()
        holder.join(10)
    assert counts == {"sent": 0, "dropped": 0, "kept": 2}
    assert relay.calls == []


def test_flush_deletes_entry_unreadable_after_pruning(monkeypatch, relay,
                                                      tmp_path):
    """An entry that lost its summary is deleted; the next is still sent."""
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"queued_at": 1}), encoding="utf-8")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"summary": _summary()}), encoding="utf-8")
    monkeypatch.setattr(cr, "_prune_queue", lambda now=None: [broken, good])
    counts = cr.flush_report_queue(1.0)
    assert counts == {"sent": 1, "dropped": 0, "kept": 0}
    assert not broken.exists() and not good.exists()
    assert len(relay.calls_to("/report")) == 1


@pytest.mark.parametrize("answer, detail", [
    ((200, {"advice": "   "}), "HTTP 200"),
    ((500, {"error": "model offline"}), "model offline"),
])
def test_diagnose_without_advice_is_an_error(relay, answer, detail):
    """A relay answer with no advice raises ``RelayRefused("error")``."""
    relay.answers = [answer]
    with pytest.raises(cr.RelayRefused) as caught:
        cr.diagnose_remotely(_summary(), "why?")
    assert caught.value.reason == "error"
    assert str(caught.value) == detail


def test_normal_allowance_ignores_unreadable_percentages():
    """Percentages that are not numbers are left out, not guessed."""
    out = cr._normal_allowance({"ai": {"best_left": 3, "share_pct": "half"},
                                "pool_pct_left": "most"})
    assert "share_pct" not in out["ai"]
    assert "pool_pct_left" not in out
    assert out["ai"]["best_left"] == 3


@pytest.mark.parametrize("answer", [(200, {}), (503, {"error": "busy"})])
def test_fetch_allowance_none_without_a_usable_answer(relay, answer):
    """An empty or failed allowance answer gives ``None``."""
    relay.answers = [answer]
    assert cr.fetch_allowance(1.0) is None
    assert len(relay.calls_to("/allowance")) == 1


def test_report_anonymously_never_raises(monkeypatch, relay):
    """A crash while building the summary is swallowed and nothing is sent."""
    def broken(*args, **kwargs):
        """Fail like a summary builder bug."""
        raise RuntimeError("summary broke")

    monkeypatch.setattr(cr, "public_summary", broken)
    assert cr._report_anonymously(ValueError("x"), "mask") is None
    assert relay.calls == []
    assert cr._queue_files() == []


def test_fault_only_modes_send_faults_alone():
    """The fault-only choices file faults and keep user errors local."""
    for mode in ("faults", "send_faults"):
        assert cr.mode_sends(cr.KIND_FAULT, mode) is True
        assert cr.mode_sends(cr.KIND_USER_ERROR, mode) is False


def test_turning_reports_on_keeps_a_choice_that_already_sends():
    """Switching on while a sending choice is saved leaves that choice."""
    cr.set_error_report_mode("faults")
    cr.set_anonymous_reports_enabled(True)
    assert cr.error_report_mode() == "faults"


def test_parse_traceback_joins_indented_message_lines():
    """Indented lines after the exception line continue its message."""
    text = ("Traceback (most recent call last):\n"
            '  File "x.py", line 1, in f\n'
            "ValueError: first part\n"
            "  second part\n")
    kind, message, frames = cr._parse_traceback_text(text)
    assert (kind, message) == ("ValueError", "first part second part")
    assert len(frames) == 1


def test_validated_summary_keeps_summarize_request():
    """A request for the relay to summarise survives without a summary."""
    out = cr._validated_summary({"summarize": True})
    assert out["summarize"] is True and "ai_summary" not in out
    assert "summarize" not in cr._validated_summary({"summarize": "yes"})


def test_public_summary_without_any_error_is_empty():
    """With no exception and no traceback the summary has no error data."""
    out = cr.public_summary(module="mask")
    assert (out["exception_type"], out["message"], out["frames"]) == (
        "", "", [])
    assert out["module"] == "mask"


def test_public_summary_from_text_keeps_spacr_code_on_request():
    """A spaCR frame's source line is kept only when code context is asked."""
    text = ("Traceback (most recent call last):\n"
            f'  File "{cr.__file__}", line 7, in run\n'
            "    value = compute(plate)\n"
            "RuntimeError: broke\n")
    with_code = cr.public_summary(traceback_text=text, code_context=True)
    without = cr.public_summary(traceback_text=text)
    assert with_code["frames"][0]["code"] == "value = compute(plate)"
    assert "code" not in without["frames"][0]


def test_ai_summary_prompt_shows_only_the_error():
    """The prompt carries the error but not the id, note or summary fields."""
    summary = dict(_summary(), user_note="my note", ai_summary="old",
                   install_id=cr.install_id(), dry_run=True)
    prompt = cr.ai_summary_prompt(summary)
    shown = json.loads(prompt.split("\n\n", 1)[1])
    assert shown["exception_type"] == "ValueError"
    for hidden in ("install_id", "user_note", "ai_summary", "dry_run",
                   "summarize"):
        assert hidden not in shown
    assert str(cr.AI_SUMMARY_TARGET_CHARS) in prompt
