"""``spacr-leakage`` end-to-end: exit codes and the ``--output`` side file.

Drives :func:`spacr.cli_leakage.main` against real crop folders rather than
a mocked auditor, so the exit code the shell sees is the one the audit
actually produced.
"""
from __future__ import annotations

import json

import pytest


def _crop(root, split, cls, name, payload=b"crop"):
    path = root / split / cls / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_a_clean_split_exits_zero_and_prints_the_report(tmp_path, capsys):
    """A split with no shared identity exits 0 and prints parsable JSON."""
    from spacr.cli_leakage import main

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"one")
    _crop(tmp_path, "test", "neg", "plate2_B02_f2_o2.png", b"two")

    code = main([str(tmp_path), "--group-by", "well"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["split_name"] == "train_vs_test"


def test_leakage_exits_one(tmp_path, capsys):
    """A well present on both sides is leakage, which is exit code 1."""
    from spacr.cli_leakage import main

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"one")
    _crop(tmp_path, "test", "neg", "plate1_A01_f1_o2.png", b"two")

    assert main([str(tmp_path), "--group-by", "well"]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_an_unusable_dataset_exits_two_and_says_why_on_stderr(tmp_path, capsys):
    """A folder with no train/ images cannot be audited: exit 2, not 0."""
    from spacr.cli_leakage import main

    _crop(tmp_path, "test", "neg", "plate1_A01_f1_o1.png", b"one")

    code = main([str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "spacr-leakage:" in captured.err
    assert "train" in captured.err


def test_a_bad_group_by_is_rejected_by_the_parser(tmp_path):
    """``--group-by`` is a closed choice list, so argparse exits, not main."""
    from spacr.cli_leakage import main

    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path), "--group-by", "quadrant"])
    assert excinfo.value.code == 2


def test_output_writes_the_same_payload_that_was_printed(tmp_path, capsys):
    """``--output`` writes the report to disk as well as to stdout."""
    from spacr.cli_leakage import main

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"one")
    _crop(tmp_path, "test", "neg", "plate2_B02_f2_o2.png", b"two")
    report_path = tmp_path / "reports" / "audit.json"

    code = main([str(tmp_path), "--group-by", "well",
                 "--output", str(report_path)])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == printed


def test_no_content_hash_skips_the_byte_identical_check(tmp_path, capsys):
    """Without content hashing a renamed byte-identical crop is not found."""
    from spacr.cli_leakage import main

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"same pixels")
    _crop(tmp_path, "test", "neg", "plate9_Z99_f9_o9.png", b"same pixels")

    with_hash = main([str(tmp_path), "--group-by", "none"])
    hashed = json.loads(capsys.readouterr().out)
    without_hash = main([str(tmp_path), "--group-by", "none",
                         "--no-content-hash"])
    unhashed = json.loads(capsys.readouterr().out)

    assert with_hash == 1
    assert hashed["overlap_counts"]["content_sha256"] == 1
    assert without_hash == 0
    assert unhashed["overlap_counts"]["content_sha256"] == 0


def test_allow_unverifiable_downgrades_an_unparseable_name(tmp_path, capsys):
    """``--allow-unverifiable`` turns a critical identity gap into a warning."""
    from spacr.cli_leakage import main

    _crop(tmp_path, "train", "neg", "unparseable.png", b"one")
    _crop(tmp_path, "test", "neg", "also-unparseable.png", b"two")

    strict = main([str(tmp_path), "--group-by", "well"])
    strict_payload = json.loads(capsys.readouterr().out)
    lenient = main([str(tmp_path), "--group-by", "well",
                    "--allow-unverifiable"])
    json.loads(capsys.readouterr().out)

    assert strict == 1
    assert "unverifiable_well" in strict_payload["critical_levels"]
    assert lenient == 0


@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_the_module_runs_as_a_script(tmp_path, capsys, monkeypatch):
    """``python -m spacr.cli_leakage`` raises SystemExit carrying the code.

    The ``__main__`` guard is the shape the console-script entry point and a
    ``-m`` invocation both take, so it is exercised the way a shell does:
    by executing the module under the name ``__main__``.
    """
    import runpy
    import sys

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"one")
    _crop(tmp_path, "test", "neg", "plate1_A01_f1_o2.png", b"two")
    monkeypatch.setattr(
        sys, "argv", ["spacr-leakage", str(tmp_path), "--group-by", "well"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.cli_leakage", run_name="__main__")

    assert excinfo.value.code == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
