"""What the bundle says when there is less than usual to say.

A crash report is read by someone who was not there, so the summary has to
distinguish "the doctor did not run" from "the doctor passed", and "nothing
failed" from "nothing was gathered".  The last path here is the one that
matters most and is the hardest to reach in the field: the report itself could
not be written, in which case the reporter must fall silent about the bundle
and let the user's own traceback through untouched.
"""
from __future__ import annotations

import runpy
import sys
import zipfile

import pytest

from spacr import crashreport as cr


@pytest.fixture()
def logs(tmp_path, monkeypatch):
    """Point every spaCR log path at a private directory for this test."""
    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setenv("SPACR_LOG_DIR", str(root))
    return root


def test_a_summary_with_no_doctor_verdict_does_not_invent_one():
    """No ``doctor_summary`` means the checks never ran; the line is left out."""
    report = cr.CrashReport(created_utc="2020-01-01T00:00:00+00:00",
                            run_id="3f9c1a2b")
    report.sections["spacr.log"] = "one line\n"
    report.manifest["sections"] = {"spacr.log": {"status": "ok", "bytes": 9}}

    text = report.summary()

    assert "run id       3f9c1a2b" in text
    assert "contents     spacr.log" in text
    assert "doctor" not in text
    assert "FAILED to gather" not in text
    assert "Nothing to gather" not in text
    assert report.problems == [] and report.omitted == []


def test_a_summary_lists_a_failure_without_inventing_an_omission():
    """A report with a broken collector and no empty one gets one heading."""
    report = cr.CrashReport(created_utc="2020-01-01T00:00:00+00:00")
    report.sections["spacr.log"] = "one line\n"
    report.manifest["doctor_summary"] = {"FAIL": 2, "PASS": 0}
    report.manifest["sections"] = {
        "doctor.txt": {"status": "failed", "error": "ImportError: no doctor"},
        "spacr.log": {"status": "ok"},
    }

    text = report.summary()

    assert "run id       (none identified)" in text
    assert "\ndoctor       FAIL 2\n" in text, (
        "a zero count is not worth a column, and the whole line has to end "
        "there: 'FAIL 2, PASS 0' starts with the same eight characters")
    assert "PASS" not in text
    assert "FAILED to gather (see manifest.json for the error):" in text
    assert "  - doctor.txt" in text
    assert "Nothing to gather" not in text
    assert report.problems == ["doctor.txt"] and report.omitted == []


def test_a_report_that_could_not_be_written_keeps_quiet_and_defers(
        logs, tmp_path, monkeypatch, capsys):
    """A bundle that never landed must not be announced to the user.

    The hook still hands the exception to the hook it replaced, so the user
    sees exactly the traceback they would have seen without the reporter
    installed -- plus one line on stderr saying the report could not be made.
    """
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a))
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)

    hook = cr.install_excepthook(blocked / "report.zip", checkout=str(tmp_path))
    try:
        error = ValueError("boom")
        hook(ValueError, error, None)
    finally:
        sys.excepthook = sys.__excepthook__
        blocked.chmod(0o700)

    assert len(seen) == 1 and seen[0][1] is error
    err = capsys.readouterr().err
    assert "could not write a crash report" in err
    assert "wrote a crash report" not in err
    assert list(blocked.iterdir()) == []


def test_running_the_module_as_a_script_writes_the_bundle_and_exits_zero(
        logs, tmp_path, monkeypatch, capsys):
    """``python -m spacr.crashreport`` is the documented way to make a bundle."""
    target = tmp_path / "bug.zip"
    monkeypatch.setattr(sys, "argv", [
        "spacr-crashreport", "-o", str(target), "--checkout", str(tmp_path),
        "--note", "it stopped after the second plate"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("spacr.crashreport", run_name="__main__")

    assert caught.value.code == 0
    assert target.is_file()
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        note = archive.read("note.txt").decode("utf-8")
    assert names[0] == "summary.txt" and names[-1] == "manifest.json"
    assert note == "it stopped after the second plate\n"
    assert f"Wrote {target.resolve()}" in capsys.readouterr().out
