"""A crash report is written from a machine that is already misbehaving.

Every fact in a report is read from something that can be broken: the metadata
database, torch, the run log, the settings file the run was started from. The
whole point of the bundle is that it survives all of them -- a report that
raised while gathering the GPU section would leave the user with a crash and
nothing to send.

So each reader answers in place: "not importable", "unreadable: <error>", or
nothing at all for a section that simply had nothing in it. The distinction
between "failed" and "empty" is kept, because a section listed as a problem
when it merely had nothing to say trains the reader to skip both.
"""
from __future__ import annotations

import sys
import types

import pytest

from spacr import crashreport


def _report():
    return crashreport.CrashReport(created_utc="2020-01-01T00:00:00Z")


# -- the environment readers ------------------------------------------------

def test_a_metadata_database_that_misbehaves_is_reported_not_raised(
        monkeypatch):
    """A failure that is not "absent" is recorded against that package.

    Falling through would abort the whole environment section over one
    unreadable distribution.
    """
    import importlib.metadata as metadata

    def _explode(name):
        raise ValueError("corrupt dist-info")

    monkeypatch.setattr(metadata, "version", _explode)

    versions = crashreport._package_versions()

    assert set(versions) == set(crashreport.REPORTED_PACKAGES)
    assert all(v.startswith("unreadable: ") for v in versions.values())


def test_a_missing_torch_is_reported_as_not_importable(monkeypatch):
    """No torch is an ordinary answer, not a missing GPU section."""
    monkeypatch.setitem(sys.modules, "torch", None)

    facts = crashreport._gpu_facts()

    assert list(facts) == ["torch"]
    assert facts["torch"].startswith("not importable: ")


def _fake_torch(**over):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.0"
    torch.version = types.SimpleNamespace(cuda="12.4")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i: "Fake GPU",
        get_device_capability=lambda i: (8, 6),
    )
    for key, value in over.items():
        setattr(torch.cuda, key, value)
    return torch


def test_a_cuda_probe_that_fails_stops_at_the_probe(monkeypatch):
    """An unreadable CUDA availability is recorded and nothing further asked.

    Going on to enumerate devices after the probe failed is how the reporter
    itself segfaults on a machine whose driver is the reason for the crash.
    """
    def _explode():
        raise RuntimeError("driver mismatch")

    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(is_available=_explode))

    facts = crashreport._gpu_facts()

    assert facts["torch"] == "2.9.0"
    assert facts["cuda_available"].startswith("unreadable: ")
    assert "devices" not in facts


def test_devices_that_cannot_be_enumerated_are_reported_as_unreadable(
        monkeypatch):
    """A device list that raises leaves a note rather than no GPU section."""
    def _explode(index):
        raise RuntimeError("device lost")

    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(get_device_name=_explode))

    facts = crashreport._gpu_facts()

    assert facts["cuda_available"] is True
    assert facts["devices"].startswith("unreadable: ")


def test_devices_are_listed_when_they_can_be_read(monkeypatch):
    """The same path does enumerate the devices on a healthy machine."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    facts = crashreport._gpu_facts()

    assert facts["devices"] == [
        {"index": 0, "name": "Fake GPU", "capability": "8.6"}]


def test_an_unimportable_spacr_still_has_a_version_and_a_location(
        monkeypatch):
    """The report says "unknown" rather than failing to open at all.

    A broken installation is exactly the case where somebody is writing a
    crash report, so the two facts that identify it must not need it.
    """
    monkeypatch.setitem(sys.modules, "spacr", None)

    assert crashreport._spacr_version() == "unknown"
    assert crashreport._spacr_location().startswith("unreadable: ")


# -- the run id --------------------------------------------------------------

def test_no_runctx_module_leaves_a_note_and_no_run_id(monkeypatch):
    """A run id that cannot be looked up is empty, and the reason is recorded."""
    monkeypatch.setitem(sys.modules, "spacr.runctx", None)
    notes = []

    assert crashreport.find_last_run_id(notes) == ""
    assert any("spacr.runctx could not be imported" in note for note in notes)


def test_an_unreadable_active_run_id_falls_back_to_the_log_folder(
        monkeypatch, tmp_path):
    """A failing ``current_run_id`` is noted and the search goes on.

    The newest run log on disk is still a usable answer, and giving up would
    file the report against no run at all.
    """
    import spacr.runctx as runctx

    def _explode():
        raise RuntimeError("no run context")

    (tmp_path / "run-7.jsonl").write_text("{}\n")
    monkeypatch.setattr(runctx, "current_run_id", _explode)
    monkeypatch.setattr(runctx, "runs_log_dir", lambda: str(tmp_path))
    notes = []

    assert crashreport.find_last_run_id(notes) == "run-7"
    assert any("active run id could not be read" in note for note in notes)


# -- the sections ------------------------------------------------------------

def test_a_run_with_no_log_file_is_empty_not_failed(monkeypatch, tmp_path):
    """A run that logged nothing is an ordinary answer, not a problem."""
    import spacr.runctx as runctx

    monkeypatch.setattr(runctx, "run_log_path",
                        lambda run_id: str(tmp_path / "absent.jsonl"))
    report = _report()

    crashreport._run_log_section(report, "run-7")

    assert "run-run-7.jsonl" not in report.sections
    assert report.manifest["sections"]["run-run-7.jsonl"]["status"] == "empty"
    assert report.problems == []


def test_a_run_with_no_warnings_has_no_problem_summary(monkeypatch):
    """No warnings or errors means no summary section, not an empty one."""
    import spacr.runctx as runctx

    monkeypatch.setattr(runctx, "read_run_log",
                        lambda run_id, level=None: [])
    report = _report()

    crashreport._run_summary_section(report, "run-7")

    assert "run-problems.txt" not in report.sections
    assert report.manifest["sections"]["run-problems.txt"]["status"] == "empty"


def test_a_logged_traceback_is_carried_into_the_summary(monkeypatch):
    """A record's traceback is included under its line, not dropped.

    The traceback is the reason the record is in the summary at all.
    """
    import spacr.runctx as runctx

    monkeypatch.setattr(runctx, "read_run_log", lambda run_id, level=None: [
        {"utc": "2020-01-01T00:00:00Z", "level": "ERROR",
         "logger": "spacr.core", "message": "segmentation failed",
         "traceback": "Traceback (most recent call last):\n  boom\n"}])
    report = _report()

    crashreport._run_summary_section(report, "run-7")

    text = report.sections["run-problems.txt"]
    assert "segmentation failed" in text
    assert "Traceback (most recent call last):" in text


def test_explicit_settings_values_are_written_without_a_file():
    """A caller holding the settings in memory needs no file on disk."""
    import json

    report = _report()

    crashreport._settings_section(report, None, {"src": "/data", "gpu": True})

    written = json.loads(report.sections["settings.json"])
    assert written == {"src": "/data", "gpu": True}


def test_a_settings_file_that_is_not_there_is_named_in_the_manifest(tmp_path):
    """A missing settings file is recorded by path rather than passed over.

    "the defaults" is almost never what was run, so the reader has to be able
    to see which file was expected and was not found.
    """
    report = _report()
    missing = tmp_path / "settings.csv"

    crashreport._settings_section(report, missing, None)

    assert "settings.json" not in report.sections
    assert report.manifest["settings"]["missing"] == str(missing)


def test_a_database_that_is_not_there_leaves_no_run_status(tmp_path):
    """A crash before the database was created has no per-well status."""
    report = _report()

    crashreport._run_status_section(report, tmp_path / "absent.db")

    assert "run-status.json" not in report.sections
    assert report.manifest["sections"]["run-status.json"]["status"] == "empty"


def test_a_database_with_no_status_rows_leaves_no_run_status(monkeypatch,
                                                             tmp_path):
    """An empty status table is empty, not a failed section."""
    import spacr.errors as errors

    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    monkeypatch.setattr(errors, "read_run_status", lambda path: [])
    report = _report()

    crashreport._run_status_section(report, db)

    assert "run-status.json" not in report.sections
    assert report.manifest["sections"]["run-status.json"]["status"] == "empty"


def test_a_value_json_cannot_hold_is_kept_as_its_repr():
    """An unserialisable setting is recorded rather than dropped.

    A settings dump missing the one key that was unusual is a dump that
    cannot reproduce the run.
    """
    class _Odd:
        def __repr__(self):
            return "<a model object>"

    assert crashreport._jsonable(_Odd()) == "<a model object>"
    assert crashreport._jsonable({"m": _Odd()}) == {"m": "<a model object>"}
    assert crashreport._jsonable([_Odd()]) == ["<a model object>"]
