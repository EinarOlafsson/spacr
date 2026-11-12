"""A native crash report is built from whatever survives, and never from guesses.

The crash log and the macOS crash reports are written by a process that was
dying, so any line may be cut short or belong to another program. GitHub
#182 taught the replay to name the crashed session and the task it was
running; these tests feed it the damaged and partial inputs a real crash
leaves behind -- malformed context lines, sessions it cannot vouch for,
reports with no faulting thread, unreadable report folders -- and pin that
each is passed over rather than trusted.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from spacr import crashreport as cr

OPEN = "11111111-2222-4333-8444-555555555555"
OTHER = "66666666-7777-4888-9999-aaaaaaaaaaaa"


def _session(owner):
    return "=== spaCR session " + json.dumps({"id": owner}) + " ===\n"


def _context(owner, **values):
    return "=== spaCR context " + json.dumps(dict(values, _session=owner)) + " ===\n"


def test_damaged_or_foreign_context_lines_never_pick_the_session():
    log = (_session(OPEN) + _session("not-a-uuid")
           + "=== spaCR context {broken ===\n"
           + _context("not-a-uuid", screen="Mask")
           + _context(OTHER, screen="Overview")
           + _context(OPEN, screen="Mask"))
    assert cr._native_session_guess(log) == (OPEN, "open")
    assert cr._native_session_guess(_session(OPEN) + _context(OTHER, screen="x")) == ("", "")

    contexts = cr._native_contexts_of(
        log + "=== spaCR context {not json} ===\n" + _context(OPEN, _thread="0x1"), OPEN)
    assert contexts == [{"screen": "Mask"}]


def test_the_running_task_is_the_newest_one_never_ended():
    contexts = [{"task": "mask", "task_state": "started"},
                {"task": "train", "task_state": "started"},
                {"task": "measure", "task_state": "ended"},
                {"screen": "Mask"},
                {"task": "train", "task_state": "ended"}]
    assert cr._native_running_task(contexts) == "mask"
    assert cr._native_running_task(contexts + [{"task": "mask", "task_state": "ended"}]) == ""


def test_an_unknown_session_still_names_the_build_its_context_recorded():
    note, _screen = cr._native_crash_note({"known": False}, [{"spacr": "1.5.1.9"}],
                                         [(False, [])])
    assert note.startswith("Crashed build: spaCR v1.5.1.9.")


def test_macos_reports_live_in_the_users_diagnostic_reports_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cr._macos_crash_reports_dir() == tmp_path / "Library" / "Logs" / "DiagnosticReports"


IMAGES = [{"name": "libsystem_kernel.dylib", "path": "/usr/lib/system/libsystem_kernel.dylib"},
          {"name": "libomp.dylib", "path": "/opt/conda/lib/libomp.dylib"}]


@pytest.mark.parametrize(
    ("body", "note", "frames"),
    [
        # No faultingThread: the triggered thread is the one that crashed,
        # and a frame that is not an object is skipped.
        ({"exception": {"type": "EXC_CRASH", "signal": "SIGABRT"}, "usedImages": IMAGES,
          "threads": [{"frames": []},
                      {"triggered": True, "frames": ["junk", {"imageIndex": 0, "symbol": "abort"}]}]},
         "macOS crash report: EXC_CRASH SIGABRT on a background thread. Innermost native "
         "frames: libsystem_kernel.dylib abort. OpenMP runtimes loaded: 1 (libomp.dylib).",
         [{"file": "libsystem_kernel.dylib", "function": "abort", "line": 0}]),
        # A report with neither an exception nor a frame says nothing.
        ({"threads": []}, "", []),
    ],
)
def test_an_ips_report_reads_only_what_it_can_vouch_for(body, note, frames):
    header = json.dumps({"app_name": "Python"})
    assert cr._macos_crash_facts(header + "\n" + json.dumps(body)) == (note, frames)


def test_a_legacy_report_without_a_crashed_thread_still_names_the_exception():
    text = ("Exception Type:  EXC_BAD_ACCESS (SIGSEGV)\n"
            "Thread 0:: Dispatch queue: com.apple.main-thread\n")
    note, frames = cr._macos_crash_facts(text)
    assert note == ("macOS crash report: EXC_BAD_ACCESS SIGSEGV. "
                    "OpenMP runtimes loaded: 0.")
    assert frames == []

    crashed_last = text + ("Thread 1 Crashed:\n"
                           "0   libsystem_kernel.dylib  \t0x0000000101 __pthread_kill + 8")
    note, frames = cr._macos_crash_facts(crashed_last)
    assert "on a background thread" in note
    assert frames == [{"file": "<user-code>", "function": "<unknown>", "line": 0}]


@pytest.fixture
def darwin_reports(monkeypatch, tmp_path):
    reports = tmp_path / "DiagnosticReports"
    monkeypatch.setattr(cr.sys, "platform", "darwin")
    monkeypatch.setattr(cr, "_macos_crash_reports_dir", lambda: reports)
    return reports


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_an_unreadable_report_folder_or_report_gives_no_excerpt(darwin_reports):
    now = time.time()
    assert cr._macos_crash_excerpt(now) == ("", [])

    darwin_reports.mkdir()
    assert cr._macos_crash_excerpt(now) == ("", [])

    (darwin_reports / "Python-dangling.ips").symlink_to(darwin_reports / "gone.ips")
    (darwin_reports / "Python-a-folder.ips").mkdir()
    assert cr._macos_crash_excerpt(now) == ("", [])
    (darwin_reports / "Python-a-folder.ips").rmdir()

    report = darwin_reports / "spacr-2026-11-12.crash"
    report.write_text("Exception Type:  EXC_BAD_ACCESS (SIGSEGV)\n", encoding="utf-8")
    note, _frames = cr._macos_crash_excerpt(Path(report).stat().st_mtime)
    assert note.startswith("macOS crash report: EXC_BAD_ACCESS SIGSEGV")
