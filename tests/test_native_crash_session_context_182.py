"""A native crash whose fault dump names no thread still reports its session (GitHub #182)."""
import json
import os
import time

from spacr import crashreport as cr
from tests.test_700_structured_error_dispatch import configured

OPEN = "11111111-2222-4333-8444-555555555555"
ENDED = "66666666-7777-4888-9999-aaaaaaaaaaaa"


def _header(owner, version="1.5.1.9"):
    session = {"known": True, "id": owner, "started_at": "2026-11-10T08:00:00Z",
               "spacr_version": version, "python": "3.12.14", "qt": "unknown", "source_sha256": {}}
    return "\n=== spaCR session " + json.dumps(session, separators=(",", ":")) + " ===\n"


def _context(owner, thread="0x10", **values):
    return "=== spaCR context " + json.dumps(dict(values, _thread=thread, _session=owner),
                                              separators=(",", ":"), sort_keys=True) + " ===\n"


def _replay(path, monkeypatch):
    delivered = []
    monkeypatch.setattr(cr, "_dispatch_error", lambda message, **kwargs: delivered.append((message, kwargs)) or True)
    assert cr._consume_native_crash(path) is True
    [(message, payload)] = delivered
    assert message == "Native process termination: Segmentation fault"
    return payload["summary"]


def test_empty_fault_dump_reports_the_only_open_session(configured, tmp_path, monkeypatch):
    path = tmp_path / "spacr-crash.log"
    path.write_text(
        _header(ENDED, "1.5.1.7") + _context(ENDED, spacr="1.5.1.7", screen="Overview")
        + _context(ENDED, action=cr._NATIVE_SHUTDOWN_ACTION)
        + _header(OPEN) + _context(OPEN, thread="0x20", spacr="1.5.1.9", os="macos", machine="arm64")
        + _context(OPEN, thread="0x20", screen="Mask") + _context(OPEN, thread="0x20", task="mask", task_state="started")
        + "Fatal Python error: Segmentation fault\n\n", encoding="utf-8")
    summary = _replay(path, monkeypatch)
    assert summary["native_session"]["id"] == OPEN
    assert summary["native_session"]["spacr_version"] == "1.5.1.9"
    assert summary["screen"] == "Mask"
    assert summary["run"] == {"stage": "mask"}
    note = summary["user_note"]
    assert "Crashed build: spaCR v1.5.1.9" in note
    assert "Task running at the crash: mask." in note
    assert "the only spaCR session still open" in note
    assert "no Python stack for any thread" in note
    assert "Overview" not in note


def test_ended_task_is_not_reported_as_running(configured, tmp_path, monkeypatch):
    path = tmp_path / "spacr-crash.log"
    path.write_text(
        _header(OPEN) + _context(OPEN, task="train-cellpose", task_state="started")
        + _context(OPEN, task="train-cellpose", task_state="ended")
        + _context(OPEN, action=cr._NATIVE_SHUTDOWN_ACTION)
        + _header(ENDED) + _context(ENDED, action=cr._NATIVE_SHUTDOWN_ACTION, thread="0x30")
        + "Fatal Python error: Segmentation fault\n\n", encoding="utf-8")
    summary = _replay(path, monkeypatch)
    assert summary["native_session"] == {"known": False}
    assert "best guess" in summary["user_note"]
    assert "Last step recorded before the crash: interpreter shutdown began." in summary["user_note"]
    assert "Task running" not in summary["user_note"]
    assert "run" not in summary


def test_sessions_before_an_earlier_fault_or_without_context_are_not_candidates():
    before = _header(ENDED) + _context(ENDED, os="macos") + "Fatal Python error: Aborted\n\n" + _header(OPEN)
    assert cr._native_session_guess(before) == ("", "")
    assert cr._native_session_guess(before + _context(OPEN, os="macos")) == (OPEN, "open")
    assert cr._native_session_guess("no spaCR lines") == ("", "")
    assert cr._native_session_guess("=== spaCR session {bad} ===\n=== spaCR context [] ===\n") == ("", "")


def test_log_without_a_complete_header_stays_unknown_and_says_so(configured, tmp_path, monkeypatch):
    path = tmp_path / "spacr-crash.log"
    path.write_text("=== spaCR session {\"known\": tr\n"
                    "Fatal Python error: Segmentation fault\n\n", encoding="utf-8")
    summary = _replay(path, monkeypatch)
    assert summary["native_session"] == {"known": False}
    assert "No crashed-session context was recorded." in summary["user_note"]
    assert "no Python stack for any thread" in summary["user_note"]
    assert summary["frames"] == []


def test_a_dump_with_frames_but_no_recorded_thread_is_never_guessed(configured, tmp_path, monkeypatch):
    path = tmp_path / "spacr-crash.log"
    path.write_text(_header(ENDED) + _header(OPEN) + _context(OPEN, thread="0x20", screen="Mask")
                    + "Fatal Python error: Aborted\n\nCurrent thread 0x99 (most recent call first):\n"
                    '  File "/usr/lib/python3.12/threading.py", line 3 in run\n', encoding="utf-8")
    delivered = []
    monkeypatch.setattr(cr, "_dispatch_error", lambda message, **kwargs: delivered.append(kwargs) or True)
    assert cr._consume_native_crash(path) is True
    summary = delivered[0]["summary"]
    assert summary["native_session"] == {"known": False}
    assert "screen" not in summary and "matched as" not in summary.get("user_note", "")


def test_matched_thread_dump_keeps_every_thread_and_no_private_path(configured, tmp_path, monkeypatch):
    path = tmp_path / "spacr-crash.log"
    path.write_text(
        _header(ENDED) + _header(OPEN) + _context(OPEN, thread="0x20", task="mask", task_state="started")
        + "Fatal Python error: Segmentation fault\n\n"
        "Thread 0x30 (most recent call first):\n"
        '  File "/Users/alice/secret/lib/python3.12/site-packages/torch/nn/modules/module.py", line 5 in forward\n\n'
        "Current thread 0x20 (most recent call first):\n"
        '  File "/Users/alice/secret/lib/python3.12/site-packages/PySide6/__init__.py", line 9 in exec\n',
        encoding="utf-8")
    summary = _replay(path, monkeypatch)
    assert summary["native_session"]["id"] == OPEN
    assert [frame["file"] for frame in summary["frames"]] == ["__init__.py", "module.py"]
    assert "matched as" not in summary["user_note"]
    assert "Task running at the crash: mask." in summary["user_note"]
    text = json.dumps(summary)
    assert "alice" not in text and "/Users" not in text


IPS_HEADER = json.dumps({"app_name": "Python", "bug_type": "309", "name": "Python"})
IPS_BODY = {
    "procName": "Python", "faultingThread": 1,
    "exception": {"type": "EXC_BAD_ACCESS", "signal": "SIGSEGV", "subtype": "KERN_INVALID_ADDRESS at 0x10"},
    "usedImages": [
        {"name": "libomp.dylib",
         "path": "/Users/alice/miniconda3/envs/spacr/lib/python3.12/site-packages/torch/lib/libomp.dylib"},
        {"name": "libomp.dylib", "path": "/Users/alice/miniconda3/envs/spacr/lib/libomp.dylib"},
        {"name": "alice_private_plugin.so", "path": "/Users/alice/projects/secret/alice_private_plugin.so"},
        {"name": "libsystem_kernel.dylib", "path": "/usr/lib/system/libsystem_kernel.dylib"},
    ],
    "threads": [
        {"queue": "com.apple.main-thread", "frames": [{"imageIndex": 3, "symbol": "mach_msg2_trap"}]},
        {"triggered": True, "name": "alice's worker", "frames": [
            {"imageIndex": 2, "symbol": "alice_secret_function"},
            {"imageIndex": 0, "symbol": "__kmp_invoke_microtask(int, void*)"},
            {"imageIndex": 3, "symbol": "_pthread_start"}]},
    ],
}

LEGACY = """Process:               Python [4242]
Path:                  /Users/alice/miniconda3/bin/python3.12
Exception Type:        EXC_BAD_ACCESS (SIGSEGV)

Thread 0 Crashed::  Dispatch queue: com.apple.main-thread
0   libtorch_cpu.dylib            \t0x0000000101 at::native::add_kernel(at::TensorIterator&) + 44
1   alice_private_plugin.so       \t0x0000000102 alice_secret + 8

Binary Images:
       0x100000000 -        0x100ffffff +libtorch_cpu.dylib (0) <0000> /Users/alice/miniconda3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.dylib
       0x200000000 -        0x200ffffff +alice_private_plugin.so (0) <0000> /Users/alice/projects/alice_private_plugin.so
       0x300000000 -        0x300ffffff +libomp.dylib (0) <0000> /Users/alice/miniconda3/lib/libomp.dylib
"""


def test_ips_report_gives_public_native_frames_and_openmp_count():
    note, frames = cr._macos_crash_facts(IPS_HEADER + "\n" + json.dumps(IPS_BODY))
    assert "macOS crash report: EXC_BAD_ACCESS SIGSEGV on a background thread." in note
    assert "OpenMP runtimes loaded: 2 (libomp.dylib, libomp.dylib)." in note
    assert frames[-1] == {"file": "<user-code>", "function": "<unknown>", "line": 0}
    assert frames[-2] == {"file": "libomp.dylib", "function": "__kmp_invoke_microtask", "line": 0}
    assert frames[0]["file"] == "libsystem_kernel.dylib"
    text = note + json.dumps(frames)
    assert "alice" not in text and "/Users" not in text and "0x10" not in text


def test_legacy_crash_text_is_read_the_same_way():
    note, frames = cr._macos_crash_facts(LEGACY)
    assert "EXC_BAD_ACCESS SIGSEGV on the main thread" in note
    assert "OpenMP runtimes loaded: 1 (libomp.dylib)." in note
    assert frames == [{"file": "<user-code>", "function": "<unknown>", "line": 0},
                      {"file": "libtorch_cpu.dylib", "function": "at.native.add_kernel", "line": 0}]
    assert "alice" not in note + json.dumps(frames)
    assert cr._macos_crash_facts("not a crash report") == ("", [])


def test_replay_on_macos_attaches_the_nearest_crash_report(configured, tmp_path, monkeypatch):
    reports = tmp_path / "DiagnosticReports"
    reports.mkdir()
    (reports / "Python-2026-11-10-080500.ips").write_text(
        IPS_HEADER + "\n" + json.dumps(IPS_BODY), encoding="utf-8")
    old = reports / "Python-2026-10-01-080500.ips"
    old.write_text(IPS_HEADER + "\n" + json.dumps(dict(IPS_BODY, exception={"type": "EXC_CRASH"})),
                   encoding="utf-8")
    stale = time.time() - 30 * 86400
    os.utime(old, (stale, stale))
    (reports / "Mail-2026-11-10.ips").write_text("{}\n{}", encoding="utf-8")
    monkeypatch.setattr(cr.sys, "platform", "darwin")
    monkeypatch.setattr(cr, "_macos_crash_reports_dir", lambda: reports)
    path = tmp_path / "spacr-crash.log"
    path.write_text(_header(OPEN) + "Fatal Python error: Segmentation fault\n\n", encoding="utf-8")
    summary = _replay(path, monkeypatch)
    assert "EXC_BAD_ACCESS SIGSEGV" in summary["user_note"]
    assert "EXC_CRASH" not in summary["user_note"]
    assert summary["frames"][-2]["file"] == "libomp.dylib"
    assert "alice" not in json.dumps(summary)


def test_macos_excerpt_is_empty_elsewhere_or_without_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(cr.sys, "platform", "linux")
    assert cr._macos_crash_excerpt(time.time()) == ("", [])
    monkeypatch.setattr(cr.sys, "platform", "darwin")
    monkeypatch.setattr(cr, "_macos_crash_reports_dir", lambda: tmp_path / "absent")
    assert cr._macos_crash_excerpt(time.time()) == ("", [])


class _Sink:
    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(json.loads(line.removeprefix("=== spaCR context ").removesuffix(" ===\n")))


def test_task_and_shutdown_lines_are_written_as_public_tokens(monkeypatch):
    sink = _Sink()
    monkeypatch.setattr(cr, "_NATIVE_CAPTURE_HANDLE", sink)
    assert cr._note_native_task("train_cellpose", "started") is True
    assert cr._note_native_task("", "started") is False
    assert cr._note_native_shutdown() is True
    assert sink.lines[0]["task"] == "train-cellpose" and sink.lines[0]["task_state"] == "started"
    assert sink.lines[1]["action"] == cr._NATIVE_SHUTDOWN_ACTION
    assert cr._native_running_task([{"task": "mask", "task_state": "started"}, {"screen": "Mask"}]) == "mask"


def test_platform_context_carries_the_spacr_version(monkeypatch):
    monkeypatch.setattr(cr, "_spacr_version", lambda: "1.5.1.9")
    assert cr._native_platform_context()["spacr"] == "1.5.1.9"
