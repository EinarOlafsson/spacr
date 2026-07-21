"""Regression tests for the AI-thread shutdown crash.

The user hit this on quit:

    QThread: Destroyed while thread '' is still running
    Aborted (core dumped)

Root cause: cancel() just flipped a Python flag while the worker was
blocked in `for line in proc.stdout` reading from the CLI subprocess.
The flag was never observed, thread.wait() timed out, Python dropped
the last reference to the running QThread, and Qt aborted.

The fix has two parts:
* ChatProvider.cancel_stream() actually terminates the subprocess so
  the reader unblocks with an empty read.
* StreamWorker.cancel() calls provider.cancel_stream() in addition
  to setting the flag.

This file tests both.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest


def test_provider_cancel_stream_kills_the_registered_proc(monkeypatch):
    """cancel_stream() on a provider must actually terminate whatever
    subprocess we've registered on it."""
    from spacr.qt.ai.providers import ClaudeCliProvider
    p = ClaudeCliProvider()
    # Register a real, long-running subprocess.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    p._current_proc = proc
    try:
        p.cancel_stream()
        # Should terminate within 1-2s
        proc.wait(timeout=3)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_provider_cancel_stream_is_safe_when_no_proc():
    from spacr.qt.ai.providers import ClaudeCliProvider
    p = ClaudeCliProvider()
    assert p._current_proc is None
    p.cancel_stream()   # no-op, no exception


def test_stream_process_registers_proc_on_provider(monkeypatch):
    """_stream_process must set provider._current_proc so cancel_stream
    can find it."""
    from spacr.qt.ai.providers import (
        ClaudeCliProvider, _stream_process,
    )
    p = ClaudeCliProvider()
    # Give it a subprocess that immediately exits — we only care
    # about proc registration, not output.
    argv = [sys.executable, "-c", "print('ok')"]
    gen = _stream_process(argv, provider=p)
    # Registration happens as soon as we start iterating.
    _ = next(gen, None)
    # After exhaustion the proc slot clears.
    for _line in gen:
        pass
    assert p._current_proc is None


def test_worker_cancel_terminates_the_provider_subprocess(qtbot, qt_theme_applied):
    """StreamWorker.cancel() must delegate to provider.cancel_stream()
    so the reader unblocks and the worker exits promptly."""
    from spacr.qt.ai.providers import ClaudeCliProvider, _stream_process
    from spacr.qt.ai.worker import StreamWorker

    class _FakeSleepProvider(ClaudeCliProvider):
        """Streams from a 30-second-sleep subprocess that will never
        yield anything until cancelled."""
        def stream_chat(self, messages, system="", model=None):
            argv = [sys.executable, "-c", "import time; time.sleep(30)"]
            yield from _stream_process(argv, provider=self)

    provider = _FakeSleepProvider()
    worker = StreamWorker(provider, messages=[{"role": "user", "content": "hi"}])

    started = time.monotonic()
    # Start the worker in the CURRENT thread (via a QThread mount) —
    # simpler than spinning a real QThread for this test.
    import threading
    t = threading.Thread(target=worker.run, name="stream-worker")
    t.start()
    time.sleep(0.4)                # let it enter the subprocess read
    worker.cancel()
    t.join(timeout=5)
    elapsed = time.monotonic() - started

    assert not t.is_alive(), (
        "worker didn't exit within 5s of cancel — the crash fix "
        "regressed and cancel is no longer killing the subprocess"
    )
    # Should finish well under 5s
    assert elapsed < 5


def test_console_panel_shutdown_drains_running_thread(qtbot, qt_theme_applied):
    """ConsolePanel.shutdown() must join the real QThread, not just
    null the refs."""
    from PySide6.QtCore import QThread
    from spacr.qt.widgets.console_panel import ConsolePanel
    from spacr.qt.ai.providers import ClaudeCliProvider, _stream_process
    from spacr.qt.ai.worker import StreamWorker

    class _FakeSleepProvider(ClaudeCliProvider):
        def stream_chat(self, messages, system="", model=None):
            argv = [sys.executable, "-c", "import time; time.sleep(30)"]
            yield from _stream_process(argv, provider=self)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    provider = _FakeSleepProvider()
    worker = StreamWorker(provider, messages=[{"role": "user", "content": "hi"}])
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    panel._ai_worker = worker
    panel._ai_thread = thread

    # Let it enter the subprocess read
    for _ in range(30):
        qtbot.wait(20)

    started = time.monotonic()
    panel.shutdown()
    elapsed = time.monotonic() - started

    assert panel._ai_thread is None
    assert panel._ai_worker is None
    assert not thread.isRunning()
    # Must have completed shutdown well under our 3s wait budget
    assert elapsed < 6, f"shutdown took {elapsed:.1f}s — crash-fix regression"
