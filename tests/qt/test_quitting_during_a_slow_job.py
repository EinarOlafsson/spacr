"""Quitting while background work is still running must exit cleanly.

Qt aborts the process when a running QThread is destroyed ("QThread:
Destroyed while thread is still running", SIGABRT, rc 134). The quit-time
drain in `spacr.qt.app` only runs on `aboutToQuit`, which needs an event
loop to quit; a process that builds a window and then simply ends never
emits it. `spacr.qt.bridge` therefore also stops live work from an
`atexit` hook. Each case runs in its own interpreter, because the abort
happens during interpreter teardown and would take pytest with it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import spacr

REPO_ROOT = Path(spacr.__file__).resolve().parents[1]
ABORT_TEXT = "Destroyed while thread"

PREAMBLE = f'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SPACR_NO_GL"] = "1"
sys.path.insert(0, {str(REPO_ROOT)!r})
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication, QWidget
app = QApplication.instance() or QApplication(["quit-test"])
'''


def _run(body: str, tmp_path: Path, timeout: int = 180):
    """Run ``body`` after the preamble in a fresh, isolated interpreter."""
    env = dict(os.environ)
    env.update(SPACR_HOME=str(tmp_path / "spacr-home"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               SPACR_LOG_DIR=str(tmp_path / "logs"),
               QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="")
    script = tmp_path / "quit_case.py"
    script.write_text(PREAMBLE + textwrap.dedent(body), encoding="utf-8")
    started = time.monotonic()
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=timeout, env=env)
    return out, time.monotonic() - started


def _assert_clean(out) -> None:
    assert out.returncode == 0, (
        f"rc {out.returncode}; stderr tail:\n{out.stderr[-3000:]}")
    assert ABORT_TEXT not in out.stderr, out.stderr[-3000:]
    assert "CASE-END" in out.stdout, out.stdout[-2000:]


def test_a_script_that_ends_during_a_job_runner_job_exits_cleanly(tmp_path):
    """No event loop ever quits, so only the exit hook can stop the job."""
    out, _ = _run('''
        from spacr.qt.job_runner import JobRunner
        owner = QWidget()
        runner = JobRunner(owner, app_key="slow read")
        runner.submit(lambda: time.sleep(0.8), lambda _r: None)
        for _ in range(5):
            app.processEvents()
        assert runner.active_jobs() == 1
        print("CASE-END", flush=True)
    ''', tmp_path)
    _assert_clean(out)


def test_quitting_the_event_loop_during_a_slow_job_exits_cleanly(tmp_path):
    """The real quit path: `aboutToQuit` runs the bounded drain."""
    out, _ = _run('''
        from spacr.qt.bridge import retire_live_work
        from spacr.qt.job_runner import JobRunner
        window = QWidget()
        window.show()
        runner = JobRunner(window, app_key="slow read")
        delivered = []
        runner.submit(lambda: time.sleep(0.8), delivered.append)
        app.aboutToQuit.connect(retire_live_work)
        QTimer.singleShot(50, app.quit)
        app.exec()
        assert delivered == [], "a cancelled job's result reached its handler"
        del window
        print("CASE-END", flush=True)
    ''', tmp_path)
    _assert_clean(out)


def test_a_job_longer_than_the_budget_is_parked_not_destroyed(tmp_path):
    """Past the 2 s budget the thread is parked and waited for, never killed."""
    out, elapsed = _run('''
        from spacr.qt.job_runner import JobRunner
        owner = QWidget()
        runner = JobRunner(owner, app_key="stubborn read")
        runner.submit(lambda: time.sleep(3.0), lambda _r: None)
        app.processEvents()
        print("CASE-END", flush=True)
    ''', tmp_path)
    _assert_clean(out)
    assert elapsed < 60, f"exit took {elapsed:.1f} s"


def test_a_qthread_owned_by_a_widget_does_not_abort_the_exit(tmp_path):
    """A screen's own QThread subclass is neither a runner nor a run."""
    out, _ = _run('''
        import spacr.qt.bridge
        class _Fetch(QThread):
            def run(self):
                time.sleep(0.8)

        window = QWidget()
        fetch = _Fetch(window)
        fetch.start()
        app.processEvents()
        print("CASE-END", flush=True)
    ''', tmp_path)
    _assert_clean(out)


@pytest.mark.slow
def test_home_with_a_slow_journal_read_exits_cleanly(tmp_path):
    """The safe-mode launch shape that aborted: Home's journal read in flight.

    The journal read is stubbed to sleep, which is what a large real run
    journal did on the maintainer's machine.
    """
    out, _ = _run('''
        from spacr.qt.widgets import home
        _read = home.RecentRunsPanel.read
        def _slow(self):
            time.sleep(1.0)
            return _read(self)
        home.RecentRunsPanel.read = _slow
        from spacr.qt.preferences import enable_safe_mode
        enable_safe_mode()
        import spacr.qt as q
        q.register_self_registering_modules()
        from spacr.qt.app import MainWindow
        window = MainWindow()
        window.resize(1200, 800)
        window.show()
        for _ in range(40):
            app.processEvents()
        print("CASE-END", flush=True)
    ''', tmp_path, timeout=600)
    _assert_clean(out)
