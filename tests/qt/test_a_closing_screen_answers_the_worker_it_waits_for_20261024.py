"""A screen closing on the GUI thread answers the worker it is waiting for.

A finishing worker closes its run journal, and that reads the workspace
sections on the GUI thread. Model Compare's ``closeEvent`` waited for its
threads with a plain ``thread.wait(5000)``: the GUI thread sat in the wait
while the worker sat waiting for the GUI thread, and after five seconds the
close gave up and left the thread running. In the fold cohort that was
``test_closing_the_screen_closes_a_module_that_was_never_opened``, whenever
earlier tests had left workspace contributors registered. The wait now runs
the calls posted to the GUI thread between short slices.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import bridge  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_close_answers_a_worker_that_asks_the_gui_thread(
        qtbot, qt_theme_applied):
    from spacr.qt.screens.model_compare import ModelCompareScreen

    bridge._install_the_workspace_dispatcher()
    asking = threading.Event()
    answered = []

    def job():
        asking.set()
        answered.append(bridge._call_on_the_gui_thread(
            lambda: threading.current_thread() is threading.main_thread()))
        return None

    screen = ModelCompareScreen()
    qtbot.addWidget(screen)
    screen._run_job(job, lambda _result: None, operation="probe")
    assert asking.wait(30)
    time.sleep(0.05)
    started = time.monotonic()
    screen.close()
    took = time.monotonic() - started

    assert answered == [True]
    assert all(not thread.isRunning() for thread, _worker in screen._jobs)
    assert took < 2.0


def test_the_wait_returns_at_once_for_a_stopped_or_missing_thread():
    assert bridge._wait_serving_the_interface(None, 1000)
