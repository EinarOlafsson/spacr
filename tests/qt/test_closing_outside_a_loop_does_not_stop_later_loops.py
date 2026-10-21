"""Closing the main window with no event loop running leaves later loops working.

2026-10-20 macOS: the darwin close path called ``QApplication.exit(0)`` even
at a test's teardown, where no loop runs. ``exit`` marks the application as
quitting, every later ``QEventLoop.exec`` returned -1 at once, and all 49
thread and timer tests after the first closed main window timed out.
"""
import time

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from spacr.qt import app as app_mod


def test_ending_with_no_loop_running_leaves_the_next_loop_alone(monkeypatch):
    monkeypatch.setattr(app_mod.sys, "platform", "darwin")
    application = QApplication.instance()
    app_mod._end_the_event_loop(application)
    assert not getattr(application, "_spacr_leaving", False)

    loop = QEventLoop()
    QTimer.singleShot(50, loop.quit)
    start = time.perf_counter()
    assert loop.exec() == 0
    assert time.perf_counter() - start >= 0.04
