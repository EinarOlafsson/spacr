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


def test_mac_exit_waits_until_the_accepted_close_has_returned(monkeypatch):
    monkeypatch.setattr(app_mod.sys, "platform", "darwin")
    monkeypatch.setattr(app_mod, "_an_event_loop_is_running", lambda: True)
    queued, calls = [], []
    monkeypatch.setattr(QTimer, "singleShot",
                        lambda delay, callback: queued.append((delay, callback)))

    class Application:
        def closeAllWindows(self):
            calls.append("closeAllWindows")
            app_mod._end_the_event_loop(self)

        def exit(self, code):
            calls.append(("exit", code))

        def quit(self):
            raise AssertionError("Darwin must not request Cocoa termination")

    application = Application()
    app_mod._end_the_event_loop(application)
    app_mod._end_the_event_loop(application)
    assert calls == ["closeAllWindows"]
    assert application._spacr_leaving is True
    assert len(queued) == 1
    delay, leave = queued.pop()
    assert delay == 0
    leave()
    assert calls == ["closeAllWindows", ("exit", 0)]


def test_other_platforms_keep_the_direct_quit_path(monkeypatch):
    monkeypatch.setattr(app_mod.sys, "platform", "linux")
    calls = []

    class Application:
        def quit(self):
            calls.append("quit")

    app_mod._end_the_event_loop(Application())
    assert calls == ["quit"]
