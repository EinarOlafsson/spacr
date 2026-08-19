"""A timer started off its own thread never fires. Instruction 163.

REPORTED 2026-08-18, four at once, arriving with the output of a pip run:

    Qt warning: QBasicTimer::start: Timers cannot be started from another thread

THIS IS NOT LOG NOISE. Qt refuses the start, so whatever the timer was driving
silently stops -- the warning is reporting a dead feature. pip runs on
`_UpdateWorker`, a QThread, and its captured stdout reaches the console from
that thread, which is how a widget's animation got started from the wrong one.
"""

import threading

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import pytest


def test_start_from_another_thread_does_not_warn_and_does_not_lose_the_timer(
        qtbot):
    """Marshalled, not refused.

    Refusing the call on the wrong thread would trade a warning for a spinner
    that never appears -- the same defect with no message attached.
    """
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.console_panel import _WorkingDots

    dots = _WorkingDots()
    qtbot.addWidget(dots)

    warnings = []

    def collect(mode, context, message):
        if "QBasicTimer" in str(message):
            warnings.append(str(message))

    from PySide6.QtCore import qInstallMessageHandler

    previous = qInstallMessageHandler(collect)
    try:
        done = threading.Event()

        def off_thread():
            dots.start()
            done.set()

        worker = threading.Thread(target=off_thread)
        worker.start()
        assert done.wait(10)
        worker.join(10)
        # The queued call runs on the widget's thread when the loop turns.
        for _ in range(50):
            QApplication.processEvents()
    finally:
        qInstallMessageHandler(previous)

    assert not warnings, warnings
    assert dots._timer.isActive(), (
        "the timer did not start, so the animation is dead -- which is the "
        "bug the warning was reporting")


def test_stop_is_safe_from_another_thread_too(qtbot):
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.console_panel import _WorkingDots

    dots = _WorkingDots()
    qtbot.addWidget(dots)
    dots.start()
    assert dots._timer.isActive()

    done = threading.Event()

    def off_thread():
        dots.stop()
        done.set()

    worker = threading.Thread(target=off_thread)
    worker.start()
    assert done.wait(10)
    worker.join(10)
    for _ in range(50):
        QApplication.processEvents()

    assert not dots._timer.isActive()


def test_on_its_own_thread_it_is_still_immediate(qtbot):
    """The common path must not become asynchronous.

    A spinner that appears one event-loop turn late on every call would be a
    cost paid by every caller for the benefit of the rare one.
    """
    from spacr.qt.widgets.console_panel import _WorkingDots

    dots = _WorkingDots()
    qtbot.addWidget(dots)
    dots.start()
    assert dots._timer.isActive(), "the same-thread call must take effect now"
