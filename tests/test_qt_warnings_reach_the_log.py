"""A Qt warning belongs in the log, not only on a terminal.

The handler printed to stderr and nowhere else, so every Qt warning was visible
to whoever was watching the terminal and invisible to anyone reading
~/.spacr/logs/spacr.log afterwards.

That cost real time on 2026-08-19. "QBasicTimer::start: Timers cannot be started
from another thread" arrives immediately before a crash on the maintainer's
machine, and grepping the log for it returned ZERO -- so the one line that
mattered could only be had by asking them to copy it out of a terminal the crash
had already taken with it.

A crash report is written from the log, not from a screen somebody happened to
be looking at.
"""

import io
import logging

import spacr


import pytest
from PySide6.QtCore import qCritical, qWarning


def _capture(qtbot):
    import spacr.qt as module

    module._install_quiet_qt_logging()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("spacr.qt")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return stream, logger, handler


def test_a_qt_warning_is_logged(qtbot):
    stream, logger, handler = _capture(qtbot)
    try:
        qWarning("QBasicTimer::start: Timers cannot be started from another "
                 "thread")
        assert "QBasicTimer" in stream.getvalue()
    finally:
        logger.removeHandler(handler)


def test_the_level_follows_qt_s_own(qtbot):
    """A critical must not arrive as a warning: the level is what a reader
    filters on when the log is three megabytes."""
    stream, logger, handler = _capture(qtbot)
    records = []
    logger.addHandler(type("H", (logging.Handler,), {
        "emit": lambda self, record: records.append(record)})())
    try:
        qCritical("a critical Qt message")
        assert any(r.levelno >= logging.ERROR for r in records), (
            [r.levelname for r in records])
    finally:
        logger.removeHandler(handler)


def test_it_still_reaches_the_terminal(qtbot, capsys):
    """Both, not one instead of the other -- somebody watching a run should
    not have to tail a file to see a warning."""
    stream, logger, handler = _capture(qtbot)
    try:
        qWarning("a message for the terminal")
        assert "a message for the terminal" in capsys.readouterr().err
    finally:
        logger.removeHandler(handler)


def test_a_thread_affinity_warning_carries_a_python_stack(qtbot):
    """`QBasicTimer::start` comes from Qt's C++ internals.

    The Python-level guard on `QTimer.start` never sees it -- measured on the
    maintainer's machine, where the warning arrives 3-14 ms after every run
    closes and the guard recorded nothing. THIS handler runs in the emitting
    thread at the moment of the warning, so a stack taken here names the
    Python call that entered Qt.
    """
    import threading

    from PySide6.QtCore import QTimer

    stream, logger, handler = _capture(qtbot)
    timer = QTimer()
    try:
        def the_guilty_function():
            timer.start(50)          # illegal: the timer lives on this thread

        worker = threading.Thread(target=the_guilty_function,
                                  name="pipeline-worker")
        worker.start()
        worker.join(10)

        out = stream.getvalue()
        assert "Python stack at that warning" in out
        assert "the_guilty_function" in out, (
            "the stack does not name the caller, which is the whole point")
    finally:
        logger.removeHandler(handler)


def test_an_ordinary_warning_gets_no_stack(qtbot):
    """A stack on every Qt warning would bury the one that matters."""
    stream, logger, handler = _capture(qtbot)
    try:
        qWarning("something ordinary and unrelated")
        assert "Python stack at that warning" not in stream.getvalue()
    finally:
        logger.removeHandler(handler)
