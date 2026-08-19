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

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

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
