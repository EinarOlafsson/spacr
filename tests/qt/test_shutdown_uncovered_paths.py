"""A force quit must not be stoppable by the things it is flushing.

``force_quit_now`` exists because the ordinary exit paths can block on the
very thread that is already wedged. It flushes what it can on the way out as
a courtesy -- but a log handler writing to a full disk, or a stdout that is
a closed pipe, raises on ``flush``, and an exception there would leave the
process running with the user's Force quit unanswered.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import shutdown                          # noqa: E402

pytestmark = pytest.mark.qt


class _BrokenHandler(logging.Handler):
    """A log sink whose backing store has gone away."""

    def emit(self, record):
        pass

    def flush(self):
        raise OSError("no space left on device")


class _RecordingHandler(logging.Handler):
    """A working sink, to prove the broken one did not stop the sweep."""

    def __init__(self):
        super().__init__()
        self.flushes = 0

    def emit(self, record):
        pass

    def flush(self):
        self.flushes += 1


class _BrokenStream:
    """A standard stream that has been closed under the process."""

    def __init__(self):
        self.writes = []

    def write(self, text):
        self.writes.append(text)
        return len(text)

    def flush(self):
        raise ValueError("I/O operation on closed file")


@pytest.fixture
def exits(monkeypatch):
    """Catch the ``os._exit`` that would otherwise end the test run."""
    codes = []
    monkeypatch.setattr(shutdown.os, "_exit", codes.append)
    return codes


def test_a_log_sink_that_cannot_flush_does_not_stop_the_force_quit(
        monkeypatch, exits):
    """The remaining handlers are still flushed, and the process still goes."""
    root = logging.getLogger()
    broken = _BrokenHandler()
    working = _RecordingHandler()
    monkeypatch.setattr(root, "handlers",
                        [broken, working] + list(root.handlers))

    shutdown.force_quit_now(4)

    assert working.flushes == 1
    assert exits == [4]


def test_a_closed_standard_stream_does_not_stop_the_force_quit(
        monkeypatch, exits):
    """stdout and stderr are tried independently; neither can veto the exit."""
    broken_out = _BrokenStream()
    broken_err = _BrokenStream()
    monkeypatch.setattr(sys, "stdout", broken_out)
    monkeypatch.setattr(sys, "stderr", broken_err)

    shutdown.force_quit_now(7)

    assert exits == [7]


def test_the_default_force_quit_code_is_a_failure(monkeypatch, exits):
    """Leaving without cleanup is not a successful exit."""
    shutdown.force_quit_now()

    assert exits == [1]
