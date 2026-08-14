"""Extra-performance mode logged two lines every couple of seconds saying
nothing had happened.

Reported as issue #83. The pre-run cleanup runs before every job in
``extra_performance`` mode, and most of the time there is nothing to
reclaim -- the caches are already empty, or the allocator has not handed the
pages back to the OS. Both results were logged at INFO::

    [INFO] before ml_analyze usage: RAM: freed nothing measurable. ...
    [INFO] before ml_analyze usage: VRAM: nothing to measure — ...
    [INFO] before ml_analyze usage: RAM: freed nothing measurable. ...
    ... every two seconds

That drowns the console and trains the reader to ignore the log, which costs
the one line that eventually does matter.

INFO is now reserved for a cleanup that MOVED something -- freed memory, or
found the process larger than before. Both are worth a line. Everything else
is DEBUG, still available when someone is diagnosing memory on purpose.
"""

import logging

import pytest

pytest.importorskip("PySide6")

from spacr.qt import resource_cleanup


class _Result:
    """Stand-in with the two properties `_report` reads."""

    def __init__(self, freed=0, grew=0, text="RAM: freed nothing measurable."):
        self.freed = freed
        self.grew = grew
        self._text = text

    def summary(self):
        return self._text


def _levels(caplog, result, **kwargs):
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=resource_cleanup.LOG.name):
        resource_cleanup._report(result, **kwargs)
    return [r.levelno for r in caplog.records]


def test_a_cleanup_that_freed_nothing_is_debug(caplog):
    """The exact case from the report, fired every couple of seconds."""
    assert _levels(caplog, _Result()) == [logging.DEBUG]


def test_nothing_to_measure_is_debug(caplog):
    """The VRAM half: no CUDA context, so there is nothing to say."""
    assert _levels(caplog, _Result(
        text="VRAM: nothing to measure — no initialised CUDA context")) \
        == [logging.DEBUG]


def test_a_cleanup_that_freed_memory_is_info(caplog):
    """The line worth keeping."""
    assert _levels(caplog, _Result(freed=64 * 1024 * 1024,
                                   text="RAM: freed 64.0 MB.")) \
        == [logging.INFO]


def test_a_process_that_GREW_is_info(caplog):
    """Also news, and arguably worse news: the cleanup made it bigger."""
    assert _levels(caplog, _Result(grew=96 * 1024,
                                   text="RAM: freed nothing — 96.0 KB more")) \
        == [logging.INFO]


def test_the_message_still_reaches_debug_for_diagnosis(caplog):
    """Quieting must not mean losing it -- the numbers still exist for
    someone reading the log on purpose."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=resource_cleanup.LOG.name):
        resource_cleanup._report(_Result(), prefix="before mask: ")
    assert "before mask: RAM: freed nothing measurable." in caplog.text


def test_neither_log_site_calls_info_directly():
    """Both sites must go through the helper, or one of them stays loud."""
    import inspect

    source = inspect.getsource(resource_cleanup)
    body = source[source.index("def _cleanup("):]
    assert 'LOG.info("%s", result.summary())' not in body
    assert 'LOG.info("before %s: %s"' not in body
