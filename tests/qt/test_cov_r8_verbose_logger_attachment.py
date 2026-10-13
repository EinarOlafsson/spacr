"""Where the verbose-logging handlers attach, and why only in one place.

Both `_ensure_handler` and `_ensure_file_handler` attach at spaCR's
package root and then REMOVE themselves from every descendant logger.
The reason is in the source: descendants propagate upward, so a handler
attached to both a child and its ancestor delivers one record
repeatedly as logging walks the chain.

That removal loop had never run with anything to remove. It is the arm
that stops a duplicate attachment turning every log line into several --
which, with verbose logging on, is a console that scrolls faster than
the run produces work.
"""
from __future__ import annotations

import logging

import pytest

from spacr.qt import verbose_logger as VL


@pytest.fixture(autouse=True)
def _restore_logging():
    """Put every touched logger back exactly as it was found."""
    saved = {name: list(logging.getLogger(name).handlers)
             for name in VL._ATTACHED_LOGGERS}
    levels = {name: logging.getLogger(name).level
              for name in VL._ATTACHED_LOGGERS}
    try:
        yield
    finally:
        for name in VL._ATTACHED_LOGGERS:
            logger = logging.getLogger(name)
            logger.handlers[:] = saved[name]
            logger.setLevel(levels[name])


class TestTheConsoleSink:

    def test_it_attaches_at_the_package_root(self):
        handler = VL._ensure_handler()
        assert handler in logging.getLogger(VL._SINK_LOGGER).handlers

    def test_calling_it_twice_attaches_one_handler(self):
        """Idempotent: a second call must not double every line."""
        first = VL._ensure_handler()
        second = VL._ensure_handler()
        assert first is second
        root = logging.getLogger(VL._SINK_LOGGER)
        assert root.handlers.count(first) == 1

    def test_a_handler_that_leaked_onto_a_child_is_taken_off(self):
        """THE UNCOVERED ARM.

        A descendant carrying the same handler delivers the record once
        itself and again at the root, so a line appears twice. The loop
        removes it wherever it is not the sink logger.
        """
        handler = VL._ensure_handler()
        child = logging.getLogger("spacr.qt")
        child.addHandler(handler)
        assert handler in child.handlers

        VL._ensure_handler()

        assert handler not in child.handlers, (
            "a duplicate attachment survived; every record through this "
            "logger would be delivered twice")
        assert handler in logging.getLogger(VL._SINK_LOGGER).handlers, (
            "the removal loop took the handler off the sink as well")

    def test_the_sink_logger_itself_is_never_stripped(self):
        """`name != _SINK_LOGGER` -- the one place it belongs."""
        handler = VL._ensure_handler()
        for _ in range(3):
            VL._ensure_handler()
        assert handler in logging.getLogger(VL._SINK_LOGGER).handlers


class TestTheRotatingFileHandler:

    def test_it_attaches_at_the_package_root(self):
        handler = VL._ensure_file_handler()
        assert handler in logging.getLogger(VL._SINK_LOGGER).handlers

    def test_its_removal_loop_can_never_have_anything_to_remove(self):
        """The file handler's strip loop is unreachable, unlike the sink's.

        The two functions differ in one way that decides it. The console
        sink CONTINUES past its `if _handler is None:` block, so a second
        call runs the loop against the existing handler -- which may by
        then have been attached to a child. `_ensure_file_handler`
        RETURNS EARLY once `_file_handler` is set, so its loop runs only
        on the first call, against a handler it created three lines
        above and which therefore cannot be on any child yet.

        Pinned to that early return. If it ever goes, the loop becomes
        live and wants the test the console one has.
        """
        import inspect

        source = inspect.getsource(VL._ensure_file_handler)
        assert "if _file_handler is not None:" in source
        early = source.index("if _file_handler is not None:")
        loop = source.index("for name in _ATTACHED_LOGGERS:")
        assert early < loop, (
            "the file handler no longer returns early, so its removal loop "
            "is now reachable")

        # and the second call really does return the same object untouched
        first = VL._ensure_file_handler()
        child = logging.getLogger("spacr.updater")
        child.addHandler(first)
        assert VL._ensure_file_handler() is first
        assert first in child.handlers, (
            "the early return no longer holds; this test is asserting "
            "nothing")
        child.removeHandler(first)

    def test_every_attached_logger_admits_info_records(self):
        """The levels are lowered so a quiet descendant cannot swallow
        records the sink was switched on to see."""
        VL._ensure_file_handler()
        for name in VL._ATTACHED_LOGGERS:
            logger = logging.getLogger(name)
            assert logger.level <= logging.INFO, (
                f"{name} would drop INFO records the sink wants")
