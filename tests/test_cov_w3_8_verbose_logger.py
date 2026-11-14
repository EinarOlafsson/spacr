"""Console/file log plumbing: the sink's refusals, and the trace helpers.

The module keeps process-wide singletons (``_handler``, ``_relay``,
``_console_ref``, ``_file_handler``) and attaches handlers to the real
spaCR loggers, so every test runs inside a fixture that snapshots those
globals and detaches anything it added.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class Widget:
    """A class at module scope: its methods' qualnames are ``Widget.method``.

    Defined here rather than inside a test because ``_label_for`` and
    ``_looks_bound`` both read ``__qualname__``, and a class built inside a
    function carries a ``<locals>`` prefix that is not the class name.
    """

    def press(self, which="Run"):
        return f"pressed {which}"


def free(which="Run"):
    """A plain function, whose qualname has no dot at all."""
    return f"called {which}"


@pytest.fixture()
def vlog(tmp_path, monkeypatch):
    """The module with its singletons snapshotted and its log dir redirected."""
    from spacr.qt import verbose_logger as module

    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    saved = {name: getattr(module, name) for name in
             ("_console_ref", "_handler", "_relay", "_file_handler",
              "_verbose")}
    saved_handlers = {name: list(logging.getLogger(name).handlers)
                      for name in module._ATTACHED_LOGGERS}
    saved_levels = {name: logging.getLogger(name).level
                    for name in module._ATTACHED_LOGGERS}
    try:
        yield module
    finally:
        # apply_verbose_logging(True) installs a process-wide sys.setprofile
        # hook. Leaving it on makes every later test log its own call graph
        # into spacr.trace, and it outlives the interpreter's own teardown.
        from spacr.logging_util import disable_function_trace
        disable_function_trace()
        for name, value in saved.items():
            setattr(module, name, value)
        for name, handlers in saved_handlers.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.setLevel(saved_levels[name])


def test_the_forwarder_says_nothing_when_no_console_is_registered(vlog, qapp):
    """No panel means no work, not an exception.

    No work is measurable: the record is never even formatted. Formatting
    it and throwing the line away costs the same as formatting it and
    keeping it, on every record the app logs before a console exists.
    """
    formatted = []

    class _Counting(vlog._ConsoleForwarder):
        def format(self, record):
            formatted.append(record)
            return super().format(record)

    handler = _Counting()
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "hello", (), None)

    vlog._console_ref = None
    handler.emit(record)
    vlog._console_ref = lambda: None
    handler.emit(record)

    assert formatted == []


def test_a_formatting_failure_inside_the_forwarder_is_swallowed(vlog, qapp,
                                                               monkeypatch):
    """Never let a logging failure escape into the app.

    The format was attempted and nothing reached the relay: a line that
    still went out after the formatter failed would be some half-built
    string, and a format that was never attempted would swallow every
    record whether or not the formatter was broken.
    """
    tried = []

    class _BrokenFormat(vlog._ConsoleForwarder):
        def format(self, record):
            tried.append(record)
            raise ValueError("bad format string")

    emitted = []

    class _Line:
        def emit(self, text):
            emitted.append(text)

    class _Relay:
        line = _Line()

    monkeypatch.setattr(vlog, "_ensure_relay", lambda: _Relay())

    handler = _BrokenFormat()
    target = object()
    vlog._console_ref = lambda: target
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1,
                               "hello", (), None)

    handler.emit(record)

    assert tried == [record]
    assert emitted == []


def test_the_label_is_the_qualified_name(vlog):
    """A bound method is labelled ``Class.method``; a function by its name."""
    assert vlog._label_for(Widget.press, ()) == "Widget.press"
    assert vlog._label_for(free, ()) == "free"

