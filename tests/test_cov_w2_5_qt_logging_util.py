"""The Qt log sink never takes the app down with it.

``QtLogHandler.emit`` runs inside every ``logging`` call in the process, so
its two guards are load-bearing: it must stay silent while a console panel is
mid-write (re-entering one is a documented segfault), and a failure to format
or emit must route to ``handleError`` instead of propagating out of the
logging call that produced it.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from spacr.qt import logging_util as lu


def _record(message="hello", level=logging.INFO):
    return logging.LogRecord("spacr.test", level, __file__, 1, message,
                             None, None)


def test_the_path_shims_agree_with_the_package_logger():
    """``log_dir``/``log_path`` are aliases, not a second opinion."""
    from spacr import logging_util as package

    assert lu.log_dir() == package.log_dir()
    assert lu.log_path() == package.log_path()
    assert lu.log_path().parent == lu.log_dir()


def test_the_shared_handler_is_shared():
    """``get_signal_handler`` hands out one instance for the process."""
    first = lu.get_signal_handler()

    assert first is lu.get_signal_handler()
    assert isinstance(first, lu.QtLogHandler)


def test_get_logger_names_a_child_under_the_qt_logger():
    """The convenience wrapper returns the named logger."""
    assert lu.get_logger().name == "spacr.qt"
    assert lu.get_logger("spacr.qt.console").name == "spacr.qt.console"


def test_a_record_arrives_on_the_signal_with_its_level():
    """The formatted line and the numeric level both reach the panel."""
    handler = lu.QtLogHandler(level=logging.DEBUG)
    seen = []
    handler.record_ready.connect(lambda text, level: seen.append((text, level)))

    handler.emit(_record("a thing happened", logging.WARNING))

    assert len(seen) == 1
    text, level = seen[0]
    assert level == logging.WARNING
    assert text.endswith("\n")
    assert "a thing happened" in text
    assert "[WARNING]" in text
    assert "spacr.test" in text


def test_nothing_is_emitted_while_a_console_panel_is_writing(monkeypatch):
    """The re-entrancy latch drops the record instead of re-entering."""
    from spacr.qt import verbose_logger

    handler = lu.QtLogHandler(level=logging.DEBUG)
    seen = []
    handler.record_ready.connect(lambda text, level: seen.append(text))
    monkeypatch.setattr(verbose_logger, "console_write_in_progress",
                        lambda: True)

    handler.emit(_record("dropped on the floor"))

    assert seen == []


def test_a_missing_latch_does_not_stop_the_record(monkeypatch):
    """If the latch cannot be consulted the record still gets through."""
    import builtins

    handler = lu.QtLogHandler(level=logging.DEBUG)
    seen = []
    handler.record_ready.connect(lambda text, level: seen.append(text))
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if "verbose_logger" in name or "verbose_logger" in (fromlist or ()):
            raise ImportError("no latch here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)

    handler.emit(_record("still delivered"))

    monkeypatch.undo()
    assert len(seen) == 1
    assert "still delivered" in seen[0]


def test_a_formatting_failure_is_handled_not_raised(monkeypatch):
    """A broken formatter routes to ``handleError`` and never propagates."""
    handler = lu.QtLogHandler(level=logging.DEBUG)
    handled = []
    monkeypatch.setattr(type(handler), "handleError",
                        lambda self, record: handled.append(record))

    class Exploding(logging.Formatter):
        def format(self, record):
            raise ValueError("cannot format this")

    handler.setFormatter(Exploding())
    record = _record("never formatted")

    handler.emit(record)                       # must not raise

    assert handled == [record]


def test_setup_is_idempotent_and_installs_both_sinks(monkeypatch, tmp_path):
    """A second ``setup_logging`` does not add a second handler."""
    monkeypatch.setattr(lu, "_INITIALISED", False)
    monkeypatch.setattr(lu, "log_path", lambda: tmp_path / "spacr.log")
    installed = []
    monkeypatch.setattr(lu, "_package_setup_logging",
                        lambda **kwargs: installed.append(kwargs))
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        lu.setup_logging(level=logging.DEBUG, console_level=logging.ERROR)
        after_first = list(root.handlers)
        lu.setup_logging()
        after_second = list(root.handlers)
    finally:
        root.handlers[:] = before

    assert installed == [{"level": logging.DEBUG,
                          "log_file": tmp_path / "spacr.log"}]
    assert lu.get_signal_handler() in after_first
    assert after_second == after_first
    assert lu.get_signal_handler().level == logging.ERROR
