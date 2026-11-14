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
