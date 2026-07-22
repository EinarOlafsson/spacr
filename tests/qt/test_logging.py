"""Tests for spacr.qt.logging_util — the real logger.

We isolate the tests from the on-disk log file by monkey-patching
log_path() to point inside tmp_path, and by resetting _INITIALISED
between runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch, tmp_path):
    """Redirect the log file into tmp_path and force re-init per test."""
    from spacr.qt import logging_util
    monkeypatch.setattr(
        logging_util, "log_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        logging_util, "log_path",
        lambda: tmp_path / "spacr-qt.log",
    )
    # Force re-init on next setup_logging call
    monkeypatch.setattr(logging_util, "_INITIALISED", False)
    monkeypatch.setattr(logging_util, "_SIGNAL_HANDLER", None)
    # Remove any handlers that a previous test attached
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)


def test_setup_logging_creates_file_and_signal_handler(qt_theme_applied):
    from spacr.qt.logging_util import (
        setup_logging, log_path, get_signal_handler,
    )
    setup_logging()

    root = logging.getLogger()
    handler_types = {type(h).__name__ for h in root.handlers}
    assert "RotatingFileHandler" in handler_types
    assert "QtLogHandler" in handler_types

    # File was created
    assert log_path().exists()

    # Handler is a singleton
    assert get_signal_handler() is get_signal_handler()


def test_log_records_emit_qt_signal(qtbot, qt_theme_applied):
    from spacr.qt.logging_util import setup_logging, get_signal_handler
    setup_logging()

    handler = get_signal_handler()
    with qtbot.waitSignal(handler.record_ready, timeout=1000) as blocker:
        logging.getLogger("spacr.qt.tests").info("hello there")
    text, level = blocker.args
    assert "hello there" in text
    assert level == logging.INFO


def test_console_panel_receives_log_records(qtbot, qt_theme_applied):
    """A ConsolePanel wired to the QtLogHandler must render records
    into its stdout stream (INFO) or its error stream (WARNING+)."""
    from spacr.qt.logging_util import setup_logging
    from spacr.qt.widgets.console_panel import ConsolePanel, _StdoutBlock

    setup_logging()
    panel = ConsolePanel()
    qtbot.addWidget(panel)

    log = logging.getLogger("spacr.qt.tests.console")
    log.info("routine info")
    log.warning("something odd")

    # Give Qt a beat to fire the queued signal
    for _ in range(5):
        qtbot.wait(20)

    text = ""
    for i in range(panel._entries.count() - 1):
        w = panel._entries.itemAt(i).widget()
        if isinstance(w, _StdoutBlock):
            text += w.text()
    assert "routine info" in text
    assert "something odd" in text


def test_quiet_loggers_are_muted(qt_theme_applied):
    from spacr.qt.logging_util import setup_logging
    setup_logging()
    # PIL is one of the quieted namespaces — verify its level is
    # WARNING or stricter, so an INFO record wouldn't propagate.
    assert logging.getLogger("PIL").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("matplotlib").getEffectiveLevel() >= logging.WARNING


def test_setup_logging_is_idempotent(qt_theme_applied):
    from spacr.qt.logging_util import setup_logging
    setup_logging()
    count_before = len(logging.getLogger().handlers)
    setup_logging()
    count_after = len(logging.getLogger().handlers)
    assert count_before == count_after
