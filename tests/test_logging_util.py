"""Tests for spacr.logging_util — the package-scope logger.

Verifies the config primitives without depending on any Qt state:
- setup_logging is idempotent and honours SPACR_LOG_LEVEL
- get_logger returns a proper spacr.* child logger
- enable_debug / disable_debug flip levels correctly
- quiet loggers are pinned
- log_path respects an override
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    """Reset module state + swap the on-disk log to tmp_path."""
    from spacr import logging_util
    # Redirect log_dir so tests never touch ~/.spacr/logs
    monkeypatch.setattr(logging_util, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(logging_util, "_INITIALISED", False)
    monkeypatch.setattr(logging_util, "_LOG_PATH", None)
    monkeypatch.setattr(logging_util, "_SESSION_LEVEL", logging.INFO)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)


def test_setup_installs_file_handler_at_tmp_path(tmp_path):
    from spacr.logging_util import setup_logging, log_path
    path = setup_logging()
    assert Path(path) == tmp_path / "spacr.log"
    assert log_path() == tmp_path / "spacr.log"
    root = logging.getLogger()
    assert any(
        type(h).__name__ == "RotatingFileHandler"
        for h in root.handlers
    )
    # Emitting a record actually writes to the file
    logging.getLogger("spacr").info("hello world")
    assert (tmp_path / "spacr.log").exists()


def test_setup_is_idempotent(tmp_path):
    from spacr.logging_util import setup_logging
    setup_logging()
    before = len(logging.getLogger().handlers)
    setup_logging()
    after = len(logging.getLogger().handlers)
    assert before == after


def test_setup_honours_env_var(monkeypatch, tmp_path):
    from spacr import logging_util
    monkeypatch.setenv("SPACR_LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(logging_util, "_INITIALISED", False)
    logging_util.setup_logging()
    root = logging.getLogger()
    file_h = next(h for h in root.handlers
                    if type(h).__name__ == "RotatingFileHandler")
    assert file_h.level == logging.DEBUG


def test_setup_accepts_custom_log_file(tmp_path):
    from spacr.logging_util import setup_logging, log_path
    custom = tmp_path / "sub" / "custom.log"
    result = setup_logging(log_file=custom)
    assert result == custom
    assert log_path() == custom


def test_get_logger_returns_child(tmp_path):
    from spacr.logging_util import get_logger
    log = get_logger("spacr.tests.child")
    assert log.name == "spacr.tests.child"
    assert isinstance(log, logging.Logger)


def test_enable_disable_debug_toggle(tmp_path):
    from spacr.logging_util import setup_logging, enable_debug, disable_debug
    setup_logging(level=logging.INFO)
    spacr_log = logging.getLogger("spacr")
    assert spacr_log.getEffectiveLevel() == logging.INFO
    enable_debug()
    assert spacr_log.level == logging.DEBUG
    disable_debug()
    assert spacr_log.level == logging.INFO


def test_quiet_loggers_are_pinned(tmp_path):
    from spacr.logging_util import setup_logging
    setup_logging()
    # PIL is in QUIET_LOGGERS — should be at WARNING or higher
    assert (logging.getLogger("PIL")
            .getEffectiveLevel() >= logging.WARNING)
    assert (logging.getLogger("matplotlib")
            .getEffectiveLevel() >= logging.WARNING)


def test_stream_handler_optional(tmp_path):
    from spacr.logging_util import setup_logging
    setup_logging(stream=True)
    root = logging.getLogger()
    stream_handlers = [
        h for h in root.handlers if type(h).__name__ == "StreamHandler"
    ]
    assert stream_handlers, "stream=True should attach a StreamHandler"
