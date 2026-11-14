"""Diagnostics that never become the reason a run cannot start.

Logging is infrastructure: a read-only home directory, a level log that
cannot be opened, or a Qt console that is not installed must each cost the
user that one facility and nothing else. These drive the real handlers into
a temporary directory and put the module's globals back afterwards, because
the module owns process-wide logging state.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import types

import pytest

from spacr import logging_util as lu


@pytest.fixture
def logging_sandbox():
    """Restore every piece of process-wide logging state a test may move."""
    root = logging.getLogger()
    spacr_logger = logging.getLogger("spacr")
    saved = {
        "handlers": list(root.handlers),
        "root_level": root.level,
        "spacr_level": spacr_logger.level,
        "initialised": lu._INITIALISED,
        "session_level": lu._SESSION_LEVEL,
        "log_path": lu._LOG_PATH,
        "file_filter": lu._FILE_FILTER,
        "level_handlers": dict(lu._LEVEL_HANDLERS),
        "trace_enabled": lu._TRACE_ENABLED,
        "sys_profile": sys.getprofile(),
    }
    yield
    for handler in list(root.handlers):
        if handler not in saved["handlers"]:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    root.handlers[:] = saved["handlers"]
    root.setLevel(saved["root_level"])
    spacr_logger.setLevel(saved["spacr_level"])
    lu._INITIALISED = saved["initialised"]
    lu._SESSION_LEVEL = saved["session_level"]
    lu._LOG_PATH = saved["log_path"]
    lu._FILE_FILTER = saved["file_filter"]
    lu._LEVEL_HANDLERS.clear()
    lu._LEVEL_HANDLERS.update(saved["level_handlers"])
    lu._TRACE_ENABLED = saved["trace_enabled"]
    sys.setprofile(saved["sys_profile"])


# --------------------------------------------------------------------------
# a log file that cannot be opened
# --------------------------------------------------------------------------

def test_a_log_file_that_cannot_be_opened_does_not_stop_the_analysis(
        tmp_path, monkeypatch, capsys, logging_sandbox):
    """A read-only home directory must not prevent analysis from starting.
    The failure goes to stderr instead of being swallowed."""
    def _refuse(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(logging.handlers, "RotatingFileHandler", _refuse)
    lu._INITIALISED = False
    resolved = lu.setup_logging(logging.INFO, log_file=tmp_path / "spacr.log",
                                quiet=())
    assert resolved == tmp_path / "spacr.log"
    assert "could not open diagnostic log" in capsys.readouterr().err
    # ... and the console handler is attached instead, so records still land
    # somewhere a user can see them.
    assert any(isinstance(h, logging.StreamHandler)
               for h in logging.getLogger().handlers)


def test_a_level_log_that_cannot_be_opened_costs_that_level_only(
        tmp_path, monkeypatch, capsys, logging_sandbox):
    real = logging.handlers.RotatingFileHandler

    def _refuse_only_errors(filename, *args, **kwargs):
        if str(filename).endswith("spacr-error.log"):
            raise OSError(13, "Permission denied")
        return real(filename, *args, **kwargs)

    monkeypatch.setattr(logging.handlers, "RotatingFileHandler",
                        _refuse_only_errors)
    lu._LEVEL_HANDLERS.clear()
    lu._install_level_handlers(tmp_path / "spacr.log", lu.LEVELS)
    assert "spacr-error.log" in capsys.readouterr().err
    assert logging.ERROR not in lu._LEVEL_HANDLERS
    assert logging.WARNING in lu._LEVEL_HANDLERS
    assert (tmp_path / "spacr-warning.log").exists()


# --------------------------------------------------------------------------
# re-gating live handlers from the Preferences switches
# --------------------------------------------------------------------------

def test_switching_a_level_off_leaves_its_handler_filtered_to_nothing(
        tmp_path, logging_sandbox):
    """Detaching a handler from a live root logger races with any thread
    that is logging, so an off level keeps its handler and passes nothing."""
    lu._LEVEL_HANDLERS.clear()
    lu._install_level_handlers(tmp_path / "spacr.log", lu.LEVELS)
    lu._install_level_handlers(tmp_path / "spacr.log", {logging.ERROR})
    assert set(lu._LEVEL_HANDLERS) == set(lu.LEVELS)
    for level, handler in lu._LEVEL_HANDLERS.items():
        gate, = [f for f in handler.filters
                 if isinstance(f, lu.LevelSetFilter)]
        assert gate.levels == ({logging.ERROR} if level == logging.ERROR
                               else set())


# --------------------------------------------------------------------------
# the function trace
# --------------------------------------------------------------------------

def test_asking_for_the_trace_twice_installs_it_once(monkeypatch,
                                                     logging_sandbox):
    calls = []
    monkeypatch.setattr(threading, "setprofile_all_threads",
                        lambda hook: calls.append(hook), raising=False)
    lu._TRACE_ENABLED = False
    try:
        lu.enable_function_trace()
        assert lu.function_trace_enabled() is True
        lu.enable_function_trace()
        assert calls == [lu._trace_profile]
    finally:
        lu.disable_function_trace()
    assert lu.function_trace_enabled() is False


# --------------------------------------------------------------------------
# timing a module
# --------------------------------------------------------------------------

def test_only_functions_the_module_defines_are_wrapped_for_timing():
    """Wrapping an imported name would time another module's function and
    report it under this one."""
    module = types.ModuleType("fake_spacr_module")
    module.__name__ = "fake_spacr_module"

    def mine():
        return "mine"

    mine.__module__ = "fake_spacr_module"

    def borrowed():
        return "borrowed"

    borrowed.__module__ = "somewhere.else"

    module.mine = mine
    module.borrowed = borrowed
    module._private = mine
    module.NOT_CALLABLE = 3

    assert lu.time_module(module) == 1
    assert module.borrowed is borrowed
    assert module.mine is not mine
    assert module.mine() == "mine"
    # A second pass finds nothing left to wrap.
    assert lu.time_module(module) == 0
