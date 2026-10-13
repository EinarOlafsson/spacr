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
import os
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


def test_the_preference_switches_reach_the_live_level_files(tmp_path,
                                                            logging_sandbox):
    lu._LEVEL_HANDLERS.clear()
    lu._LOG_PATH = tmp_path / "spacr.log"
    files, console = lu.apply_level_policy({logging.WARNING, logging.ERROR},
                                           {logging.ERROR})
    assert files == frozenset({logging.WARNING, logging.ERROR})
    assert console == frozenset({logging.ERROR})
    # spacr.* carries its own threshold, which would veto the switches
    # before any handler filter ran.
    assert logging.getLogger("spacr").level == logging.WARNING
    assert set(lu._LEVEL_HANDLERS) == set(lu.LEVELS)


def test_a_console_level_the_file_does_not_keep_is_clamped_away(
        tmp_path, logging_sandbox):
    """The console cannot show what the file was never given."""
    lu._LOG_PATH = None
    files, console = lu.apply_level_policy({logging.ERROR},
                                           {logging.DEBUG, logging.ERROR})
    assert console == frozenset({logging.ERROR})
    assert files == frozenset({logging.ERROR})


def test_a_build_without_the_qt_console_still_applies_the_file_switches(
        monkeypatch, logging_sandbox):
    """Qt is optional; the CLI has no console panel and must not be refused
    a logging policy because of it."""
    monkeypatch.setitem(sys.modules, "spacr.qt.verbose_logger", None)
    lu._LOG_PATH = None
    files, console = lu.apply_level_policy({logging.INFO, logging.ERROR},
                                           {logging.ERROR})
    assert files == frozenset({logging.INFO, logging.ERROR})
    assert console == frozenset({logging.ERROR})


def test_switching_everything_off_pins_spacr_at_critical(logging_sandbox):
    lu._LOG_PATH = None
    files, console = lu.apply_level_policy(set(), set())
    assert files == frozenset() and console == frozenset()
    assert logging.getLogger("spacr").level == logging.CRITICAL


# --------------------------------------------------------------------------
# the function trace
# --------------------------------------------------------------------------

def _spacr_frame(name="run", qualname="Runner.run",
                 filename=None, module="spacr.core"):
    """A stand-in for what CPython hands a profile hook."""
    code = types.SimpleNamespace(
        co_name=name, co_qualname=qualname,
        co_filename=filename or os.path.join(
            os.path.dirname(lu.__file__), "core.py"))
    return types.SimpleNamespace(code=code, f_code=code,
                                 f_globals={"__name__": module})


def test_a_traced_call_and_return_are_logged_in_opposite_directions(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.trace")
    lu._trace_profile(_spacr_frame(), "call", None)
    lu._trace_profile(_spacr_frame(), "return", None)
    said = [record.getMessage() for record in caplog.records]
    assert said == ["→ spacr.core.Runner.run", "← spacr.core.Runner.run"]


def test_only_call_and_return_events_are_traced(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.trace")
    for event in ("line", "c_call", "c_return", "exception"):
        lu._trace_profile(_spacr_frame(), event, None)
    assert caplog.records == []


def test_qt_event_delivery_is_never_traced(caplog):
    """Tracing those feeds the GUI console from inside event delivery, and
    the console's repaint is another event."""
    caplog.set_level(logging.DEBUG, logger="spacr.trace")
    for name in ("paintEvent", "eventFilter", "sizeHint"):
        lu._trace_profile(_spacr_frame(name=name), "call", None)
    assert caplog.records == []


def test_code_outside_spacr_and_the_logger_itself_are_not_traced(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.trace")
    lu._trace_profile(_spacr_frame(filename="/usr/lib/python3/json.py"),
                      "call", None)
    lu._trace_profile(_spacr_frame(filename=lu._TRACE_THIS_FILE), "call", None)
    assert caplog.records == []


def test_the_trace_does_not_re_enter_itself(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.trace")
    lu._TRACE_STATE.busy = True
    try:
        lu._trace_profile(_spacr_frame(), "call", None)
    finally:
        lu._TRACE_STATE.busy = False
    assert caplog.records == []


def test_a_tracing_aid_never_alters_the_code_it_observes():
    """A frame the hook cannot read must not raise into the traced call, and
    the re-entry guard must be released either way."""
    class Hostile(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("no globals here")

    frame = _spacr_frame()
    frame.f_globals = Hostile()
    lu._trace_profile(frame, "call", None)
    assert getattr(lu._TRACE_STATE, "busy", False) is False


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


def test_on_a_runtime_without_all_threads_the_hook_still_goes_on_and_off(
        monkeypatch, logging_sandbox):
    """Pre-3.12 has no ``setprofile_all_threads``; the calling thread and
    future threads are still covered, and the previous hook comes back."""
    monkeypatch.delattr(threading, "setprofile_all_threads", raising=False)
    lu._TRACE_ENABLED = False
    before = sys.getprofile()
    lu.enable_function_trace()
    try:
        assert sys.getprofile() is lu._trace_profile
    finally:
        lu.disable_function_trace()
    assert sys.getprofile() is before
    assert lu.function_trace_enabled() is False
    assert lu.disable_function_trace() is None      # idempotent


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
