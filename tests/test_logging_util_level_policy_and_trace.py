"""Table-driven tests of ``spacr.logging_util``: the level policy and the trace.

``apply_level_policy`` is what the Preferences switches call: it decides which
levels reach the log files and the in-app console, clamps the console to the
files, and moves the ``spacr`` and root thresholds so a switch is not vetoed
before any handler filter runs.  ``_trace_profile`` is the opt-in function
trace's profile hook; it must log only spaCR's own Python calls and returns
and must never raise into the code it observes.

The module owns process-wide logging state, so every test runs inside a
sandbox fixture that puts the handlers, levels, module globals and the profile
hook back afterwards.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import types

import pytest

from spacr import logging_util as lu

D, I, W, E = logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR


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
        "thread_profile": threading.getprofile(),
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
    threading.setprofile(saved["thread_profile"])


# id, file switches, console switches, live level files?, Qt console importable?,
# -> applied files, applied console, spacr level, root level
POLICY_CASES = [
    ("live level files re-gated", {W, E}, {E}, True, True,
     frozenset({W, E}), frozenset({E}), W, W),
    ("console clamped to the files", {E}, {D, E}, False, True,
     frozenset({E}), frozenset({E}), E, E),
    ("no Qt console still applies the files", {I, E}, {E}, False, False,
     frozenset({I, E}), frozenset({E}), I, W),
    ("everything off pins spacr at CRITICAL", set(), set(), False, True,
     frozenset(), frozenset(), logging.CRITICAL, logging.CRITICAL),
]


def test_apply_level_policy_reports_and_applies_exactly_the_switches(
        subtests, tmp_path, monkeypatch, logging_sandbox):
    """The applied (files, console) pair, the spacr/root levels and the live level-file gates follow the switches."""
    for (case_id, files_in, console_in, live, qt_ok,
         files_out, console_out, spacr_level, root_level) in POLICY_CASES:
        with subtests.test(msg=case_id):
            lu._LEVEL_HANDLERS.clear()
            lu._LOG_PATH = (tmp_path / case_id.replace(" ", "_") / "spacr.log"
                            if live else None)
            if live:
                lu._LOG_PATH.parent.mkdir()
            with monkeypatch.context() as patch:
                if not qt_ok:
                    patch.setitem(sys.modules, "spacr.qt.verbose_logger", None)
                files, console = lu.apply_level_policy(files_in, console_in)
            assert (files, console) == (files_out, console_out)
            assert logging.getLogger("spacr").level == spacr_level
            assert logging.getLogger().level == root_level
            if live:
                # Every level keeps a handler; an off level passes nothing.
                assert set(lu._LEVEL_HANDLERS) == set(lu.LEVELS)
                for level, handler in lu._LEVEL_HANDLERS.items():
                    gate, = [f for f in handler.filters
                             if isinstance(f, lu.LevelSetFilter)]
                    assert gate.levels == ({level} if level in files_out
                                           else set())
            else:
                assert lu._LEVEL_HANDLERS == {}


def _spacr_frame(name="run", qualname="Runner.run", filename=None,
                 module="spacr.core"):
    """A stand-in for the frame CPython hands a profile hook."""
    code = types.SimpleNamespace(
        co_name=name, co_qualname=qualname,
        co_filename=filename or os.path.join(
            os.path.dirname(lu.__file__), "core.py"))
    return types.SimpleNamespace(code=code, f_code=code,
                                 f_globals={"__name__": module})


class _HostileGlobals(dict):
    """Globals the hook cannot read: ``get`` raises."""

    def get(self, *args, **kwargs):
        raise RuntimeError("no globals here")


def _hostile_frame():
    frame = _spacr_frame()
    frame.f_globals = _HostileGlobals()
    return frame


# id, [(frame factory, event)], hook already busy?, expected spacr.trace lines
TRACE_CASES = [
    ("call then return, opposite arrows",
     [(_spacr_frame, "call"), (_spacr_frame, "return")], False,
     ["→ spacr.core.Runner.run", "← spacr.core.Runner.run"]),
    ("a method without a qualname falls back to its name",
     [(lambda: _without_qualname(_spacr_frame(name="press")), "call")], False,
     ["→ spacr.core.press"]),
    ("non call/return events are ignored",
     [(_spacr_frame, ev) for ev in ("line", "c_call", "c_return", "exception")],
     False, []),
    ("Qt event delivery is never traced",
     [(lambda n=n: _spacr_frame(name=n), "call")
      for n in ("paintEvent", "eventFilter", "sizeHint")], False, []),
    ("code outside spacr is not traced",
     [(lambda: _spacr_frame(filename="/usr/lib/python3/json.py"), "call")],
     False, []),
    ("the logger's own file is not traced",
     [(lambda: _spacr_frame(filename=lu._TRACE_THIS_FILE), "call")], False, []),
    ("the hook does not re-enter itself", [(_spacr_frame, "call")], True, []),
    ("an unreadable frame never raises", [(_hostile_frame, "call")], False, []),
]


def _without_qualname(frame):
    del frame.f_code.co_qualname
    return frame


def test_trace_profile_logs_only_spacr_calls_and_returns(subtests, caplog):
    """Exactly the spaCR call/return lines are logged; every skip rule logs nothing, never raises and releases the guard."""
    caplog.set_level(logging.DEBUG, logger="spacr.trace")
    for case_id, events, busy, expected in TRACE_CASES:
        with subtests.test(msg=case_id):
            caplog.clear()
            lu._TRACE_STATE.busy = busy
            try:
                answers = [lu._trace_profile(make(), event, None)
                           for make, event in events]
            finally:
                still_busy = getattr(lu._TRACE_STATE, "busy", False)
                lu._TRACE_STATE.busy = False
            assert answers == [None] * len(events)
            assert [r.getMessage() for r in caplog.records
                    if r.name == "spacr.trace"] == expected
            # The guard is left exactly as it was found.
            assert still_busy is busy


def test_trace_profile_is_silent_when_spacr_trace_is_above_debug(caplog):
    """With spacr.trace at INFO the hook formats nothing for a traceable frame."""
    caplog.set_level(logging.INFO, logger="spacr.trace")
    assert lu._trace_profile(_spacr_frame(), "call", None) is None
    assert [r for r in caplog.records if r.name == "spacr.trace"] == []


def test_without_setprofile_all_threads_the_trace_goes_on_and_off(
        monkeypatch, logging_sandbox):
    """Pre-3.12: the calling and future threads get the hook, and both previous hooks come back, idempotently."""
    monkeypatch.delattr(threading, "setprofile_all_threads", raising=False)
    lu._TRACE_ENABLED = False
    before_sys, before_thread = sys.getprofile(), threading.getprofile()
    lu.enable_function_trace()
    try:
        assert lu.function_trace_enabled() is True
        assert sys.getprofile() is lu._trace_profile
        assert threading.getprofile() is lu._trace_profile
    finally:
        lu.disable_function_trace()
    assert sys.getprofile() is before_sys
    assert threading.getprofile() is before_thread
    assert lu.function_trace_enabled() is False
    assert lu.disable_function_trace() is None
    assert lu.function_trace_enabled() is False


def test_a_paced_log_file_writes_a_burst_out_within_one_interval(tmp_path):
    """The first record is written at once; one right behind it waits in the buffer until the late-flush timer writes it."""
    import logging.handlers

    path = tmp_path / "paced.log"
    handler = lu._quicken(logging.handlers.RotatingFileHandler(
        str(path), maxBytes=1 << 20, backupCount=1, encoding="utf-8"))
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        handler.handle(logging.makeLogRecord({"msg": "first", "levelno": I}))
        assert path.read_text(encoding="utf-8") == "first\n"
        handler.handle(logging.makeLogRecord({"msg": "second", "levelno": I}))
        timer = handler._spacr_late_flush
        assert isinstance(timer, threading.Timer)
        timer.join(timeout=10)
        assert path.read_text(encoding="utf-8") == "first\nsecond\n"
        assert handler._spacr_late_flush is None
    finally:
        handler.close()
