"""Verbose logging must not trace the animation, or it makes spaCR unusable."""
from __future__ import annotations

import logging
import sys
import threading
import time

from spacr import logging_util


def test_gui_verbose_never_installs_an_interpreter_profile_hook(caplog):
    """Keep the 2-second startup path instead of the measured 60-second one.

    Hook identity is a deterministic performance contract: timing a quiet CI
    host is noisy, while any installed Python profile callback necessarily
    intercepts every function call in every thread.
    """
    from spacr.qt import verbose_logger

    logging_util.disable_function_trace()
    verbose_logger.apply_verbose_logging(False)
    sys_profile = sys.getprofile()
    get_thread_profile = getattr(threading, "getprofile", lambda: None)
    thread_profile = get_thread_profile()
    try:
        verbose_logger.apply_verbose_logging(True)
        assert sys.getprofile() is sys_profile
        assert get_thread_profile() is thread_profile
        assert not logging_util.function_trace_enabled()

        with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
            verbose_logger.log_button_press("Run", {"screen": "home"})
        assert any("[button:Run]" in record.getMessage()
                   and "home" in record.getMessage()
                   for record in caplog.records)
    finally:
        verbose_logger.apply_verbose_logging(False)
        # If this regression fails, do not leak its hook into later tests.
        logging_util.disable_function_trace()


def test_the_animation_modules_are_never_traced():
    """The paint path is where the cost was: 5 MB of log a minute."""
    assert "spacr.qt.widgets.ambient" in logging_util._TRACE_SKIP_MODULES
    assert "spacr.qt.widgets.fractal_travel" in logging_util._TRACE_SKIP_MODULES
    assert "spacr.qt.widgets.fractal_cascade" in logging_util._TRACE_SKIP_MODULES


def test_a_paint_helper_emits_nothing_while_verbose_is_on(caplog):
    """Drive a frame's worth of the real helper and assert the log is quiet."""
    from spacr.qt.widgets import ambient

    logging_util.enable_function_trace()
    try:
        with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
            from PySide6.QtGui import QColor
            for _ in range(50):
                ambient._with_alpha(QColor(10, 20, 30), 0.5)
    finally:
        logging_util.disable_function_trace()

    traced = [r for r in caplog.records
              if r.name == "spacr.trace" and "ambient" in r.getMessage()]
    assert traced == [], (
        f"the paint path was traced {len(traced)} times in 50 calls")


def test_something_outside_the_animation_is_still_traced(caplog):
    """The exclusion is narrow: verbose still traces ordinary spaCR code."""
    from spacr import version

    logging_util.enable_function_trace()
    try:
        with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
            version.get_version()
    finally:
        logging_util.disable_function_trace()

    assert any(r.name == "spacr.trace" for r in caplog.records), (
        "excluding the animation silenced the whole tracer")


def test_the_hook_is_cheap_when_verbose_is_off():
    """The hook runs on every call in the process; deciding must be quick."""
    logger = logging.getLogger("spacr.trace")
    previous = logger.level
    logger.setLevel(logging.INFO)          # verbose off
    logging_util.enable_function_trace()
    try:
        def _work():
            total = 0
            for i in range(20000):
                total += i
            return total

        start = time.perf_counter()
        _work()
        traced = time.perf_counter() - start
    finally:
        logging_util.disable_function_trace()
        logger.setLevel(previous)

    start = time.perf_counter()
    total = 0
    for i in range(20000):
        total += i
    plain = time.perf_counter() - start

    # A generous ceiling: the point is that the hook returns after one level
    # check rather than doing a realpath syscall per event.
    assert traced < max(plain * 200, 0.5), (
        f"tracing cost {traced:.4f}s against {plain:.4f}s untraced")


def test_verbose_is_off_by_default():
    """Measured: 3.05 s to Home with it off, 65.28 s with it on.

    It was briefly the default -- a trail that exists before the bug is
    genuinely worth having. Twenty times the startup is not something to
    give a user who did not ask for it.
    """
    from spacr.qt import preferences

    assert preferences.DEFAULT_VERBOSE_LOGGING is False

    class _Empty:
        def value(self, key, default=None, type=None):
            return default

        def setValue(self, key, value):
            pass

        def sync(self):
            pass

    real = preferences.QSettings
    preferences.QSettings = lambda *a, **k: _Empty()
    try:
        assert preferences.get_verbose_logging() is False
    finally:
        preferences.QSettings = real


def test_the_hook_survives_interpreter_shutdown():
    """At teardown the module globals are None while finalisers still run."""
    import types

    frame = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_name="close", co_filename=__file__,
                                     co_qualname="ZipFile.close"),
        f_globals={"__name__": "zipfile"})

    saved = logging_util._TRACE_SKIP_NAMES
    logging_util._TRACE_SKIP_NAMES = None
    try:
        # Must not raise: Python prints "Exception ignored in" for every
        # finaliser that trips over a tracing aid.
        assert logging_util._trace_profile(frame, "call", None) is None
    finally:
        logging_util._TRACE_SKIP_NAMES = saved


def test_verbose_actually_writes_debug_to_the_file(monkeypatch):
    """All of the cost and none of the trail was the defect (297).

    The hook was installed and `spacr.trace` set to DEBUG, then the file
    level policy dropped DEBUG at the handler -- so a record was built for
    every call in the process and thrown away.
    """
    import logging as _logging

    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_verbose_logging", lambda: True)
    with_verbose = preferences.get_log_file_levels()
    assert _logging.DEBUG in with_verbose

    monkeypatch.setattr(preferences, "get_verbose_logging", lambda: False)
    without = preferences.get_log_file_levels()
    assert _logging.DEBUG not in without, (
        "DEBUG survived verbose being turned off")


def test_verbose_does_not_rewrite_the_users_level_choice(monkeypatch):
    """DEBUG is added on read, never stored over what the user chose."""
    from spacr.qt import preferences

    written = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return written.get(key, default)

        def setValue(self, key, value):
            written[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(preferences, "_settings", lambda: _Mem())
    monkeypatch.setattr(preferences, "get_verbose_logging", lambda: True)

    preferences.get_log_file_levels()
    assert preferences._KEY_LOG_FILE_LEVELS not in written, (
        "reading the levels wrote to the user's preferences")


def test_the_tracer_survives_logging_being_torn_down():
    """`logging`'s own globals go too, so `getLogger` fails in a finaliser."""
    import types

    frame = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_name="_removeHandlerRef",
                                     co_filename=__file__,
                                     co_qualname="_removeHandlerRef"),
        f_globals={"__name__": "logging"})

    saved = logging_util.logging
    class _Dead:
        DEBUG = 10

        @staticmethod
        def getLogger(_name=None):
            raise TypeError("'NoneType' object is not callable")

    logging_util.logging = _Dead
    try:
        # Must not raise: Python prints "Exception ignored in" otherwise,
        # once per finaliser, for every process that enabled verbose.
        assert logging_util._trace_profile(frame, "call", None) is None
    finally:
        logging_util.logging = saved


def test_a_trace_line_is_not_mostly_prefix():
    """297: the prefix was longer than the message it introduced.

    On a trace record the level is always DEBUG, the logger is always
    `spacr.trace`, and the file and line are this module's own tracer rather
    than the traced code -- so three of the ordinary prefix's fields say
    nothing here, and one of them misleads.
    """
    import logging as _logging

    from spacr.logging_util import FILE_FORMAT, _CompactTraceFormat

    formatter = _CompactTraceFormat(FILE_FORMAT)

    traced = _logging.LogRecord(
        "spacr.trace", _logging.DEBUG, __file__, 1,
        "%s %s", ("→", "spacr.version.get_version"), None)
    line = formatter.format(traced)
    assert "DEBUG" not in line
    assert "spacr.trace" not in line
    assert "logging_util" not in line
    assert "spacr.version.get_version" in line
    assert len(line) < 60, f"a trace line is still {len(line)} characters"

    # Everything else keeps the full prefix, which is where it earns its keep.
    ordinary = _logging.LogRecord(
        "spacr.core", _logging.INFO, __file__, 42, "started", (), None)
    assert "INFO" in formatter.format(ordinary)
    assert "spacr.core" in formatter.format(ordinary)


def test_the_tooltip_says_what_verbose_costs():
    """A default nobody can weigh is a default nobody can turn off knowingly."""
    import inspect

    from spacr.qt import preferences

    source = inspect.getsource(preferences.PreferencesDialog)
    assert "156 bytes" in source, "the per-call cost is not stated"
    assert "65 seconds" in source, "the startup cost is not stated"
    assert "never traced" in source, "the paint-path exclusion is not stated"
