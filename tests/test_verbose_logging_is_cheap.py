"""Verbose logging must not trace the animation, or it makes spaCR unusable."""
from __future__ import annotations

import logging
import time

from spacr import logging_util


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


def test_verbose_is_on_by_default():
    """Asked for 2026-08-28 — and only defensible now that it is cheap."""
    from spacr.qt import preferences

    assert preferences.DEFAULT_VERBOSE_LOGGING is True

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
        assert preferences.get_verbose_logging() is True
    finally:
        preferences.QSettings = real
