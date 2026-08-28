"""Safe mode is a way IN when a stored preference is what breaks the start."""
from __future__ import annotations

import os

import pytest

from spacr.qt import preferences


@pytest.fixture
def a_clean_process(monkeypatch):
    """Safe mode is process-local; never let it leak into another test."""
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    monkeypatch.setattr(preferences, "_SAFE_OVERRIDES", {})
    yield
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)


def test_a_stored_value_that_breaks_the_start_is_not_read(
        a_clean_process, monkeypatch):
    """The whole point: the poisoned value is never consulted."""
    written = {}

    class _Poisoned:
        def value(self, key, default=None, type=None):
            raise AssertionError(
                f"safe mode read the stored value for {key!r}")

        def setValue(self, key, value):
            written[key] = value

        def remove(self, key):
            written.pop(key, None)

        def sync(self):
            written["synced"] = True

    monkeypatch.setattr(preferences, "QSettings",
                        lambda *a, **k: _Poisoned())
    preferences.enable_safe_mode()

    # An ordinary getter answers, without touching the store that raises.
    assert preferences.get_font_scale() == preferences.DEFAULT_FONT_SCALE


def test_saving_reaches_the_real_store(a_clean_process, monkeypatch):
    """Reads are shadowed; WRITES are not, or safe mode fixes nothing."""
    written = {}
    synced = []

    class _Real:
        def value(self, key, default=None, type=None):
            raise AssertionError("reads must not reach the real store")

        def setValue(self, key, value):
            written[key] = value

        def remove(self, key):
            written.pop(key, None)

        def sync(self):
            synced.append(True)

    monkeypatch.setattr(preferences, "QSettings", lambda *a, **k: _Real())
    preferences.enable_safe_mode()

    preferences.set_ambient_enabled(True)

    assert written, "the value the user re-saved never reached the store"
    assert synced, "a re-saved value was left unflushed"
    assert written[preferences._KEY_AMBIENT_ENABLED] is True


def test_the_backdrop_is_forced_off_not_merely_defaulted(
        a_clean_process, monkeypatch):
    """Defaults are not safe by themselves: ambient defaults to ON."""
    class _Empty:
        def value(self, key, default=None, type=None):
            return default

        def setValue(self, key, value):
            pass

        def remove(self, key):
            pass

        def sync(self):
            pass

    monkeypatch.setattr(preferences, "QSettings", lambda *a, **k: _Empty())

    # Without safe mode the default really is on -- so forcing it matters.
    assert preferences.get_ambient_enabled() is True

    preferences.enable_safe_mode()
    assert preferences.get_ambient_enabled() is False
    assert preferences.get_verbose_logging() is False
    assert preferences.get_preload_policy() == "on_demand"


def test_safe_mode_refuses_a_gl_canvas(monkeypatch):
    """The crash log points at the GL path, so safe mode must not build one."""
    from spacr.qt.widgets import fractal_travel

    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("DISPLAY", ":0")
    assert fractal_travel.platform_can_do_opengl() is True

    monkeypatch.setenv("SPACR_NO_GL", "1")
    assert fractal_travel.platform_can_do_opengl() is False


def test_the_launcher_disarms_gl_and_timing_before_qt(monkeypatch):
    """`safespacr` sets the environment before anything reads it."""
    from spacr.qt import safespacr

    monkeypatch.setenv("SPACR_TIMING", "1")
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    seen = {}

    def _fake_run(argv):
        seen["timing"] = os.environ.get("SPACR_TIMING")
        seen["no_gl"] = os.environ.get("SPACR_NO_GL")
        seen["safe"] = preferences.in_safe_mode()
        return 0

    import spacr.qt as qt_pkg
    monkeypatch.setattr(qt_pkg, "run", _fake_run)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)

    assert safespacr.main([]) == 0
    assert seen["timing"] is None, "timing instrumentation survived safe mode"
    assert seen["no_gl"] == "1"
    assert seen["safe"] is True
