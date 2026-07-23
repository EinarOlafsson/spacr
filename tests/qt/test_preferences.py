"""Tests for spacr.qt.preferences — theme + font scale + CB mode."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    """Route QSettings into a temp .ini so tests don't touch real prefs."""
    from PySide6.QtCore import QCoreApplication, QSettings
    QCoreApplication.setOrganizationName("spacr-test")
    QCoreApplication.setApplicationName("qt-prefs-test")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                        str(tmp_path))
    QSettings("spacr", "qt").clear()
    # Re-mark first-launch tour seen after the QSettings clear so the
    # autouse conftest fixture keeps its promise.
    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def test_theme_default_is_dark(qt_theme_applied):
    from spacr.qt.preferences import get_theme
    assert get_theme() == "dark"


def test_theme_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import get_theme, set_theme
    set_theme("light")
    assert get_theme() == "light"
    set_theme("system")
    assert get_theme() == "system"


def test_theme_invalid_raises(qt_theme_applied):
    from spacr.qt.preferences import set_theme
    with pytest.raises(ValueError):
        set_theme("purple")


def test_theme_recovers_from_corrupt_value(qt_theme_applied):
    from spacr.qt.preferences import get_theme
    from PySide6.QtCore import QSettings
    QSettings("spacr", "qt").setValue("prefs/theme", "garbage")
    assert get_theme() == "dark"


def test_resolve_effective_theme_dark_and_light(qt_theme_applied):
    from spacr.qt.preferences import (
        resolve_effective_theme, set_theme,
    )
    set_theme("dark")
    assert resolve_effective_theme() == "dark"
    set_theme("light")
    assert resolve_effective_theme() == "light"


def test_resolve_system_returns_valid_choice(qt_theme_applied):
    from spacr.qt.preferences import (
        resolve_effective_theme, set_theme,
    )
    set_theme("system")
    assert resolve_effective_theme() in ("dark", "light")


# ---------------------------------------------------------------------------
# Font scale
# ---------------------------------------------------------------------------

def test_font_scale_default_is_100pct(qt_theme_applied):
    from spacr.qt.preferences import get_font_scale
    assert get_font_scale() == 1.0


def test_font_scale_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import get_font_scale, set_font_scale
    set_font_scale(1.25)
    assert get_font_scale() == pytest.approx(1.25)
    set_font_scale(1.5)
    assert get_font_scale() == pytest.approx(1.5)


def test_font_scale_clamps_out_of_range(qt_theme_applied):
    from spacr.qt.preferences import (
        get_font_scale, set_font_scale,
        FONT_SCALE_MIN, FONT_SCALE_MAX,
    )
    set_font_scale(10.0)
    assert get_font_scale() == FONT_SCALE_MAX
    set_font_scale(0.01)
    assert get_font_scale() == FONT_SCALE_MIN


def test_font_scale_recovers_from_corrupt_value(qt_theme_applied):
    from spacr.qt.preferences import get_font_scale
    from PySide6.QtCore import QSettings
    QSettings("spacr", "qt").setValue("prefs/font_scale", "garbage")
    assert get_font_scale() == 1.0


# ---------------------------------------------------------------------------
# Colour-blind mode
# ---------------------------------------------------------------------------

def test_cb_mode_default_is_off(qt_theme_applied):
    from spacr.qt.preferences import get_color_blind_mode
    assert get_color_blind_mode() == "off"


def test_cb_mode_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import (
        get_color_blind_mode, set_color_blind_mode,
    )
    set_color_blind_mode("deuteranopia")
    assert get_color_blind_mode() == "deuteranopia"


def test_cb_mode_invalid_raises(qt_theme_applied):
    from spacr.qt.preferences import set_color_blind_mode
    with pytest.raises(ValueError):
        set_color_blind_mode("technicolor")


def test_categorical_palette_switches_with_cb_mode(qt_theme_applied):
    from spacr.qt.preferences import (
        color_blind_categorical_palette, set_color_blind_mode,
    )
    set_color_blind_mode("off")
    off = color_blind_categorical_palette()
    set_color_blind_mode("deuteranopia")
    on = color_blind_categorical_palette()
    assert off != on
    # Okabe-Ito starts with blue #0072B2
    assert on[0] == "#0072B2"


def test_continuous_cmap_switches_with_cb_mode(qt_theme_applied):
    from spacr.qt.preferences import (
        color_blind_continuous_cmap, set_color_blind_mode,
    )
    set_color_blind_mode("off")
    assert color_blind_continuous_cmap() == "viridis"
    set_color_blind_mode("protanopia")
    assert color_blind_continuous_cmap() == "cividis"


# ---------------------------------------------------------------------------
# Theme + font-scale integration
# ---------------------------------------------------------------------------

def test_stylesheet_accepts_theme_and_font_scale(qt_theme_applied):
    from spacr.qt.theme import stylesheet
    dark = stylesheet(theme="dark", font_scale=1.0)
    light = stylesheet(theme="light", font_scale=1.0)
    scaled = stylesheet(theme="dark", font_scale=1.5)
    assert dark != light            # palette-driven
    assert dark != scaled           # font-size driven
    assert "font-size" in dark


def test_palette_for_returns_light_for_light(qt_theme_applied):
    from spacr.qt.theme import palette_for, PALETTE, LIGHT_PALETTE
    assert palette_for("light") is LIGHT_PALETTE
    assert palette_for("dark") is PALETTE
    assert palette_for("bogus") is PALETTE     # fallback


def test_apply_preferences_to_app_does_not_raise(qt_theme_applied):
    from spacr.qt.preferences import (
        apply_preferences_to_app, set_theme, set_font_scale,
    )
    set_theme("light"); set_font_scale(1.25)
    # Should complete cleanly — no exception even mid-swap
    apply_preferences_to_app()


def test_preferences_dialog_builds_and_closes(qtbot, qt_theme_applied):
    from spacr.qt.preferences import PreferencesDialog
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    # Cancel — no persistence
    dlg.reject()
