"""The vendor flat-field profile setting is an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For vendor flat-field profiles that is the
one path setting under Illumination Correction on the Measure form; a value
saved while it is hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEY = "illumination_vendor_profile"


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_setting_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    assert ALPHA_FEATURES[543] == {"settings": (KEY,)}
    assert _is_alpha("settings", KEY)


def test_the_gate_answers_for_the_setting(prefs):
    assert not prefs._is_alpha_visible("settings", KEY)
    assert prefs._is_alpha_visible("settings", "illumination_model")
    prefs._set_show_alpha_features(True)
    assert prefs._is_alpha_visible("settings", KEY)


@pytest.mark.parametrize("app", ["measure", "illumination"])
def test_the_profile_row_follows_the_switch(qtbot, prefs, tmp_path, app):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen(app)
    try:
        screen._open_the_heading_of(KEY)
        screen._open_the_heading_of("illumination_model")
        screen._refresh_alpha_visibility()

        assert not screen.setting_row_is_visible(KEY)
        assert screen.setting_row_is_visible("illumination_model")
        profile = str(tmp_path / "Index.idx.xml")
        assert screen._settings_model.set_value_for_key(KEY, profile)
        assert screen._settings_model.collect()[KEY] == profile

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible(KEY)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
