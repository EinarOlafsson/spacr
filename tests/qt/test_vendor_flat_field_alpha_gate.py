"""The vendor flat-field profile setting is an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For vendor flat-field profiles that is the
profile path and channel assignments under Illumination Correction; values
saved while hidden or disabled still reach the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

KEY = "illumination_vendor_profile"
MAP_KEY = "illumination_vendor_channel_map"
KEYS = (KEY, MAP_KEY)


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

    assert ALPHA_FEATURES[543] == {"settings": KEYS}
    assert all(_is_alpha("settings", key) for key in KEYS)


@pytest.mark.parametrize("key", KEYS)
def test_the_gate_answers_for_the_setting(prefs, key):
    assert not prefs._is_alpha_visible("settings", key)
    assert prefs._is_alpha_visible("settings", "illumination_model")
    prefs._set_show_alpha_features(True)
    assert prefs._is_alpha_visible("settings", key)


@pytest.mark.parametrize("app", ["measure", "illumination"])
@pytest.mark.parametrize("key", KEYS)
def test_the_profile_row_follows_the_switch(qtbot, prefs, tmp_path, app, key):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen(app)
    try:
        screen._open_the_heading_of(key)
        screen._open_the_heading_of("illumination_model")
        screen._refresh_alpha_visibility()

        assert not screen.setting_row_is_visible(key)
        assert screen.setting_row_is_visible("illumination_model")
        profile = str(tmp_path / "Index.idx.xml") if key == KEY else "0:2,1:1"
        assert screen._settings_model.set_value_for_key(key, profile)
        assert screen._settings_model.collect()[key] == profile

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible(key)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(key)
        assert screen._settings_model.collect()[key] == profile
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


@pytest.mark.parametrize("app", ["measure", "illumination"])
def test_mapping_dependency_preserves_value_through_each_inactive_state(qtbot, prefs, app):
    """The actual editor follows all three prerequisites without losing its map."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    prefs._set_show_alpha_features(True)
    screen = AppScreen(app)
    qtbot.addWidget(screen)
    try:
        for key in (*KEYS, "illumination_model", "illumination_correction"):
            screen._open_the_heading_of(key)
        model = screen._settings_model
        control = model._built_control(MAP_KEY)
        assert control is not None
        assert model.set_value_for_key(MAP_KEY, "0:2,1:1")
        for correction, profile, saved_model, enabled in (
                (False, "profile.xml", "", False),
                (True, "", "", False),
                (True, "   ", "", False),
                (True, "profile.xml", "", True),
                (True, "profile.xml", "saved.npz", False),
                (True, "profile.xml", "   ", True)):
            assert model.set_value_for_key("illumination_correction", correction)
            assert model.set_value_for_key(KEY, profile)
            assert model.set_value_for_key("illumination_model", saved_model)
            model._refresh_setting_dependencies()
            assert control.isEnabled() is enabled
            assert model.collect()[MAP_KEY] == "0:2,1:1"
            label = control._spacr_setting_label
            help_text = str(label.property("apiTooltipHtml") or label.toolTip() or "")
            assert ("The value is kept and saved" in help_text) is (not enabled)
        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(MAP_KEY)
        assert model.collect()[MAP_KEY] == "0:2,1:1"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
