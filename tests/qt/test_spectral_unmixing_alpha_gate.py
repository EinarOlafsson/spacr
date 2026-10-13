"""The spectral unmixing settings are an alpha feature.

Hidden on Make Masks and Measure until Preferences -> Show alpha features is
turned on; a value saved while they are hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

UNMIX_SETTINGS = ("unmix", "unmix_controls", "unmix_background_percentile")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_settings_are_registered_under_their_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    assert ALPHA_FEATURES[538] == {"settings": UNMIX_SETTINGS}
    for key in UNMIX_SETTINGS:
        assert _is_alpha("settings", key)


@pytest.mark.parametrize("app", ["mask", "measure"])
def test_the_unmixing_settings_follow_the_switch(qtbot, prefs, app):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen(app)
    try:
        for key in UNMIX_SETTINGS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()

        assert not any(screen.setting_row_is_visible(key)
                       for key in UNMIX_SETTINGS)
        assert screen._settings_model.set_value_for_key(
            "unmix_controls", "0:A01; 1:B01")
        assert screen._settings_model.set_value_for_key("unmix", True)
        collected = screen._settings_model.collect()
        assert collected["unmix_controls"] == "0:A01; 1:B01"
        assert collected["unmix"] is True

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key)
                   for key in UNMIX_SETTINGS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in UNMIX_SETTINGS)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
