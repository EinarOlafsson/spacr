"""The measurement backend on Measure is an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For the measurement backend that is its two
settings on the Measure form, under their own "Measurement Backend α"
heading; a value saved while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEY = "measurement_backend"
KEYS = (KEY, "measurement_backend_target")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_setting_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, categories

    assert ALPHA_FEATURES[576] == {"settings": KEYS}
    assert tuple(categories["Measurement Backend α"]) == KEYS
    assert all(_is_alpha("settings", key) for key in KEYS)


def test_the_gate_hides_it_until_alpha_features_are_on(prefs):
    assert prefs._is_alpha_visible("settings", KEY) is False
    prefs._set_show_alpha_features(True)
    assert prefs._is_alpha_visible("settings", KEY) is True


def test_the_measure_row_follows_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("measure")
    try:
        screen._open_the_heading_of(KEY)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.set_value_for_key(KEY, "duckdb")
        assert screen._settings_model.collect()[KEY] == "duckdb"

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible(KEY)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.collect()[KEY] == "duckdb"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
