"""Lineage trees from tracks on the Timelapse app are an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For lineage trees that is its four
settings under the "Lineage Trees α" heading of the Timelapse form; a
value saved while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEYS = ("timelapse_lineage", "timelapse_lineage_color_by",
        "timelapse_lineage_max_distance", "timelapse_lineage_min_division_h")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_settings_are_registered_under_their_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, timelapse_settings

    assert ALPHA_FEATURES[537] == {"settings": KEYS}
    assert set(KEYS) <= set(timelapse_settings)
    assert all(_is_alpha("settings", key) for key in KEYS)


def test_the_gate_hides_them_until_alpha_features_are_on(prefs):
    assert not any(prefs._is_alpha_visible("settings", k) for k in KEYS)
    prefs._set_show_alpha_features(True)
    assert all(prefs._is_alpha_visible("settings", k) for k in KEYS)


def test_the_timelapse_rows_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("timelapse")
    try:
        screen._open_the_heading_of(KEYS[0])
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert screen._settings_model.set_value_for_key(KEYS[0], True)
        assert screen._settings_model.collect()[KEYS[0]] is True
        assert screen._settings_model.set_value_for_key(KEYS[3], 12.0)

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(k) for k in KEYS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert screen._settings_model.collect()[KEYS[0]] is True
        assert screen._settings_model.collect()[KEYS[3]] == 12.0
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_mask_generation_does_not_offer_them(qtbot):
    from PySide6.QtWidgets import QWidget
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets("mask", parent=owner)
    rendered = {widget.property("settingKey")
                for _title, rows in model.build_sections()
                for _label, widget in rows}
    assert not set(KEYS) & rendered
