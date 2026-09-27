"""The Parquet and R export settings are an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For the table export that is its two
settings on the AnnData Export form; a value saved while they are hidden
still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

TABLE_SETTINGS = ("anndata_format", "anndata_tidy_dir")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_both_settings_are_registered_under_their_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    assert ALPHA_FEATURES[581] == {"settings": TABLE_SETTINGS}
    for key in TABLE_SETTINGS:
        assert _is_alpha("settings", key)


def test_the_gate_answers_for_both_settings(prefs):
    assert not any(prefs._is_alpha_visible("settings", key)
                   for key in TABLE_SETTINGS)
    assert prefs._is_alpha_visible("settings", "anndata_out")
    prefs._set_show_alpha_features(True)
    assert all(prefs._is_alpha_visible("settings", key)
               for key in TABLE_SETTINGS)


def test_the_table_settings_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("anndata_export")
    try:
        for key in TABLE_SETTINGS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()

        assert not any(screen.setting_row_is_visible(key)
                       for key in TABLE_SETTINGS)
        assert screen.setting_row_is_visible("anndata_out")
        assert screen._settings_model.set_value_for_key(
            "anndata_format", "parquet")
        assert screen._settings_model.collect()["anndata_format"] == "parquet"

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key)
                   for key in TABLE_SETTINGS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in TABLE_SETTINGS)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
