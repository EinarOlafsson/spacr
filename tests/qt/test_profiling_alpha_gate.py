"""Image-based profiling on Measure is an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For profiling that is its nine settings on
the Measure form, under their own "Profiling (Alpha)" heading; a profiling
value saved while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

PROFILING_SETTINGS = (
    "profiling", "profiling_metadata", "profiling_treatment_column",
    "profiling_negative_control", "profiling_normalization",
    "profiling_feature_selection", "profiling_correlation_threshold",
    "profiling_phenotype_column", "profiling_databases",
)


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_every_profiling_setting_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, categories

    entry = ALPHA_FEATURES[547]
    assert set(entry) == {"settings"}
    assert set(entry["settings"]) == set(PROFILING_SETTINGS)
    assert tuple(categories["Profiling (Alpha)"]) == PROFILING_SETTINGS
    for key in PROFILING_SETTINGS:
        assert _is_alpha("settings", key)


def test_the_gate_hides_profiling_until_alpha_features_are_on(prefs):
    for key in PROFILING_SETTINGS:
        assert prefs._is_alpha_visible("settings", key) is False
    prefs._set_show_alpha_features(True)
    for key in PROFILING_SETTINGS:
        assert prefs._is_alpha_visible("settings", key) is True


def test_the_measure_rows_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("measure")
    try:
        for key in PROFILING_SETTINGS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in PROFILING_SETTINGS)
        assert screen._settings_model.set_value_for_key("profiling", True)
        assert screen._settings_model.collect()["profiling"] is True

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key)
                   for key in PROFILING_SETTINGS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in PROFILING_SETTINGS)
        assert screen._settings_model.collect()["profiling"] is True
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
