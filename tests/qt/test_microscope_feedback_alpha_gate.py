"""Microscope feedback on Make Masks is an alpha feature.

Its ten settings stay hidden until Preferences -> Show alpha features is
turned on, and a value saved while they are hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

FEEDBACK_SETTINGS = (
    "microscope_feedback", "microscope_driver", "microscope_simulated_folder",
    "microscope_positions", "microscope_stage_transform",
    "microscope_event_table", "microscope_event_query",
    "microscope_max_events", "microscope_timepoints",
    "microscope_interval_seconds")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_every_feedback_setting_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    assert set(ALPHA_FEATURES[549]["settings"]) == set(FEEDBACK_SETTINGS)
    for key in FEEDBACK_SETTINGS:
        assert _is_alpha("settings", key)


def test_the_feedback_settings_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("mask")
    try:
        for key in FEEDBACK_SETTINGS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in FEEDBACK_SETTINGS)
        model = screen._settings_model
        assert model.set_value_for_key("microscope_feedback", True)
        assert model.set_value_for_key("microscope_driver", "pycromanager")
        collected = model.collect()
        assert collected["microscope_feedback"] is True
        assert collected["microscope_driver"] == "pycromanager"

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key)
                   for key in FEEDBACK_SETTINGS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in FEEDBACK_SETTINGS)
        assert model.collect()["microscope_driver"] == "pycromanager"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
