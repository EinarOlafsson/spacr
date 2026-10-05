"""The folder watch on Make Masks is an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For the folder watch that is its six
settings on the Make Masks form and the progress line beside the Run button;
a watch value saved while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

WATCH_SETTINGS = ("watch_folder", "watch_pipeline", "watch_measure_settings",
                  "watch_classify_settings",
                  "watch_settle_seconds", "watch_poll_seconds",
                  "watch_idle_minutes")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_every_part_of_the_watch_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    entry = ALPHA_FEATURES[548]
    assert set(entry["settings"]) == set(WATCH_SETTINGS)
    assert entry["widgets"] == ("WatchFolderProgress", "WatchLivePlate")
    for key in WATCH_SETTINGS:
        assert _is_alpha("settings", key)


def test_the_watch_settings_and_progress_follow_the_switch(qtbot, prefs, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("mask")
    try:
        for key in WATCH_SETTINGS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        line = "watch_folder: 2 analysed, 1 waiting, 0 failed\n"

        assert not any(screen.setting_row_is_visible(key)
                       for key in WATCH_SETTINGS)
        screen._show_watch_progress(line)
        assert screen._watch_progress.isHidden()
        assert screen._watch_live_plate.isHidden()
        assert screen._settings_model.set_value_for_key("watch_folder", True)
        assert screen._settings_model.collect()["watch_folder"] is True

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key)
                   for key in WATCH_SETTINGS)
        screen._show_watch_progress(line)
        assert not screen._watch_progress.isHidden()
        assert "2" in screen._watch_progress.text()
        assert screen._watch_live_plate.isHidden()

        screen._watch_live_plate.begin(str(tmp_path), "mask")
        assert not screen._watch_live_plate.isHidden()

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key)
                       for key in WATCH_SETTINGS)
        assert screen._watch_progress.isHidden()
        assert screen._watch_live_plate.isHidden()
        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert not screen._watch_live_plate.isHidden()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_output_without_a_watch_line_leaves_the_label_alone(qtbot, prefs):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.screens.app_screen import AppScreen

    class Host:
        _watch_progress = QLabel()

    host = Host()
    host._watch_progress.setObjectName("WatchFolderProgress")
    qtbot.addWidget(host._watch_progress)
    prefs._set_show_alpha_features(True)
    AppScreen._show_watch_progress(host, "Progress: 3/4\n")
    assert host._watch_progress.text() == ""
    AppScreen._show_watch_progress(
        host, "watch_folder: 5 analysed, 0 waiting, 2 failed\n")
    assert host._watch_progress.text() == (
        "Watching: 5 analysed, 0 waiting, 2 failed")
