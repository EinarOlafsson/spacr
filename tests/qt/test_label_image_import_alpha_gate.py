"""Measuring another tool's label images is an alpha feature (item 790).

Its ten settings sit on the Measure form under their own "Label Image
Import α" heading and stay hidden until Preferences -> Show alpha features
is on; a value saved while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEYS = ("label_import", "label_import_images", "label_import_image_regex",
        "label_import_channels", "cell_label_folder", "nucleus_label_folder",
        "pathogen_label_folder", "cell_label_regex", "nucleus_label_regex",
        "pathogen_label_regex")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_settings_are_registered_under_their_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, categories

    assert ALPHA_FEATURES[790] == {"settings": KEYS}
    assert set(categories["Label Image Import α"]) == set(KEYS)
    assert all(_is_alpha("settings", key) for key in KEYS)


def test_the_measure_rows_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("measure")
    try:
        screen._open_the_heading_of("cell_label_folder")
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in KEYS)
        assert screen._settings_model.set_value_for_key(
            "cell_label_folder", "/lab/cellprofiler/labels")
        assert screen._settings_model.collect()["cell_label_folder"] == \
            "/lab/cellprofiler/labels"

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key) for key in KEYS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible("cell_label_folder")
        assert screen._settings_model.collect()["cell_label_folder"] == \
            "/lab/cellprofiler/labels"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
