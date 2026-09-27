"""The segmentation-robustness report sits behind Preferences -> Show alpha features.

Make Masks' "Segmentation Robustness (Alpha)" card and its eight settings are
off the form and out of the settings search with the switch off (the
default), come back when it is turned on, and a value set while hidden still
reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr.settings import ALPHA_FEATURES, _is_alpha              # noqa: E402

KEYS = ALPHA_FEATURES[578]["settings"]


def _heading(screen, name):
    """The rendered settings card whose title is ``name``."""
    for section in screen.rendered_settings_sections():
        title = getattr(section, "title", None)
        text = title() if callable(title) else ""
        if str(text).upper() == name.upper():
            return section
    raise AssertionError(f"no {name!r} card on the form")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_every_robustness_setting_is_registered_as_alpha():
    assert len(KEYS) == 8
    assert all(key.startswith("robustness_") for key in KEYS)
    assert all(_is_alpha("settings", key) for key in KEYS)
    assert not _is_alpha("settings", "seg_qc")


def test_the_mask_form_hides_and_shows_the_robustness_settings(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL, install
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("mask")
    try:
        bar = install(screen) or getattr(screen, "_settings_search", None)
        if bar is not None:
            bar.set_level(ALL)
        for key in KEYS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert prefs._is_alpha_visible("settings", "robustness_report") is False
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert _heading(screen, "Segmentation Robustness (Alpha)").isHidden()
        assert not _heading(screen, "Quality Control").isHidden()
        if bar is not None:
            assert not set(KEYS) & set(bar.indexed_keys())
        model = screen._settings_model
        assert model.set_value_for_key("robustness_report", True)
        assert model.set_value_for_key("robustness_tolerance", 0.3)
        collected = model.collect()
        assert collected["robustness_report"] is True
        assert float(collected["robustness_tolerance"]) == 0.3
        assert set(KEYS) <= set(collected)

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert prefs._is_alpha_visible("settings", "robustness_report") is True
        assert all(screen.setting_row_is_visible(k) for k in KEYS)
        assert not _heading(screen, "Segmentation Robustness (Alpha)").isHidden()
        if bar is not None:
            assert set(KEYS) <= set(bar.indexed_keys())

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert _heading(screen, "Segmentation Robustness (Alpha)").isHidden()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()

