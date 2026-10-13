"""The real / not-real object classifier sits behind Preferences -> Show alpha features.

Make Masks' two classifier settings, under "Object Filtration (all
objects)", are off the form and out of the settings search with the switch
off (the default), come back when it is turned on, and a value set while
hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr.settings import ALPHA_FEATURES, _is_alpha              # noqa: E402

KEYS = ALPHA_FEATURES[470]["settings"]


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


def test_both_classifier_settings_are_registered_as_alpha():
    assert KEYS == ("real_object_classifier", "real_object_threshold")
    assert all(_is_alpha("settings", key) for key in KEYS)
    assert not _is_alpha("settings", "object_filters")


def test_the_mask_form_hides_and_shows_the_classifier_settings(qtbot, prefs):
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
        assert prefs._is_alpha_visible("settings", KEYS[0]) is False
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert not _heading(screen, "Object Filtration (all objects)").isHidden()
        if bar is not None:
            assert not set(KEYS) & set(bar.indexed_keys())
        model = screen._settings_model
        assert model.set_value_for_key("real_object_classifier", "/models/real")
        assert model.set_value_for_key("real_object_threshold", 0.7)
        collected = model.collect()
        assert collected["real_object_classifier"] == "/models/real"
        assert float(collected["real_object_threshold"]) == 0.7

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(k) for k in KEYS)
        if bar is not None:
            assert set(KEYS) <= set(bar.indexed_keys())

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
