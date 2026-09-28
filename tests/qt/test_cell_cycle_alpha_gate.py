"""The cell-cycle settings sit behind Preferences -> Show alpha features.

Measure's "Cell Cycle (Alpha)" card and its nine settings are off the form
and out of the settings search with the switch off (the default), come back
when it is turned on, and a value set while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr.settings import ALPHA_FEATURES, _is_alpha              # noqa: E402

KEYS = ALPHA_FEATURES[535]["settings"]


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


def test_every_cell_cycle_setting_is_registered_as_alpha():
    assert len(KEYS) == 9
    assert all(key.startswith("cell_cycle") for key in KEYS)
    assert all(_is_alpha("settings", key) for key in KEYS)
    assert not _is_alpha("settings", "nucleus_channel")


def test_the_measure_form_hides_and_shows_the_cell_cycle_settings(qtbot,
                                                                  prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL, install
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("measure")
    try:
        bar = install(screen) or getattr(screen, "_settings_search", None)
        if bar is not None:
            bar.set_level(ALL)
        for key in KEYS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert prefs._is_alpha_visible("settings", "cell_cycle") is False
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert _heading(screen, "Cell Cycle (Alpha)").isHidden()
        assert not _heading(screen, "Features").isHidden()
        assert screen.setting_row_is_visible("radial_dist")
        if bar is not None:
            assert not set(KEYS) & set(bar.indexed_keys())
        model = screen._settings_model
        assert model.set_value_for_key("cell_cycle_method", "xgboost")
        collected = model.collect()
        assert collected["cell_cycle_method"] == "xgboost"
        assert set(KEYS) <= set(collected)

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert prefs._is_alpha_visible("settings", "cell_cycle") is True
        assert all(screen.setting_row_is_visible(k) for k in KEYS)
        assert not _heading(screen, "Cell Cycle (Alpha)").isHidden()
        if bar is not None:
            assert set(KEYS) <= set(bar.indexed_keys())

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert _heading(screen, "Cell Cycle (Alpha)").isHidden()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
