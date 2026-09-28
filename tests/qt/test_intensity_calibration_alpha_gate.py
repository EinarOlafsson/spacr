"""The intensity calibration settings sit behind Show alpha features.

Measure's "Intensity Calibration (Alpha)" card and its four settings are off
the form and out of the settings search with the switch off (the default),
come back when it is turned on, and a value set while hidden still reaches
the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr.settings import ALPHA_FEATURES, _is_alpha              # noqa: E402

KEYS = ALPHA_FEATURES[580]["settings"]


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


def test_every_calibration_setting_is_registered_as_alpha():
    assert len(KEYS) == 4
    assert all(key.startswith("intensity_calibration") for key in KEYS)
    assert all(_is_alpha("settings", key) for key in KEYS)
    assert not _is_alpha("settings", "illumination_correction")


def test_the_measure_form_hides_and_shows_the_calibration_settings(
        qtbot, prefs):
    from spacr.intensity_rescale import _calibration_wells
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL, install
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    title = "Intensity Calibration (Alpha)"
    screen = AppScreen("measure")
    try:
        bar = install(screen) or getattr(screen, "_settings_search", None)
        if bar is not None:
            bar.set_level(ALL)
        for key in KEYS:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert prefs._is_alpha_visible("settings", KEYS[0]) is False
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert _heading(screen, title).isHidden()
        assert not _heading(screen, "Features").isHidden()
        if bar is not None:
            assert not set(KEYS) & set(bar.indexed_keys())
        model = screen._settings_model
        assert model.set_value_for_key("intensity_calibration", True)
        assert model.set_value_for_key("intensity_calibration_wells",
                                       "A01, P24")
        assert model.set_value_for_key("intensity_calibration_statistic",
                                       "median")
        assert model.set_value_for_key("intensity_calibration_offset", "100")
        collected = model.collect()
        assert set(KEYS) <= set(collected)
        assert collected["intensity_calibration"] is True
        assert collected["intensity_calibration_statistic"] == "median"
        assert float(collected["intensity_calibration_offset"]) == 100.0
        assert sorted(_calibration_wells(collected).values()) == [
            "A01", "P24"]

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(k) for k in KEYS)
        assert not _heading(screen, title).isHidden()
        if bar is not None:
            assert set(KEYS) <= set(bar.indexed_keys())

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert _heading(screen, title).isHidden()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
