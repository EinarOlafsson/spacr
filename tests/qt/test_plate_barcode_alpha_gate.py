"""The plate barcode linkage settings sit behind Show alpha features.

Measure's "Plate Barcode Linkage (Alpha)" card and its four settings are off
the form and out of the settings search with the switch off (the default),
come back when it is turned on, and a value set while hidden still reaches
the run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr.settings import ALPHA_FEATURES, _is_alpha              # noqa: E402

KEYS = ALPHA_FEATURES[583]["settings"]


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


def test_every_barcode_setting_is_registered_as_alpha():
    assert len(KEYS) == 4
    assert all(key.startswith("plate_barcode") for key in KEYS)
    assert all(_is_alpha("settings", key) for key in KEYS)
    assert not _is_alpha("settings", "save_measurements")


def test_the_measure_form_hides_and_shows_the_barcode_settings(
        qtbot, prefs, tmp_path):
    from spacr.measure import _run_plate_barcode_step
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL, install
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    records = tmp_path / "lims.csv"
    records.write_text("barcode,well,strain\nBC9,A01,RH\n")
    src = tmp_path / "plate1" / "merged"
    src.mkdir(parents=True)
    np.save(src / "plate1_A01_1.npy", np.zeros((2, 2, 1)))

    title = "Plate Barcode Linkage (Alpha)"
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
        if bar is not None:
            assert not set(KEYS) & set(bar.indexed_keys())
        model = screen._settings_model
        assert model.set_value_for_key("plate_barcode_source", str(records))
        assert model.set_value_for_key("plate_barcodes", "{'plate1': 'BC9'}")
        collected = model.collect()
        assert set(KEYS) <= set(collected)
        assert collected["plate_barcode_source"] == str(records)
        collected.update(src=str(src), profiling_metadata="",
                         viability_plate_map="", timelapse=False)
        plate_map, mismatches = _run_plate_barcode_step(collected)
        assert plate_map["strain"].tolist() == ["RH"]
        assert plate_map["plate_barcode"].tolist() == ["BC9"]
        assert mismatches.empty

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
