"""The division analysis settings and results panel are alpha (item 771).

Its four settings sit under "Division Analysis α" on the Timelapse form and
its results panel sits in the Track preview card; both are hidden until
Preferences -> Show alpha features is turned on. The panel draws the three
figures from a results folder written by the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEYS = ("timelapse_division_analysis", "timelapse_division_mode",
        "timelapse_division_window", "timelapse_division_size_tolerance")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


@pytest.fixture
def results(tmp_path):
    """Two fields' division results under one tracks/division folder."""
    from spacr import division_analysis as da
    from tests.test_division_analysis import _synthetic_movie

    meas, lineage = _synthetic_movie(n_lineages=12)
    result = da.analyse_divisions(meas, lineage, columns=("area", "minor_axis"), n_boot=50)
    root = tmp_path / "tracks" / "division"
    for field in ("trackastra_tracks_cell_A01", "trackastra_tracks_cell_A02"):
        da.write_division_results(result, root / field, plot=False)
    return root


def test_the_settings_and_panel_are_registered_under_their_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, timelapse_settings

    assert ALPHA_FEATURES[771] == {"settings": KEYS, "widgets": ("DivisionAnalysisPanel",)}
    assert set(KEYS) <= set(timelapse_settings)
    assert all(_is_alpha("settings", key) for key in KEYS)
    assert _is_alpha("widgets", "DivisionAnalysisPanel")


def test_the_timelapse_rows_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("timelapse")
    try:
        screen._open_the_heading_of(KEYS[0])
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(k) for k in KEYS)
        assert screen._settings_model.set_value_for_key(KEYS[1], "budding")
        assert screen._settings_model.collect()[KEYS[1]] == "budding"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_the_panel_is_hidden_in_the_track_preview_until_alpha_is_on(qtbot, prefs):
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.qt.widgets.division_panel import DivisionAnalysisPanel
    from spacr.qt.widgets.timelapse_preview import build_timelapse_preview_card

    _panel, card = build_timelapse_preview_card(object())
    qtbot.addWidget(card)
    panel = card.findChild(DivisionAnalysisPanel, "DivisionAnalysisPanel")
    assert panel is not None and panel.isHidden()
    prefs._set_show_alpha_features(True)
    _apply_alpha_widgets(card)
    assert not panel.isHidden()


def test_the_panel_draws_a_results_folder_and_lists_several(qtbot, results):
    from spacr.qt.widgets.division_panel import DivisionAnalysisPanel

    panel = DivisionAnalysisPanel()
    qtbot.addWidget(panel)
    assert panel.load_folder(str(results))
    assert panel._choice.count() == 2 and not panel._choice.isHidden()
    assert [panel._tabs.tabText(i) for i in range(panel._tabs.count())] == [
        "Trajectories", "Effect sizes", "Size control"]
    assert "divisions aligned" in panel._status.text()
    panel._choice.setCurrentIndex(1)
    assert panel._tabs.count() == 3

    single = results / "trackastra_tracks_cell_A01"
    assert panel.load_folder(str(single))
    assert panel._choice.isHidden()


def test_the_panel_says_what_is_missing(qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets import division_panel

    panel = division_panel.DivisionAnalysisPanel()
    qtbot.addWidget(panel)
    assert not panel.load_folder(str(tmp_path))
    assert "No division analysis results" in panel._status.text()
    assert panel._tabs.isHidden()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "summary.json").write_text("{not json", encoding="utf-8")
    assert not panel.load_folder(str(broken))
    assert "Could not read" in panel._status.text()
    monkeypatch.setattr(division_panel.QFileDialog, "getExistingDirectory",
                        lambda *a, **k: "")
    assert not panel.browse()
