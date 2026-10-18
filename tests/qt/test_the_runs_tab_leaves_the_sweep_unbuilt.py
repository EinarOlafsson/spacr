"""Opening Regression's Runs tab does not build the parameter-sweep panel.

The Runs tab asks the sweep card where it writes, and asking built the whole
sweep panel -- half a second on the tab's first open in lesson 13 -- only to
read an output folder a panel nobody opened leaves empty.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


def test_the_runs_tab_reads_no_folder_from_an_unopened_sweep(
        qtbot, qt_theme_applied):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    tabs = screen._results_tabs
    runs = screen._sweep_runs
    assert not screen._sweep.built()

    tabs.setCurrentWidget(runs)
    screen._on_results_tab_changed(tabs.indexOf(runs))

    assert screen._sweep_destination() == ""
    assert not screen._sweep.built(), "the Runs tab built the sweep panel"


def test_an_opened_sweep_still_names_its_folder(qtbot, qt_theme_applied,
                                                tmp_path):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    screen._sweep.destination.setText(str(tmp_path))

    assert screen._sweep.built()
    assert screen._sweep_destination() == str(tmp_path)
