"""A finished Regression run rebuilds the figure grid on the grid's timer.

The run's result arrived and, in the same GUI-thread turn as drawing every
results tab, rebuilt the all-figures grid and photographed nine live plots
for it: one freeze of over half a second in lesson 13. The grid page is
raised at once and rebuilt on its debounce timer, a turn later.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


def _frame():
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(6)],
        "coefficient": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        "p_value": [0.001, 0.1, 0.8, 0.3, 0.02, 0.0001],
    })


def test_the_grid_waits_for_its_timer_and_then_rebuilds(
        qtbot, qt_theme_applied, monkeypatch):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    rebuilt = []
    real = AppScreen._refresh_figure_grid
    monkeypatch.setattr(
        AppScreen, "_refresh_figure_grid",
        lambda self: (rebuilt.append(1), real(self))[1])
    screen._figures_stack.setCurrentIndex(1)
    screen._grid_refresh.timeout.disconnect()
    screen._grid_refresh.timeout.connect(screen._refresh_figure_grid)

    screen._on_pipeline_result({"results": _frame(), "res_folder": ""})

    assert rebuilt == [], "the grid was rebuilt inside the result's turn"
    assert screen._figures_stack.currentIndex() == 0
    assert screen._grid_refresh.isActive()
    qtbot.waitUntil(lambda: bool(rebuilt), timeout=5000)
    assert screen._results_loaded_in_memory is True


def test_returning_to_the_grid_by_hand_still_rebuilds_at_once(
        qtbot, qt_theme_applied, monkeypatch):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    rebuilt = []
    monkeypatch.setattr(AppScreen, "_refresh_figure_grid",
                        lambda self: rebuilt.append(1))

    screen._show_figure_grid()

    assert rebuilt == [1]
