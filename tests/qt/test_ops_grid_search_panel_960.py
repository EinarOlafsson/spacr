"""Item 960: the Grid search panel of the OPS Live preview's Spots & genotyping tab.

The panel reads its inputs from the form, runs the grid off the GUI thread,
fills a sortable table with every point, and writes the selected point back
into the form with the grid run.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from spacr.qt.widgets import ops_grid_search_panel as W  # noqa: E402

pytestmark = pytest.mark.qt


def _result():
    """A grid run's result with two points: the default and one fixed point."""
    table = pd.DataFrame([
        {"point": 0, "is_default": True, "pareto": True, "ops_read_mode": "log",
         "ops_read_threshold": 80.0, "ops_prefix_min_read_quality": 0.0,
         "ops_prefix_max_distance": 1, "ops_quality_min_reads": 2, "ops_prefix_crosstalk": "none",
         "spots": 100, "reads": 90, "mapping_pct": 80.0, "assignment_pct": 40.0,
         "decoy_fdr_pct": 0.5, "shuffle_rate_pct": 0.3},
        {"point": 1, "is_default": False, "pareto": True, "ops_read_mode": "raw",
         "ops_read_threshold": 40.0, "ops_prefix_min_read_quality": 0.0,
         "ops_prefix_max_distance": 0, "ops_quality_min_reads": 2, "ops_prefix_crosstalk": "none",
         "spots": 150, "reads": 140, "mapping_pct": 70.0, "assignment_pct": 45.0,
         "decoy_fdr_pct": 0.9, "shuffle_rate_pct": 0.6}])
    return {"run_id": "abc123def456", "table": table, "points": 2, "tiles": ["P:W1:1"],
            "seconds": 1.5, "seed": 3, "plot": None}


def test_the_panel_runs_the_grid_and_applies_the_selected_point(qtbot):
    form = {"ops_grid_tile_mode": "edge_centre", "ops_grid_seed": 3,
            "ops_grid_params": {"ops_read_threshold": [40, 80]},
            "ops_prefix_threshold_tuning": "per_well_decoy"}
    seen, written = [], []

    def run(settings):
        seen.append(settings)
        return _result()

    panel = W._OpsGridSearchPanel(lambda: form, written.append, run=run)
    qtbot.addWidget(panel)
    assert panel.mode.currentText() == "edge_centre" and panel.seed.value() == 3
    assert panel.grid.text() == "ops_read_threshold=40,80"
    assert panel.run_grid()
    qtbot.waitUntil(lambda: panel.apply_button.isEnabled(), timeout=10000)
    assert seen[0]["ops_grid_tile_mode"] == "edge_centre" and seen[0]["ops_grid_n_tiles"] == 8
    assert panel.table.rowCount() == 2
    assert "abc123def456" in panel.status.text()
    assert not panel.apply_selected()
    panel.table.selectRow([panel.table.item(r, 0).text() for r in range(2)].index("1"))
    assert panel.apply_selected()
    applied = written[-1]
    assert applied["ops_read_mode"] == "raw" and applied["ops_read_threshold"] == 40.0
    assert applied["ops_prefix_threshold_tuning"] == "none"
    assert applied["ops_grid_run_id"] == "abc123def456" and applied["ops_grid_search"] is False


def test_a_grid_that_does_not_parse_never_starts(qtbot):
    calls = []
    panel = W._OpsGridSearchPanel(lambda: {}, calls.append, run=calls.append)
    qtbot.addWidget(panel)
    panel.grid.setText("ops_min_fraction=0.5")
    assert not panel.run_grid()
    assert "cannot vary" in panel.status.text() and not calls


def test_a_failed_run_says_why(qtbot):
    def run(settings):
        raise ValueError("no tiles")

    panel = W._OpsGridSearchPanel(lambda: {}, lambda s: None, run=run)
    qtbot.addWidget(panel)
    assert panel.run_grid()
    qtbot.waitUntil(lambda: panel.run_button.isEnabled(), timeout=10000)
    assert panel.status.text() == "ValueError: no tiles"
    assert not panel.apply_button.isEnabled()


def test_attach_puts_the_panel_on_the_spots_tab(qtbot):
    class Live:
        def __init__(self):
            self.spots = QWidget()
            QVBoxLayout(self.spots)

        def page(self, step):
            assert step == "spots"
            return self.spots

    class Model:
        def collect(self):
            return {"ops_grid_n_tiles": 4}

    class Host:
        _settings_model = Model()

        def __init__(self):
            self.written = []

        def apply_settings_dict(self, settings):
            self.written.append(settings)
            return len(settings)

    live, host = Live(), Host()
    qtbot.addWidget(live.spots)
    panel = W._attach_grid_search(live, host)
    assert panel.parent() is live.spots and panel.n_tiles.value() == 4
    assert live.spots.layout().indexOf(panel) >= 0


def test_the_ops_screen_puts_the_panel_on_the_spots_tab(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    screen = AppScreen(app_key="ops")
    qtbot.addWidget(screen)
    try:
        panel = screen._ops_grid_search
        assert isinstance(panel, W._OpsGridSearchPanel)
        spots = screen._ops_live_preview.page("spots")
        assert panel.parent() is spots and spots.layout().indexOf(panel) >= 0
        assert isinstance(dict(panel._collect()), dict)
    finally:
        screen._ops_live_preview.shutdown()
        screen.close()
