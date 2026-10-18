"""Showing a run whose plot state is remembered draws every tab once.

Lesson 13's finished regression arrived on a panel that already remembered
its run, so ``set_frame`` drew all the tabs -- two coefficient tables, the
guide support and every plot -- and the restore of the remembered state
then drew them all again: two seconds of the GUI thread for one frame.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytestmark = pytest.mark.qt


def _frame():
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(6)],
        "coefficient": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        "p_value": [0.001, 0.1, 0.8, 0.3, 0.02, 0.0001],
        "condition": ["positive", "negative", "other"] * 2,
    })


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


def _count_refreshes(panel, monkeypatch):
    calls = []
    real = type(panel).refresh_views

    def counted(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(type(panel), "refresh_views", counted)
    return calls


def test_the_same_run_again_is_drawn_once_with_its_view(
        panel, monkeypatch, tmp_path):
    source = str(tmp_path / "run_a")
    assert panel.set_frame(_frame(), source=source)
    panel.set_threshold_multiplier(2.0)
    panel.volcano.set_axis_limits(x=(-1.5, 1.5))
    before = panel.plot_state()
    calls = _count_refreshes(panel, monkeypatch)

    assert panel.set_frame(_frame(), source=source)

    assert len(calls) == 1
    after = panel.plot_state()
    assert after["threshold_multiplier"] == before["threshold_multiplier"]
    assert after["x_limits"] == before["x_limits"]
    assert panel.table.table.rowCount() == len(_frame())


def test_a_run_with_nothing_remembered_is_drawn_once(
        panel, monkeypatch, tmp_path):
    calls = _count_refreshes(panel, monkeypatch)

    assert panel.set_frame(_frame(), source=str(tmp_path / "run_b"))

    assert len(calls) == 1
    assert panel.table.table.rowCount() == len(_frame())


def test_a_state_recorded_mid_draw_is_not_drawn_again(
        panel, monkeypatch, tmp_path):
    """Lesson 13's case: the run journal saved the workspace mid-draw."""
    calls = []
    real = type(panel).refresh_views

    def saving(self):
        calls.append(1)
        real(self)
        self.workspace_state()

    monkeypatch.setattr(type(panel), "refresh_views", saving)
    source = str(tmp_path / "run_c")

    assert panel.set_frame(_frame(), source=source)

    assert len(calls) == 1
    assert panel.table.table.rowCount() == len(_frame())
