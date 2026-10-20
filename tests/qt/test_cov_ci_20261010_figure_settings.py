"""Applying statistics skips panel slots the figure no longer has."""
from __future__ import annotations

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from spacr.figures import bundle
from spacr.qt.widgets.figure_settings import _StatisticsDialog


def _two_panel_figure():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "condition": ["a"] * 20 + ["b"] * 20,
        "group": (["x"] * 10 + ["y"] * 10) * 2,
        "value": np.concatenate([rng.normal(0, 1, 10), rng.normal(3, 1, 10),
                                 rng.normal(0, 1, 10), rng.normal(3, 1, 10)])})
    figure = Figure()
    bundle._register_figure_data(
        figure, frame, x="group", y="value", kind="box", grid=[1, 2],
        panels=[{"slot": 0, "where": {"condition": ["a"]}},
                {"slot": 1, "where": {"condition": ["b"]}}])
    bundle._draw(figure, frame, figure._spacr_spec)
    return figure


def _notes(axes):
    return [text.get_text() for text in axes.texts
            if text.get_gid() == "spacr-stats"]


def test_apply_annotates_present_panels_and_skips_a_removed_slot(qapp):
    figure = _two_panel_figure()
    first, second = figure.axes
    figure.delaxes(second)
    dialog = _StatisticsDialog(figure)
    try:
        assert set(dialog.table["panel"]) == {0, 1}
        dialog._apply()
        assert _notes(first)
        assert not _notes(second)
        assert [panel["slot"] for panel in figure._spacr_spec["panels"]] == [0, 1]
        assert all(panel["stats_note"] for panel in figure._spacr_spec["panels"])
    finally:
        dialog.deleteLater()
