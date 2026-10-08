"""Recorded panel views must not clip a new graph or grow on repeated Apply."""
from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from spacr.figures import bundle
from spacr.qt.widgets.figure_settings import _StatisticsDialog, _retype
from tests.test_controls_multipanel_recipe import controls_figure


def test_retyping_loaded_panel_recipe_releases_old_axis_limits(qapp, controls_figure):
    _frame, figure = controls_figure
    for axes in figure.axes:
        if axes.has_data():
            axes.set_ylim(-1.0, 0.01)
    _data, spec = bundle._figure_record(figure)
    saved = bundle._capture_view(figure, dict(spec, keep_limits=True))
    assert all(panel["ylim"] == [-1.0, 0.01] for panel in saved["panels"])
    figure._spacr_spec = saved
    assert _retype(figure, "box")
    for axes in figure.axes:
        if axes.has_data():
            low, high = axes.get_ylim()
            values = [value for line in axes.lines for value in line.get_ydata()]
            assert min(values) >= low
            assert max(values) <= high


@pytest.mark.parametrize("controls_figure", ["paired irregular"], indirect=True)
def test_repeated_statistics_apply_keeps_panel_limits_and_hide_restores_view(
        qapp, controls_figure):
    _frame, figure = controls_figure
    dialog = _StatisticsDialog(figure)
    try:
        dialog.pair.setCurrentIndex(dialog.pair.findData("source_object_id"))
        dialog.paired.setChecked(True)
        dialog.test.setCurrentIndex(dialog.test.findData("Friedman"))
        before = [axes.get_ylim() for axes in figure.axes]
        dialog._apply()
        first = [axes.get_ylim() for axes in figure.axes]
        assert any(after != original for after, original in zip(first, before))
        dialog._apply()
        assert [axes.get_ylim() for axes in figure.axes] == first
        dialog.show_on_plot.setChecked(False)
        dialog._apply()
        assert [axes.get_ylim() for axes in figure.axes] == before
    finally:
        dialog.deleteLater()


def test_removing_statistics_respects_subsequent_manual_axis_edit(qapp, controls_figure):
    _frame, figure = controls_figure
    dialog = _StatisticsDialog(figure)
    try:
        dialog._apply()
        target = figure.axes[0]
        target.set_ylim(-10.0, 100.0)
        dialog.show_on_plot.setChecked(False)
        dialog._apply()
        assert target.get_ylim() == (-10.0, 100.0)
        assert not any(artist.get_gid() == "spacr-stats"
                       for artist in [*target.texts, *target.lines])
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("controls_figure", ["paired irregular"], indirect=True)
def test_captured_statistics_view_recreates_the_same_panel_headroom(qapp, controls_figure):
    frame, figure = controls_figure
    dialog = _StatisticsDialog(figure)
    try:
        dialog.pair.setCurrentIndex(dialog.pair.findData("source_object_id"))
        dialog.paired.setChecked(True)
        dialog.test.setCurrentIndex(dialog.test.findData("Friedman"))
        dialog._apply()
        original = [figure.axes[panel["slot"]].get_ylim()
                    for panel in figure._spacr_spec["panels"]]
        captured = bundle._capture_view(figure, dict(figure._spacr_spec, keep_limits=True))
        recreated = Figure()
        bundle._draw(recreated, frame, captured)
        actual = [recreated.axes[panel["slot"]].get_ylim() for panel in captured["panels"]]
        assert actual == original
    finally:
        dialog.deleteLater()
