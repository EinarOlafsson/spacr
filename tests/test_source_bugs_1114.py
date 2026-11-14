"""Regressions for source bugs logged in the batch-12 test audit."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr.annotation_power import screen_size_for  # noqa: E402
from spacr.classifier_evaluation import (  # noqa: E402
    _content_sha256, cross_calibrate_probabilities)


def test_one_guide_per_well_screen_sizes_without_dividing_by_zero():
    out = screen_size_for({'A1': {'g1': 1.0}, 'A2': {'g2': 1.0}},
                          sensitivity=0.95, specificity=0.99)
    assert out["specificity_needed_at_current_shape"] == 0.0
    assert out["wells_now"] == 2


def test_temperature_fold_warning_names_the_fold_as_a_plain_integer():
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2],
                                [0.2, 0.8], [0.1, 0.9]])
    warnings = []
    cross_calibrate_probabilities([0, 0, 1, 1], probabilities,
                                  np.asarray([0, 0, 1, 1]),
                                  warnings_out=warnings)
    assert warnings
    assert "np.int64" not in warnings[0]
    assert "fold 0 " in warnings[0]


def test_round_features_reads_the_database_without_writing_to_stdout(
        tmp_path, capsys):
    from tests import test_cov_288_active_learning_20261019 as helper
    db, _ = helper._project(tmp_path)
    capsys.readouterr()
    helper.al.round_features(db, tables=["cell", "nucleus"])
    assert capsys.readouterr().out == ""


def test_an_empty_path_is_reported_as_empty_not_as_a_missing_dot():
    digest, error = _content_sha256("")
    assert digest == ""
    assert "empty path" in error
    assert not error.startswith(".")


def _scene_plot():
    pytest.importorskip("PySide6")
    from spacr.figures.scene import pyqtgraph_ready
    ready, why = pyqtgraph_ready()
    if not ready:
        pytest.skip(why)
    import pyqtgraph as pg
    widget = pg.GraphicsLayoutWidget()
    return widget, widget.addPlot()


def test_an_axes_title_is_drawn_once_in_the_translated_scene():
    from spacr.figures.scene import SceneReport, _Look, _translate_axes
    widget, plot = _scene_plot()
    try:
        figure, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])
        axes.set_title("only once")
        _translate_axes(plot, axes, _Look(dpi=100.0), SceneReport())
        texts = [item for item in plot.items
                 if type(item).__name__ == "TextItem"
                 and "only once" in item.toPlainText()]
        assert texts == []
        assert plot.titleLabel.text == "only once"
    finally:
        plt.close("all")
        widget.deleteLater()


def test_log_axis_tick_labels_sit_at_log10_positions():
    from spacr.figures.scene import _carry_ticks
    widget, plot = _scene_plot()
    try:
        figure, axes = plt.subplots()
        axes.set_xscale("log")
        axes.plot([1, 10, 100], [1, 2, 3])
        axes.set_xticks([1, 10, 100])
        axes.set_xticklabels(["low", "mid", "high"])
        figure.canvas.draw()
        _carry_ticks(plot, axes)
        levels = plot.getAxis("bottom")._tickLevels
        positions = [pos for pos, _ in levels[0]]
        assert positions == pytest.approx([0.0, 1.0, 2.0])
    finally:
        plt.close("all")
        widget.deleteLater()
