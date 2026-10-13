"""Grouped-line and training-curve recipes rebuild figures without Qt or torch.

The 658 recipe tests go through ``spacr.plot.spacrGraph`` and the Qt
preferences store, so plain CPU CI never reached the rendering code in
``spacr.figures.bundle``. These tests drive ``_recipe_frame`` and ``_draw``
directly with recorded specs, as the standalone script does.
"""
import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from spacr.figures import bundle


def _grouped(**overrides):
    recipe = {"grouping_column": "condition", "columns": ["time", "signal"],
              "representation": "object", "summary_func": "mean",
              "order": ["b", "a"], "mode": "line", "log_x": False, "log_y": False,
              "palette": ["#112233", "#445566"]}
    recipe.update(overrides)
    return recipe


def _frame():
    return pd.DataFrame({
        "condition": ["a", "a", "a", "a", "b", "b", "b", "b", "a"],
        "time": [1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0, np.nan],
        "signal": [2.0, 4.0, 6.0, 8.0, 10.0, 30.0, 100.0, 300.0, 1.0],
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r1_c2", "p1_r1_c2",
                "p2_r1_c1", "p2_r1_c1", "p2_r1_c2", "p2_r1_c2", "p1_r2_c1"],
    })


def test_object_representation_keeps_rows_and_orders_categories():
    frame = _frame()
    data = bundle._recipe_frame(frame, {"grouped_line": _grouped()})
    assert len(data) == 8
    assert list(data["condition"].cat.categories) == ["b", "a"]
    assert data["condition"].cat.ordered
    assert len(frame) == 9


def test_well_representation_aggregates_per_well():
    data = bundle._recipe_frame(_frame(), {"grouped_line": _grouped(
        representation="well", summary_func="median")})
    rows = data.set_index("prc")
    assert len(rows) == 4
    assert rows.loc["p1_r1_c1", "signal"] == pytest.approx(3.0)
    assert rows.loc["p2_r1_c2", "signal"] == pytest.approx(200.0)
    assert rows.loc["p2_r1_c2", "condition"] == "b"


def test_plate_representation_splits_prc_and_aggregates_per_plate():
    data = bundle._recipe_frame(_frame(), {"grouped_line": _grouped(
        representation="plate")})
    assert set(data.columns) >= {"plateID", "condition", "time", "signal"}
    rows = {(r.plateID, r.condition): r.signal for r in data.itertuples()
            if not np.isnan(r.signal)}
    assert rows == {("p1", "a"): pytest.approx(5.0), ("p2", "b"): pytest.approx(110.0)}


def test_plate_representation_grouped_by_plate_uses_existing_plate_column():
    frame = _frame().assign(plateID=["x"] * 4 + ["y"] * 5)
    data = bundle._recipe_frame(frame, {"grouped_line": _grouped(
        grouping_column="plateID", representation="plate", order=["y", "x"])})
    assert "rowID" not in data.columns
    values = dict(zip(data["plateID"].astype(str), data["signal"]))
    assert values == {"x": pytest.approx(5.0), "y": pytest.approx(110.0)}


def test_invalid_representation_is_rejected():
    with pytest.raises(ValueError, match="Invalid grouped line representation"):
        bundle._recipe_frame(_frame(), {"grouped_line": _grouped(
            representation="field")})


def test_remove_outliers_drops_iqr_outliers_per_group_and_column():
    frame = pd.DataFrame({
        "condition": ["a"] * 6 + ["b"] * 6,
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
        "signal": [1.0, 2.0, 2.0, 3.0, 2.0, 50.0,
                   5.0, 5.0, 6.0, 6.0, 5.0, -40.0],
    })
    data = bundle._recipe_frame(frame, {"grouped_line": _grouped(
        order=["a", "b"], remove_outliers=True)})
    assert len(data) == 10
    assert 50.0 not in data["signal"].tolist()
    assert -40.0 not in data["signal"].tolist()


def test_log_axes_transform_only_selected_columns_outside_group_mode():
    frame = _frame().dropna()
    data = bundle._recipe_frame(frame, {"grouped_line": _grouped(log_y=True)})
    assert data["time"].tolist() == frame["time"].tolist()
    assert data["signal"].tolist() == pytest.approx(np.log10(frame["signal"]).tolist())
    both = bundle._recipe_frame(frame, {"grouped_line": _grouped(log_x=True)})
    assert both["time"].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0] * 2)
    assert both["signal"].tolist() == frame["signal"].tolist()


def test_group_mode_ignores_log_flags():
    frame = _frame().dropna()
    data = bundle._recipe_frame(frame, {"grouped_line": _grouped(
        mode="groups", log_x=True, log_y=True)})
    assert data["signal"].tolist() == frame["signal"].tolist()


def test_groups_mode_draws_errorbars_in_recorded_order():
    frame = pd.DataFrame({"condition": ["a", "a", "b", "b"],
                          "signal": [1.0, 3.0, 10.0, 10.0]})
    figure = Figure()
    ax = bundle._draw(figure, frame, {
        "kind": "line", "x": "condition", "y": "signal",
        "grouped_line": _grouped(mode="groups", columns=["signal"])})
    line = ax.lines[0]
    assert list(line.get_xdata()) == [0, 1]
    assert list(line.get_ydata()) == pytest.approx([10.0, 2.0])
    assert line.get_color() == "#112233"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["b", "a"]


def test_std_area_mode_draws_mean_line_and_sd_band():
    frame = pd.DataFrame({"condition": ["a"] * 4, "time": [1.0, 1.0, 2.0, 2.0],
                          "signal": [1.0, 3.0, 4.0, 8.0]})
    figure = Figure()
    ax = bundle._draw(figure, frame, {
        "kind": "line", "x": "time", "y": "signal",
        "grouped_line": _grouped(mode="std_area", order=["a"],
                                 band_color="#aa0000", band_width=2.5)})
    line = ax.lines[0]
    assert line.get_label() == "mean_signal"
    assert list(line.get_ydata()) == pytest.approx([2.0, 6.0])
    assert line.get_linewidth() == pytest.approx(2.5)
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)
             and c.get_alpha() == pytest.approx(0.25)]
    assert len(bands) == 1
    assert tuple(bands[0].get_facecolor()[0][:3]) == pytest.approx(
        matplotlib.colors.to_rgb("#aa0000"))
    vertices = bands[0].get_paths()[0].vertices
    sd_first = np.std([1.0, 3.0], ddof=1)
    assert vertices[:, 1].min() == pytest.approx(2.0 - sd_first)


def test_grouped_line_mode_draws_one_mean_line_per_group():
    frame = _frame().dropna()
    figure = Figure()
    ax = bundle._draw(figure, frame, {
        "kind": "line", "x": "time", "y": "signal", "hue": "condition",
        "grouped_line": _grouped(order=["b", "a"])})
    drawn = {tuple(np.round(l.get_ydata(), 6)) for l in ax.lines if len(l.get_ydata()) == 2}
    assert (20.0, 200.0) in drawn
    assert (3.0, 7.0) in drawn


def _training_frame():
    return pd.DataFrame({
        "curve": ["train"] * 3 + ["val"] * 3 + ["fold"] * 3,
        "epoch": [1, 2, 3] * 3,
        "loss": [0.9, 0.4, 0.6, 0.8, 0.7, 0.75, 0.5, 0.3, 0.2],
        "loss__sd": [0.1, 0.1, 0.1, np.nan, np.nan, np.nan, 0.05, 0.05, 0.05],
    })


def _training(**overrides):
    recipe = {"band": True, "mark_best": True, "direction": "min", "curves": [
        {"label": "train", "kind": "mean", "color": "#ff0000", "linestyle": "-",
         "linewidth": 2.0, "alpha": 0.9},
        {"label": "val", "kind": "mean", "color": "#00ff00", "alpha": 0.8,
         "marker": "s", "markersize": 4},
        {"label": "fold", "kind": "fold", "color": "#0000ff", "alpha": 0.5,
         "band_style": {"color": "#000000"},
         "best_style": {"color": "#123456", "marker": "x", "linestyle": "None"}},
    ], "notes": [{"x": 0.1, "y": 0.9, "text": "best 0.2", "fontsize": 7}]}
    recipe.update(overrides)
    return recipe


def _draw_training(recipe, frame=None):
    figure = Figure()
    return bundle._draw(figure, _training_frame() if frame is None else frame, {
        "kind": "line", "x": "epoch", "y": "loss", "hue": "curve",
        "training_curves": recipe})


def test_training_curves_draw_lines_bands_best_marks_and_notes():
    ax = _draw_training(_training())
    curves = {l.get_label(): l for l in ax.lines if not l.get_label().startswith("_")}
    assert list(curves) == ["train", "val", "fold"]
    assert list(curves["train"].get_ydata()) == pytest.approx([0.9, 0.4, 0.6])
    assert curves["train"].get_linewidth() == pytest.approx(2.0)
    assert curves["val"].get_marker() == "s"
    best = [l for l in ax.lines if l.get_label().startswith("_")]
    assert [(float(l.get_xdata()[0]), float(l.get_ydata()[0])) for l in best] == [
        (2.0, 0.4), (2.0, 0.7), (3.0, 0.2)]
    assert best[0].get_marker() == "o"
    assert best[0].get_alpha() == pytest.approx(0.9)
    assert best[2].get_marker() == "x"
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(bands) == 1
    ys = bands[0].get_paths()[0].vertices[:, 1]
    assert ys.min() == pytest.approx(0.3)
    assert ys.max() == pytest.approx(1.0)
    assert [t.get_text() for t in ax.texts] == ["best 0.2"]
    assert ax.texts[0].get_transform() == ax.transAxes
    assert ax.texts[0].get_fontsize() == 7
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["train", "val", "fold"]


def test_training_curves_max_direction_picks_highest_finite_value():
    frame = _training_frame()
    frame.loc[0, "loss"] = np.nan
    ax = _draw_training(_training(band=False, notes=[]), frame)
    best = [l for l in ax.lines if l.get_label().startswith("_")]
    assert [float(l.get_ydata()[0]) for l in best][:1] == [0.4]
    ax = _draw_training(_training(band=False, direction="max"), frame)
    best = [l for l in ax.lines if l.get_label().startswith("_")]
    assert [float(l.get_ydata()[0]) for l in best] == [0.6, 0.8, 0.5]
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_training_curves_skip_marks_without_direction_or_finite_values():
    frame = _training_frame()
    frame.loc[frame["curve"] == "val", "loss"] = np.nan
    ax = _draw_training(_training(direction=None), frame)
    assert not [l for l in ax.lines if l.get_label().startswith("_")]
    ax = _draw_training(_training(mark_best=False), frame)
    assert not [l for l in ax.lines if l.get_label().startswith("_")]
    ax = _draw_training(_training(), frame)
    best = [l for l in ax.lines if l.get_label().startswith("_")]
    assert len(best) == 2


def test_training_band_needs_sd_column():
    frame = _training_frame().drop(columns="loss__sd")
    ax = _draw_training(_training(), frame)
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len([l for l in ax.lines if not l.get_label().startswith("_")]) == 3


def test_training_without_curves_draws_notes_only():
    ax = _draw_training(_training(curves=[]))
    assert len(ax.lines) == 0
    assert ax.get_legend() is None
    assert [t.get_text() for t in ax.texts] == ["best 0.2"]
    assert not ax.xaxis._major_tick_kw.get("gridOn", False)


def test_plain_line_keeps_raw_rows_in_recorded_order():
    frame = pd.DataFrame({"time": [3.0, 1.0, 2.0], "signal": [30.0, 10.0, 20.0]})
    figure = Figure()
    ax = bundle._draw(figure, frame, {"kind": "line", "x": "time", "y": "signal",
                                      "line_options": {"linewidth": 3.0}})
    line = ax.lines[0]
    assert list(line.get_xdata()) == [3.0, 1.0, 2.0]
    assert list(line.get_ydata()) == [30.0, 10.0, 20.0]
    assert line.get_linewidth() == pytest.approx(3.0)
