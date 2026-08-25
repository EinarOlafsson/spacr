"""Figure settings: reading a chart back out of its artists, and the guards.

``derive_replot_recipe`` exists because most spaCR figures arrive with no
record of the data behind them, so everything here is asserted against a real
matplotlib figure -- bars, points and lines that were actually drawn.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


def _bar_figure():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.bar(["a", "b", "c"], [3.0, 5.0, 4.0])
    return figure, axes


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_every_shape_matplotlib_stores_a_colour_in_becomes_a_hex():
    assert fs._as_hex("red") == "#ff0000"
    assert fs._as_hex((1.0, 0.0, 0.0, 1.0)) == "#ff0000"
    assert fs._as_hex(np.array([[1.0, 0.0, 0.0, 1.0]])) == "#ff0000"
    assert fs._as_hex([(1.0, 0.0, 0.0, 1.0)]) == "#ff0000"
    assert fs._as_hex([[0.0, 0.0, 1.0, 1.0]]) == "#0000ff"


def test_a_colour_nothing_can_read_falls_back():
    assert fs._as_hex(object(), fallback="#123456") == "#123456"


# ---------------------------------------------------------------------------
# Tick labels
# ---------------------------------------------------------------------------

def test_the_tick_labels_are_read_off_the_axes():
    figure, axes = _bar_figure()
    figure.canvas.draw()
    labels = fs._tick_labels(axes)
    assert set(labels.values()) == {"a", "b", "c"}


def test_an_axes_that_cannot_be_asked_has_no_tick_labels():
    class Awkward:
        def get_xticks(self):
            raise RuntimeError("no renderer")

    assert fs._tick_labels(Awkward()) == {}


def test_a_position_with_no_tick_label_is_named_by_its_number():
    labels = {0.0: "a", 1.0: "b"}
    assert fs._named(labels, 0.0) == "a"
    assert fs._named(labels, 1.2) == "b"      # within half a tick
    assert fs._named(labels, 7.0) == "7"      # nothing near it
    assert fs._named({}, 2.5) == "2.5"


# ---------------------------------------------------------------------------
# Reading the data back out of the artists
# ---------------------------------------------------------------------------

def test_bars_give_up_their_group_and_their_height():
    figure, axes = _bar_figure()
    figure.canvas.draw()
    pairs = fs._pairs_from_axes(axes)
    assert sorted(pairs) == [("a", 3.0), ("b", 4.0), ("c", 5.0)] or \
        sorted(pairs) == [("a", 3.0), ("b", 5.0), ("c", 4.0)]


def test_a_background_span_and_a_non_rectangle_are_not_data():
    figure, axes = _bar_figure()
    low, high = axes.get_xlim()
    # A patch as wide as the axes is a background, not a bar.
    axes.add_patch(Rectangle((low, 0.0), high - low, 1.0))
    axes.add_patch(Circle((0.5, 0.5), 0.1))  # not a Rectangle
    figure.canvas.draw()
    assert len(fs._pairs_from_axes(axes)) == 3


def test_a_bar_with_no_finite_height_is_left_out():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.set_xlim(-0.5, 2.5)
    axes.add_patch(Rectangle((0.4, 0.0), 0.2, float("nan")))
    axes.add_patch(Rectangle((1.4, 0.0), 0.2, 2.0))
    figure.canvas.draw()
    assert fs._pairs_from_axes(axes) == [("1.5", 2.0)]


def test_an_empty_scatter_and_an_empty_line_contribute_nothing():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.scatter([], [])
    axes.plot([], [])
    figure.canvas.draw()
    assert fs._pairs_from_axes(axes) == []


def test_a_two_point_reference_line_is_not_data_but_a_marked_one_is():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [5, 5])                       # a reference line
    figure.canvas.draw()
    assert fs._pairs_from_axes(axes) == []

    axes.plot([0, 1], [2, 3], marker="o")           # two marked points
    figure.canvas.draw()
    assert len(fs._pairs_from_axes(axes)) == 2


def test_scatter_points_are_read_with_their_group():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["a", "b"])
    axes.scatter([0, 0, 1], [1.0, 2.0, 3.0])
    figure.canvas.draw()
    assert sorted(fs._pairs_from_axes(axes)) == [
        ("a", 1.0), ("a", 2.0), ("b", 3.0)]


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

def test_a_figure_of_bars_yields_a_recipe_that_can_be_redrawn():
    figure, _axes = _bar_figure()
    figure.canvas.draw()
    recipe = fs.derive_replot_recipe(figure)
    assert recipe is not None
    assert set(recipe["df"].columns) == {recipe["grouping_column"],
                                         recipe["data_column"]}
    assert len(recipe["df"]) == 3
    assert recipe["summary_func"] == "mean"


def test_a_grid_of_panels_is_not_redrawn_as_one():
    figure = Figure()
    for index in range(2):
        axes = figure.add_subplot(1, 2, index + 1)
        axes.bar(["a", "b"], [1.0, 2.0])
    figure.canvas.draw()
    assert fs.derive_replot_recipe(figure) is None
    assert fs.derive_replot_recipe(Figure()) is None


def test_a_figure_with_almost_nothing_drawn_has_no_recipe():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.bar(["a"], [1.0])
    figure.canvas.draw()
    assert fs.derive_replot_recipe(figure) is None


def test_without_pandas_no_recipe_can_be_built(monkeypatch):
    figure, _axes = _bar_figure()
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert fs.derive_replot_recipe(figure) is None


def test_a_recipe_with_no_rows_fits_no_type_in_particular():
    import pandas as pd

    assert fs._which_types_fit({"df": None}) == ((), {})
    assert fs._which_types_fit({"df": pd.DataFrame()}) == ((), {})


def test_a_recipe_that_cannot_be_read_fits_nothing():
    assert fs._which_types_fit({}) == ((), {})
    assert fs._which_types_fit(object()) == ((), {})


def test_a_real_recipe_names_the_types_that_fit_it():
    import pandas as pd

    frame = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5,
                          "value": list(range(10))})
    fits, why = fs._which_types_fit({"df": frame, "grouping_column": "group",
                                     "data_column": "value"})
    assert fits or why
    if "bar_jitter" in fits:
        assert "jitter_bar" in fits and "jitter_box" in fits


# ---------------------------------------------------------------------------
# Naming a figure
# ---------------------------------------------------------------------------

def test_a_figure_is_named_by_its_suptitle_then_by_its_axes():
    figure = Figure()
    axes = figure.add_subplot(111)
    assert fs._figure_title(figure) == ""
    axes.set_title("per-plate rate")
    assert fs._figure_title(figure) == "per-plate rate"
    figure.suptitle("the whole screen")
    assert fs._figure_title(figure) == "the whole screen"


def test_something_that_is_not_a_figure_has_no_title():
    assert fs._figure_title(object()) == ""


# ---------------------------------------------------------------------------
# Text size
# ---------------------------------------------------------------------------

def test_the_size_control_opens_at_the_size_the_figure_uses():
    """The mode, so one big heading does not speak for the whole figure."""
    figure = Figure()
    figure.suptitle("title", fontsize=18)
    for index in range(3):
        figure.text(0.1 * index, 0.5, f"body {index}", fontsize=9)
    assert fs._current_text_size(figure) == 9


def test_a_tie_opens_at_the_smaller_size():
    figure = Figure()
    figure.text(0.1, 0.5, "one", fontsize=14)
    figure.text(0.2, 0.5, "two", fontsize=8)
    assert fs._current_text_size(figure) == 8


def test_a_figure_with_no_text_opens_at_the_default():
    assert fs._current_text_size(Figure(), default=11) == 11


def test_text_whose_size_cannot_be_read_is_skipped(monkeypatch):
    class Awkward:
        def get_text(self):
            return "something"

        def get_fontsize(self):
            raise RuntimeError("no font")

    monkeypatch.setattr(fs, "_every_text", lambda _figure: [Awkward()])
    assert fs._current_text_size(Figure(), default=12) == 12


# ---------------------------------------------------------------------------
# A house style as a file
# ---------------------------------------------------------------------------

def test_a_style_with_no_path_is_not_written():
    assert fs.save_graph_style("", {"font_size": 12}) == ""


def test_a_style_that_cannot_be_written_says_nothing_was(tmp_path):
    target = tmp_path / "no such folder" / "style.json"
    assert fs.save_graph_style(str(target), {"font_size": 12}) == ""


def test_a_saved_style_round_trips(tmp_path):
    target = tmp_path / "style.json"
    written = fs.save_graph_style(str(target), {"font_size": 12},
                                  {"bar": {"grid_width": 0.6}})
    assert written == str(target)
    general, per_graph = fs.load_graph_style(str(target))
    assert general["font_size"] == 12
    assert per_graph["bar"]["grid_width"] == pytest.approx(0.6)
    assert json.loads(target.read_text())


# ---------------------------------------------------------------------------
# Style vocabulary
# ---------------------------------------------------------------------------

def test_a_setting_is_labelled_by_its_own_name():
    assert fs.style_setting_label("grid_colour") == "Grid colour"
    assert fs.style_setting_label("aspect") == "Graph shape"


def test_a_colour_and_a_transparent_ground_are_told_apart():
    assert fs._looks_like_a_colour("#ff0000")
    assert not fs._looks_like_a_colour("red")
    assert not fs._looks_like_a_colour(3)
    assert fs._is_transparent_ground("none")
    assert fs._is_transparent_ground(" TRANSPARENT ")
    assert fs._is_transparent_ground("")
    assert not fs._is_transparent_ground("#000000")


def test_a_free_form_setting_offers_no_choices():
    assert fs.style_choices_for("not_a_setting") == ()
    assert isinstance(fs.style_choices_for("palette"), tuple)


def test_a_default_is_told_from_an_edit_within_a_spinboxs_precision():
    assert fs._same_setting(True, True)
    assert not fs._same_setting(False, True)
    assert fs._same_setting(0.6000000000000001, 0.6)
    assert not fs._same_setting(0.7, 0.6)
    assert fs._same_setting("deep", "deep")
    assert not fs._same_setting("muted", "deep")


# ---------------------------------------------------------------------------
# The preferences panel
# ---------------------------------------------------------------------------

def test_reset_puts_every_control_back_to_the_package_default(qtbot):
    panel = fs.FigureStylePreferences()
    qtbot.addWidget(panel)
    before = panel.values()
    for name, (getter, setter, default) in panel._general_controls.items():
        if isinstance(default, bool):
            setter(not default)
        elif isinstance(default, (int, float)):
            setter(default + 1)
    changed, _per_graph = panel.values()
    assert changed
    panel.reset()
    assert panel.values() == before
