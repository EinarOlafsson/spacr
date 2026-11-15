"""Figure settings: reading a chart back out of its artists, and the house style.

Most spaCR figures arrive with no record of the data behind them, so
``derive_replot_recipe`` reads groups and values off real bars, points and
lines, ignores what is not data (background spans, circles, NaN bars,
two-point reference lines, empty series), refuses grids and near-empty
figures, and ``_which_types_fit`` names exactly the graph types the drawer can
use for the recipe (with the jitter aliases following ``bar_jitter``).  The
text-size control opens at the modal size, and a house style round-trips
through a JSON file or reports that nothing was written.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import spacr.graph_types as graph_types  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


def _axes(draw):
    figure = Figure()
    axes = figure.add_subplot(111)
    draw(axes)
    figure.canvas.draw()
    return figure, axes


def _bars(axes):
    axes.bar(["a", "b", "c"], [3.0, 5.0, 4.0])


def _bars_with_noise(axes):
    _bars(axes)
    low, high = axes.get_xlim()
    axes.add_patch(Rectangle((low, 0.0), high - low, 1.0))
    axes.add_patch(Circle((0.5, 0.5), 0.1))


def _nan_bar(axes):
    axes.set_xlim(-0.5, 2.5)
    axes.add_patch(Rectangle((0.4, 0.0), 0.2, float("nan")))
    axes.add_patch(Rectangle((1.4, 0.0), 0.2, 2.0))


def _empty_series(axes):
    axes.scatter([], [])
    axes.plot([], [])


def _reference_line(axes):
    axes.plot([0, 1], [5, 5])


def _reference_and_marked(axes):
    axes.plot([0, 1], [5, 5])
    axes.plot([0, 1], [2, 3], marker="o")


def _labelled_scatter(axes):
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["a", "b"])
    axes.scatter([0, 0, 1], [1.0, 2.0, 3.0])


def test_pairs_from_axes_reads_only_the_artists_that_are_data(subtests):
    """Bars, scatter points and marked lines become (group, value) pairs; spans, circles, NaNs and reference lines do not."""
    cases = {
        "bars": (_bars, [("a", 3.0), ("b", 5.0), ("c", 4.0)]),
        "background_span_and_circle_ignored": (
            _bars_with_noise, [("a", 3.0), ("b", 5.0), ("c", 4.0)]),
        "nan_bar_left_out": (_nan_bar, [("1.5", 2.0)]),
        "empty_scatter_and_line": (_empty_series, []),
        "two_point_reference_line": (_reference_line, []),
        "marked_line_is_data": (_reference_and_marked, [("0.0", 2.0), ("1.0", 3.0)]),
        "scatter_with_groups": (_labelled_scatter, [("a", 1.0), ("a", 2.0), ("b", 3.0)]),
    }
    for case_id, (draw, pairs) in cases.items():
        with subtests.test(msg=case_id):
            _figure, axes = _axes(draw)
            assert fs._pairs_from_axes(axes) == pairs


def test_derive_replot_recipe_rebuilds_one_panel_of_bars_or_nothing(monkeypatch, subtests):
    """A single bar panel yields an exact recipe; grids, blank figures, one bar and missing pandas yield None."""
    with subtests.test(msg="bars_yield_a_recipe"):
        figure, _axes_ = _axes(_bars)
        recipe = fs.derive_replot_recipe(figure)
        assert recipe["df"].to_dict("list") == {"group": ["a", "b", "c"],
                                                "value": [3.0, 5.0, 4.0]}
        assert {k: v for k, v in recipe.items() if k != "df"} == {
            "grouping_column": "group", "data_column": "value",
            "graph_type": "", "summary_func": "mean", "order": None,
            "colors": None, "y_lim": None, "error_bar_type": "std"}
    with subtests.test(msg="named_bars_give_named_groups"):
        figure, _axes_ = _axes(lambda ax: ax.bar(["ctrl", "treat"], [3.0, 5.0]))
        assert sorted(fs.derive_replot_recipe(figure)["df"]["group"]) == ["ctrl", "treat"]
    with subtests.test(msg="grid_of_panels"):
        figure = Figure()
        for index in range(2):
            figure.add_subplot(1, 2, index + 1).bar(["a", "b"], [1.0, 2.0])
        figure.canvas.draw()
        assert fs.derive_replot_recipe(figure) is None
    with subtests.test(msg="blank_figure"):
        assert fs.derive_replot_recipe(Figure()) is None
    with subtests.test(msg="one_bar"):
        figure, _axes_ = _axes(lambda ax: ax.bar(["a"], [1.0]))
        assert fs.derive_replot_recipe(figure) is None
    with subtests.test(msg="no_pandas"):
        figure, _axes_ = _axes(_bars)
        monkeypatch.setitem(sys.modules, "pandas", None)
        assert fs.derive_replot_recipe(figure) is None


def test_which_types_fit_names_drawable_types_and_carries_reasons(monkeypatch, subtests):
    """Unreadable or empty recipes fit nothing; a real one gets the exact type list; aliases follow bar_jitter."""
    for case_id, recipe in (("df_none", {"df": None}), ("empty_df", {"df": pd.DataFrame()}),
                            ("empty_mapping", {}), ("not_a_mapping", object())):
        with subtests.test(msg=case_id):
            assert fs._which_types_fit(recipe) == ((), {})
    with subtests.test(msg="real_recipe"):
        frame = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "value": list(range(10))})
        assert fs._which_types_fit({"df": frame, "grouping_column": "group",
                                    "data_column": "value"}) == (
            ("bar", "bar_jitter", "box_jitter", "jitter", "box", "violin",
             "jitter_bar", "jitter_box"),
            {"line": "a line through unordered categories is a row of markers "
                     "joined for no reason",
             "scatter": "a scatter of one categorical against one continuous "
                        "is a jitter under another name"})
    small = {"df": pd.DataFrame({"group": ["a", "a", "b", "b"],
                                 "value": [1.0, 2.0, 3.0, 4.0]}),
             "grouping_column": "group", "data_column": "value"}
    offers = {
        "alias_offered_with_bar_jitter": (
            [("bar", "Bar", ""), ("bar_jitter", "Bar with jitter", "")],
            (("bar", "bar_jitter", "jitter_bar", "jitter_box"), {})),
        "alias_carries_the_refusal": (
            [("bar", "Bar", ""), ("bar_jitter", "Bar with jitter", "needs the observations")],
            (("bar",), {"bar_jitter": "needs the observations",
                        "jitter_bar": "needs the observations",
                        "jitter_box": "needs the observations"})),
        "no_alias_for_an_absent_type": (
            [("bar", "Bar", ""), ("violin", "Violin", "too few points")],
            (("bar",), {"violin": "too few points"})),
    }
    for case_id, (offer, expected) in offers.items():
        with subtests.test(msg=case_id):
            monkeypatch.setattr(graph_types, "offer",
                                lambda f, x="", y="", _o=offer: _o)
            assert fs._which_types_fit(small) == expected


def test_current_text_size_and_house_style_file(monkeypatch, tmp_path, subtests):
    """The size control opens at the smallest modal size or the default; a style round-trips or returns ''."""
    def texts(*sizes):
        figure = Figure()
        for index, size in enumerate(sizes):
            figure.text(0.1 * index, 0.5, f"t{index}", fontsize=size)
        return figure

    for case_id, figure, default, size in (
            ("mode_wins_over_one_heading", texts(18, 9, 9, 9), 11, 9),
            ("tie_opens_smaller", texts(14, 8), 11, 8),
            ("no_text_opens_default", Figure(), 11, 11)):
        with subtests.test(msg=case_id):
            assert fs._current_text_size(figure, default=default) == size

    with subtests.test(msg="unreadable_size_skipped"):
        class Awkward:
            def get_text(self):
                return "something"

            def get_fontsize(self):
                raise RuntimeError("no font")

        monkeypatch.setattr(fs, "_every_text", lambda _figure: [Awkward()])
        assert fs._current_text_size(Figure(), default=12) == 12
        monkeypatch.undo()

    for case_id, path in (("no_path", ""),
                          ("unwritable", str(tmp_path / "no such folder" / "style.json"))):
        with subtests.test(msg=case_id):
            assert fs.save_graph_style(path, {"font_size": 12}) == ""
    with subtests.test(msg="round_trip"):
        target = tmp_path / "style.json"
        assert fs.save_graph_style(str(target), {"font_size": 12},
                                   {"bar": {"grid_width": 0.6}}) == str(target)
        general, per_graph = fs.load_graph_style(str(target))
        assert general["font_size"] == 12
        assert per_graph["bar"]["grid_width"] == pytest.approx(0.6)
        assert json.loads(target.read_text(encoding="utf-8"))
