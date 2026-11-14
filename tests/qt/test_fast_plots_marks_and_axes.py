"""FastPlot's marks, keys, scales and print repaint when something is missing.

Every case here is one a user walks into without doing anything wrong: no
pyqtgraph, a group of blanks, a violin with no spread, a key never plotted, a
log scale over a bar, a split above every point, a shape probe that raises, an
axis the plot never built.  The contract is always the same -- draw what CAN be
drawn and say how much that was -- so each test asks what a user would see.
One general test per unit, each walking a case table with ``subtests``.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtWidgets import QLabel, QMenu  # noqa: E402

from spacr.figure_style import saved_figure_appearance as _real_look  # noqa: E402
from spacr.qt.widgets import fast_plots as fp  # noqa: E402

pytestmark = pytest.mark.qt

_ATTRIBUTES = (
    "plot", "_background", "_foreground", "_status", "_headline", "_note",
    "_labels", "_legend_colours", "_items", "_keys", "_key_rows", "_row_xy",
    "_selected_key", "_selected_keys", "_extra_highlights", "_highlight",
    "_refit", "_baselines", "_compartments", "_corrections", "_encodings",
    "_p_values", "_thresholds", "_marks", "_frame", "_style_note",
    "_legend_box",
)


@pytest.fixture
def make(qtbot):
    """A FastPlot with both axes named (or as given), nothing drawn yet."""
    def build(bare=False, **kwargs):
        if not (bare or kwargs):
            kwargs = dict(title="fast", x_label="ex", y_label="why")
        widget = fp.FastPlot(**kwargs)
        qtbot.addWidget(widget)
        return widget
    return build


def _items(plot, kind):
    return [item for item in plot.plot.plotItem.items if isinstance(item, kind)]


def test_the_plot_builds_and_closes_whatever_is_missing(make, monkeypatch, subtests):
    """No pyqtgraph still builds a labelled stand-in; a failing cleanup still closes."""
    with subtests.test(msg="without_pyqtgraph_it_builds_and_says_what_to_install"):
        with monkeypatch.context() as mp:
            mp.setattr(fp, "HAVE_PYQTGRAPH", False)
            plot = make(title="Guide effects")
        texts = " ".join(label.text() for label in plot.findChildren(QLabel))
        assert "Guide effects" in texts and fp.PYQTGRAPH_MISSING_MESSAGE in texts
        assert [name for name in _ATTRIBUTES if not hasattr(plot, name)] == []
        assert plot._legend_box.isEnabled() is False
        assert plot._thresholds == ([], None, None)
        assert list(plot.plot.listDataItems()) == []
        assert bool(plot.plot) is False
        assert plot.plot.scene().sigMouseClicked is not None
        assert "absent" in repr(plot.plot)
    with subtests.test(msg="without_pyqtgraph_an_untitled_plot_shows_only_the_notice"):
        with monkeypatch.context() as mp:
            mp.setattr(fp, "HAVE_PYQTGRAPH", False)
            plot = make(bare=True)
        assert [label.text() for label in plot.findChildren(QLabel)] == [
            fp.PYQTGRAPH_MISSING_MESSAGE, ""]
    with subtests.test(msg="a_plot_closes_when_its_menus_cannot_be_retired"):
        from spacr.qt import widget_cleanup

        def _gone(_widget):
            raise RuntimeError("wrapped C/C++ object already deleted")

        plot = make()
        with monkeypatch.context() as mp:
            mp.setattr(widget_cleanup, "retire_pyqtgraph_menus", _gone)
            assert plot.close() is True
        assert plot.isVisible() is False


def test_a_group_mark_draws_what_its_values_support(make, subtests):
    """``add_group_mark`` counts what it represented and draws only honest marks."""
    with subtests.test(msg="an_all_blank_group_draws_nothing"):
        for mark in ("bar", None):
            plot = make()
            before = len(plot.plot.plotItem.items)
            args = (0.0, [np.nan, np.nan]) + ((mark,) if mark else ())
            assert plot.add_group_mark(*args) == 0
            assert len(plot.plot.plotItem.items) == before
        assert plot.add_group_mark(1.0, [1.0, 3.0], "bar") == 2
        assert len(plot.plot.plotItem.items) > before
    for case_id, values in (("a_flat_violin_of_twelve", [2.0] * 12),
                            ("a_flat_violin_of_four", [2.5] * 4)):
        with subtests.test(msg=case_id):
            plot = make()
            spread = list(np.linspace(0.0, 1.0, len(values)))
            assert plot.add_group_mark(1.0, spread, "violin") == len(values)
            assert len(_items(plot, pg.PlotCurveItem)) == 1
            assert plot.add_group_mark(3.0, values, "violin") == len(values)
            assert len(_items(plot, pg.PlotCurveItem)) == 1
            assert len(_items(plot, pg.ScatterPlotItem)) == 1
    for name in ("swarm", "candlestick"):
        with subtests.test(msg=f"an_unknown_mark_{name}_lists_the_known_ones"):
            plot = make()
            assert plot.add_group_mark(0.0, [1.0, 2.0], "box") == 2
            with pytest.raises(ValueError) as raised:
                plot.add_group_mark(0.0, [1.0, 2.0, 3.0], name)
            assert name in str(raised.value) and "violin" in str(raised.value)

    def added(plot, *args, **kwargs):
        before = len(plot.plot.plotItem.items)
        count = plot.add_group_mark(*args, **kwargs)
        return count, len(plot.plot.plotItem.items) - before

    with subtests.test(msg="a_group_of_one_gets_no_interval"):
        plot = make()
        n_one, alone = added(plot, 0.0, [4.0], "line")
        n_three, together = added(plot, 1.0, [4.0, 6.0, 5.0], "line")
        assert (n_one, n_three, together - alone) == (1, 3, 1)
    with subtests.test(msg="an_overflowing_spread_gets_no_whisker"):
        plot = make()
        with np.errstate(over="ignore", invalid="ignore"):
            n_over, overflowed = added(plot, 0.0, [1e308, -1e308], "bar", spread="var")
        n_ok, measurable = added(plot, 1.0, [1.0, 3.0], "bar", spread="var")
        assert (n_over, n_ok, measurable - overflowed) == (2, 2, 1)


def test_exported_tables_exist_only_for_what_was_drawn(make, subtests):
    """``curve_frame`` / ``ranked_frame`` are None until a finite mark is drawn."""
    with subtests.test(msg="curves"):
        plot = make()
        assert plot.curve_frame() is None
        for xs, ys in (([np.nan, 1.0], [2.0, np.nan]), ([np.nan, 1.0], [np.nan, np.nan])):
            assert plot.add_curve(xs, ys) == 0
            assert plot.curve_frame() is None
        assert plot.add_curve([1.0, 2.0], [3.0, 4.0]) == 2
        assert plot.curve_frame().to_dict("list") == {"x": [1.0, 2.0], "y": [3.0, 4.0]}
    with subtests.test(msg="ranked_bars"):
        plot = make()
        assert plot.add_ranked_bars(["a", "b"], [np.nan, np.nan]) == 0
        assert plot.ranked_frame() is None
        assert plot.add_ranked_bars(["a", "b"], [1.0, 2.0]) == 2
        assert list(plot.ranked_frame()["name"]) == ["b", "a"]
    with subtests.test(msg="a_sweep_curve_with_its_band_and_threshold_line"):
        sizes = np.arange(2, 40, 4, dtype=float)
        middle = 1.0 / sizes
        plot = make(title="sweep", x_label="Sample size", y_label="Mean absolute difference")
        assert plot.add_curve(sizes, middle, low=middle * 0.8, high=middle * 1.2) == len(sizes)
        plot.add_line(x=15.0, label="minimum cell count")
        marks = [i for i in plot.plot.items() if isinstance(i, pg.InfiniteLine)]
        assert [float(m.value()) for m in marks] == [15.0]
        assert len([i for i in plot.plot.items() if isinstance(i, pg.FillBetweenItem)]) == 1
        assert plot.curve_frame().shape == (len(sizes), 2)


def test_a_selection_keeps_its_keys_and_rings_only_drawn_points(make, subtests):
    """Clicks, repeats and undrawn keys: membership is kept, rings are counted."""
    with subtests.test(msg="a_click_on_empty_space_keeps_the_selection"):
        plot = make()
        item = plot.add_scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        plot.set_keys(["a", "b"])
        plot._on_points_clicked(item, [item.points()[1]])
        assert plot.selected_keys() == ["b"]
        plot._on_points_clicked(item, [])
        assert plot.selected_keys() == ["b"]
    for case_id, keys, rings, selected in (
            ("a_repeated_key_is_one_member", ["a", "a", "b"], 2, ["a", "b"]),
            ("undrawn_members_are_kept_not_ringed", ["c", "zz", "b"], 1, ["c", "zz", "b"])):
        with subtests.test(msg=case_id):
            plot = make()
            plot.set_keys(["a", "b", "c"])
            plot.add_scatter(np.array([0.0, 1.0]), np.array([0.0, 1.0]), rows=[0, 1])
            assert plot.highlight_keys(keys) == rings
            assert plot.selected_keys() == selected


def _points(make):
    plot = make()
    plot.add_scatter(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 300.0]), size=6)
    return plot


def test_the_log_scale_moves_every_live_mark_and_its_labels(make, subtests):
    """``set_log_axes`` transforms curves and bar edges and relabels named axes."""
    with subtests.test(msg="live_curves_move_and_stale_records_drop"):
        plot = _points(make)
        gone = plot.plot.plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        curve = plot.plot.plot(np.array([1.0, 2.0, 3.0]), np.array([10.0, 100.0, 1000.0]))
        plot.plot.addItem(pg.PlotDataItem())
        assert len(plot._drawn) == 3
        plot.plot.removeItem(gone)
        assert plot.set_log_axes(y=True) == (False, True)
        assert [entry.kind for entry in plot._drawn] == ["points", "line"]
        assert np.allclose(curve.getData()[1], [1.0, 2.0, 3.0])
        assert np.allclose(plot._drawn[-1].y, [10.0, 100.0, 1000.0])
    with subtests.test(msg="bars_move_by_all_four_edges"):
        plot = make()
        bar = pg.BarGraphItem(x=[1.0, 2.0], height=[3.0, 4.0], width=0.5, y0=[1.0, 1.0])
        plot.plot.addItem(bar)
        assert plot.set_log_axes(y=True) == (False, True)
        assert np.allclose(bar.opts["y0"], [0.0, 0.0])
        assert np.allclose(bar.opts["y1"], np.log10([4.0, 5.0]))
        assert np.allclose(bar.opts["x0"], [0.75, 1.75])
        assert np.allclose(bar.opts["x1"], [1.25, 2.25])
    for case_id, kwargs, flags, expected in (
            ("an_unnamed_axis_grows_no_label", dict(bare=True), dict(x=False, y=False), ""),
            ("a_named_axis_says_it_is_logged", dict(x_label="effect size"), dict(x=True),
             "effect size (log scale)")):
        with subtests.test(msg=case_id):
            plot = make(**kwargs)
            assert plot.set_log_axes(**flags) == (flags.get("x", False), flags.get("y", False))
            assert plot.plot.getAxis("bottom").labelText == expected
    with subtests.test(msg="font_size_and_colour_survive_a_scale_change"):
        class _LegendText:
            def __init__(self):
                self.text, self.colour = "sgRNA", None

            def setText(self, text, color=None):
                self.text, self.colour = text, color

        plot = _points(make)
        label = _LegendText()
        plot.plot.plotItem.legend = SimpleNamespace(items=[object(), ("sample", label)])
        plot.set_font_size(17)
        plot.set_font_colour("#ff0000")
        assert plot.set_log_axes(y=True) == (False, True)
        left = plot.plot.getAxis("left")
        assert (left.labelStyle["font-size"], left.labelText) == ("17pt", "why (log scale)")
        assert (plot.font_size(), plot.font_colour()) == (17, "#ff0000")
        assert label.colour == "#ff0000"
        assert left.textPen().color().name() == "#ff0000"


def test_a_y_split_is_sized_from_what_it_keeps_and_numbers_data_units(make, subtests):
    """``set_y_split``: the gap is 6% of the kept span, ticks are wrapped once."""
    with subtests.test(msg="a_split_above_everything_leaves_points_alone"):
        plot = make()
        plot.add_scatter(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), size=6)
        assert plot.set_y_split(10.0, 100.0) == ""
        assert plot._split_gap == pytest.approx(2.0 * 0.06)
        assert float(plot._compress(np.array([3.0]))[0]) == 3.0
        assert "10 to 100 is not drawn" in plot._status.text()
    with subtests.test(msg="duplicate_layers_do_not_widen_the_gap"):
        x, y = np.arange(4.0), np.array([1.0, 3.0, 1000.0, 1200.0])
        gaps = []
        for layers in (1, 3):
            plot = make()
            for _ in range(layers):
                plot.add_scatter(x, y, size=6)
            assert plot.set_y_split(10.0, 900.0) == ""
            gaps.append(plot._split_gap)
        assert gaps == pytest.approx([(2.0 + 200.0) * 0.06] * 2)
    with subtests.test(msg="tick_functions_installed_once_and_removed_with_the_split"):
        plot = _points(make)
        assert plot.set_y_split(10.0, 100.0) == ""
        axis = plot.plot.getAxis("left")
        installed = plot._axis_ticks
        wrapped_values, wrapped_strings = axis.tickValues, axis.tickStrings
        assert plot.set_y_split(20.0, 90.0) == ""
        assert plot._axis_ticks is installed
        assert axis.tickValues is wrapped_values
        hidden = float(plot._compress(np.array([200.0]))[0])
        assert axis.tickStrings([hidden], 1.0, 1.0) == ["200"]
        plot.clear_y_split()
        assert axis.tickValues is not wrapped_values
        assert wrapped_values(0.0, 300.0, 400) == axis.tickValues(0.0, 300.0, 400)
        assert (wrapped_strings([0.0, 100.0], 1.0, 1.0)
                == axis.tickStrings([0.0, 100.0], 1.0, 1.0))


class _Spec:
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    group = "g"
    value = "v"
    kind = "bar"


class _EmptySpec(_Spec):
    frame = pd.DataFrame({"g": [], "v": []})


def test_the_show_as_submenu_survives_a_failing_shape_probe(make, monkeypatch, subtests):
    """``_offer_graph_kinds`` offers kinds, and 'Always start with' only for a known shape."""
    from spacr import graph_types

    def refuse(*_a, **_k):
        raise ValueError("that column is not a shape")

    cases = [
        # case_id, spec, shape_of, expected captions (None = no submenu)
        ("a_probe_that_raises_keeps_the_kinds", _Spec, refuse, ["Bar", "Box"]),
        ("a_probe_that_answers_adds_always", _Spec, lambda *a, **k: "one value per group",
         ["Bar", "Box", "always"]),
        ("no_graph_spec_adds_nothing", None, None, None),
        ("an_empty_frame_adds_nothing", _EmptySpec, None, None),
    ]
    for case_id, spec, shape_of, expected in cases:
        with subtests.test(msg=case_id):
            plot = make()
            menu = QMenu()
            with monkeypatch.context() as mp:
                mp.setattr(plot, "graph_spec", (lambda: spec()) if spec else (lambda: None))
                mp.setattr(graph_types, "offer",
                           lambda *_a, **_k: [("bar", "Bar", ""), ("box", "Box", "")])
                if shape_of is not None:
                    mp.setattr(graph_types, "shape_of", shape_of)
                plot._offer_graph_kinds(menu)
            if expected is None:
                assert menu.actions() == []
                continue
            submenus = [a.menu() for a in menu.actions() if a.menu()]
            assert [m.title() for m in submenus] == ["Show as"]
            captions = [a.text() for a in submenus[0].actions() if a.text()]
            assert captions[:2] == ["Bar", "Box"]
            assert any(c.startswith("Always start with") for c in captions) == (
                "always" in expected)


def _style_stub(mp, **names):
    """Put a partial ``spacr.figure_style`` in front of the real one."""
    stub = types.ModuleType("spacr.figure_style")
    for name, value in names.items():
        setattr(stub, name, value)
    mp.setitem(sys.modules, "spacr.figure_style", stub)


class _PlainAxis:
    """An axis with a pen and no ``textPen`` -- older pyqtgraph's shape."""

    def __init__(self, colour: str):
        self._pen = pg.mkPen(colour)

    def pen(self):
        return self._pen

    def setPen(self, pen):
        self._pen = pen


def test_the_print_look_repaints_only_what_it_can_read_and_undo(make, monkeypatch, subtests):
    """``_wear_the_print_look`` returns [] unless a print look is readable."""
    def _black(current, kind, styling):
        return "#000000"

    def _boom():
        raise RuntimeError("the saved look cannot be read")

    print_look = SimpleNamespace(mode="print", ground="#ffffff")
    screen_look = SimpleNamespace(mode="screen", ground="#ffffff")
    stand_downs = [
        ("a_print_look_repaints", dict(saved_figure_appearance=lambda: print_look,
                                       export_colour=_black), True),
        ("the_module_lacks_the_names", {}, False),
        ("the_lookup_raises", dict(saved_figure_appearance=_boom, export_colour=_black), False),
        ("no_look_saved", dict(saved_figure_appearance=lambda: None, export_colour=_black), False),
        ("a_screen_look", dict(saved_figure_appearance=lambda: screen_look,
                               export_colour=_black), False),
    ]
    for case_id, names, repaints in stand_downs:
        with subtests.test(msg=case_id):
            plot = make()
            with monkeypatch.context() as mp:
                _style_stub(mp, **names)
                undo = fp.FastPlot._wear_the_print_look(plot.plot)
            assert (len(undo) > 0) is repaints
    with subtests.test(msg="an_edge_with_no_axis_is_stepped_over"):
        plot = make()
        repainted = []
        real = plot.plot.plotItem

        class _HalfDressed:
            titleLabel = None

            def getAxis(self, edge):
                return None if edge in ("top", "right") else real.getAxis(edge)

        with monkeypatch.context() as mp:
            _style_stub(mp, saved_figure_appearance=lambda: print_look,
                        export_colour=lambda c, kind, s: repainted.append(kind) or "#010203")
            assert fp.FastPlot._wear_the_print_look(_HalfDressed()) != []
        assert set(repainted) == {"chrome"}
    with subtests.test(msg="the_real_style_module_screen_then_print"):
        import spacr.figure_style as style

        plot = make()
        with monkeypatch.context() as mp:
            mp.setattr(style, "saved_figure_appearance", lambda: SimpleNamespace(mode="screen"))
            plot.plot.getAxis("bottom").setPen(pg.mkPen("#ffffff"))
            assert fp.FastPlot._wear_the_print_look(plot.plot) == []
            mp.setattr(style, "saved_figure_appearance", lambda: _real_look("print"))
            assert fp.FastPlot._wear_the_print_look(plot.plot) != []
    with subtests.test(msg="old_axes_without_text_pen_are_repainted_and_undone"):
        import spacr.figure_style as style

        axes = {"bottom": _PlainAxis("#ffffff"), "left": _PlainAxis("#ffffff")}
        item = SimpleNamespace(getAxis=lambda edge: axes.get(edge), titleLabel=None)
        with monkeypatch.context() as mp:
            mp.setattr(style, "saved_figure_appearance", lambda: _real_look("print"))
            undo = fp.FastPlot._wear_the_print_look(item)
        assert len(undo) == 2
        assert axes["bottom"].pen().color().name() == "#222222"
        for put_back in undo:
            put_back()
        assert axes["bottom"].pen().color().name() == "#ffffff"


def test_the_violin_profile_is_a_closed_outline_or_nothing(subtests):
    """``_violin_profile`` peaks at the half-width, closes at both ends, refuses no spread."""
    rng = np.random.default_rng(1)
    shaped = [
        ("two_points", [0.0, 1.0], 0.4),
        ("a_dense_ramp", list(np.linspace(-3.0, 3.0, 200)), 0.4),
        ("a_tiny_range", [1e-9, 2e-9], 0.4),
        ("a_lopsided_trio", [0.0, 0.0, 5.0], 0.4),
        ("a_small_spread", [1.0, 2.0, 2.0, 3.0], 0.4),
        ("a_normal_sample", list(rng.normal(size=200)), 0.4),
        ("three_values_half_width_half", [1.0, 2.0, 3.0], 0.5),
    ]
    for case_id, values, width in shaped:
        with subtests.test(msg=case_id):
            centres, density = fp._violin_profile(np.asarray(values, float), width)
            assert len(centres) == len(density)
            assert float(np.max(density)) == pytest.approx(width)
            assert (density[0], density[-1]) == (0.0, 0.0)
            assert (centres[0], centres[-1]) == pytest.approx((min(values), max(values)))
    with subtests.test(msg="three_values_give_eight_points"):
        centres, widths = fp._violin_profile([1.0, 2.0, 3.0], 0.5)
        assert centres.size == widths.size == 8
    for case_id, values in (("identical_values", [2.0, 2.0, 2.0]),
                            ("fifty_identical", list(np.full(50, 3.0))),
                            ("four_fours", [4.0, 4.0, 4.0]),
                            ("all_nan", [np.nan, np.nan]),
                            ("an_infinite_end", [np.inf, 1.0]),
                            ("an_infinite_top", [0.0, np.inf])):
        with subtests.test(msg=f"refused_{case_id}"):
            assert fp._violin_profile(values, 0.4) == (None, None)


class _Item:
    def __init__(self, data):
        self.data = data


def test_the_rows_behind_a_scatter_are_frame_positions_or_none(qtbot, subtests):
    """``_rows_of`` converts per-point data to int64 rows, or None if any is missing."""
    rows = np.zeros(4, dtype=[("data", "int64")])
    rows["data"] = [3, 1, 4, 1]
    cases = [
        ("an_object_array_with_a_gap", _Item(np.array([0, None], dtype=object)), None),
        ("no_data_at_all", _Item(None), None),
        ("an_empty_list", _Item([]), None),
        ("an_already_integer_array", _Item(rows), [3, 1, 4, 1]),
    ]
    for case_id, item, expected in cases:
        with subtests.test(msg=case_id):
            found = fp.FastPlot._rows_of(item)
            if expected is None:
                assert found is None
            else:
                assert (found.dtype != object, found.tolist()) == (True, expected)
    with subtests.test(msg="a_ranked_plot_reorders_rows_and_colours_follow"):
        ranked = fp.EffectRankPlot()
        qtbot.addWidget(ranked)
        frame = pd.DataFrame({"feature": ["a", "b", "c"],
                              "coefficient": [-2.0, 1.0, 2.0],
                              "p_value": [0.01, 0.5, 0.02]})
        assert ranked.set_results(frame) == 3
        item = _items(ranked, pg.ScatterPlotItem)[0]
        assert item.data["data"].dtype == object
        found = fp.FastPlot._rows_of(item)
        assert (found.dtype, list(found)) == (np.dtype("int64"), [0, 2, 1])
        assert ranked.colour_by_column("coefficient") == 3
        lookup = pg.colormap.get("viridis").getLookupTable(nPts=fp.COLORMAP_STEPS, alpha=True)
        from PySide6.QtGui import QColor
        coldest = QColor(*(int(c) for c in lookup[0])).name()
        hottest = QColor(*(int(c) for c in lookup[fp.COLORMAP_STEPS - 1])).name()
        painted = [brush.color().name() for brush in item.data["brush"]]
        assert painted[:2] == [coldest, hottest]
