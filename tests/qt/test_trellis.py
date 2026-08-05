"""V5 — small multiples: the grid, the panels that are empty, and the scales.

The frame is six rows, so every limit below is arithmetic that can be checked
without running anything::

    plateID  gene  area
    p1       a       10
    p1       a       20
    p1       b      100
    p2       a       30
    p2       a       40
    p2       a       50

Faceted by ``plateID`` down and ``gene`` across, the grid is 2 × 2 = **four**
panels, and one of them — (p2, b) — has no rows at all. It is drawn anyway:
"plate 2 has no gene b" and "plate 2's gene b was measured and everything was
filtered out" are different facts, and a grid that closes the gap tells the
reader the wrong one.

``_limits`` pads a range by 5% of its span, and widens a degenerate one by 5%
of its value, so the four scale modes are:

* **shared** — every panel gets [10, 100] padded: ``(5.5, 104.5)``
* **free** — (p1,a) sees [10, 20] -> ``(9.5, 20.5)``; (p1,b) sees the single
  value 100 -> ``(95, 105)``; (p2,a) sees [30, 50] -> ``(29, 51)``
* **per row** — row p1 is [10, 20, 100] -> ``(5.5, 104.5)``; row p2 is
  [30, 50] -> ``(29, 51)``
* **per column** — column a is [10, 50] -> ``(8, 52)``; column b is
  ``(95, 105)``

With two bins over the shared range the edges are 5.5, 55, 104.5, so the
tallest bin in the grid is (p2, a)'s three objects, and a shared count axis
must therefore top out at 3 × 1.08 = 3.24.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.graph_spec import (
    CONTINUOUS, HISTOGRAM, SCATTER, GraphSpec, SpecError)
from spacr.qt.widgets.trellis_spec import (
    LOW_N, MAX_WRAP, SCALE_COL, SCALE_FREE, SCALE_ROW, SCALE_SHARED,
    TrellisSpec, trellis, wrap_positions,
)

#: The one thing about the fixture that is not the fixture's own arithmetic.
#:
#: `classify_columns` calls a numeric column with twelve or fewer distinct
#: values a *category* — the right rule, and the reason `plateID` as 1/2/3 is
#: a tick list rather than a slider. A six-row table's `area` has six distinct
#: values, so it would be one too. `GraphSpec.roles` is the designed override
#: for exactly this, and saying it here keeps the frame small enough that
#: every limit below can be worked out by hand.
CONTINUOUS_AREA = {"area": CONTINUOUS}


@pytest.fixture
def hand() -> pd.DataFrame:
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "gene": ["a", "a", "b", "a", "a", "a"],
        "area": [10.0, 20.0, 100.0, 30.0, 40.0, 50.0],
    })


def two_way(**kwargs) -> TrellisSpec:
    return TrellisSpec(
        graph=GraphSpec(x="area", facet_row="plateID", facet_col="gene",
                        bins=2, roles=CONTINUOUS_AREA),
        **kwargs)


# ---------------------------------------------------------------------------
# The grid, including the panels with nothing in them
# ---------------------------------------------------------------------------

def test_the_grid_is_the_full_product_including_the_empty_panel(hand):
    result = trellis(hand, two_way())
    assert result.shape == (2, 2)
    assert result.n_panels == 4              # NOT three
    assert result.n_occupied == 4
    assert result.n_empty == 1
    assert [(p.row_level, p.col_level, p.n) for p in result.panels] == [
        ("p1", "a", 2), ("p1", "b", 1), ("p2", "a", 3), ("p2", "b", 0)]


def test_every_panel_prints_its_n(hand):
    result = trellis(hand, two_way())
    assert result.panel(0, 0).label() == "p1 · a  ·  n = 2"
    assert result.panel(1, 1).label() == "p2 · b  ·  n = 0"
    assert result.n_at(1, 0) == 3
    assert result.n_range() == (1, 3)


def test_small_panels_are_flagged(hand):
    result = trellis(hand, two_way())
    low = result.low_n_panels()
    assert {p.title() for p in low} == {"p1 · a", "p1 · b", "p2 · a"}
    assert all(p.n <= LOW_N for p in low)
    assert f"n ≤ {LOW_N}" in result.summary()
    # An empty panel is not a small panel; it is a different fact.
    assert result.panel(1, 1) not in low


def test_the_summary_counts_the_panels_and_the_empties(hand):
    summary = trellis(hand, two_way()).summary()
    assert "2 × 2 = 4 panels" in summary
    assert "1 with no rows" in summary
    assert "n per panel 1–3" in summary


def test_one_way_faceting_is_a_strip(hand):
    result = trellis(hand, TrellisSpec(
        graph=GraphSpec(x="area", facet_col="gene", roles=CONTINUOUS_AREA)))
    assert result.shape == (1, 2)
    assert result.n_panels == 2


def test_no_faceting_is_one_panel(hand):
    result = trellis(hand, TrellisSpec(graph=GraphSpec(x="area", roles=CONTINUOUS_AREA)))
    assert result.shape == (1, 1)
    assert result.panel(0, 0).n == 6
    assert result.panel(0, 0).label() == "n = 6"


# ---------------------------------------------------------------------------
# Shared axes really are shared
# ---------------------------------------------------------------------------

def test_shared_is_the_default_and_every_panel_gets_the_same_limits(hand):
    spec = two_way()
    assert spec.scale_x == SCALE_SHARED and spec.scale_y == SCALE_SHARED
    result = trellis(hand, spec)
    limits = [p.scales.x_limits for p in result.panels]
    assert limits == [(5.5, 104.5)] * 4        # hand-computed, all identical
    assert len(set(limits)) == 1
    assert result.shared.x_limits == (5.5, 104.5)
    assert result.axes_are_comparable()


def test_free_scales_give_every_panel_its_own_limits(hand):
    result = trellis(hand, two_way(scale_x=SCALE_FREE))
    assert result.panel(0, 0).scales.x_limits == (9.5, 20.5)
    assert result.panel(0, 1).scales.x_limits == (95.0, 105.0)
    assert result.panel(1, 0).scales.x_limits == (29.0, 51.0)
    # The empty panel has nothing to scale to, and says so with None rather
    # than borrowing the neighbours'.
    assert result.panel(1, 1).scales.x_limits is None
    assert not result.axes_are_comparable()


def test_per_row_scales_share_along_a_row(hand):
    result = trellis(hand, two_way(scale_x=SCALE_ROW))
    assert result.panel(0, 0).scales.x_limits == (5.5, 104.5)
    assert result.panel(0, 1).scales.x_limits == (5.5, 104.5)
    assert result.panel(1, 0).scales.x_limits == (29.0, 51.0)
    assert result.panel(1, 1).scales.x_limits == (29.0, 51.0)


def test_per_column_scales_share_down_a_column(hand):
    result = trellis(hand, two_way(scale_x=SCALE_COL))
    assert result.panel(0, 0).scales.x_limits == (8.0, 52.0)
    assert result.panel(1, 0).scales.x_limits == (8.0, 52.0)
    assert result.panel(0, 1).scales.x_limits == (95.0, 105.0)
    assert result.panel(1, 1).scales.x_limits == (95.0, 105.0)


def test_the_two_axes_take_their_modes_independently(hand):
    frame = hand.assign(intensity=[1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
    spec = TrellisSpec(
        graph=GraphSpec(x="area", y="intensity", facet_row="plateID",
                        facet_col="gene",
                        roles={"area": CONTINUOUS, "intensity": CONTINUOUS}),
        scale_x=SCALE_SHARED, scale_y=SCALE_FREE)
    result = trellis(frame, spec)
    x_limits = {p.scales.x_limits for p in result.panels}
    assert len(x_limits) == 1                     # x shared
    y_limits = [p.scales.y_limits for p in result.panels]
    assert y_limits[0] != y_limits[2]             # y free


def test_a_shared_count_axis_tops_out_at_the_tallest_bin(hand):
    """Two bins over (5.5, 104.5): p2/a puts all three objects in the first."""
    result = trellis(hand, two_way())
    limits = [p.scales.count_limit for p in result.panels]
    assert limits == [pytest.approx(3.24)] * 4


def test_a_free_count_axis_is_per_panel(hand):
    result = trellis(hand, two_way(scale_y=SCALE_FREE))
    assert result.panel(0, 0).scales.count_limit == pytest.approx(2 * 1.08)
    assert result.panel(1, 0).scales.count_limit == pytest.approx(3 * 1.08)
    assert result.panel(1, 1).scales.count_limit is None


def test_bin_edges_are_shared_when_the_axis_is(hand):
    shared = trellis(hand, two_way())
    edges = [p.scales.x_edges for p in shared.panels]
    for other in edges[1:]:
        np.testing.assert_array_equal(edges[0], other)
    free = trellis(hand, two_way(scale_x=SCALE_FREE))
    assert not np.array_equal(free.panel(0, 0).scales.x_edges,
                              free.panel(1, 0).scales.x_edges)


def test_colour_and_size_are_never_per_panel(hand):
    """A gene that is blue in one panel and orange in the next is a trap."""
    frame = hand.assign(cls=["x", "y", "x", "y", "x", "y"],
                        weight=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spec = TrellisSpec(
        graph=GraphSpec(x="area", y="weight", colour="cls", size="weight",
                        facet_row="plateID", facet_col="gene",
                        roles={"area": CONTINUOUS, "weight": CONTINUOUS}),
        scale_x=SCALE_FREE, scale_y=SCALE_FREE)
    result = trellis(frame, spec)
    assert {p.scales.colour_levels for p in result.panels} == {("x", "y")}
    assert len({p.scales.size_limits for p in result.panels}) == 1


# ---------------------------------------------------------------------------
# Saying so when they are not shared
# ---------------------------------------------------------------------------

def test_a_shared_grid_carries_no_warning(hand):
    assert "not shared" not in trellis(hand, two_way()).notice


def test_a_free_grid_says_the_panels_are_not_comparable(hand):
    notice = trellis(hand, two_way(scale_x=SCALE_FREE)).notice
    assert "axes are not shared" in notice
    assert "NOT comparable" in notice
    assert "horizontally" in notice


def test_a_per_row_grid_says_which_direction_is_comparable(hand):
    notice = trellis(hand, two_way(scale_y=SCALE_ROW)).notice
    assert "only panels in the same row are comparable" in notice
    assert "vertically" in notice


# ---------------------------------------------------------------------------
# Wrapping a long strip
# ---------------------------------------------------------------------------

def test_wrap_positions_are_row_major():
    assert wrap_positions(5, 3) == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert wrap_positions(3, 0) == ((0, 0), (1, 0), (2, 0))


def test_seven_levels_wrapped_at_three_is_a_three_by_three_with_two_blanks():
    frame = pd.DataFrame({"plateID": [f"p{i}" for i in range(1, 8)],
                          "area": np.arange(7.0)})
    result = trellis(frame, TrellisSpec(
        graph=GraphSpec(x="area", facet_col="plateID", roles=CONTINUOUS_AREA),
        wrap=3))
    assert result.shape == (3, 3)
    assert result.n_panels == 9
    assert result.n_occupied == 7
    # The two blanks are the remainder of a division, not groups with no rows.
    assert result.n_empty == 0
    assert [p.occupied for p in result.panels[-2:]] == [False, False]
    assert result.panels[-1].label() == ""
    assert result.panel(0, 1).row_level is None
    assert result.panel(0, 1).col_level == "p2"


def test_wrap_is_ignored_for_a_two_way_grid_and_says_so(hand):
    result = trellis(hand, two_way(wrap=3))
    assert result.shape == (2, 2)
    assert "wrap ignored" in result.notice


def test_a_wrap_wider_than_the_levels_does_not_pad(hand):
    result = trellis(hand, TrellisSpec(
        graph=GraphSpec(x="area", facet_col="plateID", roles=CONTINUOUS_AREA),
        wrap=8))
    assert result.shape == (1, 2)
    assert result.n_panels == 2


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_the_spec_round_trips_through_json(hand):
    spec = two_way(scale_x=SCALE_FREE, scale_y=SCALE_ROW, wrap=4)
    again = TrellisSpec.from_json(spec.to_json())
    assert again == spec
    assert again.graph == spec.graph
    assert json.loads(spec.to_json())["graph"]["facet_row"] == "plateID"


def test_an_unknown_scale_mode_is_refused_where_it_is_written():
    with pytest.raises(SpecError, match="scale_x"):
        TrellisSpec(scale_x="whatever")
    with pytest.raises(SpecError, match="wrap"):
        TrellisSpec(wrap=MAX_WRAP + 1)


def test_edits_return_new_specs():
    spec = TrellisSpec()
    assert spec.with_scales(scale_x=SCALE_FREE).scale_x == SCALE_FREE
    assert spec.scale_x == SCALE_SHARED                 # unchanged
    assert spec.with_channel("x", "area").graph.x == "area"
    assert spec.with_wrap(3).wrap == 3
    assert not spec.is_faceted
    assert two_way().is_two_way


def test_describe_mentions_the_scales_only_when_they_are_not_shared():
    assert "x scale" not in two_way().describe({})
    assert "x scale: free" in two_way(scale_x=SCALE_FREE).describe({})
    assert "wrapped at 4" in two_way(wrap=4).describe({})


# ---------------------------------------------------------------------------
# Brushing one panel
# ---------------------------------------------------------------------------

def test_a_brush_selects_only_the_swept_rows_of_that_panel(hand):
    result = trellis(hand, two_way())
    # Panel (1, 0) is plate p2 / gene a: areas 30, 40, 50. Sweeping 35..55
    # takes 40 and 50 and nothing from any other panel.
    mask = result.brush(35.0, 0.0, 55.0, 10.0, row=1, col=0)
    assert mask.tolist() == [False, False, False, False, True, True]
    assert hand.loc[mask, "area"].tolist() == [40.0, 50.0]


def test_a_brush_on_one_panel_never_reaches_another(hand):
    result = trellis(hand, two_way())
    # The same x range swept on the p1/a panel catches nothing: p1's areas
    # are 10 and 20, and the rectangle is a predicate on the panel's rows.
    mask = result.brush(35.0, 0.0, 55.0, 10.0, row=0, col=0)
    assert not mask.any()


def test_a_brush_on_a_blank_slot_selects_nothing():
    frame = pd.DataFrame({"plateID": [f"p{i}" for i in range(1, 8)],
                          "area": np.arange(7.0)})
    result = trellis(frame, TrellisSpec(
        graph=GraphSpec(x="area", facet_col="plateID", roles=CONTINUOUS_AREA),
        wrap=3))
    assert not result.brush(-100.0, -100.0, 100.0, 100.0, row=2, col=2).any()


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

@pytest.fixture
def canvas(qtbot, hand):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.trellis_view import TrellisCanvas
    widget = TrellisCanvas(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(hand)
    return widget


def test_the_canvas_draws_every_panel_including_the_empty_one(canvas):
    canvas.set_trellis_spec(two_way())
    axes = canvas.panel_axes()
    assert len(axes) == 4
    assert canvas.trellis.n_empty == 1
    assert all(ax.get_visible() for ax in axes.values())


def test_the_drawn_axes_really_share_their_limits(canvas):
    """The property, asserted against matplotlib rather than against Scales."""
    canvas.set_trellis_spec(two_way())
    limits = {ax.get_xlim() for ax in canvas.panel_axes().values()}
    assert len(limits) == 1
    assert limits.pop() == pytest.approx((5.5, 104.5))
    # The count axis is shared too -- the usual way a faceted histogram lies.
    tops = {round(ax.get_ylim()[1], 6) for ax in canvas.panel_axes().values()}
    assert len(tops) == 1


def test_free_scales_really_produce_different_drawn_limits(canvas):
    canvas.set_trellis_spec(two_way(scale_x=SCALE_FREE))
    axes = canvas.panel_axes()
    assert axes[(0, 0)].get_xlim() == pytest.approx((9.5, 20.5))
    assert axes[(1, 0)].get_xlim() == pytest.approx((29.0, 51.0))


def test_panel_titles_carry_n(canvas):
    canvas.set_trellis_spec(two_way())
    titles = {ax.get_title() for ax in canvas.panel_axes().values()}
    assert "p1 · a  ·  n = 2" in titles
    assert "p2 · b  ·  n = 0" in titles


def test_blank_wrap_slots_are_hidden_rather_than_drawn_empty(qtbot):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.trellis_view import TrellisCanvas
    frame = pd.DataFrame({"plateID": [f"p{i}" for i in range(1, 8)],
                          "area": np.arange(7.0)})
    widget = TrellisCanvas(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    widget.set_trellis_spec(TrellisSpec(
        graph=GraphSpec(x="area", facet_col="plateID", roles=CONTINUOUS_AREA),
        wrap=3))
    axes = widget.panel_axes()
    assert len(axes) == 9
    assert sum(1 for ax in axes.values() if not ax.get_visible()) == 2


def test_the_notice_carries_the_warning_and_the_shape(canvas):
    canvas.set_trellis_spec(two_way(scale_x=SCALE_FREE))
    assert "2 × 2 = 4 panels" in canvas.notice()
    assert "axes are not shared" in canvas.notice()


def test_inner_ticks_are_hidden_only_when_the_axis_is_shared(canvas):
    canvas.set_trellis_spec(two_way())
    top_left = canvas.panel_axes()[(0, 0)]
    assert not any(label.get_visible()
                   for label in top_left.get_xticklabels())
    canvas.set_trellis_spec(two_way(scale_x=SCALE_FREE))
    top_left = canvas.panel_axes()[(0, 0)]
    assert any(label.get_visible() for label in top_left.get_xticklabels())


def test_an_empty_spec_asks_for_a_column_rather_than_drawing_nothing(canvas):
    canvas.set_trellis_spec(TrellisSpec())
    assert canvas.trellis is None
    assert canvas.panel_axes() == {}


# ---------------------------------------------------------------------------
# The panel and the screen
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot, hand):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.trellis_view import TrellisPanelWidget
    widget = TrellisPanelWidget(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(hand)
    return widget


def test_dropping_columns_on_the_zones_builds_the_grid(panel):
    panel.zone("x").set_column("area")
    panel.zone("facet_row").set_column("plateID")
    panel.zone("facet_col").set_column("gene")
    assert panel.spec.graph.x == "area"
    assert panel.canvas.trellis.shape == (2, 2)


def test_the_scale_picker_changes_the_mode(panel):
    panel.zone("x").set_column("area")
    panel.zone("facet_col").set_column("gene")
    index = panel._scale_x.findData(SCALE_FREE)
    panel._scale_x.setCurrentIndex(index)
    assert panel.spec.scale_x == SCALE_FREE
    assert not panel.canvas.trellis.axes_are_comparable()


def test_pushing_a_spec_in_updates_the_controls(panel):
    panel.set_spec(two_way(scale_y=SCALE_ROW, wrap=4))
    assert panel.zone("facet_row").column == "plateID"
    assert panel._scale_y.currentData() == SCALE_ROW
    assert panel._wrap.value() == 4


def test_the_screen_gives_computed_columns_to_the_grid(qtbot, hand):
    """A formula defined on the screen is faceted like any measured column."""
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.formula import ColumnFormula
    from spacr.qt.screens.trellis import TrellisScreen

    screen = TrellisScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    screen.formulas.add_formula(ColumnFormula("half", "area / 2"))
    assert "half" in screen.panel.well.columns()
    assert "half" in screen.filters.available_columns()
    screen.panel.zone("x").set_column("half")
    screen.panel.zone("facet_col").set_column("gene")
    grid = screen.panel.canvas.trellis
    assert grid.shape == (1, 2)
    assert [p.n for p in grid.panels] == [5, 1]
    assert "half" in grid.frame.columns
    screen.close()


def test_the_screen_registers_once(qtbot):
    """One row in `spacr.qt.SELF_REGISTERING_MODULES` turns it on."""
    from spacr.qt import app as app_mod
    from spacr.qt.screens import trellis as screen_mod

    apps = list(app_mod.APPS)
    try:
        if any(row[0] == screen_mod.APP_KEY for row in app_mod.APPS):
            assert screen_mod.register() is False
        else:
            assert screen_mod.register() is True
            assert screen_mod.register() is False
            row = next(r for r in app_mod.APPS
                       if r[0] == screen_mod.APP_KEY)
            assert row[1] == screen_mod.APP_NAME
            meta = app_mod.APP_META[screen_mod.APP_KEY]
            assert meta["intro"] == screen_mod.APP_INTRO
            assert meta["cli_note"] == screen_mod.APP_CLI_NOTE
            assert len(meta["translations"]) == 9
    finally:
        app_mod.APPS[:] = apps
        app_mod.APP_META.pop(screen_mod.APP_KEY, None)
        app_mod.APP_FACTORIES.pop(screen_mod.APP_KEY, None)
        app_mod.APP_STAGE.pop(screen_mod.APP_KEY, None)


def test_the_factory_builds_a_working_screen(qtbot, hand):
    from spacr.qt.screens.trellis import make_trellis_screen

    screen = make_trellis_screen()
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    assert screen.spec.scale_x == SCALE_SHARED
    assert not screen.is_busy()
    screen.close()
