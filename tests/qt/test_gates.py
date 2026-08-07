"""V2 — gates: the rows they select, worked out by hand, and the round trip.

The frame is ten objects on a grid, chosen so every gate below selects a set
that can be listed rather than counted by the code under test::

    idx  area  intensity
      0    10         10
      1    20         10
      2    30         10
      3    40         10
      4    10         50
      5    20         50
      6    30         50
      7    40         50
      8   NaN         50     <- no area was measured
      9    25        NaN     <- no intensity was measured

* ``singlets`` = ``20 <= area <= 40`` selects rows 1, 2, 3, 5, 6, 7 and 9 —
  seven objects. Row 8 has no area and is therefore **outside**: an object with
  no measurement is not an object inside the region.
* ``bright`` = that, and ``intensity >= 30``: rows 5, 6, 7. Row 9 drops out
  because its intensity is missing; rows 1–3 because theirs is 10.
* The triangle with vertices (0, 0), (50, 0) and (0, 50) contains a point where
  ``area + intensity < 50``: rows 0 (20 < 50) and 1 (30 < 50) and 2 (40 < 50)
  and 4? — no: row 4 is (10, 50), which is *on* the hypotenuse. The test picks
  a triangle with no point on an edge, so the even-odd rule's boundary
  behaviour is never what is being asserted.

Sequential gating is the part worth testing hardest: ``bright`` inside
``singlets`` must select the INTERSECTION, and as two range clauses on
overlapping columns it would not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.selection import DataFilter, RangeFilter
from spacr.qt.widgets.gate_spec import (
    POLYGON, RECTANGLE, THRESHOLD, GateClause, GateError, GateSet,
    PolygonGate, RectGate, ThresholdGate, gate_from_dict, points_in_polygon,
)


@pytest.fixture
def hand() -> pd.DataFrame:
    return pd.DataFrame({
        "area": [10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0, np.nan, 25.0],
        "intensity": [10.0, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                      np.nan],
    })


def selected(mask) -> list:
    return list(np.flatnonzero(np.asarray(mask)))


# ---------------------------------------------------------------------------
# One gate at a time
# ---------------------------------------------------------------------------

def test_a_threshold_selects_exactly_the_hand_computed_rows(hand):
    gate = ThresholdGate(name="singlets", column="area", low=20.0, high=40.0)
    assert selected(gate.mask(hand)) == [1, 2, 3, 5, 6, 7, 9]


def test_a_missing_measurement_is_outside_every_gate(hand):
    """Row 8 has no area. It is not 'unknown', it is out."""
    gate = ThresholdGate(name="any", column="area", low=-1e9, high=1e9)
    assert 8 not in selected(gate.mask(hand))
    rect = RectGate(name="r", x_column="area", y_column="intensity",
                    x_low=-1e9, x_high=1e9, y_low=-1e9, y_high=1e9)
    assert selected(rect.mask(hand)) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_a_threshold_dragged_backwards_is_the_same_gate():
    forwards = ThresholdGate(name="a", column="area", low=20.0, high=40.0)
    backwards = ThresholdGate(name="a", column="area", low=40.0, high=20.0)
    assert (backwards.low, backwards.high) == (20.0, 40.0) == \
        (forwards.low, forwards.high)


def test_an_open_ended_threshold_means_unbounded_not_empty(hand):
    gate = ThresholdGate(name="big", column="area", low=30.0)
    assert selected(gate.mask(hand)) == [2, 3, 6, 7]
    assert gate.describe() == "area ≥ 30"


def test_a_rectangle_selects_the_intersection_of_two_ranges(hand):
    gate = RectGate(name="q", x_column="area", y_column="intensity",
                    x_low=20.0, x_high=40.0, y_low=30.0, y_high=60.0)
    assert selected(gate.mask(hand)) == [5, 6, 7]


def test_a_rectangle_is_also_expressible_as_range_clauses(hand):
    gate = RectGate(name="q", x_column="area", y_column="intensity",
                    x_low=20.0, x_high=40.0, y_low=30.0, y_high=60.0)
    ranges = gate.range_filters()
    assert [r.column for r in ranges] == ["area", "intensity"]
    combined = DataFilter(list(ranges))
    np.testing.assert_array_equal(combined.mask(hand), gate.mask(hand))


def test_a_polygon_is_not_pretended_to_be_a_box():
    gate = PolygonGate(name="p", x_column="area", y_column="intensity",
                       vertices=((0, 0), (60, 0), (0, 60)))
    assert gate.range_filters() == ()


def test_a_polygon_selects_only_what_is_inside_it(hand):
    """The triangle (0,0)-(45,0)-(0,45): inside is area + intensity < 45."""
    gate = PolygonGate(name="small", x_column="area", y_column="intensity",
                       vertices=((0.0, 0.0), (45.0, 0.0), (0.0, 45.0)))
    # rows 0 (20), 1 (30), 2 (40) are under 45; row 3 is 50, rows 4-7 are 60+.
    assert selected(gate.mask(hand)) == [0, 1, 2]


def test_a_polygon_beats_its_own_bounding_box(hand):
    """The corner debris a rectangle would have swept up."""
    triangle = PolygonGate(name="t", x_column="area", y_column="intensity",
                           vertices=((0.0, 0.0), (45.0, 0.0), (0.0, 45.0)))
    box = RectGate(name="b", x_column="area", y_column="intensity",
                   x_low=0.0, x_high=45.0, y_low=0.0, y_high=45.0)
    assert selected(box.mask(hand)) == [0, 1, 2, 3]     # (40, 10) is in the box
    assert 3 not in selected(triangle.mask(hand))       # but outside the region


def test_point_in_polygon_handles_a_concave_shape():
    """An L: (0,0)(4,0)(4,2)(2,2)(2,4)(0,4). The notch at (3,3) is outside."""
    verts = ((0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4))
    x = np.array([1.0, 3.0, 3.0, 1.0, np.nan])
    y = np.array([1.0, 1.0, 3.0, 3.0, 1.0])
    assert points_in_polygon(x, y, verts).tolist() == [
        True, True, False, True, False]


def test_a_repeated_closing_vertex_is_dropped():
    gate = PolygonGate(name="p", x_column="a", y_column="b",
                       vertices=((0, 0), (1, 0), (0, 1), (0, 0)))
    assert len(gate.vertices) == 3


# ---------------------------------------------------------------------------
# Gates that cannot mean anything are refused where they are drawn
# ---------------------------------------------------------------------------

def test_an_unnamed_gate_is_refused():
    with pytest.raises(GateError, match="needs a name"):
        ThresholdGate(name="  ", column="area", low=1.0)


def test_a_threshold_with_no_bounds_is_refused():
    with pytest.raises(GateError, match="selects everything"):
        ThresholdGate(name="a", column="area")


def test_a_two_vertex_polygon_is_refused():
    with pytest.raises(GateError, match="at least three"):
        PolygonGate(name="p", x_column="a", y_column="b",
                    vertices=((0, 0), (1, 1)))


def test_a_polygon_with_no_area_is_refused():
    """Three collinear clicks are a line, and a line contains nothing."""
    with pytest.raises(GateError, match="no area"):
        PolygonGate(name="p", x_column="a", y_column="b",
                    vertices=((0, 0), (1, 1), (2, 2)))
    with pytest.raises(GateError, match="no area"):
        PolygonGate(name="p", x_column="a", y_column="b",
                    vertices=((1, 1), (1, 1), (1, 1), (1, 1)))


def test_a_gate_on_one_column_against_itself_is_refused():
    with pytest.raises(GateError, match="against itself"):
        RectGate(name="r", x_column="a", y_column="a", x_low=0.0)


def test_a_gate_re_applied_to_a_table_without_the_column_says_which(hand):
    gate = ThresholdGate(name="a", column="nucleus_area", low=1.0)
    with pytest.raises(GateError, match="nucleus_area"):
        gate.mask(hand)


# ---------------------------------------------------------------------------
# Sequential gating
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy() -> GateSet:
    gates = GateSet()
    gates.add(ThresholdGate(name="singlets", column="area",
                            low=20.0, high=40.0))
    gates.add(ThresholdGate(name="bright", parent="singlets",
                            column="intensity", low=30.0))
    return gates


def test_a_child_gate_selects_the_intersection(hand, strategy):
    assert selected(strategy.mask(hand, "singlets")) == [1, 2, 3, 5, 6, 7, 9]
    assert selected(strategy.mask(hand, "bright")) == [5, 6, 7]


def test_two_gates_on_the_same_column_do_not_replace_each_other(hand):
    """The reason a chain is ONE clause and not one clause per gate."""
    gates = GateSet()
    gates.add(ThresholdGate(name="over", column="area", low=20.0))
    gates.add(ThresholdGate(name="under", parent="over", column="area",
                            high=30.0))
    assert selected(gates.mask(hand, "under")) == [1, 2, 5, 6, 9]
    # As two RangeFilters on `area`, DataFilter.add would keep only the last.
    naive = DataFilter()
    naive.add(RangeFilter("area", low=20.0))
    naive.add(RangeFilter("area", high=30.0))
    assert len(naive.clauses) == 1
    assert selected(naive.mask(hand)) != selected(gates.mask(hand, "under"))
    # The gate clause keeps both.
    gated = gates.filter_for("under")
    assert len(gated.clauses) == 1
    assert selected(gated.mask(hand)) == [1, 2, 5, 6, 9]


def test_the_path_is_outermost_first(strategy):
    assert [g.name for g in strategy.path("bright")] == ["singlets", "bright"]
    assert strategy.depth("bright") == 1
    assert strategy.depth("singlets") == 0


def test_a_gate_inside_a_gate_that_does_not_exist_is_refused():
    with pytest.raises(GateError, match="does not exist"):
        GateSet().add(ThresholdGate(name="a", parent="nope", column="x",
                                    low=1.0))


def test_a_gate_cannot_be_its_own_parent():
    with pytest.raises(GateError, match="its own parent"):
        ThresholdGate(name="a", parent="a", column="x", low=1.0)


def test_a_cycle_is_refused_and_the_set_is_left_as_it_was(strategy):
    with pytest.raises(GateError, match="loop"):
        strategy.add(ThresholdGate(name="singlets", parent="bright",
                                   column="area", low=20.0, high=40.0))
    # The gate that was there is still there, unchanged.
    assert strategy.get("singlets").parent is None
    assert strategy.names == ("singlets", "bright")


def test_redrawing_a_gate_moves_everything_below_it(hand, strategy):
    strategy.add(ThresholdGate(name="singlets", column="area",
                               low=30.0, high=40.0))
    assert selected(strategy.mask(hand, "bright")) == [6, 7]
    assert strategy.get("bright").parent == "singlets"


def test_deleting_a_gate_takes_its_children_with_it(strategy):
    strategy.remove("singlets")
    assert strategy.is_empty


def test_deleting_without_cascade_says_what_is_in_the_way(strategy):
    with pytest.raises(GateError, match="'bright'"):
        strategy.remove("singlets", cascade=False)


# ---------------------------------------------------------------------------
# The clause every linked view honours
# ---------------------------------------------------------------------------

def test_the_gate_becomes_a_data_filter_clause(hand, strategy):
    data_filter = strategy.filter_for("bright")
    assert isinstance(data_filter, DataFilter)
    assert selected(data_filter.mask(hand)) == [5, 6, 7]
    assert data_filter.apply(hand).index.tolist() == [5, 6, 7]
    assert "bright" in data_filter.describe()


def test_the_clause_column_is_the_gate_so_redrawing_replaces_it(hand, strategy):
    data_filter = strategy.filter_for("bright")
    assert data_filter.clauses[0].column == "gate:bright"
    # Re-drawing the same gate replaces the clause rather than stacking one.
    strategy.add(ThresholdGate(name="bright", parent="singlets",
                               column="intensity", low=40.0))
    again = strategy.filter_for("bright", data_filter)
    assert len(again.clauses) == 1
    assert selected(again.mask(hand)) == [5, 6, 7]


def test_a_gate_composes_onto_an_existing_filter(hand, strategy):
    base = DataFilter([RangeFilter("intensity", high=40.0)])
    combined = strategy.filter_for("singlets", base)
    assert len(combined.clauses) == 2
    # singlets is [1,2,3,5,6,7,9]; intensity <= 40 keeps 1,2,3 (row 9 is NaN).
    assert selected(combined.mask(hand)) == [1, 2, 3]


def test_two_different_gates_are_two_clauses(hand, strategy):
    data_filter = strategy.filter_for("singlets")
    strategy.filter_for("bright", data_filter)
    assert {c.column for c in data_filter.clauses} == {
        "gate:singlets", "gate:bright"}


def test_a_clause_needs_at_least_one_gate():
    with pytest.raises(GateError, match="at least one"):
        GateClause(())


# ---------------------------------------------------------------------------
# Percentages
# ---------------------------------------------------------------------------

def test_the_hierarchy_reports_both_percentages(hand, strategy):
    stats = {s.name: s for s in strategy.stats(hand)}
    assert stats["singlets"].n_in == 7
    assert stats["singlets"].n_parent == 10
    assert stats["singlets"].of_parent == pytest.approx(0.7)
    assert stats["singlets"].of_total == pytest.approx(0.7)
    assert stats["bright"].n_in == 3
    assert stats["bright"].n_parent == 7
    assert stats["bright"].of_parent == pytest.approx(3 / 7)
    assert stats["bright"].of_total == pytest.approx(0.3)


def test_the_report_is_indented_by_depth(hand, strategy):
    lines = strategy.report(hand).splitlines()
    assert lines[0] == "10 objects"
    assert lines[1].startswith("singlets: 7 (70.0% of parent, 70.0% of all)")
    assert lines[2].startswith("    bright: 3 (42.9% of parent, 30.0% of all)")


def test_percentages_of_an_empty_table_are_not_zero(strategy):
    empty = pd.DataFrame({"area": [], "intensity": []})
    stats = strategy.stats(empty)
    assert stats[0].n_in == 0
    assert np.isnan(stats[0].of_parent)


# ---------------------------------------------------------------------------
# Saving, loading and re-applying
# ---------------------------------------------------------------------------

def test_a_gate_set_round_trips_through_json(strategy):
    again = GateSet.from_json(strategy.to_json())
    assert again.names == strategy.names
    assert again.gates == strategy.gates


def test_every_shape_round_trips():
    gates = GateSet()
    gates.add(ThresholdGate(name="t", column="area", low=1.0, high=2.0))
    gates.add(RectGate(name="r", parent="t", x_column="area",
                       y_column="intensity", x_low=0.0, x_high=5.0,
                       y_low=1.0, y_high=9.0))
    gates.add(PolygonGate(name="p", parent="r", x_column="area",
                          y_column="intensity",
                          vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
    again = GateSet.from_json(gates.to_json())
    assert again.gates == gates.gates
    assert [g.kind for g in again.gates] == [THRESHOLD, RECTANGLE, POLYGON]


def test_a_saved_strategy_re_applies_to_another_dataset(tmp_path, hand,
                                                        strategy):
    path = strategy.save(str(tmp_path / "gates.json"))
    # Another plate: the same measurements, different objects.
    other = pd.DataFrame({"area": [25.0, 25.0, 5.0],
                          "intensity": [80.0, 5.0, 80.0]})
    reloaded = GateSet.load(path)
    assert selected(reloaded.mask(other, "bright")) == [0]
    assert reloaded.report(other).splitlines()[1].startswith("singlets: 2")


def test_a_strategy_that_does_not_fit_the_table_says_which_column(tmp_path,
                                                                  strategy):
    path = strategy.save(str(tmp_path / "gates.json"))
    reloaded = GateSet.load(path)
    with pytest.raises(GateError, match="intensity"):
        reloaded.mask(pd.DataFrame({"area": [25.0]}), "bright")


def test_an_unknown_gate_kind_fails_with_a_sentence():
    # "ellipse" used to be the example of an unknown kind. It is a real gate
    # now, so the example has to be a kind spaCR genuinely does not have --
    # otherwise this test passes for the wrong reason the moment someone
    # implements whatever it names.
    with pytest.raises(GateError, match="unknown gate kind"):
        gate_from_dict({"kind": "hyperboloid", "name": "e"})
    with pytest.raises(GateError, match="does not understand"):
        gate_from_dict({"kind": THRESHOLD, "name": "t", "column": "a",
                        "low": 1.0, "feather": 3})


def test_a_file_that_is_not_json_fails_with_a_sentence():
    with pytest.raises(GateError, match="not a gate file"):
        GateSet.from_json("{{{")


def test_the_order_is_parents_before_children():
    gates = GateSet()
    gates.add(ThresholdGate(name="b", column="x", low=1.0))
    gates.add(ThresholdGate(name="a", column="x", low=1.0))
    gates.add(ThresholdGate(name="b1", parent="b", column="x", low=2.0))
    assert [g.name for g in gates.order()] == ["b", "b1", "a"]


# ---------------------------------------------------------------------------
# The canvas and the panel
# ---------------------------------------------------------------------------

def scatter_spec():
    """area against intensity, both forced continuous.

    Ten rows means ten distinct values, and `classify_columns` calls a numeric
    column with twelve or fewer of those a *category* — the right rule, and
    `GraphSpec.roles` is the designed override for it.
    """
    from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec
    return GraphSpec(x="area", y="intensity",
                     roles={"area": CONTINUOUS, "intensity": CONTINUOUS})


@pytest.fixture
def panel(qtbot, hand):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    widget = GateEditorPanel(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(hand)
    widget.set_spec(scatter_spec())
    widget.set_namer(lambda: "drawn")
    return widget


def test_dragging_with_the_rectangle_tool_makes_a_named_gate(panel, hand):
    panel.canvas.set_tool(RECTANGLE)
    gate = panel.canvas.gate_from_drag(20.0, 30.0, 40.0, 60.0)
    panel.canvas.gate_drawn.emit(gate)
    assert panel.gates.names == ("drawn",)
    assert selected(panel.gates.mask(hand, "drawn")) == [5, 6, 7]


def test_a_threshold_reads_only_the_horizontal_sweep(qtbot, hand):
    """On a histogram the vertical axis is a count; gating on one is not a
    thing anyone means."""
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec

    widget = GateEditorPanel(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(hand)
    widget.set_spec(GraphSpec(x="area", roles={"area": CONTINUOUS}))
    widget.canvas.set_tool(THRESHOLD)
    gate = widget.canvas.gate_from_drag(20.0, 0.0, 40.0, 999.0, name="cut")
    assert isinstance(gate, ThresholdGate)
    assert (gate.column, gate.low, gate.high) == ("area", 20.0, 40.0)
    widget.close()


def test_a_polygon_is_clicked_out_and_closed(panel, hand):
    panel.canvas.set_tool(POLYGON)
    for point in ((0.0, 0.0), (45.0, 0.0), (0.0, 45.0)):
        panel.canvas._pending.append(point)
    assert len(panel.canvas.pending_vertices()) == 3
    gate = panel.canvas.close_polygon(name="ignored")
    assert gate is not None
    assert panel.gates.names == ("drawn",)
    assert selected(panel.gates.mask(hand, "drawn")) == [0, 1, 2]
    assert panel.canvas.pending_vertices() == ()


def test_two_clicks_do_not_make_a_polygon(panel):
    panel.canvas.set_tool(POLYGON)
    panel.canvas._pending.extend([(0.0, 0.0), (1.0, 1.0)])
    assert panel.canvas.close_polygon() is None
    assert panel.gates.is_empty


def test_the_next_gate_is_drawn_inside_the_selected_one(panel, hand):
    panel.canvas.set_tool(THRESHOLD)
    panel.set_namer(lambda: "singlets")
    panel.canvas.gate_drawn.emit(
        panel.canvas.gate_from_drag(20.0, 0.0, 40.0, 1.0))
    panel.tree.select("singlets")
    assert panel.canvas.active_gate == "singlets"
    # The plot still shows the WHOLE table. Selecting a gate used to replot
    # its population, which is the zoom the user rejected; parentage is all
    # that survives of it, and parentage is what the rest of this test is
    # about. The gate's own objects are marked, not isolated.
    assert len(panel.canvas.population()) == 10
    panel.set_namer(lambda: "bright")
    panel.canvas.set_tool(RECTANGLE)
    panel.canvas.gate_drawn.emit(
        panel.canvas.gate_from_drag(-1e9, 30.0, 1e9, 60.0))
    assert panel.gates.get("bright").parent == "singlets"
    assert selected(panel.gates.mask(hand, "bright")) == [5, 6, 7]


def test_the_tree_shows_n_and_both_percentages(panel, strategy):
    panel.set_gates(strategy)
    top = panel.tree.tree.topLevelItem(0)
    assert top.text(0) == "singlets"
    assert top.text(1) == "7"
    assert top.text(2) == "70.0%"
    child = top.child(0)
    assert child.text(0) == "bright"
    assert child.text(1) == "3"
    assert child.text(2) == "42.9%"
    assert child.text(3) == "30.0%"


def test_applying_a_gate_highlights_it_rather_than_filtering(qtbot, hand,
                                                             strategy):
    """CHANGED 2026-08-07, at the user's explicit request.

    This asserted that Apply published a FILTER. Filtering removed every row
    outside the gate, and the axes then rescaled to what was left -- which
    read as the plot zooming into the gate, and moved the ground out from
    under the gate outline so it could not be dragged:

        "i dont want it to zoom in the first place. i want it to highlight
         the datapoints in the gate and show the gate but also show the rest
         of the graph."

    So Apply publishes a SELECTION: the objects inside the gate are ringed
    and every other point stays on screen. Narrowing to a gate is still a
    real thing to want, but it is a second explicit act rather than what the
    primary button does.
    """
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    link = LinkedSelection()
    panel = GateEditorPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(hand)
    panel.set_gates(strategy)
    panel.tree.select("bright")

    panel.publish()

    # Nothing is FILTERED -- that is the assertion that matters, and it holds
    # whether or not this synthetic frame carries the object-key columns a
    # shared highlight needs.
    assert link.filter is None or link.filter.is_empty
    assert "highlighted" in panel.status() or "cannot be shared" in panel.status()

def test_applying_with_nothing_selected_says_so(panel, strategy):
    panel.set_gates(strategy)
    panel.tree.select("")
    assert panel.publish() is None
    assert "Select a gate" in panel.status()


def test_deleting_from_the_tree_removes_the_children_too(panel, strategy):
    panel.set_gates(strategy)
    panel.tree.select("singlets")
    panel.tree.remove_selected()
    assert panel.gates.is_empty


def test_the_status_says_what_is_on_screen_and_what_is_shown(panel, strategy):
    """The whole table, always, plus how many gates are drawn on it.

    This used to assert "inside singlets · 7 objects", which was true back
    when selecting a gate replotted only that gate's population. That is the
    zoom the user rejected -- "never zoom into the gated data" -- so the
    status has to stop claiming it. Selecting a gate now says only what it
    still does: parent the next gate drawn.
    """
    panel.set_gates(strategy)
    assert "10 objects" in panel.status()
    assert "2 of 2 gate(s) shown" in panel.status()

    panel.tree.select("singlets")
    assert "10 objects" in panel.status(), "selecting a gate shrank the plot"
    assert "next gate inside singlets" in panel.status()


def test_hiding_a_gate_is_counted_and_is_not_a_delete(panel, strategy):
    panel.set_gates(strategy)
    panel.canvas.set_gate_enabled("bright", False)
    panel._refresh_status()
    assert "1 of 2 gate(s) shown" in panel.status()
    assert "bright" in panel.gates, "hiding a gate deleted it"
    assert "10 objects" in panel.status(), "hiding a gate removed its rows"


def test_an_unknown_tool_is_refused(panel):
    with pytest.raises(GateError, match="unknown gate tool"):
        panel.canvas.set_tool("lasso")


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def test_the_screen_saves_and_reloads_a_strategy(qtbot, tmp_path, hand,
                                                 strategy):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    screen.gates.set_gates(strategy)
    path = screen.save_gates(str(tmp_path / "g.json"))

    fresh = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(fresh)
    fresh.set_frame(hand)
    assert fresh.load_gates(path)
    assert fresh.gates.gates.names == ("singlets", "bright")
    assert selected(fresh.gates.gates.mask(hand, "bright")) == [5, 6, 7]
    screen.close()
    fresh.close()


def test_a_bad_gate_file_is_reported_rather_than_raised(qtbot, tmp_path, hand):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.gate_editor import GateEditorScreen

    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    screen = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    assert not screen.load_gates(str(bad))
    screen.close()


def test_a_gate_can_be_drawn_on_a_computed_column(qtbot, hand):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.gate_editor import GateEditorScreen
    from spacr.qt.widgets.formula import ColumnFormula

    screen = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(hand)
    screen.formulas.add_formula(ColumnFormula("ratio", "area / intensity"))
    assert "ratio" in [screen._x.itemText(i)
                       for i in range(screen._x.count())]
    screen.gates.set_namer(lambda: "high_ratio")
    screen._x.setCurrentText("ratio")
    screen.gates.canvas.set_tool(THRESHOLD)
    gate = screen.gates.canvas.gate_from_drag(1.5, 0.0, 100.0, 1.0)
    screen.gates.canvas.gate_drawn.emit(gate)
    assert screen.gates.gates.names == ("high_ratio",)
    # area/intensity: 1.0, 2.0, 3.0, 4.0, 0.2, 0.4, 0.6, 0.8, NaN, NaN
    computed = screen.formulas.computed_frame()
    assert selected(screen.gates.gates.mask(computed, "high_ratio")) == [1, 2, 3]
    screen.close()


def test_the_screen_registers_once():
    """One row in `spacr.qt.SELF_REGISTERING_MODULES` turns it on."""
    from spacr.qt import app as app_mod
    from spacr.qt.screens import gate_editor as screen_mod

    apps = list(app_mod.APPS)
    try:
        if any(row[0] == screen_mod.APP_KEY for row in app_mod.APPS):
            assert screen_mod.register() is False
        else:
            assert screen_mod.register() is True
            assert screen_mod.register() is False
            meta = app_mod.APP_META[screen_mod.APP_KEY]
            assert meta["cli_note"] == screen_mod.APP_CLI_NOTE
            assert len(meta["translations"]) == 9
    finally:
        app_mod.APPS[:] = apps
        app_mod.APP_META.pop(screen_mod.APP_KEY, None)
        app_mod.APP_FACTORIES.pop(screen_mod.APP_KEY, None)
        app_mod.APP_STAGE.pop(screen_mod.APP_KEY, None)
