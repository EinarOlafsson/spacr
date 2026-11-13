"""Table-driven tests of the gate shapes in ``spacr.qt.widgets.gate_spec``.

A gate is a statement about measurements, so every edge here is a place where
the statement could quietly change meaning: a shape drawn on one measurement
against itself, an unbounded side, a bound dragged past its opposite, a handle
the shape never offered, a population that ignores its parent gate.  One
general test per unit, each walking a case table with pytest's ``subtests``
fixture and asserting the exact value or the exact refusal message.
Pure Python: no Qt widget is built here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from spacr import filters as flt
from spacr.filters import FilterError
from spacr.qt.widgets import gate_spec as gs
from spacr.qt.widgets.gate_spec import (BoxGate, EditableVolumeGate,
                                        EllipseGate, Gate, GateClause,
                                        GateError, GateSet, PolygonGate,
                                        RectGate, ThresholdGate)

SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
TRIANGLE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def _threshold(**kw):
    base = dict(name="live", column="area", low=2.0, high=8.0)
    base.update(kw)
    return ThresholdGate(**base)


def _polygon(**kw):
    base = dict(name="blob", x_column="area", y_column="intensity",
                vertices=SQUARE)
    base.update(kw)
    return PolygonGate(**base)


def _ellipse(**kw):
    base = dict(name="oval", x_column="area", y_column="intensity",
                x_centre=10.0, y_centre=20.0, x_radius=2.0, y_radius=4.0)
    base.update(kw)
    return EllipseGate(**base)


def _box(**kw):
    base = dict(name="cube", x_column="area", y_column="intensity",
                z_column="depth", x_low=0.0, x_high=10.0,
                y_low=0.0, y_high=20.0, z_low=0.0, z_high=30.0)
    base.update(kw)
    return BoxGate(**base)


def _ellipsoid(**kw):
    base = dict(name="e", x_column="a", y_column="b", z_column="c",
                x_centre=1.0, y_centre=2.0, z_centre=0.0,
                x_radius=-2.0, y_radius=2.0, z_radius=2.0)
    base.update(kw)
    return gs._EllipsoidGate(**base)


@dataclass(frozen=True)
class _Sphere(Gate):
    """A three-axis gate type the editable volume does not know."""

    @property
    def columns(self):
        return ("x", "y", "z")


def test_a_gate_that_cannot_mean_anything_is_refused_by_name(subtests):
    """Every shape refuses a missing / repeated measurement or an empty statement."""
    cases = [
        ("threshold_blank_column",
         lambda: ThresholdGate(name="live", column="  ", low=1.0),
         "threshold gate 'live' has no column; a cut has to be a cut on something"),
        ("rect_blank_y",
         lambda: RectGate(name="live", x_column="area", y_column="", x_low=1.0),
         "rectangle gate 'live' has no y_column; a rectangle is drawn on two measurements"),
        ("rect_no_bounds",
         lambda: RectGate(name="live", x_column="area", y_column="intensity"),
         "rectangle gate 'live' has no bounds at all, so it selects everything"),
        ("polygon_blank_x",
         lambda: PolygonGate(name="blob", x_column="", y_column="intensity",
                             vertices=TRIANGLE),
         "polygon gate 'blob' has no x_column; a polygon is drawn on two measurements"),
        ("polygon_against_itself",
         lambda: PolygonGate(name="blob", x_column="area", y_column="area",
                             vertices=TRIANGLE),
         "polygon gate 'blob' is drawn on 'area' against itself"),
        ("ellipse_blank_y",
         lambda: EllipseGate(name="oval", x_column="area", y_column="",
                             x_radius=1.0, y_radius=1.0),
         "ellipse gate 'oval' has no y_column; an ellipse is drawn on two measurements"),
        ("ellipse_against_itself",
         lambda: EllipseGate(name="oval", x_column="area", y_column="area",
                             x_radius=1.0, y_radius=1.0),
         "ellipse gate 'oval' is drawn on 'area' against itself"),
        ("box_blank_z",
         lambda: BoxGate(name="cube", x_column="area", y_column="intensity",
                         z_column="", x_low=0.0, x_high=1.0),
         "box gate 'cube' has no z_column; a box is drawn on three measurements"),
        ("box_from_two_limits",
         lambda: BoxGate.from_limits("cube", ("area", "intensity"),
                                     ((0.0, 1.0), (0.0, 1.0))),
         "a box gate needs three measurements and three ranges"),
        ("ellipsoid_blank_y", lambda: _ellipsoid(y_column=" "),
         "ellipsoid gate 'e' has no y_column; an ellipsoid is drawn on three measurements"),
        ("ellipsoid_repeated", lambda: _ellipsoid(z_column="a"),
         "ellipsoid gate 'e' names the same measurement twice"),
        ("editable_from_two_axes",
         lambda: EditableVolumeGate.from_gate(RectGate(
             name="flat", x_column="x", y_column="y",
             x_low=0, x_high=1, y_low=0, y_high=1)),
         "Only a three-dimensional gate can become an editable volume."),
        ("editable_unknown_kind",
         lambda: EditableVolumeGate.from_gate(_Sphere(name="sphere"), [(0, 1)] * 3),
         None),
        ("editable_too_few_segments",
         lambda: EditableVolumeGate.from_gate(
             BoxGate.from_limits("volume", ("x", "y", "z"), [(0, 1)] * 3),
             segments=4),
         "Volume sampling needs 8 to 128 circumference samples and 1 to 16 depth intervals."),
    ]
    for case_id, build, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(GateError) as excinfo:
                build()
            if message is None:
                assert "no supported finite editable volume" in str(excinfo.value)
            else:
                assert str(excinfo.value) == message


def test_a_gate_reads_back_its_kind_columns_and_statement(subtests):
    """kind / columns / describe() and the normalised bounds are exactly the gate."""
    cases = [
        ("threshold_both", _threshold(), gs.THRESHOLD, ("area",), "2 ≤ area ≤ 8"),
        ("threshold_open_above", _threshold(name="bright", low=5.0, high=None),
         gs.THRESHOLD, ("area",), "area ≥ 5"),
        ("threshold_open_below", _threshold(name="dim", low=None, high=5.0),
         gs.THRESHOLD, ("area",), "area ≤ 5"),
        ("rectangle", RectGate(name="r", x_column="area", y_column="i",
                               x_low=1.0, x_high=9.0, y_low=2.0, y_high=8.0),
         gs.RECTANGLE, ("area", "i"), "1 ≤ area ≤ 9 and 2 ≤ i ≤ 8"),
        ("ellipse", _ellipse(), gs.ELLIPSE, ("area", "intensity"),
         "area/intensity within (10±2, 20±4)"),
        ("polygon", _polygon(), gs.POLYGON, ("area", "intensity"),
         "4-sided region on area × intensity"),
        ("box", _box(), gs.BOX, ("area", "intensity", "depth"), None),
    ]
    for case_id, gate, kind, columns, text in cases:
        with subtests.test(msg=case_id):
            assert gate.kind == kind
            assert gate.columns == columns
            if text is not None:
                assert gate.describe() == text
    with subtests.test(msg="box_side_typed_backwards_is_reordered"):
        gate = _box(z_low=30.0, z_high=5.0)
        assert (gate.z_low, gate.z_high) == (5.0, 30.0)
    with subtests.test(msg="ellipsoid_negative_radius_is_its_magnitude"):
        assert (_ellipsoid().x_radius, _ellipsoid().y_radius) == (2.0, 2.0)
    with subtests.test(msg="re_parenting_is_a_new_value"):
        gate = ThresholdGate(name="live", column="area", low=1.0)
        assert gate.with_parent("singlets").parent == "singlets"
        assert gate.with_parent("singlets").describe() == gate.describe()
        assert gate.parent is None
    with subtests.test(msg="handles_are_the_roles_the_drag_accepts"):
        assert [(h.x, h.y, h.role) for h in _threshold().handles((0, 100, 0, 100))] \
            == [(2.0, 50.0, "low"), (8.0, 50.0, "high")]
        assert [h.role for h in _ellipse().handles((0, 100, 0, 100))] == [
            "x_radius", "x_radius", "y_radius", "y_radius"] + ["x_radius,y_radius"] * 4


def test_dragging_a_handle_moves_exactly_that_bound_or_is_refused(subtests):
    """with_handle moves one bound/vertex/radius; a role never offered names itself."""
    moves = [
        ("threshold_low_past_high_swaps", _threshold(), ("low", 12.0, 0.0),
         lambda g: (g.low, g.high), (8.0, 12.0)),
        ("threshold_high_only", _threshold(), ("high", 20.0, 0.0),
         lambda g: (g.low, g.high), (2.0, 20.0)),
        ("polygon_vertex_two", _polygon(), ("vertex:2", 9.0, 9.0),
         lambda g: g.vertices, ((0.0, 0.0), (4.0, 0.0), (9.0, 9.0), (0.0, 4.0))),
        ("ellipse_y_radius_only", _ellipse(), ("y_radius", 10.0, 30.0),
         lambda g: (g.x_centre, g.y_centre, g.x_radius, g.y_radius),
         (10.0, 20.0, 2.0, 10.0)),
    ]
    for case_id, gate, (role, x, y), read, want in moves:
        with subtests.test(msg=case_id):
            moved = gate.with_handle(role, x, y)
            assert type(moved) is type(gate)
            assert read(moved) == want
            assert moved.name == gate.name
    with subtests.test(msg="ellipse_handle_onto_its_centre_is_a_quiet_no"):
        gate = _ellipse()
        assert gate.with_handle("x_radius", 10.0, 20.0) is gate
    refusals = [
        ("threshold_y_high", _threshold(), "y_high", "threshold gate has no handle 'y_high'"),
        ("polygon_side", _polygon(), "x_low", "polygon gate has no handle 'x_low'"),
        ("polygon_unindexed_vertex", _polygon(), "vertex:corner",
         "polygon gate has no handle 'vertex:corner'"),
        ("ellipse_side", _ellipse(), "x_low", "ellipse gate has no handle 'x_low'"),
    ]
    for case_id, gate, role, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(GateError) as excinfo:
                gate.with_handle(role, 1.0, 2.0)
            assert str(excinfo.value) == message


def test_a_gate_set_holds_names_populations_and_clauses(subtests):
    """remove/clear/describe/population/clause, and the annotation refusal."""
    def cells_and_bright():
        return GateSet([
            ThresholdGate(name="cells", column="area", low=2.0),
            ThresholdGate(name="bright", parent="cells", column="intensity", low=5.0),
        ])

    with subtests.test(msg="removing_an_absent_gate_changes_nothing"):
        gates = GateSet([ThresholdGate(name="cells", column="area", low=1.0)])
        assert gates.remove("nope") is gates
        assert gates.names == ("cells",)
    with subtests.test(msg="clearing_leaves_nothing"):
        gates = cells_and_bright()
        assert gates.clear() is gates
        assert gates.names == ()
        assert gates.describe() == "no gates"
    with subtests.test(msg="describe_names_every_gate"):
        assert cells_and_bright().describe() == \
            "cells: area ≥ 2 · bright: intensity ≥ 5"
    with subtests.test(msg="population_is_the_gate_and_its_ancestors"):
        frame = pd.DataFrame({"area": [1.0, 3.0, 3.0], "intensity": [9.0, 1.0, 9.0]})
        gates = cells_and_bright()
        assert list(gates.population(frame, "bright").index) == [2]
        assert list(gates.population(frame, "cells").index) == [1, 2]
    with subtests.test(msg="clause_is_named_for_the_leaf"):
        root = ThresholdGate(name="cells", column="area", low=1.0)
        leaf = ThresholdGate(name="bright", parent="cells", column="intensity", low=5.0)
        clause = GateClause((root, leaf))
        assert (clause.name, clause.column) == ("bright", "gate:bright")
    with subtests.test(msg="a_gate_the_table_cannot_take_names_itself"):
        gates = GateSet().add(RectGate(name="big", x_column="area",
                                       y_column="perimeter", x_low=0.0,
                                       x_high=100.0, y_low=0.0, y_high=100.0))
        with pytest.raises(FilterError, match="gate 'big' cannot be applied"):
            flt.annotate_from_gates(pd.DataFrame({"area": [1.0, 2.0]}), gates, ["big"])
