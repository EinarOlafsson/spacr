"""Gate geometry: the open ends, the bad drags and the refusals.

A gate is a statement about measurements, so every edge here is a place where
the statement could quietly change meaning. An unbounded side has no midpoint
and must not acquire one; a bound dragged past its opposite has to swap rather
than invert; a shape drawn on one measurement against itself puts every point
on the diagonal; a composite whose operand was deleted must not become the
union of what is left. Each is refused or answered explicitly rather than
approximated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import gate_spec as gs
from spacr.qt.widgets.gate_spec import (BoxGate, CompositeGate, CylinderGate,
                                        EllipseGate, Gate, GateError, GateSet, PolygonGate,
                                        PrismGate, RectGate, ThresholdGate,
                                        WandError, wand_select)

#: A visible window, as the canvas hands one over: (x0, x1, y0, y1).
VIEW = (0.0, 100.0, 0.0, 100.0)


# ---------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------

def test_the_base_shape_implements_no_geometry():
    """Every kind states its own; a silent default would move a gate."""
    bare = Gate(name="bare")
    with pytest.raises(NotImplementedError):
        bare.translated(1.0, 1.0)
    with pytest.raises(NotImplementedError):
        bare.scaled(2.0)
    with pytest.raises(NotImplementedError):
        bare.centre()


def test_a_shape_with_nothing_to_pull_offers_no_handles():
    """Empty, not a guess: an invented handle invents a bound."""
    assert Gate(name="bare").handles(VIEW) == ()


def test_dragging_a_handle_a_shape_does_not_have_is_a_caller_bug():
    """A role the gate never offered is a bug in the canvas, not a user act."""
    with pytest.raises(GateError) as excinfo:
        Gate(name="bare").with_handle("x_low", 1.0, 2.0)
    assert "x_low" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ThresholdGate
# ---------------------------------------------------------------------------

def test_setting_a_threshold_on_its_own_column_reorders_the_pair():
    """A high typed below the low is the user's arithmetic, not their intent."""
    gate = ThresholdGate(name="live", column="area", low=1.0, high=9.0)
    edited = gate.with_threshold("area", 8.0, 2.0)
    assert (edited.low, edited.high) == (2.0, 8.0)


def test_a_closed_threshold_has_a_centre_on_its_own_axis_only():
    """It bounds one measurement, so the other axis has no middle."""
    gate = ThresholdGate(name="live", column="area", low=2.0, high=8.0)
    assert gate.centre() == (5.0, None)


def test_an_open_threshold_has_no_centre_to_resize_about():
    """A made-up centre would send the first resize somewhere arbitrary."""
    assert ThresholdGate(name="live", column="area", low=2.0).centre() == \
        (None, None)


def test_resizing_a_closed_threshold_grows_it_about_its_middle():
    """Both bounds move outward by the same factor from the centre."""
    gate = ThresholdGate(name="live", column="area", low=2.0, high=8.0)
    grown = gate.scaled(2.0)
    assert (grown.low, grown.high) == (-1.0, 11.0)


def test_resizing_an_open_threshold_leaves_the_open_end_open():
    """Adding to infinity would close a gate the user never drew."""
    gate = ThresholdGate(name="live", column="area", low=2.0)
    grown = gate.scaled(2.0, about=(0.0, 0.0))
    assert grown.low == 4.0
    assert grown.high is None


# ---------------------------------------------------------------------------
# RectGate
# ---------------------------------------------------------------------------

def _rect(**kwargs):
    base = dict(name="live", x_column="area", y_column="intensity",
                x_low=1.0, x_high=9.0, y_low=2.0, y_high=8.0)
    base.update(kwargs)
    return RectGate(**base)


def test_a_rectangle_open_on_one_side_reads_as_that_inequality():
    """The description is what the user checks the gate against."""
    gate = _rect(x_low=None, y_high=None)
    assert gate.describe() == "area ≤ 9 and intensity ≥ 2"


def test_setting_a_threshold_on_the_rectangle_s_y_axis_reorders_it():
    """Either axis can be typed in, and either can be typed backwards."""
    edited = _rect().with_threshold("intensity", 7.0, 3.0)
    assert (edited.y_low, edited.y_high) == (3.0, 7.0)


def test_setting_a_threshold_on_a_column_the_rectangle_lacks_is_refused():
    """Silently ignoring it would leave the gate saying what it said."""
    with pytest.raises(GateError):
        _rect().with_threshold("perimeter", 1.0, 2.0)


def test_a_rectangle_open_on_one_axis_has_no_centre_on_that_axis():
    """Half of a centre is still the honest answer for a half-open shape."""
    assert _rect(y_high=None).centre() == (5.0, None)


def test_a_rectangle_has_no_handle_it_did_not_offer():
    """The canvas names sides; anything else is a bug in the canvas."""
    with pytest.raises(GateError) as excinfo:
        _rect().with_handle("radius", 1.0, 2.0)
    assert "radius" in str(excinfo.value)


@pytest.mark.parametrize("role", [
    "x_low,x_high",
    "y_low,y_high",
    "x_low,x_low",
    "x_low,y_low,y_high",
])
def test_a_rectangle_refuses_side_combinations_it_did_not_offer(role):
    """Only the four sides and four emitted corners are draggable."""
    with pytest.raises(GateError) as excinfo:
        _rect().with_handle(role, 1.0, 2.0)
    assert role in str(excinfo.value)


# ---------------------------------------------------------------------------
# PolygonGate
# ---------------------------------------------------------------------------

def _polygon(**kwargs):
    base = dict(name="blob", x_column="area", y_column="intensity",
                vertices=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    base.update(kwargs)
    return PolygonGate(**base)


def test_a_polygon_reports_its_bounding_box_for_drawing():
    """For drawing only: masking through the box would take the corners."""
    assert _polygon().bounds() == (0.0, 4.0, 0.0, 4.0)


# ---------------------------------------------------------------------------
# EllipseGate
# ---------------------------------------------------------------------------

def _ellipse(**kwargs):
    base = dict(name="oval", x_column="area", y_column="intensity",
                x_centre=10.0, y_centre=20.0, x_radius=2.0, y_radius=4.0)
    base.update(kwargs)
    return EllipseGate(**base)


def test_an_ellipse_describes_its_centre_and_both_radii():
    """One radius would read as a circle on two different scales."""
    assert _ellipse().describe() == \
        "area/intensity within (10±2, 20±4)"


def test_an_ellipse_offers_axis_handles_and_bounding_box_corners():
    """The corner at 45 degrees is inside the curve and feels like a miss."""
    handles = _ellipse().handles(VIEW)
    assert len(handles) == 8
    assert [h.role for h in handles].count("x_radius") == 2
    corners = [(h.x, h.y) for h in handles if h.corner]
    assert (12.0, 24.0) in corners
    assert (8.0, 16.0) in corners


def test_resizing_an_ellipse_about_a_point_moves_it_as_well_as_grows_it():
    """Scaling about a corner is a zoom, not a growth in place."""
    grown = _ellipse().scaled(2.0, about=(0.0, 0.0))
    assert (grown.x_centre, grown.y_centre) == (20.0, 40.0)
    assert (grown.x_radius, grown.y_radius) == (4.0, 8.0)


# ---------------------------------------------------------------------------
# The hull
# ---------------------------------------------------------------------------

def test_two_points_are_their_own_hull():
    """A pair has no interior, so there is nothing to discard."""
    hull = gs._convex_hull(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert hull.shape == (2, 2)


# ---------------------------------------------------------------------------
# BoxGate
# ---------------------------------------------------------------------------

def _box(**kwargs):
    base = dict(name="cube", x_column="area", y_column="intensity",
                z_column="depth", x_low=0.0, x_high=10.0,
                y_low=0.0, y_high=20.0, z_low=0.0, z_high=30.0)
    base.update(kwargs)
    return BoxGate(**base)


def test_a_box_filters_only_on_the_sides_it_bounds():
    """An unbounded side filters on nothing and must not appear as a clause."""
    filters = _box(y_low=None, y_high=None).range_filters()
    assert [f.column for f in filters] == ["area", "depth"]


def test_a_box_describes_each_of_its_three_sides():
    """Any / ≤ / ≥ are three different statements about the same axis."""
    text = _box(x_low=None, y_low=None, y_high=None, z_high=None).describe()
    assert text == "area ≤ 10 and any intensity and depth ≥ 0"


def test_moving_a_box_leaves_its_open_ends_open():
    """Adding to an unbounded side would close it."""
    moved = _box(x_high=None).translated(2.0, 3.0)
    assert moved.x_low == 2.0
    assert moved.x_high is None
    assert (moved.y_low, moved.y_high) == (3.0, 23.0)


def test_a_box_open_on_one_axis_has_no_centre_on_that_axis():
    """A half-open axis has no middle, and inventing one moves the box."""
    assert _box(y_high=None).centre() == (5.0, None)


def test_resizing_a_box_grows_it_about_its_own_centre():
    """Every other kind scales about the centre; a box has to agree, or the
    same drag moves a box somewhere a rectangle would not go."""
    grown = _box().scaled(2.0)
    assert (grown.x_low, grown.x_high) == (-5.0, 15.0)
    assert (grown.y_low, grown.y_high) == (-10.0, 30.0)


def test_setting_a_threshold_on_a_column_the_box_lacks_is_refused():
    """The box knows three columns; a fourth is not silently ignored, and the
    refusal has to be the GateError every other kind raises."""
    with pytest.raises(GateError):
        _box().with_threshold("perimeter", 1.0, 2.0)


# ---------------------------------------------------------------------------
# CylinderGate and PrismGate
# ---------------------------------------------------------------------------

def _cylinder(**kwargs):
    base = dict(name="tube", u_column="area", v_column="intensity",
                axis_column="depth", u_centre=1.0, v_centre=2.0,
                u_radius=1.0, v_radius=2.0)
    base.update(kwargs)
    return CylinderGate(**base)


def test_a_cylinder_open_below_reads_as_an_upper_bound_on_its_axis():
    """The oval plus one inequality is the whole statement."""
    assert _cylinder(axis_high=5.0).describe().endswith("depth ≤ 5")


def test_a_cylinder_open_above_reads_as_a_lower_bound_on_its_axis():
    """The mirror case."""
    assert _cylinder(axis_low=5.0).describe().endswith("depth ≥ 5")


def _prism(**kwargs):
    base = dict(name="wedge", u_column="area", v_column="intensity",
                axis_column="depth",
                vertices=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    base.update(kwargs)
    return PrismGate(**base)


def test_a_prism_bounded_below_keeps_only_objects_above_that_depth():
    """The polygon says where, the axis bound says how deep."""
    frame = pd.DataFrame({"area": [1.0, 1.0, 1.0],
                          "intensity": [1.0, 1.0, 1.0],
                          "depth": [-1.0, 5.0, 50.0]})
    mask = _prism(axis_low=0.0, axis_high=10.0).mask(frame)
    assert list(mask) == [False, True, False]


def test_a_prism_open_below_reads_as_an_upper_bound_on_its_axis():
    """Same sentence shape as the cylinder, for the same reason."""
    assert _prism(axis_high=5.0).describe().endswith("depth ≤ 5")


def test_a_prism_open_above_reads_as_a_lower_bound_on_its_axis():
    """The mirror case."""
    assert _prism(axis_low=5.0).describe().endswith("depth ≥ 5")


def test_bounding_a_prism_s_axis_reorders_a_backwards_pair():
    """This is how the height is typed in, and it can be typed backwards."""
    bounded = _prism().with_threshold("depth", 9.0, 1.0)
    assert (bounded.axis_low, bounded.axis_high) == (1.0, 9.0)


# ---------------------------------------------------------------------------
# CompositeGate
# ---------------------------------------------------------------------------

def test_a_composite_names_its_kind():
    """The set reads the kind to know it must be evaluated with a lookup."""
    assert CompositeGate(name="either", operands=("a", "b")).kind == \
        gs.COMPOSITE


def test_a_composite_whose_operand_was_deleted_is_refused():
    """Quietly becoming the union of what is left changes what it means."""
    gate = CompositeGate(name="either", operands=("a", "b"))
    frame = pd.DataFrame({"area": [1.0, 2.0]})
    with pytest.raises(GateError) as excinfo:
        gate.mask_with(frame, {"a": np.array([True, False])})
    assert "'b'" in str(excinfo.value) or "b," in str(excinfo.value) or \
        "b" in str(excinfo.value)


def test_a_composite_has_no_centre_of_its_own():
    """Its shape is its operands', and only the set can see them."""
    assert CompositeGate(name="either", operands=("a", "b")).centre() == \
        (None, None)


# ---------------------------------------------------------------------------
# The wand
# ---------------------------------------------------------------------------

def test_a_click_on_a_plot_with_no_finite_objects_is_refused():
    """There is nothing to grow a gate from, and saying so beats an empty one."""
    frame = pd.DataFrame({"area": [np.nan, np.inf],
                          "intensity": [1.0, np.nan]})
    with pytest.raises(WandError) as excinfo:
        wand_select(frame, "area", "intensity", 1.0, 1.0)
    assert "nothing to grow" in str(excinfo.value)


def test_an_unscaled_wand_works_in_the_measurements_own_units():
    """Turning scaling off means tolerance is a distance in the data itself."""
    rng = np.random.default_rng(0)
    blob = rng.normal(loc=0.0, scale=0.05, size=(30, 2))
    far = rng.normal(loc=10.0, scale=0.05, size=(30, 2))
    points = np.vstack([blob, far])
    frame = pd.DataFrame({"area": points[:, 0], "intensity": points[:, 1]})
    mask = wand_select(frame, "area", "intensity", 0.0, 0.0,
                       tolerance=0.5, max_radius=2.0, scale=False)
    assert mask[:30].all()
    assert not mask[30:].any()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_a_silhouette_that_cannot_be_computed_is_reported_as_unknown(
        monkeypatch):
    """A score that could not be measured is not a score of zero."""
    import sklearn.metrics as metrics

    def _boom(*_args, **_kwargs):
        raise ValueError("number of labels is invalid")

    monkeypatch.setattr(metrics, "silhouette_score", _boom)
    rng = np.random.default_rng(1)
    points = np.vstack([rng.normal(loc=0.0, scale=0.1, size=(40, 2)),
                        rng.normal(loc=5.0, scale=0.1, size=(40, 2))])
    frame = pd.DataFrame({"area": points[:, 0], "intensity": points[:, 1]})
    candidates = gs.cluster_walk_candidates(
        frame, "area", "intensity", eps=0.5, min_samples=5, steps=3,
        span=2.0)
    assert candidates
    assert all(c.silhouette is None for c in candidates)


def test_a_collinear_cluster_is_skipped_rather_than_widened():
    """A line has no area, and widening it would select rows outside it."""
    line = np.column_stack([np.linspace(0.0, 1.0, 40),
                            np.linspace(0.0, 1.0, 40)])
    rng = np.random.default_rng(2)
    blob = rng.normal(loc=50.0, scale=0.2, size=(40, 2))
    points = np.vstack([line, blob])
    frame = pd.DataFrame({"area": points[:, 0], "intensity": points[:, 1]})
    gates = gs.cluster_gates(frame, "area", "intensity", eps=0.4,
                             min_samples=5, scale=True)
    assert gates
    for gate in gates:
        assert len(gate.vertices) >= 3


# ---------------------------------------------------------------------------
# GateClause and GateSet
# ---------------------------------------------------------------------------

def test_an_empty_gate_set_reports_that_it_has_no_gates():
    """A blank report reads as a report that failed to run."""
    gates = GateSet()
    assert gates.report(pd.DataFrame({"area": [1.0]})) == "no gates"
    assert gates.describe() == "no gates"


