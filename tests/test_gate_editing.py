"""Gates can be moved and resized after they are drawn.

The single biggest gap in the Gate Editor: a gate you cannot adjust is a
gate you redraw from scratch, and gating is how a screen becomes a
population.

Both operations return a NEW gate rather than mutating. These are frozen
dataclasses, a GateSet holds them by name, and an in-place edit would change
a gate something else is already holding a reference to.

The load-bearing property throughout is that the gate still means what it
looks like: after a move, the objects it selects are the objects under it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    GateError, PolygonGate, RectGate, ThresholdGate, gate_from_dict,
)


@pytest.fixture
def frame():
    xs, ys = np.meshgrid(np.arange(0, 21, 1.0), np.arange(0, 21, 1.0))
    return pd.DataFrame({"x_measure": xs.ravel(), "y_measure": ys.ravel()})


@pytest.fixture
def rect():
    return RectGate(name="box", x_column="x_measure", y_column="y_measure",
                    x_low=0, x_high=10, y_low=0, y_high=10)


@pytest.fixture
def poly():
    return PolygonGate(name="blob", x_column="x_measure",
                       y_column="y_measure",
                       vertices=((0, 0), (10, 0), (10, 10), (0, 10)))


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def test_moving_a_rectangle_moves_what_it_selects(rect, frame):
    """The point of the whole feature: the population follows the shape."""
    before = set(np.flatnonzero(rect.mask(frame)))
    moved = rect.translated(10.0, 10.0)
    after = set(np.flatnonzero(moved.mask(frame)))

    assert before != after
    assert len(after) == len(before), "a move must not change the size"
    # The moved gate covers the far corner; the original did not.
    assert moved.mask(frame)[(frame.x_measure == 20) & (frame.y_measure == 20)].all()
    assert not rect.mask(frame)[(frame.x_measure == 20) & (frame.y_measure == 20)].any()


def test_moving_a_polygon_moves_every_vertex(poly):
    moved = poly.translated(3.0, -2.0)
    assert moved.vertices == ((3.0, -2.0), (13.0, -2.0),
                              (13.0, 8.0), (3.0, 8.0))


def test_a_threshold_ignores_the_second_axis():
    """It is a cut on ONE column, so it has no y to move along."""
    gate = ThresholdGate(name="cut", column="x_measure", low=2, high=6)
    moved = gate.translated(3.0, 999.0)
    assert (moved.low, moved.high) == (5.0, 9.0)


def test_an_open_end_stays_open_when_moved():
    """None means "unbounded on this side". Adding to it would turn an open
    gate into a closed one the user never drew."""
    gate = ThresholdGate(name="cut", column="x_measure", low=5, high=None)
    moved = gate.translated(2.0, 0.0)
    assert moved.low == 7.0
    assert moved.high is None


def test_moving_returns_a_new_gate(rect):
    """Frozen dataclasses, held by name in a GateSet -- an in-place edit
    would change a gate something else already references."""
    moved = rect.translated(1.0, 1.0)
    assert moved is not rect
    assert rect.x_low == 0, "the original was mutated"


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def test_growing_a_rectangle_about_its_centre(rect):
    grown = rect.scaled(2.0)
    assert (grown.x_low, grown.x_high) == (-5.0, 15.0)
    assert (grown.y_low, grown.y_high) == (-5.0, 15.0)
    # The centre is what stays put.
    assert grown.centre() == rect.centre()


def test_shrinking_selects_a_subset(rect, frame):
    small = rect.scaled(0.5)
    assert set(np.flatnonzero(small.mask(frame))) < set(
        np.flatnonzero(rect.mask(frame)))


def test_resizing_about_a_grabbed_point_holds_that_point(poly):
    """"Click and pull" anchors on the opposite side, not the centre."""
    grown = poly.scaled(2.0, about=(0.0, 0.0))
    assert (0.0, 0.0) in grown.vertices
    assert (20.0, 20.0) in grown.vertices


def test_a_polygon_centre_is_the_vertex_centroid(poly):
    """Not the area centroid: for a strongly concave polygon that can sit
    OUTSIDE the shape, which makes a resize look like a move."""
    assert poly.centre() == (5.0, 5.0)


def test_a_half_open_threshold_has_no_centre_and_does_not_move():
    """A made-up centre would send the first resize somewhere arbitrary."""
    gate = ThresholdGate(name="cut", column="x_measure", low=5, high=None)
    assert gate.centre() == (None, None)
    assert gate.scaled(2.0) == gate


@pytest.mark.parametrize("factor", [0, -1, -0.5])
def test_a_non_positive_resize_is_refused(rect, factor):
    """Zero collapses the gate to a point and a negative turns it inside
    out. The arithmetic would accept both silently."""
    with pytest.raises(GateError, match="must be positive"):
        rect.scaled(factor)


# ---------------------------------------------------------------------------
# Per-vertex editing
# ---------------------------------------------------------------------------

def test_one_vertex_can_be_dragged(poly):
    edited = poly.with_vertex(2, 20.0, 20.0)
    assert edited.vertices[2] == (20.0, 20.0)
    assert edited.vertices[0] == poly.vertices[0], "other vertices moved"


def test_a_vertex_index_outside_the_polygon_is_refused(poly):
    """It would otherwise silently move a different corner than the one
    grabbed."""
    with pytest.raises(GateError, match="no vertex"):
        poly.with_vertex(9, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Everything still round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    lambda g: g.translated(3.0, 4.0),
    lambda g: g.scaled(1.5),
])
def test_an_edited_gate_still_serialises(rect, poly, frame, edit):
    """Persistence is free only if the edited gate is still an ordinary
    gate. If it were not, saved gates would come back in their pre-edit
    position."""
    for gate in (rect, poly):
        edited = edit(gate)
        restored = gate_from_dict(edited.to_dict())
        assert np.array_equal(restored.mask(frame), edited.mask(frame))


def test_editing_preserves_name_and_parent(poly):
    """The hierarchy must survive a drag, or moving a child re-parents it."""
    child = PolygonGate(name="child", parent="live cells",
                        x_column="x_measure", y_column="y_measure",
                        vertices=poly.vertices)
    for edited in (child.translated(1, 1), child.scaled(2.0)):
        assert edited.name == "child"
        assert edited.parent == "live cells"
