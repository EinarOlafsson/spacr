"""Per-gate thresholds. Instruction 52, point 4.

    "the user should also be able to set thresholds for each individual gate
     for the measurements they are defined by"

``bounds()`` and ``range_filters()`` answer two different questions and it
matters which is which. ``range_filters`` is "what does this gate FILTER on" --
an unbounded axis filters on nothing and is rightly absent. ``bounds`` is
"what can the user SET", and an axis they cannot see in the panel is an axis
they cannot bound, which is the whole of point 4.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    BoxGate,
    CylinderGate,
    GateError,
    PolygonGate,
    PrismGate,
    RectGate,
    ThresholdGate,
)


def _cylinder(**kw):
    base = dict(name="c", u_column="a", v_column="b", axis_column="z",
                u_radius=1.0, v_radius=1.0)
    base.update(kw)
    return CylinderGate(**base)


def _prism(**kw):
    base = dict(name="p", u_column="a", v_column="b", axis_column="z",
                vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    base.update(kw)
    return PrismGate(**base)


# ---------------------------------------------------------------------------
# What can be set, not what is set
# ---------------------------------------------------------------------------

def test_an_unbounded_cylinder_still_offers_its_normal():
    """An axis the panel does not show is an axis nobody can bound."""
    assert _cylinder().thresholds() == {"z": (None, None)}
    assert _cylinder().range_filters() == ()


def test_an_unbounded_prism_still_offers_its_normal():
    assert _prism().thresholds() == {"z": (None, None)}


def test_a_box_offers_all_three_sides_unset():
    gate = BoxGate(name="b", x_column="a", y_column="b", z_column="z")
    assert set(gate.thresholds()) == {"a", "b", "z"}


def test_a_rectangle_offers_both_sides():
    gate = RectGate(name="r", x_column="a", y_column="b", x_low=0.0, x_high=1.0)
    assert set(gate.thresholds()) == {"a", "b"}


def test_a_threshold_offers_its_one_column():
    gate = ThresholdGate(name="t", column="a", low=0.0, high=1.0)
    assert gate.thresholds() == {"a": (0.0, 1.0)}


def test_a_polygon_offers_no_thresholds_because_it_is_not_a_range():
    """A bounding box would quietly include the corners."""
    gate = PolygonGate(name="p", x_column="a", y_column="b",
                       vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    assert gate.thresholds() == {}


def test_a_cylinder_does_not_offer_its_oval():
    """The oval is not a conjunction of ranges either -- only the normal is,
    and the normal is the axis the user needs to bound the height."""
    assert set(_cylinder().thresholds()) == {"z"}


# ---------------------------------------------------------------------------
# Setting one
# ---------------------------------------------------------------------------

def test_bounding_a_cylinders_normal_is_how_its_height_is_set():
    gate = _cylinder().with_threshold("z", 1.0, 4.0)
    assert gate.thresholds() == {"z": (1.0, 4.0)}
    frame = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0], "z": [2.0, 9.0]})
    assert gate.mask(frame).tolist() == [True, False]


def test_a_reversed_pair_is_ordered_rather_than_empty():
    assert _cylinder().with_threshold("z", 9.0, 2.0).thresholds() == {"z": (2.0, 9.0)}


@pytest.mark.parametrize("column", ["a", "b", "z"])
def test_every_side_of_a_box_can_be_set(column):
    gate = BoxGate(name="b", x_column="a", y_column="b", z_column="z")
    assert gate.with_threshold(column, 1.0, 2.0).thresholds()[column] == (1.0, 2.0)


def test_clearing_a_bound_puts_it_back_to_unbounded():
    gate = _cylinder(axis_low=1.0, axis_high=2.0).with_threshold("z", None, None)
    assert gate.thresholds() == {"z": (None, None)}
    assert gate.range_filters() == ()


def test_setting_a_bound_returns_a_new_gate_and_leaves_the_old_one():
    original = _cylinder()
    changed = original.with_threshold("z", 1.0, 2.0)
    assert original.axis_low is None
    assert changed is not original


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_bounding_a_measurement_the_gate_does_not_have_is_refused():
    """A gate that silently ignored the edit would leave the panel showing a
    number the gate does not honour."""
    with pytest.raises(GateError, match="no bound on 'a'"):
        _cylinder().with_threshold("a", 0.0, 1.0)


def test_the_refusal_names_what_can_be_set():
    with pytest.raises(GateError, match="it can be given z"):
        _prism().with_threshold("b", 0.0, 1.0)


def test_a_gate_with_no_thresholds_at_all_says_that():
    gate = PolygonGate(name="p", x_column="a", y_column="b",
                       vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(GateError, match="no thresholds at all"):
        gate.with_threshold("a", 0.0, 1.0)


def test_a_threshold_gate_refuses_another_column():
    gate = ThresholdGate(name="t", column="a", low=0.0, high=1.0)
    with pytest.raises(GateError, match="no bound on 'b'"):
        gate.with_threshold("b", 0.0, 1.0)
