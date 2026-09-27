"""A view gate refuses what has no meaning, and moves on its own view.

A 3-D lasso (``ViewGate``) needs three distinct measurements, a 4 x 4
camera and an outline of at least three points; anything less is refused
with a sentence naming the gate. Once made, it can be moved and resized in
its own view coordinates, and it round-trips through its saved form with
or without the optional camera angles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import GateError, ViewGate, gate_from_dict

IDENTITY = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
SQUARE = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))


def _gate(**over):
    fields = dict(name="lasso", x_column="a", y_column="b", z_column="c",
                  projection=IDENTITY, vertices=SQUARE)
    fields.update(over)
    return ViewGate(**fields)


@pytest.mark.parametrize("blank", ["x_column", "y_column", "z_column"])
def test_a_blank_measurement_is_refused_by_name(blank):
    with pytest.raises(GateError, match=f"'lasso' has no {blank}"):
        _gate(**{blank: "   "})


def test_a_camera_that_is_not_numbers_is_refused():
    with pytest.raises(GateError, match="four rows of four numbers"):
        _gate(projection=(("near", "far"),))


def test_an_outline_of_two_points_is_refused():
    with pytest.raises(GateError, match="has 2 vertices"):
        _gate(vertices=((0.0, 0.0), (1.0, 1.0)))


def test_moving_the_outline_moves_what_it_selects():
    frame = pd.DataFrame({"a": [1.0, 3.0], "b": [1.0, 1.0], "c": [0.0, 0.0]})
    gate = _gate()
    assert gate.mask(frame).tolist() == [True, False]

    moved = gate.translated(2.0, 0.0)

    assert moved.vertices[0] == (2.0, 0.0)
    assert moved.mask(frame).tolist() == [False, True]


def test_the_centre_is_the_middle_of_the_outline():
    assert _gate().centre() == (1.0, 1.0)


def test_resizing_about_the_centre_keeps_the_centre():
    grown = _gate().scaled(2.0)

    assert grown.centre() == (1.0, 1.0)
    assert grown.vertices[0] == (-1.0, -1.0)
    assert grown.vertices[2] == (3.0, 3.0)


def test_resizing_about_a_corner_keeps_that_corner():
    shrunk = _gate().scaled(0.5, about=(0.0, 0.0))
    assert shrunk.vertices == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_a_non_positive_factor_is_refused():
    with pytest.raises((GateError, ValueError)):
        _gate().scaled(0.0)


def test_the_description_names_the_angle_only_when_it_is_known():
    assert "elevation" not in _gate().describe()
    angled = _gate(view=(30.0, -60.0, 0.0)).describe()
    assert "elevation 30°" in angled and "azimuth -60°" in angled


def test_a_saved_gate_without_its_optional_parts_reads_back():
    payload = _gate().to_dict()
    for key in ("limits", "view"):
        payload.pop(key)

    back = gate_from_dict(payload)

    assert isinstance(back, ViewGate)
    assert back.vertices == SQUARE
    assert back.view == () and back.limits == ()
    assert np.allclose(np.asarray(back.projection), np.eye(4))


def test_a_saved_gate_with_its_angles_reads_back_the_same():
    gate = _gate(view=(10.0, 20.0, 0.0), limits=((0.0, 1.0), (0.0, 2.0)))
    assert gate_from_dict(gate.to_dict()) == gate
