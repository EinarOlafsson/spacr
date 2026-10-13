"""Gates drawn on one plane of the volume and extended along the third.

Instruction 52:

    "in 3D mode the user should be able to pull a circle (which becomes a
     cylinder gate), a square/rectangle, or draw a polygon on this chosen 2d
     plane, which gets translated to 3 dims when the gate is generated."

The rectangle half already existed as ``BoxGate``. These are the other two,
and they follow BoxGate's principle rather than inventing a second one: the
anchor plane is named by ITS COLUMNS, not by a "xy"/"xz" string, so there is
no second field that can disagree with the columns, and the gate reads the
same from every camera angle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    GATE_KINDS,
    CylinderGate,
    EllipseGate,
    GateError,
    PolygonGate,
    PrismGate,
    gate_from_dict,
)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "a": [0.0, 0.5, 3.0, 0.0, 0.0],
        "b": [0.0, 0.0, 0.0, 3.0, 0.0],
        "z": [1.0, 5.0, 1.0, 1.0, 50.0],
    })


def _cyl(**kw):
    base = dict(name="c", u_column="a", v_column="b", axis_column="z",
                u_centre=0.0, v_centre=0.0, u_radius=1.0, v_radius=1.0)
    base.update(kw)
    return CylinderGate(**base)


def _prism(**kw):
    base = dict(name="p", u_column="a", v_column="b", axis_column="z",
                vertices=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    base.update(kw)
    return PrismGate(**base)


# ---------------------------------------------------------------------------
# The default along the normal is a decision, not an accident
# ---------------------------------------------------------------------------

def test_an_unbounded_cylinder_means_what_the_2d_oval_meant(frame):
    """So drawing in 3D and drawing in 2D agree, and narrowing is explicit."""
    cylinder = _cyl()
    oval = EllipseGate(name="c", x_column="a", y_column="b",
                       x_centre=0.0, y_centre=0.0, x_radius=1.0, y_radius=1.0)
    assert np.array_equal(cylinder.mask(frame), oval.mask(frame))


def test_an_unbounded_prism_means_what_the_2d_polygon_meant(frame):
    prism = _prism()
    polygon = PolygonGate(name="p", x_column="a", y_column="b",
                          vertices=prism.vertices)
    assert np.array_equal(prism.mask(frame), polygon.mask(frame))


def test_bounding_the_normal_narrows_it(frame):
    wide = _cyl().mask(frame)
    narrow = _cyl(axis_low=0.0, axis_high=2.0).mask(frame)
    assert narrow.sum() < wide.sum()
    # Row 4 is inside the oval but at z=50, so only the bound excludes it.
    assert wide[4] and not narrow[4]


def test_the_bounds_are_inclusive_at_both_ends(frame):
    gate = _cyl(axis_low=1.0, axis_high=1.0)
    assert gate.mask(frame)[0]


def test_a_reversed_axis_range_is_ordered_rather_than_empty():
    gate = _cyl(axis_low=9.0, axis_high=2.0)
    assert (gate.axis_low, gate.axis_high) == (2.0, 9.0)


# ---------------------------------------------------------------------------
# The plane is named by its columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_the_three_measurements_must_be_three(maker):
    with pytest.raises(GateError, match="same measurement twice"):
        maker(axis_column="a")


@pytest.mark.parametrize("maker,missing", [(_cyl, "u_column"),
                                           (_prism, "axis_column")])
def test_a_missing_measurement_is_named(maker, missing):
    with pytest.raises(GateError, match=missing):
        maker(**{missing: "  "})


@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_the_columns_are_the_plane_then_the_normal(maker):
    assert maker().columns == ("a", "b", "z")


def test_a_prism_needs_three_vertices():
    with pytest.raises(GateError, match="at least three"):
        _prism(vertices=((0.0, 0.0), (1.0, 1.0)))


# ---------------------------------------------------------------------------
# Only the normal is a range clause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_an_unbounded_gate_offers_no_range_clause(maker):
    assert maker().range_filters() == ()


@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_only_the_normal_becomes_a_range_clause(maker):
    """The oval and the polygon are not conjunctions of ranges, so offering
    a bounding box for them would quietly include the corners."""
    clauses = maker(axis_low=1.0, axis_high=4.0).range_filters()
    assert len(clauses) == 1
    assert clauses[0].column == "z"
    assert (clauses[0].low, clauses[0].high) == (1.0, 4.0)


def test_the_range_clause_selects_the_same_rows_the_bound_does(frame):
    gate = _cyl(axis_low=0.0, axis_high=2.0)
    clause = gate.range_filters()[0]
    by_clause = ((frame["z"] >= clause.low) & (frame["z"] <= clause.high))
    assert np.array_equal(gate.mask(frame), gate.to_ellipse().mask(frame)
                          & by_clause.to_numpy())


# ---------------------------------------------------------------------------
# Extruding, and looking back down the axis
# ---------------------------------------------------------------------------

def test_extruding_a_drawn_oval_keeps_its_geometry():
    oval = EllipseGate(name="drawn", x_column="a", y_column="b",
                       x_centre=2.0, y_centre=3.0, x_radius=4.0, y_radius=5.0)
    cylinder = CylinderGate.from_ellipse(oval, "z", axis_low=1.0)
    assert cylinder.to_ellipse() == oval
    assert cylinder.axis_column == "z" and cylinder.axis_low == 1.0


def test_extruding_a_drawn_polygon_keeps_its_vertices():
    polygon = PolygonGate(name="drawn", x_column="a", y_column="b",
                          vertices=((0.0, 0.0), (2.0, 0.0), (1.0, 2.0)))
    prism = PrismGate.from_polygon(polygon, "z")
    assert prism.to_polygon() == polygon


def test_looking_down_the_axis_leaves_the_extent_alone_rather_than_resetting():
    """The 2D view cannot express the depth; it must not silently drop it."""
    cylinder = _cyl(axis_low=1.0, axis_high=2.0)
    assert cylinder.to_ellipse().mask(pd.DataFrame({"a": [0.0], "b": [0.0]}))[0]
    # The cylinder itself is unchanged by having been viewed.
    assert (cylinder.axis_low, cylinder.axis_high) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Editing after the fact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_translating_moves_the_plane_shape_and_not_the_normal(maker):
    moved = maker(axis_low=1.0, axis_high=2.0).translated(10.0, -10.0)
    assert (moved.axis_low, moved.axis_high) == (1.0, 2.0)
    assert moved.centre() != maker().centre()


@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_scaling_leaves_the_normal_alone_too(maker):
    scaled = maker(axis_low=1.0, axis_high=2.0).scaled(2.0)
    assert (scaled.axis_low, scaled.axis_high) == (1.0, 2.0)


def test_scaling_a_cylinder_grows_its_radii(frame):
    assert _cyl().scaled(3.0).mask(frame)[2]      # a=3 was outside at r=1


def test_a_zero_radius_is_an_empty_gate_not_a_division_by_zero(frame):
    assert not _cyl(u_radius=0.0).mask(frame).any()


def test_a_negative_radius_is_read_as_its_size():
    assert _cyl(u_radius=-2.0).u_radius == 2.0


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_a_gate_survives_being_saved_and_read_back(maker):
    gate = maker(axis_low=1.0, axis_high=7.0)
    assert gate_from_dict(gate.to_dict()) == gate


@pytest.mark.parametrize("kind", ["cylinder", "prism"])
def test_the_new_kinds_are_offered(kind):
    assert kind in GATE_KINDS


def test_a_prisms_vertices_survive_json(tmp_path):
    import json

    gate = _prism()
    restored = gate_from_dict(json.loads(json.dumps(gate.to_dict())))
    assert restored == gate


@pytest.mark.parametrize("maker", [_cyl, _prism])
def test_describe_says_the_plane_and_the_extent(maker):
    text = maker(axis_low=1.0, axis_high=2.0).describe()
    assert "a/b" in text and "z" in text
    assert "any z" in maker().describe()


def test_missing_values_never_land_inside_a_gate():
    frame = pd.DataFrame({"a": [0.0, np.nan, 0.0],
                          "b": [0.0, 0.0, np.nan],
                          "z": [1.0, 1.0, 1.0]})
    assert _cyl().mask(frame).tolist() == [True, False, False]
