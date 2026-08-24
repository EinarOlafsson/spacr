"""Distances within and between objects, on geometry with known answers.

Before this, measure knew two kinds of distance: centroid-to-centroid
between objects of the SAME type, and a channel's intensity centre of mass
to the nearest nucleus or pathogen surface. Nothing between object types,
nothing about perimeters, nothing about local maxima.

The tests use squares at known separations, because the point of a distance
is the number and a synthetic field is the only place the number is known
in advance. Three distances are checked apart, because they are three
different questions that all get called "distance":

    centre to centre     large for two big objects that are touching
    centre to surface    asymmetric -- a's centre to b's edge is not b's
                         centre to a's edge
    surface to surface   ZERO when they touch; the number a biologist
                         means by "how far apart are they"
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.object_distances import (interior_distance_transform,
                                    object_distances,
                                    surface_distance_transform)


def _square(shape, top, left, size, label=1, into=None):
    mask = np.zeros(shape, dtype=np.uint16) if into is None else into
    mask[top:top + size, left:left + size] = label
    return mask


# ---------------------------------------------------------------------------
# the transforms the rest is built on
# ---------------------------------------------------------------------------

def test_the_surface_transform_is_zero_on_an_object():
    """It measures distance TO a surface, so inside is zero."""
    mask = _square((40, 40), 10, 10, 10)

    field = surface_distance_transform(mask)

    assert field[15, 15] == 0.0
    assert field[0, 15] == pytest.approx(10.0, abs=0.5)


def test_the_surface_transform_of_an_empty_mask_is_infinite():
    """No object of that type anywhere is infinitely far, not zero."""
    field = surface_distance_transform(np.zeros((10, 10), dtype=np.uint16))

    assert np.isinf(field).all()


def test_the_interior_transform_peaks_at_the_centre():
    mask = _square((40, 40), 10, 10, 10)

    field = interior_distance_transform(mask)

    assert field[0, 0] == 0.0
    assert field[14, 14] == pytest.approx(5.0, abs=0.5)


# ---------------------------------------------------------------------------
# between two object types
# ---------------------------------------------------------------------------

@pytest.fixture
def two_types():
    """A cell with a nucleus inside it, and a second, empty cell."""
    cell = _square((60, 60), 5, 5, 20, label=1)
    _square((60, 60), 35, 35, 20, label=2, into=cell)
    nucleus = _square((60, 60), 10, 10, 8, label=1)
    return {"cell": cell, "nucleus": nucleus}


def test_the_gap_between_two_objects_is_their_surface_distance(two_types):
    """Cell 1 spans 5-25, its nucleus 10-18, so the gap is 5 pixels."""
    out = object_distances(two_types, primary="cell")

    gap = dict(zip(out["label"], out["surface_to_nucleus_surface"]))
    assert gap[1] == pytest.approx(5.0, abs=1.0)
    # Cell 2 is far from the only nucleus.
    assert gap[2] > 20


def test_touching_objects_have_a_surface_distance_of_zero():
    """The whole point of measuring surface to surface."""
    a = _square((40, 40), 5, 5, 10, label=1)
    b = _square((40, 40), 5, 15, 10, label=1)   # starts where a ends

    out = object_distances({"a": a, "b": b}, primary="a")

    assert out["surface_to_b_surface"].iloc[0] == pytest.approx(0.0, abs=1.0)
    # And their centres are NOT zero apart, which is the distinction.
    assert out["centre_to_nearest_b_centre"].iloc[0] > 5


def test_an_object_inside_another_overlaps_completely():
    inner = _square((40, 40), 12, 12, 6, label=1)
    outer = _square((40, 40), 5, 5, 20, label=1)

    out = object_distances({"inner": inner, "outer": outer}, primary="inner")

    assert out["outer_overlap_fraction"].iloc[0] == pytest.approx(1.0)


def test_no_partner_is_infinitely_far_not_missing(two_types):
    """`inf` is a fact; NaN would mean "not measured"."""
    lonely = {"cell": two_types["cell"],
              "nucleus": np.zeros_like(two_types["nucleus"])}

    out = object_distances(lonely, primary="cell")

    assert np.isinf(out["surface_to_nucleus_surface"]).all()
    assert not out["surface_to_nucleus_surface"].isna().any()


def test_the_centre_knows_how_far_it_is_from_its_own_rim(two_types):
    """A 20-pixel square's centre is 10 from its edge."""
    out = object_distances(two_types, primary="cell")

    assert out["distance_to_own_boundary"].iloc[0] == pytest.approx(10.0,
                                                                    abs=1.0)


def test_an_object_near_the_edge_says_so():
    """Which is what tells you a measurement is of a fragment."""
    near = _square((60, 60), 0, 0, 10, label=1)
    far = _square((60, 60), 25, 25, 10, label=2, into=near)

    out = object_distances({"cell": far}, primary="cell")
    edge = dict(zip(out["label"], out["distance_to_field_edge"]))

    assert edge[1] < edge[2]


def test_the_columns_carry_no_object_type_prefix(two_types):
    """`measure` adds it; adding it here produced `cell_cell_...`."""
    out = object_distances(two_types, primary="cell")

    assert not [c for c in out.columns if c.startswith("cell_")]
    assert "surface_to_nucleus_surface" in out.columns


# ---------------------------------------------------------------------------
# local maxima
# ---------------------------------------------------------------------------

@pytest.fixture
def two_bright_spots():
    cell = _square((60, 60), 5, 5, 40, label=1)
    image = np.zeros((60, 60, 1), dtype=np.uint16)
    image[12:14, 12:14, 0] = 900
    image[36:38, 36:38, 0] = 900
    return {"cell": cell}, image


def test_the_maxima_are_counted(two_bright_spots):
    masks, image = two_bright_spots

    out = object_distances(masks, image, primary="cell")

    assert out["channel_0_maxima_count"].iloc[0] == 2


def test_spread_tells_clustered_from_scattered(two_bright_spots):
    """Two spots at opposite ends and two on top of each other both count 2."""
    masks, spread_out = two_bright_spots
    together = np.zeros((60, 60, 1), dtype=np.uint16)
    together[12:14, 12:14, 0] = 900
    together[18:20, 18:20, 0] = 900

    far = object_distances(masks, spread_out, primary="cell")
    near = object_distances(masks, together, primary="cell")

    assert (far["channel_0_maxima_spread"].iloc[0]
            > near["channel_0_maxima_spread"].iloc[0])


def test_an_object_with_no_signal_has_no_maxima(two_bright_spots):
    masks, _image = two_bright_spots
    dark = np.zeros((60, 60, 1), dtype=np.uint16)

    out = object_distances(masks, dark, primary="cell")

    assert out["channel_0_maxima_count"].iloc[0] == 0
    # NaN, not zero: zero would read as "the peak is right here".
    assert np.isnan(out["channel_0_maxima_to_centre_min"].iloc[0])


def test_a_polarised_channel_offsets_its_intensity_centre():
    """Uniform staining gives ~0; all the signal at one end does not."""
    cell = _square((60, 60), 10, 10, 30, label=1)
    uniform = np.zeros((60, 60, 1), dtype=np.uint16)
    uniform[10:40, 10:40, 0] = 500
    polarised = np.zeros((60, 60, 1), dtype=np.uint16)
    polarised[10:18, 10:40, 0] = 500

    flat = object_distances({"cell": cell}, uniform, primary="cell")
    skew = object_distances({"cell": cell}, polarised, primary="cell")

    assert flat["channel_0_intensity_centre_offset"].iloc[0] < 1.0
    assert skew["channel_0_intensity_centre_offset"].iloc[0] > 5.0


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------

def test_it_is_off_by_default():
    """It is real time on a 3-D field, so nobody pays for it unasked."""
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    assert settings["object_distances"] is False
