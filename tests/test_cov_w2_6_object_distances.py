"""Distances measured on masks whose right answers are known by construction.

Every case here is built from small hand-placed squares, so the expected
number can be written down rather than recorded: two squares four pixels
apart have a surface-to-surface distance of four, a square's centre sits
half its width from its own rim, and an object with no partner of the other
type is infinitely far from one rather than NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import object_distances as od


def _one_square(shape=(40, 40), value=1, row=10, col=10, size=6):
    mask = np.zeros(shape, dtype=np.int32)
    mask[row:row + size, col:col + size] = value
    return mask


# --------------------------------------------------------------------------
# the two transforms
# --------------------------------------------------------------------------

def test_a_voxel_size_scales_the_surface_transform():
    """The transform in physical units is the pixel one times the step."""
    mask = _one_square()
    plain = od.surface_distance_transform(mask)
    scaled = od.surface_distance_transform(mask, spacing=2.0)
    assert np.allclose(scaled, plain * 2.0, atol=1e-4)


def test_an_empty_mask_has_no_interior_depth_anywhere():
    """Nothing is inside an object, so the interior transform is zero --
    not inf, which is what the *surface* transform reports for the same
    field and means the opposite thing."""
    empty = np.zeros((12, 12), dtype=np.int32)
    interior = od.interior_distance_transform(empty)
    assert interior.shape == (12, 12)
    assert np.all(interior == 0)
    assert np.all(np.isinf(od.surface_distance_transform(empty)))


def test_a_voxel_size_scales_the_interior_transform():
    mask = _one_square()
    plain = od.interior_distance_transform(mask)
    scaled = od.interior_distance_transform(mask, spacing=3.0)
    assert np.allclose(scaled, plain * 3.0, atol=1e-4)


# --------------------------------------------------------------------------
# sampling and boundaries
# --------------------------------------------------------------------------

def test_a_field_with_no_objects_has_no_boundary_pixels():
    assert od._boundary_pixels(np.zeros((10, 10), dtype=np.int32)) == {}


def test_the_boundary_of_a_square_is_its_rim_and_only_its_rim():
    mask = _one_square()
    edges = od._boundary_pixels(mask)
    assert set(edges) == {1}
    rows, cols = edges[1]
    # A 6x6 square has 20 rim pixels and 16 interior ones.
    assert rows.size == 20
    assert not ((rows > 10) & (rows < 15) & (cols > 10) & (cols < 15)).any()


def test_an_object_with_no_boundary_is_infinitely_far_from_anything():
    """`inf`, deliberately: there is no surface to measure from, and 0 would
    read as 'touching'."""
    assert od._min_over_boundary(np.zeros((4, 4)), None) == float("inf")
    empty = (np.empty(0, dtype=int), np.empty(0, dtype=int))
    assert od._min_over_boundary(np.zeros((4, 4)), empty) == float("inf")


# --------------------------------------------------------------------------
# local maxima
# --------------------------------------------------------------------------

def test_two_peaks_have_a_spread_and_one_has_none():
    """Spread is what tells two bright spots at opposite ends of a cell from
    two sitting on top of each other -- the count cannot."""
    assert od._pairwise_spread(np.empty((0, 2))) == 0.0
    assert od._pairwise_spread(np.array([[3.0, 4.0]])) == 0.0
    spread = od._pairwise_spread(np.array([[0.0, 0.0], [0.0, 10.0]]))
    assert spread == pytest.approx(10.0)


def test_a_voxel_size_scales_the_spread_it_reports():
    points = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert od._pairwise_spread(points, spacing=(1.0, 0.5)) == pytest.approx(5.0)
