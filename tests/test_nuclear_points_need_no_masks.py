"""Phase A4's nuclear points, and two inputs its alignment refuses.

`spacr.ops_phenotype.nuclear_points` is how A4 gets points that mean the same
thing in two acquisitions without segmenting either, and no test called it.
The refusals are the degenerate similarity and an anchor centre that is not a
number, both named in their docstrings.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_layout import round_well_layout
from spacr.ops_phenotype import (_fit_raster, nuclear_points,
                                 similarity_from_correspondences)


def _stain(size=200, sigma=4.0):
    """A nuclear stain: 25 blobs on a 35 px grid, each brighter than the last.

    :returns: ``(image, centres)``, centres in the order of brightness.
    """
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size), np.float32)
    centres = [(30 + 35 * i, 30 + 35 * j) for i in range(5) for j in range(5)]
    for rank, (cy, cx) in enumerate(centres):
        image += (100.0 + 10.0 * rank) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    return image, centres


def _nearest(points, centres):
    """The distance from each centre to its nearest point."""
    points = np.asarray(points, float)
    return [float(np.hypot(*(points - np.asarray(c, float)).T).min())
            for c in centres]


def test_every_nucleus_answers_once_at_its_centre():
    image, centres = _stain()

    points = nuclear_points(image, sigma=2.0, min_distance=8, percentile=50.0)

    assert points.dtype == float
    assert points.shape == (25, 2)
    assert max(_nearest(points, centres)) <= 1.0


def test_top_keeps_the_brightest_nuclei():
    """The five brightest blobs are the bottom row of the grid."""
    image, centres = _stain()

    points = nuclear_points(image, sigma=2.0, min_distance=8, top=5,
                            percentile=50.0)

    assert points.shape == (5, 2)
    assert max(_nearest(points, centres[-5:])) <= 1.0


def test_a_target_collapsed_to_one_point_is_no_similarity():
    """Every source point onto one target point is scale zero, not a fit."""
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.full((3, 2), 5.0)

    assert similarity_from_correspondences(source, target) is None


def test_an_anchor_centre_that_is_not_a_number_is_refused():
    anchors = {0: (1.0, 2.0), 40: ("north", "east"), 100: (3.0, 4.0)}

    with pytest.raises(ValueError, match=r"finite \(y, x\) pair"):
        _fit_raster(round_well_layout(333), anchors)
