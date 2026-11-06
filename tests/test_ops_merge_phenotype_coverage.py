"""The OPS merge helpers on the inputs that leave nothing to solve or pair."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.ops_merge import (deduplicate_matches, field_transform,
                             merge_field, refine_field_transform)

FIELD = (400, 400)


def _seed():
    return field_transform(0.25, 0.0, (1000.0, 1000.0), FIELD)


def test_refinement_keeps_the_seed_when_too_few_nuclei_are_given():
    seed = _seed()
    found, inliers, residual, refined = refine_field_transform(
        [[0.0, 0.0], [10.0, 10.0]], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        seed)
    assert refined is False and inliers == 0 and np.isnan(residual)
    assert found[0] == 0.25
    assert np.allclose(found[1], seed[1]) and np.allclose(found[2], seed[2])
    _found, inliers, _residual, refined = refine_field_transform(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0]], seed)
    assert refined is False and inliers == 0


def test_refinement_keeps_the_seed_when_no_round_can_be_solved():
    seed = _seed()
    points = [[10.0, 10.0], [200.0, 50.0], [300.0, 350.0]]
    far = [[90000.0, 90000.0], [95000.0, 91000.0], [99000.0, 99000.0]]
    found, inliers, residual, refined = refine_field_transform(
        points, far, seed, radius=3.0)
    assert refined is False and inliers == 0 and np.isnan(residual)
    assert np.allclose(found[2], seed[2])


def test_a_field_with_no_nucleus_in_reach_pairs_nothing():
    transform = _seed()
    pairs = merge_field([[10.0, 10.0], [20.0, 20.0]],
                        [[50000.0, 50000.0]], transform, threshold=4.0)
    assert pairs.shape == (0, 3)
    assert pairs.dtype == float


def test_an_empty_or_missing_merge_frame_is_returned_as_is():
    empty = pd.DataFrame({"object_id": [], "distance": []})
    kept, dropped = deduplicate_matches(empty)
    assert kept is empty and dropped == 0
    kept, dropped = deduplicate_matches(None)
    assert kept is None and dropped == 0


def _planted_field(n=200):
    """Field nuclei and their sequencing partners under a known transform
    rotated 0.35 degrees from the seed's."""
    rng = np.random.default_rng(5)
    points = rng.uniform(20, FIELD[0] - 20, size=(n, 2))
    truth = field_transform(0.25, 0.35, (1000.0, 1000.0), FIELD)
    targets = (truth[0] * (truth[1] @ points.T).T + truth[2]
               + rng.normal(0, 0.1, size=(n, 2)))
    return points, targets


def test_refinement_keeps_the_seed_when_too_few_nuclei_agree():
    points, targets = _planted_field()
    seed = _seed()
    found, inliers, residual, refined = refine_field_transform(
        points, targets, seed, min_inliers=10_000)
    assert refined is False and np.isnan(residual)
    assert inliers > 150
    assert np.allclose(found[2], seed[2]) and np.allclose(found[1], seed[1])


def test_refinement_keeps_the_seed_when_the_rotation_moves_too_far():
    points, targets = _planted_field()
    seed = _seed()
    _found, inliers, _residual, refined = refine_field_transform(
        points, targets, seed, max_degrees_change=0.1)
    assert refined is False and inliers > 150
    found, _inliers, residual, refined = refine_field_transform(
        points, targets, seed, max_degrees_change=1.0)
    assert refined is True and residual < 1.0
    degrees = np.degrees(np.arctan2(found[1][1, 0], found[1][0, 0]))
    assert abs(degrees - 0.35) < 0.05
