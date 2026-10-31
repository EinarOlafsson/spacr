"""Degenerate inputs to the OPS merge primitives give honest empty answers."""
from __future__ import annotations

import numpy as np

from spacr import ops_merge


def test_coincident_points_define_no_similarity():
    points = np.full((4, 2), 3.5)
    assert ops_merge._similarity_from_pairs(points, points + 10.0) is None


def test_a_real_similarity_is_recovered_from_matched_points():
    rng = np.random.default_rng(1)
    src = rng.random((12, 2)) * 100
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = 2.0 * (rotation @ src.T).T + np.array([5.0, -7.0])
    scale, found_rotation, translation = ops_merge._similarity_from_pairs(src, dst)
    assert np.isclose(scale, 2.0)
    assert np.allclose(found_rotation, rotation)
    assert np.allclose(translation, [5.0, -7.0])


def test_matching_with_an_empty_side_returns_no_pairs():
    some = np.array([[0.0, 0.0], [5.0, 5.0]])
    for source, target in ((np.empty((0, 2)), some), (some, np.empty((0, 2)))):
        pairs = ops_merge.match_cells(source, target, gpu=False)
        assert pairs.shape == (0, 2)
        assert pairs.dtype == np.int64


def test_mutual_neighbours_beyond_the_threshold_are_not_paired():
    source = np.array([[0.0, 0.0], [100.0, 100.0]])
    target = np.array([[1.0, 0.0], [130.0, 100.0]])
    pairs = ops_merge.match_cells(source, target, threshold=10.0, gpu=False)
    assert pairs.tolist() == [[0, 0]]
    wide = ops_merge.match_cells(source, target, threshold=50.0, gpu=False)
    assert wide.tolist() == [[0, 0], [1, 1]]


def test_the_transform_is_applied_before_pairing():
    source = np.array([[0.0, 0.0], [10.0, 0.0]])
    target = source * 2.0
    transform = (2.0, np.eye(2), np.zeros(2))
    assert ops_merge.match_cells(source, target, transform=transform, threshold=0.5,
                                 gpu=False).tolist() == [[0, 0], [1, 1]]
    assert ops_merge.match_cells(source, target, threshold=0.5, gpu=False).tolist() == [[0, 0]]


def test_fewer_than_three_points_give_no_triangles_and_no_alignment():
    two = np.array([[0.0, 0.0], [1.0, 1.0]])
    descriptors, vertices = ops_merge.triangle_descriptors(two)
    assert descriptors.shape == (0, 2) and vertices.shape == (0, 3)
    assert ops_merge.align_by_triangles(two, two) is None
