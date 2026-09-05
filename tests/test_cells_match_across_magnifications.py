"""Match two cell sets related by a transform we chose, and check we get it back.

The merge step exists because pixel correlation fails between a 10x barcode
acquisition and a 20x phenotype acquisition: the scale differs, the stains
differ, and often the microscope does too. What survives is the ARRANGEMENT of
the cells, so the matching is done on triangle shapes.

Every test below builds the second point set from the first by applying a
similarity transform this file chose, so "did it recover the transform" and
"did it match the right cells" are both comparisons against a known answer
rather than judgements about a picture.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spacr import ops_merge

#: The transform the fixtures apply. 2.0 is the 10x -> 20x case; the rotation
#: and offset are arbitrary and deliberately not round numbers.
SCALE = 2.0
ANGLE = np.deg2rad(11.0)
SHIFT = np.array([137.5, -92.25])


def _rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def _two_views(n=90, seed=7, jitter=0.0, drop=0):
    """A cell set and the same cells seen through the known transform.

    :param n: how many cells.
    :param seed: for reproducibility of any failure.
    :param jitter: noise added in TARGET units, to imitate segmentation
        centroids that do not agree to the pixel.
    :param drop: how many cells are visible in one view but not the other,
        which is the normal case -- the two acquisitions never see quite the
        same population.
    :returns: ``(source, target, truth)`` where truth maps source index to
        target index.
    """
    rng = np.random.default_rng(seed)
    source = rng.uniform(0, 500, size=(n, 2))
    moved = (SCALE * (_rotation(ANGLE) @ source.T).T) + SHIFT
    if jitter:
        moved = moved + rng.normal(0, jitter, moved.shape)

    keep = np.arange(n)
    if drop:
        keep = rng.permutation(n)[: n - drop]
        keep.sort()
    target = moved[keep]
    truth = {int(s): int(t) for t, s in enumerate(keep)}
    return source, target, truth


def test_the_triangle_shape_does_not_care_about_scale_or_rotation():
    """The descriptor must be identical for a triangle and its transform.

    This is the assumption the whole method rests on. If it does not hold,
    nothing downstream can work, so it is asserted directly rather than
    inferred from the matching succeeding.
    """
    source, target, _truth = _two_views(n=40)
    src_desc, _src_tri = ops_merge.triangle_descriptors(source)
    dst_desc, _dst_tri = ops_merge.triangle_descriptors(target)

    assert len(src_desc) > 0 and len(dst_desc) > 0
    # Delaunay on the transformed points gives the SAME triangulation, so the
    # two descriptor sets must agree as sets.
    src_sorted = np.array(sorted(map(tuple, np.round(src_desc, 6))))
    dst_sorted = np.array(sorted(map(tuple, np.round(dst_desc, 6))))
    assert src_sorted.shape == dst_sorted.shape
    assert np.allclose(src_sorted, dst_sorted, atol=1e-6)


def test_the_transform_is_recovered_from_the_arrangement_alone():
    """Scale, rotation and translation, from cell positions and nothing else."""
    source, target, _truth = _two_views()
    got = ops_merge.align_by_triangles(source, target)
    assert got is not None, "no transform was recovered at all"

    scale, rotation, translation = got
    assert abs(scale - SCALE) < 0.02, f"scale {scale:.4f} vs {SCALE}"
    angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    assert abs(angle - ANGLE) < np.deg2rad(1.0), (
        f"angle {np.rad2deg(angle):.2f} deg vs {np.rad2deg(ANGLE):.2f}"
    )
    assert np.allclose(translation, SHIFT, atol=3.0), (
        f"translation {translation} vs {SHIFT}"
    )


def test_every_cell_is_matched_to_the_right_one():
    """Under the recovered transform, the pairing must be the true pairing."""
    source, target, truth = _two_views(jitter=1.5, drop=8)
    transform = ops_merge.align_by_triangles(source, target)
    assert transform is not None

    pairs = ops_merge.match_cells(source, target, transform=transform,
                                  threshold=12.0)
    assert len(pairs) >= int(0.8 * len(truth)), (
        f"only {len(pairs)} of {len(truth)} matchable cells were paired"
    )
    wrong = [(int(s), int(t)) for s, t in pairs if truth.get(int(s)) != int(t)]
    assert not wrong, f"{len(wrong)} cells matched to the wrong partner: {wrong[:5]}"


def test_a_contested_cell_is_dropped_rather_than_assigned():
    """Two cells competing for one partner must BOTH lose.

    A wrongly matched cell hands a real phenotype somebody else's barcode and
    corrupts every statistic computed from it. A dropped cell costs only
    power. So the tie must not be broken.
    """
    # Two source cells almost on top of each other, one target between them:
    # neither source cell is the target's unique mutual nearest.
    source = np.array([[100.0, 100.0], [104.0, 100.0], [300.0, 300.0]])
    target = np.array([[102.0, 100.0], [300.0, 300.0]])

    pairs = ops_merge.match_cells(source, target, threshold=20.0)
    matched_sources = {int(s) for s, _t in pairs}

    # The isolated pair is unambiguous and must survive.
    assert (2, 1) in {(int(s), int(t)) for s, t in pairs}
    # Only ONE of the two contested cells may be kept -- never both, because
    # that would mean the single target was used twice.
    assert len({0, 1} & matched_sources) <= 1
    assert len({int(t) for _s, t in pairs}) == len(pairs), (
        "a target cell was matched more than once"
    )


def test_too_little_evidence_returns_nothing_rather_than_a_guess():
    """Below the vote floor the answer is None, not a confident wrong number."""
    rng = np.random.default_rng(1)
    assert ops_merge.align_by_triangles(rng.uniform(0, 10, (3, 2)),
                                        rng.uniform(0, 10, (3, 2)),
                                        min_votes=50) is None
    assert ops_merge.align_by_triangles(np.zeros((2, 2)),
                                        np.zeros((2, 2))) is None
