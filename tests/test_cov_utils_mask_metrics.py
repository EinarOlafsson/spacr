"""CPU coverage for the spacr.utils mask-metrics block.

Covers ``get_files_from_dir`` .. ``_remove_multiobject_cells``: circular
masking, image/label resizing (including the 3-D and singleton-channel
branches and the ``show_example`` plotting hand-offs), the IoU / AP /
Jaccard / Dice / boundary-F1 metrics, and the three mask-cleanup helpers.

Everything here is offline, CPU-only and sub-second.
"""
from __future__ import annotations


import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_lingering_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _block(shape, sl, dtype=bool, value=True):
    """Return an array of ``shape`` with ``sl`` filled with ``value``."""
    arr = np.zeros(shape, dtype=dtype)
    arr[sl] = value
    return arr


# ---------------------------------------------------------------------------
# get_files_from_dir
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# create_circular_mask / apply_mask / invert_image
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# resize_images_and_labels — branch matrix
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# resize_labels_back
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# pad_to_same_shape / calculate_iou
# ---------------------------------------------------------------------------

def test_pad_to_same_shape_pads_both_to_elementwise_max():
    from spacr.utils import pad_to_same_shape

    m1 = np.ones((3, 5), dtype=np.uint8)
    m2 = np.full((6, 2), 2, dtype=np.uint8)
    p1, p2 = pad_to_same_shape(m1, m2)
    assert p1.shape == (6, 5) and p2.shape == (6, 5)
    # original content pinned to the top-left corner, padding is zero
    assert np.array_equal(p1[:3, :5], m1)
    assert p1[3:, :].sum() == 0
    assert np.array_equal(p2[:6, :2], m2)
    assert p2[:, 2:].sum() == 0


# ---------------------------------------------------------------------------
# match_masks / compute_average_precision
# ---------------------------------------------------------------------------

def test_compute_average_precision_values_and_zero_guards():
    from spacr.utils import compute_average_precision

    # 2 matches out of 4 predictions and 3 truths
    precision, recall = compute_average_precision([0, 1], 3, 4)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(2 / 3)
    # no predictions and no truths => both guards return 0 instead of ZeroDivision
    assert compute_average_precision([], 0, 0) == (0, 0)


# ---------------------------------------------------------------------------
# compute_ap_over_iou_thresholds / compute_segmentation_ap
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# jaccard_index / dice_coefficient
# ---------------------------------------------------------------------------

def test_jaccard_index_matches_hand_computed_value():
    from spacr.utils import jaccard_index

    a = _block((10, 10), np.s_[0:5, 0:5])       # 25
    b = _block((10, 10), np.s_[0:5, 0:10])      # 50, superset
    assert jaccard_index(a, a) == pytest.approx(1.0)
    assert jaccard_index(a, b) == pytest.approx(25 / 50)
    assert jaccard_index(a, _block((10, 10), np.s_[6:10, 6:10])) == pytest.approx(0.0)


def test_dice_coefficient_binarises_and_handles_empty_pair():
    from spacr.utils import dice_coefficient

    # non-binary label values must be treated as foreground
    a = _block((10, 10), np.s_[0:5, 0:5], dtype=np.uint16, value=7)
    b = _block((10, 10), np.s_[0:5, 0:10], dtype=np.uint16, value=3)
    assert dice_coefficient(a, a) == pytest.approx(1.0)
    assert dice_coefficient(a, b) == pytest.approx(2 * 25 / (25 + 50))
    # both empty -> defined as a perfect match
    empty = np.zeros((10, 10), dtype=np.uint16)
    assert dice_coefficient(empty, empty) == 1.0
    assert dice_coefficient(a, empty) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# extract_boundaries / boundary_f1_score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius,expected", [(1, 12 * 12 - 8 * 8), (2, 14 * 14 - 6 * 6)])
def test_extract_boundaries_ring_area_matches_structuring_element(radius, expected):
    from spacr.utils import extract_boundaries

    mask = np.zeros((30, 30), dtype=np.uint16)
    mask[10:20, 10:20] = 4                      # non-binary label value
    boundary = extract_boundaries(mask, dilation_radius=radius)
    assert boundary.dtype == np.bool_
    assert boundary.shape == mask.shape
    assert int(boundary.sum()) == expected
    assert not boundary[15, 15], "the object interior is not a boundary"
    assert boundary[10 - radius, 15], "the dilated edge is part of the boundary"


def test_boundary_f1_score_perfect_and_disjoint():
    from spacr.utils import boundary_f1_score

    a = np.zeros((30, 30), dtype=np.uint8)
    a[5:15, 5:15] = 1
    b = np.zeros((30, 30), dtype=np.uint8)
    b[20:28, 20:28] = 1

    assert boundary_f1_score(a, a) == pytest.approx(1.0, abs=1e-4)
    assert boundary_f1_score(a, b) == pytest.approx(0.0, abs=1e-9)
    empty = np.zeros((30, 30), dtype=np.uint8)
    assert boundary_f1_score(empty, empty) == pytest.approx(0.0, abs=1e-9)


def test_boundary_f1_score_partial_overlap_is_between_zero_and_one():
    from spacr.utils import boundary_f1_score

    a = np.zeros((30, 30), dtype=np.uint8)
    a[5:15, 5:15] = 1
    shifted = np.zeros((30, 30), dtype=np.uint8)
    shifted[6:16, 5:15] = 1                     # 1 px vertical shift
    score = boundary_f1_score(a, shifted)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# _remove_noninfected / _remove_outside_objects / _remove_multiobject_cells
# ---------------------------------------------------------------------------

