"""CPU coverage for the spacr.utils mask-metrics block.

Covers ``get_files_from_dir`` .. ``_remove_multiobject_cells``: circular
masking, image/label resizing (including the 3-D and singleton-channel
branches and the ``show_example`` plotting hand-offs), the IoU / AP /
Jaccard / Dice / boundary-F1 metrics, and the three mask-cleanup helpers.

Everything here is offline, CPU-only and sub-second.
"""
from __future__ import annotations

import os

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

def test_get_files_from_dir_filters_by_extension(tmp_path):
    from spacr.utils import get_files_from_dir

    for name in ("a.tif", "b.tif", "c.csv"):
        (tmp_path / name).write_text("x")

    tifs = get_files_from_dir(str(tmp_path), "*.tif")
    assert sorted(os.path.basename(p) for p in tifs) == ["a.tif", "b.tif"]
    assert all(os.path.isabs(p) for p in tifs)


def test_get_files_from_dir_default_pattern_returns_everything(tmp_path):
    from spacr.utils import get_files_from_dir

    for name in ("a.tif", "c.csv"):
        (tmp_path / name).write_text("x")
    (tmp_path / "sub").mkdir()

    everything = get_files_from_dir(str(tmp_path))
    assert sorted(os.path.basename(p) for p in everything) == ["a.tif", "c.csv", "sub"]


def test_get_files_from_dir_empty_dir_returns_empty_list(tmp_path):
    from spacr.utils import get_files_from_dir

    out = get_files_from_dir(str(tmp_path), "*.tif")
    assert out == []


# ---------------------------------------------------------------------------
# create_circular_mask / apply_mask / invert_image
# ---------------------------------------------------------------------------

def test_create_circular_mask_defaults_to_inscribed_circle():
    from spacr.utils import create_circular_mask

    mask = create_circular_mask(10, 10)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.bool_
    # centre in, corners out, cardinal points exactly on the radius are in.
    assert mask[5, 5]
    assert not mask[0, 0] and not mask[9, 9]
    assert mask[0, 5] and mask[5, 0]
    # area of a radius-5 disc, allow for pixelation.
    assert 60 <= int(mask.sum()) <= 100


def test_create_circular_mask_explicit_center_and_radius():
    from spacr.utils import create_circular_mask

    mask = create_circular_mask(8, 8, center=(2, 3), radius=1)
    # center is (x, y) => column 2, row 3.
    true_coords = {tuple(map(int, rc)) for rc in zip(*np.nonzero(mask))}
    assert true_coords == {(3, 2), (2, 2), (4, 2), (3, 1), (3, 3)}


def test_apply_mask_2d_zeroes_outside_the_disc():
    from spacr.utils import apply_mask

    img = np.full((20, 20), 100, dtype=np.uint16)
    out = apply_mask(img)
    assert out.shape == img.shape
    assert out[10, 10] == 100        # inside the disc: untouched
    assert out[0, 0] == 0            # corner: outside the disc
    assert out[19, 19] == 0
    assert 0 < int((out > 0).sum()) < img.size


def test_apply_mask_custom_output_value_and_multichannel():
    from spacr.utils import apply_mask

    img = np.full((20, 20, 3), 7, dtype=np.uint8)
    out = apply_mask(img, output_value=9)
    assert out.shape == (20, 20, 3)
    # the mask is repeated per channel, so a corner is filled in every channel
    assert np.array_equal(out[0, 0, :], np.array([9, 9, 9], dtype=out.dtype))
    assert np.array_equal(out[10, 10, :], np.array([7, 7, 7], dtype=out.dtype))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_invert_image_uses_dtype_max_as_pivot(dtype):
    from spacr.utils import invert_image

    max_value = np.iinfo(dtype).max
    img = np.array([[0, 1, max_value]], dtype=dtype)
    out = invert_image(img)
    assert out.dtype == dtype
    assert np.array_equal(out, np.array([[max_value, max_value - 1, 0]], dtype=dtype))


# ---------------------------------------------------------------------------
# resize_images_and_labels — branch matrix
# ---------------------------------------------------------------------------

def test_resize_images_and_labels_3d_image_keeps_channels():
    """images+labels branch with a 3-D image (utils.py 4065-4066)."""
    from spacr.utils import resize_images_and_labels

    img = np.random.default_rng(3).integers(0, 255, (32, 32, 3)).astype(np.uint8)
    lbl = np.zeros((32, 32), dtype=np.uint16)
    lbl[4:20, 4:20] = 5

    imgs, lbls = resize_images_and_labels([img], [lbl], 16, 16, show_example=False)
    assert len(imgs) == 1 and len(lbls) == 1
    assert imgs[0].shape == (16, 16, 3)
    assert imgs[0].dtype == np.uint8
    assert lbls[0].shape == (16, 16)
    assert lbls[0].dtype == np.uint16
    # order=0 resizing must not invent new label values
    assert set(np.unique(lbls[0])).issubset({0, 5})


def test_resize_images_and_labels_singleton_channel_is_squeezed():
    """images+labels branch, 3-D image with one channel (utils.py 4065-4066, 4072)."""
    from spacr.utils import resize_images_and_labels

    img = np.random.default_rng(4).integers(0, 255, (32, 32, 1)).astype(np.uint8)
    lbl = np.zeros((32, 32), dtype=np.uint16)
    lbl[8:24, 8:24] = 2

    imgs, lbls = resize_images_and_labels([img], [lbl], 16, 16, show_example=False)
    assert imgs[0].shape == (16, 16), "trailing singleton channel must be squeezed"
    assert imgs[0].dtype == np.uint8
    assert lbls[0].shape == (16, 16)
    assert 2 in np.unique(lbls[0])


def test_resize_images_only_2d(tmp_path):
    """images-only branch with a 2-D image (utils.py 4081)."""
    from spacr.utils import resize_images_and_labels

    img = np.random.default_rng(5).random((32, 32)).astype(np.float32)
    imgs, lbls = resize_images_and_labels([img], None, 16, 8, show_example=False)
    assert lbls == []
    assert len(imgs) == 1
    assert imgs[0].shape == (16, 8)
    assert imgs[0].dtype == np.float32


def test_resize_images_only_singleton_channel_is_squeezed():
    """images-only branch, 3-D image with one channel (utils.py 4088)."""
    from spacr.utils import resize_images_and_labels

    img = np.random.default_rng(6).integers(0, 1000, (32, 32, 1)).astype(np.uint16)
    imgs, lbls = resize_images_and_labels([img], None, 16, 16, show_example=False)
    assert lbls == []
    assert imgs[0].shape == (16, 16), "trailing singleton channel must be squeezed"
    assert imgs[0].dtype == np.uint16


def test_resize_labels_only_uses_nearest_neighbour():
    from spacr.utils import resize_images_and_labels

    lbl = np.zeros((32, 32), dtype=np.int32)
    lbl[0:16, 0:16] = 3
    lbl[16:32, 16:32] = 9
    imgs, lbls = resize_images_and_labels(None, [lbl], 16, 16, show_example=False)
    assert imgs == []
    assert lbls[0].shape == (16, 16)
    # order=0 => only the original label ids survive
    assert set(np.unique(lbls[0])) == {0, 3, 9}


def test_resize_all_none_returns_two_empty_lists():
    from spacr.utils import resize_images_and_labels

    imgs, lbls = resize_images_and_labels(None, None, 16, 16, show_example=False)
    assert imgs == [] and lbls == []


def test_resize_show_example_images_and_labels(monkeypatch):
    """show_example with both inputs calls plot_resize(imgs, ri, lbls, rl) (4099)."""
    import spacr.plot
    from spacr.utils import resize_images_and_labels

    calls = []
    monkeypatch.setattr(spacr.plot, "plot_resize",
                        lambda *a: calls.append(a))

    img = np.random.default_rng(7).random((32, 32)).astype(np.float32)
    lbl = np.zeros((32, 32), dtype=np.uint16)
    lbl[2:10, 2:10] = 1
    imgs, lbls = resize_images_and_labels([img], [lbl], 16, 16, show_example=True)

    assert len(calls) == 1
    originals, resized, orig_labels, resized_labels = calls[0]
    assert originals[0].shape == (32, 32)
    assert resized[0].shape == (16, 16)
    assert orig_labels[0].shape == (32, 32)
    assert resized_labels[0].shape == (16, 16)
    # the returned lists are the very ones handed to the plotter
    assert resized is imgs and resized_labels is lbls


def test_resize_show_example_images_only(monkeypatch):
    """show_example with images only passes the images twice (4100-4101)."""
    import spacr.plot
    from spacr.utils import resize_images_and_labels

    calls = []
    monkeypatch.setattr(spacr.plot, "plot_resize",
                        lambda *a: calls.append(a))

    img = np.random.default_rng(8).random((32, 32)).astype(np.float32)
    imgs, lbls = resize_images_and_labels([img], None, 16, 16, show_example=True)

    assert len(calls) == 1
    a, b, c, d = calls[0]
    assert a is c and b is d
    assert b is imgs
    assert lbls == []


def test_resize_show_example_labels_only(monkeypatch):
    """show_example with labels only passes the labels twice (4102-4103)."""
    import spacr.plot
    from spacr.utils import resize_images_and_labels

    calls = []
    monkeypatch.setattr(spacr.plot, "plot_resize",
                        lambda *a: calls.append(a))

    lbl = np.zeros((32, 32), dtype=np.uint16)
    lbl[2:20, 2:20] = 4
    imgs, lbls = resize_images_and_labels(None, [lbl], 16, 16, show_example=True)

    assert len(calls) == 1
    a, b, c, d = calls[0]
    assert a is c and b is d
    assert b is lbls
    assert imgs == []
    assert lbls[0].shape == (16, 16)


# ---------------------------------------------------------------------------
# resize_labels_back
# ---------------------------------------------------------------------------

def test_resize_labels_back_preserves_label_values_and_dtype():
    from spacr.utils import resize_labels_back

    lbl = np.zeros((16, 16), dtype=np.int32)
    lbl[0:8, 0:8] = 11
    out = resize_labels_back([lbl], [(32, 24)])
    assert len(out) == 1
    assert out[0].shape == (32, 24)
    assert out[0].dtype == np.int32
    assert set(np.unique(out[0])) == {0, 11}


def test_resize_labels_back_rejects_non_tuple_dims():
    from spacr.utils import resize_labels_back

    with pytest.raises(ValueError, match="tuple of two integers"):
        resize_labels_back([np.zeros((4, 4), np.uint8)], [[8, 8]])


def test_resize_labels_back_rejects_length_mismatch_before_resizing():
    from spacr.utils import resize_labels_back

    with pytest.raises(ValueError, match="length of labels and orig_dims must match"):
        resize_labels_back([np.zeros((4, 4), np.uint8)], [(8, 8), (8, 8)])


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


def test_calculate_iou_identical_disjoint_and_empty():
    from spacr.utils import calculate_iou

    a = _block((10, 10), np.s_[0:5, 0:5])
    b = _block((10, 10), np.s_[5:10, 5:10])
    assert calculate_iou(a, a) == pytest.approx(1.0)
    assert calculate_iou(a, b) == pytest.approx(0.0)
    # union == 0 => the guard returns 0 rather than dividing by zero
    empty = np.zeros((10, 10), dtype=bool)
    assert calculate_iou(empty, empty) == 0


def test_calculate_iou_pads_mismatched_shapes():
    from spacr.utils import calculate_iou

    a = np.ones((4, 4), dtype=bool)          # 16 px
    b = np.ones((4, 8), dtype=bool)          # 32 px, superset after padding
    assert calculate_iou(a, b) == pytest.approx(16 / 32)


# ---------------------------------------------------------------------------
# match_masks / compute_average_precision
# ---------------------------------------------------------------------------

def test_match_masks_consumes_each_true_mask_at_most_once():
    from spacr.utils import match_masks

    a = _block((10, 10), np.s_[0:5, 0:5])
    b = _block((10, 10), np.s_[6:10, 6:10])
    # two identical predictions both look like `a`
    matches = match_masks([a, b], [a.copy(), a.copy()], 0.5)
    assert len(matches) == 1, "a true mask must not be matched twice"
    true_matched, pred_matched = matches[0]
    assert np.array_equal(true_matched, a)
    assert np.array_equal(pred_matched, a)


def test_match_masks_threshold_excludes_weak_overlap():
    from spacr.utils import match_masks

    true = _block((10, 10), np.s_[0:5, 0:5])       # 25 px
    pred = _block((10, 10), np.s_[0:5, 0:4])       # 20 px, IoU 0.8
    assert len(match_masks([true], [pred], 0.5)) == 1
    assert match_masks([true], [pred], 0.9) == []


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

def _ap_fixture_masks():
    """true: one big + one small object; pred: an 80%-overlap copy of the big one."""
    true0 = _block((20, 20), np.s_[0:10, 0:10])
    true1 = _block((20, 20), np.s_[15:19, 15:19])
    pred0 = _block((20, 20), np.s_[0:10, 0:8])   # IoU 0.8 with true0
    return [true0, true1], [pred0]


def test_compute_ap_over_iou_thresholds_area_under_pr_curve():
    from spacr.utils import compute_ap_over_iou_thresholds

    true_masks, pred_masks = _ap_fixture_masks()
    # threshold 0.5 -> (P=1.0, R=0.5); threshold 0.9 -> (P=0, R=0)
    # trapz over recalls [0, 0.5] with precisions [0, 1] == 0.25
    ap = compute_ap_over_iou_thresholds(true_masks, pred_masks, [0.5, 0.9])
    assert ap == pytest.approx(0.25)


def test_compute_ap_over_iou_thresholds_no_predictions_is_zero():
    from spacr.utils import compute_ap_over_iou_thresholds

    true_masks, _ = _ap_fixture_masks()
    ap = compute_ap_over_iou_thresholds(true_masks, [], [0.5, 0.9])
    assert ap == pytest.approx(0.0)


def test_compute_ap_over_iou_thresholds_rejects_out_of_range_precision():
    """The defensive bounds check (utils.py 4190) fires on an inconsistent input.

    ``compute_average_precision`` derives FP as ``num_pred_masks - TP``; feeding a
    sequence whose ``__len__`` under-reports its own contents makes FP negative and
    precision > 1, which must be rejected rather than silently integrated.
    """
    from spacr.utils import compute_ap_over_iou_thresholds

    class LyingLenList(list):
        def __len__(self):  # noqa: D105 - deliberately inconsistent with __iter__
            return 1

    a = _block((10, 10), np.s_[0:5, 0:5])
    b = _block((10, 10), np.s_[6:10, 6:10])
    true_masks = [a, b]
    pred_masks = LyingLenList([a.copy(), b.copy()])   # iterates 2, reports len 1

    with pytest.raises(ValueError) as excinfo:
        compute_ap_over_iou_thresholds(true_masks, pred_masks, [0.5])
    msg = str(excinfo.value)
    assert "out of bounds" in msg
    assert "Precision: 2.0" in msg


def test_compute_segmentation_ap_on_labeled_images():
    from spacr.utils import compute_segmentation_ap

    true_img = np.zeros((20, 20), dtype=np.uint8)
    true_img[0:10, 0:10] = 1
    true_img[15:19, 15:19] = 1          # second connected component
    pred_img = np.zeros((20, 20), dtype=np.uint8)
    pred_img[0:10, 0:8] = 1             # 80% of the first object only

    ap = compute_segmentation_ap(true_img, pred_img, iou_thresholds=[0.5, 0.9])
    assert ap == pytest.approx(0.25)


def test_compute_segmentation_ap_empty_prediction_is_zero():
    from spacr.utils import compute_segmentation_ap

    true_img = np.zeros((20, 20), dtype=np.uint8)
    true_img[2:8, 2:8] = 1
    pred_img = np.zeros((20, 20), dtype=np.uint8)

    ap = compute_segmentation_ap(true_img, pred_img, iou_thresholds=[0.5, 0.9])
    assert ap == pytest.approx(0.0)


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

def _three_channel_stack():
    """(40, 40, 3) stack: ch0 cells, ch1 nuclei, ch2 pathogens.

    cell 1 (with pathogen 1 + nucleus 1) and cell 2 (nucleus 2, no pathogen).
    pathogen 2 sits outside every cell.
    """
    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1        # cell 1
    stack[20:30, 2:12, 0] = 2       # cell 2
    stack[5:8, 5:8, 1] = 1          # nucleus 1 inside cell 1
    stack[23:26, 5:8, 1] = 2        # nucleus 2 inside cell 2
    stack[6:9, 6:9, 2] = 1          # pathogen 1 inside cell 1
    stack[33:37, 33:37, 2] = 2      # pathogen 2 outside every cell
    return stack


def test_remove_noninfected_clears_cells_without_pathogen():
    from spacr.utils import _remove_noninfected

    out = _remove_noninfected(_three_channel_stack(), 0, 1, 2)
    cell, nucleus, pathogen = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    assert set(np.unique(cell)) == {0, 1}, "the pathogen-free cell must be removed"
    assert set(np.unique(nucleus)) == {0, 1}, "its nucleus must go with it"
    # the infected cell keeps its exact footprint
    assert int((cell == 1).sum()) == 100
    # pathogens are untouched by this helper
    assert set(np.unique(pathogen)) == {0, 1, 2}


def test_remove_noninfected_with_cell_dim_none_is_a_noop():
    from spacr.utils import _remove_noninfected

    stack = _three_channel_stack()
    out = _remove_noninfected(stack.copy(), None, 1, 2)
    assert np.array_equal(out, stack)


def test_remove_noninfected_with_nucleus_and_pathogen_dims_none():
    from spacr.utils import _remove_noninfected

    stack = _three_channel_stack()
    out = _remove_noninfected(stack.copy(), 0, None, None)
    # with no pathogen channel every cell counts as non-infected
    assert set(np.unique(out[:, :, 0])) == {0}
    # the nucleus channel was never written back, so it survives untouched
    assert np.array_equal(out[:, :, 1], stack[:, :, 1])


def test_remove_outside_objects_clears_pathogens_with_no_host_cell():
    from spacr.utils import _remove_outside_objects

    out = _remove_outside_objects(_three_channel_stack(), 0, 1, 2)
    pathogen = out[:, :, 2]
    assert set(np.unique(pathogen)) == {0, 1}, "pathogen 2 has no host cell"
    assert int((pathogen == 1).sum()) == 9
    # cells are untouched
    assert set(np.unique(out[:, :, 0])) == {0, 1, 2}
    assert int((out[:, :, 0] == 2).sum()) == 100


def test_remove_outside_objects_must_not_erase_unrelated_nucleus():
    from spacr.utils import _remove_outside_objects

    # pathogen 2 lies outside every cell; nucleus 2 lies inside cell 2, which
    # survives. Removing pathogen 2 must not touch nucleus 2.
    out = _remove_outside_objects(_three_channel_stack(), 0, 1, 2)
    assert int((out[:, :, 0] == 2).sum()) == 100, "cell 2 must survive"
    assert int((out[:, :, 1] == 2).sum()) == 9, "the nucleus of surviving cell 2 must survive"


def test_remove_noninfected_keeps_a_fully_infected_cell():
    from spacr.utils import _remove_noninfected

    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1        # cell 1
    stack[4:8, 4:8, 1] = 1          # nucleus 1
    stack[2:12, 2:12, 2] = 1        # pathogen 1 covers cell 1 exactly

    out = _remove_noninfected(stack, 0, 1, 2)
    assert int((out[:, :, 0] == 1).sum()) == 100, "a fully infected cell must be kept"
    assert int((out[:, :, 1] == 1).sum()) == 16


def test_remove_multiobject_cells_removes_a_fully_tiled_cell():
    from spacr.utils import _remove_multiobject_cells

    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1        # cell 1
    stack[4:6, 4:6, 1] = 1          # nucleus 1
    stack[2:12, 2:7, 2] = 1         # pathogen 1 } together they tile cell 1
    stack[2:12, 7:12, 2] = 2        # pathogen 2 }

    out = _remove_multiobject_cells(stack, 0, 0, 1, 2, 2)
    assert set(np.unique(out[:, :, 0])) == {0}, "a cell with two pathogens must be removed"


def test_remove_outside_objects_returns_early_without_cell_dim():
    from spacr.utils import _remove_outside_objects

    stack = _three_channel_stack()
    out = _remove_outside_objects(stack.copy(), None, 1, 2)
    assert np.array_equal(out, stack)


def test_remove_multiobject_cells_clears_cells_with_two_pathogens():
    from spacr.utils import _remove_multiobject_cells

    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1        # cell 1 -> will hold two pathogens
    stack[20:30, 2:12, 0] = 2       # cell 2 -> a single pathogen
    stack[5:8, 5:8, 1] = 1
    stack[23:26, 5:8, 1] = 2
    stack[3:5, 3:5, 2] = 1          # pathogen 1 in cell 1
    stack[9:11, 9:11, 2] = 2        # pathogen 2 in cell 1
    stack[23:25, 5:7, 2] = 3        # pathogen 3 in cell 2

    out = _remove_multiobject_cells(stack, 0, 0, 1, 2, 2)
    assert set(np.unique(out[:, :, 0])) == {0, 2}, "the multi-pathogen cell is removed"
    assert set(np.unique(out[:, :, 1])) == {0, 2}, "its nucleus is removed too"
    assert set(np.unique(out[:, :, 2])) == {0, 3}, "both of its pathogens are removed"
    assert int((out[:, :, 0] == 2).sum()) == 100


def test_remove_multiobject_cells_keeps_single_object_cells():
    from spacr.utils import _remove_multiobject_cells

    stack = np.zeros((30, 30, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1
    stack[4:6, 4:6, 1] = 1
    stack[6:8, 6:8, 2] = 1
    before = stack.copy()

    out = _remove_multiobject_cells(stack, 0, 0, 1, 2, 2)
    assert np.array_equal(out, before)
