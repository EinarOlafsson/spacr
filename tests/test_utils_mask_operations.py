"""Table-driven tests of the mask metrics and mask filtering helpers in ``spacr.utils``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: file listing, circular masks, image inversion,
resizing, IoU / AP metrics, the multi-channel mask clean-up helpers, the
object filters, the filename regexes, test-mode sampling and Cellpose model
resolution.  Everything is synthetic, tiny, CPU-only and offline; Cellpose is
replaced by a recording constructor double so no weights are loaded.
"""
from __future__ import annotations

import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.utils as U  # noqa: E402
from spacr.model_zoo import ModelZooError  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _block(shape, sl, dtype=bool, value=True):
    """Return an array of ``shape`` with ``sl`` filled with ``value``."""
    arr = np.zeros(shape, dtype=dtype)
    arr[sl] = value
    return arr


def _uniq(arr):
    """Sorted list of the distinct values of ``arr``."""
    return sorted(np.unique(arr).tolist())


def _two_object_mask(shape=(32, 32)):
    """Label mask with a small (16 px) object 1 and a big (100 px) object 2."""
    m = np.zeros(shape, dtype=np.int32)
    m[2:6, 2:6] = 1
    m[10:20, 10:20] = 2
    return m


def _intensity_batch_image(mask, values, shape=(32, 32), channels=2):
    """(H, W, C) float32 image whose channel 1 carries per-label ``values``."""
    img = np.zeros((*shape, channels), dtype=np.float32)
    for lbl, val in values.items():
        img[..., 1][mask == lbl] = val
    return img


def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# get_files_from_dir
# ---------------------------------------------------------------------------

def test_get_files_from_dir_lists_absolute_paths_matching_the_pattern(subtests, tmp_path):
    """Files (and folders) matching the glob are listed as absolute paths."""
    cases = [
        dict(id="tif_only", files=["a.tif", "b.tif", "c.csv"], dirs=[],
             pattern="*.tif", want=["a.tif", "b.tif"]),
        dict(id="default_everything", files=["a.tif", "c.csv"], dirs=["sub"],
             pattern=None, want=["a.tif", "c.csv", "sub"]),
        dict(id="empty_dir", files=[], dirs=[], pattern="*.tif", want=[]),
        dict(id="tif_among_txt", files=["a.tif", "b.tif", "c.txt"], dirs=[],
             pattern="*.tif", want=["a.tif", "b.tif"]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            d = _case_dir(tmp_path, case["id"])
            for name in case["files"]:
                (d / name).write_text("x", encoding="utf-8")
            for name in case["dirs"]:
                (d / name).mkdir()
            if case["pattern"] is None:
                out = U.get_files_from_dir(str(d))
            else:
                out = U.get_files_from_dir(str(d), case["pattern"])
            assert isinstance(out, list)
            assert sorted(os.path.basename(p) for p in out) == case["want"]
            assert all(os.path.isabs(p) for p in out)
            assert all(os.path.dirname(p) == str(d) for p in out)


# ---------------------------------------------------------------------------
# create_circular_mask / apply_mask
# ---------------------------------------------------------------------------

def test_create_circular_mask_fills_the_disc_around_the_centre(subtests):
    """A boolean disc: inscribed by default, else at (x, y) centre with a radius."""
    cases = [
        dict(id="default_10", h=10, w=10, kw={}, inside=[(5, 5), (0, 5), (5, 0)],
             outside=[(0, 0), (9, 9)], area=79, coords=None),
        dict(id="explicit_8", h=8, w=8, kw=dict(center=(2, 3), radius=1),
             inside=[], outside=[], area=5,
             coords={(3, 2), (2, 2), (4, 2), (3, 1), (3, 3)}),
        dict(id="default_20", h=20, w=20, kw={}, inside=[(10, 10)],
             outside=[(0, 0)], area=315, coords=None),
        dict(id="explicit_20", h=20, w=20, kw=dict(center=(5, 5), radius=3),
             inside=[(5, 5)], outside=[(15, 15)], area=29, coords=None),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            mask = U.create_circular_mask(case["h"], case["w"], **case["kw"])
            assert mask.shape == (case["h"], case["w"])
            assert mask.dtype == np.bool_
            for rc in case["inside"]:
                assert mask[rc]
            for rc in case["outside"]:
                assert not mask[rc]
            assert int(mask.sum()) == case["area"]
            if case["coords"] is not None:
                got = {tuple(map(int, rc)) for rc in zip(*np.nonzero(mask))}
                assert got == case["coords"]


def test_apply_mask_replaces_everything_outside_the_inscribed_disc(subtests):
    """Pixels outside the inscribed disc take ``output_value`` in every channel."""
    cases = [
        dict(id="gray_uint16_20", img=np.full((20, 20), 100, np.uint16), kw={},
             fill=0, inside=[(10, 10)], outside=[(0, 0), (19, 19)], n_inside=315),
        dict(id="rgb_uint8_value9", img=np.full((20, 20, 3), 7, np.uint8),
             kw=dict(output_value=9), fill=9, inside=[(10, 10)], outside=[(0, 0)],
             n_inside=315),
        dict(id="gray_uint8_16", img=np.ones((16, 16), np.uint8) * 100, kw={},
             fill=0, inside=[(8, 8)], outside=[(0, 0)], n_inside=195),
        dict(id="rgb_uint8_16", img=np.ones((16, 16, 3), np.uint8) * 100, kw={},
             fill=0, inside=[(8, 8)], outside=[(0, 0)], n_inside=195),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            img = case["img"]
            original = img.copy()
            out = U.apply_mask(img, **case["kw"])
            assert out.shape == img.shape
            for rc in case["inside"]:
                assert np.all(out[rc] == original[rc])
            for rc in case["outside"]:
                assert np.all(out[rc] == case["fill"])
            plane = out if out.ndim == 2 else out[..., 0]
            orig_plane = original if original.ndim == 2 else original[..., 0]
            kept = plane == orig_plane
            assert int(kept.sum()) == case["n_inside"]
            if out.ndim == 3:
                # the mask is repeated per channel
                for c in range(out.shape[2]):
                    assert np.array_equal(out[..., c] == original[..., c], kept)
            assert np.all(plane[~kept] == case["fill"])


# ---------------------------------------------------------------------------
# invert_image
# ---------------------------------------------------------------------------

def _invert_cases():
    cases = []
    for dt in (np.uint8, np.uint16):
        mx = np.iinfo(dt).max
        cases.append(dict(id=f"pivot_{dt.__name__}", dtype=dt, img=[[0, 1, mx]],
                          want=[[mx, mx - 1, 0]]))
    for dt in (np.int8, np.int16, np.int32):
        info = np.iinfo(dt)
        cases.append(dict(id=f"signed_{dt.__name__}", dtype=dt,
                          img=[[info.min, -100, 0, 1, info.max]],
                          want=[[info.max, 99, -1, -2, info.min]]))
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dt)
        cases.append(dict(id=f"range_{dt.__name__}", dtype=dt,
                          img=[info.min, info.max], want=[info.max, info.min]))
    cases.append(dict(id="uint16_ends", dtype=np.uint16, img=[[0, 65535]],
                      want=[[65535, 0]]))
    cases.append(dict(id="uint8_grid", dtype=np.uint8, img=[[0, 255], [100, 200]],
                      want=[[255, 0], [155, 55]]))
    for bad in (np.float32, np.float64, bool):
        cases.append(dict(id=f"refuse_{getattr(bad, '__name__', bad)}", dtype=bad,
                          img=np.zeros((2, 2)), want=None))
    return cases


def test_invert_image_reflects_through_the_integer_dtype_range(subtests):
    """min + max - image, same dtype, no wrap, an involution; non-integers refused."""
    for case in _invert_cases():
        with subtests.test(case=case["id"]):
            img = np.array(case["img"], dtype=case["dtype"])
            if case["want"] is None:
                with pytest.raises(ValueError, match="integer dtype"):
                    U.invert_image(img)
                continue
            out = U.invert_image(img)
            assert out.dtype == case["dtype"]
            assert np.array_equal(out, np.array(case["want"], dtype=case["dtype"]))
            assert np.array_equal(U.invert_image(out), img)


# ---------------------------------------------------------------------------
# resize_images_and_labels / resize_labels_back
# ---------------------------------------------------------------------------

def _rng_img(seed, shape, dtype, hi=None):
    rng = np.random.default_rng(seed)
    if hi is None:
        return rng.random(shape).astype(dtype)
    return rng.integers(0, hi, shape).astype(dtype)


def _lbl(shape, dtype, *blocks):
    m = np.zeros(shape, dtype=dtype)
    for sl, v in blocks:
        m[sl] = v
    return m


def test_resize_images_and_labels_resizes_each_frame_to_the_target(subtests):
    """Images keep dtype/channels (singleton squeezed); labels resize nearest-neighbour."""
    cases = [
        dict(id="rgb_with_labels",
             images=[_rng_img(3, (32, 32, 3), np.uint8, 255)],
             labels=[_lbl((32, 32), np.uint16, (np.s_[4:20, 4:20], 5))],
             h=16, w=16, img_shapes=[(16, 16, 3)], img_dtype=np.uint8,
             lbl_shapes=[(16, 16)], lbl_dtype=np.uint16, lbl_values=[[0, 5]]),
        dict(id="singleton_with_labels",
             images=[_rng_img(4, (32, 32, 1), np.uint8, 255)],
             labels=[_lbl((32, 32), np.uint16, (np.s_[8:24, 8:24], 2))],
             h=16, w=16, img_shapes=[(16, 16)], img_dtype=np.uint8,
             lbl_shapes=[(16, 16)], lbl_dtype=np.uint16, lbl_values=[[0, 2]]),
        dict(id="images_only_2d", images=[_rng_img(5, (32, 32), np.float32)],
             labels=None, h=16, w=8, img_shapes=[(16, 8)], img_dtype=np.float32,
             lbl_shapes=[], lbl_dtype=None, lbl_values=[]),
        dict(id="images_only_singleton",
             images=[_rng_img(6, (32, 32, 1), np.uint16, 1000)], labels=None,
             h=16, w=16, img_shapes=[(16, 16)], img_dtype=np.uint16,
             lbl_shapes=[], lbl_dtype=None, lbl_values=[]),
        dict(id="images_only_rgb_float",
             images=[_rng_img(1, (32, 32, 3), np.float32)], labels=None,
             h=16, w=16, img_shapes=[(16, 16, 3)], img_dtype=np.float32,
             lbl_shapes=[], lbl_dtype=None, lbl_values=[]),
        dict(id="labels_only_nearest", images=None,
             labels=[_lbl((32, 32), np.int32, (np.s_[0:16, 0:16], 3),
                          (np.s_[16:32, 16:32], 9))],
             h=16, w=16, img_shapes=[], img_dtype=None,
             lbl_shapes=[(16, 16)], lbl_dtype=np.int32, lbl_values=[[0, 3, 9]]),
        dict(id="labels_only_empty", images=None,
             labels=[np.zeros((32, 32), np.uint16)], h=16, w=16, img_shapes=[],
             img_dtype=None, lbl_shapes=[(16, 16)], lbl_dtype=np.uint16,
             lbl_values=[[0]]),
        dict(id="gray_float_with_labels",
             images=[_rng_img(0, (32, 32), np.float32)],
             labels=[_lbl((32, 32), np.uint16, (np.s_[4:12, 4:12], 1))],
             h=16, w=16, img_shapes=[(16, 16)], img_dtype=np.float32,
             lbl_shapes=[(16, 16)], lbl_dtype=np.uint16, lbl_values=[[0, 1]]),
        dict(id="plane_8_to_4", images=[np.zeros((8, 8), np.uint8)],
             labels=[np.zeros((8, 8), np.uint16)], h=4, w=4,
             img_shapes=[(4, 4)], img_dtype=np.uint8, lbl_shapes=[(4, 4)],
             lbl_dtype=np.uint16, lbl_values=[[0]]),
        dict(id="all_none", images=None, labels=None, h=16, w=16, img_shapes=[],
             img_dtype=None, lbl_shapes=[], lbl_dtype=None, lbl_values=[]),
        # A 4-D frame gets no target shape of its own and scikit-image refuses it.
        dict(id="4d_frame_with_labels",
             images=[np.zeros((8, 8), np.uint8), np.zeros((8, 8, 2, 2), np.uint8)],
             labels=[np.zeros((8, 8), np.uint16)] * 2, h=4, w=4, raises=ValueError),
        dict(id="4d_frame_images_only",
             images=[np.zeros((8, 8), np.uint8), np.zeros((8, 8, 2, 2), np.uint8)],
             labels=None, h=4, w=4, raises=ValueError),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            if case.get("raises"):
                with pytest.raises(case["raises"]):
                    U.resize_images_and_labels(case["images"], case["labels"],
                                               case["h"], case["w"], show_example=False)
                continue
            imgs, lbls = U.resize_images_and_labels(case["images"], case["labels"],
                                                    case["h"], case["w"],
                                                    show_example=False)
            assert [i.shape for i in imgs] == case["img_shapes"]
            assert [l.shape for l in lbls] == case["lbl_shapes"]
            for i in imgs:
                assert i.dtype == case["img_dtype"]
            for l, values in zip(lbls, case["lbl_values"]):
                assert l.dtype == case["lbl_dtype"]
                assert _uniq(l) == values


def test_resize_images_and_labels_show_example_hands_the_pair_to_the_plotter(
        subtests, monkeypatch):
    """show_example calls plot_resize once with (orig, resized, orig, resized); nothing to show draws nothing."""
    import spacr.plot

    img = _rng_img(7, (32, 32), np.float32)
    lbl = _lbl((32, 32), np.uint16, (np.s_[2:10, 2:10], 1))
    lbl_only = _lbl((32, 32), np.uint16, (np.s_[2:20, 2:20], 4))
    cases = [
        dict(id="images_and_labels", images=[img], labels=[lbl], mode="both"),
        dict(id="images_only", images=[img], labels=None, mode="images"),
        dict(id="labels_only", images=None, labels=[lbl_only], mode="labels"),
        dict(id="nothing", images=None, labels=None, mode="none"),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            calls = []
            with monkeypatch.context() as mp:
                mp.setattr(spacr.plot, "plot_resize",
                           lambda *a, **k: calls.append(a))
                imgs, lbls = U.resize_images_and_labels(case["images"], case["labels"],
                                                        16, 16, show_example=True)
            if case["mode"] == "none":
                assert (imgs, lbls) == ([], [])
                assert calls == []
                continue
            assert len(calls) == 1
            a, b, c, d = calls[0]
            if case["mode"] == "both":
                assert a[0].shape == (32, 32) and c[0].shape == (32, 32)
                assert b[0].shape == (16, 16) and d[0].shape == (16, 16)
                assert b is imgs and d is lbls
            elif case["mode"] == "images":
                assert a is c and b is d
                assert b is imgs
                assert lbls == []
            else:
                assert a is c and b is d
                assert b is lbls
                assert imgs == []
                assert lbls[0].shape == (16, 16)


def test_resize_labels_back_restores_each_label_to_its_original_dims(subtests):
    """Nearest-neighbour resize to each (w, h) tuple, dtype kept; malformed input refused."""
    cases = [
        dict(id="values_and_dtype",
             labels=[_lbl((16, 16), np.int32, (np.s_[0:8, 0:8], 11))],
             dims=[(32, 24)], shapes=[(32, 24)], dtypes=[np.int32], values=[[0, 11]]),
        dict(id="blank_uint16", labels=[np.zeros((16, 16), np.uint16)],
             dims=[(32, 24)], shapes=[(32, 24)], dtypes=[np.uint16], values=[[0]]),
        dict(id="list_dims", labels=[np.zeros((4, 4), np.uint8)], dims=[[8, 8]],
             raises="tuple of two integers"),
        dict(id="one_element_tuple", labels=[np.zeros((4, 4))], dims=[(8,)],
             raises="tuple of two integers"),
        dict(id="length_mismatch_uint8", labels=[np.zeros((4, 4), np.uint8)],
             dims=[(8, 8), (8, 8)],
             raises="length of labels and orig_dims must match"),
        dict(id="length_mismatch_float", labels=[np.zeros((4, 4))],
             dims=[(8, 8), (8, 8)],
             raises="length of labels and orig_dims must match"),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            if "raises" in case:
                with pytest.raises(ValueError, match=case["raises"]):
                    U.resize_labels_back(case["labels"], case["dims"])
                continue
            out = U.resize_labels_back(case["labels"], case["dims"])
            assert [o.shape for o in out] == case["shapes"]
            assert [o.dtype for o in out] == case["dtypes"]
            assert [_uniq(o) for o in out] == case["values"]


# ---------------------------------------------------------------------------
# calculate_iou / match_masks / AP
# ---------------------------------------------------------------------------

def test_calculate_iou_pads_to_a_common_shape_and_guards_empty_union(subtests):
    """IoU after zero padding; an empty union is 0 rather than a division error."""
    a = _block((10, 10), np.s_[0:5, 0:5])
    b = _block((10, 10), np.s_[5:10, 5:10])
    empty = np.zeros((10, 10), dtype=bool)
    cases = [
        dict(id="identical", m1=a, m2=a, want=1.0),
        dict(id="disjoint", m1=a, m2=b, want=0.0),
        dict(id="both_empty", m1=empty, m2=empty, want=0),
        dict(id="float_zeros", m1=np.zeros((3, 3)), m2=np.zeros((3, 3)), want=0),
        dict(id="pad_4x4_vs_4x8", m1=np.ones((4, 4), bool), m2=np.ones((4, 8), bool),
             want=16 / 32, padded=(4, 8)),
        dict(id="pad_4x4_vs_6x5", m1=np.ones((4, 4), bool), m2=np.ones((6, 5), bool),
             want=16 / 30, padded=(6, 5)),
        dict(id="pad_empty_4x4", m1=np.zeros((4, 4), bool), m2=np.zeros((4, 4), bool),
             want=0),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            got = U.calculate_iou(case["m1"], case["m2"])
            assert got == pytest.approx(case["want"])
            if "padded" in case:
                pa, pb = U.pad_to_same_shape(case["m1"], case["m2"])
                assert pa.shape == pb.shape == case["padded"]


def test_match_masks_pairs_each_true_mask_at_most_once_above_threshold(subtests):
    """Greedy matching: one prediction per true mask, IoU must exceed the threshold."""
    a = _block((10, 10), np.s_[0:5, 0:5])
    b = _block((10, 10), np.s_[6:10, 6:10])
    true = _block((10, 10), np.s_[0:5, 0:5])
    pred80 = _block((10, 10), np.s_[0:5, 0:4])           # IoU 0.8
    t1 = _block((8, 8), np.s_[1:4, 1:4])
    t2 = _block((8, 8), np.s_[5:7, 5:7])
    cases = [
        dict(id="duplicate_predictions", true=[a, b], pred=[a.copy(), a.copy()],
             thr=0.5, pairs=[(a, a)], ap=None),
        dict(id="iou_08_at_05", true=[true], pred=[pred80], thr=0.5,
             pairs=[(true, pred80)], ap=None),
        dict(id="iou_08_at_09", true=[true], pred=[pred80], thr=0.9, pairs=[], ap=None),
        dict(id="one_of_two_found", true=[t1, t2], pred=[t1.copy()], thr=0.5,
             pairs=[(t1, t1)], ap=(2, 1, 1.0, 0.5)),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            matches = U.match_masks(case["true"], case["pred"], case["thr"])
            assert len(matches) == len(case["pairs"])
            for (gt, gp), (wt, wp) in zip(matches, case["pairs"]):
                assert np.array_equal(gt, wt)
                assert np.array_equal(gp, wp)
            if case["ap"] is not None:
                n_true, n_pred, p, r = case["ap"]
                assert U.compute_average_precision(matches, n_true, n_pred) == (p, r)


def _ap_fixture_masks():
    """true: one big + one small object; pred: an 80%-overlap copy of the big one."""
    true0 = _block((20, 20), np.s_[0:10, 0:10])
    true1 = _block((20, 20), np.s_[15:19, 15:19])
    pred0 = _block((20, 20), np.s_[0:10, 0:8])
    return [true0, true1], [pred0]


class _LyingLenList(list):
    """Iterates its items but reports length 1 (makes FP negative)."""

    def __len__(self):  # noqa: D105
        return 1


def test_compute_ap_over_iou_thresholds_integrates_precision_over_recall(subtests):
    """AP is the trapezoid area under (recall, precision); out-of-range precision is refused."""
    true_masks, pred_masks = _ap_fixture_masks()
    t1 = _block((8, 8), np.s_[1:4, 1:4])
    a = _block((10, 10), np.s_[0:5, 0:5])
    b = _block((10, 10), np.s_[6:10, 6:10])
    cases = [
        # 0.5 -> (P=1, R=0.5); 0.9 -> (0, 0); trapz == 0.25
        dict(id="two_thresholds", true=true_masks, pred=pred_masks, thr=[0.5, 0.9],
             want=0.25),
        dict(id="no_predictions", true=true_masks, pred=[], thr=[0.5, 0.9], want=0.0),
        # equal recalls at both thresholds -> zero area
        dict(id="perfect_flat_recall", true=[t1], pred=[t1.copy()], thr=[0.5, 0.75],
             want=0.0),
        dict(id="lying_len", true=[a, b], pred=_LyingLenList([a.copy(), b.copy()]),
             thr=[0.5], raises=["out of bounds", "Precision: 2.0"]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            if "raises" in case:
                with pytest.raises(ValueError) as err:
                    U.compute_ap_over_iou_thresholds(case["true"], case["pred"], case["thr"])
                for frag in case["raises"]:
                    assert frag in str(err.value)
                continue
            ap = U.compute_ap_over_iou_thresholds(case["true"], case["pred"], case["thr"])
            assert ap == pytest.approx(case["want"])


def test_compute_segmentation_ap_scores_connected_components(subtests):
    """Binary images are split into components and scored like mask lists."""
    true_img = np.zeros((20, 20), np.uint8)
    true_img[0:10, 0:10] = 1
    true_img[15:19, 15:19] = 1
    pred_img = np.zeros((20, 20), np.uint8)
    pred_img[0:10, 0:8] = 1
    single = np.zeros((20, 20), np.uint8)
    single[2:8, 2:8] = 1
    same = np.zeros((16, 16), np.int32)
    same[2:6, 2:6] = 1
    cases = [
        dict(id="one_of_two_at_80pct", true=true_img, pred=pred_img, thr=[0.5, 0.9],
             want=0.25),
        dict(id="empty_prediction", true=single, pred=np.zeros((20, 20), np.uint8),
             thr=[0.5, 0.9], want=0.0),
        dict(id="identical_one_threshold", true=same, pred=same.copy(), thr=[0.5],
             want=0.0),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            ap = U.compute_segmentation_ap(case["true"], case["pred"],
                                           iou_thresholds=case["thr"])
            assert ap == pytest.approx(case["want"])


# ---------------------------------------------------------------------------
# _remove_noninfected / _remove_outside_objects / _remove_multiobject_cells
# ---------------------------------------------------------------------------

def _three_channel_stack():
    """(40, 40, 3): ch0 cells, ch1 nuclei, ch2 pathogens.

    cell 1 (pathogen 1 + nucleus 1), cell 2 (nucleus 2, no pathogen),
    pathogen 2 outside every cell.
    """
    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1
    stack[20:30, 2:12, 0] = 2
    stack[5:8, 5:8, 1] = 1
    stack[23:26, 5:8, 1] = 2
    stack[6:9, 6:9, 2] = 1
    stack[33:37, 33:37, 2] = 2
    return stack


def _small_stack(size=24):
    """(24, 24, 3): cell 1 infected (pathogen 1), cell 2 uninfected."""
    cell = np.zeros((size, size), dtype=np.int32)
    cell[2:10, 2:10] = 1
    cell[14:22, 14:22] = 2
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[3:6, 3:6] = 1
    nucleus[15:18, 15:18] = 2
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[6:9, 6:9] = 1
    return np.stack([cell, nucleus, pathogen], axis=-1)


def _fully_infected_stack():
    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1
    stack[4:8, 4:8, 1] = 1
    stack[2:12, 2:12, 2] = 1         # pathogen covers cell 1 exactly
    return stack


def _check_channels(out, before, expect):
    """``expect`` maps channel -> ("same",) | (unique values, {label: px count})."""
    for ch, spec in expect.items():
        if spec == "same":
            assert np.array_equal(out[:, :, ch], before[:, :, ch]), ch
            continue
        values, counts = spec
        assert _uniq(out[:, :, ch]) == values, ch
        for lbl, n in counts.items():
            assert int((out[:, :, ch] == lbl).sum()) == n, (ch, lbl)


def test_remove_noninfected_clears_cells_without_a_pathogen(subtests):
    """Cells (and their nuclei) with no pathogen are zeroed; pathogens untouched."""
    cases = [
        dict(id="clears_cell_2", stack=_three_channel_stack, dims=(0, 1, 2),
             expect={0: ([0, 1], {1: 100}), 1: ([0, 1], {}), 2: "same"}),
        dict(id="cell_dim_none_noop", stack=_three_channel_stack, dims=(None, 1, 2),
             unchanged=True),
        dict(id="no_nucleus_no_pathogen_dims", stack=_three_channel_stack,
             dims=(0, None, None), expect={0: ([0], {}), 1: "same", 2: "same"}),
        dict(id="fully_infected_kept", stack=_fully_infected_stack, dims=(0, 1, 2),
             expect={0: ([0, 1], {1: 100}), 1: ([0, 1], {1: 16}), 2: "same"}),
        dict(id="small_stack", stack=_small_stack, dims=(0, 1, 2),
             expect={0: ([0, 1], {1: 64}), 1: ([0, 1], {1: 9}), 2: "same"}),
        dict(id="all_dims_none", stack=_small_stack, dims=(None, None, None),
             unchanged=True),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            before = case["stack"]()
            out = U._remove_noninfected(before.copy(), *case["dims"])
            assert out.shape == before.shape
            if case.get("unchanged"):
                assert np.array_equal(out, before)
            else:
                _check_channels(out, before, case["expect"])


def _stray_pathogen_stack():
    cell = np.zeros((16, 16), np.int32)
    cell[0:5, 0:5] = 1
    pathogen = np.zeros((16, 16), np.int32)
    pathogen[9:15, 9:15] = 4             # outside every cell
    nucleus = np.zeros((16, 16), np.int32)
    nucleus[10:13, 10:13] = 8            # inside that pathogen
    nucleus[1:3, 1:3] = 9                # unrelated, inside the cell
    return np.stack([cell, nucleus, pathogen], axis=-1).astype(np.int32)


def _small_stack_with_stray():
    stack = _small_stack()
    stack[20:22, 0:2, 2] = 5
    return stack


def test_remove_outside_objects_clears_pathogens_with_no_host_cell(subtests):
    """A pathogen outside every cell goes, with only the nuclei inside its footprint."""
    cases = [
        dict(id="stray_pathogen_takes_its_nucleus", stack=_stray_pathogen_stack,
             dims=(0, 1, 2),
             expect={0: "same", 1: ([0, 9], {9: 4}), 2: ([0], {})}),
        dict(id="unrelated_nucleus_survives", stack=_three_channel_stack, dims=(0, 1, 2),
             expect={0: ([0, 1, 2], {2: 100}), 1: ([0, 1, 2], {2: 9}),
                     2: ([0, 1], {1: 9})}),
        dict(id="no_cell_dim", stack=_three_channel_stack, dims=(None, 1, 2),
             unchanged=True),
        dict(id="small_stack_stray_5", stack=_small_stack_with_stray, dims=(0, 1, 2),
             expect={0: "same", 1: "same", 2: ([0, 1], {1: 9})}),
        dict(id="small_stack_no_cell_dim", stack=_small_stack, dims=(None, 1, 2),
             unchanged=True),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            before = case["stack"]()
            out = U._remove_outside_objects(before.copy(), *case["dims"])
            if case.get("unchanged"):
                assert np.array_equal(out, before)
            else:
                _check_channels(out, before, case["expect"])


def _tiled_stack():
    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1
    stack[4:6, 4:6, 1] = 1
    stack[2:12, 2:7, 2] = 1
    stack[2:12, 7:12, 2] = 2
    return stack


def _two_pathogen_stack():
    stack = np.zeros((40, 40, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1
    stack[20:30, 2:12, 0] = 2
    stack[5:8, 5:8, 1] = 1
    stack[23:26, 5:8, 1] = 2
    stack[3:5, 3:5, 2] = 1
    stack[9:11, 9:11, 2] = 2
    stack[23:25, 5:7, 2] = 3
    return stack


def _single_object_stack():
    stack = np.zeros((30, 30, 3), dtype=np.uint16)
    stack[2:12, 2:12, 0] = 1
    stack[4:6, 4:6, 1] = 1
    stack[6:8, 6:8, 2] = 1
    return stack


def _no_pathogen_channel_stack():
    stack = np.zeros((12, 12, 3), np.uint16)
    stack[2:8, 2:8, 0] = 1
    stack[9:11, 9:11, 0] = 2
    stack[3:5, 3:5, 1] = 1
    stack[6:7, 6:7, 1] = 2
    stack[9:10, 9:10, 1] = 3
    stack[3:6, 3:6, 2] = 7
    stack[9:11, 9:11, 2] = 8
    return stack


def _four_channel_stack():
    size = 24
    mask = np.zeros((size, size), dtype=np.int32)
    mask[2:12, 2:12] = 1
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[3:5, 3:5] = 1
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[3:5, 3:5] = 1
    pathogen[8:10, 8:10] = 2
    return np.stack([mask, mask.copy(), nucleus, pathogen], axis=-1)


def test_remove_multiobject_cells_clears_cells_holding_several_objects(subtests):
    """A cell with >1 object is removed with its nucleus and objects; others stay."""
    cases = [
        dict(id="fully_tiled_cell", stack=_tiled_stack, dims=(0, 0, 1, 2, 2),
             expect={0: ([0], {})}),
        dict(id="two_pathogens_in_cell_1", stack=_two_pathogen_stack,
             dims=(0, 0, 1, 2, 2),
             expect={0: ([0, 2], {2: 100}), 1: ([0, 2], {}), 2: ([0, 3], {})}),
        dict(id="single_object_cells_kept", stack=_single_object_stack,
             dims=(0, 0, 1, 2, 2), unchanged=True),
        dict(id="no_pathogen_channel", stack=_no_pathogen_channel_stack,
             dims=(0, None, 2, None, 1),
             expect={0: ([0, 2], {}), 2: ([0, 8], {}), 1: ([0, 1, 2, 3], {})}),
        dict(id="four_channel_stack", stack=_four_channel_stack, dims=(0, 1, 2, 3, 3),
             expect={1: ([0], {})}),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            before = case["stack"]()
            mask_dim, cell_dim, nucleus_dim, pathogen_dim, object_dim = case["dims"]
            out = U._remove_multiobject_cells(
                before.copy(), mask_dim=mask_dim, cell_dim=cell_dim,
                nucleus_dim=nucleus_dim, pathogen_dim=pathogen_dim,
                object_dim=object_dim)
            if case.get("unchanged"):
                assert np.array_equal(out, before)
            else:
                _check_channels(out, before, case["expect"])


# ---------------------------------------------------------------------------
# remove_intensity_objects / _filter_closest_to_stat
# ---------------------------------------------------------------------------

def test_remove_intensity_objects_drops_objects_on_the_wrong_side(subtests):
    """mode='low' drops dim objects, mode='high' bright ones, footprints kept."""
    mask = _two_object_mask()
    split = np.zeros(mask.shape, np.float32)
    split[mask == 1] = 10.0
    split[mask == 2] = 200.0
    flat = np.zeros(mask.shape, np.float32)
    flat[mask > 0] = 50.0
    cases = [
        dict(id="low", image=split, thr=100.0, mode="low", want=[0, 2], counts={2: 100}),
        dict(id="high", image=split, thr=100.0, mode="high", want=[0, 1], counts={1: 16}),
        dict(id="extreme_threshold", image=flat, thr=0.0, mode="low", want=[0, 1, 2],
             counts={1: 16, 2: 100}),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            out = U.remove_intensity_objects(case["image"], mask.copy(),
                                             intensity_threshold=case["thr"],
                                             mode=case["mode"])
            assert _uniq(out) == case["want"]
            for lbl, n in case["counts"].items():
                assert int((out == lbl).sum()) == n


def test_filter_closest_to_stat_keeps_the_rows_nearest_the_mean_or_median(subtests):
    """The n rows closest to the column mean (or median) survive; no helper column leaks."""
    base = [0.0, 10.0, 11.0, 12.0, 67.0]
    cases = [
        dict(id="mean_two", data={"v": base, "tag": list("abcde")}, n=2, med=None,
             want=[11.0, 12.0], tags=["c", "d"]),
        dict(id="mean_one", data={"v": base}, n=1, med=None, want=[12.0]),
        dict(id="median_one", data={"v": base}, n=1, med=True, want=[11.0]),
        dict(id="median_two", data={"v": [1.0, 5.0, 5.1, 5.2, 100.0]}, n=2, med=True,
             want=[5.0, 5.1]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            df = pd.DataFrame(case["data"])
            kw = {} if case["med"] is None else {"use_median": case["med"]}
            out = U._filter_closest_to_stat(df.copy(), "v", n_rows=case["n"], **kw)
            assert "diff" not in out.columns
            assert len(out) == case["n"]
            assert sorted(out["v"].tolist()) == case["want"]
            if "tags" in case:
                assert sorted(out["tag"].tolist()) == case["tags"]


# ---------------------------------------------------------------------------
# _find_similar_sized_images
# ---------------------------------------------------------------------------

def test_find_similar_sized_images_returns_the_largest_same_crop_group(subtests, tmp_path):
    """Images grouped by their non-zero bounding-box size; blanks skipped; nothing readable raises."""
    import cv2

    def gray(i):
        img = np.zeros((24, 24), np.uint8)
        img[6:14, 6:14] = 180
        return img

    def gray_odd():
        img = np.zeros((24, 24), np.uint8)
        img[2:20, 2:5] = 180
        return img

    def rgb(i):
        img = np.zeros((24, 24, 3), np.uint8)
        img[4:16, 4:10, i % 3] = 200
        return img

    def rgb_odd():
        img = np.zeros((24, 24, 3), np.uint8)
        img[2:6, 2:22] = 200
        return img

    def rgb20(i):
        img = np.zeros((20, 20, 3), np.uint8)
        img[4:12, 4:12] = 200
        return img

    def rgb20_odd():
        img = np.zeros((20, 20, 3), np.uint8)
        img[2:18, 2:6] = 200
        return img

    cases = [
        dict(id="grayscale_with_blank", group=[gray(i) for i in range(3)],
             others=[np.zeros((24, 24), np.uint8), gray_odd()]),
        dict(id="colour_per_pixel_any", group=[rgb(i) for i in range(3)],
             others=[rgb_odd()]),
        dict(id="colour_20px", group=[rgb20(i) for i in range(3)], others=[rgb20_odd()]),
        dict(id="nothing_readable", group=[], others=[], missing=True),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            d = _case_dir(tmp_path, case["id"])
            if case.get("missing"):
                with pytest.raises(ValueError):
                    U._find_similar_sized_images([str(d / "does_not_exist.png")])
                continue
            group_paths, other_paths = [], []
            for i, img in enumerate(case["group"]):
                p = str(d / f"g{i}.png")
                assert cv2.imwrite(p, img)
                group_paths.append(p)
            for i, img in enumerate(case["others"]):
                p = str(d / f"o{i}.png")
                assert cv2.imwrite(p, img)
                other_paths.append(p)
            got = U._find_similar_sized_images(group_paths + other_paths)
            assert len(got) == len(group_paths)
            assert set(got) == set(group_paths)


# ---------------------------------------------------------------------------
# _relabel_parent_with_child_labels / _exclude_objects / _merge_overlapping_objects
# ---------------------------------------------------------------------------

def test_relabel_parent_with_child_labels_gives_parents_their_child_ids(subtests):
    """A parent takes its child's id (the last one when several); children standardise on the first."""
    def single(shape, psl, csl, cid):
        parent = np.zeros(shape, np.int32)
        parent[psl] = 1
        child = np.zeros(shape, np.int32)
        child[csl] = cid
        return parent, child

    p2 = np.zeros((24, 24), np.int32)
    p2[4:20, 4:20] = 1
    c2 = np.zeros((24, 24), np.int32)
    c2[6:9, 6:9] = 3
    c2[14:17, 14:17] = 7
    cases = [
        dict(id="single_child_24", masks=single((24, 24), np.s_[4:16, 4:16],
                                                np.s_[6:10, 6:10], 5),
             parent=[0, 5], child=[0, 5], parent_px={5: 144}, child_px={5: 16}),
        dict(id="single_child_16", masks=single((16, 16), np.s_[2:12, 2:12],
                                                np.s_[4:7, 4:7], 5),
             parent=[0, 5], child=[0, 5], parent_px={5: 100}, child_px={5: 9}),
        dict(id="two_children", masks=(p2, c2), parent=[0, 7], child=[0, 3],
             parent_px={7: 256}, child_px={3: 18}),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            parent, child = case["masks"]
            new_parent, new_child = U._relabel_parent_with_child_labels(parent.copy(),
                                                                        child.copy())
            assert _uniq(new_parent) == case["parent"]
            assert _uniq(new_child) == case["child"]
            for lbl, n in case["parent_px"].items():
                assert int((new_parent == lbl).sum()) == n
            for lbl, n in case["child_px"].items():
                assert int((new_child == lbl).sum()) == n


def _exclude_inputs(kind):
    cell = np.zeros((32, 32), np.int32)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    if kind == "one_cell_lacks_nucleus":
        cell[2:12, 2:12] = 1
        cell[18:28, 18:28] = 2
        nucleus[4:8, 4:8] = 1
        pathogen[20:22, 20:22] = 1
    elif kind == "no_pathogen":
        cell[2:12, 2:12] = 1
        nucleus[4:8, 4:8] = 1
    elif kind == "one_cell_lacks_pathogen":
        cell[2:12, 2:12] = 1
        cell[18:28, 18:28] = 2
        nucleus[4:8, 4:8] = 1
        nucleus[20:24, 20:24] = 2
        pathogen[9:11, 9:11] = 1
    else:  # "complete_16"
        cell = np.zeros((16, 16), np.int32)
        cell[2:10, 2:10] = 1
        nucleus = np.zeros_like(cell)
        nucleus[3:6, 3:6] = 1
        pathogen = np.zeros_like(cell)
        pathogen[7:9, 7:9] = 1
    cyto = cell.copy()
    cyto[nucleus > 0] = 0
    return cell, nucleus, pathogen, cyto


def test_exclude_objects_keeps_only_complete_cells(subtests):
    """Cells need a nucleus (and a pathogen unless uninfected=True); orphans are wiped."""
    # expected per output: (unique labels, foreground px)
    cases = [
        dict(id="uninfected_drops_nucleus_free", kind="one_cell_lacks_nucleus",
             uninfected=True, want=[([0, 1], 100), ([0, 1], 16), ([0], 0), ([0, 1], 84)]),
        dict(id="infected_requires_pathogen", kind="no_pathogen", uninfected=False,
             want=[([0], 0), ([0], 0), ([0], 0), ([0], 0)]),
        dict(id="infected_keeps_populated", kind="one_cell_lacks_pathogen",
             uninfected=False, want=[([0, 1], 100), ([0, 1], 16), ([0, 1], 4),
                                     ([0, 1], 84)]),
        dict(id="complete_cell_uninfected", kind="complete_16", uninfected=True,
             want=[([0, 1], 64), ([0, 1], 9), ([0, 1], 4), ([0, 1], 55)]),
        dict(id="complete_cell_infected", kind="complete_16", uninfected=False,
             want=[([0, 1], 64), ([0, 1], 9), ([0, 1], 4), ([0, 1], 55)]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            arrays = _exclude_inputs(case["kind"])
            out = U._exclude_objects(*(a.copy() for a in arrays),
                                     uninfected=case["uninfected"])
            assert len(out) == 4
            got = [(_uniq(a), int((a > 0).sum())) for a in out]
            assert got == case["want"]


def test_merge_overlapping_objects_resolves_split_overlaps(subtests):
    """A >=90% dominant overlap trims mask1; an even split merges mask2 labels; one overlap is a no-op."""
    m1 = np.zeros((20, 20), np.int32)
    m1[2:12, 2:12] = 1
    dom2 = np.zeros((20, 20), np.int32)
    dom2[2:12, 2:12] = 1
    dom2[11, 2:7] = 2
    even2 = np.zeros((20, 20), np.int32)
    even2[2:8, 2:12] = 1
    even2[8:12, 2:12] = 2
    s1 = np.zeros((16, 16), np.int32)
    s1[2:10, 2:10] = 1
    s2 = np.zeros((16, 16), np.int32)
    s2[2:10, 2:10] = 3
    cases = [
        dict(id="dominant_trims_mask1", m1=m1, m2=dom2, check="dominant"),
        dict(id="even_split_merges_mask2", m1=m1, m2=even2, check="even"),
        dict(id="single_overlap_noop", m1=s1, m2=s2, check="noop"),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            out1, out2 = U._merge_overlapping_objects(case["m1"].copy(), case["m2"].copy())
            if case["check"] == "dominant":
                assert int((out1 > 0).sum()) == 95
                assert not out1[11, 2:7].any()
                assert np.array_equal(out2, case["m2"])
            elif case["check"] == "even":
                assert _uniq(out2) == [0, 1]
                assert int((out2 == 1).sum()) == 100
                assert np.array_equal(out1, case["m1"])
            else:
                assert np.array_equal(out1, case["m1"])
                assert np.array_equal(out2, case["m2"])


# ---------------------------------------------------------------------------
# _filter_cp_masks
# ---------------------------------------------------------------------------

def _cp_inputs(masks, images=None):
    """Wrap masks/images into the (masks, flows, batch) shape _filter_cp_masks wants."""
    n = len(masks)
    if images is None:
        images = [np.zeros((*masks[0].shape, 2), dtype=np.float32) for _ in range(n)]
    flows = [np.zeros(masks[0].shape, dtype=np.float32) for _ in range(n)]
    return masks, [flows], images


def _size_mask():
    mask = np.zeros((40, 40), np.int32)
    mask[1:3, 1:3] = 1
    mask[6:12, 6:12] = 2
    mask[16:36, 16:36] = 3
    return mask


def _border_mask():
    mask = np.zeros((24, 24), np.int32)
    mask[0:4, 0:4] = 1
    mask[10:16, 10:16] = 2
    return mask


def _single_mask():
    mask = np.zeros((32, 32), np.int32)
    mask[10:20, 10:20] = 4
    return mask


def _staged_mask():
    m = np.zeros((32, 32), np.int32)
    m[2:8, 2:8] = 1
    m[12:22, 12:22] = 2
    m[0:3, 28:31] = 3
    return m


def test_filter_cp_masks_applies_each_enabled_filter(subtests, monkeypatch, capsys):
    """Size, intensity-cluster, border and merge filters each remove exactly their objects."""
    off = dict(filter_size=False, filter_intensity=False, minimum_size=0,
               maximum_size=10**6, remove_border_objects=False, merge=False)
    two = _two_object_mask()
    cases = [
        dict(id="all_disabled", masks=[two, two.copy()], images=None, kw={},
             want=[two, two]),
        dict(id="size_bounds", masks=[_size_mask()], images=None,
             kw=dict(filter_size=True, minimum_size=10, maximum_size=200),
             want=[np.where(_size_mask() == 2, 2, 0)]),
        dict(id="intensity_keeps_bright", masks=[two],
             images=[_intensity_batch_image(two, {1: 5.0, 2: 400.0})],
             kw=dict(filter_intensity=True), want=[np.where(two == 2, 2, 0)]),
        dict(id="intensity_close_clusters_noop", masks=[two],
             images=[_intensity_batch_image(two, {1: 5.0, 2: 5.1})],
             kw=dict(filter_intensity=True), want=[two]),
        dict(id="intensity_single_object", masks=[_single_mask()],
             images=[_intensity_batch_image(_single_mask(), {4: 123.0})],
             kw=dict(filter_intensity=True), want=[_single_mask()]),
        dict(id="border_objects", masks=[_border_mask()], images=None,
             kw=dict(remove_border_objects=True),
             want=[np.where(_border_mask() == 2, 2, 0)]),
        dict(id="plot_every_stage_first_image", masks=[_staged_mask(), _staged_mask()],
             images=[_intensity_batch_image(_staged_mask(), {1: 5.0, 2: 400.0, 3: 350.0})
                     for _ in range(2)],
             kw=dict(filter_size=True, filter_intensity=True, minimum_size=10,
                     remove_border_objects=True, merge=True),
             plot=True, shows=5, uniques=[0, 2],
             prints=["Number of objects before filtration", "after merging adjacent objects",
                     "after size filtration", "potential intensity clustering",
                     "after removing border objects"]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            capsys.readouterr()
            shows = []
            m, f, b = _cp_inputs([x.copy() for x in case["masks"]], case["images"])
            kw = {**off, **case["kw"]}
            with monkeypatch.context() as mp:
                mp.setattr(plt, "show", lambda *a, **k: shows.append(1))
                out = U._filter_cp_masks(m, f, batch=b, plot=case.get("plot", False),
                                         figuresize=2, **kw)
            assert len(out) == len(case["masks"])
            assert len(shows) == case.get("shows", 0)
            if "want" in case:
                for got, want in zip(out, case["want"]):
                    assert np.array_equal(got, want)
            if "uniques" in case:
                for got in out:
                    assert _uniq(got) == case["uniques"]
            text = capsys.readouterr().out
            for frag in case.get("prints", []):
                assert frag in text


# ---------------------------------------------------------------------------
# _object_filter
# ---------------------------------------------------------------------------

def test_object_filter_applies_only_integer_list_bounds(subtests):
    """[lo, hi] integer lists filter area and own-channel mean; floats, tuples and None do not."""
    frame4 = {"cell_area": [5, 50, 500, 5000],
              "cell_channel_1_mean_intensity": [1, 100, 100, 100]}
    frame3 = {"cell_area": [5, 50, 500], "cell_channel_1_mean_intensity": [1, 100, 100]}
    frame_r4 = {"cell_area": [10, 20, 30], "cell_channel_1_mean_intensity": [5, 50, 500]}
    cases = [
        dict(id="size_and_intensity", data=frame4, size=[10, 1000], inten=[10, 1000],
             chans=[0, 1], chan=1, want=[50, 500]),
        dict(id="float_size_bounds", data=frame3, size=[1.5, 2.5], inten=None,
             chans=[0, 1], chan=1, want=[5, 50, 500]),
        dict(id="tuple_ranges", data=frame3, size=(10, 100), inten=(10, 100),
             chans=[0, 1], chan=1, want=[5, 50, 500]),
        dict(id="wide_bounds", data={"cell_area": [50, 200, 500, 1000],
                                     "cell_channel_1_mean_intensity": [10, 100, 1000, 5000]},
             size=[100, 800], inten=[50, 2000], chans=[0, 1], chan=1, want=[200, 500]),
        dict(id="no_ranges_no_intensity_column", data={"cell_area": [1, 2, 3]},
             size=None, inten=None, chans=[0], chan=0, want=[1, 2, 3]),
        dict(id="none_intensity_halves", data=frame_r4, size=None, inten=[None, None],
             chans=[0, 1], chan=1, want=[10, 20, 30]),
        dict(id="intensity_only", data=frame_r4, size=None, inten=[10, 100],
             chans=[0, 1], chan=1, want=[20]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            df = pd.DataFrame(case["data"])
            out = U._object_filter(df, "cell", case["size"], case["inten"],
                                   case["chans"], case["chan"])
            assert out["cell_area"].tolist() == case["want"]


# ---------------------------------------------------------------------------
# _get_regex
# ---------------------------------------------------------------------------

def test_get_regex_builds_named_group_patterns(subtests):
    """Built-in patterns parse real filenames; custom wraps the user pattern; None means tif."""
    cases = [
        dict(id="cellvoyager", mt="cellvoyager", fmt="tif",
             name="plate1_A01_T0001F001L01A01Z01C02.tif", groups=("wellID", "chanID")),
        dict(id="cq1", mt="cq1", fmt="tif", name="W1F001T0001Z01C2.tif",
             groups=("wellID", "chanID")),
        dict(id="auto", mt="auto", fmt="tif", name="plate1_A01_T0001F001L01C02.tif",
             groups=("wellID", "chanID")),
        dict(id="cellvoyager_none_format", mt="cellvoyager", fmt=None,
             name="plate1_A01_T0001F001L01A01Z01C02.tif", groups=("wellID", "chanID")),
        dict(id="custom_well", mt="custom", fmt="tif",
             custom=r"(?P<wellID>[A-H]\d{2})", exact=r"((?P<wellID>[A-H]\d{2})).(?i:tif)",
             name="A01.tif", groups=("wellID",)),
        dict(id="custom_plate", mt="custom", fmt="tif", custom=r"(?P<plateID>.*)",
             exact=r"((?P<plateID>.*)).(?i:tif)", name="p1.tif", groups=("plateID",)),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            kw = {"custom_regex": case["custom"]} if "custom" in case else {}
            rx = U._get_regex(case["mt"], case["fmt"], **kw)
            if "exact" in case:
                assert rx == case["exact"]
            match = re.compile(rx).match(case["name"])
            assert match is not None
            for g in case["groups"]:
                assert g in rx
                assert match.group(g)


# ---------------------------------------------------------------------------
# _run_test_mode
# ---------------------------------------------------------------------------

_SIMPLE_RX = (r"(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_"
              r"(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif")
_ANCHORED_RX = (r"^(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_"
                r"(?P<fieldID>f\d+)_(?P<chanID>C\d)\.tif$")


def _cellvoyager_names():
    return [f"plate1_{w}_T0001F{f}L01A01Z01C{c}.tif"
            for w in ("A01", "A02") for f in ("001", "002") for c in ("01", "02")]


def _cq1_names():
    return [f"W{w}F{f}T0001Z01C{c}.tif"
            for w in (1, 25) for f in ("001", "002") for c in ("1", "2")]


def test_run_test_mode_copies_whole_sampled_image_sets(subtests, tmp_path, capsys):
    """Matched files are grouped by (plate, well, field) and whole sets are copied to test/."""
    cv_rx = U._get_regex("cellvoyager", "tif")
    cases = [
        dict(id="two_random_sets", folder="cellvoyager", files=_cellvoyager_names(),
             rx=cv_rx, kw=dict(timelapse=False, test_images=2, random_test=True),
             n_copied=4, n_sets=2, found=8, used=2),
        dict(id="timelapse_single_set", folder="cellvoyager", files=_cellvoyager_names(),
             rx=cv_rx, kw=dict(timelapse=True, test_images=10), n_copied=2, n_sets=1,
             found=8, used=1),
        dict(id="cq1_no_plate_group", folder="cq1", files=_cq1_names(),
             rx=U._get_regex("cq1", "tif"),
             kw=dict(timelapse=False, test_images=10, random_test=False),
             copied=sorted(_cq1_names()), found=8, used=4),
        dict(id="reads_orig_subfolder", folder="plate",
             orig={f"plate1_A01_T0001F00{i}L01A01Z01C01.tif": b"not-a-real-tif"
                   for i in (1, 2)},
             rx=cv_rx, kw=dict(test_images=10),
             copied=sorted(f"plate1_A01_T0001F00{i}L01A01Z01C01.tif" for i in (1, 2)),
             found=2, used=2),
        dict(id="orig_and_beside", folder="plate",
             orig={f"plate1_A01_T0001F00{i}L01A01Z01C01.tif": b"from-orig" for i in (1, 2)},
             beside={"plate1_B02_T0001F009L01A01Z01C01.tif": b"beside",
                     "plate1_A01_T0001F001L01A01Z01C01.tif": b"stale-duplicate"},
             rx=cv_rx, kw=dict(test_images=10),
             copied=sorted([f"plate1_A01_T0001F00{i}L01A01Z01C01.tif" for i in (1, 2)]
                           + ["plate1_B02_T0001F009L01A01Z01C01.tif"]),
             contents={"plate1_A01_T0001F001L01A01Z01C01.tif": b"from-orig",
                       "plate1_B02_T0001F009L01A01Z01C01.tif": b"beside"},
             found=3, used=3),
        dict(id="no_matching_files", folder="empty_plate", files=["README.txt"], rx=cv_rx,
             kw={}, copied=[], found=0, used=0),
        dict(id="anchored_regex_filters", folder="plate",
             files=["plate1_A1_f1_C1.tif", "plate1_A1_f1_C2.tif", "notes.txt",
                    "plate1_A1.tif"],
             rx=_ANCHORED_RX, kw=dict(test_images=1, random_test=False),
             copied=["plate1_A1_f1_C1.tif", "plate1_A1_f1_C2.tif"], found=2, used=1),
        dict(id="non_matching_tiffs_never_listed", folder="plate1",
             files=["plate1_A01_1_1.tif", "overview.tif", "flatfield_correction.tif"],
             rx=_SIMPLE_RX, kw={}, copied=["plate1_A01_1_1.tif"], found=1, used=1),
        dict(id="grouped_by_set", folder="plate1",
             files=["plate1_A01_1_1.tif", "plate1_A01_1_2.tif", "plate1_A02_1_1.tif"],
             rx=_SIMPLE_RX, kw={},
             copied=["plate1_A01_1_1.tif", "plate1_A01_1_2.tif", "plate1_A02_1_1.tif"],
             found=3, used=2),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            capsys.readouterr()
            src = _case_dir(tmp_path, case["id"]) / case["folder"]
            src.mkdir()
            for name in case.get("files", []):
                (src / name).write_bytes(b"II*\x00")
            if "orig" in case:
                (src / "orig").mkdir()
                for name, data in case["orig"].items():
                    (src / "orig" / name).write_bytes(data)
            for name, data in case.get("beside", {}).items():
                (src / name).write_bytes(data)
            out = U._run_test_mode(str(src), case["rx"], **case["kw"])
            assert out == os.path.join(str(src), "test")
            assert os.path.isdir(out)
            copied = sorted(os.listdir(out))
            if "copied" in case:
                assert copied == case["copied"]
            else:
                assert len(copied) == case["n_copied"]
                rx = re.compile(case["rx"])
                sets = {(m.group("plateID"), m.group("wellID"), m.group("fieldID"))
                        for m in (rx.match(f) for f in copied)}
                assert len(sets) == case["n_sets"]
                # whole sets: every channel of a sampled field is copied
                assert len(copied) == 2 * case["n_sets"]
            for name, data in case.get("contents", {}).items():
                with open(os.path.join(out, name), "rb") as handle:
                    assert handle.read() == data
            printed = capsys.readouterr().out
            assert f"Found {case['found']} files" in printed
            assert f"Using {case['used']} random image set(s)" in printed


# ---------------------------------------------------------------------------
# _choose_model
# ---------------------------------------------------------------------------
#
# Cellpose 4 ships exactly one model (cpsam). _choose_model maps every legacy
# name to cpsam explicitly and says so, loads fine-tuned checkpoints by path,
# and refuses (rather than silently substituting stock weights) when an
# explicit name or path cannot be resolved.

class _FakeCellposeModel:
    """Records the kwargs Cellpose would have been constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Swap the Cellpose constructor for a recording double (no weights, no GPU)."""
    U.reset_cellpose_model_reports()
    monkeypatch.setattr(U, "cp_models", SimpleNamespace(CellposeModel=_FakeCellposeModel))
    yield _FakeCellposeModel
    U.reset_cellpose_model_reports()


def test_choose_model_resolves_names_and_checkpoints_to_weights(
        subtests, fake_cellpose, tmp_path, capsys):
    """Legacy/removed/None names load cpsam (announced); existing checkpoints load by path."""
    cases = []
    for name in ["cyto", "cyto2", "cyto3", "cyto_2", "cyto_3", "nuclei", "nucleus"]:
        cases.append(dict(id=f"legacy_{name}", name=name, kw={}, want="cpsam",
                          says=["cpsam"], no_legacy_kwargs=True))
    for name in ["toxo_pv_lumen", "toxo_cyto"]:
        cases.append(dict(id=f"removed_{name}", name=name,
                          kw=dict(object_type="pathogen", object_settings={"diameter": 30}),
                          want="cpsam", says=[name, "cpsam"]))
    for restore in ["denoise", "deblur", "upsample"]:
        cases.append(dict(id=f"restore_{restore}", name="cyto",
                          kw=dict(restore_type=restore), want="cpsam",
                          says=[restore, "not supported"]))
    cases += [
        dict(id="sam", name="sam", kw={}, want="cpsam"),
        dict(id="none", name=None, kw={}, want="cpsam"),
        dict(id="cpsam", name="cpsam", kw={}, want="cpsam"),
        dict(id="device_cuda1", name="cyto", kw={}, device="cuda:1", want="cpsam"),
        dict(id="no_object_settings", name="cyto",
             kw=dict(object_type="pathogen", object_settings=None), want="cpsam"),
        dict(id="fine_tuned", ckpt="my_cells_model", kw=dict(object_type="cell"),
             says_path=True),
        dict(id="ckpt_pth", ckpt="cp_model.pth", kw={}),
        dict(id="ckpt_pt", ckpt="cp_model.pt", kw={}),
        dict(id="ckpt_extensionless", ckpt="cp_model", kw={}),
        dict(id="ckpt_whitespace", ckpt="model", pad="  ", kw={}),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            U.reset_cellpose_model_reports()
            capsys.readouterr()
            device = case.get("device", "cpu")
            if "ckpt" in case:
                ckpt = _case_dir(tmp_path, case["id"]) / case["ckpt"]
                ckpt.write_bytes(b"not really weights, but it exists")
                pad = case.get("pad", "")
                model = U._choose_model(f"{pad}{ckpt}{pad}", device=device, **case["kw"])
                want = str(ckpt)
            else:
                model = U._choose_model(case["name"], device=device, **case["kw"])
                want = case["want"]
            assert isinstance(model, _FakeCellposeModel)
            assert model.kwargs["pretrained_model"] == want
            assert model.kwargs["device"] == device
            out = capsys.readouterr().out
            if case.get("no_legacy_kwargs"):
                assert "model_type" not in model.kwargs
                assert "diam_mean" not in model.kwargs
            for frag in case.get("says", []):
                assert frag in out
            if case.get("says_path"):
                assert want in out


def test_choose_model_refuses_to_substitute_stock_weights(
        subtests, fake_cellpose, tmp_path, monkeypatch):
    """Unknown names and missing checkpoint paths raise ModelZooError before any construction."""
    from spacr import model_zoo as zoo

    cases = [
        dict(id="unknown_name_empty_catalogue", name="no_such_model", empty_zoo=True,
             frags=["no_such_model"]),
        dict(id="missing_checkpoint", path=("gone", "model.pth"),
             kw=dict(object_type="nucleus"),
             frags=["{path}", "nucleus", "stock weights were not substituted"]),
        dict(id="missing_extensionless_path", path=("models", "trained_on_hela"),
             frags=["{path}"]),
        dict(id="typo_word_not_a_path", name="cytoo",
             frags=["cytoo", "stock weights were not substituted", "model_name"]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            U.reset_cellpose_model_reports()
            name = case.get("name")
            if "path" in case:
                name = str(_case_dir(tmp_path, case["id"]).joinpath(*case["path"]))
            with monkeypatch.context() as mp:
                if case.get("empty_zoo"):
                    mp.setattr(zoo, "catalogue", lambda **kwargs: [])
                mp.setattr(U.cp_models, "CellposeModel",
                           lambda **kwargs: pytest.fail("refused model reached constructor"))
                with pytest.raises(ModelZooError) as error:
                    U._choose_model(name, device="cpu", **case.get("kw", {}))
            msg = str(error.value)
            for frag in case["frags"]:
                assert frag.format(path=name) in msg
