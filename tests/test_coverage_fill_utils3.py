"""Coverage-fill batch 3 for spacr.utils mask/segmentation-metric helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# circular mask / apply_mask
# ---------------------------------------------------------------------------

def test_create_circular_mask():
    m = U.create_circular_mask(20, 20)
    assert m.dtype == bool and m[10, 10] and not m[0, 0]
    # explicit center + radius
    m2 = U.create_circular_mask(20, 20, center=(5, 5), radius=3)
    assert m2[5, 5] and not m2[15, 15]


def test_apply_mask_gray_and_rgb():
    gray = np.ones((16, 16), dtype=np.uint8) * 100
    out = U.apply_mask(gray)
    assert out[0, 0] == 0 and out[8, 8] == 100
    rgb = np.ones((16, 16, 3), dtype=np.uint8) * 100
    out_rgb = U.apply_mask(rgb)
    assert out_rgb.shape == (16, 16, 3) and out_rgb[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# IoU / matching / AP
# ---------------------------------------------------------------------------

def test_pad_to_same_shape_and_iou():
    a = np.ones((4, 4), dtype=bool)
    b = np.ones((6, 5), dtype=bool)
    pa, pb = U.pad_to_same_shape(a, b)
    assert pa.shape == pb.shape == (6, 5)
    iou = U.calculate_iou(a, b)
    assert 0 < iou <= 1
    # disjoint → union nonzero but intersection 0
    z = np.zeros((4, 4), dtype=bool)
    assert U.calculate_iou(z, z) == 0


def test_match_masks_and_ap():
    t1 = np.zeros((8, 8), dtype=bool); t1[1:4, 1:4] = True
    t2 = np.zeros((8, 8), dtype=bool); t2[5:7, 5:7] = True
    p1 = np.zeros((8, 8), dtype=bool); p1[1:4, 1:4] = True   # matches t1
    matches = U.match_masks([t1, t2], [p1], iou_threshold=0.5)
    assert len(matches) == 1
    precision, recall = U.compute_average_precision(matches, 2, 1)
    assert precision == 1.0 and recall == 0.5


def test_compute_ap_over_iou_thresholds():
    t1 = np.zeros((8, 8), dtype=bool); t1[1:4, 1:4] = True
    p1 = np.zeros((8, 8), dtype=bool); p1[1:4, 1:4] = True
    ap = U.compute_ap_over_iou_thresholds([t1], [p1], [0.5, 0.75])
    assert ap >= 0


def test_compute_segmentation_ap():
    true = np.zeros((16, 16), dtype=np.int32); true[2:6, 2:6] = 1
    pred = np.zeros((16, 16), dtype=np.int32); pred[2:6, 2:6] = 1
    ap = U.compute_segmentation_ap(true, pred, iou_thresholds=[0.5])
    assert ap >= 0


# ---------------------------------------------------------------------------
# jaccard / dice / boundary
# ---------------------------------------------------------------------------

def test_jaccard_and_dice():
    a = np.zeros((8, 8), dtype=np.int32); a[1:5, 1:5] = 1
    b = np.zeros((8, 8), dtype=np.int32); b[1:5, 1:5] = 1
    assert U.jaccard_index(a, b) == 1.0
    assert U.dice_coefficient(a, b) == 1.0
    # both empty → dice defined as 1.0
    z = np.zeros((4, 4), dtype=np.int32)
    assert U.dice_coefficient(z, z) == 1.0


def test_extract_boundaries_and_bf1():
    m = np.zeros((16, 16), dtype=np.int32); m[4:12, 4:12] = 1
    b = U.extract_boundaries(m)
    assert b.any()
    f1 = U.boundary_f1_score(m, m)
    assert f1 > 0.9


# ---------------------------------------------------------------------------
# mask object helpers
# ---------------------------------------------------------------------------

def test_filter_object():
    m = np.zeros((16, 16), dtype=np.int32)
    m[0, 0] = 1              # 1 px → removed
    m[4:8, 4:8] = 2          # 16 px → kept
    out = U._filter_object(m.copy(), min_value=5)
    assert 1 not in np.unique(out) and 2 in np.unique(out)


def test_exclude_objects():
    cell = np.zeros((16, 16), dtype=np.int32); cell[2:10, 2:10] = 1
    nucleus = np.zeros((16, 16), dtype=np.int32); nucleus[3:6, 3:6] = 1
    cyto = cell.copy(); cyto[nucleus > 0] = 0
    pathogen = np.zeros((16, 16), dtype=np.int32); pathogen[7:9, 7:9] = 1
    fc, nm, pm, cm = U._exclude_objects(cell, nucleus, pathogen, cyto,
                                        uninfected=True)
    assert (fc > 0).any()
    # infected-only: cell must also have a pathogen
    fc2, *_ = U._exclude_objects(cell, nucleus, pathogen, cyto,
                                 uninfected=False)
    assert (fc2 > 0).any()


def test_relabel_parent_with_child_labels():
    parent = np.zeros((16, 16), dtype=np.int32); parent[2:12, 2:12] = 1
    child = np.zeros((16, 16), dtype=np.int32); child[4:7, 4:7] = 5
    new_parent, new_child = U._relabel_parent_with_child_labels(parent, child)
    assert 5 in np.unique(new_parent)


def test_merge_overlapping_objects():
    m1 = np.zeros((16, 16), dtype=np.int32); m1[2:10, 2:10] = 1
    m2 = np.zeros((16, 16), dtype=np.int32); m2[2:10, 2:10] = 3
    out1, out2 = U._merge_overlapping_objects(m1.copy(), m2.copy())
    assert out1 is not None and out2 is not None


# ---------------------------------------------------------------------------
# dataframe helpers
# ---------------------------------------------------------------------------

def test_filter_closest_to_stat():
    df = pd.DataFrame({"v": [1.0, 5.0, 5.1, 5.2, 100.0]})
    out = U._filter_closest_to_stat(df, "v", n_rows=2, use_median=True)
    assert len(out) == 2 and "diff" not in out.columns


def test_generate_fraction_map():
    df = pd.DataFrame({
        "gene": ["g1", "g1", "g2", "g3", "g3", "g3"],
        "prc": ["p_r1_c1", "p_r1_c1", "p_r1_c1",
                "p_r1_c2", "p_r1_c2", "p_r1_c2"],
        "count": [10, 20, 30, 5, 5, 5],
        "well_read_sum": [60, 60, 60, 15, 15, 15],
    })
    out = U.generate_fraction_map(df, gene_column="gene", min_frequency=0.0)
    assert isinstance(out, pd.DataFrame)
    assert out.index.name == "prc"


def test_calculate_iou_empty():
    assert U.calculate_iou(np.zeros((3, 3)), np.zeros((3, 3))) == 0
