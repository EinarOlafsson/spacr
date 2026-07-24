"""Coverage-fill for spacr.timelapse pure-logic tracking helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import timelapse as TL


# ---------------------------------------------------------------------------
# _sort_key
# ---------------------------------------------------------------------------

def test_sort_key():
    assert TL._sort_key("/x/1_A02_3_5.npy") == ("1", "A02", "3", 5)
    # non-matching → default lowest key
    assert TL._sort_key("garbage.npy") == ("", "", "", 0)


# ---------------------------------------------------------------------------
# _prepare_for_tracking
# ---------------------------------------------------------------------------

def test_prepare_for_tracking():
    frame0 = np.zeros((16, 16), dtype=np.int32); frame0[2:6, 2:6] = 1
    frame1 = np.zeros((16, 16), dtype=np.int32); frame1[3:7, 3:7] = 1
    df = TL._prepare_for_tracking(np.stack([frame0, frame1]))
    assert {"frame", "y", "x", "mass", "original_label"} <= set(df.columns)
    assert set(df["frame"]) == {0, 1}


# ---------------------------------------------------------------------------
# link_by_iou / _track_by_iou / _filter_short_tracks
# ---------------------------------------------------------------------------

def _moving_masks(n=4, size=24):
    masks = []
    for t in range(n):
        m = np.zeros((size, size), dtype=np.int32)
        m[2 + t:6 + t, 2:6] = 1        # object 1 drifts down
        m[14:18, 14 - t:18 - t] = 2    # object 2 drifts left
        masks.append(m)
    return np.stack(masks)


def test_link_by_iou():
    m0 = np.zeros((16, 16), dtype=np.int32); m0[2:8, 2:8] = 1
    m1 = np.zeros((16, 16), dtype=np.int32); m1[3:9, 3:9] = 1
    matches = TL.link_by_iou(m0, m1, iou_threshold=0.1)
    assert matches == [(1, 1)]


def test_track_by_iou_and_filter():
    masks = _moving_masks()
    df = TL._track_by_iou(masks, iou_threshold=0.1)
    assert {"frame", "original_label", "track_id"} <= set(df.columns)
    filt = TL._filter_short_tracks(df, min_length=3)
    assert isinstance(filt, pd.DataFrame)


# ---------------------------------------------------------------------------
# _remove_objects_from_first_frame
# ---------------------------------------------------------------------------

def test_remove_objects_from_first_frame():
    masks = _moving_masks()
    n_before = len(np.unique(masks[0])) - 1
    out = TL._remove_objects_from_first_frame(masks.copy(), percentage=50)
    n_after = len(np.unique(out[0])) - 1
    assert n_after < n_before


# ---------------------------------------------------------------------------
# exponential_decay
# ---------------------------------------------------------------------------

def test_exponential_decay():
    assert TL.exponential_decay(0, 2.0, 1.0, 0.5) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# _infer_plate_well_meta_tag
# ---------------------------------------------------------------------------

def test_infer_plate_well_meta_tag():
    single = pd.DataFrame({"plateID": ["plate1"], "wellID": ["A02"]})
    assert TL._infer_plate_well_meta_tag(single) == "plate1_A02"
    multi_w = pd.DataFrame({"plateID": ["p1", "p1"], "wellID": ["A01", "A02"]})
    assert TL._infer_plate_well_meta_tag(multi_w) == "p1_MULTI_WELLS"
    multi_p = pd.DataFrame({"plateID": ["p1", "p2"], "wellID": ["A01", "A01"]})
    assert TL._infer_plate_well_meta_tag(multi_p) == "MULTI_PLATES_A01"
    multi_both = pd.DataFrame({"plateID": ["p1", "p2"], "wellID": ["A01", "A02"]})
    assert TL._infer_plate_well_meta_tag(multi_both) == "MULTI_PLATES_MULTI_WELLS"


# ---------------------------------------------------------------------------
# _parse_merged_filename
# ---------------------------------------------------------------------------

def test_parse_merged_filename():
    meta = TL._parse_merged_filename("plate1_B03_2_t005.npy")
    assert meta["plateID"] == "plate1" and meta["wellID"] == "B03"
    assert meta["rowID"] == "B" and meta["columnID"] == 3
    assert meta["fieldID"] == "2" and meta["timeID"] == 5
    assert meta["prcft"] == "plate1_B03_2_5"


def test_parse_merged_filename_minimal():
    meta = TL._parse_merged_filename("plateX.npy")
    assert meta["plateID"] == "plateX" and meta["timeID"] == 0
    assert meta["fieldID"] == "1"


# ---------------------------------------------------------------------------
# _relabel_masks_based_on_tracks
# ---------------------------------------------------------------------------

def test_relabel_masks_based_on_tracks():
    masks = _moving_masks()
    tracks = TL._track_by_iou(masks, iou_threshold=0.1)
    relabeled = TL._relabel_masks_based_on_tracks(masks, tracks)
    assert relabeled.shape == masks.shape and relabeled.dtype == masks.dtype


# ---------------------------------------------------------------------------
# _reorient_merged_array
# ---------------------------------------------------------------------------

def test_reorient_merged_array_planes_first():
    arr = np.zeros((4, 16, 16), dtype=np.float32)   # already planes-first
    out, planes, H, W = TL._reorient_merged_array(arr, n_channels=3)
    assert planes == 4 and (H, W) == (16, 16)


def test_reorient_merged_array_planes_last():
    arr = np.zeros((16, 16, 4), dtype=np.float32)   # planes last → moved
    out, planes, H, W = TL._reorient_merged_array(arr, n_channels=3)
    assert out.shape[0] == 4 and (H, W) == (16, 16)


def test_reorient_merged_array_fallback():
    # no axis in [n, n+max] → smallest axis chosen
    arr = np.zeros((2, 16, 16), dtype=np.float32)
    out, planes, H, W = TL._reorient_merged_array(arr, n_channels=8)
    assert planes == 2


def test_reorient_merged_array_bad_ndim():
    with pytest.raises(ValueError):
        TL._reorient_merged_array(np.zeros((16, 16)), n_channels=3)


# ---------------------------------------------------------------------------
# preprocess_pathogen_data
# ---------------------------------------------------------------------------

def test_preprocess_pathogen_data():
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n,
        "column_name": ["c1"] * n, "fieldID": ["f1"] * n,
        "timeid": rng.integers(0, 3, n),
        "pathogen_cell_id": rng.integers(1, 4, n),
        "object_label": range(n),
        "area": rng.uniform(10, 100, n),
    })
    out = TL.preprocess_pathogen_data(df)
    assert "parasite_count" in out.columns
    assert "object_label" in out.columns   # renamed from pathogen_cell_id


# ---------------------------------------------------------------------------
# save_figure / save_results_dataframe
# ---------------------------------------------------------------------------

def test_save_figure_and_results(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
    src = str(tmp_path / "run" / "data")
    (tmp_path / "run").mkdir()
    TL.save_figure(fig, src, 1)
    plt.close(fig)
    assert (tmp_path / "run" / "results" / "figure_1.pdf").exists()
    TL.save_results_dataframe(pd.DataFrame({"a": [1, 2]}), src, "res")
    assert (tmp_path / "run" / "results" / "res.csv").exists()
