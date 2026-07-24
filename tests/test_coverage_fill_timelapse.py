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
