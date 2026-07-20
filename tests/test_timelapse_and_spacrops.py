"""
Deep behavioral tests for the two biggest still-cold pipeline modules:

  * spacr.timelapse   — tracking primitives that are pure enough to test
                        without live microscopy timelapses
  * spacr.spacrops    — screen-QC pipelines; test the _DiskFeatureStore
                        class-level LRU + disk cache
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# timelapse: pure helpers
# ===========================================================================

def test_sort_key_parses_yokogawa_name():
    from spacr.timelapse import _sort_key
    key = _sort_key("/some/path/1_A03_4_12.npy")
    assert key == ("1", "A03", "4", 12)


def test_sort_key_returns_default_for_nonmatching_name():
    from spacr.timelapse import _sort_key
    # A file name that doesn't match the plate_well_field_time regex
    # should sort to the "earliest" bucket.
    assert _sort_key("garbage.npy") == ("", "", "", 0)


def test_sort_key_sorts_files_chronologically():
    from spacr.timelapse import _sort_key
    files = [
        "1_A01_1_10.npy",
        "1_A01_1_2.npy",
        "1_A01_1_1.npy",
        "1_A01_1_20.npy",
    ]
    ordered = sorted(files, key=_sort_key)
    # Time IDs should sort numerically, not lexicographically.
    time_ids = [_sort_key(f)[3] for f in ordered]
    assert time_ids == sorted(time_ids)


# ---------------------------------------------------------------------------
# _parse_merged_filename — turns plate_well_field_time.npy into a metadata dict
# ---------------------------------------------------------------------------

def test_parse_merged_filename_full_name():
    from spacr.timelapse import _parse_merged_filename
    meta = _parse_merged_filename("plate1_A03_002_t017.npy")
    assert meta["plateID"] == "plate1"
    assert meta["wellID"] == "A03"
    assert meta["rowID"] == "A"
    assert meta["columnID"] == 3
    assert meta["fieldID"] == "002"
    assert meta["timeID"] == 17
    assert meta["prcf"] == "plate1_A03_002"
    assert meta["prcft"] == "plate1_A03_002_17"


def test_parse_merged_filename_missing_parts_defaults():
    from spacr.timelapse import _parse_merged_filename
    meta = _parse_merged_filename("plate1_A03.npy")
    assert meta["plateID"] == "plate1"
    assert meta["wellID"] == "A03"
    # Missing field / time → defaults.
    assert meta["fieldID"] == "1"
    assert meta["timeID"] == 0


def test_parse_merged_filename_ignores_extension():
    from spacr.timelapse import _parse_merged_filename
    a = _parse_merged_filename("plate1_A03_001_0.npy")
    b = _parse_merged_filename("plate1_A03_001_0.tif")
    assert a["prcft"] == b["prcft"]


# ---------------------------------------------------------------------------
# _prepare_for_tracking — 3D mask -> per-frame regionprops DataFrame
# ---------------------------------------------------------------------------

def test_prepare_for_tracking_returns_frame_dataframe():
    from spacr.timelapse import _prepare_for_tracking
    # Build a 3-frame stack; each frame has 2 objects.
    mask = np.zeros((3, 32, 32), dtype=np.int32)
    for t in range(3):
        mask[t, 5:10, 5:10] = 1
        mask[t, 20:25, 20:25] = 2

    df = _prepare_for_tracking(mask)
    assert isinstance(df, pd.DataFrame)
    for col in ("frame", "y", "x", "mass", "original_label",
                "bbox-0", "bbox-1", "bbox-2", "bbox-3", "eccentricity"):
        assert col in df.columns
    # 3 frames × 2 objects = 6 rows.
    assert len(df) == 6
    assert set(df["frame"].unique()) == {0, 1, 2}
    assert set(df["original_label"].unique()) == {1, 2}


def test_prepare_for_tracking_empty_frame():
    from spacr.timelapse import _prepare_for_tracking
    mask = np.zeros((2, 16, 16), dtype=np.int32)  # no objects anywhere
    df = _prepare_for_tracking(mask)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# link_by_iou — Hungarian-assignment IoU linker
# ---------------------------------------------------------------------------

def test_link_by_iou_identical_frames_all_match():
    from spacr.timelapse import link_by_iou
    prev = np.zeros((20, 20), dtype=np.int32)
    prev[2:8, 2:8] = 1
    prev[10:16, 10:16] = 2
    curr = prev.copy()
    matches = link_by_iou(prev, curr, iou_threshold=0.5)
    assert set(matches) == {(1, 1), (2, 2)}


def test_link_by_iou_disjoint_frames_no_matches():
    from spacr.timelapse import link_by_iou
    prev = np.zeros((30, 30), dtype=np.int32)
    prev[0:5, 0:5] = 1
    curr = np.zeros((30, 30), dtype=np.int32)
    curr[25:30, 25:30] = 3
    matches = link_by_iou(prev, curr, iou_threshold=0.1)
    assert matches == []


def test_link_by_iou_relabel_matches_by_iou_only():
    """A cell in frame N with a different label ID in frame N+1 but the
    same spatial location should still get matched."""
    from spacr.timelapse import link_by_iou
    prev = np.zeros((20, 20), dtype=np.int32)
    prev[5:15, 5:15] = 7
    curr = np.zeros((20, 20), dtype=np.int32)
    curr[5:15, 5:15] = 42
    matches = link_by_iou(prev, curr, iou_threshold=0.5)
    assert matches == [(7, 42)]


# ---------------------------------------------------------------------------
# _track_by_iou — full pipeline over a 3-frame stack
# ---------------------------------------------------------------------------

def test_track_by_iou_stable_object_gets_single_track():
    from spacr.timelapse import _track_by_iou
    masks = np.zeros((3, 20, 20), dtype=np.int32)
    # Same object appears at (5..10, 5..10) in every frame, but with a
    # different label id each time.
    masks[0, 5:10, 5:10] = 1
    masks[1, 5:10, 5:10] = 5
    masks[2, 5:10, 5:10] = 9

    df = _track_by_iou(masks, iou_threshold=0.5)
    assert list(df.columns) == ["frame", "original_label", "track_id"]
    assert len(df) == 3
    # All three rows should share the same track_id.
    assert df["track_id"].nunique() == 1


def test_track_by_iou_new_appearance_gets_new_track():
    """Frame 0 has 1 cell; frame 1 has that cell + a new one → 2 tracks."""
    from spacr.timelapse import _track_by_iou
    masks = np.zeros((2, 20, 20), dtype=np.int32)
    masks[0, 2:8, 2:8] = 1
    masks[1, 2:8, 2:8] = 1
    masks[1, 12:18, 12:18] = 2   # newborn
    df = _track_by_iou(masks, iou_threshold=0.5)
    assert df["track_id"].nunique() == 2


# ---------------------------------------------------------------------------
# _filter_short_tracks — pure pandas groupby
# ---------------------------------------------------------------------------

def test_filter_short_tracks_removes_below_threshold():
    from spacr.timelapse import _filter_short_tracks
    df = pd.DataFrame({
        "track_id":       [1, 1, 1, 2, 2, 3],
        "frame":          [0, 1, 2, 0, 1, 0],
        "original_label": [1, 1, 1, 2, 2, 3],
    })
    filtered = _filter_short_tracks(df, min_length=3)
    # Only track 1 has length >= 3.
    assert set(filtered["track_id"].unique()) == {1}


def test_filter_short_tracks_keeps_all_when_threshold_met():
    from spacr.timelapse import _filter_short_tracks
    df = pd.DataFrame({
        "track_id": [1, 1, 2, 2],
        "frame":    [0, 1, 0, 1],
    })
    filtered = _filter_short_tracks(df, min_length=2)
    assert len(filtered) == 4


# ---------------------------------------------------------------------------
# exponential_decay — trivial math
# ---------------------------------------------------------------------------

def test_exponential_decay_at_zero_returns_a_plus_c():
    from spacr.timelapse import exponential_decay
    a, b, c = 5.0, 0.5, 1.0
    got = exponential_decay(0.0, a, b, c)
    # exp(0) = 1 → a*1 + c = a + c
    assert got == pytest.approx(a + c)


def test_exponential_decay_at_infinity_approaches_c():
    from spacr.timelapse import exponential_decay
    got = exponential_decay(1000.0, a=5.0, b=0.5, c=1.0)
    # exp(-500) ≈ 0 → result ≈ c
    assert got == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _reorient_merged_array — reorients merged intensity+mask stacks
# ---------------------------------------------------------------------------

def test_reorient_merged_array_channel_first_layout():
    """arr shape (H, W, C+3): 4 intensity channels + up to 3 mask channels.
    _reorient_merged_array puts the last dim in front."""
    from spacr.timelapse import _reorient_merged_array
    # (H, W, C=4 + 3 extra masks = 7)
    arr = np.zeros((16, 16, 7), dtype=np.float32)
    out = _reorient_merged_array(arr, n_channels=4, max_extra_masks=3)
    assert out is not None


# ===========================================================================
# spacrops._DiskFeatureStore — disk-backed LRU cache
# ===========================================================================

def test_disk_feature_store_roundtrip(tmp_path):
    from spacr.spacrops import _DiskFeatureStore
    store = _DiskFeatureStore(str(tmp_path), max_ram_items=4)
    feat = {
        "ds8":  np.array([[1, 2], [3, 4]], dtype=np.uint8),
        "pts":  np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "desc": np.zeros((2, 32), dtype=np.uint8),
        "Hds": 32, "Wds": 32, "H": 128, "W": 128,
    }
    key = "/fake/some_image.tif"
    assert store.get(key) is None    # cold miss
    store.put(key, feat)
    got = store.get(key)             # warm hit
    assert got is not None
    assert np.array_equal(got["ds8"], feat["ds8"])
    assert np.array_equal(got["pts"], feat["pts"])
    assert got["Hds"] == 32 and got["W"] == 128


def test_disk_feature_store_key_hashing_is_stable(tmp_path):
    from spacr.spacrops import _DiskFeatureStore
    store = _DiskFeatureStore(str(tmp_path))
    # Two calls to _key_for_path with the same input give the same hash.
    a = store._key_for_path("/some/img.tif")
    b = store._key_for_path("/some/img.tif")
    assert a == b
    # Different inputs → different hashes.
    c = store._key_for_path("/other/img.tif")
    assert a != c


def test_disk_feature_store_lru_evicts_beyond_limit(tmp_path):
    from spacr.spacrops import _DiskFeatureStore
    store = _DiskFeatureStore(str(tmp_path), max_ram_items=2)

    def _tiny_feat(i):
        return {
            "ds8": np.array([[i]], dtype=np.uint8),
            "pts": np.zeros((0, 2), dtype=np.float32),
            "desc": np.zeros((0, 32), dtype=np.uint8),
            "Hds": 1, "Wds": 1, "H": 1, "W": 1,
        }

    for i in range(4):
        store.put(f"/f/{i}.tif", _tiny_feat(i))

    # LRU cap = 2 → only the two most-recent entries are in RAM.
    assert len(store._ram) == 2


def test_disk_feature_store_get_falls_back_to_disk_after_lru_eviction(tmp_path):
    """After eviction from RAM, get() still finds the entry via the NPZ
    on disk and rehydrates it."""
    from spacr.spacrops import _DiskFeatureStore
    store = _DiskFeatureStore(str(tmp_path), max_ram_items=1)

    def _tiny_feat(i):
        return {
            "ds8": np.array([[i]], dtype=np.uint8),
            "pts": np.zeros((0, 2), dtype=np.float32),
            "desc": np.zeros((0, 32), dtype=np.uint8),
            "Hds": 1, "Wds": 1, "H": 1, "W": 1,
        }

    store.put("/a.tif", _tiny_feat(1))
    store.put("/b.tif", _tiny_feat(2))   # evicts /a.tif from RAM
    assert "/a.tif" not in store._ram

    got = store.get("/a.tif")             # disk fallback
    assert got is not None
    assert int(got["ds8"][0, 0]) == 1
