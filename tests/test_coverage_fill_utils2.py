"""Coverage-fill batch 2 for spacr.utils pure-logic helpers (no GPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------

def test_union_find_root_and_merge():
    parent = list(range(5))
    U._union_find_merge(parent, 1, 2)
    U._union_find_merge(parent, 2, 3)
    assert U._union_find_root(parent, 3) == U._union_find_root(parent, 1)
    # merging already-joined nodes is a no-op
    U._union_find_merge(parent, 1, 3)
    assert U._union_find_root(parent, 3) == 1


# ---------------------------------------------------------------------------
# _convert_cq1_well_id
# ---------------------------------------------------------------------------

def test_convert_cq1_well_id():
    assert U._convert_cq1_well_id(1) == "A01"
    assert U._convert_cq1_well_id(25) == "B01"   # 24 cols per row
    assert U._convert_cq1_well_id(24) == "A24"


# ---------------------------------------------------------------------------
# _map_values
# ---------------------------------------------------------------------------

def test_map_values_with_locs():
    row = {"rowID": "r2"}
    out = U._map_values(row, ["ctrl", "trt"], [["r1"], ["r2"]])
    assert out == "trt"


def test_map_values_no_locs():
    assert U._map_values({"rowID": "r1"}, ["only"], None) == "only"
    assert U._map_values({"rowID": "r1"}, [], None) is None


# ---------------------------------------------------------------------------
# _safe_int_convert
# ---------------------------------------------------------------------------

def test_safe_int_convert():
    assert U._safe_int_convert("42") == 42
    assert U._safe_int_convert("nope", default=7) == 7


# ---------------------------------------------------------------------------
# _map_wells / _map_wells_png
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _get_diam / _get_object_settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obj,expected", [
    ("cell", 2 * 20 + 80),
    ("cell_large", 2 * 20 + 120),
    ("nucleus", int(0.75 * 20 + 45)),
    ("pathogen", 20),
])
def test_get_diam(obj, expected):
    assert U._get_diam(20, obj) == expected


def test_get_diam_unsupported_raises():
    with pytest.raises(ValueError):
        U._get_diam(20, "bogus")


# ---------------------------------------------------------------------------
# normalize_to_dtype
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _filter_objects
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# annotate_conditions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _split_data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# boundary / perimeter helpers
# ---------------------------------------------------------------------------

def _adjacent_labels(size=16):
    m = np.zeros((size, size), dtype=np.int32)
    m[2:12, 2:8] = 1     # left block
    m[2:12, 8:14] = 2    # right block — touches label 1 along column 8
    return m


def test_compute_label_perimeters():
    m = _adjacent_labels()
    per = U._compute_label_perimeters(m)
    assert 1 in per and 2 in per and per[1] > 0


def test_compute_shared_boundaries():
    m = _adjacent_labels()
    shared = U._compute_shared_boundaries(m)
    assert (1, 2) in shared and shared[(1, 2)] > 0


def test_merge_by_perimeter():
    m = _adjacent_labels()
    parent = {1: 1, 2: 2}
    U._merge_by_perimeter(m, perimeter_fraction=0.01, parent=parent)
    # low threshold → the two touching labels get merged
    assert U._union_find_root(parent, 1) == U._union_find_root(parent, 2)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def test_gen_rgb_image():
    img = np.random.default_rng(0).random((16, 16, 4)).astype(np.float32)
    rgb = U._gen_rgb_image(img, channels=[0, 1, 2])
    assert rgb.shape == (16, 16, 3)
    # out-of-range channel index is skipped gracefully
    rgb2 = U._gen_rgb_image(img, channels=[0, 9, 2])
    assert rgb2.shape == (16, 16, 3)


def test_find_bounding_box():
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[10:20, 12:18] = 7
    out = U._find_bounding_box(mask, 7, buffer=2)
    assert (out == 7).any()
    ys, xs = np.where(out == 7)
    assert ys.min() >= 8 and xs.min() >= 10   # buffered box


def test_smooth_hull_lines():
    rng = np.random.default_rng(1)
    pts = rng.random((30, 2))
    x, y = U.smooth_hull_lines(pts)
    assert len(x) == 100 and len(y) == 100


# ---------------------------------------------------------------------------
# print_progress
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# check_mask_folder
# ---------------------------------------------------------------------------

def test_check_mask_folder(tmp_path):
    # missing mask folder → True
    assert U.check_mask_folder(str(tmp_path), "cell") is True
    # equal counts → False
    (tmp_path / "stack").mkdir()
    (tmp_path / "masks" / "cell").mkdir(parents=True)
    # whole arrays: an empty mask is not counted (2026-09-19)
    for d in ("stack", "masks/cell"):
        np.save(tmp_path / d / "a.npy", np.zeros((4, 4), np.uint16))
    assert U.check_mask_folder(str(tmp_path), "cell") is False
    # fewer masks than stacks → True
    np.save(tmp_path / "stack" / "b.npy", np.zeros((4, 4), np.uint16))
    assert U.check_mask_folder(str(tmp_path), "cell") is True


# ---------------------------------------------------------------------------
# _extract_filename_metadata
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _generate_names
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _pivot_counts_table
# ---------------------------------------------------------------------------

def test_pivot_counts_table(tmp_path):
    import sqlite3
    db = tmp_path / "m.db"
    with sqlite3.connect(str(db)) as conn:
        pd.DataFrame({
            "file_name": ["a", "a", "b"],
            "count_type": ["cell", "nucleus", "cell"],
            "object_count": [10, 5, 7],
        }).to_sql("object_counts", conn, index=False)
    U._pivot_counts_table(str(db))
    with sqlite3.connect(str(db)) as conn:
        out = pd.read_sql_query("SELECT * FROM pivoted_counts", conn)
    assert "cell" in out.columns and "nucleus" in out.columns
    assert len(out) == 2   # one row per file_name
