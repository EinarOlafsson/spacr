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

def test_map_wells():
    plate, row, col, field, prcf = U._map_wells("plate1_B03_2")
    assert row == "r2" and col == "c3" and field == "f2"
    assert prcf == "plate1_r2_c3_f2"


def test_map_wells_timelapse():
    out = U._map_wells("plate1_B03_2_5", timelapse=True)
    assert len(out) == 6 and out[4] == "t5"


def test_map_wells_error():
    # too few parts → error path returns the error sentinel tuple
    out = U._map_wells("garbage")
    assert out[0] == "error"


def test_map_wells_png():
    out = U._map_wells_png("plate1_B03_2_10.png")
    plate, row, col, field, prcfo, object_id = out
    assert object_id == "o10" and row == "r2" and col == "c3"


def test_map_wells_png_timelapse():
    out = U._map_wells_png("plate1_B03_2_5_10.png", timelapse=True)
    assert len(out) == 7 and out[4] == "t5"


# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------

def test_crop_center():
    img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    out = U._crop_center(img, mask, new_width=16, new_height=16)
    assert out.shape == (16, 16, 3)


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


def test_get_object_settings_cell_variants():
    base = {"magnification": 20, "verbose": False, "merge_pathogens": False}
    s_no_nuc = U._get_object_settings("cell", {**base, "nucleus_channel": None})
    assert s_no_nuc["model_name"] == "cyto"
    s_nuc = U._get_object_settings("cell", {**base, "nucleus_channel": 1})
    assert s_nuc["model_name"] == "cyto2"


def test_get_object_settings_nucleus_pathogen():
    base = {"magnification": 20, "verbose": True, "merge_pathogens": True,
            "nucleus_channel": 0}
    nuc = U._get_object_settings("nucleus", base)
    assert nuc["model_name"] == "nuclei"
    pat = U._get_object_settings("pathogen", base)
    assert pat["model_name"] == "cyto" and pat["merge"] is True


# ---------------------------------------------------------------------------
# normalize_to_dtype
# ---------------------------------------------------------------------------

def test_normalize_to_dtype_variants():
    arr = (np.random.default_rng(1).random((16, 16, 2)) * 4000).astype(np.uint16)
    assert U.normalize_to_dtype(arr).shape == arr.shape
    assert U.normalize_to_dtype(arr, new_dtype=np.uint8).shape == arr.shape
    assert U.normalize_to_dtype(arr, new_dtype="uint16").shape == arr.shape
    # explicit per-channel percentile list
    out = U.normalize_to_dtype(arr, percentile_list=[(10, 3000), (5, 3500)])
    assert out.shape == arr.shape


# ---------------------------------------------------------------------------
# _filter_objects
# ---------------------------------------------------------------------------

def test_filter_objects_min_max_area():
    m = np.zeros((32, 32), dtype=np.uint16)
    m[0, 0] = 1                # 1 px
    m[4:8, 4:8] = 2            # 16 px
    m[12:24, 12:24] = 3        # 144 px
    out = U._filter_objects(m, min_area=5, max_area=100)
    kept = set(np.unique(out)) - {0}
    assert len(kept) == 1      # only the 16-px object survives


def test_filter_objects_remove_border():
    m = np.zeros((16, 16), dtype=np.uint16)
    m[0:4, 0:4] = 1            # touches border
    m[7:11, 7:11] = 2          # interior
    out = U._filter_objects(m, remove_border=True)
    assert len(set(np.unique(out)) - {0}) == 1


def test_filter_objects_intensity_percentile():
    m = np.zeros((16, 16), dtype=np.uint16)
    m[2:6, 2:6] = 1
    m[10:14, 10:14] = 2
    intensity = np.zeros((16, 16), dtype=np.float32)
    intensity[2:6, 2:6] = 10.0     # dim object
    intensity[10:14, 10:14] = 100.0  # bright object
    out = U._filter_objects(m, intensity_img=intensity,
                            min_intensity_percentile=50)
    assert 0 in np.unique(out)


def test_filter_objects_empty():
    empty = np.zeros((8, 8), dtype=np.uint16)
    assert np.array_equal(U._filter_objects(empty), empty)


# ---------------------------------------------------------------------------
# annotate_conditions
# ---------------------------------------------------------------------------

def _cond_df():
    return pd.DataFrame({
        "rowID": ["r1", "r2", "r1", "r2"],
        "columnID": ["c1", "c1", "c2", "c2"],
    })


def test_annotate_conditions_single_string():
    out = U.annotate_conditions(_cond_df(), cells="HeLa")
    assert (out["host_cells"] == "HeLa").all()


def test_annotate_conditions_list_default():
    out = U.annotate_conditions(_cond_df(), cells=["HeLa", "Vero"])
    assert (out["host_cells"] == "HeLa").all()   # first value broadcast


def test_annotate_conditions_loc_based():
    out = U.annotate_conditions(
        _cond_df(),
        pathogens=["wt", "ko"], pathogen_loc=[["r1"], ["r2"]],
        treatments=["dmso", "drug"], treatment_loc=[["c1"], ["c2"]])
    assert "condition" in out.columns
    # r1/c1 → wt_dmso
    r1c1 = out[(out.rowID == "r1") & (out.columnID == "c1")].iloc[0]
    assert r1c1["pathogen"] == "wt" and r1c1["treatment"] == "dmso"


# ---------------------------------------------------------------------------
# _split_data
# ---------------------------------------------------------------------------

def test_split_data():
    df = pd.DataFrame({
        "plateID": ["p1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1"] * 4, "fieldID": ["f1"] * 4,
        "object_label": [1, 2, 3, 4],
        "cell_area": [100.0, 200, 300, 400],
        "cell_channel_0_mean_intensity": [10.0, 20, 30, 40],
    })
    numeric, non_numeric = U._split_data(df, "prcf", "object_label")
    # area columns are summed, intensity averaged
    assert isinstance(numeric, pd.DataFrame)
    assert "cell_area" in numeric.columns
