"""Table-driven tests for the feature-table, crop, recruitment, results-path,
augmentation and statistics helpers of :mod:`spacr.utils`.

One general test per unit: each walks a case table under ``subtests`` and
checks the unit's invariants (exact values, shapes, files, refusals) on every
row. The plotting / dimensionality-reduction units live in
``test_utils_feature_tables_and_plots_2.py``.

CPU-only, offline, tiny synthetic inputs; every figure is closed.
"""
from __future__ import annotations

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from spacr import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _na_list(series):
    """A Series as a list with every missing value spelled ``None``."""
    return [None if pd.isna(v) else v for v in series.tolist()]


# ---------------------------------------------------------------------------
# _get_percentiles
# ---------------------------------------------------------------------------

def _percentile_reference(arr, p1, p2):
    """Per channel: percentiles of the >0 pixels, or of the raw channel if none."""
    out = []
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        nz = ch[ch > 0]
        src = nz if nz.size else ch
        out.append([float(np.percentile(src, p1)), float(np.percentile(src, p2))])
    return out


def test_get_percentiles_uses_positive_pixels_and_falls_back_to_raw_channel(subtests):
    """Each channel's (p1, p2) come from its >0 pixels, else from the raw channel."""
    ramp = np.arange(64, dtype=np.float32).reshape(8, 8)
    populated = ramp[:, :, None].copy()
    with_zero_channel = np.zeros((8, 8, 2), dtype=np.float32)
    with_zero_channel[:, :, 0] = ramp
    negative = np.linspace(-10.0, -1.0, 16).reshape(4, 4, 1)
    random3 = np.random.default_rng(0).integers(0, 1000, size=(16, 16, 3)).astype(np.uint16)

    cases = [
        ("populated", populated, dict(p1=2, p2=98), None),
        ("all-zero channel", with_zero_channel, dict(p1=2, p2=98), [None, [0.0, 0.0]]),
        ("all-negative channel", negative, dict(p1=0, p2=100), [[-10.0, -1.0]]),
        ("defaults on 3 channels", random3, {}, None),
    ]
    for name, arr, kw, exact in cases:
        with subtests.test(case=name):
            out = U._get_percentiles(arr, **kw)
            ref = _percentile_reference(arr, kw.get("p1", 2), kw.get("p2", 98))
            assert len(out) == arr.shape[2]
            assert all(len(p) == 2 for p in out)
            for got, want in zip(out, ref):
                assert got[0] == pytest.approx(want[0])
                assert got[1] == pytest.approx(want[1])
            if exact:
                for got, want in zip(out, exact):
                    if want is not None:
                        assert list(got) == pytest.approx(want)
            if name == "populated":
                # the zero pixel must not drag the low percentile to 0
                assert out[0][0] > 0.0
            if name == "all-zero channel":
                assert out[1] == [0.0, 0.0]
                assert out[0][0] > 0.0 and out[0][1] > out[0][0]


# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------

def test_crop_center_keeps_exactly_the_masked_pixels_in_a_window_of_requested_size(subtests):
    """The crop has the requested (h, w, c), keeps only masked pixels, zero-pads."""
    label_img = np.full((40, 40, 2), 9, dtype=np.uint8)
    label_mask = np.zeros((40, 40), dtype=np.int32)
    label_mask[16:24, 16:24] = 7

    rand_img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    rand_mask = np.zeros((32, 32), dtype=np.uint8)
    rand_mask[10:22, 10:22] = 1

    small_img = np.full((16, 16, 1), 5, dtype=np.uint8)
    small_mask = np.zeros((16, 16), dtype=np.uint8)
    small_mask[6:10, 6:10] = 1

    cases = [
        ("label mask", label_img, label_mask, 12, 12, 64),
        ("random image", rand_img, rand_mask, 16, 16, None),
        ("empty mask", np.full((20, 20, 3), 7, np.uint8), np.zeros((20, 20), np.uint8), 8, 6, 0),
        ("window larger than image", small_img, small_mask, 32, 32, 16),
    ]
    for name, img, mask, w, h, n_kept in cases:
        with subtests.test(case=name):
            with np.errstate(invalid="ignore"):
                out = U._crop_center(img, mask, new_width=w, new_height=h)
            assert out.shape == (h, w, img.shape[2])
            assert out.dtype == np.uint8
            inside = img[mask > 0]
            want = np.sort(inside[inside > 0].ravel())
            np.testing.assert_array_equal(np.sort(out[out > 0].ravel()), want)
            if n_kept is not None:
                assert int((out[:, :, 0] > 0).sum()) == n_kept
                assert int((out[:, :, 0] == 0).sum()) == h * w - n_kept
            if name == "label mask":
                ys, xs = np.nonzero(out[:, :, 0])
                assert ys.min() > 0 and ys.max() < 11
                assert xs.min() > 0 and xs.max() < 11
            if name == "empty mask":
                assert not out.any()


# ---------------------------------------------------------------------------
# _masks_to_masks_stack
# ---------------------------------------------------------------------------

def test_masks_to_masks_stack_lists_every_mask_in_order_without_copying(subtests):
    """A list comes back as the same objects in order; an array splits on axis 0."""
    int_list = [np.zeros((2, 2), dtype=np.int32) + i for i in range(3)]
    mixed = [np.zeros((4, 4)), np.ones((4, 4))]
    cases = [
        ("list of int masks", int_list, 3, (2, 2), True),
        ("list of float masks", mixed, 2, (4, 4), True),
        ("3-D array", np.zeros((3, 2, 2), dtype=np.int32), 3, (2, 2), False),
    ]
    for name, masks, n, shape, same_objects in cases:
        with subtests.test(case=name):
            out = U._masks_to_masks_stack(masks)
            assert isinstance(out, list)
            assert len(out) == n
            assert all(m.shape == shape for m in out)
            for i, m in enumerate(out):
                np.testing.assert_array_equal(m, masks[i])
            if same_objects:
                assert all(out[i] is masks[i] for i in range(n))


# ---------------------------------------------------------------------------
# _get_object_settings
# ---------------------------------------------------------------------------

def _object_settings_expected(diam, **extra):
    base = {
        "diameter": diam, "minimum_size": diam ** 2 / 4,
        "maximum_size": diam ** 2 * 10, "merge": False, "resample": True,
        "remove_border_objects": False, "model_name": "cpsam",
    }
    base.update(extra)
    return base


def test_get_object_settings_derives_bounds_from_diameter_per_object_type(subtests, capsys):
    """Size bounds follow the diameter; each type sets its own keys; bad types refuse."""
    seg = dict(filter_size=False, filter_intensity=False, restore_type=None)
    cases = [
        ("cell without nucleus", "cell",
         {"magnification": 20, "verbose": False, "merge_pathogens": False, "nucleus_channel": None},
         _object_settings_expected(120, **seg), None),
        ("cell with nucleus", "cell",
         {"magnification": 20, "verbose": False, "merge_pathogens": False, "nucleus_channel": 1},
         _object_settings_expected(120, **seg), None),
        ("nucleus verbose", "nucleus",
         {"magnification": 20, "verbose": True, "merge_pathogens": True, "nucleus_channel": 0},
         _object_settings_expected(60, **seg), "'diameter': 60"),
        ("pathogen merges", "pathogen",
         {"magnification": 20, "verbose": True, "merge_pathogens": True, "nucleus_channel": 0},
         _object_settings_expected(20, **{**seg, "resample": False, "merge": True}), "'merge': True"),
        ("cell_large is unsupported", "cell_large",
         {"magnification": 20, "verbose": False},
         _object_settings_expected(160), "cell_large not supported"),
    ]
    for name, obj, settings, expected, printed in cases:
        with subtests.test(case=name):
            capsys.readouterr()
            out = U._get_object_settings(obj, settings)
            assert out == expected
            text = capsys.readouterr().out
            if printed:
                assert printed in text
            if obj == "cell_large":
                for key in ("filter_size", "filter_intensity", "restore_type"):
                    assert key not in out
    with subtests.test(case="unknown type raises before settings"):
        with pytest.raises(ValueError, match="unsupported object type"):
            U._get_object_settings("mitochondria", {"magnification": 20, "verbose": False})


# ---------------------------------------------------------------------------
# _get_cellpose_channels
# ---------------------------------------------------------------------------

def test_get_cellpose_channels_extracts_each_source_once_and_remaps_densely(subtests):
    """Channels are deduplicated, sorted and remapped; cell gets nucleus second."""
    cases = [
        ("all four roles",
         {"cellpose_nucleus_channel": 2, "cellpose_cell_channel": 5,
          "cellpose_pathogen_channel": 1, "cellpose_organelle_channel": 7},
         [1, 2, 5, 7],
         {"nucleus": [1], "cell": [2, 1], "pathogen": [0], "organelle": [3]}),
        ("organelle only", {"cellpose_organelle_channel": 4}, [4], {"organelle": [0]}),
        ("shared channel", {"cellpose_cell_channel": 3, "cellpose_organelle_channel": 3},
         [3], {"cell": [0], "organelle": [0]}),
        ("identity remap",
         {"cellpose_nucleus_channel": 0, "cellpose_cell_channel": 1,
          "cellpose_pathogen_channel": 2, "cellpose_organelle_channel": None},
         [0, 1, 2], {"nucleus": [0], "cell": [1, 0], "pathogen": [2]}),
        ("cell without nucleus",
         {"cellpose_nucleus_channel": None, "cellpose_cell_channel": 2,
          "cellpose_pathogen_channel": None, "cellpose_organelle_channel": None},
         [2], {"cell": [0]}),
    ]
    for name, settings, extract, channels in cases:
        with subtests.test(case=name):
            got_extract, got_channels = U._get_cellpose_channels(settings)
            assert got_extract == extract
            assert got_channels == channels


# ---------------------------------------------------------------------------
# annotate_conditions
# ---------------------------------------------------------------------------

def _cond_df3():
    return pd.DataFrame({"rowID": ["r1", "r2", "r3"], "columnID": ["c1", "c2", "c3"]})


def _cond_df4():
    return pd.DataFrame({"rowID": ["r1", "r2", "r1", "r2"], "columnID": ["c1", "c1", "c2", "c2"]})


def test_annotate_conditions_assigns_by_row_or_column_and_joins_the_condition(subtests):
    """Values go to wells whose r*/c* id is listed; others stay NA, never ''."""
    cases = [
        ("unprefixed loc ignored", _cond_df3,
         dict(cells=["HeLa", "Vero"], cell_loc=[["x1"], ["r2"]]),
         {"host_cells": [None, "Vero", None], "condition": [None, "Vero", None]}),
        ("non-string loc ignored", _cond_df3,
         dict(pathogens=["wt", "ko"], pathogen_loc=[[1], ["c3"]]),
         {"pathogen": [None, None, "ko"], "condition": [None, None, "ko"]}),
        ("nothing matches", _cond_df3,
         dict(treatments=["drug"], treatment_loc=[["zz"]]),
         {"treatment": [None, None, None], "condition": [None, None, None]}),
        ("single string broadcast", _cond_df4, dict(cells="HeLa"),
         {"host_cells": ["HeLa"] * 4, "condition": ["HeLa"] * 4}),
        ("list without loc uses first", _cond_df4, dict(cells=["HeLa", "Vero"]),
         {"host_cells": ["HeLa"] * 4, "condition": ["HeLa"] * 4}),
        ("row and column locs", _cond_df4,
         dict(pathogens=["wt", "ko"], pathogen_loc=[["r1"], ["r2"]],
              treatments=["dmso", "drug"], treatment_loc=[["c1"], ["c2"]]),
         {"pathogen": ["wt", "ko", "wt", "ko"], "treatment": ["dmso", "dmso", "drug", "drug"],
          "condition": ["wt_dmso", "ko_dmso", "wt_drug", "ko_drug"]}),
    ]
    for name, make, kwargs, expected in cases:
        with subtests.test(case=name):
            out = U.annotate_conditions(make(), **kwargs)
            for col, values in expected.items():
                assert _na_list(out[col]) == values, col
            assert (out["condition"] == "").sum() == 0


# ---------------------------------------------------------------------------
# _split_data
# ---------------------------------------------------------------------------

def _split_cases():
    return [
        dict(
            name="missing fieldID reuses existing prcf",
            df=pd.DataFrame({
                "prcf": ["p1_r1_c1_f1"] * 2 + ["p1_r2_c1_f1"] * 2,
                "plateID": ["p1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
                "columnID": ["c1"] * 4, "object_label": [1, 2, 3, 4],
                "cell_area": [100.0, 200.0, 300.0, 400.0]}),
            by="prcf", printed=["Exception 'fieldID'"],
            numeric_index=["p1_r1_c1_f1", "p1_r2_c1_f1"],
            numeric={("p1_r1_c1_f1", "cell_area"): 300.0, ("p1_r2_c1_f1", "cell_area"): 700.0,
                     ("p1_r1_c1_f1", "object_label"): 1, ("p1_r2_c1_f1", "object_label"): 3},
            non_numeric={("p1_r1_c1_f1", "prcfo"): "p1_r1_c1_f1_1"},
            absent=["prcft"]),
        dict(
            name="all object columns -> empty numeric frame",
            df=pd.DataFrame({
                "plateID": ["p1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
                "columnID": ["c1"] * 4, "fieldID": ["f1"] * 4,
                "object_label": ["1", "2", "3", "4"], "note": ["a", "b", "c", "d"]}),
            by="prcf", printed=[], numeric_shape=(2, 0),
            numeric_index=["p1_r1_c1_f1", "p1_r2_c1_f1"],
            non_numeric={("p1_r1_c1_f1", "note"): "a", ("p1_r2_c1_f1", "note"): "c",
                         ("p1_r2_c1_f1", "prcfo"): "p1_r2_c1_f1_3"}),
        dict(
            name="all numeric -> empty non-numeric frame",
            df=pd.DataFrame({
                "plateID": [1] * 4, "rowID": [1, 1, 2, 2], "columnID": [1] * 4,
                "fieldID": [1] * 4, "object_label": [1, 2, 3, 4],
                "cell_area": [100.0, 200.0, 300.0, 400.0],
                "cell_channel_0_mean_intensity": [10.0, 20.0, 30.0, 40.0]}),
            by=["prcf", "prcfo"], non_numeric_shape=(4, 0),
            non_numeric_index=[("1_1_1_1", "1_1_1_1_1"), ("1_1_1_1", "1_1_1_1_2"),
                               ("1_2_1_1", "1_2_1_1_3"), ("1_2_1_1", "1_2_1_1_4")],
            numeric_cols={"cell_area": [100.0, 200.0, 300.0, 400.0],
                          "cell_channel_0_mean_intensity": [10.0, 20.0, 30.0, 40.0]}),
        dict(
            name="areas sum, lengths and intensities average, timepoint kept",
            df=pd.DataFrame({
                "plateID": ["p1"] * 4, "rowID": ["r1"] * 4, "columnID": ["c1"] * 4,
                "fieldID": ["f1"] * 4, "timeID": [1] * 4, "object_label": [1] * 4,
                "cell_area": [1.0, 2.0, 3.0, 4.0], "cell_perimeter": [1.0] * 4,
                "cell_equivalent_diameter": [2.0] * 4,
                "cell_channel_0_mean_intensity": [10.0, 20.0, 30.0, 40.0]}),
            by="prcfo", numeric_index=["p1_r1_c1_f1_1_1"],
            numeric={("p1_r1_c1_f1_1_1", "cell_area"): 10.0,
                     ("p1_r1_c1_f1_1_1", "cell_perimeter"): 1.0,
                     ("p1_r1_c1_f1_1_1", "cell_equivalent_diameter"): 2.0,
                     ("p1_r1_c1_f1_1_1", "cell_channel_0_mean_intensity"): 25.0},
            non_numeric={("p1_r1_c1_f1_1_1", "prcft"): "p1_r1_c1_f1_1",
                         ("p1_r1_c1_f1_1_1", "prcf"): "p1_r1_c1_f1_1"}),
        dict(
            name="per-well sum and mean",
            df=pd.DataFrame({
                "plateID": ["p1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
                "columnID": ["c1"] * 4, "fieldID": ["f1"] * 4,
                "object_label": [1, 2, 3, 4], "cell_area": [100.0, 200, 300, 400],
                "cell_channel_0_mean_intensity": [10.0, 20, 30, 40]}),
            by="prcf", numeric_index=["p1_r1_c1_f1", "p1_r2_c1_f1"],
            numeric={("p1_r1_c1_f1", "cell_area"): 300.0, ("p1_r2_c1_f1", "cell_area"): 700.0,
                     ("p1_r1_c1_f1", "cell_channel_0_mean_intensity"): 15.0,
                     ("p1_r2_c1_f1", "cell_channel_0_mean_intensity"): 35.0}),
        dict(
            name="numeric prcf rewritten as text",
            df=pd.DataFrame({
                "plateID": ["plate1"] * 2, "rowID": ["r1"] * 2, "columnID": ["c1"] * 2,
                "fieldID": ["f1"] * 2, "prcf": [1, 2], "object_label": [1, 2],
                "area": [1.0, 2.0]}),
            by="prcfo", non_numeric_cols={"prcf": ["plate1_r1_c1_f1"] * 2},
            numeric_absent=["prcf"]),
    ]


def test_split_data_groups_objects_and_separates_numeric_from_metadata(subtests, capsys):
    """Grouped numeric/non-numeric halves: exact keys, aggregates, printed notes."""
    for case in _split_cases():
        with subtests.test(case=case["name"]):
            df = case["df"]
            before = list(df.columns)
            capsys.readouterr()
            numeric, non_numeric = U._split_data(df, case["by"], "object_label")
            printed = capsys.readouterr().out
            assert isinstance(numeric, pd.DataFrame)
            assert isinstance(non_numeric, pd.DataFrame)
            assert list(df.columns) == before  # caller's frame not mutated
            if "printed" in case:
                assert printed.splitlines() == case["printed"]
            if "numeric_shape" in case:
                assert numeric.shape == case["numeric_shape"]
            if "numeric_index" in case:
                assert sorted(numeric.index.tolist()) == case["numeric_index"]
            for (idx, col), val in case.get("numeric", {}).items():
                assert numeric.loc[idx, col] == val, (idx, col)
            for col, vals in case.get("numeric_cols", {}).items():
                assert numeric[col].tolist() == vals
            for (idx, col), val in case.get("non_numeric", {}).items():
                assert non_numeric.loc[idx, col] == val, (idx, col)
            for col, vals in case.get("non_numeric_cols", {}).items():
                assert list(non_numeric[col]) == vals
            if "non_numeric_shape" in case:
                assert non_numeric.shape == case["non_numeric_shape"]
            if "non_numeric_index" in case:
                assert non_numeric.index.tolist() == case["non_numeric_index"]
            for col in case.get("absent", []):
                assert col not in numeric.columns and col not in non_numeric.columns
            for col in case.get("numeric_absent", []):
                assert col not in numeric.columns
                assert col in non_numeric.columns
            assert "prcfo" not in df.columns or "prcfo" in before


# ---------------------------------------------------------------------------
# _calculate_recruitment
# ---------------------------------------------------------------------------

def _recruitment_df(channel=1):
    return pd.DataFrame({
        f"pathogen_channel_{channel}_mean_intensity": [100.0, 200.0],
        f"cell_channel_{channel}_mean_intensity": [10.0, 50.0],
        f"cytoplasm_channel_{channel}_mean_intensity": [5.0, 25.0],
        f"nucleus_channel_{channel}_mean_intensity": [4.0, 8.0],
        f"pathogen_channel_{channel}_percentile_75": [300.0, 400.0],
        f"pathogen_channel_{channel}_outside_mean": [20.0, 30.0],
        f"pathogen_channel_{channel}_outside_75_percentile": [60.0, 90.0],
        f"pathogen_channel_{channel}_periphery_mean": [8.0, 16.0],
    })


_RECRUITMENT_RATIOS = {
    "cell_mean_ratio": [10.0, 4.0], "cytoplasm_mean_ratio": [20.0, 8.0],
    "nucleus_mean_ratio": [25.0, 25.0],
    "cell_q75_ratio": [30.0, 8.0], "cytoplasm_q75_ratio": [60.0, 16.0],
    "nucleus_q75_ratio": [75.0, 50.0],
    "cell_outside_mean_ratio": [2.0, 0.6], "cytoplasm_outside_mean_ratio": [4.0, 1.2],
    "nucleus_outside_mean_ratio": [5.0, 3.75],
    "cell_outside_q75_ratio": [6.0, 1.8], "cytoplasm_outside_q75_ratio": [12.0, 3.6],
    "nucleus_outside_q75_ratio": [15.0, 11.25],
    "cell_periphery_mean_ratio": [0.8, 0.32], "cytoplasm_periphery_mean_ratio": [1.6, 0.64],
    "nucleus_periphery_mean_ratio": [2.0, 2.0],
}


def test_calculate_recruitment_writes_every_ratio_in_place_or_refuses(subtests):
    """All 15 ratio columns are added in place; missing inputs raise KeyError."""
    with subtests.test(case="every ratio column"):
        df = _recruitment_df(1)
        out = U._calculate_recruitment(df, channel=1)
        assert out is df
        for suffix, values in _RECRUITMENT_RATIOS.items():
            assert out[f"pathogen_channel_1_{suffix}"].tolist() == pytest.approx(values), suffix

    with subtests.test(case="spatial slopes are not fabricated"):
        src = _recruitment_df(0)
        src["pathogen_slope_channel_0"] = [2.3, 5.4]
        out = U._calculate_recruitment(src, channel=0)
        assert out["pathogen_slope_channel_0"].tolist() == [2.3, 5.4]
        assert [k for k in out if "slope_channel_" in k] == ["pathogen_slope_channel_0"]

    with subtests.test(case="zero denominator -> inf"):
        df = _recruitment_df(2)
        df["nucleus_channel_2_mean_intensity"] = [0.0, 8.0]
        out = U._calculate_recruitment(df, channel=2)
        assert np.isinf(out.loc[0, "pathogen_channel_2_nucleus_mean_ratio"])
        assert out.loc[1, "pathogen_channel_2_nucleus_mean_ratio"] == 25.0

    refusals = [
        ("missing column", _recruitment_df(1).drop(columns=["pathogen_channel_1_periphery_mean"]),
         1, "periphery_mean"),
        ("wrong channel", _recruitment_df(1), 3, None),
    ]
    for name, df, channel, match in refusals:
        with subtests.test(case=name):
            with pytest.raises(KeyError, match=match):
                U._calculate_recruitment(df, channel=channel)


# ---------------------------------------------------------------------------
# _filter_objects
# ---------------------------------------------------------------------------

def test_filter_objects_keeps_only_objects_inside_every_bound_and_relabels(subtests):
    """Area, border and intensity bounds drop objects; survivors are relabelled 1..n."""
    area = np.zeros((32, 32), dtype=np.uint16)
    area[0, 0] = 1
    area[4:8, 4:8] = 2
    area[12:24, 12:24] = 3
    border = np.zeros((16, 16), dtype=np.uint16)
    border[0:4, 0:4] = 1
    border[7:11, 7:11] = 2
    two = np.zeros((16, 16), dtype=np.uint16)
    two[2:6, 2:6] = 1
    two[10:14, 10:14] = 2
    intensity = np.zeros((16, 16), dtype=np.float32)
    intensity[2:6, 2:6] = 10.0
    intensity[10:14, 10:14] = 100.0

    cases = [
        ("min/max area", area, dict(min_area=5, max_area=100), (area == 2).astype(np.uint16)),
        ("remove border", border, dict(remove_border=True), (border == 2).astype(np.uint16)),
        ("min intensity", two, dict(intensity_img=intensity, min_intensity=50),
         (two == 2).astype(np.uint16)),
        ("max intensity", two, dict(intensity_img=intensity, max_intensity=50),
         (two == 1).astype(np.uint16)),
        ("inclusive bounds keep all", two,
         dict(intensity_img=intensity, min_intensity=10, max_intensity=100), two),
        ("empty mask", np.zeros((8, 8), np.uint16), {}, np.zeros((8, 8), np.uint16)),
    ]
    for name, mask, kwargs, expected in cases:
        with subtests.test(case=name):
            out = U._filter_objects(mask.copy(), **kwargs)
            np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# _generate_names
# ---------------------------------------------------------------------------

def test_generate_names_builds_crop_name_and_class_folder(subtests):
    """Crop file name, class folder and table name follow the object counts."""
    from spacr.schema import parse_object_stem

    src = "SRC"
    j = os.path.join
    cases = [
        ("cell, one nucleus one pathogen", "plate1_A01_1", [5], [2], [3], "cell", False,
         "plate1_A01_1_5.png", j(src, "data", "single_nucleus", "single_pathogen", "plate1_A01"),
         None),
        ("underscored plate escaped", "my_plate_A01_1_1", [5], [2], [3], "cell", False,
         "my%5Fplate_A01_1_1_5.png",
         j(src, "data", "single_nucleus", "single_pathogen", "my%5Fplate_A01"),
         ("plateID", "my_plate")),
        ("ordinary plate untouched", "plate1_A01_1_1", [5], [2], [3], "cell", False,
         "plate1_A01_1_1_5.png", j(src, "data", "single_nucleus", "single_pathogen", "plate1_A01"),
         ("prcfo", "plate1_r1_c1_f1_o5")),
        ("nucleus, several nuclei, uninfected", "plate1_A01_1", [5], [2, 3], [0], "nucleus", False,
         "plate1_A01_1_5_multi.png",
         j(src, "data", "multiple_nucleus", "uninfected", "plate1_A01"), None),
        ("pathogen timelapse infected", "plate1_A01_1_2", [5], [0], [3], "pathogen", True,
         "plate1_A01_1_2_5_3.png", j(src, "data", "no_nucleus", "infected", "plate1_A01_1"),
         None),
    ]
    for name, stem, cell, nuc, pat, mode, tl, img_name, fldr, parsed in cases:
        with subtests.test(case=name):
            got_name, got_fldr, table = U._generate_names(
                stem, np.array(cell), np.array(nuc), np.array(pat), src,
                crop_mode=mode, timelapse=tl)
            assert got_name == img_name
            assert got_fldr == fldr
            assert table == fldr.replace(os.sep, "_")
            if parsed:
                assert getattr(parse_object_stem(got_name), parsed[0]) == parsed[1]


# ---------------------------------------------------------------------------
# _map_wells
# ---------------------------------------------------------------------------

CV_REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)"
            r"F(?P<fieldID>\d+)L(?P<laserID>\d+)A(?P<AID>\d+)"
            r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")


def _cv_name(well="A01", time="0001", field="001", chan="01", ext=".tif"):
    return f"plate1_{well}_T{time}F{field}L01A01Z01C{chan}{ext}"


def test_map_wells_parses_plate_row_column_field_and_errors_out(subtests, capsys, tmp_path):
    """A stem maps to (plate, rN, cN, fN[, tN], key); bad stems give 'error' slots."""
    cases = [
        ("letter well", ("plate1_B03_2",), {},
         ("plate1", "r2", "c3", "f2", "plate1_r2_c3_f2")),
        ("timelapse", ("plate1_B03_2_5",), dict(timelapse=True),
         ("plate1", "r2", "c3", "f2", "t5", "plate1_r2_c3_f2_t5")),
        ("numeric well copied into row and column", ("plate1_12_3",), {},
         ("plate1", "12", "12", "f3", "plate1_12_12_f3")),
        ("too few parts", ("garbage",), {}, ("error",) * 5),
    ]
    for name, args, kwargs, expected in cases:
        with subtests.test(case=name):
            capsys.readouterr()
            assert U._map_wells(*args, **kwargs) == expected
            if expected[0] == "error":
                assert "Error processing filename: garbage" in capsys.readouterr().out

    with subtests.test(case="reads back the stack names spacr.io writes"):
        from spacr.io import _rename_and_organize_image_files

        src = tmp_path / "plate1"
        src.mkdir()
        tifffile.imwrite(str(src / _cv_name(well="A01", field="002", chan="01")),
                         np.full((6, 8), 5, np.uint16))
        tifffile.imwrite(str(src / _cv_name(well="A01", field="002", chan="02")),
                         np.full((6, 8), 6, np.uint16))
        _rename_and_organize_image_files(
            str(src), CV_REGEX, batch_size=10, metadata_type="custom",
            img_format=[".tif"], save_original_images=False)
        written = sorted(os.listdir(src / "stack"))
        assert written == ["plate1_A01_2_1.npy"]
        assert U._map_wells(written[0], timelapse=False) == (
            "plate1", "r1", "c1", "f2", "plate1_r1_c1_f2")


# ---------------------------------------------------------------------------
# correct_paths
# ---------------------------------------------------------------------------

def test_correct_paths_reroots_folder_paths_and_refuses_other_inputs(subtests, capsys):
    """Paths containing '/<folder>/' move under base_path; frames update in place."""
    frame_paths = [
        "/old/root/data/plate1/A01_1_cell_1.png", "/old/root/data/plate1/A01_1_cell_2.png",
        "/old/root/misc/no_folder_here.png", "/new/base/data/plate1/already_ok.png"]
    frame_expected = [
        "/new/base/data/plate1/A01_1_cell_1.png", "/new/base/data/plate1/A01_1_cell_2.png",
        "/old/root/misc/no_folder_here.png", "/new/base/data/plate1/already_ok.png"]
    frame_cases = [
        ("frame", frame_paths, dict(base_path="/new/base", folder="data"), frame_expected),
        ("frame default folder", ["/old/data/cell_png/a.png"], dict(base_path="/new"),
         ["/new/data/cell_png/a.png"]),
    ]
    for name, paths, kwargs, expected in frame_cases:
        with subtests.test(case=name):
            df = pd.DataFrame({"png_path": paths})
            out_df, out_paths = U.correct_paths(df, **kwargs)
            assert out_df is df
            assert out_df["png_path"].tolist() == expected
            if name == "frame":
                assert out_paths == expected

    list_cases = [
        ("list", ["/old/data/p1/x.png", "/old/nope/y.png"], "data",
         ["/mnt/base/data/p1/x.png", "/old/nope/y.png"]),
        ("custom folder", ["/old/root/pngs/p1/x.png"], "pngs", ["/mnt/base/pngs/p1/x.png"]),
    ]
    for name, paths, folder, expected in list_cases:
        with subtests.test(case=name):
            got = U.correct_paths(paths, base_path="/mnt/base", folder=folder)
            assert isinstance(got, list)
            assert got == expected

    with subtests.test(case="frame without png_path"):
        capsys.readouterr()
        df = pd.DataFrame({"something_else": ["a", "b"]})
        out_df, out_paths = U.correct_paths(df, base_path="/new/base", folder="data")
        assert out_df is df and out_paths is None
        assert "No 'png_path' column found in the dataframe." in capsys.readouterr().out

    with subtests.test(case="neither frame nor list"):
        with pytest.raises(UnboundLocalError, match="image_paths"):
            U.correct_paths({"png_path": ["/old/data/cell_png/a.png"]}, "/new")


# ---------------------------------------------------------------------------
# delete_folder
# ---------------------------------------------------------------------------

def test_delete_folder_removes_only_existing_directories(subtests, tmp_path, capsys):
    """A directory tree is removed and reported; a file or missing path is left alone."""
    tree = tmp_path / "victim"
    (tree / "sub" / "deeper").mkdir(parents=True)
    (tree / "top.txt").write_text("x", encoding="utf-8")
    (tree / "sub" / "mid.txt").write_text("y", encoding="utf-8")
    (tree / "sub" / "deeper" / "leaf.txt").write_text("z", encoding="utf-8")
    a_file = tmp_path / "a_file.txt"
    a_file.write_text("keep me", encoding="utf-8")

    cases = [
        ("nested tree", tree, False, f"Folder '{tree}' has been deleted."),
        ("missing path", tmp_path / "does_not_exist", False, "does not exist or is not a directory"),
        ("a file", a_file, True, "does not exist or is not a directory"),
    ]
    for name, path, survives, message in cases:
        with subtests.test(case=name):
            capsys.readouterr()
            assert U.delete_folder(str(path)) is None
            assert path.exists() is survives
            assert message in capsys.readouterr().out
            if survives:
                assert path.read_text(encoding="utf-8") == "keep me"


# ---------------------------------------------------------------------------
# generate_path_list_from_db
# ---------------------------------------------------------------------------

def _png_db(path, paths):
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE png_list (png_path TEXT)")
        con.executemany("INSERT INTO png_list (png_path) VALUES (?)", [(p,) for p in paths])
        con.commit()
    finally:
        con.close()
    return str(path)


def test_generate_path_list_from_db_selects_by_metadata_substring(subtests, tmp_path, monkeypatch, capsys):
    """None = all rows, str/list = LIKE (OR-ed), other types = nothing, errors -> None."""
    all_paths = [f"/data/plate1/A{i:02d}_1_cell_{i}.png" for i in range(5)]
    wells = ["/data/p1/A01_1.png", "/data/p1/B02_1.png", "/data/p1/C03_1.png", "/data/p1/D04_1.png"]
    db_all = _png_db(tmp_path / "all.db", all_paths)
    db_wells = _png_db(tmp_path / "wells.db", wells)

    cases = [
        ("no metadata", db_all, None, sorted(all_paths)),
        ("single string", db_wells, "B02", ["/data/p1/B02_1.png"]),
        ("list OR-ed", db_wells, ["A01", "C03"], ["/data/p1/A01_1.png", "/data/p1/C03_1.png"]),
        ("one-item list", db_wells, ["A01"], ["/data/p1/A01_1.png"]),
        ("tuple selects nothing", db_wells, ("A01",), []),
    ]
    for name, db, meta, expected in cases:
        with subtests.test(case=name):
            assert sorted(U.generate_path_list_from_db(db, meta)) == expected

    with subtests.test(case="sqlite error -> None"):
        empty = tmp_path / "empty.db"
        sqlite3.connect(empty).close()
        capsys.readouterr()
        assert U.generate_path_list_from_db(str(empty), None) is None
        assert "Database error:" in capsys.readouterr().out

    with subtests.test(case="generic error -> None"):
        def boom(*a, **k):
            raise ValueError("connect exploded")

        with monkeypatch.context() as m:
            m.setattr(U.sqlite3, "connect", boom)
            capsys.readouterr()
            assert U.generate_path_list_from_db(str(tmp_path / "nope.db"), "A01") is None
        out = capsys.readouterr().out
        assert "Error: connect exploded" in out
        assert "Database error:" not in out


# ---------------------------------------------------------------------------
# measure_test_mode
# ---------------------------------------------------------------------------

def test_measure_test_mode_points_src_at_a_fresh_test_subset(subtests, tmp_path, capsys):
    """Test mode copies test_nr files into <parent>/test/<name>; off or already-test is a no-op."""
    with subtests.test(case="copies subset and clears stale folder"):
        root = tmp_path / "a"
        src = root / "merged"
        src.mkdir(parents=True)
        for i in range(6):
            (src / f"plate1_A01_{i}.npy").write_text(str(i), encoding="utf-8")
        stale = root / "test" / "merged"
        (stale / "leftover_dir").mkdir(parents=True)
        (stale / "stale.npy").write_text("old", encoding="utf-8")
        capsys.readouterr()
        out = U.measure_test_mode({"src": str(src), "test_mode": True, "test_nr": 3})
        expected = os.path.join(str(root), "test", "merged")
        assert out["src"] == expected
        copied = sorted(os.listdir(expected))
        assert len(copied) == 3
        assert "stale.npy" not in copied and "leftover_dir" not in copied
        assert all(n.startswith("plate1_A01_") for n in copied)
        assert len(os.listdir(src)) == 6
        assert f"Changed source folder to {expected} for test mode" in capsys.readouterr().out

    with subtests.test(case="src already a test folder"):
        root = tmp_path / "b"
        src = root / "test"
        src.mkdir(parents=True)
        capsys.readouterr()
        out = U.measure_test_mode({"src": str(src), "test_mode": True, "test_nr": 2})
        assert out["src"] == str(src)
        assert not (src / "merged").exists()
        assert f"Test mode enabled, using source folder {src}" in capsys.readouterr().out

    with subtests.test(case="disabled"):
        root = tmp_path / "c"
        src = root / "merged"
        src.mkdir(parents=True)
        out = U.measure_test_mode({"src": str(src), "test_mode": False, "test_nr": 2})
        assert out["src"] == str(src)
        assert not (root / "test").exists()


# ---------------------------------------------------------------------------
# preprocess_data
# ---------------------------------------------------------------------------

def _channel_feature_df(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cell_channel_1_mean_intensity": rng.normal(10.0, 2.0, n),
        "nucleus_channel_1_mean_intensity": rng.normal(5.0, 1.0, n),
        "cell_channel_2_mean_intensity": rng.normal(20.0, 3.0, n),
        "cell_channel_3_mean_intensity": rng.normal(30.0, 4.0, n),
        "cell_area": rng.normal(500.0, 50.0, n),
        "object_id": np.arange(n),
        "pathogen_count": rng.integers(0, 4, n),
    })


def _zscore(series):
    return ((series - series.mean()) / series.std(ddof=0)).to_numpy()


def _pp(df, **kw):
    args = dict(filter_by=None, remove_highly_correlated=False, log_data=False, exclude=None)
    args.update(kw)
    return U.preprocess_data(df, **args)


def test_preprocess_data_returns_a_scaled_finite_feature_matrix(subtests, capsys):
    """Selected numeric features come back z-scored, finite, with exact shapes."""
    df = _channel_feature_df()
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 1.0, 200)
    b = 0.9 * a + rng.normal(0.0, 0.9, 200)
    corr = pd.DataFrame({"a": a, "b": b, "c": rng.normal(0.0, 1.0, 200)})
    assert 0.5 < abs(np.corrcoef(a, b)[0, 1]) < 0.95
    rng5 = np.random.default_rng(5)
    logdf = pd.DataFrame({"a": rng5.uniform(1.0, 10.0, 30), "b": rng5.uniform(1.0, 10.0, 30)})
    logdf.loc[0, "a"] = np.nan
    log_expected = logdf["a"].copy()
    log_expected.iloc[0] = log_expected.median()
    log_expected = _zscore(np.log(log_expected + 1e-6))
    cols = ["cell_area", "cell_channel_2_mean_intensity"]

    cases = [
        ("filter by channel", lambda: _pp(df.copy(), filter_by=1), (60, 2),
         "Dropped 0 columns with NaN values", None),
        ("sqlite numeric text", lambda: _pp(pd.DataFrame({
            "cell_channel_0_mode_intensity": [str(v) for v in np.linspace(2.5, 25.0, 20)],
            "object_label": np.arange(20)}), filter_by="channel_0"), (20, 1),
         "Dropped 0 columns with NaN values", None),
        ("column list subsets first", lambda: _pp(df.copy(), column_list=cols), (60, 2), None,
         (0, _zscore(df["cell_area"]))),
        ("float correlation threshold", lambda: _pp(corr.copy(), remove_highly_correlated=0.5),
         (200, 2), None, None),
        ("default correlation threshold", lambda: _pp(corr.copy(), remove_highly_correlated=True),
         (200, 3), None, None),
        ("log transform after median fill", lambda: _pp(logdf.copy(), log_data=True), (30, 2),
         None, (0, log_expected)),
        ("wholly missing feature dropped", lambda: _pp(pd.DataFrame({
            "usable": np.linspace(1.0, 5.0, 12), "unobserved": np.full(12, np.nan)})),
         (12, 1), "Dropped 1 columns with NaN values", None),
    ]
    for name, run, shape, printed, column in cases:
        with subtests.test(case=name):
            capsys.readouterr()
            out = run()
            assert isinstance(out, np.ndarray)
            assert out.shape == shape
            assert np.isfinite(out).all()
            assert np.allclose(out.mean(axis=0), 0.0, atol=1e-10)
            if name != "sqlite numeric text":
                assert np.allclose(out.std(axis=0), 1.0, atol=1e-10)
            if printed:
                assert printed in capsys.readouterr().out
            if column:
                assert np.allclose(out[:, column[0]], column[1])

    expected_all = _pp(df)
    for filter_by in ["", "None", " null ", "all", "*"]:
        with subtests.test(case=f"none-like filter {filter_by!r}"):
            out = _pp(df, filter_by=filter_by)
            assert out.shape == expected_all.shape
            assert np.allclose(out, expected_all)

    with subtests.test(case="infinity is a missing measurement"):
        repaired = _pp(pd.DataFrame({"cell_area": [1.0, 2.0, np.inf, 4.0],
                                     "cell_perimeter": [np.inf, -np.inf, np.inf, -np.inf]}))
        expected = _pp(pd.DataFrame({"cell_area": [1.0, 2.0, 2.0, 4.0]}))
        assert repaired.shape == expected.shape == (4, 1)
        assert np.isfinite(repaired).all()
        assert np.allclose(repaired, expected)

    with subtests.test(case="unknown filter lists available channels"):
        with pytest.raises(ValueError) as error:
            _pp(_channel_feature_df(), filter_by="channel_9")
        message = str(error.value)
        assert "filter_by='channel_9' matched no measurement features" in message
        assert "channel_1" in message and "channel_2" in message
        assert "Set filter_by to None" in message

    with subtests.test(case="no numeric columns"):
        with pytest.raises(ValueError, match="No numeric measurement columns"):
            _pp(pd.DataFrame({"well": ["A01", "A02", "A03"],
                              "prc": ["p1_A01_1", "p1_A02_1", "p1_A03_1"]}),
                remove_highly_correlated=True)


# ---------------------------------------------------------------------------
# filter_dataframe_features
# ---------------------------------------------------------------------------

def _self_pair_df(n=50, seed=13):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cell_channel_1_mean_intensity": rng.normal(10.0, 2.0, n),
        "cell_channel_2_mean_intensity": rng.normal(20.0, 3.0, n),
        "cell_channel_3_mean_intensity": rng.normal(30.0, 4.0, n),
        "cell_channel_4_mean_intensity": rng.normal(40.0, 5.0, n),
        "cell_area": rng.normal(500.0, 50.0, n),
        "object_id": np.arange(n),
        "pathogen_count": rng.integers(0, 4, n),
        "pathogen_pathogen": rng.normal(1.0, 0.5, n),
        "cell_cell": rng.normal(2.0, 0.5, n),
        "nucleus_nucleus": rng.normal(3.0, 0.5, n),
        "cytoplasm_cytoplasm": rng.normal(4.0, 0.5, n),
    })


def _plain_feature_df(n=30):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "cell_channel_0_mean_intensity": rng.normal(10, 2, n),
        "cell_channel_1_mean_intensity": rng.normal(20, 3, n),
        "cell_area": rng.normal(500, 50, n),
        "cell_perimeter": rng.normal(90, 5, n),
        "label_text": ["a"] * n,
    })


def _morph_df(n=40):
    rng = np.random.default_rng(21)
    return pd.DataFrame({
        "cell_area": rng.normal(500.0, 50.0, n),
        "cell_perimeter": rng.normal(90.0, 9.0, n),
        "cell_solidity": rng.uniform(0.5, 0.99, n),
        "cell_channel_1_mean_intensity": rng.normal(10.0, 2.0, n),
        "object_id": np.arange(n),
    })


_PLAIN_FEATURES = ["cell_channel_0_mean_intensity", "cell_channel_1_mean_intensity",
                   "cell_area", "cell_perimeter"]
_NO_PRUNE = dict(remove_low_variance_features=False, remove_highly_correlated_features=False)
_ID_LIKE = ["pathogen_pathogen", "cell_cell", "nucleus_nucleus", "cytoplasm_cytoplasm",
            "object_id", "pathogen_count"]


def test_filter_dataframe_features_keeps_exactly_the_requested_feature_columns(subtests, capsys):
    """Returned features equal the filtered frame's columns and honour every selector."""
    partial = _self_pair_df(n=5)
    partial_name = "cell_channel_1_homogeneity_distance_32"
    partial[partial_name] = [0.1, np.nan, 0.9, np.nan, 0.5]
    with_nan = _self_pair_df()
    with_nan["cell_channel_1_std_intensity"] = np.nan

    cases = [
        dict(name="self-measurement and id columns dropped", df=_self_pair_df(),
             kw=dict(channel_of_interest=None, verbose=True), absent=_ID_LIKE,
             present=["cell_area"],
             printed=["Columns to remove:", "pathogen_pathogen", "cytoplasm_cytoplasm"]),
        dict(name="channel list", df=_self_pair_df(),
             kw=dict(channel_of_interest=[1, 2], verbose=True),
             features=["cell_channel_1_mean_intensity", "cell_channel_2_mean_intensity"],
             shape=(50, 2), printed=["Removed columns:"]),
        dict(name="channel string", df=_self_pair_df(), kw=dict(channel_of_interest="channel_3"),
             features=["cell_channel_3_mean_intensity"], shape=(50, 1)),
        dict(name="exclude list (pruning on)", df=_self_pair_df(),
             kw=dict(channel_of_interest=None, exclude=["cell_area"]), absent=["cell_area"]),
        dict(name="exclude str (pruning on)", df=_self_pair_df(),
             kw=dict(channel_of_interest=None, exclude="cell_area"), absent=["cell_area"]),
        dict(name="exclude list", df=_plain_feature_df(),
             kw=dict(channel_of_interest=None,
                     exclude=["cell_area", "cell_channel_1_mean_intensity"], **_NO_PRUNE),
             features=["cell_channel_0_mean_intensity", "cell_perimeter"], shape=(30, 2)),
        dict(name="exclude str", df=_plain_feature_df(),
             kw=dict(channel_of_interest=None, exclude="cell_perimeter", **_NO_PRUNE),
             features=["cell_channel_0_mean_intensity", "cell_channel_1_mean_intensity",
                       "cell_area"], shape=(30, 3)),
        dict(name="all-NaN column dropped", df=with_nan,
             kw=dict(channel_of_interest="channel_1", **_NO_PRUNE),
             features=["cell_channel_1_mean_intensity"],
             printed=["Dropped 1 columns with NaN values"]),
        dict(name="partial NaN median-imputed", df=partial,
             kw=dict(channel_of_interest="channel_1", **_NO_PRUNE), present=[partial_name],
             values={partial_name: [0.1, 0.5, 0.9, 0.5, 0.5]},
             printed=["Dropped 0 columns with NaN values",
                      "Median-imputed 2 missing value(s) in 1 partially observed"]),
        dict(name="morphology", df=_morph_df(), kw=dict(channel_of_interest="morphology"),
             features=["cell_area", "cell_perimeter", "cell_solidity"], shape=(40, 3)),
    ]
    for case in cases:
        with subtests.test(case=case["name"]):
            df = case["df"]
            capsys.readouterr()
            filtered, features = U.filter_dataframe_features(df.copy(), **case["kw"])
            printed = capsys.readouterr().out
            assert list(filtered.columns) == list(features)
            assert len(filtered) == len(df)
            if "features" in case:
                assert sorted(features) == sorted(case["features"])
            for col in case.get("absent", []):
                assert col not in features and col not in filtered.columns
            for col in case.get("present", []):
                assert col in features
            if "shape" in case:
                assert filtered.shape == case["shape"]
            for col, vals in case.get("values", {}).items():
                assert filtered[col].tolist() == vals
            for fragment in case.get("printed", []):
                assert fragment in printed

    with subtests.test(case="unmatched exclusion warns and keeps data"):
        df = _plain_feature_df()
        with pytest.warns(UserWarning, match=r"Feature exclusion 'not_a_column' matched no columns\."):
            filtered, features = U.filter_dataframe_features(
                df.copy(), channel_of_interest=None, exclude="not_a_column", **_NO_PRUNE)
        assert features == _PLAIN_FEATURES
        pd.testing.assert_frame_equal(filtered, df.loc[:, _PLAIN_FEATURES])


# ---------------------------------------------------------------------------
# find_non_overlapping_position
# ---------------------------------------------------------------------------

def test_find_non_overlapping_position_jitters_within_ten_or_returns_the_original(subtests):
    """A free spot within +/-10 is found when possible; else the input comes back."""
    cases = [
        ("free spot, tiny threshold", (0.0, 0.0), [(0.0, 0.0), (50.0, 50.0)], 1e-9, 10, False),
        ("free spot, far neighbour", (0.0, 0.0), [(100.0, 100.0)], 5.0, 10, False),
        ("gives up, threshold beyond jitter", (3.0, -4.0), [(0.0, 0.0)], 10_000.0, 5, True),
        ("gives up at origin", (0.0, 0.0), [(0.0, 0.0)], 1000.0, 5, True),
        ("zero attempts", (1.5, 2.5), [(1.5, 2.5)], 0.1, 0, True),
    ]
    for name, (x0, y0), occupied, threshold, attempts, unchanged in cases:
        with subtests.test(case=name):
            x, y = U.find_non_overlapping_position(x0, y0, occupied, threshold=threshold,
                                                   max_attempts=attempts)
            if unchanged:
                assert (x, y) == (x0, y0)
                assert U.check_overlap((x, y), occupied, threshold=threshold) is True
            else:
                assert (x, y) != (x0, y0)
                assert abs(x - x0) <= 10.0 and abs(y - y0) <= 10.0
                assert U.check_overlap((x, y), occupied, threshold=threshold) is False


# ---------------------------------------------------------------------------
# filter_columns
# ---------------------------------------------------------------------------

def test_filter_columns_selects_by_substring_or_morphology(subtests):
    """A substring keeps matching columns; 'morphology' keeps non-channel columns."""
    cases = [
        ("substring", pd.DataFrame({"cell_channel_0_mean": [1.0], "cell_channel_1_mean": [2.0],
                                    "cell_area": [3.0]}), "channel_1",
         ["cell_channel_1_mean"], [2.0]),
        ("morphology", pd.DataFrame({"cell_area": [3.0], "cell_channel_0_mean": [1.0],
                                     "nucleus_perimeter": [4.0], "pathogen_channel_2_max": [5.0]}),
         "morphology", ["cell_area", "nucleus_perimeter"], [3.0, 4.0]),
        ("morphology with non-string columns",
         pd.DataFrame({0: [1.0], "cell_channel_0": [2.0], 7: [3.0]}), "morphology",
         [0, 7], [1.0, 3.0]),
    ]
    for name, df, filter_by, cols, row in cases:
        with subtests.test(case=name):
            out = U.filter_columns(df, filter_by=filter_by)
            assert list(out.columns) == cols
            assert out.shape == (1, len(cols))
            assert out.iloc[0].tolist() == row


# ---------------------------------------------------------------------------
# get_ml_results_paths
# ---------------------------------------------------------------------------

_EXPECTED_BASENAMES = [
    "results.csv", "permutation.csv", "feature_importance.csv", None,
    "permutation.pdf", "feature_importance.pdf", "shap.pdf", "plate_heatmap.pdf",
    "ml_settings.csv", "ml_features.csv",
]


def test_get_ml_results_paths_names_ten_files_in_one_feature_space_folder(subtests, tmp_path):
    """Ten distinct paths under results/<model>/<feature space>; bad spaces refuse."""
    cases = [
        ("int channel", "xgboost", 2, "channel_2"),
        ("list of channels", "random_forest", [0, 1, 3], "channels_0_1_3"),
        ("tuple of channels", "xgboost", (1, 2), "channels_1_2"),
        ("morphology", "logistic", "morphology", "morphology"),
        ("object name", "xgboost", "nucleus", "nucleus"),
        ("column fragment", "xgboost", "mean_intensity", "mean_intensity"),
        ("None means all", "xgboost", None, "all_features"),
    ]
    for i, (name, model, channel, folder) in enumerate(cases):
        with subtests.test(case=name):
            src = str(tmp_path / f"run{i}")
            paths = U.get_ml_results_paths(src, model_type=model, channel_of_interest=channel)
            assert isinstance(paths, tuple) and len(paths) == 10
            res = os.path.join(src, "results", model, folder)
            assert os.path.isdir(res)
            for path, base in zip(paths, _EXPECTED_BASENAMES):
                assert os.path.dirname(path) == res
                if base is not None:
                    assert os.path.basename(path) == base
            assert os.path.basename(paths[3]) == f"{model}_model.csv"
            assert len(set(paths)) == 10

    with subtests.test(case="idempotent"):
        src = str(tmp_path / "again")
        first = U.get_ml_results_paths(src, channel_of_interest=1)
        second = U.get_ml_results_paths(src, channel_of_interest=1)
        assert first == second
        assert os.path.isdir(os.path.dirname(first[0]))

    for j, bad in enumerate([2.5, {"a": 1}, object()]):
        with subtests.test(case=f"rejects {type(bad).__name__}"):
            src = str(tmp_path / f"bad{j}")
            with pytest.raises(ValueError, match="channel_of_interest"):
                U.get_ml_results_paths(src, channel_of_interest=bad)
            assert not os.path.exists(os.path.join(src, "results"))


# ---------------------------------------------------------------------------
# merge_regression_res_with_metadata
# ---------------------------------------------------------------------------

def _csvs(tmp_path, tag, results, metadata):
    r = tmp_path / f"{tag}_results.csv"
    m = tmp_path / f"{tag}_meta.csv"
    pd.DataFrame(results).to_csv(r, index=False)
    pd.DataFrame(metadata).to_csv(m, index=False)
    return r, m


def test_merge_regression_res_with_metadata_left_joins_parsed_genes(subtests, tmp_path):
    """Genes parsed from feature names left-join metadata; nothing leaks, misses refuse."""
    full = ({"feature": ["C(gene)[T.ABC123_2]", "C(gene)[DEF456_11]", "C(gene)[T.GHI789]",
                         "Intercept"],
             "coefficient": [1.5, -0.5, 0.25, 10.0]},
            {"Gene ID": ["TGME49_ABC123", "TGME49_DEF456", "TGME49_ZZZ000"],
             "Description": ["kinase", "transporter", "unrelated"]})
    cases = [
        ("parses and merges", full, {}, ["ABC123", "DEF456", "GHI789", None],
         ["kinase", "transporter", None, None]),
        ("disjoint genes", ({"feature": ["C(gene)[T.AAA_1]"], "coefficient": [0.1]},
                            {"Gene ID": ["TGME49_BBB"], "Description": ["nope"]}), {},
         ["AAA"], [None]),
        ("no gene never joins", ({"feature": ["Intercept", "C(gene)[T.AAA_1]"],
                                  "coefficient": [1.0, 2.0]},
                                 {"Gene ID": ["NOUNDERSCORE", "TGME49_AAA"],
                                  "Description": ["SPURIOUS", "real"]}), {},
         [None, "AAA"], [None, "real"]),
    ]
    for i, (name, (res, meta), kw, genes, descriptions) in enumerate(cases):
        with subtests.test(case=name):
            r, m = _csvs(tmp_path, f"c{i}", res, meta)
            merged = U.merge_regression_res_with_metadata(str(r), str(m), **kw)
            assert len(merged) == len(res["feature"])
            assert _na_list(merged["gene"]) == genes
            assert _na_list(merged["Description"]) == descriptions
            assert merged["coefficient"].tolist() == res["coefficient"]
            written = tmp_path / f"c{i}_results_metadata.csv"
            assert written.is_file()

    with subtests.test(case="custom suffix writes that csv"):
        r, m = _csvs(tmp_path, "sfx", *full)
        merged = U.merge_regression_res_with_metadata(str(r), str(m), name="_annotated")
        expected = tmp_path / "sfx_results_annotated.csv"
        assert expected.is_file()
        assert not (tmp_path / "sfx_results_metadata.csv").exists()
        roundtrip = pd.read_csv(expected)
        assert list(roundtrip.columns) == list(merged.columns)
        assert len(roundtrip) == len(merged)
        assert roundtrip.loc[0, "Description"] == "kinase"
        assert "Unnamed: 0" not in roundtrip.columns

    with subtests.test(case="no identifier column"):
        r, m = _csvs(tmp_path, "noid", {"feature": ["C(gene)[T.TGGT1_225160_2]"],
                                        "coefficient": [0.5]},
                     {"description": ["a kinase"], "notes": ["x"]})
        with pytest.raises(ValueError, match="no column holding a gene"):
            U.merge_regression_res_with_metadata(str(r), str(m))


# ---------------------------------------------------------------------------
# process_vision_results
# ---------------------------------------------------------------------------

def test_process_vision_results_maps_wells_and_applies_inclusive_threshold(subtests):
    """Path -> plate/row/column/field/object/prc; pred >= threshold is positive."""
    cases = [
        ("letter wells", ["plate1_A01_3_7.png", "plate1_B12_10_2.png", "plate2_C03_1_5.png"],
         [0.20, 0.50, 0.99], 0.5,
         dict(plateID=["plate1", "plate1", "plate2"], rowID=["r1", "r2", "r3"],
              columnID=["c1", "c12", "c3"], fieldID=["f3", "f10", "f1"], object=["7", "2", "5"],
              prc=["plate1_r1_c1", "plate1_r2_c12", "plate2_r3_c3"], cv_predictions=[0, 1, 1])),
        ("higher threshold", ["p_A01_1_1.png", "p_A02_1_1.png"], [0.6, 0.95], 0.9,
         dict(cv_predictions=[0, 1])),
        ("numeric well kept raw", ["plateX_12_4_9.png"], [1.0], 0.5,
         dict(rowID=["12"], columnID=["12"], prc=["plateX_12_12"], cv_predictions=[1])),
    ]
    for name, paths, preds, threshold, expected in cases:
        with subtests.test(case=name):
            df = pd.DataFrame({"path": paths, "pred": preds})
            out = U.process_vision_results(df, threshold=threshold)
            assert out is df
            for col, vals in expected.items():
                assert list(out[col]) == vals, col
            assert out["cv_predictions"].dtype.kind == "i"


# ---------------------------------------------------------------------------
# augment_image
# ---------------------------------------------------------------------------

def _eight_views(a):
    """original, hflip, rot90cw, +hflip, rot180, +hflip, rot90ccw, +hflip."""
    views = []
    for k in (0, -1, 2, 1):
        r = np.rot90(a, k, axes=(0, 1))
        views += [r, r[:, ::-1]]
    return views


def test_augment_image_returns_the_eight_dihedral_views_as_rgb(subtests):
    """Eight RGB PIL images in dihedral order; grey input is replicated to 3 channels."""
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    gray4 = np.arange(16, dtype=np.uint8).reshape(4, 4) * 15
    rgb = np.random.default_rng(0).integers(0, 256, size=(4, 6, 3), dtype=np.uint8)
    rgb5 = np.random.default_rng(1).integers(0, 256, size=(5, 5, 3), dtype=np.uint8)
    cases = [
        ("grey array", gray, np.stack([gray] * 3, axis=-1)),
        ("non-square grey", np.zeros((3, 5), np.uint8), np.zeros((3, 5, 3), np.uint8)),
        ("grey PIL", Image.fromarray(gray4, mode="L"), np.stack([gray4] * 3, axis=-1)),
        ("colour array", rgb, rgb),
        ("RGB PIL", Image.fromarray(rgb5, mode="RGB"), rgb5),
    ]
    for name, image, as_rgb in cases:
        with subtests.test(case=name):
            out = U.augment_image(image)
            assert len(out) == 8
            assert all(isinstance(im, Image.Image) for im in out)
            arrs = [np.asarray(im) for im in out]
            for got, want in zip(arrs, _eight_views(as_rgb)):
                assert got.shape == want.shape
                np.testing.assert_array_equal(got, want)
            h, w = as_rgb.shape[:2]
            assert [im.size for im in out] == [(w, h), (w, h), (h, w), (h, w),
                                               (w, h), (w, h), (h, w), (h, w)]
            assert sorted(int(a.sum()) for a in arrs) == [int(as_rgb.sum())] * 8


# ---------------------------------------------------------------------------
# augment_dataset
# ---------------------------------------------------------------------------

def test_augment_dataset_expands_each_tensor_eightfold_and_keeps_metadata(subtests):
    """Each (tensor, label, name) becomes 8 dihedral views; non-tensors raise TypeError."""
    a = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    b = torch.zeros(3, 4, 4)
    sq = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    small = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    cases = [
        ("two samples", [(a, 0, "a.png"), (b, 1, "b.png")], {}),
        ("single square", [(sq, 7, "x.png")], {}),
        ("grayscale flag", [(small, 0, "g.png")], dict(is_grayscale=True)),
        ("grayscale flag off", [(small, 0, "g.png")], dict(is_grayscale=False)),
        ("empty", [], {}),
    ]
    for name, dataset, kw in cases:
        with subtests.test(case=name):
            out = U.augment_dataset(dataset, **kw)
            assert len(out) == 8 * len(dataset)
            for i, (img, lbl, fn) in enumerate(dataset):
                block = out[8 * i: 8 * i + 8]
                assert [x[1] for x in block] == [lbl] * 8
                assert [x[2] for x in block] == [fn] * 8
                expected = []
                for k in (0, 1, 2, 3):
                    rot = torch.rot90(img, k, dims=(-2, -1))
                    expected += [rot, rot.flip(-1)]
                for t, want in zip((x[0] for x in block), expected):
                    assert isinstance(t, torch.Tensor)
                    assert t.dtype == torch.float32
                    assert t.shape == img.shape
                    assert torch.equal(t, want)
                    assert sorted(t.flatten().tolist()) == sorted(img.flatten().tolist())
            if name == "empty":
                assert out == []

    rejects = [
        ("numpy image", [(np.zeros((1, 4, 4), np.float32), 0, "bad.png")], "Expected torch.Tensor"),
        ("non-tensor after a valid one",
         [(torch.zeros(1, 3, 3), 0, "ok.png"), ([[0, 1], [2, 3]], 1, "bad.png")], None),
    ]
    for name, dataset, match in rejects:
        with subtests.test(case=name):
            with pytest.raises(TypeError, match=match):
                U.augment_dataset(dataset)


# ---------------------------------------------------------------------------
# augment_classes
# ---------------------------------------------------------------------------

def _fill(dirpath, n, prefix):
    os.makedirs(dirpath, exist_ok=True)
    names = []
    for i in range(n):
        name = f"{prefix}_{i:02d}.png"
        with open(os.path.join(dirpath, name), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        names.append(name)
    return names


def test_augment_classes_generates_then_splits_into_train_and_test(subtests, tmp_path, monkeypatch, capsys):
    """generate makes aug_nc/aug_pc (augmenting only as __main__); move splits 90/10."""
    with subtests.test(case="generate only, not main"):
        dst = str(tmp_path / "gen")
        assert U.augment_classes(dst, nc=["/nope/a.png"], pc=["/nope/b.png"],
                                 generate=True, move=False) is None
        for sub in ("aug_nc", "aug_pc"):
            assert os.path.isdir(os.path.join(dst, sub))
            assert os.listdir(os.path.join(dst, sub)) == []
        assert not os.path.exists(os.path.join(dst, "aug"))

    with subtests.test(case="generate as __main__ augments both classes"):
        dst = str(tmp_path / "main")
        calls = []
        nc = ["/src/nc1.png", "/src/nc2.png"]
        pc = ["/src/pc1.png"]
        with monkeypatch.context() as m:
            m.setattr(U, "augment_images", lambda file_paths, dst: calls.append((list(file_paths), dst)))
            m.setattr(U, "__name__", "__main__")
            U.augment_classes(dst, nc=nc, pc=pc, generate=True, move=False)
        assert calls == [(nc, os.path.join(dst, "aug_nc")), (pc, os.path.join(dst, "aug_pc"))]

    with subtests.test(case="move splits and reports"):
        dst = str(tmp_path / "move")
        nc_names = _fill(os.path.join(dst, "aug_nc"), 20, "nc")
        pc_names = _fill(os.path.join(dst, "aug_pc"), 10, "pc")
        capsys.readouterr()
        U.augment_classes(dst, nc=["x"] * 20, pc=["y"] * 10, generate=False,
                          move=True, group_by="cell")
        aug = os.path.join(dst, "aug")
        got = {(split, cls): sorted(os.listdir(os.path.join(aug, split, cls)))
               for split in ("train", "test") for cls in ("nc", "pc")}
        assert (len(got["train", "nc"]), len(got["test", "nc"])) == (18, 2)
        assert (len(got["train", "pc"]), len(got["test", "pc"])) == (9, 1)
        assert sorted(got["train", "nc"] + got["test", "nc"]) == sorted(nc_names)
        assert sorted(got["train", "pc"] + got["test", "pc"]) == sorted(pc_names)
        assert os.listdir(os.path.join(dst, "aug_nc")) == []
        assert os.listdir(os.path.join(dst, "aug_pc")) == []
        assert "Train nc: 18, Train pc:9, Test nc:2, Test pc:1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# MLR
# ---------------------------------------------------------------------------

def _mlr_df(n=150):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "gene": rng.choice(["g1", "g2", "g3"], n), "grna": rng.choice(["r1", "r2"], n),
        "plate": rng.choice(["p1", "p2"], n), "row": rng.choice(["A", "B"], n),
        "column": rng.choice(["1", "2"], n),
    })
    pred = rng.normal(0.0, 0.25, n)
    pred += (df["gene"] == "g2") * (df["grna"] == "r2") * 1.5
    pred -= (df["gene"] == "g3") * (df["grna"] == "r2") * 0.8
    df["pred"] = pred
    df.loc[0, "pred"] = 25.0
    df.loc[1, "pred"] = -25.0
    return df


def _two_guide_df():
    rng = np.random.default_rng(0)
    rows = []
    for gene in ("g1", "g2"):
        for grna in ("s1", "s2", "s3"):
            for rep in range(6):
                boost = 0.8 if (gene == "g2" and grna == "s2") else 0.0
                boost += 0.2 if (gene == "g2" and grna == "s3") else 0.0
                rows.append({"gene": gene, "grna": grna, "plate": f"p{rep % 2 + 1}",
                             "row": f"r{rep % 3 + 1}", "column": f"c{rep % 2 + 1}",
                             "pred": rng.normal(0.5 if gene == "g1" else 1.0, 0.1) + boost})
    return pd.DataFrame(rows)


def test_MLR_reports_each_genes_strongest_interaction(subtests, monkeypatch, capsys):
    """Per gene the largest |gene x gRNA| effect and its own p-value; table sorted desc."""
    import spacr.plot as P

    cases = [
        ("refined", _mlr_df, True, True, {"g2", "g3"}),
        ("unrefined", _mlr_df, False, True, None),
        ("real volcano plot", _mlr_df, False, False, None),
        ("two guides per gene", _two_guide_df, False, False, {"g2"}),
    ]
    for name, make, refine, spy, genes in cases:
        with subtests.test(case=name):
            df = make()
            seen = []
            capsys.readouterr()
            with monkeypatch.context() as m:
                if spy:
                    m.setattr(P, "_reg_v_plot", lambda *a, **k: seen.append((a, k)))
                result = U.MLR(df, refine_model=refine)
            assert len(result) == 4
            max_effects, max_pvalues, model, out = result
            printed = capsys.readouterr().out
            assert len(max_effects) > 0
            assert set(max_pvalues) == set(max_effects)
            # the real volcano plot adds its own -log10(p) column to the table
            assert list(out.columns) == (["effect", "p"] if spy else ["effect", "p", "-log10(p)"])
            assert out["effect"].is_monotonic_decreasing
            interactions = {k: v for k, v in model.params.items()
                            if "gene[T." in k and ":grna[T." in k}
            for gene in max_effects:
                terms = {k: v for k, v in interactions.items() if k.startswith(f"gene[T.{gene}]:")}
                strongest = max(terms, key=lambda k: abs(terms[k]))
                assert max_effects[gene] == pytest.approx(terms[strongest])
                assert max_pvalues[gene] == pytest.approx(model.pvalues[strongest])
                assert out.loc[gene, "effect"] == pytest.approx(terms[strongest])
            if genes is not None:
                assert set(max_effects) == genes
            if refine:
                assert int(model.nobs) < len(df)
                assert "Number of outliers detected by standardized residuals:" in printed
                assert "Durbin-Watson" in printed
                assert max_effects["g2"] > 0 > max_effects["g3"]
                assert list(out.index) == ["g2", "g3"]
                assert out.loc["g2", "p"] < 0.05
            else:
                assert int(model.nobs) == len(df)
            if name == "two guides per gene":
                assert len(interactions) == 2
            if spy:
                assert len(seen) == 1
                pd.testing.assert_frame_equal(seen[0][0][0], out)


# ---------------------------------------------------------------------------
# load_image / extract_features
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _write_png(path, value=255, size=(37, 53)):
    Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8), mode="L").save(path)
    return str(path)


def test_load_image_gives_a_normalised_1x3x224x224_batch(subtests, tmp_path):
    """Any PNG is resized to 224x224 RGB and ImageNet-normalised; missing files raise."""
    for value in (255, 0):
        with subtests.test(case=f"constant {value}"):
            tensor = U.load_image(_write_png(tmp_path / f"v{value}.png", value=value))
            assert isinstance(tensor, torch.Tensor)
            assert tuple(tensor.shape) == (1, 3, 224, 224)
            assert tensor.dtype == torch.float32
            expected = (value / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            got = tensor[0].reshape(3, -1).mean(dim=1)
            assert torch.allclose(got, expected, atol=1e-3)
    with subtests.test(case="missing file"):
        with pytest.raises(FileNotFoundError):
            U.load_image(str(tmp_path / "does_not_exist.png"))


class _TinyResNet(torch.nn.Module):
    """Stand-in for torchvision resnet: a trunk plus a classifier head."""

    def __init__(self, n_feat=5):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(3, n_feat, kernel_size=3, stride=4, padding=1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(n_feat, 2)

    def forward(self, x):
        return self.fc(torch.flatten(self.trunk(x), 1))


def test_extract_features_returns_one_headless_embedding_per_image(subtests, tmp_path):
    """The classifier head is dropped and a detached float32 row comes back per image."""
    cases = [
        ("three images", [255, 0, 128], 5, False),
        ("grad-tracking weights", [200], 4, True),
        ("no images", [], 5, False),
    ]
    for i, (name, values, n_feat, grad) in enumerate(cases):
        with subtests.test(case=name):
            paths = [_write_png(tmp_path / f"e{i}_{j}.png", value=v) for j, v in enumerate(values)]
            calls = []

            def factory(pretrained=False, n_feat=n_feat, grad=grad):
                calls.append(pretrained)
                torch.manual_seed(0)
                model = _TinyResNet(n_feat=n_feat)
                if grad:
                    for p in model.parameters():
                        p.requires_grad_(True)
                return model

            feats = U.extract_features(paths, resnet=factory)
            assert calls == [True]
            assert isinstance(feats, np.ndarray)
            if values:
                assert feats.shape == (len(values), n_feat)
                assert feats.dtype == np.float32
                assert np.isfinite(feats).all()
            else:
                assert feats.size == 0
            if len(values) > 1:
                assert not np.allclose(feats[0], feats[1])
