"""Coverage-fill batch 7 for spacr.utils settings/df/file helpers."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# load_settings — every parse_value branch
# ---------------------------------------------------------------------------

def test_load_settings_all_types(tmp_path):
    csv = tmp_path / "s.csv"
    pd.DataFrame({
        "setting_key": ["flag_t", "flag_f", "empty", "lst", "tup",
                        "dct", "n_int", "n_float", "text", "bad_lit"],
        "setting_value": ["True", "False", "", "[1, 2]", "(3, 4)",
                          "{'a': '1'}", "42", "3.14", "hello", "[unclosed"],
    }).to_csv(csv, index=False)
    out = U.load_settings(str(csv))
    assert out["flag_t"] is True and out["flag_f"] is False
    assert out["empty"] is None
    assert out["lst"] == [1, 2] and out["tup"] == (3, 4)
    assert out["dct"] == {"a": 1}          # nested value parsed
    assert out["n_int"] == 42 and out["n_float"] == 3.14
    assert out["text"] == "hello"
    assert out["bad_lit"] == "[unclosed"   # unparseable literal → as-is


def test_load_settings_missing_columns(tmp_path):
    csv = tmp_path / "s.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        U.load_settings(str(csv))


# ---------------------------------------------------------------------------
# save_settings  (+ round-trip via explicit column names)
# ---------------------------------------------------------------------------

def test_save_settings_str_src(tmp_path):
    settings = {"src": str(tmp_path), "diameter": 30,
                "test_mode": True, "plot": True}
    U.save_settings(settings, name="exp")
    out = tmp_path / "settings" / "exp.csv"
    assert out.exists()
    # test_mode/plot forced False in the saved copy
    reload = U.load_settings(str(out), setting_key="Key", setting_value="Value")
    assert reload["test_mode"] is False and reload["plot"] is False
    assert reload["diameter"] == 30


def test_save_settings_list_src(tmp_path):
    d1 = tmp_path / "a"; d1.mkdir()
    settings = {"src": [str(d1), "/other"], "diameter": 30}
    U.save_settings(settings, name="exp")
    assert (d1 / "settings" / "exp_list.csv").exists()


# ---------------------------------------------------------------------------
# _group_by_well
# ---------------------------------------------------------------------------

def test_group_by_well():
    df = pd.DataFrame({
        "plateID": ["p1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c1", "c1", "c1"],
        "value": [10.0, 20.0, 30.0, 40.0],
        "label": ["a", "b", "c", "d"],
    })
    out = U._group_by_well(df)
    # r1/c1 numeric mean = 15, non-numeric first = 'a'
    assert out.loc[("p1", "r1", "c1"), "value"] == 15.0


# ---------------------------------------------------------------------------
# annotate_predictions
# ---------------------------------------------------------------------------

def test_annotate_predictions(tmp_path):
    csv = tmp_path / "preds.csv"
    pd.DataFrame({
        "path": [
            "/x/1_A05_1_1.png",   # plate1 col5 → screen
            "/x/5_B06_1_2.png",   # plate5 col6 → pc
            "/x/1_C02_1_3.png",   # col2 → nc
        ],
        "pred": [0.9, 0.1, 0.5],
    }).to_csv(csv, index=False)
    out = U.annotate_predictions(str(csv))
    conds = set(out["cond"])
    assert "screen" in conds and "pc" in conds and "nc" in conds


# ---------------------------------------------------------------------------
# get_files_from_dir / _find_similar_sized_images
# ---------------------------------------------------------------------------

def test_get_files_from_dir(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "b.tif").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    tifs = U.get_files_from_dir(str(tmp_path), "*.tif")
    assert len(tifs) == 2


def test_find_similar_sized_images(tmp_path):
    paths = []
    for i in range(3):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[4:12, 4:12] = 200          # same non-zero bbox size across all
        p = str(tmp_path / f"img{i}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    # one odd-sized image
    odd = np.zeros((20, 20, 3), dtype=np.uint8); odd[2:18, 2:6] = 200
    p_odd = str(tmp_path / "odd.png"); cv2.imwrite(p_odd, odd)
    paths.append(p_odd)
    group = U._find_similar_sized_images(paths)
    assert len(group) == 3   # the 3 same-sized crops
