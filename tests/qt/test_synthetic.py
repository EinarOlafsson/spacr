"""Tests for spacr.qt.synthetic — the demo-dataset generators.

Every test asserts the file layout matches what the corresponding
pipeline app expects: cellvoyager-named .tif images, populated
measurements.db, class-labeled PNG crops, etc.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt import synthetic as syn


# ---------------------------------------------------------------------------
# Filename builder
# ---------------------------------------------------------------------------

def test_cellvoyager_filename_matches_regex():
    from spacr.utils import _get_regex
    regex = _get_regex("cellvoyager", "tif")
    fn = syn.cellvoyager_filename(
        plate="plate1", well="A01", time=1, field=2, chan=3,
    )
    m = re.match(regex, fn)
    assert m is not None, f"{fn!r} did not match cellvoyager regex"
    assert m.group("plateID") == "plate1"
    assert m.group("wellID") == "A01"
    assert m.group("chanID") == "03"


# ---------------------------------------------------------------------------
# Mask demo
# ---------------------------------------------------------------------------

def test_mask_demo_files_and_settings(tmp_path: Path):
    layout = syn.generate_mask_demo(
        tmp_path, wells=("A01",), fields=2, channels=(0, 1, 2, 3),
    )
    # 1 well × 2 fields × 4 channels = 8 files
    assert len(layout.image_files) == 8
    for p in layout.image_files:
        assert p.exists()
        arr = tifffile.imread(p)
        assert arr.dtype == np.uint16
        assert arr.shape == (256, 256)
    assert layout.settings_csv is not None
    assert layout.settings_csv.exists()

    # CSV parses cleanly with the same reader spacr.utils.load_settings uses
    with open(layout.settings_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Key", "Value"]
    settings = {k: v for k, v in rows[1:]}
    assert settings["metadata_type"] == "cellvoyager"
    assert settings["src"] == str(layout.src)
    assert int(settings["cell_channel"]) == syn.CHANNEL_LAYOUT["cell_channel"]


def test_mask_demo_settings_can_be_reloaded_by_spacr(tmp_path: Path):
    from spacr.utils import load_settings
    layout = syn.generate_mask_demo(tmp_path)
    loaded = load_settings(
        str(layout.settings_csv),
        setting_key="Key", setting_value="Value",
    )
    assert isinstance(loaded, dict)
    assert loaded["src"] == str(layout.src)
    assert loaded["metadata_type"] == "cellvoyager"


# ---------------------------------------------------------------------------
# Measure demo
# ---------------------------------------------------------------------------

def test_measure_demo_has_masks_and_db(tmp_path: Path):
    layout = syn.generate_measure_demo(
        tmp_path, wells=("A01",), fields=1, channels=(0, 1),
    )
    assert layout.mask_files, "measure demo must include a masks/ folder"
    assert (layout.src / "masks").is_dir()
    assert layout.db_path is not None and layout.db_path.exists()
    # measurements.db has an empty png_list waiting for cropped rows
    with sqlite3.connect(layout.db_path) as conn:
        cols = {row[1] for row in conn.execute('PRAGMA table_info("png_list")')}
    assert "png_path" in cols


def test_crop_demo_has_data_dir(tmp_path: Path):
    layout = syn.generate_crop_demo(tmp_path, wells=("A01",),
                                      fields=1, channels=(0, 1))
    assert (layout.src / "data").is_dir()
    assert layout.db_path is not None


# ---------------------------------------------------------------------------
# Classify demo
# ---------------------------------------------------------------------------

def test_classify_demo_produces_labeled_pngs(tmp_path: Path):
    layout = syn.generate_classify_demo(tmp_path, n_crops=8)
    assert len(layout.image_files) == 8
    for p in layout.image_files:
        assert p.suffix == ".png"
    # DB has annotate column with alternating 1/2 labels
    with sqlite3.connect(layout.db_path) as conn:
        rows = conn.execute(
            'SELECT annotate FROM "png_list"'
        ).fetchall()
    assert sorted(v for (v,) in rows) == [1, 1, 1, 1, 2, 2, 2, 2]


def test_classify_demo_settings_carry_annotation_column(tmp_path: Path):
    layout = syn.generate_classify_demo(tmp_path, n_crops=4)
    with open(layout.settings_csv) as f:
        settings = {k: v for k, v in csv.reader(f)}
    assert settings["annotation_column"] == "annotate"


# ---------------------------------------------------------------------------
# Timelapse demo
# ---------------------------------------------------------------------------

def test_timelapse_demo_has_multiple_frames_per_well(tmp_path: Path):
    layout = syn.generate_timelapse_demo(
        tmp_path, wells=("A01",), fields=1, times=6, channels=(0, 1),
    )
    # 1 well × 1 field × 6 times × 2 channels = 12 files
    assert len(layout.image_files) == 12
    # Every filename should carry T01..T06
    times = {
        int(re.search(r"_T(\d+)F", p.name).group(1))
        for p in layout.image_files
    }
    assert times == {1, 2, 3, 4, 5, 6}
    with open(layout.settings_csv) as f:
        settings = {k: v for k, v in csv.reader(f)}
    assert settings["timelapse"] == "True"


# ---------------------------------------------------------------------------
# save_settings_csv + demo_settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", ["mask", "measure", "crop",
                                       "classify", "timelapse"])
def test_demo_settings_include_src(app_key, tmp_path: Path):
    settings = syn.demo_settings(app_key, str(tmp_path))
    assert settings["src"] == str(tmp_path)


def test_save_settings_csv_roundtrip(tmp_path: Path):
    settings = {"src": str(tmp_path), "n": 42, "flag": True, "opt": None}
    p = syn.save_settings_csv(tmp_path / "s.csv", settings)
    with open(p) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Key", "Value"]
    kv = {k: v for k, v in rows[1:]}
    assert kv["src"] == str(tmp_path)
    assert kv["n"] == "42"
    assert kv["flag"] == "True"
    assert kv["opt"] == ""     # None → empty string


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_generates_all_demos(tmp_path: Path):
    rc = syn.main(["all", str(tmp_path)])
    assert rc == 0
    for name in ("mask", "measure", "crop", "classify", "timelapse"):
        assert (tmp_path / name).is_dir(), f"missing {name} demo dir"
