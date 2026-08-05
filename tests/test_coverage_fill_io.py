"""Coverage-fill for spacr.io pure-logic helpers (no GPU)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import io as IO


# ---------------------------------------------------------------------------
# _read_mask / _read_db
# ---------------------------------------------------------------------------

def test_read_mask_tif(tmp_path):
    import tifffile
    p = tmp_path / "m.tif"
    tifffile.imwrite(str(p), np.ones((8, 8), dtype=np.uint16))
    out = IO._read_mask(str(p))
    assert out.dtype == np.uint16 and out.shape == (8, 8)


def test_read_db_returns_list(tmp_path):
    db = tmp_path / "m.db"
    with sqlite3.connect(str(db)) as conn:
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_sql(
            "cell", conn, index=False)
    out = IO._read_db(str(db), tables=["cell"])
    assert isinstance(out, list) and len(out) == 1
    assert len(out[0]) == 2


# ---------------------------------------------------------------------------
# _get_avg_object_size
# ---------------------------------------------------------------------------

def test_get_avg_object_size():
    m = np.zeros((10, 10), dtype=np.int32)
    m[1:4, 1:4] = 1
    m[6:8, 6:8] = 2
    avg_n, avg_size = IO._get_avg_object_size([m])
    assert avg_n >= 1 and avg_size > 0


def test_get_avg_object_size_empty():
    avg_n, avg_size = IO._get_avg_object_size([np.zeros((5, 5), dtype=np.int32)])
    assert avg_n == 0


# ---------------------------------------------------------------------------
# _generate_time_lists
# ---------------------------------------------------------------------------

def test_generate_time_lists_groups_by_field():
    files = [
        "plate1_A01_1_0.npy", "plate1_A01_1_2.npy", "plate1_A01_1_1.npy",
        "plate1_A02_1_0.npy",
        "malformed.npy",           # skipped (too few parts)
        "plate1_A01_1_bad.npy",    # skipped (bad timepoint)
    ]
    out = IO._generate_time_lists(files)
    # Two groups: (plate1,A01,1) and (plate1,A02,1).
    assert len(out) == 2
    # The A01 group is sorted by timepoint.
    a01 = [g for g in out if len(g) == 3][0]
    assert a01[0].endswith("_0.npy")


# ---------------------------------------------------------------------------
# parse_gz_files
# ---------------------------------------------------------------------------

def test_parse_gz_files(tmp_path):
    for name in ("s1_R1_001.fastq.gz", "s1_R2_001.fastq.gz",
                 "s2_R1_001.fastq.gz"):
        (tmp_path / name).write_bytes(b"")
    out = IO.parse_gz_files(str(tmp_path))
    assert out["s1"]["R1"].endswith("s1_R1_001.fastq.gz")
    assert out["s1"]["R2"].endswith("s1_R2_001.fastq.gz")
    assert "R1" in out["s2"]


# ---------------------------------------------------------------------------
# convert_numpy_to_tiff
# ---------------------------------------------------------------------------

def test_convert_numpy_to_tiff(tmp_path):
    np.save(str(tmp_path / "a.npy"),
            np.ones((8, 8), dtype=np.uint16))
    IO.convert_numpy_to_tiff(str(tmp_path))
    tiffs = list(tmp_path.rglob("*.tif")) + list(tmp_path.rglob("*.tiff"))
    assert tiffs


# ---------------------------------------------------------------------------
# _read_and_join_tables
# ---------------------------------------------------------------------------

def test_read_and_join_tables(tmp_path):
    db = tmp_path / "m.db"
    with sqlite3.connect(str(db)) as conn:
        pd.DataFrame({"prcfo": ["a", "b"], "cell_area": [1.0, 2.0]}
                     ).to_sql("cell", conn, index=False)
    out = IO._read_and_join_tables(str(db), table_names=["cell"])
    assert out is None or isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# _save_figure + _save_settings_to_db
# ---------------------------------------------------------------------------

def test_save_figure(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    # _save_figure writes to dirname(src)/figure/ — give a nested src
    # so the PDF lands inside tmp_path.
    src = str(tmp_path / "cell" / "objroot")
    IO._save_figure(fig, src, text="test")
    assert list(tmp_path.rglob("*.pdf"))


def test_save_settings_to_db(tmp_path):
    """The settings really land in ``<plate>/measurements/measurements.db``.

    The old shape -- call it, swallow anything, skip -- passed on a machine
    where the function wrote no database at all. It writes two tables and the
    difference between them is the whole point of the function, so both are
    checked: ``settings`` is replaced by the latest run, ``settings_history``
    accumulates one tagged row set per run.
    """
    import sqlite3

    src = tmp_path / "plate1" / "images"
    src.mkdir(parents=True)
    db = tmp_path / "plate1" / "measurements" / "measurements.db"

    IO._save_settings_to_db({"src": str(src), "diameter": 30, "plot": False},
                            stage="measure_crop")
    assert db.exists(), "no measurements.db was written"

    with sqlite3.connect(db) as conn:
        latest = dict(conn.execute(
            f'SELECT setting_key, setting_value FROM "{IO.SETTINGS_TABLE}"'
        ).fetchall())
        history = conn.execute(
            f'SELECT run_id, stage, setting_key, setting_value '
            f'FROM "{IO.SETTINGS_HISTORY_TABLE}"').fetchall()

    # Values are stringified on the way in -- that is why resume compares text.
    assert latest["diameter"] == "30"
    assert latest["plot"] == "False"
    assert latest["src"] == str(src)
    assert {stage for _rid, stage, _k, _v in history} == {"measure_crop"}
    assert len({rid for rid, *_ in history}) == 1

    # A second stage replaces `settings` and APPENDS to the history: the
    # repair this function exists for. Without both assertions an implementation
    # that only ever wrote `settings` would pass.
    IO._save_settings_to_db({"src": str(src), "diameter": 99},
                            stage="preprocess")
    with sqlite3.connect(db) as conn:
        latest = dict(conn.execute(
            f'SELECT setting_key, setting_value FROM "{IO.SETTINGS_TABLE}"'
        ).fetchall())
        stages = [row[0] for row in conn.execute(
            f'SELECT DISTINCT stage FROM "{IO.SETTINGS_HISTORY_TABLE}"')]
    assert latest["diameter"] == "99" and "plot" not in latest
    assert set(stages) == {"measure_crop", "preprocess"}
