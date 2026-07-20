"""
Tests for spacr.io — the sqlite/database boundary.

Full pipelines like preprocess_img_data / concatenate_and_normalize are
covered elsewhere; here we focus on the pure/testable helpers.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spacr.io as IO


# ---------------------------------------------------------------------------
# _create_database — creates a valid, openable sqlite file
# ---------------------------------------------------------------------------

def test_create_database_produces_valid_sqlite(tmp_path):
    db = tmp_path / "new.db"
    IO._create_database(str(db))
    assert db.exists()
    # sqlite3 header magic is: "SQLite format 3\x00" but a freshly-created
    # empty database may have a 0-byte file until first write. Instead,
    # verify it can be opened and queried.
    conn = sqlite3.connect(db)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        # Empty database: no tables. Assertion succeeds because the
        # connection worked at all.
        assert isinstance(tables, list)
    finally:
        conn.close()


def test_create_database_idempotent(tmp_path):
    db = tmp_path / "twice.db"
    IO._create_database(str(db))
    IO._create_database(str(db))
    assert db.exists()


# ---------------------------------------------------------------------------
# _read_db — reads existing tables, quotes identifiers safely
# ---------------------------------------------------------------------------

def test_read_db_returns_one_frame_per_table(synth_sqlite_db, capsys):
    frames = IO._read_db(str(synth_sqlite_db), ["cell"])
    assert isinstance(frames, list)
    assert len(frames) == 1
    df = frames[0]
    assert isinstance(df, pd.DataFrame)
    # Rows should match the synth fixture (40 measurements).
    assert len(df) == 40


def test_read_db_raises_on_missing_table(synth_sqlite_db, capsys):
    with pytest.raises(ValueError):
        IO._read_db(str(synth_sqlite_db), ["not_a_real_table"])


def test_read_db_handles_multiple_tables(synth_sqlite_db, capsys):
    # png_list is present in the fixture db (see conftest.synth_sqlite_db).
    frames = IO._read_db(str(synth_sqlite_db), ["cell", "png_list"])
    assert len(frames) == 2
    assert isinstance(frames[0], pd.DataFrame)
    assert isinstance(frames[1], pd.DataFrame)


# ---------------------------------------------------------------------------
# delete_empty_subdirectories
# ---------------------------------------------------------------------------

def test_delete_empty_subdirectories_removes_only_empties(tmp_path):
    (tmp_path / "empty_a").mkdir()
    (tmp_path / "empty_b").mkdir()
    non_empty = tmp_path / "with_file"
    non_empty.mkdir()
    (non_empty / "keep.txt").write_text("hello")

    IO.delete_empty_subdirectories(str(tmp_path))

    assert not (tmp_path / "empty_a").exists()
    assert not (tmp_path / "empty_b").exists()
    assert non_empty.exists()
    assert (non_empty / "keep.txt").exists()


def test_delete_empty_subdirectories_noop_when_all_populated(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "f.txt").write_text("x")
    IO.delete_empty_subdirectories(str(tmp_path))
    assert d.exists()


# ---------------------------------------------------------------------------
# _is_dir_empty
# ---------------------------------------------------------------------------

def test_is_dir_empty_true(tmp_path):
    empty = tmp_path / "e"
    empty.mkdir()
    assert IO._is_dir_empty(str(empty)) is True


def test_is_dir_empty_false(tmp_path):
    d = tmp_path / "with"
    d.mkdir()
    (d / "f").write_text("")
    assert IO._is_dir_empty(str(d)) is False


# ---------------------------------------------------------------------------
# _get_avg_object_size — trivial arithmetic on synthetic label masks
# ---------------------------------------------------------------------------

def test_get_avg_object_size_returns_count_and_size(synth_mask_2d):
    """The helper returns (num_objects, avg_size) — both positive on a
    populated mask."""
    v = IO._get_avg_object_size([synth_mask_2d])
    assert isinstance(v, tuple) and len(v) == 2
    n, avg = v
    assert n > 0
    assert avg > 0


def test_get_avg_object_size_empty_masks_zero(capsys):
    empty = np.zeros((16, 16), dtype=np.int32)
    v = IO._get_avg_object_size([empty])
    n, avg = v
    # Zero objects; avg may be 0 or NaN — either is acceptable.
    assert n == 0
    assert avg == 0 or np.isnan(avg)
