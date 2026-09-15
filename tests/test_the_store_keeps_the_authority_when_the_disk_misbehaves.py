"""ops_store's failure paths, each one driven rather than argued.

tests/test_the_object_table_is_on_disk_before_phase_c_starts.py holds the
contract when the disk behaves. These are the paths 372 PART 14-M wrote for
when it does not -- a frame Arrow cannot hold, a sidecar path taken by a
directory, a cache that is not parquet, a database that is not SQLite, keys
SQLite folds together, a table from before the key, and an append SQLite
itself refuses -- and in each one SQLite stays the authority and the caller
is told what happened.
"""
from __future__ import annotations

import logging
import os
import sqlite3

import pandas as pd
import pytest

from spacr.ops_store import (OBJECTS_TABLE, StoreError, objects_ready,
                             read_table, row_count, write_table)


def _db(tmp_path):
    return str(tmp_path / "measurements.db")


def _sidecar(tmp_path, table):
    return str(tmp_path / f"measurements.{table}.parquet")


def test_a_frame_parquet_cannot_hold_is_still_written_to_the_authority(
        tmp_path, caplog):
    """A column of mixed types is fine in SQLite and refused by Arrow."""
    path = _db(tmp_path)
    reads = pd.DataFrame({"object_id": [1, 2], "cycle": [1, 1],
                          "base": [1, "A"]})

    with caplog.at_level(logging.WARNING, logger="spacr.ops_store"):
        assert write_table(path, "ops_reads", reads) == 2

    assert not os.path.exists(_sidecar(tmp_path, "ops_reads"))
    assert "no parquet cache for ops_reads" in caplog.text
    assert len(read_table(path, "ops_reads")) == 2


def test_a_directory_where_the_sidecar_goes_costs_the_cache_not_the_write(
        tmp_path, caplog):
    """Neither parquet nor the stale-cache removal can use a directory.

    Both failures are logged, and the table is written and read anyway.
    """
    path = _db(tmp_path)
    os.makedirs(_sidecar(tmp_path, "ops_reads"))
    reads = pd.DataFrame({"object_id": [1, 2, 3], "cycle": [1, 1, 1],
                          "base": list("ACG")})

    with caplog.at_level(logging.WARNING, logger="spacr.ops_store"):
        assert write_table(path, "ops_reads", reads) == 3

    assert "could not remove the stale parquet cache" in caplog.text
    assert "no parquet cache for ops_reads" in caplog.text
    assert list(read_table(path, "ops_reads",
                           prefer_cache=False)["base"]) == list("ACG")


def test_an_unreadable_cache_falls_back_to_the_authority(tmp_path):
    path = _db(tmp_path)
    write_table(path, "ops_reads",
                pd.DataFrame({"object_id": [1, 2], "cycle": [1, 1],
                              "base": list("AC")}))
    with open(_sidecar(tmp_path, "ops_reads"), "wb") as handle:
        handle.write(b"not a parquet file")

    assert list(read_table(path, "ops_reads")["base"]) == ["A", "C"]


def test_a_file_that_is_not_a_database_has_no_object_table(tmp_path):
    """The gate answers "no table" rather than raising on a stray file."""
    path = _db(tmp_path)
    with open(path, "wb") as handle:
        handle.write(b"this is not sqlite " * 100)

    assert row_count(path, OBJECTS_TABLE) is None
    verdict = objects_ready(path)
    assert not verdict and verdict.rows is None
    assert "has no ops_objects table" in verdict.reason


def test_keys_sqlite_treats_as_equal_leave_no_table_behind(tmp_path):
    """1 and "1" are two keys to a DataFrame and one to a TEXT column.

    The frame check passes them; the index SQLite builds does not, and the
    table it was built on is removed rather than left for the gate to pass.
    """
    path = _db(tmp_path)
    frame = pd.DataFrame({"object_id": pd.Series([1, "1"], dtype=object),
                          "area": [300, 301]})

    with pytest.raises(StoreError, match="SQLite treats as equal"):
        write_table(path, OBJECTS_TABLE, frame)

    assert row_count(path, OBJECTS_TABLE) is None
    assert not objects_ready(path)


def test_a_table_written_before_the_key_is_named_before_an_append(tmp_path):
    """V11c's own case: a table on disk from before the constraint, holding a
    duplicate. The append is refused and the caller's table is left alone."""
    path = _db(tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ops_objects "
                     "(plate TEXT, well TEXT, object_id INTEGER)")
        conn.executemany("INSERT INTO ops_objects VALUES (?, ?, ?)",
                         [("p1", "A1", 1), ("p1", "A1", 1), ("p1", "A1", 2)])
    conn.close()
    new = pd.DataFrame({"plate": ["p1"], "well": ["A1"], "object_id": [3]})

    with pytest.raises(StoreError, match="without the uniqueness constraint"):
        write_table(path, OBJECTS_TABLE, new, if_exists="append")

    assert row_count(path, OBJECTS_TABLE) == 3


def test_an_append_with_columns_the_table_lacks_fails_as_what_it_is(tmp_path):
    """The table on disk has no plate or well to key on, so nothing is
    constrained before the append, and SQLite's refusal comes back as itself
    rather than as a collision between objects."""
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, pd.DataFrame({"object_id": [1, 2]}))
    extra = pd.DataFrame({"plate": ["p1"], "well": ["A1"], "object_id": [3]})

    with pytest.raises(Exception) as caught:
        write_table(path, OBJECTS_TABLE, extra, if_exists="append")

    assert not isinstance(caught.value, StoreError)
    assert "no column named plate" in (
        f"{caught.value} {caught.value.__cause__}")
    assert row_count(path, OBJECTS_TABLE) == 2
