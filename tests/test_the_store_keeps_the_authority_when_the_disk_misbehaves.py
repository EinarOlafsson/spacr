"""ops_store's failure paths, each one driven rather than argued.

tests/test_the_object_table_is_on_disk_before_phase_c_starts.py holds the
contract when the disk behaves. These are the paths 372 PART 14-M wrote for
when it does not, as they stand since item 0an moved the OPS tables into the
run's spaCR store: a frame Arrow cannot hold, a stray file where the
database was, keys a frame keeps apart but a column cannot, an older
database holding a duplicate, and an append that adds a cycle. In each one
nothing half-written becomes visible and the caller is told what happened.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.ops_store import (OBJECTS_TABLE, StoreError, objects_ready,
                             read_table, row_count, write_table)


def _db(tmp_path):
    return str(tmp_path / "measurements.db")


def test_a_frame_arrow_cannot_hold_is_refused_and_keeps_the_table(tmp_path):
    """A column of mixed types cannot be one typed column; nothing is written."""
    path = _db(tmp_path)
    good = pd.DataFrame({"object_id": [1, 2], "cycle": [1, 1],
                         "base": ["A", "C"]})
    assert write_table(path, "ops_reads", good) == 2
    mixed = pd.DataFrame({"object_id": [1, 2], "cycle": [1, 1],
                          "base": [1, "A"]})

    with pytest.raises(Exception) as caught:
        write_table(path, "ops_reads", mixed)

    assert not isinstance(caught.value, StoreError)
    assert list(read_table(path, "ops_reads")["base"]) == ["A", "C"]


def test_a_file_that_is_not_a_database_has_no_object_table(tmp_path):
    """The gate answers "no store" rather than raising on a stray file."""
    path = _db(tmp_path)
    with open(path, "wb") as handle:
        handle.write(b"this is not sqlite " * 100)

    assert row_count(path, OBJECTS_TABLE) is None
    verdict = objects_ready(path)
    assert not verdict and verdict.rows is None
    assert "no store" in verdict.reason


def test_keys_a_column_cannot_tell_apart_leave_no_table_behind(tmp_path):
    """1 and "1" are two keys to a DataFrame but cannot share one column.

    The frame check passes them; the store's writer refuses the column, and
    no table is left for the gate to pass.
    """
    path = _db(tmp_path)
    frame = pd.DataFrame({"object_id": pd.Series([1, "1"], dtype=object),
                          "area": [300, 301]})

    with pytest.raises(Exception):
        write_table(path, OBJECTS_TABLE, frame)

    assert row_count(path, OBJECTS_TABLE) is None
    assert not objects_ready(path)


def test_an_older_table_holding_a_duplicate_is_named_before_an_append(
        tmp_path):
    """V11c's own case: a measurements.db from before the constraint, holding
    a duplicate. The append is refused, nothing is copied into the store and
    the older table is left alone."""
    path = _db(tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ops_objects "
                     "(plate TEXT, well TEXT, object_id INTEGER)")
        conn.executemany("INSERT INTO ops_objects VALUES (?, ?, ?)",
                         [("p1", "A1", 1), ("p1", "A1", 1), ("p1", "A1", 2)])
    conn.close()
    new = pd.DataFrame({"plate": ["p1"], "well": ["A1"], "object_id": [3]})

    with pytest.raises(StoreError, match="would hold 2 rows"):
        write_table(path, OBJECTS_TABLE, new, if_exists="append")

    assert row_count(path, OBJECTS_TABLE) == 3


def test_an_older_table_is_copied_into_the_store_before_one_well_is_rewritten(
        tmp_path):
    """Rewriting one well of an older run keeps the other wells it held."""
    from spacr.ops_engine import _replace_well_rows, _well_rows

    path = _db(tmp_path)
    with sqlite3.connect(path) as conn:
        pd.DataFrame({"plate": ["p1"] * 3, "well": ["A1", "A2", "A2"],
                      "object_id": [1, 1, 2]}).to_sql(OBJECTS_TABLE, conn,
                                                      index=False)
    conn.close()
    _replace_well_rows(path, OBJECTS_TABLE, pd.DataFrame(
        {"plate": ["p1"], "well": ["A1"], "object_id": [7]}), "p1", "A1")

    assert row_count(path, OBJECTS_TABLE) == 3
    assert _well_rows(path, OBJECTS_TABLE, "p1", "A1")["object_id"].tolist() == [7]
    assert _well_rows(path, OBJECTS_TABLE, "p1", "A2")["object_id"].tolist() == [1, 2]


def test_appending_a_cycle_keeps_all_reads(tmp_path):
    """A newly appended cycle joins the reads already stored for the well."""
    path = _db(tmp_path)
    first = pd.DataFrame({"object_id": [11, 12], "cycle": [1, 1],
                          "base": ["A", "C"]})
    second = pd.DataFrame({"object_id": [11, 12], "cycle": [2, 2],
                           "base": ["G", "T"]})
    assert write_table(path, "ops_reads", first) == 2
    assert write_table(path, "ops_reads", second, if_exists="append") == 4

    expected = [(11, 1, "A"), (12, 1, "C"),
                (11, 2, "G"), (12, 2, "T")]
    recovered = read_table(path, "ops_reads").sort_values(
        ["cycle", "object_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        recovered, pd.DataFrame(expected, columns=["object_id", "cycle", "base"]))


def test_a_refused_append_preserves_prior_objects(tmp_path):
    """An append whose objects the table already holds loses nothing."""
    path = _db(tmp_path)
    original = pd.DataFrame({"object_id": ["prior"], "area": [307]})
    assert write_table(path, OBJECTS_TABLE, original) == 1
    incoming = pd.DataFrame({"object_id": ["prior", "next"],
                             "area": [401, 402]})

    with pytest.raises(StoreError, match="already holds 1 of the 2"):
        write_table(path, OBJECTS_TABLE, incoming, if_exists="append")

    assert row_count(path, OBJECTS_TABLE) == 1
    verdict = objects_ready(path)
    assert verdict and verdict.rows == 1
    pd.testing.assert_frame_equal(read_table(path, OBJECTS_TABLE), original)
