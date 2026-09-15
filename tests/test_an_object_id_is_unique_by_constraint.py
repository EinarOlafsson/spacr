"""One row per object in `ops_objects` and `ops_barcodes` is a constraint.

MEASURED BEFORE THE FIX (372 PART 14-L, V11c): the storage contract says
``UNIQUE(plate, well, object_id)``, and the table on disk had no constraint
and no index. The first real well's ids were unique only because `number`
made them so; appending the same well a second time would have doubled every
object without an error, and every later join on the id would have counted
each cell twice.

A duplicate is refused before anything is written, whether it sits inside
one frame or between a frame and the table it is appended to, and the
refusal names the keys.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.ops_store import (OBJECTS_TABLE, StoreError, objects_ready,
                             read_table, row_count, write_table)


def _objects(n=5, well="A1", plate="p1", start=1):
    """An ops_objects frame of ``n`` objects numbered from ``start``."""
    return pd.DataFrame({
        "plate": [plate] * n, "well": [well] * n,
        "object_id": list(range(start, start + n)),
        "centroid_x": [10.0 * i for i in range(n)],
        "centroid_y": [20.0 * i for i in range(n)],
        "area": [300 + i for i in range(n)],
    })


def _distinct(path, table=OBJECTS_TABLE):
    """How many distinct (plate, well, object_id) keys the table holds."""
    with sqlite3.connect(path) as conn:
        return conn.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT plate, well, object_id "
            f"FROM {table})").fetchone()[0]


def test_a_frame_with_duplicate_ids_is_refused_and_writes_nothing(tmp_path):
    path = str(tmp_path / "measurements.db")
    frame = _objects(5)
    frame.loc[4, "object_id"] = 2

    with pytest.raises(StoreError, match="object_id=2"):
        write_table(path, OBJECTS_TABLE, frame)

    assert row_count(path, OBJECTS_TABLE) is None
    assert not objects_ready(path)


def test_appending_the_same_well_twice_is_refused(tmp_path):
    path = str(tmp_path / "measurements.db")
    write_table(path, OBJECTS_TABLE, _objects(5))

    with pytest.raises(StoreError, match="already"):
        write_table(path, OBJECTS_TABLE, _objects(5), if_exists="append")

    assert row_count(path, OBJECTS_TABLE) == 5
    assert _distinct(path) == 5


def test_an_append_that_overlaps_by_one_id_is_refused_whole(tmp_path):
    """Atomic: the four new rows do not land beside the one duplicate."""
    path = str(tmp_path / "measurements.db")
    write_table(path, OBJECTS_TABLE, _objects(5))

    with pytest.raises(StoreError, match="object_id=5"):
        write_table(path, OBJECTS_TABLE, _objects(5, start=5),
                    if_exists="append")

    assert row_count(path, OBJECTS_TABLE) == 5


def test_two_wells_with_overlapping_id_ranges_are_accepted(tmp_path):
    """Ids are numbered per well, so the key is the well and the id together."""
    path = str(tmp_path / "measurements.db")
    write_table(path, OBJECTS_TABLE, _objects(5, well="A1"))
    write_table(path, OBJECTS_TABLE, _objects(5, well="A2"),
                if_exists="append")

    assert row_count(path, OBJECTS_TABLE) == 10
    assert _distinct(path) == 10
    assert objects_ready(path)


def test_the_constraint_is_in_the_schema_and_survives_a_rewrite(tmp_path):
    """A replace drops the table's indexes with it, so the key is rebuilt."""
    path = str(tmp_path / "measurements.db")
    write_table(path, OBJECTS_TABLE, _objects(5))
    write_table(path, OBJECTS_TABLE, _objects(5))

    with sqlite3.connect(path) as conn:
        indexes = [sql for (sql,) in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND tbl_name=?", (OBJECTS_TABLE,))]
    assert any("UNIQUE" in sql.upper() and "plate" in sql
               and "well" in sql and "object_id" in sql for sql in indexes)
    with pytest.raises(StoreError):
        write_table(path, OBJECTS_TABLE, _objects(1), if_exists="append")


def test_a_replace_with_duplicates_keeps_the_table_it_would_have_replaced(
        tmp_path):
    path = str(tmp_path / "measurements.db")
    write_table(path, OBJECTS_TABLE, _objects(5))
    bad = _objects(6)
    bad.loc[5, "object_id"] = 1

    with pytest.raises(StoreError):
        write_table(path, OBJECTS_TABLE, bad)

    verdict = objects_ready(path)
    assert verdict and verdict.rows == 5
    assert list(read_table(path, OBJECTS_TABLE)["object_id"]) == [1, 2, 3, 4, 5]


def test_without_plate_and_well_the_id_alone_is_the_key(tmp_path):
    path = str(tmp_path / "measurements.db")
    frame = pd.DataFrame({"object_id": [1, 2, 2], "centroid_x": [1.0, 2, 3]})

    with pytest.raises(StoreError, match="object_id=2"):
        write_table(path, OBJECTS_TABLE, frame)

    write_table(path, OBJECTS_TABLE, frame.iloc[:2])
    with pytest.raises(StoreError):
        write_table(path, OBJECTS_TABLE, frame.iloc[1:2], if_exists="append")


def test_one_barcode_row_per_object(tmp_path):
    path = str(tmp_path / "measurements.db")
    frame = _objects(3)
    frame["barcode"] = ["ACGT", "TTGA", "GGCA"]
    write_table(path, "ops_barcodes", frame)

    with pytest.raises(StoreError, match="ops_barcodes"):
        write_table(path, "ops_barcodes", frame.iloc[:1], if_exists="append")
    assert row_count(path, "ops_barcodes") == 3


def test_reads_may_repeat_an_object(tmp_path):
    """Many reads per object is the point of ops_reads; it carries no key."""
    path = str(tmp_path / "measurements.db")
    reads = pd.DataFrame({"object_id": [1, 1, 1, 2], "cycle": [1, 2, 3, 1],
                          "base": list("ACGT")})

    assert write_table(path, "ops_reads", reads) == 4
