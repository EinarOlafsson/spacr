"""372 Phase D and the B5 gate: where the tables live, and when C may start.

Phase D's contract has one sentence that decides the whole design -- sqlite is
authoritative, parquet is a cache, "written in the same step, with the row
counts asserted equal, so the sidecar cannot silently drift from the
authority". Two stores that can disagree need one of them to be right BY
DEFINITION, or a reader that picked the faster one would sometimes return
yesterday's numbers with no way to tell.

B5 is the gate that keeps Phase C from starting early: "no sequencing or
phenotype channel is read until `ops_objects` is on disk and its row count
checked."
"""

import os
import sqlite3

import pandas as pd
import pytest

from spacr.ops_store import (
    CACHED_TABLES, OBJECTS_TABLE, OPS_TABLES, StoreError,
    objects_ready, read_table, row_count, write_table,
)


def _objects(n=5):
    return pd.DataFrame({
        "plate": ["p1"] * n, "well": ["A1"] * n,
        "object_id": list(range(1, n + 1)),
        "centroid_x": [10.0 * i for i in range(n)],
        "centroid_y": [20.0 * i for i in range(n)],
        "area": [300 + i for i in range(n)],
    })


def _reads(n=6):
    return pd.DataFrame({
        "object_id": list(range(1, n + 1)),
        "cycle": [1] * n,
        "channel_0": [0.5] * n, "channel_1": [0.7] * n,
        "base": list("ACGTAC")[:n], "quality": [30.0] * n,
    })


def _db(tmp_path):
    return str(tmp_path / "measurements.db")


# -- B5, the gate ----------------------------------------------------------

def test_no_database_is_refused_with_the_reason(tmp_path):
    verdict = objects_ready(str(tmp_path / "absent.db"))
    assert not verdict
    assert "no database" in verdict.reason


def test_a_database_without_the_table_is_refused_by_name(tmp_path):
    path = _db(tmp_path)
    sqlite3.connect(path).close()
    verdict = objects_ready(path)

    assert not verdict
    assert OBJECTS_TABLE in verdict.reason
    assert verdict.rows is None


def test_an_empty_object_table_does_not_pass_the_gate(tmp_path):
    """The half that 'on disk' alone would miss.

    An empty ops_objects makes every later phase produce nothing while
    appearing to run -- results that look finished and are not.
    """
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, _objects(0))
    verdict = objects_ready(path)

    assert not verdict
    assert verdict.rows == 0
    assert "look like a finished run" in verdict.reason


def test_a_populated_table_opens_the_gate(tmp_path):
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, _objects(5))
    verdict = objects_ready(path)

    assert verdict
    assert bool(verdict) is True
    assert verdict.rows == 5
    assert verdict.reason == ""


def test_a_caller_who_knows_their_plate_can_raise_the_floor(tmp_path):
    """The default floor is 1 because a one-nucleus well is bad, not impossible."""
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, _objects(5))

    assert objects_ready(path, minimum=5)
    assert not objects_ready(path, minimum=6)


# -- Phase D, the two stores ----------------------------------------------

def test_a_cached_table_is_written_to_both_stores(tmp_path):
    path = _db(tmp_path)
    written = write_table(path, "ops_reads", _reads(6))

    assert written == 6
    sidecar = str(tmp_path / "measurements.ops_reads.parquet")
    assert os.path.exists(sidecar), "the parquet cache was not written"
    assert row_count(path, "ops_reads") == 6


def test_an_uncached_table_gets_no_sidecar(tmp_path):
    """Only the wide numeric tables are cached; the rest are read whole.

    A cache for a small table is two things to keep in step for no gain.
    """
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, _objects(5))

    assert OBJECTS_TABLE not in CACHED_TABLES
    assert not os.path.exists(str(tmp_path / "measurements.ops_objects.parquet"))


def test_the_reader_prefers_the_cache_and_gets_the_same_rows(tmp_path):
    path = _db(tmp_path)
    write_table(path, "ops_reads", _reads(6))

    cached = read_table(path, "ops_reads")
    authority = read_table(path, "ops_reads", prefer_cache=False)

    assert len(cached) == len(authority) == 6
    assert set(cached["object_id"]) == set(authority["object_id"])


def test_a_missing_cache_falls_back_silently(tmp_path):
    """A missing sidecar is a speed question, not a correctness one."""
    path = _db(tmp_path)
    write_table(path, "ops_reads", _reads(6))
    os.remove(str(tmp_path / "measurements.ops_reads.parquet"))

    assert len(read_table(path, "ops_reads")) == 6


def test_a_cache_that_disagrees_is_removed_rather_than_returned(tmp_path):
    """The one failure this arrangement can produce, and it must not be quiet.

    A stale sidecar is the only way a reader here can be handed wrong data,
    so the disagreement is resolved in favour of the authority AND the liar
    is deleted -- leaving it would mean the next read hits it again.
    """
    path = _db(tmp_path)
    write_table(path, "ops_reads", _reads(6))
    sidecar = str(tmp_path / "measurements.ops_reads.parquet")

    _reads(3).to_parquet(sidecar, index=False)          # plant the drift
    frame = read_table(path, "ops_reads")

    assert len(frame) == 6, "the stale cache was returned"
    assert not os.path.exists(sidecar), "the stale cache was left in place"


def test_a_table_outside_the_contract_is_refused(tmp_path):
    """A table nothing reads would look like data somebody kept."""
    with pytest.raises(StoreError, match="not an OPS table"):
        write_table(_db(tmp_path), "ops_guesses", _objects(2))


def test_every_contract_table_can_be_written(tmp_path):
    """All four of Phase D's names are accepted, not just the two in use."""
    path = _db(tmp_path)
    for table in OPS_TABLES:
        write_table(path, table, _objects(2))
        assert row_count(path, table) == 2


def test_reading_a_table_that_was_never_written_says_so(tmp_path):
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, _objects(2))

    with pytest.raises(StoreError, match="ops_barcodes is not in"):
        read_table(path, "ops_barcodes")


def test_rewriting_replaces_rather_than_appends(tmp_path):
    """Re-running a well must not double its object count."""
    path = _db(tmp_path)
    write_table(path, OBJECTS_TABLE, _objects(5))
    write_table(path, OBJECTS_TABLE, _objects(5))

    assert row_count(path, OBJECTS_TABLE) == 5


def test_the_object_ids_survive_the_round_trip(tmp_path):
    """They are the join key for every later phase, so they must come back."""
    path = _db(tmp_path)
    original = _objects(5)
    write_table(path, OBJECTS_TABLE, original)
    back = read_table(path, OBJECTS_TABLE)

    assert list(back["object_id"]) == list(original["object_id"])
    assert back["centroid_x"].tolist() == original["centroid_x"].tolist()
