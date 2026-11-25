"""372 Phase D and the B5 gate: where the tables live, and when C may start.

Item 0an: the OPS tables live in the run's spaCR store (``measurements/``
beside the ``measurements.db`` path the phases pass), written through the
store's one writer and read through its one reader. There is ONE copy of
each table, so the old SQLite authority and its parquet cache -- two stores
that could disagree -- are gone. An older run's ``measurements.db`` is still
read, through the same reader.

B5 is the gate that keeps Phase C from starting early: "no sequencing or
phenotype channel is read until `ops_objects` is on disk and its row count
checked."
"""

import os

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


def test_a_store_without_the_table_is_refused_by_name(tmp_path):
    path = _db(tmp_path)
    write_table(path, "ops_reads", _reads(2))
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


# -- Phase D, one store (item 0an) ---------------------------------------

def test_a_table_is_written_to_the_run_store_and_not_to_sqlite(tmp_path):
    """The table is a part of the run's spaCR store; no SQLite file is made."""
    from spacr import store

    path = _db(tmp_path)
    assert write_table(path, "ops_reads", _reads(6)) == 6

    assert not os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "measurements.ops_reads.parquet"))
    with store.open(str(tmp_path / "measurements")) as opened:
        assert "reads" in opened.tables()
        assert len(opened.query(table="reads")) == 6
    assert row_count(path, "ops_reads") == 6


def test_an_older_run_database_is_read_through_the_shared_reader(tmp_path):
    """A run from before 0an kept its tables in measurements.db; they still read."""
    import sqlite3

    path = _db(tmp_path)
    with sqlite3.connect(path) as conn:
        _objects(4).to_sql(OBJECTS_TABLE, conn, index=False)

    assert row_count(path, OBJECTS_TABLE) == 4
    assert objects_ready(path)
    back = read_table(path, OBJECTS_TABLE)
    assert back["object_id"].tolist() == [1, 2, 3, 4]
    assert row_count(path, "ops_reads") is None


def test_the_readers_get_the_types_they_were_written_against(tmp_path):
    """The cached tables keep their written types; the rest SQLite's classes."""
    import numpy as np

    path = _db(tmp_path)
    frame = _objects(3).assign(flag=[True, False, True],
                               small=np.array([1, 2, 3], np.int16))
    write_table(path, OBJECTS_TABLE, frame)
    write_table(path, "ops_reads", _reads(3).assign(flag=[True, False, True]))

    objects = read_table(path, OBJECTS_TABLE)
    assert objects["flag"].dtype == np.int64 and objects["small"].dtype == np.int64
    assert objects["flag"].tolist() == [1, 0, 1]
    assert read_table(path, "ops_reads")["flag"].dtype == bool
    assert OBJECTS_TABLE not in CACHED_TABLES


def test_rewriting_one_well_leaves_the_other_wells(tmp_path):
    """A well is its own part: replacing it does not read or touch the others."""
    from spacr.ops_engine import _replace_well_rows, _well_rows

    path = _db(tmp_path)
    _replace_well_rows(path, OBJECTS_TABLE, _objects(3), "p1", "A1")
    _replace_well_rows(path, OBJECTS_TABLE, _objects(4).assign(well="A2"),
                       "p1", "A2")
    _replace_well_rows(path, OBJECTS_TABLE, _objects(2), "p1", "A1")

    assert row_count(path, OBJECTS_TABLE) == 6
    assert len(_well_rows(path, OBJECTS_TABLE, "p1", "A1")) == 2
    assert len(_well_rows(path, OBJECTS_TABLE, "p1", "A2")) == 4


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


def test_a_path_with_a_tilde_means_one_file_to_every_entry_point(tmp_path,
                                                                 monkeypatch):
    """`~/run.db` must name the same store to the writer, the counter and
    the reader.

    Found by review. `write_table` writes through
    :func:`spacr.tabular.write_database`, which resolves ``~`` and ``$VARS``
    itself. Nothing else in the module did. So a caller passing
    ``~/run/measurements.db`` had the database written to the EXPANDED path
    while `row_count` opened the literal one, `_sidecar_path` put the parquet
    cache beside a directory actually named ``~``, and `read_table` failed to
    open a database that existed.

    NOTHING RAISED IN THE OBVIOUS PLACE. The write succeeded, so a run looked
    healthy; what was lost was the row-count verification that is the whole
    point of writing both halves in one step, and a cache that could never be
    read. That is why this is a test and not a docstring: the failure is
    silent at the moment it happens and expensive much later.
    """
    import pandas as pd

    from spacr.ops_store import objects_ready, read_table, row_count, write_table

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "run").mkdir()
    db = "~/run/measurements.db"
    frame = pd.DataFrame({"object_id": [1, 2, 3],
                          "centroid_x": [1.0, 2.0, 3.0]})

    assert write_table(db, "ops_objects", frame) == 3
    assert row_count(db, "ops_objects") == 3
    back = read_table(db, "ops_objects")
    assert len(back) == 3
    assert list(back.columns) == ["object_id", "centroid_x"]
    assert bool(objects_ready(db)) is True

    # The expanded store is the real one, and no directory named `~` was made.
    assert (tmp_path / "run" / "measurements" / "spacr.duckdb").is_file()
    assert not (tmp_path / "~").exists()
