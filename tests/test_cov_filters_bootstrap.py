"""The filters bootstrap, its fallback, and sampled reads.

Instruction 60. ``spacr.filters`` was at 80%, and the uncovered part was
mostly the paths a database takes when it is NOT the ideal shape: no
relationships table, only some object tables, a table with no object label.

That is the part worth testing. The ideal database already works -- what
decides whether a user can gate their screen at all is what happens to the
awkward one, and the module's own comment says the fallback exists to keep
such a database "gateable rather than blocked".
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import filters


def _write(path, tables):
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _objects(n=4, label_column="object_label"):
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        label_column: range(1, n + 1),
        "area": [10.0 * i for i in range(1, n + 1)],
    })


# --------------------------------------------------------------------------- #
#  object_tables -- what the database actually has
# --------------------------------------------------------------------------- #

def test_only_tables_carrying_an_object_label_count(tmp_path):
    """A table can be present and empty of the identity a filter needs.
    Discovering that later is worse than not counting it now."""
    path = _write(tmp_path / "m.db", {
        "cell": _objects(),
        "notes": pd.DataFrame({"text": ["hello"]}),
    })

    found = filters.object_tables(path)

    assert "cell" in found
    assert "notes" not in found


def test_a_database_with_no_object_tables_reports_none(tmp_path):
    path = _write(tmp_path / "m.db",
                  {"notes": pd.DataFrame({"text": ["hello"]})})
    assert filters.object_tables(path) == ()


# --------------------------------------------------------------------------- #
#  choose_anchor -- "usually cell, but not always"
# --------------------------------------------------------------------------- #

def test_cell_is_the_anchor_when_it_is_there():
    assert filters.choose_anchor(("nucleus", "cell", "pathogen")) == "cell"


def test_a_database_of_only_nuclei_anchors_on_nuclei():
    """There is no table that must exist, so the preference order has to
    keep working all the way down."""
    assert filters.choose_anchor(("nucleus",)) == "nucleus"


def test_no_tables_is_refused_rather_than_answered_emptily():
    """A database with no object table cannot be gated at all, so this
    refuses and NAMES what it looked for -- an empty string would be handed
    onward and fail somewhere less informative."""
    with pytest.raises(filters.FilterError) as caught:
        filters.choose_anchor(())

    message = str(caught.value)
    assert "no object table" in message
    assert "cell" in message and "nucleus" in message


# --------------------------------------------------------------------------- #
#  read_identity -- identity only, on purpose
# --------------------------------------------------------------------------- #

def test_reading_identity_does_not_drag_in_the_measurements(tmp_path):
    """A measurement table is hundreds of columns wide and four matter. This
    is the difference between a bootstrap that takes a second and one that
    reads the whole screen."""
    path = _write(tmp_path / "m.db", {"cell": _objects()})

    frame = filters.read_identity(path, "cell")

    assert "plateID" in frame.columns
    assert "area" not in frame.columns


def test_key_columns_are_the_identity_in_join_order(tmp_path):
    frame = _objects()
    keys = filters.key_columns(frame)

    assert keys
    assert all(column in frame.columns for column in keys)
    assert "area" not in keys


# --------------------------------------------------------------------------- #
#  build_filters_frame -- including the fallback that keeps a database usable
# --------------------------------------------------------------------------- #

def test_the_bootstrap_builds_one_row_per_object(tmp_path):
    path = _write(tmp_path / "m.db", {"cell": _objects(n=4)})

    frame = filters.build_filters_frame(path)

    assert len(frame) == 4
    assert "plateID" in frame.columns


def test_the_bootstrap_falls_back_when_relationships_cannot_be_built(
        tmp_path, monkeypatch):
    """The module's own reason: an unreadable relationships table, or a
    schema nobody anticipated, must leave the database GATEABLE rather than
    blocked."""
    path = _write(tmp_path / "m.db", {"cell": _objects(n=3)})

    def _explode(_db_path):
        raise RuntimeError("relationships table is not readable")

    monkeypatch.setattr(filters, "build_filters_from_relationships", _explode)

    frame = filters.build_filters_frame(path)

    assert len(frame) == 3, "the fallback must still produce the objects"


def test_a_filter_error_from_relationships_is_not_swallowed(tmp_path,
                                                            monkeypatch):
    """The fallback catches surprises, not refusals. A FilterError is the
    module saying something is genuinely wrong, and hiding it behind a
    fallback would turn a clear message into a confusing frame."""
    path = _write(tmp_path / "m.db", {"cell": _objects()})

    def _refuse(_db_path):
        raise filters.FilterError("this database has no object table")

    monkeypatch.setattr(filters, "build_filters_from_relationships", _refuse)

    with pytest.raises(filters.FilterError):
        filters.build_filters_frame(path)


# --------------------------------------------------------------------------- #
#  read_sampled -- the row cap that makes a huge screen openable
# --------------------------------------------------------------------------- #

def test_a_whole_table_is_read_when_nothing_is_asked_for(tmp_path):
    path = _write(tmp_path / "m.db", {"cell": _objects(n=10)})
    assert len(filters.read_sampled(path, "cell")) == 10


def test_a_limit_caps_the_rows(tmp_path):
    path = _write(tmp_path / "m.db", {"cell": _objects(n=10)})
    assert len(filters.read_sampled(path, "cell", limit=4)) == 4


def test_a_fraction_takes_a_share_of_the_rows(tmp_path):
    """Sampling happens in SQL where it can, because the point is to NOT
    read the whole table."""
    path = _write(tmp_path / "m.db", {"cell": _objects(n=100)})

    sampled = filters.read_sampled(path, "cell", fraction=0.2)

    assert 0 < len(sampled) < 100


def test_a_fraction_and_a_limit_together_respect_the_limit(tmp_path):
    path = _write(tmp_path / "m.db", {"cell": _objects(n=100)})

    sampled = filters.read_sampled(path, "cell", fraction=0.9, limit=5)

    assert len(sampled) <= 5
