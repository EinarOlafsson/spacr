"""A table rewrite that fails part-way leaves the old table in place.

``_update_database_with_merged_info`` rewrites ``png_list`` with the merged
annotation columns. It used to run ``DROP TABLE`` and then ``to_sql``. Python's
sqlite3 driver opens an implicit transaction only for DML, so the DROP was
committed on its own, and a write that failed after it (disk full, I/O error,
a value SQLite refuses) left measurements.db with no ``png_list`` at all. The
error was printed and swallowed, so the run carried on and the next step
found the table gone.

``_pivot_counts_table`` had the same weakness through
``to_sql(if_exists='replace')``, which pandas implements as DROP + CREATE +
INSERT.

Both now stage the rows under a scratch name and swap them in inside one
transaction. These tests force the write to fail and check that the original
table, its columns and its rows survive, and that a successful rewrite gives
the same table as before.
"""

import sqlite3

import pandas as pd
import pytest

from spacr import utils as U


_ORIGINAL = pd.DataFrame({
    "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2", "plate1_r1_c1_f1_o3"],
    "png_path": ["a/1.png", "a/2.png", "a/3.png"],
    "png_width": [64, 64, 80],
})


def _db_with_png_list(tmp_path):
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db, timeout=30)
    try:
        _ORIGINAL.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db


def _tables(db):
    con = sqlite3.connect(db, timeout=30)
    try:
        return sorted(row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))
    finally:
        con.close()


def _read(db, table):
    con = sqlite3.connect(db, timeout=30)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    finally:
        con.close()


def _schema(db, table):
    con = sqlite3.connect(db, timeout=30)
    try:
        return con.execute(f'PRAGMA table_info("{table}")').fetchall()
    finally:
        con.close()


def _annotations():
    return pd.DataFrame({
        "plateID": ["plate1"] * 3, "rowID": ["r1"] * 3,
        "columnID": ["c1"] * 3, "fieldID": ["f1"] * 3,
        "object_label": [1, 2, 3],
        "pathogen": ["rh", "rh", "rh"], "treatment": ["cm", "dmso", "cm"],
        "host_cells": ["HeLa"] * 3,
        "condition": ["HeLa_rh_cm", "HeLa_rh_dmso", "HeLa_rh_cm"],
    })


def _failing_to_sql(monkeypatch, after_rows=False):
    """Make every DataFrame.to_sql fail; optionally after writing some rows."""
    real = pd.DataFrame.to_sql

    def boom(self, name, con, *args, **kwargs):
        if after_rows:
            real(self.iloc[:1], name, con, *args, **kwargs)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(pd.DataFrame, "to_sql", boom)


@pytest.mark.parametrize("after_rows", [False, True],
                         ids=["fails-before-any-row", "fails-after-a-row"])
def test_a_failed_png_list_update_keeps_the_table_and_its_rows(
        tmp_path, monkeypatch, capsys, after_rows):
    db = _db_with_png_list(tmp_path)
    schema_before = _schema(db, "png_list")
    _failing_to_sql(monkeypatch, after_rows=after_rows)

    U._update_database_with_merged_info(str(db), _annotations())

    out = capsys.readouterr().out
    assert "Failed to update table png_list in the database" in out
    assert "disk I/O error" in out
    assert _tables(db) == ["png_list"], "no scratch table may be left behind"
    assert _schema(db, "png_list") == schema_before
    pd.testing.assert_frame_equal(_read(db, "png_list"), _ORIGINAL)


def test_a_successful_png_list_update_is_unchanged(tmp_path, capsys):
    """Same table, columns, types and row order as the old DROP + to_sql."""
    db = _db_with_png_list(tmp_path)
    U._update_database_with_merged_info(str(db), _annotations())
    assert "successfully updated" in capsys.readouterr().out

    expected_db = tmp_path / "expected.db"
    expected = pd.merge(
        _ORIGINAL,
        _annotations().assign(prcfo=_ORIGINAL["prcfo"])[
            ["pathogen", "treatment", "host_cells", "condition", "prcfo"]],
        on="prcfo", how="left", validate="many_to_one")
    con = sqlite3.connect(expected_db, timeout=30)
    try:
        expected.to_sql("png_list", con, index=False)
    finally:
        con.close()

    assert _tables(db) == ["png_list"]
    assert _schema(db, "png_list") == _schema(expected_db, "png_list")
    pd.testing.assert_frame_equal(_read(db, "png_list"),
                                  _read(expected_db, "png_list"))


def _db_with_counts(tmp_path):
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db, timeout=30)
    try:
        pd.DataFrame({
            "file_name": ["b.npy", "a.npy", "b.npy", "a.npy"],
            "count_type": ["cell", "cell", "nucleus", "nucleus"],
            "object_count": [3, 1, 2, 5],
        }).to_sql("object_counts", con, index=False)
    finally:
        con.close()
    return db


def test_a_failed_counts_pivot_keeps_the_previous_pivot(tmp_path, monkeypatch):
    db = _db_with_counts(tmp_path)
    U._pivot_counts_table(str(db))
    before = _read(db, "pivoted_counts")
    schema_before = _schema(db, "pivoted_counts")
    assert len(before) == 2

    _failing_to_sql(monkeypatch, after_rows=True)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        U._pivot_counts_table(str(db))

    assert _tables(db) == ["object_counts", "pivoted_counts"]
    assert _schema(db, "pivoted_counts") == schema_before
    pd.testing.assert_frame_equal(_read(db, "pivoted_counts"), before)


def test_a_successful_counts_pivot_is_unchanged(tmp_path):
    db = _db_with_counts(tmp_path)
    U._pivot_counts_table(str(db))
    U._pivot_counts_table(str(db))  # a rerun replaces, it does not append

    source = _read(db, "object_counts")
    expected = source.pivot(index="file_name", columns="count_type",
                            values="object_count").reset_index().fillna(0)
    expected_db = tmp_path / "expected.db"
    con = sqlite3.connect(expected_db, timeout=30)
    try:
        expected.to_sql("pivoted_counts", con, index=False)
    finally:
        con.close()

    assert _tables(db) == ["object_counts", "pivoted_counts"]
    assert _schema(db, "pivoted_counts") == _schema(expected_db, "pivoted_counts")
    pd.testing.assert_frame_equal(_read(db, "pivoted_counts"),
                                  _read(expected_db, "pivoted_counts"))
