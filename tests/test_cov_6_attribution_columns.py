"""A record with no match key is skipped, not written and not counted.

:func:`spacr.attribution_columns.write` matches each attribution record to a
``png_list`` row by ``key_column``. A record whose key is missing or ``NULL``
cannot name a row. Issuing its UPDATE anyway would compare ``prcfo = NULL``,
which SQLite evaluates as unknown for every row: the write would touch
nothing while ``matched`` still counted it, so the caller would be told the
attribution landed on cells it never reached.
"""

from __future__ import annotations

import sqlite3

import pytest

from spacr.attribution_columns import write


def _make_db(path, keys):
    """``png_list`` with a ``prcfo`` key column and one row per key."""
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE png_list (prcfo TEXT, png_path TEXT)")
        con.executemany("INSERT INTO png_list VALUES (?, ?)",
                        [(k, f"/data/{k}.png") for k in keys])
        con.commit()
    finally:
        con.close()
    return str(path)


def _attributions(path, key="prcfo"):
    con = sqlite3.connect(path)
    try:
        return dict(con.execute(
            "SELECT prcfo, grna_attributed FROM png_list").fetchall())
    finally:
        con.close()


def test_a_record_without_the_key_is_skipped_and_not_counted(tmp_path):
    """A keyless record names no row, so it must not inflate ``matched``."""
    db = _make_db(tmp_path / "measurements.db", ["o1", "o2"])
    result = write(db, [
        {"prcfo": "o1", "grna_attributed": "gRNA_1"},
        {"grna_attributed": "gRNA_2"},
    ], confirmed=True)
    assert result["matched"] == 1
    assert _attributions(db) == {"o1": "gRNA_1", "o2": None}


def test_an_explicit_none_key_is_skipped_the_same_way(tmp_path):
    """``prcfo=None`` is what a frame with a missing key yields on to_dict."""
    db = _make_db(tmp_path / "measurements.db", ["o1"])
    result = write(db, [{"prcfo": None, "grna_attributed": "gRNA_9"}],
                   confirmed=True)
    assert result["matched"] == 0
    assert _attributions(db) == {"o1": None}


def test_records_after_a_keyless_one_are_still_written(tmp_path):
    """Skipping is a ``continue``: one bad record must not stop the write."""
    db = _make_db(tmp_path / "measurements.db", ["o1", "o2"])
    result = write(db, [
        {"grna_attributed": "gRNA_lost"},
        {"prcfo": "o2", "grna_attributed": "gRNA_2"},
    ], confirmed=True)
    assert result["matched"] == 1
    assert _attributions(db) == {"o1": None, "o2": "gRNA_2"}
