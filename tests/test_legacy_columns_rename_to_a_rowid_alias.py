"""Legacy measurement columns still migrate where SQLite refuses a rowid-alias rename.

Found natively 2026-10-19: SQLite 3.54 (conda-forge) refuses ``RENAME COLUMN "row" TO "rowID"`` and
``ADD COLUMN "rowID"`` ("cannot rename to rowid alias"), so every legacy measurement database with a
``row`` column failed to migrate. The table is now rebuilt with every row and rowid kept; tables the
rebuild could not reproduce exactly are refused inside the migration's transaction.
"""
import sqlite3

import pytest

from spacr import database_schema


def _legacy(path):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE cell ("row" TEXT NOT NULL, "column" TEXT DEFAULT \'x\', value REAL)')
    con.executemany("INSERT INTO cell VALUES (?, ?, ?)", [("A", "1", 0.5), ("B", "2", 1.5)])
    con.execute("DELETE FROM cell WHERE value = 0.5")
    con.execute("INSERT INTO cell VALUES ('C', '3', 2.5)")
    con.commit()
    return con


def test_the_rebuild_keeps_every_row_its_rowid_and_its_definition(tmp_path):
    con = _legacy(tmp_path / "legacy.db")
    before = con.execute("SELECT _rowid_, * FROM cell ORDER BY _rowid_").fetchall()
    cursor = con.cursor()
    database_schema._rebuild_with_column_renamed(cursor, "cell", "row", "rowID")
    con.commit()
    info = con.execute("PRAGMA table_info(cell)").fetchall()
    assert [row[1] for row in info] == ["rowID", "column", "value"]
    assert info[0][3] == 1 and info[1][4] == "'x'"
    assert con.execute('SELECT _rowid_, "rowID", "column", value FROM cell ORDER BY _rowid_').fetchall() == before
    assert not con.execute("SELECT name FROM sqlite_master WHERE name LIKE '_spacr_rebuild_%'").fetchall()


def test_an_index_naming_the_column_is_refused_and_nothing_changes(tmp_path):
    con = _legacy(tmp_path / "legacy.db")
    con.execute('CREATE INDEX by_row ON cell ("row")')
    con.commit()
    with pytest.raises(database_schema.DatabaseMigrationError):
        database_schema._rebuild_with_column_renamed(con.cursor(), "cell", "row", "rowID")
    assert [row[1] for row in con.execute("PRAGMA table_info(cell)")] == ["row", "column", "value"]


def test_a_legacy_database_migrates_whatever_this_sqlite_allows(tmp_path):
    path = tmp_path / "legacy.db"
    _legacy(path).close()
    from spacr.utils import rename_columns_in_db

    rename_columns_in_db(str(path))
    con = sqlite3.connect(path)
    assert [row[1] for row in con.execute("PRAGMA table_info(cell)")][:2] == ["rowID", "columnID"]
    assert con.execute('SELECT "rowID" FROM cell ORDER BY _rowid_').fetchall() == [("B",), ("C",)]
