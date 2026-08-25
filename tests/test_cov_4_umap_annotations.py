"""A database with no ``png_list`` table skips its records instead of
inventing one.

Annotations are written back into a table the segmentation step created. If
the caller points at the wrong ``.db``, the writer must count those records
as skipped rather than ALTER a table that does not exist -- the user needs to
see that nothing landed, not a silent success.
"""
from __future__ import annotations

import sqlite3

from spacr.umap_annotations import write_umap_annotations


def _empty_db(tmp_path, name="no_table.db"):
    path = tmp_path / name
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (a INTEGER)")
    return str(path)


def test_a_database_without_png_list_skips_every_record(tmp_path):
    """No table means nothing was updated and everything was skipped."""
    db_path = _empty_db(tmp_path)
    records = [{"db_path": db_path, "db_png_path": f"/img/{i}.png"}
               for i in range(3)]
    updated, skipped = write_umap_annotations(records, [1, 0, 1], "keep")
    assert updated == 0
    assert skipped == 3


def test_the_missing_table_is_not_created_behind_the_users_back(tmp_path):
    """A skipped database is left exactly as it was found."""
    db_path = _empty_db(tmp_path, "untouched.db")
    write_umap_annotations(
        [{"db_path": db_path, "db_png_path": "/img/0.png"}], [1], "keep")
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "png_list" not in tables, tables


def test_a_real_png_list_still_receives_its_column(tmp_path):
    """The skip path must not shadow the ordinary write."""
    path = tmp_path / "good.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE png_list (png_path TEXT)")
        connection.execute("INSERT INTO png_list VALUES ('/img/0.png')")
    updated, skipped = write_umap_annotations(
        [{"db_path": str(path), "db_png_path": "/img/0.png"}], [1], "keep")
    assert (updated, skipped) == (1, 0)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT keep FROM png_list").fetchone()[0] == 1
