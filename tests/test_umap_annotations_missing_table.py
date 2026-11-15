"""UMAP annotations land in ``png_list`` or are counted as skipped (``spacr.umap_annotations``).

A database with no ``png_list`` table skips its records and is left exactly as
found -- the writer never invents the table -- while a real ``png_list`` still
receives its column and values.
"""
from __future__ import annotations

import sqlite3

from spacr.umap_annotations import write_umap_annotations


def _tables_and_rows(path):
    with sqlite3.connect(path) as connection:
        tables = sorted(row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))
        if "png_list" not in tables:
            return tables, None, None
        cursor = connection.execute("SELECT * FROM png_list")
        columns = [description[0] for description in cursor.description]
        return tables, columns, cursor.fetchall()


def test_annotations_are_written_to_png_list_or_skipped(subtests, tmp_path):
    """Exact (updated, skipped) counts and the database state afterwards."""
    no_table = tmp_path / "no_table.db"
    with sqlite3.connect(no_table) as connection:
        connection.execute("CREATE TABLE unrelated (a INTEGER)")
    good = tmp_path / "good.db"
    with sqlite3.connect(good) as connection:
        connection.execute("CREATE TABLE png_list (png_path TEXT)")
        connection.execute("INSERT INTO png_list VALUES ('/img/0.png')")

    cases = [
        ("no png_list skips every record",
         [{"db_path": str(no_table), "db_png_path": f"/img/{i}.png"}
          for i in range(3)], [1, 0, 1], (0, 3),
         no_table, (["unrelated"], None, None)),
        ("a real png_list receives its column",
         [{"db_path": str(good), "db_png_path": "/img/0.png"}], [1], (1, 0),
         good, (["png_list"], ["png_path", "keep"], [("/img/0.png", 1)])),
    ]
    for case_id, records, values, counts, db, state in cases:
        with subtests.test(msg=case_id):
            assert write_umap_annotations(records, values, "keep") == counts
            assert _tables_and_rows(db) == state
