from __future__ import annotations

import sqlite3

import pytest

from spacr.umap_annotations import write_umap_annotations


def _database(path, png_paths=("a.png", "b.png", "c.png")):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE png_list "
            "(png_path TEXT PRIMARY KEY, plateID TEXT)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, 'plate1')",
            [(png_path,) for png_path in png_paths],
        )


def _values(path, column):
    escaped = column.replace('"', '""')
    with sqlite3.connect(path) as connection:
        return connection.execute(
            f'SELECT "{escaped}" FROM png_list ORDER BY png_path'
        ).fetchall()


def test_writes_selected_rows_and_creates_column(tmp_path):
    database = tmp_path / "measurements.db"
    _database(database)
    records = [
        {"db_path": database, "db_png_path": "a.png"},
        {"db_path": database, "db_png_path": "c.png"},
    ]

    updated, skipped = write_umap_annotations(
        records, [7, 9], "umap_annotation")

    assert (updated, skipped) == (2, 0)
    assert _values(database, "umap_annotation") == [(7,), (None,), (9,)]


def test_groups_multiple_databases_and_quotes_column_name(tmp_path):
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    _database(first, ("a.png",))
    _database(second, ("b.png",))
    records = [
        {"db_path": first, "db_png_path": "a.png"},
        {"db_path": second, "db_png_path": "b.png"},
    ]

    updated, skipped = write_umap_annotations(
        records, [0, 4], 'review "label"')

    assert (updated, skipped) == (2, 0)
    assert _values(first, 'review "label"') == [(0,)]
    assert _values(second, 'review "label"') == [(4,)]


def test_skips_non_database_records_and_missing_png_rows(tmp_path):
    database = tmp_path / "measurements.db"
    _database(database, ("a.png",))
    records = [
        {"db_path": None, "db_png_path": None},
        {"db_path": tmp_path / "missing.db", "db_png_path": "x.png"},
        {"db_path": database, "db_png_path": "missing.png"},
    ]

    updated, skipped = write_umap_annotations(
        records, [1, 2, 3], "umap_annotation")

    assert (updated, skipped) == (0, 3)


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="same length"):
        write_umap_annotations([], [1], "label")
    with pytest.raises(ValueError, match="non-empty"):
        write_umap_annotations([], [], "")
