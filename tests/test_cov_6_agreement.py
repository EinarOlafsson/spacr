"""Two guards in the agreement reader: the read-only URI and the class cap.

:func:`spacr.agreement._read_only_uri` is how the module states, in the
connection string itself, that it will never write to somebody's
``measurements.db``. Its escaping has to survive the paths real spaCR runs
produce -- Windows separators, spaces, ``#`` -- because a URI that loses a
character opens the wrong file, or opens a writable one.

:func:`annotation_columns` offers a user the columns that *look like* an
annotation pass. A column holding a continuous measurement has as many
distinct values as rows; offering it would put a per-cell intensity in the
annotator picker and score it as a third annotator.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from spacr.agreement import (
    PNG_KEY, PNG_TABLE, _read_only_uri, annotation_columns,
)

_META = ("png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
         "prcfo", "cell_id")


def _make_db(path, columns, rows):
    """``png_list`` with the spaCR metadata plus one column per name given."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    try:
        meta_sql = ", ".join(f'"{c}" TEXT' for c in _META)
        con.execute(f"CREATE TABLE {PNG_TABLE} ({meta_sql})")
        for name, kind in columns:
            con.execute(f'ALTER TABLE {PNG_TABLE} ADD COLUMN "{name}" {kind}')
        placeholders = ", ".join("?" * (len(_META) + len(columns)))
        payload = []
        for i, values in enumerate(rows):
            crop = f"/data/plate1/cell_png/plate1_A01_1_{i}.png"
            payload.append((crop, os.path.basename(crop), "plate1", "r1",
                            "c1", "f1", f"plate1_A01_1_o{i}", f"o{i}",
                            *values))
        con.executemany(
            f"INSERT INTO {PNG_TABLE} VALUES ({placeholders})", payload)
        con.commit()
    finally:
        con.close()
    return path


def test_the_read_only_uri_says_read_only_and_keeps_the_path():
    """``mode=ro`` is the promise; the path has to arrive intact beside it."""
    uri = _read_only_uri("/data/run1/measurements/measurements.db")
    assert uri.startswith("file:/data/run1/measurements/measurements.db")
    assert uri.endswith("?mode=ro")


def test_the_read_only_uri_normalises_windows_separators():
    """A backslash in a URI is not a separator; SQLite would miss the file."""
    uri = _read_only_uri(r"C:\data\run1\measurements.db")
    assert "\\" not in uri
    assert "C:/data/run1/measurements.db" in uri


def test_the_read_only_uri_escapes_a_space_and_a_fragment_marker():
    """A raw ``#`` truncates the URI, so it must be percent-encoded."""
    uri = _read_only_uri("/data/my runs/plate #2/measurements.db")
    assert " " not in uri
    assert "%20" in uri
    assert uri.count("#") == 0
    assert "%23" in uri
    assert uri.endswith("?mode=ro")


def test_a_column_with_too_many_distinct_values_is_not_an_annotation(tmp_path):
    """A per-cell measurement must not be offered as an annotator column."""
    rows = [(1, i) for i in range(40)]
    db = _make_db(tmp_path / "run" / "measurements.db",
                  [("alice", "INTEGER"), ("cell_area", "INTEGER")], rows)
    offered = annotation_columns(db, max_classes=10, min_labelled=1)
    assert "alice" in offered
    assert "cell_area" not in offered


def test_raising_the_class_cap_lets_the_same_column_through(tmp_path):
    """The cap is what excludes it -- nothing about the column's name."""
    rows = [(1, i) for i in range(40)]
    db = _make_db(tmp_path / "run" / "measurements.db",
                  [("alice", "INTEGER"), ("cell_area", "INTEGER")], rows)
    assert "cell_area" in annotation_columns(
        db, max_classes=100, min_labelled=1)


def test_the_key_column_is_never_offered_as_an_annotation(tmp_path):
    """``prcfo`` is the row identity, so it is unique by construction."""
    db = _make_db(tmp_path / "run" / "measurements.db",
                  [("alice", "INTEGER")], [(1,), (2,)])
    assert PNG_KEY not in annotation_columns(db, min_labelled=1)
