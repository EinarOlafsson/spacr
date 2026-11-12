"""Opening an annotated source asks a read-only question first (issue #184).

``_annotation_schema_ready`` decides whether an open needs the write lock at
all. It must answer False -- "go and write, and report what goes wrong
there" -- for every database it cannot read, and True only when the column
and the ``png_path`` index are both already in place.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from spacr.qt.annotate_engine import _annotation_schema_ready


def _png_list(path, *, column=True, index=True):
    connection = sqlite3.connect(path)
    columns = "png_path, label" if column else "png_path"
    connection.execute(f"CREATE TABLE png_list ({columns})")
    if index:
        connection.execute("CREATE INDEX idx_png_path ON png_list (png_path)")
    connection.commit()
    connection.close()
    return str(path)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root opens a file whatever its permissions")
def test_only_a_readable_table_with_the_column_and_index_is_ready(tmp_path):
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"not an SQLite database " * 64)
    locked = _png_list(tmp_path / "locked.db")
    os.chmod(locked, 0)
    try:
        answers = {
            "ready": _annotation_schema_ready(
                _png_list(tmp_path / "ready.db"), "label", "png_list"),
            "no column": _annotation_schema_ready(
                _png_list(tmp_path / "bare.db", column=False), "label", "png_list"),
            "no index": _annotation_schema_ready(
                _png_list(tmp_path / "noindex.db", index=False), "label", "png_list"),
            "not sqlite": _annotation_schema_ready(str(garbage), "label", "png_list"),
            "unreadable": _annotation_schema_ready(locked, "label", "png_list"),
        }
    finally:
        os.chmod(locked, 0o600)

    assert answers == {"ready": True, "no column": False, "no index": False,
                       "not sqlite": False, "unreadable": False}
    assert garbage.read_bytes().startswith(b"not an SQLite database")
