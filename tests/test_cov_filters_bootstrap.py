"""Which object tables a database offers to the filters bootstrap.

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
