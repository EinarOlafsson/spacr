"""A record with no match key is skipped, not written and not counted.

:func:`spacr.attribution_columns.write` matches each attribution record to a
``png_list`` row by ``key_column``.  A record whose key is missing or ``None``
cannot name a row: issuing its UPDATE anyway would compare ``prcfo = NULL``,
which SQLite evaluates as unknown for every row, so the write would touch
nothing while ``matched`` still counted it.
"""
from __future__ import annotations

import sqlite3

from spacr.attribution_columns import COLUMNS, write


def _make_db(path, keys):
    """``png_list`` with a ``prcfo`` key column and one row per key."""
    with sqlite3.connect(str(path)) as con:
        con.execute("CREATE TABLE png_list (prcfo TEXT, png_path TEXT)")
        con.executemany("INSERT INTO png_list VALUES (?, ?)",
                        [(k, f"/data/{k}.png") for k in keys])
    return str(path)


def _attributions(path):
    with sqlite3.connect(path) as con:
        return dict(con.execute("SELECT prcfo, grna_attributed FROM png_list").fetchall())


CASES = [
    ("a_missing_key_is_skipped", ["o1", "o2"],
     [{"prcfo": "o1", "grna_attributed": "gRNA_1"}, {"grna_attributed": "gRNA_2"}],
     1, {"o1": "gRNA_1", "o2": None}),
    ("an_explicit_none_key_is_skipped", ["o1"],
     [{"prcfo": None, "grna_attributed": "gRNA_9"}],
     0, {"o1": None}),
    ("records_after_a_keyless_one_are_still_written", ["o1", "o2"],
     [{"grna_attributed": "gRNA_lost"}, {"prcfo": "o2", "grna_attributed": "gRNA_2"}],
     1, {"o1": None, "o2": "gRNA_2"}),
    ("a_key_naming_no_row_matches_nothing", ["o1"],
     [{"prcfo": "o7", "grna_attributed": "gRNA_7"}],
     0, {"o1": None}),
]


def test_write_counts_only_records_whose_key_names_a_row(tmp_path, subtests):
    """``matched`` equals the rows actually updated, and keyless records change nothing."""
    for case_id, keys, records, matched, stored in CASES:
        with subtests.test(msg=case_id):
            db = _make_db(tmp_path / f"{case_id}.db", keys)
            result = write(db, records, confirmed=True)
            assert result == {"matched": matched, "added": len(COLUMNS)}
            assert _attributions(db) == stored
