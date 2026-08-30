"""Six blocks in the foreign-table importer: its messages, and its rollbacks.

Two kinds of code, both untested for the same reason -- neither runs on a
successful import. The message helpers only ever build text for a REFUSAL, and
the rollbacks only ever run when a rename or an insert has already failed. So a
suite made of imports that work reaches none of it, and the first time either
runs is in front of a user whose import has gone wrong.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest


def _db(tmp_path, name="measurements.db"):
    return str(tmp_path / name)


# ---------------------------------------------------------------------------
# _describe_table and _sample — the ellipsis when there is more to say
# ---------------------------------------------------------------------------

def test_a_long_field_list_is_truncated_with_an_ellipsis(tmp_path):
    """Line 2481. The message must show it is a sample, not the whole set.

    This text is built only for a refusal, and a refusal that listed four
    fields out of six hundred without saying so would send the user looking
    for a problem in the four.
    """
    from spacr.foreign import _describe_table

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (prcf TEXT)')
        connection.executemany('INSERT INTO cell VALUES (?)',
                               [(f"p1_r1_c1_f{i}",) for i in range(9)])
        connection.commit()

        described = _describe_table(connection, "cell", limit=4)
    finally:
        connection.close()

    assert "9 row(s)" in described
    assert described.rstrip().endswith("…")


def test_a_short_field_list_is_shown_whole(tmp_path):
    """The other side: no ellipsis when nothing was left out."""
    from spacr.foreign import _describe_table

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (prcf TEXT)')
        connection.execute('INSERT INTO cell VALUES ("p1_r1_c1_f1")')
        connection.commit()
        described = _describe_table(connection, "cell", limit=4)
    finally:
        connection.close()

    assert "…" not in described
    assert "p1_r1_c1_f1" in described


def test_a_long_value_sample_says_how_many_it_left_out():
    """Line 2490. The count is the useful part -- "4 of 6" versus "4 of 600"."""
    from spacr.foreign import _sample

    text = _sample([f"g{i}" for i in range(10)], limit=4)

    assert text.startswith("g0, g1, g2, g3")
    assert "(6 more)" in text


def test_a_short_value_sample_is_listed_in_full():
    """The other side, and the empty case, which must read as 'none'."""
    from spacr.foreign import _sample

    assert _sample(["b", "a"]) == "a, b"
    assert _sample([]) == "none"


# ---------------------------------------------------------------------------
# _twin_condition — nothing in common to match on
# ---------------------------------------------------------------------------

def test_tables_with_no_shared_columns_have_no_twin_condition(tmp_path):
    """Line 2593. ``None`` means "cannot decide", which is not "no twins".

    The condition is used to find rows the importer added. With no shared
    column there is no way to pair a row with its foreign original, and
    returning a condition that matched everything -- or nothing -- would
    either delete the user's own rows or silently leave the importer's behind.
    """
    from spacr.foreign import _twin_condition

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (a INTEGER, b INTEGER)')
        connection.execute('CREATE TABLE foreign_cell (x INTEGER, y INTEGER)')
        connection.commit()

        assert _twin_condition(connection, "cell") is None
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# release_canonical_copy — a database that is not there
# ---------------------------------------------------------------------------

def test_releasing_from_a_missing_database_removes_nothing(tmp_path):
    """Line 2673. Zero, not an exception.

    Release runs during cleanup, where the database may already have been
    moved or removed. A raise here would turn a finished run into a failed one
    at the last step.
    """
    from spacr.foreign import release_canonical_copy

    assert release_canonical_copy(_db(tmp_path, "gone.db"), "cell") == 0


# ---------------------------------------------------------------------------
# _replace_table_atomically — the rollback
# ---------------------------------------------------------------------------

def test_a_failed_table_replacement_leaves_the_original_in_place(tmp_path):
    """Lines 2886-2888: ROLLBACK and re-raise.

    The docstring calls this "``to_sql(if_exists='replace')``, but
    all-or-nothing", and the all-or-nothing half is entirely in these three
    lines. Without them a failed rename leaves the table DROPPED and the
    staging table unrenamed -- the user's measurements gone and the run
    reporting an error about something else.
    """
    from spacr import foreign

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (a INTEGER)')
        connection.execute('INSERT INTO cell VALUES (1)')
        connection.commit()

        frame = pd.DataFrame({"a": [2, 3]})

        class _RefusesTheRename:
            """A cursor that fails exactly where a real rename can fail.

            Wrapping rather than patching sqlite3.Cursor, which is an
            immutable type. The failure is placed on the RENAME because that
            is the step after the DROP -- the window in which the table is
            gone and nothing has replaced it.
            """

            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args, **kwargs):
                if sql.startswith('ALTER TABLE') and 'RENAME TO' in sql:
                    raise sqlite3.OperationalError("the rename failed")
                return self._real.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._real, name)

        class _Connection:
            def __init__(self, real):
                self._real = real

            def cursor(self):
                return _RefusesTheRename(self._real.cursor())

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __setattr__(self, name, value):
                if name == "_real":
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._real, name, value)

        with pytest.raises(sqlite3.OperationalError):
            foreign._replace_table_atomically(_Connection(connection),
                                              "cell", frame)

        # The original row survived: the DROP was rolled back with the rename.
        kept = connection.execute('SELECT a FROM cell').fetchall()
        assert kept == [(1,)]
    finally:
        connection.close()


def test_a_successful_replacement_swaps_the_contents(tmp_path):
    """The committed path, so the rollback above is visibly the other outcome."""
    from spacr import foreign

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (a INTEGER)')
        connection.execute('INSERT INTO cell VALUES (1)')
        connection.commit()

        foreign._replace_table_atomically(connection, "cell",
                                          pd.DataFrame({"a": [2, 3]}))

        assert connection.execute(
            'SELECT a FROM cell ORDER BY a').fetchall() == [(2,), (3,)]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# _insert_rows — an empty frame
# ---------------------------------------------------------------------------

def test_inserting_an_empty_frame_does_nothing_and_does_not_raise(tmp_path):
    """Line 3073.

    The docstring's reason for this function existing is that ``to_sql``
    commits on its own and would break the caller's delete/insert pair in
    half. An empty frame is an ordinary outcome of that pair -- everything
    filtered out -- and it must leave the caller's transaction untouched
    rather than raising on a zero-column INSERT.
    """
    from spacr.foreign import _insert_rows

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (a INTEGER)')
        connection.commit()
        cursor = connection.cursor()

        _insert_rows(cursor, "cell", pd.DataFrame({"a": []}))

        assert connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == 0
    finally:
        connection.close()


def test_a_non_empty_frame_is_inserted_with_an_explicit_column_list(tmp_path):
    """The other side, which the empty check must not prevent."""
    from spacr.foreign import _insert_rows

    path = _db(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE cell (a INTEGER, b TEXT)')
        connection.commit()
        cursor = connection.cursor()

        _insert_rows(cursor, "cell", pd.DataFrame({"a": [1, 2],
                                                   "b": ["x", "y"]}))
        connection.commit()

        assert connection.execute(
            'SELECT a, b FROM cell ORDER BY a').fetchall() == [(1, "x"),
                                                               (2, "y")]
    finally:
        connection.close()
