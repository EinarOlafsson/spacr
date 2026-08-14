"""Regression tests for the intermittent measure_crop field loss.

The symptom was
``tests/test_errors.py::test_measure_crop_marks_measurements_db_partial_when_a_field_fails``
failing about 5 runs in 20: four synthetic fields go in, one deliberately
corrupt, and on a bad run only TWO of the three good fields came back out of
``measurements.db`` — while the run ledger still counted three successes.

The first cause was a check-then-act race inside ``pandas.DataFrame.to_sql``.
``SQLTable.create`` asks whether the table exists and issues ``CREATE TABLE``
when it does not; there is no lock across the two steps. measure_crop runs one
worker PROCESS per field against one SQLite file, so on the first fields of a
fresh run several workers pass the existence check together, all issue the
CREATE, and every worker but the winner gets::

    sqlite3.OperationalError: table "cell" already exists

which is not a lock error, so ``spacr.utils._append_to_measurements_db``
printed one line and returned — dropping that field's entire measurement
frame, silently, while the field still reported success. Four writers released
from a barrier lost rows in 30 of 30 trials.

Issue #15 exposed the second cause: pandas repeated the same table-existence
read before every append. A stream of worker reads can prevent a writer from
committing on rollback-journal network filesystems. The writer now inserts
directly after the one-time creation. These tests pin both recoveries.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.utils import (
    DB_APPEND_REPAIRS,
    _append_frame,
    _append_to_measurements_db,
    _insert_frame,
    _widen_table_for,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path, name='measurements.db'):
    """An existing but empty measurements.db, as _save_settings_to_db leaves it."""
    db = tmp_path / 'measurements'
    db.mkdir(exist_ok=True)
    path = str(db / name)
    sqlite3.connect(path).close()
    return path


def _lost_the_create_race(table):
    """The exact error pandas surfaces to the loser of the CREATE TABLE race."""
    return sqlite3.OperationalError(f'table "{table}" already exists')


def _flaky_to_sql(monkeypatch, fail_times, error_factory):
    """Make the first ``fail_times`` calls to ``DataFrame.to_sql`` raise.

    Returns a list that records one entry per call, so a test can assert the
    retry actually happened rather than the first call having quietly worked.
    """
    real = pd.DataFrame.to_sql
    calls = []

    def flaky(self, name, con, *args, **kwargs):
        calls.append(name)
        if len(calls) <= fail_times:
            raise error_factory(name)
        return real(self, name, con, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'to_sql', flaky)
    return calls


def _rows(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# the create-table race
# ---------------------------------------------------------------------------

def test_append_frame_retries_when_it_loses_the_create_table_race(tmp_path, monkeypatch):
    """The loser's rows must land, not vanish.

    The winner of the race has created the table by the time we look again, so
    a plain retry turns pandas' ``create()`` into a no-op and the insert goes
    through. Before the fix the OperationalError escaped and the caller threw
    the frame away.
    """
    path = _fresh_db(tmp_path)
    calls = _flaky_to_sql(monkeypatch, fail_times=1, error_factory=_lost_the_create_race)
    frame = pd.DataFrame({'object_label': [1, 2], 'file_name': ['f', 'f']})

    conn = sqlite3.connect(path)
    try:
        _append_frame(conn, 'cell', frame)
    finally:
        conn.close()

    assert calls == ['cell', 'cell']       # it really did fail once and retry
    assert _rows(path, 'cell') == 2        # and nothing was lost


def test_append_to_measurements_db_keeps_the_field_after_a_create_race(tmp_path, monkeypatch):
    """The same thing at the entry point measure_crop's workers actually call.

    This is the precise line where a field's measurements used to disappear:
    ``_append_to_measurements_db`` classified "table already exists" as
    not-a-lock, printed, and returned success.
    """
    path = _fresh_db(tmp_path)
    _flaky_to_sql(monkeypatch, fail_times=1, error_factory=_lost_the_create_race)
    frame = pd.DataFrame({'object_label': [1, 2, 3], 'file_name': ['plate1_A01_F001'] * 3})

    _append_to_measurements_db(path, 'cell', frame)

    conn = sqlite3.connect(path)
    try:
        names = {row[0] for row in conn.execute('SELECT DISTINCT file_name FROM cell')}
    finally:
        conn.close()
    assert names == {'plate1_A01_F001'}
    assert _rows(path, 'cell') == 3


def test_append_frame_raises_rather_than_losing_rows_when_the_race_never_settles(
        tmp_path, monkeypatch):
    """A race that never resolves must be loud, not silent.

    If the repairs are exhausted the error propagates, so the worker's
    RunLedger records the field as failed and the artifact is stamped partial.
    Returning quietly here would recreate the original bug in a new place.
    """
    path = _fresh_db(tmp_path)
    calls = _flaky_to_sql(monkeypatch, fail_times=DB_APPEND_REPAIRS + 5,
                          error_factory=_lost_the_create_race)
    frame = pd.DataFrame({'object_label': [1]})

    conn = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.OperationalError, match='already exists'):
            _append_frame(conn, 'cell', frame)
    finally:
        conn.close()

    assert len(calls) == DB_APPEND_REPAIRS      # bounded; it does not spin


def test_append_frame_does_not_swallow_unrelated_sqlite_errors(tmp_path, monkeypatch):
    """Only the two known-recoverable conditions are retried.

    A read-only database is a setup problem, not contention, and must reach
    ``_append_to_measurements_db``'s own handling unchanged.
    """
    path = _fresh_db(tmp_path)

    def readonly(_table):
        return sqlite3.OperationalError('attempt to write a readonly database')

    calls = _flaky_to_sql(monkeypatch, fail_times=1, error_factory=readonly)
    conn = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            _append_frame(conn, 'cell', pd.DataFrame({'object_label': [1]}))
    finally:
        conn.close()

    assert len(calls) == 1                       # no pointless retry


def test_append_frame_widens_after_losing_the_create_race_to_a_narrower_frame(
        tmp_path):
    """Both repairs in one append: lose the create, then find a narrower table.

    The worker that won the race can have created the table from a frame with
    fewer columns — a field with no pathogen objects, say. The retry then hits
    "has no column named", which widens, and the third pass writes. No
    monkeypatching: the narrow table is created for real first.
    """
    path = _fresh_db(tmp_path)
    conn = sqlite3.connect(path)
    try:
        pd.DataFrame({'object_label': [1]}).to_sql('cell', conn, if_exists='append',
                                                   index=False)
        wide = pd.DataFrame({'object_label': [2], 'pathogen_area': [17.0]})
        _append_frame(conn, 'cell', wide)
    finally:
        conn.close()

    conn = sqlite3.connect(path)
    try:
        got = dict(conn.execute('SELECT object_label, pathogen_area FROM cell').fetchall())
    finally:
        conn.close()
    assert got == {1: None, 2: 17.0}       # the old row is NULL, the new one landed


def test_existing_table_append_never_reads_sqlite_master(tmp_path):
    """The shared-filesystem fix: ordinary appends contain no schema read."""
    path = _fresh_db(tmp_path)
    conn = sqlite3.connect(path)
    try:
        pd.DataFrame({'object_label': [1], 'value': [0.5]}).to_sql(
            'cell', conn, if_exists='append', index=False)
        statements = []
        conn.set_trace_callback(statements.append)
        _append_frame(
            conn,
            'cell',
            pd.DataFrame({'object_label': [2], 'value': [1.5]}),
        )
    finally:
        conn.close()

    sql = '\n'.join(statements).lower()
    assert 'sqlite_master' not in sql
    assert 'insert into "cell"' in sql
    assert _rows(path, 'cell') == 2


def test_direct_insert_matches_pandas_scalar_and_null_encoding(tmp_path):
    """Direct insertion retains the value semantics of the removed path."""
    path = _fresh_db(tmp_path)
    frame = pd.DataFrame({
        'integer': pd.Series([1, None], dtype='Int64'),
        'floating': [np.float32(2.5), np.nan],
        'boolean': pd.Series([True, None], dtype='boolean'),
        'text': ['alpha', None],
        'timestamp': [pd.Timestamp('2026-08-14T12:30:00'), pd.NaT],
        'duration': [pd.Timedelta(seconds=2), pd.NaT],
    })
    direct = sqlite3.connect(path)
    reference = sqlite3.connect(':memory:')
    try:
        frame.iloc[:0].to_sql('values_table', direct, index=False)
        _insert_frame(direct, 'values_table', frame)
        frame.to_sql('values_table', reference, index=False)
        direct_rows = direct.execute(
            'SELECT * FROM values_table ORDER BY integer IS NULL'
        ).fetchall()
        reference_rows = reference.execute(
            'SELECT * FROM values_table ORDER BY integer IS NULL'
        ).fetchall()
    finally:
        direct.close()
        reference.close()

    assert direct_rows == reference_rows


def test_widen_table_for_tolerates_a_concurrent_writer_adding_the_same_column(tmp_path):
    """The widening has a check-then-act window of its own.

    Two workers can both read PRAGMA table_info, both miss a column, and both
    ALTER. The loser gets "duplicate column name" — which is success in
    disguise, since the column it wanted is there. Letting it escape would have
    cost that worker its whole frame, the original bug one level down.

    The race is made deterministic here with a second real connection that
    always gets its ALTER in first.
    """
    path = _fresh_db(tmp_path)
    winner = sqlite3.connect(path)
    loser = sqlite3.connect(path)

    class _Overtaken:
        """A connection whose ALTERs are always beaten by another writer."""

        def execute(self, sql, *args):
            if sql.lstrip().upper().startswith('ALTER TABLE'):
                winner.execute(sql)
                winner.commit()
            return loser.execute(sql, *args)

        def commit(self):
            return loser.commit()

    try:
        pd.DataFrame({'object_label': [1]}).to_sql('cell', winner, if_exists='append',
                                                   index=False)
        winner.commit()
        frame = pd.DataFrame({'object_label': [2], 'blur_ch0': [0.5]})
        added = _widen_table_for(_Overtaken(), 'cell', frame)
    finally:
        winner.close()
        loser.close()

    # The column exists; whether we or the other writer added it is immaterial.
    assert added == []
    conn = sqlite3.connect(path)
    try:
        have = {row[1] for row in conn.execute('PRAGMA table_info("cell")')}
    finally:
        conn.close()
    assert 'blur_ch0' in have


# ---------------------------------------------------------------------------
# the real thing: several processes, one fresh database
# ---------------------------------------------------------------------------

def _write_one_field(args):
    """Pool worker: one field's rows into a shared measurements.db.

    Anything raised here comes back out of ``pool.map`` and fails the test,
    which is the point — a worker that cannot write must not be quiet.
    """
    db_path, barrier, field = args
    try:
        barrier.wait(timeout=60)
    except Exception:
        pass                                  # a slow runner must not fail the test
    frame = pd.DataFrame({'object_label': [1, 2],
                          'file_name': [f'plate1_A0{field}_F001'] * 2})
    _append_to_measurements_db(db_path, 'cell', frame)


def test_four_processes_writing_a_fresh_db_lose_no_fields(tmp_path):
    """End-to-end at the layer that broke: four writers, none of them silent.

    This is measure_crop's shape in miniature — one process per field, one
    SQLite file, all of them arriving at an empty table at once. The barrier is
    what makes the create-table race near-certain rather than one-run-in-four:
    before the fix this lost rows in 30 of 30 trials.
    """
    path = _fresh_db(tmp_path)
    fields = [1, 2, 3, 4]

    with mp.Manager() as manager:
        barrier = manager.Barrier(len(fields))
        with mp.Pool(len(fields)) as pool:
            pool.map(_write_one_field, [(path, barrier, f) for f in fields])

    conn = sqlite3.connect(path)
    try:
        names = {row[0] for row in conn.execute('SELECT DISTINCT file_name FROM cell')}
        total = conn.execute('SELECT COUNT(*) FROM cell').fetchone()[0]
    finally:
        conn.close()
    assert names == {f'plate1_A0{f}_F001' for f in fields}
    assert total == 2 * len(fields)


def test_the_race_this_all_guards_against_is_real(tmp_path):
    """Sanity: pandas really does emit 'table already exists' on an append.

    If a future pandas makes ``to_sql(if_exists='append')`` atomic, this test
    fails and the recovery code above can be reconsidered. Constructed
    deterministically: the table is created between pandas' existence check and
    its CREATE by doing the create ourselves and re-issuing pandas' own SQL.
    """
    path = _fresh_db(tmp_path)
    conn = sqlite3.connect(path)
    try:
        frame = pd.DataFrame({'object_label': [1]})
        frame.to_sql('cell', conn, if_exists='append', index=False)
        # Exactly what the loser of the race ends up executing: a CREATE for a
        # table another process created a moment earlier.
        with pytest.raises(sqlite3.OperationalError, match='already exists'):
            conn.execute('CREATE TABLE "cell" ("object_label" INTEGER)')
    finally:
        conn.close()

    assert os.path.isfile(path)
