"""Legacy column names in a measurements database are migrated on first read.

Two defects, one mechanism.

**One concept, two spellings.** ``utils.filepaths_to_database`` wrote the
timepoint into ``png_list`` as ``time_id`` while
``utils._merge_and_save_to_database`` wrote ``timeID`` onto every object table.
The visible consequence was that ``_split_data`` printed ``Exception 'timeID'``
and silently built no ``prcft`` for ``png_list``, and any join between
``png_list`` and ``cell`` on time matched nothing.

**Half a rename.** ``utils.correct_metadata`` knew the ``plate_name`` /
``row_name`` / ``column_name`` / ``field_name`` aliases and
``utils.rename_columns_in_db`` did not, so a database carrying them was only
partly repaired -- the generic cause behind several helpers that merged on
``column_name`` and had never worked against a real measurements.db.

``rename_columns_in_db`` runs at the top of ``io._read_db`` and
``io._read_and_join_tables``, so these migrations happen the first time an old
database is opened. That makes its idempotence and its behaviour on a database
that already has the target column the load-bearing properties, and they are
what most of this file tests.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.utils import (DB_COLUMN_RENAMES, TIME_COLUMN_ALIASES, _split_data,
                         _time_column, filepaths_to_database,
                         rename_columns_in_db)


def _columns(db, table):
    conn = sqlite3.connect(db)
    try:
        return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
    finally:
        conn.close()


def _read(db, table):
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(f'SELECT * FROM {table}', conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# the writer now spells it timeID
# --------------------------------------------------------------------------

def test_filepaths_to_database_writes_timeID(tmp_path):
    """The real writer, on a real timelapse crop name."""
    (tmp_path / 'measurements').mkdir(parents=True)
    paths = [str(tmp_path / 'cell_png' / 'plate1_A01_1_3_17.png')]
    filepaths_to_database(paths, {'timelapse': True}, str(tmp_path), 'cell')

    db = tmp_path / 'measurements' / 'measurements.db'
    cols = _columns(db, 'png_list')
    assert 'timeID' in cols
    assert 'time_id' not in cols

    row = _read(db, 'png_list').iloc[0]
    assert row['timeID'] == 't3'
    assert row['fieldID'] == 'f1'
    assert row['cell_id'] == 'o17'


def test_png_list_and_object_tables_now_agree_on_the_time_column(tmp_path):
    """``png_list`` and ``cell`` can be joined on the timepoint."""
    from spacr.utils import _merge_and_save_to_database

    (tmp_path / 'measurements').mkdir(parents=True)
    filepaths_to_database(
        [str(tmp_path / 'cell_png' / 'plate1_A01_1_3_17.png')],
        {'timelapse': True}, str(tmp_path), 'cell')
    _merge_and_save_to_database(
        pd.DataFrame({'label': [17], 'cell_area': [10.0]}),
        pd.DataFrame({'label': [17], 'cell_channel_0_mean_intensity': [1.0]}),
        'cell', str(tmp_path), 'plate1_A01_1_3', 'exp', timelapse=True)

    db = tmp_path / 'measurements' / 'measurements.db'
    png = _read(db, 'png_list')
    cell = _read(db, 'cell')
    assert 'timeID' in png.columns and 'timeID' in cell.columns
    merged = png.merge(cell, on=['plateID', 'rowID', 'columnID', 'fieldID', 'timeID'])
    assert len(merged) == 1


# --------------------------------------------------------------------------
# the migration
# --------------------------------------------------------------------------

def _legacy_png_list(tmp_path):
    """A png_list table written by the REAL writer, then renamed back to time_id.

    Going through ``filepaths_to_database`` and undoing the one column keeps
    the rest of the schema honest -- this is the table a pre-fix spaCR left on
    disk, not a hand-rolled approximation of it.
    """
    (tmp_path / 'measurements').mkdir(parents=True, exist_ok=True)
    filepaths_to_database(
        [str(tmp_path / 'cell_png' / 'plate1_A01_1_3_17.png'),
         str(tmp_path / 'cell_png' / 'plate1_A01_1_4_18.png')],
        {'timelapse': True}, str(tmp_path), 'cell')
    db = tmp_path / 'measurements' / 'measurements.db'
    conn = sqlite3.connect(db)
    try:
        conn.execute('ALTER TABLE png_list RENAME COLUMN "timeID" TO "time_id"')
        conn.commit()
    finally:
        conn.close()
    assert 'time_id' in _columns(db, 'png_list')
    return db


def test_time_id_is_migrated_and_the_data_survives(tmp_path):
    db = _legacy_png_list(tmp_path)
    before = _read(db, 'png_list')

    renamed = rename_columns_in_db(str(db))
    assert ('png_list', 'time_id', 'timeID') in renamed

    after = _read(db, 'png_list')
    assert 'timeID' in after.columns
    assert 'time_id' not in after.columns
    # The values, not just the name.
    assert after['timeID'].tolist() == before['time_id'].tolist() == ['t3', 't4']
    assert after['png_path'].tolist() == before['png_path'].tolist()
    assert after['cell_id'].tolist() == before['cell_id'].tolist() == ['o17', 'o18']
    assert len(after) == len(before) == 2


def test_migrating_twice_is_a_no_op(tmp_path):
    db = _legacy_png_list(tmp_path)
    rename_columns_in_db(str(db))
    first = _read(db, 'png_list')

    assert rename_columns_in_db(str(db)) == []
    second = _read(db, 'png_list')
    pd.testing.assert_frame_equal(first, second)


def test_a_database_that_already_uses_timeID_is_untouched(tmp_path):
    (tmp_path / 'measurements').mkdir(parents=True)
    filepaths_to_database(
        [str(tmp_path / 'cell_png' / 'plate1_A01_1_3_17.png')],
        {'timelapse': True}, str(tmp_path), 'cell')
    db = tmp_path / 'measurements' / 'measurements.db'
    before = _read(db, 'png_list')

    assert rename_columns_in_db(str(db)) == []
    pd.testing.assert_frame_equal(before, _read(db, 'png_list'))


def test_both_spellings_present_keeps_both_and_does_not_crash(tmp_path):
    """Decision: keep both, drop neither, raise nothing.

    A table carrying ``time_id`` *and* ``timeID`` is not something spaCR can
    produce, but a hand-edited or merged database can. Renaming would collide,
    so the rename is skipped -- and dropping or overwriting one of them to tidy
    a name would destroy data, which is never the right trade. Both columns
    stay reachable and every reader accepts either spelling, so a human can
    decide which is authoritative.
    """
    db = _legacy_png_list(tmp_path)
    conn = sqlite3.connect(db)
    try:
        conn.execute('ALTER TABLE png_list ADD COLUMN "timeID" TEXT')
        conn.execute('UPDATE png_list SET "timeID" = \'t99\'')
        conn.commit()
    finally:
        conn.close()

    assert rename_columns_in_db(str(db)) == []          # no crash, no rename
    after = _read(db, 'png_list')
    assert set(after['time_id']) == {'t3', 't4'}        # legacy values intact
    assert set(after['timeID']) == {'t99'}              # new values intact


def test_the_four_name_aliases_are_renamed(tmp_path):
    db = tmp_path / 'legacy.db'
    conn = sqlite3.connect(db)
    try:
        pd.DataFrame({
            'plate_name': ['p1'], 'row_name': ['r2'],
            'column_name': ['c3'], 'field_name': ['f4'], 'value': [1.0],
        }).to_sql('cell', conn, index=False)
    finally:
        conn.close()

    renamed = rename_columns_in_db(str(db))
    assert {(t, o, n) for t, o, n in renamed} == {
        ('cell', 'plate_name', 'plateID'), ('cell', 'row_name', 'rowID'),
        ('cell', 'column_name', 'columnID'), ('cell', 'field_name', 'fieldID')}

    row = _read(db, 'cell').iloc[0]
    assert (row['plateID'], row['rowID'], row['columnID'], row['fieldID']) == \
        ('p1', 'r2', 'c3', 'f4')
    assert row['value'] == 1.0


def test_canonical_names_are_untouched(tmp_path):
    db = tmp_path / 'canonical.db'
    conn = sqlite3.connect(db)
    try:
        pd.DataFrame({
            'plateID': ['p1'], 'rowID': ['r2'], 'columnID': ['c3'],
            'fieldID': ['f4'], 'timeID': ['t5'], 'value': [1.0],
        }).to_sql('cell', conn, index=False)
    finally:
        conn.close()
    before = _read(db, 'cell')
    assert rename_columns_in_db(str(db)) == []
    pd.testing.assert_frame_equal(before, _read(db, 'cell'))


def test_an_alias_is_skipped_when_the_canonical_column_is_already_there(tmp_path):
    """``column_name`` next to an existing ``columnID`` must not collide."""
    db = tmp_path / 'mixed.db'
    conn = sqlite3.connect(db)
    try:
        pd.DataFrame({'columnID': ['c1'], 'column_name': ['c9']}).to_sql(
            'cell', conn, index=False)
    finally:
        conn.close()
    assert rename_columns_in_db(str(db)) == []
    after = _read(db, 'cell')
    assert after['columnID'].tolist() == ['c1']
    assert after['column_name'].tolist() == ['c9']


def test_two_aliases_of_one_canonical_name_do_not_both_fire(tmp_path):
    """``col`` and ``column`` both map to ``columnID``; only one may win."""
    db = tmp_path / 'twoaliases.db'
    conn = sqlite3.connect(db)
    try:
        pd.DataFrame({'col': ['c1'], 'column': ['c2']}).to_sql(
            'cell', conn, index=False)
    finally:
        conn.close()
    renamed = rename_columns_in_db(str(db))
    assert len(renamed) == 1
    cols = _columns(db, 'cell')
    assert cols.count('columnID') == 1
    assert len(cols) == 2                      # nothing was dropped


def test_migration_is_all_or_nothing_when_a_rename_fails(tmp_path, monkeypatch):
    """A failure part-way leaves the database exactly as it was.

    SQLite's DDL is transactional and every rename runs in the one transaction
    committed at the end, so this is a property of the implementation rather
    than of the loop -- worth pinning, because the function runs on every read
    of every user database.
    """
    db = tmp_path / 'atomic.db'
    conn = sqlite3.connect(db)
    try:
        pd.DataFrame({'plate_name': ['p1'], 'row_name': ['r2']}).to_sql(
            'cell', conn, index=False)
    finally:
        conn.close()
    before = _columns(db, 'cell')

    real_connect = sqlite3.connect

    class _FailingCursor:
        def __init__(self, inner):
            self._inner = inner
            self._n = 0

        def execute(self, sql, *args):
            if sql.startswith('ALTER TABLE'):
                self._n += 1
                if self._n == 2:
                    raise sqlite3.OperationalError('boom')
            return self._inner.execute(sql, *args)

        def close(self):
            return self._inner.close()

        def fetchall(self):
            return self._inner.fetchall()

        def __iter__(self):
            return iter(self._inner)

    class _Wrapped:
        def __init__(self, inner):
            self._inner = inner
            self._cursor = None

        def cursor(self):
            self._cursor = _FailingCursor(self._inner.cursor())
            return self._cursor

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(sqlite3, 'connect',
                        lambda *a, **k: _Wrapped(real_connect(*a, **k)))
    with pytest.raises(sqlite3.OperationalError):
        rename_columns_in_db(str(db))
    monkeypatch.undo()

    assert _columns(db, 'cell') == before


# --------------------------------------------------------------------------
# the readers
# --------------------------------------------------------------------------

def test_time_column_helper_prefers_the_canonical_spelling():
    assert _time_column(['plateID', 'timeID']) == 'timeID'
    assert _time_column(['plateID', 'time_id']) == 'time_id'
    assert _time_column(['plateID', 'timeID', 'time_id']) == 'timeID'
    assert _time_column(['plateID']) is None
    assert TIME_COLUMN_ALIASES[0] == 'timeID'


@pytest.mark.parametrize('time_col', ['timeID', 'time_id'])
def test_split_data_builds_prcft_for_png_list(time_col):
    """This is the bug the spelling caused: no prcft, and a printed exception.

    Before the fix, ``_split_data`` hard-coded ``timeID`` inside a bare
    ``try/except``; on ``png_list`` it printed ``Exception 'timeID'`` and moved
    on, so ``prcft`` was simply absent from a timelapse analysis.
    """
    df = pd.DataFrame({
        'plateID': ['p1', 'p1'], 'rowID': ['r1', 'r1'],
        'columnID': ['c1', 'c1'], 'fieldID': ['f1', 'f1'],
        time_col: ['t1', 't2'], 'cell_id': ['o1', 'o2'],
        'value': [1.0, 3.0],
    })
    numeric, non_numeric = _split_data(df, 'prcfo', 'cell_id')
    assert 'prcft' in non_numeric.columns
    assert set(non_numeric['prcft']) == {'p1_r1_c1_f1_t1', 'p1_r1_c1_f1_t2'}


def test_split_data_without_a_time_column_is_silent(capsys):
    df = pd.DataFrame({
        'plateID': ['p1'], 'rowID': ['r1'], 'columnID': ['c1'],
        'fieldID': ['f1'], 'cell_id': ['o1'], 'value': [1.0],
    })
    numeric, non_numeric = _split_data(df, 'prcfo', 'cell_id')
    assert 'prcft' not in non_numeric.columns
    assert 'prcft' not in numeric.columns
    # No "Exception 'timeID'" for a perfectly ordinary non-timelapse frame.
    assert 'Exception' not in capsys.readouterr().out


def test_agreement_treats_both_time_spellings_as_metadata():
    from spacr.agreement import _METADATA_COLUMNS
    assert 'timeID' in _METADATA_COLUMNS
    assert 'time_id' in _METADATA_COLUMNS


def test_resume_honours_both_time_spellings():
    from spacr.resume import TIME_KEY_COLUMNS
    assert set(TIME_KEY_COLUMNS) == {'timeID', 'time_id'}


def test_feature_dict_still_explains_both_spellings():
    from spacr.feature_dict import parse_column
    for name in ('timeID', 'time_id'):
        entry = parse_column(name)
        assert entry.family == 'meta'
        assert entry.description


def test_rename_map_covers_every_alias_correct_metadata_knows():
    """The DataFrame path and the database path must not know different aliases.

    ``correct_metadata`` knowing an alias that ``rename_columns_in_db`` does not
    is exactly how a database ends up half-repaired.
    """
    df_aliases = {'plate_name': 'plateID', 'row_name': 'rowID',
                  'column_name': 'columnID', 'field_name': 'fieldID',
                  'row': 'rowID', 'col': 'columnID', 'column': 'columnID',
                  'field': 'fieldID'}
    for alias, canonical in df_aliases.items():
        assert DB_COLUMN_RENAMES.get(alias) == canonical, alias
