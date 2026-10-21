"""SQLite fallbacks preserve constraints and file locks close their descriptors."""
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest


















def test_sqlite_rebuild_preserves_primary_key_defaults_and_independent_index():
    from spacr.database_schema import _rebuild_with_column_renamed

    with sqlite3.connect(':memory:') as connection:
        connection.execute('CREATE TABLE cells (row_name INTEGER PRIMARY KEY, note, score REAL DEFAULT 1.5)')
        connection.execute('CREATE INDEX cells_score ON cells(score)')
        connection.execute('INSERT INTO cells(row_name,note) VALUES (42,?)', ('cell',))
        _rebuild_with_column_renamed(connection.cursor(), 'cells', 'row_name', 'rowID')
        assert connection.execute('SELECT rowid,rowID,note,score FROM cells').fetchall() == [(42, 42, 'cell', 1.5)]
        info = {row[1]: row for row in connection.execute('PRAGMA table_info(cells)')}
        assert info['rowID'][5] == 1
        assert info['note'][2] == ''
        assert info['score'][4] == '1.5'
        assert connection.execute('PRAGMA index_info(cells_score)').fetchall() == [(0, 2, 'score')]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute('INSERT INTO cells(rowID) VALUES (42)')


def test_column_rename_propagates_non_alias_sqlite_errors_without_rebuilding():
    from spacr.database_schema import _rename_legacy_columns

    with sqlite3.connect(':memory:') as connection:
        connection.execute('CREATE TABLE cells(row_name TEXT)')
        connection.execute("INSERT INTO cells VALUES ('r1')")

        class LockedCursor:
            def __init__(self):
                self.cursor = connection.cursor()

            def execute(self, sql, *args):
                if sql.startswith('ALTER TABLE'):
                    raise sqlite3.OperationalError('database is locked')
                return self.cursor.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self.cursor, name)

        with pytest.raises(sqlite3.OperationalError, match='database is locked'):
            _rename_legacy_columns(SimpleNamespace(cursor=LockedCursor))
        assert connection.execute('SELECT row_name FROM cells').fetchall() == [('r1',)]
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name LIKE '_spacr_rebuild_%'").fetchall()


@pytest.mark.parametrize('failed_mode', [None, 1, 2])
def test_windows_advisory_lock_closes_its_descriptor_even_if_locking_is_unavailable(
        tmp_path, monkeypatch, failed_mode):
    from spacr import remote_execution

    calls = []

    def locking(descriptor, mode, size):
        os.fstat(descriptor)
        calls.append((descriptor, mode, size))
        if mode == failed_mode:
            raise OSError('advisory locks unavailable')

    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(locking=locking, LK_LOCK=1, LK_UNLCK=2))
    monkeypatch.setattr(remote_execution, 'os', SimpleNamespace(name='nt'))
    path = tmp_path / 'state.lock'
    entered = []
    with remote_execution._file_lock(path):
        entered.append(True)
        assert path.is_file()
    assert entered == [True]
    assert [call[1:] for call in calls] == [(1, 1), (2, 1)]
    assert calls[0][0] == calls[1][0]
    with pytest.raises(OSError):
        os.fstat(calls[0][0])






def test_sqlite_alias_rename_fallback_preserves_rows_primary_key_and_index():
    from spacr.database_schema import _rename_legacy_columns

    with sqlite3.connect(':memory:') as connection:
        connection.execute('CREATE TABLE cells(row_name INTEGER PRIMARY KEY, note TEXT)')
        connection.execute('CREATE INDEX cells_note ON cells(note)')
        connection.execute("INSERT INTO cells VALUES (42, 'cell')")
        refused = []

        class AliasRefusingCursor:
            def __init__(self):
                self.cursor = connection.cursor()

            def execute(self, sql, *args):
                if sql.startswith('ALTER TABLE') and 'RENAME COLUMN' in sql:
                    refused.append(sql)
                    raise sqlite3.OperationalError('cannot rename to rowid alias')
                return self.cursor.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self.cursor, name)

        renamed = _rename_legacy_columns(SimpleNamespace(cursor=AliasRefusingCursor))
        assert renamed == (('cells', 'row_name', 'rowID'),)
        assert len(refused) == 1
        assert connection.execute('SELECT rowid,rowID,note FROM cells').fetchall() == [(42, 42, 'cell')]
        assert connection.execute('PRAGMA table_info(cells)').fetchall()[0][5] == 1
        assert connection.execute('PRAGMA index_info(cells_note)').fetchall() == [(0, 1, 'note')]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute('INSERT INTO cells(rowID) VALUES (42)')
