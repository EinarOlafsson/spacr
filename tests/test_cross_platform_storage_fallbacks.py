"""Storage fallbacks preserve labels, recorded paths and database constraints."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_recorded_posix_path_does_not_become_a_windows_path():
    from spacr.cli_repro import _recorded_path

    assert _recorded_path('/images/../outputs') == PurePosixPath('/outputs')
    assert _recorded_path('/images/../outputs', normalise=False).parts == (
        '/', 'images', '..', 'outputs')


def test_windows_crop_destination_matches_components_without_case():
    from spacr import crops

    original = crops.os
    crops.os = SimpleNamespace(name='nt')
    try:
        assert crops.path_is_under('C:/LAB/experiment/cell.npy', 'c:/lab/experiment')
        assert not crops.path_is_under('C:/LAB/experiment2/cell.npy', 'c:/lab/experiment')
    finally:
        crops.os = original


@pytest.mark.parametrize('path', ['C:\\images\\cell.npy', '\\\\server\\share\\cell.npy'])
def test_plain_windows_paths_keep_their_original_spelling(path):
    from spacr.portable_paths import _plain_path

    assert _plain_path(path, windows=True) == path


def test_mapped_mask_save_can_rewrite_without_truncating(tmp_path, monkeypatch):
    from spacr import mask_io

    path = tmp_path / 'mask.npy'
    original = np.array([[0, 3], [3, 0]], dtype=np.uint16)
    np.save(path, original)
    view = np.load(path, mmap_mode='r')
    updated = np.array([[0, 65535], [2, 0]], dtype=np.uint16)
    save = np.save

    def refuse_truncation(destination, *args, **kwargs):
        if isinstance(destination, (str, Path)):
            raise PermissionError('The destination has a live mapping')
        return save(destination, *args, **kwargs)

    monkeypatch.setattr(mask_io.np, 'save', refuse_truncation)
    assert mask_io.save_mask(path, updated) == path
    np.testing.assert_array_equal(np.load(path), updated)
    np.testing.assert_array_equal(view, updated)
    np.testing.assert_array_equal(original, [[0, 3], [3, 0]])


def test_in_place_mask_fallback_preserves_a_different_sized_file(tmp_path, monkeypatch):
    from spacr import mask_io

    path = tmp_path / 'mask.npy'
    np.save(path, np.array([[7]], dtype=np.uint16))
    before = path.read_bytes()
    save = np.save

    def refuse_truncation(destination, *args, **kwargs):
        if isinstance(destination, (str, Path)):
            raise PermissionError('mapped destination')
        return save(destination, *args, **kwargs)

    monkeypatch.setattr(mask_io.np, 'save', refuse_truncation)
    with pytest.raises(PermissionError, match='mapped destination'):
        mask_io.save_mask(path, np.zeros((2, 2), dtype=np.uint16))
    assert path.read_bytes() == before


def test_in_place_mask_fallback_refuses_a_missing_destination(tmp_path):
    from spacr import mask_io

    path = tmp_path / 'mask.npy'
    assert not mask_io._overwrite_mapped_npy_in_place(path, np.zeros((1, 1), dtype=np.uint16))


@pytest.mark.parametrize('directory', [True, False])
def test_chunk_permission_error_is_empty_only_for_a_directory(tmp_path, monkeypatch, directory):
    from spacr import ome_zarr

    path = tmp_path / 'chunk'
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b'chunk data')

    def denied(*args, **kwargs):
        raise PermissionError('access denied')

    monkeypatch.setattr(ome_zarr, 'open', denied, raising=False)
    if directory:
        assert ome_zarr._read_chunk_bytes(path) is None
    else:
        with pytest.raises(PermissionError, match='access denied'):
            ome_zarr._read_chunk_bytes(path)
        assert path.read_bytes() == b'chunk data'


def test_foreign_parquet_folder_errors_keep_the_native_destination(tmp_path, monkeypatch):
    from spacr import tabular

    table = 'cell:area'
    native = os.path.join(str(tmp_path), 'cell%3Aarea')

    def unavailable(path):
        if path == native:
            return False
        raise ValueError('this filesystem refuses the foreign spelling')

    monkeypatch.setattr(tabular, 'os', SimpleNamespace(
        name='nt', path=SimpleNamespace(join=os.path.join, isdir=unavailable)))
    assert tabular._parquet_table_folder(str(tmp_path), table) == native
    assert not tabular._parquet_verbatim_folder(str(tmp_path), table)


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


def test_curation_reports_missing_destinations_and_retains_out_of_folder_counts(tmp_path):
    from spacr.curation_queue import CurationQueueError, QueueItem, QueueLayout, QueueSummary

    item = QueueItem('field', 'nested')
    layout = QueueLayout('nested', tmp_path, tmp_path, None, (item,))
    with pytest.raises(CurationQueueError, match='no masks folder'):
        layout.mask_destination('field')
    with pytest.raises(CurationQueueError, match='not in the queue'):
        layout.mask_destination('other')
    summary = QueueSummary(total=5, done=1, skip=1, recropped=1, remaining=2, unknown=3)
    assert summary.describe() == '5 bundles, 1 done, 1 skip, 1 recropped, 2 remaining, 3 elsewhere'


def test_recorded_windows_path_keeps_drive_and_optional_parent_components(monkeypatch):
    from spacr import cli_repro

    monkeypatch.setattr(cli_repro, 'os', SimpleNamespace(name='nt'))
    path = r"C:\images\..\outputs"
    assert cli_repro._recorded_path(path) == PureWindowsPath('C:/outputs')
    assert cli_repro._recorded_path(path, normalise=False).parts == (
        'C:\\', 'images', '..', 'outputs')


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
