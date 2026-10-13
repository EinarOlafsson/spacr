"""Restart a verified collection checkpoint without mixing analysis attempts."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import core
from spacr.cancellation import PipelineCancelled
from tests.test_watch_pipeline_resume_f548 import _images, _settings, _bytes


class Analysis:
    """Write real TIFF-derived pixels and rows; make any repeat distinguishable."""

    def __init__(self, wal=False):
        """Start with no completed analysis attempts."""
        self.calls = 0
        self.wal = wal
        self.connections = []

    def __call__(self, folder, settings):
        """Produce a merged array and SQLite values from this attempt's pixels."""
        self.calls += 1
        folder = Path(folder)
        source = sorted(folder.glob('*.tif'))[0]
        pixels = tifffile.imread(source) + self.calls
        (folder / 'merged').mkdir()
        np.save(folder / 'merged' / (folder.name + '.npy'), pixels)
        (folder / 'measurements').mkdir()
        connection = sqlite3.connect(folder / 'measurements/measurements.db')
        if self.wal:
            connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE cell (field TEXT, intensity REAL)')
        connection.execute('INSERT INTO cell VALUES (?, ?)',
                           (folder.name, float(pixels.mean())))
        connection.commit()
        if self.wal:
            self.connections.append(connection)
        else:
            connection.close()


def _fail_collection(tmp_path, monkeypatch, *, after_commit=False, cancel=False, wal=False):
    """Fail once at the SQLite collection boundary after a real analysis."""
    _images(tmp_path, 'A01')
    analysis = Analysis(wal=wal)
    merge = core._watch_merge_database

    def fail(source, destination, key):
        """Model failure before or after the exactly-once database append."""
        if after_commit:
            merge(source, destination, key)
        if cancel:
            raise PipelineCancelled('stop at collection boundary')
        raise OSError('injected collection failure')

    with monkeypatch.context() as patch:
        patch.setattr(core, '_watch_merge_database', fail)
        if cancel:
            with pytest.raises(PipelineCancelled):
                core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), analysis)
        else:
            result = core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), analysis)
            assert result['failed'] == ['plate1_A01_0001_001']
    return analysis


@pytest.mark.parametrize('after_commit,cancel', [(False, False), (True, False), (True, True)])
def test_resume_collects_same_analysis_once(tmp_path, monkeypatch, after_commit, cancel):
    analysis = _fail_collection(tmp_path, monkeypatch, after_commit=after_commit, cancel=cancel)
    work = tmp_path / 'spacr_watch'
    original = (work / 'merged/plate1_A01_0001_001.npy').read_bytes()
    result = core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), analysis)
    assert result['done'] == ['plate1_A01_0001_001']
    assert analysis.calls == 1
    assert (work / 'merged/plate1_A01_0001_001.npy').read_bytes() == original
    pixels = np.load(work / 'merged/plate1_A01_0001_001.npy')
    with sqlite3.connect(work / 'measurements/measurements.db') as connection:
        assert connection.execute('SELECT intensity FROM cell').fetchall() == [(float(pixels.mean()),)]
        assert connection.execute('SELECT COUNT(*) FROM spacr_watch_fields').fetchone() == (1,)


@pytest.mark.parametrize('damage', ['missing', 'changed', 'conflict', 'source_changed'])
def test_unverifiable_collection_refuses_without_reanalysis(tmp_path, monkeypatch, damage):
    analysis = _fail_collection(tmp_path, monkeypatch)
    work = tmp_path / 'spacr_watch'
    field = work / 'fields/plate1_A01_0001_001'
    if damage == 'missing':
        (field / '.watch_collection/measurements.db').unlink()
    elif damage == 'changed':
        with sqlite3.connect(field / '.watch_collection/measurements.db') as connection:
            connection.execute('UPDATE cell SET intensity=999')
    elif damage == 'conflict':
        path = work / 'merged/plate1_A01_0001_001.npy'
        path.unlink()  # break the hard link before making a conflicting artifact
        np.save(path, np.full((16, 16), 999))
    else:
        source = sorted(tmp_path.glob('*.tif'))[0]
        tifffile.imwrite(source, np.full((16, 16), 99, np.uint16))
    before = _bytes(tmp_path)
    result = core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), analysis)
    assert result['failed'] == ['plate1_A01_0001_001']
    assert analysis.calls == 1
    after = _bytes(tmp_path)
    before.pop('spacr_watch/watch_ledger.json')
    after.pop('spacr_watch/watch_ledger.json')
    assert after == before
    entry = json.loads(Path(result['ledger']).read_text())['fields']['plate1_A01_0001_001']
    assert 'collection' in entry['error'].lower()


def test_running_checkpoint_after_process_exit_is_recoverable(tmp_path, monkeypatch):
    analysis = _fail_collection(tmp_path, monkeypatch)
    path = tmp_path / 'spacr_watch/watch_ledger.json'
    ledger = json.loads(path.read_text())
    ledger['fields']['plate1_A01_0001_001']['status'] = 'running'
    path.write_text(json.dumps(ledger))
    assert core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), analysis)['done']
    assert analysis.calls == 1


def test_live_wal_rows_are_captured_without_changing_producer_bytes(tmp_path):
    folder = tmp_path / 'measurements'
    folder.mkdir()
    database = folder / 'measurements.db'
    connection = sqlite3.connect(database)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE cell (value INTEGER)')
    connection.commit()
    connection.close()
    before = database.read_bytes()
    connection = sqlite3.connect(database)
    try:
        connection.execute('INSERT INTO cell VALUES (42)')
        connection.commit()
        assert database.read_bytes() == before
        wal = Path(str(database) + '-wal').read_bytes()
        core._watch_snapshot_database(str(tmp_path))
        snapshot = tmp_path / '.watch_collection/measurements.db'
        captured = sqlite3.connect(snapshot)
        try:
            assert captured.execute('SELECT value FROM cell').fetchall() == [(42,)]
        finally:
            captured.close()
        assert core._watch_collection_artifacts(str(tmp_path))
        assert database.read_bytes() == before
        assert Path(str(database) + '-wal').read_bytes() == wal
    finally:
        connection.close()
    assert core._watch_collection_artifacts(str(tmp_path))


@pytest.mark.parametrize('suffix', ['-journal', '-shm', '-wal'])
def test_sqlite_sidecars_are_refused_without_deleting_them(tmp_path, suffix):
    folder = tmp_path / '.watch_collection'
    folder.mkdir()
    database = folder / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (value INTEGER)')
    sidecar = Path(str(database) + suffix)
    sidecar.write_bytes(b'pending database state')
    with pytest.raises(ValueError, match='SQLite.*sidecars'):
        core._watch_collection_artifacts(str(tmp_path))
    assert sidecar.read_bytes() == b'pending database state'


def test_merged_child_directory_is_not_silently_omitted(tmp_path):
    (tmp_path / 'merged/nested').mkdir(parents=True)
    with pytest.raises(ValueError, match='Collection artifact.*unsafe'):
        core._watch_collection_artifacts(str(tmp_path))


def test_retry_preserves_pixel_row_parity_with_live_producer_wal(tmp_path, monkeypatch):
    analysis = _fail_collection(tmp_path, monkeypatch, wal=True)
    work = tmp_path / 'spacr_watch'
    producer = work / 'fields/plate1_A01_0001_001/measurements/measurements.db'
    before = producer.read_bytes()
    wal = Path(str(producer) + '-wal').read_bytes()
    try:
        assert core._watch_folder_and_analyse(_settings(tmp_path, 'mask_measure'), analysis)['done']
        assert analysis.calls == 1
        pixels = np.load(work / 'merged/plate1_A01_0001_001.npy')
        with sqlite3.connect(work / 'measurements/measurements.db') as connection:
            assert connection.execute('SELECT intensity FROM cell').fetchall() == [(float(pixels.mean()),)]
        assert producer.read_bytes() == before
        assert Path(str(producer) + '-wal').read_bytes() == wal
    finally:
        for connection in analysis.connections:
            connection.close()


def test_interrupted_database_backup_never_publishes_snapshot(tmp_path, monkeypatch):
    folder = tmp_path / 'measurements'
    folder.mkdir()
    database = folder / 'measurements.db'
    connection = sqlite3.connect(database)
    connection.execute('CREATE TABLE cell (value INTEGER)')
    connection.commit()
    connection.close()
    before = database.read_bytes()

    def interrupt(status, remaining, total):
        """Stop at a real SQLite backup callback boundary."""
        raise PipelineCancelled('stop backup')

    monkeypatch.setattr(core, '_watch_backup_progress', interrupt)
    with pytest.raises(PipelineCancelled):
        core._watch_snapshot_database(str(tmp_path))
    assert not (tmp_path / '.watch_collection/measurements.db').exists()
    assert database.read_bytes() == before
    with pytest.raises(ValueError, match='no completed analysis artifacts'):
        core._watch_collection_artifacts(str(tmp_path))
