"""Callers preserve scientific inputs and reject invalid queued writes."""

import errno
import json
import multiprocessing
from pathlib import Path

import pytest

from spacr import measure, resource_log, sim
from spacr.cancellation import PipelineCancelled
from spacr.database_concurrency import connect
from spacr.runctx import _DeferredOverloadRetries


@pytest.mark.parametrize('caller', ['measure', 'simulation'])
def test_processing_without_writer_endpoint_never_starts_computation(monkeypatch, caller):
    module = measure if caller == 'measure' else sim
    endpoint = '_MEASURE_WRITE_ENDPOINT' if caller == 'measure' else '_SIMULATION_WRITE_ENDPOINT'
    compute = '_measure_crop_core' if caller == 'measure' else 'run_and_save'
    monkeypatch.setattr(module, endpoint, None)
    monkeypatch.setattr(module, compute, lambda *_: pytest.fail('Computed without writer'))
    with pytest.raises(RuntimeError, match='no database writer endpoint'):
        if caller == 'measure':
            measure._measure_crop_queued(0, [], 'field', {})
        else:
            sim._run_and_save_queued(0, {}, [], 1)


def test_measure_rejects_unknown_operation_without_committing_a_ticket(tmp_path):
    path = tmp_path / 'measurements.db'
    packet = {'ticket': 'ticket', 'field': 'field',
              'operations': [('unapproved', (), {})]}
    with pytest.raises(ValueError, match='Unsupported Measure'):
        measure._commit_measure_packet(path, packet)
    with connect(path, readonly=True) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


@pytest.mark.parametrize('damage', ['path_type', 'index_type', 'operation', 'keywords', 'other_database', 'empty'])
def test_simulation_rejects_invalid_packet_before_creating_sqlite(tmp_path, damage):
    path = str(tmp_path / 'simulations.db')
    field = [path, 1]
    operations = [('simulation', (str(tmp_path), 'rows', 'table'), {})]
    if damage == 'path_type':
        field[0] = 1
    elif damage == 'index_type':
        field[1] = '1'
    elif damage == 'operation':
        operations[0] = ('unapproved', operations[0][1], {})
    elif damage == 'keywords':
        operations[0] = ('simulation', operations[0][1], {'unapproved': True})
    elif damage == 'other_database':
        operations[0] = ('simulation', (str(tmp_path / 'other'), 'rows', 'table'), {})
    else:
        operations = []
    with pytest.raises(ValueError):
        sim._commit_simulation_packet({'ticket': 'ticket', 'field': json.dumps(field),
                                       'operations': operations})
    assert not list(tmp_path.rglob('*.db'))


@pytest.mark.parametrize('value', [-1, 65, float('nan'), float('inf'), 'invalid'])
def test_preference_rejects_invalid_budget_without_overwriting_saved_value(tmp_path, monkeypatch, value):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    settings = QSettings(str(tmp_path / 'prefs.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: settings)
    preferences.set_database_write_queue_gib(0.25)
    with pytest.raises(ValueError):
        preferences.set_database_write_queue_gib(value)
    assert preferences.get_database_write_queue_gib() == 0.25


@pytest.mark.parametrize('value, expected', [('invalid', 1), (None, 1), (float('nan'), 1), (float('inf'), 1), (-1, 0), (65, 64)])
def test_damaged_saved_budget_has_a_finite_bounded_fallback(tmp_path, monkeypatch, value, expected):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    settings = QSettings(str(tmp_path / 'prefs.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: settings)
    settings.setValue(preferences._KEY_DATABASE_WRITE_QUEUE, value)
    assert preferences.get_database_write_queue_gib() == expected


@pytest.mark.parametrize('workers, context', [(0, None), (1, None), (1, 'spawn')])
def test_loader_argument_adapter_preserves_inputs_and_only_wraps_workers(workers, context):
    original = {'num_workers': workers, 'batch_size': 3, 'drop_last': True,
                'multiprocessing_context': context}
    result = resource_log._data_loader_arguments((), original)
    assert original['multiprocessing_context'] == context
    assert result['batch_size'] == 3 and result['drop_last'] is True
    if workers:
        assert isinstance(result['multiprocessing_context'], resource_log._StaggeredContext)
    else:
        assert result == original


def test_loader_adapter_does_not_double_wrap_an_existing_context():
    context = resource_log._StaggeredContext(multiprocessing.get_context('spawn'))
    result = resource_log._data_loader_arguments((), {'num_workers': 1, 'multiprocessing_context': context})
    assert result['multiprocessing_context'] is context


def test_real_torch_loader_adapter_keeps_values_and_order():
    loader = resource_log._parallel_data_loader([3, 4], batch_size=1, num_workers=0)
    assert [batch.item() for batch in loader] == [3, 4]
    assert loader.num_workers == 0


def test_single_worker_cloudpickle_calls_finish_primaries_before_one_retry():
    seen = []

    def call(value):
        seen.append(value)
        if value == 0 and seen.count(0) == 1:
            raise MemoryError('primary overload')
        if value == 0:
            assert seen == [0, 1, 0]
        return value + 4

    result = resource_log._parallel_cloudpickle_map(
        [(call, (value,), {}) for value in [0, 1]], workers=1)
    assert result == [4, 5] and seen == [0, 1, 0]


@pytest.mark.parametrize('delay', [-1, float('nan'), float('inf')])
def test_invalid_start_gate_delay_is_refused(delay):
    with pytest.raises(ValueError):
        resource_log._WorkerStartGate(delay=delay)


def test_cancelled_final_retry_does_not_run_or_readmit_later_tasks():
    retries, seen = _DeferredOverloadRetries(), []

    def cancel():
        raise PipelineCancelled('user stopped final pass')

    assert retries.defer('stop', MemoryError(), cancel)
    assert retries.defer('later', MemoryError(), lambda: seen.append('ran'))
    with pytest.raises(PipelineCancelled):
        list(retries.drain())
    assert seen == [] and list(retries.drain()) == []
    with pytest.raises(RuntimeError, match='cannot accept'):
        retries.defer('another', MemoryError(), lambda: None)


@pytest.mark.parametrize('failure', [MemoryError('overload'), OSError(errno.ENOMEM, 'overload')])
def test_intensity_inspection_overload_retries_instead_of_becoming_a_bad_field(tmp_path, monkeypatch, failure):
    import numpy as np
    from spacr import intensity_rescale

    seen = []
    names = ['plate1_A01_1.npy', 'plate1_A01_2.npy']
    for name in names:
        np.save(tmp_path / name, np.ones((3, 3, 2), dtype=np.uint16))
    original = np.load

    def load(path, *args, **kwargs):
        name = Path(path).name
        seen.append(name)
        if name == names[0] and seen.count(name) == 1:
            raise failure
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', load)
    result = intensity_rescale.build_plate_plan(tmp_path, names,
        {'n_jobs': 1, 'cell_mask_dim': 1, 'timelapse': False})
    assert seen == [names[0], names[1], names[0]]
    assert set(result['fields']) == set(names) and result['failures'] == {}


def test_ops_resource_exhaustion_is_not_reported_as_a_corrupt_image(tmp_path, monkeypatch):
    import tifffile
    from spacr import ops_engine

    path = tmp_path / 'plane.tif'
    path.write_bytes(b'fixture source')

    def exhausted(*_args, **_kwargs):
        raise OSError(errno.EMFILE, 'too many open files')

    monkeypatch.setattr(tifffile, 'TiffFile', exhausted)
    unreadable = []
    with pytest.raises(OSError) as failure:
        ops_engine._read_plane((str(path), None), unreadable)
    assert failure.value.errno == errno.EMFILE and unreadable == []


@pytest.mark.parametrize('notification', ['success', 'overload', 'invalid'])
def test_mask_commit_preserves_filtered_pixels_and_permissions_before_notification(
        tmp_path, notification):
    import os
    import stat
    import numpy as np
    from spacr import utils

    mask = np.zeros((8, 8), np.uint16)
    mask[1:3, 1:3] = 1
    mask[4:7, 4:7] = 2
    paths = [tmp_path / 'field1.npy', tmp_path / 'field2.npy']
    for path in paths:
        np.save(path, mask)
        os.chmod(path, 0o640)
    seen = []

    def notify(index, total, duration, operation):
        seen.append(index)
        assert total == 2 and duration >= 0 and operation == 'filter'
        if notification == 'overload':
            raise MemoryError('progress notification overloaded')
        if notification == 'invalid':
            raise ValueError('invalid notification')

    if notification == 'invalid':
        with pytest.raises(ValueError, match='invalid notification'):
            utils.merge_split_objects(str(tmp_path), perimeter_fraction=0,
                                      min_area=5, n_jobs=1,
                                      progress_callback=notify, op_name='filter')
    else:
        utils.merge_split_objects(str(tmp_path), perimeter_fraction=0,
                                  min_area=5, n_jobs=1,
                                  progress_callback=notify, op_name='filter')
    assert seen == [0, 1]
    for path in paths:
        saved = np.load(path)
        assert np.array_equal(saved > 0, mask == 2)
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert not list(tmp_path.glob('.spacr-mask-*'))


def test_confluency_capture_saves_original_result_through_the_owned_writer(tmp_path):
    import numpy as np
    from spacr.database_concurrency import _capture_write_packet, _DatabaseWriteQueue

    path = tmp_path / 'measurements' / 'measurements.db'
    result = measure._mask_coverage(np.array([[1, 0], [1, 0]], dtype=np.uint16))
    with _capture_write_packet() as operations:
        measure._write_confluency_record(str(tmp_path), 'plate1_A01_F001.npy', {}, result)
    assert not path.exists() and len(operations) == 1
    path.parent.mkdir()
    outcomes = []
    writer = _DatabaseWriteQueue(path.parent / '.write_queue',
        lambda packet: measure._commit_measure_packet(path, packet),
        lambda *row: outcomes.append(row), ram_gib=0)
    writer.endpoint.enqueue('field', operations)
    writer.start()
    writer.finish()
    assert len(outcomes) == 1 and outcomes[0][2] is None
    with connect(path, readonly=True) as connection:
        assert connection.execute('SELECT confluency, covered_px, field_px FROM confluency').fetchall() == [(0.5, 2, 4)]


@pytest.mark.parametrize('message, attempts', [('database is locked', 2), ('no such table: invalid', 1)])
def test_optional_append_cannot_hide_an_error_inside_an_atomic_packet(tmp_path, monkeypatch, message, attempts):
    import pandas as pd
    import sqlite3
    from spacr import utils
    from spacr.database_concurrency import _capture_write_packet, _DatabaseWriteQueue

    path = tmp_path / 'measurements.db'
    with _capture_write_packet() as operations:
        utils._append_to_measurements_db(str(path), 'cell', pd.DataFrame({'label': [1]}), required=False)
    seen, outcomes = [], []

    def fail(*_args, **_kwargs):
        seen.append(message)
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(utils, '_append_frame', fail)
    monkeypatch.setattr(utils, 'DB_WRITE_ATTEMPTS', 1)
    writer = _DatabaseWriteQueue(tmp_path / 'spool',
        lambda packet: measure._commit_measure_packet(path, packet),
        lambda *row: outcomes.append(row), ram_gib=0)
    ticket = writer.endpoint.enqueue('field', operations)
    writer.start()
    writer.finish()
    assert len(seen) == attempts and len(outcomes) == 1
    assert isinstance(outcomes[0][2], sqlite3.OperationalError)
    assert (Path(writer.endpoint.folder) / (ticket + '.pkl')).exists()
    with connect(path, readonly=True) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
def _simulation_rows_for_writer_failure(index, settings, times, total):
    import os
    import pandas as pd
    from spacr import sim

    destination = Path(settings['src']) / settings['start_time'] / settings['name']
    destination.mkdir(parents=True, exist_ok=True)
    sim.append_database(str(destination), pd.DataFrame({'value': [index + 17]}), 'simulations')
    if settings['injected_failure'] == 'producer':
        raise ValueError('invalid simulated field after preparing rows')
    times.append(0.01)
    return index, os.getpid()


@pytest.mark.parametrize('failure', ['producer', 'invalid_sql', 'overloaded_sql'])
def test_simulation_failures_raise_and_join_the_only_writer(tmp_path, monkeypatch, failure):
    import multiprocessing
    import pickle
    import sqlite3
    from spacr import database_concurrency as database, sim
    from spacr.resource_log import _parallel_pool

    writers, attempts = [], []
    original = database._DatabaseWriteQueue

    def make_writer(*args, **kwargs):
        writer = original(*args, **kwargs)
        writers.append(writer)
        return writer

    def refuse_commit(packet):
        attempts.append(packet['field'])
        message = 'database is locked' if failure == 'overloaded_sql' else 'no such table: invalid'
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(database, '_DatabaseWriteQueue', make_writer)
    monkeypatch.setattr(sim, '_SIMULATION_WRITE_ENDPOINT', None)
    monkeypatch.setattr(sim, 'generate_parameters', lambda settings: [{**settings, 'name': 'failure'}])
    monkeypatch.setattr(sim, 'run_and_save', _simulation_rows_for_writer_failure)
    monkeypatch.setattr(sim, '_commit_simulation_packet', refuse_commit)
    monkeypatch.setattr(sim, 'Pool', lambda count, **options: _parallel_pool(
        count, context=multiprocessing.get_context('fork'), **options))
    settings = {'src': str(tmp_path), 'max_workers': 1, 'database_write_queue_gib': 0,
                'injected_failure': failure}
    expected = ValueError if failure == 'producer' else RuntimeError
    message = 'invalid simulated field' if failure == 'producer' else (
        'database is locked' if failure == 'overloaded_sql' else 'no such table')
    with pytest.raises(expected, match=message):
        sim.run_multiple_simulations(settings)
    assert len(writers) == 1
    writer = writers[0]
    assert not writer._thread.is_alive() and writer._finished
    assert writer.endpoint.failed.is_set() and writer.endpoint.reserved.value == 0
    pending = list(Path(writer.endpoint.folder).glob('*.pkl'))
    assert len(attempts) == {'producer': 0, 'invalid_sql': 1, 'overloaded_sql': 2}[failure]
    assert len(pending) == (0 if failure == 'producer' else 1)
    if pending:
        packet = pickle.loads(pending[0].read_bytes())
        assert packet['operations'][0][1][1]['value'].tolist() == [17]
    assert not list(tmp_path.rglob('simulations.db'))
