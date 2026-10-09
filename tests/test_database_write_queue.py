"""Bounded write transport with actual SQLite commits and durable failures."""
import multiprocessing
import sqlite3
import threading
from pathlib import Path

import pytest

from spacr.database_concurrency import (
    _DatabaseWriteQueue, _capture_write_packet, _commit_write_packet, connect, transaction,
)
from spacr.cancellation import PipelineCancelled


def test_measure_worker_sends_frozen_rows_without_opening_sqlite(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from spacr import measure, utils

    database = tmp_path / 'measurements' / 'measurements.db'
    database.parent.mkdir()
    results = []
    writer = _DatabaseWriteQueue(database.parent / '.write_queue',
        lambda packet: measure._commit_measure_packet(database, packet),
        lambda *row: results.append(row), ram_gib=0).start()
    frame = pd.DataFrame({'object_label': [1], 'intensity': [4.0]})

    def compute(index, times, file, settings, *optional):
        utils._append_to_measurements_db(str(database), 'cell', frame)
        assert not database.exists()
        frame.loc[0, 'intensity'] = 999.0
        return index, 1.0, np.array([0, 1]), {}, ''

    monkeypatch.setattr(measure, '_MEASURE_WRITE_ENDPOINT', writer.endpoint)
    monkeypatch.setattr(measure, '_measure_crop_core', compute)
    result = measure._measure_crop_queued(0, [], 'plate1_A01_F001.npy', {})
    writer.finish()
    assert result[5] == results[0][1] and results[0][2] is None
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT intensity FROM cell').fetchall() == [(4.0,)]


def test_failed_measure_field_discards_earlier_captured_operations(tmp_path, monkeypatch):
    import pandas as pd
    from spacr import measure, utils

    database = tmp_path / 'measurements.db'
    outcomes = []
    writer = _DatabaseWriteQueue(tmp_path / '.write_queue',
        lambda packet: measure._commit_measure_packet(database, packet),
        lambda *row: outcomes.append(row), ram_gib=0).start()

    def compute(index, *_args):
        utils._append_to_measurements_db(str(database), 'cell',
                                        pd.DataFrame({'object_label': [1]}))
        return index, 1.0, 0, {}, 'original computation traceback'

    monkeypatch.setattr(measure, '_MEASURE_WRITE_ENDPOINT', writer.endpoint)
    monkeypatch.setattr(measure, '_measure_crop_core', compute)
    result = measure._measure_crop_queued(0, [], 'field', {})
    writer.finish()
    assert result[4] == 'original computation traceback'
    assert outcomes == [] and not database.exists()


def test_measure_packet_cannot_write_another_database(tmp_path):
    from spacr import measure
    import pandas as pd

    packet = {'field': 'field', 'ticket': 'ticket', 'operations': [
        ('append', (str(tmp_path / 'other.db'), 'cell',
                    pd.DataFrame({'object_label': [1]}), True, None), {})]}
    with pytest.raises(ValueError, match='different database'):
        measure._commit_measure_packet(tmp_path / 'central.db', packet)
    assert not (tmp_path / 'other.db').exists()


@pytest.mark.parametrize('value', [-1, 65, float('nan'), float('inf')])
def test_invalid_headless_write_budget_is_refused(value):
    from spacr import measure

    with pytest.raises(ValueError):
        measure._measure_write_queue_budget({'database_write_queue_gib': value})


def _sqlite_dispatch(path, packet):
    with connect(path) as connection:
        with transaction(connection):
            connection.execute('CREATE TABLE IF NOT EXISTS tickets (ticket TEXT PRIMARY KEY)')
            connection.execute('CREATE TABLE IF NOT EXISTS rows (field TEXT, value TEXT)')
            if connection.execute('SELECT 1 FROM tickets WHERE ticket=?',
                                  (packet['ticket'],)).fetchone():
                return
            if packet['operations'] is None:
                raise ValueError('An uncommitted packet has no durable data')
            for value in packet['operations']:
                connection.execute('INSERT INTO rows VALUES (?, ?)',
                                   (packet['field'], value))
            connection.execute('INSERT INTO tickets VALUES (?)', (packet['ticket'],))


def _enqueue_in_child(endpoint, field, value):
    endpoint.enqueue(field, [value])


def _simulation_in_child(index, settings):
    import os
    import pandas as pd
    from spacr import sim

    def compute(i, options, times, total):
        root = Path(options['src'])
        primary = root / f'primary-{i}'
        retry = primary.exists()
        primary.write_text('primary complete')
        if i == 0 and not retry:
            raise MemoryError('injected primary simulation overload')
        if i == 0:
            assert all((root / f'primary-{other}').exists() for other in range(3))
        folder = root / options['start_time'] / options['name']
        folder.mkdir(parents=True, exist_ok=True)
        sim.append_database(str(folder), pd.DataFrame({
            'simulation_index': [i], 'producer_pid': [os.getpid()]}), 'simulations')
        return i, 0.1

    def refuse_worker_database(*args, **kwargs):
        raise AssertionError('A simulation processing worker opened SQLite')

    sim.run_and_save = compute
    sqlite3.connect = refuse_worker_database
    return sim._run_and_save_queued(index, settings, [], 3)


def _large_simulation_driver(folder):
    import os
    import time
    import pandas as pd
    from spacr import sim
    from spacr.resource_log import _parallel_pool, _WorkerStartGate

    os.setsid()
    sim.generate_parameters = lambda settings: [dict(settings) for _ in range(12)]
    sim.Pool = lambda workers, **kwargs: _parallel_pool(workers,
        context=multiprocessing.get_context('fork'), gate=_WorkerStartGate(delay=0), **kwargs)

    def calculate(index, settings, times, total):
        destination = Path(settings['src']) / settings['start_time'] / settings['name']
        destination.mkdir(parents=True, exist_ok=True)
        sim.append_database(str(destination), pd.DataFrame({
            'simulation_index': [index], 'payload': ['x' * 300000]}), 'simulations')
        return index, 0.01

    original = sim._commit_simulation_packet
    def slow_commit(packet):
        time.sleep(0.03)
        return original(packet)

    sim.run_and_save = calculate
    sim._commit_simulation_packet = slow_commit
    sim.run_multiple_simulations({'src': folder, 'name': 'queued', 'max_workers': 2,
        'database_write_queue_gib': 1})


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(),
    reason='bounded production shutdown probe needs inherited synthetic calculation')
def test_simulation_shutdown_flushes_real_ram_packets_before_writer_sentinel(tmp_path):
    import os
    import signal

    child = multiprocessing.get_context('spawn').Process(
        target=_large_simulation_driver, args=(str(tmp_path),))
    child.start()
    try:
        child.join(25)
        assert child.exitcode == 0, 'Simulation producer/writer shutdown did not complete'
        databases = list(tmp_path.glob('*/queued/simulations.db'))
        assert len(databases) == 1
        with connect(databases[0], readonly=True) as connection:
            rows = connection.execute('SELECT simulation_index, LENGTH(payload) FROM simulations ORDER BY simulation_index').fetchall()
        assert rows == [(index, 300000) for index in range(12)]
        assert not list(tmp_path.glob('*/.simulation_write_queue/*/*.pkl'))
    finally:
        if child.is_alive():
            os.killpg(child.pid, signal.SIGTERM)
            child.join(5)
            if child.is_alive():
                os.killpg(child.pid, signal.SIGKILL)
                child.join(5)
        child.close()


def test_spawned_simulation_pool_spills_and_commits_after_separate_retry_queues(tmp_path):
    import os
    import time
    from spacr import sim
    from spacr.resource_log import _parallel_pool

    context = multiprocessing.get_context('spawn')
    seen, outcomes = [], []
    destination = tmp_path / 'run' / 'batch' / 'simulations.db'

    def dispatch(packet):
        import json
        _, index = json.loads(packet['field'])
        assert threading.current_thread().name == 'spacr-database-writer'
        if index == 0 and 0 not in seen:
            seen.append(index)
            raise sqlite3.OperationalError('database is locked')
        if index == 0:
            assert set(seen) == {0, 1, 2}
        seen.append(index)
        return sim._commit_simulation_packet(packet)

    writer = _DatabaseWriteQueue(destination.parent / '.simulation_write_queue',
        dispatch, lambda *row: outcomes.append(row), ram_gib=0, context=context, slots=1)
    started = time.monotonic()
    try:
        with _parallel_pool(2, context=context,
                initializer=sim._initialize_simulation_writer,
                initargs=(writer.endpoint,)) as pool:
            assert time.monotonic() - started >= 9.8
            writer.start()
            options = {'src': str(tmp_path), 'start_time': 'run', 'name': 'batch'}
            results = pool.starmap_async(_simulation_in_child,
                [(index, options) for index in range(3)]).get(timeout=40)
            assert results == [(0, 0.1), (1, 0.1), (2, 0.1)]
        writer.finish()
    finally:
        if not writer._finished and writer._thread is not None:
            writer.cancel()
            writer.finish()
    assert all(error is None for _, _, error in outcomes)
    assert len(outcomes) == 3 and seen.count(0) == 2
    assert writer.endpoint.reserved.value == 0
    assert not list(Path(writer.endpoint.folder).glob('*.pkl'))
    with connect(destination, readonly=True) as connection:
        rows = connection.execute('SELECT simulation_index, producer_pid FROM simulations ORDER BY simulation_index').fetchall()
    assert [row[0] for row in rows] == [0, 1, 2]
    assert all(row[1] != os.getpid() for row in rows)


@pytest.mark.parametrize('ram_gib', [0.0, 0.001])
def test_only_the_writer_thread_commits_and_spool_disappears_after_commit(tmp_path, ram_gib):
    database = tmp_path / 'measurements.db'
    writes, outcomes = [], []
    def dispatch(packet):
        writes.append(threading.current_thread().name)
        _sqlite_dispatch(database, packet)
    writer = _DatabaseWriteQueue(tmp_path / '.write_queue', dispatch,
                                lambda *row: outcomes.append(row), ram_gib=ram_gib)
    ticket = writer.endpoint.enqueue('field-1', ['original pixels'])
    assert list(Path(writer.endpoint.folder).glob('*.pkl'))
    assert 0 <= writer.endpoint.reserved.value <= writer.endpoint.limit_bytes
    writer.start()
    writer.finish()
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT * FROM rows').fetchall() == [
            ('field-1', 'original pixels')]
    assert writes == ['spacr-database-writer']
    assert outcomes == [('field-1', ticket, None)]
    assert writer.endpoint.reserved.value == 0
    assert not list(Path(writer.endpoint.folder).glob('*.pkl'))


def test_budget_overflow_goes_to_disk_without_growing_the_reserved_counter(tmp_path):
    writer = _DatabaseWriteQueue(tmp_path, lambda _: None, lambda *_: None,
                                ram_gib=2048 / 1024 ** 3)
    writer.endpoint.enqueue('fits', ['small'])
    first = writer.endpoint.reserved.value
    assert 1024 < first <= 2048
    writer.endpoint.enqueue('overflow', ['large' * 2048])
    assert writer.endpoint.reserved.value == first
    writer.start()
    writer.finish()
    assert writer.endpoint.reserved.value == 0


def test_sql_overload_final_retry_follows_all_primary_packets(tmp_path):
    seen, outcomes = [], []
    database = tmp_path / 'measurements.db'
    def dispatch(packet):
        identity = packet['field']
        if identity == 'busy' and identity not in seen:
            seen.append(identity)
            raise sqlite3.OperationalError('database is locked')
        if identity == 'busy':
            assert seen == ['busy', 'other-1', 'other-2']
        seen.append(identity)
        _sqlite_dispatch(database, packet)
    writer = _DatabaseWriteQueue(tmp_path / 'spool', dispatch,
                                lambda *row: outcomes.append(row), ram_gib=0)
    for identity in ['busy', 'other-1', 'other-2']:
        writer.endpoint.enqueue(identity, [identity])
    writer.start()
    writer.finish()
    assert seen == ['busy', 'other-1', 'other-2', 'busy']
    assert all(error is None for _, _, error in outcomes)
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT COUNT(*) FROM rows').fetchone()[0] == 3


def test_exhausted_final_retry_retains_data_and_full_original_traceback(tmp_path):
    calls, outcomes = [], []
    def busy(packet):
        calls.append(packet['field'])
        raise sqlite3.OperationalError('database is locked')
    writer = _DatabaseWriteQueue(tmp_path, busy, lambda *row: outcomes.append(row), ram_gib=0)
    ticket = writer.endpoint.enqueue('field', ['unchanged data'])
    writer.start()
    writer.finish()
    assert calls == ['field', 'field']
    assert isinstance(outcomes[0][2], sqlite3.OperationalError)
    assert isinstance(outcomes[0][2].__cause__, sqlite3.OperationalError)
    spool = Path(writer.endpoint.folder) / (ticket + '.pkl')
    assert spool.exists()
    evidence = Path(str(spool) + '.error.json').read_text()
    assert 'in busy' in evidence and 'database is locked' in evidence


def test_duplicate_ticket_does_not_append_rows_twice_in_disk_only_mode(tmp_path):
    database = tmp_path / 'measurements.db'
    outcomes = []
    writer = _DatabaseWriteQueue(tmp_path / 'spool',
        lambda packet: _sqlite_dispatch(database, packet),
        lambda *row: outcomes.append(row), ram_gib=0)
    first = writer.endpoint.enqueue('same field', ['one'])
    assert writer.endpoint.enqueue('same field', ['one']) == first
    writer.start()
    writer.finish()
    assert all(error is None for _, _, error in outcomes)
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT COUNT(*) FROM rows').fetchone()[0] == 1


def test_spawned_producer_does_not_open_or_write_sqlite(tmp_path):
    database = tmp_path / 'measurements.db'
    outcomes = []
    context = multiprocessing.get_context('spawn')
    writer = _DatabaseWriteQueue(tmp_path / 'spool',
        lambda packet: _sqlite_dispatch(database, packet),
        lambda *row: outcomes.append(row), ram_gib=0, context=context).start()
    child = context.Process(target=_enqueue_in_child,
                            args=(writer.endpoint, 'child-field', 'child data'))
    child.start()
    child.join(20)
    assert child.exitcode == 0
    writer.finish()
    assert len(outcomes) == 1 and outcomes[0][2] is None
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT * FROM rows').fetchall() == [
            ('child-field', 'child data')]


def test_invalid_data_is_not_retried_and_remains_on_disk(tmp_path):
    calls, outcomes = [], []
    def invalid(packet):
        calls.append(packet['field'])
        raise ValueError('invalid column')
    writer = _DatabaseWriteQueue(tmp_path, invalid, lambda *row: outcomes.append(row), ram_gib=0)
    ticket = writer.endpoint.enqueue('invalid', ['data'])
    writer.start()
    writer.finish()
    assert calls == ['invalid'] and isinstance(outcomes[0][2], ValueError)
    assert (Path(writer.endpoint.folder) / (ticket + '.pkl')).exists()


def test_cancel_preserves_queued_payloads_and_does_not_begin_new_writes(tmp_path):
    calls = []
    writer = _DatabaseWriteQueue(tmp_path, lambda packet: calls.append(packet),
                                lambda *_: None, ram_gib=0)
    ticket = writer.endpoint.enqueue('unfinished', ['data'])
    writer.cancel()
    writer.start()
    with pytest.raises(PipelineCancelled):
        writer.finish()
    assert calls == []
    assert (Path(writer.endpoint.folder) / (ticket + '.pkl')).exists()


def test_disk_full_during_serialization_does_not_charge_ram_or_enqueue(tmp_path, monkeypatch):
    import errno
    import pickle
    writer = _DatabaseWriteQueue(tmp_path, lambda _: None, lambda *_: None)
    def full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, 'disk full')
    monkeypatch.setattr(pickle, 'dump', full)
    with pytest.raises(OSError):
        writer.endpoint.enqueue('not saved', ['data'])
    assert writer.endpoint.reserved.value == 0
    assert not list(Path(writer.endpoint.folder).iterdir())
    writer.start()
    writer.finish()


def test_packet_failure_rolls_back_earlier_helper_commits_and_schema(tmp_path):
    database = tmp_path / 'measurements.db'
    def helper(operation):
        connection = connect(database)
        try:
            connection.execute('CREATE TABLE IF NOT EXISTS cells (value INTEGER)')
            connection.execute('INSERT INTO cells VALUES (?)', (operation,))
            connection.commit()
            if operation == 2:
                raise ValueError('invalid later operation')
        finally:
            connection.close()
    packet = {'ticket': 'original ticket', 'field': 'field', 'operations': [1, 2]}
    with pytest.raises(ValueError):
        _commit_write_packet(database, packet, helper)
    with connect(database, readonly=True) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


def test_packet_ticket_and_rows_commit_together_and_replay_is_idempotent(tmp_path):
    database = tmp_path / 'measurements.db'
    def helper(operation):
        with connect(database) as connection:
            with transaction(connection):
                connection.execute('CREATE TABLE IF NOT EXISTS cells (value INTEGER)')
                connection.execute('INSERT INTO cells VALUES (?)', (operation,))
    packet = {'ticket': 'same ticket', 'field': 'field', 'operations': [1, 2]}
    assert _commit_write_packet(database, packet, helper)
    assert not _commit_write_packet(database, {**packet, 'operations': None}, helper)
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT * FROM cells').fetchall() == [(1,), (2,)]
        assert connection.execute('SELECT * FROM _spacr_write_queue_commits').fetchall() == [
            ('same ticket', 'field')]


def test_real_measurement_append_widening_remains_atomic_under_owned_connection(tmp_path):
    import pandas as pd
    from spacr.utils import _append_to_measurements_db
    database = tmp_path / 'measurements.db'
    first = pd.DataFrame({'original': [1]})
    second = pd.DataFrame({'original': [2], 'added': [3]})
    def helper(frame):
        _append_to_measurements_db(str(database), 'diagnostic', frame)
    packet = {'ticket': 'widening', 'field': 'field', 'operations': [first, second]}
    assert _commit_write_packet(database, packet, helper)
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT original, added FROM diagnostic').fetchall() == [
            (1, None), (2, 3)]


def test_captured_append_does_not_open_database_and_freezes_original_values(tmp_path):
    import pandas as pd
    from spacr.utils import _append_to_measurements_db
    database = tmp_path / 'measurements.db'
    frame = pd.DataFrame({'value': [7]})
    with _capture_write_packet() as operations:
        _append_to_measurements_db(str(database), 'diagnostic', frame)
        frame.loc[0, 'value'] = 99
    assert not database.exists()
    assert len(operations) == 1 and operations[0][1][2].value.iloc[0] == 7
    def dispatch(operation):
        name, arguments, keywords = operation
        assert name == 'append'
        _append_to_measurements_db(*arguments, **keywords)
    packet = {'ticket': 'captured', 'field': 'field', 'operations': operations}
    _commit_write_packet(database, packet, dispatch)
    with connect(database, readonly=True) as connection:
        assert connection.execute('SELECT * FROM diagnostic').fetchall() == [(7,)]


def test_nonlock_sql_error_cannot_acknowledge_a_partially_saved_packet(tmp_path, monkeypatch):
    import pandas as pd
    import spacr.utils as utils
    database = tmp_path / 'measurements.db'
    original_append = utils._append_frame
    def append(connection, table, frame):
        if frame.value.iloc[0] == 2:
            raise sqlite3.OperationalError('no such column: broken')
        original_append(connection, table, frame)
    monkeypatch.setattr(utils, '_append_frame', append)
    packet = {'ticket': 'not committed', 'field': 'field', 'operations': [
        pd.DataFrame({'value': [1]}), pd.DataFrame({'value': [2]})]}
    with pytest.raises(sqlite3.OperationalError, match='broken'):
        _commit_write_packet(database, packet, lambda frame:
            utils._append_to_measurements_db(str(database), 'diagnostic', frame))
    with connect(database, readonly=True) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='diagnostic'").fetchall() == []


def test_capture_refuses_external_mirrors_before_sqlite_writes(tmp_path):
    import pandas as pd
    from spacr.utils import _append_to_measurements_db
    database = tmp_path / 'measurements.db'
    with _capture_write_packet():
        with pytest.raises(ValueError, match='external-store'):
            _append_to_measurements_db(str(database), 'diagnostic',
                pd.DataFrame({'value': [1]}), store=str(tmp_path / 'mirror.duckdb'))
    assert not database.exists()


def test_simulation_tables_use_one_atomic_ticket_through_the_normal_table_writer(tmp_path):
    import json
    import pandas as pd
    from spacr import sim
    from spacr.database_concurrency import _WRITE_TICKETS_TABLE
    path = tmp_path / 'simulations.db'
    packet = {'field': json.dumps([str(path), 1]), 'ticket': 'simulation-1',
              'operations': [('simulation', (str(tmp_path),
                             pd.DataFrame({'value': [4]}), 'simulations'), {})]}
    assert sim._commit_simulation_packet(packet)
    assert sim._commit_simulation_packet(packet) is False
    assert sim._commit_simulation_packet(dict(packet, operations=None)) is False
    with connect(path, readonly=True) as db:
        assert db.execute('SELECT value FROM simulations').fetchall() == [(4,)]
        assert db.execute(f'SELECT COUNT(*) FROM {_WRITE_TICKETS_TABLE}').fetchone()[0] == 1


def test_simulation_packet_rolls_back_all_prior_tables_when_a_later_append_fails(tmp_path):
    import json
    import pandas as pd
    from spacr import sim
    path = tmp_path / 'simulations.db'
    packet = {'field': json.dumps([str(path), 2]), 'ticket': 'simulation-2',
              'operations': [('simulation', (str(tmp_path), frame, 'simulations'), {})
                             for frame in (pd.DataFrame({'value': [4]}),
                                           pd.DataFrame({'unknown_column': [5]}))]}
    with pytest.raises((sqlite3.OperationalError, pd.errors.DatabaseError)) as raised:
        sim._commit_simulation_packet(packet)
    cause = (raised.value.__cause__
             if isinstance(raised.value, pd.errors.DatabaseError)
             else raised.value)
    assert isinstance(cause, sqlite3.OperationalError)
    assert 'unknown_column' in str(cause)
    with connect(path, readonly=True) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='simulations'").fetchall() == []


def test_simulation_capture_failure_does_not_enqueue_or_commit_a_partial_result(tmp_path, monkeypatch):
    from spacr import sim
    from types import SimpleNamespace
    import pandas as pd
    calls = []
    monkeypatch.setattr(sim, '_SIMULATION_WRITE_ENDPOINT', SimpleNamespace(
        enqueue=lambda *args: calls.append(args)))
    def fail(_i, settings, *_args):
        sim.append_database(str(tmp_path), pd.DataFrame({'value': [4]}), 'simulations')
        raise MemoryError('failed after preparing the first table')
    monkeypatch.setattr(sim, 'run_and_save', fail)
    with pytest.raises(MemoryError):
        sim._run_and_save_queued(0, {}, [], 1)
    assert calls == [] and not (tmp_path / 'simulations.db').exists()


def test_simulation_invalid_capture_is_not_hidden_as_success(tmp_path):
    from spacr import sim
    import pandas as pd
    with _capture_write_packet():
        with pytest.raises(IndexError):
            sim.save_data(str(tmp_path), [pd.DataFrame({'value': [4]})], {})
    assert not (tmp_path / 'simulations.db').exists()
