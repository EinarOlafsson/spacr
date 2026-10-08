"""Durable queue behavior at failure, backpressure and shutdown boundaries."""

import errno
from dataclasses import replace
import pickle
import queue
import sqlite3
from pathlib import Path

import pytest

from spacr import database_concurrency as database
from spacr.cancellation import PipelineCancelled


def test_nested_capture_rejection_preserves_the_outer_packet():
    with database._capture_write_packet() as operations:
        with pytest.raises(RuntimeError, match='already being captured'):
            with database._capture_write_packet():
                pytest.fail('Nested capture was accepted')
        with pytest.raises(ValueError, match='Unsupported'):
            database._capture_write_operation('unexpected', ())
        assert database._capture_write_operation('simulation', ('scientific rows',))
        assert operations == [('simulation', ('scientific rows',), {})]
    assert not database._write_capture_active()


@pytest.mark.parametrize('budget', [-1, float('nan'), float('inf')])
def test_invalid_writer_budget_never_creates_a_spool(tmp_path, budget):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        database._DatabaseWriteQueue(tmp_path, lambda _: None, lambda *_: None,
                                     ram_gib=budget)
    assert list(tmp_path.iterdir()) == []


def test_writer_start_and_finish_boundaries_do_not_repeat_dispatch(tmp_path):
    packets = []
    writer = database._DatabaseWriteQueue(tmp_path, packets.append, lambda *_: None)
    with pytest.raises(RuntimeError, match='not started'):
        writer.finish()
    writer.endpoint.enqueue('field', ['original rows'])
    writer.start()
    with pytest.raises(RuntimeError, match='already started'):
        writer.start()
    writer.finish()
    writer.finish()
    assert len(packets) == 1 and packets[0]['operations'] == ['original rows']


@pytest.mark.parametrize('budget', [0, 0.001])
def test_failed_writer_rejects_producer_without_losing_durable_rows(tmp_path, budget):
    writer = database._DatabaseWriteQueue(tmp_path, lambda _: pytest.fail('Dispatched after stop'),
                                         lambda *_: None, ram_gib=budget)
    writer.cancel()
    with pytest.raises(RuntimeError, match='writer stopped'):
        writer.endpoint.enqueue('unfinished', ['original rows'])
    assert writer.endpoint.reserved.value == 0
    paths = list(Path(writer.endpoint.folder).glob('*.pkl'))
    assert len(paths) == 1
    assert pickle.loads(paths[0].read_bytes())['operations'] == ['original rows']
    writer.start()
    for _ in range(2):
        with pytest.raises(PipelineCancelled):
            writer.finish()
    assert paths[0].exists()


def test_transport_backpressure_preserves_one_packet_and_shutdown(tmp_path):
    dispatched, outcomes = [], []
    writer = database._DatabaseWriteQueue(tmp_path, dispatched.append,
                                         lambda *row: outcomes.append(row))
    original = writer.endpoint.inbox

    class TemporarilyFull:
        def __init__(self):
            self.failed_packet = self.failed_sentinel = False

        def put(self, item, **kwargs):
            attribute = 'failed_sentinel' if item is None else 'failed_packet'
            if not getattr(self, attribute):
                setattr(self, attribute, True)
                raise queue.Full
            return original.put(item, **kwargs)

        def __getattr__(self, name):
            return getattr(original, name)

    transport = TemporarilyFull()
    writer.endpoint = replace(writer.endpoint, inbox=transport)
    ticket = writer.endpoint.enqueue('field', ['original rows'])
    writer.start()
    writer.finish()
    assert transport.failed_packet and transport.failed_sentinel
    assert len(dispatched) == 1 and dispatched[0]['operations'] == ['original rows']
    assert outcomes == [('field', ticket, None)]
    assert writer.endpoint.reserved.value == 0
    assert not list(Path(writer.endpoint.folder).glob('*.pkl'))


@pytest.mark.parametrize('damage', ['wrong_ticket', 'wrong_field', 'missing'])
def test_damaged_disk_packet_cannot_dispatch_or_report_success(tmp_path, damage):
    dispatched, outcomes = [], []
    writer = database._DatabaseWriteQueue(tmp_path, dispatched.append,
                                         lambda *row: outcomes.append(row), ram_gib=0)
    ticket = writer.endpoint.enqueue('field', ['original rows'])
    path = Path(writer.endpoint.folder) / (ticket + '.pkl')
    if damage == 'missing':
        path.unlink()
    else:
        packet = pickle.loads(path.read_bytes())
        packet['ticket' if damage == 'wrong_ticket' else 'field'] = 'different'
        path.write_bytes(pickle.dumps(packet))
    writer.start()
    writer.finish()
    assert dispatched == [] and len(outcomes) == 1
    assert isinstance(outcomes[0][2], FileNotFoundError if damage == 'missing' else ValueError)
    assert Path(str(path) + '.error.json').is_file()
    assert writer.endpoint.reserved.value == 0


def test_committed_ticket_rejects_another_field_and_absent_payload(tmp_path):
    path = tmp_path / 'measurements.db'
    seen = []
    packet = {'ticket': 'stable', 'field': 'field', 'operations': ['rows']}
    assert database._commit_write_packet(path, packet, seen.append)
    with pytest.raises(ValueError, match='another field'):
        database._commit_write_packet(path, {**packet, 'field': 'other'}, seen.append)
    with pytest.raises(ValueError, match='no scientific payload'):
        database._commit_write_packet(path, {**packet, 'ticket': 'unsaved', 'operations': None}, seen.append)
    assert not database._commit_write_packet(path, packet, seen.append)
    assert seen == ['rows']
    with database.connect(path, readonly=True) as connection:
        assert connection.execute('SELECT ticket, field FROM _spacr_write_queue_commits').fetchall() == [('stable', 'field')]


def test_packet_connection_outside_writer_keeps_normal_context_commit(tmp_path):
    connection = sqlite3.connect(tmp_path / 'normal.db', factory=database._PacketConnection)
    try:
        with connection:
            connection.execute('CREATE TABLE rows (value INTEGER)')
            connection.execute('INSERT INTO rows VALUES (4)')
        with connection:
            assert connection.execute('SELECT value FROM rows').fetchall() == [(4,)]
    finally:
        connection.close()


@pytest.mark.parametrize('stage', ['commit_marker', 'retry_spool'])
def test_disk_full_after_primary_read_retains_scientific_packet(tmp_path, monkeypatch, stage):
    import json

    outcomes, attempts = [], []

    def dispatch(packet):
        attempts.append(packet['field'])
        if stage == 'retry_spool':
            raise sqlite3.OperationalError('database is locked')

    writer = database._DatabaseWriteQueue(tmp_path, dispatch,
                                         lambda *row: outcomes.append(row), ram_gib=0)
    ticket = writer.endpoint.enqueue('field', ['original rows'])
    path = Path(writer.endpoint.folder) / (ticket + '.pkl')

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, 'disk full at durable metadata')

    monkeypatch.setattr(json if stage == 'commit_marker' else pickle,
                        'dump', disk_full)
    writer.start()
    if stage == 'retry_spool':
        with pytest.raises(OSError, match='disk full'):
            writer.finish()
        assert writer.endpoint.failed.is_set() and outcomes == []
    else:
        writer.finish()
        assert len(outcomes) == 1 and isinstance(outcomes[0][2], OSError)
    assert attempts == ['field']
    assert pickle.loads(path.read_bytes())['operations'] == ['original rows']
    assert not list(Path(writer.endpoint.folder).glob('*.partial'))
    assert writer.endpoint.reserved.value == 0


def test_cancel_between_primary_and_final_retry_keeps_pending_disk_data(tmp_path):
    attempts, outcomes = [], []

    def dispatch(packet):
        attempts.append(packet['field'])
        raise sqlite3.OperationalError('database is locked')

    writer = database._DatabaseWriteQueue(tmp_path, dispatch,
                                         lambda *row: outcomes.append(row), ram_gib=0)
    original = writer.endpoint.inbox

    class CancelAtPrimaryEnd:
        def get(self, **kwargs):
            item = original.get(**kwargs)
            if item is None:
                writer.cancel()
            return item

        def __getattr__(self, name):
            return getattr(original, name)

    writer.endpoint = replace(writer.endpoint, inbox=CancelAtPrimaryEnd())
    ticket = writer.endpoint.enqueue('busy', ['original rows'])
    writer.start()
    with pytest.raises(PipelineCancelled):
        writer.finish()
    assert attempts == ['busy']
    assert not any(field == 'busy' for field, _, _ in outcomes)
    assert (Path(writer.endpoint.folder) / (ticket + '.pkl')).exists()
    assert writer.endpoint.reserved.value == 0


@pytest.mark.parametrize('final_failure', [False, True])
def test_duplicate_overloaded_delivery_has_one_final_attempt_and_atomic_rows(
        tmp_path, final_failure):
    path = tmp_path / 'measurements.db'
    attempts, outcomes = [], []

    def dispatch(packet):
        attempts.append(packet['ticket'])
        if len(attempts) <= 2 or final_failure:
            raise sqlite3.OperationalError('database is locked')

        def write(operation):
            with database.connect(path) as connection:
                connection.execute('CREATE TABLE IF NOT EXISTS results (value INTEGER)')
                connection.execute('INSERT INTO results VALUES (?)', operation)

        database._commit_write_packet(path, packet, write)

    writer = database._DatabaseWriteQueue(tmp_path / '.write_queue', dispatch,
                                         lambda *row: outcomes.append(row), ram_gib=0)
    ticket = writer.endpoint.enqueue('same field', [(17,), (23,)])
    writer.endpoint.inbox.put((ticket, 'same field',
                              str(Path(writer.endpoint.folder) / (ticket + '.pkl')),
                              None, 0))
    writer.start()
    writer.finish()
    assert attempts == [ticket, ticket, ticket]
    assert len(outcomes) == 1 and outcomes[0][:2] == ('same field', ticket)
    assert not list(Path(writer.endpoint.folder).glob('*.retry'))
    assert writer.endpoint.reserved.value == 0
    if final_failure:
        assert len(list(Path(writer.endpoint.folder).glob('*.exhausted'))) == 1
        assert isinstance(outcomes[0][2], sqlite3.OperationalError)
        assert not path.exists()
        assert (Path(writer.endpoint.folder) / (ticket + '.pkl')).is_file()
    else:
        assert not list(Path(writer.endpoint.folder).glob('*.exhausted'))
        assert outcomes[0][2] is None
        with database.connect(path, readonly=True) as connection:
            assert connection.execute('SELECT value FROM results').fetchall() == [(17,), (23,)]
            assert connection.execute('SELECT ticket FROM _spacr_write_queue_commits').fetchall() == [(ticket,)]
        assert not list(Path(writer.endpoint.folder).glob('*.pkl'))
