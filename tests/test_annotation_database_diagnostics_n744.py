"""Annotation startup reports only allowlisted facts, never source identities."""
import json
import sqlite3
import sys

import pytest

from spacr import crashreport as cr
from spacr.qt import annotate_engine as engine
from tests.test_700_structured_error_dispatch import configured, joined


def _failure(path, monkeypatch):
    captured = []
    monkeypatch.setattr(engine.LOG, 'exception', lambda *args, **kwargs: captured.append(sys.exc_info()[1]))
    worker = engine.SaveWorker(str(path), 'annotate')
    worker.submit({'/private/patient/image.png': 1})
    worker.submit({'/private/patient/second.png': 2})
    worker._run()
    assert len(captured) == 1
    return worker, captured[0]


def test_failed_connection_collects_facts_without_losing_queued_labels(tmp_path, monkeypatch):
    target = tmp_path / 'private_patient' / 'not_mounted' / 'measurements.db'
    worker, exc = _failure(target, monkeypatch)
    facts = exc._spacr_annotation_database
    assert facts['target'] == facts['parent'] == 'missing'
    assert facts['path'] == 'absolute'
    assert facts['code'] == getattr(exc, 'sqlite_errorcode', None)
    if facts['code'] is not None:
        assert facts['code'] == getattr(sqlite3, 'SQLITE_CANTOPEN', 14)
    assert facts['queued'] == 2 and facts['read'] is None
    assert worker.pending_batches == 2 and 'could not start' in worker.last_error
    assert worker._q.get_nowait() == {'/private/patient/image.png': 1}
    assert worker._q.get_nowait() == {'/private/patient/second.png': 2}
    worker.submit({'later.png': 3})
    assert worker._failed_batch == {'later.png': 3}
    assert not target.parent.exists()
    message = cr._annotation_database_message(exc)
    assert len(message) <= cr.MAX_MESSAGE_CHARS
    assert 'private_patient' not in message and 'not_mounted' not in message


def test_directory_target_is_distinguished_without_opening_contents(tmp_path, monkeypatch):
    worker, exc = _failure(tmp_path, monkeypatch)
    facts = exc._spacr_annotation_database
    assert facts['target'] == facts['parent'] == 'directory'
    assert type(facts['read']) is bool and type(facts['parent_write']) is bool
    assert worker.last_error and not worker.busy


def test_report_reconstructs_allowlist_and_drops_planted_private_values(tmp_path, monkeypatch):
    worker, exc = _failure(tmp_path / 'private_patient' / 'db', monkeypatch)
    private = 'private_person secret@private.org 192.168.9.9 host.private.org /home/person/patient.npy'
    facts = exc._spacr_annotation_database
    facts.update(path=private, target=private, parent=private, read=private,
                 write=1, parent_read=private, parent_write=private,
                 code=private, queued=True, filename=private, content=private)
    summary = cr.public_summary(exc, module='annotate', diagnostics=True,
                                error_text=private, log_text=private)
    text = json.dumps(summary)
    for token in ('private_person', 'secret@', '192.168', 'host.private', '/home/person', 'patient.npy'):
        assert token not in text
    assert 'Annotation database startup' in summary['message']
    assert '"code":null' in summary['message'] and '"queued":null' in summary['message']
    assert summary['frames'] and not summary.get('chain') and not summary.get('log_tail')


def test_consented_report_carries_real_startup_facts_and_original_stack(configured, tmp_path, monkeypatch):
    relay, outcomes, ready = configured
    worker, exc = _failure(tmp_path / 'private_person' / 'measurements.db', monkeypatch)
    joined(cr._dispatch_error('writer failed', exception=exc, module='spacr.qt.annotate_engine'))
    assert ready.wait(5)
    [call] = relay.calls_to('/report')
    body = call['body']
    assert body['exception_type'] == 'OperationalError'
    assert any(frame['function'] == '_run' for frame in body['frames'])
    assert '"target":"missing"' in body['message'] and '"queued":2' in body['message']
    assert 'private_person' not in json.dumps(relay.calls)
    assert 'patient' not in json.dumps(relay.calls)


@pytest.mark.parametrize('policy', ['off', 'no-consent', 'none'])
def test_existing_reporting_and_ai_choices_are_respected(configured, tmp_path, monkeypatch, policy):
    relay, outcomes, ready = configured
    if policy == 'off':
        cr.set_error_report_mode('off')
    else:
        state = cr._read_state()
        if policy == 'no-consent':
            state.pop('affirmed_policy')
        else:
            state['affirmed_policy']['provider'] = 'none'
        cr._write_state(state)
    worker, exc = _failure(tmp_path / 'missing' / 'db', monkeypatch)
    thread = cr._dispatch_error('writer failed', exception=exc, module='spacr.qt.annotate_engine')
    if policy == 'none':
        joined(thread)
        assert ready.wait(5)
        assert not relay.calls_to('/diagnose')
        assert len(relay.calls_to('/report')) == 1
    else:
        assert thread is None and relay.calls == outcomes == []
        assert cr.queued_summaries() == []


def _queued_startup_failure(path, monkeypatch):
    """Observe the public worker's failed start without consuming its accepted edits."""
    captured = []
    monkeypatch.setattr(engine.LOG, 'exception',
                        lambda *args, **kwargs: captured.append(sys.exc_info()[1]))
    worker = engine.SaveWorker(path, 'annotate')
    labels = {'/private/patient/image.png': 1, '/private/patient/second.png': 2}
    worker.submit(labels)
    worker.start()
    worker.stop(wait=True)
    assert not worker.is_alive and worker.pending_batches == 1
    assert worker.last_save_ts is None and worker.last_error
    assert len(captured) == 1
    assert worker._q.get_nowait() == labels
    assert worker._q.get_nowait() is worker._SENTINEL
    return worker, captured[0]


def test_empty_configured_path_connection_failure_retains_labels_without_path_facts(monkeypatch):
    import errno

    real_connect = engine.connect_database

    def disk_unavailable(path, *args, **kwargs):
        if path == '':
            raise OSError(errno.ENOSPC, 'no database storage available')
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(engine, 'connect_database', disk_unavailable)
    _worker, error = _queued_startup_failure('', monkeypatch)
    facts = error._spacr_annotation_database
    assert facts['path'] == 'empty'
    assert facts['target'] == facts['parent'] == 'unknown'
    assert facts['read'] is facts['write'] is None and facts['queued'] == 1
    assert 'no database storage available' in _worker.last_error


def test_nul_in_configured_path_cannot_break_failure_reporting_or_leak_it(tmp_path, monkeypatch):
    path = str(tmp_path / 'private_patient\x00' / 'measurements.db')
    _worker, error = _queued_startup_failure(path, monkeypatch)
    assert isinstance(error, ValueError)
    facts = error._spacr_annotation_database
    assert facts['path'] == 'absolute' and facts['queued'] == 1
    assert facts['target'] == facts['parent'] == 'unknown'
    text = cr._annotation_database_message(error)
    assert 'private_patient' not in text and '\x00' not in text
    assert not (tmp_path / 'private_patient').exists()


def test_unavailable_path_provider_retains_labels_and_unknown_allowlisted_facts(tmp_path, monkeypatch):
    import errno
    import os

    existing = tmp_path / 'private_patient.db'
    existing.write_bytes(b'keep acquisition bytes')

    class UnavailablePath(os.PathLike):
        """A filesystem path provider can report a transient stale handle."""
        def __fspath__(self):
            raise OSError(errno.ESTALE, 'path provider unavailable')

    _worker, error = _queued_startup_failure(UnavailablePath(), monkeypatch)
    facts = error._spacr_annotation_database
    assert facts['path'] == facts['target'] == facts['parent'] == 'unknown'
    assert facts['queued'] == 1 and facts['code'] is None
    assert existing.read_bytes() == b'keep acquisition bytes'
    assert 'private_patient' not in cr._annotation_database_message(error)


def test_cursor_allocation_failure_closes_real_connection_and_preserves_rows(tmp_path, monkeypatch):
    path = tmp_path / 'private_patient.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE png_list(png_path TEXT, annotate INTEGER)')
        connection.execute('INSERT INTO png_list VALUES (?, ?)', ('/private/patient/image.png', 7))
    closed = []

    class CursorAllocationFailure:
        """Keep real SQLite ownership while injecting its cursor-allocation boundary."""
        def cursor(self):
            raise MemoryError('database cursor allocation failed')

        def __init__(self):
            self.connection = sqlite3.connect(path)

        def close(self):
            self.connection.close()
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                self.connection.execute('SELECT 1')
            closed.append(True)

    monkeypatch.setattr(engine, 'connect_database', lambda *args, **kwargs: CursorAllocationFailure())
    _worker, error = _queued_startup_failure(str(path), monkeypatch)
    assert isinstance(error, MemoryError) and not hasattr(error, '_spacr_annotation_database')
    assert closed == [True]
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT * FROM png_list').fetchall() == [('/private/patient/image.png', 7)]
