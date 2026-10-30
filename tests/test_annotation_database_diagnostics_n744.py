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
