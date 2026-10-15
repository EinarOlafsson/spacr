"""Pruning from another thread preserves an active run's temporary journal."""
import json
import os
import threading
import time

import pytest

from spacr import run_journal as journal


@pytest.fixture
def private_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, '_spacr_home', lambda: tmp_path)
    root = tmp_path / 'runs'
    root.mkdir()
    monkeypatch.setattr(journal, 'runs_root', lambda: root)
    monkeypatch.setattr(journal, '_env_snapshot', lambda: {'packages': {}})
    monkeypatch.setattr(journal, '_write_environment_lock', lambda *args: {})
    monkeypatch.setattr(journal, '_check_lock_for_run', lambda *args: None)
    monkeypatch.setattr(journal, 'begin_recording', lambda *args, **kwargs: None)
    monkeypatch.setattr(journal, 'finish_recording', lambda *args, **kwargs: None)
    monkeypatch.setattr(journal, '_notify_run_finished', lambda *args: None)
    for name in ('_capture_initial_provenance', '_capture_final_provenance', '_snapshot_log_tail'):
        monkeypatch.setattr(journal.Run, name, lambda self: None)
    monkeypatch.setattr(journal.Run, '_write_settings', lambda self: None)
    monkeypatch.setattr(journal.Run, '_write_manifest',
                        lambda self: (self.dir / 'manifest.json').write_text(json.dumps({'status': self.status})))
    from spacr import workspace
    monkeypatch.setattr(workspace, 'save_for_run', lambda *args, **kwargs: None)
    return tmp_path


@pytest.mark.parametrize('operation', ['plan', 'delete', 'delete_child'])
def test_active_worker_run_is_protected_from_other_threads(private_journal, operation):
    opened, release = threading.Event(), threading.Event()
    active, failures = [], []

    def work():
        try:
            with journal.open_run('mask', {}) as run:
                sentinel = run.dir / 'output.txt'
                sentinel.write_bytes(b'in-progress output must survive')
                child = run.dir / 'subtask'
                child.mkdir()
                (child / 'keep.txt').write_bytes(b'active child output')
                then = time.time() - 3 * 86400
                for child in run.dir.iterdir():
                    os.utime(child, (then, then))
                os.utime(run.dir, (then, then))
                active.append(run.dir)
                opened.set()
                if not release.wait(10):
                    raise RuntimeError('test did not release its temporary worker')
        except BaseException as error:
            failures.append(error)
            opened.set()

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert opened.wait(10)
        assert not failures
        folder = active[0]
        assert folder.is_relative_to(private_journal)
        assert journal.current_run() is None
        if operation == 'plan':
            assert journal._prune_plan('run_folders', 1, 0)['delete'] == []
        else:
            chosen = folder / 'subtask' if operation == 'delete_child' else folder
            deleted, refused = journal.delete_runs([chosen])
            assert deleted == 0
            assert refused == [chosen.name + ': still running']
        assert (folder / 'output.txt').read_bytes() == b'in-progress output must survive'
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert failures == []
    assert journal.delete_runs([folder]) == (1, [])
    assert not folder.exists()


def test_failed_run_releases_its_cross_thread_prune_protection(private_journal):
    folder = None
    with pytest.raises(ValueError, match='controlled pipeline failure'):
        with journal.open_run('mask', {}) as run:
            folder = run.dir
            assert journal.delete_runs([folder])[0] == 0
            raise ValueError('controlled pipeline failure')
    assert journal.current_run() is None
    assert journal.delete_runs([folder]) == (1, [])


def test_nested_runs_keep_the_outer_run_protected(private_journal):
    with journal.open_run('mask', {}) as outer:
        with journal.open_run('measure', {}) as inner:
            assert journal.delete_runs([outer.dir, inner.dir])[0] == 0
        assert journal.current_run() is outer
        assert journal.delete_runs([inner.dir]) == (1, [])
        assert journal.delete_runs([outer.dir])[0] == 0
    assert journal.delete_runs([outer.dir]) == (1, [])


def test_finalizer_failure_releases_prune_protection(private_journal, monkeypatch):
    def fail_notification(run):
        raise ValueError('controlled notification failure')

    monkeypatch.setattr(journal, '_notify_run_finished', fail_notification)
    with pytest.raises(ValueError, match='controlled notification failure'):
        with journal.open_run('mask', {}) as run:
            folder = run.dir
    assert journal.current_run() is None
    assert journal.delete_runs([folder]) == (1, [])
