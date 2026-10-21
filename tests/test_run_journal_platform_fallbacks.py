"""Run ownership and notification persistence fail without exposing secrets."""
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def test_owner_probe_failure_is_cached_as_unknown(monkeypatch):
    from spacr import install_cleanup, run_journal

    calls = []

    def unavailable(pid):
        calls.append(pid)
        raise OSError('process start stamp unavailable')

    monkeypatch.setattr(run_journal, '_OWN_RUN_OWNER', {})
    monkeypatch.setattr(install_cleanup, '_process_started', unavailable)
    assert run_journal._run_owner() is None
    assert run_journal._run_owner() is None
    assert calls == [os.getpid()]


def test_an_unreadable_existing_owner_is_not_marked_dead(monkeypatch):
    from spacr import install_cleanup, run_journal

    owner = {'pid': str(os.getpid() + 100000), 'started': 'test:1',
             'host': install_cleanup._lease_host(), 'pidns': install_cleanup._pid_namespace()}
    monkeypatch.setattr(install_cleanup, '_process_gone', lambda pid: False)
    monkeypatch.setattr(install_cleanup, '_process_started', lambda pid: None)
    assert not run_journal._run_owner_gone(owner)


@pytest.fixture
def windows_journal(monkeypatch):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, 'os', SimpleNamespace(
        name='nt', getpid=os.getpid, environ={'USERNAME': 'review-user', 'USERDOMAIN': 'LAB'}))
    return run_journal


@pytest.mark.parametrize('folder, grant', [(False, 'LAB\\review-user:F'), (True, 'LAB\\review-user:(OI)(CI)F')])
def test_private_windows_acl_removes_inheritance_before_granting_owner(
        windows_journal, tmp_path, monkeypatch, folder, grant):
    calls = []

    def apply(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', apply)
    target = tmp_path / 'notification-secrets'
    windows_journal._restrict_to_owner(target, folder=folder)
    command, options = calls[0]
    assert command == ['icacls', str(target), '/inheritance:r', '/grant:r', grant]
    assert options['timeout'] == 30
    assert options['capture_output'] and options['text']


@pytest.mark.parametrize('failure', ['missing-account', 'launch', 'nonzero-stderr', 'nonzero-stdout'])
def test_unsettable_windows_acl_refuses_secret_persistence(
        windows_journal, tmp_path, monkeypatch, failure):
    calls = []
    if failure == 'missing-account':
        windows_journal.os.environ.clear()

    def apply(command, **kwargs):
        calls.append(command)
        if failure == 'launch':
            raise OSError('ACL tool unavailable')
        return SimpleNamespace(returncode=1,
                               stdout='denied' if failure == 'nonzero-stdout' else '',
                               stderr='denied' if failure == 'nonzero-stderr' else '')

    monkeypatch.setattr(subprocess, 'run', apply)
    with pytest.raises(OSError):
        windows_journal._restrict_to_owner(tmp_path / 'empty-secret')
    assert len(calls) == (0 if failure == 'missing-account' else 1)
    assert not list(tmp_path.iterdir())


def test_private_staging_is_secured_before_becoming_available(windows_journal, tmp_path, monkeypatch):
    secured = []

    def restrict(path, *, folder=False):
        assert folder and path.is_dir()
        assert list(path.iterdir()) == []
        secured.append(path)

    monkeypatch.setattr(windows_journal, '_restrict_to_owner', restrict)
    destination = tmp_path / 'secrets.json'
    staging = windows_journal._private_secret_staging(destination)
    assert staging.parent == destination.parent
    assert staging != destination.parent
    assert secured == [staging]
    staging.rmdir()


@pytest.mark.parametrize('cleanup_refused', [False, True])
def test_refused_staging_acl_never_creates_a_secret(
        windows_journal, tmp_path, monkeypatch, cleanup_refused):
    folders = []

    def refuse(path, *, folder=False):
        folders.append(path)
        raise OSError('private ACL refused')

    monkeypatch.setattr(windows_journal, '_restrict_to_owner', refuse)
    original = Path.rmdir
    if cleanup_refused:
        def busy(path):
            if path in folders:
                raise OSError('folder busy')
            return original(path)
        monkeypatch.setattr(Path, 'rmdir', busy)
    with pytest.raises(OSError, match='private ACL refused'):
        windows_journal._private_secret_staging(tmp_path / 'secrets.json')
    assert all(not list(path.iterdir()) for path in folders if path.exists())
    assert not (tmp_path / 'secrets.json').exists()
    for path in folders:
        if path.exists():
            original(path)


@pytest.mark.parametrize('cleanup_refused', [False, True])
def test_secret_write_cleans_its_private_staging_and_keeps_the_saved_value(
        tmp_path, monkeypatch, cleanup_refused):
    from spacr import run_journal

    destination = tmp_path / 'secrets.json'
    destination.write_text('{"unchanged":"original"}')
    staging = tmp_path / 'private-staging'
    staging.mkdir()
    monkeypatch.setattr(run_journal, '_private_secret_staging', lambda path: staging)
    monkeypatch.setattr(run_journal, '_restrict_to_owner', lambda path: None)
    original = Path.rmdir
    if cleanup_refused:
        def busy(path):
            if path == staging:
                raise OSError('staging busy')
            return original(path)
        monkeypatch.setattr(Path, 'rmdir', busy)
    run_journal._write_notify_secret_file({'test-token': 'synthetic-value'}, destination)
    assert destination.read_text() == '{"test-token": "synthetic-value"}'
    if staging.exists():
        assert list(staging.iterdir()) == []
        original(staging)


def test_windows_path_and_string_values_compare_without_reserializing_settings(windows_journal, tmp_path):
    path = tmp_path / 'field.npy'
    assert windows_journal.values_equal(path, str(path))
    assert windows_journal._same_windows_path(str(path), path)
    assert not windows_journal._same_windows_path(3, 3)


def test_unrepresentable_windows_path_does_not_break_settings_comparison(windows_journal, monkeypatch):
    class RefusingPath(type(Path())):
        def __new__(cls, value):
            if value == 'unrepresentable':
                raise ValueError('unrepresentable path')
            return super().__new__(cls, value)

    first = RefusingPath('/field.npy')
    monkeypatch.setattr(windows_journal, 'Path', RefusingPath)
    assert not windows_journal._same_windows_path(first, 'unrepresentable')


def test_windows_run_owner_records_identity_without_a_pid_namespace(windows_journal, monkeypatch):
    from spacr import install_cleanup

    def no_namespace():
        raise AssertionError('Windows ownership must not query a PID namespace')

    monkeypatch.setattr(install_cleanup, '_pid_namespace', no_namespace)
    monkeypatch.setattr(install_cleanup, '_lease_host', lambda: 'test-host')
    monkeypatch.setattr(install_cleanup, '_process_started', lambda pid: 'test:1')
    monkeypatch.setattr(windows_journal, '_OWN_RUN_OWNER', {})
    expected = {'pid': str(os.getpid()), 'started': 'test:1', 'host': 'test-host'}
    owner = windows_journal._run_owner()
    assert owner == expected
    owner['started'] = 'modified-copy'
    assert windows_journal._run_owner() == expected


@pytest.mark.parametrize('current_stamp, gone', [('test:1', False), ('test:2', True)])
def test_windows_existing_owner_is_judged_by_its_start_stamp(
        windows_journal, monkeypatch, current_stamp, gone):
    from spacr import install_cleanup

    def no_namespace():
        raise AssertionError('Windows ownership must not query a PID namespace')

    monkeypatch.setattr(install_cleanup, '_pid_namespace', no_namespace)
    monkeypatch.setattr(install_cleanup, '_lease_host', lambda: 'test-host')
    monkeypatch.setattr(install_cleanup, '_process_gone', lambda pid: False)
    monkeypatch.setattr(install_cleanup, '_process_started', lambda pid: current_stamp)
    owner = {'pid': str(os.getpid() + 100000), 'started': 'test:1', 'host': 'test-host'}
    assert windows_journal._run_owner_gone(owner) is gone
