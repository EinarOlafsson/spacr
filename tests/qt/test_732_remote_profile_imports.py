"""Portable job and connection profiles fill forms only after explicit Apply."""
import json
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QPlainTextEdit, QPushButton

from spacr.qt.screens import distributed_jobs as dj
from spacr.remote_execution import ExecutionProfile, JobStore, ProfileStore, RemoteExecutionError, RemoteJob, RemoteJobManager
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _no_remote(*args, **kwargs):
    pytest.fail('Profile transfer performed a remote operation')


@pytest.fixture
def screen(qtbot, private_ai_profile, tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / 'profiles.json')
    store.save(ExecutionProfile('cluster', 'slurm', host='gl', workdir='/project/user',
                               local_root='/local/data', remote_root='/remote/data',
                               ssh_options={'hostname': 'cluster.example', 'user': 'scientist',
                                            'port': '2222', 'identityfile': '~/.ssh/id_ed25519'},
                               auth_mode='custom', ssh_command='ssh -F ~/.ssh/custom_config {host} {command}',
                               ssh_keychain=True, resources={'cpus': '20', 'gpus': '2', 'mem': '90G'},
                               env_setup='module load mamba\nmamba activate spacr',
                               job_script='#!/bin/bash\n{env_setup}\n{run_command}\n',
                               login_hosts='login*,greatlakes*'))
    manager = RemoteJobManager(store, JobStore(tmp_path / 'jobs.json'), _no_remote)
    monkeypatch.setattr(manager, 'submit', _no_remote)
    monkeypatch.setattr(dj, '_keychain_password', _no_remote)
    monkeypatch.setattr(dj, '_signin_command', _no_remote)
    widget = dj.DistributedJobsScreen(manager=manager, threaded=False, auto_poll=False)
    qtbot.addWidget(widget)
    widget._module.setCurrentIndex(widget._module.findData('mask'))
    widget._settings_snapshot = {'src': '/local/data/plate', 'cell_channel': 0}
    widget._snapshot_module = 'mask'
    widget._settings_path.setText('[current mask settings snapshot]')
    widget._save_snapshot.setEnabled(True)
    widget._allocated_gpus.setChecked(True)
    return widget


def _preview(monkeypatch, accept, seen):
    def answer(dialog):
        view = dialog.findChild(QPlainTextEdit, 'RemoteProfileImportPreview')
        assert view is not None and view.isReadOnly()
        seen.append(json.loads(view.toPlainText()))
        buttons = dialog.findChild(QDialogButtonBox)
        assert buttons.button(QDialogButtonBox.Cancel).isDefault()
        buttons.button(QDialogButtonBox.Apply if accept else QDialogButtonBox.Cancel).click()
        return dialog.result()
    monkeypatch.setattr(QDialog, 'exec', answer)


def _file_dialogs(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))


def test_full_job_buttons_roundtrip_every_edit_without_overwriting_old_profile(screen, tmp_path, monkeypatch):
    original = screen.manager.profiles.get('cluster').to_dict()
    screen._res_cpus.setValue(32)
    screen._res_gpus.setValue(3)
    screen._res_account.setEditText('allocation-account')
    screen._res_time.setText('48:00:00')
    screen._env_edit.setPlainText('module load python\nsource /env/bin/activate')
    screen._script_edit.setPlainText('#!/bin/bash\n{env_setup}\necho ready\n{run_command}\n')
    expected = screen._profile_from_editors().to_dict()
    path = tmp_path / 'job.json'
    _file_dialogs(monkeypatch, path)
    screen.findChild(QPushButton, 'SaveJobProfile').click()
    saved = json.loads(path.read_text())
    assert saved['profile'] == expected
    assert saved['job']['settings_snapshot']['cell_channel'] == 0
    assert screen.manager.profiles.get('cluster').to_dict() == original
    screen._module.setCurrentIndex(screen._module.findData('measure'))
    screen._res_cpus.setValue(1)
    screen._env_edit.clear()
    screen._allocated_gpus.setChecked(False)
    screen._settings_snapshot = None
    screen._settings_path.setText('/different/settings.csv')
    seen = []
    _preview(monkeypatch, True, seen)
    screen.findChild(QPushButton, 'ImportJobProfile').click()
    imported = screen._profile_from_editors().to_dict()
    assert imported.pop('name') != 'cluster'
    expected.pop('name')
    assert imported == expected
    assert screen._module.currentData() == 'mask'
    assert screen._settings_snapshot == saved['job']['settings_snapshot']
    assert screen._settings_path.text() == saved['job']['settings_path']
    assert screen._allocated_gpus.isChecked()
    assert screen.manager.profiles.get('cluster').to_dict() == original
    assert screen.manager.jobs.list() == []
    assert len(seen) == 1


def test_connection_buttons_preserve_job_scope_and_custom_authentication(screen, tmp_path, monkeypatch):
    source = screen._current_profile()
    path = tmp_path / 'connection.json'
    _file_dialogs(monkeypatch, path)
    screen.findChild(QPushButton, 'SaveConnectionProfile').click()
    saved = json.loads(path.read_text())
    assert set(saved['profile']) == {'name', *dj._CONNECTION_PROFILE_FIELDS}
    assert 'job' not in saved and 'resources' not in saved['profile']
    other = replace(source, name='other', backend='pbs', host='other-host',
                    resources={'cpus': '8', 'mem': '12G'}, env_setup='source /other/env',
                    job_script='#!/bin/bash\n{env_setup}\n{run_command}\n')
    screen.manager.profiles.save(other)
    screen._reload_profiles('other')
    screen._res_cpus.setValue(42)
    screen._script_edit.setPlainText('#!/bin/bash\necho keep\n{run_command}\n')
    before = screen._profile_transfer_document('job')['job']
    job_profile = screen._profile_from_editors().to_dict()
    seen = []
    _preview(monkeypatch, True, seen)
    screen.findChild(QPushButton, 'ImportConnectionProfile').click()
    current = screen._current_profile().to_dict()
    for key in dj._CONNECTION_PROFILE_FIELDS:
        assert current[key] == source.to_dict()[key]
    for key in ('backend', 'resources', 'env_setup', 'job_script', 'runner', 'scheduler_options', 'direct_run'):
        assert current[key] == job_profile[key]
    assert screen._profile_transfer_document('job')['job'] == before
    assert screen.manager.profiles.get('other').to_dict() == other.to_dict()
    assert screen.manager.jobs.list() == []


@pytest.mark.parametrize('reason', ['chooser', 'preview', 'malformed', 'wrong_scope', 'unknown_field', 'resource_overflow'])
def test_cancel_and_invalid_import_leave_form_and_stores_untouched(screen, tmp_path, monkeypatch, reason):
    before = screen._profile_transfer_document('job')
    store_before = screen.manager.profiles.path.read_bytes()
    document = json.loads(json.dumps(before))
    if reason == 'wrong_scope':
        document['scope'] = 'connection'
    if reason == 'unknown_field':
        document['profile']['unexpected'] = 'unsupported'
    if reason == 'resource_overflow':
        document['profile']['resources']['cpus'] = '999999999'
    path = tmp_path / 'import.json'
    path.write_text('not JSON' if reason == 'malformed' else json.dumps(document))
    _file_dialogs(monkeypatch, path)
    if reason == 'chooser':
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: ('', ''))
    seen = []
    _preview(monkeypatch, False, seen)
    screen.findChild(QPushButton, 'ImportJobProfile').click()
    assert screen._profile_transfer_document('job') == before
    assert screen.manager.profiles.path.read_bytes() == store_before
    assert screen.manager.jobs.list() == []
    assert len(seen) == (1 if reason == 'preview' else 0)


@pytest.mark.parametrize('change', [
    lambda value: value['profile'].update(password='do-not-export'),
    lambda value: value['profile'].update(env_setup='export API_KEY=do-not-export'),
    lambda value: value['profile'].update(env_setup='export TOKEN=do-not-export'),
    lambda value: value['profile'].update(job_script='export SECRET=do-not-export'),
    lambda value: value['profile'].update(env_setup='export MFA=do-not-export'),
    lambda value: value['profile'].update(env_setup='export ONE_TIME_PASSWORD=do-not-export'),
    lambda value: value['profile'].update(env_setup='export TOKEN="$TOKEN do-not-export"'),
    lambda value: value['profile'].update(env_setup='some-client --password do-not-export'),
    lambda value: value['profile'].update(env_setup='some-client --access-token do-not-export'),
    lambda value: value['profile'].update(env_setup='some-client --private-key do-not-export'),
    lambda value: value['job']['settings_snapshot'].update(token='$TOKEN do-not-export'),
    lambda value: value['profile'].update(job_script='-----BEGIN OPENSSH PRIVATE KEY-----\nprivate material'),
    lambda value: value['job']['settings_snapshot'].update(token='do-not-export'),
])
def test_credentials_refused_before_saving_or_preview(screen, tmp_path, monkeypatch, change):
    document = screen._profile_transfer_document('job')
    change(document)
    with pytest.raises(RemoteExecutionError, match='may not contain passwords'):
        dj._validate_profile_transfer(document, 'job')
    path = tmp_path / 'unsafe.json'
    path.write_text(json.dumps(document))
    _file_dialogs(monkeypatch, path)
    seen = []
    _preview(monkeypatch, True, seen)
    old = screen._current_profile().to_dict()
    screen._import_profile_transfer('job')
    assert seen == []
    assert screen._current_profile().to_dict() == old
    assert 'do-not-export' not in screen._status.text()


def test_reference_settings_and_nonsecret_environment_references_roundtrip(screen, tmp_path, monkeypatch):
    settings = tmp_path / 'settings.csv'
    settings.write_text('Key,Value\ncell_channel,0\n')
    screen._settings_snapshot = None
    screen._settings_path.setText(str(settings))
    screen._env_edit.setPlainText('export API_KEY="$API_KEY"')
    document = screen._profile_transfer_document('job')
    assert document['job']['settings_snapshot'] is None
    assert document['job']['settings_path'] == str(settings)
    assert document['profile']['ssh_options']['identityfile'] == '~/.ssh/id_ed25519'
    assert document['profile']['ssh_keychain'] is True
    assert 'private material' not in json.dumps(document)


def test_allocation_reference_requires_real_compatible_running_allocation(screen):
    document = screen._profile_transfer_document('job')
    document['job']['allocation_id'] = 'missing'
    with pytest.raises(RemoteExecutionError, match='allocation is unavailable'):
        screen._prepare_profile_transfer(document, 'job')
    holder = RemoteJob('allocation-id', 'allocation', 'cluster', 'slurm', status='running')
    screen.manager.jobs.save(holder)
    document['job']['allocation_id'] = holder.job_id
    prepared, candidate, allocation = screen._prepare_profile_transfer(document, 'job')
    assert allocation == holder.job_id and candidate.name == 'cluster'
    document['profile']['host'] = 'different-cluster'
    with pytest.raises(RemoteExecutionError, match='allocation is unavailable'):
        screen._prepare_profile_transfer(document, 'job')


def test_import_repopulates_live_allocation_missing_from_screen_cache(screen, tmp_path, monkeypatch):
    holder = RemoteJob('live-allocation', 'allocation', 'cluster', 'slurm', status='running',
                       profile=screen._current_profile().to_dict())
    screen.manager.jobs.save(holder)
    assert screen._jobs == []
    document = screen._profile_transfer_document('job')
    document['job']['allocation_id'] = holder.job_id
    path = tmp_path / 'allocated.json'
    path.write_text(json.dumps(document))
    _file_dialogs(monkeypatch, path)
    seen = []
    _preview(monkeypatch, True, seen)
    screen._import_profile_transfer('job')
    assert screen._run_inside.currentData() == holder.job_id
    assert screen.manager.jobs.list() == [holder]


def test_profile_edited_after_allocation_cannot_redirect_import_to_new_host(screen):
    profile = screen._current_profile()
    holder = RemoteJob('old-allocation', 'allocation', profile.name, 'slurm', status='running',
                       profile=profile.to_dict())
    screen.manager.jobs.save(holder)
    screen.manager.profiles.save(replace(profile, host='new-host'))
    document = screen._profile_transfer_document('job')
    document['job']['allocation_id'] = holder.job_id
    with pytest.raises(RemoteExecutionError, match='allocation is unavailable'):
        screen._prepare_profile_transfer(document, 'job')


def test_nested_profile_secrets_are_refused_without_echoing_or_writing(screen, tmp_path, monkeypatch):
    document = screen._profile_transfer_document('job')
    document['job']['settings_snapshot']['steps'] = [{'token': '${TOKEN}'}, {'token': False}]
    assert dj._validate_profile_transfer(document, 'job') == document
    document['job']['settings_snapshot']['steps'][0]['token'] = 'private-token-must-not-appear'
    path = tmp_path / 'nested-secret.json'
    path.write_text(json.dumps(document))
    before = screen.manager.profiles.path.read_bytes()
    _file_dialogs(monkeypatch, path)
    seen = []
    _preview(monkeypatch, True, seen)
    screen._import_profile_transfer('job')
    assert seen == [] and screen.manager.profiles.path.read_bytes() == before
    assert 'private-token-must-not-appear' not in screen._status.text()
    assert 'may not contain passwords' in screen._status.text()
    assert screen.manager.jobs.list() == []


def test_profile_schema_types_and_metadata_budget_refuse_without_mutation(screen):
    original = screen._profile_transfer_document('job')
    before = screen.manager.profiles.path.read_bytes()
    changes = [('version', True), ('resources', {'cpus': 7}), ('allocated_gpus', 1),
               ('settings_snapshot', []), ('settings_path', '\0invalid')]
    for key, value in changes:
        document = json.loads(json.dumps(original))
        target = document if key == 'version' else document['profile'] if key == 'resources' else document['job']
        target[key] = value
        with pytest.raises(RemoteExecutionError, match='unsupported or incompatible'):
            dj._validate_profile_transfer(document, 'job')
    document = json.loads(json.dumps(original))
    document['job']['settings_snapshot']['notes'] = 'x' * (2 * 1024 * 1024)
    with pytest.raises(RemoteExecutionError, match='too large'):
        dj._validate_profile_transfer(document, 'job')
    connection = screen._profile_transfer_document('connection')
    del connection['profile']['host']
    with pytest.raises(RemoteExecutionError, match='unsupported or incompatible'):
        dj._validate_profile_transfer(connection, 'connection')
    assert screen._profile_transfer_document('job') == original
    assert screen.manager.profiles.path.read_bytes() == before and screen.manager.jobs.list() == []


def test_busy_transfers_and_failed_atomic_export_preserve_existing_store(screen, tmp_path, monkeypatch):
    before = screen.manager.profiles.path.read_bytes()
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', _no_remote)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', _no_remote)
    screen._busy = True
    try:
        screen._save_profile_transfer('job')
        screen._import_profile_transfer('job')
    finally:
        screen._busy = False
    target = tmp_path / 'existing-directory'
    target.mkdir()
    sentinel = target / 'keep.txt'
    sentinel.write_text('preserve this directory')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target), ''))
    screen._save_profile_transfer('job')
    assert sentinel.read_text() == 'preserve this directory'
    assert 'operation failed' in screen._status.text()
    assert screen.manager.profiles.path.read_bytes() == before and screen.manager.jobs.list() == []
