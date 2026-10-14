"""Explicit SSH discovery and queued MFA controls use no real account."""
from __future__ import annotations

import threading

import pytest
from PySide6.QtWidgets import QDialog, QInputDialog, QLabel, QLineEdit

from spacr.qt.screens import distributed_jobs as ui
from spacr.remote_execution import CommandResult, ExecutionProfile


def test_discovery_never_runs_on_open_and_imported_fields_roundtrip(qtbot, monkeypatch, tmp_path):
    calls = []
    row = {'alias': 'greatlakes', 'hostname': 'login.example', 'user': 'ada', 'port': '2222',
           'identityfile': str(tmp_path / 'key'), 'proxyjump': 'bastion', 'config': str(tmp_path / 'config')}
    monkeypatch.setattr(ui, '_discover_ssh_settings', lambda folder=None: calls.append(folder) or [row])
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    assert calls == []
    dialog._find_ssh()
    assert calls == [None]
    dialog._name.setText('cluster')
    dialog._workdir.setText('/shared')
    profile = dialog.profile()
    assert profile.host == 'greatlakes'
    assert profile.ssh_options == {key: value for key, value in row.items() if key != 'alias'}
    assert profile.ssh_interactive
    restored = ui.ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(restored)
    assert restored.profile() == profile
    assert calls == [None]


def test_discovery_error_and_empty_results_are_visible(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    monkeypatch.setattr(ui, '_discover_ssh_settings', lambda folder: [])
    dialog._find_ssh()
    assert 'No SSH aliases' in dialog._error.text()
    def denied(folder):
        raise PermissionError('private path must not leak')
    monkeypatch.setattr(ui, '_discover_ssh_settings', denied)
    dialog._find_ssh()
    assert 'could not be read' in dialog._error.text()
    assert 'private path' not in dialog._error.text()


@pytest.mark.parametrize('failed_stage', [None, 'hostname', 'sinfo'])
def test_connection_reports_exact_stage_off_gui_thread(qtbot, monkeypatch, failed_stage):
    calls = []
    main_thread = threading.get_ident()
    def run(argv, **kwargs):
        stage = argv[2].split()[0]
        calls.append((stage, threading.get_ident()))
        return CommandResult(1 if stage == failed_stage else 0)
    monkeypatch.setattr(ui, '_run_command', run)
    profile = ExecutionProfile('p', 'slurm', host='alias', workdir='/shared', ssh_interactive=True)
    dialog = ui.ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(dialog)
    dialog._test_connection()
    qtbot.waitUntil(lambda: dialog._connection_worker is None, timeout=20000)
    assert all(thread != main_thread for _, thread in calls)
    assert [stage for stage, _ in calls] == (['hostname'] if failed_stage == 'hostname' else ['hostname', 'sinfo'])
    assert (f'{failed_stage} failed:' if failed_stage else 'passed') in dialog._error.text()


def test_server_challenges_are_verbatim_plaintext_and_secret_values_are_ephemeral(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()
    labels = []
    def accept(prompt):
        labels.append((prompt.labelText(), prompt.textEchoMode(), [label.textFormat() for label in prompt.findChildren(QLabel)]))
        prompt.setTextValue('TEMP-SECRET')
        return QDialog.Accepted
    monkeypatch.setattr(QInputDialog, 'exec', accept)
    result = []
    thread = threading.Thread(target=lambda: result.append(broker._request('Vendor title', 'Approve <b>push</b>', [('Code verbatim:', False)])))
    thread.start()
    qtbot.waitUntil(lambda: bool(result))
    thread.join(timeout=1)
    assert result == [['TEMP-SECRET']]
    assert labels[0][0] == 'Approve <b>push</b>\n\nCode verbatim:'
    assert labels[0][1] == QLineEdit.Password
    assert all(mode == ui.Qt.PlainText for mode in labels[0][2])
    assert 'TEMP-SECRET' not in str(dialog._ssh_options)
    assert not broker._pending


def test_cancelled_prompt_is_not_an_empty_password(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()
    monkeypatch.setattr(QInputDialog, 'exec', lambda self: QDialog.Rejected)
    result = []
    thread = threading.Thread(target=lambda: result.append(broker._request('SSH', '', [('Password:', False)])))
    thread.start()
    qtbot.waitUntil(lambda: bool(result))
    thread.join(timeout=1)
    assert result == [None]


def test_invalid_imported_port_is_reported_without_accepting_bad_settings(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    monkeypatch.setattr(ui, '_discover_ssh_settings', lambda folder=None: [{'alias': 'bad', 'hostname': 'bad', 'port': 'invalid'}])
    dialog._find_ssh()
    assert dialog._host.text() == ''
    assert 'between 1 and 65535' in dialog._error.text()


def test_custom_backend_disables_ssh_controls(qtbot):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._backend.setCurrentIndex(dialog._backend.findData('command'))
    for field in (dialog._ssh_user, dialog._ssh_port, dialog._ssh_hostname, dialog._ssh_key, dialog._ssh_jump,
                  dialog._find_ssh_button, dialog._choose_ssh_button, dialog._test_ssh):
        assert not field.isEnabled()


def test_folder_selection_and_cancel_are_explicit(qtbot, monkeypatch, tmp_path):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    calls = []
    monkeypatch.setattr(dialog, '_find_ssh', calls.append)
    monkeypatch.setattr(ui.QFileDialog, 'getExistingDirectory', lambda *args: '')
    dialog._choose_ssh_folder()
    assert calls == []
    monkeypatch.setattr(ui.QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    dialog._choose_ssh_folder()
    assert calls == [str(tmp_path)]


def test_no_authentication_wait_is_allowed_on_gui_thread(qtbot):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    with pytest.raises(ui.RemoteExecutionError, match='background'):
        ui._ensure_ssh_broker()._request('SSH', '', [])


def test_instruction_only_push_is_nonmodal_and_preserves_verbatim_text(qtbot):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()
    result = []
    thread = threading.Thread(target=lambda: result.append(broker._request('Okta', 'Approve <b>push</b> on your device', [])))
    thread.start()
    qtbot.waitUntil(lambda: bool(result))
    thread.join(timeout=1)
    assert result == [[]]
    assert broker._notices[-1].text() == 'Approve <b>push</b> on your device'
    assert broker._notices[-1].textFormat() == ui.Qt.PlainText
    for notice in broker._notices:
        notice.close()
    broker._notices.clear()


def test_normal_runtime_extractor_tracks_backend_errors_and_indirect_ssh_labels():
    from tools import build_i18n_catalogs as build
    sources = set(build.extract_static_ui_sources())
    assert {'SSH host name', 'SSH user', 'SSH port', 'SSH identity file', 'SSH jump host',
            'SSH authentication was cancelled or timed out.',
            'SSH command timed out; it was not submitted again.'} <= sources
