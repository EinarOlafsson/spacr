"""Explicit SSH discovery and queued MFA controls use no real account."""
from __future__ import annotations

import threading
import types

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


def test_keychain_box_is_off_by_default_and_follows_interactive_login(qtbot, monkeypatch):
    monkeypatch.setattr(ui, '_keychain_available', lambda: True)
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._backend.setCurrentIndex(dialog._backend.findData('slurm'))
    assert not dialog._ssh_keychain.isChecked()
    assert not dialog._ssh_keychain.isEnabled()
    dialog._auth_mode.setCurrentIndex(dialog._auth_mode.findData('interactive'))
    assert dialog._ssh_keychain.isEnabled()
    dialog._name.setText('gl')
    dialog._host.setText('greatlakes')
    dialog._workdir.setText('/scratch')
    assert dialog.profile().ssh_keychain is False
    dialog._ssh_keychain.setChecked(True)
    assert dialog.profile().ssh_keychain is True
    restored = ui.ExecutionProfileDialog(profile=dialog.profile())
    qtbot.addWidget(restored)
    assert restored._ssh_keychain.isChecked()


def test_keychain_box_is_unavailable_without_keyring(qtbot, monkeypatch):
    monkeypatch.setattr(ui, '_keychain_available', lambda: False)
    dialog = ui.ExecutionProfileDialog(profile=ExecutionProfile('p', 'slurm', host='alias', workdir='/w', ssh_interactive=True))
    qtbot.addWidget(dialog)
    assert not dialog._ssh_keychain.isEnabled()
    assert 'keyring' in dialog._ssh_keychain.toolTip()


def test_unticking_keychain_forgets_the_remembered_password(qtbot, monkeypatch):
    deleted = []
    monkeypatch.setattr(ui, '_keychain_available', lambda: True)
    monkeypatch.setattr(ui, '_keychain_account', lambda host, options: f'ada@{host}:22')
    monkeypatch.setattr(ui, '_keychain_password', lambda action, account: deleted.append((action, account)))
    profile = ExecutionProfile('gl', 'slurm', host='greatlakes', workdir='/w', ssh_interactive=True, ssh_keychain=True)
    dialog = ui.ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(dialog)
    dialog._ssh_keychain.setChecked(False)
    dialog._validate_and_accept()
    qtbot.waitUntil(lambda: bool(deleted))
    assert deleted == [('delete', 'ada@greatlakes:22')]


def test_denied_login_shows_the_failing_step_and_retry_signs_in_again(qtbot, monkeypatch):
    from spacr.remote_execution import _auth_denied_message
    outcomes = ['denied', 'ok', 'ok']
    calls = []
    def run(argv, **kwargs):
        calls.append(argv[2])
        if outcomes.pop(0) == 'denied':
            raise ui.RemoteExecutionError(_auth_denied_message())
        return CommandResult(0, 'gl-login1\n')
    monkeypatch.setattr(ui, '_run_command', run)
    dialog = ui.ExecutionProfileDialog(profile=ExecutionProfile('p', 'slurm', host='greatlakes', workdir='/w', ssh_interactive=True))
    qtbot.addWidget(dialog)
    assert dialog._retry_ssh.isHidden()
    dialog._test_connection()
    qtbot.waitUntil(lambda: dialog._connection_worker is None, timeout=20000)
    assert dialog._error.text().startswith('hostname failed:')
    assert 'was denied' in dialog._error.text()
    assert not dialog._retry_ssh.isHidden()
    dialog._retry_ssh.click()
    qtbot.waitUntil(lambda: dialog._connection_worker is None and 'passed' in dialog._error.text(), timeout=20000)
    assert dialog._retry_ssh.isHidden()
    assert calls == ['hostname', 'hostname', 'sinfo --noheader']


def test_help_text_names_great_lakes_okta_and_read_only_discovery(qtbot):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    help_text = dialog._auth_mode.toolTip() + dialog._auth_label.toolTip()
    assert 'Great Lakes' in help_text and 'Okta' in help_text
    assert 'never runs ssh -G' in dialog._find_ssh_button.toolTip()


class _Signal:
    """Record the slots a connection test wires to its worker."""

    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


def _capture_connection_worker(monkeypatch):
    """Replace ``make_thread`` so the connection check runs on this thread."""
    started = []

    def fake_make_thread(operation, settings, **kwargs):
        thread = type('Thread', (), {'finished': _Signal(), 'start': lambda self: None})()
        worker = type('Worker', (), {'finished': _Signal()})()
        started.append((operation, thread, worker, kwargs))
        return thread, worker

    monkeypatch.setattr(ui, 'make_thread', fake_make_thread)
    return started


@pytest.mark.parametrize('backend, failing, stages, error', [
    ('ssh', None, ['hostname'], ''),
    ('slurm', None, ['hostname', 'sinfo'], ''),
    ('slurm', 'hostname', ['hostname'], 'hostname failed: Remote command returned a non-zero exit status.'),
    ('slurm', 'sinfo', ['hostname', 'sinfo'], 'sinfo failed: Remote command returned a non-zero exit status.'),
])
def test_the_connection_check_stops_at_the_first_failed_stage(qtbot, monkeypatch, backend, failing, stages, error):
    started = _capture_connection_worker(monkeypatch)
    calls = []

    def run(argv, **kwargs):
        stage = argv[2].split()[0]
        calls.append(stage)
        return CommandResult(1 if stage == failing else 0)

    monkeypatch.setattr(ui, '_run_command', run)
    profile = ExecutionProfile('p', backend, host='alias', workdir='/shared', ssh_interactive=True)
    dialog = ui.ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(dialog)
    dialog._test_connection()
    dialog._test_connection()
    assert len(started) == 1, 'a second click while one check runs starts nothing'
    assert not dialog._test_ssh.isEnabled()
    operation, thread, worker, kwargs = started[0]
    assert kwargs['app_key'] == 'ssh_connection_test' and kwargs['journal'] is False
    operation({})
    assert calls == stages
    assert dialog._connection_error == error
    for slot in worker.finished.slots:
        slot(True)
    for slot in thread.finished.slots:
        slot()
    assert dialog._connection_worker is None
    assert dialog._test_ssh.isEnabled()
    assert dialog._error.text() == (error or 'SSH connection test passed.')
    assert dialog._retry_ssh.isHidden() is not bool(error)


def test_an_invalid_host_is_reported_before_any_connection_is_tried(qtbot, monkeypatch):
    started = _capture_connection_worker(monkeypatch)
    dialog = ui.ExecutionProfileDialog(profile=ExecutionProfile('p', 'ssh', host='alias', workdir='/w'))
    qtbot.addWidget(dialog)
    dialog._host.setText('bad host;reboot')
    dialog._test_connection()
    assert started == []
    assert 'unsupported characters' in dialog._error.text()


def test_an_imported_alias_needs_a_port_inside_the_valid_range(qtbot):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._select_ssh(-1)
    assert dialog._host.text() == '' and dialog._error.text() == ''
    dialog._ssh_aliases.blockSignals(True)
    dialog._ssh_aliases.addItem('far', {'alias': 'far', 'hostname': 'far.example', 'port': '70000'})
    dialog._ssh_aliases.blockSignals(False)
    dialog._select_ssh(dialog._ssh_aliases.count() - 1)
    assert dialog._host.text() == ''
    assert 'between 1 and 65535' in dialog._error.text()


def test_a_password_that_cannot_be_named_is_not_forgotten_blindly(qtbot, monkeypatch):
    deleted = []
    monkeypatch.setattr(ui, '_keychain_password', lambda action, account: deleted.append(account))

    def unreadable(host, options):
        raise ui.RemoteExecutionError('SSH configuration cannot be read.')

    monkeypatch.setattr(ui, '_keychain_account', unreadable)
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._forget_password('greatlakes', {})
    assert deleted == []


def test_an_unanswered_challenge_times_out_and_is_never_shown(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()

    waits = []

    class Expired(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return False

    monkeypatch.setattr(ui, 'threading', types.SimpleNamespace(Event=Expired, Thread=threading.Thread))
    shown = []
    monkeypatch.setattr(QInputDialog, 'exec', lambda self: shown.append(self) or QDialog.Accepted)
    result = []
    thread = threading.Thread(target=lambda: result.append(broker._request('SSH', '', [('Code:', True)])))
    thread.start()
    thread.join(timeout=5)
    assert result == [None] and waits == [185]
    assert broker._pending == []
    qtbot.wait(50)
    assert shown == []


def test_a_new_instruction_replaces_the_one_still_on_screen(qtbot):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()
    first = {'title': 'Okta', 'instructions': 'Approve the first push', 'prompts': [],
             'event': threading.Event(), 'answers': None, 'cancelled': False}
    second = dict(first, instructions='Approve the second push', event=threading.Event())
    broker._show(first)
    old = broker._notices[-1]
    broker._show(second)
    try:
        assert old.isHidden()
        assert [notice.text() for notice in broker._notices] == ['Approve the second push']
        assert first['answers'] == [] and second['event'].is_set()
    finally:
        for notice in broker._notices:
            notice.close()
        broker._notices.clear()


def test_quitting_cancels_waiting_challenges_and_closes_sessions_off_the_gui_thread(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()
    closed = threading.Event()
    closers = []

    def close():
        closers.append(threading.current_thread() is not threading.main_thread())
        closed.set()

    monkeypatch.setattr(ui._SSH_SESSIONS, 'close', close)
    waiting = {'title': 'SSH', 'instructions': '', 'prompts': [('Password:', False)],
               'event': threading.Event(), 'answers': None, 'cancelled': False}
    broker._pending.append(waiting)
    try:
        broker._shutdown()
        assert waiting['cancelled'] and waiting['event'].is_set()
        assert closed.wait(5) and closers == [True]
        broker._pending.remove(waiting)
        closed.clear()
        broker._shutdown()
        assert closed.wait(5)
    finally:
        if waiting in broker._pending:
            broker._pending.remove(waiting)
    shown = []
    monkeypatch.setattr(QInputDialog, 'exec', lambda self: shown.append(self) or QDialog.Accepted)
    broker._show(waiting)
    assert shown == [] and waiting['answers'] is None


def test_a_challenge_that_expires_while_its_notice_opens_gets_no_answers(qtbot, monkeypatch):
    dialog = ui.ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    broker = ui._ensure_ssh_broker()
    request = {'title': 'Okta', 'instructions': 'Approve the push', 'prompts': [],
               'event': threading.Event(), 'answers': None, 'cancelled': False}
    monkeypatch.setattr(ui.QMessageBox, 'show', lambda self: request.__setitem__('cancelled', True))
    try:
        broker._show(request)
        assert request['answers'] is None
        assert request['event'].is_set()
    finally:
        for notice in broker._notices:
            notice.close()
        broker._notices.clear()
