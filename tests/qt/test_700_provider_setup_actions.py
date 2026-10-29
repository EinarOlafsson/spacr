"""Provider setup links and explicit sign-in actions never contact real accounts."""
import subprocess

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QDialog, QLabel, QPushButton

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


class _Provider:
    def __init__(self, name, installed=True, signed=False):
        self.name = self.cli_name = name
        self.label = name.title()
        self.install_hint = 'curl documented-installer'
        self.login_command = 'arbitrary text must never launch'
        self.installed, self.signed = installed, signed
        self.checked = []

    def is_installed(self):
        return self.installed

    def check_signed_in(self):
        self.checked.append(QThread.currentThread())
        return self.signed

    def is_configured(self):
        return self.installed


@pytest.fixture
def setup_dialog(qtbot, private_ai_profile, monkeypatch):
    from spacr.qt import ai
    from spacr.qt.ai import cli_install
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    providers = [_Provider(name) for name in ('claude', 'codex', 'gemini')]
    monkeypatch.setattr(ai, 'list_providers', lambda: providers)
    monkeypatch.setattr(cli_install, 'locate', lambda tool: '/installed/' + tool.cli_name if tool.installed else '')
    monkeypatch.setattr(cli_install, 'put_on_path', lambda path: True)
    dialog = _ProvidersDialog()
    qtbot.addWidget(dialog)
    yield dialog, providers
    dialog.reject()


def test_official_links_precede_commands_and_opening_never_signs_in(setup_dialog):
    dialog, providers = setup_dialog
    links = {
        'claude': 'https://code.claude.com/docs/en/setup',
        'codex': 'https://developers.openai.com/codex/cli',
        'gemini': 'https://geminicli.com/docs/get-started/installation/',
    }
    for provider in providers:
        link = dialog.findChild(QLabel, 'ProviderInstallLink_' + provider.name)
        assert links[provider.name] in link.text() and link.openExternalLinks()
        card = dialog._provider_rows[provider.name][1]
        assert card.layout().indexOf(link) == 1
        assert dialog.findChild(QPushButton, 'ProviderSignIn_' + provider.name).isEnabled()
        assert not provider.checked
    prose = ' '.join(label.text() for label in dialog.findChildren(QLabel))
    assert 'Node.js' in prose and 'On Windows, install Node.js' in prose


@pytest.mark.parametrize('name, arguments', [('claude', ['auth', 'login']), ('codex', ['login']), ('gemini', [])])
@pytest.mark.parametrize('signed', [False, None, True])
def test_explicit_native_signin_rechecks_real_status_not_dialog_success(
        setup_dialog, qtbot, monkeypatch, name, arguments, signed):
    from spacr.qt.ai import pty_sign_in
    dialog, providers = setup_dialog
    provider = next(item for item in providers if item.name == name)
    provider.signed = signed
    launched = []
    class SignIn(QDialog):
        succeeded = True
        def __init__(self, label, argv, parent):
            super().__init__(parent)
            launched.append((label, argv))
        def exec(self):
            return QDialog.Accepted
    monkeypatch.setattr(pty_sign_in, 'SignInDialog', SignIn)
    monkeypatch.setattr(pty_sign_in, 'pty_available', lambda: True)
    dialog.findChild(QPushButton, 'ProviderSignIn_' + name).click()
    assert launched == [(provider.label, ['/installed/' + name, *arguments])]
    label = dialog.findChild(QLabel, 'ProviderAuthStatus_' + name)
    qtbot.waitUntil(lambda: bool(provider.checked) and 'Checking' not in label.text())
    expected = 'Signed in.' if signed is True else 'Not signed in.' if signed is False else 'unavailable'
    assert expected in label.text()
    assert all(thread != QThread.currentThread() for thread in provider.checked)


class _Process:
    def __init__(self):
        self.code = None
        self.stopped = []
    def poll(self):
        return self.code
    def terminate(self):
        self.stopped.append('terminate')
    def kill(self):
        self.stopped.append('kill')
        self.code = -9
    def wait(self, timeout):
        self.stopped.append(timeout)
        if self.code is None:
            raise subprocess.TimeoutExpired('offline process', timeout)
        return self.code


@pytest.mark.parametrize('name, argv', [
    ('claude', [r'C:\Tools\claude.exe', 'auth', 'login']),
    ('codex', [r'C:\Tools\codex.exe', 'login']),
    ('gemini', [r'C:\Users\Test User\npm\gemini.cmd']),
])
def test_windows_native_console_uses_fixed_argv_and_rechecks_after_exit(
        setup_dialog, qtbot, monkeypatch, name, argv):
    dialog, providers = setup_dialog
    provider = next(item for item in providers if item.name == name)
    process, calls = _Process(), []
    monkeypatch.setenv('COMSPEC', r'C:\Windows\System32\cmd.exe')
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kwargs: (
        calls.append((command, kwargs)) or process))
    dialog._start_windows_sign_in(provider, argv)
    command, options = calls[0]
    assert options == {'shell': False, 'creationflags': getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x10)}
    if name == 'gemini':
        assert command == r'C:\Windows\System32\cmd.exe /d /s /c ""C:\Users\Test User\npm\gemini.cmd""'
    else:
        assert command == argv
    dialog._poll_provider_sign_in(provider)
    assert not provider.checked
    process.code = 0
    dialog._poll_provider_sign_in(provider)
    qtbot.waitUntil(lambda: bool(provider.checked))
    label = dialog.findChild(QLabel, 'ProviderAuthStatus_' + name)
    qtbot.waitUntil(lambda: 'Checking' not in label.text())
    assert label.text() == 'Not signed in.'


@pytest.mark.parametrize('path', ['C:/bad%PATH%/gemini.cmd', 'C:/bad!value/gemini.cmd', 'C:/bad\nvalue/gemini.cmd'])
def test_windows_batch_fallback_refuses_shell_expansion_paths(setup_dialog, monkeypatch, path):
    dialog, providers = setup_dialog
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Unsafe path launched'))
    with pytest.raises(ValueError):
        dialog._start_windows_sign_in(providers[2], [path])


def test_closing_setup_cancels_owned_console_with_bounded_waits(setup_dialog, monkeypatch):
    dialog, providers = setup_dialog
    process = _Process()
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: process)
    dialog._start_windows_sign_in(providers[1], ['C:/codex.exe', 'login'])
    dialog.reject()
    assert process.stopped == ['terminate', 0.1, 'kill', 0.1]
    assert not dialog._provider_signins


def test_missing_cli_or_unavailable_pty_never_launches_accounts(setup_dialog, monkeypatch):
    from spacr.qt.ai import pty_sign_in
    dialog, providers = setup_dialog
    monkeypatch.setattr(pty_sign_in, 'pty_available', lambda: False)
    monkeypatch.setattr(pty_sign_in, 'SignInDialog', lambda *a, **k: pytest.fail('Unexpected sign-in'))
    dialog._sign_in_provider(providers[0])
    assert 'terminal on this platform' in dialog._provider_rows['claude'][3].text()
    providers[1].installed = False
    assert dialog._provider_login_argv(providers[1]) == []


def test_failed_native_signin_refreshes_actual_status_and_missing_cli_stays_disabled(
        setup_dialog, qtbot, monkeypatch):
    from spacr.qt.ai import pty_sign_in
    dialog, providers = setup_dialog
    provider = providers[0]
    monkeypatch.setattr(pty_sign_in, 'pty_available', lambda: True)
    def refuse(*args, **kwargs):
        raise OSError('native terminal unavailable')
    monkeypatch.setattr(pty_sign_in, 'SignInDialog', refuse)
    dialog._sign_in_provider(provider)
    qtbot.waitUntil(lambda: bool(provider.checked)
                    and dialog._provider_rows[provider.name][3].text() == 'Not signed in.')
    provider.installed = False
    dialog._sign_in_provider(provider)
    qtbot.waitUntil(lambda: dialog._provider_rows[provider.name][2].text() == 'missing')
    assert not dialog._provider_rows[provider.name][4].isEnabled()
    assert dialog._provider_login_argv(_Provider('unknown')) == []


@pytest.mark.parametrize('failure', ['already_gone', 'kill_timeout', 'batch_login'])
def test_owned_console_is_singleton_and_closure_releases_even_failed_native_cleanup(
        setup_dialog, monkeypatch, failure):
    dialog, providers = setup_dialog
    launches = []
    class Uncooperative(_Process):
        def terminate(self):
            self.stopped.append('terminate')
            if failure == 'already_gone':
                raise OSError('process disappeared')
        def kill(self):
            self.stopped.append('kill')
    process = Uncooperative()
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: launches.append(a) or process)
    provider = providers[1]
    argv = ['C:/codex.exe', 'login']
    if failure == 'batch_login':
        monkeypatch.setenv('COMSPEC', r'C:\Windows\System32\cmd.exe')
        argv = ['C:/Program Files/codex.cmd', 'login']
    dialog._start_windows_sign_in(provider, argv)
    timer = dialog._provider_signins[provider.name][1]
    dialog._start_windows_sign_in(provider, argv)
    assert len(launches) == 1 and timer.isActive()
    if failure == 'batch_login':
        assert launches[0][0] == r'C:\Windows\System32\cmd.exe /d /s /c ""C:/Program Files/codex.cmd" login"'
    dialog.reject()
    assert not timer.isActive() and not dialog._provider_signins
    assert process.stopped == (['terminate'] if failure == 'already_gone'
                               else ['terminate', 0.1, 'kill', 0.1])
