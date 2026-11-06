"""Connection preferences and session failures stay bounded and deterministic."""
import sys

import pytest

from spacr import remote_execution as remote
from tests.test_ssh_discovery_mfa_699 import (
    AuthenticationException, BadAuthenticationType, Channel, FakeKeyring, ssh_fake,
)


@pytest.mark.parametrize('field', ['ssh_interactive', 'ssh_keychain'])
def test_authentication_preferences_reject_strings_instead_of_enabling_them(field):
    profile = remote.ExecutionProfile(
        'local-test', 'ssh', host='test.invalid', workdir='/work', **{field: 'false'})

    with pytest.raises(remote.RemoteExecutionError, match='must be a boolean'):
        profile.validate()

    assert getattr(profile, field) == 'false'


def test_duplicate_home_locations_and_config_comments_do_not_duplicate_discovery(tmp_path, monkeypatch):
    folder = tmp_path / '.ssh'
    folder.mkdir()
    config = folder / 'config'
    config.write_text('\n # local configuration\nHost sample\n User reader # preferred user\n\n')
    original = config.read_bytes()
    read_directives = remote._ssh_config_directives
    seen = []

    def record_paths(paths):
        seen.extend(paths)
        return read_directives(paths)

    monkeypatch.setattr(remote, '_ssh_config_directives', record_paths)
    rows = remote._discover_ssh_settings(
        home=tmp_path, environ={'USERPROFILE': str(tmp_path)}, system_config=tmp_path / 'absent')

    assert rows == [{'alias': 'sample', 'config': str(config), 'hostname': 'sample',
                     'user': 'reader', 'port': '22'}]
    assert seen.count(config) == 1
    assert config.read_bytes() == original


def test_oversized_known_hosts_is_refused_before_reading_its_contents(tmp_path, monkeypatch):
    from pathlib import Path

    known = tmp_path / 'known_hosts'
    with known.open('wb') as stream:
        stream.seek(4 * 1024 * 1024)
        stream.write(b'\n')
    size = known.stat().st_size
    real_read = Path.read_text

    def read_text(path, *args, **kwargs):
        assert path != known, 'oversized known-hosts contents were read'
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    with pytest.raises(remote.RemoteExecutionError, match='known-hosts file is too large'):
        remote._discover_ssh_settings(tmp_path)
    assert known.stat().st_size == size


def test_invalid_known_host_names_and_ports_do_not_become_connection_options(tmp_path):
    known = tmp_path / 'known_hosts'
    known.write_text(
        '[zero.invalid]:0,[high.invalid]:65536,[text.invalid]:bad ssh-ed25519 PUBLIC\n'
        '-bad.invalid,bad/name,[good.invalid]:2222 ssh-ed25519 PUBLIC\n')
    original = known.read_bytes()

    rows = {row['alias']: row for row in remote._discover_ssh_settings(tmp_path)}

    assert set(rows) == {'zero.invalid', 'high.invalid', 'text.invalid', 'good.invalid'}
    assert {name: row['port'] for name, row in rows.items()} == {
        'zero.invalid': '22', 'high.invalid': '22', 'text.invalid': '22', 'good.invalid': '2222'}
    assert known.read_bytes() == original


def test_keychain_account_uses_the_resolved_host_and_explicit_login(tmp_path):
    config = tmp_path / 'config'
    config.write_text('Host alias\n HostName destination.invalid\n User configured\n Port 2222\n')
    original = config.read_bytes()

    account = remote._keychain_account('explicit@alias', {'config': str(config)})

    assert account == 'explicit@destination.invalid:2222'
    assert config.read_bytes() == original


def test_unknown_keychain_operation_cannot_change_a_saved_login(monkeypatch):
    keyring = FakeKeyring()
    keyring.store[('spaCR SSH', 'reader@test.invalid:22')] = 'saved fixture'
    monkeypatch.setitem(sys.modules, 'keyring', keyring)

    assert remote._keychain_password('unsupported', 'reader@test.invalid:22', 'replacement') is None

    assert keyring.calls == []
    assert keyring.store == {('spaCR SSH', 'reader@test.invalid:22'): 'saved fixture'}


@pytest.mark.parametrize('first_result', ['rejected', 'no_authentication'])
def test_an_unaccepted_agent_key_does_not_skip_a_later_valid_key(ssh_fake, monkeypatch, first_result):
    state = ssh_fake
    state.publickeys = ['first', 'second']
    offered = []

    def authenticate(transport, username, key):
        assert username == 'alice'
        offered.append(key)
        if key == 'first':
            if first_result == 'rejected':
                raise AuthenticationException()
            return []
        transport.authenticated = True
        return None

    monkeypatch.setattr(sys.modules['paramiko'].Transport, 'auth_publickey', authenticate)

    result = state.pool.run(state.target, 'hostname')

    assert result.stdout == 'node\n'
    assert offered == ['first', 'second']
    assert not state.prompts
    assert len(state.channels) == 1
    assert ('agent-close',) in state.calls


def test_unsupported_authentication_does_not_prompt_or_execute_a_command(ssh_fake):
    class PublicKeyOnly(BadAuthenticationType):
        allowed_types = ['publickey']

    def refuse(_transport, _handler):
        raise PublicKeyOnly()

    state = ssh_fake
    state.interactive = refuse
    with pytest.raises(remote.RemoteExecutionError, match='was denied'):
        state.pool.run(state.target, 'hostname')

    assert not state.prompts and not state.channels and not state.pool._sessions
    assert len(state.transports) == 1 and not state.transports[0].active


def test_target_host_refusal_closes_the_already_connected_jump_transport(ssh_fake, monkeypatch):
    state = ssh_fake
    state.target['options']['proxyjump'] = 'bastion'
    state.publickeys = ['fixture-key']
    state.key_ok = True

    def check(_known, host, _key):
        state.calls.append(('verify', host))
        return host == 'bastion'

    monkeypatch.setattr(sys.modules['paramiko'].HostKeys, 'check', check)
    with pytest.raises(remote.RemoteExecutionError, match='host key is unknown or changed'):
        state.pool.run(state.target, 'hostname')

    assert [call for call in state.calls if call[0] == 'verify'] == [
        ('verify', 'bastion'), ('verify', 'fake-host')]
    assert len(state.transports) == 2 and all(not item.active for item in state.transports)
    assert not state.channels and not state.pool._sessions


def test_an_idle_command_yields_until_output_arrives_without_resubmission(ssh_fake, monkeypatch):
    import time

    state = ssh_fake

    class Delayed(Channel):
        def __init__(self):
            super().__init__(b'')
            self.finished = False

        def exit_status_ready(self):
            return self.finished

    state.channel_factory = Delayed
    pauses = []

    def release_output(delay):
        pauses.append(delay)
        assert len(pauses) == 1
        state.channels[0].output = b'delayed result\n'
        state.channels[0].finished = True

    monkeypatch.setattr(time, 'sleep', release_output)
    result = state.pool.run(state.target, 'hostname', timeout=5)

    assert result == remote.CommandResult(0, 'delayed result\n', '')
    assert pauses == [0.01]
    assert len(state.channels) == 1 and state.channels[0].command == 'hostname'
    assert state.channels[0].closed
