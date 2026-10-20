"""N699 passive discovery and verified session authentication, without a socket."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import spacr.remote_execution as remote


def test_includes_wildcards_precedence_and_match_exec_are_passive(tmp_path):
    ssh = tmp_path / '.ssh'
    ssh.mkdir()
    (ssh / 'parts').mkdir()
    (ssh / 'config').write_text('Include parts/*\nHost * !excluded\n User alice\n Port 2222\nHost excluded\n User bob\nMatch exec "touch SHOULD_NOT_EXIST"\n User wrong\n', encoding="utf-8")
    (ssh / 'parts/a').write_text('Host gl\n HostName greatlakes.example\n IdentityFile ~/.ssh/id_ed25519\n ProxyJump login\n ProxyCommand forbidden\n ControlMaster auto\n', encoding="utf-8")
    (ssh / 'known_hosts').write_text('public.example,second.example ssh-ed25519 PUBLIC\n|1|HASH|HASH ssh-ed25519 PUBLIC\n[ported.example]:2222 ssh-ed25519 PUBLIC\n', encoding="utf-8")
    system = tmp_path / 'system'
    system.write_text('User system\nHost gl\n Port 22\n', encoding="utf-8")
    rows = {row['alias']: row for row in remote._discover_ssh_settings(home=tmp_path, environ={}, system_config=system)}
    assert set(rows) == {'gl', 'excluded', 'public.example', 'second.example', 'ported.example'}
    assert rows['gl']['user'] == 'alice'
    assert rows['gl']['port'] == '2222'
    assert rows['gl']['hostname'] == 'greatlakes.example'
    assert rows['gl']['identityfile'] == str(ssh / 'id_ed25519')
    assert rows['gl']['proxyjump'] == 'login'
    assert 'proxycommand' not in rows['gl']
    assert rows['gl']['controlmaster'] == 'auto'
    assert rows['excluded']['user'] == 'bob'
    assert not (tmp_path / 'SHOULD_NOT_EXIST').exists()


def test_windows_userprofile_and_backslashes_survive(tmp_path):
    winhome = tmp_path / 'Windows'
    ssh = winhome / '.ssh'
    ssh.mkdir(parents=True)
    (ssh / 'config').write_text('Host=win\n HostName host.example\n IdentityFile "C:\\Users\\Ada\\.ssh\\key"\n', encoding="utf-8")
    rows = remote._discover_ssh_settings(home=tmp_path / 'other', environ={'USERPROFILE': str(winhome)}, system_config=tmp_path / 'missing')
    assert rows == [{'alias': 'win', 'config': str(ssh / 'config'), 'hostname': 'host.example',
                     'identityfile': r'C:\Users\Ada\.ssh\key', 'port': '22'}]


def test_known_hosts_only_does_not_select_missing_config(tmp_path):
    (tmp_path / 'known_hosts').write_text('@cert-authority host.example ssh-ed25519 PUBLIC\n', encoding="utf-8")
    assert remote._discover_ssh_settings(tmp_path)[0]['config'] == ''


@pytest.mark.parametrize('kind', ['cycle', 'large', 'private'])
def test_bad_config_is_bounded_without_executing_it(tmp_path, kind):
    config = tmp_path / 'config'
    text = {'cycle': 'Include config\n', 'large': 'x' * (1024 * 1024 + 1),
            'private': '-----BEGIN OPENSSH PRIVATE KEY-----\n'}[kind]
    config.write_text(text, encoding="utf-8")
    with pytest.raises(remote.RemoteExecutionError):
        remote._discover_ssh_settings(tmp_path)


@pytest.mark.parametrize('options', [{'password': 'never'}, {'port': '0'}, {'port': '65536'}, {'port': 'bad'},
                                     {'user': 'a\nb'}, {'proxycommand': 'one', 'proxyjump': 'two'}])
def test_only_nonsecret_valid_options_can_be_persisted(options):
    with pytest.raises(remote.RemoteExecutionError):
        remote.ExecutionProfile('p', 'ssh', host='host', workdir='/tmp', ssh_options=options).validate()


def test_old_profiles_and_external_ssh_remain_compatible():
    profile = remote.ExecutionProfile.from_dict({'name': 'p', 'backend': 'ssh', 'host': 'alias', 'workdir': '/work'})
    assert remote._remote_argv(profile, ['echo', 'a b']) == ['ssh', 'alias', "echo 'a b'"]
    options = {'config': '/temporary/config', 'user': 'alice', 'port': '2222'}
    profile = remote.ExecutionProfile('p', 'ssh', host='alias', workdir='/work', ssh_options=options)
    assert remote._remote_script(profile, 'true') == ['ssh', '-F', '/temporary/config', '-o', 'User=alice', '-o', 'Port=2222', 'alias', 'sh -c true']


class AuthenticationException(Exception):
    pass


class BadAuthenticationType(AuthenticationException):
    allowed_types = ['password']


class PasswordRequiredException(Exception):
    pass


class Channel:
    def __init__(self, output=b'node\n', error=b'', status=0):
        self.output, self.error, self.status = output, error, status
        self.sent = bytearray()
        self.closed = False
        self.command = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def exec_command(self, command):
        self.command = command

    def sendall(self, data):
        self.sent.extend(data)

    def shutdown_write(self):
        self.input_closed = True

    def recv_ready(self):
        return bool(self.output)

    def recv(self, count):
        chunk, self.output = self.output[:count], self.output[count:]
        return chunk

    def recv_stderr_ready(self):
        return bool(self.error)

    def recv_stderr(self, count):
        chunk, self.error = self.error[:count], self.error[count:]
        return chunk

    def exit_status_ready(self):
        return True

    def recv_exit_status(self):
        return self.status

    def close(self):
        self.closed = True


@pytest.fixture
def ssh_fake(monkeypatch, tmp_path):
    state = SimpleNamespace(transports=[], channels=[], prompts=[], calls=[], host_ok=True, denied=False,
                            password_only=False, publickeys=[], key_ok=False, partial_key=False, encrypted=False, channel_factory=Channel, interactive=None)
    known = tmp_path / 'known_hosts'
    known.write_text('fake-host ssh-ed25519 PUBLIC\n', encoding="utf-8")
    monkeypatch.setattr(remote, '_ssh_config_directives', lambda paths: [])
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda address, timeout: state.calls.append(('socket', address)) or SimpleNamespace(close=lambda: None))

    class HostKeys:
        def load(self, path):
            state.calls.append(('known', str(path)))

        def check(self, host, key):
            state.calls.append(('verify', host))
            return state.host_ok

    class Agent:
        def get_keys(self):
            return state.publickeys

        def close(self):
            state.calls.append(('agent-close',))

    class PKey:
        @staticmethod
        def from_path(path, passphrase=None):
            if state.encrypted and passphrase is None:
                raise PasswordRequiredException()
            return 'private-key'

    class Transport:
        def __init__(self, sock):
            self.authenticated = False
            self.active = True
            state.transports.append(self)

        def start_client(self, timeout):
            state.calls.append(('start', timeout))

        def get_remote_server_key(self):
            return 'server-key'

        def is_authenticated(self):
            return self.authenticated

        def is_active(self):
            return self.active

        def auth_publickey(self, username, key):
            state.calls.append(('publickey', username))
            if state.partial_key:
                return ["keyboard-interactive"]
            if not state.key_ok:
                raise AuthenticationException()
            self.authenticated = True

        def auth_interactive(self, username, handler):
            state.calls.append(('interactive', username))
            if state.interactive is not None:
                return state.interactive(self, handler)
            if state.password_only:
                raise BadAuthenticationType()
            first = handler('Great Lakes', 'Kerberos login', [('Password:', False)])
            second = handler('Great Lakes', 'Okta verification', [('Verification code:', False)])
            assert first == ['KERBEROS-SECRET']
            assert second == ['OTP-SECRET']
            self.authenticated = not state.denied

        def auth_password(self, username, password, fallback):
            assert password == 'KERBEROS-SECRET' and fallback is False
            self.authenticated = True

        def open_session(self, timeout):
            channel = state.channel_factory()
            state.channels.append(channel)
            return channel

        def open_channel(self, kind, destination, source, timeout):
            state.calls.append(('jump', destination))
            return SimpleNamespace(close=lambda: None)

        def close(self):
            self.active = False

    fake = SimpleNamespace(Transport=Transport, HostKeys=HostKeys, Agent=Agent, PKey=PKey,
                           AuthenticationException=AuthenticationException, BadAuthenticationType=BadAuthenticationType,
                           PasswordRequiredException=PasswordRequiredException,
                           ProxyCommand=lambda command: state.calls.append(('proxy', command)) or SimpleNamespace(close=lambda: None))
    monkeypatch.setitem(sys.modules, 'paramiko', fake)
    pool = remote._SSHSessionPool()

    def prompt(title, instructions, prompts):
        state.prompts.append((title, instructions, prompts))
        return ['OTP-SECRET' if 'code' in prompts[0][0] else 'KERBEROS-SECRET']

    pool.prompt = prompt
    state.pool = pool
    state.target = {'host': 'alias', 'options': {'hostname': 'fake-host', 'user': 'alice', 'userknownhostsfile': str(known)}}
    yield state
    pool.close()


def test_two_step_mfa_and_separate_json_channel_reuse_one_session(ssh_fake, monkeypatch):
    state = ssh_fake
    monkeypatch.setattr(remote, '_SSH_SESSIONS', state.pool)
    profile = remote.ExecutionProfile('p', 'ssh', host='alias', workdir='/work', ssh_options=state.target['options'], ssh_interactive=True).validate()
    result = remote._run_command(remote._remote_argv(profile, ['python', '-']), input_text='{"settings": true}')
    assert result.stdout == 'node\n'
    assert bytes(state.channels[0].sent) == b'{"settings": true}'
    assert state.channels[0].command == 'python -'
    remote._run_command(remote._remote_script(profile, 'true'))
    assert len(state.transports) == 1
    assert len(state.prompts) == 2
    assert state.prompts[1][1:] == ('Okta verification', [('Verification code:', False)])
    serialized = json.dumps(profile.to_dict())
    assert 'KERBEROS-SECRET' not in serialized and 'OTP-SECRET' not in serialized
    assert state.calls.index(('verify', 'fake-host')) < state.calls.index(('interactive', 'alice'))


@pytest.mark.parametrize('failure', ['host', 'denied', 'cancel', 'no-window'])
def test_authentication_failures_close_session_and_never_execute(ssh_fake, failure):
    state = ssh_fake
    if failure == 'host':
        state.host_ok = False
    elif failure == 'denied':
        state.denied = True
    elif failure == 'cancel':
        state.pool.prompt = lambda *args: None
    else:
        state.pool.prompt = None
    with pytest.raises(remote.RemoteExecutionError):
        state.pool.run(state.target, 'hostname')
    assert not state.channels
    assert not state.pool._sessions
    assert all(not transport.active for transport in state.transports)
    if failure == 'host':
        assert not state.prompts


def test_denied_push_allows_explicit_retry_without_automatic_submission(ssh_fake):
    state = ssh_fake
    state.denied = True
    with pytest.raises(remote.RemoteExecutionError):
        state.pool.run(state.target, 'sbatch job')
    assert not state.channels
    state.denied = False
    state.pool.run(state.target, 'hostname')
    assert len(state.transports) == 2
    assert len(state.channels) == 1


@pytest.mark.parametrize('kind', ['agent', 'identity', 'encrypted', 'password-only'])
def test_key_and_password_only_servers_have_explicit_authentication(ssh_fake, kind):
    state = ssh_fake
    if kind == 'agent':
        state.publickeys = ['agent-key']
        state.key_ok = True
    elif kind in {'identity', 'encrypted'}:
        state.target['options']['identityfile'] = '/temporary/key'
        state.key_ok = True
        state.encrypted = kind == 'encrypted'
    else:
        state.password_only = True
    state.pool.run(state.target, 'hostname')
    assert len(state.channels) == 1
    if kind in {'agent', 'identity'}:
        assert not state.prompts


@pytest.mark.parametrize('route', ['jump', 'proxy'])
def test_explicit_proxy_route_is_used_only_when_connecting(ssh_fake, route):
    state = ssh_fake
    if route == 'jump':
        state.target['options']['proxyjump'] = 'jump@bastion:2222'
    else:
        state.target['options']['proxycommand'] = 'proxy --host %h --port %p --user %r --alias %n %%'
    state.pool.run(state.target, 'hostname')
    assert any(call[0] == route for call in state.calls)
    state.pool.close()
    assert all(not transport.active for transport in state.transports)


def test_output_status_and_timeout_do_not_retry_remote_commands(ssh_fake):
    state = ssh_fake
    state.channel_factory = lambda: Channel(b'output', b'error', 3)
    assert state.pool.run(state.target, 'false') == remote.CommandResult(3, 'output', 'error')
    state.channel_factory = lambda: Channel(b'x' * (4 * 1024 * 1024 + 1))
    with pytest.raises(remote.RemoteExecutionError, match='capture limit'):
        state.pool.run(state.target, 'large')
    assert len(state.channels) == 2
    assert not state.pool._sessions


def test_include_never_reads_conventional_private_key_files(tmp_path):
    (tmp_path / 'config').write_text('Include *\n', encoding="utf-8")
    (tmp_path / 'id_ed25519').write_text('-----BEGIN OPENSSH PRIVATE KEY-----\nSECRET', encoding="utf-8")
    # A dedicated include matches only the private file; reading it would reject its header.
    (tmp_path / 'config').write_text('Include id_*\nHost public\nUser ada\n', encoding="utf-8")
    assert remote._discover_ssh_settings(tmp_path)[0]['user'] == 'ada'


def test_proxycommand_wins_when_it_appears_before_proxyjump(tmp_path):
    (tmp_path / 'config').write_text('Host public\n ProxyCommand proxy %h\n ProxyJump later\n', encoding="utf-8")
    row = remote._discover_ssh_settings(tmp_path)[0]
    assert row['proxycommand'] == 'proxy %h'
    assert 'proxyjump' not in row


def test_nondefault_known_host_port_is_kept(tmp_path):
    (tmp_path / 'known_hosts').write_text('[public.example]:2222 ssh-ed25519 PUBLIC\n', encoding="utf-8")
    assert remote._discover_ssh_settings(tmp_path)[0]['port'] == '2222'


def test_multiple_jump_hosts_own_and_close_their_entire_chain(ssh_fake):
    state = ssh_fake
    state.target['options']['proxyjump'] = 'first,second'
    state.pool.run(state.target, 'hostname')
    assert len(state.transports) == 3
    assert [call[1] for call in state.calls if call[0] == 'jump'] == [('second', 22), ('fake-host', 22)]
    state.pool.close()
    assert all(not transport.active for transport in state.transports)


@pytest.mark.parametrize('jump', ['alias', ','.join('host'+str(index) for index in range(8))])
def test_jump_cycles_and_hop_limit_stop_before_any_socket(ssh_fake, jump):
    state = ssh_fake
    state.target['options']['proxyjump'] = jump
    with pytest.raises(remote.RemoteExecutionError, match='cycle or hop limit'):
        state.pool.run(state.target, 'hostname')
    assert not [call for call in state.calls if call[0] == 'socket']


def test_a_dead_session_is_replaced_only_before_a_new_command(ssh_fake):
    state = ssh_fake
    state.pool.run(state.target, 'hostname')
    state.transports[0].active = False
    state.pool.run(state.target, 'hostname')
    assert len(state.transports) == 2
    assert len(state.channels) == 2


def test_remote_command_exception_drops_transport_and_does_not_leak_exception_text(ssh_fake):
    state = ssh_fake
    class BrokenChannel(Channel):
        def exec_command(self, command):
            raise OSError('PASSWORD-DO-NOT-LOG')
    state.channel_factory = BrokenChannel
    with pytest.raises(remote.RemoteExecutionError) as error:
        state.pool.run(state.target, 'sbatch one-job')
    assert 'OSError' in str(error.value)
    assert 'PASSWORD-DO-NOT-LOG' not in str(error.value)
    assert len(state.channels) == 1
    assert state.channels[0].closed
    assert not state.pool._sessions


def test_command_deadline_is_real_and_does_not_resubmit(ssh_fake, monkeypatch):
    import time
    state = ssh_fake
    class Waiting(Channel):
        def exit_status_ready(self):
            return False
    state.channel_factory = lambda: Waiting(b'')
    ticks = iter([0, 2])
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    with pytest.raises(remote.RemoteExecutionError, match='timed out'):
        state.pool.run(state.target, 'sbatch one-job', timeout=1)
    assert len(state.channels) == 1
    assert not state.pool._sessions


def test_initial_connection_error_closes_socket_and_contains_no_private_value(ssh_fake, monkeypatch):
    state = ssh_fake
    fake = sys.modules['paramiko']
    def broken(sock):
        raise OSError('SECRET-ADDRESS')
    monkeypatch.setattr(fake, 'Transport', broken)
    with pytest.raises(remote.RemoteExecutionError) as error:
        state.pool.run(state.target, 'hostname')
    assert 'SECRET-ADDRESS' not in str(error.value)
    assert not state.channels


def test_authentication_answer_count_is_checked_before_returning(ssh_fake):
    ssh_fake.pool.prompt = lambda *args: []
    with pytest.raises(remote.RemoteExecutionError, match='cancelled or timed out'):
        ssh_fake.pool._answers('SSH', '', [('Password:', False)])


def test_key_plus_mfa_moves_to_challenge_before_offering_more_agent_keys(ssh_fake):
    state = ssh_fake
    state.publickeys = ['first-key', 'second-key', 'third-key']
    state.partial_key = True
    state.pool.run(state.target, 'hostname')
    assert [call for call in state.calls if call[0] == 'publickey'] == [('publickey', 'alice')]
    assert len(state.prompts) == 2
    assert len(state.channels) == 1


@pytest.mark.parametrize('assignment', ['User=ada', 'User =ada', 'User= ada', 'User = ada'])
def test_openssh_assignment_spacing_is_equivalent(tmp_path, assignment):
    (tmp_path / 'config').write_text('Host = alias\n '+assignment+'\n', encoding="utf-8")
    assert remote._discover_ssh_settings(tmp_path)[0]['user'] == 'ada'


class FakeKeyring:
    def __init__(self):
        self.store, self.calls = {}, []

    def get_password(self, service, account):
        self.calls.append(('get', service, account))
        return self.store.get((service, account))

    def set_password(self, service, account, secret):
        self.calls.append(('set', service, account))
        self.store[(service, account)] = secret

    def delete_password(self, service, account):
        self.calls.append(('delete', service, account))
        self.store.pop((service, account), None)


def okta_server(state, outcome):
    """A fake keyboard-interactive server: Kerberos password, then an Okta push choice."""
    def interactive(transport, handler):
        password = handler('', 'Kerberos login for alice', [('Password:', False)])
        state.answers.append(list(password))
        if password != ['KERBEROS-SECRET']:
            raise AuthenticationException('Authentication failed.')
        choice = handler('', 'Okta Verify\n 1. Push to iPhone\n 2. Enter a code', [('Passcode or option (1-2): ', True)])
        state.answers.append(list(choice))
        if outcome[0] == 'denied':
            raise AuthenticationException('Authentication failed.')
        if outcome[0] == 'timeout':
            raise AuthenticationException('Authentication timeout.')
        transport.authenticated = True
    return interactive


@pytest.fixture
def okta(ssh_fake):
    state = ssh_fake
    state.answers = []
    outcome = ['approved']
    state.interactive = okta_server(state, outcome)

    def prompt(title, instructions, prompts):
        state.prompts.append((title, instructions, prompts))
        return ['1' if prompts[0][1] else 'KERBEROS-SECRET']

    state.pool.prompt = prompt
    state.outcome = outcome
    return state


def test_okta_two_step_prompts_are_verbatim_and_second_command_does_not_reprompt(okta):
    okta.pool.run(okta.target, 'hostname')
    assert okta.prompts == [('', 'Kerberos login for alice', [('Password:', False)]),
                            ('', 'Okta Verify\n 1. Push to iPhone\n 2. Enter a code', [('Passcode or option (1-2): ', True)])]
    assert okta.answers == [['KERBEROS-SECRET'], ['1']]
    okta.pool.run(okta.target, 'sinfo --noheader')
    assert len(okta.prompts) == 2
    assert len(okta.transports) == 1
    assert [channel.command for channel in okta.channels] == ['hostname', 'sinfo --noheader']


@pytest.mark.parametrize('outcome, wording', [('denied', 'was denied'), ('timeout', 'timed out')])
def test_denied_or_timed_out_push_is_explained_runs_nothing_and_retry_signs_in_again(okta, outcome, wording):
    okta.outcome[0] = outcome
    with pytest.raises(remote.RemoteExecutionError, match=wording) as caught:
        okta.pool.run(okta.target, 'sbatch job')
    assert 'Retry' in str(caught.value)
    assert 'KERBEROS-SECRET' not in str(caught.value)
    assert not okta.channels
    assert not okta.pool._sessions
    okta.outcome[0] = 'approved'
    okta.pool.run(okta.target, 'hostname')
    assert len(okta.prompts) == 4
    assert [channel.command for channel in okta.channels] == ['hostname']


def test_push_wait_gets_an_authentication_allowance_of_at_least_two_minutes(okta):
    okta.pool.run(okta.target, 'hostname', timeout=5)
    assert okta.transports[0].auth_timeout >= 120


def test_keychain_is_off_by_default_and_never_consulted(okta, monkeypatch):
    keyring = FakeKeyring()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    profile = remote.ExecutionProfile('p', 'ssh', host='alias', workdir='/work', ssh_options=okta.target['options'], ssh_interactive=True).validate()
    assert profile.ssh_keychain is False
    assert 'keychain' not in json.loads(remote._remote_argv(profile, ['hostname'])[1])
    okta.pool.run(okta.target, 'hostname')
    assert keyring.calls == []


def test_keychain_stores_only_the_password_and_reuses_it_once(okta, monkeypatch):
    keyring = FakeKeyring()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    profile = remote.ExecutionProfile('p', 'ssh', host='alias', workdir='/work', ssh_options=okta.target['options'],
                                      ssh_interactive=True, ssh_keychain=True).validate()
    target = json.loads(remote._remote_argv(profile, ['hostname'])[1])
    assert target['keychain'] is True
    okta.pool.run(target, 'hostname')
    assert keyring.store == {('spaCR SSH', 'alice@fake-host:22'): 'KERBEROS-SECRET'}
    assert 'KERBEROS-SECRET' not in json.dumps(profile.to_dict())
    okta.pool.close()
    okta.pool.run(target, 'hostname')
    assert [prompts for _, _, prompts in okta.prompts[2:]] == [[('Passcode or option (1-2): ', True)]]
    assert okta.answers[2:] == [['KERBEROS-SECRET'], ['1']]


def test_rejected_remembered_password_is_removed_from_the_keychain(okta, monkeypatch):
    keyring = FakeKeyring()
    keyring.store[('spaCR SSH', 'alice@fake-host:22')] = 'OLD-PASSWORD'
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    target = dict(okta.target, keychain=True)
    with pytest.raises(remote.RemoteExecutionError, match='was denied'):
        okta.pool.run(target, 'hostname')
    assert keyring.store == {}
    assert not okta.prompts


def test_keychain_failures_store_nothing_and_do_not_block_login(okta, monkeypatch):
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError('no keychain backend')
    monkeypatch.setitem(sys.modules, 'keyring', Broken())
    okta.pool.run(dict(okta.target, keychain=True), 'hostname')
    assert len(okta.prompts) == 2


def test_discovery_never_opens_private_key_files(tmp_path, monkeypatch):
    import builtins
    import io
    ssh = tmp_path / '.ssh'
    ssh.mkdir()
    keys = [ssh / 'id_ed25519', ssh / 'greatlakes_key', tmp_path / 'elsewhere.pem']
    for key in keys:
        key.write_text('-----BEGIN OPENSSH PRIVATE KEY-----\nSECRET\n', encoding="utf-8")
    (ssh / 'config').write_text('Host gl\n HostName greatlakes.arc-ts.umich.edu\n User ada\n'
                                f' IdentityFile ~/.ssh/greatlakes_key\n IdentityFile {keys[2]}\n'
                                'Host *\n IdentityFile ~/.ssh/id_ed25519\n', encoding="utf-8")
    (ssh / 'known_hosts').write_text('greatlakes.arc-ts.umich.edu ssh-ed25519 PUBLIC\n', encoding="utf-8")
    opened = []
    real_open = io.open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', recording_open)
    monkeypatch.setattr(io, 'open', recording_open)
    rows = {row['alias']: row for row in remote._discover_ssh_settings(home=tmp_path, environ={}, system_config=tmp_path / 'none')}
    assert rows['gl']['identityfile'] == str(ssh / 'greatlakes_key')
    assert opened and not ({str(key) for key in keys} & set(opened))
    assert 'SECRET' not in json.dumps(rows)


def test_discovery_never_runs_ssh_or_any_local_command(tmp_path, monkeypatch):
    import inspect
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError(f'discovery started a process: {args!r}')
    for name in ('run', 'Popen', 'call', 'check_call', 'check_output'):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(remote.os, 'system', forbidden)
    ssh = tmp_path / '.ssh'
    ssh.mkdir()
    (ssh / 'config').write_text('Match exec "ssh -G evil"\n User wrong\nHost gl\n HostName gl.example\n'
                                ' ProxyCommand ssh -W %h:%p bastion\n', encoding="utf-8")
    rows = remote._discover_ssh_settings(home=tmp_path, environ={}, system_config=tmp_path / 'none')
    assert rows[0]['proxycommand'] == 'ssh -W %h:%p bastion'
    assert 'user' not in rows[0]
    for function in (remote._discover_ssh_settings, remote._ssh_config_directives, remote._ssh_lookup, remote._ssh_resolve):
        assert '-G' not in inspect.getsource(function)


@pytest.mark.parametrize('field', ['ssh_interactive', 'ssh_keychain'])
def test_login_switches_must_be_real_booleans(field):
    profile = remote.ExecutionProfile('p', 'ssh', host='alias', workdir='/w', **{field: 'yes'})
    with pytest.raises(remote.RemoteExecutionError, match='must be a boolean'):
        profile.validate()


def test_blank_and_comment_lines_in_a_config_are_skipped(tmp_path):
    config = tmp_path / 'config'
    config.write_text('# cluster logins\n\nHost gl\n\n   # the login node\n HostName greatlakes.example\n', encoding="utf-8")
    assert remote._ssh_config_directives([config]) == [(None, 'top'), ('host', ['gl']), ('hostname', ['greatlakes.example'])]


def test_an_unknown_internal_marker_leaves_the_host_scope_unchanged():
    directives = [(None, 'top'), ('host', ['gl']), (None, 'not-a-marker'), ('user', ['alice'])]
    assert remote._ssh_lookup('gl', directives)['user'] == 'alice'
    assert 'user' not in remote._ssh_lookup('other', directives)


def test_a_windows_profile_folder_that_is_the_home_folder_is_read_once(tmp_path):
    ssh = tmp_path / '.ssh'
    ssh.mkdir()
    (ssh / 'config').write_text('Host gl\n HostName greatlakes.example\n', encoding="utf-8")
    (ssh / 'known_hosts').write_text('gl ssh-ed25519 PUBLIC\n', encoding="utf-8")
    rows = remote._discover_ssh_settings(home=tmp_path, environ={'USERPROFILE': str(tmp_path)},
                                         system_config=tmp_path / 'none')
    assert [row['alias'] for row in rows] == ['gl']


def test_an_oversized_known_hosts_file_is_refused(tmp_path):
    ssh = tmp_path / '.ssh'
    ssh.mkdir()
    with open(ssh / 'known_hosts', 'wb') as handle:
        handle.truncate(4 * 1024 * 1024 + 1)
    with pytest.raises(remote.RemoteExecutionError, match='too large'):
        remote._discover_ssh_settings(home=tmp_path, environ={}, system_config=tmp_path / 'none')


def test_known_hosts_keep_only_valid_ports_and_plain_host_names(tmp_path):
    ssh = tmp_path / '.ssh'
    ssh.mkdir()
    (ssh / 'known_hosts').write_text('[far.example]:99999 ssh-ed25519 PUBLIC\n'
                                     'bad$host,[odd.example]:22x ssh-ed25519 PUBLIC\n', encoding="utf-8")
    rows = {row['alias']: row for row in remote._discover_ssh_settings(
        home=tmp_path, environ={}, system_config=tmp_path / 'none')}
    assert set(rows) == {'far.example', 'odd.example'}
    assert rows['far.example']['port'] == '22'
    assert rows['odd.example']['port'] == '22'


def test_an_unknown_keychain_action_touches_nothing(monkeypatch):
    keyring = FakeKeyring()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    assert remote._keychain_password('rename', 'ada@gl:22', 'secret') is None
    assert keyring.calls == []


def test_the_keychain_account_names_user_host_and_port_only(monkeypatch):
    monkeypatch.setattr(remote, '_ssh_config_directives', lambda paths: [])
    account = remote._keychain_account('ada@gl', {'hostname': 'greatlakes.example', 'port': '2222'})
    assert account == 'ada@greatlakes.example:2222'


def test_rejected_and_silent_keys_fall_through_to_the_server_challenge(ssh_fake, monkeypatch):
    state = ssh_fake
    state.publickeys = ['rejected-key', 'silent-key']
    offered = []
    fake = sys.modules['paramiko']

    def auth_publickey(self, username, key):
        offered.append(key)
        if key == 'rejected-key':
            raise AuthenticationException()
        return []

    monkeypatch.setattr(fake.Transport, 'auth_publickey', auth_publickey)
    state.pool.run(state.target, 'hostname')
    assert offered == ['rejected-key', 'silent-key']
    assert len(state.prompts) == 2
    assert len(state.channels) == 1


def test_a_server_without_challenge_or_password_login_is_denied(ssh_fake, monkeypatch):
    state = ssh_fake
    state.password_only = True
    monkeypatch.setattr(BadAuthenticationType, 'allowed_types', ['publickey'])
    with pytest.raises(remote.RemoteExecutionError, match='denied'):
        state.pool.run(state.target, 'hostname')
    assert not state.prompts and not state.channels
    assert all(not transport.active for transport in state.transports)


def test_a_failed_final_hop_closes_every_jump_host_it_opened(ssh_fake, monkeypatch):
    state = ssh_fake
    state.target['options']['proxyjump'] = 'bastion'
    fake = sys.modules['paramiko']
    monkeypatch.setattr(fake.HostKeys, 'check', lambda self, host, key: host != 'fake-host')
    with pytest.raises(remote.RemoteExecutionError, match='host key'):
        state.pool.run(state.target, 'hostname')
    assert len(state.transports) == 2
    assert all(not transport.active for transport in state.transports)
    assert not state.channels


def test_a_quiet_command_is_polled_without_spinning(ssh_fake, monkeypatch):
    import time
    state = ssh_fake
    naps = []

    class Slow(Channel):
        def __init__(self):
            super().__init__(b'')
            self.polls = 0

        def exit_status_ready(self):
            self.polls += 1
            return self.polls > 2

    state.channel_factory = Slow
    monkeypatch.setattr(time, 'sleep', naps.append)
    assert state.pool.run(state.target, 'sleep 1') == remote.CommandResult(0, '', '')
    assert naps == [0.01, 0.01]
