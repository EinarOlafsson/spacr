"""Connection contract regressions; native OpenSSH probes never connect."""
import json
import os
import shutil
import subprocess
import sys

import pytest

import spacr.remote_execution as remote
from tests.test_ssh_discovery_mfa_699 import BadAuthenticationType, ssh_fake


def test_password_partial_auth_continues_to_otp_without_retrying_command(ssh_fake, monkeypatch):
    state = ssh_fake
    transport_class = sys.modules['paramiko'].Transport

    def password(self, username, value, fallback):
        assert username == 'alice' and value == 'KERBEROS-SECRET' and fallback is False
        self.password_accepted = True
        state.calls.append(('password-accepted',))
        return ['keyboard-interactive']

    def interactive(self, username, handler):
        if not getattr(self, 'password_accepted', False):
            raise BadAuthenticationType()
        answer = handler('Cluster', 'Second factor', [('Verification code:', False)])
        assert answer == ['OTP-SECRET']
        self.authenticated = True

    monkeypatch.setattr(transport_class, 'auth_password', password)
    monkeypatch.setattr(transport_class, 'auth_interactive', interactive)
    result = state.pool.run(state.target, 'hostname')
    assert result.stdout == 'node\n'
    assert len(state.transports) == 1 and len(state.channels) == 1
    assert [row[2][0][0] for row in state.prompts] == ['Password:', 'Verification code:']
    assert 'SECRET' not in json.dumps(state.target)


@pytest.mark.skipif(not shutil.which('ssh') or os.name != 'posix', reason='native POSIX OpenSSH required')
@pytest.mark.parametrize('option,output_key', [('identityfile', 'identityfile'), ('controlpath', 'controlpath')])
@pytest.mark.parametrize('path', ['/tmp/spacr native test/key', '/tmp/spacr "quoted" key'])
def test_native_openssh_receives_one_literal_path(option, output_key, path):
    profile = remote.ExecutionProfile('p', 'ssh', host='test.invalid', workdir='/work',
                                      ssh_options={'config': '/dev/null', option: path}).validate()
    result = subprocess.run(['ssh', '-G', '-o', 'CanonicalizeHostname=no', '-o', 'HostName=127.0.0.1',
                             *remote._ssh_overrides(profile), 'test.invalid'],
                            stdin=subprocess.DEVNULL, text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert f'{output_key} {path}' in result.stdout.splitlines()
