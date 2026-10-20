"""HostName substitutions agree with isolated OpenSSH configuration resolution."""
import getpass
import os
import shutil
import subprocess

import pytest

from spacr import remote_execution as remote


def _owner_only(tmp_path):
    """A folder Windows OpenSSH accepts config files from.

    Windows OpenSSH refuses (exit 255, "Bad owner or permissions") a config
    file whose ACL grants OWNER RIGHTS, which is what Python 3.12.4+ mkdtemp
    folders, and so pytest's tmp_path, hand down. A real ~/.ssh does not.
    """
    folder = tmp_path / 'ssh'
    folder.mkdir()
    if os.name == 'nt':
        subprocess.run(['icacls', str(folder), '/inheritance:r', '/grant:r',
                        getpass.getuser() + ':(OI)(CI)F'],
                       capture_output=True, check=True, timeout=30)
    return folder


@pytest.mark.skipif(shutil.which('ssh') is None, reason='Native OpenSSH parser unavailable')
@pytest.mark.parametrize('template', [
    '%h.cluster.invalid', '%%h.cluster.invalid', '%%%h.cluster.invalid',
    '%%%%h.cluster.invalid', '%h-%h.cluster.invalid', 'fixed.invalid',
    '%h%%.invalid', '%%.invalid', None,
])
def test_hostname_tokens_match_native_without_rescanning(tmp_path, template):
    tmp_path = _owner_only(tmp_path)
    config = tmp_path / 'config'
    child = tmp_path / 'child'
    child.write_text('Host wanted\n' + (' HostName ' + template + '\n' if template else ''), encoding="utf-8")
    config.write_text('Host wanted\n Include ' + str(child)
                      + '\n HostName first-fallback.invalid\n User controlled-user\n', encoding="utf-8")
    if template is None:
        config.write_text('Host wanted\n User controlled-user\n', encoding="utf-8")
    before = {p: p.read_bytes() for p in (config, child)}
    result = subprocess.run(
        ['ssh', '-G', '-F', str(config), '-o', 'CanonicalizeHostname=no',
         '-o', 'ControlMaster=no', 'wanted'],
        capture_output=True, text=True, check=True, timeout=5)
    native = dict(line.split(' ', 1) for line in result.stdout.splitlines() if ' ' in line)
    directives = remote._ssh_config_directives([config])
    actual = remote._ssh_lookup('wanted', directives)
    assert actual['hostname'] == native['hostname']
    assert actual['port'] == native['port']
    assert actual['user'] == native['user']
    assert {p: p.read_bytes() for p in before} == before
