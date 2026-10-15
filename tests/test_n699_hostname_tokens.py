"""HostName substitutions agree with isolated OpenSSH configuration resolution."""
import shutil
import subprocess

import pytest

from spacr import remote_execution as remote


@pytest.mark.skipif(shutil.which('ssh') is None, reason='Native OpenSSH parser unavailable')
@pytest.mark.parametrize('template', [
    '%h.cluster.invalid', '%%h.cluster.invalid', '%%%h.cluster.invalid',
    '%%%%h.cluster.invalid', '%h-%h.cluster.invalid', 'fixed.invalid',
    '%h%%.invalid', '%%.invalid', None,
])
def test_hostname_tokens_match_native_without_rescanning(tmp_path, template):
    config = tmp_path / 'config'
    child = tmp_path / 'child'
    child.write_text('Host wanted\n' + (' HostName ' + template + '\n' if template else ''))
    config.write_text('Host wanted\n Include ' + str(child)
                      + '\n HostName first-fallback.invalid\n User controlled-user\n')
    if template is None:
        config.write_text('Host wanted\n User controlled-user\n')
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
