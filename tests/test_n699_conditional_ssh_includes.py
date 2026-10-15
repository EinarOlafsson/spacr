"""Static Include scopes agree with private native OpenSSH configuration probes."""
import getpass
import shutil
import subprocess

import pytest

from spacr import remote_execution as remote


@pytest.mark.skipif(shutil.which('ssh') is None, reason='Native OpenSSH parser unavailable')
@pytest.mark.parametrize('case', [
    'inactive', 'active', 'negated', 'wildcard', 'nested_inactive',
    'nested_active', 'parent_restore', 'first_value', 'glob_restore', 'nested_restore',
])
def test_include_scopes_match_native_private_config(tmp_path, case):
    child = tmp_path / 'child.conf'
    grandchild = tmp_path / 'grandchild.conf'
    target = 'wanted'
    parent = 'Host wanted\n Include {child}\n'
    child_text = 'Host *\n User included-user\n'
    grandchild.write_text('Host *\n User nested-user\n')
    if case == 'inactive':
        parent = 'Host unrelated\n Include {child}\nHost wanted\n HostName target.invalid\n'
    elif case == 'negated':
        target = 'wanted.invalid'
        parent = 'Host *.invalid !wanted.invalid\n Include {child}\nHost wanted.invalid\n HostName target.invalid\n'
    elif case == 'wildcard':
        target = 'wanted.invalid'
        parent = 'Host *.invalid !blocked.invalid\n Include {child}\n'
    elif case == 'nested_inactive':
        parent = 'Host unrelated\n Include {child}\nHost wanted\n HostName target.invalid\n'
        child_text = 'Host wanted\n Include {grandchild}\n'
    elif case == 'nested_active':
        child_text = 'Host *\n Include {grandchild}\n'
    elif case == 'parent_restore':
        parent += ' User outer-user\n'
        child_text = 'Host unrelated\n User unrelated-user\n'
    elif case == 'first_value':
        parent = 'User first-user\n' + parent
    elif case == 'glob_restore':
        fragments = tmp_path / 'fragments'
        fragments.mkdir()
        (fragments / 'a.fragment').write_text('Host unrelated\n Port 2223\n')
        (fragments / 'b.fragment').write_text('User second-file-user\n')
        parent = 'Host wanted\n Include ' + str(fragments / '*.fragment') + '\n'
    elif case == 'nested_restore':
        child_text = 'Host wanted\n Include {grandchild}\n User child-after-user\n'
        grandchild.write_text('Host unrelated\n User unrelated-user\n')
    child.write_text(child_text.format(grandchild=grandchild))
    config = tmp_path / 'config'
    config.write_text(parent.format(child=child))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    result = subprocess.run(
        ['ssh', '-G', '-F', str(config), '-o', 'CanonicalizeHostname=no',
         '-o', 'ControlMaster=no', target],
        capture_output=True, text=True, check=True, timeout=5,
    )
    native = dict(line.split(' ', 1) for line in result.stdout.splitlines() if ' ' in line)
    directives = remote._ssh_config_directives([config])
    actual = remote._ssh_lookup(target, directives)
    assert actual.get('user', getpass.getuser()) == native['user']
    assert actual['hostname'] == native['hostname']
    assert actual['port'] == native['port']
    assert {path: path.read_bytes() for path in before} == before


def test_passive_discovery_never_executes_match_and_ignores_internal_markers(tmp_path):
    sentinel = tmp_path / 'must-not-exist'
    child = tmp_path / 'child.conf'
    child.write_text('Host *\n User wrong-included-user\n')
    config = tmp_path / 'config'
    config.write_text(
        'Match exec "touch ' + str(sentinel) + '"\n Include ' + str(child)
        + '\nHost wanted\n User wanted-user\n'
    )
    rows = remote._discover_ssh_settings(folder=tmp_path, home=tmp_path, environ={})
    wanted = next(row for row in rows if row['alias'] == 'wanted')
    assert wanted['user'] == 'wanted-user'
    assert all(isinstance(row['alias'], str) for row in rows)
    assert not sentinel.exists()


def test_unsupported_text_cannot_impersonate_internal_scope_markers(tmp_path):
    config = tmp_path / 'config'
    config.write_text(
        'Host unrelated\n __top anything\n __include_start anything\n'
        ' __include_end anything\n User wrong-user\n'
        'Host wanted\n User wanted-user\n'
    )
    directives = remote._ssh_config_directives([config])
    actual = remote._ssh_lookup('wanted', directives)
    assert actual['user'] == 'wanted-user'
    assert remote._ssh_lookup('another', directives).get('user') is None
