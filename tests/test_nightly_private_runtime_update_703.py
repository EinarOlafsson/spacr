"""A Nightly update selection remains valid through private-runtime preparation."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.test_pip_runtime_transaction_703 import installation as pip_installation
from tests.test_macos_online_update_n627 import installation as mac_installation
from tests.test_macos_runtime_transaction_703 import installation as mac_runtime_installation, _installer, _snapshot, _wheel
from tests.test_updater_runtime_identity_703 import runtime as identity_runtime

from spacr import install_cleanup as cleanup

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Exercises private POSIX runtime replacement")


@pytest.mark.parametrize('version', ['1.5.1.5.dev1', '1.5.1.5rc1', '1.5.1.5.post1'])
def test_private_pip_plan_accepts_selected_nightly_dev_release(tmp_path, version):
    prefix = tmp_path / 'private environment'
    python = prefix / 'bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    prefix.chmod(0o700)
    prefix.parent.chmod(0o700)
    (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    record = cleanup.InstallRecord(kind='environment', platform='linux', layout='pip', root=str(prefix),
                                   python=str(python), version='1.5.1.4', running=True)
    machine = cleanup._Machine(platform='linux', environ={'HOME': str(tmp_path / 'user')},
                               fs_root=str(tmp_path), executable=str(python), running_prefix=str(prefix))
    spawned = []
    plan = cleanup.start_update_helper([record], version, pid=123,
                                      workdir=str(tmp_path / 'helper'), system=machine,
                                      spawn=lambda *args: spawned.append(args))
    assert plan['error'] is None, plan['error']
    assert len(spawned) == 1
    assert plan['install'] == [str(python), '-I', '-m', 'pip', 'install', '--upgrade', 'spacr==' + version]
    assert Path(plan['command'][2]).is_file()


@pytest.fixture
def private_installation(pip_installation):
    record, machine, prefix = pip_installation
    prefix.chmod(0o700)
    prefix.parent.chmod(0o700)
    return record, machine, prefix


@pytest.mark.parametrize('first,second,expected', [
    ('1.5.1.9', '1.5.1.10', -1), ('1.5.1.5.dev1', '1.5.1.5rc1', -1),
    ('1.5.1.5rc1', '1.5.1.5', -1), ('1.5.1.5.post1', '1.5.1.5', 1),
    ('1!1.5.1.4', '1.5.1.5', 1), ('1.5.1.5rc1', '1.5.1.5rc1', 0),
])
def test_version_order_uses_private_pep440_without_mutating_its_environment(identity_runtime, first, second, expected):
    root, python, _, _ = identity_runtime
    before = _snapshot(root)
    assert cleanup._runtime_version_compare(str(python), first, second) == expected
    assert _snapshot(root) == before


@pytest.mark.parametrize('value', ['', 'nightly', '1.5.1.5.dev', '1.5.1.5 --index-url https://invalid', '1.5.1.5;echo'])
def test_invalid_version_cannot_become_a_package_command(value):
    with pytest.raises(ValueError):
        cleanup._runtime_requested_version(value)


@pytest.mark.parametrize('answer', [(1, 'unavailable'), (0, '2'), (0, '{}')])
def test_failed_or_malformed_private_comparison_does_not_guess_an_order(answer):
    with pytest.raises(ValueError):
        cleanup._runtime_version_compare('not executed', '1.5.1.5rc1', '1.5.1.5', lambda argv: answer)


@pytest.mark.parametrize('version', ['1.5.1.5.dev1', '1.5.1.5rc1', '1.5.1.5RC1'])
def test_mac_online_prerelease_request_is_pinned_and_exactly_verified(mac_installation, tmp_path, version):
    record, _, _ = mac_installation
    install, probe, _ = cleanup._macos_online_commands(record, version, str(tmp_path / 'helper'))
    assert install[-1] == 'spacr==' + version.lower()
    expected = json.loads(probe[-1])
    assert expected['version'] == version.lower()
    assert expected['minimum'] is False


def _approve(monkeypatch):
    original = cleanup._FrozenUpdateHandshake.ready
    def ready(handshake):
        original(handshake)
        handshake.approve()
    monkeypatch.setattr(cleanup._FrozenUpdateHandshake, 'ready', ready)


@pytest.mark.parametrize('version', ['1.5.1.5.dev1', '1.5.1.5rc1'])
def test_real_offline_private_pip_nightly_update_requires_approval_and_keeps_old_runtime(private_installation, tmp_path, monkeypatch, version):
    record, machine, prefix = private_installation
    wheel = _wheel(tmp_path / 'nightly-wheels', version)
    plan = cleanup.start_update_helper([record], version, pid=123, workdir=str(tmp_path / 'helper'),
                                       system=machine, spawn=lambda *args: None)
    assert plan['error'] is None, plan['error']
    _approve(monkeypatch)
    installed = []
    def run(argv):
        if argv[1:6] == ['-I', '-m', 'pip', 'install', '--upgrade']:
            assert argv[-1] == 'spacr==' + version
            assert argv[0] != record.python
            installed.append(argv)
            argv = [*argv[:-1], '--isolated', '--no-cache-dir', '--no-index', '--no-deps', str(wheel)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout + result.stderr
    reopened = []
    code = cleanup._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine,
                             run=run, wait=lambda pid: True, spawn=lambda *args: reopened.append(args),
                             remove=lambda *args: pytest.fail('No runtime retirement'),
                             fetch=lambda *args: pytest.fail('No installer download'))
    assert code == 0 and len(installed) == 1 and len(reopened) == 1
    identity = subprocess.run(cleanup._runtime_identity_command(record.python, version),
                              text=True, capture_output=True, timeout=15)
    assert identity.returncode == 0, identity.stderr
    assert json.loads(identity.stdout)['version'] == version
    assert (prefix / 'unknown-user-folder/keep.bin').read_bytes() == b'not installer-owned data'
    journals = list(prefix.parent.glob('.spacr-runtime-update-*/transaction.json'))
    assert len(journals) == 1
    state = json.loads(journals[0].read_text())
    assert state['state'] == 'committed' and state['verified_version'] == version
    assert (Path(state['rollback_runtime']) / 'unknown-user-folder/keep.bin').read_bytes() == b'not installer-owned data'


@pytest.mark.parametrize('stable_fallback', [False, True])
def test_real_mac_style_uv_nightly_staging_never_accepts_a_stable_fallback(mac_runtime_installation, tmp_path, monkeypatch, stable_fallback):
    record, machine, prefix, _ = mac_runtime_installation
    version = '1.5.1.5rc1'
    wheel = _wheel(tmp_path / 'nightly-wheels', '1.5.1.5' if stable_fallback else version)
    uv = _installer(tmp_path, wheel)
    bootstrap = Path(record.root) / 'bootstrap'
    bootstrap.mkdir()
    shutil.copy2(uv, bootstrap / 'uv')
    before = _snapshot(prefix)
    plan = cleanup.start_update_helper([record], version, pid=123, workdir=str(tmp_path / 'helper'),
                                       system=machine, spawn=lambda *args: None)
    assert plan['error'] is None and plan['install'][-1] == 'spacr==' + version
    _approve(monkeypatch)
    reopened = []
    code = cleanup._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine, wait=lambda pid: True,
                             spawn=lambda *args: reopened.append(args),
                             remove=lambda *args: pytest.fail('No unrelated installation retirement'),
                             fetch=lambda *args: pytest.fail('No installer download'))
    if stable_fallback:
        assert code != 0 and not reopened
        assert _snapshot(prefix) == before
    else:
        assert code == 0 and len(reopened) == 1
        identity = subprocess.run(cleanup._runtime_identity_command(record.python, version),
                                  text=True, capture_output=True, timeout=15)
        assert identity.returncode == 0, identity.stderr


@pytest.mark.parametrize('current,target', [('1.5.1.5.dev2', '1.5.1.5.dev1'), ('1.5.1.5', '1.5.1.5rc1')])
def test_real_prerelease_downgrade_refuses_before_approval_or_package_operation(private_installation, tmp_path, monkeypatch, current, target):
    record, machine, prefix = private_installation
    wheel = _wheel(tmp_path / 'current-wheels', current)
    result = subprocess.run([record.python, '-I', '-m', 'pip', '--isolated', '--no-cache-dir',
                             'install', '--no-index', '--no-deps', '--upgrade', str(wheel)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record = replace(record, version=current)
    before = _snapshot(prefix)
    plan = cleanup.start_update_helper([record], target, pid=123, workdir=str(tmp_path / 'helper'),
                                       system=machine, spawn=lambda *args: None)
    assert plan['error'] is None, plan['error']
    monkeypatch.setattr(cleanup._FrozenUpdateHandshake, 'ready',
                        lambda handshake: pytest.fail('Downgrade cannot request shutdown approval'))
    def run(argv):
        assert argv[1:6] != ['-I', '-m', 'pip', 'install', '--upgrade']
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout + result.stderr
    code = cleanup._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine, run=run,
                             wait=lambda pid: pytest.fail('No shutdown on downgrade'),
                             spawn=lambda *args: pytest.fail('No relaunch on downgrade'))
    assert code != 0
    assert _snapshot(prefix) == before
