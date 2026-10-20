"""A private pip venv changes only after verified and approved shutdown."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic
from tests.test_pip_runtime_transaction_703 import installation
from tests.test_macos_runtime_transaction_703 import _snapshot, _wheel
from tests.test_macos_online_update_n627 import desktop_boundary


@pytest.fixture(autouse=True)
def _private_owner_permissions(installation):
    _record, _machine, prefix = installation
    prefix.chmod(0o700)
    prefix.parent.chmod(0o700)


def _plan(tmp_path, installation):
    record, machine, _prefix = installation
    spawned = []
    plan = ic.start_update_helper([record], '1.5.1.5', pid=43210, workdir=str(tmp_path / 'helper'),
                                 system=machine, spawn=lambda *args: spawned.append(args))
    assert plan['command'] and plan['error'] is None, plan['error']
    assert len(spawned) == 1
    return plan, Path(plan['workdir']) / 'plan.json'


def test_private_pip_plan_isolated_and_no_installer_or_cleanup(installation, tmp_path):
    record, _machine, prefix = installation
    before = _snapshot(prefix)
    plan, _ = _plan(tmp_path, installation)
    assert plan['adapter'] == 'pip-private-posix-v1'
    assert plan['install'] == [record.python, '-I', '-m', 'pip', 'install', '--upgrade', 'spacr==1.5.1.5']
    assert plan['relaunch'] == [record.python, '-I', '-m', 'spacr.qt']
    assert plan['fetch'] is None
    assert plan['command'][:2] == [record.python, '-I']
    assert _snapshot(prefix) == before
    assert not (Path(plan['workdir']) / 'approved.json').exists()


@pytest.mark.parametrize('failure', [None, 'cancelled', 'pip-unavailable', 'unverified-current', 'restart'])
def test_real_offline_handoff_prepares_then_requires_approval(installation, tmp_path, monkeypatch, failure):
    record, machine, prefix = installation
    before = _snapshot(prefix)
    plan, path = _plan(tmp_path, installation)
    wheel = _wheel(tmp_path / 'new-wheels', '1.5.1.5')
    events = []
    original_ready = ic._FrozenUpdateHandshake.ready
    def ready(handshake):
        assert _snapshot(prefix) == before
        original_ready(handshake)
        events.append('ready')
        if failure == 'cancelled':
            handshake.cancel()
        else:
            handshake.approve()
    monkeypatch.setattr(ic._FrozenUpdateHandshake, 'ready', ready)
    def run(argv):
        if argv == [record.python, '-I', '-B', '-c', 'import pip'] and failure == 'pip-unavailable':
            return 1, 'pip is unavailable'
        if argv[0] == record.python and '-c' in argv and failure == 'unverified-current':
            return 0, '{}'
        if argv[1:6] == ['-I', '-m', 'pip', 'install', '--upgrade']:
            assert events == []
            assert _snapshot(prefix) == before
            assert argv[0] != record.python and argv[0] != sys.executable
            argv = [*argv[:-1], '--isolated', '--no-cache-dir', '--no-index', '--no-deps', str(wheel)]
            events.append('staged install')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout + result.stderr
    def wait(pid):
        assert pid == 43210
        events.append('exit')
        return True
    def spawn(*args):
        events.append('relaunch')
        if failure == 'restart':
            raise OSError('temporary relaunch deliberately refused')
    result = ic._run_plan(str(path), system=machine, run=run, wait=wait, spawn=spawn,
                          remove=lambda *args, **kwargs: pytest.fail('No installation or user data may be removed'),
                          fetch=lambda *args: pytest.fail('No installer download is permitted'))
    if failure:
        assert result != 0
        assert _snapshot(prefix) == before
        if failure not in {'restart', 'cancelled'}:
            assert 'staged install' not in events
    else:
        assert result == 0
        assert events == ['staged install', 'ready', 'exit', 'relaunch']
        transaction, = prefix.parent.glob('.spacr-runtime-update-*')
        assert _snapshot(transaction / 'runtime') == before


@pytest.mark.parametrize('reason', ['conda', 'system-site', 'windows', 'other-python', 'shared-parent', 'ambiguous', 'frozen', 'missing-python'])
def test_unsupported_environment_refuses_before_any_update(installation, tmp_path, monkeypatch, reason):
    record, machine, prefix = installation
    records = [record]
    if reason == 'conda':
        (prefix / 'conda-meta').mkdir()
    elif reason == 'system-site':
        # The helper rejects shared packages before readiness, not in the fast GUI preparation.
        (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = true\n', encoding="utf-8")
    elif reason == 'windows':
        machine.platform = 'windows'
        records = [replace(record, platform='windows')]
    elif reason == 'other-python':
        machine.executable = sys.executable
    elif reason == 'shared-parent':
        prefix.parent.chmod(0o777)
    elif reason == 'ambiguous':
        records.append(replace(record))
    elif reason == 'frozen':
        monkeypatch.setattr(ic.sys, 'frozen', True, raising=False)
    elif reason == 'missing-python':
        Path(record.python).unlink()
    before = _snapshot(prefix)
    calls = []
    plan = ic.start_update_helper(records, '1.5.1.5', workdir=str(tmp_path / 'helper'), system=machine,
                                 spawn=lambda *args: calls.append(args))
    if reason == 'system-site':
        assert plan['command']
        result = ic._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine,
                             run=lambda *args: pytest.fail('No package or probe may run'),
                             wait=lambda *args: pytest.fail('No shutdown may be requested'))
        assert result != 0
        assert ic._FrozenUpdateHandshake(plan).status()['state'] == 'error'
    else:
        assert not plan['command'] and plan['error']
        assert not calls
    assert _snapshot(prefix) == before


def test_changed_runtime_after_approval_is_not_downgraded(installation, tmp_path, monkeypatch):
    record, machine, prefix = installation
    before = _snapshot(prefix)
    plan, path = _plan(tmp_path, installation)
    original_ready = ic._FrozenUpdateHandshake.ready
    def ready(handshake):
        original_ready(handshake)
        handshake.approve()
    monkeypatch.setattr(ic._FrozenUpdateHandshake, 'ready', ready)
    exited = []
    probes = []
    wheel = _wheel(tmp_path / 'changed-runtime-wheel', '1.5.1.5')
    def run(argv):
        if argv[1:6] == ['-I', '-m', 'pip', 'install', '--upgrade']:
            assert argv[0] != record.python and not exited
            argv = [*argv[:-1], '--isolated', '--no-cache-dir', '--no-index', '--no-deps', str(wheel)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        if argv[0] == record.python and '-c' in argv and argv[-1] != 'import pip':
            probes.append(argv)
            if exited:
                receipt = json.loads(output)
                receipt['version'] = '1.5.1.6'
                return 0, json.dumps(receipt)
        return result.returncode, output
    result = ic._run_plan(str(path), system=machine, run=run,
                          wait=lambda pid: exited.append(pid) or True,
                          spawn=lambda *args: pytest.fail('Changed runtime cannot be relaunched automatically'))
    assert result != 0 and len(probes) == 2
    assert all('-B' in argv for argv in probes)
    assert _snapshot(prefix) == before
