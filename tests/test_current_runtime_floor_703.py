"""A saved update plan cannot lower the actual current private-runtime version."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from spacr import install_cleanup as cleanup
from tests.test_macos_runtime_transaction_703 import acknowledging_spawn, installation, _installer, _snapshot, _wheel

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='Exercises the POSIX runtime adapter')


def _install(python, wheel):
    result = subprocess.run([sys.executable, '-I', '-B', '-m', 'pip', '--python', python,
                             '--isolated', '--no-cache-dir', 'install', '--no-index', '--no-deps',
                             '--upgrade', str(wheel)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _plan(record, machine, tmp_path):
    uv = _installer(tmp_path, _wheel(tmp_path / 'target wheels', '1.5.1.5'))
    bootstrap = Path(record.root) / 'bootstrap'
    bootstrap.mkdir()
    shutil.copy2(uv, bootstrap / 'uv')
    plan = cleanup.start_update_helper([record], '1.5.1.5', pid=123,
                                       workdir=str(tmp_path / 'helper'), system=machine,
                                       spawn=lambda *args: None)
    assert plan['error'] is None, plan['error']
    return plan


def _approve(monkeypatch, events):
    original = cleanup._FrozenUpdateHandshake.ready
    def ready(handshake):
        events.append('approval')
        original(handshake)
        handshake.approve()
    monkeypatch.setattr(cleanup._FrozenUpdateHandshake, 'ready', ready)


def test_stale_mac_plan_cannot_downgrade_actual_newer_runtime(installation, tmp_path, monkeypatch):
    record, machine, prefix, _ = installation
    _install(record.python, _wheel(tmp_path / 'current wheels', '1.5.1.6'))
    before = _snapshot(prefix)
    plan = _plan(record, machine, tmp_path)
    events = []
    _approve(monkeypatch, events)
    code = cleanup._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine,
                             wait=lambda pid: events.append('shutdown') or True,
                             spawn=lambda *args: events.append('relaunch'))
    assert code != 0, 'A stale recorded version must not authorize an actual runtime downgrade'
    assert not events
    assert _snapshot(prefix) == before


def test_runtime_changed_while_waiting_is_rechecked_under_actual_lease(installation, tmp_path, monkeypatch):
    record, machine, prefix, _ = installation
    plan = _plan(record, machine, tmp_path)
    events = []
    _approve(monkeypatch, events)
    before = []
    original_identity = cleanup._retiring_runtime_identity
    probes = []
    def identity(path):
        probes.append((prefix.parent / '.venv.spacr-update-lock').is_dir())
        return original_identity(path)
    monkeypatch.setattr(cleanup, '_retiring_runtime_identity', identity)
    def wait(pid):
        events.append('shutdown')
        _install(record.python, _wheel(tmp_path / 'intervening wheels', '1.5.1.6'))
        before.append(_snapshot(prefix))
        return True
    code = cleanup._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine,
                             wait=wait, spawn=lambda *args: events.append('relaunch'))
    assert code != 0 and events == ['approval', 'shutdown']
    assert probes[0] is False and len(probes) >= 2 and all(probes[1:]), (
        'every identity check after the first runs under the actual update lease')
    assert _snapshot(prefix) == before[0]
    assert not (prefix.parent / '.venv.spacr-update-lock').exists()
    assert not list(prefix.parent.glob('.spacr-runtime-update-*'))


def test_readonly_current_identity_allows_repairing_broken_old_qt(installation, tmp_path, monkeypatch):
    record, machine, prefix, site = installation
    old_qt = site / 'spacr/qt.py'
    old_qt.write_text('raise RuntimeError("old Qt dependencies are broken")\n')
    before = _snapshot(prefix)
    plan = _plan(record, machine, tmp_path)
    events = []
    _approve(monkeypatch, events)
    launched = []
    spawn = acknowledging_spawn(launched)
    code = cleanup._run_plan(str(Path(plan['workdir']) / 'plan.json'), system=machine,
                             wait=lambda pid: events.append('shutdown') or True,
                             spawn=lambda *args: events.append('relaunch') or spawn(*args))
    assert code == 0 and events == ['approval', 'shutdown', 'relaunch'] and len(launched) == 1
    identity = subprocess.run(cleanup._runtime_identity_command(record.python, '1.5.1.5'),
                              capture_output=True, text=True, timeout=15)
    assert identity.returncode == 0, identity.stderr
    state = json.loads(next(prefix.parent.glob('.spacr-runtime-update-*/transaction.json')).read_text())
    assert _snapshot(Path(state['rollback_runtime'])) == before
