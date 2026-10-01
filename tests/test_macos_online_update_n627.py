"""macOS online updates preserve their runtime and require approved shutdown."""
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest

from spacr import install_cleanup as ic

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Models the macOS POSIX runtime and handshake")


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / 'user home' / 'Library' / 'Application Support' / 'SpaCR'
    python = root / 'venv/bin/python'
    uv = root / 'bootstrap/uv'
    for path in (python, uv):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    kept = root / 'settings-to-preserve.json'
    kept.write_text('{"retained": true}')
    record = ic.InstallRecord(kind='installer', layout='macos-runtime', platform='macos',
                              root=str(root), python=str(python), version='1.5.0.1', running=True)
    machine = ic._Machine(platform='macos', environ={'HOME': str(tmp_path / 'user home')},
                          fs_root=str(tmp_path), executable=str(python), running_prefix=str(python.parent.parent))
    return record, machine, kept


def make_plan(tmp_path, installation):
    record, machine, _ = installation
    spawned = []
    other = ic.InstallRecord(kind='installer', layout='macos-app', platform='macos',
                             root=str(tmp_path / 'Applications/spaCR.app'))
    plan = ic.start_update_helper([other, record], '1.5.1.1', pid=43210,
                                 workdir=str(tmp_path / 'private helper'), system=machine,
                                 spawn=lambda *args: spawned.append(args))
    assert plan['command'] and plan['error'] is None
    assert len(spawned) == 1
    return plan, Path(plan['workdir']) / 'plan.json'


def approve_when_ready(monkeypatch, plan):
    original = ic._FrozenUpdateHandshake.ready

    def ready(handshake):
        original(handshake)
        ic._FrozenUpdateHandshake(plan).approve()

    monkeypatch.setattr(ic._FrozenUpdateHandshake, 'ready', ready)


def no_cleanup(*args, **kwargs):
    pytest.fail('The existing environment or another copy must never be removed')


def test_plan_targets_exact_private_paths_without_installer_or_bootstrap(tmp_path, installation):
    record, _, kept = installation
    plan, _ = make_plan(tmp_path, installation)
    assert plan['adapter'] == 'macos-online-uv-v1'
    assert plan['steps'] == ['wait', 'upgrade', 'verify', 'relaunch']
    assert plan['fetch'] is None
    assert plan['install'] == [str(Path(record.root) / 'bootstrap/uv'), 'pip', 'install',
                               '--upgrade', '--python', record.python, 'spacr']
    assert plan['command'][:2] == [record.python, '-I']
    assert plan['relaunch'] == [record.python, '-m', 'spacr.qt']
    assert kept.read_text() == '{"retained": true}'


def test_upgrade_waits_verifies_then_relaunches_once(tmp_path, installation, monkeypatch):
    _, machine, kept = installation
    plan, path = make_plan(tmp_path, installation)
    approve_when_ready(monkeypatch, plan)
    events = []

    def run(argv):
        events.append(('verify' if '-c' in argv else 'upgrade', argv))
        return (0, '1.5.1.1\n') if '-c' in argv else (0, 'Updated')

    def wait(pid):
        assert not events
        events.append(('wait', pid))
        return True

    assert ic._run_plan(str(path), wait=wait, run=run, remove=no_cleanup,
                        fetch=no_cleanup, system=machine,
                        spawn=lambda *args: events.append(('relaunch', args))) == 0
    assert [event[0] for event in events] == ['wait', 'upgrade', 'verify', 'relaunch']
    assert kept.exists()
    assert 'Verified spaCR 1.5.1.1' in Path(plan['log']).read_text()


@pytest.mark.parametrize('failure', ['upgrade', 'probe', 'wrong_version', 'relaunch'])
def test_failure_is_logged_without_false_success_or_extra_launch(tmp_path, installation, monkeypatch, failure):
    _, machine, kept = installation
    plan, path = make_plan(tmp_path, installation)
    approve_when_ready(monkeypatch, plan)
    calls = []

    def run(argv):
        if '-c' not in argv:
            return (2, 'network unavailable') if failure == 'upgrade' else (0, 'Installed')
        return (1, 'probe failed') if failure == 'probe' else (0, '1.5.0.1' if failure == 'wrong_version' else '1.5.1.1')

    def spawn(*args):
        calls.append(args)
        raise OSError('launch refused')

    assert ic._run_plan(str(path), wait=lambda pid: True, run=run, remove=no_cleanup,
                        system=machine, spawn=spawn) == 6
    assert len(calls) == (1 if failure == 'relaunch' else 0)
    assert kept.exists()
    log = Path(plan['log']).read_text()
    assert 'update stopped' in log and 'relaunch requested' not in log
    assert ic._FrozenUpdateHandshake(plan).status()['state'] == 'error'


@pytest.mark.parametrize('cancel_at', ['before_ready', 'after_ready', 'while_waiting'])
def test_cancelled_shutdown_never_upgrades_on_later_exit(tmp_path, installation, monkeypatch, cancel_at):
    _, machine, kept = installation
    plan, path = make_plan(tmp_path, installation)
    controller = ic._FrozenUpdateHandshake(plan)
    original = ic._FrozenUpdateHandshake.ready
    if cancel_at == 'before_ready':
        controller.cancel()
    else:
        def ready(handshake):
            original(handshake)
            controller.approve()
            if cancel_at == 'after_ready':
                controller.cancel()
        monkeypatch.setattr(ic._FrozenUpdateHandshake, 'ready', ready)

    def wait(pid):
        controller.cancel()
        return True  # A later unrelated close must still not authorize work.

    assert ic._run_plan(str(path), wait=wait, run=no_cleanup, remove=no_cleanup,
                        system=machine, spawn=no_cleanup) == 6
    assert kept.exists()


@pytest.mark.parametrize('unavailable', ['bootstrap/uv', 'venv/bin/python'])
def test_missing_executable_refuses_without_cleanup_fallback(tmp_path, installation, unavailable):
    record, machine, kept = installation
    (Path(record.root) / unavailable).unlink()
    assert ic._macos_online_update_record([record], system=machine) == record
    plan = ic.start_update_helper([record], '1.5.1.1', workdir=str(tmp_path / 'helper'),
                                 system=machine, spawn=no_cleanup)
    assert plan['adapter'] == 'macos-online-uv-v1'
    assert plan['command'] is None and 'unavailable' in plan['error']
    assert kept.exists()


def test_non_macos_or_frozen_helper_refuses_online_plan(tmp_path, installation, monkeypatch):
    _, machine, _ = installation
    plan, path = make_plan(tmp_path, installation)
    linux = ic._Machine(platform='linux', environ={}, fs_root=str(tmp_path))
    assert ic._run_plan(str(path), system=linux, run=no_cleanup, spawn=no_cleanup) == 6
    monkeypatch.setattr(ic.sys, 'frozen', True, raising=False)
    assert ic._run_plan(str(path), system=machine, run=no_cleanup, spawn=no_cleanup) == 6


def test_changed_plan_commands_cannot_select_another_environment(tmp_path, installation):
    _, machine, _ = installation
    plan, path = make_plan(tmp_path, installation)
    plan['install'][5] = '/another/python'
    path.write_text(json.dumps(plan))
    assert ic._run_plan(str(path), system=machine, run=no_cleanup, spawn=no_cleanup) == 6


def test_recognition_leaves_other_platforms_and_frozen_layouts_unchanged(installation):
    record, machine, _ = installation
    for changed in (replace(record, platform='linux'), replace(record, layout='macos-app'),
                    replace(record, running=False), replace(record, kind='environment')):
        assert ic._macos_online_update_record([changed], system=machine) is None


def test_native_launcher_runs_newer_existing_runtime_without_pinned_reinstall(tmp_path):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('C compiler unavailable')
    source = Path(__file__).resolve().parents[1] / 'packaging/online/macos_launcher.c'
    binary = tmp_path / 'launcher'
    subprocess.run([compiler, '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary)], check=True)
    home = tmp_path / 'user home'
    python = home / 'Library/Application Support/spaCR/venv/bin/python'
    python.parent.mkdir(parents=True)
    # The old application may have a lower package pin; an available runtime
    # is launched directly without asking any installer to restore that pin.
    python.write_text('#!/bin/sh\nprintf "1.5.9\\n"\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run([str(binary), '--diagnostic'], env={**os.environ, 'HOME': str(home)},
                            check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ['1.5.9', '-m', 'spacr.qt', '--diagnostic']


@pytest.mark.parametrize('observed', ['1.5.1.1', '1.5.2', '2.0.0.0'])
def test_newer_stable_release_is_accepted_without_an_old_pin(tmp_path, installation, monkeypatch, observed):
    _, machine, _ = installation
    plan, path = make_plan(tmp_path, installation)
    approve_when_ready(monkeypatch, plan)
    launches = []
    assert ic._run_plan(str(path), wait=lambda pid: True,
                        run=lambda argv: (0, observed if '-c' in argv else 'updated'),
                        spawn=lambda *args: launches.append(args), system=machine) == 0
    assert len(launches) == 1
    assert f'Verified spaCR {observed}' in Path(plan['log']).read_text()


def test_ambiguous_online_records_refuse_instead_of_destructive_fallback(tmp_path, installation):
    record, machine, kept = installation
    duplicate = replace(record, root=str(tmp_path / 'another online install'))
    assert ic._macos_online_update_record([record, duplicate], system=machine)
    plan = ic.start_update_helper([record, duplicate], '1.5.1.1',
                                 workdir=str(tmp_path / 'helper'), system=machine, spawn=no_cleanup)
    assert plan['adapter'] == 'macos-online-uv-v1'
    assert plan['command'] is None and 'ambiguous' in plan['error']
    assert kept.exists()


def test_real_private_executables_upgrade_verify_and_launch_in_order(tmp_path, installation, monkeypatch):
    record, machine, kept = installation
    state = tmp_path / 'installed version'
    journal = tmp_path / 'process journal'
    argv_log = tmp_path / 'uv arguments'
    def quote(path):
        return shlex.quote(str(path))
    uv = Path(record.root) / 'bootstrap/uv'
    uv.write_text('#!/bin/sh\n'
                  + f'printf "upgrade\\n" >> {quote(journal)}\n'
                  + f'printf "%s\\n" "$@" > {quote(argv_log)}\n'
                  + f'printf "1.5.2\\n" > {quote(state)}\n')
    Path(record.python).write_text('#!/bin/sh\nif [ "$1" = "-I" ]; then\n'
                                  + f'printf "verify\\n" >> {quote(journal)}\n'
                                  + f'cat {quote(state)}\nelse\n'
                                  + f'printf "relaunch\\n" >> {quote(journal)}\nfi\n')
    plan, path = make_plan(tmp_path, installation)
    approve_when_ready(monkeypatch, plan)

    def wait(pid):
        journal.write_text('wait\n')
        return True

    assert ic._run_plan(str(path), wait=wait, remove=no_cleanup, fetch=no_cleanup, system=machine) == 0
    deadline = time.monotonic() + 3
    while 'relaunch' not in journal.read_text() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert journal.read_text().splitlines() == ['wait', 'upgrade', 'verify', 'relaunch']
    assert argv_log.read_text().splitlines() == ['pip', 'install', '--upgrade', '--python', record.python, 'spacr']
    assert state.read_text().strip() == '1.5.2'
    assert kept.read_text() == '{"retained": true}'
