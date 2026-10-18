"""Frozen Windows update refusals, journals and lock retention, plus plan adapters."""
import ctypes
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from spacr import install_cleanup as cleanup
from tests.test_frozen_windows_update_703 import (  # noqa: F401
    identity, plan_for, run_update, runtime, snapshot, update)


def _lock_dir(runtime):
    root = os.path.realpath(str(runtime.root))
    digest = hashlib.sha256(os.path.normcase(root).encode()).hexdigest()
    return runtime.home / '.spacr' / 'update-locks' / digest


def test_record_selection_refuses_a_non_windows_computer(runtime):
    machine = cleanup._Machine(platform='linux', environ={'HOME': str(runtime.home)})
    with pytest.raises(ValueError, match='windows_frozen_layout'):
        cleanup._windows_frozen_update_record([runtime.record], '1.5.1.5', str(runtime.helper), machine)


def test_record_selection_refuses_a_helper_folder_inside_the_installation(runtime):
    inside = runtime.root / 'helper'
    with pytest.raises(ValueError, match='windows_frozen_helper_location'):
        cleanup._windows_frozen_update_record([runtime.record], '1.5.1.5', str(inside), runtime.machine)


def test_record_selection_accepts_the_real_installation_and_reports_its_version(runtime):
    record, previous = cleanup._windows_frozen_update_record(
        [runtime.record], '1.5.1.5', str(runtime.helper), runtime.machine)
    assert record == runtime.record and previous == '1.5.1.4'


@pytest.mark.parametrize('expected', [None, 'ABC', 'g' * 64])
def test_installer_pin_rejects_malformed_published_checksum(runtime, expected):
    installer = runtime.helper / 'setup.exe'
    installer.write_bytes(b'MZ')
    with pytest.raises(ValueError, match='windows_frozen_published_checksum'):
        with cleanup._windows_frozen_installer_pin(str(installer), expected):
            pytest.fail('a malformed checksum must not open the installer')


@pytest.mark.parametrize('kind', ['empty', 'directory'])
def test_installer_pin_rejects_empty_or_non_regular_installer(runtime, kind):
    installer = runtime.helper / 'setup.exe'
    if kind == 'empty':
        installer.write_bytes(b'')
    else:
        installer.mkdir()
    with pytest.raises(ValueError, match='windows_frozen_installer'):
        with cleanup._windows_frozen_installer_pin(str(installer), 'a' * 64):
            pytest.fail('an unusable installer must not be hashed')


def _native_os(monkeypatch, kernel, osfhandle):
    proxy = types.SimpleNamespace(**vars(os))
    proxy.name = 'nt'
    proxy.O_BINARY = 0
    monkeypatch.setattr(cleanup, '_windows_update_library', lambda name: kernel)
    monkeypatch.setattr(cleanup, 'os', proxy)
    monkeypatch.setitem(sys.modules, 'msvcrt', types.SimpleNamespace(open_osfhandle=osfhandle))


def test_native_installer_pin_reports_the_windows_error_when_create_fails(runtime, monkeypatch):
    installer = runtime.helper / 'setup.exe'
    installer.write_bytes(b'MZ installer')
    errors = []
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: 32, raising=False)
    monkeypatch.setattr(ctypes, 'WinError', lambda code: errors.append(code) or OSError(code, 'sharing'),
                        raising=False)
    kernel = types.SimpleNamespace(CreateFileW=lambda *args: None, CloseHandle=lambda handle: 1)
    _native_os(monkeypatch, kernel, lambda handle, flags: pytest.fail('no handle to convert'))
    with pytest.raises(OSError) as error:
        with cleanup._windows_frozen_installer_pin(str(installer), 'a' * 64):
            pytest.fail('a failed native open must not yield')
    assert error.value.errno == 32 and errors == [32]


def test_native_installer_pin_closes_the_handle_when_conversion_fails(runtime, monkeypatch):
    installer = runtime.helper / 'setup.exe'
    installer.write_bytes(b'MZ installer')
    closed = []

    def convert(handle, flags):
        raise OSError('modeled descriptor table full')

    kernel = types.SimpleNamespace(CreateFileW=lambda *args: 77, CloseHandle=lambda handle: closed.append(handle))
    _native_os(monkeypatch, kernel, convert)
    with pytest.raises(OSError, match='descriptor table full'):
        with cleanup._windows_frozen_installer_pin(str(installer), 'a' * 64):
            pytest.fail('a failed conversion must not yield')
    assert closed == [77]


def test_installer_replaced_between_inspection_and_open_is_refused(runtime, monkeypatch):
    installer = runtime.helper / 'setup.exe'
    installer.write_bytes(b'MZ original installer')
    expected = hashlib.sha256(b'MZ original installer').hexdigest()

    def swap(path):
        replacement = runtime.helper / 'replacement.exe'
        replacement.write_bytes(b'MZ original installer')
        os.replace(replacement, path)

    monkeypatch.setattr(cleanup, '_windows_pip_private_access', swap)
    with pytest.raises(ValueError, match='windows_frozen_installer_identity'):
        with cleanup._windows_frozen_installer_pin(str(installer), expected):
            pytest.fail('a swapped installer must not yield')


def test_installer_growing_past_the_size_bound_is_refused(runtime, monkeypatch):
    installer = runtime.helper / 'setup.exe'
    installer.write_bytes(b'MZ small installer')

    class Huge(bytes):
        def __len__(self):
            return 8 * 1024 ** 3 + 1

    class Stream:
        def __init__(self, inner):
            self.inner = inner
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.inner.close()
            return False

        def fileno(self):
            return self.inner.fileno()

        def read(self, size):
            self.reads += 1
            return Huge(b'x') if self.reads == 1 else b''

    proxy = types.SimpleNamespace(**vars(os))
    real_fdopen = os.fdopen
    proxy.fdopen = lambda descriptor, mode: Stream(real_fdopen(descriptor, mode))
    monkeypatch.setattr(cleanup, 'os', proxy)
    with pytest.raises(ValueError, match='windows_frozen_installer_size'):
        with cleanup._windows_frozen_installer_pin(str(installer), 'a' * 64):
            pytest.fail('an oversized installer must not yield')


def test_failed_journal_publish_leaves_no_temporary_file(runtime, monkeypatch):
    plan = plan_for(runtime)

    def refuse(source, destination):
        raise OSError('modeled replace failure')

    monkeypatch.setattr(cleanup.os, 'replace', refuse)
    with pytest.raises(OSError, match='replace failure'):
        cleanup._windows_frozen_update_journal(str(runtime.helper), plan, 'preparing', root='r')
    assert not list(runtime.helper.glob('.journal-*'))
    assert not (runtime.helper / 'journal.json').exists()


def test_helper_interpreter_inside_the_installation_is_refused(update, monkeypatch):
    before = snapshot(update.runtime.root)
    monkeypatch.setattr(cleanup.sys, 'executable', str(update.runtime.root / 'spacr.exe'))
    with pytest.raises(ValueError, match='windows_frozen_outside_helper'):
        run_update(update)
    assert update.actions == [] and snapshot(update.runtime.root) == before


def test_tampered_relaunch_command_is_refused(update):
    update.plan['relaunch'] = [str(update.runtime.helper / 'other.exe')]
    with pytest.raises(ValueError, match='windows_frozen_plan'):
        run_update(update)
    assert update.actions == []


def test_existing_staged_installer_is_refused_and_lock_released(update):
    Path(update.plan['fetch']['path']).write_bytes(b'left over from another attempt')
    with pytest.raises(ValueError, match='windows_frozen_existing_stage'):
        run_update(update)
    assert update.actions == []
    assert not _lock_dir(update.runtime).exists()
    assert Path(update.plan['fetch']['path']).read_bytes() == b'left over from another attempt'


def test_destination_changed_during_download_is_refused_before_readiness(update):
    def download(url, path):
        update.download(url, path)
        update.runtime.executable.write_bytes(b'executable replaced during download')

    with pytest.raises(ValueError, match='windows_frozen_changed_destination'):
        run_update(update, download=download)
    assert update.handshake.status() is None
    assert 'install' not in update.actions


def _direct_coordinator(monkeypatch, before_launch=None, error=None):
    def coordinate(plan, handshake, operation, *, check, launch, wait=None):
        check()
        if error is not None:
            raise error
        if before_launch is not None:
            before_launch()
        return 'modeled output', launch()

    monkeypatch.setattr(cleanup, '_coordinate_native_update', coordinate)
    monkeypatch.setattr(cleanup, '_frozen_windows_payload', lambda root, version: None)
    monkeypatch.setattr(cleanup, '_frozen_windows_identity', lambda root, version, run: {'version': version})


def _journal(update):
    return json.loads((_lock_dir(update.runtime) / 'journal.json').read_text())


def test_launch_refuses_a_destination_whose_identity_changed(update, monkeypatch):
    changed = []
    real_state = cleanup._windows_frozen_root_state

    def state(root):
        observed_root, observed = real_state(root)
        if changed:
            observed = [(0, 0)] + observed[1:]
        return observed_root, observed

    monkeypatch.setattr(cleanup, '_windows_frozen_root_state', state)
    _direct_coordinator(monkeypatch, before_launch=lambda: changed.append(True))
    monkeypatch.setattr(cleanup, '_launch_updated_desktop', lambda *a, **k: pytest.fail('must not relaunch'))
    with pytest.raises(ValueError, match='windows_frozen_changed_destination'):
        run_update(update)
    journal = _journal(update)
    assert journal['state'] == 'desktop-unconfirmed'
    assert 'windows_frozen_changed_destination' in journal['error']


def test_launch_refuses_an_elevated_relaunch(update, monkeypatch):
    elevated = []
    monkeypatch.setattr(cleanup, '_windows_process_elevated', lambda: bool(elevated))
    _direct_coordinator(monkeypatch, before_launch=lambda: elevated.append(True))
    monkeypatch.setattr(cleanup, '_launch_updated_desktop', lambda *a, **k: pytest.fail('must not relaunch'))
    with pytest.raises(ValueError, match='windows_frozen_elevated_relaunch'):
        run_update(update)
    assert _journal(update)['state'] == 'desktop-unconfirmed'


def test_verified_launch_publishes_desktop_ready_result(update, monkeypatch):
    _direct_coordinator(monkeypatch)
    launched = []

    def launch(plan, argv, version, root, *, frozen, spawn):
        launched.append((argv, version, frozen))
        return {'version': version, 'pid': 9}

    monkeypatch.setattr(cleanup, '_launch_updated_desktop', launch)
    assert run_update(update) == ('1.5.1.5', 'modeled output')
    assert launched == [([str(update.runtime.executable)], '1.5.1.5', True)]
    results = list(update.runtime.helper.glob('windows-update-result-*.json'))
    assert [json.loads(path.read_text())['state'] for path in results] == ['desktop-ready']
    assert not _lock_dir(update.runtime).exists()


def test_timeout_before_staging_releases_the_lock(update):
    def download(url, path):
        raise TimeoutError('modeled slow download')

    with pytest.raises(TimeoutError, match='slow download'):
        run_update(update, download=download)
    assert not _lock_dir(update.runtime).exists()
    assert not list(update.runtime.helper.glob('windows-update-result-*.json'))


def test_unwritable_uncertain_journal_still_reports_the_timeout(update, monkeypatch):
    _direct_coordinator(monkeypatch, error=TimeoutError('modeled installer still running'))
    real_journal = cleanup._windows_frozen_update_journal

    def journal(directory, plan, state, **details):
        if state == 'installer-uncertain':
            raise OSError('modeled full disk')
        return real_journal(directory, plan, state, **details)

    monkeypatch.setattr(cleanup, '_windows_frozen_update_journal', journal)
    with pytest.raises(TimeoutError, match='still running'):
        run_update(update)
    assert _journal(update)['state'] == 'desktop-unconfirmed'


def test_lock_with_foreign_content_is_retained_after_refusal(update):
    def download(url, path):
        (_lock_dir(update.runtime) / 'foreign').write_bytes(b'not ours')
        raise ValueError('modeled refused download')

    with pytest.raises(ValueError, match='refused download'):
        run_update(update, download=download)
    assert (_lock_dir(update.runtime) / 'foreign').read_bytes() == b'not ours'


def _linux_machine(tmp_path, platform='linux'):
    home = tmp_path / 'home'
    home.mkdir(exist_ok=True)
    return cleanup._Machine(platform=platform, environ={'HOME': str(home)}, fs_root=str(tmp_path))


def test_plan_without_a_running_installer_reports_why(tmp_path):
    spawned = []
    plan = cleanup.start_update_helper([], '1.5.1.5', pid=4, workdir=str(tmp_path / 'work'),
                                       system=_linux_machine(tmp_path), spawn=lambda *a: spawned.append(a))
    assert plan['error'] == 'the running spaCR is not an installer-made copy'
    assert plan['command'] is None and spawned == []
    assert json.loads((tmp_path / 'work' / 'plan.json').read_text())['error'] == plan['error']


def _deb_record(tmp_path):
    return cleanup.InstallRecord(kind='installer', layout='linux-deb', platform='linux',
                                 root=str(tmp_path / 'opt' / 'spacr'), version='1.5.1.4', running=True)


def test_frozen_debian_plan_uses_the_standalone_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(cleanup, '_debian_update_commands',
                        lambda records, version, machine: (['pkexec', 'install', version], ['spacr']))
    seen = []

    def helper(records, workdir, plan_path, machine):
        seen.append(plan_path)
        return ['helper', 'run-plan', plan_path], {'MARK': '1'}, None

    monkeypatch.setattr(cleanup, '_standalone_helper_command', helper)
    spawned = []
    plan = cleanup.start_update_helper([_deb_record(tmp_path)], '1.5.1.5', pid=4,
                                       workdir=str(tmp_path / 'work'), system=_linux_machine(tmp_path),
                                       spawn=lambda *a: spawned.append(a))
    assert plan['adapter'] == 'linux-deb-v1' and plan['error'] is None
    assert plan['steps'][0] == 'prepare' and plan['install'] == ['pkexec', 'install', '1.5.1.5']
    assert plan['relaunch'] == ['spacr'] and len(plan['handshake']['token']) == 64
    assert spawned == [(plan['command'], {'MARK': '1'}, str(tmp_path / 'work'))]
    assert seen == [str(tmp_path / 'work' / 'plan.json')]


def test_frozen_debian_plan_records_an_unsupported_package(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup.sys, 'frozen', True, raising=False)

    def refuse(records, version, machine):
        raise ValueError('modeled unsupported package')

    monkeypatch.setattr(cleanup, '_debian_update_commands', refuse)
    plan = cleanup.start_update_helper([_deb_record(tmp_path)], '1.5.1.5', pid=4,
                                       workdir=str(tmp_path / 'work'), system=_linux_machine(tmp_path),
                                       spawn=lambda *a: pytest.fail('no helper launch'))
    assert plan['adapter'] == 'linux-deb-v1'
    assert plan['error'] == 'modeled unsupported package' and plan['command'] is None


def test_frozen_macos_bundle_plan_uses_the_standalone_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(cleanup, '_standalone_helper_command',
                        lambda records, workdir, plan_path, machine: (None, None, 'modeled no helper'))
    record = cleanup.InstallRecord(kind='installer', layout='macos-app', platform='macos',
                                   root=str(tmp_path / 'Applications' / 'spaCR.app'), running=True)
    plan = cleanup.start_update_helper([record], '1.5.1.5', pid=4, workdir=str(tmp_path / 'work'),
                                       system=_linux_machine(tmp_path, 'macos'),
                                       spawn=lambda *a: pytest.fail('no helper launch'))
    assert plan['adapter'] == 'macos-frozen-v1' and plan['handshake']['schema'] == 1
    assert plan['command'] is None and plan['error'] == 'modeled no helper'


def test_frozen_family_on_the_wrong_platform_requires_its_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup.sys, 'frozen', False, raising=False)
    record = cleanup.InstallRecord(kind='installer', layout='windows-offline', platform='windows',
                                   root=str(tmp_path / 'spaCR'), running=True)
    plan = cleanup.start_update_helper([record], '1.5.1.5', pid=4, workdir=str(tmp_path / 'work'),
                                       system=_linux_machine(tmp_path),
                                       spawn=lambda *a: pytest.fail('no helper launch'))
    assert 'verified frozen replacement adapter' in plan['error']
    assert plan['command'] is None and 'adapter' not in plan
