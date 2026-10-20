"""Frozen Windows updates coordinate verified NSIS transactions without elevation."""
import hashlib
import json
from pathlib import Path
import types

import pytest

from spacr import install_cleanup as cleanup


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    root = tmp_path / 'Program Files' / 'spaCR'
    internal = root / '_internal'
    metadata = internal / 'spacr-1.5.1.4.dist-info' / 'METADATA'
    metadata.parent.mkdir(parents=True)
    metadata.write_text('Name: spacr\nVersion: 1.5.1.4\n', encoding="utf-8")
    executable = root / 'spacr.exe'
    executable.write_bytes(b'old executable')
    (internal / 'spacr-update-helper.exe').write_bytes(b'modeled independent helper')
    (root / 'user-data').mkdir()
    (root / 'user-data' / 'keep.csv').write_bytes(b'unchanged user data')
    helper = tmp_path / 'private helper'
    helper.mkdir(mode=0o700)
    acl = {'owner': 'S-1-5-21-123', 'user': 'S-1-5-21-123', 'has_dacl': True,
           'rules': [{'sid': 'S-1-5-21-123', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0}]}
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda _path: acl)
    monkeypatch.setattr(cleanup, '_windows_process_elevated', lambda: False, raising=False)
    monkeypatch.setattr(cleanup.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(cleanup.sys, '_MEIPASS', str(internal), raising=False)
    record = cleanup.InstallRecord(kind='installer', layout='windows-offline', platform='windows',
                                   root=str(root), version='1.5.1.4', running=True,
                                   launchers=(str(executable),), needs_admin=True)
    machine = cleanup._Machine(platform='windows', environ={'USERPROFILE': str(home)},
                               fs_root=str(tmp_path), executable=str(executable))
    return types.SimpleNamespace(root=root, internal=internal, helper=helper, home=home,
                                 executable=executable, record=record, machine=machine)


def test_frozen_windows_plan_uses_the_outside_helper_and_native_handshake(runtime):
    spawned = []
    plan = cleanup.start_update_helper([runtime.record], '1.5.1.5', pid=1234,
        workdir=str(runtime.helper), system=runtime.machine,
        spawn=lambda *args: spawned.append(args))
    assert plan.get('error') is None
    assert plan['adapter'] == 'windows-frozen-nsis-v1'
    assert plan['command'][0] == str(runtime.helper / 'spacr-update-helper.exe')
    assert plan['command'][1:] == ['run-plan', str(runtime.helper / 'plan.json')]
    assert cleanup._WindowsPipHandshake(plan).status() is None
    assert len(spawned) == 1
    assert (runtime.root / 'user-data' / 'keep.csv').read_bytes() == b'unchanged user data'


def plan_for(runtime):
    name = 'spaCR-1.5.1.5-update.exe'
    target = runtime.helper / name
    return {'schema': 1, 'adapter': 'windows-frozen-nsis-v1', 'version': '1.5.1.5',
            'records': [cleanup.asdict(runtime.record)], 'pid': 1234, 'ticked': [],
            'workdir': str(runtime.helper), 'frozen_application': True,
            'handshake': {'schema': 1, 'token': 'ab' * 32, 'expires': cleanup.time.time() + 300},
            'fetch': {'url': cleanup._RELEASE_DOWNLOAD + '/v1.5.1.5/' + name, 'path': str(target)},
            'install': [str(target), '/S', '/D=' + str(runtime.root)],
            'relaunch': [str(runtime.executable)]}


def payload(root, version):
    root.mkdir()
    (root / 'spacr.exe').write_bytes(('new executable ' + version).encode())
    metadata = root / '_internal' / ('spacr-' + version + '.dist-info') / 'METADATA'
    metadata.parent.mkdir(parents=True)
    metadata.write_text('Name: spacr\nVersion: ' + version + '\n', encoding="utf-8")
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()}
    (root / 'spacr-payload.json').write_text(json.dumps({'schema': 1, 'version': version, 'files': files}), encoding="utf-8")
    return root


def identity(argv):
    root = Path(argv[0]).parent
    version = cleanup._frozen_windows_previous_version(root)
    Path(argv[-1]).write_text(json.dumps({'schema': 1, 'version': version, 'frozen': True,
        'executable': argv[0], 'bundle_root': str(root), 'import_root': str(root / '_internal'),
        'module': str(root / '_internal' / 'spacr' / '__init__.py')}), encoding="utf-8")
    return 0, ''


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def update(runtime, monkeypatch, tmp_path):
    plan = plan_for(runtime)
    incoming = payload(tmp_path / 'new-payload', '1.5.1.5')
    installer_bytes = b'MZ modeled checksum-bound NSIS installer'
    expected = hashlib.sha256(installer_bytes).hexdigest()
    monkeypatch.setattr(cleanup, '_published_digest', lambda _url: expected)
    actions = []
    monkeypatch.setattr(cleanup, '_desktop_process_identity', lambda _pid: (cleanup.os.getuid(), str(runtime.executable)))
    monkeypatch.setattr(cleanup, '_wait_for_exit', lambda _pid, **_kwargs: 'approved-exit' in actions)
    handshake = cleanup._WindowsPipHandshake(plan)
    real_ready = cleanup._WindowsPipHandshake.ready

    def ready_with_gui_approval(channel):
        real_ready(channel)
        handshake.approve()

    monkeypatch.setattr(cleanup._WindowsPipHandshake, 'ready', ready_with_gui_approval)

    def download(url, path):
        assert url == plan['fetch']['url']
        Path(path).write_bytes(installer_bytes)
        actions.append('download')

    def wait(pid):
        assert pid == 1234
        assert handshake.status()['state'] == 'ready'
        handshake.approve()
        actions.append('approved-exit')
        return True

    def install(installer, destination, *, phase_channel, desktop_pid):
        assert Path(installer).read_bytes() == installer_bytes
        assert Path(destination) == runtime.root
        assert handshake.status() is None
        actions.append('stage')
        with cleanup._UpdatePhaseClient(phase_channel, desktop_pid, str(runtime.executable)) as channel:
            def prepared():
                assert cleanup._frozen_windows_previous_version(runtime.root) == '1.5.1.4'
                cleanup._native_update_ready(channel, '1.5.1.5', desktop_pid)
            def activated():
                actions.append('install')
                if getattr(runtime, 'corrupt_before_launch', False):
                    runtime.executable.write_bytes(b'corrupt activated executable')
                cleanup._native_update_activated(channel, '1.5.1.5')
            cleanup._install_frozen_windows(incoming, destination, '1.5.1.5', run=identity,
                before_activate=prepared, after_activate=activated, preserve_previous=True)
        return 0, 'modeled NSIS completed'

    def spawn(argv, env, cwd):
        assert argv == [str(runtime.executable)]
        assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1" and cwd == str(runtime.helper)
        assert cleanup._frozen_windows_previous_version(runtime.root) == '1.5.1.5'
        folder, binding = json.loads(env["SPACR_UPDATE_LAUNCH"])
        channel = cleanup._UpdateLaunchHandshake(folder, binding)
        assert channel._read("launch-request.json")["details"] == {
            "version": "1.5.1.5", "executable": argv[0], "root": str(runtime.root), "frozen": True}
        channel._write("launch-result.json", "ready", details={
            "version": "1.5.1.5", "executable": argv[0], "pid": 100,
            "module": str(runtime.internal / "spacr/__init__.py")})
        actions.append('relaunch')
        return types.SimpleNamespace(pid=100, poll=lambda: None)

    return types.SimpleNamespace(plan=plan, incoming=incoming, expected=expected,
        download=download, install=install, wait=wait, spawn=spawn,
        actions=actions, handshake=handshake, runtime=runtime)


def run_update(update, **kwargs):
    return cleanup._run_windows_frozen_update(update.plan, update.runtime.machine,
        download=kwargs.get('download', update.download), install=kwargs.get('install', update.install),
        wait=kwargs.get('wait', update.wait), spawn=kwargs.get('spawn', update.spawn),
        run=kwargs.get('run', identity))


@pytest.mark.posix_only
def test_verified_nsis_update_keeps_data_and_relaunches_after_actual_identity(update):
    assert run_update(update) == ('1.5.1.5', 'modeled NSIS completed')
    assert update.actions == ['download', 'stage', 'approved-exit', 'install', 'relaunch']
    assert (update.runtime.root / 'user-data/keep.csv').read_bytes() == b'unchanged user data'
    assert not list((update.runtime.home / '.spacr/update-locks').iterdir())
    results = list(update.runtime.helper.glob('windows-update-result-*.json'))
    assert len(results) == 1
    assert json.loads(results[0].read_text(encoding="utf-8"))['state'] == 'desktop-ready'


@pytest.mark.posix_only
@pytest.mark.parametrize("failure", ["old-version", "exited", "never-ready", "cannot-stop"])
def test_unconfirmed_windows_desktop_retains_recovery_and_cannot_report_success(update, monkeypatch, failure):
    stopped = []

    def spawn(argv, env, cwd):
        process = update.spawn(argv, env, cwd)
        folder, binding = json.loads(env["SPACR_UPDATE_LAUNCH"])
        channel = cleanup._UpdateLaunchHandshake(folder, binding)
        message = channel._read("launch-result.json")
        if failure == "old-version":
            message["details"]["version"] = "1.5.1.4"
            channel._write("launch-result.json", "ready", details=message["details"])
        elif failure == "exited":
            process.poll = lambda: 9
        else:
            channel._write("launch-result.json", "starting", details=message["details"])
            monkeypatch.setattr(cleanup.time, "time", lambda: channel.expires + 1)

        def stop():
            if failure == "cannot-stop":
                raise OSError("modeled child handle cannot be stopped")
            stopped.append(True)
            process.poll = lambda: -15

        process.terminate = stop
        process.wait = lambda timeout: -15
        return process

    with pytest.raises(RuntimeError):
        run_update(update, spawn=spawn)
    assert bool(stopped) is (failure in {"old-version", "never-ready"})
    assert not list(update.runtime.helper.glob("windows-update-result-*.json"))
    journals = list((update.runtime.home / ".spacr/update-locks").glob("*/journal.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["state"] == "desktop-unconfirmed"
    assert journal["pid"] == (100 if failure == "cannot-stop" else None)
    assert (update.runtime.root / "user-data/keep.csv").read_bytes() == b"unchanged user data"


@pytest.mark.parametrize('expected', [None, '', 'not-a-checksum', 'a' * 64])
def test_missing_or_corrupt_release_checksum_cannot_request_shutdown_or_install(update, monkeypatch, expected):
    before = snapshot(update.runtime.root)
    monkeypatch.setattr(cleanup, '_published_digest', lambda _url: expected)
    with pytest.raises(ValueError, match='checksum'):
        run_update(update, wait=lambda _pid: pytest.fail('must not approve'),
                   install=lambda *_: pytest.fail('must not install'))
    assert snapshot(update.runtime.root) == before
    assert update.handshake.status() is None


def test_user_cancellation_before_approval_keeps_old_app(update):
    before = snapshot(update.runtime.root)
    update.handshake.cancel()
    with pytest.raises(RuntimeError, match='cancelled'):
        run_update(update)
    assert update.actions == []
    assert snapshot(update.runtime.root) == before


def test_cancelled_uac_never_relaunches_or_changes_old_runtime(update):
    before = snapshot(update.runtime.root)

    def cancel(*_args, **_kwargs):
        raise OSError(1223, 'modeled UAC declined')

    with pytest.raises(OSError) as error:
        run_update(update, install=cancel)
    assert error.value.errno == 1223
    assert update.actions == ['download']
    assert snapshot(update.runtime.root) == before
    assert not list((update.runtime.home / '.spacr/update-locks').iterdir())


@pytest.mark.posix_only
def test_failed_nsis_transaction_restores_old_bytes_and_never_relaunches(update):
    before = snapshot(update.runtime.root)

    def failed(installer, root, *, phase_channel, desktop_pid):
        def refuse_new(argv):
            if Path(argv[0]).parent == Path(root):
                return 1, 'modeled actual activated import failure'
            return identity(argv)
        with cleanup._UpdatePhaseClient(phase_channel, desktop_pid, str(update.runtime.executable)) as channel:
            with pytest.raises(ValueError):
                cleanup._install_frozen_windows(update.incoming, root, '1.5.1.5', run=refuse_new,
                    before_activate=lambda: cleanup._native_update_ready(channel, '1.5.1.5', desktop_pid),
                    preserve_previous=True)
        return 1, 'NSIS transaction failed'

    with pytest.raises(RuntimeError):
        run_update(update, install=failed)
    assert snapshot(update.runtime.root) == before
    assert 'relaunch' not in update.actions
    locks = list((update.runtime.home / '.spacr/update-locks').iterdir())
    assert len(locks) == 1
    assert json.loads((locks[0] / 'journal.json').read_text(encoding="utf-8"))['state'] == 'desktop-unconfirmed'


def test_live_installer_timeout_retains_lock_stage_and_journal(update):
    before = snapshot(update.runtime.root)

    def timeout(*_args, **_kwargs):
        raise TimeoutError('modeled installer still running')

    with pytest.raises(TimeoutError):
        run_update(update, install=timeout)
    assert snapshot(update.runtime.root) == before
    assert Path(update.plan['fetch']['path']).is_file()
    locks = list((update.runtime.home / '.spacr/update-locks').iterdir())
    assert len(locks) == 1
    assert json.loads((locks[0] / 'journal.json').read_text(encoding="utf-8"))['state'] == 'installer-uncertain'
    assert 'relaunch' not in update.actions
    with pytest.raises(FileExistsError):
        run_update(update)


@pytest.mark.posix_only
@pytest.mark.parametrize('change', ['root', 'executable', 'version', 'stage'])
def test_changed_destination_or_staged_installer_refused_after_approved_exit(update, change):
    def mutate(pid):
        update.wait(pid)
        root = update.runtime.root
        if change == 'root':
            root.rename(root.with_name('retained-old'))
            root.mkdir()
        elif change == 'executable':
            update.runtime.executable.write_bytes(b'foreign new executable')
        elif change == 'version':
            metadata = next(update.runtime.internal.glob('*.dist-info/METADATA'))
            metadata.write_text('Name: spacr\nVersion: 1.5.1.6\n', encoding="utf-8")
        else:
            Path(update.plan['fetch']['path']).write_bytes(b'replaced after approval')
        return True

    with pytest.raises((ValueError, FileNotFoundError)):
        run_update(update, wait=mutate)
    assert 'install' not in update.actions and 'relaunch' not in update.actions


@pytest.mark.posix_only
@pytest.mark.parametrize('change', ['version', 'runtime', 'receipt'])
def test_wrong_installed_runtime_or_receipt_is_not_relaunched(update, change):
    def wrong(installer, root, **kwargs):
        if change == 'version':
            return 0, 'dishonest installer exit'
        update.runtime.corrupt_before_launch = change == 'runtime'
        update.install(installer, root, **kwargs)
        return 0, ''

    def wrong_identity(argv):
        code, output = identity(argv)
        path = Path(argv[-1])
        result = json.loads(path.read_text(encoding="utf-8"))
        result['frozen'] = False
        path.write_text(json.dumps(result), encoding="utf-8")
        return code, output

    with pytest.raises((ValueError, OSError, RuntimeError)):
        run_update(update, install=wrong, run=wrong_identity if change == 'receipt' else identity)
    assert 'relaunch' not in update.actions


def test_already_elevated_coordinator_cannot_prepare_or_run(runtime, update, monkeypatch):
    before = snapshot(runtime.root)
    monkeypatch.setattr(cleanup, '_windows_process_elevated', lambda: True)
    plan = cleanup.start_update_helper([runtime.record], '1.5.1.5', workdir=str(runtime.helper),
        system=runtime.machine, spawn=lambda *_: pytest.fail('no helper launch'))
    assert plan['command'] is None and plan['error']
    with pytest.raises(ValueError, match='coordinator'):
        run_update(update)
    assert snapshot(runtime.root) == before


def test_downgrade_is_refused_before_helper_launch(runtime):
    before = snapshot(runtime.root)
    plan = cleanup.start_update_helper([runtime.record], '1.5.1.3', workdir=str(runtime.helper),
        system=runtime.machine, spawn=lambda *_: pytest.fail('no helper launch'))
    assert plan['command'] is None and plan['error']
    assert snapshot(runtime.root) == before


@pytest.mark.needs_symlink
@pytest.mark.parametrize('entry', ['installer', 'root', 'internal'])
def test_reparse_and_nonregular_boundaries_are_refused(update, tmp_path, entry):
    if entry == 'installer':
        def bad_download(_url, path):
            Path(path).symlink_to(update.runtime.executable)
        with pytest.raises(ValueError):
            run_update(update, download=bad_download)
    else:
        path = update.runtime.root if entry == 'root' else update.runtime.internal
        retained = tmp_path / ('retained-' + entry)
        path.rename(retained)
        path.symlink_to(retained, target_is_directory=True)
        with pytest.raises(ValueError):
            run_update(update)
    assert 'install' not in update.actions and 'relaunch' not in update.actions


@pytest.mark.posix_only
@pytest.mark.parametrize('variable', ['SPACR_HOME', 'SPACR_MODEL_CACHE', 'SPACR_NEWS_CACHE'])
@pytest.mark.parametrize('late', [False, True])
def test_configured_data_inside_payload_refuses_before_staging_or_after_approval(update, variable, late):
    location = update.runtime.internal / 'retained user files'
    location.mkdir()
    (location / 'keep.bin').write_bytes(b'configured user data')
    before = snapshot(update.runtime.root)
    if not late:
        update.runtime.machine.environ[variable] = str(location)
        wait = update.wait
    else:
        def wait(pid):
            update.wait(pid)
            update.runtime.machine.environ[variable] = str(location)
            return True
    with pytest.raises(ValueError, match='user_data_overlap'):
        run_update(update, wait=wait)
    assert snapshot(update.runtime.root) == before
    assert 'install' not in update.actions and 'relaunch' not in update.actions
    if not late:
        assert update.actions == []


@pytest.mark.needs_symlink
def test_configured_cache_alias_into_payload_is_refused(update):
    location = update.runtime.internal / 'model-cache'
    location.mkdir()
    alias = update.runtime.home / 'model-alias'
    alias.symlink_to(location, target_is_directory=True)
    update.runtime.machine.environ['SPACR_MODEL_CACHE'] = str(alias)
    with pytest.raises(ValueError, match='user_data_overlap'):
        run_update(update)
    assert alias.is_symlink() and location.is_dir()
    assert update.actions == []


def test_changed_data_alias_before_readiness_never_approves_shutdown(update):
    location = update.runtime.internal / 'model-cache'
    location.mkdir()

    def download(url, path):
        update.download(url, path)
        update.runtime.machine.environ['SPACR_MODEL_CACHE'] = str(location)

    with pytest.raises(ValueError, match='user_data_overlap'):
        run_update(update, download=download)
    assert update.handshake.status() is None
    assert update.actions == ['download']


def test_native_installer_pin_denies_write_delete_and_keeps_descriptor_live(update, monkeypatch):
    import os
    import sys

    path = Path(update.plan['fetch']['path'])
    update.download(update.plan['fetch']['url'], str(path))
    calls = []
    actual_open = os.open
    opened = []

    class NativeFunction:
        def __init__(self, fn):
            self.fn = fn
        def __call__(self, *args):
            return self.fn(*args)

    def create(name, access, sharing, security, disposition, flags, template):
        calls.append((name, access, sharing, disposition, flags))
        descriptor = actual_open(name, os.O_RDONLY)
        opened.append(descriptor)
        return descriptor

    kernel = types.SimpleNamespace(CreateFileW=NativeFunction(create),
                                    CloseHandle=NativeFunction(lambda fd: os.close(fd)))
    monkeypatch.setattr(cleanup, '_windows_update_library', lambda name: kernel)
    proxy = types.SimpleNamespace(**vars(os))
    proxy.name = 'nt'
    proxy.O_BINARY = 0
    monkeypatch.setattr(cleanup, 'os', proxy)
    monkeypatch.setitem(sys.modules, 'msvcrt', types.SimpleNamespace(open_osfhandle=lambda handle, flags: handle))
    monkeypatch.setattr(os, 'O_BINARY', 0, raising=False)
    with cleanup._windows_frozen_installer_pin(str(path), update.expected) as stream:
        assert os.fstat(opened[0]).st_ino == path.stat().st_ino
        assert not stream.closed
        assert calls == [(str(path), 0x80000000, 1, 3, 0x08200000)]
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.posix_only
def test_lost_approval_or_expired_lease_never_activates_staged_runtime(update, monkeypatch):
    original_wait = cleanup._WindowsPipHandshake.wait_for_shutdown

    def wait_then_cancel(channel, waiter=None):
        original_wait(channel, waiter)
        channel.cancel()

    monkeypatch.setattr(cleanup._WindowsPipHandshake, 'wait_for_shutdown', wait_then_cancel)
    with pytest.raises(RuntimeError, match='cancelled'):
        run_update(update)
    assert 'install' not in update.actions and 'relaunch' not in update.actions


def test_cancellation_during_final_journal_publish_never_elevates(update, monkeypatch):
    original = cleanup._windows_frozen_update_journal

    def journal(directory, plan, state, **details):
        original(directory, plan, state, **details)
        if state == 'preparing':
            update.handshake.cancel()

    monkeypatch.setattr(cleanup, '_windows_frozen_update_journal', journal)
    with pytest.raises(RuntimeError, match='cancelled'):
        run_update(update, install=lambda *_: pytest.fail('cancelled plan must not elevate'))
    assert 'install' not in update.actions and 'relaunch' not in update.actions
