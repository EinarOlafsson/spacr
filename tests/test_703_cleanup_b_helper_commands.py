"""Update-plan adapters, uninstall preparation, installer metadata and helper commands."""
import json
import os
import types
from dataclasses import asdict

import pytest

from spacr import install_cleanup as cleanup


def _plan(tmp_path, **values):
    plan = {'schema': 1, 'version': '1.5.1.5', 'pid': 41, 'records': [], 'ticked': [],
            'log': str(tmp_path / 'logs' / 'update.log'), 'workdir': str(tmp_path),
            'install': ['installer', '--quiet'], 'fetch': None, 'relaunch': None}
    plan.update(values)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan), encoding="utf-8")
    return str(path)


def _log(tmp_path):
    return (tmp_path / 'logs' / 'update.log').read_text(encoding="utf-8")


class _BrokenHandshake:
    def __init__(self, plan):
        raise ValueError('modeled readiness channel unavailable')


def _machine(tmp_path, platform='linux'):
    return cleanup._Machine(platform=platform, environ={'HOME': str(tmp_path)}, fs_root=str(tmp_path))


def test_pip_update_failure_is_logged_when_readiness_cannot_be_published(tmp_path, monkeypatch):
    def fail(plan, machine, **kwargs):
        raise RuntimeError('modeled pip failure')

    monkeypatch.setattr(cleanup, '_run_pip_environment_update', fail)
    monkeypatch.setattr(cleanup, '_FrozenUpdateHandshake', _BrokenHandshake)
    code = cleanup._run_plan(_plan(tmp_path, adapter='pip-private-posix-v1'), system=_machine(tmp_path))
    assert code == 6
    log = _log(tmp_path)
    assert 'modeled pip failure' in log
    assert 'could not be published to the readiness channel' in log


def test_windows_frozen_adapter_success_reports_ready_relaunch(tmp_path, monkeypatch):
    seen = []

    def run(plan, machine, **kwargs):
        seen.append(sorted(kwargs))
        return '1.5.1.5', 'modeled installer output'

    monkeypatch.setattr(cleanup, '_run_windows_frozen_update', run)
    code = cleanup._run_plan(_plan(tmp_path, adapter='windows-frozen-nsis-v1'), system=_machine(tmp_path))
    assert code == 0
    assert seen == [['download', 'run', 'spawn', 'wait']]
    log = _log(tmp_path)
    assert 'modeled installer output' in log
    assert json.dumps({'version': '1.5.1.5', 'relaunch': 'ready'}) in log


def test_windows_frozen_adapter_failure_stops_even_without_a_channel(tmp_path, monkeypatch):
    def run(plan, machine, **kwargs):
        raise ValueError('modeled frozen refusal')

    monkeypatch.setattr(cleanup, '_run_windows_frozen_update', run)
    monkeypatch.setattr(cleanup, '_WindowsPipHandshake', _BrokenHandshake)
    code = cleanup._run_plan(_plan(tmp_path, adapter='windows-frozen-nsis-v1'), system=_machine(tmp_path))
    assert code == 6
    assert 'modeled frozen refusal' in _log(tmp_path)


def test_debian_adapter_success_reports_the_verified_version(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, '_run_debian_update', lambda plan, machine, **kwargs: ('1.5.1.5', 'dpkg ok'))
    code = cleanup._run_plan(_plan(tmp_path, adapter='linux-deb-v1'), system=_machine(tmp_path))
    assert code == 0
    assert 'Verified Debian spaCR 1.5.1.5; updated desktop ready.' in _log(tmp_path)


@pytest.mark.parametrize('handshake', [True, False])
def test_debian_adapter_failure_is_logged_with_or_without_readiness(tmp_path, monkeypatch, handshake):
    def fail(plan, machine, **kwargs):
        raise RuntimeError('modeled dpkg failure')

    monkeypatch.setattr(cleanup, '_run_debian_update', fail)
    monkeypatch.setattr(cleanup, '_FrozenUpdateHandshake', _BrokenHandshake)
    extra = {'handshake': {'schema': 1}} if handshake else {}
    code = cleanup._run_plan(_plan(tmp_path, adapter='linux-deb-v1', **extra), system=_machine(tmp_path))
    assert code == 6
    log = _log(tmp_path)
    assert 'Debian update stopped: modeled dpkg failure' in log
    assert ('The Debian readiness error could not be published.' in log) is handshake


def test_macos_frozen_adapter_refuses_another_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, '_run_macos_frozen_update', lambda plan, **kw: pytest.fail('must not run'))
    monkeypatch.setattr(cleanup, '_FrozenUpdateHandshake', _BrokenHandshake)
    code = cleanup._run_plan(_plan(tmp_path, adapter='macos-frozen-v1', handshake={'schema': 1}),
                             system=_machine(tmp_path))
    assert code == 6
    log = _log(tmp_path)
    assert 'requires its actual native platform' in log
    assert 'shutdown remains unapproved' in log


def test_macos_frozen_adapter_ready_relaunch_succeeds(tmp_path, monkeypatch):
    receipt = {'state': 'committed', 'relaunch': {'status': 'ready'}}
    monkeypatch.setattr(cleanup, '_run_macos_frozen_update', lambda plan, **kw: receipt)
    code = cleanup._run_plan(_plan(tmp_path, adapter='macos-frozen-v1', handshake={'schema': 1}),
                             system=_machine(tmp_path, 'macos'))
    assert code == 0
    assert json.dumps(receipt, sort_keys=True) in _log(tmp_path)


def _installer_record(tmp_path, **values):
    data = dict(kind='installer', layout='linux-online', platform='linux',
                root=str(tmp_path / 'spacr-old'), version='1.5.1.4')
    data.update(values)
    return asdict(cleanup.InstallRecord(**data))


def test_generic_plan_without_a_running_desktop_requests_no_relaunch(tmp_path):
    commands = []
    path = _plan(tmp_path, records=[_installer_record(tmp_path)])
    code = cleanup._run_plan(path, wait=lambda pid: True, run=lambda argv: commands.append(argv) or 0,
                             spawn=lambda *a: pytest.fail('no relaunch'), system=_machine(tmp_path))
    assert code == 7
    assert commands == [['installer', '--quiet']]
    assert 'The desktop interpreter could not be identified' in _log(tmp_path)


def test_generic_plan_with_unreadable_identity_requests_no_relaunch(tmp_path):
    commands = []

    def run(argv):
        commands.append(argv)
        return (0, 'installed') if len(commands) == 1 else (0, 'not json')

    record = _installer_record(tmp_path, running=True, python=str(tmp_path / 'spacr-old' / 'bin' / 'python'))
    code = cleanup._run_plan(_plan(tmp_path, records=[record]), wait=lambda pid: True, run=run,
                             spawn=lambda *a: pytest.fail('no relaunch'), system=_machine(tmp_path))
    assert code == 7
    assert len(commands) == 2 and commands[1][0] == record['python']
    assert 'could not be verified; no relaunch was requested' in _log(tmp_path)


def test_download_into_a_missing_folder_fails_without_a_staged_file(tmp_path):
    target = tmp_path / 'missing' / 'spaCR.exe'
    with pytest.raises(FileNotFoundError):
        cleanup._download(cleanup._RELEASE_DOWNLOAD + '/v1.5.1.5/spaCR.exe', str(target))
    assert not (tmp_path / 'missing').exists()


@pytest.fixture
def windows_install(tmp_path):
    root = os.path.realpath(str(tmp_path / 'spaCR'))
    python = os.path.join(root, 'python')
    os.makedirs(os.path.join(root, 'venv', 'Scripts'))
    os.makedirs(os.path.join(python, 'Lib'))
    executable = os.path.join(root, 'venv', 'Scripts', 'python.exe')
    open(executable, 'wb').close()
    with open(os.path.join(python, 'python.exe'), 'wb') as handle:
        handle.write(b'private python')
    with open(os.path.join(python, 'Lib', 'site.py'), 'w') as handle:
        handle.write('modeled = True\n')
    workdir = os.path.realpath(str(tmp_path / 'work'))
    os.mkdir(workdir)
    os.chmod(workdir, 0o755)
    home = tmp_path / 'home'
    home.mkdir()
    machine = cleanup._Machine(platform='windows', environ={'USERPROFILE': str(home)},
                               fs_root=str(tmp_path), executable=executable,
                               running_prefix=os.path.join(root, 'venv'))
    return types.SimpleNamespace(root=root, python=python, workdir=workdir, machine=machine)


@pytest.mark.posix_only
def test_uninstall_preparation_refuses_a_shared_posix_folder(windows_install):
    with pytest.raises(ValueError, match='private owner folder'):
        cleanup._prepare_windows_uninstall(windows_install.root, windows_install.workdir,
                                           machine=windows_install.machine, base_prefix=windows_install.python)
    assert os.listdir(windows_install.workdir) == []


def test_uninstall_preparation_on_windows_copies_the_private_python(windows_install, monkeypatch):
    proxy = types.SimpleNamespace(**vars(os))
    proxy.name = 'nt'
    monkeypatch.setattr(cleanup, 'os', proxy)
    receipt = cleanup._prepare_windows_uninstall(windows_install.root, windows_install.workdir,
                                                 machine=windows_install.machine,
                                                 base_prefix=windows_install.python)
    runtime = os.path.join(windows_install.workdir, 'runtime')
    assert receipt['command'] == [os.path.join(runtime, 'python.exe'), '-I',
                                  os.path.join(windows_install.workdir, 'install_cleanup.py'), 'remove',
                                  '--only-root', windows_install.root, '--require-outside']
    with open(os.path.join(runtime, 'python.exe'), 'rb') as handle:
        assert handle.read() == b'private python'
    assert os.path.isfile(os.path.join(runtime, 'Lib', 'site.py'))
    assert os.path.isfile(os.path.join(windows_install.root, 'python', 'python.exe'))


def test_metadata_snapshot_refuses_a_directory_entry(tmp_path):
    entry = tmp_path / 'sidecar.d'
    entry.mkdir()
    with pytest.raises(ValueError) as error:
        cleanup._installer_metadata_snapshot(str(tmp_path / 'backup'), [str(entry)])
    assert error.value.args[0]['code'] == 'unsupported_metadata_entry'
    assert entry.is_dir()


@pytest.fixture
def metadata(tmp_path):
    sidecar = tmp_path / 'uninstall.dat'
    sidecar.write_bytes(b'original sidecar')
    link = tmp_path / 'current'
    link.symlink_to('uninstall.dat')
    folder = tmp_path / 'backup'
    state = cleanup._installer_metadata_snapshot(str(folder), [str(sidecar), str(link)])
    return types.SimpleNamespace(folder=folder, sidecar=sidecar, link=link, state=state)


def _restore_code(folder):
    with pytest.raises(ValueError) as error:
        cleanup._installer_metadata_restore(str(folder))
    return error.value.args[0]['code']


@pytest.mark.needs_symlink
def test_metadata_restore_refuses_a_shared_folder(metadata):
    os.chmod(metadata.folder, 0o755)
    assert _restore_code(metadata.folder) == 'untrusted_metadata_folder'


@pytest.mark.needs_symlink
def test_metadata_restore_refuses_a_non_regular_journal(metadata):
    journal = metadata.folder / 'metadata.json'
    journal.unlink()
    os.mkfifo(journal)
    assert _restore_code(metadata.folder) == 'invalid_metadata_journal'


@pytest.mark.needs_symlink
def test_metadata_restore_refuses_another_schema(metadata):
    journal = metadata.folder / 'metadata.json'
    state = json.loads(journal.read_text(encoding="utf-8"))
    state['schema'] = 2
    journal.write_text(json.dumps(state), encoding="utf-8")
    assert _restore_code(metadata.folder) == 'invalid_metadata_binding'


def _record_index(metadata, kind):
    return next(i for i, record in enumerate(metadata.state['records']) if record['kind'] == kind)


@pytest.mark.needs_symlink
def test_metadata_restore_refuses_a_retargeted_link_backup(metadata):
    record = metadata.state['records'][_record_index(metadata, 'link')]
    backup = metadata.folder / record['backup']
    backup.unlink()
    backup.symlink_to('elsewhere')
    assert _restore_code(metadata.folder) == 'changed_metadata_backup'
    assert os.readlink(metadata.link) == 'uninstall.dat'


@pytest.mark.needs_symlink
def test_metadata_restore_refuses_an_unknown_record_kind(metadata):
    journal = metadata.folder / 'metadata.json'
    state = json.loads(journal.read_text(encoding="utf-8"))
    state['records'][_record_index(metadata, 'file')]['kind'] = 'device'
    journal.write_text(json.dumps(state), encoding="utf-8")
    assert _restore_code(metadata.folder) == 'invalid_metadata_kind'
    assert metadata.sidecar.read_bytes() == b'original sidecar'


@pytest.mark.needs_symlink
def test_metadata_restore_puts_back_both_file_and_link(metadata):
    metadata.sidecar.write_bytes(b'changed by installer')
    metadata.link.unlink()
    receipt = cleanup._installer_metadata_restore(str(metadata.folder))
    assert receipt['state'] == 'restored'
    assert metadata.sidecar.read_bytes() == b'original sidecar'
    assert os.readlink(metadata.link) == 'uninstall.dat'


def _stopped(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_apply_macos_root_prints_its_receipt(monkeypatch, capsys):
    seen = []

    def apply(version, digest, image, channel, pid):
        seen.append(cleanup._PRIVILEGED_DEBIAN_HELPER)
        return 'applied ' + version + '\n'

    monkeypatch.setattr(cleanup, '_macos_apply_root_transaction', apply)
    code = cleanup._main(['apply-macos-root', '--version', '1.5.1.5', '--digest', 'd', '--image', 'i',
                          '--phase-channel', '3', '--desktop-pid', '7'])
    assert code == 0 and seen == [True] and cleanup._PRIVILEGED_DEBIAN_HELPER is False
    assert capsys.readouterr().out == 'applied 1.5.1.5\n'


_MACOS_UNINSTALL = ['uninstall-macos-root', '--phase-channel', '3', '--desktop-pid', '7',
                    '--desktop-executable', 'spaCR', '--previous-version', '1.5.1.4',
                    '--app-identity', 'id', '--keep-json', '[]']


def test_uninstall_macos_root_returns_the_transaction_code(monkeypatch):
    seen = []
    monkeypatch.setattr(cleanup, '_uninstall_macos_root', lambda *args: seen.append(args) or 0)
    assert cleanup._main(_MACOS_UNINSTALL) == 0
    assert seen == [('3', 7, 'spaCR', '1.5.1.4', 'id', '[]')]
    assert cleanup._PRIVILEGED_DEBIAN_HELPER is False


def test_uninstall_macos_root_failure_prints_a_stopped_state(monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError('modeled macOS refusal')

    monkeypatch.setattr(cleanup, '_uninstall_macos_root', fail)
    assert cleanup._main(_MACOS_UNINSTALL) == 6
    assert _stopped(capsys) == {'state': 'stopped', 'error': 'modeled macOS refusal'}
    assert cleanup._PRIVILEGED_DEBIAN_HELPER is False


_DEBIAN = ['--version', '1.5.1.5', '--previous-version', '1.5.1.4', '--desktop-uid', '1000']


def test_prepare_debian_root_requires_both_phase_arguments(monkeypatch, capsys):
    monkeypatch.setattr(cleanup, '_debian_prepare_root_transaction', lambda *a, **k: pytest.fail('must refuse'))
    assert cleanup._main(['prepare-debian-root', *_DEBIAN, '--phase-channel', '3']) == 6
    assert 'authorization descriptor is not private' in _stopped(capsys)['error']


def test_prepare_debian_root_without_a_channel_prints_its_output(monkeypatch, capsys):
    seen = []

    def prepare(version, previous, uid, **kwargs):
        seen.append((version, previous, uid, kwargs))
        return 0, 'prepared stage'

    monkeypatch.setattr(cleanup, '_debian_prepare_root_transaction', prepare)
    assert cleanup._main(['prepare-debian-root', *_DEBIAN]) == 0
    assert seen == [('1.5.1.5', '1.5.1.4', 1000, {})]
    assert capsys.readouterr().out == 'prepared stage\n'


def test_apply_debian_root_requires_a_stage(monkeypatch, capsys):
    monkeypatch.setattr(cleanup, '_debian_apply_root_transaction', lambda *a, **k: pytest.fail('must refuse'))
    assert cleanup._main(['apply-debian-root', *_DEBIAN]) == 6
    assert _stopped(capsys)['error'] == 'a private Debian transaction directory is required'


def test_apply_debian_root_prints_the_receipt(monkeypatch, capsys):
    seen = []

    def apply(stage, version, previous, uid, **kwargs):
        seen.append((stage, kwargs))
        return {'state': 'applied', 'version': version}

    monkeypatch.setattr(cleanup, '_debian_apply_root_transaction', apply)
    assert cleanup._main(['apply-debian-root', *_DEBIAN, '--stage', 'stage-dir',
                          '--phase-channel', '3', '--desktop-pid', '9']) == 0
    assert seen == [('stage-dir', {'phase_channel': '3', 'desktop_pid': 9})]
    assert json.loads(capsys.readouterr().out) == {'state': 'applied', 'version': '1.5.1.5'}


_WINDOWS_UNINSTALL = ['uninstall-frozen-windows', '--root', 'C:/spaCR', '--request', 'req.json',
                      '--desktop-pid', '7', '--desktop-executable', 'spacr.exe', '--phase-channel', '3']
_DEBIAN_UNINSTALL = ['uninstall-debian-root', '--previous-version', '1.5.1.4', '--desktop-uid', '1000',
                     '--desktop-pid', '7', '--desktop-executable', 'spacr', '--phase-channel', '3']


@pytest.mark.parametrize('argv,name,expected', [
    (_WINDOWS_UNINSTALL, '_uninstall_frozen_windows', ('req.json', 'C:/spaCR', '3', 7, 'spacr.exe')),
    (_DEBIAN_UNINSTALL, '_uninstall_debian_root', ('3', 7, 'spacr', 1000, '1.5.1.4')),
])
def test_privileged_uninstall_commands_forward_their_arguments(monkeypatch, argv, name, expected):
    seen = []
    monkeypatch.setattr(cleanup, name, lambda *args: seen.append((args, cleanup._PRIVILEGED_DEBIAN_HELPER)) or 0)
    assert cleanup._main(argv) == 0
    assert seen == [(expected, True)]
    assert cleanup._PRIVILEGED_DEBIAN_HELPER is False


@pytest.mark.parametrize('argv,name', [(_WINDOWS_UNINSTALL, '_uninstall_frozen_windows'),
                                       (_DEBIAN_UNINSTALL, '_uninstall_debian_root')])
def test_privileged_uninstall_failures_print_a_stopped_state(monkeypatch, capsys, argv, name):
    def fail(*args):
        raise OSError('modeled uninstall refusal')

    monkeypatch.setattr(cleanup, name, fail)
    assert cleanup._main(argv) == 6
    assert _stopped(capsys) == {'state': 'stopped', 'error': 'modeled uninstall refusal'}
    assert cleanup._PRIVILEGED_DEBIAN_HELPER is False


def test_run_uninstall_plan_returns_the_plan_code(monkeypatch):
    monkeypatch.setattr(cleanup, '_run_uninstall_plan', lambda path: 0 if path == 'plan.json' else 9)
    assert cleanup._main(['run-uninstall-plan', 'plan.json']) == 0


def test_run_uninstall_plan_failure_is_reported(monkeypatch, capsys):
    def fail(path):
        raise ValueError('modeled uninstall plan error')

    monkeypatch.setattr(cleanup, '_run_uninstall_plan', fail)
    assert cleanup._main(['run-uninstall-plan', 'plan.json']) == 1
    assert 'Uninstall stopped: modeled uninstall plan error' in capsys.readouterr().out


_FROZEN_INSTALL = ['install-frozen-windows', '--source', 'payload', '--destination', 'dest', '--version', '1.5.1.5']


def test_install_frozen_windows_with_a_phase_channel_binds_both_phases(monkeypatch, capsys):
    events = []

    class Client:
        def __init__(self, channel, pid, executable):
            events.append(('client', channel, pid, executable))

        def __enter__(self):
            return 'channel'

        def __exit__(self, *exc):
            events.append('closed')
            return False

    def install(source, destination, version, *, before_activate, after_activate, preserve_previous):
        before_activate()
        after_activate()
        return {'ok': True, 'preserved': preserve_previous}

    monkeypatch.setattr(cleanup, '_UpdatePhaseClient', Client)
    monkeypatch.setattr(cleanup, '_install_frozen_windows', install)
    monkeypatch.setattr(cleanup, '_native_update_ready', lambda channel, version, pid: events.append(('ready', channel, version, pid)))
    monkeypatch.setattr(cleanup, '_native_update_activated', lambda channel, version: events.append(('activated', channel, version)))
    assert cleanup._main([*_FROZEN_INSTALL, '--phase-channel', '3', '--desktop-pid', '7']) == 0
    assert events == [('client', '3', 7, os.path.join('dest', 'spacr.exe')),
                      ('ready', 'channel', '1.5.1.5', 7), ('activated', 'channel', '1.5.1.5'), 'closed']
    assert json.loads(capsys.readouterr().out) == {'ok': True, 'preserved': True}


def test_install_frozen_windows_refuses_a_pid_without_a_channel(monkeypatch, capsys):
    monkeypatch.setattr(cleanup, '_install_frozen_windows', lambda *a, **k: pytest.fail('must refuse'))
    assert cleanup._main([*_FROZEN_INSTALL, '--desktop-pid', '7']) == 1
    output = json.loads(capsys.readouterr().out)
    assert output['ok'] is False and 'not private' in output['error']


def test_install_frozen_windows_without_a_channel_installs_directly(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cleanup, '_install_frozen_windows',
                        lambda source, destination, version: seen.append((source, destination, version)) or {'ok': True})
    assert cleanup._main(_FROZEN_INSTALL) == 0
    assert seen == [('payload', 'dest', '1.5.1.5')]
    assert json.loads(capsys.readouterr().out) == {'ok': True}


def test_prepare_windows_uninstall_command_prints_the_receipt(monkeypatch, capsys):
    monkeypatch.setattr(cleanup, '_prepare_windows_uninstall',
                        lambda root, workdir: {'schema': 1, 'root': root, 'workdir': workdir})
    assert cleanup._main(['prepare-windows-uninstall', '--install-root', 'R', '--workdir', 'W']) == 0
    assert json.loads(capsys.readouterr().out) == {'schema': 1, 'root': 'R', 'workdir': 'W'}


def test_prepare_windows_uninstall_command_reports_refusal(monkeypatch, capsys):
    def fail(root, workdir):
        raise ValueError('modeled shared folder')

    monkeypatch.setattr(cleanup, '_prepare_windows_uninstall', fail)
    assert cleanup._main(['prepare-windows-uninstall', '--install-root', 'R', '--workdir', 'W']) == 1
    assert 'Uninstall preparation stopped: modeled shared folder' in capsys.readouterr().out


def test_recover_runtime_command_prints_the_receipt(monkeypatch, capsys):
    monkeypatch.setattr(cleanup, '_recover_runtime_update',
                        lambda transaction, pid: {'state': 'restored', 'pid': pid, 'transaction': transaction})
    assert cleanup._main(['recover-runtime', 'tx', '--pid', '12']) == 0
    assert json.loads(capsys.readouterr().out) == {'state': 'restored', 'pid': 12, 'transaction': 'tx'}


def test_verify_runtime_refuses_an_unverified_retiring_runtime(monkeypatch, capsys, tmp_path):
    def fail(prefix):
        raise ValueError('modeled unreadable runtime')

    monkeypatch.setattr(cleanup, '_user_data', lambda machine: [])
    monkeypatch.setattr(cleanup, '_retiring_runtime_identity', fail)
    monkeypatch.setattr(cleanup, '_run', lambda *a, **k: pytest.fail('must not verify'))
    assert cleanup._main(['verify-runtime', '--python', 'python', '--retiring-prefix', str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        'verified': False, 'reason': 'previous_runtime_identity_unverified'}


def test_targeted_removal_requires_one_recognized_installation(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cleanup, 'find_old_installs', lambda system: [])
    monkeypatch.setattr(cleanup, 'run_update_sequence', lambda *a, **k: pytest.fail('must not remove'))
    target = os.path.realpath(str(tmp_path / 'install'))
    code = cleanup._main(['remove', '--only-root', target, '--require-outside', '--root', str(tmp_path)])
    assert code == 1
    assert 'requires one recognized canonical installation' in capsys.readouterr().out


def test_remove_with_purge_runs_the_purge_after_removal(monkeypatch, tmp_path):
    purged = []
    monkeypatch.setattr(cleanup, 'find_old_installs', lambda system: [])
    monkeypatch.setattr(cleanup, 'run_update_sequence', lambda *a, **k: ([], 0))
    monkeypatch.setattr(cleanup, '_purge_command', lambda machine, yes: purged.append(yes) or 0)
    assert cleanup._main(['remove', '--purge', '--yes', '--root', str(tmp_path)]) == 0
    assert purged == [True]
