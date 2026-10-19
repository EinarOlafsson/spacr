"""The Windows online installation updates in place through its own uv, like the macOS online layout.

Found natively on Windows 11 2026-10-19: start_update_helper refused every root made by
install_spacr_windows.ps1 ("This installation cannot stage a verified update"), so users of the
recommended Windows installer could not update from inside spaCR. windows-online-uv-v1 stages a copy
of <root>\\venv with the installation's bundled uv and saved choices, verifies it, and after approved
shutdown swaps it in under the runtime lease, keeping the previous runtime for rollback; relaunch must
show the new main window. Older committed recovery copies are pruned, keeping the newest.

The native ACL probe is modeled, so these run on any system.
"""
import json
import os
from pathlib import Path
import types

import pytest

from spacr import install_cleanup as cleanup


USER = 'S-1-5-21-123-456-789-1001'


def acl():
    return {'owner': USER, 'user': USER, 'has_dacl': True,
            'rules': [{'sid': USER, 'type': 0, 'rights': 0x1f01ff, 'propagation': 0},
                      {'sid': 'S-1-5-18', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0}]}


def snapshot(prefix):
    return {path.relative_to(prefix).as_posix(): path.read_bytes()
            for path in prefix.rglob('*') if path.is_file()}


@pytest.fixture
def online(tmp_path, monkeypatch):
    root = tmp_path / 'home' / 'AppData' / 'Local' / 'spaCR'
    venv = root / 'venv'
    scripts = venv / 'Scripts'
    scripts.mkdir(parents=True)
    for name in ('python.exe', 'pythonw.exe', 'spacr.exe'):
        (scripts / name).write_bytes(b'MZ relocatable trampoline ..\\python.exe')
    base = root / 'python' / 'cpython-3.12.11-windows-x86_64-none'
    base.mkdir(parents=True)
    (base / 'python.exe').write_bytes(b'MZ uv-managed interpreter')
    (base / 'Lib').mkdir()
    (venv / 'pyvenv.cfg').write_text('home = ' + str(base) + '\ninclude-system-site-packages = false\n'
                                     'relocatable = true\n')
    package = venv / 'Lib' / 'site-packages' / 'spacr'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('__version__ = "1.5.1.5"\n')
    (root / 'bootstrap').mkdir()
    (root / 'bootstrap' / 'uv.exe').write_bytes(b'MZ pinned uv')
    (root / 'cache').mkdir()
    record = cleanup.InstallRecord(kind='installer', layout='windows-online', platform='windows',
                                   root=str(root), python=str(scripts / 'python.exe'), version='1.5.1.5',
                                   running=True)
    machine = cleanup._Machine(platform='windows', environ={'USERPROFILE': str(tmp_path / 'home')},
                               fs_root=str(tmp_path), running_prefix=str(venv),
                               executable=str(scripts / 'pythonw.exe'))
    monkeypatch.setattr(cleanup.sys, 'prefix', str(venv))
    monkeypatch.setattr(cleanup.sys, 'base_prefix', str(base))
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda _path: acl())
    monkeypatch.setattr(cleanup, '_retiring_runtime_identity', lambda _path: {'verified': True, 'version': '1.5.1.5'})
    return types.SimpleNamespace(root=root, venv=venv, scripts=scripts, base=base, record=record,
                                 machine=machine, helper=tmp_path / 'outside helper')


def commands(online, version='1.5.1.6'):
    return cleanup._windows_online_commands(online.record, version, str(online.helper),
                                            online.machine.running_prefix, online.machine.executable)


def test_an_online_root_stages_with_its_own_uv_and_detaches_on_the_base_python(online, monkeypatch):
    before = snapshot(online.venv)
    probes = []

    def probe(argv, **_kwargs):
        probes.append(argv)
        return 0, json.dumps({'python': argv[0], 'prefix': str(online.base), 'base_prefix': str(online.base),
                              'stdlib': str(online.base / 'Lib'), 'os_name': 'nt'})

    monkeypatch.setattr(cleanup, '_run', probe)
    spawned = []
    plan = cleanup.start_update_helper([online.record], '1.5.1.6', pid=4242, workdir=str(online.helper),
                                       system=online.machine, spawn=lambda *args: spawned.append(args))
    assert plan['error'] is None, plan['error']
    assert plan['adapter'] == 'windows-online-uv-v1'
    assert plan['command'][0] == str(online.base / 'python.exe') and plan['command'][1:3] == ['-I', '-B']
    assert plan['desktop'] == {'prefix': str(online.venv), 'executable': str(online.scripts / 'pythonw.exe')}
    uv, python = str(online.root / 'bootstrap' / 'uv.exe'), str(online.scripts / 'python.exe')
    assert plan['install'][:8] == [uv, 'pip', 'install', '--upgrade', '--cache-dir', str(online.root / 'cache'),
                                   '--python', python]
    assert plan['install'][-3:] == ['spacr==1.5.1.6', 'numba>=0.60,<1.0', 'llvmlite>=0.43,<1.0']
    assert plan['relaunch'] == [str(online.scripts / 'pythonw.exe'), '-I', '-m', 'spacr.qt']
    assert plan['fetch'] is None and len(spawned) == 1 and len(probes) == 1
    assert snapshot(online.venv) == before


def test_an_owner_only_mkdtemp_acl_is_private(monkeypatch):
    """Python 3.12.4+ gives mkdtemp folders SYSTEM, Administrators and OWNER RIGHTS; found natively 2026-10-19."""
    evidence = {'owner': USER, 'user': USER, 'has_dacl': True,
                'rules': [{'sid': 'S-1-5-18', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0},
                          {'sid': 'S-1-5-32-544', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0},
                          {'sid': 'S-1-3-4', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0}]}
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda _path: evidence)
    cleanup._windows_pip_private_access('updater folder')
    evidence['rules'].append({'sid': 'S-1-5-11', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0})
    with pytest.raises(ValueError):
        cleanup._windows_pip_private_access('updater folder')


@pytest.mark.parametrize('without_wgpu', [False, True])
def test_the_saved_backend_and_wgpu_choice_are_kept(online, without_wgpu):
    from spacr import install_profile

    install_profile.write_profile(online.root / 'install-profile.json', 'cpu', 'none', without_wgpu=without_wgpu)
    online.helper.mkdir()
    install, _relaunch = commands(online)
    assert install[install.index('--torch-backend') + 1] == 'cpu'
    assert ('--override' in install) is without_wgpu
    if without_wgpu:
        override = Path(install[install.index('--override') + 1])
        assert override.parent == online.helper and 'never' in override.read_text(encoding="utf-8")


def test_the_detached_helper_reads_saved_choices_without_importing_spacr(online, monkeypatch):
    """Found natively 2026-10-19: the helper runs on the base Python, where spacr is not importable."""
    import importlib
    import shutil

    from spacr import install_profile

    shutil.copy(install_profile.__file__, online.venv / 'Lib' / 'site-packages' / 'spacr' / 'install_profile.py')
    install_profile.write_profile(online.root / 'install-profile.json', 'cpu', 'none', without_wgpu=True)
    real = importlib.import_module

    def without_spacr(name, *args, **kwargs):
        if name.startswith('spacr'):
            raise ModuleNotFoundError("No module named 'spacr'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, 'import_module', without_spacr)
    online.helper.mkdir()
    install, _relaunch = commands(online)
    assert install[install.index('--torch-backend') + 1] == 'cpu' and '--override' in install
    (online.venv / 'Lib' / 'site-packages' / 'spacr' / 'install_profile.py').unlink()
    with pytest.raises(ValueError, match='saved installation choices'):
        commands(online)


def test_unreadable_saved_choices_refuse(online):
    (online.root / 'install-profile.json').write_text('{"schema": 1}')
    with pytest.raises(ValueError, match='saved installation choices'):
        commands(online)


@pytest.mark.parametrize('defect', ['layout', 'not_running', 'other_desktop', 'console_elsewhere', 'missing_uv',
                                    'helper_inside', 'record_python'])
def test_an_unproven_online_root_refuses_before_any_change(online, defect):
    before = snapshot(online.root)
    desktop_prefix, desktop_executable = online.machine.running_prefix, online.machine.executable
    record = online.record
    helper = str(online.helper)
    if defect == 'layout':
        record = cleanup.replace(record, layout='windows-unknown')
    elif defect == 'not_running':
        record = cleanup.replace(record, running=False)
    elif defect == 'other_desktop':
        desktop_prefix = str(online.base)
    elif defect == 'console_elsewhere':
        desktop_executable = str(online.base / 'python.exe')
    elif defect == 'missing_uv':
        (online.root / 'bootstrap' / 'uv.exe').unlink()
    elif defect == 'helper_inside':
        helper = str(online.root / 'updater')
    else:
        record = cleanup.replace(record, python=str(online.base / 'python.exe'))
    modified = snapshot(online.root)
    with pytest.raises(ValueError):
        cleanup._windows_online_commands(record, '1.5.1.6', helper, desktop_prefix, desktop_executable)
    assert snapshot(online.root) == modified
    assert all(modified.get(key) == value for key, value in before.items() if key in modified)


def test_an_unknown_windows_layout_still_refuses(online, monkeypatch):
    monkeypatch.setattr(cleanup, '_run', lambda *a, **kw: (0, '{}'))
    plan = cleanup.start_update_helper([cleanup.replace(online.record, layout='windows-unknown')], '1.5.1.6',
                                       workdir=str(online.helper), system=online.machine, spawn=lambda *a: None)
    assert plan.get('adapter') != 'windows-online-uv-v1'
    assert plan['command'] is None and plan['install'] is None


class UvRunner:
    def __init__(self, online, failure=None):
        self.online = online
        self.failure = failure
        self.commands = []

    def __call__(self, argv):
        self.commands.append(argv)
        if argv[1:4] == ['pip', 'install', '--upgrade']:
            target = Path(argv[argv.index('--python') + 1]).parent.parent
            assert target.name == 'staged'
            if self.failure == 'install':
                return 1, 'modeled resolver error: spacr==1.5.1.7 not found'
            (target / 'Lib/site-packages/spacr/__init__.py').write_text('__version__ = "1.5.1.6"\n')
            if self.failure == 'absolute_script':
                (target / 'Scripts' / 'spacr.exe').write_bytes(b'MZ #!' + str(target).encode('utf-8'))
            return 0, 'staged package installed'
        if argv[1:3] == ['pip', 'check']:
            assert argv[argv.index('--cache-dir') + 1] == str(self.online.root / 'cache')
            return (1, 'modeled dependency error') if self.failure == 'check' else (0, 'dependencies verified')
        python = Path(argv[0])
        target = python.parent.parent
        if argv[-1] == 'import spacr.qt':
            bad = self.failure == ('final_import' if target == self.online.venv else 'import')
            return (1, 'modeled import error') if bad else (0, '')
        assert '-c' in argv
        expected = json.loads(argv[-1])
        if self.failure == 'identity' or (self.failure == 'final_identity' and target == self.online.venv):
            return 1, '{}'
        return 0, json.dumps({'verified': True, 'version': '1.5.1.6', 'requested_version': expected['version']})


def transaction(online, *, failure=None, activate=None):
    install, _relaunch = commands(online)
    runtime = cleanup.replace(online.record, root=str(online.venv))
    runner = UvRunner(online, failure)
    return cleanup._windows_pip_transaction(runtime, '1.5.1.6', install, machine=online.machine, runner=runner,
                                            activate=activate, uv_tool=True), runner


def test_a_verified_uv_update_swaps_in_and_retains_the_previous_runtime(online):
    before = snapshot(online.venv)
    (version, output), runner = transaction(online)
    assert version == '1.5.1.6'
    recovery = next(online.root.glob('.spacr-windows-update-*'))
    assert snapshot(recovery / 'previous') == before
    journal = json.loads((recovery / 'transaction.json').read_text(encoding="utf-8"))
    assert journal['family'] == 'windows-online-uv' and journal['state'] == 'committed'
    assert journal['entrypoints'] == {'verified': True, 'relocatable': True, 'scripts': 3}
    assert '__version__ = "1.5.1.6"' in (online.venv / 'Lib/site-packages/spacr/__init__.py').read_text(encoding="utf-8")
    assert not (online.root / '.venv.spacr-update-lock').exists()
    install = [command for command in runner.commands if command[1:4] == ['pip', 'install', '--upgrade']]
    assert len(install) == 1 and install[0][0] == str(online.root / 'bootstrap' / 'uv.exe')
    assert Path(install[0][install[0].index('--python') + 1]).parent.parent.name == 'staged'


@pytest.mark.parametrize('failure', ['install', 'check', 'import', 'identity', 'absolute_script',
                                     'final_identity', 'final_import', 'activation'])
def test_a_failed_uv_update_leaves_or_restores_the_exact_runtime(online, failure):
    before = snapshot(online.venv)

    def activate(_version):
        raise RuntimeError('the updated application did not confirm its main window before the deadline')

    with pytest.raises((RuntimeError, ValueError, OSError)):
        transaction(online, failure=failure, activate=activate if failure == 'activation' else None)
    assert snapshot(online.venv) == before
    assert not (online.root / '.venv.spacr-update-lock').exists()
    recovery = next(online.root.glob('.spacr-windows-update-*'))
    assert json.loads((recovery / 'transaction.json').read_text(encoding="utf-8"))['state'] in {'staging_failed', 'rolled_back'}


def test_a_runtime_that_is_not_relocatable_is_never_swapped(online):
    (online.venv / 'pyvenv.cfg').write_text('home = x\ninclude-system-site-packages = false\n')
    before = snapshot(online.venv)
    with pytest.raises(ValueError, match='windows_entrypoint_verification_failed'):
        transaction(online)
    assert snapshot(online.venv) == before


def test_a_running_installer_lease_refuses_the_update(online):
    (online.root / '.venv.spacr-update-lock').write_text('pid=4100\nstarted=filetime:1\nhost=WINHOST\ntree=1\n')
    before = snapshot(online.venv)
    with pytest.raises(OSError):
        transaction(online)
    assert snapshot(online.venv) == before
    assert (online.root / '.venv.spacr-update-lock').is_file()


def _recovery(root, name, state, stamp, family='windows-online-uv'):
    folder = root / ('.spacr-windows-update-' + name)
    (folder / 'previous' / 'Lib').mkdir(parents=True)
    (folder / 'previous' / 'Lib' / 'x.py').write_text('x')
    journal = folder / 'transaction.json'
    journal.write_text(json.dumps({'family': family, 'state': state, 'prefix': str(root / 'venv')}))
    os.utime(journal, (stamp, stamp))
    return folder


def test_pruning_keeps_only_the_newest_committed_recovery_copy(online):
    oldest = _recovery(online.root, 'a', 'committed', 1_000)
    newest = _recovery(online.root, 'b', 'committed', 3_000)
    middle = _recovery(online.root, 'c', 'committed', 2_000)
    review = _recovery(online.root, 'd', 'rolled_back', 500)
    foreign = _recovery(online.root, 'e', 'committed', 100, family='windows-private-pip')
    removed = cleanup._prune_windows_online_recovery(str(online.root))
    assert sorted(removed) == sorted([os.path.realpath(oldest), os.path.realpath(middle)])
    assert newest.is_dir() and review.is_dir() and foreign.is_dir()
    assert not oldest.exists() and not middle.exists()


def test_pruning_removes_a_link_inside_a_copy_without_entering_it(online, tmp_path):
    keep = tmp_path / 'outside' / 'keep.txt'
    keep.parent.mkdir()
    keep.write_text('must survive')
    old = _recovery(online.root, 'a', 'committed', 1_000)
    _recovery(online.root, 'b', 'committed', 2_000)
    link = old / 'previous' / 'link'
    if os.name == 'nt':
        import _winapi

        _winapi.CreateJunction(str(keep.parent), str(link))
    else:
        link.symlink_to(keep.parent, target_is_directory=True)
    cleanup._prune_windows_online_recovery(str(online.root))
    assert not old.exists() and keep.read_text(encoding="utf-8") == 'must survive'
