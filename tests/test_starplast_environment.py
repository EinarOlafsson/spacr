"""External-app installation isolates packages, owns cleanup and snapshots local sources."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from spacr import _starplast as service


def ready(root):
    env = root / 'starplast'
    python = Path(service.environments._env_python(str(env)))
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    (env / service._OWNER).write_text(json.dumps({'app': 'starplast', 'ready': True}))
    return env


def preflight(spec, root):
    return (sys.executable,)


def test_install_commands_only_put_packages_inside_owned_environment(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if 'venv' in argv:
            python = Path(service.environments._env_python(str(tmp_path/'starplast')))
            python.parent.mkdir(exist_ok=True)
            python.touch()
        return 0, [json.dumps({'ok': True, 'version': '0.41.0'})]

    service.install_starplast(service.REPOSITORY, root=tmp_path, runner=runner, preflight=preflight)
    assert service.is_installed(tmp_path)
    env_python = str(tmp_path/'starplast'/'bin'/'python') if os.name != 'nt' else str(tmp_path/'starplast'/'Scripts'/'python.exe')
    for argv, kwargs in calls[1:]:
        assert argv[0] == env_python
        assert argv[1] == '-I'
        assert kwargs['env']['VIRTUAL_ENV'] == str(tmp_path/'starplast')
        assert 'PYTHONPATH' not in kwargs['env']
    assert calls[2][0][-1] == service.REPOSITORY
    assert 'paths.check()' in calls[-1][0][-1]
    assert not (tmp_path/'starplast.lock').exists()
    service.install_starplast(root=tmp_path, runner=lambda *_a, **_k: pytest.fail('installed twice'))
    assert len(calls) == 4


@pytest.mark.parametrize('failure', ['pip', 'selftest', 'cancel'])
def test_failed_or_cancelled_install_removes_only_its_environment_and_retains_log(tmp_path, failure):
    other = tmp_path/'my-project'
    other.mkdir()
    (other/'keep.txt').write_text('keep')

    def runner(argv, **kwargs):
        kwargs['on_line']('diagnostic from subprocess')
        if failure == 'cancel':
            raise service.environments._InstallCancelled('cancelled')
        if failure == 'pip' and 'install' in argv:
            return 1, ['network unavailable']
        return 0, ['no selftest response']

    with pytest.raises((service.environments._InstallFailed, service.environments._InstallCancelled)):
        service.install_starplast(service.REPOSITORY, root=tmp_path, runner=runner, preflight=preflight)
    assert not (tmp_path/'starplast').exists()
    assert not (tmp_path/'starplast.lock').exists()
    assert (other/'keep.txt').read_text() == 'keep'
    assert 'diagnostic from subprocess' in (tmp_path/'starplast-install.log').read_text()


def test_refuses_unowned_folder_and_live_install(tmp_path):
    (tmp_path/'starplast').mkdir()
    (tmp_path/'starplast'/'keep').write_text('mine')
    with pytest.raises(service.environments._InstallFailed, match='not a spaCR-owned'):
        service.install_starplast(service.REPOSITORY, root=tmp_path, preflight=preflight)
    assert (tmp_path/'starplast'/'keep').read_text() == 'mine'
    service._claim(tmp_path)
    with pytest.raises(service.environments._InstallFailed, match='already being installed'):
        service.install_starplast(service.REPOSITORY, root=tmp_path, preflight=preflight)
    assert (tmp_path/'starplast.lock').exists()


def test_local_source_is_real_git_archive_of_committed_files_without_building_in_checkout(tmp_path):
    import tarfile
    source = tmp_path/'checkout with spaces'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source/'starplast').mkdir()
    (source/'starplast'/'app.py').write_text('committed')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'fixture'], check=True)
    (source/'starplast'/'app.py').write_text('uncommitted stays untouched')
    root = tmp_path/'apps'

    def runner(argv, **kwargs):
        if argv[0] == 'git':
            done = subprocess.run(argv, capture_output=True, text=True)
            return done.returncode, done.stderr.splitlines()
        if '--no-cache-dir' in argv:
            with tarfile.open(argv[-1]) as archive:
                assert archive.extractfile('starplast/app.py').read() == b'committed'
        return 0, ['{"ok": true, "version": "fixture"}']

    service.install_starplast(source, root=root, runner=runner, preflight=preflight)
    assert (source/'starplast'/'app.py').read_text() == 'uncommitted stays untouched'
    assert not list(source.glob('*.egg-info'))
    assert not list(root.glob('starplast-source-*'))


def test_child_environment_clears_host_python_and_qt_plugins(tmp_path, monkeypatch):
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        monkeypatch.setenv(key, '/host/packages')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/frozen/libraries')
    monkeypatch.setenv('LD_LIBRARY_PATH_ORIG', '/system/libraries')
    monkeypatch.setenv('DISPLAY', ':1')
    env = ready(tmp_path)
    calls = []
    sentinel = object()
    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return sentinel
    assert service.launch_starplast(root=tmp_path, popen=popen) is sentinel
    argv, kwargs = calls[0]
    assert argv == [service.environments._env_python(str(env)), '-I', '-m', 'starplast']
    assert kwargs['env']['DISPLAY'] == ':1'
    assert kwargs['env']['LD_LIBRARY_PATH'] == '/system/libraries'
    assert kwargs['env']['PYQTGRAPH_QT_LIB'] == 'PyQt6'
    assert 'QT_PLUGIN_PATH' not in kwargs['env'] and 'PYTHONPATH' not in kwargs['env']


def test_real_subprocess_uses_environment_python_and_writes_launch_log(tmp_path, monkeypatch):
    env = tmp_path/'starplast'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True)
    python = service.environments._env_python(str(env))
    site = subprocess.check_output([python, '-I', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
    package = Path(site)/'starplast'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'__main__.py').write_text('import sys; print("isolated:", sys.prefix)')
    (env/service._OWNER).write_text('{"app":"starplast","ready":true}')
    monkeypatch.setenv('PYTHONPATH', '/should/not/be/used')
    process = service.launch_starplast(root=tmp_path)
    assert process.wait(timeout=15) == 0
    assert 'isolated: ' + str(env) in (tmp_path/'starplast-launch.log').read_text()


def test_pre_cancelled_install_does_not_create_environment(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(service.environments._InstallCancelled):
        service.install_starplast(service.REPOSITORY, root=tmp_path, cancel=cancel)
    assert not (tmp_path/'starplast').exists()


def test_default_source_prefers_override_then_sibling_then_official_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(service, '__file__', str(tmp_path/'spacr'/'spacr'/'_starplast.py'))
    monkeypatch.setenv('SPACR_STARPLAST_SOURCE', 'chosen source')
    assert service.default_source() == 'chosen source'
    monkeypatch.delenv('SPACR_STARPLAST_SOURCE')
    assert service.default_source() == service.REPOSITORY
    sibling = tmp_path/'starplast'
    (sibling/'starplast').mkdir(parents=True)
    (sibling/'starplast'/'app.py').touch()
    (sibling/'.git').touch()
    assert service.default_source() == str(sibling)


@pytest.mark.parametrize('local', [True, False])
def test_invalid_source_fails_before_preflight_or_environment_creation(tmp_path, local):
    source = tmp_path/'not-a-checkout'
    source.mkdir()
    if not local:
        source = 'git+https://example.org/unrelated.git'
    root = tmp_path/'apps'
    with pytest.raises(ValueError, match='checkout'):
        service.install_starplast(source, root=root,
                                  preflight=lambda *args: pytest.fail('preflight on invalid source'))
    assert not root.exists()


@pytest.mark.parametrize('record', ['[]', '{"app":"another-app","ready":true}', '{broken'])
def test_invalid_ownership_never_launches_or_deletes_an_existing_folder(tmp_path, record):
    env = ready(tmp_path)
    (env/service._OWNER).write_text(record)
    keep = env/'user-file'
    keep.write_text('preserve')
    assert not service.is_installed(tmp_path)
    with pytest.raises(RuntimeError, match='installed before'):
        service.launch_starplast(root=tmp_path, popen=lambda *args, **kwargs: pytest.fail('launched'))
    with pytest.raises(service.environments._InstallFailed, match='not a spaCR-owned'):
        service.install_starplast(service.REPOSITORY, root=tmp_path, preflight=preflight)
    assert keep.read_text() == 'preserve'
    assert (env/service._OWNER).read_text() == record


def test_cancellation_during_preflight_preserves_existing_partial_environment(tmp_path):
    env = ready(tmp_path)
    (env/service._OWNER).write_text('{"app":"starplast","ready":false}')
    keep = env/'old-install'
    keep.write_text('still here')
    cancel = threading.Event()

    def cancelled_preflight(*args):
        cancel.set()
        return (sys.executable,)

    with pytest.raises(service.environments._InstallCancelled):
        service.install_starplast(service.REPOSITORY, root=tmp_path, cancel=cancel,
                                  preflight=cancelled_preflight,
                                  runner=lambda *args, **kwargs: pytest.fail('command after cancellation'))
    assert keep.read_text() == 'still here'
    assert not (tmp_path/'starplast.lock').exists()


def test_retry_replaces_only_owned_partial_environment_and_recovers_invalid_lock(tmp_path):
    env = ready(tmp_path)
    (env/service._OWNER).write_text('{"app":"starplast","ready":false}')
    (env/'old-install').write_text('partial')
    (tmp_path/'starplast.lock').write_text('interrupted lock write')
    outside = tmp_path/'project.txt'
    outside.write_text('preserve')

    def runner(argv, **kwargs):
        assert not (env/'old-install').exists()
        if 'venv' in argv:
            interpreter = Path(service.environments._env_python(str(env)))
            interpreter.parent.mkdir(exist_ok=True)
            interpreter.touch()
        return 0, ['{"ok":true,"version":"fixture"}']

    service.install_starplast(service.REPOSITORY, root=tmp_path, runner=runner, preflight=preflight)
    assert service.is_installed(tmp_path)
    assert outside.read_text() == 'preserve'
    assert not (tmp_path/'starplast.lock').exists()
    assert not list(tmp_path.glob('starplast-source-*'))


@pytest.mark.parametrize('response', [[], ['[]'], ['{"ok":false}']])
def test_successful_exit_without_positive_selftest_never_marks_ready(tmp_path, response):
    with pytest.raises(service.environments._InstallFailed, match='did not confirm'):
        service.install_starplast(service.REPOSITORY, root=tmp_path, preflight=preflight,
                                  runner=lambda *args, **kwargs: (0, response))
    assert not service.is_installed(tmp_path)
    assert not (tmp_path/'starplast').exists()
    assert not (tmp_path/'starplast.lock').exists()
    assert (tmp_path/'starplast-install.log').is_file()


def test_frozen_launcher_drops_bundled_libraries_without_original_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        monkeypatch.setenv(key, '/bundled/qt-libraries')
        monkeypatch.delenv(key + '_ORIG', raising=False)
    values = service.process_environment(tmp_path/'starplast')
    assert 'LD_LIBRARY_PATH' not in values and 'DYLD_LIBRARY_PATH' not in values
    assert values['QT_API'] == 'pyqt6'


def test_completed_install_between_initial_check_and_lock_is_not_rebuilt(tmp_path, monkeypatch):
    claim = service._claim

    def finish_then_claim(root):
        ready(root)
        claim(root)

    monkeypatch.setattr(service, '_claim', finish_then_claim)
    assert service.install_starplast(service.REPOSITORY, root=tmp_path,
                                    preflight=lambda *args: pytest.fail('already installed')) == tmp_path/'starplast'
    assert service.is_installed(tmp_path)
    assert not (tmp_path/'starplast.lock').exists()


def test_selftest_reads_the_single_starplast_distribution(monkeypatch, capsys):
    """Installing Starplast succeeds without metadata from an auxiliary package."""
    import importlib.metadata
    from types import ModuleType
    package = ModuleType("starplast")
    paths = ModuleType("starplast.paths")
    paths.check = lambda: (True, "ready")
    package.paths = paths
    app = ModuleType("starplast.app")
    app.main = lambda: None
    qt = ModuleType("PyQt6")
    qt.QtWidgets = ModuleType("PyQt6.QtWidgets")
    for name, module in {"starplast": package, "starplast.paths": paths,
                         "starplast.app": app, "PyQt6": qt}.items():
        monkeypatch.setitem(sys.modules, name, module)
    def version(name):
        if name != "starplast":
            raise importlib.metadata.PackageNotFoundError(name)
        return "0.42.0"
    monkeypatch.setattr(importlib.metadata, "version", version)
    exec(service._SELFTEST, {})
    assert json.loads(capsys.readouterr().out) == {"ok": True, "version": "0.42.0"}
