"""External-app installation isolates packages, owns cleanup and snapshots local sources."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from spacr import _parascape as service


def ready(root):
    env = root / 'parascape'
    python = Path(service.environments._env_python(str(env)))
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    (env / service._OWNER).write_text(json.dumps({'app': 'parascape', 'ready': True}), encoding="utf-8")
    return env


def preflight(spec, root):
    return (sys.executable,)


def test_install_commands_only_put_packages_inside_owned_environment(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if 'venv' in argv:
            python = Path(service.environments._env_python(str(tmp_path/'parascape')))
            python.parent.mkdir(exist_ok=True)
            python.touch()
        return 0, [json.dumps({'ok': True, 'version': '0.41.0'})]

    service.install_parascape(service.REPOSITORY, root=tmp_path, runner=runner, preflight=preflight)
    assert service.is_installed(tmp_path)
    env_python = str(tmp_path/'parascape'/'bin'/'python') if os.name != 'nt' else str(tmp_path/'parascape'/'Scripts'/'python.exe')
    for argv, kwargs in calls[1:]:
        assert argv[0] == env_python
        assert argv[1] == '-I'
        assert kwargs['env']['VIRTUAL_ENV'] == str(tmp_path/'parascape')
        assert 'PYTHONPATH' not in kwargs['env']
    assert calls[2][0][-1] == service.REPOSITORY
    assert 'paths.check()' in calls[-1][0][-1]
    assert not (tmp_path/'parascape.lock').exists()
    service.install_parascape(root=tmp_path, runner=lambda *_a, **_k: pytest.fail('installed twice'))
    assert len(calls) == 4


@pytest.mark.parametrize('failure', ['pip', 'selftest', 'cancel'])
def test_failed_or_cancelled_install_removes_only_its_environment_and_retains_log(tmp_path, failure):
    other = tmp_path/'my-project'
    other.mkdir()
    (other/'keep.txt').write_text('keep', encoding="utf-8")

    def runner(argv, **kwargs):
        kwargs['on_line']('diagnostic from subprocess')
        if failure == 'cancel':
            raise service.environments._InstallCancelled('cancelled')
        if failure == 'pip' and 'install' in argv:
            return 1, ['network unavailable']
        return 0, ['no selftest response']

    with pytest.raises((service.environments._InstallFailed, service.environments._InstallCancelled)):
        service.install_parascape(service.REPOSITORY, root=tmp_path, runner=runner, preflight=preflight)
    assert not (tmp_path/'parascape').exists()
    assert not (tmp_path/'parascape.lock').exists()
    assert (other/'keep.txt').read_text(encoding="utf-8") == 'keep'
    assert 'diagnostic from subprocess' in (tmp_path/'parascape-install.log').read_text(encoding="utf-8")


def test_refuses_unowned_folder_and_live_install(tmp_path):
    (tmp_path/'parascape').mkdir()
    (tmp_path/'parascape'/'keep').write_text('mine', encoding="utf-8")
    with pytest.raises(service.environments._InstallFailed, match='not a spaCR-owned'):
        service.install_parascape(service.REPOSITORY, root=tmp_path, preflight=preflight)
    assert (tmp_path/'parascape'/'keep').read_text(encoding="utf-8") == 'mine'
    service._claim(tmp_path)
    with pytest.raises(service.environments._InstallFailed, match='already being installed'):
        service.install_parascape(service.REPOSITORY, root=tmp_path, preflight=preflight)
    assert (tmp_path/'parascape.lock').exists()


def test_local_source_is_real_git_archive_of_committed_files_without_building_in_checkout(tmp_path):
    import tarfile
    source = tmp_path/'checkout with spaces'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source/'parascape').mkdir()
    (source/'parascape'/'app.py').write_text('committed', encoding="utf-8")
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'fixture'], check=True)
    (source/'parascape'/'app.py').write_text('uncommitted stays untouched', encoding="utf-8")
    root = tmp_path/'apps'

    def runner(argv, **kwargs):
        if argv[0] == 'git':
            done = subprocess.run(argv, capture_output=True, text=True)
            return done.returncode, done.stderr.splitlines()
        if '--no-cache-dir' in argv:
            with tarfile.open(argv[-1]) as archive:
                assert archive.extractfile('parascape/app.py').read() == b'committed'
        return 0, ['{"ok": true, "version": "fixture"}']

    service.install_parascape(source, root=root, runner=runner, preflight=preflight)
    assert (source/'parascape'/'app.py').read_text(encoding="utf-8") == 'uncommitted stays untouched'
    assert not list(source.glob('*.egg-info'))
    assert not list(root.glob('parascape-source-*'))


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
    assert service.launch_parascape(root=tmp_path, popen=popen) is sentinel
    argv, kwargs = calls[0]
    assert argv == [str(service.gui_command(env))]
    assert service.gui_command(env).parent == Path(service.environments._env_python(str(env))).parent
    assert kwargs['env']['DISPLAY'] == ':1'
    assert kwargs['env']['LD_LIBRARY_PATH'] == '/system/libraries'
    assert kwargs['env']['PYQTGRAPH_QT_LIB'] == 'PyQt6'
    assert 'QT_PLUGIN_PATH' not in kwargs['env'] and 'PYTHONPATH' not in kwargs['env']


@pytest.mark.skipif(os.name == 'nt', reason='a console script on Windows is a compiled launcher')
def test_real_subprocess_uses_environment_console_script_and_writes_launch_log(tmp_path, monkeypatch):
    env = tmp_path/'parascape'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True)
    python = service.environments._env_python(str(env))
    script = service.gui_command(env)
    script.write_text(f'#!{python}\nimport sys; print("isolated:", sys.prefix)\n', encoding="utf-8")
    script.chmod(0o755)
    (env/service._OWNER).write_text('{"app":"parascape","ready":true}', encoding="utf-8")
    monkeypatch.setenv('PYTHONPATH', '/should/not/be/used')
    process = service.launch_parascape(root=tmp_path)
    assert process.wait(timeout=15) == 0
    assert 'isolated: ' + str(env) in (tmp_path/'parascape-launch.log').read_text(encoding="utf-8")


def test_pre_cancelled_install_does_not_create_environment(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(service.environments._InstallCancelled):
        service.install_parascape(service.REPOSITORY, root=tmp_path, cancel=cancel)
    assert not (tmp_path/'parascape').exists()


def test_default_source_uses_pypi_even_beside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(service, '__file__', str(tmp_path/'spacr'/'spacr'/'_parascape.py'))
    monkeypatch.setenv('SPACR_PARASCAPE_SOURCE', 'chosen source')
    assert service.default_source() == 'chosen source'
    monkeypatch.delenv('SPACR_PARASCAPE_SOURCE')
    assert service.default_source() == service.PYPI_PACKAGE
    sibling = tmp_path/'parascape'
    (sibling/'parascape').mkdir(parents=True)
    (sibling/'parascape'/'app.py').touch()
    (sibling/'.git').touch()
    assert service.default_source() == service.PYPI_PACKAGE


@pytest.mark.parametrize('local', [True, False])
def test_invalid_source_fails_before_preflight_or_environment_creation(tmp_path, local):
    source = tmp_path/'not-a-checkout'
    source.mkdir()
    if not local:
        source = 'git+https://example.org/unrelated.git'
    root = tmp_path/'apps'
    with pytest.raises(ValueError, match='checkout'):
        service.install_parascape(source, root=root,
                                  preflight=lambda *args: pytest.fail('preflight on invalid source'))
    assert not root.exists()


@pytest.mark.parametrize('record', ['[]', '{"app":"another-app","ready":true}', '{broken'])
def test_invalid_ownership_never_launches_or_deletes_an_existing_folder(tmp_path, record):
    env = ready(tmp_path)
    (env/service._OWNER).write_text(record, encoding="utf-8")
    keep = env/'user-file'
    keep.write_text('preserve', encoding="utf-8")
    assert not service.is_installed(tmp_path)
    with pytest.raises(RuntimeError, match='installed before'):
        service.launch_parascape(root=tmp_path, popen=lambda *args, **kwargs: pytest.fail('launched'))
    with pytest.raises(service.environments._InstallFailed, match='not a spaCR-owned'):
        service.install_parascape(service.REPOSITORY, root=tmp_path, preflight=preflight)
    assert keep.read_text(encoding="utf-8") == 'preserve'
    assert (env/service._OWNER).read_text(encoding="utf-8") == record


def test_cancellation_during_preflight_preserves_existing_partial_environment(tmp_path):
    env = ready(tmp_path)
    (env/service._OWNER).write_text('{"app":"parascape","ready":false}', encoding="utf-8")
    keep = env/'old-install'
    keep.write_text('still here', encoding="utf-8")
    cancel = threading.Event()

    def cancelled_preflight(*args):
        cancel.set()
        return (sys.executable,)

    with pytest.raises(service.environments._InstallCancelled):
        service.install_parascape(service.REPOSITORY, root=tmp_path, cancel=cancel,
                                  preflight=cancelled_preflight,
                                  runner=lambda *args, **kwargs: pytest.fail('command after cancellation'))
    assert keep.read_text(encoding="utf-8") == 'still here'
    assert not (tmp_path/'parascape.lock').exists()


def test_retry_replaces_only_owned_partial_environment_and_recovers_invalid_lock(tmp_path):
    env = ready(tmp_path)
    (env/service._OWNER).write_text('{"app":"parascape","ready":false}', encoding="utf-8")
    (env/'old-install').write_text('partial', encoding="utf-8")
    (tmp_path/'parascape.lock').write_text('interrupted lock write', encoding="utf-8")
    outside = tmp_path/'project.txt'
    outside.write_text('preserve', encoding="utf-8")

    def runner(argv, **kwargs):
        assert not (env/'old-install').exists()
        if 'venv' in argv:
            interpreter = Path(service.environments._env_python(str(env)))
            interpreter.parent.mkdir(exist_ok=True)
            interpreter.touch()
        return 0, ['{"ok":true,"version":"fixture"}']

    service.install_parascape(service.REPOSITORY, root=tmp_path, runner=runner, preflight=preflight)
    assert service.is_installed(tmp_path)
    assert outside.read_text(encoding="utf-8") == 'preserve'
    assert not (tmp_path/'parascape.lock').exists()
    assert not list(tmp_path.glob('parascape-source-*'))


@pytest.mark.parametrize('response', [[], ['[]'], ['{"ok":false}']])
def test_successful_exit_without_positive_selftest_never_marks_ready(tmp_path, response):
    with pytest.raises(service.environments._InstallFailed, match='did not confirm'):
        service.install_parascape(service.REPOSITORY, root=tmp_path, preflight=preflight,
                                  runner=lambda *args, **kwargs: (0, response))
    assert not service.is_installed(tmp_path)
    assert not (tmp_path/'parascape').exists()
    assert not (tmp_path/'parascape.lock').exists()
    assert (tmp_path/'parascape-install.log').is_file()


def test_frozen_launcher_drops_bundled_libraries_without_original_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        monkeypatch.setenv(key, '/bundled/qt-libraries')
        monkeypatch.delenv(key + '_ORIG', raising=False)
    values = service.process_environment(tmp_path/'parascape')
    assert 'LD_LIBRARY_PATH' not in values and 'DYLD_LIBRARY_PATH' not in values
    assert values['QT_API'] == 'pyqt6'


def test_completed_install_between_initial_check_and_lock_is_not_rebuilt(tmp_path, monkeypatch):
    claim = service._claim

    def finish_then_claim(root):
        ready(root)
        claim(root)

    monkeypatch.setattr(service, '_claim', finish_then_claim)
    assert service.install_parascape(service.REPOSITORY, root=tmp_path,
                                    preflight=lambda *args: pytest.fail('already installed')) == tmp_path/'parascape'
    assert service.is_installed(tmp_path)
    assert not (tmp_path/'parascape.lock').exists()


def test_selftest_reads_the_single_parascape_distribution(monkeypatch, capsys):
    """Installing ParaScape succeeds without metadata from an auxiliary package."""
    import importlib.metadata
    from types import ModuleType
    package = ModuleType("parascape")
    paths = ModuleType("parascape.paths")
    paths.check = lambda: (True, "ready")
    package.paths = paths
    app = ModuleType("parascape.app")
    app.main = lambda: None
    qt = ModuleType("PyQt6")
    qt.QtWidgets = ModuleType("PyQt6.QtWidgets")
    for name, module in {"parascape": package, "parascape.paths": paths,
                         "parascape.app": app, "PyQt6": qt}.items():
        monkeypatch.setitem(sys.modules, name, module)
    def version(name):
        if name != "parascape":
            raise importlib.metadata.PackageNotFoundError(name)
        return "0.42.0"
    monkeypatch.setattr(importlib.metadata, "version", version)
    exec(service._SELFTEST, {})
    assert json.loads(capsys.readouterr().out) == {"ok": True, "version": "0.42.0"}


def test_default_install_resolves_unpinned_pypi_package_not_a_same_named_folder(tmp_path, monkeypatch):
    monkeypatch.delenv('SPACR_PARASCAPE_SOURCE', raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path/'parascape').mkdir()
    root = tmp_path/'apps'
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if 'venv' in argv:
            python = Path(service.environments._env_python(str(root/'parascape')))
            python.parent.mkdir(exist_ok=True)
            python.touch()
        return 0, [json.dumps({'ok': True, 'version': '0.42.0'})]

    service.install_parascape(root=root, runner=runner, preflight=preflight)
    command = calls[2]
    assert command[-1] == 'parascape>=0.0.1.0' == service.pypi_requirement()
    assert '--upgrade' in command
    assert command[command.index('--index-url') + 1] == 'https://pypi.org/simple'
    assert not any(command[0] == 'git' for command in calls)
    record = json.loads((root/'parascape'/service._OWNER).read_text(encoding="utf-8"))
    assert record['source'] == 'parascape'
    assert record['version'] == '0.42.0'

@pytest.mark.parametrize('installed, latest, available', [
    ('0.9', '0.10', True), ('1.0', '1.0', False), ('2.0', '1.9', False),
    ('0.0.1.0', '0.0.1.1', True), ('0.0.1.9', '0.0.1.10', True), ('0.0.2.0', '0.0.1.10', False),
    ('0.0.1.0', '0.0.1', False),
])
def test_update_check_reads_actual_environment_and_compatible_index(tmp_path, installed, latest, available):
    env = ready(tmp_path)
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == service.environments._env_python(str(env))
        assert kwargs['env']['PIP_CONFIG_FILE'] == os.devnull
        if '-c' in argv:
            return 0, [installed]
        return 0, ['WARNING: index warning', f'parascape ({latest})', 'Available versions: ' + latest]
    result = service.check_parascape_update(root=tmp_path, runner=runner)
    assert result == dict(installed=installed, latest=latest, available=available)
    assert '--index-url' in calls[1] and 'https://pypi.org/simple' in calls[1]
    assert '--pre' not in calls[1] and 'install' not in calls[1]
    assert service._record(env)['ready']


def test_upgrade_ignores_developer_source_and_runs_pip_in_existing_environment(tmp_path, monkeypatch):
    env = ready(tmp_path)
    (env/'preserve').write_text('keep', encoding="utf-8")
    monkeypatch.setenv('SPACR_PARASCAPE_SOURCE', service.REPOSITORY)
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        assert not service.is_installed(tmp_path)
        return 0, ['{"ok":true,"version":"1.2"}']
    service.upgrade_parascape(service.REPOSITORY, root=tmp_path, runner=runner)
    assert len(calls) == 2
    assert calls[0][-1] == 'parascape>=0.0.1.0' and '--upgrade' in calls[0]
    assert calls[0][:4] == [service.environments._env_python(str(env)), '-I', '-m', 'pip']
    assert (env/'preserve').read_text(encoding="utf-8") == 'keep'
    assert service._record(env) == dict(app='parascape', ready=True, source='parascape', version='1.2')
    assert not (tmp_path/'parascape.lock').exists()


def test_interrupted_upgrade_keeps_environment_and_allows_pip_retry(tmp_path):
    env = ready(tmp_path)
    (env/'preserve').write_text('keep', encoding="utf-8")
    def cancelled(*args, **kwargs):
        raise service.environments._InstallCancelled('cancelled')
    with pytest.raises(service.environments._InstallCancelled):
        service.upgrade_parascape(root=tmp_path, runner=cancelled)
    assert not service.is_installed(tmp_path)
    assert (env/'preserve').exists() and not (tmp_path/'parascape.lock').exists()
    service.upgrade_parascape(root=tmp_path, runner=lambda *a, **k: (0, ['{"ok":true,"version":"1.2"}']))
    assert service.is_installed(tmp_path)


def test_update_index_failure_keeps_working_install(tmp_path):
    env = ready(tmp_path)
    with pytest.raises(RuntimeError, match='offline'):
        service.check_parascape_update(root=tmp_path, runner=lambda *a, **k: (1, ['offline']))
    assert service.is_installed(tmp_path)


def test_pip_user_overrides_cannot_change_upgrade_index_or_release_selection(monkeypatch, tmp_path):
    for key in ('PIP_PRE', 'PIP_FIND_LINKS', 'PIP_NO_INDEX', 'PIP_EXTRA_INDEX_URL'):
        monkeypatch.setenv(key, 'unwanted')
    values = service.process_environment(tmp_path/'parascape')
    assert all(key not in values for key in ('PIP_PRE', 'PIP_FIND_LINKS', 'PIP_NO_INDEX', 'PIP_EXTRA_INDEX_URL'))


def test_minimum_release_is_a_four_part_version_compared_with_packaging():
    from packaging.version import Version

    assert service.MINIMUM_VERSION == '0.0.1.0'
    assert service.pypi_requirement() == service.PYPI_PACKAGE + '>=' + service.MINIMUM_VERSION
    assert Version('0.0.1.10') > Version('0.0.1.9') > Version(service.MINIMUM_VERSION)


def test_folder_left_under_the_former_name_is_neither_used_nor_touched(tmp_path):
    former = bytes.fromhex('73746172706c617374').decode()
    legacy = tmp_path / former
    (legacy / 'bin').mkdir(parents=True)
    (legacy / 'bin' / 'python').touch()
    (legacy / f'.spacr-{former}.json').write_text(json.dumps({'app': former, 'ready': True}), encoding="utf-8")
    assert not service.is_installed(tmp_path)

    def runner(argv, **kwargs):
        if 'venv' in argv:
            python = Path(service.environments._env_python(str(tmp_path/'parascape')))
            python.parent.mkdir(exist_ok=True)
            python.touch()
        return 0, [json.dumps({'ok': True, 'version': '0.0.1.0'})]

    assert service.install_parascape(root=tmp_path, runner=runner, preflight=preflight) == tmp_path/'parascape'
    assert service.is_installed(tmp_path)
    assert (legacy / 'bin' / 'python').is_file()


def test_console_script_takes_the_interpreter_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(service.environments, '_env_python',
                        lambda env: str(Path(env) / 'Scripts' / 'python.exe'))
    assert service.gui_command(tmp_path) == tmp_path / 'Scripts' / 'parascape.exe'
