"""A desktop update verifies actual code and keeps the previous user data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest

from spacr import install_cleanup as cleanup, updater


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / 'private runtime'
    venv.EnvBuilder(with_pip=False).create(root)
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    sites = subprocess.check_output([str(python), '-I', '-c',
                                    'import sysconfig; print(sysconfig.get_path("purelib"))'],
                                   text=True, timeout=15).strip()
    site = Path(sites)
    packaging = Path(importlib.util.find_spec('packaging').origin).parent
    shutil.copytree(packaging, site / 'packaging', ignore=shutil.ignore_patterns('__pycache__'))
    package = site / 'spacr'
    package.mkdir()
    (package / '__init__.py').write_text('from ._version import __version__\n', encoding="utf-8")
    (package / '_version.py').write_text('__version__ = "1.5.1.5"\n', encoding="utf-8")
    info = site / 'spacr-1.5.1.5.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: spacr\nVersion: 1.5.1.5\n', encoding="utf-8")
    return root, python, package, info


@pytest.mark.parametrize('failure', [None, 'version', 'metadata', 'shadow', 'prefix'])
def test_runtime_probe_detects_wrong_code_owner_version_and_environment(runtime, tmp_path, failure):
    root, python, package, info = runtime
    if failure == 'version':
        (package / '_version.py').write_text('__version__ = "1.5.1.4"\n', encoding="utf-8")
    elif failure == 'metadata':
        (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: spacr\nVersion: 1.5.1.4\n', encoding="utf-8")
    elif failure == 'shadow':
        shadow = tmp_path / 'obsolete application'
        shutil.copytree(package, shadow / 'spacr')
        (package.parent / 'old-copy.pth').write_text(
            f'import sys; sys.path.insert(0, {str(shadow)!r})\n', encoding="utf-8")
    command = cleanup._runtime_identity_command(str(python), '1.5.1.5',
                                               prefix=str(tmp_path) if failure == 'prefix' else str(root))
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) == (failure is None), result.stderr
    if failure is None:
        receipt = json.loads(result.stdout)
        assert receipt['verified'] is True
        assert receipt['version'] == '1.5.1.5'
        assert Path(receipt['module']) == package / '__init__.py'
        assert Path(receipt['prefix']).resolve() == root.resolve()


def test_newer_runtime_can_be_verified_without_a_downgrade(runtime):
    root, python, _, _ = runtime
    result = subprocess.run(cleanup._runtime_identity_command(
        str(python), '1.5.1.4', prefix=str(root), minimum=True),
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['version'] == '1.5.1.5'


def test_installer_cli_roundtrip_verifies_the_private_desktop_runtime(runtime):
    root, python, package, _ = runtime
    result = subprocess.run([sys.executable, '-I', cleanup.__file__, 'verify-runtime',
                             '--python', str(python), '--version', '1.5.1.5',
                             '--retiring-prefix', str(root / 'unused previous runtime')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['verified'] is True
    assert Path(receipt['module']) == package / '__init__.py'
    assert receipt['requested_version'] == '1.5.1.5'


def test_updater_refuses_a_downgrade_before_running_any_package_command(monkeypatch):
    monkeypatch.setattr(updater, 'editable_install_location', lambda: None)
    monkeypatch.setattr(updater, '_installed_version', lambda: '1.5.1.5')
    monkeypatch.setattr(updater, 'run_install_command',
                        lambda *args, **kwargs: pytest.fail('No package command for a downgrade'))
    code, message = updater.run_pip_upgrade(target_version='1.5.1.4')
    assert code != 0
    assert 'newer' in message


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_downgrade_refusal_has_a_runtime_translation_in_every_locale(language):
    from spacr.qt.i18n import has_translation, tr
    source = 'The running spaCR is newer than the requested release. Nothing was changed.'
    assert has_translation(source, language)
    assert tr(source, language) != source


def test_retiring_old_application_keeps_nested_cache_and_custom_project(tmp_path):
    root = tmp_path / 'spacr old'
    owned = root / 'venv' / 'spacr-code.py'
    retained = [root / 'cache' / 'wheel.whl', root / 'test_data' / 'annotated.tif',
                root / 'my original project' / 'judgments.csv']
    for path in [owned, *retained]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original bytes')
    machine = cleanup._Machine(platform='linux', environ={'HOME': str(tmp_path / 'user')},
                               fs_root=str(tmp_path))
    record = cleanup.InstallRecord(kind='installer', layout='linux-online',
                                   platform='linux', root=str(root))
    report = cleanup.remove_install(record, system=machine)
    assert report.ok
    assert not owned.exists()
    assert all(path.read_bytes() == b'original bytes' for path in retained)


def test_keeping_install_root_also_keeps_its_current_launcher(tmp_path):
    root = tmp_path / 'spacr current'
    root.mkdir()
    launcher = tmp_path / 'spacr launcher'
    launcher.write_bytes(b'working launcher')
    machine = cleanup._Machine(platform='linux', environ={'HOME': str(tmp_path / 'user')},
                               fs_root=str(tmp_path))
    record = cleanup.InstallRecord(kind='installer', layout='linux-online',
                                   platform='linux', root=str(root), launchers=(str(launcher),))
    report = cleanup.remove_install(record, keep=[str(root)], system=machine)
    assert report.ok and not report.removed
    assert launcher.read_bytes() == b'working launcher'


def test_mixed_runtime_and_declared_user_data_refuses_swap_before_verification(tmp_path, monkeypatch):
    runtime = tmp_path / 'venv'
    data = runtime / 'my project'
    data.mkdir(parents=True)
    saved = data / 'annotations.csv'
    saved.write_bytes(b'irreplaceable judgments')
    machine = cleanup._Machine(platform='linux', fs_root=str(tmp_path),
                               environ={'HOME': str(tmp_path / 'user'), 'SPACR_HOME': str(data)})
    monkeypatch.setattr(cleanup, '_Machine', lambda: machine)
    monkeypatch.setattr(cleanup, '_run', lambda *args, **kwargs: pytest.fail('Refuse before any runtime replacement'))
    assert cleanup._main(['verify-runtime', '--python', 'unused',
                          '--retiring-prefix', str(runtime)]) == 2
    assert saved.read_bytes() == b'irreplaceable judgments'


@pytest.mark.parametrize('minimum,expected_code', [(False, 1), (True, 0)])
def test_cli_minimum_version_keeps_newer_runtime_without_weakening_exact_install_pin(runtime, minimum, expected_code):
    _, python, _, _ = runtime
    argv = [sys.executable, '-I', cleanup.__file__, 'verify-runtime', '--python', str(python),
            '--version', '1.5.1.4']
    if minimum:
        argv.append('--minimum')
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert result.returncode == expected_code, result.stderr
    if minimum:
        assert json.loads(result.stdout)['version'] == '1.5.1.5'


def _identity_file_inventory(root):
    return {str(path.relative_to(root)): ('link', os.readlink(path)) if path.is_symlink()
            else ('file', hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob('*') if '__pycache__' not in path.parts
            and (path.is_file() or path.is_symlink())}


@pytest.mark.parametrize('foreign', ['application-and-metadata', 'linked-application', 'metadata', 'linked-metadata'])
def test_private_runtime_rejects_external_application_or_distribution(runtime, tmp_path, foreign):
    root, python, package, info = runtime
    outside = tmp_path / 'unrelated application'
    outside.mkdir()
    if foreign in ('application-and-metadata', 'linked-application'):
        shutil.move(str(package), str(outside / 'spacr'))
    if foreign in ('application-and-metadata', 'metadata'):
        shutil.move(str(info), str(outside / info.name))
        (package.parent / 'foreign-wheel.pth').write_text(str(outside) + '\n', encoding="utf-8")
    if foreign == 'linked-metadata':
        shutil.move(str(info), str(outside / info.name))
        try:
            info.symlink_to(outside / info.name, target_is_directory=True)
        except OSError:
            pytest.skip('native filesystem does not permit disposable directory symlinks')
    elif foreign == 'linked-application':
        try:
            package.symlink_to(outside / 'spacr', target_is_directory=True)
        except OSError:
            pytest.skip('native filesystem does not permit disposable directory symlinks')
    elif foreign == 'metadata':
        try:
            (outside / 'spacr').symlink_to(package, target_is_directory=True)
        except OSError:
            pytest.skip('native filesystem does not permit disposable directory symlinks')
    retained = outside / 'model weights.bin'
    retained.write_bytes(b'keep unrelated model and user files exactly')
    before = _identity_file_inventory(tmp_path)
    result = subprocess.run(cleanup._runtime_identity_command(
        str(python), '1.5.1.5', prefix=str(root)),
        text=True, capture_output=True, timeout=15)
    assert result.returncode != 0, result.stdout + result.stderr
    expected = ('Application distribution is outside the launcher environment' if foreign in ('metadata', 'linked-metadata')
                else 'Imported application is outside the launcher environment')
    assert expected in result.stdout + result.stderr
    assert _identity_file_inventory(tmp_path) == before
