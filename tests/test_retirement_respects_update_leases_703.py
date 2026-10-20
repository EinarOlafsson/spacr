"""Real installer/retirement concurrency preserves active runtimes and user data."""
import json
import os
from pathlib import Path
import select
import subprocess

import pytest

from spacr import install_cleanup as ic


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'This installation could not be locked for removal. Nothing was removed.'


def _installer(tmp_path, root):
    text = (ROOT / 'packaging/online/install_spacr_unix.sh').read_text(encoding="utf-8")
    block = text.split('# @SPACR_INSTALLER_LEASE_BEGIN@\n', 1)[1].split(
        '# @SPACR_INSTALLER_LEASE_END@', 1)[0]
    script = tmp_path / 'installer.sh'
    script.write_text('set -Eeuo pipefail\nPLATFORM=linux\n'
                      'spacr_say() { printf "%s\\n" "$*" >&2; }\n' + block +
                      '\nprintf "ready\\n"\nread -r finish\n')
    return ['bash', str(script)], dict(os.environ, INSTALL_ROOT=str(root), HOME=str(tmp_path / 'home'))


@pytest.fixture
def installation(tmp_path, monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')
    root = tmp_path / 'application'
    (root / 'venv/lib').mkdir(parents=True)
    (root / 'venv/lib/spacr.py').write_text('original runtime')
    (root / 'cache').mkdir()
    (root / 'cache/model').write_text('weights')
    launcher = tmp_path / 'spacr'
    launcher.write_text(str(root / 'venv/bin/python'))
    machine = ic._Machine(platform='linux', fs_root=str(tmp_path),
                          environ={'HOME': str(tmp_path / 'home')}, running_prefix=None)
    record = ic.InstallRecord(kind='installer', layout='linux-online', platform='linux',
                              root=str(root), launchers=(str(launcher),))
    return root, launcher, machine, record


def test_real_active_installer_cannot_be_retired_from_another_root(installation, tmp_path):
    root, launcher, machine, record = installation
    command, environment = _installer(tmp_path, root)
    process = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline().strip() == 'ready'
        before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
        report = ic.remove_install(record, system=machine)
        assert not report.ok and not report.removed
        assert report.failed == [(str(root), SOURCE)]
        assert {path: path.read_bytes() for path in before} == before
        assert launcher.read_text(encoding="utf-8") == str(root / 'venv/bin/python')
    finally:
        process.communicate('finish\n', timeout=10)
    assert process.returncode == 0
    report = ic.remove_install(record, system=machine)
    assert report.ok
    assert not (root / 'venv').exists() and not launcher.exists()
    assert (root / 'cache/model').read_text(encoding="utf-8") == 'weights'


def test_installer_cannot_enter_between_retirement_check_and_actual_delete(installation, tmp_path, monkeypatch):
    root, launcher, machine, record = installation
    command, environment = _installer(tmp_path, root)
    original = ic._delete
    attempts = []

    def during_delete(path, *args):
        if path == str(root):
            attempts.append(subprocess.run(command, env=environment, text=True,
                                           capture_output=True, timeout=10))
        return original(path, *args)

    monkeypatch.setattr(ic, '_delete', during_delete)
    report = ic.remove_install(record, system=machine)
    assert report.ok and len(attempts) == 1
    assert attempts[0].returncode == 73
    assert 'already_running' in attempts[0].stderr
    assert not (root / 'venv').exists() and not launcher.exists()
    assert (root / 'cache/model').read_text(encoding="utf-8") == 'weights'


@pytest.mark.parametrize('kind', ['directory', 'file', 'symlink'])
def test_existing_unknown_or_interrupted_outer_lease_is_preserved(installation, tmp_path, kind):
    root, launcher, machine, record = installation
    lease = root.parent / ('.' + root.name + '.spacr-installer-lock')
    if kind == 'directory':
        lease.mkdir()
        (lease / 'evidence').write_text('retain')
    elif kind == 'file':
        lease.write_text('unknown owner')
    else:
        target = tmp_path / 'user-file'
        target.write_text('preserve')
        lease.symlink_to(target)
    before = lease.lstat()
    report = ic.remove_install(record, system=machine)
    assert not report.ok and not report.removed
    assert lease.lstat().st_ino == before.st_ino
    assert (root / 'venv/lib/spacr.py').read_text(encoding="utf-8") == 'original runtime'
    assert launcher.exists()


def test_existing_private_update_lease_refuses_before_removal_and_releases_only_own_outer_lease(installation):
    root, launcher, machine, record = installation
    lease = root / '.venv.spacr-update-lock'
    lease.mkdir()
    (lease / 'journal').write_text('interrupted update proof')
    report = ic.remove_install(record, system=machine)
    assert not report.ok and not report.removed
    assert (lease / 'journal').read_text(encoding="utf-8") == 'interrupted update proof'
    assert not (root.parent / ('.' + root.name + '.spacr-installer-lock')).exists()
    assert launcher.exists()


def test_selected_environment_cannot_uninstall_packages_during_private_update(installation, monkeypatch):
    root, _launcher, machine, _record = installation
    prefix = root / 'venv'
    record = ic.InstallRecord(kind='environment', layout='venv', platform='linux',
                              root=str(prefix), python=str(prefix / 'bin/python'))
    (root / '.venv.spacr-update-lock').mkdir()
    monkeypatch.setattr(ic, '_uninstall_from_environment', lambda *a: pytest.fail('pip must not run'))
    report = ic.remove_install(record, ticked=True, system=machine)
    assert not report.ok and not report.removed
    assert (prefix / 'lib/spacr.py').read_text(encoding="utf-8") == 'original runtime'


def test_kept_or_running_root_does_not_try_to_take_its_own_update_lease(installation):
    root, launcher, machine, record = installation
    (root / '.venv.spacr-update-lock').mkdir()
    assert ic.remove_install(record, keep=[str(root)], system=machine).ok
    record = ic.replace(record, running=True)
    report = ic.remove_install(record, system=machine)
    assert not report.ok
    assert launcher.exists()


def test_empty_removed_root_is_not_retained_only_because_of_our_temporary_lease(installation):
    root, _launcher, machine, record = installation
    (root / 'cache/model').unlink()
    (root / 'cache').rmdir()
    report = ic.remove_install(record, system=machine)
    assert report.ok
    assert not root.exists()


@pytest.mark.parametrize('relative', ['.spacr', '.local/share/applications'])
def test_empty_protected_user_or_shared_folder_is_not_removed_during_lease_release(installation, relative):
    _root, _launcher, machine, _record = installation
    protected = Path(machine.home) / relative
    protected.mkdir(parents=True)
    record = ic.InstallRecord(kind='installer', layout='linux-online', platform='linux',
                              root=str(protected))
    report = ic.remove_install(record, system=machine)
    assert protected.is_dir()
    assert report.failed or report.skipped


def test_symlinked_record_does_not_remove_its_empty_user_directory_target(installation, tmp_path):
    _root, _launcher, machine, _record = installation
    target = tmp_path / 'unrelated empty folder'
    target.mkdir()
    link = tmp_path / 'spacr-old'
    link.symlink_to(target, target_is_directory=True)
    record = ic.InstallRecord(kind='installer', layout='linux-online', platform='linux', root=str(link))
    report = ic.remove_install(record, system=machine)
    assert report.ok
    assert not link.exists()
    assert target.is_dir() and not list(target.iterdir())


def test_core_failure_is_not_relabelled_as_a_nothing_removed_lease_refusal(installation, monkeypatch):
    root, _launcher, machine, record = installation

    def fail_after_mutation(*args):
        (root / 'venv/lib/spacr.py').unlink()
        raise OSError('actual mutation failure')

    monkeypatch.setattr(ic, '_delete', fail_after_mutation)
    with pytest.raises(OSError, match='actual mutation failure'):
        ic.remove_install(record, system=machine)
    assert not (root / 'venv/lib/spacr.py').exists()
    assert not (root / '.venv.spacr-update-lock').exists()


@pytest.mark.parametrize('platform', ['linux', 'windows'])
def test_retirement_release_never_removes_a_replaced_outer_lease(installation, tmp_path, monkeypatch, platform):
    root, _launcher, machine, record = installation
    machine.platform = platform
    record = ic.replace(record, platform=platform, layout=platform + '-online')
    lease = root.parent / ('.' + root.name + '.spacr-installer-lock')
    retained = tmp_path / 'previous-owned-lease'
    original = ic._delete

    def replace_during_delete(path, *args):
        if path == str(root):
            lease.rename(retained)
            if platform == 'windows':
                lease.write_text('other owner')
            else:
                lease.mkdir()
                (lease / 'proof').write_text('other owner')
        return original(path, *args)

    monkeypatch.setattr(ic, '_delete', replace_during_delete)
    report = ic.remove_install(record, system=machine)
    assert report.ok
    assert retained.exists()
    assert (lease.read_text(encoding="utf-8") if platform == 'windows' else (lease / 'proof').read_text(encoding="utf-8")) == 'other owner'
    assert (root / 'cache/model').read_text(encoding="utf-8") == 'weights'


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_actual_removal_refusal_uses_exact_reviewed_locale(installation, monkeypatch, language):
    root, _launcher, machine, record = installation
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    (root / '.venv.spacr-update-lock').mkdir()
    payload = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language /
                          '2026-10-15-n703-retirement-lease.json').read_text(encoding="utf-8"))
    reviewed = payload['records'][0]
    assert reviewed['source'] == SOURCE
    assert reviewed['translation'] != SOURCE
    assert ic.remove_install(record, system=machine).failed == [(str(root), reviewed['translation'])]
