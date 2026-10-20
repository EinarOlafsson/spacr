"""Retirement preserves configured user bytes reached through native symlinks."""
from dataclasses import replace

import pytest

from spacr import install_cleanup as cleanup


def installation(tmp_path, **environment):
    home = tmp_path / 'home'
    root = home / 'spacr-old'
    code = root / 'venv' / 'bin' / 'python'
    code.parent.mkdir(parents=True)
    code.write_bytes(b'disposable old application')
    machine = cleanup._Machine(platform='linux', environ={'HOME': str(home), **environment},
                               fs_root=str(tmp_path), running_prefix=None)
    record = cleanup.InstallRecord(kind='installer', layout='linux-online', platform='linux',
                                   root=str(root))
    return home, root, code, machine, record


@pytest.mark.needs_symlink
@pytest.mark.parametrize('variable', ['SPACR_HOME', 'HF_HOME'])
@pytest.mark.parametrize('alias_kind', ['directory', 'parent'])
def test_configured_external_alias_into_old_runtime_preserves_real_user_bytes(
    tmp_path, variable, alias_kind,
):
    home, root, code, machine, record = installation(tmp_path)
    state = root / 'venv' / 'user-state'
    state.mkdir()
    marker = state / 'settings-and-models.bin'
    marker.write_bytes(b'user bytes, not application code\x00\xff')
    alias = home / 'spacr-data-alias'
    alias.symlink_to(state if alias_kind == 'directory' else state.parent, target_is_directory=True)
    configured = alias if alias_kind == 'directory' else alias / state.name
    machine.environ[variable] = str(configured)
    report = cleanup.remove_install(record, system=machine)
    assert marker.is_file(), 'Configured user data reached through an alias was deleted'
    assert marker.read_bytes() == b'user bytes, not application code\x00\xff'
    assert configured.resolve(strict=True) == state
    assert alias.is_symlink()
    assert not code.exists()
    assert report.ok
    assert not (root / '.venv.spacr-update-lock').exists()
    assert not (home / '.spacr-old.spacr-installer-lock').exists()


@pytest.mark.needs_symlink
def test_final_lease_cleanup_keeps_empty_aliased_user_root(tmp_path):
    home, root, code, machine, record = installation(tmp_path)
    code.unlink()
    code.parent.rmdir()
    code.parent.parent.rmdir()
    alias = home / 'spacr-empty-data-alias'
    alias.symlink_to(root, target_is_directory=True)
    machine.environ['SPACR_HOME'] = str(alias)
    cleanup.remove_install(record, system=machine)
    assert root.is_dir(), 'Lease release removed an empty configured user-data target'
    assert alias.resolve(strict=True) == root
    assert not list(root.iterdir())
    assert not (home / '.spacr-old.spacr-installer-lock').exists()


@pytest.mark.needs_symlink
def test_configured_link_inside_runtime_does_not_follow_or_delete_external_data(tmp_path):
    home, root, code, machine, record = installation(tmp_path)
    state = home / 'external-user-data'
    state.mkdir()
    marker = state / 'model.bin'
    marker.write_bytes(b'keep external bytes')
    alias = root / 'venv' / 'spacr-user-data'
    alias.symlink_to(state, target_is_directory=True)
    machine.environ['SPACR_HOME'] = str(alias)
    report = cleanup.remove_install(record, system=machine)
    assert report.ok
    assert marker.read_bytes() == b'keep external bytes'
    assert alias.is_symlink() and alias.resolve(strict=True) == state
    assert not code.exists()


@pytest.mark.needs_symlink
def test_moved_and_retargeted_configured_alias_is_resolved_at_removal(tmp_path):
    home, root, code, machine, record = installation(tmp_path)
    state = root / 'venv' / 'before-move'
    state.mkdir()
    (state / 'data.bin').write_bytes(b'keep after move')
    alias = home / 'spacr-moved-data-alias'
    alias.symlink_to(state, target_is_directory=True)
    machine.environ['SPACR_HOME'] = str(alias)
    cleanup._user_data(machine)
    moved = state.with_name('after-move')
    state.rename(moved)
    alias.unlink()
    alias.symlink_to(moved, target_is_directory=True)
    report = cleanup.remove_install(record, system=machine)
    assert (moved / 'data.bin').is_file(), 'Retargeted configured user data was deleted'
    assert (moved / 'data.bin').read_bytes() == b'keep after move'
    assert alias.resolve(strict=True) == moved
    assert not code.exists()
    assert report.ok


@pytest.mark.needs_symlink
def test_broken_external_alias_does_not_protect_or_delete_unrelated_roots(tmp_path):
    home, root, code, machine, record = installation(tmp_path)
    unrelated = home / 'unrelated'
    unrelated.mkdir()
    sentinel = unrelated / 'untouched.bin'
    sentinel.write_bytes(b'outside application')
    alias = home / 'spacr-broken-data-alias'
    alias.symlink_to(unrelated / 'missing', target_is_directory=True)
    machine.environ['SPACR_HOME'] = str(alias)
    report = cleanup.remove_install(record, system=machine)
    assert report.ok
    assert alias.is_symlink() and not alias.exists()
    assert sentinel.read_bytes() == b'outside application'
    assert not root.exists() and not code.exists()


@pytest.mark.needs_symlink
def test_aliased_installer_parent_and_separate_data_alias_preserve_same_native_bytes(tmp_path):
    home, root, code, machine, record = installation(tmp_path)
    state = root / 'venv' / 'user-state'
    state.mkdir()
    marker = state / 'data.bin'
    marker.write_bytes(b'keep through two independent aliases')
    data_alias = home / 'spacr-data-alias'
    data_alias.symlink_to(state, target_is_directory=True)
    machine.environ['SPACR_HOME'] = str(data_alias)
    parent_alias = tmp_path / 'installer-parent-alias'
    parent_alias.symlink_to(home, target_is_directory=True)
    record = replace(record, root=str(parent_alias / root.name))
    report = cleanup.remove_install(record, system=machine)
    assert marker.is_file(), 'Aliased installation parent bypassed user-data protection'
    assert marker.read_bytes() == b'keep through two independent aliases'
    assert data_alias.resolve(strict=True) == state
    assert parent_alias.is_symlink() and not code.exists()
    assert report.ok


@pytest.mark.needs_symlink
def test_relative_configured_alias_preserves_data_through_installer_parent_alias(tmp_path, monkeypatch):
    home, root, code, machine, record = installation(tmp_path)
    state = root / 'venv' / 'user-state'
    state.mkdir()
    marker = state / 'relative-data.bin'
    marker.write_bytes(b'relative configured path remains protected')
    alias = tmp_path / 'relative-data-alias'
    alias.symlink_to(state, target_is_directory=True)
    machine.environ['SPACR_HOME'] = alias.name
    parent_alias = tmp_path / 'installer-parent-alias'
    parent_alias.symlink_to(home, target_is_directory=True)
    record = replace(record, root=str(parent_alias / root.name))
    monkeypatch.chdir(tmp_path)
    report = cleanup.remove_install(record, system=machine)
    assert marker.read_bytes() == b'relative configured path remains protected'
    assert alias.is_symlink() and parent_alias.is_symlink()
    assert not code.exists() and report.ok
