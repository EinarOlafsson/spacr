"""Windows alias spelling is modeled separately from real link/reparse identity."""
import json
import os
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as cleanup


@pytest.fixture
def windows_guard(monkeypatch):
    guard = getattr(cleanup, '_installer_metadata_folder', None)
    if guard is not None:
        monkeypatch.setattr(cleanup, '_installer_metadata_folder',
                            lambda folder, code: guard(folder, code, windows=True))


def _native_alias(monkeypatch, alias, physical):
    original_realpath, original_lstat, original_stat = os.path.realpath, os.lstat, os.stat
    def native(path):
        if isinstance(path, int):
            return path
        text = os.fspath(path)
        if text == str(alias) or text.startswith(str(alias) + os.sep):
            return str(physical) + text[len(str(alias)):]
        return text
    monkeypatch.setattr(os.path, 'realpath', lambda path, *args, **kwargs:
                        original_realpath(native(path), *args, **kwargs))
    monkeypatch.setattr(os, 'lstat', lambda path, *args, **kwargs:
                        original_lstat(native(path), *args, **kwargs))
    monkeypatch.setattr(os, 'stat', lambda path, *args, **kwargs:
                        original_stat(native(path), *args, **kwargs))


@pytest.mark.parametrize('spelling', ['RUNNER~1', 'runneradmin'])
def test_native_short_or_case_alias_snapshots_and_restores_the_same_folder(tmp_path, monkeypatch, windows_guard, spelling):
    profile = tmp_path / 'RunnerAdmin'
    work = profile / 'Temp/installer'
    work.mkdir(parents=True)
    alias = tmp_path / spelling
    _native_alias(monkeypatch, alias, profile)
    sidecar = tmp_path / 'installed/runtime-identity.json'
    sidecar.parent.mkdir()
    sidecar.write_bytes(b'previous verified identity')
    folder = alias / 'Temp/installer/previous-metadata'
    state = cleanup._installer_metadata_snapshot(str(folder), [str(sidecar)])
    actual = work / 'previous-metadata'
    assert state['folder'] == str(actual)
    assert state['identity'] == [actual.stat().st_dev, actual.stat().st_ino]
    sidecar.write_bytes(b'new identity')
    restored = cleanup._installer_metadata_restore(str(folder))
    assert restored['folder'] == str(actual)
    assert sidecar.read_bytes() == b'previous verified identity'
    assert json.loads((actual / 'metadata.json').read_text(encoding="utf-8")) == state


@pytest.mark.parametrize('operation', ['snapshot', 'restore'])
def test_actual_symlink_ancestor_is_still_refused(tmp_path, windows_guard, operation):
    physical = tmp_path / 'private work'
    physical.mkdir()
    link = tmp_path / 'redirected work'
    link.symlink_to(physical, target_is_directory=True)
    source = tmp_path / 'identity.json'
    source.write_bytes(b'original')
    if operation == 'restore':
        cleanup._installer_metadata_snapshot(str(physical / 'backup'), [str(source)])
        source.write_bytes(b'new identity')
    with pytest.raises(ValueError) as error:
        if operation == 'snapshot':
            cleanup._installer_metadata_snapshot(str(link / 'backup'), [str(source)])
        else:
            cleanup._installer_metadata_restore(str(link / 'backup'))
    assert error.value.args[0]['code'] == ('linked_metadata_folder' if operation == 'snapshot'
                                         else 'untrusted_metadata_folder')
    assert source.read_bytes() == (b'original' if operation == 'snapshot' else b'new identity')
    if operation == 'snapshot':
        assert not (physical / 'backup').exists()


@pytest.mark.parametrize('operation', ['snapshot', 'restore'])
def test_native_reparse_attribute_is_refused_even_without_spelling_difference(tmp_path, monkeypatch, windows_guard, operation):
    parent = tmp_path / 'reparse directory'
    parent.mkdir()
    source = tmp_path / 'identity.json'
    source.write_bytes(b'original')
    folder = parent / 'backup'
    if operation == 'restore':
        cleanup._installer_metadata_snapshot(str(folder), [str(source)])
        source.write_bytes(b'new identity')
    original = os.lstat
    def lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if os.fspath(path) == str(parent):
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result
    monkeypatch.setattr(os, 'lstat', lstat)
    with pytest.raises(ValueError) as error:
        if operation == 'snapshot':
            cleanup._installer_metadata_snapshot(str(folder), [str(source)])
        else:
            cleanup._installer_metadata_restore(str(folder))
    assert error.value.args[0]['code'] == ('linked_metadata_folder' if operation == 'snapshot'
                                         else 'untrusted_metadata_folder')
    assert source.read_bytes() == (b'original' if operation == 'snapshot' else b'new identity')
    if operation == 'snapshot':
        assert not folder.exists()


def test_alias_is_not_accepted_when_existing_native_identity_does_not_match(tmp_path, monkeypatch, windows_guard):
    profile = tmp_path / 'RunnerAdmin'
    (profile / 'Temp').mkdir(parents=True)
    alias = tmp_path / 'RUNNER~1'
    _native_alias(monkeypatch, alias, profile)
    original = os.path.samefile
    monkeypatch.setattr(os.path, 'samefile', lambda first, second:
                        False if str(first).startswith(str(alias)) else original(first, second))
    source = tmp_path / 'identity.json'
    source.write_bytes(b'original')
    with pytest.raises(ValueError):
        cleanup._installer_metadata_snapshot(str(alias / 'Temp/backup'), [str(source)])
    assert not (profile / 'Temp/backup').exists()
    assert source.read_bytes() == b'original'


def test_posix_folder_policy_still_refuses_a_symlink_parent(tmp_path, monkeypatch):
    physical = tmp_path / 'physical'
    physical.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(physical, target_is_directory=True)
    source = tmp_path / 'identity.json'
    source.write_bytes(b'original')
    original = cleanup._installer_metadata_folder
    monkeypatch.setattr(cleanup, '_installer_metadata_folder',
                        lambda folder, code: original(folder, code, windows=False))
    with pytest.raises(ValueError) as error:
        cleanup._installer_metadata_snapshot(str(alias / 'backup'), [str(source)])
    assert error.value.args[0]['code'] == 'linked_metadata_folder'
    assert not (physical / 'backup').exists()
    assert source.read_bytes() == b'original'
