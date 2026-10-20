"""A failed cross-device copy never exposes a partial destination cache."""
import errno
import os
from pathlib import Path
import shutil

import pytest

from spacr import run_journal as journal


@pytest.fixture
def location(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setattr(journal, '_spacr_home', lambda: home)
    monkeypatch.setenv('SPACR_NEWS_CACHE', '')
    source = home / 'news'
    source.mkdir()
    (source / 'first.json').write_bytes(b'complete-first-cache-entry')
    (source / 'second.json').write_bytes(b'complete-second-cache-entry')
    return source, tmp_path / 'other-drive', home / 'cache_locations.json'


def test_interrupted_cross_device_copy_keeps_source_and_unpublished_destination(location, monkeypatch):
    source, parent, record = location
    original_rename = os.rename
    original_copy = shutil.copy2

    def cross_device(src, dst, *args, **kwargs):
        if Path(src) == source:
            raise OSError(errno.EXDEV, 'controlled cross-device rename')
        return original_rename(src, dst, *args, **kwargs)

    def interrupted_copy(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b'partial-copy')
        raise OSError('controlled interrupted copy')

    original_move = shutil.move
    monkeypatch.setattr(os, 'rename', cross_device)
    monkeypatch.setattr(shutil, 'copy2', interrupted_copy)
    monkeypatch.setattr(shutil, 'move', lambda src, dst: original_move(src, dst, copy_function=interrupted_copy))
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises((OSError, shutil.Error)):
        journal._relocate_cache('news', parent)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert not (parent / 'spacr-news').exists()
    assert list(parent.iterdir()) == []
    assert not record.exists()
    assert os.environ['SPACR_NEWS_CACHE'] == ''
    monkeypatch.setattr(shutil, 'copy2', original_copy)
    monkeypatch.setattr(shutil, 'move', original_move)
    moved = journal._relocate_cache('news', parent)
    assert {path.name: path.read_bytes() for path in moved.iterdir()} == before


@pytest.fixture
def cross_drive(location, monkeypatch):
    source, _parent, _record = location
    rename = os.rename

    def cross_device(src, dst, *args, **kwargs):
        if Path(src) == source:
            raise OSError(errno.EXDEV, 'controlled cross-device rename')
        return rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, 'rename', cross_device)
    return location


def test_cross_device_metadata_failure_preserves_complete_source_and_target(cross_drive, monkeypatch):
    source, parent, record = cross_drive
    before = {path.name: path.read_bytes() for path in source.iterdir()}

    def fail_record(*args):
        assert {path.name: path.read_bytes() for path in source.iterdir()} == before
        raise OSError('controlled metadata failure')

    monkeypatch.setattr(journal, '_atomic_write_text', fail_record)
    with pytest.raises(OSError, match='controlled metadata failure'):
        journal._relocate_cache('news', parent)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    target = parent / 'spacr-news'
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert not record.exists()
    assert os.environ['SPACR_NEWS_CACHE'] == ''
    assert list(parent.iterdir()) == [target]


def test_source_cleanup_failure_keeps_committed_new_cache(cross_drive, monkeypatch):
    source, parent, record = cross_drive
    rmtree = shutil.rmtree

    def fail_old_cache(path, *args, **kwargs):
        if Path(path) == source:
            raise OSError('controlled old-cache cleanup failure')
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, 'rmtree', fail_old_cache)
    target = journal._relocate_cache('news', parent)
    assert target == parent / 'spacr-news'
    assert (target / 'first.json').read_bytes() == (source / 'first.json').read_bytes()
    assert journal._cache_folder('news') == target
    assert record.is_file()


def test_concurrent_private_target_is_preserved_and_source_remains_intact(cross_drive, monkeypatch):
    source, parent, record = cross_drive
    original_copy = shutil.copy2

    def create_private_target(src, dst, *args, **kwargs):
        target = parent / 'spacr-news'
        target.mkdir(exist_ok=True)
        (target / 'private.json').write_bytes(b'concurrent private file')
        return original_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, 'copy2', create_private_target)
    with pytest.raises(OSError):
        journal._relocate_cache('news', parent)
    assert (source / 'first.json').read_bytes() == b'complete-first-cache-entry'
    target = parent / 'spacr-news'
    assert [path.name for path in target.iterdir()] == ['private.json']
    assert (target / 'private.json').read_bytes() == b'concurrent private file'
    assert list(parent.iterdir()) == [target]
    assert not record.exists()
    assert os.environ['SPACR_NEWS_CACHE'] == ''


def test_copy_failure_restores_preexisting_empty_destination(cross_drive, monkeypatch):
    source, parent, record = cross_drive
    target = parent / 'spacr-news'
    target.mkdir(mode=0o700, parents=True)
    original_mode = target.stat().st_mode

    def fail_copy(*args, **kwargs):
        raise OSError('controlled copy failure')

    monkeypatch.setattr(shutil, 'copy2', fail_copy)
    with pytest.raises(shutil.Error):
        journal._relocate_cache('news', parent)
    assert target.is_dir() and not list(target.iterdir())
    assert target.stat().st_mode == original_mode
    assert list(parent.iterdir()) == [target]
    assert (source / 'first.json').read_bytes() == b'complete-first-cache-entry'
    assert not record.exists()


@pytest.mark.parametrize('existing_target', [False, True])
def test_native_rename_failure_preserves_cache_preferences_and_empty_target(
        location, monkeypatch, existing_target):
    import json
    import stat

    source, parent, record = location
    parent.mkdir()
    target = parent / 'spacr-news'
    original_mode = None
    if existing_target:
        target.mkdir(mode=0o700)
        original_mode = stat.S_IMODE(target.stat().st_mode)
    record.write_text(json.dumps({'SPACR_MODEL_CACHE': str(source.parent / 'other-models')}, indent=2) + '\n')
    manifest_before = record.read_bytes()
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    env_before = {env: os.environ.get(env) for _key, _label, env in journal._CACHE_ROWS if env}
    rename = os.rename
    attempted = []
    def denied(src, dst, *args, **kwargs):
        if Path(src) == source and Path(dst) == target:
            attempted.append((Path(src), Path(dst)))
            raise PermissionError(errno.EACCES, 'owned fixture native rename denied')
        return rename(src, dst, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('a non-EXDEV rename failure must not copy or publish a new location')
    monkeypatch.setattr(os, 'rename', denied)
    monkeypatch.setattr(shutil, 'copytree', forbidden)
    monkeypatch.setattr(journal, '_atomic_write_text', forbidden)
    with pytest.raises(PermissionError) as caught:
        journal._relocate_cache('news', parent)
    assert caught.value.errno == errno.EACCES and attempted == [(source, target)]
    assert {path.name: path.read_bytes() for path in source.iterdir()} == source_before
    assert record.read_bytes() == manifest_before
    assert {env: os.environ.get(env) for env in env_before} == env_before
    if existing_target:
        assert target.is_dir() and list(target.iterdir()) == []
        assert list(parent.iterdir()) == [target]
        assert stat.S_IMODE(target.stat().st_mode) == original_mode
    else:
        assert not target.exists() and list(parent.iterdir()) == []


@pytest.mark.parametrize('rollback', ['restored', 'denied', 'target-reappeared'])
def test_metadata_failure_preserves_preexisting_private_destination(location, monkeypatch, rollback):
    import json
    import stat

    source, parent, record = location
    target = parent / 'spacr-news'
    target.mkdir(mode=0o700, parents=True)
    mode_before = stat.S_IMODE(target.stat().st_mode)
    record.write_text(json.dumps({'SPACR_MODEL_CACHE': str(source.parent / 'other-models')}) + '\n')
    manifest_before = record.read_bytes()
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    env_before = os.environ['SPACR_NEWS_CACHE']
    rename = Path.rename
    def rollback_move(path, destination):
        if path != target or Path(destination) != source:
            return rename(path, destination)
        if rollback == 'denied':
            raise PermissionError(errno.EACCES, 'owned fixture rollback denied')
        result = rename(path, destination)
        if rollback == 'target-reappeared':
            target.mkdir(mode=0o700)
            (target / 'concurrent-private.bin').write_bytes(b'concurrent destination must survive')
        return result
    def failed_metadata(*args):
        assert not source.exists()
        assert {path.name: path.read_bytes() for path in target.iterdir()} == source_before
        raise OSError('owned fixture metadata write failed')
    monkeypatch.setattr(Path, 'rename', rollback_move)
    monkeypatch.setattr(journal, '_atomic_write_text', failed_metadata)
    with pytest.raises(OSError, match='metadata write failed'):
        journal._relocate_cache('news', parent)
    assert record.read_bytes() == manifest_before and os.environ['SPACR_NEWS_CACHE'] == env_before
    if rollback == 'denied':
        assert not source.exists()
        assert {path.name: path.read_bytes() for path in target.iterdir()} == source_before
    else:
        assert {path.name: path.read_bytes() for path in source.iterdir()} == source_before
        assert target.is_dir()
        if rollback == 'target-reappeared':
            assert [path.name for path in target.iterdir()] == ['concurrent-private.bin']
            assert (target / 'concurrent-private.bin').read_bytes() == b'concurrent destination must survive'
        else:
            assert list(target.iterdir()) == []
            restored_mode = stat.S_IMODE(target.stat().st_mode)
            assert restored_mode & ~mode_before == 0
            if os.name != 'nt':
                assert mode_before == 0o700 and restored_mode & 0o077 == 0
