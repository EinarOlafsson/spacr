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
    target.mkdir(parents=True)

    def fail_copy(*args, **kwargs):
        raise OSError('controlled copy failure')

    monkeypatch.setattr(shutil, 'copy2', fail_copy)
    with pytest.raises(shutil.Error):
        journal._relocate_cache('news', parent)
    assert target.is_dir() and not list(target.iterdir())
    assert list(parent.iterdir()) == [target]
    assert (source / 'first.json').read_bytes() == b'complete-first-cache-entry'
    assert not record.exists()
