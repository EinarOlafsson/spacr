"""Optional cache cleanup refuses links, unknown layouts, budgets and concurrent changes."""
import json
import os
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from spacr import install_cleanup as ic
from tests.test_n703_guarded_optional_cleanup import caches, fresh, request  # noqa: F401

SOUND = 'chime-0123456789ab'


def _open_dir(path, dir_fd=None):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY, dir_fd=dir_fd)


def _tree(path, key):
    descriptor = _open_dir(str(path))
    try:
        return ic._optional_cleanup_tree(descriptor, key)
    finally:
        os.close(descriptor)


@pytest.mark.posix_only
def test_directory_walk_requires_no_follow_support(monkeypatch, tmp_path):
    monkeypatch.delattr(ic.os, 'O_NOFOLLOW')
    with pytest.raises(ValueError, match='no_nofollow_directory_support'):
        ic._optional_cleanup_directory(str(tmp_path))


@pytest.mark.posix_only
def test_directory_walk_releases_its_descriptor_on_a_missing_component(tmp_path):
    before = len(os.listdir('/proc/self/fd'))
    with pytest.raises(FileNotFoundError):
        ic._optional_cleanup_directory(str(tmp_path / 'missing' / 'cache'))
    assert len(os.listdir('/proc/self/fd')) == before


@pytest.mark.posix_only
def test_tree_refuses_more_than_the_entry_budget(tmp_path):
    for index in range(515):
        (tmp_path / ('icon%03d-0123456789abcdef0123.png' % index)).write_bytes(b'i')
    with pytest.raises(ValueError, match='budget'):
        _tree(tmp_path, 'icons')


@pytest.mark.posix_only
def test_tree_refuses_an_unknown_sound_folder(tmp_path):
    (tmp_path / 'not a sound pack').mkdir()
    with pytest.raises(ValueError, match='layout'):
        _tree(tmp_path, 'sounds')


@pytest.mark.posix_only
def test_tree_refuses_an_unknown_file_inside_a_sound_pack(tmp_path):
    (tmp_path / SOUND).mkdir()
    (tmp_path / SOUND / 'notes.txt').write_bytes(b'x')
    with pytest.raises(ValueError, match='layout'):
        _tree(tmp_path, 'sounds')


@pytest.mark.posix_only
def test_tree_refuses_a_sound_budget_reached_inside_a_pack(tmp_path):
    names = ['click-%d.wav' % i for i in range(4)] + ['hover-%d.wav' % i for i in range(4)] + [
        'run_finished.wav', 'run_failed.wav', 'bed.wav']
    for pack in range(43):
        folder = tmp_path / ('pack%02d-0123456789ab' % pack)
        folder.mkdir()
        for name in names:
            (folder / name).write_bytes(b'w')
    with pytest.raises(ValueError, match='budget'):
        _tree(tmp_path, 'sounds')


@pytest.mark.posix_only
def test_tree_lists_a_valid_sound_pack(tmp_path):
    (tmp_path / SOUND).mkdir()
    (tmp_path / SOUND / 'bed.wav').write_bytes(b'wave')
    _root, entries, contents, total = _tree(tmp_path, 'sounds')
    assert [name for name, _details in entries] == ['', SOUND, SOUND + '/bed.wav']
    assert [dict(row) for row in contents] == [{'name': SOUND + '/bed.wav', 'size_bytes': 4}]
    assert total == 4


@pytest.mark.posix_only
def test_tree_refuses_a_sound_pack_replaced_while_opened(tmp_path, monkeypatch):
    (tmp_path / SOUND).mkdir()
    real_fstat = os.fstat
    calls = []

    def fstat(descriptor):
        details = real_fstat(descriptor)
        calls.append(descriptor)
        if len(calls) == 2:
            return types.SimpleNamespace(st_mode=details.st_mode, st_dev=details.st_dev,
                                         st_ino=details.st_ino + 1, st_size=details.st_size,
                                         st_mtime_ns=details.st_mtime_ns, st_ctime_ns=details.st_ctime_ns)
        return details

    proxy = types.SimpleNamespace(**vars(os))
    proxy.fstat = fstat
    monkeypatch.setattr(ic, 'os', proxy)
    with pytest.raises(ValueError, match='changed'):
        _tree(tmp_path, 'sounds')


@pytest.fixture
def home(tmp_path):
    base = tmp_path / 'home' / '.spacr'
    (base / 'news').mkdir(parents=True)
    (base / 'news' / 'releases.json').write_bytes(b'{"releases": []}')
    machine = ic._Machine(environ={'HOME': str(tmp_path / 'home')}, platform='linux')
    return types.SimpleNamespace(base=base, machine=machine)


def _row(preview, key):
    return next(row for row in preview['folders'] if row['key'] == key)


@pytest.mark.posix_only
@pytest.mark.parametrize('kind', ['list', 'fifo', 'oversized'])
def test_unusable_relocation_manifest_is_ignored(home, kind):
    manifest = home.base / ic._CACHE_LOCATIONS_FILE
    if kind == 'list':
        manifest.write_text(json.dumps(['SPACR_NEWS_CACHE', '/elsewhere']), encoding="utf-8")
    elif kind == 'fifo':
        os.mkfifo(manifest)
    else:
        manifest.write_text(json.dumps({'SPACR_NEWS_CACHE': 'x' * (1024 * 1024 + 8)}), encoding="utf-8")
    preview = ic.preview_optional_cleanup('manifest-op', system=home.machine)
    news = _row(preview, 'news')
    assert news['path'] == str(home.base / 'news')
    assert news['reason'] == 'coordinator_unavailable' and news['size_bytes'] == 16


def test_valid_relocation_manifest_marks_the_cache_external(home, tmp_path):
    (home.base / ic._CACHE_LOCATIONS_FILE).write_text(json.dumps({'SPACR_NEWS_CACHE': str(tmp_path / 'news')}), encoding="utf-8")
    news = _row(ic.preview_optional_cleanup('manifest-op', system=home.machine), 'news')
    assert news['path'] == str(tmp_path / 'news') and news['reason'] == 'unverified'


def test_cache_root_replaced_during_inspection_is_unverified(home, monkeypatch):
    real_tree = ic._optional_cleanup_tree

    def tree(descriptor, key):
        identity, entries, contents, total = real_tree(descriptor, key)
        return (identity[0], identity[1] + 1) + identity[2:], entries, contents, total

    monkeypatch.setattr(ic, '_optional_cleanup_tree', tree)
    news = _row(ic.preview_optional_cleanup('race-op', system=home.machine), 'news')
    assert news['reason'] == 'unverified' and news['size_bytes'] is None and news['contents'] == ()


def test_spacr_home_changed_during_inspection_is_unverified(home, monkeypatch):
    real_tree = ic._optional_cleanup_tree

    def tree(descriptor, key):
        result = real_tree(descriptor, key)
        (home.base / ('new-entry-' + key)).write_bytes(b'concurrent')
        return result

    monkeypatch.setattr(ic, '_optional_cleanup_tree', tree)
    news = _row(ic.preview_optional_cleanup('race-op', system=home.machine), 'news')
    assert news['reason'] == 'unverified' and not news['selectable']


@pytest.mark.posix_only
def test_failing_guard_leaves_every_cache_unselectable(home):
    @contextmanager
    def guard(paths):
        raise RuntimeError('modeled guard failure')
        yield

    preview = ic.preview_optional_cleanup('guard-op', system=home.machine, guard=guard)
    news = _row(preview, 'news')
    assert news['reason'] == 'coordinator_unavailable' and not news['selectable']


@pytest.fixture
def held(home):
    home_fd = _open_dir(str(home.base))
    descriptors = [home_fd]

    def open_root(key):
        descriptor = _open_dir(key, dir_fd=home_fd)
        descriptors.append(descriptor)
        _root, expected, _contents, _size = ic._optional_cleanup_tree(descriptor, key)
        return descriptor, expected

    yield types.SimpleNamespace(home_fd=home_fd, open_root=open_root, base=home.base)
    for descriptor in descriptors:
        os.close(descriptor)


def _private(base):
    found = list(base.glob('.spacr-cleanup-*'))
    assert len(found) == 1
    return found[0]


@pytest.mark.posix_only
def test_remove_refuses_a_root_replaced_before_detachment(held):
    root_fd, expected = held.open_root('news')
    (held.base / 'news').rename(held.base / 'news-old')
    (held.base / 'news').mkdir()
    path = str(held.base / 'news')
    outcome = ic._optional_cleanup_remove('news', held.home_fd, root_fd, expected, path)
    assert outcome == {'key': 'news', 'ok': False, 'reason': 'retained', 'removed': 0, 'retained_path': path}
    assert (held.base / 'news-old' / 'releases.json').is_file()
    assert not list(held.base.glob('.spacr-cleanup-*'))


@pytest.mark.posix_only
def test_remove_keeps_the_cache_when_no_private_folder_can_be_made(held):
    root_fd, expected = held.open_root('news')
    os.chmod(held.base, 0o500)
    try:
        outcome = ic._optional_cleanup_remove('news', held.home_fd, root_fd, expected, str(held.base / 'news'))
    finally:
        os.chmod(held.base, 0o700)
    assert not outcome['ok'] and outcome['retained_path'] == str(held.base / 'news')
    assert (held.base / 'news' / 'releases.json').is_file()


@pytest.mark.posix_only
def test_remove_tolerates_a_vanished_private_folder_after_a_failed_detach(held, monkeypatch):
    root_fd, expected = held.open_root('news')
    real_rename, real_rmdir = os.rename, os.rmdir

    def rename(*args, **kwargs):
        real_rmdir(_private(held.base))
        raise OSError('modeled rename failure')

    monkeypatch.setattr(ic.os, 'rename', rename)
    outcome = ic._optional_cleanup_remove('news', held.home_fd, root_fd, expected, str(held.base / 'news'))
    monkeypatch.setattr(ic.os, 'rename', real_rename)
    assert not outcome['ok'] and outcome['removed'] == 0
    assert (held.base / 'news' / 'releases.json').is_file()
    assert not list(held.base.glob('.spacr-cleanup-*'))


@pytest.mark.posix_only
def test_remove_retains_a_detached_tree_that_differs_from_the_preview(held):
    root_fd, expected = held.open_root('news')
    stale = [(name, details if name != 'releases.json' else details[:3] + (999,) + details[4:])
             for name, details in expected]
    outcome = ic._optional_cleanup_remove('news', held.home_fd, root_fd, tuple(stale), str(held.base / 'news'))
    private = _private(held.base)
    assert not outcome['ok'] and outcome['retained_path'] == str(private)
    assert (private / 'cache' / 'releases.json').read_bytes() == b'{"releases": []}'


def _sound_root(held):
    pack = held.base / 'sounds' / SOUND
    pack.mkdir(parents=True)
    (pack / 'bed.wav').write_bytes(b'wave')
    return held.open_root('sounds')


@pytest.mark.posix_only
def test_remove_retains_a_sound_pack_replaced_after_detachment(held, monkeypatch):
    root_fd, expected = _sound_root(held)
    real_mkdir = os.mkdir

    def mkdir(path, *args, **kwargs):
        if path == 'sounds':
            cache = _private(held.base) / 'cache'
            (cache / SOUND).rename(cache / 'moved')
            real_mkdir(cache / SOUND)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(ic.os, 'mkdir', mkdir)
    outcome = ic._optional_cleanup_remove('sounds', held.home_fd, root_fd, expected, str(held.base / 'sounds'))
    monkeypatch.setattr(ic.os, 'mkdir', real_mkdir)
    assert not outcome['ok'] and outcome['removed'] == 0
    assert (_private(held.base) / 'cache' / 'moved' / 'bed.wav').read_bytes() == b'wave'


@pytest.mark.posix_only
def test_remove_retains_a_sound_pack_replaced_after_its_files_are_removed(held, monkeypatch):
    root_fd, expected = _sound_root(held)
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        result = real_unlink(path, *args, **kwargs)
        if path == 'bed.wav':
            cache = _private(held.base) / 'cache'
            (cache / SOUND).rename(cache / 'moved')
            (cache / SOUND).mkdir()
        return result

    monkeypatch.setattr(ic.os, 'unlink', unlink)
    outcome = ic._optional_cleanup_remove('sounds', held.home_fd, root_fd, expected, str(held.base / 'sounds'))
    monkeypatch.setattr(ic.os, 'unlink', real_unlink)
    assert not outcome['ok'] and outcome['removed'] == 1
    assert (_private(held.base) / 'cache' / SOUND).is_dir()


@pytest.mark.posix_only
def test_remove_retains_a_detached_root_replaced_after_removal(held, monkeypatch):
    root_fd, expected = held.open_root('news')
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        result = real_unlink(path, *args, **kwargs)
        if path == 'releases.json':
            private = _private(held.base)
            (private / 'cache').rename(private / 'moved')
            (private / 'cache').mkdir()
        return result

    monkeypatch.setattr(ic.os, 'unlink', unlink)
    outcome = ic._optional_cleanup_remove('news', held.home_fd, root_fd, expected, str(held.base / 'news'))
    monkeypatch.setattr(ic.os, 'unlink', real_unlink)
    assert not outcome['ok'] and outcome['removed'] == 1
    assert (_private(held.base) / 'cache').is_dir() and (_private(held.base) / 'moved').is_dir()


@pytest.mark.posix_only
def test_remove_retains_a_private_folder_replaced_before_cleanup(held, monkeypatch):
    root_fd, expected = held.open_root('news')
    real_rmdir = os.rmdir

    def rmdir(path, *args, **kwargs):
        result = real_rmdir(path, *args, **kwargs)
        if path == 'cache':
            private = _private(held.base)
            private.rename(held.base / 'moved-private')
            private.mkdir()
        return result

    monkeypatch.setattr(ic.os, 'rmdir', rmdir)
    outcome = ic._optional_cleanup_remove('news', held.home_fd, root_fd, expected, str(held.base / 'news'))
    monkeypatch.setattr(ic.os, 'rmdir', real_rmdir)
    assert not outcome['ok'] and outcome['removed'] == 1
    assert outcome['retained_path'] == str(_private(held.base))
    assert (held.base / 'news').is_dir() and not list((held.base / 'news').iterdir())


@pytest.mark.posix_only
def test_apply_refuses_a_cache_changed_after_its_guarded_reinspection(caches, monkeypatch):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    real_preview = ic.preview_optional_cleanup

    def reinspect(operation_id, *, system=None, guard=None):
        current = real_preview(operation_id, system=system, guard=guard)
        (base / 'icons' / 'late-abcdefabcdefabcdefab.png').write_bytes(b'late icon')
        return current

    monkeypatch.setattr(ic, 'preview_optional_cleanup', reinspect)
    result = ic.apply_optional_cleanup(preview, request(preview, ['icons']), system=machine, guard=guard)
    assert result == {'ok': False, 'reason': 'preview_changed'}
    assert files['icons'][0].read_bytes() == files['icons'][1]
    assert Path(base / 'icons' / 'late-abcdefabcdefabcdefab.png').read_bytes() == b'late icon'


@pytest.mark.posix_only
def test_apply_reports_an_unavailable_coordinator_when_the_guard_fails(caches):
    machine, base, files, state, guard = caches
    preview = fresh(caches)

    @contextmanager
    def failing(paths):
        raise OSError('modeled guard I/O failure')
        yield

    result = ic.apply_optional_cleanup(preview, request(preview, ['news']), system=machine, guard=failing)
    assert result == {'ok': False, 'reason': 'coordinator_unavailable'}
    assert files['news'][0].read_bytes() == files['news'][1]
