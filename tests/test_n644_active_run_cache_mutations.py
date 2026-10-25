"""Cache mutations must wait for worker-thread runs through finalization."""
import importlib
import json
import threading

import pytest

from spacr import run_journal as rj


CAPTION = 'A run is in progress. Try again when it has finished.'


def _unchanged(details):
    """What a refused mutation must not alter; reading a file may move its access time."""
    return (details.st_mode, details.st_ino, details.st_dev, details.st_size,
            details.st_mtime_ns, details.st_ctime_ns)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    state = tmp_path / 'private-spacr'
    state.mkdir()
    folder = state / 'news'
    folder.mkdir()
    (folder / 'kept.json').write_bytes(b'important cache bytes')
    metadata = state / 'cache_locations.json'
    metadata.write_text('{"unrelated": "preserve-me"}', encoding="utf-8")
    monkeypatch.setenv('SPACR_HOME', str(state))
    monkeypatch.delenv('SPACR_NEWS_CACHE', raising=False)
    monkeypatch.setattr(rj, '_spacr_home', lambda: state)
    monkeypatch.setattr(rj, 'runs_root', lambda: tmp_path / 'runs')
    return folder, metadata


@pytest.mark.parametrize('operation', ['clear', 'move'])
@pytest.mark.parametrize('phase', ['body', 'finalization'])
def test_cache_mutation_refuses_active_worker_then_succeeds_after_finalization(cache, tmp_path, monkeypatch, operation, phase):
    folder, metadata = cache
    entered, finish_body, finalizing, finish_finalizing = [threading.Event() for _ in range(4)]
    errors = []
    writer = rj.Run._write_manifest
    def write_manifest(run):
        if run.end_ts is not None and phase == 'finalization':
            finalizing.set()
            assert finish_finalizing.wait(15)
        return writer(run)
    monkeypatch.setattr(rj.Run, '_write_manifest', write_manifest)
    def worker():
        try:
            with rj.open_run('mask', {}):
                entered.set()
                assert finish_body.wait(15)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    before_metadata = metadata.read_bytes()
    before_stat = metadata.stat()
    before_cache = (folder / 'kept.json').stat()
    try:
        assert entered.wait(15)
        assert rj.current_run() is None
        if phase == 'finalization':
            finish_body.set()
            assert finalizing.wait(15)
        if operation == 'clear':
            removed, refused = rj._clear_cache('news')
            assert removed == 0
            assert refused == [CAPTION]
        else:
            with pytest.raises(ValueError, match='run is in progress'):
                rj._relocate_cache('news', tmp_path / 'destination')
        assert _unchanged((folder / 'kept.json').stat()) == _unchanged(before_cache)
        assert _unchanged(metadata.stat()) == _unchanged(before_stat)
        assert (folder / 'kept.json').read_bytes() == b'important cache bytes'
        assert metadata.read_bytes() == before_metadata
        assert not (tmp_path / 'destination').exists()
    finally:
        finish_body.set()
        finish_finalizing.set()
        thread.join(15)
    assert not thread.is_alive()
    assert not errors
    if operation == 'clear':
        assert rj._clear_cache('news') == (1, [])
        assert folder.is_dir() and not list(folder.iterdir())
    else:
        target = rj._relocate_cache('news', tmp_path / 'destination')
        assert (target / 'kept.json').read_bytes() == b'important cache bytes'
        assert not folder.exists()
        assert json.loads(metadata.read_text(encoding="utf-8")) == {'unrelated': 'preserve-me', 'SPACR_NEWS_CACHE': str(target)}


def test_missing_source_relocation_cannot_write_metadata_during_active_run(cache, tmp_path):
    folder, metadata = cache
    (folder / 'kept.json').unlink()
    folder.rmdir()
    before = metadata.read_bytes()
    with rj.open_run('mask', {}):
        with pytest.raises(ValueError, match='run is in progress'):
            rj._relocate_cache('news', tmp_path / 'destination')
        assert metadata.read_bytes() == before
        assert not (tmp_path / 'destination').exists()


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_busy_cache_refusal_uses_existing_shipped_locale(cache, monkeypatch, language):
    from spacr.qt import i18n
    folder, metadata = cache
    translated = importlib.import_module('spacr.qt.i18n_catalogs.' + language).UI[CAPTION]
    assert translated != CAPTION
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    before = metadata.read_bytes()
    with rj.open_run('mask', {}):
        assert rj._clear_cache('news') == (0, [translated])
        assert (folder / 'kept.json').read_bytes() == b'important cache bytes'
        assert metadata.read_bytes() == before


def test_run_registration_waits_until_cache_mutation_finishes_without_reentrant_lock(cache, tmp_path, monkeypatch):
    from pathlib import Path
    folder, _metadata = cache
    clearing, allow_clear, registering, began = [threading.Event() for _ in range(4)]
    errors, result = [], []
    lock = threading.Lock()
    class RegistrationLock:
        def __enter__(self):
            if threading.current_thread().name == 'new-run':
                registering.set()
            lock.acquire()
        def __exit__(self, *_args):
            lock.release()
    monkeypatch.setattr(rj, '_ACTIVE_RUNS_LOCK', RegistrationLock())
    unlink = Path.unlink
    def blocked_unlink(path, *args, **kwargs):
        if path == folder / 'kept.json':
            clearing.set()
            assert allow_clear.wait(15)
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', blocked_unlink)
    def clear():
        try:
            result.append(rj._clear_cache('news'))
        except BaseException as error:
            errors.append(error)
    def run():
        try:
            with rj.open_run('mask', {}):
                began.set()
        except BaseException as error:
            errors.append(error)
    clearer = threading.Thread(target=clear, name='clear-cache')
    starter = threading.Thread(target=run, name='new-run')
    clearer.start()
    try:
        assert clearing.wait(15)
        starter.start()
        assert registering.wait(15)
        assert not began.is_set()
    finally:
        allow_clear.set()
        clearer.join(15)
        if starter.ident is not None:
            starter.join(15)
    assert not clearer.is_alive()
    assert not starter.is_alive()
    assert not errors
    assert began.is_set()
    assert result == [(1, [])]
