"""Raw fields cross the readiness boundary only as verified private snapshots."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import core
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
from tests.test_watch_folder_and_analyse import Recorder, _name
from tests.test_watch_nested_f548 import _fast, _write


def _ledger(folder):
    """Read every persisted field, including deferred copies and real failures."""
    return json.loads((folder / 'spacr_watch/watch_ledger.json').read_text())['fields']


def _change_pixels(path, value, replace=False):
    """Write real TIFF pixels while deliberately preserving size and mtime."""
    before = path.stat()
    target = path.with_suffix('.replacement') if replace else path
    tifffile.imwrite(target, np.full((16, 16), value, np.uint16))
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    if replace:
        os.replace(target, path)
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize('nested', [False, True])
def test_unchanged_private_copy_preserves_pixels_and_binds_sha256(tmp_path, nested):
    folder = tmp_path / 'plate/well' if nested else tmp_path
    paths = [_write(folder, _name('A01', c), c * 11) for c in (1, 2)]
    before = {path: path.read_bytes() for path in paths}
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert len(result['done']) == len(recorder.calls) == 1
    entry = _ledger(tmp_path)[result['done'][0]]
    assert set(entry['snapshot_sha256']) == {str(p.relative_to(tmp_path)) for p in paths}
    for path, content in before.items():
        name = str(path.relative_to(tmp_path))
        assert entry['snapshot_sha256'][name] == hashlib.sha256(content).hexdigest()
        assert entry['source_identity'][name] == core._watch_file_identity(path)
        staged = tmp_path / 'spacr_watch/fields' / result['done'][0] / path.name
        assert staged.read_bytes() == content == path.read_bytes()
    assert len(core._watch_folder_and_analyse(_fast(tmp_path), recorder)['done']) == 1
    assert len(recorder.calls) == 1


@pytest.mark.parametrize('replacement', [False, True])
def test_mutation_during_copy_retries_without_analyzing_stale_bytes(tmp_path, monkeypatch, replacement):
    paths = [_write(tmp_path, _name('A01', c), c) for c in (1, 2)]
    target = paths[0]
    wanted = target.stat().st_ino
    original_read = os.read
    changed = []

    def read_then_change(descriptor, count):
        """Change a real source after one bounded read from its open descriptor."""
        data = original_read(descriptor, count)
        if not changed and os.fstat(descriptor).st_ino == wanted:
            assert count <= 1024 * 1024
            changed.append(True)
            _change_pixels(target, 99, replace=replacement)
        return data

    monkeypatch.setattr(os, 'read', read_then_change)
    analysed = []

    def analyse(folder, settings):
        """Record the actual staged pixels before creating normal test outputs."""
        analysed.append([int(tifffile.imread(Path(folder) / p.name)[0, 0]) for p in paths])
        Recorder()(folder, settings)

    result = core._watch_folder_and_analyse(_fast(tmp_path), analyse)
    assert changed and analysed == [[99, 2]]
    entry = _ledger(tmp_path)[result['done'][0]]
    assert entry['snapshot_sha256'][target.name] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert entry['status'] == 'done' and not result['failed']


def test_previous_companion_change_during_later_copy_rechecks_whole_field(tmp_path, monkeypatch):
    folder = tmp_path / 'nested'
    paths = [_write(folder, _name('A01', c), c) for c in (1, 2)]
    original_copy = core._watch_copy_snapshot
    changed = []

    def copy_then_change_previous(source, destination, expected):
        """Invalidate the first companion only after the second is copied."""
        digest = original_copy(source, destination, expected)
        if Path(source).name == paths[1].name and not changed:
            changed.append(True)
            _change_pixels(paths[0], 77)
        return digest

    monkeypatch.setattr(core, '_watch_copy_snapshot', copy_then_change_previous)
    observed = []

    def analyse(directory, settings):
        """Only the second coherent snapshot may reach analysis."""
        observed.append(int(tifffile.imread(Path(directory) / paths[0].name)[0, 0]))
        Recorder()(directory, settings)

    result = core._watch_folder_and_analyse(_fast(tmp_path), analyse)
    assert changed and observed == [77] and len(result['done']) == 1


def test_deleted_only_companion_remains_waiting_without_staging(tmp_path, monkeypatch):
    path = _write(tmp_path, _name('A01', 1))
    original_copy = core._watch_copy_snapshot

    def copy_then_delete(source, destination, expected):
        """Delete the acquired field at the end of its staging copy."""
        digest = original_copy(source, destination, expected)
        Path(source).unlink()
        return digest

    monkeypatch.setattr(core, '_watch_copy_snapshot', copy_then_delete)
    recorder = Recorder()
    result = core._watch_folder_and_analyse(dict(_fast(tmp_path), channels=[0]), recorder)
    assert not recorder.calls and not result['done'] and not result['failed']
    assert result['incomplete'] == ['plate1_A01_0001_001']
    entry = _ledger(tmp_path)[result['incomplete'][0]]
    assert entry['status'] == 'waiting' and 'snapshot_sha256' not in entry
    assert not path.exists()
    assert not (tmp_path / 'spacr_watch/fields/plate1_A01_0001_001').exists()


def test_destination_failure_stays_failed_and_retries_on_next_start(tmp_path, monkeypatch):
    import builtins

    paths = [_write(tmp_path, _name('A01', c), c) for c in (1, 2)]
    before = {path: path.read_bytes() for path in paths}
    recorder = Recorder()
    original_open = builtins.open

    def fail_staging_write(path, mode='r', *args, **kwargs):
        """Simulate an actual destination error without changing source inputs."""
        if mode == 'xb' and str(path).endswith('.tif'):
            raise OSError('simulated destination full')
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(core, 'open', fail_staging_write, raising=False)
        result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert result['failed'] == ['plate1_A01_0001_001'] and not recorder.calls
    assert 'destination full' in _ledger(tmp_path)[result['failed'][0]]['error']
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert len(result['done']) == len(recorder.calls) == 1 and not result['failed']
    assert all(path.read_bytes() == content for path, content in before.items())


def test_large_copy_is_chunked_and_cancelled_before_pipeline(tmp_path, monkeypatch):
    path = tmp_path / _name('A01', 1)
    tifffile.imwrite(path, np.ones((1024, 1024), np.uint16))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    wanted = path.stat().st_ino
    original_read = os.read
    reads = []
    token = CancellationToken()

    def read_then_stop(descriptor, count):
        """Stop after the first chunk; no second source chunk may be read."""
        data = original_read(descriptor, count)
        if os.fstat(descriptor).st_ino == wanted:
            reads.append(count)
            token.cancel()
        return data

    monkeypatch.setattr(os, 'read', read_then_stop)
    recorder = Recorder()
    with installed_token(token), pytest.raises(PipelineCancelled):
        core._watch_folder_and_analyse(dict(_fast(tmp_path), channels=[0]), recorder)
    assert reads == [1024 * 1024] and not recorder.calls
    entry = _ledger(tmp_path)['plate1_A01_0001_001']
    assert entry['status'] == 'interrupted' and 'snapshot_sha256' not in entry
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_new_inode_with_preserved_size_and_mtime_restarts_settle_observation(tmp_path):
    path = _write(tmp_path, _name('A01', 1), 1)
    context = {'src': str(tmp_path), 'seen': {}}
    assert core._watch_observe(context, 1.0)
    before = dict(context['seen'][path.name])
    _change_pixels(path, 9, replace=True)
    assert core._watch_observe(context, 2.0)
    after = context['seen'][path.name]
    assert before['signature'] == after['signature']
    assert before['identity'] != after['identity']
    assert after['changed'] == 2.0 and after['first'] == 1.0
    assert not after['readable']


def test_source_symlink_swap_at_open_does_not_read_external_target(tmp_path, monkeypatch):
    source = _write(tmp_path, _name('A01', 1))
    external = _write(tmp_path / 'elsewhere', 'external.tif', 99)
    expected = core._watch_file_identity(source)
    before = external.read_bytes()
    target = tmp_path / 'staged.tif'
    original_open = os.open

    def swap_then_open(path, flags, *args, **kwargs):
        """Swap the already checked source for a link at the open boundary."""
        if Path(path) == source:
            source.unlink()
            source.symlink_to(external)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', swap_then_open)
    assert core._watch_copy_snapshot(source, target, expected) is None
    assert not target.exists() and external.read_bytes() == before


def test_growing_source_copy_never_reads_beyond_observed_size_plus_one(tmp_path, monkeypatch):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'a' * 64)
    target = tmp_path / 'staged.bin'
    expected = core._watch_file_identity(source)
    original_read = os.read
    read_sizes = []

    def read_then_append(descriptor, count):
        """Keep appending to demonstrate that snapshot reads have a fixed ceiling."""
        result = original_read(descriptor, count)
        if os.fstat(descriptor).st_ino == expected[1]:
            read_sizes.append(len(result))
            with source.open('ab') as handle:
                handle.write(b'b' * 64)
        return result

    monkeypatch.setattr(os, 'read', read_then_append)
    assert core._watch_copy_snapshot(source, target, expected) is None
    assert sum(read_sizes) == 65 and target.stat().st_size == 64


@pytest.mark.parametrize('replacement', ['symlink', 'directory'])
def test_replacement_between_listing_and_stat_discards_stale_observation(tmp_path, monkeypatch, replacement):
    folder = tmp_path / 'watched'
    path = _write(folder, _name('A01', 1), 1)
    external = _write(tmp_path / 'outside', 'external.tif', 99)
    context = {'src': str(folder), 'seen': {}}
    assert core._watch_observe(context, 1.0)
    original_images = core._watch_images

    def list_then_replace(src):
        """Invalidate a regular filename after discovery but before its stat check."""
        names = original_images(src)
        path.unlink()
        if replacement == 'symlink':
            path.symlink_to(external)
        else:
            path.mkdir()
        return names

    monkeypatch.setattr(core, '_watch_images', list_then_replace)
    assert core._watch_observe(context, 2.0)
    assert not context['seen']
