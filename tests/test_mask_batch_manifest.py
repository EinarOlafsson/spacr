"""Prepared batches must have exclusive, verifiable output ownership."""

import hashlib
import json

import numpy as np
import pytest

from spacr.io import _mask_batch_manifest


def _archive(root, name, fields, *, shape=None):
    path = root / name
    np.savez_compressed(path, data=np.zeros(shape or (len(fields), 8, 9, 2),
                                          dtype=np.float32), filenames=np.asarray(fields))
    return path


def test_manifest_names_each_field_and_hashes_the_actual_archive(tmp_path, monkeypatch):
    second = _archive(tmp_path, 'b.npz', ['field3.npy'])
    first = _archive(tmp_path, 'a.npz', ['field1.npy', 'field2.npy'])
    (tmp_path / '._ignored.npz').write_bytes(b'not an archive')
    before = {path.name: path.read_bytes() for path in (first, second)}
    original = np.lib.format.read_array

    def no_pixels(stream, *args, **kwargs):
        assert getattr(stream, 'name', None) != 'data.npy'
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, 'read_array', no_pixels)
    result = _mask_batch_manifest(tmp_path)
    assert json.loads(json.dumps(result)) == result
    assert [record['fields'] for record in result] == [
        ['field1.npy', 'field2.npy'], ['field3.npy']]
    for record, path in zip(result, (first, second)):
        assert record['path'] == str(path)
        assert record['sha256'] == hashlib.sha256(before[path.name]).hexdigest()
        assert record['bytes'] == len(before[path.name])
        assert record['planes'] == 2
        assert path.read_bytes() == before[path.name]


@pytest.mark.parametrize('same_archive', [False, True])
def test_duplicate_field_ownership_is_rejected(tmp_path, same_archive):
    _archive(tmp_path, 'a.npz', ['field.npy', 'field.npy'] if same_archive else ['field.npy'])
    if not same_archive:
        _archive(tmp_path, 'b.npz', ['field.npy'])
    with pytest.raises(ValueError, match='belongs to both'):
        _mask_batch_manifest(tmp_path)


@pytest.mark.parametrize('name', ['../outside.npy', '/outside.npy',
                                 'folder\\outside.npy', 'C:outside.npy',
                                 '.hidden.npy', 'field.tif', '', 'a\x00b.npy'])
def test_output_filenames_cannot_escape_or_hide_from_resume(tmp_path, name):
    _archive(tmp_path, 'batch.npz', [name])
    with pytest.raises(ValueError, match='unsafe field filename'):
        _mask_batch_manifest(tmp_path)


def test_time_axis_manifest_uses_the_declared_axis(tmp_path):
    _archive(tmp_path, 'batch.npz', ['t0.npy', 't1.npy', 't2.npy'],
             shape=(2, 3, 8, 9, 1))
    with pytest.raises(ValueError, match='filenames count'):
        _mask_batch_manifest(tmp_path)
    assert _mask_batch_manifest(tmp_path, field_axis=1)[0]['fields'] == [
        't0.npy', 't1.npy', 't2.npy']


def test_missing_batches_are_not_a_successful_empty_plan(tmp_path):
    with pytest.raises(FileNotFoundError, match='No prepared NPZ'):
        _mask_batch_manifest(tmp_path)


def test_legacy_pickled_names_cannot_establish_worker_ownership(tmp_path):
    np.savez(tmp_path / 'batch.npz', data=np.zeros((1, 8, 9, 1)),
             filenames=np.array(['field.npy'], dtype=object))
    with pytest.raises(ValueError, match='unreadable field identities'):
        _mask_batch_manifest(tmp_path)


def test_replaced_archive_during_inspection_is_refused(tmp_path, monkeypatch):
    from spacr import io

    _archive(tmp_path, 'batch.npz', ['field.npy'])
    inspect = io._inspect_normalized_archive

    def replace_after_reading(path, **kwargs):
        result = inspect(path, **kwargs)
        replacement = _archive(tmp_path, 'replacement.npz', ['other.npy'])
        replacement.replace(path)
        return result

    monkeypatch.setattr(io, '_inspect_normalized_archive', replace_after_reading)
    with pytest.raises(ValueError, match='changed while building'):
        _mask_batch_manifest(tmp_path)
