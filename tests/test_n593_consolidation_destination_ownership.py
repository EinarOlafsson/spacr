"""Consolidation never writes or failure-cleans another writer's image bytes."""
from pathlib import Path

import pytest

from spacr import folder_consolidation as fc


def source_tree(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    original = source / 'image.tif'
    original.write_bytes(b'original microscopy bytes')
    return source, original


@pytest.mark.parametrize('arrival', ['file', 'symlink', 'broken_symlink'])
def test_arrival_after_name_selection_is_preserved(tmp_path, monkeypatch, arrival):
    source, original = source_tree(tmp_path)
    outside = tmp_path / 'outside.tif'
    if arrival != 'broken_symlink':
        outside.write_bytes(b'private outside image')
    real_available = fc.available_filename
    selected = []

    def available(*args, **kwargs):
        target = real_available(*args, **kwargs)
        selected.append(target)
        if arrival == 'file':
            target.write_bytes(b'private arriving image')
        else:
            try:
                target.symlink_to(outside)
            except OSError as error:
                pytest.skip(str(error))
        return target

    monkeypatch.setattr(fc, 'available_filename', available)
    result = fc.consolidate_folder(source, tmp_path / 'out', log=lambda *_: None)
    assert (result.copied, result.failed) == (0, 1)
    assert original.read_bytes() == b'original microscopy bytes'
    target = selected[0]
    if arrival == 'file':
        assert target.read_bytes() == b'private arriving image'
    else:
        assert target.is_symlink()
        assert target.readlink() == outside
    if arrival == 'broken_symlink':
        assert not outside.exists()
    else:
        assert outside.read_bytes() == b'private outside image'
    assert result.rows[0][1:3] == ('', 'error')


@pytest.mark.parametrize('replacement', ['file', 'symlink'])
def test_failed_copy_never_cleans_a_replaced_destination(tmp_path, monkeypatch, replacement):
    source, original = source_tree(tmp_path)
    outside = tmp_path / 'outside.tif'
    outside.write_bytes(b'private outside image')
    selected = []

    def replace_then_fail(target):
        target = Path(target)
        target.unlink(missing_ok=True)
        if replacement == 'file':
            target.write_bytes(b'private replacement image')
        else:
            try:
                target.symlink_to(outside)
            except OSError as error:
                pytest.skip(str(error))
        selected.append(target)
        raise OSError('interrupted image copy')

    def legacy_copy(original_path, target):
        Path(target).write_bytes(b'partial owned copy')
        replace_then_fail(target)

    def fd_copy(incoming, outgoing, *args, **kwargs):
        outgoing.write(b'partial owned copy')
        outgoing.flush()
        replace_then_fail(outgoing.name)

    monkeypatch.setattr(fc.shutil, 'copy2', legacy_copy)
    monkeypatch.setattr(fc.shutil, 'copyfileobj', fd_copy)
    result = fc.consolidate_folder(source, tmp_path / 'out', log=lambda *_: None)
    assert (result.copied, result.failed) == (0, 1)
    target = selected[0]
    assert target.is_symlink() if replacement == 'symlink' else target.is_file()
    assert target.read_bytes() == (b'private outside image' if replacement == 'symlink' else b'private replacement image')
    assert outside.read_bytes() == b'private outside image'
    assert original.read_bytes() == b'original microscopy bytes'
    assert result.rows[0][1:3] == ('', 'error')


def test_arriving_file_is_not_deleted_when_copy_errors_before_open(tmp_path, monkeypatch):
    source, original = source_tree(tmp_path)
    real_available = fc.available_filename
    selected = []

    def available(*args, **kwargs):
        target = real_available(*args, **kwargs)
        target.write_bytes(b'private arriving image')
        selected.append(target)
        return target

    def failed(*args, **kwargs):
        raise OSError('copy refused before write')

    monkeypatch.setattr(fc, 'available_filename', available)
    monkeypatch.setattr(fc.shutil, 'copy2', failed)
    monkeypatch.setattr(fc.shutil, 'copyfileobj', failed)
    result = fc.consolidate_folder(source, tmp_path / 'out', log=lambda *_: None)
    assert (result.copied, result.failed) == (0, 1)
    assert selected[0].read_bytes() == b'private arriving image'
    assert original.read_bytes() == b'original microscopy bytes'


def test_replacement_at_metadata_seam_preserves_bytes_and_is_not_accepted(tmp_path, monkeypatch):
    source, original = source_tree(tmp_path)
    selected = []
    original_copystat = fc.shutil.copystat

    def replaced_metadata(source_path, target_path, *args, **kwargs):
        target = Path(target_path)
        target.unlink()
        target.write_bytes(b'private metadata-race replacement')
        selected.append(target)
        return original_copystat(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(fc.shutil, 'copystat', replaced_metadata)
    result = fc.consolidate_folder(source, tmp_path / 'out', log=lambda *_: None)
    assert (result.copied, result.failed) == (0, 1)
    assert selected[0].read_bytes() == b'private metadata-race replacement'
    assert original.read_bytes() == b'original microscopy bytes'
    assert result.rows[0][1:3] == ('', 'error')
