"""Real archive transport and publication boundaries retain scientific evidence."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

import spacr.io as images
from tests.test_normalized_archive_channel_provenance import _settings, _stack


def _member(array):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), version=(2, 0), allow_pickle=False)
    return stream.getvalue()


def _transport_archive(path, raw, channels, *, provenance=None):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data.npy', _member(np.take(raw, channels, axis=-1)[None]))
        archive.writestr('filenames.npy', _member(['plate1_A01_1.npy']))
        archive.writestr('source_channels.npy',
                         _member(np.asarray(channels, dtype=np.int64)) if provenance is None else provenance)


def test_numpy_v2_members_keep_nonascending_channel_identity_without_decoding_pixels(tmp_path, monkeypatch):
    stack, raw = _stack(tmp_path)
    masks = tmp_path / 'masks'
    masks.mkdir()
    path = masks / 'stack_0_norm.npz'
    _transport_archive(path, raw, [3, 2])
    before = path.read_bytes()
    source = (stack / 'plate1_A01_1.npy').read_bytes()
    load = images.np.load
    loads = []

    def observed_load(*args, **kwargs):
        location = getattr(args[0], 'name', args[0])
        if (isinstance(location, (str, Path)) and
                (Path(location).absolute().is_relative_to(tmp_path) or str(location) == 'data.npy')):
            loads.append(location)
        return load(*args, **kwargs)

    monkeypatch.setattr(images.np, 'load', observed_load)
    assert images._check_archives_without_preprocessing(str(tmp_path), settings=_settings(tmp_path)) == [path.name]
    assert images._validate_normalized_source_channels(path, [3, 2]) == [3, 2]
    assert loads == []
    assert path.read_bytes() == before and (stack / 'plate1_A01_1.npy').read_bytes() == source


@pytest.mark.parametrize('damage', ['oversized', 'truncated'])
def test_external_channel_metadata_is_bounded_and_refused_without_changing_scientific_files(tmp_path, damage):
    stack, raw = _stack(tmp_path)
    masks = tmp_path / 'masks'
    masks.mkdir()
    path = masks / 'stack_0_norm.npz'
    metadata = (_member(np.arange(8192, dtype=np.int64)) if damage == 'oversized'
                else _member(np.array([3, 2], dtype=np.int64))[:-8])
    _transport_archive(path, raw, [3, 2], provenance=metadata)
    before = path.read_bytes()
    source = (stack / 'plate1_A01_1.npy').read_bytes()
    with pytest.raises(ValueError, match='channel order is unknown'):
        images._check_archives_without_preprocessing(str(tmp_path), settings=_settings(tmp_path))
    assert path.read_bytes() == before and (stack / 'plate1_A01_1.npy').read_bytes() == source
    assert not list(tmp_path.glob('previous_segmentation_channel_order_*'))


@pytest.mark.parametrize('evidence', ['none', 'masks-only', 'merged-only'])
def test_failed_channel_publication_restores_exactly_the_outputs_that_existed(tmp_path, monkeypatch, evidence):
    stack, _ = _stack(tmp_path)
    settings = _settings(tmp_path)
    images.concatenate_and_normalize(str(stack), [2, 3], settings=settings)
    archive, = (tmp_path / 'masks').glob('*.npz')
    before = archive.read_bytes()
    source = (stack / 'plate1_A01_1.npy').read_bytes()
    previous = None
    if evidence != 'none':
        folder = tmp_path / ('masks/cell_mask_stack' if evidence == 'masks-only' else 'merged')
        folder.mkdir()
        previous = folder / 'previous.npy'
        np.save(previous, np.arange(12, dtype=np.uint16).reshape(3, 4))
        previous_bytes = previous.read_bytes()
    replace = images.os.replace
    failed = []

    def fail_publication(source_path, destination):
        if '.spacr_v1_npz_' in str(source_path) and str(source_path).endswith('.npz') and not failed:
            failed.append(str(source_path))
            raise OSError('publication boundary unavailable')
        return replace(source_path, destination)

    monkeypatch.setattr(images.os, 'replace', fail_publication)
    with pytest.raises(OSError, match='publication boundary unavailable'):
        images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    assert failed and archive.read_bytes() == before
    assert (stack / 'plate1_A01_1.npy').read_bytes() == source
    if previous is not None:
        assert previous.read_bytes() == previous_bytes
    assert not list(tmp_path.glob('previous_segmentation_channel_order_*'))
    assert not list(tmp_path.glob('.spacr_v1_npz_*'))
    if evidence == 'merged-only':
        removed, merged = images._invalidate_v1_segmentation_outputs(str(tmp_path))
        assert removed == () and merged is True and not previous.parent.exists()
        assert archive.read_bytes() == before
        assert (stack / 'plate1_A01_1.npy').read_bytes() == source


@pytest.mark.parametrize('replacement', ['missing-channels', 'incomplete-header'])
def test_stack_replaced_after_initial_checks_cannot_rebuild_unknown_channel_order(tmp_path, monkeypatch, replacement):
    stack, raw = _stack(tmp_path)
    source = stack / 'plate1_A01_1.npy'
    original = source.read_bytes()
    masks = tmp_path / 'masks'
    masks.mkdir()
    archive = masks / 'stack_0_norm.npz'
    np.savez(archive, data=raw[None, ..., :2], filenames=[source.name],
             source_channels=np.array([2, 3], dtype=np.int64))
    archive_bytes = archive.read_bytes()
    original_stack = tmp_path / 'original-stack.npy'
    checked = images._check_normalized_archives
    changed = []

    def check_then_replace(*args, **kwargs):
        actual = checked(*args, **kwargs)
        if not changed:
            source.rename(original_stack)
            if replacement == 'missing-channels':
                np.save(source, raw[..., :2])
            else:
                source.write_bytes(b'\x93NUMPY\x01')
            changed.append(source.read_bytes())
        return actual

    monkeypatch.setattr(images, '_check_normalized_archives', check_then_replace)
    with pytest.raises(ValueError, match='original image stacks are unavailable'):
        images._resume_normalized_archives(_settings(tmp_path), str(tmp_path), [3, 2])
    assert changed and source.read_bytes() == changed[0]
    assert original_stack.read_bytes() == original and archive.read_bytes() == archive_bytes
    assert sorted(path.name for path in masks.iterdir()) == [archive.name]


@pytest.mark.parametrize('replace_archive', [False, True])
def test_missing_field_resume_respects_the_observed_archive_channel_count(tmp_path, monkeypatch, capsys, replace_archive):
    stack, raw = _stack(tmp_path)
    second = stack / 'plate1_A01_2.npy'
    np.save(second, raw + 7)
    source_bytes = {path: path.read_bytes() for path in stack.iterdir()}
    masks = tmp_path / 'masks'
    masks.mkdir()
    archive = masks / 'stack_0_norm.npz'
    channels = [0, 1, 2] if replace_archive else [0, 1]
    _transport_archive(archive, raw, channels)
    original = archive.read_bytes()
    retained = tmp_path / 'previous-archive.npz'
    replacement = tmp_path / 'replacement-archive.npz'
    _transport_archive(replacement, raw, [0, 1])
    replacement_bytes = replacement.read_bytes()
    checked = images._check_normalized_archives
    changes = []

    def check_then_replace(*args, **kwargs):
        actual = checked(*args, **kwargs)
        if replace_archive and not changes:
            archive.rename(retained)
            images.os.replace(replacement, archive)
            changes.append(True)
        return actual

    monkeypatch.setattr(images, '_check_normalized_archives', check_then_replace)
    settings = _settings(tmp_path, nucleus_channel=0, cell_channel=1, pathogen_channel=None)
    assert images._resume_normalized_archives(settings, str(tmp_path), [0, 1]) is True
    assert all(path.read_bytes() == value for path, value in source_bytes.items())
    if replace_archive:
        assert changes and retained.read_bytes() == original
        assert archive.read_bytes() == replacement_bytes
        assert sorted(path.name for path in masks.iterdir()) == [archive.name]
        assert 'so none is added' in capsys.readouterr().out
    else:
        assert archive.read_bytes() == original
        appended, = (path for path in masks.glob('*.npz') if path != archive)
        assert images._validate_normalized_source_channels(appended, [0, 1]) == [0, 1]
        with np.load(appended, allow_pickle=False) as values:
            assert values['filenames'].tolist() == [second.name]
            assert values['data'].shape[-1] == 2
