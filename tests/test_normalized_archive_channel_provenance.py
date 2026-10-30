"""Normalized image archives bind their final axis to ordered raw channels."""
import io
import zipfile

import numpy as np
import pytest

import spacr.io as images
from spacr.settings import set_default_settings_preprocess_generate_masks


def _settings(root, **updates):
    settings = dict(
        src=str(root), nucleus_channel=3, cell_channel=3, pathogen_channel=2,
        organelle_channel=None, batch_size=1, randomize=False, timelapse=False,
        plot=False, verbose=False, lower_percentile=2)
    settings.update(updates)
    return set_default_settings_preprocess_generate_masks(settings)


def _stack(root, *, volume=False):
    stack = root / 'stack'
    stack.mkdir(parents=True)
    shape = (2, 8, 9) if volume else (8, 9)
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    raw = np.stack([values * (c + 1) + c * 100 for c in range(4)], axis=-1)
    np.save(stack / 'plate1_A01_1.npy', raw)
    return stack, raw


@pytest.mark.parametrize('volume', [False, True])
def test_real_writer_preserves_duplicate_nonascending_roles_and_metadata(tmp_path, volume):
    stack, raw = _stack(tmp_path, volume=volume)
    settings = _settings(tmp_path)
    images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    path = next((tmp_path / 'masks').glob('*.npz'))
    expected = images._normalize_img_batch(raw[None].copy(), [3, 2], np.float32, settings)[..., [3, 2]]
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['source_channels'], [3, 2])
        assert archive['source_channels'].dtype == np.int64
        np.testing.assert_array_equal(archive['data'], expected)
    assert images._validate_normalized_source_channels(path, [3, 2]) == [3, 2]
    assert images._check_archives_without_preprocessing(str(tmp_path), settings=settings) == [path.name]


@pytest.mark.parametrize('stored', [None, [2, 3], [3], [3, 3], [-1, 2], [3., 2.],
                                    np.array([3, 2], dtype=np.int32), np.array([3, 2], dtype=np.uint64)])
def test_unknown_or_incompatible_order_cannot_pass_full_coverage_resume(tmp_path, stored):
    stack, raw = _stack(tmp_path)
    masks = tmp_path / 'masks'
    masks.mkdir()
    archive = masks / 'stack_0_norm.npz'
    arrays = dict(data=raw[None, ..., [2, 3]], filenames=['plate1_A01_1.npy'])
    if stored is not None:
        arrays['source_channels'] = np.asarray(stored)
    np.savez(archive, **arrays)
    original = archive.read_bytes()
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match='channel order is unknown'):
        images._check_archives_without_preprocessing(str(tmp_path), settings=settings)
    assert images._resume_normalized_archives(settings, str(tmp_path), [3, 2]) is False
    assert archive.read_bytes() == original
    images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    assert images._validate_normalized_source_channels(archive, [3, 2]) == [3, 2]
    np.testing.assert_array_equal(np.load(next(stack.glob('*.npy'))), raw)


@pytest.mark.parametrize('incomplete_stack', [False, True])
def test_no_recoverable_stack_refuses_unknown_order_without_modifying_archive(tmp_path, incomplete_stack):
    masks = tmp_path / 'masks'
    masks.mkdir()
    archive = masks / 'legacy.npz'
    np.savez(archive, data=np.zeros((1, 3, 4, 2)), filenames=['plate1_A01_1.npy'])
    original = archive.read_bytes()
    if incomplete_stack:
        stack = tmp_path / 'stack'
        stack.mkdir()
        np.save(stack / 'plate1_A01_1.npy', np.zeros((3, 4, 2)))
    assert images._validate_normalized_source_channels(archive, [3, 2], required=False) is None
    with pytest.raises(ValueError, match='original image stacks are unavailable'):
        images._resume_normalized_archives(_settings(tmp_path), str(tmp_path), [3, 2])
    assert archive.read_bytes() == original


def test_same_count_swapped_order_rebuild_is_staged_and_failure_retains_old_outputs(tmp_path, monkeypatch):
    stack, _raw = _stack(tmp_path)
    settings = _settings(tmp_path)
    images.concatenate_and_normalize(str(stack), [2, 3], settings=settings)
    archive = next((tmp_path / 'masks').glob('*.npz'))
    original = archive.read_bytes()
    old_masks = tmp_path / 'masks/pathogen_mask_stack'
    old_masks.mkdir()
    (old_masks / 'old.npy').write_bytes(b'previous segmentation')
    merged = tmp_path / 'merged'
    merged.mkdir()
    (merged / 'old.npy').write_bytes(b'previous merged evidence')
    assert images._resume_normalized_archives(settings, str(tmp_path), [3, 2]) is False
    normalize = images._normalize_img_batch

    def fail(*args, **kwargs):
        raise RuntimeError('normalization failed before publication')

    monkeypatch.setattr(images, '_normalize_img_batch', fail)
    with pytest.raises(RuntimeError, match='normalization failed'):
        images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    assert archive.read_bytes() == original
    assert (old_masks / 'old.npy').read_bytes() == b'previous segmentation'
    assert (merged / 'old.npy').read_bytes() == b'previous merged evidence'
    assert not list(tmp_path.glob('previous_segmentation_channel_order_*'))
    monkeypatch.setattr(images, '_normalize_img_batch', normalize)
    images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    assert images._validate_normalized_source_channels(archive, [3, 2]) == [3, 2]
    assert not old_masks.exists()
    assert not merged.exists()
    backup, = tmp_path.glob('previous_segmentation_channel_order_*')
    assert (backup / 'masks/pathogen_mask_stack/old.npy').read_bytes() == b'previous segmentation'
    assert (backup / 'merged/old.npy').read_bytes() == b'previous merged evidence'
    old_masks.mkdir()
    (old_masks / 'new.npy').write_bytes(b'regenerated segmentation')
    assert (old_masks / 'new.npy').read_bytes() == b'regenerated segmentation'
    assert (backup / 'masks/pathogen_mask_stack/old.npy').read_bytes() == b'previous segmentation'
    assert not list(tmp_path.glob('.spacr_v1_npz_*'))


@pytest.mark.parametrize('failure', ['preserve', 'publish', 'interrupt'])
def test_channel_rebuild_failure_restores_archive_and_active_evidence(tmp_path, monkeypatch, failure):
    stack, _ = _stack(tmp_path)
    settings = _settings(tmp_path)
    images.concatenate_and_normalize(str(stack), [2, 3], settings=settings)
    archive = next((tmp_path / 'masks').glob('*.npz'))
    original = archive.read_bytes()
    mask = tmp_path / 'masks/pathogen_mask_stack'
    mask.mkdir()
    (mask / 'old.npy').write_bytes(b'old mask')
    merged = tmp_path / 'merged'
    merged.mkdir()
    (merged / 'old.npy').write_bytes(b'old merged')
    replace = images.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        moving_evidence = str(source) == str(merged)
        publishing = '.spacr_v1_npz_' in str(source) and str(source).endswith('.npz')
        if not failed and ((failure == 'preserve' and moving_evidence)
                           or (failure != 'preserve' and publishing)):
            failed = True
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            raise OSError('injected output replacement failure')
        return replace(source, destination)

    monkeypatch.setattr(images.os, 'replace', fail_once)
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else OSError):
        images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    assert failed
    assert archive.read_bytes() == original
    assert images._validate_normalized_source_channels(archive, [2, 3]) == [2, 3]
    assert (mask / 'old.npy').read_bytes() == b'old mask'
    assert (merged / 'old.npy').read_bytes() == b'old merged'
    assert not list(tmp_path.glob('previous_segmentation_channel_order_*'))
    images.concatenate_and_normalize(str(stack), [3, 2], settings=settings)
    assert images._validate_normalized_source_channels(archive, [3, 2]) == [3, 2]
    assert not mask.exists() and not merged.exists()
    backup, = tmp_path.glob('previous_segmentation_channel_order_*')
    assert (backup / 'masks/pathogen_mask_stack/old.npy').read_bytes() == b'old mask'
    assert (backup / 'merged/old.npy').read_bytes() == b'old merged'


def test_metadata_validation_reads_headers_without_loading_image_or_unpickling(tmp_path, monkeypatch):
    archive = tmp_path / 'valid.npz'
    np.savez_compressed(archive, data=np.zeros((1, 8, 9, 2)), filenames=['field.npy'],
                        source_channels=np.array([3, 2], dtype=np.int64))
    monkeypatch.setattr(images.np, 'load', lambda *args, **kwargs: pytest.fail('image payload loaded'))
    assert images._validate_normalized_source_channels(archive, [3, 2]) == [3, 2]
    forged = tmp_path / 'forged.npz'
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '<i8', 'fortran_order': False,
                                                'shape': (10**12,)})
    with zipfile.ZipFile(forged, 'w') as output:
        output.writestr('source_channels.npy', header.getvalue())
    with pytest.raises(ValueError, match='channel order is unknown'):
        images._validate_normalized_source_channels(forged, [3, 2])


def test_time_writer_records_the_final_channel_axis_order(tmp_path):
    stack, raw = _stack(tmp_path)
    (stack / 'plate1_A01_1.npy').rename(stack / 'plate1_A01_1_1.npy')
    np.save(stack / 'plate1_A01_1_2.npy', raw + 3)
    images.concatenate_and_normalize(str(stack), [3, 2], settings=_settings(tmp_path, timelapse=True))
    archive = next((tmp_path / 'masks').glob('*.npz'))
    assert images._validate_normalized_source_channels(archive, [3, 2]) == [3, 2]
    with np.load(archive, allow_pickle=False) as content:
        assert content['data'].shape == (2, 8, 9, 2)
        assert len(content['filenames']) == 2


def test_native_time_volume_writer_records_deduplicated_nonascending_channels(tmp_path):
    from tests.test_native_tzyx_batch_f548 import _settings as native_settings
    from tests.test_watch_timelapse_series_f548 import _converted_series

    source, _rows = _converted_series(tmp_path)
    images.preprocess_img_data(native_settings(source, nucleus_channel=1, cell_channel=1, pathogen_channel=0))
    archive = next((source / 'masks').glob('*.npz'))
    assert images._validate_normalized_source_channels(archive, [1, 0]) == [1, 0]
    with np.load(archive, allow_pickle=False) as content:
        assert content['data'].ndim == 5 and content['data'].shape[-1] == 2
