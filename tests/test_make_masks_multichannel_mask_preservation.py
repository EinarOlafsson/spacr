"""Color display and reviewed image channels retain their paired label identities."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as engine


@pytest.mark.parametrize('axes', ['YXS', 'SYX'])
@pytest.mark.parametrize('samples', [3, 4])
def test_explicit_rgb_samples_flatten_without_changing_labels_or_source(tmp_path, axes, samples):
    pixels = np.zeros((9, 11, samples), dtype=np.uint8)
    pixels[2:7, 3:8, :3] = (100, 160, 220)
    if samples == 4:
        pixels[..., 3] = 255
    stored = pixels if axes == 'YXS' else np.moveaxis(pixels, -1, 0)
    source = tmp_path / 'malnio.tif'
    tifffile.imwrite(source, stored, photometric='rgb',
                     planarconfig='contig' if axes == 'YXS' else 'separate',
                     metadata={'axes': axes})
    labels = np.zeros(pixels.shape[:2], np.uint32)
    labels[2:5, 3:6], labels[5:7, 6:8] = 7, 70000
    mask = Path(engine.save_mask(str(tmp_path), source.name, labels, preserve_ids=True))
    before = source.read_bytes(), mask.read_bytes()
    metadata = engine.load_mask_metadata(str(tmp_path), source.name)

    header = engine._container_image_header(source)
    assert engine._native_rgb_field(header)
    engine._retain_native_image_header(header)
    image, loaded = engine.load_image_and_mask(str(tmp_path), source.name)

    assert image.shape == labels.shape and image.dtype == np.uint16
    np.testing.assert_array_equal(loaded, labels)
    np.testing.assert_array_equal(engine.read_image(str(source)), pixels)
    assert engine.load_mask_metadata(str(tmp_path), source.name) == metadata
    assert (source.read_bytes(), mask.read_bytes()) == before


def test_scientific_channels_reuse_paired_labels_then_save_independently(tmp_path):
    source = tmp_path / 'field.tif'
    pixels = np.arange(3 * 9 * 11, dtype=np.uint16).reshape(3, 9, 11)
    tifffile.imwrite(source, pixels, photometric='minisblack', metadata={'axes': 'CYX'})
    paired = tmp_path / 'paired'
    labels = np.zeros((9, 11), np.uint32)
    labels[2:6, 3:7] = 70000
    mask = Path(engine.save_mask(str(tmp_path), source.name, labels,
                                 masks_dir=str(paired), preserve_ids=True))
    metadata = engine.load_mask_metadata(str(tmp_path), source.name, str(paired))
    originals = source.read_bytes(), mask.read_bytes()
    header = engine._container_image_header(source)
    assert not engine._native_rgb_field(header)
    first = engine._set_container_image_view(header, channel=0, source_masks_dir=str(paired))
    image, loaded = engine.load_image_and_mask(str(tmp_path), source.name, first['masks_dir'])
    np.testing.assert_array_equal(image, pixels[0])
    np.testing.assert_array_equal(loaded, labels)
    assert engine.load_mask_metadata(str(tmp_path), source.name, first['masks_dir']) == metadata
    edited = loaded.copy()
    edited[0, 0] = 17
    engine.save_mask(str(tmp_path), source.name, edited,
                     masks_dir=first['masks_dir'], preserve_ids=True)
    np.testing.assert_array_equal(engine.load_image_and_mask(
        str(tmp_path), source.name, first['masks_dir'])[1], edited)
    second = engine._set_container_image_view(header, channel=2, source_masks_dir=str(paired))
    image, loaded = engine.load_image_and_mask(str(tmp_path), source.name, second['masks_dir'])
    np.testing.assert_array_equal(image, pixels[2])
    np.testing.assert_array_equal(loaded, labels)
    assert (source.read_bytes(), mask.read_bytes()) == originals


@pytest.mark.parametrize('axes,shape', [('TCYX', (2, 3, 9, 11)), ('ZCYX', (2, 3, 9, 11))])
def test_ambiguous_time_or_depth_never_imports_a_plain_paired_mask(tmp_path, axes, shape):
    source = tmp_path / 'stack.tif'
    tifffile.imwrite(source, np.zeros(shape, np.uint16), photometric='minisblack',
                     metadata={'axes': axes})
    engine.save_mask(str(tmp_path), source.name, np.ones((9, 11), np.uint16), preserve_ids=True)
    view = engine._set_container_image_view(engine._container_image_header(source),
                                            source_masks_dir=str(tmp_path / 'masks'))
    assert not engine.load_image_and_mask(str(tmp_path), source.name, view['masks_dir'])[1].any()
    assert engine.load_mask_metadata(str(tmp_path), source.name, view['masks_dir']) == {}


def test_changed_paired_mask_is_not_silently_imported_after_review(tmp_path):
    source = tmp_path / 'field.tif'
    tifffile.imwrite(source, np.zeros((3, 9, 11), np.uint16),
                     photometric='minisblack', metadata={'axes': 'CYX'})
    path = engine.save_mask(str(tmp_path), source.name, np.ones((9, 11), np.uint16))
    view = engine._set_container_image_view(engine._container_image_header(source),
                                            source_masks_dir=str(tmp_path / 'masks'))
    tifffile.imwrite(path, np.full((9, 11), 9, np.uint16))
    with pytest.raises(ValueError, match='source changed'):
        engine.load_image_and_mask(str(tmp_path), source.name, view['masks_dir'])


def test_mismatched_paired_shape_cannot_supply_labels_or_metadata(tmp_path):
    source = tmp_path / 'field.tif'
    tifffile.imwrite(source, np.zeros((3, 9, 11), np.uint16),
                     photometric='minisblack', metadata={'axes': 'CYX'})
    engine.save_mask(str(tmp_path), source.name, np.ones((8, 11), np.uint16))
    view = engine._set_container_image_view(engine._container_image_header(source),
                                            source_masks_dir=str(tmp_path / 'masks'))
    for load in (engine.load_image_and_mask, engine.load_mask_metadata):
        with pytest.raises(ValueError, match='does not match image shape'):
            load(str(tmp_path), source.name, view['masks_dir'])
