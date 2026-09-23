"""Primary-mask pairing cannot reuse the wrong field or overwrite its source."""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr.qt.secondary_masks import read_primary_source
from spacr.tiff_io import write_tiff


def _files(tmp_path):
    primary = tmp_path / 'primary'
    primary.mkdir()
    labels = np.zeros((10, 12), dtype=np.uint16)
    labels[2:5, 3:6] = 900
    source = primary / 'field.tif'
    write_tiff(source, labels)
    image = tmp_path / 'field.tif'
    output = tmp_path / 'masks' / 'field.tif'
    return source, image, output, labels


def test_folder_pairs_each_field_and_records_read_only_source_identity(tmp_path):
    source, image, output, labels = _files(tmp_path)
    before = source.read_bytes()
    snapshot = read_primary_source(source.parent, image, labels.shape, output)
    np.testing.assert_array_equal(snapshot.labels, labels)
    assert not snapshot.labels.flags.writeable
    assert snapshot.identity[-1] == snapshot.sha256
    assert len(snapshot.sha256) == 64
    assert json.loads(json.dumps(snapshot.provenance()))['primary_class'] == 'nucleus'
    crop = snapshot.crop((3, 2, 6, 5))
    assert np.all(crop == 900)
    crop[:] = 0
    assert np.any(snapshot.labels == 900)
    assert source.read_bytes() == before
    assert not output.exists()
    with pytest.raises(ValueError, match='found 0'):
        read_primary_source(source.parent, image.with_name('next.tif'), labels.shape, output)


def test_explicit_file_is_bound_to_one_field_even_if_next_field_has_same_shape(tmp_path):
    source, image, output, labels = _files(tmp_path)
    chosen = read_primary_source(source, image, labels.shape, output, bound_image=image)
    assert chosen.image_path == str(image)
    with pytest.raises(ValueError, match='another field'):
        read_primary_source(source, image.with_name('next.tif'), labels.shape, output, bound_image=image)


@pytest.mark.parametrize('alias', ['same', 'symbolic', 'hard'])
def test_primary_destination_alias_is_refused_without_writing(tmp_path, alias):
    source, image, output, labels = _files(tmp_path)
    before = source.read_bytes()
    target = source if alias == 'same' else tmp_path / 'alias.tif'
    if alias == 'symbolic':
        target.symlink_to(source)
    elif alias == 'hard':
        target.hardlink_to(source)
    with pytest.raises(ValueError, match='different files'):
        read_primary_source(source, image, labels.shape, target, bound_image=image)
    snapshot = read_primary_source(source, image, labels.shape, output, bound_image=image)
    with pytest.raises(ValueError, match='different files'):
        snapshot.validate_destination(target)
    assert source.read_bytes() == before


def test_ambiguous_folder_mask_requires_an_explicit_choice(tmp_path):
    source, image, output, labels = _files(tmp_path)
    np.save(source.with_suffix('.npy'), labels)
    with pytest.raises(ValueError, match='found 2'):
        read_primary_source(source.parent, image, labels.shape, output)
    snapshot = read_primary_source(source, image, labels.shape, output, bound_image=image)
    assert snapshot.path == str(source)


def test_changed_source_has_a_new_request_identity(tmp_path):
    source, image, output, labels = _files(tmp_path)
    first = read_primary_source(source.parent, image, labels.shape, output)
    labels[7, 7] = 7
    write_tiff(source, labels)
    second = read_primary_source(source.parent, image, labels.shape, output)
    assert first.identity != second.identity
    assert first.labels[7, 7] == 0 and second.labels[7, 7] == 7


@pytest.mark.parametrize('options', [{'shape': (3, 4)}, {'primary_class': ''},
                                    {'secondary_class': 'NUCLEUS'}])
def test_invalid_shape_and_roles_are_refused(tmp_path, options):
    source, image, output, labels = _files(tmp_path)
    shape = options.pop('shape', labels.shape)
    with pytest.raises(ValueError):
        read_primary_source(source.parent, image, shape, output, **options)
