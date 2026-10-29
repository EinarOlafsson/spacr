"""Reader revisions, metadata and explicit choices stay bound to source pixels."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import convert
from spacr.qt import mask_engine as engine


@pytest.mark.parametrize('review', ['native', 'plane'])
def test_disappearing_reviewed_source_discards_its_current_interpretation(tmp_path, review):
    path = tmp_path / 'removed.tif'
    tifffile.imwrite(path, np.arange(48, dtype=np.uint16).reshape(6, 8))
    header = engine._container_image_header(path)
    if review == 'native':
        engine._retain_native_image_header(header)
    else:
        engine._set_container_image_view(header)
    path.unlink()
    assert not engine._container_reviewed(path)
    assert engine._current_native_image_header(path) is None
    assert engine._current_container_image_view(path) is None


@pytest.mark.parametrize('physical,expected', [(dict(PhysicalSizeX=.01725, PhysicalSizeY=.01725), 17.25),
                                               ({}, None)])
def test_ome_square_calibration_is_attached_to_the_reviewed_series(tmp_path, physical, expected):
    path = tmp_path / 'calibrated.ome.tif'
    tifffile.imwrite(path, np.zeros((6, 8), np.uint16), ome=True,
                     metadata=dict(axes='YX', **physical))
    before = path.read_bytes()
    entry = engine._container_image_header(path)['series'][0]
    assert entry['pixel_size_nm'] == (pytest.approx(expected) if expected is not None else None)
    assert path.read_bytes() == before


@pytest.mark.parametrize('metadata,message', [
    (dict(n_series=513), 'too many series'),
    (dict(n_series=0), 'no bounded image series'),
    (dict(n_t=0), 'unsupported spatial layout'),
    (dict(shape=(0, 8)), 'unsupported spatial layout'),
])
def test_vendor_metadata_outside_the_editor_contract_never_decodes(tmp_path, monkeypatch, metadata, message):
    path = tmp_path / 'metadata-contract.nd2'
    path.write_bytes(b'ND2 metadata API double; no vendor pixels')
    described = dict(shape=(6, 8), axes='ND2', dtype='uint16', n_series=1,
                     n_t=1, n_z=1, n_c=1, reader='nd2reader')
    described.update(metadata)
    monkeypatch.setattr(convert, 'reader_available', lambda ext: True)
    monkeypatch.setattr(convert, '_describe', lambda *args: described)
    monkeypatch.setattr(convert, '_import_reader', lambda *args: pytest.fail('Invalid metadata reached decoding'))
    with pytest.raises(ValueError, match=message):
        engine._container_image_header(path)
    assert not engine._container_reviewed(path)
    assert path.read_bytes() == b'ND2 metadata API double; no vendor pixels'


@pytest.mark.parametrize('axes,shape', [('BYX', (1, 6, 8)), ('BYX', (2, 6, 8))])
def test_legacy_czi_metadata_double_budgets_whole_decode_and_rejects_extra_spatial_axes(tmp_path, monkeypatch, axes, shape):
    path = tmp_path / 'legacy-contract.czi'
    path.write_bytes(b'czifile metadata API double; no vendor pixels')
    class Reader:
        dtype = np.dtype('uint16')
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def metadata(self):
            return ''
    Reader.shape = shape
    monkeypatch.setattr(convert, 'reader_available', lambda ext: True)
    monkeypatch.setattr(convert, '_describe', lambda *args: dict(
        shape=shape, axes=axes, dtype='uint16', n_series=1, reader='czifile'))
    monkeypatch.setattr(convert, '_import_reader', lambda ext: SimpleNamespace(CziFile=lambda path: Reader()))
    if shape[0] > 1:
        with pytest.raises(ValueError, match='unsupported spatial layout'):
            engine._container_image_header(path)
        assert not engine._container_reviewed(path)
    else:
        header = engine._container_image_header(path)
        entry = header['series'][0]
        assert entry['axes'] == 'YX' and entry['shape'] == (6, 8)
        assert entry['decode_bytes'] == np.prod(shape) * 2
        assert entry['pixel_size_nm'] is None
        view = engine._set_container_image_view(header)
        assert view['shape'] == (6, 8)


def test_source_replaced_during_metadata_inspection_requires_a_new_review(tmp_path, monkeypatch):
    path = tmp_path / 'acquisition.tif'
    tifffile.imwrite(path, np.zeros((6, 8), np.uint16))
    replacement = tmp_path / 'replacement.tif'
    tifffile.imwrite(replacement, np.ones((7, 9), np.uint16))
    inspect = engine._tiff_series_pixel_size
    def replace(handle, image):
        result = inspect(handle, image)
        replacement.replace(path)
        return result
    with monkeypatch.context() as context:
        context.setattr(engine, '_tiff_series_pixel_size', replace)
        with pytest.raises(ValueError, match='changed while its image metadata was read'):
            engine._container_image_header(path)
    assert not engine._container_reviewed(path)
    assert engine._container_image_header(path)['series'][0]['shape'] == (7, 9)


def test_explicit_plane_choices_cannot_escape_the_reviewed_tiff(tmp_path):
    path = tmp_path / 'planes.tif'
    pixels = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8)
    tifffile.imwrite(path, pixels, metadata={'axes': 'CYX'})
    header = engine._container_image_header(path)
    for choices, message in [(dict(series=True), 'valid image series'),
                             (dict(axes='QYX'), 'stored spatial'),
                             (dict(channel=2), 'valid time')]:
        with pytest.raises(ValueError, match=message):
            engine._set_container_image_view(header, **choices)
        assert not engine._container_reviewed(path)
    view = engine._set_container_image_view(header, channel=1)
    np.testing.assert_array_equal(engine._read_container_image_view(path), pixels[1])
    replacement = tmp_path / 'replacement.tif'
    tifffile.imwrite(replacement, pixels + 100, metadata={'axes': 'CYX'})
    replacement.replace(path)
    labels = np.zeros((6, 8), np.uint16)
    with pytest.raises(ValueError, match='source changed'):
        engine._set_container_image_view(header)
    with pytest.raises(ValueError, match='source changed'):
        engine.save_mask(str(tmp_path), path.name, labels, masks_dir=view['masks_dir'])
    assert not Path(engine.mask_save_path(str(tmp_path), path.name, view['masks_dir'])).exists()
    with pytest.raises(ValueError, match='Review this source'):
        engine.save_mask(str(tmp_path), path.name, labels,
                         masks_dir=str(tmp_path / 'masks' / '.spacr-reader-unreviewed'))


def test_numeric_reader_requires_explicit_axes_and_channel_only_when_needed(tmp_path):
    path = tmp_path / 'plane.npy'
    pixels = np.arange(48, dtype=np.uint16).reshape(6, 8)
    np.save(path, pixels)
    read, axes = engine._read_npy_image_view(path)
    assert axes == 'YX'
    np.testing.assert_array_equal(read, pixels)
    channel_path = tmp_path / 'channels.npy'
    np.save(channel_path, np.stack((pixels, pixels + 100), axis=-1))
    with pytest.raises(ValueError, match='Review the NPY spatial axes'):
        engine._read_npy_image_view(channel_path)
    for layout, channel in [('ZYXC', 0), ('YXC', True), ('YXC', 2)]:
        with pytest.raises(ValueError, match='Review the NPY spatial axes'):
            engine._set_npy_image_view(channel_path, layout, channel)
    with pytest.raises(ValueError, match='Review the NPY spatial axes'):
        engine._set_npy_image_view(path, 'YX', 1)
    engine._set_npy_image_view(channel_path, 'YXC', 1)
    read, axes = engine._read_npy_image_view(channel_path)
    assert axes == 'YX'
    np.testing.assert_array_equal(read, pixels + 100)


def test_numeric_source_replaced_when_mapping_is_refused_before_using_new_pixels(tmp_path, monkeypatch):
    path = tmp_path / 'plane.npy'
    np.save(path, np.zeros((6, 8), np.uint16))
    engine._set_npy_image_view(path, 'YX')
    replacement = tmp_path / 'replacement.npy'
    np.save(replacement, np.ones((7, 9), np.uint16))
    load = np.load
    def replace_then_load(*args, **kwargs):
        replacement.replace(path)
        return load(*args, **kwargs)
    monkeypatch.setattr(np, 'load', replace_then_load)
    with pytest.raises(ValueError, match='source changed'):
        engine._read_npy_image_view(path)


def test_boolean_numeric_volume_keeps_truth_values_in_the_editable_session(tmp_path):
    path = tmp_path / 'binary.npy'
    pixels = np.zeros((3, 6, 8), bool)
    pixels[:, 2:4, 3:5] = True
    np.save(path, pixels)
    engine._set_npy_image_view(path, 'ZYX')
    session = engine.load_volume_image_and_mask(str(tmp_path), path.name)
    assert session.image.dtype == np.uint8 and session.axes == 'ZYX'
    np.testing.assert_array_equal(session.image, pixels)
    assert not session.mask.any()


@pytest.mark.parametrize('shape,operation,message', [
    ((16385, 16385), 'native', 'selected native image'),
    ((6000, 6000), 'plane', 'selected image plane'),
])
def test_sparse_tiff_metadata_refuses_unsafe_decode_before_reading_pixels(tmp_path, monkeypatch, shape, operation, message):
    path = tmp_path / 'large.tif'
    tifffile.imwrite(path, data=None, shape=shape, dtype=np.uint8, metadata={'axes': 'YX'})
    header = engine._container_image_header(path)
    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray',
                        lambda *args, **kwargs: pytest.fail('Oversized image decoded'))
    with pytest.raises(ValueError, match=message):
        if operation == 'native':
            engine._retain_native_image_header(header)
        else:
            engine._set_container_image_view(header)
    assert not engine._container_reviewed(path)


def test_bounded_review_caches_evict_old_sources_and_require_renewed_channel_choice(tmp_path):
    first_tiff = first_npy = None
    for index in range(129):
        path = tmp_path / f'field{index}.tif'
        pixels = np.full((2, 3), index, np.uint16)
        tifffile.imwrite(path, pixels)
        engine._retain_native_image_header(engine._container_image_header(path))
        array = path.with_suffix('.npy')
        np.save(array, np.stack((pixels, pixels + 1), axis=-1))
        engine._set_npy_image_view(array, 'YXC', 1)
        if index == 0:
            first_tiff, first_npy = path, array
    assert len(engine._CONTAINER_NATIVE_HEADERS) == len(engine._NPY_IMAGE_VIEWS) == 128
    assert engine._current_native_image_header(first_tiff) is None
    assert engine._current_native_image_header(path) is not None
    with pytest.raises(ValueError, match='Review the NPY spatial axes'):
        engine._read_npy_image_view(first_npy)
    latest, axes = engine._read_npy_image_view(array)
    assert axes == 'YX'
    np.testing.assert_array_equal(latest, pixels + 1)
    engine._set_npy_image_view(first_npy, 'YXC', 0)
    oldest, axes = engine._read_npy_image_view(first_npy)
    np.testing.assert_array_equal(oldest, np.zeros((2, 3), np.uint16))


def test_xy_tiff_is_transposed_and_change_during_decode_is_refused(tmp_path, monkeypatch):
    path = tmp_path / 'column-first.tif'
    pixels = np.arange(48, dtype=np.uint16).reshape(8, 6)
    tifffile.imwrite(path, pixels, metadata={'axes': 'XY'})
    header = engine._container_image_header(path)
    engine._set_container_image_view(header)
    np.testing.assert_array_equal(engine._read_container_image_view(path), pixels.T)
    replacement = tmp_path / 'replacement.tif'
    tifffile.imwrite(replacement, pixels + 100, metadata={'axes': 'XY'})
    decode = tifffile.TiffPageSeries.asarray
    def replaced_after_decode(series, *args, **kwargs):
        result = decode(series, *args, **kwargs)
        replacement.replace(path)
        return result
    monkeypatch.setattr(tifffile.TiffPageSeries, 'asarray', replaced_after_decode)
    with pytest.raises(ValueError, match='invalid or changed image plane'):
        engine._read_container_image_view(path)


def test_lif_metadata_double_with_incomplete_scale_does_not_invent_calibration(tmp_path, monkeypatch):
    path = tmp_path / 'uncalibrated-contract.lif'
    path.write_bytes(b'LIF metadata API double; no vendor pixels')
    monkeypatch.setattr(convert, 'reader_available', lambda ext: True)
    monkeypatch.setattr(convert, '_describe', lambda *args: dict(
        shape=(6, 8), axes='LIF', dtype='uint16', reader='readlif',
        per_series=[dict(lif_image=0, lif_tile=0)]))
    monkeypatch.setattr(convert, '_lif_images', lambda path: [SimpleNamespace(scale=(4.,))])
    header = engine._container_image_header(path)
    assert header['series'][0]['pixel_size_nm'] is None


def test_imagej_unit_calibration_survives_native_metadata_review(tmp_path):
    path = tmp_path / 'imagej.tif'
    tifffile.imwrite(path, np.zeros((6, 8), np.uint16), imagej=True,
                     metadata={'unit': 'nm'}, resolution=(1 / 17.25, 1 / 17.25))
    header = engine._container_image_header(path)
    assert header['series'][0]['pixel_size_nm'] == pytest.approx(17.25, rel=1e-6)


def test_shared_container_route_refuses_numeric_arrays_before_reading_them(tmp_path, monkeypatch):
    path = tmp_path / 'numeric.npy'
    np.save(path, np.arange(48, dtype=np.uint16).reshape(6, 8))
    before = path.read_bytes()
    monkeypatch.setattr(convert, 'reader_available', lambda ext: pytest.fail('NPY reached a container reader'))
    with pytest.raises(ValueError, match='no shared image reader'):
        engine._container_image_header(path)
    assert path.read_bytes() == before


def test_legacy_czi_plane_decode_double_receives_the_reviewed_dimensions(tmp_path, monkeypatch):
    path = tmp_path / 'legacy-plane-contract.czi'
    path.write_bytes(b'legacy CZI reader API double; no vendor pixels')
    expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
    class Reader:
        shape = (1, 1, 2, 6, 8)
        dtype = np.dtype('uint16')
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(convert, 'reader_available', lambda ext: True)
    monkeypatch.setattr(convert, '_describe', lambda *args: dict(
        shape=Reader.shape, axes='TZCYX', dtype='uint16', n_series=1, reader='czifile'))
    monkeypatch.setattr(convert, '_import_reader', lambda ext: SimpleNamespace(CziFile=lambda path: Reader()))
    decoded = []
    def decode(source):
        decoded.append(source)
        return np.stack((expected, expected + 100)).reshape(1, 1, 2, 6, 8)
    monkeypatch.setattr(convert, '_read_czi', decode)
    view = engine._set_container_image_view(engine._container_image_header(path), channel=1)
    np.testing.assert_array_equal(engine._read_container_image_view(path), expected + 100)
    assert view['shape'] == (6, 8) and len(decoded) == 1
    assert decoded[0].n_channels == 2 and decoded[0].meta['series'] == 0
    assert decoded[0].path == str(path)
