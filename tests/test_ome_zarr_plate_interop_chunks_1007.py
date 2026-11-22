"""NGFF 0.6 plate interoperability and crop-friendly shared chunk policy."""
from importlib.metadata import version
import json
from time import perf_counter

import numpy as np
import pytest

from spacr.image_store import PlateWriter
from spacr.ome_zarr import read_ome_zarr, write_ome_zarr


@pytest.fixture
def field_data():
    """A two-channel field and integer masks spanning several spatial chunks."""
    pixels = (np.arange(2 * 317 * 421).reshape(2, 317, 421) % 4093).astype(np.uint16)
    y, x = np.indices(pixels.shape[-2:])
    labels = ((y // 29 + x // 31) % 7).astype(np.uint32)
    return pixels, labels


def _same_pixel_bytes(actual, expected):
    """Compare dtype, shape and every pixel bit, including signed zero."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_ngff06_plate_well_image_and_labels_validate_with_models(tmp_path, field_data, record_property):
    """The pinned independent 0.6 model reader accepts every HCS group type."""
    models = pytest.importorskip('ome_zarr_models')
    pytest.importorskip('ome_zarr_models.v06')
    record_property('ome_zarr_models_version', version('ome-zarr-models'))
    pixels, labels = field_data
    writer = PlateWriter(tmp_path, 'P1', ngff_version='0.6', channel_names=['DNA', 'GFP'])
    key = writer.write_field('A01', 1, pixels, {'cells': labels})
    locations = [writer.path, writer.path / 'A' / '1', writer.path / key,
                 writer.path / key / 'labels', writer.path / key / 'labels' / 'cells']
    for location in locations:
        model = models.open_ome_zarr(str(location), version='0.6')
        assert type(model).__module__.startswith('ome_zarr_models.v06.')
    _same_pixel_bytes(read_ome_zarr(writer.path / key).read(), pixels)
    _same_pixel_bytes(read_ome_zarr(locations[-1]).read(), labels)


@pytest.mark.parametrize('ngff_version', ['0.5', '0.6'])
def test_plate_opens_with_a_supporting_ome_zarr_reader(tmp_path, field_data, record_property, ngff_version):
    """Record the reader version and skip only a missing declared format capability."""
    reader = pytest.importorskip('ome_zarr.reader')
    formats = pytest.importorskip('ome_zarr.format')
    io = pytest.importorskip('ome_zarr.io')
    reader_version = version('ome-zarr')
    record_property('ome_zarr_version', reader_version)
    record_property('ngff_version', ngff_version)
    supported = [fmt.version for fmt in formats.format_implementations()]
    if ngff_version not in supported:
        pytest.skip(f'ome-zarr {reader_version} declares formats {supported}; no NGFF {ngff_version} reader')
    pixels, _ = field_data
    writer = PlateWriter(tmp_path, 'P1', ngff_version=ngff_version)
    writer.write_field('A01', 1, pixels)
    nodes = list(reader.Reader(io.parse_url(str(writer.path), mode='r'))())
    assert nodes and nodes[0].data
    _same_pixel_bytes(nodes[0].data[0].compute(scheduler='single-threaded'), pixels)


@pytest.mark.parametrize('ngff_version', ['0.5', '0.6'])
@pytest.mark.parametrize('image_mode', ['channels', 'plane', 'labels-only'])
@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_tiled_plate_pixels_match_whole_field_reference_bytes(tmp_path, field_data, ngff_version, image_mode, dtype):
    """Spatial tiling preserves image and mask bytes at every pyramid level."""
    pixels, labels = field_data
    pixels = pixels.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        pixels.flat[0] = -0.0
        pixels.flat[1] = np.nextafter(np.float32(1), np.float32(2))
    if image_mode == 'plane':
        pixels = pixels[0]
    elif image_mode == 'labels-only':
        pixels = None
    writer = PlateWriter(tmp_path / 'tiled', 'P1', ngff_version=ngff_version)
    key = writer.write_field('A01', 1, pixels, {'cells': labels})
    locations = [(writer.path / key / 'labels' / 'cells', labels, ['y', 'x'], 'stride')]
    if pixels is not None:
        axes = ['c', 'y', 'x'] if pixels.ndim == 3 else ['y', 'x']
        locations.append((writer.path / key, pixels, axes, 'mean'))
    for index, (location, expected, axes, downsample) in enumerate(locations):
        actual = read_ome_zarr(location)
        chunks = (1, *expected.shape[1:]) if expected.ndim == 3 else expected.shape
        reference = write_ome_zarr(tmp_path / f'reference_{index}', expected, axes=axes,
            ngff_version=ngff_version, chunks=chunks, shards=False, levels=len(actual.levels),
            downsample=downsample, compressor='lz4', compression_level=5)
        for level in range(len(actual.levels)):
            _same_pixel_bytes(actual.read(level), reference.read(level))
        _same_pixel_bytes(actual.read(), expected)
        metadata = json.loads((location / '0' / 'zarr.json').read_text())
        assert metadata['chunk_grid']['configuration']['chunk_shape'] == ([1, 256, 256] if expected.ndim == 3 else [256, 256])
        assert [codec['name'] for codec in metadata['codecs']] == ['bytes', 'blosc']
        assert metadata['codecs'][1]['configuration']['cname'] == 'lz4'
        assert metadata['codecs'][1]['configuration']['clevel'] == 5
        assert metadata['codecs'][1]['configuration']['shuffle'] == 'shuffle'


def test_one_thousand_random_cell_crops_have_a_generous_time_bound(tmp_path, record_property):
    """A thousand 64-pixel crops avoid the former whole-field decoding cost."""
    rng = np.random.default_rng(20261121)
    pixels = rng.integers(0, 4096, size=(2, 1024, 1280), dtype=np.uint16)
    writer = PlateWriter(tmp_path, 'P1', ngff_version='0.6')
    key = writer.write_field('A01', 1, pixels)
    image = read_ome_zarr(writer.path / key)
    positions = rng.integers([0, 0], [1024 - 64, 1280 - 64], size=(1000, 2))
    image.read(region={'y': (0, 64), 'x': (0, 64)})
    observed = expected = 0
    started = perf_counter()
    for y, x in positions:
        crop = image.read(region={'y': (int(y), int(y + 64)), 'x': (int(x), int(x + 64))})
        assert crop.shape == (2, 64, 64)
        observed += int(crop.sum(dtype=np.uint64))
        expected += int(pixels[:, y:y + 64, x:x + 64].sum(dtype=np.uint64))
    elapsed = perf_counter() - started
    record_property('crop_count', 1000)
    record_property('crop_seconds', elapsed)
    assert observed == expected
    assert elapsed < 30.0, f'1000 random 64-pixel crops took {elapsed:.2f}s'
