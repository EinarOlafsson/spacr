"""OME camera bit depth is bound to the displayed TIFF plane, never guessed."""
import hashlib
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
from PIL import Image
import tifffile

from spacr import plot

NS = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'


@pytest.fixture(autouse=True)
def _output_preferences(monkeypatch):
    """Keep real exports small and independent of personal preferences."""
    monkeypatch.delenv(plot._INTEGRITY_ENV, raising=False)
    monkeypatch.setattr(plot, 'figure_output_preferences', lambda: ('png', 100))


def _raw(bits=12):
    """Two percent of pixels at the explicit detector ceiling."""
    raw = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    raw.ravel()[::50] = (1 << bits) - 1
    return raw


def _xml(pixels_specs):
    """Build standard OME Pixels/TiffData mappings for deterministic edge cases."""
    root = ET.Element('OME', xmlns=NS, UUID='urn:uuid:current-file')
    for number, spec in enumerate(pixels_specs):
        attrs = dict(ID=f'Pixels:{number}', Type='uint16', SizeX='32', SizeY='32',
                     SizeZ='1', SizeC='1', SizeT='1', DimensionOrder='XYZCT',
                     SignificantBits=str(spec.get('bits', 12)))
        attrs.update(spec.get('attributes', {}))
        image = ET.SubElement(root, 'Image', ID=f'Image:{number}')
        pixels = ET.SubElement(image, 'Pixels', attrs)
        ET.SubElement(pixels, 'Channel', ID=f'Channel:{number}:0', SamplesPerPixel='1')
        mapping = ET.SubElement(pixels, 'TiffData', spec.get('mapping', {'IFD': str(number), 'PlaneCount': '1'}))
        if 'uuid' in spec:
            ET.SubElement(mapping, 'UUID').text = spec['uuid']
    return ET.tostring(root, encoding='unicode')


def _opened(description, ifd=0):
    """Already-read metadata with an explicit displayed IFD, without pixel IO."""
    return SimpleNamespace(tag_v2={270: description}, tell=lambda: ifd)


@pytest.mark.parametrize('bits', [12, 14])
def test_real_ome_tiff_grid_export_warns_and_records_camera_ceiling(tmp_path, bits):
    raw = _raw(bits)
    original = raw.copy()
    source = tmp_path / f'camera_{bits}.ome.tif'
    tifffile.imwrite(source, raw, ome=True, metadata={'SignificantBits': bits})
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    figure = plot.plot_image_grid([str(source)], (0, 100))
    written = plot.save_figure(figure, tmp_path / f'camera_{bits}.png', integrity=True)
    report = json.loads(open(plot._provenance_sidecar_path(written), encoding='utf-8').read())
    panel = report['panels'][0]
    assert panel['sensor_range']['source'] == 'OME Pixels SignificantBits'
    assert panel['sensor_range']['ceiling'] == (1 << bits) - 1
    assert panel['sensor_range']['ifd'] == 0
    assert panel['sensor_range']['pixels_id'] == 'Pixels:0'
    assert panel['sensor_saturated'] == pytest.approx(np.mean(raw == (1 << bits) - 1))
    findings = report['integrity']['findings']
    assert len(findings) == 1
    assert findings[0]['severity'] == 'warning'
    assert str((1 << bits) - 1) in findings[0]['message']
    assert report['integrity']['warnings'] == 1
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(tifffile.imread(source), original)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    rebuilt, exact = plot._reproduce_panel(plot._provenance_sidecar_path(written), 0)
    assert exact
    assert rebuilt.shape[:2] == raw.shape


def test_real_multiseries_tiff_uses_displayed_series_only(tmp_path):
    source = tmp_path / 'two_series.ome.tif'
    with tifffile.TiffWriter(source, ome=True) as writer:
        writer.write(_raw(12), metadata={'SignificantBits': 12})
        writer.write(_raw(14), metadata={'SignificantBits': 14})
    with Image.open(source) as opened:
        raw = np.array(opened)
        info = plot._source_sensor_range(opened, raw)
    assert info['pixels_id'] == 'Pixels:0'
    assert info['significant_bits'] == 12


def test_reordered_metadata_and_selected_plane_resolve_by_ifd():
    description = _xml([{'bits': 14, 'mapping': {'IFD': '1'}},
                        {'bits': 12, 'mapping': {'IFD': '0'}}])
    first = plot._source_sensor_range(_opened(description), _raw(12))
    second = plot._source_sensor_range(_opened(description, 1), _raw(14))
    assert first['pixels_id'] == 'Pixels:1' and first['significant_bits'] == 12
    assert second['pixels_id'] == 'Pixels:0' and second['significant_bits'] == 14


@pytest.mark.parametrize('specs,reason', [
    ([{'bits': 12}, {'bits': 14, 'mapping': {'IFD': '0'}}], 'unique'),
    ([{'bits': 0}], 'outside'),
    ([{'bits': 17}], 'outside'),
    ([{'bits': '12.5'}], 'invalid literal'),
    ([{'attributes': {'Type': 'uint8'}}], 'contradict'),
    ([{'attributes': {'SizeX': '64'}}], 'contradict'),
    ([{'mapping': {'IFD': '1'}}], 'unique'),
    ([{'mapping': {'IFD': '-1'}}], 'invalid'),
    ([{'mapping': {'PlaneCount': '2'}}], 'invalid'),
    ([{'attributes': {'SizeT': '2'},
       'mapping': {'FirstT': '1', 'PlaneCount': '2'}}], 'exceeds'),
    ([{'attributes': {'DimensionOrder': 'unknown'}}], 'dimension order'),
])
def test_invalid_or_ambiguous_metadata_records_dtype_fallback(specs, reason):
    info = plot._source_sensor_range(_opened(_xml(specs)), _raw())
    assert info['source'] == 'storage dtype'
    assert info['ceiling'] == 65535
    assert reason in info['reason']


def test_other_file_uuid_cannot_supply_this_planes_bit_depth():
    description = _xml([{'bits': 14, 'uuid': 'urn:uuid:other-file'},
                        {'bits': 12, 'mapping': {'IFD': '0'}}])
    info = plot._source_sensor_range(_opened(description), _raw())
    assert info['pixels_id'] == 'Pixels:1'
    assert info['significant_bits'] == 12


def test_implicit_plane_count_falls_back_without_enumerating_file_ifds():
    info = plot._source_sensor_range(_opened(_xml([{'mapping': {}}])), _raw())
    assert info['source'] == 'storage dtype'
    assert 'no explicit IFD or PlaneCount' in info['reason']


def test_pixels_above_metadata_ceiling_refuse_metadata_and_keep_source_immutable():
    raw = _raw()
    raw[0, 0] = 5000
    before = raw.copy()
    info = plot._source_sensor_range(_opened(_xml([{}])), raw)
    assert info['source'] == 'storage dtype'
    assert 'exceed' in info['reason']
    np.testing.assert_array_equal(raw, before)
    assert plot._raw_clip_stats(raw, [], sensor_ceiling=info['ceiling'])['sensor_saturated'] == 0


@pytest.mark.parametrize('description,reason', [
    ('x' * (1024 * 1024 + 1), 'exceeds 1 MiB'),
    ('<!DOCTYPE OME [<!ENTITY test "x">]><OME/>', 'entities'),
    ('<OME>', 'no element found'),
])
def test_metadata_parse_is_bounded_without_additional_image_reads(description, reason):
    info = plot._source_sensor_range(_opened(description), _raw())
    assert info['ceiling'] == 65535
    assert reason in info['reason']


def test_unannotated_12_bit_values_do_not_imply_camera_saturation(tmp_path):
    source = tmp_path / 'plain.tif'
    raw = _raw()
    tifffile.imwrite(source, raw, metadata=None)
    figure = plot.plot_image_grid([str(source)], (0, 100))
    report = plot._integrity_report(figure, fmt='png', dpi=100)
    assert report['panels'][0]['sensor_saturated'] == 0
    assert report['panels'][0]['sensor_range']['source'] == 'storage dtype'
    assert report['integrity']['findings'] == []


def test_packed_channel_boundary_is_not_guessed():
    root = ET.fromstring(_xml([{'attributes': {'SizeC': '3'},
                               'mapping': {'FirstC': '1', 'IFD': '0'}}]))
    channel = root.find(f'{{{NS}}}Image/{{{NS}}}Pixels/{{{NS}}}Channel')
    channel.set('SamplesPerPixel', '3')
    info = plot._source_sensor_range(_opened(ET.tostring(root, encoding='unicode')), _raw())
    assert info['source'] == 'storage dtype'
    assert 'packed-channel' in info['reason']
