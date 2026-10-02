"""Organized-stack Live Preview uses the batch spectral correction."""
from copy import deepcopy

import numpy as np
import pytest

from spacr.cancellation import PipelineCancelled
from spacr.psf_pipeline import _prepare_unmixing, prepare_psf
from spacr.qt.widgets import live_preview as lp


@pytest.fixture
def plate(tmp_path):
    stack = tmp_path / 'stack'
    stack.mkdir()
    matrix = np.array([[1., .25], [.5, 1.]])
    first = np.zeros((32, 32))
    first[4:12, 4:12] = 1200
    second = np.zeros((32, 32))
    second[19:27, 18:26] = 800
    fields = {}
    for well, dyes in [('A01', (first, first * 0)),
                       ('B01', (second * 0, second)),
                       ('C01', (first, second))]:
        values = np.stack(dyes, axis=-1) @ matrix.T + 100
        path = stack / f'plate1_{well}_1.npy'
        np.save(path, values.astype(np.uint16))
        fields[well] = path
    return fields, np.stack((first, second), axis=-1) + 100


def config(**extra):
    return dict(unmix=True, unmix_controls='0:A01; 1:B01',
                unmix_background_percentile=5., organelle_method='otsu',
                **extra)


def request(plate, settings=None):
    path = plate[0]['C01']
    loaded = lp.load_source_payload(path)
    assert loaded['error'] == ''
    return lp.PreviewRequest(
        loaded['array'], source_path=str(loaded['path']),
        object_types=('organelle', 'organelleb'),
        channels={'organelle': 1, 'organelleb': 0},
        preprocess_settings=settings if settings is not None else config())


@pytest.mark.parametrize('psf', [False, True])
def test_actual_loader_to_segmentation_matches_batch_pixels_and_keeps_source(
        plate, monkeypatch, psf):
    settings = config()
    if psf:
        settings.update(psf_operation='convolve', psf_source='gaussian',
                        psf_image_sampling_um=[1., 1.], psf_fwhm_um=[2., 2.])
    req = request(plate, settings)
    original = req.image.copy()
    before = {p: p.read_bytes() for p in plate[0].values()}
    batch = _prepare_unmixing(settings, plate[0]['C01'].parent)
    expected = batch.apply(original)
    np.testing.assert_allclose(expected, plate[1], atol=1e-4)
    if psf:
        expected = prepare_psf(settings).apply(expected)
    inputs = []
    actual_segment = lp._classical_organelle_mask

    def inspect_pixels(image, role, options):
        inputs.append(image.copy())
        return actual_segment(image, role, options)

    monkeypatch.setattr(lp, '_classical_organelle_mask', inspect_pixels)
    monkeypatch.setattr(lp, 'preview_cellpose_model',
                        lambda name: pytest.fail('classical preview loaded a model'))
    masks, flows = lp._segment_multi(req)
    assert not flows
    assert all(np.any(mask) for mask in masks.values())
    for actual, channel in zip(inputs, [1, 0]):
        np.testing.assert_allclose(actual, expected[..., channel], atol=1e-4)
    np.testing.assert_array_equal(req.image, original)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert req.provenance['unmixing']['matrix'] == batch.provenance()['matrix']
    assert req.provenance['unmixing']['fields'] == batch.provenance()['fields']
    assert req.provenance['filter_intensity_source'] == 'original loaded preview field'
    assert req.provenance['input_modified'] is False


def test_disabled_unmixing_is_exact_identity_without_control_reads(plate):
    req = request(plate, {'unmix': False})
    req.source_path = 'nonexistent.tif'
    actual, provenance = lp._unmix_preview_field(req)
    assert actual is req.image and provenance is None


@pytest.mark.parametrize('failure', ['missing', 'flat', 'channels', 'raw', 'merged', 'shape'])
def test_invalid_sources_fail_before_segmentation(plate, monkeypatch, failure):
    req = request(plate)
    if failure == 'missing':
        req.preprocess_settings['unmix_controls'] = '0:D01'
    elif failure == 'flat':
        np.save(plate[0]['A01'], np.ones_like(req.image))
    elif failure == 'channels':
        np.save(plate[0]['A01'], np.ones((*req.image.shape[:2], 3)))
    elif failure == 'raw':
        req.source_path = str(plate[0]['C01'].with_suffix('.tif'))
    elif failure == 'merged':
        path = plate[0]['C01'].parent.parent / 'merged'
        path.mkdir()
        target = path / plate[0]['C01'].name
        np.save(target, req.image)
        req.source_path = str(target)
    else:
        req.image = req.image[..., 0]
    monkeypatch.setattr(lp, '_classical_organelle_mask',
                        lambda *a: pytest.fail('invalid source reached segmentation'))
    with pytest.raises(ValueError):
        lp._segment_multi(req)


def test_cancel_during_control_read_stops_before_next_stage(plate, monkeypatch):
    req = request(plate)
    real_load = np.load
    reads = []

    def read(path, **kwargs):
        reads.append(path)
        result = real_load(path, **kwargs)
        req.cancel.set()
        return result

    monkeypatch.setattr(np, 'load', read)
    with pytest.raises(PipelineCancelled):
        lp._segment_multi(req)
    assert len(reads) == 1


def test_request_settings_not_modified_by_unmix_plan(plate):
    req = request(plate)
    original = deepcopy(req.preprocess_settings)
    lp._unmix_preview_field(req)
    assert req.preprocess_settings == original


@pytest.mark.parametrize('array', [np.zeros((2, 3, 4, 5)), np.zeros((0, 4)),
                                   np.array([['not pixels']])])
def test_bad_npy_layout_is_reported_by_actual_loader(tmp_path, array):
    path = tmp_path / 'bad.npy'
    np.save(path, array)
    payload = lp.load_source_payload(path)
    assert payload['array'] is None and payload['error']


def test_stack_directory_discovery_can_load_numpy_fields(plate):
    payload = lp.load_source_payload(plate[0]['C01'].parent)
    assert payload['error'] == ''
    assert payload['path'].suffix == '.npy'
    assert payload['array'].shape == (32, 32, 2)
