"""Scientific input, cancellation, and archive reuse contracts for Mask PSFs."""
import json
import pickle

import numpy as np
import pytest

from spacr.psf_pipeline import (
    prepare_psf, validate_psf_resume, _prepare_segmentation_psf,
)
from tests.test_io_v1_illumination import _settings


def _input(root, shapes=((15, 17), (12, 13))):
    stack = root / 'stack'
    stack.mkdir(parents=True)
    for i, shape in enumerate(shapes):
        rows, cols = np.indices(shape)
        a = ((rows * 23 + cols * 47) % 79 + 100).astype(np.uint16)
        a[0, -1] = 1500
        np.save(stack / f'plate1_A01_F00{i}.npy',
                np.stack([a, a * 2, np.full(shape, 777, dtype=np.uint16)], axis=-1))
    return stack


def _configuration(stack, operation='convolve'):
    settings = _settings(stack)
    settings.update(psf_operation=operation, psf_image_sampling_um=[.5, .5],
                    psf_fwhm_um=[1., 1.], psf_iterations=3)
    return settings


def _run(stack, settings, illumination=None):
    from spacr.io import concatenate_and_normalize
    session = _prepare_segmentation_psf(settings, stack.parent, [0, 1])
    concatenate_and_normalize(str(stack), [0, 1], settings=settings,
                              illumination_session=illumination,
                              psf_session=session)
    return session


@pytest.mark.parametrize('operation', ['convolve', 'deconvolve'])
def test_mask_uses_full_field_float_psf_before_padding_and_normalizing(tmp_path, operation):
    from spacr.io import _normalize_img_batch
    stack = _input(tmp_path)
    settings = _configuration(stack, operation)
    originals = {p.name: p.read_bytes() for p in stack.glob('*.npy')}
    session = _run(stack, settings)
    transformed = []
    for p in sorted(stack.glob('*.npy')):
        raw = np.load(p)
        processed = session.plan.apply(raw[..., :2])
        assert np.any(processed != processed.astype(np.uint16))
        private = raw.astype(np.float32)
        private[..., :2] = processed
        transformed.append(private)
        assert p.read_bytes() == originals[p.name]
    expected = np.stack([np.pad(a, ((0, 15-a.shape[0]), (0, 17-a.shape[1]), (0, 0)))
                         for a in transformed])
    expected = _normalize_img_batch(expected, [0, 1], np.float32, settings)[..., :2]
    with np.load(next((tmp_path / 'masks').glob('*.npz'))) as result:
        indices = [int(str(name).removesuffix('.npy')[-1]) for name in result['filenames']]
        np.testing.assert_allclose(result['data'], expected[indices], atol=2e-6)
        fields = [str(n).removesuffix('.npy') for n in result['filenames']]
    validate_psf_resume(settings, tmp_path, [0, 1], expected_fields=fields)
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    assert record['complete'] and len(record['archives']) == 1
    assert record['configuration']['measurement_intensity_source'] == 'original persisted intensities'
    assert record['configuration']['processing']['operation'] == operation


def test_illumination_precedes_psf_and_only_selected_channels_are_processed(tmp_path):
    stack = _input(tmp_path, ((15, 17),))
    settings = _configuration(stack)
    session = _prepare_segmentation_psf(settings, tmp_path, [1])
    from spacr.io import _correct_v1_segmentation_batch
    image = np.load(next(stack.glob('*.npy')))[None]
    original = image.copy()

    class Illumination:
        def correct(self, field, values, context):
            assert values.dtype == np.uint16
            result = values.copy()
            result[:, :5] //= 2
            return result

    result, _ = _correct_v1_segmentation_batch(
        image, ['plate1_A01_F000.npy'], [1], settings, Illumination(), session)
    illuminated = image[0, ..., [1]].transpose(1, 2, 0).copy()
    illuminated[:, :5] //= 2
    np.testing.assert_allclose(result[0, ..., 1], session.plan.apply(illuminated)[..., 0])
    np.testing.assert_array_equal(result[..., [0, 2]], original[..., [0, 2]])
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('change', ['operation', 'kernel', 'channels', 'archive', 'incomplete', 'field'])
def test_reuse_refuses_mismatched_or_incomplete_inputs(tmp_path, change):
    stack = _input(tmp_path, ((15, 17),))
    settings = _configuration(stack)
    _run(stack, settings)
    fields, channels = ['plate1_A01_F000'], [0, 1]
    if change == 'operation':
        settings['psf_operation'] = 'none'
    elif change == 'kernel':
        settings['psf_fwhm_um'] = [2., 2.]
    elif change == 'channels':
        channels.reverse()
    elif change == 'archive':
        with open(next((tmp_path / 'masks').glob('*.npz')), 'ab') as stream:
            stream.write(b'changed')
    elif change == 'field':
        fields.append('missing')
    else:
        _prepare_segmentation_psf(settings, tmp_path, channels)
    with pytest.raises(ValueError, match='Enable preprocessing'):
        validate_psf_resume(settings, tmp_path, channels, expected_fields=fields)


def test_switching_off_rebuilds_unprocessed_inputs_and_invalidates_old_masks(tmp_path):
    from spacr.io import _resume_normalized_archives
    stack = _input(tmp_path, ((15, 17),))
    settings = _configuration(stack)
    _run(stack, settings)
    settings['psf_operation'] = 'none'
    assert not _resume_normalized_archives(settings, str(tmp_path), [0, 1])
    stale = tmp_path / 'masks/cell_mask_stack'
    stale.mkdir()
    (stale / 'old.npy').write_bytes(b'old')
    _run(stack, settings)
    assert not stale.exists()
    validate_psf_resume(settings, tmp_path, [0, 1], expected_fields=['plate1_A01_F000'])
    baseline = tmp_path / 'baseline'
    baseline_stack = _input(baseline, ((15, 17),))
    _run(baseline_stack, _configuration(baseline_stack, 'none'))
    with np.load(next((tmp_path / 'masks').glob('*.npz'))) as actual, \
            np.load(next((baseline / 'masks').glob('*.npz'))) as expected:
        np.testing.assert_array_equal(actual['data'], expected['data'])


def test_cancel_leaves_old_archives_intact_and_provenance_incomplete(tmp_path):
    from spacr.cancellation import CancellationToken, installed_token, PipelineCancelled
    stack = _input(tmp_path)
    settings = _configuration(stack)
    _run(stack, settings)
    old = {p.name: p.read_bytes() for p in (tmp_path / 'masks').glob('*.npz')}
    token = CancellationToken()
    token.cancel()
    with installed_token(token), pytest.raises(PipelineCancelled):
        _run(stack, settings)
    assert {p.name: p.read_bytes() for p in (tmp_path / 'masks').glob('*.npz')} == old
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    assert not record['complete']
    assert not list(tmp_path.glob('.spacr_v1_npz_*'))


def test_measured_snapshot_is_picklable_and_changes_only_on_next_prepare(tmp_path):
    path = tmp_path / 'kernel.npy'
    np.save(path, np.ones((3, 3)))
    settings = {'psf_operation': 'convolve', 'psf_source': 'measured',
                'psf_path': str(path), 'psf_image_sampling_um': [1., 1.],
                'psf_kernel_sampling_um': [1., 1.]}
    plan = prepare_psf(settings)
    np.save(path, np.eye(3))
    assert plan == pickle.loads(pickle.dumps(plan))
    assert plan.provenance() != prepare_psf(settings).provenance()
    settings['psf_kernel_sampling_um'] = [.5, .5]
    with pytest.raises(ValueError, match='sampling'):
        prepare_psf(settings)


@pytest.mark.parametrize('key,value', [('psf_image_sampling_um', None),
                                     ('psf_iterations', 0), ('psf_iterations', True),
                                     ('psf_operation', 'invalid'),
                                     ('psf_source', 'invalid')])
def test_invalid_configuration_is_refused(key, value):
    settings = {'psf_operation': 'convolve', 'psf_image_sampling_um': [1., 1.],
                'psf_fwhm_um': [2., 2.], key: value}
    with pytest.raises(ValueError):
        prepare_psf(settings)


def test_disabled_plan_requires_no_calibration_or_kernel():
    assert prepare_psf({}) is None
    assert prepare_psf({'psf_operation': 'none', 'psf_path': '/missing'}) is None


def test_preprocess_entry_rebuilds_changed_psf_and_then_reuses_exact_inputs(tmp_path):
    from spacr.io import preprocess_img_data
    stack = _input(tmp_path, ((15, 17),))
    settings = _configuration(stack)
    settings.update(verbose=False, test_mode=False)
    settings, _ = preprocess_img_data(settings)
    first = next((tmp_path / 'masks').glob('*.npz')).read_bytes()
    record = (tmp_path / 'psf/segmentation_application.json').read_bytes()
    preprocess_img_data(settings)
    assert (tmp_path / 'psf/segmentation_application.json').read_bytes() == record
    assert next((tmp_path / 'masks').glob('*.npz')).read_bytes() == first
    settings['psf_fwhm_um'] = [2., 2.]
    preprocess_img_data(settings)
    assert next((tmp_path / 'masks').glob('*.npz')).read_bytes() != first


def test_v2_model_sees_processed_channels_while_stored_intensities_stay_raw(tmp_path, monkeypatch):
    from tests.test_pipeline_v2_illumination import _stack, _raw, _CaptureModel
    from spacr.pipeline_v2 import stream_masks_from_stack
    monkeypatch.setattr('cellpose.models.CellposeModel', _CaptureModel)
    stack = _stack(tmp_path)
    settings = _configuration(tmp_path / 'stack')
    session = _prepare_segmentation_psf(settings, tmp_path, [1, 0], pipeline_style='v2')
    stream_masks_from_stack([stack], channels_for_cellpose=(1, 0),
                            batch_fields=1, psf_session=session)
    expected = session.plan.apply(_raw()[..., [1, 0]])
    np.testing.assert_allclose(_CaptureModel.received[0], expected / expected.max())
    stored = np.load(stack.path)
    np.testing.assert_array_equal(stored[..., :2], _raw())
    assert stored[..., -1].max() == 7
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    assert record['complete']
    assert record['configuration']['channels'] == [1, 0]


def test_run_v2_captures_psf_from_pipeline_settings(tmp_path, monkeypatch):
    import tifffile
    from tests.test_pipeline_v2_illumination import _CaptureModel
    from spacr.pipeline_v2 import run_v2
    monkeypatch.setattr('cellpose.models.CellposeModel', _CaptureModel)
    raw = np.arange(256, dtype=np.uint16).reshape(16, 16)
    tifffile.imwrite(tmp_path / 'plate1_A01_T01F01L01A01Z01C00.tif', raw)
    settings = _configuration(tmp_path / 'stack')
    settings.update(cell_channel=0, nucleus_channel=None, channels=[0])
    result = run_v2(tmp_path, channels=(0,), channels_for_cellpose=(0,),
                    postprocess_settings=settings)
    np.testing.assert_array_equal(np.load(result['stacks'][0].path)[..., 0], raw)
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    assert record['complete'] and record['configuration']['pipeline_style'] == 'v2'


def test_preflight_reports_uncalibrated_psf_before_any_processing(tmp_path):
    from spacr.validate import validate_settings
    problems = validate_settings({'src': str(tmp_path), 'psf_operation': 'convolve'}, 'mask')
    assert any(p.setting == 'psf_operation' and 'PSF preparation failed' in p.message for p in problems)
    assert not (tmp_path / 'psf').exists()


def test_timelapse_processes_each_frame_independently(tmp_path):
    from spacr.io import concatenate_and_normalize
    stack = tmp_path / 'stack'
    stack.mkdir()
    images = []
    for time in (1, 2):
        image = np.full((8, 8, 1), 100, np.uint16)
        image[time, time] = 3000
        np.save(stack / f'plate1_A01_1_{time}.npy', image)
        images.append(image)
    settings = _configuration(stack)
    settings.update(timelapse=True, cell_channel=0, nucleus_channel=None)
    session = _prepare_segmentation_psf(settings, tmp_path, [0])
    concatenate_and_normalize(str(stack), [0], settings=settings, psf_session=session)
    with np.load(next((tmp_path / 'masks').glob('*.npz'))) as archive:
        assert archive['data'].shape == (2, 8, 8, 1)
        assert not np.array_equal(archive['data'][0], archive['data'][1])
    validate_psf_resume(settings, tmp_path, [0],
                        expected_fields=['plate1_A01_1_1', 'plate1_A01_1_2'])
