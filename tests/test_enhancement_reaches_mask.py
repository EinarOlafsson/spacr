"""Item 508: the Make Masks enhancement chain reaches Mask generation.

The chain is written as ``enhance_*`` settings by
:func:`spacr.qt.detect_chain.chain_settings` and read back by
:func:`spacr.psf_pipeline.prepare_chain`; a plate run applies it per
selected channel at the PSF stage, records it, and refuses a resume made
with a different chain. With every step off nothing changes.
"""
import json

import numpy as np
import pytest

from spacr.psf_pipeline import (
    _prepare_segmentation_psf, apply_chain, chain_problems, prepare_chain,
    prepare_psf, processing_requested, validate_psf_resume,
)
from spacr.qt import detect_chain as dc
from tests.test_psf_mask_pipeline import _configuration, _input, _run


def _field(shape=(40, 48), seed=3):
    rng = np.random.default_rng(seed)
    image = rng.normal(200.0, 20.0, shape).astype(np.float32)
    image[10:20, 12:24] += 900.0
    image[0, 0] = 60000.0
    return image


def _unit(image):
    low, high = float(image.min()), float(image.max())
    return (image - low) / (high - low), low, high - low


def test_log_transform_is_the_documented_curve():
    image = _field()
    out = dc.prepare(image, dc.Chain(log=True, log_gain=5.0))
    unit, low, span = _unit(image)
    expected = np.log1p(unit * 5.0) / np.log1p(5.0) * span + low
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-2)


def test_square_root_is_gamma_one_half():
    image = _field()
    np.testing.assert_allclose(dc.prepare(image, dc.Chain(sqrt=True)),
                               dc.prepare(image, dc.Chain(gamma=0.5)),
                               rtol=1e-5, atol=1e-2)


def test_percentile_clip_keeps_units_and_needs_no_round_trip():
    image = _field()
    out = dc.prepare(image, dc.Chain(percentile_clip=True, percentile_low=2,
                                     percentile_high=98))
    low, high = np.percentile(image, (2, 98))
    np.testing.assert_array_equal(out, np.clip(image, low, high))


def test_clip_runs_before_the_curves():
    image = _field()
    clipped = np.clip(image, *np.percentile(image, (1, 99))).astype(np.float32)
    np.testing.assert_array_equal(
        dc.prepare(image, dc.Chain(percentile_clip=True, gamma=0.7)),
        dc.prepare(clipped, dc.Chain(gamma=0.7)))


def test_gaussian_filter_and_total_variation_denoise():
    from skimage.filters import gaussian
    from skimage.restoration import denoise_tv_chambolle
    image = _field()
    np.testing.assert_allclose(
        dc.prepare(image, dc.Chain(denoise='gaussian', denoise_strength=2.0)),
        gaussian(image, sigma=2.0, preserve_range=True), rtol=1e-5)
    unit, low, span = _unit(image)
    np.testing.assert_allclose(
        dc.prepare(image, dc.Chain(denoise='tv', denoise_strength=1.0)),
        denoise_tv_chambolle(unit, weight=0.1) * span + low, rtol=1e-4, atol=1e-2)


def test_new_steps_are_named_and_recorded():
    chain = dc.Chain(percentile_clip=True, log=True, sqrt=True, denoise='tv')
    names = dc.step_names(chain)
    assert 'square root' in names and 'total-variation denoise' in names
    assert any(name.startswith('logarithm') for name in names)
    steps = dc.provenance(chain)['enhancement']
    assert steps['percentile_clip'] == [1.0, 99.0]
    assert steps['log'] and steps['log_gain'] == 10.0 and steps['sqrt']


def test_strict_prepare_raises_instead_of_skipping(monkeypatch):
    def broken(image, chain):
        raise RuntimeError('broken step')
    monkeypatch.setattr(dc, '_contrast', broken)
    chain = dc.Chain(gamma=0.5)
    assert dc.prepare(_field(), chain).shape == (40, 48)
    with pytest.raises(RuntimeError, match='broken step'):
        dc.prepare(_field(), chain, strict=True)


CHAIN = dc.Chain(background='tophat', background_radius=5,
                 background_scale=1.0, denoise='gaussian',
                 denoise_strength=1.5, percentile_clip=True, gamma=0.8,
                 log=True, log_gain=4.0, sqrt=True, clahe=True, clahe_tile=16,
                 equalize=False, sharpen=True, sharpen_radius=1.2,
                 sharpen_amount=0.5, morphology='open', split=True)


def test_settings_round_trip_to_the_same_chain_and_the_same_image():
    settings = dc.chain_settings(CHAIN)
    assert set(settings) == {'enhance_' + f for f in dc.SETTINGS_FIELDS}
    json.dumps(settings)
    chain = prepare_chain(json.loads(json.dumps(settings)))
    image = np.stack([_field(seed=1), _field(seed=2)], axis=-1)
    applied = apply_chain(image, chain)
    image_only = CHAIN._replace(morphology='none', split=False)
    assert chain == image_only
    for channel in range(2):
        np.testing.assert_array_equal(
            applied[..., channel], dc.prepare(image[..., channel], CHAIN))


def test_settings_file_words_are_read():
    chain = prepare_chain({'enhance_log': 'true', 'enhance_log_gain': '3',
                           'enhance_background_radius': '7.0'})
    assert chain.log is True and chain.log_gain == 3.0
    assert chain.background_radius == 7


def test_defaults_are_no_chain_and_match_the_settings_module():
    from spacr.settings import set_default_settings_preprocess_generate_masks
    defaults = set_default_settings_preprocess_generate_masks({})
    written = dc.chain_settings()
    assert {k: defaults[k] for k in written} == written
    assert prepare_chain(defaults) is None
    assert not processing_requested(defaults)
    assert processing_requested({'enhance_gamma': 0.5})
    assert processing_requested({'enhance_denoise': 'bogus'})


@pytest.mark.parametrize('key,value', [
    ('enhance_background', 'bogus'), ('enhance_background_radius', 0),
    ('enhance_background_scale', 1.5), ('enhance_denoise', 'bogus'),
    ('enhance_denoise_strength', 0), ('enhance_percentile_low', 99.5),
    ('enhance_gamma', 0), ('enhance_log_gain', -1), ('enhance_clahe_tile', 4),
    ('enhance_clahe_clip', 0), ('enhance_sharpen_radius', 0),
    ('enhance_sharpen_amount', -1), ('enhance_log', 'maybe'),
    ('enhance_gamma', 'x'),
])
def test_bad_values_are_refused_before_processing(key, value):
    assert [k for k, _ in chain_problems({key: value})] == [key]
    with pytest.raises(ValueError, match=key):
        prepare_chain({key: value})


def test_preflight_reports_a_bad_chain(tmp_path):
    from spacr.validate import validate_settings
    problems = validate_settings({'src': str(tmp_path),
                                  'enhance_denoise': 'bogus'}, 'mask')
    assert any(p.setting == 'enhance_denoise' for p in problems)


def test_three_dimensional_psf_is_refused_with_a_chain():
    plan = prepare_psf({'psf_operation': 'convolve',
                        'psf_image_sampling_um': [1., .5, .5],
                        'psf_fwhm_um': [2., 1., 1.]}, ndim=3)
    with pytest.raises(ValueError, match='two-dimensional'):
        prepare_chain({'enhance_gamma': 0.5}, plan)


def test_apply_chain_wants_a_channel_axis():
    with pytest.raises(ValueError, match='channel axis'):
        apply_chain(_field(), dc.Chain(gamma=0.5))


def _enhanced(stack, operation='none'):
    settings = _configuration(stack, operation)
    settings.update(enhance_background='rolling_ball',
                    enhance_background_radius=4,
                    enhance_background_scale=1.0,
                    enhance_log=True, enhance_percentile_clip=True)
    return settings


@pytest.mark.parametrize('operation', ['none', 'convolve'])
def test_v1_applies_the_chain_per_selected_channel_and_records_it(
        tmp_path, operation):
    from spacr.io import _normalize_img_batch
    stack = _input(tmp_path, ((15, 17),))
    settings = _enhanced(stack, operation)
    original = {p.name: p.read_bytes() for p in stack.glob('*.npy')}
    session = _run(stack, settings)
    raw = np.load(next(stack.glob('*.npy')))
    chain = prepare_chain(settings, prepare_psf(settings))
    assert (chain.psf_operation == operation)
    private = raw.astype(np.float32)
    private[..., :2] = apply_chain(raw[..., :2], chain)
    expected = _normalize_img_batch(private[None], [0, 1], np.float32,
                                    settings)[..., :2]
    with np.load(next((tmp_path / 'masks').glob('*.npz'))) as result:
        np.testing.assert_allclose(result['data'], expected, atol=2e-6)
    assert {p.name: p.read_bytes() for p in stack.glob('*.npy')} == original
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    steps = record['configuration']['enhancement']
    assert record['complete'] and steps['background'] == 'rolling_ball'
    assert steps['log'] and steps['percentile_clip'] == [1.0, 99.0]
    assert ('psf' in steps) == (operation != 'none')
    assert session.processes
    validate_psf_resume(settings, tmp_path, [0, 1],
                        expected_fields=['plate1_A01_F000'])


@pytest.mark.parametrize('change', ['off', 'gamma', 'radius'])
def test_resume_refuses_inputs_made_with_another_chain(tmp_path, change):
    stack = _input(tmp_path, ((15, 17),))
    settings = _enhanced(stack)
    _run(stack, settings)
    if change == 'off':
        settings.update(dc.chain_settings())
    elif change == 'gamma':
        settings['enhance_gamma'] = 0.5
    else:
        settings['enhance_background_radius'] = 5
    with pytest.raises(ValueError, match='Enable preprocessing'):
        validate_psf_resume(settings, tmp_path, [0, 1],
                            expected_fields=['plate1_A01_F000'])


def test_resume_entry_rebuilds_after_a_chain_change(tmp_path):
    from spacr.io import _resume_normalized_archives
    stack = _input(tmp_path, ((15, 17),))
    settings = _enhanced(stack)
    _run(stack, settings)
    assert _resume_normalized_archives(settings, str(tmp_path), [0, 1])
    settings['enhance_sqrt'] = True
    assert not _resume_normalized_archives(settings, str(tmp_path), [0, 1])


def test_chain_off_is_bit_identical_and_writes_no_record(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    plain = _configuration(_input(first, ((15, 17),)), 'none')
    assert _prepare_segmentation_psf(plain, first, [0, 1]) is None
    defaulted = _configuration(_input(second, ((15, 17),)), 'none')
    defaulted.update(dc.chain_settings())
    _run(first / 'stack', plain)
    _run(second / 'stack', defaulted)
    assert not (second / 'psf').exists()
    a = next((first / 'masks').glob('*.npz')).read_bytes()
    with np.load(next((first / 'masks').glob('*.npz'))) as x, \
            np.load(next((second / 'masks').glob('*.npz'))) as y:
        np.testing.assert_array_equal(x['data'], y['data'])
    assert a


def test_psf_alone_keeps_its_record_without_an_enhancement_key(tmp_path):
    stack = _input(tmp_path, ((15, 17),))
    settings = _configuration(stack)
    settings.update(dc.chain_settings())
    session = _run(stack, settings)
    assert session.chain is None
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    assert 'enhancement' not in record['configuration']


def test_v2_model_sees_the_chain(tmp_path, monkeypatch):
    from tests.test_pipeline_v2_illumination import _stack, _raw, _CaptureModel
    from spacr.pipeline_v2 import stream_masks_from_stack
    monkeypatch.setattr('cellpose.models.CellposeModel', _CaptureModel)
    monkeypatch.setattr(_CaptureModel, 'received', [])
    stack = _stack(tmp_path)
    settings = _enhanced(tmp_path / 'stack')
    session = _prepare_segmentation_psf(settings, tmp_path, [1, 0],
                                        pipeline_style='v2')
    stream_masks_from_stack([stack], channels_for_cellpose=(1, 0),
                            batch_fields=1, psf_session=session)
    expected = apply_chain(_raw()[..., [1, 0]], session.chain)
    np.testing.assert_allclose(_CaptureModel.received[0],
                               expected / expected.max(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.load(stack.path)[..., :2], _raw())
    record = json.loads((tmp_path / 'psf/segmentation_application.json').read_text())
    assert record['complete'] and record['configuration']['enhancement']['log']


def test_cancel_reaches_the_chain():
    from threading import Event
    from spacr.cancellation import PipelineCancelled
    event = Event()
    event.set()
    with pytest.raises(PipelineCancelled):
        apply_chain(np.stack([_field()], -1), dc.Chain(gamma=0.5), cancel=event)
