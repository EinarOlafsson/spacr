"""Spectral unmixing: bleed-through estimated from single-stain controls.

Synthetic two-dye plates with a known bleed-through matrix: each dye is
read in the other channel at a fixed fraction, over a flat background with
noise. The estimate recovers the matrix, the controls' off-target channel
falls to background after unmixing, and a double-stained field's
intensities move to the amounts put in.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from spacr.psf_pipeline import (
    _UNMIX_RECORD_KEY, _apply_recorded_unmixing, _configuration,
    _parse_unmix_controls, _prepare_measure_unmixing,
    _prepare_segmentation_psf, _prepare_unmixing, processing_requested)

DESCRIPTOR_RELATIVE_READS = (hasattr(os, 'O_DIRECTORY') and hasattr(os, 'O_NOFOLLOW')
                             and os.open in os.supports_dir_fd)
needs_descriptor_relative_reads = pytest.mark.skipif(
    not DESCRIPTOR_RELATIVE_READS,
    reason='signed cell loading refuses platforms without descriptor-relative reads')

BACKGROUND = 100.0
NOISE = 3.0
TRUE = np.array([[1.0, 0.12],
                 [0.35, 1.0]])
SHAPE = (96, 96)


def _dye(rng, amount):
    """A field of bright square objects of ``amount`` on zero."""
    plane = np.zeros(SHAPE)
    for _ in range(12):
        y, x = rng.integers(0, SHAPE[0] - 10, size=2)
        plane[y:y + 8, x:x + 8] = amount
    return plane


def _field(rng, amounts):
    """Readings of two channels, plus a nuclear-like third plane, as uint16."""
    dyes = np.stack(amounts, axis=-1)
    readings = dyes @ TRUE.T + BACKGROUND
    readings = readings + rng.normal(0, NOISE, readings.shape)
    third = np.full(SHAPE + (1,), 50.0)
    return np.clip(np.concatenate([readings, third], axis=-1), 0,
                   65535).astype(np.uint16), dyes


@pytest.fixture
def plate(tmp_path):
    """A stack folder with single-stain A01/B01 and double-stain C01 fields."""
    rng = np.random.default_rng(538)
    stack = tmp_path / 'stack'
    stack.mkdir()
    truth = {}
    for index in (1, 2):
        field, dyes = _field(rng, [_dye(rng, 2000.0), np.zeros(SHAPE)])
        np.save(stack / f'plate1_A01_{index}.npy', field)
        truth[f'plate1_A01_{index}'] = dyes
        field, dyes = _field(rng, [np.zeros(SHAPE), _dye(rng, 1500.0)])
        np.save(stack / f'plate1_B01_{index}.npy', field)
        truth[f'plate1_B01_{index}'] = dyes
    field, dyes = _field(rng, [_dye(rng, 1800.0), _dye(rng, 1200.0)])
    np.save(stack / 'plate1_C01_1.npy', field)
    truth['plate1_C01_1'] = dyes
    return stack, truth


def _settings(**extra):
    settings = {'unmix': True, 'unmix_controls': '0:A01; 1:B01',
                'unmix_background_percentile': 5.0}
    settings.update(extra)
    return settings


def test_controls_parse_from_text_and_mapping():
    assert _parse_unmix_controls('0:A01,A02; 1:b1\n2:C03') == {
        0: ('A01', 'A02'), 1: ('b1',), 2: ('C03',)}
    assert _parse_unmix_controls({1: 'B01', 0: ['A01']}) == {
        1: ('B01',), 0: ('A01',)}
    assert _parse_unmix_controls('') == {}
    assert _parse_unmix_controls(None) == {}
    for bad in ('A01', 'x:A01', '0:', '0:notawell'):
        with pytest.raises(ValueError):
            _parse_unmix_controls(bad)


def test_off_prepares_nothing_and_on_needs_controls(plate):
    stack, _ = plate
    assert _prepare_unmixing({'unmix': False}, stack) is None
    assert not processing_requested({'unmix': False})
    assert processing_requested({'unmix': True})
    with pytest.raises(ValueError, match='names no single-stain'):
        _prepare_unmixing(_settings(unmix_controls=''), stack)
    with pytest.raises(ValueError, match='no field'):
        _prepare_unmixing(_settings(unmix_controls='0:D01'), stack)
    with pytest.raises(ValueError, match='outside'):
        _prepare_unmixing(_settings(unmix_controls='7:A01'), stack)
    with pytest.raises(ValueError, match='percentile'):
        _prepare_unmixing(_settings(unmix_background_percentile=100), stack)


def test_the_estimate_recovers_the_bleed_through_matrix(plate):
    stack, _ = plate
    plan = _prepare_unmixing(_settings(), stack)
    matrix = np.asarray(plan.matrix)
    assert plan.channels == (0, 1, 2)
    assert np.allclose(matrix[:2, :2], TRUE, atol=0.01)
    assert np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=0.01)
    assert plan.fields == {0: ('plate1_A01_1', 'plate1_A01_2'),
                           1: ('plate1_B01_1', 'plate1_B01_2')}
    record = plan.provenance()
    assert json.loads(json.dumps(record)) == record
    assert record['controls'] == {'0': ['A01'], '1': ['B01']}


def test_off_target_signal_falls_to_background_on_the_controls(plate):
    stack, truth = plate
    plan = _prepare_unmixing(_settings(), stack)
    for stem, dye, other in (('plate1_A01_1', 0, 1), ('plate1_B01_1', 1, 0)):
        field = np.load(stack / f'{stem}.npy')
        stained = truth[stem][..., dye] > 0
        unmixed = plan.apply(field)[..., other]
        before = (field[..., other][stained].mean()
                  - field[..., other][~stained].mean())
        after = unmixed[stained].mean() - unmixed[~stained].mean()
        assert before > 100
        assert abs(after) < NOISE


def test_a_double_stain_moves_toward_the_known_amounts(plate):
    stack, truth = plate
    plan = _prepare_unmixing(_settings(), stack)
    field = np.load(stack / 'plate1_C01_1.npy')
    unmixed = plan.apply(field)
    dyes = truth['plate1_C01_1']
    empty = (dyes == 0).all(axis=-1)
    for channel in (0, 1):
        stained = dyes[..., channel] > 0
        known = dyes[..., channel][stained].mean()
        before = (field[..., channel][stained].mean()
                  - field[..., channel][empty].mean())
        after = (unmixed[..., channel][stained].mean()
                 - unmixed[..., channel][empty].mean())
        assert abs(after - known) < 0.1 * abs(before - known)
        assert abs(after - known) < 0.02 * known
    assert np.array_equal(unmixed[..., 2], field[..., 2])


def test_make_masks_unmixes_whole_fields_and_records_the_matrix(plate):
    stack, _ = plate
    root = stack.parent
    session = _prepare_segmentation_psf(_settings(), root, [1])
    assert session is not None and session.processes
    record = session.configuration['unmixing']
    assert record['matrix'] == [list(row) for row in session.unmixing.matrix]
    written = json.loads((root / 'psf' / 'segmentation_application.json')
                         .read_text(encoding="utf-8"))
    assert written['configuration']['unmixing'] == record
    assert _configuration(None, [1], 'v1')['processing'] == {'operation': 'none'}
    assert 'unmixing' not in _configuration(None, [1], 'v1')

    from spacr.io import _correct_v1_segmentation_batch

    field = np.load(stack / 'plate1_A01_1.npy')
    working, ids = _correct_v1_segmentation_batch(
        field[np.newaxis], ['plate1_A01_1.npy'], [1],
        {'timelapse': False}, None, session)
    assert ids == ('plate1_A01_1',)
    assert np.allclose(working[0][..., 1], session.unmix(field)[..., 1])
    assert np.array_equal(working[0][..., 0], field[..., 0])


def test_measure_records_the_matrix_and_unmixes_in_the_field_dtype(plate):
    stack, truth = plate
    merged = stack.parent / 'merged'
    stack.rename(merged)
    settings = _settings(src=str(merged), channels=[0, 1])
    plan = _prepare_measure_unmixing(settings)
    record = json.loads(settings[_UNMIX_RECORD_KEY])
    assert record == plan.provenance()
    assert record['channels'] == [0, 1]
    saved = json.loads((stack.parent / 'measurements' / 'bleed_through.json')
                       .read_text(encoding="utf-8"))
    assert saved == record

    field = np.load(merged / 'plate1_A01_1.npy')[..., [0, 1]]
    out = _apply_recorded_unmixing(field, settings)
    assert out.dtype == np.uint16
    stained = truth['plate1_A01_1'][..., 0] > 0
    assert abs(out[..., 1][stained].mean() - out[..., 1][~stained].mean()) < NOISE
    assert field[..., 1][stained].mean() - field[..., 1][~stained].mean() > 100

    off = {'unmix': False, 'src': str(merged), 'channels': [0, 1],
           _UNMIX_RECORD_KEY: 'stale'}
    assert _prepare_measure_unmixing(off) is None
    assert _UNMIX_RECORD_KEY not in off
    assert _apply_recorded_unmixing(field, off) is field


def _calibrated_test_blur(plane, kernel):
    """Convolve a small oracle plane by explicit centered spatial summation."""
    ry, rx = np.array(kernel.shape) // 2
    padded = np.pad(np.asarray(plane, dtype=np.float64), ((ry, ry), (rx, rx)))
    result = np.zeros_like(plane, dtype=np.float64)
    for y, x in np.ndindex(kernel.shape):
        result += kernel[y, x] * padded[2 * ry - y:2 * ry - y + plane.shape[0],
                                       2 * rx - x:2 * rx - x + plane.shape[1]]
    return result


def _calibrated_test_camera(dyes, matrix, kernels, shifts, noise=None):
    """Form nonnegative camera readings with declared sensor offsets and noise."""
    mixed = dyes @ matrix.T
    readings = np.zeros_like(mixed)
    for channel, ((dy, dx), kernel) in enumerate(zip(shifts, kernels)):
        aligned = _calibrated_test_blur(mixed[..., channel], kernel)
        if noise is not None:
            aligned += noise[..., channel]
        for y, x in np.ndindex(dyes.shape[:2]):
            if 0 <= y + dy < dyes.shape[0] and 0 <= x + dx < dyes.shape[1]:
                readings[y, x, channel] = aligned[y + dy, x + dx]
    return readings.astype(np.float32)


@pytest.fixture
def calibrated_plate(tmp_path):
    """Two single-stain wells have known spill, different PSFs and pixel shifts."""
    stack = tmp_path / 'stack'
    stack.mkdir()
    shape = (48, 56)
    matrix = np.array([[1., .6], [.8, 1.]])
    rows = (np.array([1., 2., 1.]), np.array([1., 4., 6., 4., 1.]))
    columns = np.array([1., 2., 1.])
    kernels = tuple(np.outer(row, columns) / (row.sum() * columns.sum()) for row in rows)
    shifts = ((0, 0), (1, -2))
    for channel, well in enumerate(('A01', 'B01')):
        dyes = np.zeros(shape + (2,))
        dyes[10:22, 14:27, channel] = 200.
        dyes[28:36, 30:40, channel] = 80.
        np.save(stack / f'plate1_{well}_1.npy',
                _calibrated_test_camera(dyes, matrix, kernels, shifts))
    dyes = np.zeros(shape + (2,))
    dyes[9:25, 10:27, 0] = 180.
    dyes[28:38, 31:44, 1] = 120.
    dyes[30:36, 35:41, 0] = 60.
    target = _calibrated_test_camera(dyes, matrix, kernels, shifts)
    np.save(stack / 'plate1_C01_1.npy', target)
    calibration = {
        'version': 1, 'axes': 'YX', 'pixel_size_um': [.4, .3], 'reference_channel': 0,
        'channels': [{'channel': channel, 'shift_to_reference_px': list(shift),
                      'psf': kernel.tolist(),
                      'psf_origin_px': [kernel.shape[0] // 2, kernel.shape[1] // 2],
                      'psf_pixel_size_um': [.4, .3]}
                     for channel, (shift, kernel) in enumerate(zip(shifts, kernels))]}
    calibration_path = tmp_path / 'calibration.json'
    calibration_path.write_text(json.dumps(calibration), encoding='utf-8')
    settings = _settings(unmix_background_percentile=0.,
                         unmix_calibration=str(calibration_path), unmix_pixel_size_um=[.4, .3])
    common_rows = np.convolve(rows[0], rows[1])
    common_columns = np.convolve(columns, columns)
    common = np.outer(common_rows, common_columns)
    common /= common.sum()
    expected = np.stack([_calibrated_test_blur(dyes[..., channel], common)
                         for channel in range(2)], axis=-1)
    return stack, settings, calibration_path, calibration, dyes, target, matrix, kernels, shifts, expected


def test_calibrated_controls_recover_spill_and_exact_common_blur_roi(calibrated_plate):
    stack, settings, path, _, _, target, matrix, _, _, expected = calibrated_plate
    original_files = {file: file.read_bytes() for file in stack.iterdir()}
    calibration_bytes = path.read_bytes()
    original_target = target.copy()
    plan = _prepare_unmixing(settings, stack, allow_calibrated=True)
    np.testing.assert_allclose(plan.matrix, matrix, atol=2e-6, rtol=0)
    result = plan.apply_calibrated(target)
    assert result.valid_slices == (slice(2, 46), slice(1, 53))
    assert result.source_origin_px == (2, 1)
    assert result.image.shape == (44, 52, 2) and result.image.dtype == np.float32
    np.testing.assert_allclose(result.image, expected[2:46, 1:53], atol=2e-4, rtol=0)
    assert json.loads(json.dumps(result.record)) == result.record
    assert result.record
    np.testing.assert_array_equal(target, original_target)
    assert path.read_bytes() == calibration_bytes
    assert all(file.read_bytes() == content for file, content in original_files.items())


def test_calibrated_camera_noise_keeps_signed_residuals_without_clipping(calibrated_plate):
    stack, settings, _, _, dyes, _, matrix, kernels, shifts, expected = calibrated_plate
    noise = np.zeros_like(dyes)
    noise[14:20, 15:22, 0] = .5
    noise[14:20, 15:22, 1] = -.5
    camera = _calibrated_test_camera(dyes, matrix, kernels, shifts, noise)
    assert camera.min() >= 0
    original = camera.copy()
    common_noise = np.stack((_calibrated_test_blur(noise[..., 0], kernels[1]),
                             _calibrated_test_blur(noise[..., 1], kernels[0])), axis=-1)
    expected_residual = common_noise @ np.linalg.inv(matrix).T
    plan = _prepare_unmixing(settings, stack, allow_calibrated=True)
    actual = plan.apply_calibrated(camera)
    np.testing.assert_allclose(actual.image, (expected + expected_residual)[2:46, 1:53],
                               atol=2e-4, rtol=0)
    assert actual.image[..., 1].min() < -.5
    assert np.max(np.abs(actual.image - expected[2:46, 1:53])) < 2.
    np.testing.assert_array_equal(camera, original)


def test_calibration_is_explicit_preview_only_and_legacy_apply_is_unchanged(calibrated_plate):
    stack, settings, _, _, _, target, _, _, _, _ = calibrated_plate
    with pytest.raises(ValueError):
        _prepare_unmixing(settings, stack)
    plan = _prepare_unmixing(settings, stack, allow_calibrated=True)
    with pytest.raises(ValueError):
        plan.apply(target)
    legacy_settings = {key: value for key, value in settings.items()
                       if key not in ('unmix_calibration', 'unmix_pixel_size_um')}
    legacy = _prepare_unmixing(legacy_settings, stack)
    explicit = _prepare_unmixing(legacy_settings, stack, allow_calibrated=True)
    assert explicit.matrix == legacy.matrix
    np.testing.assert_array_equal(explicit.apply(target), legacy.apply(target))
    assert legacy.apply(target).shape == target.shape


@pytest.mark.parametrize('invalid', ('axes', 'pixel-grid', 'channel-set', 'reference-shift',
                                     'psf-origin', 'psf-grid'))
def test_calibrated_geometry_refuses_incompatible_metadata(calibrated_plate, invalid):
    stack, settings, path, calibration, _, _, _, _, _, _ = calibrated_plate
    if invalid == 'axes':
        calibration['axes'] = 'ZYX'
    elif invalid == 'pixel-grid':
        settings = dict(settings, unmix_pixel_size_um=[.8, .3])
    elif invalid == 'channel-set':
        calibration['channels'][1]['channel'] = 2
    elif invalid == 'reference-shift':
        calibration['channels'][0]['shift_to_reference_px'] = [1, 0]
    elif invalid == 'psf-origin':
        calibration['channels'][1]['psf_origin_px'] = [0, 1]
    else:
        calibration['channels'][1]['psf_pixel_size_um'] = [.4, .6]
    path.write_text(json.dumps(calibration), encoding='utf-8')
    before = {file: file.read_bytes() for file in stack.iterdir()}
    with pytest.raises(ValueError):
        _prepare_unmixing(settings, stack, allow_calibrated=True)
    assert all(file.read_bytes() == content for file, content in before.items())


def test_calibrated_oversized_control_is_mapped_and_refused_before_channel_copy(
        calibrated_plate, monkeypatch):
    stack, settings, calibration_path, _, _, _, _, _, _, _ = calibrated_plate
    path = stack / 'plate1_A01_1.npy'
    path.unlink()
    oversized = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                         shape=(4097, 2048, 2))
    oversized[0, 0, 0], oversized[-1, -1, 1] = 17, 29
    oversized.flush()
    del oversized
    before_stat = path.stat()
    calibration_bytes = calibration_path.read_bytes()
    other_controls = {file: file.read_bytes() for file in stack.iterdir() if file != path}
    before_paths = {file.relative_to(stack.parent) for file in stack.parent.rglob('*')}
    real_load = np.load
    loads = []

    def observe_real_load(file, *args, **kwargs):
        result = real_load(file, *args, **kwargs)
        loads.append((str(file), dict(kwargs), isinstance(result, np.memmap)))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(np, 'load', observe_real_load)
        with pytest.raises(ValueError, match='at most 16,777,216 channel pixels'):
            _prepare_unmixing(settings, stack, allow_calibrated=True)
    assert loads and loads[0][0] == str(path)
    assert all(options.get('mmap_mode') == 'r' and options.get('allow_pickle') is False
               and mapped for _, options, mapped in loads)
    unchanged = real_load(path, mmap_mode='r', allow_pickle=False)
    assert isinstance(unchanged, np.memmap)
    assert unchanged.shape == (4097, 2048, 2) and unchanged.dtype == np.uint8
    assert unchanged[0, 0, 0] == 17 and unchanged[-1, -1, 1] == 29
    assert unchanged[0, 0, 1] == 0 and unchanged[-1, -1, 0] == 0
    del unchanged
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before_stat.st_size, before_stat.st_mtime_ns)
    assert calibration_path.read_bytes() == calibration_bytes
    assert all(file.read_bytes() == content for file, content in other_controls.items())
    assert {file.relative_to(stack.parent) for file in stack.parent.rglob('*')} == before_paths


def test_calibrated_controls_refuse_cumulative_budget_before_copying_the_next_field(
        calibrated_plate, monkeypatch):
    from spacr import psf_pipeline

    stack, settings, calibration_path, _, _, _, _, _, _, _ = calibrated_plate
    before = {file: file.read_bytes() for file in stack.iterdir()}
    calibration_bytes = calibration_path.read_bytes()
    before_paths = {file.relative_to(stack.parent) for file in stack.parent.rglob('*')}
    controls = [np.load(stack / f'plate1_{well}_1.npy', mmap_mode='r', allow_pickle=False)
                for well in ('A01', 'B01')]
    assert all(control.shape == (48, 56, 2) and control.size < 8192 for control in controls)
    assert sum(control.size for control in controls) > 8192
    monkeypatch.setattr(psf_pipeline, '_UNMIX_CALIBRATED_CONTROL_VALUES', 8192)
    with pytest.raises(ValueError, match='exceed the calibrated control-pixel budget') as failure:
        _prepare_unmixing(settings, stack, allow_calibrated=True)
    traceback = failure.value.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    assert traceback.tb_frame.f_code is _prepare_unmixing.__code__
    state = traceback.tb_frame.f_locals
    assert state['name'] == 'plate1_B01_1.npy'
    assert state['field'].shape == controls[1].shape
    np.testing.assert_array_equal(state['field'], controls[1])
    assert len(state['fields_by_dye'][0]) == 1 and state['fields_by_dye'][1] == []
    assert state['selected_field'] is state['fields_by_dye'][0][0]
    assert state['selected_field'].shape == (44, 52, 2)
    assert all(file.read_bytes() == content for file, content in before.items())
    assert calibration_path.read_bytes() == calibration_bytes
    assert {file.relative_to(stack.parent) for file in stack.parent.rglob('*')} == before_paths


@pytest.fixture
def calibrated_batch_inputs(calibrated_plate, tmp_path):
    """Use real control files packed in physical-channel order 3, 1 with well aliases."""
    from shutil import copyfile

    original_stack, settings, path, calibration, dyes, target, matrix, kernels, shifts, expected = calibrated_plate
    stack = tmp_path / 'assembled-controls'
    stack.mkdir()
    identities = {}
    for index, well in enumerate(('A01', 'B01', 'C01')):
        filename = f'stack_{index:06d}.npy'
        copyfile(original_stack / f'plate1_{well}_1.npy', stack / filename)
        identity = {'plate': 'plate1', 'well': well, 'field': '1', 'time': '', 'z': ''}
        identities[filename] = identity
        identities[f'{index:06d}'] = identity
    calibration['reference_channel'] = 3
    for physical, channel in zip((3, 1), calibration['channels']):
        channel['channel'] = physical
    path.write_text(json.dumps(calibration), encoding='utf-8')
    settings = dict(settings, unmix_controls='3:a1; 1:b1')
    return dict(stack=stack, settings=settings, calibration_path=path, dyes=dyes,
                target=target, matrix=matrix, kernels=kernels, shifts=shifts,
                expected=expected, identities=identities, output=tmp_path / 'calibrated-output')


def _calibrated_batch_session(inputs):
    """Prepare the real V2 session using read-only native NPY control loading."""
    from spacr.psf_pipeline import _initialize_calibrated_output

    def load(path):
        return np.load(path, mmap_mode='r', allow_pickle=False)

    _initialize_calibrated_output(inputs['output'], inputs['settings'])
    return _prepare_segmentation_psf(inputs['settings'], inputs['output'], [1],
        pipeline_style='v2', stack_dir=inputs['stack'], load=load,
        source_channels=(3, 1), field_identities=inputs['identities'])


def test_calibrated_v2_session_keeps_physical_order_aliases_signed_crop_and_source(
        calibrated_batch_inputs):
    inputs = calibrated_batch_inputs
    source_files = {file: file.read_bytes() for file in inputs['stack'].glob('*.npy')}
    calibration_bytes = inputs['calibration_path'].read_bytes()
    session = _calibrated_batch_session(inputs)
    assert session.calibrated and session.processes
    noise = np.zeros_like(inputs['dyes'])
    noise[14:20, 15:22, 0] = .5
    noise[14:20, 15:22, 1] = -.5
    camera = _calibrated_test_camera(inputs['dyes'], inputs['matrix'],
                                     inputs['kernels'], inputs['shifts'], noise)
    assert camera.min() >= 0
    original = camera.copy()
    result = session.prepare_calibrated(camera, '000002')
    assert result.valid_slices == (slice(2, 46), slice(1, 53))
    assert result.source_origin_px == (2, 1)
    assert result.image.shape == (44, 52, 2) and result.image.dtype == np.float32
    common_noise = np.stack((_calibrated_test_blur(noise[..., 0], inputs['kernels'][1]),
                             _calibrated_test_blur(noise[..., 1], inputs['kernels'][0])), axis=-1)
    oracle = inputs['expected'] + common_noise @ np.linalg.inv(inputs['matrix']).T
    np.testing.assert_allclose(result.image, oracle[2:46, 1:53], atol=2e-4, rtol=0)
    assert result.image[..., 1].min() < -.5
    signed = result.image.copy()
    display = session.calibrated_display(camera, result)
    assert display.shape == camera.shape and display.dtype == np.float32
    expected_display = np.zeros(camera.shape, dtype=np.float32)
    expected_display[2:46, 1:53] = np.maximum(oracle[2:46, 1:53], 0)
    np.testing.assert_allclose(display, expected_display, atol=2e-4, rtol=0)
    assert np.all(display >= 0)
    assert not np.shares_memory(display, camera) and not np.shares_memory(display, result.image)
    np.testing.assert_array_equal(result.image, signed)
    np.testing.assert_array_equal(camera, original)
    assert all(file.read_bytes() == before for file, before in source_files.items())
    assert inputs['calibration_path'].read_bytes() == calibration_bytes
    assert not (inputs['stack'].parent / 'psf').exists()
    assert json.loads(json.dumps(result.record)) == result.record
    assert result.record['source_identity'] == inputs['identities']['000002']
    assert result.record['physical_channel_order'] == [3, 1]
    record = json.loads((inputs['output'] / 'psf' / 'segmentation_application.json').read_text())
    field = record['calibrated_fields']['000002']
    assert field['source_identity'] == inputs['identities']['stack_000002.npy']
    assert field['configuration_sha256'] == hashlib.sha256(json.dumps(
        record['configuration'], sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert 'matrix' not in field and 'calibration' not in field
    assert record['configuration']['unmixing']['calibration']['common_psf']
    assert json.dumps(record).count('"common_psf":') == 1


def test_calibrated_v2_restores_exact_uint16_boundary_and_interior_ids(calibrated_batch_inputs):
    inputs = calibrated_batch_inputs
    session = _calibrated_batch_session(inputs)
    result = session.prepare_calibrated(inputs['target'], '000002')
    labels = np.zeros(result.image.shape[:2], dtype=np.uint16)
    labels[0:4, 3:8] = 7
    labels[10:13, 12:15] = 7
    labels[17:20, 21:24] = 65535
    original = labels.copy()
    restored = session.restore_calibrated(labels, result, '000002')
    expected = np.zeros(inputs['target'].shape[:2], dtype=np.uint16)
    expected[19:22, 22:25] = 65535
    np.testing.assert_array_equal(restored, expected)
    assert np.max(restored) == 65535 and 7 not in restored
    np.testing.assert_array_equal(labels, original)
    session._write(False)
    record = json.loads((inputs['output'] / 'psf' / 'segmentation_application.json').read_text())
    field = record['calibrated_fields']['000002']
    assert field['excluded_boundary_label_ids'] == [7]
    assert field['valid_roi_yx'] == [[2, 46], [1, 53]]
    assert not record['complete'] and record['fields'] == []
    assert field['source_identity'] == inputs['identities']['000002']
    assert field['configuration_sha256'] == hashlib.sha256(json.dumps(
        record['configuration'], sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert 'matrix' not in field and 'calibration' not in field
    assert record['configuration']['unmixing']['calibration']['common_psf']
    assert json.dumps(record).count('"common_psf":') == 1


@pytest.mark.parametrize('bad', ['fraction', 'negative', 'overflow', 'float32-loss', 'shape'])
def test_calibrated_v2_refuses_unrepresentable_or_wrong_roi_labels(calibrated_batch_inputs, bad):
    inputs = calibrated_batch_inputs
    session = _calibrated_batch_session(inputs)
    result = session.prepare_calibrated(inputs['target'], '000002')
    labels = np.zeros(result.image.shape[:2], dtype=np.float64)
    labels[4:8, 4:8] = {'fraction': 1.5, 'negative': -1., 'overflow': 65536.,
                         'float32-loss': float(2 ** 24 + 1), 'shape': 1.}[bad]
    if bad == 'shape':
        labels = np.pad(labels, ((1, 0), (0, 0)))
    before = labels.copy()
    with pytest.raises(ValueError):
        session.restore_calibrated(labels, result, '000002')
    np.testing.assert_array_equal(labels, before)
    assert not (inputs['output'] / 'stack_000002.npy').exists()


def test_calibrated_output_marker_precedes_images_and_refuses_prior_output(calibrated_batch_inputs):
    from spacr.psf_pipeline import _initialize_calibrated_output

    inputs = calibrated_batch_inputs
    before = {file: file.read_bytes() for file in inputs['stack'].glob('*.npy')}
    _initialize_calibrated_output(inputs['output'], inputs['settings'])
    path = inputs['output'] / 'psf' / 'segmentation_application.json'
    marker = json.loads(path.read_text())
    assert marker['configuration']['version'] == 1
    assert marker['configuration']['pipeline_style'] == 'v2'
    assert marker['configuration']['coordinate_system'] == 'calibrated_reference_canvas'
    assert marker['configuration']['measurement_intensity_source'] == 'nonquantitative calibrated display; measurements unsupported'
    assert marker['complete'] is False and marker['fields'] == [] and marker['archives'] == {}
    assert not tuple(inputs['output'].glob('*.npy'))
    marker_bytes = path.read_bytes()
    with pytest.raises(ValueError):
        _initialize_calibrated_output(inputs['output'], inputs['settings'])
    assert path.read_bytes() == marker_bytes
    assert all(file.read_bytes() == content for file, content in before.items())


@pytest.fixture
def completed_calibrated_cells(calibrated_batch_inputs):
    """Persist a real signed session with an explicit cell role and matching canvas."""
    inputs = calibrated_batch_inputs
    session = _calibrated_batch_session(inputs)
    result = session.prepare_calibrated(inputs['target'], '000002')
    labels = np.zeros(result.image.shape[:2], dtype=np.uint16)
    labels[0:3, 2:5] = 7
    labels[12:17, 14:20] = 11
    mask = session.restore_calibrated(labels, result, '000002')
    display = session.calibrated_display(inputs['target'], result)
    output = inputs['output'] / 'merged' / 'stack_000002.npy'
    signed = inputs['output'] / 'signed' / 'stack_000002.npy'
    output.parent.mkdir(); signed.parent.mkdir()
    np.save(output, np.concatenate((display, mask[..., None]), axis=-1).astype(np.float32))
    np.save(signed, result.image)
    session.mark_completed('000002', output_path=output, signed_path=signed, mask_role='cell')
    session.finish({'000002'})
    return dict(inputs=inputs, session=session, result=result, mask=mask,
                output=output, signed=signed,
                record=inputs['output'] / 'psf/segmentation_application.json')


def _rewrite_calibrated_cell_record(path, record):
    """Rebind deliberate metadata changes so semantic refusals cannot rely on stale hashes."""
    from spacr.psf_pipeline import _cell_record_hash

    configuration_sha = _cell_record_hash(record['configuration'])
    for row in record['calibrated_fields'].values():
        row['configuration_sha256'] = configuration_sha
        row['binding_sha256'] = _cell_record_hash({key: value for key, value in row.items()
                                                 if key != 'binding_sha256'})
    path.write_text(json.dumps(record, allow_nan=False))


@pytest.mark.skipif(DESCRIPTOR_RELATIVE_READS, reason='this platform reads signed cells')
def test_signed_cell_loader_refuses_a_platform_without_descriptor_relative_reads(completed_calibrated_cells):
    """Refuse the whole dataset before exposing any field or touching its bytes."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    root = completed_calibrated_cells['inputs']['output']
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='POSIX descriptor-relative file access'):
        with _open_calibrated_cell_dataset(root):
            pytest.fail('a platform without descriptor-relative reads exposed a field')
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


@needs_descriptor_relative_reads
def test_signed_cell_record_serializes_role_and_streams_matching_readonly_maps(completed_calibrated_cells):
    """Consume signed pixels, explicit physical positions and exact ROI cell labels."""
    from spacr.psf_pipeline import _cell_record_hash, _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    root = sample['inputs']['output']
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    record = json.loads(sample['record'].read_text())
    row = record['calibrated_fields']['000002']
    assert record['complete'] is True
    assert row['measurement_schema_version'] == 1 and row['mask_role'] == 'cell'
    assert row['mask_channel_position'] == 2
    assert row['binding_sha256'] == _cell_record_hash({key: value for key, value in row.items()
                                                     if key != 'binding_sha256'})
    with _open_calibrated_cell_dataset(root, channels=(1,)) as dataset:
        assert dataset.record_sha256 == hashlib.sha256(sample['record'].read_bytes()).hexdigest()
        assert dataset.channels == (1,) and dataset.physical_channel_order == (3, 1)
        assert dataset.pixel_size_um == tuple(record['configuration']['unmixing']['calibration']['pixel_size_um'])
        assert dataset.provenance() == record
        iterator = dataset.fields()
        field = next(iterator)
        signed = field['signed']
        assert isinstance(signed, np.memmap) and not signed.flags.writeable
        assert field['labels'].dtype == np.uint16 and not field['labels'].flags.writeable
        np.testing.assert_array_equal(signed, sample['result'].image)
        np.testing.assert_array_equal(field['labels'], sample['mask'][2:46, 1:53])
        assert field['channel_positions'] == {3: 0, 1: 1}
        assert field['source_origin_px'] == (2, 1) and field['source_shape'] == (48, 56, 2)
        assert field['source_identity'] == dict(sample['inputs']['identities']['000002'],
                                                source_dataset=record['configuration']['source_dataset'])
        assert field['provenance'] == row and set(np.unique(field['labels'])) == {0, 11}
        with pytest.raises(ValueError):
            signed[0, 0, 0] = 5
        with pytest.raises(StopIteration):
            next(iterator)
        assert signed._mmap.closed
    assert all(path.read_bytes() == content for path, content in before.items())
    import subprocess
    import sys

    headless = subprocess.run([sys.executable, '-c', """
import sys
from spacr.psf_pipeline import _open_calibrated_cell_dataset
with _open_calibrated_cell_dataset(sys.argv[1]) as dataset:
    for field in dataset.fields():
        assert not field['signed'].flags.writeable
try:
    with _open_calibrated_cell_dataset(sys.argv[1], channels=(999,)):
        raise AssertionError('unknown physical channel accepted')
except ValueError:
    pass
assert not any(name == 'PySide6' or name.startswith('PySide6.') for name in sys.modules)
print('headless signed cell loader passed')
""", str(root)], capture_output=True, text=True, timeout=30)
    assert headless.returncode == 0, headless.stdout + headless.stderr
    assert headless.stdout.strip() == 'headless signed cell loader passed'


@needs_descriptor_relative_reads
@pytest.mark.parametrize('defect', ['role', 'old-record', 'stale-hash', 'shape', 'origin',
                                  'spacing', pytest.param('symlink', marks=pytest.mark.needs_symlink),
                                  'common-psf', 'source-identity',
                                  'duplicate-field', 'duplicate-artifact', 'high-label'])
def test_signed_cell_loader_refuses_real_inconsistent_producer_outputs(completed_calibrated_cells, defect):
    """Reject actual incompatible records and artifacts before exposing any field."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    record = json.loads(sample['record'].read_text())
    row = record['calibrated_fields']['000002']
    if defect == 'role':
        row['mask_role'] = 'nucleus'
    elif defect == 'old-record':
        row.pop('measurement_schema_version'); row.pop('mask_role')
    elif defect == 'stale-hash':
        pixels = np.load(sample['signed'], allow_pickle=False)
        pixels[0, 0, 0] += 1
        np.save(sample['signed'], pixels)
    elif defect == 'shape':
        pixels = np.load(sample['signed'], allow_pickle=False)
        np.save(sample['signed'], pixels[:-1])
        row['signed_crop']['sha256'] = hashlib.sha256(sample['signed'].read_bytes()).hexdigest()
    elif defect == 'origin':
        row['source_origin_px'][0] += 1
    elif defect == 'spacing':
        record['configuration']['unmixing']['calibration']['pixel_size_um'][0] *= 2
    elif defect == 'common-psf':
        record['configuration']['unmixing']['calibration']['common_psf'][0][0] += .1
    elif defect == 'source-identity':
        row['source_identity']['source_dataset'] += '-other'
    elif defect in ('duplicate-field', 'duplicate-artifact'):
        duplicate = json.loads(json.dumps(row))
        duplicate['field_id'] = '000003'
        if defect == 'duplicate-artifact':
            duplicate['source_identity']['field'] = 'different-field'
        record['fields'].append('000003')
        record['calibrated_fields']['000003'] = duplicate
    elif defect == 'high-label':
        canvas = np.load(sample['output'], allow_pickle=False)
        canvas[20, 20, -1] = 65536
        np.save(sample['output'], canvas)
        row['display_mask']['sha256'] = hashlib.sha256(sample['output'].read_bytes()).hexdigest()
    else:
        outside = sample['inputs']['stack'].parent / 'outside-signed.npy'
        outside.write_bytes(sample['signed'].read_bytes())
        sample['signed'].unlink(); sample['signed'].symlink_to(outside)
    _rewrite_calibrated_cell_record(sample['record'], record)
    before = {path: path.read_bytes() for path in sample['inputs']['output'].rglob('*') if path.is_file()}
    with pytest.raises(ValueError):
        with _open_calibrated_cell_dataset(sample['inputs']['output'], channels=(1,)):
            pytest.fail('inconsistent data reached the consumer')
    assert all(path.read_bytes() == content for path, content in before.items())


@needs_descriptor_relative_reads
def test_signed_cell_loader_closes_maps_and_refuses_postyield_mutation(completed_calibrated_cells):
    """Reject changed durable pixels after a real borrowed view has been consumed."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    with pytest.raises(ValueError):
        with _open_calibrated_cell_dataset(sample['inputs']['output']) as dataset:
            field = next(dataset.fields())
            mapped = field['signed']
            assert float(mapped[0, 0, 0]) == float(sample['result'].image[0, 0, 0])
            with sample['signed'].open('r+b') as stream:
                stream.seek(-4, 2)
                stream.write(np.float32(123).tobytes())
    assert mapped._mmap.closed


@needs_descriptor_relative_reads
def test_signed_cell_loader_cancellation_closes_borrowed_maps(completed_calibrated_cells):
    """Cancel a real open field without leaking its mappings or modifying source bytes."""
    import threading
    from spacr.cancellation import PipelineCancelled
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    event = threading.Event()
    before = sample['signed'].read_bytes()
    with pytest.raises(PipelineCancelled):
        with _open_calibrated_cell_dataset(sample['inputs']['output'], cancelled=event.is_set) as dataset:
            mapped = next(dataset.fields())['signed']
            event.set()
    assert mapped._mmap.closed
    assert sample['signed'].read_bytes() == before


def test_signed_cell_record_snapshot_caps_a_concurrent_append(completed_calibrated_cells):
    """Bound the actual JSON read even when the real metadata file grows during capture."""
    from spacr.psf_pipeline import _cell_record_snapshot

    path = completed_calibrated_cells['record']
    with path.open('rb') as stream:
        class GrowingStream:
            """Append to the real descriptor's file immediately before its bounded read."""

            def fileno(self):
                """Expose the actual metadata descriptor to identity validation."""
                return stream.fileno()

            def seek(self, offset):
                """Keep the genuine metadata stream positioned by the loader."""
                return stream.seek(offset)

            def read(self, size):
                """Grow real durable metadata without allowing an unbounded allocation."""
                assert 0 < size <= 16 * 1024 ** 2 + 1
                with path.open('ab') as writer:
                    writer.write(b' ' * (16 * 1024 ** 2 + 1))
                return stream.read(size)

        with pytest.raises(ValueError):
            _cell_record_snapshot(GrowingStream(), None)


@pytest.fixture
def plate_scoped_inputs(calibrated_batch_inputs):
    """The actual physical-ID fixture with an explicit reusable plate master map."""
    inputs = calibrated_batch_inputs
    master = {'plate1': {'3': ['A01'], '1': ['B01']},
              'plate2': {'3': ['B01'], '1': ['A01']}}
    for identity in inputs['identities'].values():
        identity['source_dataset'] = str(inputs['stack'].parent.resolve())
    inputs['settings'] = dict(inputs['settings'], unmix_per_plate=True,
                              unmix_controls=json.dumps(master))
    inputs['master'] = master
    return inputs


def _plate_scoped_plan(inputs):
    """Build the actual plan with captured physical positions and source identities."""
    return _prepare_unmixing(
        inputs['settings'], inputs['stack'], channels=(3, 1), allow_calibrated=True,
        channel_positions={3: 0, 1: 1}, field_identities=inputs['identities'])


def test_per_plate_controls_capture_exact_inputs_and_immutable_selection(plate_scoped_inputs):
    import hashlib

    inputs = plate_scoped_inputs
    before = {path: path.read_bytes() for path in inputs['stack'].glob('*.npy')}
    plan = _plate_scoped_plan(inputs)
    np.testing.assert_allclose(plan.matrix, inputs['matrix'], atol=2e-6, rtol=0)
    selection = plan.provenance()['control_selection']
    assert selection['scope'] == 'plate' and selection['plate'] == 'plate1'
    assert selection['physical_channel_order'] == [3, 1]
    assert selection['selected_controls'] == {'3': ['A01'], '1': ['B01']}
    assert selection['canonical_controls_by_plate'] == inputs['master']
    assert selection['controls_map_sha256'] == hashlib.sha256(json.dumps(
        inputs['master'], sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert set(selection['sampled_fields']) == {'stack_000000', 'stack_000001'}
    for stem, facts in selection['sampled_fields'].items():
        path = inputs['stack'] / f'{stem}.npy'
        raw = np.load(path)
        assert facts['source_identity'] == inputs['identities'][path.name]
        assert facts['input_stage'] == 'assembled_pre_unmixing'
        assert facts['input_shape'] == list(raw.shape)
        assert facts['input_dtype'] == raw.dtype.str
        assert facts['input_size_bytes'] == len(before[path])
        assert facts['input_sha256'] == hashlib.sha256(before[path]).hexdigest()
    captured = plan.provenance()
    inputs['settings']['unmix_controls'] = '{}'
    inputs['identities']['stack_000000.npy']['well'] = 'H12'
    selection['selected_controls']['3'].append('H12')
    assert plan.provenance() == captured
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('controls', [
    '{"3":["A01"],"1":["B01"]}',
    '{"another":{"3":["A01"],"1":["B01"]}}',
    '{"plate1":{"3":["A01"]}}',
    '{"plate1":{"3":["A01"],"3":["B01"],"1":["B01"]}}',
    '{"plate1":{"3":["A01"],"1":["a1"]}}',
    '{"plate1":{"3":["A99"],"1":["B01"]}}',
], ids=['flat', 'missing-plate', 'missing-channel', 'duplicate-key', 'shared-control', 'missing-well'])
def test_per_plate_controls_refuse_incomplete_or_ambiguous_maps(plate_scoped_inputs, controls):
    inputs = plate_scoped_inputs
    before = {path: path.read_bytes() for path in inputs['stack'].glob('*.npy')}
    inputs['settings']['unmix_controls'] = controls
    with pytest.raises(ValueError, match='plate-to-channel-to-well-list'):
        _plate_scoped_plan(inputs)
    assert not inputs['output'].exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_per_plate_controls_refuse_mixed_actual_identities(plate_scoped_inputs):
    inputs = plate_scoped_inputs
    inputs['identities']['stack_000002.npy']['plate'] = 'plate2'
    with pytest.raises(ValueError, match='exact source plate'):
        _plate_scoped_plan(inputs)
    assert not inputs['output'].exists()


def test_per_plate_false_keeps_global_numbers_and_provenance_exact(calibrated_batch_inputs):
    inputs = calibrated_batch_inputs
    options = dict(channels=(3, 1), allow_calibrated=True,
                   channel_positions={3: 0, 1: 1}, field_identities=inputs['identities'])
    original = _prepare_unmixing(inputs['settings'], inputs['stack'], **options)
    explicit_false = _prepare_unmixing(dict(inputs['settings'], unmix_per_plate=False),
                                      inputs['stack'], **options)
    assert explicit_false.provenance() == original.provenance()
    assert 'control_selection' not in explicit_false.provenance()
    np.testing.assert_array_equal(explicit_false.apply_calibrated(inputs['target']).image,
                                  original.apply_calibrated(inputs['target']).image)


@pytest.fixture
def plate_scoped_cells(plate_scoped_inputs):
    """Complete a real scoped session while preserving its independent raw controls."""
    inputs = plate_scoped_inputs
    session = _calibrated_batch_session(inputs)
    result = session.prepare_calibrated(inputs['target'], '000002')
    labels = np.zeros(result.image.shape[:2], dtype=np.uint16)
    labels[12:17, 14:20] = 11
    mask = session.restore_calibrated(labels, result, '000002')
    output = inputs['output'] / 'merged' / 'stack_000002.npy'
    signed = inputs['output'] / 'signed' / 'stack_000002.npy'
    output.parent.mkdir(); signed.parent.mkdir()
    np.save(output, np.concatenate((session.calibrated_display(inputs['target'], result),
                                   mask[..., None]), axis=-1).astype(np.float32))
    np.save(signed, result.image)
    session.mark_completed('000002', output_path=output, signed_path=signed, mask_role='cell')
    session.finish({'000002'})
    return dict(inputs=inputs, record=inputs['output'] / 'psf/segmentation_application.json',
                signed=signed, mask=mask)


@needs_descriptor_relative_reads
@pytest.mark.parametrize('defect', [None, 'wrong-plate', 'wrong-map-hash', 'swapped-dye-fields', 'wrong-source'],
                         ids=['valid', 'wrong-plate', 'wrong-map-hash', 'swapped-dye-fields', 'wrong-source'])
def test_per_plate_signed_reader_validates_selection_with_rebound_hashes(plate_scoped_cells, defect):
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = plate_scoped_cells
    record = json.loads(sample['record'].read_text())
    unmixing = record['configuration']['unmixing']
    selection = unmixing['control_selection']
    if defect == 'wrong-plate':
        selection['plate'] = 'plate2'
    elif defect == 'wrong-map-hash':
        selection['controls_map_sha256'] = '0' * 64
    elif defect == 'swapped-dye-fields':
        unmixing['fields']['3'], unmixing['fields']['1'] = unmixing['fields']['1'], unmixing['fields']['3']
    if defect == 'wrong-source':
        stem = next(iter(selection['sampled_fields']))
        selection['sampled_fields'][stem]['source_identity']['source_dataset'] = '/other/source'
        record['configuration']['control_identities'][stem]['source_dataset'] = '/other/source'
    if defect is not None:
        _rewrite_calibrated_cell_record(sample['record'], record)
        with pytest.raises(ValueError):
            with _open_calibrated_cell_dataset(sample['inputs']['output']):
                pytest.fail('inconsistent plate selection was accepted')
    else:
        with _open_calibrated_cell_dataset(sample['inputs']['output']) as dataset:
            fields = dataset.fields()
            field = next(fields)
            assert field['source_identity']['plate'] == 'plate1'
            np.testing.assert_array_equal(field['signed'], np.load(sample['signed']))
            np.testing.assert_array_equal(field['labels'], sample['mask'][2:46, 1:53])
            assert dataset.provenance()['configuration']['unmixing']['control_selection'] == selection


def test_per_plate_controls_refuse_source_changes_during_matching(plate_scoped_inputs, monkeypatch):
    from spacr.psf_pipeline import _UnmixCalibration

    inputs = plate_scoped_inputs
    original_apply = _UnmixCalibration.apply
    path = inputs['stack'] / 'stack_000000.npy'
    other = {file: file.read_bytes() for file in inputs['stack'].glob('*.npy') if file != path}
    changed = []

    def apply_then_change(calibration, pixels):
        result = original_apply(calibration, pixels)
        if not changed:
            raw = np.load(path, mmap_mode='r+')
            raw[0, 0, 0] += 1
            raw.flush()
            raw._mmap.close()
            changed.append(True)
        return result

    monkeypatch.setattr(_UnmixCalibration, 'apply', apply_then_change)
    with pytest.raises(ValueError, match='stable bounded numeric'):
        _plate_scoped_plan(inputs)
    assert changed == [True] and not inputs['output'].exists()
    assert all(file.read_bytes() == content for file, content in other.items())


def test_per_plate_control_hash_stops_at_original_size_plus_one(plate_scoped_inputs):
    from spacr.psf_pipeline import _plate_control_hash

    path = plate_scoped_inputs['stack'] / 'stack_000000.npy'
    original_size = path.stat().st_size
    consumed = []
    appended = []
    with path.open('rb') as source:
        class GrowingReader:
            def fileno(self):
                return source.fileno()

            def seek(self, position):
                return source.seek(position)

            def read(self, count):
                block = source.read(count)
                consumed.append(len(block))
                if not appended:
                    with path.open('ab') as writer:
                        writer.write(b'x' * (1024 ** 2))
                    appended.append(True)
                return block

        with pytest.raises(ValueError):
            _plate_control_hash(GrowingReader(), original_size)
    assert appended == [True]
    assert sum(consumed) == original_size + 1
    assert path.stat().st_size == original_size + 1024 ** 2
    assert not plate_scoped_inputs['output'].exists()


def _signed_source_descriptors(root):
    """Snapshot this dataset's actual open descriptors where procfs exposes them."""
    if not os.path.isdir('/proc/self/fd'):
        return None
    prefix = os.fspath(root.resolve())
    descriptors = set()
    for name in os.listdir('/proc/self/fd'):
        try:
            target = os.readlink('/proc/self/fd/' + name)
        except FileNotFoundError:
            continue
        if target == prefix or target.startswith(prefix + '/'):
            descriptors.add((name, target))
    return descriptors


@pytest.fixture
def signed_reader_resources(monkeypatch):
    """Observe real reader streams and maps without replacing their file operations."""
    import spacr.psf_pipeline as pipeline

    streams, maps = [], []
    original_open, original_map = pipeline._cell_record_open, pipeline._cell_record_map

    def opened(*args, **kwargs):
        stream = original_open(*args, **kwargs)
        streams.append(stream)
        return stream

    def mapped(*args, **kwargs):
        array = original_map(*args, **kwargs)
        maps.append(array)
        return array

    monkeypatch.setattr(pipeline, '_cell_record_open', opened)
    monkeypatch.setattr(pipeline, '_cell_record_map', mapped)
    yield
    assert streams and all(stream.closed for stream in streams)
    assert all(array._mmap.closed for array in maps)


@needs_descriptor_relative_reads
@pytest.mark.parametrize('artifact', ['signed', 'output', 'record'])
def test_signed_reader_rejects_identical_byte_replacement_after_borrow(
        completed_calibrated_cells, signed_reader_resources, artifact):
    """Content equality cannot conceal replacement of a borrowed input's inode."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    root = sample['inputs']['output']
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    descriptors = _signed_source_descriptors(root)
    path = sample[artifact]
    with pytest.raises(ValueError):
        with _open_calibrated_cell_dataset(root) as dataset:
            signed = next(dataset.fields())['signed']
            np.testing.assert_array_equal(signed, sample['result'].image)
            original_inode = path.stat().st_ino
            replacement = path.with_name(path.name + '.replacement')
            replacement.write_bytes(before[path])
            os.replace(replacement, path)
            assert path.stat().st_ino != original_inode
            assert path.read_bytes() == before[path]
    assert signed._mmap.closed
    assert _signed_source_descriptors(root) == descriptors
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


@needs_descriptor_relative_reads
@pytest.mark.parametrize('caller_error', [False, True])
def test_signed_reader_early_consumer_exit_preserves_errors_and_closes_inputs(
        completed_calibrated_cells, signed_reader_resources, caller_error):
    """Early consumers release borrowed resources and retain their own failure identity."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    root = sample['inputs']['output']
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    descriptors = _signed_source_descriptors(root)
    failure = OSError('the consumer destination is full')
    try:
        with _open_calibrated_cell_dataset(root) as dataset:
            signed = next(dataset.fields())['signed']
            assert not signed.flags.writeable
            np.testing.assert_array_equal(signed, sample['result'].image)
            if caller_error:
                raise failure
    except OSError as error:
        assert caller_error and error is failure
    else:
        assert not caller_error
    assert signed._mmap.closed
    dataset.close()
    with pytest.raises(ValueError):
        dataset.fields()
    assert _signed_source_descriptors(root) == descriptors
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


@needs_descriptor_relative_reads
@pytest.mark.parametrize('defect', ['fortran', 'trailing-bytes', 'nonfinite'])
def test_signed_reader_rejects_rebound_unsafe_pixel_artifacts(
        completed_calibrated_cells, signed_reader_resources, defect):
    """Validate actual NPY layout and finite signed values even with current hashes."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    pixels = np.load(sample['signed'], allow_pickle=False)
    if defect == 'fortran':
        np.save(sample['signed'], np.asfortranarray(pixels))
    elif defect == 'trailing-bytes':
        with sample['signed'].open('ab') as stream:
            stream.write(b'unexpected trailing payload')
    else:
        pixels[0, 0, 0] = np.inf
        np.save(sample['signed'], pixels)
    record = json.loads(sample['record'].read_text())
    row = record['calibrated_fields']['000002']
    row['signed_crop']['sha256'] = hashlib.sha256(sample['signed'].read_bytes()).hexdigest()
    _rewrite_calibrated_cell_record(sample['record'], record)
    root = sample['inputs']['output']
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    descriptors = _signed_source_descriptors(root)
    with pytest.raises(ValueError):
        with _open_calibrated_cell_dataset(root):
            pytest.fail('unsafe pixels were exposed to a consumer')
    assert _signed_source_descriptors(root) == descriptors
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


@needs_descriptor_relative_reads
@pytest.mark.parametrize('defect', ['psf-hash', 'reference-shift', 'negative-fraction'])
def test_signed_reader_rejects_rebound_calibration_and_residual_misstatements(
        completed_calibrated_cells, signed_reader_resources, defect):
    """Hash-consistent records must still describe the measured calibration and pixels."""
    from spacr.psf_pipeline import _open_calibrated_cell_dataset

    sample = completed_calibrated_cells
    record = json.loads(sample['record'].read_text())
    calibration = record['configuration']['unmixing']['calibration']
    if defect == 'psf-hash':
        calibration['channels'][0]['normalized_psf_sha256'] = '0' * 64
    elif defect == 'reference-shift':
        reference = next(entry for entry in calibration['channels']
                         if entry['channel'] == calibration['reference_channel'])
        reference['shift_to_reference_px'] = [1, 0]
    else:
        fractions = record['calibrated_fields']['000002']['negative_fraction_per_channel']
        channel = str(record['configuration']['physical_channel_order'][0])
        fractions[channel] = 1.0 if fractions[channel] != 1.0 else 0.0
    _rewrite_calibrated_cell_record(sample['record'], record)
    root = sample['inputs']['output']
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    descriptors = _signed_source_descriptors(root)
    with pytest.raises(ValueError):
        with _open_calibrated_cell_dataset(root):
            pytest.fail('incorrect calibration or residual metadata reached a consumer')
    assert _signed_source_descriptors(root) == descriptors
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before
