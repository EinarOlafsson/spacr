"""Spectral unmixing: bleed-through estimated from single-stain controls.

Synthetic two-dye plates with a known bleed-through matrix: each dye is
read in the other channel at a fixed fraction, over a flat background with
noise. The estimate recovers the matrix, the controls' off-target channel
falls to background after unmixing, and a double-stained field's
intensities move to the amounts put in.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr.psf_pipeline import (
    _UNMIX_RECORD_KEY, _apply_recorded_unmixing, _configuration,
    _parse_unmix_controls, _prepare_measure_unmixing,
    _prepare_segmentation_psf, _prepare_unmixing, processing_requested)

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
