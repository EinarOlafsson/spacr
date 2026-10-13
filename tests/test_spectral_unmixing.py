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
                         .read_text())
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
                       .read_text())
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
