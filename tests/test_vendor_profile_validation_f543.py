"""Reject ambiguous or unusable vendor calibration before touching run outputs."""
import copy
import json

import numpy as np
import pytest

from spacr import illumination as I
from spacr.measure_hooks import PreprocessingContext


def surface(coefficients, shape=(4, 3)):
    return {'Profile': {'Type': 'Polynomial', 'Dims': list(shape),
                        'Origin': [0, 0], 'Scale': [1, 1],
                        'Coefficients': coefficients}}


def profile():
    return {'Channel': 1, 'Foreground': surface([[2.]]),
            'Background': surface([[10.]])}


def write(path, profiles):
    path.write_text('<Root>' + ''.join(
        '<FlatfieldProfile>' + json.dumps(row) + '</FlatfieldProfile>'
        for row in profiles) + '</Root>')
    return path


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1e100])
def test_spatial_background_must_remain_finite_in_model_dtype(tmp_path, value):
    record = profile()
    record['Background'] = surface([[value]])
    path = write(tmp_path / 'index.xml', [record])
    before = path.read_bytes()
    with pytest.raises(I.IlluminationError, match='background.*finite'):
        I._vendor_illumination(path, [0], verbose=False)
    assert path.read_bytes() == before


def test_gain_with_finite_minimum_but_overflowing_pixels_is_rejected(tmp_path):
    record = profile()
    # First column gain=1; other columns exceed float32. min() alone misses it.
    record['Foreground'] = surface([[1.], [1e39, 0.]])
    path = write(tmp_path / 'index.xml', [record])
    with pytest.raises(I.IlluminationError, match='flat field.*finite'):
        I._vendor_illumination(path, [0], verbose=False)


@pytest.mark.parametrize('shape', [(4, 1), (1, 3), (2, 2)])
def test_background_cannot_broadcast_or_resize_onto_different_sensor_grid(tmp_path, shape):
    record = profile()
    record['Background'] = surface([[10.]], shape=shape)
    path = write(tmp_path / 'index.xml', [record])
    with pytest.raises(I.IlluminationError, match='calibration grids must match'):
        I._vendor_illumination(path, [0], verbose=False)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1e100, None, 'invalid'])
def test_scalar_offset_must_be_finite_and_representable(tmp_path, value):
    path = write(tmp_path / 'index.xml', [profile()])
    with pytest.raises(I.IlluminationError, match='camera offset'):
        I._vendor_illumination(path, [0], dark=value, verbose=False)


@pytest.mark.parametrize('changed', ['Foreground', 'Background'])
def test_duplicate_channel_with_conflicting_calibration_is_not_last_wins(tmp_path, changed):
    first = profile()
    second = copy.deepcopy(first)
    second[changed] = surface([[20.]])
    for records in ([first, second], [second, first]):
        path = write(tmp_path / 'index.xml', records)
        with pytest.raises(I.IlluminationError, match='conflicting.*channel 1'):
            I._vendor_illumination(path, [0], verbose=False)


def test_identical_repeated_channel_profiles_keep_exact_correction_and_sources(tmp_path):
    first = profile()
    second = copy.deepcopy(first)
    second['ChannelName'] = 'Repeated image metadata'
    path = write(tmp_path / 'index.xml', [first, second])
    model = I._vendor_illumination(path, [0], dark=3., verbose=False)
    raw = np.full((3, 4, 1), 113., dtype=np.float32)
    before = raw.copy()
    context = PreprocessingContext(file_name='plate1_A01_1', channels=[0], settings={})
    result = I.IlluminationCorrector(model, verbose=False)(raw, context)
    np.testing.assert_array_equal(result, np.full_like(raw, 50.))
    np.testing.assert_array_equal(raw, before)
    saved = tmp_path / 'model.npz'
    model.save(saved)
    loaded = I.IlluminationModel.load(saved)
    np.testing.assert_array_equal(
        I.IlluminationCorrector(loaded, verbose=False)(raw, context), result)


def test_invalid_profile_preflight_preserves_existing_outputs(tmp_path):
    merged = tmp_path / 'plate1' / 'merged'
    merged.mkdir(parents=True)
    raw_path = merged / 'plate1_A01_1.npy'
    np.save(raw_path, np.full((3, 4, 1), 100, dtype=np.uint16))
    output = merged.parent / 'illumination'
    output.mkdir()
    existing = output / 'illumination_model.npz'
    existing.write_bytes(b'existing output must survive')
    record = profile()
    record['Background'] = surface([[np.nan]])
    path = write(tmp_path / 'index.xml', [record])
    originals = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    settings = {'src': str(merged), 'channels': [0], 'illumination_correction': True,
                'illumination_vendor_profile': str(path), 'illumination_qc': False,
                'verbose': False}
    with pytest.raises(I.IlluminationError, match='background.*finite'):
        I.prepare_illumination_model(settings)
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == originals
