import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spacr import measure


SOURCE = 'Wound calibration {name} must be a finite number greater than zero, or blank.'
BASE = dict(channels=[0], cell_mask_dim=1, wound_source='masks',
            wound_window=7, wound_threshold=None, plot=False)


def _field(width=40):
    mask = np.ones((128, 128), dtype=np.uint16)
    mask[:, 64 - width // 2:64 + width // 2] = 0
    return np.stack([np.zeros_like(mask), mask], axis=-1)


@pytest.mark.parametrize('name', ['voxel_size_xy_um', 'wound_hours_per_frame'])
@pytest.mark.parametrize('value', [0., -.5, np.nan, np.inf, -np.inf, 'bad'])
def test_invalid_calibration_refuses_in_existing_settings_preflight(name, value):
    settings = dict(BASE, **{name: value})
    before = copy.deepcopy(settings)
    with pytest.raises(ValueError) as error:
        measure._wound_settings_check(settings)
    assert str(error.value) == SOURCE.format(name=name)
    np.testing.assert_equal(settings, before)


@pytest.mark.parametrize('value', [0., -.5, np.nan, np.inf])
def test_direct_series_refuses_before_consuming_any_image(value):
    read = []

    def planes():
        read.append(True)
        raise AssertionError('Invalid physical scale consumed an image')
        yield

    with pytest.raises(ValueError) as error:
        measure._wound_series(planes(), [0.], source='masks', pixel_size_um=value)
    assert str(error.value) == SOURCE.format(name='pixel_size_um')
    assert not read


@pytest.mark.parametrize('value', [0., -.5, np.nan])
def test_actual_preview_refuses_invalid_physical_size_without_changing_pixels(value):
    field = _field()
    original = field.tobytes()
    with pytest.raises(ValueError) as error:
        measure._measure_field_wound(field, dict(BASE, voxel_size_xy_um=value))
    assert str(error.value) == SOURCE.format(name='voxel_size_xy_um')
    assert field.tobytes() == original


@pytest.mark.parametrize('value', [0., -2., np.nan, np.inf])
def test_actual_table_refuses_before_listing_any_input_directory(monkeypatch, value):
    listed = []

    def list_directory(path):
        listed.append(path)
        raise AssertionError('Invalid time scale reached input directory I/O')

    monkeypatch.setattr(measure, 'os', SimpleNamespace(listdir=list_directory, path=os.path))
    with pytest.raises(ValueError) as error:
        measure._wound_closure_tables('/must-not-read', dict(BASE, wound_hours_per_frame=value))
    assert str(error.value) == SOURCE.format(name='wound_hours_per_frame')
    assert not listed


def test_valid_spatial_scale_preserves_pixel_math_and_exact_physical_conversion():
    planes = [_field(width)[..., 1] for width in [40, 20]]
    original = [plane.tobytes() for plane in planes]
    plain, status, _ = measure._wound_series(planes, [0., 1.], source='masks', pixel_size_um=None)
    scaled, scaled_status, _ = measure._wound_series(planes, [0., 1.], source='masks', pixel_size_um='0.5')
    assert status == scaled_status == 'ok'
    pixel_columns = [column for column in plain if '_um' not in column]
    pd.testing.assert_frame_equal(plain[pixel_columns], scaled[pixel_columns])
    np.testing.assert_array_equal(scaled.mean_width_um, scaled.mean_width_px * .5)
    np.testing.assert_array_equal(scaled.open_area_um2, scaled.open_area_px * .25)
    assert plain.mean_width_um.isna().all()
    assert [plane.tobytes() for plane in planes] == original


@pytest.mark.parametrize('hours,expected,unit', [
    (None, [0., 1.], 'frame'), ('', [0., 1.], 'frame'),
    (2., [0., 2.], 'h'), ('2.0', [0., 2.], 'h'),
])
def test_valid_and_unspecified_time_calibration_preserve_native_table_inputs(
        tmp_path, hours, expected, unit):
    for time, width in enumerate([40, 20]):
        np.save(tmp_path / f'plate1_A01_1_{time}.npy', _field(width))
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tmp_path.glob('*.npy')}
    result = measure._wound_closure_tables(
        str(tmp_path), dict(BASE, wound_hours_per_frame=hours, voxel_size_xy_um=.5))
    assert result.time.tolist() == expected
    assert result.time_unit.tolist() == [unit, unit]
    assert result.status.tolist() == ['ok', 'ok']
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.glob('*.npy')} == before


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'ko', 'hi', 'is', 'fr'])
@pytest.mark.parametrize('route', ['preview', 'table'])
def test_installed_reviewed_language_at_actual_wound_boundary(monkeypatch, language, route):
    path = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-wound-calibration.json'
    target = json.loads(path.read_text())['records'][0]['translation']
    assert target != SOURCE
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    name = 'voxel_size_xy_um' if route == 'preview' else 'wound_hours_per_frame'
    settings = dict(BASE, **{name: 0.})
    if route == 'preview':
        field = _field()
        original = field.tobytes()
        with pytest.raises(ValueError) as error:
            measure._measure_field_wound(field, settings)
        assert field.tobytes() == original
    else:
        listed = []

        def list_directory(path):
            listed.append(path)
            raise AssertionError('Invalid calibration reached directory I/O')

        monkeypatch.setattr(measure, 'os', SimpleNamespace(listdir=list_directory, path=os.path))
        with pytest.raises(ValueError) as error:
            measure._wound_closure_tables('/must-not-read', settings)
        assert not listed
    assert str(error.value) == target.format(name=name)
