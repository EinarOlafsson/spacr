"""Spatial image columns cannot masquerade as spectral channel indices."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spacr import psf_pipeline as psf

SOURCE = 'Spectral unmixing requires images with an explicit channel axis and at least three dimensions.'


@pytest.fixture(autouse=True)
def english_errors(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


def settings(controls='0:A01; 1:B01'):
    return dict(unmix=True, unmix_controls=controls, unmix_background_percentile=5.0)


def physical_fields():
    plane = np.arange(64, dtype=np.float64).reshape(8, 8)
    return [np.stack([20 + 3 * plane, 20 + .6 * plane], axis=-1),
            np.stack([20 + .4 * plane, 20 + 4 * plane], axis=-1)]


def write_fields(folder, fields):
    folder.mkdir()
    for index, value in enumerate(fields):
        np.save(folder / f'plate1_{"AB"[index]}01_1.npy', value)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.glob('*.npy')}


def assert_files_exact(folder, before):
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.glob('*.npy')}


def low_dimension(ndim):
    if ndim == 0:
        return np.array(20.)
    if ndim == 1:
        return np.arange(8, dtype=float) + 20
    field = np.full((8, 8), 20.)
    field[:, 0] = 20 + 3 * np.arange(8)
    field[:, 1] = 20 + .6 * np.arange(8)
    return field


@pytest.mark.parametrize('ndim', [0, 1, 2])
@pytest.mark.parametrize('channels', [None, (0, 1)])
def test_control_without_spatial_and_channel_axes_refuses(tmp_path, ndim, channels):
    field = low_dimension(ndim)
    before = field.copy()
    folder = tmp_path / 'controls'
    hashes = write_fields(folder, [field])
    with pytest.raises(ValueError) as error:
        psf._prepare_unmixing(settings('0:A01'), folder, channels=channels)
    assert str(error.value) == SOURCE
    np.testing.assert_array_equal(field, before)
    assert_files_exact(folder, hashes)


@pytest.fixture
def actual_plan(tmp_path):
    folder = tmp_path / 'valid-controls'
    write_fields(folder, physical_fields())
    return psf._prepare_unmixing(settings(), folder)


@pytest.mark.parametrize('ndim', [0, 1, 2])
def test_plan_refuses_missing_axes_before_copying_or_indexing(actual_plan, ndim):
    image = low_dimension(ndim)
    before = image.copy()
    with pytest.raises(ValueError) as error:
        actual_plan.apply(image)
    assert str(error.value) == SOURCE
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('leading', [(), (2,), (2, 2)])
@pytest.mark.parametrize('channels', [None, (1, 0)])
def test_channel_last_leading_axes_and_selected_order_remain_valid(tmp_path, leading, channels):
    fields = [np.broadcast_to(field, leading + field.shape).copy() for field in physical_fields()]
    before = [field.copy() for field in fields]
    folder = tmp_path / 'controls'
    hashes = write_fields(folder, fields)
    plan = psf._prepare_unmixing(settings(), folder, channels=channels)
    assert plan.channels == ((0, 1) if channels is None else channels)
    corrected = plan.apply(fields[0])
    assert corrected.shape == fields[0].shape and corrected.dtype == np.float32
    assert np.isfinite(corrected).all()
    for actual, expected in zip(fields, before):
        np.testing.assert_array_equal(actual, expected)
    assert_files_exact(folder, hashes)


def test_one_explicit_channel_axis_is_still_default_identity(tmp_path):
    field = physical_fields()[0][..., :1]
    before = field.copy()
    folder = tmp_path / 'controls'
    hashes = write_fields(folder, [field])
    plan = psf._prepare_unmixing(settings('0:A01'), folder)
    assert plan.channels == (0,)
    np.testing.assert_array_equal(plan.apply(field), field.astype(np.float32))
    np.testing.assert_array_equal(field, before)
    assert_files_exact(folder, hashes)


def test_disabled_unmixing_does_not_load_or_validate_controls(tmp_path):
    folder = tmp_path / 'controls'
    hashes = write_fields(folder, [low_dimension(2)])
    config = settings('0:A01')
    config['unmix'] = False
    calls = []
    assert psf._prepare_unmixing(config, folder, load=lambda path: calls.append(path)) is None
    assert not calls
    assert_files_exact(folder, hashes)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
@pytest.mark.parametrize('entrypoint', ['controls', 'apply'])
def test_installed_reviewed_language_guards_both_boundaries(tmp_path, actual_plan, language, entrypoint, monkeypatch):
    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-unmix-shape-boundary.json'
    record = json.loads(review.read_text(encoding="utf-8"))['records'][0]
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    image = low_dimension(2)
    before = image.copy()
    folder = tmp_path / 'invalid-controls'
    hashes = write_fields(folder, [image])
    with pytest.raises(ValueError) as error:
        if entrypoint == 'controls':
            psf._prepare_unmixing(settings('0:A01'), folder)
        else:
            actual_plan.apply(image)
    assert str(error.value) == record['translation']
    assert str(error.value) != SOURCE
    np.testing.assert_array_equal(image, before)
    assert_files_exact(folder, hashes)
