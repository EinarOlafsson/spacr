"""Physical unmixing channels cannot truncate or overwrite one another."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from spacr import psf_pipeline as psf

SOURCE = 'Spectral unmixing requires distinct nonnegative integer channel indices.'


@pytest.fixture(autouse=True)
def english_errors(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


@pytest.fixture
def plate(tmp_path):
    stack = tmp_path / 'stack'
    stack.mkdir()
    plane = np.arange(64, dtype=np.float64).reshape(8, 8)
    for name, data in [('plate1_A01_1.npy', np.stack([20 + 3 * plane, 20 + .25 * plane], axis=-1)),
                       ('plate1_B01_1.npy', np.stack([20 + .1 * plane, 20 + 4 * plane], axis=-1))]:
        np.save(stack / name, data)
    return stack


def settings(stack, channels):
    return dict(src=str(stack), unmix=True, unmix_controls='0:A01; 1:B01',
                unmix_background_percentile=5.0, channels=channels)


def source_hashes(stack):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in stack.glob('*.npy')}


@pytest.mark.parametrize('selection', [[.5, 1], [True, 1], [np.bool_(False), 1],
                                      [-1, 1], [0, 0, 1], [], [np.nan, 1],
                                      [np.inf, 1], [complex(0), 1], [np.array([0]), 1],
                                      ['0.5', 1], '01', 0])
@pytest.mark.parametrize('entrypoint', ['measure', 'direct'])
def test_invalid_physical_selection_refuses_before_control_load_or_output(plate, selection, entrypoint, monkeypatch):
    config = settings(plate, selection)
    before = copy.deepcopy(config)
    hashes = source_hashes(plate)
    calls = []
    real_load = np.load

    def load(path):
        calls.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(psf.np, 'load', load)
    with pytest.raises(ValueError) as error:
        if entrypoint == 'measure':
            psf._prepare_measure_unmixing(config)
        else:
            psf._prepare_unmixing(config, plate, channels=selection, load=load)
    assert str(error.value) == SOURCE
    assert not calls
    assert not (plate.parent / 'measurements').exists()
    assert source_hashes(plate) == hashes
    np.testing.assert_equal(config, before)


@pytest.mark.parametrize('selection', [(0, 1), (1, 0), [0.0, 1.0], ['0', '1'],
                                      [np.int64(0), np.int64(1)], [np.array(0), np.array(1)]])
def test_valid_physical_order_and_saved_integral_selectors_are_preserved(plate, selection):
    config = settings(plate, selection)
    before = copy.deepcopy(selection)
    hashes = source_hashes(plate)
    plan = psf._prepare_measure_unmixing(config)
    assert plan.channels == tuple(int(value) for value in selection)
    assert plan.provenance()['channels'] == list(plan.channels)
    assert source_hashes(plate) == hashes
    np.testing.assert_equal(config['channels'], before)


def test_none_still_auto_selects_every_physical_channel(plate):
    plan = psf._prepare_unmixing(settings(plate, None), plate, channels=None)
    assert plan.channels == (0, 1)


@pytest.mark.parametrize('entrypoint', ['measure', 'direct'])
def test_disabled_unmixing_keeps_early_return_even_with_invalid_selection(plate, entrypoint):
    config = settings(plate, [False, .5, 0])
    config['unmix'] = False
    before = copy.deepcopy(config)
    if entrypoint == 'measure':
        assert psf._prepare_measure_unmixing(config) is None
    else:
        assert psf._prepare_unmixing(config, plate, channels=config['channels']) is None
    np.testing.assert_equal(config, before)
    assert not (plate.parent / 'measurements').exists()


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_unmix_refusal_uses_installed_reviewed_language_before_loading(plate, language, monkeypatch):
    import json

    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-unmix-channel-boundary.json'
    record = json.loads(review.read_text())['records'][0]
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    calls = []
    real_load = np.load

    def load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(psf.np, 'load', load)
    config = settings(plate, [.5, 1])
    before = copy.deepcopy(config)
    hashes = source_hashes(plate)
    with pytest.raises(ValueError) as error:
        psf._prepare_measure_unmixing(config)
    assert str(error.value) == record['translation']
    assert str(error.value) != record['source']
    assert not calls
    assert not (plate.parent / 'measurements').exists()
    assert source_hashes(plate) == hashes
    np.testing.assert_equal(config, before)
