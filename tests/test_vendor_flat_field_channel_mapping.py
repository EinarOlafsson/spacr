"""Correct reordered/subset acquisitions using the explicitly assigned profile."""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr import illumination as ill
from spacr.measure_hooks import PreprocessingContext


def profiles(tmp_path, ids=(1, 2)):
    """Write actual Harmony XML spelling, with independent known gain/offsets."""
    def polynomial(coefficients):
        """Represent a polynomial in the vendor's documented coordinates."""
        return {"Character": "NonFlat", "Profile": {
            "Type": "Polynomial", "Dims": [7, 5], "Origin": [0, 0],
            "Scale": [1, 1], "Coefficients": coefficients}}

    entries = []
    for channel, gain, slope, dark, dark_slope in zip(ids, (1, 2), (.02, -.03), (10, 40), (1, 2)):
        blob = {"Channel": channel,
                "Foreground": polynomial([[gain], [slope, 0]]),
                "Background": polynomial([[dark], [0, dark_slope]])}
        entries.append('<FlatfieldProfile>' + json.dumps(blob) + '</FlatfieldProfile>')
    path = tmp_path / 'Index.idx.xml'
    path.write_text('<Root>' + ''.join(entries) + '</Root>')
    y, x = np.mgrid[:5, :7]
    return path, [1 + .02 * x, 2 - .03 * x], [10 + y, 40 + 2 * y]


def correct(model, raw, channels):
    """Apply through the real Measure hook without modifying input pixels."""
    context = PreprocessingContext(file_name='plate1_A01_1_1.npy', channels=channels, settings={})
    return ill.IlluminationCorrector(model, verbose=False)(raw, context)


def test_reordered_channels_keep_their_own_gain_and_spatial_background(tmp_path):
    path, gains, backgrounds = profiles(tmp_path)
    model = ill._vendor_illumination(path, [0, 1], channel_map='0:2,1:1', dark=3, verbose=False)
    truth = np.stack([np.full((5, 7), 700), np.full((5, 7), 250)], axis=-1)
    raw = np.stack([truth[..., i] * gains[j] + backgrounds[j] + 3
                    for i, j in enumerate((1, 0))], axis=-1).astype(np.float32)
    before = raw.copy()
    np.testing.assert_allclose(correct(model, raw, [0, 1]), truth, rtol=1e-6)
    np.testing.assert_array_equal(raw, before)
    assert model.meta['vendor_channel_map'] == {'0': 2, '1': 1}
    assert model.meta['vendor_channel_map_explicit'] is True


def test_noncontiguous_profile_ids_and_subset_channel_order(tmp_path):
    path, gains, backgrounds = profiles(tmp_path, ids=(3, 7))
    model = ill._vendor_illumination(path, [4, 1], channel_map='0:3,1:7,4:3', verbose=False)
    raw = np.stack([500 * gains[j] + backgrounds[j] for j in (0, 1)], axis=-1).astype(np.float32)
    np.testing.assert_allclose(correct(model, raw, [4, 1]), 500, rtol=1e-6)
    assert model.fields[ill.ALL_PLATES].channels == (4, 1)
    assert model.meta['vendor_channel_map'] == {'4': 3, '1': 7}


@pytest.mark.parametrize('mapping', ['0:1', '0:1,0:2,1:2', '0:0,1:2', '-1:2,1:1',
                                     '0=2,1=1', '0:1.5,1:2', '0:1,1:2,', 'abc'])
def test_invalid_or_incomplete_mapping_is_refused(tmp_path, mapping):
    path, _, _ = profiles(tmp_path)
    with pytest.raises(ill.IlluminationError, match='illumination_vendor_channel_map'):
        ill._vendor_illumination(path, [0, 1], channel_map=mapping, verbose=False)


def test_unknown_vendor_channel_is_refused(tmp_path):
    path, _, _ = profiles(tmp_path)
    with pytest.raises(ill.IlluminationError, match='Harmony channel'):
        ill._vendor_illumination(path, [0], channel_map='0:8', verbose=False)


def test_reference_image_planes_are_one_based_and_reordered(tmp_path):
    y, x = np.mgrid[:5, :7]
    planes = np.stack([100 + 3 * x + y, 200 + x + 7 * y]).astype(np.float32)
    path = tmp_path / 'shading.npy'
    np.save(path, planes + 5)
    model = ill._vendor_illumination(path, [0, 2], channel_map='0:2,2:1', dark=5, verbose=False)
    truth = np.full((5, 7, 2), 400, dtype=np.float32)
    raw = np.stack([400 * planes[j] / planes[j].mean() + 5 for j in (1, 0)], axis=-1)
    np.testing.assert_allclose(correct(model, raw, [0, 2]), truth, rtol=1e-6)


def test_single_reference_broadcast_stays_default_but_explicit_invalid_plane_fails(tmp_path):
    path = tmp_path / 'single.npy'
    np.save(path, np.ones((5, 7)))
    legacy = ill._vendor_illumination(path, [2, 5], verbose=False)
    assert legacy.meta['vendor_channel_map'] == {'2': 1, '5': 1}
    assert legacy.meta['vendor_channel_map_explicit'] is False
    explicit = ill._vendor_illumination(path, [2, 5], channel_map='2:1,5:1', verbose=False)
    np.testing.assert_array_equal(legacy.fields[ill.ALL_PLATES].flatfield,
                                  explicit.fields[ill.ALL_PLATES].flatfield)
    with pytest.raises(ill.IlluminationError, match='vendor plane'):
        ill._vendor_illumination(path, [2], channel_map='2:2', verbose=False)


def test_settings_propagate_mapping_and_saved_model_keeps_it(tmp_path):
    path, gains, backgrounds = profiles(tmp_path)
    source = tmp_path / 'plate' / 'merged'
    source.mkdir(parents=True)
    settings = {'src': str(source), 'channels': [0, 1], 'illumination_correction': True,
                'illumination_vendor_profile': str(path), 'illumination_vendor_channel_map': '0:2,1:1',
                'illumination_qc': False, 'verbose': False}
    prepared = ill.prepare_illumination_model(settings)
    loaded = ill.load_illumination_model(prepared.model_path)
    assert loaded.meta['vendor_channel_map'] == {'0': 2, '1': 1}
    raw = np.stack([300 * gains[j] + backgrounds[j] for j in (1, 0)], axis=-1).astype(np.float32)
    np.testing.assert_allclose(correct(loaded, raw, [0, 1]), 300, rtol=1e-6)
    # A saved model's exact calibration retains priority over vendor settings.
    settings.update(illumination_model=prepared.model_path, illumination_vendor_channel_map='invalid')
    reused = ill.prepare_illumination_model(settings)
    assert reused.model_sha256 == prepared.model_sha256


def test_blank_mapping_retains_legacy_profile_order_and_default(tmp_path):
    path, _, _ = profiles(tmp_path)
    assert ill.illumination_settings()['illumination_vendor_channel_map'] == ''
    implicit = ill._vendor_illumination(path, [1, 0], verbose=False)
    explicit = ill._vendor_illumination(path, [1, 0], channel_map='0:1,1:2', verbose=False)
    np.testing.assert_array_equal(implicit.fields[ill.ALL_PLATES].flatfield,
                                  explicit.fields[ill.ALL_PLATES].flatfield)
    np.testing.assert_array_equal(implicit.fields[ill.ALL_PLATES].darkfield,
                                  explicit.fields[ill.ALL_PLATES].darkfield)
