"""Item 725 round 3: Measure's measurements at image level, on images with known answers."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, skew

from spacr.measure import (
    _IMAGE_ALL_FEATURE_GROUPS,
    _IMAGE_COSTLY_FEATURE_GROUPS,
    _IMAGE_FEATURE_GROUPS,
    _IMAGE_FEATURE_SETTING,
    _IMAGE_OBJECT_FEATURE_GROUPS,
    _IMAGE_SETTINGS_FEATURE_GROUPS,
    _image_feature_groups,
    _image_level_features,
)
from spacr.settings import _overview_measurement_declarations


def _features(planes, group):
    return _image_level_features(planes, groups=(group,), confluency=False)


def test_intensity_extras_match_their_definitions():
    rng = np.random.default_rng(3)
    image = rng.gamma(2.0, 50.0, (64, 64))
    image[0, :8] = 0
    record = _features({0: image}, 'intensity')
    values = image.ravel()
    assert record['channel_0_intensity_min'] == 0
    assert record['channel_0_intensity_max'] == pytest.approx(values.max())
    assert record['channel_0_intensity_variance'] == pytest.approx(values.var())
    assert record['channel_0_intensity_mad'] == pytest.approx(np.median(np.abs(values - np.median(values))))
    assert record['channel_0_intensity_iqr'] == pytest.approx(np.subtract(*np.percentile(values, (75, 25))))
    assert record['channel_0_intensity_zero_fraction'] == pytest.approx(8 / values.size)
    assert record['channel_0_intensity_skewness'] == pytest.approx(skew(values))
    assert record['channel_0_intensity_kurtosis'] == pytest.approx(kurtosis(values))


def test_saturation_reports_the_underexposed_fraction():
    image = np.full((10, 10), 500, np.uint16)
    image[1, :] = 0
    record = _features({0: image}, 'saturation')
    assert record['channel_0_saturation_floor_fraction'] == pytest.approx(0.1)
    assert np.isnan(_features({0: image.astype(float)}, 'saturation')['channel_0_saturation_floor_fraction'])


def test_distribution_is_measures_whole_image_region():
    rng = np.random.default_rng(0)
    image = rng.exponential(100.0, (80, 80))
    image[5, 5] = np.nan
    record = _features({0: image}, 'distribution')
    data = image[np.isfinite(image)]
    assert record['channel_0_distribution_skewness'] == pytest.approx(skew(data))
    assert record['channel_0_distribution_kurtosis'] == pytest.approx(kurtosis(data))
    assert record['channel_0_distribution_cv'] == pytest.approx(data.std() / data.mean())
    assert record['channel_0_distribution_frac_high90'] == pytest.approx(0.1, abs=0.002)
    assert record['channel_0_distribution_frac_low10'] == pytest.approx(0.1, abs=0.002)
    assert record['channel_0_distribution_p85'] == pytest.approx(np.percentile(data, 85))
    flat = _features({0: np.full((20, 20), 7.0)}, 'distribution')
    assert flat['channel_0_distribution_gini'] == pytest.approx(0.0)
    assert flat['channel_0_distribution_range'] == 0
    assert flat['channel_0_distribution_entropy'] == pytest.approx(0.0)
    two = np.zeros((20, 20))
    two[:, 10:] = 1.0
    assert _features({0: two}, 'distribution')['channel_0_distribution_entropy'] == pytest.approx(1.0)


def test_radial_profile_follows_distance_from_the_centre():
    rows, columns = np.indices((101, 101))
    cone = np.hypot(rows - 50, columns - 50)
    record = _features({0: cone}, 'radial')
    bins = [record[f'channel_0_radial_bin_{index}'] for index in range(6)]
    assert all(later > earlier for earlier, later in zip(bins, bins[1:]))
    assert record['channel_0_radial_edge_centre_ratio'] > 5
    assert record['channel_0_radial_centroid_offset'] == pytest.approx(0.0, abs=1e-9)
    uniform = _features({0: np.full((64, 64), 9.0)}, 'radial')
    assert uniform['channel_0_radial_edge_centre_ratio'] == pytest.approx(1.0)
    assert uniform['channel_0_radial_block_cv'] == pytest.approx(0.0)
    corner = np.zeros((64, 64))
    corner[:16, :16] = 100.0
    shifted = _features({0: corner}, 'radial')
    assert shifted['channel_0_radial_centroid_offset'] == pytest.approx(
        np.hypot(31.5 - 7.5, 31.5 - 7.5) / (0.5 * np.hypot(63, 63)))
    assert shifted['channel_0_radial_block_cv'] > 1


def _spots(radius, spacing=24, size=192):
    image = np.zeros((size, size))
    rows, columns = np.indices(image.shape)
    for y in range(spacing // 2, size, spacing):
        for x in range(spacing // 2, size, spacing):
            image[np.hypot(rows - y, columns - x) <= radius] = 1000.0
    return image


def test_granularity_peaks_at_the_size_of_the_granules():
    fine = _features({0: _spots(1)}, 'granularity')
    coarse = _features({0: _spots(5)}, 'granularity')
    fine_spectrum = [fine[f'channel_0_granularity_{k}'] for k in range(1, 9)]
    coarse_spectrum = [coarse[f'channel_0_granularity_{k}'] for k in range(1, 9)]
    assert int(np.argmax(fine_spectrum)) + 1 <= 2
    assert int(np.argmax(coarse_spectrum)) + 1 >= 5
    assert sum(fine_spectrum) == pytest.approx(100.0, abs=1.0)
    flat = _features({0: np.full((64, 64), 3.0)}, 'granularity')
    assert all(flat[f'channel_0_granularity_{k}'] == 0 for k in range(1, 9))


def test_colocalization_of_identical_opposite_and_independent_channels():
    rng = np.random.default_rng(5)
    signal = gaussian_filter(rng.normal(0, 1, (128, 128)), 2)
    base = (1000 + 400 * signal).clip(0)
    record = _image_level_features({0: base, 1: base.copy(), 2: 3000 - base,
                                    3: rng.normal(1000, 50, (128, 128))},
                                   groups=('cross_channel',), confluency=False)
    assert record['channels_0_1_coloc_pearson'] == pytest.approx(1.0)
    assert record['channels_0_1_coloc_spearman'] == pytest.approx(1.0)
    assert record['channels_0_1_coloc_manders_m1'] == pytest.approx(1.0)
    assert record['channels_0_1_coloc_manders_m2'] == pytest.approx(1.0)
    assert record['channels_0_1_coloc_overlap_coefficient'] == pytest.approx(1.0)
    assert record['channels_0_1_coloc_icq'] == pytest.approx(0.5)
    assert record['channels_0_2_coloc_pearson'] == pytest.approx(-1.0)
    assert record['channels_0_2_coloc_icq'] == pytest.approx(-0.5)
    assert abs(record['channels_0_3_coloc_pearson']) < 0.05
    assert abs(record['channels_0_3_coloc_icq']) < 0.05
    assert not any(key.startswith('channel_0_') for key in record)


def test_spectral_focus_falls_with_blur():
    rng = np.random.default_rng(2)
    sharp = rng.normal(1000, 200, (256, 256))
    blurred = gaussian_filter(sharp, 2)
    sharp_record = _features({0: sharp}, 'focus')
    blurred_record = _features({0: blurred}, 'focus')
    assert blurred_record['channel_0_focus_power_spectrum_slope'] < sharp_record['channel_0_focus_power_spectrum_slope'] - 1
    assert blurred_record['channel_0_focus_high_frequency_ratio'] < 0.2 * sharp_record['channel_0_focus_high_frequency_ratio']
    assert sharp_record['channel_0_focus_normalized_variance'] == pytest.approx(sharp.var() / sharp.mean())
    flat = _features({0: np.full((64, 64), 5.0)}, 'focus')
    assert np.isnan(flat['channel_0_focus_power_spectrum_slope'])


def test_homogeneity_matches_the_stripe_period():
    columns = np.indices((64, 64))[1]
    stripes = np.where((columns // 2) % 2 == 0, 10.0, 200.0)
    record = _features({0: stripes}, 'homogeneity')
    assert record['channel_0_homogeneity_distance_4'] == pytest.approx(1.0)
    assert record['channel_0_homogeneity_distance_2'] < 0.01
    assert set(record) == {f'channel_0_homogeneity_distance_{d}' for d in (2, 4, 8, 16, 32, 64)}


def test_fibres_count_skeleton_length_and_branch_points():
    image = np.full((96, 96), 10.0)
    image[48, 10:86] = 1000.0
    image[10:86, 48] = 1000.0
    record = _features({0: image}, 'fibres')
    assert 151 * 0.9 <= record['channel_0_fibres_skeleton_length'] <= 151 * 1.15
    assert record['channel_0_fibres_branch_points'] >= 1
    assert record['channel_0_fibres_skeleton_density'] == pytest.approx(
        record['channel_0_fibres_skeleton_length'] / image.size)
    single = image.copy()
    single[10:86, 48] = 10.0
    assert _features({0: single}, 'fibres')['channel_0_fibres_branch_points'] == 0


def test_costly_groups_are_off_unless_chosen():
    defaults = _overview_measurement_declarations()['defaults']
    assert _image_feature_groups(defaults) == _IMAGE_SETTINGS_FEATURE_GROUPS
    assert not set(_IMAGE_COSTLY_FEATURE_GROUPS) & set(_image_feature_groups(None))
    assert not set(_IMAGE_COSTLY_FEATURE_GROUPS) & set(_image_feature_groups({}))
    chosen = dict(defaults, overview_measure_homogeneity=True)
    assert 'homogeneity' in _image_feature_groups(chosen) and 'fibres' not in _image_feature_groups(chosen)
    assert _image_feature_groups(['fibres', 'intensity']) == ('intensity', 'fibres')
    assert set(_IMAGE_FEATURE_SETTING) == set(_IMAGE_ALL_FEATURE_GROUPS) - set(_IMAGE_OBJECT_FEATURE_GROUPS)
    declared = _overview_measurement_declarations()
    advanced = declared['categories']['Advanced Measurements']
    for group in _IMAGE_COSTLY_FEATURE_GROUPS:
        assert _IMAGE_FEATURE_SETTING[group] in advanced
    regrouped = declared['categories']['Intensity'] + declared['categories']['Texture']
    for group in _IMAGE_SETTINGS_FEATURE_GROUPS:
        assert _IMAGE_FEATURE_SETTING[group] in regrouped
        assert declared['defaults'][_IMAGE_FEATURE_SETTING[group]] is True


def test_every_new_per_channel_column_is_named_channel_group_statistic():
    rng = np.random.default_rng(1)
    planes = {0: rng.poisson(30, (64, 64)).astype(np.uint16),
              1: rng.poisson(40, (64, 64)).astype(np.uint16)}
    for group in ('distribution', 'radial', 'granularity', 'homogeneity', 'fibres', 'radial_texture'):
        record = _features(planes, group)
        assert record and all(key.startswith((f'channel_0_{group}_', f'channel_1_{group}_'))
                              for key in record), group
    record = _features(planes, 'cross_channel')
    assert {key for key in record if '_coloc_' in key} == {
        f'channels_0_1_coloc_{name}' for name in
        ('pearson', 'spearman', 'manders_m1', 'manders_m2', 'overlap_coefficient', 'icq')}


def test_colocalization_without_shared_finite_pixels_is_undefined():
    from spacr.measure import _image_colocalization
    a, b = np.ones((4, 4)), np.ones((4, 4))
    fa, fb = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    fa[0, 0], fb[1, 1] = True, True
    record = _image_colocalization(a, fa, b, fb)
    assert len(record) == 6 and all(np.isnan(value) for value in record.values())
