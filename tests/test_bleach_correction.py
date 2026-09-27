"""Photobleaching correction of timelapse intensity measurements.

A synthetic bleaching series -- every object's intensity decaying as
``a * exp(-b * t) + c`` with its own brightness and noise -- must come back
with a flat background trend under each of the three methods, and the
corrected table must say which method made it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr.tabular import database_tables, read_table, write_database
from spacr.timelapse import (
    _bleach_channel_columns, _bleach_correct_table, _bleach_factors,
    _correct_timelapse_bleaching, _fit_bleach_decay, _histogram_match,
)

DECAY = {0: (600.0, 0.15, 200.0), 1: (300.0, 0.05, 50.0)}


def _bleaching_table(n_frames=12, n_objects=40, fields=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for field in fields:
        brightness = rng.lognormal(0.0, 0.3, n_objects)
        for t in range(n_frames):
            for label in range(1, n_objects + 1):
                row = {'object_label': label, 'plateID': 'plate1',
                       'rowID': 'r1', 'columnID': 'c1', 'fieldID': field,
                       'timeID': t, 'prcf': f'plate1_r1_c1_f{field}',
                       'file_name': f'plate1_r1_c1_f{field}_t{t}',
                       'cell_area': 100.0}
                for channel, (a, b, c) in DECAY.items():
                    level = (a * np.exp(-b * t) + c) * brightness[label - 1]
                    level *= rng.normal(1.0, 0.01)
                    row[f'cell_channel_{channel}_mean_intensity'] = level
                    row[f'cell_channel_{channel}_integrated_intensity'] = level * 100
                    row[f'cell_channel_{channel}_percentile_95'] = level * 1.4
                    row[f'cell_channel_{channel}_cv_intensity'] = 0.2
                rows.append(row)
    return pd.DataFrame(rows)


def _trend_drift(frame, column):
    trend = frame.groupby('timeID')[column].median()
    return float(trend.iloc[-1] / trend.iloc[0] - 1.0)


def test_level_columns_are_grouped_by_channel_with_the_mean_first():
    df = _bleaching_table(n_frames=2, n_objects=3, fields=(1,))
    channels = _bleach_channel_columns(df, 'cell')
    assert sorted(channels) == [0, 1]
    assert channels[0][0] == 'cell_channel_0_mean_intensity'
    assert 'cell_channel_0_integrated_intensity' in channels[0]
    assert 'cell_channel_0_percentile_95' in channels[0]
    assert 'cell_channel_0_cv_intensity' not in channels[0]


def test_the_exponential_fit_recovers_the_decay():
    t = np.arange(15, dtype=float)
    a, b, c = 500.0, 0.2, 100.0
    params = _fit_bleach_decay(t + 3, a * np.exp(-b * t) + c)
    assert params is not None
    assert params == pytest.approx((a, b, c), rel=1e-3)
    assert _fit_bleach_decay(t[:2], (a * np.exp(-b * t) + c)[:2]) is None


def test_too_short_a_series_falls_back_to_the_ratio():
    trend = pd.Series([100.0, 80.0], index=[0, 1])
    factors, applied, params = _bleach_factors(trend, 'exponential')
    assert applied == 'ratio_fallback'
    assert params is None
    assert factors.tolist() == pytest.approx([1.0, 1.25])


def test_histogram_matching_takes_the_reference_distribution():
    reference = np.linspace(10, 20, 101)
    matched = _histogram_match(np.array([1.0, np.nan, 3.0, 2.0]), reference)
    assert np.isnan(matched[1])
    assert matched[0] < matched[3] < matched[2]
    assert reference.min() <= np.nanmin(matched)
    assert np.nanmax(matched) <= reference.max()


@pytest.mark.parametrize('method', ['ratio', 'exponential', 'histogram'])
def test_a_bleaching_series_corrects_to_a_flat_trend(method):
    df = _bleaching_table()
    for channel in DECAY:
        assert _trend_drift(df, f'cell_channel_{channel}_mean_intensity') < -0.2
    corrected, fits = _bleach_correct_table(df, 'cell', method)
    assert (corrected['bleach_correction_method'] == method).all()
    assert len(corrected) == len(df)
    for channel in DECAY:
        for suffix in ('mean_intensity', 'integrated_intensity',
                       'percentile_95'):
            column = f'cell_channel_{channel}_{suffix}'
            for _, part in corrected.groupby('fieldID'):
                assert abs(_trend_drift(part, column)) < 0.02, (method, column)
        assert (corrected[f'cell_channel_{channel}_mean_intensity'].notna()).all()
    assert 'cell_channel_0_cv_intensity' not in corrected
    assert len(fits) == 2 * len(DECAY)
    assert set(fits['method']) == {method}
    drift = fits['corrected_last'] / fits['corrected_first'] - 1
    assert drift.abs().max() < 0.02
    if method == 'exponential':
        # The slow channel decays only 45 % over the series, so its rate is
        # less well determined than the fast one's; the correction is flat
        # either way.
        tolerance = {0: 0.1, 1: 0.25}
        for channel, (a, b, c) in DECAY.items():
            rates = fits.loc[fits['channel'] == channel, 'decay_b']
            assert rates.to_numpy() == pytest.approx(b, rel=tolerance[channel])
            half = fits.loc[fits['channel'] == channel, 'half_life']
            assert half.to_numpy() == pytest.approx(
                np.log(2) / b, rel=tolerance[channel] * 1.5)
    else:
        assert fits['decay_b'].isna().all()


def test_the_table_needs_a_time_axis_and_a_known_method():
    df = _bleaching_table(n_frames=3, n_objects=4, fields=(1,))
    with pytest.raises(ValueError, match='bleach_correction'):
        _bleach_correct_table(df, 'cell', 'gaussian')
    with pytest.raises(ValueError, match='timepoint'):
        _bleach_correct_table(df.drop(columns='timeID'), 'cell', 'ratio')


def test_the_database_gets_corrected_tables_fits_and_a_figure(tmp_path):
    db = tmp_path / 'plate1' / 'measurements' / 'measurements.db'
    df = _bleaching_table(n_frames=8, n_objects=20)
    write_database(df, db, 'cell', if_exists='replace', canonicalise=False)
    fits = _correct_timelapse_bleaching(str(db), 'exponential')
    tables = database_tables(str(db))
    assert {'cell', 'cell_bleach_corrected', 'bleach_correction'} <= set(tables)
    stored = read_table(str(db), table='cell_bleach_corrected', report=None)
    assert set(stored['bleach_correction_method']) == {'exponential'}
    assert abs(_trend_drift(stored, 'cell_channel_0_mean_intensity')) < 0.02
    measured = read_table(str(db), table='cell', report=None)
    assert _trend_drift(measured, 'cell_channel_0_mean_intensity') < -0.2
    recorded = read_table(str(db), table='bleach_correction', report=None)
    assert len(recorded) == len(fits) == 4
    figures = os.listdir(tmp_path / 'plate1' / 'results' / 'bleach_correction')
    assert len(figures) == 1 and figures[0].startswith('cell.')


def test_measure_runs_the_step_and_survives_a_failure(tmp_path, capsys):
    from spacr.measure import _run_bleach_correction_step

    db = tmp_path / 'measurements' / 'measurements.db'
    write_database(_bleaching_table(n_frames=5, n_objects=10), db, 'cell',
                   if_exists='replace', canonicalise=False)
    fits = _run_bleach_correction_step(str(db), {'bleach_correction': 'ratio'})
    assert set(fits['method']) == {'ratio'}
    assert 'cell_bleach_corrected' in capsys.readouterr().out

    empty = tmp_path / 'other' / 'measurements.db'
    write_database(pd.DataFrame({'x': [1]}), empty, 'misc',
                   if_exists='replace', canonicalise=False)
    assert _run_bleach_correction_step(
        str(empty), {'bleach_correction': 'ratio'}) is None
    assert 'could not be applied' in capsys.readouterr().out


def test_measure_offers_the_setting_off_by_default():
    from spacr.settings import get_measure_crop_settings
    from spacr.timelapse import _BLEACH_METHODS

    settings = get_measure_crop_settings({'src': '/tmp/none'})
    assert settings['bleach_correction'] == 'none'
    assert _BLEACH_METHODS[0] == 'none'
