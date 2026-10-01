"""Opt-in calcium traces share field-local bleach correction without refitting."""
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as TL

LEVEL = 'cell_channel_1_mean_intensity'
RING = 'cell_channel_1_outside_percentile_50'
TIMES = [0, 1, 3, 4, 6, 8, 10, 12]


@pytest.fixture(autouse=True)
def figures():
    yield
    plt.close('all')


def measured_frame():
    rows = []
    for field, initial, decay, offset in [('f1', 500., .08, 33000.),
                                          ('f2', 1000., .03, 2000.)]:
        for time in TIMES:
            for label in [7, 42, 99]:
                pulse = 1.5 if label == 7 and time == 4 else 1.
                background = offset + time * 5
                signal = initial * np.exp(-decay * time) * pulse
                rows.append(dict(plateID='plate1', rowID='r1', columnID='c1',
                                 fieldID=field, timeID=f't{time}', time=time,
                                 object_label=label, prcf=f'plate1_r1_c1_{field}',
                                 cell_area=20., **{LEVEL: background + signal,
                                                  RING: background}))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('method', ['ratio', 'exponential'])
def test_independent_fields_preserve_absolute_levels_and_calcium_pulse(method):
    raw = measured_frame()
    original = raw.copy(deep=True)
    result, fits = TL._calcium_shared_bleaching(raw, LEVEL, method)
    expected = np.where((raw.object_label == 7) & (raw.time == 4), 1.5, 1.)
    np.testing.assert_allclose(result['corrected_' + LEVEL], expected, atol=1e-7)
    bases = np.where(raw.fieldID == 'f1', 500., 1000.)
    np.testing.assert_allclose(result['bleach_corrected_' + LEVEL],
                               raw[RING] + bases * expected, atol=1e-5)
    np.testing.assert_allclose(result['bleach_baseline_' + LEVEL], bases)
    pd.testing.assert_series_equal(result[LEVEL], raw[LEVEL])
    pd.testing.assert_frame_equal(raw, original)
    assert set(fits['fieldID']) == {'f1', 'f2'}
    assert set(fits['method']) == {method}
    assert set(fits['normalized_units']) == {'dimensionless'}


def test_integrated_levels_subtract_ring_times_area():
    raw = measured_frame()
    column = 'cell_channel_1_integrated_intensity'
    raw[column] = raw[LEVEL] * raw.cell_area
    result, _ = TL._calcium_shared_bleaching(raw, column, 'ratio')
    np.testing.assert_allclose(result['corrected_' + column],
                               np.where((raw.object_label == 7) & (raw.time == 4), 1.5, 1.))
    np.testing.assert_allclose(result['bleach_background_' + column], raw[RING] * 20)


@pytest.mark.parametrize('unknown', ['missing_ring', 'zero_baseline', 'negative_baseline',
                                    'missing_initial', 'infinite_ring'])
def test_unavailable_background_or_initial_signal_stays_unknown(unknown):
    raw = measured_frame()
    if unknown == 'missing_ring':
        raw = raw.drop(columns=RING)
    elif unknown == 'infinite_ring':
        raw[RING] = np.inf
    else:
        first = raw.time == 0
        raw.loc[first, LEVEL] = (np.nan if unknown == 'missing_initial' else
                                 raw.loc[first, RING] - (1 if unknown == 'negative_baseline' else 0))
    result, fits = TL._calcium_shared_bleaching(raw, LEVEL, 'ratio')
    assert result['corrected_' + LEVEL].isna().all()
    assert result['bleach_baseline_' + LEVEL].isna().all()
    assert fits['baseline_signal'].isna().all()


def test_short_exponential_series_records_actual_ratio_fallback():
    raw = measured_frame().query('time < 2')
    result, fits = TL._calcium_shared_bleaching(raw, LEVEL, 'exponential')
    assert set(result.bleach_correction_method) == {'ratio_fallback'}
    assert set(fits.method) == {'ratio_fallback'}
    assert set(fits.requested_method) == {'exponential'}


def test_histogram_path_records_method_and_uses_shared_absolute_result():
    raw = measured_frame()
    expected, _ = TL._bleach_correct_table(raw, 'cell', 'histogram')
    result, fits = TL._calcium_shared_bleaching(raw, LEVEL, 'histogram')
    np.testing.assert_allclose(result['bleach_corrected_' + LEVEL], expected[LEVEL])
    assert set(fits.method) == {'histogram'}


def test_ambiguous_row_identity_and_nonlevel_selection_fail():
    raw = measured_frame()
    with pytest.raises(ValueError, match='unique'):
        TL._calcium_shared_bleaching(pd.concat([raw, raw.iloc[:1]]), LEVEL, 'ratio')
    with pytest.raises(ValueError, match='intensity level'):
        TL._calcium_shared_bleaching(raw, 'cell_area', 'ratio')


def database(tmp_path, raw):
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as conn:
        raw.to_sql('cell', conn, index=False)
        # A stale saved correction must never be consumed or corrected again.
        raw.assign(**{LEVEL: 999999.}).to_sql('cell_bleach_corrected', conn, index=False)
    return db


def test_actual_analysis_uses_raw_once_and_exports_fits_without_mutating_database(
        tmp_path, monkeypatch):
    raw = measured_frame()
    db = database(tmp_path, raw)
    before = db.read_bytes()
    monkeypatch.setattr(TL, 'curve_fit', lambda *a, **k: pytest.fail('global refit attempted'))
    result, peaks, _ = TL.analyze_calcium_oscillations(
        str(db), measurement=LEVEL, bleach_correction='ratio', remove_transient=False)
    found = peaks.dropna(subset=['time'])
    assert found['time'].tolist() == [4, 4]
    np.testing.assert_allclose(found.amplitude, [.5, .5], atol=1e-10)
    assert set(result.bleach_correction_method) == {'ratio'}
    assert db.read_bytes() == before
    fits = pd.read_csv(tmp_path / 'results' / 'bleach_correction_fits.csv')
    assert set(fits.source_database) == {str(db)}
    assert set(fits.source_table) == {'cell'}
    assert sorted(fits.baseline_signal) == [500., 1000.]
    assert (tmp_path / 'results' / 'results.csv').is_file()


def test_missing_observation_does_not_become_zero_delta_or_bridge_peak(tmp_path):
    raw = measured_frame()
    raw.loc[(raw.object_label == 7) & (raw.time == 3), RING] = np.nan
    db = database(tmp_path, raw)
    result, peaks, _ = TL.analyze_calcium_oscillations(
        str(db), measurement=LEVEL, bleach_correction='ratio', remove_transient=False)
    unknown = result[(result.object_label == 7) & result.time.isin([3, 4])]
    assert unknown['delta_' + LEVEL].isna().all()
    assert peaks['time'].isna().all()


def test_absent_object_row_does_not_bridge_an_observed_field_timepoint(tmp_path):
    raw = measured_frame()
    raw = raw[~((raw.object_label == 7) & (raw.time == 3))]
    db = database(tmp_path, raw)
    result, peaks, _ = TL.analyze_calcium_oscillations(
        str(db), measurement=LEVEL, bleach_correction='ratio', remove_transient=False)
    after_gap = result[(result.object_label == 7) & (result.time == 4)]
    assert after_gap['delta_' + LEVEL].isna().all()
    assert peaks['time'].isna().all()


def test_invalid_method_fails_before_creating_database(tmp_path):
    db = tmp_path / 'absent.db'
    with pytest.raises(ValueError, match='bleach_correction'):
        TL.analyze_calcium_oscillations(str(db), bleach_correction='invented')
    assert not db.exists()
