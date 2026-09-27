"""Time to event on tracked objects: Kaplan-Meier, log-rank and Cox.

The estimators are checked against the published Gehan (1965) leukaemia
remission data, the textbook example of R's ``survival`` package and of
lifelines: the numbers pinned below are lifelines 0.30.3 on the same table,
and agree with R's ``survfit(conf.type = "log-log")``, ``survdiff`` and
``coxph`` output for it to the printed digits. When lifelines is installed
the two engines are also run side by side on a simulated timelapse.

The event extraction is checked on hand-built tracks whose event frame and
censoring are known, for every event mode.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats
from spacr.measure import (_time_to_event, _time_to_event_groups,
                           _time_to_event_objects, _tte_first_run,
                           _tte_settings, _run_time_to_event_step)

MP_T = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32,
        34, 35]
MP_E = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
PL_T = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
PL_E = [1] * 21


def _gehan():
    return pd.DataFrame({'T': MP_T + PL_T, 'E': MP_E + PL_E,
                         'placebo': [0.0] * 21 + [1.0] * 21})


def test_kaplan_meier_matches_lifelines_and_r_on_gehan():
    curve = sp_stats._kaplan_meier(MP_T, MP_E, engine='builtin')
    at = curve.set_index('time')
    assert list(at.index) == [0, 6, 7, 9, 10, 11, 13, 16, 17, 19, 20, 22, 23,
                              25, 32, 34, 35]
    assert at.loc[6, 'at_risk'] == 21 and at.loc[6, 'events'] == 3
    assert at.loc[32, 'censored'] == 2
    expected = {6: (0.857143, 0.619718, 0.951552),
                13: (0.690196, 0.431610, 0.849066),
                22: (0.537815, 0.267779, 0.746791),
                23: (0.448179, 0.188052, 0.680143)}
    for time, values in expected.items():
        got = at.loc[time, ['survival', 'ci_lower', 'ci_upper']].to_numpy()
        np.testing.assert_allclose(got, values, atol=1e-6)
    median, low, high = sp_stats._median_survival(curve)
    assert (median, low) == (23.0, 13.0)
    assert np.isnan(high)


def test_logrank_matches_lifelines_and_r_on_gehan():
    result = sp_stats._logrank(MP_T + PL_T, MP_E + PL_E,
                               ['6-MP'] * 21 + ['placebo'] * 21,
                               engine='builtin')
    assert result['df'] == 1
    assert result['statistic'] == pytest.approx(16.792940989216547, rel=1e-9)
    assert result['p_value'] == pytest.approx(4.168809109334511e-05,
                                              rel=1e-7)
    assert result['observed'] == {'6-MP': 9.0, 'placebo': 21.0}


def test_cox_matches_lifelines_and_r_on_gehan():
    table, model = sp_stats._cox_regression(_gehan(), 'T', 'E', ['placebo'],
                                            engine='builtin')
    row = table.iloc[0]
    assert row['coef'] == pytest.approx(1.572125, abs=1e-5)
    assert row['hazard_ratio'] == pytest.approx(4.816873, abs=1e-4)
    assert row['se'] == pytest.approx(0.412397, abs=1e-5)
    assert row['hr_lower'] == pytest.approx(2.146508, abs=1e-4)
    assert row['hr_upper'] == pytest.approx(10.809307, abs=1e-3)
    assert row['p_value'] == pytest.approx(0.000138, abs=1e-6)
    assert model['log_likelihood'] == pytest.approx(-85.00842457737184,
                                                    abs=1e-6)
    assert model['lr_statistic'] == pytest.approx(16.35169083895181,
                                                  abs=1e-5)
    assert model['engine'] == 'builtin'


def test_a_curve_that_falls_to_zero_keeps_a_finite_band_before_it():
    curve = sp_stats._kaplan_meier([1, 2, 3], [1, 1, 1], engine='builtin')
    assert curve['survival'].iloc[-1] == 0.0
    assert np.isfinite(curve['ci_lower'].iloc[1:3]).all()
    assert curve.loc[0, 'ci_lower'] == curve.loc[0, 'ci_upper'] == 1.0


def test_the_estimators_refuse_bad_input():
    with pytest.raises(ValueError, match='non-negative'):
        sp_stats._kaplan_meier([1, -2], [1, 1], engine='builtin')
    with pytest.raises(ValueError, match='two groups'):
        sp_stats._logrank([1, 2], [1, 1], ['a', 'a'], engine='builtin')
    with pytest.raises(ValueError, match='unknown survival engine'):
        sp_stats._survival_engine('R')


def _tracks(spec, frames=10):
    """Object rows from ``{label: (first, last, values)}`` in one field."""
    rows = []
    for label, (first, last, values) in spec.items():
        for index, frame in enumerate(range(first, last + 1)):
            rows.append({'plateID': 'p1', 'rowID': 'r1', 'columnID': 'c1',
                         'fieldID': 'f1', 'object_label': label,
                         'timeID': f't{frame}',
                         'value': values[index] if values else 0.0})
    movies = {('p1', 'r1', 'c1', 'f1'): np.arange(frames, dtype=float)}
    return pd.DataFrame(rows), movies


def _config(**overrides):
    settings = {'time_to_event_min_frames': 1}
    settings.update({f'time_to_event_{k}': v for k, v in overrides.items()})
    return _tte_settings(settings)


def test_track_end_is_an_event_before_the_movie_ends_and_censored_at_it():
    frame, movies = _tracks({1: (0, 4, None), 2: (0, 9, None),
                             3: (2, 6, None)})
    objects, dropped = _time_to_event_objects(frame, _config(), movies)
    by = objects.set_index('object_label')
    assert by.loc[1, ['event', 'event_frame', 'duration']].tolist() == [
        1, 5.0, 5.0]
    assert by.loc[2, ['event', 'duration', 'censored_at']].tolist() == [
        0, 9.0, 'movie_end']
    assert by.loc[3, ['event', 'duration']].tolist() == [1, 5.0]
    assert sum(dropped.values()) == 0


def test_movie_origin_keeps_the_first_frame_cohort_and_counts_from_it():
    frame, movies = _tracks({1: (0, 4, None), 3: (2, 6, None)})
    objects, dropped = _time_to_event_objects(
        frame, _config(origin='movie', hours_per_frame=0.5), movies)
    assert objects['object_label'].tolist() == [1]
    assert objects['duration'].tolist() == [2.5]
    assert objects['time_unit'].tolist() == ['h']
    assert dropped['late'] == 1


def test_threshold_modes_date_the_event_and_censor_a_lost_track():
    frame, movies = _tracks({
        1: (0, 6, [1, 1, 5, 1, 5, 5, 5]),
        2: (0, 3, [1, 1, 1, 1]),
        3: (0, 5, [9, 9, 9, 9, 9, 9])})
    objects, dropped = _time_to_event_objects(
        frame, _config(mode='above', column='value', threshold=4), movies)
    by = objects.set_index('object_label')
    assert by.loc[1, ['event', 'event_frame']].tolist() == [1, 2.0]
    assert by.loc[2, ['event', 'duration', 'censored_at']].tolist() == [
        0, 3.0, 'track_lost']
    assert 3 not in by.index and dropped['at_first_frame'] == 1

    persisted, _ = _time_to_event_objects(
        frame, _config(mode='above', column='value', threshold=4,
                       persist=2), movies)
    assert persisted.set_index('object_label').loc[1, 'event_frame'] == 4.0

    below, dropped = _time_to_event_objects(
        frame, _config(mode='below', column='value', threshold=1), movies)
    assert below['object_label'].tolist() == [3]
    assert below['event'].tolist() == [0]
    assert dropped['at_first_frame'] == 2


def test_fold_change_is_relative_to_the_first_frame():
    frame, movies = _tracks({1: (0, 4, [2, 3, 4, 4, 4]),
                             2: (0, 4, [0, 3, 4, 4, 4])})
    objects, dropped = _time_to_event_objects(
        frame, _config(mode='fold_change', column='value', threshold=2),
        movies)
    assert objects.set_index('object_label').loc[1, 'event_frame'] == 2.0
    assert dropped['no_baseline'] == 1


def test_annotated_labels_count_any_non_zero_or_the_named_label():
    frame, movies = _tracks({1: (0, 4, [0, None, 2, 1, 1])})
    any_label, _ = _time_to_event_objects(
        frame, _config(mode='annotated', column='value'), movies)
    assert any_label['event_frame'].tolist() == [2.0]
    named, _ = _time_to_event_objects(
        frame, _config(mode='annotated', column='value', threshold=1),
        movies)
    assert named['event_frame'].tolist() == [3.0]


def test_the_first_run_needs_consecutive_rows():
    assert _tte_first_run([False, True, False, True, True], 2) == 3
    assert _tte_first_run([True, False], 2) is None
    assert _tte_first_run([True], 1) == 0


def test_short_tracks_are_left_out():
    frame, movies = _tracks({1: (0, 1, None), 2: (0, 5, None)})
    objects, dropped = _time_to_event_objects(
        frame, _config(min_frames=3), movies)
    assert objects['object_label'].tolist() == [2]
    assert dropped['short'] == 1


def test_settings_that_cannot_work_are_refused_by_name():
    with pytest.raises(ValueError, match='time_to_event_column'):
        _tte_settings({'time_to_event_mode': 'above'})
    with pytest.raises(ValueError, match='time_to_event_threshold'):
        _tte_settings({'time_to_event_mode': 'above',
                       'time_to_event_column': 'x'})
    with pytest.raises(ValueError, match='time_to_event_mode'):
        _tte_settings({'time_to_event_mode': 'death'})


def test_conditions_are_named_by_wells_and_the_reference_leads():
    objects = pd.DataFrame({
        'plateID': ['p1'] * 3, 'rowID': ['r1'] * 3,
        'columnID': ['c1', 'c2', 'c9'], 'fieldID': ['f1'] * 3,
        'object_label': [1, 2, 3], 'duration': [1.0, 2.0, 3.0],
        'event': [1, 1, 0]})
    config = _config(conditions=['mock=c1', 'drug=c2'], reference='drug')
    grouped, order = _time_to_event_groups(objects, config)
    assert grouped['condition'].tolist() == ['mock', 'drug']
    assert order == ['drug', 'mock']
    with pytest.raises(ValueError, match='not one of the conditions'):
        _time_to_event_groups(objects, _config(conditions=['mock=c1'],
                                               reference='drug'))
    with pytest.raises(ValueError, match='name=wells'):
        _time_to_event_groups(objects, _config(conditions=['c1']))
    by_well, order = _time_to_event_groups(objects, _config())
    assert len(order) == 3 and by_well['condition'].nunique() == 3


def _simulated_database(tmp_path, seed=571):
    """A two-condition timelapse with death annotated on png_list."""
    rng = np.random.default_rng(seed)
    cells, crops = [], []
    for column, rate in ((1, 0.03), (2, 0.03), (3, 0.09), (4, 0.09)):
        for label in range(1, 26):
            size = rng.normal(0, 1)
            death = int(np.ceil(rng.exponential(
                1 / (rate * np.exp(0.5 * size)))))
            last = 29 if rng.random() > 0.2 else int(rng.integers(5, 29))
            for t in range(last + 1):
                prcf = f'plate1_r1_c{column}_f1_t{t}'
                cells.append({'plateID': 'plate1', 'rowID': 'r1',
                              'columnID': f'c{column}', 'fieldID': 'f1',
                              'timeID': f't{t}', 'prcf': prcf,
                              'object_label': label,
                              'cell_area': 1000 + 100 * size})
                crops.append({'prcfo': f'{prcf}_o{label}',
                              'dead': 1 if t >= death else None})
    folder = tmp_path / 'measurements'
    folder.mkdir()
    db = folder / 'measurements.db'
    with sqlite3.connect(db) as conn:
        pd.DataFrame(cells).to_sql('cell', conn, index=False)
        pd.DataFrame(crops).to_sql('png_list', conn, index=False)
    return str(db)


SIMULATED = {'time_to_event': True, 'time_to_event_mode': 'annotated',
             'time_to_event_column': 'dead', 'time_to_event_min_frames': 1,
             'time_to_event_hours_per_frame': 0.5,
             'time_to_event_conditions': ['mock=c1,c2', 'drug=c3,c4'],
             'time_to_event_reference': 'mock',
             'time_to_event_covariates': ['cell_area']}


def test_the_whole_step_writes_its_tables_and_figures(tmp_path):
    db = _simulated_database(tmp_path)
    result = _time_to_event(db, SIMULATED, plot=True, engine='builtin')
    with sqlite3.connect(db) as conn:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        tests = pd.read_sql('SELECT * FROM time_to_event_tests', conn)
        cox = pd.read_sql('SELECT * FROM time_to_event_cox', conn)
        summary = pd.read_sql('SELECT * FROM time_to_event_summary', conn)
    assert {'time_to_event', 'time_to_event_curves', 'time_to_event_summary',
            'time_to_event_tests', 'time_to_event_cox'} <= names
    assert len(result['objects']) == 100
    assert set(summary['level']) == {'condition', 'well'}
    assert summary[summary['level'] == 'well']['condition'].tolist() == [
        'mock', 'mock', 'drug', 'drug']
    drug = cox.set_index('covariate').loc['condition=drug']
    assert drug['hazard_ratio'] > 1.5 and drug['reference'] == 'mock'
    assert tests['comparison'].tolist() == ['all conditions', 'drug vs mock']
    assert tests['p_value'].iloc[0] < 0.01
    assert [os.path.basename(p) for p in result['figures']] == [
        'kaplan_meier.pdf', 'hazard_ratios.pdf']
    assert all(os.path.isfile(p) for p in result['figures'])


def test_a_failed_step_says_why_and_does_not_raise(tmp_path, capsys):
    db = _simulated_database(tmp_path)
    assert _run_time_to_event_step(db, {'time_to_event_object': 'vacuole'}) \
        is None
    assert 'has no vacuole table' in capsys.readouterr().out


def test_a_table_measured_without_time_is_refused(tmp_path):
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as conn:
        pd.DataFrame({'plateID': ['p'], 'rowID': ['r1'], 'columnID': ['c1'],
                      'fieldID': ['f1'], 'object_label': [1]}).to_sql(
            'cell', conn, index=False)
    with pytest.raises(ValueError, match='timelapse on'):
        _time_to_event(str(db), {}, plot=False, engine='builtin')


def test_both_engines_give_the_same_tables(tmp_path):
    pytest.importorskip('lifelines')
    db = _simulated_database(tmp_path)
    ours = _time_to_event(db, SIMULATED, plot=False, engine='builtin')
    theirs = _time_to_event(db, SIMULATED, plot=False, engine='lifelines')
    columns = ['time', 'at_risk', 'survival', 'ci_lower', 'ci_upper']
    np.testing.assert_allclose(ours['curves'][columns].to_numpy(float),
                               theirs['curves'][columns].to_numpy(float),
                               atol=1e-9)
    np.testing.assert_allclose(ours['tests']['statistic'],
                               theirs['tests']['statistic'], rtol=1e-9)
    np.testing.assert_allclose(ours['cox']['hazard_ratio'],
                               theirs['cox']['hazard_ratio'], rtol=1e-4)
    np.testing.assert_allclose(ours['cox']['se'], theirs['cox']['se'],
                               rtol=1e-4)
    assert set(theirs['cox']['engine']) == {'lifelines'}
