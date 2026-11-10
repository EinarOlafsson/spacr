"""When each measure's pre-division change starts (item 901, alpha).

The onset is read back from the division: the direction is the sign of g at
the last frame, and the onset is the earliest frame of the unbroken run of
frames whose interval excludes zero in that direction. The reference is the
stage-2 script of the BMC reanalysis, whose per-frame effects and onsets for
13 datasets are pinned in tests/data/division_onsets_901.
"""
import os

import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da
from tests.test_division_analysis import _synthetic_movie
from tests.test_division_analysis_step import _tracked_movie

DATA = os.path.join(os.path.dirname(__file__), 'data', 'division_onsets_901')


def _effects(measure, rows):
    """A per-frame effect table from ``(rel_t, g, ci_low, ci_high)`` rows."""
    return pd.DataFrame([{'measure': measure, 'rel_t': r, 'g': g, 'ci_low': lo, 'ci_high': hi}
                         for r, g, lo, hi in rows])


def test_onsets_match_the_reference_script_on_all_13_datasets():
    effects = pd.read_csv(os.path.join(DATA, 'per_frame_effects.csv'))
    reference = pd.read_csv(os.path.join(DATA, 'reference_onsets.csv'))
    found = []
    for dataset, ref in reference.groupby('dataset', sort=False):
        out = da.onset_times(effects[effects['dataset'] == dataset], measures=list(ref['measure']))
        found.append(out.assign(dataset=dataset))
    found = pd.concat(found, ignore_index=True).merge(
        reference, on=['dataset', 'measure'], suffixes=('', '_ref'))
    assert len(found) == 130 and found['dataset'].nunique() == 13
    pd.testing.assert_series_equal(found['onset_rel_t'].astype(float),
                                   found['onset_rel_t_ref'].astype(float), check_names=False)
    pd.testing.assert_series_equal(found['direction'].astype(float),
                                   found['direction_ref'].astype(float), check_names=False)
    assert found['onset_rel_t'].notna().sum() == 100


def test_the_onset_is_the_start_of_the_unbroken_run_before_division():
    rows = [(-5, 1.0, 0.2, 2.0), (-4, 0.1, -0.5, 0.6), (-3, 1.0, 0.3, 2.0),
            (-2, 1.2, 0.4, 2.0), (-1, 1.5, 0.5, 2.0), (0, 2.0, 1.0, 3.0), (1, 2.0, 1.0, 3.0)]
    out = da.onset_times(_effects('area', rows))
    assert out.to_dict('records') == [{'measure': 'area', 'onset_rel_t': -3, 'direction': 1.0}]


def test_a_falling_measure_uses_the_upper_bound_and_unsorted_rows():
    rows = [(0, -1.0, -2.0, -0.2), (-2, -1.0, -2.0, -0.1), (-1, -1.0, -2.0, -0.3)]
    out = da.onset_times(_effects('speed', rows))
    assert out.loc[0, 'onset_rel_t'] == -2 and out.loc[0, 'direction'] == -1


def test_no_onset_when_the_last_frame_is_not_significant_or_g_is_missing():
    effects = pd.concat([
        _effects('area', [(-1, 1.0, 0.5, 2.0), (0, 0.5, -0.1, 1.0)]),
        _effects('solidity', [(-1, 1.0, 0.5, 2.0), (0, np.nan, np.nan, np.nan)]),
    ])
    out = da.onset_times(effects, measures=['area', 'solidity', 'turning']).set_index('measure')
    assert np.isnan(out.loc['area', 'onset_rel_t']) and out.loc['area', 'direction'] == 1
    assert out.loc[['solidity', 'turning']].isna().all().all()


def test_the_interval_columns_come_from_the_settings():
    effects = _effects('area', [(-1, 1.0, -1.0, 2.0), (0, 1.0, -1.0, 2.0)]).assign(
        lo90=0.1, hi90=1.9)
    assert np.isnan(da.onset_times(effects).loc[0, 'onset_rel_t'])
    tight = da.onset_times(effects, {'timelapse_division_onset_ci': 'lo90 / hi90'})
    assert tight.loc[0, 'onset_rel_t'] == -1
    with pytest.raises(ValueError):
        da.onset_times(effects, {'timelapse_division_onset_ci': 'lo90'})
    with pytest.raises(ValueError):
        da.onset_times(effects, {'timelapse_division_onset_ci': 'a/b'})
    empty = da.onset_times(pd.DataFrame(columns=['measure', 'rel_t', 'g']))
    assert list(empty.columns) == ['measure', 'onset_rel_t', 'direction'] and not len(empty)


def test_the_planted_rounding_starts_where_it_was_planted_and_other_results_do_not_change():
    meas, lineage = _synthetic_movie()
    plain = da.analyse_divisions(meas, lineage, mode='fission', window=6, n_boot=200)
    with_onsets = da.analyse_divisions(meas, lineage, mode='fission', window=6, n_boot=200,
                                       onsets=True, frame_boot=200)
    pd.testing.assert_frame_equal(plain['effects'], with_onsets['effects'])
    assert 'onsets' not in plain and 'frame_effects' not in plain
    frames = with_onsets['frame_effects']
    assert set(frames['rel_t']) == set(range(-6, 1))
    minor = frames[frames['measure'] == 'minor_axis'].set_index('rel_t')['g']
    assert (minor.loc[-2:0] > 10).all() and (minor.loc[-6:-3].abs() < 1).all()
    onsets = with_onsets['onsets'].set_index('measure')
    onset = onsets.loc['minor_axis', 'onset_rel_t']
    assert -6 <= onset <= -2 and onsets.loc['minor_axis', 'direction'] == 1
    assert onsets.loc['minor_axis', 'onset_cycle_fraction'] == pytest.approx(onset / 12)
    assert np.isnan(onsets.loc['minor_axis', 'onset_minutes'])
    assert with_onsets['summary']['onset_ci'] == 'ci_low/ci_high'
    calibrated = da.calibrate_divisions(with_onsets, frame_interval_min=5.0)
    assert calibrated['onsets'].set_index('measure').loc['minor_axis', 'onset_minutes'] == onset * 5
    again = da.calibrate_divisions(calibrated)
    assert again['onsets']['onset_minutes'].isna().all()


def test_frame_effects_skip_absent_measures_and_stay_empty_without_units():
    meas, lineage = _synthetic_movie(n_lineages=4)
    results = da.analyse_divisions(meas, lineage, mode='fission', window=6, n_boot=0,
                                   columns=('area',), onsets=True, frame_boot=0)
    assert set(results['frame_effects']['measure']) <= {'area'}
    none = da.frame_effects(pd.DataFrame(columns=['role', 'rel_t']),
                            pd.DataFrame(columns=['role', 'rel_t']), 3)
    assert list(none.columns) == da._FRAME_EFFECT_COLUMNS and not len(none)
    holder = {'summary': {'median_cycle_frames': float('nan')}}
    da._add_onsets(holder, {}, 3, ['area'], 'ci_low/ci_high', 10, 0)
    assert holder['frame_effects'].empty and holder['onsets']['onset_rel_t'].isna().all()
    assert holder['summary']['n_onsets'] == 0


def test_the_step_writes_onsets_csv_and_reads_it_back(tmp_path):
    from spacr.tabular import write_table

    stack, tracks = _tracked_movie()
    src = tmp_path / 'plate' / 'masks'
    os.makedirs(src)
    tracks_dir = tmp_path / 'plate' / 'tracks'
    write_table(tracks, tracks_dir / 'trackastra_tracks_cell_A01.csv')
    settings = {'timelapse_division_window': 6, 'save': False,
                'timelapse_division_size_tolerance': 0.4, 'timelapse_division_onset': True,
                'frame_interval_s': 120}
    result = da._run_division_step(str(src), 'A01', 'cell', 'trackastra', settings,
                                   label_stack=list(stack))
    out = tracks_dir / 'division' / 'trackastra_tracks_cell_A01'
    assert os.path.isfile(out / 'onsets.csv') and os.path.isfile(out / 'frame_effects.csv')
    back = da.read_division_results(out)
    onsets = back['onsets'].set_index('measure')
    assert onsets.loc['minor_axis', 'direction'] == 1
    assert onsets.loc['minor_axis', 'onset_rel_t'] <= -1
    assert onsets.loc['minor_axis', 'onset_minutes'] == onsets.loc['minor_axis', 'onset_rel_t'] * 2
    assert result['summary']['n_onsets'] == int(back['onsets']['onset_rel_t'].notna().sum())
    plain = tmp_path / 'plain'
    da.write_division_results({'summary': {}}, plain, plot=False)
    assert 'onsets' not in da.read_division_results(plain)


def test_onset_settings_are_registered_alpha_and_validated():
    from spacr.settings import (ALPHA_FEATURES, expected_types,
                                set_default_settings_preprocess_generate_masks, tooltips)

    defaults = set_default_settings_preprocess_generate_masks({})
    keys = ('timelapse_division_onset', 'timelapse_division_onset_ci')
    assert [defaults[k] for k in keys] == [False, 'ci_low/ci_high']
    assert set(keys) <= set(ALPHA_FEATURES[901]['settings'])
    for key in keys:
        assert key in expected_types and tooltips[key].endswith(f'Default {defaults[key]}.')
    assert 'onsets' not in da._division_settings(defaults)
    on = da._division_settings({**defaults, 'timelapse_division_onset': True})
    assert on['onsets'] is True and on['onset_ci'] == 'ci_low/ci_high'
    with pytest.raises(ValueError):
        da._division_settings({'timelapse_division_onset': True,
                               'timelapse_division_onset_ci': 'nope'})
