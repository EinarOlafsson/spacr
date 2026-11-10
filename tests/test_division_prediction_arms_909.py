"""Division prediction options of item 909 (alpha): scoring unit and guards, the
shape-and-motion arm, single-movie grouping, reference features, learner and
label options. Every option defaults to the item-866 behaviour."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da
from spacr import division_prediction as dp
from tests.test_division_prediction_866 import _features, _fission_movie

ARMS = ['model', 'age_only', 'shape_motion', 'shuffled', 'majority']


def test_a_planted_shape_change_beats_age_and_a_pure_timer_does_not():
    planted = dp.predict_divisions(_features(True), model='gradient_boosting', n_boot=200,
                                   arms=ARMS, n_repeats=0, min_positive_tracks=5)
    m = planted['metrics'].set_index('arm')
    assert m.at['shape_motion', 'auroc'] > m.at['age_only', 'auroc'] + 0.1
    assert m.at['shape_motion', 'auroc'] > m.at['shuffled', 'auroc'] + 0.1
    assert 'shape_motion_minus_age_only_auroc' in planted['summary']
    assert planted['summary']['shape_motion_minus_age_only_auroc'][0] > 0
    assert m.at['shape_motion', 'label'] == 'shape and motion only'
    timer = dp.predict_divisions(_features(False, min_wait=20), model='gradient_boosting',
                                 n_boot=0, arms=ARMS, n_repeats=0)
    t = timer['metrics'].set_index('arm')
    assert t.at['age_only', 'auroc'] > t.at['shape_motion', 'auroc'] + 0.1
    assert abs(t.at['age_only', 'auroc'] - t.at['model', 'auroc']) < 0.2
    curves = planted['pr_curves']
    assert set(curves['arm']) == {'model', 'age_only', 'shape_motion', 'shuffled'}
    assert (curves.groupby('arm').size() <= 120).all()


def test_defaults_are_unchanged_and_counts_are_always_reported():
    features = _features(True)
    before = dp.predict_divisions(features, model='gradient_boosting', n_boot=50, n_repeats=0)
    again = dp.predict_divisions(features, model='gradient_boosting', n_boot=50, n_repeats=0,
                                 learner=dict(dp.LEARNER_DEFAULTS), min_positive_tracks=0)
    pd.testing.assert_frame_equal(before['metrics'].drop(columns='informative'),
                                  again['metrics'].drop(columns='informative'))
    assert list(before['metrics']['arm']) == ['model', 'age_only', 'shuffled', 'majority']
    s = before['summary']
    assert s['unit'] == 'track' and s['grouping'] == 'movie'
    assert s['n_positive_rows'] == s['n_positive'] and s['n_positive_tracks'] == s['n_positive']


def test_the_row_cap_and_both_guards():
    features = _features(True)
    every = dp.predict_divisions(features, model='gradient_boosting', n_boot=0, n_repeats=0,
                                 one_per_track=False, max_rows=400, row_seed=1)
    assert every['summary']['n_rows'] == 400 and every['summary']['unit'] == 'frame'
    assert every['summary']['n_positive_tracks'] < every['summary']['n_positive_rows'] + 1
    flagged = dp.predict_divisions(features, model='gradient_boosting', n_boot=0, n_repeats=0,
                                   min_positive_tracks=10 ** 6)
    assert flagged['summary']['informative'] is False and 'informative_note' in flagged['summary']
    assert not flagged['metrics']['informative'].any()
    figure = dp.division_prediction_figure(flagged)
    assert 'not be informative' in figure._suptitle.get_text() or \
        'informative' in figure._suptitle.get_text()
    skipped = dp.predict_divisions(features, model='gradient_boosting', min_positive_rows=10 ** 6)
    assert skipped['metrics'].empty and 'fewer than' in skipped['summary']['skipped']


def test_one_movie_needs_auto_or_lineage_tree_and_folds_never_split_a_tree():
    m, lin = _fission_movie(3, True, founders=12)
    single = dp.division_prediction_features(m, lin, frames=5).assign(movie='only')
    with pytest.raises(ValueError, match='lineage_tree'):
        dp.predict_divisions(single, model='gradient_boosting')
    result = dp.predict_divisions(single, model='gradient_boosting', group='auto', n_boot=20,
                                  n_repeats=0, arms=ARMS)
    s = result['summary']
    assert s['grouping'] == 'lineage_tree' and 'weaker' in s['grouping_note']
    p = result['predictions']
    assert (p.groupby('lineage_tree')['fold'].nunique() == 1).all()
    assert 'lineage trees' in dp.division_prediction_figure(result)._suptitle.get_text()
    with pytest.raises(ValueError):
        dp.predict_divisions(single.drop(columns='lineage_tree'), group='lineage_tree')
    with pytest.raises(ValueError):
        dp.predict_divisions(single, group='field')
    with pytest.raises(ValueError):
        dp.predict_divisions(single, arms=['age_only'])


def test_reference_features_labels_and_born_flag():
    lineage = pd.DataFrame({'label': [1, 2, 3, 4], 't_start': [0, 10, 10, 2],
                            't_end': [9, 14, 19, 12], 'parent': [0, 1, 1, 0]})
    rows = [(1, t) for t in range(10)] + [(2, t) for t in range(10, 15)] + \
        [(3, t) for t in range(10, 20)] + [(4, t) for t in range(2, 13)]
    measures = pd.DataFrame(rows, columns=['track_id', 't']).assign(
        area=lambda d: 100.0 + d['t'], minor_axis=5.0)
    f = dp.division_prediction_features(measures, lineage, frames=3, mode='fission',
                                        feature_set='reference', lag=2,
                                        negative_rule='last_seen')
    assert list(f.columns[:5]) == ['track_id', 't', 'label', 'frames_to_division', 'lineage_tree']
    assert list(f.columns[5:]) == ['area', 'area_d1', 'area_d2', 'area_roll2', 'minor_axis',
                                   'minor_axis_d1', 'minor_axis_d2', 'minor_axis_roll2',
                                   'age', 'born_in_view']
    one = f[f['track_id'] == 1].set_index('t')
    assert list(one.loc[7:9, 'label']) == [1, 1, 1] and one.at[3, 'area_d2'] == 2
    assert one.at[3, 'area_roll2'] == 102.5 and np.isnan(one.at[0, 'area_d1'])
    assert one.at[0, 'born_in_view'] == 0 and f.loc[f['track_id'] == 4, 'born_in_view'].iloc[0] == 1
    assert (f.loc[f['track_id'] == 3, 'lineage_tree'] == 1).all()
    assert f.loc[f['track_id'] == 4, 't'].max() == 10
    ends = dp.division_prediction_features(measures, lineage, frames=3, mode='fission',
                                           feature_set='reference', lag=2)
    assert ends.loc[ends['track_id'] == 4, 't'].max() == 9
    spacr = dp.division_prediction_features(measures, lineage, frames=3, mode='fission',
                                            negative_rule='last_seen', born_flag=True)
    assert spacr.loc[spacr['track_id'] == 4, 't'].max() == 10 and 'born_in_view' in spacr
    three = pd.concat([lineage, pd.DataFrame({'label': [5], 't_start': [10], 't_end': [12],
                                              'parent': [1]})], ignore_index=True)
    both = dp.division_prediction_features(measures, three, frames=3, mode='fission')
    two = dp.division_prediction_features(measures, three, frames=3, mode='fission',
                                          events='two_daughters')
    assert both.loc[both['track_id'] == 1, 'label'].sum() == 3
    assert two.loc[two['track_id'] == 1, 'label'].sum() == 0
    with pytest.raises(ValueError):
        dp.division_prediction_features(measures, lineage, feature_set='funk')
    assert dp.lineage_trees(lineage) == {1: 1, 2: 1, 3: 1, 4: 4}


def test_settings_become_options_and_the_reference_preset_loads():
    assert dp.prediction_run_options({})['learner'] is None
    preset_path = os.path.join(os.path.dirname(dp.__file__), 'resources',
                               'division_prediction_reference.json')
    with open(preset_path) as handle:
        preset = json.load(handle)
    run = dp.prediction_run_options(preset['settings'])
    assert run['one_per_track'] is False and run['max_rows'] == 120000 and run['group'] == 'auto'
    assert run['learner']['max_bin'] == 64 and run['seed'] == 20261104
    assert run['arms'] == ARMS and preset['horizons_frames']['bacmother_ctc'] == 2
    feats = dp.prediction_feature_options({**preset['settings'],
                                           'timelapse_division_prediction_frames': 'auto'},
                                          {'window': 9, 'median_cycle_frames': 40})
    assert feats == {'frames': 6, 'history': 3, 'feature_set': 'reference', 'lag': 5,
                     'born_flag': True, 'events': 'two_daughters', 'negative_rule': 'last_seen'}
    assert dp.prediction_feature_options({'timelapse_division_prediction_frames': 'auto'})['frames'] == 3
    for bad in ({'timelapse_division_prediction_unit': 'cell'},
                {'timelapse_division_prediction_arms': 'model,coin'}):
        with pytest.raises(ValueError):
            dp.prediction_run_options(bad)
    with pytest.raises(ValueError):
        dp.prediction_feature_options({'timelapse_division_prediction_events': 'three'})
    assert dp.division_prediction_pr_curve([1, 1], [0.2, 0.4]).empty
    from spacr.settings import ALPHA_FEATURES, expected_types, tooltips
    for key in dp.PREDICTION_DEFAULTS:
        assert key in ALPHA_FEATURES[866]['settings']
        assert key in expected_types and key in tooltips


def test_the_run_step_reads_the_new_settings(tmp_path, capsys):
    from spacr.tabular import write_table

    for seed in range(3):
        m, lin = _fission_movie(seed, True)
        folder = tmp_path / f'trackastra_tracks_cell_F{seed}'
        os.makedirs(folder)
        write_table(dp.division_prediction_features(m, lin, frames=5), folder / 'prediction_features.csv')
    result = dp._run_prediction_step(str(tmp_path), 'trackastra_tracks_cell_',
                                     {'save': False, 'timelapse_division_prediction_model': 'gradient_boosting',
                                      'timelapse_division_prediction_arms': ','.join(ARMS),
                                      'timelapse_division_prediction_min_positive_tracks': 5})
    assert list(result['metrics']['arm']) == ARMS
    assert os.path.isfile(tmp_path / 'prediction_trackastra_tracks_cell' / 'pr_curves.csv')
    skipped = dp._run_prediction_step(str(tmp_path), 'trackastra_tracks_cell_',
                                      {'save': False,
                                       'timelapse_division_prediction_model': 'gradient_boosting',
                                       'timelapse_division_prediction_min_positive_rows': 10 ** 6})
    assert skipped['summary']['skipped'] and 'not fitted' in capsys.readouterr().out
