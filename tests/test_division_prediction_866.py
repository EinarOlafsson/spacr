"""Predicting whether a tracked cell divides within N frames (item 866)."""
import json

import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da
from spacr import division_prediction as dp


def _fission_movie(seed, planted, founders=6, frames=70, horizon=5, min_wait=6):
    """Tracks that divide after a memoryless wait; with ``planted`` they elongate before it.

    Division waits are ``min_wait`` frames plus a geometric wait; with
    ``min_wait`` 1 they are memoryless and a cell's age says nothing about
    when it will divide, with more a young cell cannot divide yet. Size
    stays flat. The planted change raises the aspect ratio
    by 0.6 over the last ``horizon`` frames before each division.
    """
    rng = np.random.default_rng(seed)
    measures, lineage = [], []
    queue = [(i + 1, 0, 0) for i in range(founders)]
    next_id = founders + 1
    while queue:
        tid, start, parent = queue.pop()
        wait = min_wait + int(rng.geometric(0.08))
        end = min(start + wait - 1, frames - 1)
        divides = start + wait - 1 <= frames - 2
        for t in range(start, end + 1):
            ar = 1.5 + rng.normal(0, 0.1)
            if planted and divides and end - t < horizon:
                ar += 0.6 * (horizon - (end - t)) / horizon
            measures.append((tid, t, 300 + rng.normal(0, 10), ar, 1 + rng.normal(0, 0.05)))
        lineage.append((tid, start, end, parent))
        if divides:
            for _ in range(2):
                queue.append((next_id, end + 1, tid))
                next_id += 1
    m = pd.DataFrame(measures, columns=['track_id', 't', 'area', 'aspect_ratio',
                                        'mean_intensity'])
    return m, pd.DataFrame(lineage, columns=list(da.LINEAGE_COLUMNS))


def _features(planted, movies=6, frames=5, min_wait=6):
    tables = []
    for seed in range(movies):
        m, lin = _fission_movie(seed + (100 if planted else 0), planted, min_wait=min_wait)
        tables.append(dp.division_prediction_features(m, lin, frames=frames)
                      .assign(movie=f'movie{seed}'))
    return pd.concat(tables, ignore_index=True)


def test_labels_follow_the_horizon_and_censor_unseen_futures():
    lineage = pd.DataFrame({'label': [1, 2, 3, 4], 't_start': [0, 10, 10, 0],
                            't_end': [9, 14, 19, 12], 'parent': [0, 1, 1, 0]})
    rows = [(1, t) for t in range(10)] + [(2, t) for t in range(10, 15)] + \
        [(3, t) for t in range(10, 20)] + [(4, t) for t in range(13)]
    measures = pd.DataFrame(rows, columns=['track_id', 't']).assign(area=100.0)
    f = dp.division_prediction_features(measures, lineage, frames=3, mode='fission', history=2)
    one = f[f['track_id'] == 1].set_index('t')
    assert list(one.loc[7:9, 'label']) == [1, 1, 1] and one.at[6, 'label'] == 0
    assert one.at[9, 'frames_to_division'] == 0 and one.at[0, 'age'] == 0
    assert np.isnan(one.at[0, 'd_area']) and one.at[5, 'd_area'] == 0
    lost = f[f['track_id'] == 4]
    assert lost['t'].max() == 9 and (lost['label'] == 0).all()
    assert f.loc[f['track_id'] == 2, 't'].max() == 11
    assert f.loc[f['track_id'] == 3, 't'].max() == 16


def test_budding_mothers_restart_their_age_at_each_bud():
    lineage = pd.DataFrame({'label': [1, 2, 3], 't_start': [0, 5, 12], 't_end': [20, 20, 20],
                            'parent': [0, 1, 1]})
    measures = pd.DataFrame({'track_id': [1] * 21, 't': range(21), 'area': 200.0})
    f = dp.division_prediction_features(measures, lineage, frames=2)
    mother = f.set_index('t')
    assert mother.at[4, 'label'] == 1 and mother.at[3, 'label'] == 1 and mother.at[2, 'label'] == 0
    assert mother.at[5, 'age'] == 0 and mother.at[11, 'age'] == 6 and mother.at[12, 'age'] == 0
    assert mother.index.max() == 18


def test_the_planted_change_beats_every_control_on_held_out_movies():
    results = dp.predict_divisions(_features(True, movies=8), model='gradient_boosting', folds=4,
                                   n_boot=200, rng=0, min_movies=5, n_repeats=2)
    metrics = results['metrics'].set_index('arm')
    assert metrics.at['model', 'auroc'] > 0.75
    assert abs(metrics.at['shuffled', 'auroc'] - 0.5) < 0.15
    assert metrics.at['majority', 'auroc'] == 0.5
    assert metrics.at['majority', 'auprc'] == pytest.approx(results['summary']['prevalence'])
    assert metrics.at['shuffled', 'delta_auroc_ci_low'] > 0
    assert metrics.at['age_only', 'delta_auroc_ci_low'] > 0
    summary = results['summary']
    assert summary['beats_shuffled'] and summary['beats_age_only'] and summary['beats_majority']
    assert summary['ci_method'] == 'movie' and summary['n_folds'] == 4
    assert 'beats the shuffled-label and age-only' in summary['verdict']
    folds = results['folds']
    assert (folds['n_shared_movies'] == 0).all()
    for _, g in folds[folds['arm'] == 'model'].iterrows():
        assert g['test_movies']
    tested = folds[folds['arm'] == 'model']['test_movies'].str.split(';').sum()
    assert sorted(tested) == sorted(set(tested)) == sorted(summary['movies'])
    top = results['importances'].iloc[0]['feature']
    assert 'aspect_ratio' in top
    assert (results['predictions']['fold'] >= 0).all()
    keys = results['predictions'][['movie', 'track_id']]
    assert not keys.duplicated().any()
    assert summary['n_rows'] == len(keys) == _features(True, movies=8)[
        ['movie', 'track_id']].drop_duplicates().shape[0]
    again = dp.predict_divisions(_features(True, movies=8), model='gradient_boosting', folds=4,
                                 n_boot=0, rng=0, n_repeats=0)
    pd.testing.assert_frame_equal(again['predictions'][['movie', 'track_id', 't']],
                                  results['predictions'][['movie', 'track_id', 't']])
    every = dp.predict_divisions(_features(True, movies=3), model='logistic', folds=3,
                                 n_boot=0, n_repeats=0, one_per_track=False)
    assert every['summary']['row_sampling'] == 'every labelled frame'
    assert every['summary']['n_rows'] == len(_features(True, movies=3))


def test_without_a_planted_change_the_model_does_not_beat_shuffled_labels():
    results = dp.predict_divisions(_features(False, min_wait=1), model='logistic', folds=6,
                                   n_boot=200, rng=1, n_repeats=0)
    summary = results['summary']
    assert not summary['beats_shuffled']
    assert summary['verdict'] == 'the model does not beat the shuffled-label control'
    assert results['metrics'].set_index('arm').at['model', 'auroc'] < 0.6
    assert not len(results['importances'])


def test_a_cell_cycle_clock_is_caught_by_the_age_only_control():
    results = dp.predict_divisions(_features(False), model='logistic', folds=6,
                                   n_boot=200, rng=1, n_repeats=0)
    summary = results['summary']
    assert summary['beats_shuffled'] and not summary['beats_age_only']
    assert 'cell cycle clock' in summary['verdict']


def test_few_movies_resample_rows_and_say_so(tmp_path):
    feats = _features(True, movies=3)
    results = dp.predict_divisions(feats, model='auto', folds=5, n_boot=50, n_repeats=1)
    summary = results['summary']
    assert summary['n_folds'] == 3 and summary['ci_method'] == 'row'
    assert 'too narrow' in summary['ci_note']
    assert summary['learner'] in ('xgboost', 'gradient_boosting')
    paths = dp.write_division_prediction(results, tmp_path / 'out')
    assert {'metrics', 'folds', 'importances', 'predictions', 'summary', 'figure'} <= set(paths)
    back = dp.read_division_prediction(tmp_path / 'out')
    assert len(back['metrics']) == 4 and back['summary']['n_movies'] == 3
    json.loads((tmp_path / 'out' / 'summary.json').read_text())
    fig = dp.division_prediction_figure(back)
    assert len(fig.axes) == 2 and fig.axes[0].get_ylabel() == 'AUROC'
    none = dp.predict_divisions(feats, model='gradient_boosting', folds=3, n_boot=0, n_repeats=0)
    assert none['summary']['ci_method'] == 'none' and np.isnan(none['metrics']['auroc_ci_low']).all()


def test_xgboost_missing_falls_back_to_scikit_learn(monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == 'xgboost':
            raise ImportError('no xgboost')
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    _, name = dp._learner('xgboost', 0)
    assert name == 'gradient_boosting'


def test_bad_input_is_refused(tmp_path):
    feats = _features(True, movies=2)
    with pytest.raises(ValueError, match='unknown model'):
        dp.predict_divisions(feats, model='forest')
    with pytest.raises(ValueError, match='lack'):
        dp.predict_divisions(feats.drop(columns='movie'))
    with pytest.raises(ValueError, match='two movies'):
        dp.predict_divisions(feats.assign(movie='one'))
    with pytest.raises(ValueError, match='one class'):
        dp.predict_divisions(feats.assign(label=0))
    with pytest.raises(ValueError, match='folds'):
        dp.predict_divisions(feats, folds=1)
    with pytest.raises(ValueError):
        dp.division_prediction_features(pd.DataFrame({'t': [0]}), pd.DataFrame())
    with pytest.raises(ValueError):
        dp.division_prediction_features(pd.DataFrame(), pd.DataFrame(), mode='other')
    with pytest.raises(ValueError):
        dp.division_prediction_features(pd.DataFrame(), pd.DataFrame(), frames=0)
    with pytest.raises(FileNotFoundError):
        dp.read_division_prediction(tmp_path)
    empty = dp.division_prediction_features(
        pd.DataFrame(columns=['track_id', 't', 'area']),
        pd.DataFrame(columns=list(da.LINEAGE_COLUMNS)), mode='fission')
    assert not len(empty) and 'd_area' in empty
    fig = dp.division_prediction_figure({})
    assert fig.axes[0].texts


def test_metrics_match_scikit_learn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200) + 0.3 * y, 1)
    assert dp._auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert dp._auprc(y, s) == pytest.approx(average_precision_score(y, s))
    assert np.isnan(dp._auroc(np.zeros(5), s[:5])) and np.isnan(dp._auprc(np.zeros(5), s[:5]))
    assert np.isnan(dp._interval([np.nan])[0])


def test_options_from_settings():
    assert dp._prediction_options({}) == {'frames': 5, 'folds': 5, 'model': 'auto'}
    assert dp._prediction_options({'timelapse_division_prediction_frames': 8,
                                   'timelapse_division_prediction_folds': '',
                                   'timelapse_division_prediction_model': 'logistic'}) == {
        'frames': 8, 'folds': 5, 'model': 'logistic'}
    for bad in ({'timelapse_division_prediction_frames': 0},
                {'timelapse_division_prediction_folds': 1},
                {'timelapse_division_prediction_folds': True},
                {'timelapse_division_prediction_model': 'forest'}):
        with pytest.raises(ValueError):
            dp._prediction_options(bad)


def test_the_step_waits_for_three_fields_then_predicts(tmp_path, capsys):
    division = tmp_path / 'division'
    settings = {'timelapse_division_prediction': True, 'timelapse_division_prediction_folds': 3,
                'timelapse_division_prediction_model': 'gradient_boosting', 'save': False}
    for seed in range(4):
        m, lin = _fission_movie(seed + 100, True)
        out = division / f'trackpy_tracks_cell_f{seed}'
        out.mkdir(parents=True)
        results = {'summary': {'mode': 'fission'}}
        assert dp._write_field_features(m, lin, results, str(out), settings)
        if seed == 1:
            assert dp._run_prediction_step(str(division), 'trackpy_tracks_cell_', settings) is None
            assert 'waits for 3 fields' in capsys.readouterr().out
    (division / 'trackpy_tracks_cell_empty').mkdir()
    (division / 'trackpy_tracks_cell_empty' / 'prediction_features.csv').write_text('')
    predicted = dp._run_prediction_step(str(division), 'trackpy_tracks_cell_', settings)
    assert predicted is not None and predicted['summary']['n_movies'] == 4
    assert (division / 'prediction_trackpy_tracks_cell' / 'metrics.csv').is_file()
    assert 'Division prediction over 4 fields' in capsys.readouterr().out
    bad = dict(settings, timelapse_division_prediction_frames=0)
    assert dp._write_field_features(m, lin, {'summary': {'mode': 'fission'}}, str(out), bad) is None
    assert dp._run_prediction_step(str(division), 'trackpy_tracks_cell_', bad) is None
    assert 'could not' in capsys.readouterr().out
