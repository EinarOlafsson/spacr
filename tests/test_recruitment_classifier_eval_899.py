"""Held-out classifier evaluation with controls, and the annotation sample (item 899, alpha).

The reference is the BMC reanalysis script train_eval_recruitment.py: a
class-balanced histogram gradient boosting trained leave-one-well-out, a
majority-class control and a shuffled-label control (labels permuted within
each training fold, seeds 0..19, averaged). Its algorithm is restated in
``_reference_lowo`` and the spaCR evaluation must give the same predictions
and metrics. The GBP plate77 rows pinned in tests/data/recruitment_eval_899
come from the reference script itself on the same synthetic labels.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from spacr import ml

DATA = os.path.join(os.path.dirname(__file__), 'data', 'recruitment_eval_899')
FEATURES = ['f1', 'f2', 'f3']


def _objects(n_wells=4, per_well=30, seed=0):
    """Three classes separated along f1 and f2, spread over wells."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        for i in range(per_well):
            label = 1 + i % 3
            rows.append({'well': f'W{w}', 'plateID': f'p{w % 2}', 'label': label,
                         'f1': label + rng.normal(0, 0.4), 'f2': -label + rng.normal(0, 0.4),
                         'f3': rng.normal()})
    return pd.DataFrame(rows)


def _reference_lowo(df, shuffle_seed=None):
    """train_eval_recruitment.lowo, restated."""
    preds = pd.Series(index=df.index, dtype=float)
    rng = np.random.default_rng(shuffle_seed)
    for well in sorted(df['well'].unique()):
        tr, te = df[df['well'] != well], df[df['well'] == well]
        y = tr['label'].to_numpy()
        if shuffle_seed is not None:
            y = rng.permutation(y)
        m = HistGradientBoostingClassifier(max_iter=50, class_weight='balanced', random_state=0)
        m.fit(tr[FEATURES], y)
        preds[te.index] = m.predict(te[FEATURES])
    return preds.astype(int)


SETTINGS = {'ml_eval_split': 'leave_one_well_out', 'ml_eval_controls': 'majority, shuffled',
            'ml_eval_shuffle_seed': 0, 'ml_eval_shuffle_repeats': 2,
            'ml_model_type': 'hist_gbm', 'n_estimators': 50}


def test_model_and_controls_equal_the_reference_algorithm():
    df = _objects()
    out = ml.evaluate_classifier(df, FEATURES, SETTINGS)
    pooled = out[out['fold'] == 'pooled'].set_index('model')
    model = pooled.loc['hist_gbm, leave_one_well_out']
    reference = _reference_lowo(df)
    assert model['accuracy'] == pytest.approx((reference == df['label']).mean(), abs=0)
    assert model['accuracy'] > 0.8 and model['auroc'] > 0.9
    majority = pooled.loc['control: majority class (1)']
    assert majority['accuracy'] == pytest.approx(1 / 3) and majority['auroc'] == 0.5
    shuffled = pooled.loc['control: shuffled training labels (mean of 2)']
    expected = np.mean([(_reference_lowo(df, s) == df['label']).mean() for s in (0, 1)])
    assert shuffled['accuracy'] == pytest.approx(expected, abs=1e-12)
    assert shuffled['accuracy'] < model['accuracy']
    folds = out[(out['role'] == 'model') & (out['fold'] != 'pooled')]
    assert list(folds['fold']) == ['W0', 'W1', 'W2', 'W3'] and (folds['n'] == 30).all()
    assert list(out.columns[:3]) == ['role', 'model', 'fold']
    assert {'1 precision', '3 recall', '2 n', 'balanced_accuracy', 'f1_macro'} <= set(out.columns)


def test_plate_and_random_splits_and_their_folds():
    df = _objects()
    plates = ml.evaluation_folds(df, 'leave_one_plate_out')
    assert [f for f, _, _ in plates] == ['p0', 'p1']
    for _, train, test in plates:
        assert not set(df.loc[train, 'plateID']) & set(df.loc[test, 'plateID'])
    random = ml.evaluation_folds(df, 'random', seed=3, n_splits=4)
    assert len(random) == 4 and sorted(np.concatenate([t for *_, t in random])) == list(df.index)
    wells = ml.evaluation_folds(df.drop(columns='well').assign(rowID='r1', columnID=df['well']),
                                'leave_one_well_out')
    assert len(wells) == 4
    with pytest.raises(ValueError):
        ml.evaluation_folds(df, 'leave_one_field_out')
    with pytest.raises(ValueError):
        ml.evaluation_folds(df.drop(columns=['plateID']), 'leave_one_plate_out')
    with pytest.raises(ValueError):
        ml.evaluation_folds(df.drop(columns=['well']), 'leave_one_well_out')


def test_other_models_binary_auroc_and_bad_settings():
    df = _objects()
    binary = df[df['label'] < 3]
    for model in ('random_forest', 'logistic_regression'):
        out = ml.evaluate_classifier(binary, FEATURES, {**SETTINGS, 'ml_model_type': model,
                                                        'ml_eval_controls': '',
                                                        'n_estimators': 20})
        assert len(out) == 5 and out.loc[0, 'auroc'] > 0.9
    with pytest.raises(ValueError):
        ml.evaluate_classifier(df, FEATURES, {**SETTINGS, 'ml_model_type': 'svm'})
    with pytest.raises(ValueError):
        ml.evaluate_classifier(df, FEATURES, {**SETTINGS, 'ml_eval_controls': 'coin'})
    with pytest.raises(ValueError):
        ml.evaluate_classifier(df[df['label'] == 1], FEATURES, SETTINGS)
    one = ml.classifier_metrics([1, 1], [1, 2], [1, 2], 'x', y_score=[[1, 0], [0, 1]])
    assert np.isnan(one['auroc']) and one['accuracy'] == 0.5
    assert np.isnan(ml.classifier_metrics([1, 2, 3], [1, 2, 3], [1, 2, 3], 'x')['auroc'])
    assert ml._ml_eval_list(float('nan')) == [] and ml._ml_eval_list(['a', ' ']) == ['a']


def _project(tmp_path, labels, features):
    """A plate folder whose png_list holds ``labels`` and a features CSV."""
    src = tmp_path / 'plate'
    os.makedirs(src / 'measurements')
    with sqlite3.connect(src / 'measurements' / 'measurements.db') as con:
        labels.to_sql('png_list', con, index=False)
    path = tmp_path / 'features.csv'
    features.to_csv(path, index=False)
    return str(src), str(path)


def test_the_plate77_settings_run_reproduces_the_reference_script(tmp_path):
    selection = pd.read_csv(os.path.join(DATA, 'gbp_plate77_features.csv'))
    labels = selection[['file_name', 'annotate']].assign(
        png_path=lambda d: '/crops/' + d['file_name'], rowID='r1', columnID='c1')
    src, features = _project(tmp_path, labels, selection.drop(columns=['annotate']))
    settings = {'src': src, 'ml_eval_report': True, 'ml_eval_features_table': features,
                'ml_eval_classes': '1,2,3', 'ml_eval_controls': 'majority',
                'ml_eval_features': ('rim_enrichment, ring_median_bgsub, ref_cyto_median_bgsub, '
                                     'area_px, tox_mean, ring_sat_frac'),
                'n_estimators': 200, 'ml_model_type': 'hist_gbm', 'save': False}
    result = ml.run_ml_evaluation(settings)
    assert os.path.isfile(result['path']) and result['n'] == 500
    pooled = result['metrics'][result['metrics']['fold'] == 'pooled'].reset_index(drop=True)
    reference = pd.read_csv(os.path.join(DATA, 'reference_classifier_metrics.csv'))
    names = {1: 'recruitment', 2: 'no recruitment', 3: 'artefact'}
    for ours, ref in ((0, 1), (1, 2)):
        assert pooled.loc[ours, 'accuracy'] == pytest.approx(reference.loc[ref, 'accuracy'], abs=1e-15)
        for value, name in names.items():
            for kind in ('precision', 'recall', 'n'):
                assert pooled.loc[ours, f'{value} {kind}'] == pytest.approx(
                    reference.loc[ref, f'{name} {kind}'], abs=1e-15)


def test_generate_ml_scores_dispatches_the_evaluation_and_the_sample(tmp_path, monkeypatch):
    df = _objects()
    labels = pd.DataFrame({'file_name': [f'o{i}.png' for i in range(len(df))],
                           'png_path': [f'/c/o{i}.png' for i in range(len(df))],
                           'annotate': df['label'].where(df.index % 10 != 0)})
    src, features = _project(tmp_path, labels,
                             df.assign(file_name=labels['file_name']).drop(columns='label'))
    monkeypatch.setattr('spacr.utils.save_settings', lambda *a, **k: None)
    result = ml.generate_ml_scores({'src': src, 'ml_eval_report': True,
                                    'ml_eval_features_table': features,
                                    'ml_eval_features': 'f1,f2,f3', 'ml_eval_controls': '',
                                    'n_estimators': 20})
    assert result['n'] == int(labels['annotate'].notna().sum())
    sample = ml.generate_ml_scores({'src': src, 'annotate_sample_n': 12,
                                    'annotate_sample_strata': 'well, f1:3',
                                    'ml_eval_features_table': features,
                                    'ml_label_column': 'fresh_labels'})
    assert len(sample) == 12 and sample['annotate_sample'].tolist() == list(range(1, 13))
    with sqlite3.connect(os.path.join(src, 'measurements', 'measurements.db')) as con:
        marked = pd.read_sql('SELECT png_path, annotate_sample, fresh_labels FROM png_list', con)
    assert marked['annotate_sample'].notna().sum() == 12 and marked['fresh_labels'].isna().all()
    with pytest.raises(ValueError):
        ml.run_ml_evaluation({'src': src, 'ml_eval_features_table': features,
                              'ml_eval_features': 'nope'})
    with pytest.raises(ValueError):
        ml.run_ml_evaluation({'src': src, 'ml_eval_features_table': features,
                              'ml_label_column': 'missing'})


def test_annotation_sample_is_stratified_topped_up_and_seeded():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'strain': ['a'] * 50 + ['b'] * 50 + ['c'] * 5,
                       'score': rng.normal(size=105)})
    sample = ml.annotation_sample(df, 30, seed=0, strata='strain, score:5')
    # 15 strata x 2 = 25, the top-up adds 2 from a and 2 from b: one short of 30, as the reference
    assert len(sample) == 29 and (sample['strain'] == 'c').sum() == 5
    assert sample.index.is_unique
    again = ml.annotation_sample(df, 30, seed=0, strata='strain, score:5')
    pd.testing.assert_frame_equal(sample, again)
    plain = ml.annotation_sample(df, 10, seed=1)
    assert len(plain) == 10 and set(plain['stratum']) == {'all'}
    with pytest.raises(ValueError):
        ml.annotation_sample(df, 10, strata='missing')


def test_evaluation_after_a_training_run_needs_annotation_labels(capsys):
    assert ml._evaluate_after_training({}, None, None) is None
    assert 'needs annotation labels' in capsys.readouterr().out


def test_evaluation_settings_are_registered_alpha():
    from spacr.settings import ALPHA_FEATURES, expected_types, set_default_analyze_screen, tooltips

    defaults = set_default_analyze_screen({})
    names = [k for k in ALPHA_FEATURES[772]['settings'] if k.startswith(('ml_', 'annotate_sample'))]
    assert len(names) == 15
    assert defaults['ml_eval_report'] is False and defaults['annotate_sample_n'] == 0
    assert defaults['ml_eval_split'] == 'leave_one_well_out'
    for key in names:
        assert key in expected_types and tooltips[key].endswith(f'Default {defaults[key]}.')


def test_evaluation_after_training_and_default_features(tmp_path):
    df = _objects(n_wells=3)
    frame = df.rename(columns={'label': 'annotate'})
    output = [frame] + [None] * 8 + [['f1', 'f2', 'absent']]
    result = ml._evaluate_after_training({'src': str(tmp_path), 'ml_eval_controls': '',
                                          'n_estimators': 20}, output, 'annotate')
    assert result['n'] == 90 and os.path.isfile(result['path'])
    everything = ml.run_ml_evaluation({'src': str(tmp_path), 'ml_eval_controls': '',
                                       'n_estimators': 20}, df=frame.drop(columns='plateID'))
    assert everything['metrics'].loc[0, 'accuracy'] > 0.7
    two = ml.classifier_metrics([1, 2, 1], [1, 2, 2], [1, 2, 3], 'x',
                                y_score=[[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
    assert np.isnan(two['auroc']) and two['3 n'] == 0
