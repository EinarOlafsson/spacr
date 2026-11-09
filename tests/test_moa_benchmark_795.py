"""Item 795: the leave-one-compound-out mechanism-of-action benchmark."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import moa_benchmark as mb
from spacr.embeddings import evaluate_moa
from spacr.tabular import read_table, write_table

MOAS = ('actin', 'dna', 'tubulin', 'kinase')


def _wells(seed=0, noise=0.6, compounds_per_moa=3, concentrations=2, replicates=3,
           batches=('Week1', 'Week2', 'Week3'), dmso=True):
    """Per-well profiles: each MoA a direction, each compound a shifted copy, three weeks."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(MOAS), 12)) * 3
    rows = []
    for m, moa in enumerate(MOAS):
        for c in range(compounds_per_moa):
            compound = f'{moa}_{c}'
            shift = rng.normal(size=12) * 0.3
            for k in range(concentrations):
                for r in range(replicates):
                    batch = batches[(c + k + (r == 2 and m == 0)) % len(batches)]
                    values = centres[m] * (1 + 0.5 * k) + shift + rng.normal(size=12) * noise
                    rows.append(dict(compound=compound, concentration=float(k + 1),
                                     moa=moa, batch=batch, plate=f'{batch}_p',
                                     **{f'f{i}': v for i, v in enumerate(values)}))
    if dmso:
        for r in range(6):
            rows.append(dict(compound='DMSO', concentration=0.0, moa='DMSO',
                             batch=batches[r % len(batches)], plate='x',
                             **{f'f{i}': v for i, v in enumerate(rng.normal(size=12))}))
    return pd.DataFrame(rows)


def _reference_knn(profiles, features, batch_column='batch'):
    """The run_and_score.py rule, written out: first maximum among the allowed candidates."""
    X = profiles[features].to_numpy(float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    comp = profiles['compound'].to_numpy()
    y = profiles['moa'].to_numpy()
    sets = [set(str(b).split('|')) for b in profiles[batch_column]]
    out = {}
    for rule in ('NSC', 'NSCB'):
        correct = n = 0
        for i in range(len(X)):
            ok = comp != comp[i]
            if rule == 'NSCB':
                ok &= np.array([not (sets[i] & b) for b in sets])
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            j = idx[np.argmax(S[i, idx])]
            correct += y[j] == y[i]
            n += 1
        out[rule] = (int(correct), n)
    return out


def test_the_1nn_arm_is_evaluate_moa_and_the_ljosa_rule():
    wells = _wells(noise=2.5)
    features = [f'f{i}' for i in range(12)]
    result = mb.moa_benchmark(wells, arms=('1-NN',), batch_column='batch', n_permutations=5)
    profiles = result['profiles']
    assert len(profiles) == 24 and set(profiles['n_wells']) == {3}
    assert profiles['batch'].str.contains('|', regex=False).any()
    table = result['table']
    expected = _reference_knn(profiles, features)
    direct = evaluate_moa(profiles, batch_column='batch', features=features, n_permutations=0)
    for rule in ('NSC', 'NSCB'):
        row = table[(table['arm'] == '1-NN cosine') & (table['evaluation'] == rule)].iloc[0]
        assert (int(row['correct']), int(row['n_treatments'])) == expected[rule]
        same = direct['table'][(direct['table']['profiles'] == 'profiles')
                               & (direct['table']['evaluation'] == rule)].iloc[0]
        assert row['accuracy'] == same['accuracy'] and row['ci_low'] == same['ci_low']
        assert row['n_solvable'] == same['n_solvable']
    assert list(table.columns) == list(mb._TABLE_COLUMNS)
    assert table.attrs['glossary'] == mb.GLOSSARY == result['glossary']
    assert set(mb.GLOSSARY) >= {'MoA', '1-NN', 'NSC', 'NSCB', 'LOCO', 'CI'} | set(table.columns)
    assert result['features'] == features


def test_mean_replicates_and_treatment_level_input():
    wells = _wells()
    mean = mb.moa_benchmark(wells, arms=('1-NN',), replicates='mean', n_permutations=0)
    profiles = mean['profiles']
    one = wells[(wells['compound'] == 'actin_0') & (wells['concentration'] == 1.0)]
    assert np.isclose(profiles.loc[(profiles['compound'] == 'actin_0')
                                   & (profiles['concentration'] == 1.0), 'f0'].iloc[0],
                      one['f0'].mean())
    again = mb.moa_benchmark(profiles, arms=('1-NN',), n_permutations=0)
    assert again['profiles'] is not profiles and len(again['profiles']) == len(profiles)
    assert again['table'].iloc[0]['accuracy'] == mean['table'].iloc[0]['accuracy']
    assert set(mean['table']['evaluation']) == {'NSC'}


def test_the_shuffled_control_is_at_chance_and_matches_the_reference_permutation():
    wells = _wells(noise=0.3)
    result = mb.moa_benchmark(wells, arms=('1-NN',), batch_column='batch', n_permutations=200)
    table = result['table'].set_index(['arm', 'evaluation'])
    real = table.loc[('1-NN cosine', 'NSC'), 'accuracy']
    shuffled = table.loc[('control: shuffled MoA labels (1-NN cosine)', 'NSC')]
    assert real == 1.0
    assert abs(shuffled['accuracy'] - 0.2) < 0.08
    assert shuffled['n_permutations'] == 200 and shuffled['sd'] > 0
    assert shuffled['ci_low'] <= shuffled['accuracy'] <= shuffled['ci_high']
    profiles = result['profiles']
    rng = np.random.default_rng(0)
    cmoa = profiles.groupby('compound')['moa'].first()
    shares = []
    for _ in range(200):
        mapping = dict(zip(cmoa.index, rng.permutation(cmoa.values)))
        relabelled = profiles.assign(moa=profiles['compound'].map(mapping))
        correct, n = _reference_knn(relabelled, result['features'])['NSC']
        shares.append(correct / n)
    assert np.isclose(shuffled['accuracy'], np.mean(shares))
    assert np.isclose(shuffled['sd'], np.std(shares))
    majority = [arm for arm, _ in table.index if arm.startswith('control: majority')]
    assert len(majority) == 2 and majority[0][len('control: majority class ('):-1] in MOAS
    assert table.loc[(majority[0], 'NSC'), 'accuracy'] == 0.25


def test_logistic_learns_the_moa_and_its_shuffled_control_does_not():
    wells = _wells(noise=0.5)
    result = mb.moa_benchmark(wells, arms=('logistic',), batch_column='batch',
                              supervised_permutations=3, workers=2)
    table = result['table'].set_index(['arm', 'evaluation'])
    nsc = table.loc[('logistic regression', 'NSC')]
    assert nsc['accuracy'] == 1.0 and nsc['n_treatments'] == 24 and nsc['n_wells'] == 72
    assert nsc['well_accuracy'] == 1.0 and nsc['accuracy_solvable'] == 1.0
    assert table.loc[('control: shuffled MoA labels (logistic regression)', 'NSC'),
                     'accuracy'] < 0.6
    predictions = result['predictions']
    assert set(predictions['evaluation']) == {'NSC', 'NSCB'}
    assert predictions['probability'].dropna().between(0, 1).all()


def test_nscb_leaves_out_the_held_out_treatments_batches():
    wells = _wells(batches=('Week1',), dmso=False)
    wells.loc[wells['compound'] == 'actin_0', 'batch'] = 'Week9'
    out = mb.loco_supervised(wells, batch_column='batch', model='logistic')
    predictions = out['predictions']
    nscb = predictions[predictions['evaluation'] == 'NSCB']
    assert nscb[nscb['compound'] == 'actin_0']['scored'].all()
    assert not nscb[nscb['compound'] != 'actin_0']['scored'].any()
    row = out['table'].set_index('evaluation').loc['NSCB']
    assert row['n_treatments'] == 2 and row['n_solvable'] == 2
    assert row['accuracy_solvable'] == row['accuracy']
    majority = mb.majority_class_control(predictions)
    assert majority.set_index('evaluation').loc['NSCB', 'n_treatments'] == 2


def test_a_fold_with_one_training_class_is_not_scored():
    wells = _wells(dmso=False)
    wells = wells[wells['moa'].isin(['actin', 'dna'])]
    wells = wells[~wells['compound'].isin(['dna_1', 'dna_2'])]
    out = mb.loco_supervised(wells, model='logistic')
    predictions = out['predictions']
    held = predictions[predictions['compound'] == 'dna_0']
    assert not held['scored'].any() and held['predicted_moa'].isna().all()
    assert not held['solvable'].any()
    row = out['table'].iloc[0]
    assert row['n_treatments'] == 6 and row['n_wells'] == 18


def test_xgboost_runs_on_the_cpu(monkeypatch):
    wells = _wells(noise=0.4, compounds_per_moa=2, concentrations=1, replicates=2)
    monkeypatch.setitem(mb.XGBOOST_PARAMS, 'n_estimators', 20)
    out = mb.loco_supervised(wells, model='XGBoost', batch_column='batch', seed=3)
    assert out['table'].set_index('evaluation').loc['NSC', 'accuracy'] >= 0.5
    estimator = mb._make_model('xgboost', 1)
    assert estimator.get_params()['device'] == 'cpu'


def test_controls_with_a_callable_arm_and_row_permutations():
    wells = _wells(noise=0.3)
    profiles = mb.moa_benchmark(wells, arms=('1-NN',), n_permutations=0)['profiles']
    control = mb.shuffled_label_control(mb.loco_nearest_neighbour, profiles,
                                        n_permutations=4, seed=1)
    assert control['arm'].iloc[0] == 'control: shuffled MoA labels (loco_nearest_neighbour)'
    mixed = profiles.copy()
    mixed.loc[0, 'moa'] = 'dna'
    rows = mb.shuffled_label_control('1nn', mixed, n_permutations=3)
    assert rows['n_permutations'].iloc[0] == 3
    assert mb.shuffled_label_control('1-NN', profiles, n_permutations=0).empty
    with pytest.raises(ValueError, match='no .moa. column'):
        mb.shuffled_label_control('1-NN', profiles.drop(columns='moa'))
    with pytest.raises(ValueError, match='arm must be one of'):
        mb.shuffled_label_control('svm', profiles)
    with pytest.raises(ValueError, match='columns not in the predictions'):
        mb.majority_class_control(pd.DataFrame({'moa': ['a']}))
    empty = pd.DataFrame({'moa': ['a'], 'evaluation': ['NSC'], 'scored': [False]})
    assert mb.majority_class_control(empty)['n_treatments'].iloc[0] == 0


def test_labels_are_joined_and_inputs_are_checked(tmp_path):
    wells = _wells()
    labels = wells[['compound', 'concentration', 'moa']].drop_duplicates()
    path = write_table(labels, tmp_path / 'labels.csv', canonicalise=False)
    joined = mb.loco_nearest_neighbour(wells.drop(columns='moa'), labels=path)
    assert joined['table'].iloc[0]['n_treatments'] == 72
    with pytest.raises(ValueError, match='no .moa. column'):
        mb.loco_nearest_neighbour(wells.drop(columns='moa'), labels=labels.drop(columns='moa'))
    with pytest.raises(ValueError, match='share no column'):
        mb.loco_nearest_neighbour(wells.drop(columns='moa'),
                                  labels=pd.DataFrame({'x': [1], 'moa': ['a']}))
    with pytest.raises(ValueError, match='columns not in the profiles'):
        mb.loco_nearest_neighbour(wells, batch_column='week')
    with pytest.raises(ValueError, match='fewer than two'):
        mb.loco_nearest_neighbour(wells[wells['compound'].isin(['actin_0', 'DMSO'])])
    broken = wells.copy()
    broken.loc[3, 'f2'] = np.nan
    with pytest.raises(ValueError, match='1 missing or infinite'):
        mb.loco_supervised(broken)
    with pytest.raises(ValueError, match='feature columns not in the table'):
        mb.loco_nearest_neighbour(wells, features=['f0', 'nope'])
    with pytest.raises(ValueError, match='no numeric feature columns'):
        mb.loco_nearest_neighbour(wells[['compound', 'concentration', 'moa']])
    with pytest.raises(ValueError, match='model must be one of'):
        mb.loco_supervised(wells, model='svm')
    with pytest.raises(ValueError, match='replicates must be one of'):
        mb.moa_benchmark(wells, replicates='mode')
    with pytest.raises(ValueError, match='at least one arm'):
        mb.moa_benchmark(wells, arms='')
    with pytest.raises(ValueError, match='confidence'):
        mb.loco_supervised(wells, confidence=1.5)
    only_compound = mb.loco_nearest_neighbour(wells, treatment_columns=('concentration',))
    assert only_compound['table'].iloc[0]['n_treatments'] == 72
    assert mb._names(None) == [] and mb._names(['a', 2]) == ['a', '2']


def test_write_run_and_command_line(tmp_path, caplog):
    wells = _wells()
    path = write_table(wells, tmp_path / 'wells.csv', canonicalise=False)
    settings = mb.moa_benchmark_defaults({'moa_profiles': str(path), 'moa_arms': '1-NN',
                                          'moa_batch_column': 'batch', 'moa_shuffles': 3})
    assert settings['moa_replicates'] == 'median' and settings['moa_arms'] == '1-NN'
    caplog.set_level('INFO', logger='spacr.moa_benchmark')
    result = mb.run_moa_benchmark(settings)
    folder = tmp_path / 'moa_benchmark'
    assert sorted(p.name for p in folder.iterdir()) == [
        'moa_accuracy.csv', 'moa_glossary.csv', 'moa_predictions.csv', 'moa_summary.json']
    assert len(result['paths']) == 4
    table = read_table(folder / 'moa_accuracy.csv', canonicalise=False)
    assert list(table['arm'])[:2] == ['1-NN cosine', '1-NN cosine']
    glossary = read_table(folder / 'moa_glossary.csv', canonicalise=False)
    assert dict(zip(glossary['term'], glossary['meaning'])) == mb.GLOSSARY
    summary = json.loads((folder / 'moa_summary.json').read_text(encoding='utf-8'))
    assert summary['n_features'] == 12 and len(summary['table']) == len(result['table'])
    assert '1-NN cosine NSC' in caplog.text
    framed = mb.run_moa_benchmark({'moa_profiles': wells, 'moa_arms': '1-NN',
                                   'moa_shuffles': 0, 'moa_dst': str(tmp_path / 'f')})
    assert (tmp_path / 'f' / 'moa_accuracy.csv').is_file() and len(framed['table']) == 2
    with pytest.raises(ValueError, match='moa_profiles is blank'):
        mb.run_moa_benchmark({})
    code = mb.main(['--profiles', str(path), '--arms', 'logistic', '--supervised-shuffles', '0',
                    '--dst', str(tmp_path / 'cli'), '--controls', 'DMSO'])
    assert code == 0 and (tmp_path / 'cli' / 'moa_predictions.csv').is_file()


def test_the_settings_are_registered_once_and_alpha():
    from spacr import settings as S

    assert S.has_registered_defaults(mb.APP_KEY)
    mb._register()
    defaults = S.defaults_for(mb.APP_KEY, {})
    assert set(defaults) == set(mb._EXPECTED_TYPES) == set(mb._TOOLTIPS) == set(mb._DEFAULTS)
    for key, kind in mb._EXPECTED_TYPES.items():
        assert S.expected_types[key] is kind and isinstance(defaults[key], kind)
        assert S.tooltips[key].startswith(f'({kind.__name__}) - ') and len(S.tooltips[key]) <= 600
    categorised = {key for keys in mb._CATEGORIES.values() for key in keys}
    assert categorised == set(defaults)
    for name, keys in mb._CATEGORIES.items():
        assert set(keys) <= set(S.categories[name])
    assert set(S.ALPHA_FEATURES[795]['module_settings'][mb.APP_KEY]) == set(defaults)
    assert set(defaults) <= S._alpha_names('settings', app_key=mb.APP_KEY)


def test_float32_features_stay_float32_and_integers_are_cast():
    wells = _wells(dmso=False)
    features = [f'f{i}' for i in range(12)]
    single = wells.astype({column: 'float32' for column in features})
    whole = wells.assign(**{column: (wells[column] * 10).round().astype(int) for column in features})
    assert mb.loco_supervised(single)['table'].iloc[0]['accuracy'] >= 0.9
    assert mb.loco_supervised(whole)['table'].iloc[0]['accuracy'] >= 0.9
