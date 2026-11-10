"""Item 897: well-set training of the well transformer, and the cell-level path with pre-training.

Tiny CPU data only: a synthetic BBBC021-like screen (two weeks, three MoAs
plus DMSO, per-plate embeddings and cell tables on disk) for the well-set
path, and random crops whose class is carried by one stain for the cell path.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from spacr import architecture_training as at
from spacr import well_set_training as wst
from spacr.tabular import read_table, write_table


def _screen(tmp_path, D=12, seed=0):
    """Two plates per week, two weeks; three MoAs of two compounds each plus DMSO wells."""
    rng = np.random.default_rng(seed)
    moas = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'b2': 'B', 'c1': 'C', 'c2': 'C', 'taxol': 'A'}
    dirs = {m: rng.normal(size=D) * 2 for m in 'ABC'}
    for plate in ('Week1_1', 'Week1_2', 'Week2_1', 'Week2_2'):
        rows, X = [], []
        wells = [c for c in moas if c != 'taxol' or plate.startswith('Week1')] + ['DMSO', 'DMSO']
        for i, comp in enumerate(wells):
            moa = 'DMSO' if comp == 'DMSO' else moas[comp]
            n = int(rng.integers(8, 14))
            conc = 1.0 if comp == 'taxol' else 0.5
            x = rng.normal(size=(n, D)) + (0 if moa == 'DMSO' else dirs[moa]) + rng.normal(size=D) * 0.2
            X.append(x)
            rows += [{'plate': plate, 'well': f'w{i:02d}', 'compound': comp, 'concentration': conc, 'moa': moa}] * n
        np.save(tmp_path / f'{plate}.npy', np.concatenate(X).astype(np.float32))
        write_table(pd.DataFrame(rows), tmp_path / f'{plate}.cells.parquet', canonicalise=False)
    return tmp_path


def test_normalisation_centres_each_plate_on_its_controls_and_whitens(tmp_path):
    _screen(tmp_path)
    t, Z = wst.load_cell_embeddings(tmp_path, method='plate_control', control='DMSO')
    assert Z.dtype == np.float32 and len(t) == len(Z)
    for plate in t['plate'].unique():
        d = Z[((t['plate'] == plate) & (t['moa'] == 'DMSO')).to_numpy()]
        np.testing.assert_allclose(d.mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(d.std(0), 1, atol=1e-4)
    t2, Zp = wst.load_cell_embeddings(tmp_path, pca_dims=5)
    assert Zp.dtype == np.float16 and Zp.shape == (len(t2), 5)
    dm = Zp[(t2['moa'] == 'DMSO').to_numpy()].astype(np.float64)
    np.testing.assert_allclose(dm.std(0, ddof=1), 1, atol=0.02)
    assert set(t2['week']) == {'Week1', 'Week2'}
    _, Zn = wst.load_cell_embeddings(tmp_path, method='none')
    assert Zn.shape[1] == 12
    with pytest.raises(ValueError, match='train_set_normalisation'):
        wst.normalise_embeddings([], [], method='zca')
    with pytest.raises(ValueError, match='no'):
        wst.normalise_embeddings([np.zeros((2, 3))], [pd.DataFrame({'moa': ['x', 'y']})])
    with pytest.raises(ValueError, match='table rows'):
        wst.normalise_embeddings([np.zeros((2, 3))], [pd.DataFrame({'moa': ['x']})])
    with pytest.raises(FileNotFoundError):
        wst.load_cell_embeddings(tmp_path / 'nothing')


def test_folds_follow_the_reference_rules(tmp_path):
    _screen(tmp_path)
    t, _ = wst.load_cell_embeddings(tmp_path, method='none')
    W = wst.group_wells(t)
    assert W['n'].sum() == len(t) and all(len(r) == n for r, n in zip(W['rows'], W['n']))
    lab = W[W['moa'] != 'DMSO'].reset_index(drop=True)
    folds = wst.moa_folds(lab, 'NSCB')
    assert [f[0] for f in folds] == ['Week1', 'Week2', 'taxol_1_3uM']
    for name, tr, te in folds[:2]:
        assert not (lab['week'][tr] == name).any() and (lab['week'][te] == name).all()
        assert not (lab['compound'][te] == 'taxol').any()
    _, tr, te = folds[2]
    assert (lab['compound'][te] == 'taxol').all()
    assert set(lab['week'][tr]) == {'Week2'} and not (lab['compound'][tr] == 'taxol').any()
    nsc = wst.moa_folds(lab, 'NSC')
    assert len(nsc) == 4
    for _, tr, te in nsc:
        assert not set(lab['compound'][tr]) & set(lab['compound'][te])
    with pytest.raises(ValueError, match='train_protocol'):
        wst.moa_folds(lab, 'LOO')


def test_scoring_zeroes_untrained_classes_and_bootstraps():
    P = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    Z = wst.zero_untrained(P, ['A', 'B', 'C'], {'A', 'C'})
    np.testing.assert_allclose(Z, [[0.4, 0, 0.6], [0.6 / 0.9, 0, 0.3 / 0.9]])
    wells = pd.DataFrame({'plate': ['p'] * 4, 'well': list('wxyz'), 'compound': ['a', 'a', 'b', 'c'],
                          'concentration': [1.0] * 4, 'moa': ['A', 'A', 'B', 'C'], 'fold': ['f'] * 4})
    P = np.array([[0.9, 0.1, 0], [0.8, 0.2, 0], [0.1, 0.9, 0], [0.7, 0.3, 0.0]])
    t, res = wst.score_treatments(wells, P, ['A', 'B', 'C'])
    assert len(t) == 3 and res['accuracy'] == pytest.approx(2 / 3)
    assert res['acc_ci'][0] <= res['accuracy'] <= res['acc_ci'][1]
    assert res['majority_class_accuracy'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('arm', ['settf', 'abmil', 'n1', 'mixbag', 'mean_lr', 'mean_knn'])
def test_every_arm_runs_on_the_tiny_screen(tmp_path, arm):
    _screen(tmp_path)
    t, Z = wst.load_cell_embeddings(tmp_path, method='plate_control')
    cfg = {'arm': arm, 'design': 'NSCB', 'steps': 30, 'bag_batch': 6, 'bag_cells': [4, 8], 'dim': 16, 'depth': 1,
           'lr': 3e-3, 'threads': 1}
    out = tmp_path / 'out'
    res = wst.run_well_set_arm(cfg, t, Z, out)
    assert res['arm'] == arm and res['n_treatments'] == 13
    assert 0 <= res['accuracy'] <= 1 and res['acc_ci'][0] <= res['acc_ci'][1]
    assert (out / 'treatment_probs.csv').is_file()
    if arm in ('settf', 'abmil', 'mixbag'):
        pen = read_table(out / 'well_penetrance.csv', canonicalise=False, report=None)
        assert pen['penetrant_fraction'].between(0, 1).all() and 0 <= res['median_penetrant_fraction'] <= 1
    if arm in wst.TRANSFORMER_ARMS:
        assert res['params'] > 0
    if arm in ('mean_knn', 'settf'):
        assert res['accuracy'] > 1 / 3


def test_an_arm_is_identical_when_rerun_with_the_same_seed(tmp_path):
    _screen(tmp_path)
    t, Z = wst.load_cell_embeddings(tmp_path, method='plate_control')
    cfg = {'arm': 'settf', 'design': 'NSCB', 'steps': 8, 'bag_batch': 4, 'bag_cells': [4, 8], 'dim': 16,
           'depth': 1, 'threads': 1, 'folds': [0, 2]}
    a, b = wst.run_well_set_arm(cfg, t, Z), wst.run_well_set_arm(cfg, t, Z)
    assert a == b
    assert wst.run_well_set_arm(dict(cfg, seed=3), t, Z)['seed'] == 3
    with pytest.raises(ValueError, match='arm'):
        wst.run_well_set_arm(dict(cfg, arm='gbm'), t, Z)


def test_settings_run_writes_summary_with_arms_and_controls(tmp_path):
    (tmp_path / 'emb').mkdir()
    _screen(tmp_path / 'emb')
    dst = tmp_path / 'res'
    res = at.run_spacr_architectures({
        'train_set_input': str(tmp_path / 'emb'), 'train_set_normalisation': 'plate_control',
        'train_set_arms': 'settf,mean_knn', 'train_set_steps': 6, 'train_set_bag_batch': 4,
        'train_set_bag_cells': (4, 8), 'train_set_dim': 16, 'train_set_depth': 1, 'train_set_threads': 1,
        'train_set_shuffle_control': True, 'train_set_dst': str(dst), 'train_set_folds': None})
    s = res['summary']
    assert list(s['run']) == ['NSCB_resnet18_settf', 'NSCB_resnet18_mean_knn', 'NSCB_resnet18_settf_shuffled',
                              'NSCB_resnet18_mean_knn_shuffled', 'NSCB_resnet18_majority_class']
    assert s['shuffled'].tolist() == [False, False, True, True, False]
    assert s.iloc[-1]['accuracy'] == s.iloc[0]['majority_class_accuracy']
    disk = read_table(dst / 'summary.csv', canonicalise=False, report=None)
    assert list(disk.columns) == list(wst.SUMMARY_COLUMNS) and len(disk) == 5
    cfg = json.loads((dst / 'NSCB_resnet18_settf' / 'config.json').read_text())
    assert cfg['bag_cells'] == [4, 8] and cfg['steps'] == 6 and cfg['folds'] is None
    assert 'NSCB' in json.loads((dst / 'glossary.json').read_text())
    with pytest.raises(ValueError, match='train_set_input'):
        wst.run_well_set_training({})
    with pytest.raises(ValueError, match='not among'):
        wst.run_well_set_training({'train_set_input': str(tmp_path), 'train_set_arms': 'gbm'})
    with pytest.raises(ValueError, match='well_set'):
        wst.run_well_set_training({'train_level': 'cell'})


def test_synthetic_arrayed_and_pooled_learn_the_planted_signal():
    small = wst.synthetic_arrayed(0, n_wells=40, D=16, cells=(20, 40), shift=4.0)
    res = wst.run_synthetic_arrayed({'arm': 'settf', 'steps': 150, 'bag_batch': 8, 'bag_cells': [8, 16], 'dim': 16,
                                     'depth': 1, 'lr': 3e-3}, small)
    assert res['well_accuracy'] > 0.5
    assert -1 <= res['penetrance_pearson_r'] <= 1 and res['penetrance_mae'] >= 0
    assert wst.run_synthetic_arrayed({'arm': 'n1', 'steps': 5, 'bag_batch': 4, 'dim': 16, 'depth': 1},
                                     small)['well_accuracy'] >= 0
    for arm in ('ctx', 'noctx', 'ctxshuf'):
        r = wst.run_synthetic_pooled({'arm': arm, 'steps': 4, 'well_batch': 2, 'targets': 8, 'context': 16,
                                      'dim': 16, 'depth': 1})
        assert r['arm'] == arm and 0 <= r['cell_accuracy'] <= 1 and r['chance'] == pytest.approx(1 / 3)


def _cells(tmp_path, n=48, C=3, size=32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    x = rng.gamma(2.0, 50.0, size=(n, C, size, size)).astype(np.float32)
    x[y == 1, 1] *= 3
    np.save(tmp_path / 'crops.npy', x)
    write_table(pd.DataFrame({'label': np.where(y == 1, 'hit', 'ctrl'), 'well': [f'w{i % 8}' for i in range(n)]}),
                tmp_path / 'cells.csv', canonicalise=False)
    return {'train_level': 'cell', 'train_cell_crops': str(tmp_path / 'crops.npy'),
            'train_cell_table': str(tmp_path / 'cells.csv'), 'train_cell_stains': 'dna,tubulin,actin',
            'train_cell_size': 32, 'train_cell_epochs': 1, 'train_cell_batch': 8, 'train_set_threads': 1,
            'train_set_dst': str(tmp_path / 'out'), 'train_cell_test_fraction': 0.25}


def test_cell_cnn_with_missing_stain_subsets_and_controls(tmp_path):
    s = dict(_cells(tmp_path), train_arch='spacr_channel_cnn', eval_channel_subsets='dna,tubulin;dna',
             train_set_shuffle_control=True)
    res = at.run_spacr_architectures(s)
    t = res['summary']
    assert t['channels'].tolist() == ['dna,tubulin,actin', 'dna,tubulin', 'dna'] * 2 + ['dna,tubulin,actin']
    assert t['arm'].tolist()[-1] == 'majority_class' and t['shuffled'].tolist()[3] is True
    assert t['accuracy'].between(0, 1).all() and (t['n_test'] == 12).all()
    assert (tmp_path / 'out' / 'model.pt').is_file()
    cfg = json.loads((tmp_path / 'out' / 'config.json').read_text())
    assert cfg['stains'] == ['dna', 'tubulin', 'actin'] and len(cfg['scale']) == 3


@pytest.mark.parametrize('method', ['mae', 'dino'])
def test_cell_vit_pretraining_then_few_shot(tmp_path, method):
    s = dict(_cells(tmp_path), train_arch='spacr_channel_vit', train_cell_vit_size='mini', train_cell_patch=16,
             pretrain_method=method, pretrain_epochs=1, train_labels_per_class=3)
    res = at.run_spacr_architectures(s)
    row = res['summary'].iloc[0]
    assert row['pretrain'] == method and row['n_train'] == 6 and row['arm'].endswith('_3pc')
    losses = json.loads((tmp_path / 'out' / 'losses.json').read_text())
    assert len(losses['pretrain']) == 5 and all(np.isfinite(losses['pretrain']))


def test_cell_level_refuses_bad_settings(tmp_path):
    s = _cells(tmp_path)
    with pytest.raises(ValueError, match='train_level'):
        at.run_spacr_architectures({'train_level': 'pixel'})
    with pytest.raises(ValueError, match='train_arch'):
        at.run_spacr_architectures({'train_arch': 'resnet18'})
    with pytest.raises(ValueError, match="'cell' trains"):
        at.run_cell_architecture(dict(s, train_arch='spacr_well_transformer'))
    with pytest.raises(ValueError, match='built for spacr_channel_vit'):
        at.run_cell_architecture(dict(s, train_arch='spacr_channel_cnn', pretrain_method='mae'))
    with pytest.raises(ValueError, match='pretrain_method'):
        at.run_cell_architecture(dict(s, train_arch='spacr_channel_vit', pretrain_method='simclr'))
    with pytest.raises(ValueError, match='name the crops'):
        at.run_cell_architecture({'train_arch': 'spacr_channel_cnn'})
    with pytest.raises(ValueError, match='not among the stains'):
        at.run_cell_architecture(dict(s, train_arch='spacr_channel_cnn', eval_channel_subsets='golgi'))
    np.save(tmp_path / 'crops.npy', np.zeros((3, 3, 32, 32), np.float32))
    with pytest.raises(ValueError, match='table rows'):
        at.run_cell_architecture(dict(s, train_arch='spacr_channel_cnn'))
    with pytest.raises(ValueError, match='two groups'):
        at.split_by_group(['a', 'a'], 0.5, 0)
    with pytest.raises(ValueError, match="'mae' or 'dino'"):
        at.pretrain_encoder(None, None, None, None, method='byol', epochs=1)


def test_helpers_split_sample_and_scale():
    tr, te = at.split_by_group(['a', 'a', 'b', 'c', 'c', 'd'], 0.5, 0)
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'd'])
    assert not set(groups[tr]) & set(groups[te]) and len(tr) + len(te) == 6
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert at.few_shot_rows(np.arange(6), labels, 2, 0).tolist() != [] and len(
        at.few_shot_rows(np.arange(6), labels, 2, 0)) == 4
    assert at.few_shot_rows(np.arange(6), labels, 0, 0).tolist() == list(range(6))
    crops = np.ones((5, 2, 4, 4), np.float32)
    crops[:, 1] = 0
    assert at.fixed_channel_scale(crops, np.arange(5)).tolist() == [1.0, 1.0]
    x = at.prepare_batch(crops, np.arange(2), np.array([2.0, 1.0], np.float32), 8,
                         generator=torch.Generator().manual_seed(0))
    assert x.shape == (2, 2, 8, 8) and float(x[:, 1].max()) == -2.0
    assert at.channel_subsets('dna, tubulin; ;dna') == [['dna', 'tubulin'], ['dna']]
    assert at.channel_subsets([['dna'], []]) == [['dna']] and at.channel_subsets(None) == []


def test_settings_file_round_trip_through_main(tmp_path):
    from spacr.cli import write_settings_file

    s = dict(_cells(tmp_path), train_arch='spacr_channel_cnn')
    path = tmp_path / 'settings.csv'
    write_settings_file(at.architecture_defaults(s), str(path))
    assert at.main(['--settings', str(path)]) == 0
    assert (tmp_path / 'out' / 'summary.csv').is_file()


def test_the_settings_are_registered_under_the_app_key():
    from spacr import settings as S

    defaults = at.architecture_defaults()
    assert defaults['train_arch'] == 'spacr_well_transformer' and S.has_registered_defaults(at.APP_KEY)
    assert set(at._TOOLTIPS) == set(defaults) == set(at._EXPECTED_TYPES)
    assert {k for keys in at._CATEGORIES.values() for k in keys} == set(defaults)
    assert at.architecture_defaults({'train_set_seed': 4})['train_set_seed'] == 4
