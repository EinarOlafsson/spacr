"""Item 889, image arms: crops, samplers, the saved transform, the fold loop and the BBBC021 scorer.

Tiny CPU fixtures only: a BBBC021-like crop array with its index (two
weeks, three MoAs) and an OPS-like shard of TIFF crops with byte offsets,
genotype, scales and gene-class tables.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
import torch

from spacr import gene_class_images as gci
from spacr.tabular import write_table


def _bbbc(tmp_path, n_per=6, px=16):
    rng = np.random.default_rng(0)
    rows, crops = [], []
    comps = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'b2': 'B', 'c1': 'C', 'taxol': 'A'}
    for week in ('Week1', 'Week2'):
        for wi, (comp, moa) in enumerate(comps.items()):
            if comp == 'taxol' and week == 'Week2':
                continue
            for _ in range(n_per):
                x = rng.gamma(2, 100, size=(px, px, 3))
                x[..., 'ABC'.index(moa)] *= 3
                crops.append(x.astype(np.uint16))
                rows.append({'plate': f'{week}_1', 'well': f'w{wi}', 'compound': comp,
                             'concentration': 1.0 if comp == 'taxol' else 0.5, 'moa': moa, 'week': week,
                             'scale0': 600.0, 'scale1': 600.0, 'scale2': 600.0})
    np.save(tmp_path / 'crops.npy', np.stack(crops))
    write_table(pd.DataFrame(rows), tmp_path / 'index.parquet', canonicalise=False)
    return {'dataset': 'bbbc021', 'memmap': str(tmp_path / 'crops.npy'), 'index': str(tmp_path / 'index.parquet'),
            'design': 'NSCB', 'steps': 4, 'batch': 6, 'size': 32, 'threads': 1, 'loader_threads': 2,
            'cpu': True}


def test_transform_spec_and_check(tmp_path):
    cfg = _bbbc(tmp_path)
    pre = gci.transform_spec(dict(cfg, arch='resnet18'), 4)
    assert pre['kind'] == 'pretrained' and pre['mean'][3] == pytest.approx(np.mean(gci.IMAGENET_MEAN))
    assert pre['scales_md5'] and pre['clip'] == [0.0, 1.0]
    scratch = gci.transform_spec(dict(cfg, arch='spacr_cnn'), 3)
    assert scratch['kind'] == 'dataset_standardised' and scratch['mean'] == [0.5] * 3
    legacy = gci.transform_spec(dict(cfg, arch='resnet18', legacy_transform=True), 3)
    assert legacy['clip'] == [0.0, 1.5] and legacy['version'] == 1
    gci.check_transform(pre, dict(pre))
    with pytest.raises(RuntimeError, match='preprocessing differs'):
        gci.check_transform(pre, dict(pre, size=128))


def test_device_prep_augments_only_in_training():
    spec = {'mean': [0.5, 0.5], 'std': [0.25, 0.25], 'size': 8, 'clip': [0.0, 1.0], 'interpolation': 'bilinear',
            'antialias': True}
    x = torch.rand(3, 2, 8, 8)
    te = gci.DevicePrep(spec, False)(x)
    torch.testing.assert_close(te, (x - 0.5) / 0.25)
    a = gci.DevicePrep(spec, True, seed=1)(x)
    b = gci.DevicePrep(spec, True, seed=1)(x)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, te)
    assert gci.DevicePrep(dict(spec, size=4), False)(x).shape == (3, 2, 4, 4)


def test_balanced_sampler_and_class_gene_sampler():
    s = gci.BalancedSampler([np.array([1, 2]), np.array([], int), np.array([9])], [1, 5, 1], 0)
    draws = s.draw(400)
    assert set(draws) <= {1, 2, 9} and 0.4 < (draws == 9).mean() < 0.6
    Y = np.array([[1, 0], [1, 1], [0, 0], [0, 0]], np.float32)
    gene_of_cell = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    sm = gci.class_gene_sampler({'Y': Y, 'gene_of_cell': gene_of_cell}, np.arange(20), 0)
    d = sm.draw(4000)
    pos = np.isin(gene_of_cell[d], [0, 1]).mean()
    assert 0.45 < pos < 0.55
    assert (gene_of_cell[d] == 1).mean() > (gene_of_cell[d] == 0).mean()


def test_adapted_stem_keeps_a_uniform_response():
    conv = torch.nn.Conv2d(3, 4, 3, bias=True)
    new = gci.adapt_first_conv(conv, 5)
    x3, x5 = torch.ones(1, 3, 5, 5), torch.ones(1, 5, 5, 5)
    assert new.bias is not None
    torch.testing.assert_close(conv(x3).detach(), new(x5).detach())
    assert new.weight.shape == (4, 5, 3, 3)
    torch.testing.assert_close(new.weight.sum(1), conv.weight.mean(1) * 3)


@pytest.mark.parametrize('arch', ['resnet18', 'convnext_tiny', 'spacr_cnn'])
def test_models_build_with_any_channel_count(arch):
    model, backbone = gci.make_image_net({'arch': arch, 'pretrained': False, 'mode': 'finetune'}, 4, 5)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 4, 32, 32)).shape == (2, 5)
    assert (backbone is None) == (arch == 'spacr_cnn')


def test_frozen_mode_trains_only_the_head():
    model, backbone = gci.make_image_net({'arch': 'resnet18', 'pretrained': False, 'mode': 'frozen'}, 3, 2)
    assert not any(p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in model[1].parameters())
    with pytest.raises(ValueError, match='no image architecture'):
        gci.build_image_model('alexnet', 3, 2)


def test_bbbc021_fold_trains_scores_and_is_reproducible(tmp_path):
    cfg = dict(_bbbc(tmp_path), arch='spacr_cnn', pretrained=False, save_weights=True)
    run = tmp_path / 'run'
    t = gci.train_image_fold(cfg, 0, run / 'fold_00')
    assert t['steps'] == 4 and t['test_cells'] == 30
    p1 = np.load(run / 'fold_00' / 'test_probs.npy')
    np.testing.assert_allclose(p1.sum(1), 1, rtol=1e-5)
    gci.train_image_fold(cfg, 0, tmp_path / 'again' / 'fold_00')
    assert np.array_equal(p1, np.load(tmp_path / 'again' / 'fold_00' / 'test_probs.npy'))
    saved = torch.load(run / 'fold_00' / 'model.pt', weights_only=False)
    assert saved['transform'] == json.loads((run / 'fold_00' / 'transform.json').read_text())
    gci.train_image_fold(cfg, 1, run / 'fold_01')
    res = gci.score_bbbc_run(run, cfg)
    assert res['folds_done'] == 2 and res['n_treatments'] == 10
    assert 0 <= res['accuracy'] <= 1 and (run / 'eval' / 'summary.json').is_file()
    with pytest.raises(FileNotFoundError):
        gci.score_bbbc_run(tmp_path / 'again' / 'fold_00', cfg)


def test_smoke_mode_and_shuffled_labels(tmp_path):
    cfg = dict(_bbbc(tmp_path), arch='resnet18', pretrained=False, smoke=True, smoke_steps=2,
               smoke_test_cells=5, shuffle_labels=True, export_embeddings=True, sampling='none')
    t = gci.train_image_fold(cfg, 2, tmp_path / 'f')
    assert t['steps'] == 2 and t['batch'] == 8 and t['test_cells'] == 5
    assert np.load(tmp_path / 'f' / 'test_embeddings_f16.npy').shape == (5, 512)
    S = gci.bbbc_fold_setup(cfg, 0)
    S0 = gci.bbbc_fold_setup(dict(cfg, shuffle_labels=False), 0)
    assert not np.array_equal(S['lab'], S0['lab']) and S['c_in'] == 3


def _ops(tmp_path):
    import tifffile

    rng = np.random.default_rng(0)
    genes = ['G1', 'G2', 'G3', 'G4', 'nontargeting']
    shard = tmp_path / 'shards' / 'GC_A_0.tar'
    shard.parent.mkdir()
    rows, gt, offset = [], [], 0
    with open(shard, 'wb') as fh:
        for i in range(40):
            buf = io.BytesIO()
            tifffile.imwrite(buf, rng.integers(0, 1000, size=(4, 12, 12)).astype(np.uint16))
            data = buf.getvalue()
            fh.write(data)
            rows.append({'plate': 'A', 'well': 'A1', 'site': i // 10, 'label': i, 'offset_data': offset,
                         'size': len(data), 'window_edge': 0, 'prcfo': i})
            offset += len(data)
            g = genes[i % 5]
            gt.append({'plate': 'A', 'well': 'A1', 'site': i // 10, 'label': i, 'gene': g,
                       'sgRNA': f'sg{i % 2}' if g == 'nontargeting' else f'sg_{g}'})
    write_table(pd.DataFrame(rows), str(shard) + '.index.csv', canonicalise=False)
    write_table(pd.DataFrame(gt), tmp_path / 'genotype.parquet', canonicalise=False)
    (tmp_path / 'scales.json').write_text(json.dumps({'A': [500.0] * 4}))
    tb = tmp_path / 'tables'
    tb.mkdir()
    write_table(pd.DataFrame({'class_id': ['C1', 'C1', 'C2', 'C2'], 'gene': ['G1', 'G2', 'G2', 'G3'],
                              'trainable': [True, True, True, False]}), tb / 'class_members.csv', canonicalise=False)
    write_table(pd.DataFrame({'gene': ['G1', 'G2', 'G3', 'G4'], 'fold': [0, 1, 0, 1]}), tb / 'gene_folds.csv',
                canonicalise=False)
    write_table(pd.DataFrame({'sgRNA': ['sg0', 'sg1'], 'pseudogene': ['NT_0', 'NT_1'], 'fold': [0, 1]}),
                tb / 'nt_pseudogenes.csv', canonicalise=False)
    return {'dataset': 'ops', 'shard_dirs': [str(shard.parent)], 'genotype': str(tmp_path / 'genotype.parquet'),
            'scales': str(tmp_path / 'scales.json'), 'tables': str(tb), 'steps': 3, 'batch': 4, 'size': 16,
            'threads': 1, 'loader_threads': 2, 'cpu': True, 'out': str(tmp_path / 'run')}


def test_ops_setup_masks_unknown_labels_and_adds_decoys(tmp_path):
    cfg = _ops(tmp_path)
    (tmp_path / 'run').mkdir()
    S = gci.ops_fold_setup(dict(cfg, decoy_heads=2, gene_weighting='inv_sqrt'), 0)
    gi = {g: i for i, g in enumerate(S['genes'])}
    assert S['classes'] == ['C1', 'C2', 'DECOY000', 'DECOY001']
    assert S['Y'][gi['G2'], :2].tolist() == [1, 1] and S['M'][gi['G3'], 1] == 0
    assert S['W'][gi['G2']] == pytest.approx(1 / np.sqrt(2))
    test_genes = set(np.array(S['genes'])[S['gene_of_cell'][S['test_rows']]])
    assert test_genes == {'G1', 'G3', 'NT_0'}
    assert not test_genes & set(np.array(S['genes'])[S['gene_of_cell'][S['train_rows']]])
    assert (tmp_path / 'run' / 'decoys.csv').is_file()
    ds = gci.OpsCropDataset(S['cells'], np.arange(3))
    x, j = ds[1]
    assert x.shape == (4, 12, 12) and j == 1 and 0 <= x.min() and x.max() <= 1
    assert len(ds) == 3
    sh = gci.ops_fold_setup(dict(cfg, shuffle_labels=True), 0)
    assert sh['Y'].sum() == gci.ops_fold_setup(cfg, 0)['Y'].sum()


def test_ops_fold_writes_gene_scores(tmp_path):
    cfg = dict(_ops(tmp_path), arch='spacr_cnn', pretrained=False, cells_per_gene_cap=6)
    t = gci.train_image_fold(cfg, 1, tmp_path / 'run' / 'fold_01')
    gs = pd.read_parquet(tmp_path / 'run' / 'fold_01' / 'gene_scores.parquet')
    assert set(gs.index) == {'G2', 'G4', 'NT_1'} and gs['n_cells'].sum() == t['test_cells']
    assert gs[['C1', 'C2']].stack().between(0, 1).all()


def test_settings_config_and_errors(tmp_path):
    base = {'train_arch': 'dinov2_frozen', 'train_balanced_sampling': 'class', 'train_image_dataset': 'bbbc021',
            'train_input_size': 224, 'train_steps': 10, 'train_batch': 8, 'train_seed': 2,
            'train_shuffle_control': False, 'train_crops': 'c.npy', 'train_crop_index': 'i.parquet',
            'train_protocol': 'NSC', 'train_genotype': '', 'train_crop_scales': '', 'train_gene_classes': '',
            'train_decoy_heads': 0}
    cfg = gci.image_fold_config(base)
    assert cfg['arch'] == 'vit_s_dinov2' and cfg['mode'] == 'frozen' and cfg['design'] == 'NSC'
    assert gci.image_fold_config(dict(base, train_arch='spacr_cnn'))['pretrained'] is False
    with pytest.raises(ValueError, match='train_arch'):
        gci.image_fold_config(dict(base, train_arch='alexnet'))
    with pytest.raises(ValueError, match='train_balanced_sampling'):
        gci.image_fold_config(dict(base, train_balanced_sampling='gene'))
    with pytest.raises(ValueError, match='train_image_dataset'):
        gci.image_fold_config(dict(base, train_image_dataset='jump'))
    with pytest.raises(ValueError, match='BBBC021 crop'):
        gci.image_fold_config(dict(base, train_crops=''))
    with pytest.raises(ValueError, match='OPS crops'):
        gci.image_fold_config(dict(base, train_image_dataset='ops'))
    ops = gci.image_fold_config(dict(base, train_image_dataset='ops', train_crops='a, b', train_genotype='g',
                                     train_crop_scales='s', train_gene_classes='t', train_decoy_heads=3))
    assert ops['shard_dirs'] == ['a', 'b'] and ops['decoy_heads'] == 3 and ops['classes'] is None


def test_run_from_settings_trains_every_fold_and_scores(tmp_path):
    cfg = _bbbc(tmp_path)
    s = {'train_arch': 'spacr_cnn', 'train_balanced_sampling': 'class', 'train_image_dataset': 'bbbc021',
         'train_input_size': 32, 'train_steps': 2, 'train_batch': 4, 'train_seed': 0,
         'train_shuffle_control': False, 'train_crops': cfg['memmap'], 'train_crop_index': cfg['index'],
         'train_protocol': 'NSCB', 'gene_class_dst': str(tmp_path / 'img'), 'train_folds': 5}
    lines = []
    out = gci.run_gene_class_images(s, log=lines.append)
    assert len(out['timings']) == 3 and len(lines) == 3
    assert out['summary']['folds_done'] == 3 and (tmp_path / 'img' / 'config.json').is_file()


def test_gene_classes_settings_run_the_image_arms(tmp_path, monkeypatch):
    from spacr import gene_classes as gc

    cfg = _bbbc(tmp_path)
    s = {'ml_model': 'image', 'train_arch': 'spacr_cnn', 'train_image_dataset': 'bbbc021',
         'train_balanced_sampling': 'class', 'train_input_size': 32, 'train_steps': 1, 'train_batch': 4,
         'train_crops': cfg['memmap'], 'train_crop_index': cfg['index'], 'gene_class_dst': str(tmp_path / 'b')}
    res = gc.run_gene_classes(s)
    assert res['summary']['folds_done'] == 3 and res['run'] == str(tmp_path / 'b')
    (tmp_path / 'o').mkdir()
    ops = _ops(tmp_path / 'o')
    seen = {}

    def fake_eval(run, **kw):
        seen['run'] = run
        return {'median_auroc': 0.5, 'classes_scored': 2}
    monkeypatch.setattr(gc, 'evaluate_gene_class_run', fake_eval)
    res = gc.run_gene_classes({'ml_model': 'image', 'train_arch': 'spacr_cnn', 'train_image_dataset': 'ops',
                               'train_input_size': 16, 'train_steps': 1, 'train_batch': 4, 'train_folds': 2,
                               'train_decoy_heads': 1, 'train_crops': ops['shard_dirs'][0],
                               'train_genotype': ops['genotype'], 'train_crop_scales': ops['scales'],
                               'train_gene_classes': ops['tables'], 'gene_class_dst': str(tmp_path / 'oo')})
    assert seen['run'] == str(tmp_path / 'oo') and res['summary']['median_auroc'] == 0.5
    assert (tmp_path / 'oo' / 'fold_01' / 'gene_scores.parquet').is_file()
    assert (tmp_path / 'oo' / 'decoys.csv').is_file()
    with pytest.raises(ValueError, match="'logistic_regression' or 'image'"):
        gc.run_gene_classes({'ml_model': 'xgboost', 'train_gene_classes': 'x'})
