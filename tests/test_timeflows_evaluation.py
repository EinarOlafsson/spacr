"""The Timeflows evaluator measures explicit truths and preserves its controls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import tifffile

from spacr import timeflows_model as tm

_SPEC = importlib.util.spec_from_file_location(
    'timeflows_evaluator', Path(__file__).resolve().parents[1] / 'tools/evaluate_timeflows.py')
evaluate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate)


def _labels(moving=0):
    labels = np.zeros((40, 50), np.uint16)
    labels[5:9, 5 + moving:9 + moving] = 7
    labels[25:29, 30:34] = 91
    return labels


def _movie(root, missing=None):
    for folder in ('01', '01_GT/SEG', '01_GT/TRA'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for index in range(5):
        mask = _labels(index)
        for folder, prefix, array in [('01', 't', (mask > 0).astype(np.uint16) * 400),
                                      ('01_GT/SEG', 'man_seg', mask),
                                      ('01_GT/TRA', 'man_track', mask)]:
            if (folder, index) != missing:
                tifffile.imwrite(root / folder / f'{prefix}{index:03d}.tif', array)
    return root


def test_pair_selection_uses_real_gaps_and_complete_annotations(tmp_path):
    movie = _movie(tmp_path / 'movie', missing=('01_GT/SEG', 2))
    selected = evaluate.selected_pairs(movie, '01', 'GT', [1, 3], 1)
    assert [pair['frame_numbers'] for pair in selected] == [[0, 1], [0, 3]]
    all_pairs = evaluate.selected_pairs(movie, '01', 'GT', [1], 0)
    assert [pair['frame_numbers'] for pair in all_pairs] == [[0, 1], [3, 4]]
    with pytest.raises(ValueError, match='No matched images'):
        evaluate.selected_pairs(movie, '01', 'ST', [1], 1)
    with pytest.raises(ValueError, match='two-digit'):
        evaluate.selected_pairs(movie, '../training', 'GT', [1], 1)


def test_slice_masks_and_duplicate_frame_names_are_not_silently_accepted(tmp_path):
    (tmp_path / 'man_seg_000_001.tif').touch()
    assert evaluate.indexed(tmp_path, 'man_seg') == {}
    (tmp_path / 'man_seg000.tif').touch()
    (tmp_path / 'man_seg000.tiff').touch()
    with pytest.raises(ValueError, match='Duplicate frame'):
        evaluate.indexed(tmp_path, 'man_seg')


def test_ambiguous_or_duplicate_marker_assignments_are_excluded():
    seg = np.zeros((8, 16), np.uint16)
    for index in range(4):
        seg[:, index * 4:(index + 1) * 4] = index + 1
    markers = np.zeros_like(seg)
    markers[1, 1], markers[2, 2] = 10, 11
    markers[1, 5], markers[1, 9] = 20, 20
    output, counts = evaluate.tracked_masks(seg, markers)
    assert not output.any()
    assert counts['multi_marker_objects'] == 1
    assert counts['duplicate_track_objects'] == 2
    assert counts['unmarked_objects'] == 1
    assert counts['excluded_track_ids'] == [10, 11, 20]


def test_scrambling_preserves_sparse_label_objects():
    labels = _labels()
    scrambled, mapping = evaluate.scramble(labels, seed=4)
    assert set(mapping) == set(mapping.values()) == {7, 91}
    for old, new in mapping.items():
        np.testing.assert_array_equal(labels == old, scrambled == new)


def test_oracle_beats_position_only_on_large_motion():
    first, second = _labels(), _labels(12)
    target = tm.time_targets(first, second)
    predictions = {'oracle': target, 'zero': {'vector': np.zeros_like(target['vector']),
                                             'successor': np.ones_like(target['successor'])}}
    rows = evaluate.score_pair(first, second, predictions, seed=2)
    hard = next(row for row in rows if row['label'] == 7)
    assert hard['motion_bin'] == 'at_least_1'
    assert hard['correct']['oracle'] and not hard['correct']['zero']
    summary = evaluate.summarise(rows)
    assert summary['motion']['at_least_1']['true_successors'] == 1
    assert summary['motion']['at_least_1']['arms']['oracle']['successor_accuracy'] == 1
    assert summary['motion']['0.5_to_1']['arms']['oracle']['successor_accuracy'] is None
    assert sum(group['true_successors'] for group in summary['motion_by_density'].values()) == 2


def test_false_successor_links_are_counted_but_missing_annotations_are_not():
    first = _labels()
    second = first.copy()
    second[second == 7] = 15
    prediction = {'zero': {'vector': np.zeros((2,) + first.shape), 'successor': np.ones(first.shape)}}
    rows = evaluate.score_pair(first, second, prediction)
    result = evaluate.summarise(rows)['overall']
    assert result['true_successors'] == 1 and result['no_successor'] == 1
    assert result['arms']['zero']['false_links_without_successor'] == 1
    censored = evaluate.score_pair(first, second, prediction, unknown_successors=[7])
    assert [row['label'] for row in censored] == [91]


def test_holdout_check_rejects_aliases_and_missing_provenance(tmp_path):
    training = tmp_path / 'training'
    training.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(training, target_is_directory=True)
    for path in (training, alias, training / '01', tmp_path):
        with pytest.raises(ValueError, match='overlaps'):
            evaluate.check_holdout(path, {'movies': [str(training)]})
    with pytest.raises(ValueError, match='nonempty training'):
        evaluate.check_holdout(tmp_path / 'heldout', {})
    evaluate.check_holdout(tmp_path / 'heldout', {'movies': [str(training)]})


def test_real_cli_writes_only_complete_reports_with_explicit_scope(tmp_path, monkeypatch):
    movie = _movie(tmp_path / 'heldout')
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'checkpoint stand-in')
    checkpoint.with_suffix('.pt.json').write_text(json.dumps({'movies': [str(tmp_path / 'training')]}))
    before = {path: evaluate.digest(path) for path in movie.rglob('*.tif')}

    def factory(path, device, seed, precision):
        assert path == checkpoint and device == 'cpu'
        assert precision == 'float32'

        def predict(a, b, random_head=False):
            return {'vector': np.zeros((2,) + a.shape), 'successor': np.ones(a.shape)}
        predict.precision = {'requested': precision, 'effective_encoder': 'torch.float32'}
        return predict

    monkeypatch.setattr(evaluate, 'checkpoint_predictors', factory)
    out = tmp_path / 'result'
    argv = ['--checkpoint', str(checkpoint), '--movie', str(movie), '--sequences', '01',
            '--segmentation', 'GT', '--data-kind', 'synthetic', '--gaps', '1',
            '--pairs-per-gap', '2', '--precision', 'float32', '--out', str(out)]
    assert evaluate.main(argv) == 0
    report = json.loads((out / 'summary.json').read_text())
    assert report['complete'] and report['selected_pairs'] == 2
    assert report['data_kind'] == 'synthetic'
    assert report['precision']['effective_encoder'] == 'torch.float32'
    assert report['requested_precision'] == 'float32'
    assert json.loads((out / 'run.json').read_text())['precision'] == report['precision']
    assert report['results']['overall']['true_successors'] == 4
    assert report['copied_frame_control']['overall']['arms']['trained']['successor_accuracy'] == 1
    assert len((out / 'pairs.jsonl').read_text().splitlines()) == 2
    assert before == {path: evaluate.digest(path) for path in before}
    with pytest.raises(FileExistsError):
        evaluate.main(argv)

    def failing_factory(*args):
        raise RuntimeError('checkpoint cannot load')

    monkeypatch.setattr(evaluate, 'checkpoint_predictors', failing_factory)
    failed = tmp_path / 'failed'
    argv[-1] = str(failed)
    with pytest.raises(RuntimeError, match='checkpoint cannot load'):
        evaluate.main(argv)
    assert (failed / 'run.json').exists()
    assert not (failed / 'summary.json').exists()


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('precision', ['checkpoint', 'float32'])
def test_random_head_control_restores_trained_parameters(tmp_path, monkeypatch, dtype_name, precision):
    torch = pytest.importorskip('torch')

    class Backbone(torch.nn.Module):
        channels = 4

        def __init__(self, ps=2, dtype=torch.float32):
            super().__init__()
            self.ps = ps
            self.encoder = torch.nn.Module()
            self.encoder.patch_embed = torch.nn.Module()
            self.encoder.patch_embed.proj = torch.nn.Conv2d(3, 4, ps, stride=ps)
            self.encoder.patch_embed.proj.weight.data = self.encoder.patch_embed.proj.weight.data.to(dtype)

        def forward(self, x):
            return self.encoder.patch_embed.proj(x.to(self.encoder.patch_embed.proj.weight.dtype)).float()

    monkeypatch.setitem(sys.modules, 'cellpose.vit', types.SimpleNamespace(CPSAM=Backbone))
    monkeypatch.setattr(tm, 'CellposeSamFeatures', lambda net: net)
    dtype = getattr(torch, dtype_name)
    net = tm.TimeflowsNet(Backbone(dtype=dtype).to(dtype=dtype))
    checkpoint = tmp_path / 'tiny.pt'
    torch.save(net.state_dict(), checkpoint)
    before = evaluate.digest(checkpoint)
    expected_dtype = torch.float32 if precision == 'float32' else dtype
    ordinary_predict = tm.predict_pair

    def checked_predict(model, *args, **kwargs):
        weight = model.encoder.encoder.patch_embed.proj.weight
        assert weight.dtype == expected_dtype
        torch.testing.assert_close(weight, net.encoder.encoder.patch_embed.proj.weight.to(expected_dtype), rtol=0, atol=0)
        return ordinary_predict(model, *args, **kwargs)

    monkeypatch.setattr(tm, 'predict_pair', checked_predict)
    predict = evaluate.checkpoint_predictors(checkpoint, 'cpu', 91, precision)
    assert predict.precision['checkpoint_encoder'] == str(dtype)
    assert predict.precision['effective_encoder'] == str(expected_dtype)
    a = np.ones((16, 16), np.float32)
    first = predict(a, a)
    random = predict(a, a, random_head=True)
    restored = predict(a, a)
    assert not np.allclose(first['vector'], random['vector'])
    for key in first:
        np.testing.assert_allclose(first[key], restored[key])
    assert evaluate.digest(checkpoint) == before
