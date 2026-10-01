"""Timeflows 2026-09-25: wider training gaps, harder held-out pairs, whole-movie
ids and the opt-in timelapse backend.

Nothing here loads real weights. The network is a stand-in and predictions
come from the true displacement (an oracle) or from zero motion, so these
tests pin the plumbing, not the model's accuracy.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import timeflows_model as tm

_SPEC = importlib.util.spec_from_file_location(
    'timeflows_evaluator_movie', Path(__file__).resolve().parents[1] / 'tools/evaluate_timeflows.py')
evaluate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate)


def _square(frame, y, x, label, size=4):
    frame[y:y + size, x:x + size] = label
    return frame


def _stack(shifts, labels_per_frame, shape=(40, 60)):
    """Two squares per frame, moving right by ``shifts[t]`` with given ids."""
    masks = np.zeros((len(shifts),) + shape, np.uint16)
    for t, (shift, (a, b)) in enumerate(zip(shifts, labels_per_frame)):
        if a:
            _square(masks[t], 5, 5 + shift, a)
        if b:
            _square(masks[t], 25, 30 + shift, b)
    return masks


def _oracle(net, frame_t, frame_t1, device='cpu'):
    """A predict_pair stand-in reading the true motion from the frames.

    ``net['truth']`` holds the same objects with one id per object across
    frames, while the masks being tracked shuffle their ids; this isolates
    linking and stitching from any model.
    """
    target = tm.time_targets(np.asarray(net['truth'][net['index']]),
                             np.asarray(net['truth'][net['index'] + 1]))
    net['index'] += 1
    return {'vector': target['vector'], 'successor': target['successor']}


def test_stitched_ids_follow_links_and_new_objects_start_new_tracks():
    masks = _stack([0, 1, 2], [(3, 9), (4, 8), (6, 0)])
    out = tm._stitch_links(masks, [{3: 4, 9: 8}, {4: 6}])
    assert set(np.unique(out[0])) == {0, 1, 2}
    assert out[1][5, 6] == out[0][5, 5] == 1
    assert out[1][25, 31] == out[0][25, 30] == 2
    assert out[2][5, 7] == 1


def test_unlinked_objects_are_new_tracks_in_ascending_label_order():
    masks = _stack([0, 0], [(5, 7), (2, 1)])
    out = tm._stitch_links(masks, [{}])
    assert out[1][25, 30] == 3 and out[1][5, 5] == 4


def test_stitched_ids_are_identical_on_a_rerun_whatever_the_link_order():
    masks = _stack([0, 1, 2], [(3, 9), (4, 8), (6, 1)])
    first = tm._stitch_links(masks, [{3: 4, 9: 8}, {8: 1, 4: 6}])
    second = tm._stitch_links(masks, [{9: 8, 3: 4}, {4: 6, 8: 1}])
    np.testing.assert_array_equal(first, second)


def test_links_naming_absent_labels_or_a_claimed_target_are_ignored():
    masks = _stack([0, 1], [(3, 9), (4, 8)])
    out = tm._stitch_links(masks, [{3: 4, 9: 4, 77: 8}])
    assert out[1][5, 6] == 1
    assert out[1][25, 31] == 3


def test_stitching_rejects_bad_shapes_and_link_counts():
    with pytest.raises(ValueError, match='T, H, W'):
        tm._stitch_links(np.zeros((4, 4)), [])
    with pytest.raises(ValueError, match='one link dict'):
        tm._stitch_links(np.zeros((3, 4, 4), int), [{}])
    assert tm._stitch_links(np.zeros((1, 4, 4), int), []).shape == (1, 4, 4)
    assert not tm._stitch_links(np.zeros((2, 4, 4), int), [{}]).any()


def test_track_movie_links_shuffled_ids_with_oracle_motion():
    masks = _stack([0, 3, 6, 9], [(1, 2), (2, 1), (5, 7), (7, 5)])
    net = {'truth': _stack([0, 3, 6, 9], [(1, 2)] * 4), 'index': 0}
    tracked, links = tm._track_movie(net, list(masks.astype(np.float32)), masks, predict=_oracle)
    assert links[0] == {1: 2, 2: 1}
    top = [tracked[t][5, 5 + 3 * t] for t in range(4)]
    bottom = [tracked[t][25, 30 + 3 * t] for t in range(4)]
    assert len(set(top)) == 1 and len(set(bottom)) == 1 and top[0] != bottom[0]


def test_track_movie_rejects_mismatched_frames():
    masks = _stack([0, 1], [(1, 2), (1, 2)])
    with pytest.raises(ValueError, match='one frame per'):
        tm._track_movie(None, [masks[0]], masks, predict=_oracle)
    with pytest.raises(ValueError, match='match its mask'):
        tm._track_movie(None, [np.zeros((5, 5))] * 2, masks, predict=_oracle)


def _ctc(root, frames=7, step=2):
    for folder in ('01', '01_ST/SEG', '01_GT/TRA'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for t in range(frames):
        labels = _square(np.zeros((30, 60), np.uint16), 10, 3 + step * t, 4)
        tifffile.imwrite(root / '01' / f't{t:03d}.tif', (labels > 0).astype(np.uint16) * 500)
        tifffile.imwrite(root / '01_ST/SEG' / f'man_seg{t:03d}.tif', labels)
        tifffile.imwrite(root / '01_GT/TRA' / f'man_track{t:03d}.tif', labels)
    return root


def test_ctc_pairs_default_is_unchanged_and_gaps_pair_wider_frames(tmp_path, monkeypatch):
    movie = _ctc(tmp_path / 'movie')
    assert len(tm.ctc_pairs(str(movie))) == 6
    reads = []
    original = tifffile.imread
    monkeypatch.setattr(tifffile, 'imread', lambda path: reads.append(Path(path).name) or original(path))
    pairs = tm.ctc_pairs(str(movie), gaps=(1, 3, 3), max_pairs=2)
    assert len(pairs) == 4
    moves = [tm.object_centroids(p.labels_t1)[4][1] - tm.object_centroids(p.labels_t)[4][1] for p in pairs]
    assert moves == [2.0, 2.0, 6.0, 6.0]
    assert len(reads) == len(set(reads))
    with pytest.raises(ValueError, match='gaps must be positive'):
        tm.ctc_pairs(str(movie), gaps=(0,))
    with pytest.raises(ValueError, match='gaps must be positive'):
        tm.ctc_pairs(str(movie), gaps=())


def test_a_wider_gap_gives_the_motion_sampler_rarer_heavier_pairs(tmp_path):
    movie = _ctc(tmp_path / 'movie', frames=13)
    pairs = tm.ctc_pairs(str(movie), gaps=(1, 12))
    weights = tm._training_pair_sampling_weights(pairs)
    assert len(pairs) == 13
    assert weights[-1] > weights[0]
    assert np.isclose(weights.sum(), 1)


def test_training_cli_rejects_non_positive_gaps(tmp_path):
    with pytest.raises(SystemExit):
        tm.main(['--movies', str(tmp_path), '--out', str(tmp_path / 'x.pt'), '--gaps', '0'])


def _eval_movie(root):
    for folder in ('01', '01_GT/SEG', '01_GT/TRA'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    offsets = [0, 0, 1, 8, 9, 9]
    for t, offset in enumerate(offsets):
        labels = np.zeros((30, 60), np.uint16)
        labels[5:9, 5 + offset:9 + offset] = 7
        labels[20:24, 40:44] = 3
        for folder, prefix, array in [('01', 't', (labels > 0).astype(np.uint16) * 300),
                                      ('01_GT/SEG', 'man_seg', labels),
                                      ('01_GT/TRA', 'man_track', labels)]:
            tifffile.imwrite(root / folder / f'{prefix}{t:03d}.tif', array)
    return root


def test_hardest_selection_ranks_candidates_by_annotated_motion(tmp_path):
    movie = _eval_movie(tmp_path / 'movie')
    chosen = evaluate.selected_pairs(movie, '01', 'GT', [1], 1, 'hardest', 5)
    assert [pair['frame_numbers'] for pair in chosen] == [[2, 3]]
    assert chosen[0]['hardness']['fast_share'] == 0.5
    assert chosen[0]['hardness']['tracked_objects'] == 2
    assert chosen[0]['hardness']['mean_motion'] > 0.5
    even = evaluate.selected_pairs(movie, '01', 'GT', [1], 1)
    assert 'hardness' not in even[0]
    with pytest.raises(ValueError, match='even or hardest'):
        evaluate.selected_pairs(movie, '01', 'GT', [1], 1, 'random', 5)
    with pytest.raises(ValueError, match='at least that many'):
        evaluate.selected_pairs(movie, '01', 'GT', [1], 3, 'hardest', 2)


def test_select_only_then_pairs_from_scores_the_same_pairs(tmp_path, monkeypatch):
    movie = _eval_movie(tmp_path / 'movie')
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'stand-in')
    checkpoint.with_suffix('.pt.json').write_text(json.dumps({'movies': [str(tmp_path / 'training')]}))
    base = ['--checkpoint', str(checkpoint), '--movie', str(movie), '--sequences', '01',
            '--segmentation', 'GT', '--data-kind', 'synthetic']
    picked = tmp_path / 'picked'

    def refuse(*args):
        raise AssertionError('select-only must not load a model')

    monkeypatch.setattr(evaluate, 'checkpoint_predictors', refuse)
    assert evaluate.main(base + ['--gaps', '1', '--pairs-per-gap', '2', '--selection', 'hardest',
                                 '--candidates-per-gap', '5', '--select-only', '--out', str(picked)]) == 0
    run = json.loads((picked / 'run.json').read_text())
    assert not (picked / 'summary.json').exists()
    assert [pair['frame_numbers'] for pair in run['selected']] == [[1, 2], [2, 3]]

    def factory(path, device, seed, precision):
        def predict(a, b, random_head=False):
            return {'vector': np.zeros((2,) + a.shape), 'successor': np.ones(a.shape)}
        predict.precision = {'requested': precision}
        return predict

    monkeypatch.setattr(evaluate, 'checkpoint_predictors', factory)
    scored = tmp_path / 'scored'
    assert evaluate.main(base + ['--pairs-from', str(picked / 'run.json'), '--out', str(scored)]) == 0
    summary = json.loads((scored / 'summary.json').read_text())
    assert [pair['frame_numbers'] for pair in summary['selected']] == \
        [pair['frame_numbers'] for pair in run['selected']]
    assert summary['selection'].startswith('reused from')
    rows = [json.loads(line) for line in (scored / 'pairs.jsonl').read_text().splitlines()]
    assert all('hardness' in row for row in rows)
    other = _eval_movie(tmp_path / 'other')
    with pytest.raises(ValueError, match='same movie'):
        evaluate.reselected_pairs(other, 'GT', picked / 'run.json')


def test_timeflows_is_an_opt_in_timelapse_mode_with_no_default_model():
    from spacr.settings import set_default_settings_preprocess_generate_masks
    from spacr.settings import categories, expected_types, tooltips
    from spacr.settings_spec import convert_settings_dict_for_gui

    settings = set_default_settings_preprocess_generate_masks({'src': 'x'})
    assert settings['timelapse_mode'] == 'trackastra'
    assert settings['timeflows_model'] is None
    spec = convert_settings_dict_for_gui({'timelapse_mode': 'trackastra'})
    kind, choices, default = spec['timelapse_mode']
    assert 'timeflows' in choices and default == 'trackastra'
    assert 'timeflows' in tooltips['timelapse_mode']
    assert 'timeflows_model' in categories['Timelapse']
    assert expected_types['timeflows_model'] == (str, type(None))
    assert len(tooltips['timeflows_model'].split()) >= 15


def _backend_inputs():
    masks = _stack([0, 3, 6], [(1, 2), (2, 1), (4, 9)])
    truth = _stack([0, 3, 6], [(1, 2)] * 3)
    return masks, masks.astype(np.float32), {'truth': truth, 'index': 0}


def test_timelapse_backend_writes_consistent_tracks(tmp_path, monkeypatch):
    from spacr import timelapse

    masks, images, net = _backend_inputs()
    monkeypatch.setattr(tm, 'predict_pair', _oracle)
    src = tmp_path / 'run' / 'masks'
    src.mkdir(parents=True)
    stack = timelapse._timeflows_track_cells(str(src), 'plate1_A01', ['a', 'b', 'c'], 'cell',
                                             masks, images=images, net=net)
    table = pd.read_csv(tmp_path / 'run' / 'tracks' / 'timeflows_tracks_cell_plate1_A01.csv')
    assert list(table.columns) == ['frame', 'track_id', 'original_label', 'x', 'y']
    assert table.groupby('track_id')['frame'].nunique().tolist() == [3, 3]
    assert np.asarray(stack).shape == masks.shape


def test_timelapse_backend_remove_transient_keeps_full_length_tracks(tmp_path, monkeypatch):
    from spacr import timelapse

    masks = _stack([0, 3, 6], [(1, 2), (2, 0), (4, 0)])
    net = {'truth': _stack([0, 3, 6], [(1, 2), (1, 0), (1, 0)]), 'index': 0}
    monkeypatch.setattr(tm, 'predict_pair', _oracle)
    src = tmp_path / 'masks'
    src.mkdir()
    stack = timelapse._timeflows_track_cells(str(src), 'n', ['a', 'b', 'c'], 'cell', masks,
                                             images=masks.astype(np.float32), net=net,
                                             timelapse_remove_transient=True)
    assert all(len(np.unique(plane)) == 2 for plane in stack)


def test_timelapse_backend_refuses_missing_inputs_before_loading(tmp_path, capsys):
    from spacr import timelapse

    masks, images, _net = _backend_inputs()
    with pytest.raises(ValueError, match='timeflows_model'):
        timelapse._timeflows_track_cells(str(tmp_path), 'n', [], 'cell', masks, images=images,
                                         model_path=str(tmp_path / 'absent.pt'))
    with pytest.raises(ValueError, match='image stack'):
        timelapse._timeflows_track_cells(str(tmp_path), 'n', [], 'cell', masks, images=None)
    with pytest.raises(ValueError, match='does not match'):
        timelapse._timeflows_track_cells(str(tmp_path), 'n', [], 'cell', masks, images=images[:, :5])
    with pytest.raises(ValueError, match='mask stack'):
        timelapse._timeflows_track_cells(str(tmp_path), 'n', [], 'cell', masks[0], images=images)
    single = timelapse._timeflows_track_cells(str(tmp_path), 'n', [], 'cell', masks[:1], images=None)
    assert len(single) == 1 and 'nothing to link' in capsys.readouterr().out


def test_timelapse_backend_loads_the_named_checkpoint_on_cpu_in_float32(tmp_path, monkeypatch):
    from spacr import timelapse

    masks, images, net = _backend_inputs()
    checkpoint = tmp_path / 'tf.pt'
    checkpoint.write_bytes(b'stand-in')
    calls = []

    def load(path, device='cpu', precision='checkpoint'):
        calls.append((path, device, precision))
        return net

    monkeypatch.setattr(tm, '_load_timeflows', load)
    monkeypatch.setattr(tm, 'predict_pair', _oracle)
    timelapse._timeflows_track_cells(str(tmp_path / 'masks'), 'n', [], 'cell', masks,
                                     images=images, model_path=str(checkpoint), device='cpu')
    assert calls == [(str(checkpoint), 'cpu', 'float32')]


def test_load_timeflows_rebuilds_the_saved_network_without_download(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')

    class Encoder(torch.nn.Module):
        def __init__(self, ps=8, dtype=torch.float32):
            super().__init__()
            self.ps = ps
            self.weight = torch.nn.Parameter(torch.zeros(2, dtype=dtype))

    state = {'encoder.encoder.patch_embed.proj.weight': torch.zeros(1, dtype=torch.bfloat16),
             'up.weight': torch.zeros(1, 1, 8, 8)}
    path = tmp_path / 'tf.pt'
    torch.save(state, path)
    loaded = {}
    monkeypatch.setitem(sys.modules, 'cellpose.vit', types.SimpleNamespace(
        CPSAM=lambda ps, dtype: loaded.update(ps=ps, dtype=dtype) or Encoder(ps, dtype)))

    class Fake(torch.nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = features

        def load_state_dict(self, state, strict=True):
            loaded['keys'] = sorted(state)
            loaded['strict'] = strict

    monkeypatch.setattr(tm, 'TimeflowsNet', Fake)
    monkeypatch.setattr(tm, 'CellposeSamFeatures', lambda encoder: encoder)
    net = tm._load_timeflows(str(path))
    assert loaded['ps'] == 8 and loaded['dtype'] == torch.bfloat16 and loaded['strict']
    assert not net.training
    tm._load_timeflows(str(path), precision='float32')
    assert loaded['dtype'] == torch.float32
    with pytest.raises(ValueError, match='precision'):
        tm._load_timeflows(str(path), precision='half')
