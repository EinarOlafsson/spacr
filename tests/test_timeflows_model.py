"""Timeflows, item 426 steps 3-6: targets, augmentation, linking, training.

Synthetic movies of discs that move a known amount per frame, so every
answer is known exactly. Nothing here needs a GPU or Cellpose weights: the
backbone is a small stand-in with the same ``features`` contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import timeflows_model as tm

torch = pytest.importorskip("torch")


def _disc(shape, cy, cx, r, label, out):
    yy, xx = np.indices(shape)
    out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label


def _movie(frames=4, step=12, shape=(64, 64)):
    """Two discs moving right and down by ``step`` px per frame."""
    labels = np.zeros((frames,) + shape, np.int32)
    images = np.zeros((frames,) + shape, np.float32)
    for t in range(frames):
        _disc(shape, 12, 8 + step * t, 5, 1, labels[t])
        _disc(shape, 8 + step * t, 40, 5, 2, labels[t])
        images[t] = (labels[t] > 0) * 0.8 + 0.1
    return images, labels


def test_targets_point_at_the_successor_centre_in_diameters():
    _images, labels = _movie()
    target = tm.time_targets(labels[0], labels[1])
    here = tm.object_centroids(labels[0])
    inside = labels[0] == 1
    dx = target["vector"][1][inside]
    diameter = here[1][2]
    assert np.allclose(dx.mean(), 12 / diameter, atol=0.05)
    assert target["successor"][inside].min() == 1


def test_an_object_that_leaves_has_no_successor_and_no_vector_loss():
    _images, labels = _movie()
    gone = labels[1].copy()
    gone[gone == 2] = 0
    target = tm.time_targets(labels[0], gone)
    inside = labels[0] == 2
    assert target["successor"][inside].max() == 0
    assert target["vector_weight"][inside].max() == 0
    assert target["object_weight"][inside].min() == 1


def test_augmentation_is_the_same_for_both_frames():
    images, labels = _movie()
    rng = np.random.default_rng(3)
    frames, labs = tm.augment_pair([images[0], images[1]], [labels[0], labels[1]], rng)
    before = tm.time_targets(labels[0], labels[1])
    after = tm.time_targets(labs[0], labs[1])
    moved = np.linalg.norm(before["vector"][:, labels[0] == 1].mean(1))
    moved_after = np.linalg.norm(after["vector"][:, labs[0] == 1].mean(1))
    assert np.isclose(moved, moved_after, atol=1e-4)
    assert (frames[0] > 0.5).sum() == (images[0] > 0.5).sum()


def test_sampling_weights_favour_the_rare_large_moves():
    """Six small steps then one large one: the large pair is its own bin and
    so carries as much weight as all six small ones together."""
    positions = [8, 9, 10, 11, 12, 13, 38]
    stack = np.zeros((len(positions), 64, 64), np.int32)
    for t, x in enumerate(positions):
        _disc((64, 64), 20, x, 5, 1, stack[t])
    weights = tm.pair_sampling_weights(stack, bins=2)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[-1] > weights[0]
    assert np.isclose(weights[-1], weights[:-1].sum())


def _independent_pairs(mixed_sizes):
    """Six small moves and one large move, with unrelated starting positions."""
    pairs = []
    for index, step in enumerate([1, 1, 1, 1, 1, 1, 14]):
        shape = (80, 96) if mixed_sizes and index % 2 else (64, 64)
        labels = np.zeros((2,) + shape, np.int32)
        x = 10 if index % 2 else 35
        for frame in (0, 1):
            _disc(shape, 24, x + frame * step, 5, 1, labels[frame])
        pairs.append(tm._Pair((labels[0] > 0).astype(np.float32),
                              (labels[1] > 0).astype(np.float32),
                              labels[0], labels[1]))
    return pairs


@pytest.mark.parametrize("mixed_sizes", [False, True])
def test_pair_weights_follow_real_endpoints_across_movie_boundaries(mixed_sizes):
    pairs = _independent_pairs(mixed_sizes)
    weights = tm._training_pair_sampling_weights(pairs)
    np.testing.assert_allclose(weights, [1 / 12] * 6 + [1 / 2])
    order = np.array([6, 2, 0, 4, 1, 5, 3])
    reordered = tm._training_pair_sampling_weights([pairs[index] for index in order])
    np.testing.assert_allclose(reordered, weights[order])


def test_independent_and_contiguous_sampling_agree_for_one_complete_movie():
    images, labels = _movie(frames=4, step=6)
    pairs = [tm._Pair(images[t], images[t + 1], labels[t], labels[t + 1])
             for t in range(3)]
    np.testing.assert_allclose(tm._training_pair_sampling_weights(pairs),
                               tm.pair_sampling_weights(labels))
    assert tm._training_pair_sampling_weights([]).shape == (0,)


@pytest.mark.parametrize("mixed_sizes", [False, True])
def test_cli_trains_with_pair_weights_and_records_them(tmp_path, monkeypatch, mixed_sizes):
    import json
    import sys
    import types

    pairs = _independent_pairs(mixed_sizes)
    monkeypatch.setattr(tm, "ctc_pairs", lambda movie, sequence, max_pairs: pairs if sequence == "01" else [])
    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(
        models=types.SimpleNamespace(CellposeModel=lambda **kwargs: types.SimpleNamespace(net=None))))
    monkeypatch.setattr(tm, "CellposeSamFeatures", lambda net: None)
    monkeypatch.setattr(tm, "TimeflowsNet", lambda net: types.SimpleNamespace(state_dict=lambda: {}))
    seen = {}

    def train(net, actual_pairs, **kwargs):
        assert actual_pairs == pairs
        seen.update(kwargs)
        return [0.1]

    monkeypatch.setattr(tm, "train_timeflows", train)
    target = tmp_path / "model.pt"
    assert tm.main(["--movies", "ctc-example", "--out", str(target), "--device", "cpu"]) == 0
    expected = [1 / 12] * 6 + [1 / 2]
    np.testing.assert_allclose(seen["weights"], expected)
    record = json.loads(target.with_suffix(".pt.json").read_text())
    assert record["sampling"]["strategy"] == "inverse_frequency_displacement_bins"
    assert record["sampling"]["bins"] == 5
    assert record["window_supervision"] == {
        "policy": "complete_source_and_present_successor_masks",
        "tile_size": tm.TILE, "maximum_attempts_per_step": 32,
        "full_frame_disappearances_supervised": True}
    assert record["annotation_assignment"] == {
        "policy": "one_object_per_track_one_track_per_object",
        "missing_successor_full_mask": "censor_source_supervision"}
    np.testing.assert_allclose(record["sampling"]["weights"], expected)


def test_ctc_markers_relabel_the_silver_masks():
    seg = np.zeros((20, 20), np.int32)
    seg[2:8, 2:8] = 5
    seg[10:18, 10:18] = 9
    seg[0:2, 15:20] = 3
    markers = np.zeros((20, 20), np.int32)
    markers[4:6, 4:6] = 101
    markers[13:15, 13:15] = 202
    out = tm.track_masks_from_ctc(seg, markers)
    assert set(np.unique(out)) == {0, 101, 202}
    assert (out == 101).sum() == 36


def _oracle(labels_t, labels_t1):
    target = tm.time_targets(labels_t, labels_t1)
    return lambda _a, _b: {"vector": target["vector"],
                           "successor": target["successor"]}


def test_a_perfect_prediction_recovers_scrambled_identities_iou_cannot():
    images, labels = _movie(step=14)
    result = tm.scramble_test(labels[0], labels[1], _oracle(labels[0], labels[1]),
                              images[0], images[1])
    assert result["model"] == 1.0
    assert result["iou"] < 1.0
    assert result["objects"] == 2


def test_training_runs_on_a_cpu_and_the_loss_falls():
    images, labels = _movie(frames=4, step=6)
    nn = torch.nn

    class Stub(nn.Module):
        channels, ps = 16, 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 8, stride=8)

        def forward(self, x):
            return self.conv(x)

    net = tm.TimeflowsNet(Stub())
    pairs = [tm._Pair(images[t], images[t + 1], labels[t], labels[t + 1])
             for t in range(3)]
    losses = tm.train_timeflows(net, pairs, head_steps=60, full_steps=40,
                                lr_head=3e-3, lr_full=1e-3, seed=1)
    assert len(losses) == 100
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    prediction = tm.predict_pair(net, images[0], images[1])
    assert prediction["vector"].shape == (2, 64, 64)


def test_a_ctc_movie_becomes_consecutive_track_labelled_pairs(tmp_path):
    import tifffile

    _images, labels = _movie(frames=3, step=4)
    movie = tmp_path / "ctc_movie"
    for sub in ("01", "01_ST/SEG", "01_GT/TRA"):
        (movie / sub).mkdir(parents=True)
    for t in range(3):
        tifffile.imwrite(movie / "01" / f"t{t:03d}.tif", (labels[t] > 0).astype(np.uint16) * 900)
        silver = np.where(labels[t] == 1, 7, np.where(labels[t] == 2, 3, 0)).astype(np.uint16)
        tifffile.imwrite(movie / "01_ST" / "SEG" / f"man_seg{t:03d}.tif", silver)
        markers = np.zeros_like(silver)
        for label, track in ((1, 11), (2, 22)):
            ys, xs = np.nonzero(labels[t] == label)
            markers[ys[len(ys) // 2], xs[len(xs) // 2]] = track
        tifffile.imwrite(movie / "01_GT" / "TRA" / f"man_track{t:03d}.tif", markers)
    pairs = tm.ctc_pairs(str(movie), "01")
    assert len(pairs) == 2
    assert set(np.unique(pairs[0].labels_t)) == {0, 11, 22}
    assert pairs[0].frame_t.max() <= 1.0


def test_a_training_window_is_the_encoder_tile_and_shared_by_both_frames():
    """2026-09-22: whole CTC frames failed the encoder's fixed 32 x 32 position
    embedding; training reads one shared 256 px window per pair."""
    import numpy as np

    from spacr import timeflows_model as tm

    rng = np.random.default_rng(0)
    frame = rng.random((600, 700)).astype(np.float32)
    labels = np.zeros((600, 700), dtype=np.int32)
    labels[300:320, 400:420] = 7
    moved = np.roll(labels, 5, axis=1)
    frames, labs = tm.random_window([frame, frame * 2], [labels, moved], rng)
    assert all(f.shape == (tm.TILE, tm.TILE) for f in frames + labs)
    assert np.allclose(frames[1], frames[0] * 2), "the same window of both frames"
    assert (labs[0] == 7).any(), "the window holds the object it was centred on"


def test_a_small_frame_is_padded_to_the_tile():
    import numpy as np

    from spacr import timeflows_model as tm

    rng = np.random.default_rng(1)
    frames, labs = tm.random_window([np.ones((100, 120), np.float32)] * 2,
                                    [np.zeros((100, 120), np.int32)] * 2, rng)
    assert frames[0].shape == (tm.TILE, tm.TILE)
