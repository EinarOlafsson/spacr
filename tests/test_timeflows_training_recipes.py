"""Timeflows training recipes: cosine schedule, forward-backward consistency,
group-balanced sampling and fine-tuning from a saved checkpoint."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from spacr import timeflows_model as tm

torch = pytest.importorskip("torch")


def _disc(shape, cy, cx, r, label, out):
    yy, xx = np.indices(shape)
    out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label


def _pair_labels(step=6, shape=(40, 40)):
    first = np.zeros(shape, np.int32)
    second = np.zeros(shape, np.int32)
    _disc(shape, 12, 10, 4, 1, first)
    _disc(shape, 12, 10 + step, 4, 1, second)
    _disc(shape, 28, 28, 4, 2, first)
    return first, second


class _Stub(torch.nn.Module):
    channels, ps = 8, 8

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 8, stride=8)

    def forward(self, x):
        return self.conv(x)


def test_cosine_factor_warms_up_then_decays_to_five_percent():
    steps = 200
    factors = [tm._cosine_factor(step, steps) for step in range(steps)]
    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert max(factors) == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.05, abs=1e-3)
    assert all(a >= b for a, b in zip(factors[10:], factors[11:]))
    assert tm._cosine_factor(0, 1) == pytest.approx(1.0)


def test_cycle_consistency_is_zero_when_displacements_cancel_and_positive_otherwise():
    first, second = _pair_labels()
    forward_target = tm.time_targets(first, second)["vector"]
    backward_target = tm.time_targets(second, first)["vector"]
    forward = torch.zeros(1, 3, *first.shape)
    backward = torch.zeros(1, 3, *first.shape)
    forward[0, :2] = torch.from_numpy(forward_target)
    backward[0, :2] = torch.from_numpy(backward_target)
    assert tm._cycle_consistency(forward, backward, first, second).item() == pytest.approx(0.0, abs=1e-6)
    diameter = tm.object_centroids(first)[1][2]
    same_way = torch.zeros_like(forward)
    same_way[0, 1][torch.from_numpy(second == 1)] = 6.0 / diameter
    forward.requires_grad_(True)
    value = tm._cycle_consistency(forward, same_way, first, second)
    assert value.item() == pytest.approx((12.0 / diameter) ** 2, rel=1e-4)
    value.backward()
    assert forward.grad[0, :2][:, first == 1].abs().sum() > 0
    assert forward.grad[0, :2][:, first == 2].abs().sum() == 0
    empty = tm._cycle_consistency(forward, backward, np.zeros_like(first), second)
    assert empty.item() == 0.0


def test_group_balancing_gives_each_group_an_equal_share_and_keeps_the_order_within():
    weights = np.array([1.0, 3.0, 1.0, 1.0, 1.0, 0.0])
    groups = ["mouse", "mouse", "human", "rat", "rat", "rat"]
    out = tm._group_balanced_weights(weights, groups)
    assert out.sum() == pytest.approx(1.0)
    for name in ("mouse", "human", "rat"):
        assert out[[g == name for g in groups]].sum() == pytest.approx(1 / 3)
    assert out[1] == pytest.approx(3 * out[0])
    assert out[5] == 0.0
    zero = tm._group_balanced_weights(np.zeros(2), ["a", "b"])
    np.testing.assert_allclose(zero, [0.5, 0.5])
    with pytest.raises(ValueError, match="one group per pair"):
        tm._group_balanced_weights(weights, groups[:2])


def test_training_with_cosine_and_consistency_runs_and_rejects_bad_options():
    first, second = _pair_labels(shape=(64, 64))
    images = [(first > 0).astype(np.float32), (second > 0).astype(np.float32)]
    pairs = [tm._Pair(images[0], images[1], first, second)]
    net = tm.TimeflowsNet(_Stub())
    losses = tm.train_timeflows(net, pairs, head_steps=6, full_steps=4,
                                lr_schedule="cosine", consistency_weight=0.5, seed=2)
    assert len(losses) == 10 and np.isfinite(losses).all()
    with pytest.raises(ValueError, match="lr_schedule"):
        tm.train_timeflows(net, pairs, head_steps=1, full_steps=0, lr_schedule="step")
    with pytest.raises(ValueError, match="consistency_weight"):
        tm.train_timeflows(net, pairs, head_steps=1, full_steps=0, consistency_weight=-1.0)


def test_cli_fine_tunes_from_a_checkpoint_balances_groups_and_records_the_recipe(tmp_path, monkeypatch):
    first, second = _pair_labels()
    pair = tm._Pair((first > 0).astype(np.float32), (second > 0).astype(np.float32), first, second)
    per_movie = {"m1": [pair, pair, pair], "m2": [pair]}
    monkeypatch.setattr(tm, "ctc_pairs", lambda movie, sequence, max_pairs, gaps=(1,):
                        per_movie[movie] if sequence == "01" else [])
    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(
        models=types.SimpleNamespace(CellposeModel=lambda **kwargs: types.SimpleNamespace(net=None))))
    monkeypatch.setattr(tm, "CellposeSamFeatures", lambda net: _Stub())
    saved = tm.TimeflowsNet(_Stub())
    with torch.no_grad():
        for parameter in saved.parameters():
            parameter.fill_(0.25)
    init = tmp_path / "v1.pt"
    torch.save(saved.state_dict(), init)
    seen = {}

    def train(net, actual_pairs, **kwargs):
        seen.update(kwargs, head=net.head[0].weight.detach().clone())
        return [0.2]

    monkeypatch.setattr(tm, "train_timeflows", train)
    target = tmp_path / "ft.pt"
    assert tm.main(["--movies", "m1", "m2", "--groups", "mouse", "human", "--init", str(init),
                    "--lr-schedule", "cosine", "--consistency-weight", "0.5",
                    "--lr-head", "3e-4", "--out", str(target), "--device", "cpu"]) == 0
    assert torch.all(seen["head"] == 0.25)
    assert seen["lr_schedule"] == "cosine" and seen["consistency_weight"] == 0.5
    assert seen["lr_head"] == 3e-4 and seen["lr_full"] == 1e-5
    np.testing.assert_allclose(seen["weights"], [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    record = json.loads(target.with_suffix(".pt.json").read_text())
    assert record["init"] == str(init) and record["lr_schedule"] == "cosine"
    assert record["consistency_weight"] == 0.5
    assert record["sampling"]["groups"] == ["mouse", "human"]
    with pytest.raises(SystemExit):
        tm.main(["--movies", "m1", "m2", "--groups", "mouse", "--out", str(target)])
