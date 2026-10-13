"""Crop boundaries must not teach false deaths or truncated motion targets."""
import numpy as np
import pytest

from spacr import timeflows_model as tm

torch = pytest.importorskip("torch")


class _Centered:
    def integers(self, low, high):
        return 0


class _Encoder(torch.nn.Module):
    channels, ps = 4, 8

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 8, stride=8)

    def forward(self, frame):
        return self.conv(frame)


def _pair(case):
    first = np.zeros((512, 512), np.int32)
    second = np.zeros_like(first)
    first[10:20, 10:20] = 1
    second[15:25, 15:25] = 1
    first[70:80, 70:80] = 30
    if case == "source_cut":
        first[40:50, 250:270] = 20
        second[40:50, 60:80] = 20
    else:
        first[40:50, 40:50] = 20
        if case == "target_cut":
            second[40:50, 250:270] = 20
        else:
            second[400:410, 400:410] = 20
    return tm._Pair(first.astype(np.float32), second.astype(np.float32), first, second)


@pytest.mark.parametrize("case", ["source_cut", "target_cut", "target_outside"])
@pytest.mark.parametrize("seed", [0, 3])
def test_actual_training_loss_excludes_crop_artifacts_but_keeps_true_death(monkeypatch, case, seed):
    pair = _pair(case)
    original_window = tm.random_window
    original_loss = tm.timeflows_loss
    seen = []

    def fixed_window(frames, labels, rng):
        return original_window(frames, labels, _Centered())

    def inspect(output, targets):
        seen.append({key: value.detach().cpu().numpy().copy() for key, value in targets.items()})
        return original_loss(output, targets)

    monkeypatch.setattr(tm, "random_window", fixed_window)
    monkeypatch.setattr(tm, "timeflows_loss", inspect)
    net = tm.TimeflowsNet(_Encoder())
    before = [array.copy() for array in (pair.frame_t, pair.frame_t1, pair.labels_t, pair.labels_t1)]
    losses = tm.train_timeflows(net, [pair], head_steps=1, full_steps=1, seed=seed)
    assert len(losses) == 2 and np.isfinite(losses).all()
    assert len(seen) == 2
    for targets in seen:
        assert targets["object_weight"].sum() == 200
        assert targets["successor"].sum() == 100
        assert targets["vector_weight"].sum() == 100
        assert ((targets["object_weight"] > 0) & (targets["successor"] == 0)).sum() == 100
    for actual, expected in zip((pair.frame_t, pair.frame_t1, pair.labels_t, pair.labels_t1), before):
        np.testing.assert_array_equal(actual, expected)


def test_no_usable_window_fails_before_weight_decay_or_any_parameter_update(monkeypatch):
    pair = _pair("target_outside")
    pair.labels_t[pair.labels_t != 20] = 0
    original_window = tm.random_window
    calls = []

    def fixed_window(frames, labels, rng):
        calls.append(1)
        return original_window(frames, labels, _Centered())

    monkeypatch.setattr(tm, "random_window", fixed_window)
    net = tm.TimeflowsNet(_Encoder())
    before = {key: value.clone() for key, value in net.state_dict().items()}
    with pytest.raises(ValueError, match="No usable temporal supervision in 32 sampled windows"):
        tm.train_timeflows(net, [pair], head_steps=1, full_steps=0)
    assert len(calls) == 32
    for key, expected in before.items():
        torch.testing.assert_close(net.state_dict()[key], expected, rtol=0, atol=0)


def test_a_rejected_crop_is_retried_and_a_valid_crop_can_train(monkeypatch):
    pair = _pair("target_outside")
    real_window = tm._training_window
    calls = []

    def first_empty(pair, rng):
        frames, labels = real_window(pair, rng)
        calls.append(1)
        if len(calls) == 1:
            labels[0] = np.zeros_like(labels[0])
        return frames, labels

    monkeypatch.setattr(tm, "_training_window", first_empty)
    losses = tm.train_timeflows(tm.TimeflowsNet(_Encoder()), [pair], head_steps=1, full_steps=0)
    assert len(calls) == 2
    assert len(losses) == 1 and np.isfinite(losses[0])


def test_small_padded_frames_keep_complete_motion_and_true_absence():
    first = np.zeros((32, 48), np.int32)
    first[2:6, 2:6] = 1
    first[12:16, 12:16] = 2
    second = np.zeros_like(first)
    second[5:9, 8:12] = 1
    pair = tm._Pair(first.astype(np.float32), second.astype(np.float32), first, second)
    frames, labels = tm._training_window(pair, _Centered())
    actual = tm.time_targets(*labels)
    expected = tm.time_targets(first, second)
    for key in actual:
        np.testing.assert_array_equal(actual[key][..., :32, :48], expected[key])
    assert frames[0].shape == labels[0].shape == (tm.TILE, tm.TILE)


def test_censoring_never_changes_a_contiguous_view_of_the_input():
    first = np.zeros((512, tm.TILE), np.int32)
    second = np.zeros_like(first)
    first[4:8, 4:8] = 1
    second[400:404, 4:8] = 1
    pair = tm._Pair(first.astype(np.float32), second.astype(np.float32), first, second)
    before = first.copy()
    frames, labels = tm._training_window(pair, _Centered())
    assert not labels[0].any()
    assert frames[0].any()
    np.testing.assert_array_equal(first, before)
