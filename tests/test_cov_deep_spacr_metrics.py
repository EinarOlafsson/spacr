"""Coverage fill for the metric helpers of :mod:`spacr.deep_spacr`.

Covers ``_to_numpy_labels``, ``_binary_metrics``, ``_multiclass_metrics``,
``evaluate_model_performance``, ``test_model_core`` and
``test_model_performance``.

Everything here is CPU-only and offline: the "models" are tiny fixed-weight
``nn.Module`` heads and the "loaders" are plain lists of
``(data, target, filenames)`` tuples, which is all the functions under test
require.

The one branch that no other test in the suite reaches is the one-hot target
normalization inside ``evaluate_model_performance``
(``if target.ndim == 2: target = target.argmax(dim=1)``); it is exercised by
``test_evaluate_multiclass_one_hot_targets_match_index_targets``.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (after importorskip on purpose)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Force the CPU device branch even on a CUDA-equipped machine.

    ``evaluate_model_performance`` / ``test_model_core`` both pick their
    device with ``torch.cuda.is_available()``; pinning it to False keeps
    these tests CPU-only and lets us compare against reference tensors
    computed on the CPU.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def _no_figures():
    """Never leak matplotlib figures out of a test in this module."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# Deterministic tiny heads / loaders
# ---------------------------------------------------------------------------

class _FixedLinear(nn.Module):
    """Linear head with deterministic weights (no randomness at test time)."""

    def __init__(self, in_dim: int = 4, out_dim: int = 3, seed: int = 0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(out_dim, in_dim, generator=g))
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


class _Flat1DLogits(nn.Module):
    """Head emitting a *1-D* logit vector of shape ``(N,)`` (no class axis)."""

    def forward(self, x):
        return x.sum(dim=1)


def _fixed_data(n: int, in_dim: int = 4, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, in_dim, generator=g)


# ---------------------------------------------------------------------------
# _to_numpy_labels
# ---------------------------------------------------------------------------

def test_to_numpy_labels_column_vector_is_not_treated_as_onehot():
    """A ``(N, 1)`` float tensor must fall through to the rounding branch.

    ``size(1) > 1`` is False, so argmax (which would return all zeros) must
    not be used; the float values are rounded instead.
    """
    from spacr.deep_spacr import _to_numpy_labels

    t = torch.tensor([[0.2], [0.7], [1.0], [0.5]], dtype=torch.float32)
    out = _to_numpy_labels(t)

    assert out.dtype == np.dtype(int)
    # numpy/torch round-half-to-even: 0.5 -> 0
    assert out.reshape(-1).tolist() == [0, 1, 1, 0]
    # ... and definitely not the all-zero argmax answer.
    assert out.reshape(-1).tolist() != [0, 0, 0, 0]


def test_to_numpy_labels_bool_tensor_casts_without_rounding():
    """A bool tensor is not floating point, so it takes the plain-cast branch."""
    from spacr.deep_spacr import _to_numpy_labels

    t = torch.tensor([False, True, True, False])
    out = _to_numpy_labels(t)

    assert out.dtype == np.dtype(int)
    assert out.tolist() == [0, 1, 1, 0]


def test_to_numpy_labels_detaches_grad_tensor():
    """Targets carrying grad history must still convert (``detach`` branch)."""
    from spacr.deep_spacr import _to_numpy_labels

    t = torch.tensor([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]], requires_grad=True)
    out = _to_numpy_labels(t)

    assert out.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# _binary_metrics
# ---------------------------------------------------------------------------

def test_binary_metrics_nan_f1_threshold_is_skipped():
    """A precision/recall pair of (0, 0) yields a NaN F1 that ``nanargmax`` must skip.

    y=[0,1] with probs=[0.9,0.1] produces f1 = [0.667, nan, 0.0]; the optimal
    threshold has to come from index 0, not from the NaN entry.
    """
    from spacr.deep_spacr import _binary_metrics

    m = _binary_metrics(np.array([0, 1]), np.array([0.9, 0.1]))

    assert m["optimal_threshold"] == pytest.approx(0.1)
    assert m["prauc"] == pytest.approx(0.25)
    # Predictions at the fixed 0.5 cut are exactly inverted -> everything wrong.
    assert m["accuracy"] == 0.0
    assert m["neg_accuracy"] == 0.0
    assert m["pos_accuracy"] == 0.0


def test_binary_metrics_empty_input_is_all_nan():
    """No samples at all: every accuracy collapses to NaN and prauc is NaN."""
    from spacr.deep_spacr import _binary_metrics

    m = _binary_metrics(np.array([], dtype=int), np.array([], dtype=float))

    assert np.isnan(m["accuracy"])
    assert np.isnan(m["neg_accuracy"])
    assert np.isnan(m["pos_accuracy"])
    assert np.isnan(m["prauc"])
    assert m["optimal_threshold"] == 0.5


def test_binary_metrics_column_vector_labels_are_flattened():
    """``(N, 1)`` labels are reshaped to 1-D and give identical metrics.

    Without the reshape the ``pred == y_true`` comparison would broadcast to
    an ``(N, N)`` matrix and the accuracy would be wrong.
    """
    from spacr.deep_spacr import _binary_metrics

    probs = np.array([0.2, 0.7, 0.9, 0.4])
    flat = _binary_metrics(np.array([0, 1, 1, 0]), probs)
    column = _binary_metrics(np.array([[0], [1], [1], [0]]), probs)

    assert column["accuracy"] == 1.0
    assert column == flat


def test_binary_metrics_probability_exactly_at_threshold_counts_positive():
    """The 0.5 cut is inclusive (``>= 0.5``), which decides the tie case."""
    from spacr.deep_spacr import _binary_metrics

    m = _binary_metrics(np.array([0, 1, 1, 0]),
                        np.array([0.5, 0.5, 0.9, 0.05]))

    # preds -> [1, 1, 1, 0]; only the first sample is wrong.
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["neg_accuracy"] == pytest.approx(0.5)
    assert m["pos_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# _multiclass_metrics
# ---------------------------------------------------------------------------

def test_multiclass_metrics_empty_inputs_hit_average_precision_failure():
    """With zero rows ``average_precision_score`` raises -> prauc must be NaN."""
    from spacr.deep_spacr import _multiclass_metrics

    m = _multiclass_metrics(np.array([], dtype=int), np.empty((0, 3)))

    assert np.isnan(m["accuracy"])
    assert np.isnan(m["prauc"])          # the ``except`` branch fired
    assert m["num_classes"] == 3
    assert m["per_class_accuracy"] == [0.0, 0.0, 0.0]
    assert np.isnan(m["neg_accuracy"]) and np.isnan(m["pos_accuracy"])
    assert np.isnan(m["optimal_threshold"])


def test_multiclass_metrics_accuracy_and_macro_ap_on_perfect_predictions():
    """A perfectly separated 3-class problem: accuracy 1.0 and macro-AP 1.0."""
    from spacr.deep_spacr import _multiclass_metrics

    y = np.array([0, 1, 2, 1])
    prob = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.10, 0.80, 0.10],
    ])
    m = _multiclass_metrics(y, prob)

    assert m["accuracy"] == 1.0
    assert m["prauc"] == pytest.approx(1.0)   # macro average precision
    assert m["num_classes"] == 3


def test_multiclass_metrics_counts_wrong_predictions():
    """One misclassified row out of four -> accuracy 0.75 and macro-AP 7/9.

    Per-class AP is 0.5 (class 0: a negative outranks the positive), 1.0
    (class 1) and 5/6 (class 2), so the macro average is 7/9.
    """
    from spacr.deep_spacr import _multiclass_metrics

    y = np.array([0, 1, 2, 2])
    prob = np.array([
        [0.50, 0.30, 0.20],   # correct  (class 0)
        [0.10, 0.80, 0.10],   # correct  (class 1)
        [0.10, 0.10, 0.80],   # correct  (class 2)
        [0.80, 0.05, 0.15],   # wrong: says 0, truth is 2
    ])
    m = _multiclass_metrics(y, prob)

    assert m["accuracy"] == pytest.approx(0.75)
    assert m["prauc"] == pytest.approx(7.0 / 9.0)
    assert len(m["per_class_accuracy"]) == 3


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _multiclass_metrics per-class accuracy uses "
           "cm.sum(axis=1, where=..., initial=1), so every row sum is "
           "inflated by 1 and a perfectly classified class scores <1.0",
)
def test_multiclass_metrics_per_class_accuracy_perfect_is_one():
    """Per-class accuracy of a perfect classifier must be 1.0 for every class.

    ``np.sum(..., initial=1)`` adds 1 to *every* row sum (it is not a
    divide-by-zero guard), so the reported per-class accuracies are
    ``diag / (rowsum + 1)`` — e.g. [0.5, 0.667, 0.5] instead of [1, 1, 1].
    """
    from spacr.deep_spacr import _multiclass_metrics

    y = np.array([0, 1, 2, 1])
    prob = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.10, 0.80, 0.10],
    ])
    m = _multiclass_metrics(y, prob)

    assert m["per_class_accuracy"] == pytest.approx([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# evaluate_model_performance
# ---------------------------------------------------------------------------

def test_evaluate_multiclass_one_hot_targets_match_index_targets():
    """One-hot ``(N, C)`` targets must be argmax-ed into class indices.

    This is the ``if target.ndim == 2: target = target.argmax(dim=1)`` branch.
    Running the identical model/data with index targets must produce
    bit-identical metrics and labels.
    """
    from spacr.deep_spacr import evaluate_model_performance

    model = _FixedLinear(in_dim=4, out_dim=3, seed=3)
    data = _fixed_data(6)
    idx_target = torch.tensor([0, 2, 1, 1, 0, 2])
    one_hot = torch.nn.functional.one_hot(idx_target, num_classes=3).float()

    idx_metrics, (idx_probs, idx_labels) = evaluate_model_performance(
        model, [(data, idx_target, ["a"] * 6)], epoch=4)
    oh_metrics, (oh_probs, oh_labels) = evaluate_model_performance(
        model, [(data, one_hot, ["a"] * 6)], epoch=4)

    # The one-hot run recovered the class indices ...
    assert oh_labels == idx_target.tolist()
    assert oh_labels == idx_labels
    # ... and produced exactly the same probabilities/metrics.
    assert oh_probs.shape == (6, 3)
    np.testing.assert_allclose(oh_probs, idx_probs)
    np.testing.assert_allclose(oh_probs.sum(axis=1), np.ones(6), atol=1e-6)
    assert oh_metrics["loss"] == pytest.approx(idx_metrics["loss"])
    assert oh_metrics["accuracy"] == pytest.approx(idx_metrics["accuracy"])
    assert oh_metrics["num_classes"] == 3
    assert oh_metrics["epoch"] == 4
    assert oh_metrics["Accuracy"] == oh_metrics["accuracy"]


def test_evaluate_multiclass_one_hot_targets_produce_finite_ce_loss():
    """The argmax-ed one-hot target must still be a valid CE target.

    If the one-hot tensor reached ``F.cross_entropy`` unconverted the call
    would either raise or silently compute a soft-label loss; here we pin the
    loss against a hand-computed cross entropy on the argmax indices.
    """
    import torch.nn.functional as F
    from spacr.deep_spacr import evaluate_model_performance

    model = _FixedLinear(in_dim=4, out_dim=3, seed=11)
    data = _fixed_data(5, seed=21)
    idx_target = torch.tensor([2, 0, 1, 2, 0])
    one_hot = torch.nn.functional.one_hot(idx_target, num_classes=3).float()

    metrics, (probs, labels) = evaluate_model_performance(
        model, [(data, one_hot, ["x"] * 5)], epoch=0)

    with torch.no_grad():
        expected = F.cross_entropy(model(data), idx_target).item()

    assert np.isfinite(metrics["loss"])
    assert metrics["loss"] == pytest.approx(expected, rel=1e-5)
    assert labels == [2, 0, 1, 2, 0]
    assert probs.shape == (5, 3)


def test_evaluate_binary_accepts_column_targets_and_thresholds_at_half():
    """Binary mode takes ``(N, 1)`` float targets and binarises at 0.5."""
    from spacr.deep_spacr import evaluate_model_performance

    model = _FixedLinear(in_dim=4, out_dim=1, seed=5)
    data = _fixed_data(4, seed=9)
    target = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    metrics, (probs, labels) = evaluate_model_performance(
        model, [(data, target, ["f"] * 4)], epoch=2)

    assert labels == [0, 1, 1, 0]
    assert probs.shape == (4,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert np.isfinite(metrics["loss"])
    assert metrics["epoch"] == 2


def test_evaluate_uses_supplied_loss_fn_verbatim():
    """When ``loss_fn`` is given, ``build_loss`` must not be consulted."""
    from spacr.deep_spacr import evaluate_model_performance

    calls = []

    def _const_loss(logits, target):
        calls.append((tuple(logits.shape), tuple(target.shape)))
        return torch.tensor(3.5)

    model = _FixedLinear(in_dim=4, out_dim=3, seed=1)
    loader = [
        (_fixed_data(4, seed=1), torch.tensor([0, 1, 2, 1]), ["a"] * 4),
        (_fixed_data(2, seed=2), torch.tensor([2, 0]), ["b"] * 2),
    ]

    metrics, (probs, labels) = evaluate_model_performance(
        model, loader, epoch=7, loss_fn=_const_loss)

    assert len(calls) == 2                       # called once per batch
    assert calls[0] == ((4, 3), (4,))
    # sample-weighted mean of a constant loss is the constant itself
    assert metrics["loss"] == pytest.approx(3.5)
    assert len(labels) == 6
    assert probs.shape == (6, 3)


def test_evaluate_empty_loader_without_num_classes_returns_1d_probs():
    """An empty loader with no ``num_classes`` hint falls back to binary rank."""
    from spacr.deep_spacr import evaluate_model_performance

    metrics, (probs, labels) = evaluate_model_performance(
        _FixedLinear(out_dim=1), [], epoch=9)

    assert probs.shape == (0,)
    assert probs.ndim == 1
    assert labels == []
    assert metrics["loss"] == 0.0                # total_loss / max(1, 0)
    assert metrics["epoch"] == 9
    assert np.isnan(metrics["accuracy"])


def test_evaluate_empty_loader_with_num_classes_returns_2d_probs():
    """``num_classes >= 2`` on an empty loader yields an ``(0, C)`` array."""
    from spacr.deep_spacr import evaluate_model_performance

    metrics, (probs, labels) = evaluate_model_performance(
        _FixedLinear(out_dim=4), [], epoch=1, num_classes=4)

    assert probs.shape == (0, 4)
    assert metrics["num_classes"] == 4
    assert np.isnan(metrics["prauc"])
    assert metrics["Accuracy"] is metrics["Accuracy"]  # key present
    assert np.isnan(metrics["Accuracy"])


def test_evaluate_loss_is_sample_weighted_across_uneven_batches():
    """Batches of different size must be weighted by their sample count."""
    from spacr.deep_spacr import evaluate_model_performance

    seen = []

    def _loss_by_size(logits, target):
        n = logits.size(0)
        seen.append(n)
        return torch.tensor(float(n))          # loss == batch size

    model = _FixedLinear(in_dim=4, out_dim=2, seed=4)
    loader = [
        (_fixed_data(1, seed=31), torch.tensor([0]), ["a"]),
        (_fixed_data(3, seed=32), torch.tensor([1, 0, 1]), ["b", "c", "d"]),
    ]

    metrics, _ = evaluate_model_performance(
        model, loader, epoch=0, loss_fn=_loss_by_size)

    assert seen == [1, 3]
    # (1*1 + 3*3) / 4 == 2.5
    assert metrics["loss"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# test_model_core
# ---------------------------------------------------------------------------

def test_model_core_handles_1d_logit_head():
    """A head returning shape ``(N,)`` is treated as a single-logit binary head."""
    from spacr.deep_spacr import test_model_core

    data = _fixed_data(5, seed=13)
    target = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    names = [f"img_{i}.png" for i in range(5)]

    metrics, probs, labels, df = test_model_core(
        _Flat1DLogits(), [(data, target, names)], "test", epoch=3,
        loss_type="auto")

    expected_probs = torch.sigmoid(data.sum(dim=1)).numpy()
    assert probs.ndim == 1
    np.testing.assert_allclose(probs, expected_probs, rtol=1e-6)
    assert labels == [0, 1, 1, 0, 1]
    assert list(df.columns) == ["filename", "true_label", "predicted_label",
                                "class_1_probability"]
    assert df["filename"].tolist() == names
    assert df["predicted_label"].tolist() == (expected_probs >= 0.5).astype(int).tolist()
    assert metrics["epoch"] == 3
    assert np.isfinite(metrics["loss"])


def test_model_core_empty_loader_produces_empty_frame():
    """No batches at all: empty (0, 1) prob matrix, NaN accuracy, empty frame."""
    from spacr.deep_spacr import test_model_core

    metrics, probs, labels, df = test_model_core(
        _FixedLinear(out_dim=1), [], "test", epoch=0, loss_type="auto")

    assert probs.shape == (0,)
    assert labels == []
    assert len(df) == 0
    assert list(df.columns) == ["filename", "true_label", "predicted_label",
                                "class_1_probability"]
    assert metrics["loss"] == 0.0
    assert np.isnan(metrics["accuracy"])
    assert np.isnan(metrics["Accuracy"])


def test_model_core_multiclass_emits_one_column_per_class():
    """Multiclass path: probs are softmax rows and each class gets a column."""
    from spacr.deep_spacr import test_model_core

    model = _FixedLinear(in_dim=4, out_dim=3, seed=6)
    data = _fixed_data(4, seed=17)
    target = torch.tensor([0, 1, 2, 2])
    names = ["a.png", "b.png", "c.png", "d.png"]

    metrics, probs, labels, df = test_model_core(
        model, [(data, target, names)], "test", epoch=5, loss_type="auto")

    with torch.no_grad():
        expected = torch.softmax(model(data), dim=1).numpy()

    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs, expected, rtol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)
    assert labels == [0, 1, 2, 2]
    for k in range(3):
        assert f"prob_class_{k}" in df.columns
    assert "class_1_probability" not in df.columns
    assert df["predicted_label"].tolist() == expected.argmax(1).tolist()
    assert metrics["num_classes"] == 3
    assert metrics["Accuracy"] == metrics["accuracy"]


def test_model_core_concatenates_multiple_batches_in_order():
    """Rows from consecutive batches are stacked in loader order."""
    from spacr.deep_spacr import test_model_core

    model = _FixedLinear(in_dim=4, out_dim=1, seed=2)
    loader = [
        (_fixed_data(2, seed=41), torch.tensor([1.0, 0.0]), ["x0.png", "x1.png"]),
        (_fixed_data(3, seed=42), torch.tensor([0.0, 1.0, 1.0]),
         ["y0.png", "y1.png", "y2.png"]),
    ]

    metrics, probs, labels, df = test_model_core(
        model, loader, "test", epoch=1, loss_type="auto")

    assert df["filename"].tolist() == ["x0.png", "x1.png",
                                        "y0.png", "y1.png", "y2.png"]
    assert labels == [1, 0, 0, 1, 1]
    assert probs.shape == (5,)
    assert df["true_label"].tolist() == labels


def test_model_core_multiclass_accepts_one_hot_targets():
    """``_to_numpy_labels`` turns one-hot targets into indices for the frame."""
    from spacr.deep_spacr import test_model_core

    model = _FixedLinear(in_dim=4, out_dim=3, seed=8)
    data = _fixed_data(4, seed=19)
    idx = torch.tensor([2, 0, 1, 2])
    one_hot = torch.nn.functional.one_hot(idx, num_classes=3).float()

    metrics, probs, labels, df = test_model_core(
        model, [(data, one_hot, ["p", "q", "r", "s"])], "test", epoch=0,
        loss_type="auto")

    assert labels == [2, 0, 1, 2]
    assert df["true_label"].tolist() == [2, 0, 1, 2]
    assert probs.shape == (4, 3)


# ---------------------------------------------------------------------------
# test_model_performance
# ---------------------------------------------------------------------------

def test_model_performance_wrapper_matches_core_output():
    """The wrapper is a thin shim: one summary row + the per-file frame."""
    from spacr.deep_spacr import test_model_core, test_model_performance

    model = _FixedLinear(in_dim=4, out_dim=1, seed=12)
    data = _fixed_data(6, seed=23)
    target = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    names = [f"n{i}.png" for i in range(6)]

    core_metrics, _, _, core_df = test_model_core(
        model, [(data, target, names)], "test", epoch=2, loss_type="auto")
    result_df, results_df = test_model_performance(
        [(data, target, names)], model, ["test"], epoch=2, loss_type="auto")

    assert len(result_df) == 1
    assert result_df.loc[0, "epoch"] == 2
    assert result_df.loc[0, "loss"] == pytest.approx(core_metrics["loss"])
    assert result_df.loc[0, "Accuracy"] == pytest.approx(core_metrics["accuracy"])
    assert {"accuracy", "prauc", "optimal_threshold"} <= set(result_df.columns)
    assert results_df["filename"].tolist() == names
    assert results_df["true_label"].tolist() == core_df["true_label"].tolist()


def test_model_performance_wrapper_multiclass_summary_row():
    """Multiclass summary rows carry the per-class accuracy list and class count."""
    from spacr.deep_spacr import test_model_performance

    model = _FixedLinear(in_dim=4, out_dim=3, seed=15)
    data = _fixed_data(6, seed=29)
    target = torch.tensor([0, 1, 2, 0, 1, 2])

    result_df, results_df = test_model_performance(
        [(data, target, [f"m{i}.png" for i in range(6)])], model, ["test"],
        epoch=11, loss_type="auto")

    assert len(result_df) == 1
    assert result_df.loc[0, "num_classes"] == 3
    assert len(result_df.loc[0, "per_class_accuracy"]) == 3
    assert result_df.loc[0, "epoch"] == 11
    assert len(results_df) == 6
    assert {"prob_class_0", "prob_class_1", "prob_class_2"} <= set(results_df.columns)
