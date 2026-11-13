"""Evaluation metrics of ``spacr.deep_spacr``: label conversion, the binary and
multiclass metric dicts, and the three evaluation passes built on them.

One table-driven test per unit, each row run as a subtest so a failing case
names itself.  Everything is CPU-only and offline: the "models" are tiny
fixed-weight linear heads and the "loaders" are plain lists of
``(data, target, filenames)`` tuples, which is all the passes iterate.

Units: _to_numpy_labels, _binary_metrics, _multiclass_metrics,
evaluate_model_performance, test_model_core and test_model_performance.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from spacr import deep_spacr as D  # noqa: E402
from spacr import utils as U  # noqa: E402

NAN = float("nan")

#: What ``pick_device`` says when a busy card sent the run to the CPU.
_NOTE = ("The GPU has 300 MiB free of 24576 and this run needs about 2000 "
         "MiB, so it is running on the CPU instead.")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Pin every pass to the CPU, whatever an earlier test cached."""
    import spacr.accelerator as accelerator

    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _same(got, want):
    """Exact equality for metric dicts, with NaN equal to NaN."""
    assert set(got) == set(want), sorted(set(got) ^ set(want))
    for key, value in want.items():
        g = got[key]
        if isinstance(value, float) and math.isnan(value):
            assert isinstance(g, float) and math.isnan(g), (key, g)
        elif isinstance(value, float):
            assert g == pytest.approx(value, abs=1e-12), (key, g, value)
        else:
            assert g == value, (key, g, value)


class _Fixed(nn.Module):
    """Linear head with seeded weights and zero bias."""

    def __init__(self, out_dim=3, in_dim=4, seed=0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(out_dim, in_dim, generator=g))
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


class _Flat(nn.Module):
    """Head returning a 1-D logit vector ``(N,)``: the row sum."""

    def forward(self, x):
        return x.sum(dim=1)


class _Stop(Exception):
    """Raised by a model once the device notice has been printed."""


class _Refuses(nn.Module):
    """A model whose ``eval`` stops the pass."""

    def eval(self):
        raise _Stop("stopped after the notice")


def _data(n, seed=7, in_dim=4):
    return torch.randn(n, in_dim, generator=torch.Generator().manual_seed(seed))


# ===========================================================================
# _to_numpy_labels
# ===========================================================================

def test_to_numpy_labels_turns_every_target_encoding_into_class_ids(subtests):
    """One-hot rows are arg-maxed, 1-D or column floats are rounded half to
    even, bool and integer tensors are cast, and grad history is detached."""
    cases = [
        ("one-hot", torch.tensor([[0.1, 0.9], [0.8, 0.2]]), [1, 0]),
        ("soft-three-class-with-grad",
         torch.tensor([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]], requires_grad=True), [1, 0]),
        ("float-1d-rounds", torch.tensor([0.2, 0.7, 1.4]), [0, 1, 1]),
        ("half-rounds-to-even", torch.tensor([0.5, 1.5, 2.5]), [0, 2, 2]),
        ("column-is-not-one-hot", torch.tensor([[0.2], [0.7], [1.0], [0.5]]),
         [[0], [1], [1], [0]]),
        ("bool", torch.tensor([False, True, True, False]), [0, 1, 1, 0]),
        ("int64", torch.tensor([0, 2, 1], dtype=torch.int64), [0, 2, 1]),
        ("int32", torch.tensor([3, 0], dtype=torch.int32), [3, 0]),
        ("empty", torch.tensor([], dtype=torch.long), []),
    ]
    for cid, target, want in cases:
        with subtests.test(case=cid):
            out = D._to_numpy_labels(target)
            assert isinstance(out, np.ndarray)
            assert out.dtype == np.dtype(int)
            assert out.tolist() == want


# ===========================================================================
# _binary_metrics
# ===========================================================================

def _binary(acc, neg, pos, prauc, thr, f1, per, support):
    return {"accuracy": acc, "neg_accuracy": neg, "pos_accuracy": pos,
            "prauc": prauc, "optimal_threshold": thr, "f1_macro": f1,
            "per_class_accuracy": per, "class_support": support,
            "num_classes": 2}


def test_binary_metrics_scores_every_label_mix_exactly(subtests):
    """Accuracy cuts at an inclusive 0.5, the PR-AUC and F1-optimal threshold
    need both classes (a NaN F1 is skipped), a missing class has NaN accuracy
    but 0.0 per-class accuracy, column labels are flattened and an empty split
    is NaN throughout."""
    cases = [
        ("perfect", [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
         _binary(1.0, 1.0, 1.0, 1.0, 0.6, 1.0, [1.0, 1.0], [2, 2])),
        ("inverted-nan-f1-is-skipped", [0, 1], [0.9, 0.1],
         _binary(0.0, 0.0, 0.0, 0.25, 0.1, 0.0, [0.0, 0.0], [1, 1])),
        ("tie-at-half-counts-positive", [0, 1, 1, 0], [0.5, 0.5, 0.9, 0.05],
         _binary(0.75, 0.5, 1.0, 11 / 12, 0.5, 11 / 15, [0.5, 1.0], [2, 2])),
        ("only-positives", [1, 1, 1], [0.6, 0.7, 0.8],
         _binary(1.0, NAN, 1.0, NAN, 0.5, 1.0, [0.0, 1.0], [0, 3])),
        ("only-negatives", [0, 0, 0], [0.2, 0.6, 0.1],
         _binary(2 / 3, 2 / 3, NAN, NAN, 0.5, 0.4, [2 / 3, 0.0], [3, 0])),
        ("empty", [], [],
         _binary(NAN, NAN, NAN, NAN, 0.5, NAN, [0.0, 0.0], [0, 0])),
    ]
    for cid, y, p, want in cases:
        with subtests.test(case=cid):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = D._binary_metrics(np.array(y, dtype=int),
                                        np.array(p, dtype=float))
            _same(got, want)

    with subtests.test(case="column-labels-equal-flat-labels"):
        probs = np.array([0.2, 0.7, 0.9, 0.4])
        flat = D._binary_metrics(np.array([0, 1, 1, 0]), probs)
        column = D._binary_metrics(np.array([[0], [1], [1], [0]]), probs)
        assert column["accuracy"] == 1.0
        _same(column, flat)


# ===========================================================================
# _multiclass_metrics
# ===========================================================================

def _multi(acc, prauc, f1, per, support):
    return {"accuracy": acc, "neg_accuracy": NAN, "pos_accuracy": NAN,
            "prauc": prauc, "optimal_threshold": NAN, "f1_macro": f1,
            "per_class_accuracy": per, "class_support": support,
            "num_classes": len(per)}


def test_multiclass_metrics_scores_every_split_exactly(subtests):
    """Accuracy, macro one-vs-rest average precision, macro F1 and per-class
    accuracy = diagonal / row sum (0.0 for a class with no samples); an empty
    split answers NaN with the class schema of the probability matrix."""
    sharp = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05],
                      [0.05, 0.05, 0.90], [0.10, 0.80, 0.10]])
    cases = [
        ("perfect", [0, 1, 2, 1], sharp,
         _multi(1.0, 1.0, 1.0, [1.0, 1.0, 1.0], [1, 2, 1])),
        ("perfect-two-of-class-0", [0, 1, 2, 0],
         np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6],
                   [0.7, 0.2, 0.1]]),
         _multi(1.0, 1.0, 1.0, [1.0, 1.0, 1.0], [2, 1, 1])),
        ("one-wrong-of-four", [0, 1, 2, 2],
         np.array([[0.50, 0.30, 0.20], [0.10, 0.80, 0.10],
                   [0.10, 0.10, 0.80], [0.80, 0.05, 0.15]]),
         _multi(0.75, 7 / 9, 7 / 9, [1.0, 1.0, 0.5], [1, 1, 2])),
        ("class-without-samples-scores-zero", [0, 0, 1], np.eye(3)[[0, 0, 1]],
         _multi(1.0, 2 / 3, 1.0, [1.0, 1.0, 0.0], [2, 1, 0])),
        ("all-wrong", [0, 0, 1, 1], np.eye(2)[[1, 1, 0, 0]],
         _multi(0.0, 0.5, 0.0, [0.0, 0.0], [2, 2])),
        ("one-row-one-hot-fill", [0], np.full((1, 2), 0.5),
         _multi(1.0, 0.5, 1.0, [1.0, 0.0], [1, 0])),
        ("seven-rows-one-hot-fill", [0] * 7, np.full((7, 2), 0.5),
         _multi(1.0, 0.5, 1.0, [1.0, 0.0], [7, 0])),
    ]
    for cid, y, prob, want in cases:
        with subtests.test(case=cid):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = D._multiclass_metrics(np.array(y, dtype=int), prob)
            _same(got, want)

    for cid, dtype, classes in (("empty-int-1", int, 1), ("empty-int-3", int, 3),
                                ("empty-int-4", int, 4), ("empty-int-5", int, 5),
                                ("empty-float-labels", float, 2)):
        with subtests.test(case=cid):
            got = D._multiclass_metrics(np.array([], dtype=dtype),
                                        np.zeros((0, classes)))
            _same(got, {"accuracy": NAN, "neg_accuracy": NAN,
                        "pos_accuracy": NAN, "prauc": NAN,
                        "optimal_threshold": NAN, "f1_macro": NAN,
                        "per_class_accuracy": [0.0] * classes,
                        "class_support": [0] * classes,
                        "num_classes": classes})

    with subtests.test(case="average-precision-failure-is-logged-as-nan"):
        import logging

        records = []

        class _Keep(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Keep()
        log = logging.getLogger("spacr.deep_spacr")
        log.addHandler(handler)
        try:
            # a diverged model: NaN probabilities, which sklearn refuses
            got = D._multiclass_metrics(np.array([0, 1]),
                                        np.array([[NAN, 0.5], [0.2, 0.8]]))
        finally:
            log.removeHandler(handler)
        assert math.isnan(got["prauc"])
        assert got["accuracy"] == 1.0
        assert got["class_support"] == [1, 1]
        assert len(records) == 1
        assert records[0].startswith("macro average-precision could not be computed (")
        assert records[0].endswith("prauc will be NaN for this evaluation")


# ===========================================================================
# evaluate_model_performance
# ===========================================================================

def test_evaluate_model_performance_scores_binary_and_multiclass_heads(subtests, capsys,
                                                                       monkeypatch):
    """Softmax/CE for a multi-logit head (one-hot targets arg-maxed), sigmoid/BCE
    with a 0.5 label cut for a single logit, the loss sample-weighted across
    batches, a supplied loss used verbatim, an empty loader shaped by
    ``num_classes`` and the device notice printed first."""
    idx = torch.tensor([0, 2, 1, 1, 0, 2])
    one_hot = F.one_hot(idx, num_classes=3).float()
    data6 = _data(6)
    for cid, target in (("index-targets", idx), ("one-hot-targets", one_hot)):
        with subtests.test(case=cid):
            model = _Fixed(3, seed=3)
            with torch.no_grad():
                logits = model(data6)
            metrics, (probs, labels) = D.evaluate_model_performance(
                model, [(data6, target, ["a"] * 6)], epoch=4)
            np.testing.assert_allclose(probs, torch.softmax(logits, 1).numpy(),
                                       rtol=1e-6)
            assert labels == [0, 2, 1, 1, 0, 2]
            assert metrics["loss"] == pytest.approx(
                F.cross_entropy(logits, idx).item(), rel=1e-6)
            want = D._multiclass_metrics(np.array(labels), probs)
            for key in ("accuracy", "prauc", "f1_macro", "per_class_accuracy",
                        "class_support", "num_classes"):
                assert metrics[key] == want[key]
            assert metrics["epoch"] == 4
            assert metrics["Accuracy"] == metrics["accuracy"]

    for cid, target in (("binary-column-targets", torch.tensor([[0.0], [1.0], [1.0], [0.0]])),
                        ("binary-soft-targets-cut-at-half",
                         torch.tensor([[0.2], [0.9], [0.6], [0.4]]))):
        with subtests.test(case=cid):
            model = _Fixed(1, seed=5)
            data = _data(4, seed=9)
            with torch.no_grad():
                logits = model(data)
            metrics, (probs, labels) = D.evaluate_model_performance(
                model, [(data, target, ["f"] * 4)], epoch=2)
            assert labels == [0, 1, 1, 0]
            assert probs.shape == (4,)
            np.testing.assert_allclose(probs, torch.sigmoid(logits.view(-1)).numpy(),
                                       rtol=1e-6)
            assert metrics["loss"] == pytest.approx(
                F.binary_cross_entropy_with_logits(logits, target).item(), rel=1e-6)
            want = D._binary_metrics(np.array(labels), probs)
            for key in ("accuracy", "neg_accuracy", "pos_accuracy",
                        "optimal_threshold", "class_support"):
                assert metrics[key] == want[key]
            assert metrics["epoch"] == 2

    with subtests.test(case="supplied-loss-sample-weighted"):
        seen = []

        def by_size(logits, target):
            seen.append((tuple(logits.shape), tuple(target.shape)))
            return torch.tensor(float(logits.size(0)))

        loader = [(_data(1, seed=31), torch.tensor([0]), ["a"]),
                  (_data(3, seed=32), torch.tensor([1, 0, 1]), ["b", "c", "d"])]
        metrics, (probs, labels) = D.evaluate_model_performance(
            _Fixed(2, seed=4), loader, epoch=0, loss_fn=by_size)
        assert seen == [((1, 2), (1,)), ((3, 2), (3,))]
        assert metrics["loss"] == pytest.approx((1 * 1 + 3 * 3) / 4)
        assert labels == [0, 1, 0, 1]
        assert probs.shape == (4, 2)

    for cid, kw, model, shape, schema in (
            ("empty-no-hint-is-binary", {}, _Fixed(1), (0,), 2),
            ("empty-one-class", {"num_classes": 1}, _Fixed(1), (0,), 2),
            ("empty-three-classes", {"num_classes": 3}, _Fixed(3), (0, 3), 3),
            ("empty-four-classes", {"num_classes": 4}, _Fixed(4), (0, 4), 4)):
        with subtests.test(case=cid):
            metrics, (probs, labels) = D.evaluate_model_performance(
                model, [], epoch=9, **kw)
            assert probs.shape == shape
            assert labels == []
            assert metrics["loss"] == 0.0
            assert metrics["epoch"] == 9
            assert metrics["num_classes"] == schema
            assert metrics["class_support"] == [0] * schema
            assert math.isnan(metrics["accuracy"])
            assert math.isnan(metrics["Accuracy"])
            assert math.isnan(metrics["prauc"])

    with subtests.test(case="device-notice-comes-first"):
        capsys.readouterr()
        with monkeypatch.context() as mp:
            mp.setattr(D, "pick_device", lambda *a, **k: (torch.device("cpu"), _NOTE))
            with pytest.raises(_Stop):
                D.evaluate_model_performance(_Refuses(), [], epoch=1)
        assert capsys.readouterr().out == _NOTE + "\n"


# ===========================================================================
# test_model_core
# ===========================================================================

_BINARY_COLUMNS = ["filename", "true_label", "predicted_label", "class_1_probability"]


def test_test_model_core_builds_the_per_image_frame(subtests, capsys, monkeypatch):
    """Rows stack in loader order; a 1-D or single-logit head gives sigmoid
    probabilities cut at 0.5 and one ``class_1_probability`` column, a wider
    head softmax rows and one ``prob_class_<k>`` column per class; the loss is
    always the focal loss and ``majority_class_accuracy`` is the commonest
    label's share."""
    data5 = _data(5, seed=13)
    t5 = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    two = [(_data(2, seed=41), torch.tensor([1.0, 0.0]), ["x0.png", "x1.png"]),
           (_data(3, seed=42), torch.tensor([0.0, 1.0, 1.0]),
            ["y0.png", "y1.png", "y2.png"])]
    idx = torch.tensor([2, 0, 1, 2])
    cases = [
        # (id, model, loader, labels, n_classes)
        ("flat-1d-head", _Flat(), [(data5, t5, [f"img_{i}.png" for i in range(5)])],
         [0, 1, 1, 0, 1], 1),
        ("single-logit-two-batches", _Fixed(1, seed=2), two, [1, 0, 0, 1, 1], 1),
        ("three-class-index", _Fixed(3, seed=6),
         [(_data(4, seed=17), torch.tensor([0, 1, 2, 2]), ["a", "b", "c", "d"])],
         [0, 1, 2, 2], 3),
        ("three-class-one-hot", _Fixed(3, seed=8),
         [(_data(4, seed=19), F.one_hot(idx, 3).float(), ["p", "q", "r", "s"])],
         [2, 0, 1, 2], 3),
    ]
    for cid, model, loader, labels_want, n in cases:
        with subtests.test(case=cid):
            with torch.no_grad():
                logits = [model(d) for d, _t, _f in loader]
            want_loss = sum(U.calculate_loss(lg, t, prefer_focal=True).item() * len(t)
                            for lg, (_d, t, _f) in zip(logits, loader)) / len(labels_want)
            names = [f for _d, _t, fs in loader for f in fs]
            metrics, probs, labels, df = D.test_model_core(
                model, loader, "ignored", epoch=3.7, loss_type="ignored")
            assert labels == labels_want
            assert metrics["epoch"] == 3
            assert metrics["loss"] == pytest.approx(want_loss, rel=1e-6)
            assert metrics["majority_class_accuracy"] == pytest.approx(
                np.bincount(labels_want).max() / len(labels_want))
            assert df["filename"].tolist() == names
            assert df["true_label"].tolist() == labels_want
            if n == 1:
                want = torch.sigmoid(torch.cat([lg.view(-1) for lg in logits])).numpy()
                assert probs.shape == (len(labels_want),)
                np.testing.assert_allclose(probs, want, rtol=1e-6)
                assert list(df.columns) == _BINARY_COLUMNS
                assert df["predicted_label"].tolist() == (want >= 0.5).astype(int).tolist()
                np.testing.assert_allclose(df["class_1_probability"], want, rtol=1e-6)
                assert metrics["num_classes"] == 2
                assert metrics["accuracy"] == D._binary_metrics(
                    np.array(labels_want), probs)["accuracy"]
            else:
                want = torch.softmax(torch.cat(logits), 1).numpy()
                np.testing.assert_allclose(probs, want, rtol=1e-6)
                assert list(df.columns) == ["filename", "true_label", "predicted_label",
                                            "prob_class_0", "prob_class_1", "prob_class_2"]
                assert df["predicted_label"].tolist() == want.argmax(1).tolist()
                np.testing.assert_allclose(df[[f"prob_class_{k}" for k in range(3)]],
                                           want, rtol=1e-6)
                assert metrics["num_classes"] == 3
                assert metrics["per_class_accuracy"] == D._multiclass_metrics(
                    np.array(labels_want), probs)["per_class_accuracy"]
            assert metrics["Accuracy"] == metrics["accuracy"]

    with subtests.test(case="empty-loader-takes-the-binary-branch"):
        metrics, probs, labels, df = D.test_model_core(
            _Fixed(3), [], "test", epoch=0, loss_type="auto")
        assert probs.shape == (0,)
        assert labels == []
        assert len(df) == 0
        assert list(df.columns) == _BINARY_COLUMNS
        assert metrics["loss"] == 0.0
        assert metrics["num_classes"] == 2
        assert "majority_class_accuracy" not in metrics
        assert math.isnan(metrics["accuracy"]) and math.isnan(metrics["Accuracy"])

    with subtests.test(case="two-item-batch-is-refused"):
        with pytest.raises(ValueError):
            D.test_model_core(_Fixed(2), [(_data(2), torch.tensor([0, 1]))],
                              "test", 1, "auto")

    with subtests.test(case="device-notice-comes-first"):
        capsys.readouterr()
        with monkeypatch.context() as mp:
            mp.setattr(D, "pick_device", lambda *a, **k: (torch.device("cpu"), _NOTE))
            with pytest.raises(_Stop):
                D.test_model_core(_Refuses(), [], "test", 1, "auto")
        assert capsys.readouterr().out == _NOTE + "\n"


# ===========================================================================
# test_model_performance
# ===========================================================================

def test_test_model_performance_wraps_the_core_in_one_summary_row(subtests):
    """The summary frame is the core's metrics dict as one row (list-valued
    per-class cells included) and the per-file frame is the core's frame."""
    cases = [
        ("binary", _Fixed(1, seed=12), _data(6, seed=23),
         torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])),
        ("three-class", _Fixed(3, seed=15), _data(6, seed=29),
         torch.tensor([0, 1, 2, 0, 1, 2])),
    ]
    for cid, model, data, target in cases:
        with subtests.test(case=cid):
            names = [f"n{i}.png" for i in range(6)]
            loader = [(data, target, names)]
            core, _p, _l, core_df = D.test_model_core(model, loader, "x", 11, "auto")
            summary, per_file = D.test_model_performance(
                loader, model, ["test"], epoch=11, loss_type="auto")
            assert list(summary.index) == [0]
            assert list(summary.columns) == list(core)
            for key, value in core.items():
                cell = summary.loc[0, key]
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(cell), key
                elif isinstance(value, list):
                    assert list(cell) == value, key
                else:
                    assert cell == pytest.approx(value), key
            assert summary.loc[0, "epoch"] == 11
            assert per_file.equals(core_df)
            assert per_file["filename"].tolist() == names
