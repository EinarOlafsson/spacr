"""Coverage-fill for spacr.deep_spacr pure-logic metric helpers (no GPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use("Agg")

from spacr import deep_spacr as D


# ---------------------------------------------------------------------------
# _to_numpy_labels — three branches
# ---------------------------------------------------------------------------

def test_to_numpy_labels_onehot():
    t = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    out = D._to_numpy_labels(t)
    assert list(out) == [1, 0]


def test_to_numpy_labels_float_1d():
    t = torch.tensor([0.2, 0.7, 1.4])
    out = D._to_numpy_labels(t)
    assert list(out) == [0, 1, 1]


def test_to_numpy_labels_int_1d():
    t = torch.tensor([0, 2, 1], dtype=torch.int64)
    out = D._to_numpy_labels(t)
    assert list(out) == [0, 2, 1]


# ---------------------------------------------------------------------------
# _binary_metrics
# ---------------------------------------------------------------------------

def test_binary_metrics_two_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = D._binary_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert 0.0 <= m["prauc"] <= 1.0
    assert 0.0 <= m["optimal_threshold"] <= 1.0


def test_binary_metrics_single_class():
    y = np.array([1, 1, 1])
    p = np.array([0.6, 0.7, 0.8])
    m = D._binary_metrics(y, p)
    assert np.isnan(m["prauc"])
    assert m["optimal_threshold"] == 0.5
    assert m["pos_accuracy"] == 1.0
    assert np.isnan(m["neg_accuracy"])   # no negatives present


def test_binary_metrics_reshapes_2d():
    y = np.array([[0], [1], [1]])         # 2-D → reshaped to 1-D
    p = np.array([0.2, 0.7, 0.9])
    m = D._binary_metrics(y, p)
    assert m["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# _multiclass_metrics
# ---------------------------------------------------------------------------

def test_multiclass_metrics():
    y = np.array([0, 1, 2, 1])
    prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
    ])
    m = D._multiclass_metrics(y, prob)
    assert m["accuracy"] == 1.0
    assert m["num_classes"] == 3
    assert len(m["per_class_accuracy"]) == 3
    assert np.isnan(m["neg_accuracy"])


# ---------------------------------------------------------------------------
# evaluate_model_performance / test_model_core / test_model_performance
# ---------------------------------------------------------------------------

import torch.nn as nn


class _Bin(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc(x)


class _Multi(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.fc = nn.Linear(8, c)

    def forward(self, x):
        return self.fc(x)


def _binary_loader(n_batches=2, bs=4):
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(n_batches):
        data = torch.randn(bs, 8, generator=g)
        target = torch.randint(0, 2, (bs,), generator=g).float()
        names = [f"img{i}_{j}.png" for j in range(bs)]
        out.append((data, target, names))
    return out


def _multi_loader(n_batches=2, bs=4, c=3):
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(n_batches):
        data = torch.randn(bs, 8, generator=g)
        target = torch.randint(0, c, (bs,), generator=g)
        names = [f"img{i}_{j}.png" for j in range(bs)]
        out.append((data, target, names))
    return out


def test_evaluate_model_performance_binary():
    metrics, (probs, labels) = D.evaluate_model_performance(
        _Bin(), _binary_loader(), epoch=1)
    assert "loss" in metrics and metrics["epoch"] == 1
    assert probs.ndim == 1 and len(labels) == 8


def test_evaluate_model_performance_multiclass():
    metrics, (probs, labels) = D.evaluate_model_performance(
        _Multi(), _multi_loader(), epoch=2)
    assert probs.ndim == 2 and probs.shape[1] == 3
    assert metrics["num_classes"] == 3


def test_evaluate_model_performance_empty_binary():
    metrics, (probs, labels) = D.evaluate_model_performance(
        _Bin(), [], epoch=0, num_classes=1)
    assert probs.shape == (0,)


def test_evaluate_model_performance_empty_multiclass():
    metrics, (probs, labels) = D.evaluate_model_performance(
        _Multi(), [], epoch=0, num_classes=3)
    assert probs.shape == (0, 3)


def test_test_model_core_binary():
    import pandas as pd
    metrics, probs, labels, df = D.test_model_core(
        _Bin(), _binary_loader(), "val", epoch=1, loss_type="auto")
    assert "class_1_probability" in df.columns
    assert len(df) == 8
    assert "filename" in df.columns


def test_test_model_core_multiclass():
    metrics, probs, labels, df = D.test_model_core(
        _Multi(), _multi_loader(), "val", epoch=1, loss_type="auto")
    assert "prob_class_0" in df.columns and "prob_class_2" in df.columns


def test_test_model_performance_wrapper():
    result_df, results_df = D.test_model_performance(
        _binary_loader(), _Bin(), ["val"], epoch=1, loss_type="auto")
    assert len(result_df) == 1              # one summary row
    assert len(results_df) == 8             # per-file rows
