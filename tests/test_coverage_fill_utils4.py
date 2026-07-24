"""Coverage-fill batch 4 for spacr.utils loss/metric/model helpers."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_two_classes():
    labels = [0, 0, 1, 1, 1, 0]
    probs = [0.1, 0.4, 0.6, 0.9, 0.8, 0.2]
    loss = torch.tensor(0.5)
    df = U.classification_metrics(labels, probs, loss, epoch=1)
    assert "accuracy" in df.columns and "prauc" in df.columns
    assert df.index[0] == "1"


def test_classification_metrics_single_class():
    labels = [1, 1, 1]
    probs = [0.6, 0.7, 0.8]
    df = U.classification_metrics(labels, probs, torch.tensor(0.1), epoch=2)
    assert np.isnan(df["prauc"].iloc[0])
    assert df["optimal_threshold"].iloc[0] == 0.5


def test_classification_metrics_length_mismatch():
    with pytest.raises(ValueError):
        U.classification_metrics([0, 1], [0.5], torch.tensor(0.1), epoch=0)


# ---------------------------------------------------------------------------
# calculate_loss — binary / multiclass / multilabel, plain + focal
# ---------------------------------------------------------------------------

def test_calculate_loss_binary():
    logits = torch.randn(8, 1)
    target = torch.randint(0, 2, (8,)).float()
    plain = U.calculate_loss(logits, target, prefer_focal=False)
    focal = U.calculate_loss(logits, target, prefer_focal=True)
    assert plain.ndim == 0 and focal.ndim == 0


def test_calculate_loss_binary_1d_output():
    logits = torch.randn(8)          # (N,) → unsqueezed to (N,1)
    target = torch.randint(0, 2, (8,)).float()
    assert U.calculate_loss(logits, target).ndim == 0


def test_calculate_loss_multiclass():
    logits = torch.randn(8, 4)
    target = torch.randint(0, 4, (8,))     # long (N,)
    plain = U.calculate_loss(logits, target, prefer_focal=False)
    focal = U.calculate_loss(logits, target, prefer_focal=True)
    assert plain.ndim == 0 and focal.ndim == 0


def test_calculate_loss_multilabel():
    logits = torch.randn(8, 4)
    target = torch.randint(0, 2, (8, 4)).float()   # (N,C) float
    assert U.calculate_loss(logits, target).ndim == 0
    assert U.calculate_loss(logits, target, prefer_focal=True).ndim == 0


def test_calculate_loss_reduction_none():
    logits = torch.randn(8, 1)
    target = torch.randint(0, 2, (8,)).float()
    out = U.calculate_loss(logits, target, reduction="none")
    assert out.shape[0] == 8


# ---------------------------------------------------------------------------
# split_my_dataset
# ---------------------------------------------------------------------------

def test_split_my_dataset():
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.arange(100).float().view(100, 1))
    train, val = U.split_my_dataset(ds, split_ratio=0.2)
    assert len(train) + len(val) == 100
    assert abs(len(val) - 20) <= 1


# ---------------------------------------------------------------------------
# pick_best_model
# ---------------------------------------------------------------------------

def test_pick_best_model(tmp_path):
    for name in ["m_epoch_1_acc_0.80.pth", "m_epoch_5_acc_0.95.pth",
                 "m_epoch_3_acc_0.90.pth", "unrelated.txt"]:
        (tmp_path / name).write_bytes(b"")
    best = U.pick_best_model(str(tmp_path))
    assert best.endswith("m_epoch_5_acc_0.95.pth")


# ---------------------------------------------------------------------------
# get_paths_from_db
# ---------------------------------------------------------------------------

def test_get_paths_from_db():
    df = pd.DataFrame(index=["p1_r1_c1_f1_o1", "p1_r1_c1_f1_o2"])
    png_df = pd.DataFrame({
        "png_path": ["/x/cell_png/a.png", "/x/nucleus_png/b.png",
                     "/x/cell_png/c.png"],
        "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c1_f1_o1", "p1_r1_c1_f1_o9"],
    })
    out = U.get_paths_from_db(df, png_df, image_type="cell_png")
    assert len(out) == 1   # only the matching prcfo + cell_png row


# ---------------------------------------------------------------------------
# _list_torchvision_model_names
# ---------------------------------------------------------------------------

def test_list_torchvision_model_names():
    names = U._list_torchvision_model_names()
    assert isinstance(names, set) and "resnet50" in names
