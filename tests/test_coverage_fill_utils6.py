"""Coverage-fill batch 6 for spacr.utils loss-builder / regression / resize."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# _infer_indices
# ---------------------------------------------------------------------------

def test_infer_indices():
    assert list(U._infer_indices(torch.tensor([0, 2, 1]), 3)) == [0, 2, 1]
    onehot = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    assert list(U._infer_indices(onehot, 2)) == [1, 0]
    binf = torch.tensor([0.2, 0.7, 0.9])
    assert list(U._infer_indices(binf, 1)) == [0, 1, 1]


# ---------------------------------------------------------------------------
# build_loss
# ---------------------------------------------------------------------------

def _multiclass_batch(C=3, n=8):
    return torch.randn(n, C), torch.randint(0, C, (n,))


def _binary_batch(n=8):
    return torch.randn(n, 1), torch.randint(0, 2, (n,)).float()


@pytest.mark.parametrize("lt", ["ce", "ce_smooth", "focal_ce", "auto"])
def test_build_loss_multiclass(lt):
    fn = U.build_loss(lt, num_classes=3, label_smoothing=0.1)
    logits, target = _multiclass_batch()
    loss = fn(logits, target)
    assert loss.ndim == 0


def test_build_loss_ce_weighted():
    counts = torch.tensor([10.0, 30.0, 60.0])
    fn = U.build_loss("ce_weighted", num_classes=3, class_counts=counts)
    logits, target = _multiclass_batch()
    assert fn(logits, target).ndim == 0


@pytest.mark.parametrize("lt", ["bce", "focal_bce", "auto"])
def test_build_loss_binary(lt):
    fn = U.build_loss(lt, num_classes=1)
    logits, target = _binary_batch()
    assert fn(logits, target).ndim == 0


def test_build_loss_unknown():
    with pytest.raises(ValueError):
        U.build_loss("bogus_loss", num_classes=3)


# ---------------------------------------------------------------------------
# lasso_reg
# ---------------------------------------------------------------------------

def _reg_df(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "gene": rng.choice(["g1", "g2", "g3"], n),
        "grna": rng.choice(["r1", "r2"], n),
        "plateID": rng.choice(["p1", "p2"], n),
        "rowID": rng.choice(["A", "B"], n),
        "columnID": rng.choice(["1", "2"], n),
        "pred": rng.normal(0, 1, n),
    })


def test_lasso_reg_lasso_and_ridge():
    df = _reg_df()
    lasso = U.lasso_reg(df, alpha_value=0.01, reg_type="lasso")
    assert {"Feature", "Coefficient"} <= set(lasso.columns)
    ridge = U.lasso_reg(df, alpha_value=0.01, reg_type="ridge")
    assert {"Feature", "Coefficient"} <= set(ridge.columns)


# ---------------------------------------------------------------------------
# resize_images_and_labels / resize_labels_back
# ---------------------------------------------------------------------------

def test_resize_images_and_labels():
    imgs = [np.random.default_rng(0).random((32, 32)).astype(np.float32)]
    lbls = [np.zeros((32, 32), dtype=np.uint16)]
    lbls[0][4:12, 4:12] = 1
    ri, rl = U.resize_images_and_labels(imgs, lbls, 16, 16, show_example=False)
    assert ri[0].shape == (16, 16) and rl[0].shape == (16, 16)


def test_resize_images_only():
    imgs = [np.random.default_rng(1).random((32, 32, 3)).astype(np.float32)]
    ri, rl = U.resize_images_and_labels(imgs, None, 16, 16, show_example=False)
    assert ri[0].shape[:2] == (16, 16) and rl == []


def test_resize_labels_only():
    lbls = [np.zeros((32, 32), dtype=np.uint16)]
    ri, rl = U.resize_images_and_labels(None, lbls, 16, 16, show_example=False)
    assert rl[0].shape == (16, 16) and ri == []


def test_resize_labels_back():
    lbls = [np.zeros((16, 16), dtype=np.uint16)]
    out = U.resize_labels_back(lbls, [(32, 24)])
    assert out[0].shape == (32, 24)


def test_resize_labels_back_length_mismatch():
    with pytest.raises(ValueError):
        U.resize_labels_back([np.zeros((4, 4))], [(8, 8), (8, 8)])


def test_resize_labels_back_bad_dims():
    with pytest.raises(ValueError):
        U.resize_labels_back([np.zeros((4, 4))], [(8,)])
