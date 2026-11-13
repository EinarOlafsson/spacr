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


def test_lasso_reg_rejects_unknown_regression_type():
    with pytest.raises(ValueError, match="expected 'lasso' or 'ridge'"):
        U.lasso_reg(_reg_df(), reg_type="elastic-net")


# ---------------------------------------------------------------------------
# resize_images_and_labels / resize_labels_back
# ---------------------------------------------------------------------------
