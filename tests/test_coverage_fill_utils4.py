"""Coverage-fill batch 4 for spacr.utils loss/metric/model helpers."""
from __future__ import annotations


import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# calculate_loss — binary / multiclass / multilabel, plain + focal
# ---------------------------------------------------------------------------

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
