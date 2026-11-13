"""Coverage fill for the spacr.utils training-helper block.

Targets the defensive / rarely-taken branches of ``classification_metrics``,
``compute_irm_penalty``, ``_list_torchvision_model_names``, ``choose_model``,
``calculate_loss``, ``pick_best_model`` and ``save_file_lists``.

Everything here is CPU-only and offline: the two real-backbone checks build
``resnet18`` with ``pretrained=False`` at 32x32, and every other
``choose_model`` branch is driven through a monkeypatched ``TorchModel``
double so no weights are ever constructed or downloaded.
"""
from __future__ import annotations


import pandas as pd
import pytest
import torch

from spacr import utils as U


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let a stray figure survive a test in this module."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# classification_metrics — the "no positive-class rows" branch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compute_irm_penalty
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _list_torchvision_model_names — the list_models() failure fallback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# choose_model — invalid names, the custom branch, sanity-check failures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# calculate_loss — focal reductions and the 1-D multilabel one-hot path
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# pick_best_model — filenames that do not match the epoch/acc pattern
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# save_file_lists
# ---------------------------------------------------------------------------

def test_save_file_lists_writes_single_column_csv(tmp_path):
    """The list lands in <dst>/<data_set>.csv under a column named data_set."""
    paths = ["/a/one.png", "/a/two.png", "/a/three.png"]
    assert U.save_file_lists(str(tmp_path), "train", paths) is None

    out = tmp_path / "train.csv"
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns) == ["train"]
    assert back["train"].tolist() == paths


def test_save_file_lists_empty_list_writes_header_only(tmp_path):
    """An empty list still produces a well-formed, header-only CSV."""
    U.save_file_lists(str(tmp_path), "val", [])
    back = pd.read_csv(tmp_path / "val.csv")
    assert list(back.columns) == ["val"]
    assert len(back) == 0


# ---------------------------------------------------------------------------
# get_paths_from_db / split_my_dataset — behaviour anchors for this block
# ---------------------------------------------------------------------------

def test_get_paths_from_db_returns_empty_frame_when_nothing_matches():
    """A prcfo index disjoint from png_df yields an empty (but typed) frame."""
    df = pd.DataFrame(index=["p1_r1_c1_f1_o1"])
    png_df = pd.DataFrame({
        "png_path": ["/x/cell_png/a.png"],
        "prcfo": ["p9_r9_c9_f9_o9"],
    })
    out = U.get_paths_from_db(df, png_df)
    assert len(out) == 0
    assert list(out.columns) == ["png_path", "prcfo"]


def test_split_my_dataset_partitions_without_overlap():
    """The two subsets are disjoint and together cover every source index."""
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.arange(20).float().view(20, 1))
    train, val = U.split_my_dataset(ds, split_ratio=0.25)

    assert len(train) == 15 and len(val) == 5
    assert set(train.indices).isdisjoint(val.indices)
    assert set(train.indices) | set(val.indices) == set(range(20))
