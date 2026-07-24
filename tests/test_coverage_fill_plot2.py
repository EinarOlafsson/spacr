"""Coverage-fill batch 2 for spacr.plot pure/small helpers (Agg)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import plot as P


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# colormap / colour helpers
# ---------------------------------------------------------------------------

def test_generate_mask_random_cmap_private():
    m = np.zeros((16, 16), dtype=np.int32); m[2:6, 2:6] = 1; m[8:12, 8:12] = 2
    cmap = P._generate_mask_random_cmap(m)
    assert cmap.N >= 3   # background + 2 objects


@pytest.mark.parametrize("fmt,expected0", [
    ("rgb", [1, 0, 0]), ("bgr", [0, 0, 1]),
    ("gbr", [0, 1, 0]), ("rbg", [1, 0, 0]), ("other", [1, 0, 0])])
def test_get_colours_merged(fmt, expected0):
    out = P._get_colours_merged(fmt)
    assert out[0] == expected0 and len(out) == 3


# ---------------------------------------------------------------------------
# _filter_objects_in_plot
# ---------------------------------------------------------------------------

def test_filter_objects_in_plot():
    size = 24
    cell = np.zeros((size, size), dtype=np.int32)
    cell[2:10, 2:10] = 1
    cell[14:22, 14:22] = 2
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[3:6, 3:6] = 1
    nucleus[15:18, 15:18] = 2
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[6:9, 6:9] = 1
    stack = np.stack([cell, nucleus, pathogen], axis=-1)
    out = P._filter_objects_in_plot(
        stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
        pathogen_mask_dim=2, mask_dims=[0, 1, 2],
        filter_min_max=[[1, 100000], [1, 100000], [1, 100000]],
        nuclei_limit=True, pathogen_limit=True)
    assert out.shape == stack.shape


# ---------------------------------------------------------------------------
# _plot_histograms_and_stats
# ---------------------------------------------------------------------------

def test_plot_histograms_and_stats():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "condition": rng.choice(["ctrl", "trt"], 100),
        "pred": rng.uniform(0, 1, 100),
    })
    P._plot_histograms_and_stats(df)   # prints + plots; just must not raise


# ---------------------------------------------------------------------------
# random_cmap / generate_mask_random_cmap (public)
# ---------------------------------------------------------------------------

def test_random_cmap_and_public():
    assert P.random_cmap(30).N >= 30
    m = np.zeros((8, 8), dtype=np.int32); m[1:4, 1:4] = 1
    assert P.generate_mask_random_cmap(m) is not None
