"""
Tests for spacr.plot — matplotlib helpers exercised with the Agg backend.

Full "plot everything from a database" pipelines are covered by higher-level
integration tests; here we focus on the pure/computable pieces.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)  # safety net if pytest ran before conftest
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

import spacr.plot as P


# ---------------------------------------------------------------------------
# random_cmap / generate_mask_random_cmap
# ---------------------------------------------------------------------------

def test_random_cmap_default_size():
    cmap = P.random_cmap()
    # default num_objects=100, plus the +1 for background slot → 101 colors
    assert cmap.N == 101


@pytest.mark.parametrize("n", [1, 5, 20, 200])
def test_random_cmap_size(n):
    cmap = P.random_cmap(num_objects=n)
    assert cmap.N == n + 1
    # First color slot is background (black, alpha 1).
    r, g, b, a = cmap(0)
    assert (r, g, b, a) == (0.0, 0.0, 0.0, 1.0)


def test_generate_mask_random_cmap_uses_unique_labels(synth_mask_2d):
    cmap = P.generate_mask_random_cmap(synth_mask_2d)
    n_labels = len(np.unique(synth_mask_2d)) - 1  # subtract background 0
    assert cmap.N == n_labels + 1


def test_generate_mask_random_cmap_alpha_channel_is_one(synth_mask_2d):
    cmap = P.generate_mask_random_cmap(synth_mask_2d)
    # Every color in the cmap should have alpha == 1.
    for i in range(cmap.N):
        _, _, _, a = cmap(i)
        assert a == 1.0


def test_generate_mask_random_cmap_empty_mask_returns_bg_only():
    empty = np.zeros((16, 16), dtype=np.int32)
    cmap = P.generate_mask_random_cmap(empty)
    # 0 objects + 1 background slot
    assert cmap.N == 1


# ---------------------------------------------------------------------------
# generate_plate_heatmap — with a synthetic 3-part prc DataFrame
# ---------------------------------------------------------------------------

def _fake_heatmap_df(n_wells=48, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = [f"r{int(i)}" for i in rng.integers(1, 17, size=n_wells)]
    cols = [f"c{int(i)}" for i in rng.integers(1, 25, size=n_wells)]
    prc = [f"p1_{r}_{c}" for r, c in zip(rows, cols)]
    return pd.DataFrame(
        {
            "prc": prc,
            "value": rng.uniform(0, 100, size=n_wells),
        }
    )


def test_generate_plate_heatmap_count_grouping():
    df = _fake_heatmap_df()
    plate_map, (vmin, vmax) = P.generate_plate_heatmap(
        df, plate_number="p1", variable="value", grouping="count",
        min_max="all", min_count=0,
    )
    assert isinstance(plate_map, pd.DataFrame)
    # After pivot, index is rows and columns are the well columns.
    assert plate_map.index.name in ("rowID", None)  # accept either style
    assert vmin <= vmax


def test_generate_plate_heatmap_mean_grouping():
    df = _fake_heatmap_df()
    plate_map, _ = P.generate_plate_heatmap(
        df, plate_number="p1", variable="value", grouping="mean",
        min_max="all", min_count=0,
    )
    # Means of the "value" column should be within its input range.
    vals = plate_map.values.flatten()
    vals = vals[~np.isnan(vals)]
    assert (vals >= 0).all() and (vals <= 100).all()


def test_generate_plate_heatmap_rejects_bad_grouping():
    df = _fake_heatmap_df()
    with pytest.raises(ValueError):
        P.generate_plate_heatmap(df, "p1", "value", "median", "all", 0)


def test_generate_plate_heatmap_missing_variable_raises():
    df = _fake_heatmap_df()
    with pytest.raises(KeyError):
        P.generate_plate_heatmap(df, "p1", "not_a_column", "mean", "all", 0)


# ---------------------------------------------------------------------------
# Matplotlib import path is Agg — no display needed.
# ---------------------------------------------------------------------------

def test_matplotlib_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_generate_plate_heatmap_output_is_plottable():
    """Sanity: the heatmap DataFrame renders without raising on Agg."""
    df = _fake_heatmap_df()
    plate_map, (vmin, vmax) = P.generate_plate_heatmap(
        df, "p1", "value", "mean", "all", 0,
    )
    fig, ax = plt.subplots()
    im = ax.imshow(plate_map.values, vmin=vmin, vmax=vmax)
    fig.canvas.draw()
    plt.close(fig)
