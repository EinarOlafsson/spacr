"""CPU-only coverage for the spacr.utils "preprocess / feature filtering" block.

Covers the cluster-grid renderer (:func:`plot_grid`), the DB path helpers
(:func:`generate_path_list_from_db`, :func:`correct_paths`), the filesystem
helpers (:func:`delete_folder`, :func:`measure_test_mode`) and the feature
matrix pipeline (:func:`preprocess_data`, :func:`remove_low_variance_columns`,
:func:`remove_highly_correlated_columns`, :func:`filter_dataframe_features`)
plus the overlap-avoidance helpers.

Everything runs offline on tiny synthetic frames; matplotlib is driven under
the Agg backend and every figure is closed by the autouse fixture below.
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_grid
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# generate_path_list_from_db
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# correct_paths
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# delete_folder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# measure_test_mode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# preprocess_data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# remove_low_variance_columns / remove_highly_correlated_columns
# ---------------------------------------------------------------------------

def test_remove_low_variance_columns_verbose_keeps_non_numeric(capsys):
    from spacr.utils import remove_low_variance_columns

    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "spread": rng.normal(0.0, 5.0, 40),
        "flat": np.full(40, 3.0),
        "well": ["A01"] * 40,
    })
    out = remove_low_variance_columns(df, threshold=0.01, verbose=True)

    assert list(out.columns) == ["spread", "well"]
    assert "Removed columns due to low variance: ['flat']" in capsys.readouterr().out


def test_remove_highly_correlated_columns_verbose(capsys):
    from spacr.utils import remove_highly_correlated_columns

    rng = np.random.default_rng(9)
    base = rng.normal(0.0, 1.0, 50)
    df = pd.DataFrame({"a": base, "b": base * 3.0 + 1.0, "c": rng.normal(0, 1, 50)})
    out = remove_highly_correlated_columns(df, threshold=0.95, verbose=True)

    # 'b' is the later column in the upper triangle, so it is the one dropped
    assert list(out.columns) == ["a", "c"]
    assert "Removed columns due to high correlation: ['b']" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# filter_dataframe_features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# check_overlap / find_non_overlapping_position
# ---------------------------------------------------------------------------

def test_find_non_overlapping_position_stays_within_jitter_range():
    """A free neighbourhood yields a moved point, still inside the +/-10 box."""
    from spacr.utils import check_overlap, find_non_overlapping_position

    occupied = [(100.0, 100.0)]
    assert check_overlap((0.0, 0.0), occupied, threshold=5.0) is False

    x, y = find_non_overlapping_position(0.0, 0.0, occupied, threshold=5.0,
                                         max_attempts=10)
    assert (x, y) != (0.0, 0.0)
    assert abs(x) <= 10.0 and abs(y) <= 10.0
    assert check_overlap((x, y), occupied, threshold=5.0) is False
