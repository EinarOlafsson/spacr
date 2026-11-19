"""Grid application preserves per-well tuning and yield without decoys."""

import os

import pandas as pd
import pytest

from spacr import ops_grid_search as grid, ops_prefix as prefix
from tests.test_ops_grid_search_960 import _settings


@pytest.mark.parametrize("current_tuning", ["per_well_decoy", "none"])
def test_applying_a_partial_grid_preserves_the_per_well_tuned_values(current_tuning):
    """A result with different well thresholds must keep the per-well decode path."""
    settings = {"ops_prefix_threshold_tuning": current_tuning,
                "ops_read_threshold": 80, "ops_prefix_min_read_quality": 0,
                "ops_prefix_max_distance": 1, "ops_read_mode": "log",
                "ops_prefix_crosstalk": "none", "ops_quality_min_reads": 2}
    point = {key: settings[key] for key in grid._GRID_KEYS}
    point.update(ops_read_threshold="per well", ops_quality_min_reads=3, point=1)
    result = {"table": pd.DataFrame([point]), "run_id": "two-wells",
              "tiles": ["P:W1:1", "P:W2:1"],
              "defaults": {"P:W1": {"ops_read_threshold": 10},
                           "P:W2": {"ops_read_threshold": 20}}}
    applied = grid._apply_grid_choice(settings, result, 1)
    assert applied["ops_prefix_threshold_tuning"] == "per_well_decoy"
    assert applied["ops_quality_min_reads"] == 3
    assert applied["ops_read_threshold"] == 80
    assert applied["ops_grid_run_id"] == "two-wells"
    assert applied["ops_grid_search"] is False


def test_grid_without_decoys_still_counts_real_assignments(tmp_path):
    """The grid reports the real decoder calls when the decoy control is disabled."""
    settings = _settings(tmp_path, "grid", ops_prefix_threshold_tuning="none",
                         ops_prefix_decoy_k=0, ops_grid_tiles="1")
    result = grid._run_grid_search(settings)
    decoded = prefix.run_prefix_decode({**settings, "cpg_sites": [1],
                                        "dst_root": str(tmp_path / "decode")})
    calls = pd.read_csv(os.path.join(decoded["folder"], "cells_CP9_W1.csv"))
    real = calls["quality"].isin(("perfect", "great")) & ~calls["gene"].isin(
        (prefix.DECOY_GENE, prefix.NEARMISS_GENE))
    expected = int(real.sum())
    assert expected > 0
    row = result["table"].iloc[0]
    assert row["cells_real"] == expected
    assert row["assignment_pct"] == pytest.approx(100.0 * expected / row["cells_in_labels"])
    assert row["cells_decoy"] == 0
