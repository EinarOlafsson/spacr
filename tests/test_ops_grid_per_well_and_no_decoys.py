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


@pytest.mark.parametrize("method", ["spacr", "weighted_majority"])
def test_applying_a_partial_grid_keeps_the_values_it_fixes_out_of_the_tuner(method):
    """The per-well tuner chooses only the columns left to each well, not the applied ones."""
    settings = {"ops_prefix_threshold_tuning": "per_well_decoy", "ops_library": "Barcodes.csv",
                "ops_prefix_decoy_k": 4,
                "ops_prefix_quality_method": method,
                "ops_read_threshold": 80, "ops_prefix_min_read_quality": 0,
                "ops_prefix_max_distance": 1, "ops_read_mode": "log",
                "ops_prefix_crosstalk": "none", "ops_quality_min_reads": 2}
    point = {key: settings[key] for key in grid._GRID_KEYS}
    point.update(ops_read_threshold=40, ops_prefix_min_read_quality="per well", point=2)
    result = {"table": pd.DataFrame([point]), "run_id": "threshold-grid",
              "tiles": ["P:W1:1", "P:W2:1"]}
    applied = grid._apply_grid_choice(settings, result, 2)
    assert applied["ops_prefix_threshold_tuning"] == "per_well_decoy"
    axes = dict(prefix._tuning_axes(prefix.prefix_settings(applied)))
    assert axes["ops_read_threshold"] == [40.0]
    assert axes["ops_prefix_max_distance"] == [1.0]
    default = (prefix.PREFIX_TUNING_GRID_WEIGHTED if method == "weighted_majority"
               else prefix.PREFIX_TUNING_GRID)
    assert axes["ops_prefix_min_read_quality"] == sorted(default["ops_prefix_min_read_quality"])
    if method == "weighted_majority":
        assert len(axes["ops_read_neighbour_discount"]) > 1


def test_applying_a_partial_grid_pins_axes_of_an_entered_tuning_grid():
    """An entered tuning grid keeps its per-well axes and loses only the fixed ones."""
    settings = {"ops_prefix_threshold_tuning": "per_well_decoy", "ops_library": "Barcodes.csv",
                "ops_prefix_decoy_k": 4,
                "ops_prefix_tuning_grid": '{"ops_read_threshold": [60, 80], '
                                          '"ops_prefix_min_read_quality": [0, 0.2]}',
                "ops_read_threshold": 80, "ops_prefix_min_read_quality": 0,
                "ops_prefix_max_distance": 1, "ops_read_mode": "log",
                "ops_prefix_crosstalk": "none", "ops_quality_min_reads": 2}
    point = {key: settings[key] for key in grid._GRID_KEYS}
    point.update(ops_read_threshold="per well", ops_prefix_min_read_quality=0.2, point=1)
    result = {"table": pd.DataFrame([point]), "run_id": "entered", "tiles": ["P:W1:1"]}
    applied = grid._apply_grid_choice(settings, result, 1)
    axes = dict(prefix._tuning_axes(prefix.prefix_settings(applied)))
    assert axes["ops_read_threshold"] == [60.0, 80.0]
    assert axes["ops_prefix_min_read_quality"] == [0.2]
