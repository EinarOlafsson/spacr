"""Item 905: responder fraction, responder profiles and distribution distances."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (distribution_distances,
                                            responder_profiles)

DIMS = 5


def _cells(seed=0):
    """Gene HALF moves 40 % of its cells by 4 in the first dimension; gene
    NONE is like the controls; gene FEW has too few cells."""
    rng = np.random.default_rng(seed)
    rows = []
    for gene, count in (("nontargeting", 4000), ("HALF", 600), ("NONE", 600), ("FEW", 10)):
        for k in range(count):
            x = rng.normal(0, 1, DIMS)
            if gene == "HALF" and k < 240:
                x[0] += 4.0
            rows.append({"gene": gene if gene != "nontargeting" else f"NTpseudo_{k % 2:02d}",
                         "guide": f"{gene}_{k % 3}" if gene != "nontargeting" else f"nt{k % 8}", **{f"PC_{d}": x[d] for d in range(DIMS)}})
    cells = pd.DataFrame(rows)
    return cells, cells["gene"].str.startswith("NTpseudo")


def test_responders_are_the_share_that_moved_and_their_profile_is_undiluted():
    cells, control = _cells()
    columns = [f"PC_{d}" for d in range(DIMS)]
    result = responder_profiles(cells, columns, control=control,
                                control_pool=3000, energy_controls=1000,
                                directions=8, quantiles=4)
    summary = result["summary"].set_index("gene")
    assert "FEW" not in summary.index
    assert summary.loc["HALF", "responder_fraction"] == pytest.approx(0.4, abs=0.08)
    assert summary.loc["NONE", "responder_fraction"] < 0.05
    assert summary.loc["HALF", "energy_distance_vs_NT"] > summary.loc["NONE", "energy_distance_vs_NT"]
    profile = result["genes"].set_index("gene")["PC_0"]
    assert profile["HALF"] > 1.0 and profile["HALF"] > profile["NONE"] + 1.0
    assert result["sliced_wasserstein"].shape == (len(summary), 8 * 4)
    assert set(result["guides"]["gene"]) == set(summary.index)
    distances, embedding = distribution_distances(
        cells, columns, control=control, control_pool=3000,
        energy_controls=1000, directions=8, quantiles=4)
    assert np.array_equal(distances["energy_distance_vs_NT"],
                          result["summary"]["energy_distance_vs_NT"])
    assert np.array_equal(embedding, result["sliced_wasserstein"])


def test_the_used_cells_and_the_subsample_cap_are_honoured():
    cells, control = _cells(seed=1)
    columns = [f"PC_{d}" for d in range(DIMS)]
    use = cells["gene"] != "NONE"
    result = responder_profiles(cells, columns, control=control, use=use,
                                max_cells=100, control_pool=2000,
                                energy_controls=500, directions=4, quantiles=2)
    summary = result["summary"].set_index("gene")
    assert "NONE" not in summary.index
    assert summary.loc["HALF", "cells_used"] == 100


def test_the_aggregate_phase_writes_responder_tables(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_profile_similarity="sliced_wasserstein",
                  ops_profile_cell_cycle="G1",
                  ops_profile_responder_alpha=0.05)["profile_options"]
    summary = pd.read_parquet(report["outputs"]["responder_summary_parquet"])
    assert summary["responder_fraction"].between(0, 1).all()
    quantiles = pd.read_parquet(report["outputs"]["genes_sliced_wasserstein_parquet"])
    assert quantiles.shape[1] == 1 + 64 * 32
    assert report["responders"]["alpha"] == 0.05
