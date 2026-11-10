"""Item 903: fitness and generic-axis removal from gene profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (fitness_bins, generic_covariates,
                                            remove_generic_axes)


def _genes(seed=0, n=300, dims=6, axis_strength=0.0):
    """Profiles whose first column follows log cells per guide."""
    rng = np.random.default_rng(seed)
    names = [f"G{k:03d}" for k in range(n)] + [f"NTpseudo_{k:02d}" for k in range(10)]
    cpg = np.r_[rng.lognormal(5, 0.6, n), np.full(10, np.exp(5))]
    covariates = pd.DataFrame({"gene": names, "cells_per_guide": cpg,
                               "shift_G1": rng.normal(0, 0.05, len(names)),
                               "shift_M": rng.normal(0, 0.02, len(names)),
                               "shift_SG2": rng.normal(0, 0.03, len(names))})
    x = rng.normal(0, 1, (len(names), dims))
    x[:, 0] += 2.0 * (np.log(cpg) - np.log(cpg).mean())
    if axis_strength:
        direction = np.ones(dims) / np.sqrt(dims)
        x += axis_strength * np.outer(-(np.log(cpg) - np.log(cpg).mean()), direction)
    return x, np.array(names), covariates


def test_the_covariate_part_is_regressed_out_across_genes():
    x, names, covariates = _genes()
    residual, report, _axis, _g = remove_generic_axes(
        x, names, covariates, axes=["log_cells_per_guide", "shift_G1"])
    targeting = ~pd.Series(names).str.startswith("NTpseudo").to_numpy()
    log_cpg = np.log(covariates.set_index("gene").loc[names, "cells_per_guide"])
    before = abs(np.corrcoef(x[targeting, 0], log_cpg[targeting])[0, 1])
    after = abs(np.corrcoef(residual[targeting, 0], log_cpg[targeting])[0, 1])
    assert before > 0.7 and after < 1e-6
    assert report["covariates"] == ["log cells per guide", "G1 fraction shift"]
    assert 0 < report["var_removed_by_covariates"] < 1
    assert "essential_axis_removed" not in report


def test_the_essential_axis_goes_only_when_it_tracks_fitness():
    x, names, covariates = _genes(seed=1, axis_strength=3.0)
    guides = np.vstack([x[:5], x[:5]])
    residual, report, axis, guide_residual = remove_generic_axes(
        x, names, covariates, axes=["essential_axis"], min_rho=0.3,
        guide_values=guides, guide_genes=list(names[:5]) * 2)
    assert report["essential_axis_removed"]
    assert abs(report["essential_axis_rho_with_log_cells_per_guide"]) > 0.3
    assert np.allclose(residual @ axis, 0, atol=1e-9)
    assert np.allclose(guide_residual @ axis, 0, atol=1e-9)
    kept, report, _a, _g = remove_generic_axes(
        x, names, covariates, axes=["essential_axis"], min_rho=0.999)
    assert not report["essential_axis_removed"] and np.allclose(kept, x)
    with pytest.raises(ValueError, match="unknown generic axes"):
        remove_generic_axes(x, names, covariates, axes=["batch"])


def test_covariates_give_phase_shares_against_the_controls():
    genes = pd.DataFrame({"gene": ["A", "nontargeting"], "n_guides": [4, 2],
                          "n_cells": [40, 20]})
    cells = pd.DataFrame({"gene": ["A"] * 4 + ["nontargeting"] * 4,
                          "phase": ["M", "M", "G1", "SG2", "G1", "G1", "G1", "M"]})
    control = cells["gene"] == "nontargeting"
    table = generic_covariates(genes, cells, control=control).set_index("gene")
    assert table.loc["A", "cells_per_guide"] == 10
    assert table.loc["A", "frac_M"] == 0.5
    assert table.loc["A", "shift_M"] == pytest.approx(0.25)
    assert table.loc["nontargeting", "shift_G1"] == 0
    bins = fitness_bins(["A", "B", "C", "nontargeting_x"], [10, 100, 1000, 5], bins=3)
    assert list(bins) == [0, 1, 2, -1]


def test_the_aggregate_phase_removes_generic_axes(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_profile_generic_axes="log_cells_per_guide,shift_M,essential_axis",
                  ops_profile_essential_axis_min_rho=0.3)["profile_options"]
    info = report["generic_axes"]
    assert info["covariates"] == ["log cells per guide", "M fraction shift"]
    assert "essential_axis_removed" in info
    table = pd.read_parquet(report["outputs"]["genes_generic_parquet"])
    assert table.filter(like="PC_").notna().all().all()
