"""Item 887: JACKS-style guide efficacy, blended with the all-cell profile."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (_rank_one_efficacy,
                                            guide_efficacy_profiles)


def _planted(seed=0, dims=6):
    """Two genes whose guides are e_g m exactly, and a gene row with no
    guide rows."""
    rng = np.random.default_rng(seed)
    efficacy = {"A": [1.0, 0.7, 0.25, 0.0], "B": [0.4, 1.0, 0.9]}
    rows, genes = [], []
    for name, values in efficacy.items():
        m = rng.normal(0, 1, dims)
        for k, e in enumerate(values):
            rows.append({"guide": f"{name}{k}", "gene": name,
                         "n_cells": int(rng.integers(20, 300)),
                         **{f"w_{d}": e * m[d] for d in range(dims)}})
        genes.append({"gene": name, "n_cells": 100,
                      **{f"w_{d}": m[d] + rng.normal(0, 0.1) for d in range(dims)}})
    genes.append({"gene": "nontargeting_x", "n_cells": 100,
                  **{f"w_{d}": rng.normal(0, 1) for d in range(dims)}})
    return pd.DataFrame(genes), pd.DataFrame(rows), efficacy


def test_planted_efficacies_are_recovered():
    genes, guides, planted = _planted()
    _rows, efficacy = guide_efficacy_profiles(genes, guides, iters=2000)
    for name, values in planted.items():
        fitted = efficacy[[f"{name}{k}" for k in range(len(values))]].to_numpy()
        assert np.allclose(fitted, values, atol=1e-4)
        assert fitted.max() == pytest.approx(1.0)
    _rows, early = guide_efficacy_profiles(genes, guides, iters=2)
    assert np.abs(early.to_numpy() - efficacy.to_numpy()).max() > 1e-4


def test_rows_blend_the_gene_profile_and_rows_without_guides_keep_it():
    genes, guides, _planted_e = _planted(seed=1)
    rows, _e = guide_efficacy_profiles(genes, guides)
    columns = [c for c in genes.columns if c.startswith("w_")]
    z = genes[columns].to_numpy(float)
    unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    control = rows.set_index("gene").loc["nontargeting_x", columns].to_numpy(float)
    assert np.allclose(control, 2 * unit[2])
    assert list(rows["gene"]) == list(genes["gene"])
    norms = np.linalg.norm(rows[columns].to_numpy(float), axis=1)
    assert (norms <= 2 + 1e-12).all() and (norms > 1.9).all()


def test_fewer_than_two_directions_are_refused():
    genes, guides, _planted_e = _planted(seed=2)
    columns = [c for c in genes.columns if c.startswith("w_")]
    genes[columns[1:]] = 0.0
    with pytest.raises(ValueError, match="refused"):
        guide_efficacy_profiles(genes, guides)


def test_the_fit_stops_at_the_iteration_cap_or_on_convergence():
    rng = np.random.default_rng(4)
    rows = rng.normal(0, 1, (5, 8))
    weights = np.full(5, 100.0)
    _m, _e, capped = _rank_one_efficacy(rows, weights, iters=1, emax=1.0)
    _m, _e, free = _rank_one_efficacy(rows, weights, iters=5000, emax=1.0)
    assert capped == 1
    assert 1 < free < 5000
    planted = np.outer([1.0, 0.5, 0.2], np.ones(4))
    _m, e, converged = _rank_one_efficacy(planted, np.ones(3), iters=5000, emax=1.0)
    assert converged < 5000 and np.allclose(e, [1.0, 0.5, 0.2], atol=1e-4)
    _m, e, _n = _rank_one_efficacy(-planted, np.ones(3), iters=20, emax=1.0)
    assert np.isfinite(e).all()


def test_the_aggregate_phase_blends_efficacy_profiles(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_nt_rows="barcode",
                  ops_profile_weighting="contrastive_wiener",
                  ops_guide_efficacy="jacks_blend", ops_guide_efficacy_iters=5,
                  ops_guide_efficacy_emax=1.0)
    options = report["profile_options"]
    assert options["guide_efficacy"]["iters"] == 5
    efficacy = pd.read_parquet(options["outputs"]["guides_efficacy_parquet"])
    assert efficacy["efficacy"].dropna().between(0, 1).all()
    rows = pd.read_parquet(options["outputs"]["genes_efficacy_parquet"])
    norms = np.linalg.norm(rows.filter(like="w_").to_numpy(), axis=1)
    assert (norms <= 2 + 1e-9).all()


def test_efficacy_needs_the_weighted_space(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    with pytest.raises(ValueError, match="weighted space"):
        _run(tmp_path, ops_guide_efficacy="jacks_blend")
    with pytest.raises(ValueError, match="ops_guide_efficacy must be"):
        _run(tmp_path, ops_profile_weighting="contrastive_wiener",
             ops_guide_efficacy="median")
