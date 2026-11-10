"""Item 902: guide-reliability empirical-Bayes shrinkage of gene profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import shrink_gene_profiles


def _guides(seed=0, genes=80, dims=5):
    """Genes with a common effect; the first half have guides that agree,
    the second half guides that disagree."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(genes):
        effect = rng.normal(0, 1, dims)
        spread = 0.1 if g < genes // 2 else 3.0
        for k in range(4):
            rows.append({"guide": f"g{g}_{k}", "gene": f"G{g:02d}", "n_cells": 200,
                         **{f"PC_{d}": effect[d] + rng.normal(0, spread)
                            for d in range(dims)}})
    for k in range(40):
        rows.append({"guide": f"nt{k}", "gene": f"NTpseudo_{k // 4:02d}",
                     "n_cells": 200,
                     **{f"PC_{d}": rng.normal(0, 0.1) for d in range(dims)}})
    return pd.DataFrame(rows), [f"PC_{d}" for d in range(dims)]


def test_genes_whose_guides_disagree_are_shrunk_more():
    guides, columns = _guides()
    genes, factor, params = shrink_gene_profiles(guides, columns)
    agree = factor[[f"G{g:02d}" for g in range(40)]]
    disagree = factor[[f"G{g:02d}" for g in range(40, 80)]]
    assert agree.mean() > disagree.mean() + 0.2
    assert ((factor > 0) & (factor < 1)).all()
    assert params["prior_df"] == 4.0 and len(params["tau2"]) == len(columns)
    means = guides.groupby("gene")[columns].mean()
    centre = guides[guides.gene.str.startswith("NTpseudo")][columns].mean()
    shrunk = genes.set_index("gene").loc["G00", columns]
    assert np.all(np.abs(shrunk) <= np.abs(means.loc["G00"] - centre) + 1e-12)


def test_the_prior_weight_moves_the_pooled_noise():
    guides, columns = _guides(seed=1)
    _g, light, _p = shrink_gene_profiles(guides, columns, prior_df=0.5)
    _g, heavy, _p = shrink_gene_profiles(guides, columns, prior_df=50.0)
    noisy = [f"G{g:02d}" for g in range(40, 80)]
    assert heavy[noisy].mean() > light[noisy].mean()


def test_a_single_guide_gene_uses_the_control_noise_alone():
    guides, columns = _guides(seed=2)
    single = guides[guides.gene != "G00"]
    single = pd.concat([single, guides[guides.gene == "G00"].iloc[:1]])
    genes, factor, _p = shrink_gene_profiles(single, columns,
                                             n_cells=single["n_cells"])
    assert np.isfinite(factor["G00"])
    assert set(genes.columns) == {"gene", *columns}


def test_the_aggregate_phase_shrinks_plain_and_phase_profiles(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    for name in ("plain", "phase", "bad"):
        (tmp_path / name).mkdir()
    plain = _run(tmp_path / "plain", ops_profile_shrinkage="eb_nt",
                 ops_profile_shrinkage_prior_df=4)["profile_options"]
    assert 0 < plain["shrinkage"]["median_factor"] < 1
    table = pd.read_parquet(plain["outputs"]["genes_shrink_parquet"])
    assert list(table.columns[:3]) == ["gene", "n_guides", "n_cells"]
    phase = _run(tmp_path / "phase", ops_profile_shrinkage="eb_nt",
                 ops_profile_cell_cycle="G1")["profile_options"]
    assert phase["shrinkage"]["per_phase"]
    table = pd.read_parquet(phase["outputs"]["genes_shrink_parquet"])
    assert all(c.startswith("G1_") for c in table.columns[3:])
    with pytest.raises(ValueError, match="ops_profile_shrinkage must be"):
        _run(tmp_path / "bad", ops_profile_shrinkage="james_stein")
