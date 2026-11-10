"""Item 907: contrastive generalised-eigen Wiener weighting of gene profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (contrastive_weighting,
                                            shrink_gene_profiles)


def _screen(seed=0, genes=60, guides=4, dims=12, controls=40):
    """Guides of genes with signal in the first three dimensions, plus
    control guides that carry only noise."""
    rng = np.random.default_rng(seed)
    signal = np.zeros((genes, dims))
    signal[:, :3] = rng.normal(0, 3, (genes, 3))
    rows = []
    for g in range(genes):
        for k in range(guides):
            rows.append({"guide": f"g{g}_{k}", "gene": f"G{g:02d}",
                         "n_cells": int(rng.integers(50, 200)),
                         **{f"PC_{d}": signal[g, d] + rng.normal(0, 1)
                            for d in range(dims)}})
    for k in range(controls):
        rows.append({"guide": f"nt{k}", "gene": f"nontargeting_nt{k}",
                     "n_cells": 150,
                     **{f"PC_{d}": rng.normal(0, 1) for d in range(dims)}})
    guides_table = pd.DataFrame(rows)
    columns = [f"PC_{d}" for d in range(dims)]
    grouped = guides_table.groupby("gene", sort=True)
    genes_table = grouped[columns].median()
    genes_table.insert(0, "n_cells", grouped["n_cells"].sum())
    return genes_table.reset_index(), guides_table, columns


def test_signal_directions_are_kept_and_noise_directions_weighted_down():
    genes, guides, columns = _screen()
    rows, guide_rows, info = contrastive_weighting(genes, guides, columns)
    assert info["directions_kept"] >= 2
    weights = info["weights"]
    assert (np.diff(info["lambda"]) <= 0).all()
    assert weights[0] > 0.5 and (weights >= 0).all() and (weights < 1).all()
    assert list(rows.columns[:2]) == ["gene", "n_cells"]
    assert len(rows) == len(genes) and len(guide_rows) == len(guides)
    assert info["nt_guides"] == 40 and info["kbar"] == pytest.approx(4.0)


def test_dropping_zero_weights_keeps_every_cosine():
    genes, guides, columns = _screen(seed=1)
    full, _g, info = contrastive_weighting(genes, guides, columns)
    kept, _g2, _i = contrastive_weighting(genes, guides, columns, drop_zero=True)
    a = full.filter(like="w_").to_numpy()
    b = kept.filter(like="w_").to_numpy()
    assert b.shape[1] == int((info["weights"] > 0).sum()) < a.shape[1]
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    assert np.allclose(unit(a) @ unit(a).T, unit(b) @ unit(b).T, atol=1e-10)


def test_genes_with_few_cells_are_left_out_first():
    genes, guides, columns = _screen(seed=2)
    genes.loc[0, "n_cells"] = 5
    rows, _g, _i = contrastive_weighting(genes, guides, columns, min_cells=10)
    assert genes.loc[0, "gene"] not in set(rows["gene"])
    rows, _g, _i = contrastive_weighting(genes, guides, columns, min_cells=None)
    assert genes.loc[0, "gene"] in set(rows["gene"])


def test_profiles_shrunk_toward_the_controls_are_refused():
    rng = np.random.default_rng(3)
    genes, guides, columns = _screen(seed=3)
    noise = guides.copy()
    noise[columns] = rng.normal(0, 1, (len(noise), len(columns)))
    shrunk, _f, _p = shrink_gene_profiles(noise, columns)
    shrunk.insert(1, "n_cells", 500)
    with pytest.raises(ValueError, match="refused"):
        contrastive_weighting(shrunk, noise, columns)


def _run(tmp_path, **options):
    """The aggregate phase of the profile plate with ``options``."""
    from spacr import ops_engine
    from tests.test_ops_gene_level_aggregation_883 import (PLATE, profile_plate,
                                                           profile_settings)

    db, wells = profile_plate(tmp_path)
    return ops_engine._ph_aggregate(db, PLATE, wells,
                                    profile_settings(tmp_path, **options),
                                    str(tmp_path))


def test_the_aggregate_phase_writes_weighted_profiles(tmp_path):
    report = _run(tmp_path, ops_nt_rows="barcode",
                  ops_profile_weighting="contrastive_wiener")
    options = report["profile_options"]
    assert options["weighting"]["directions_kept"] >= 2
    rows = pd.read_parquet(options["outputs"]["genes_weighted_parquet"])
    guides = pd.read_parquet(options["outputs"]["guides_weighted_parquet"])
    assert rows.columns[2].startswith("w_") and guides.columns[3].startswith("w_")
    assert rows["gene"].str.startswith("nontargeting_").sum() == 12


def test_profile_options_need_tvn_and_weighting_is_not_stacked(tmp_path):
    with pytest.raises(ValueError, match="TVN profiles"):
        _run(tmp_path, ops_profile_normalisation="robust_z",
             ops_profile_weighting="contrastive_wiener")
    with pytest.raises(ValueError, match="choose one"):
        _run(tmp_path, ops_profile_weighting="contrastive_wiener",
             ops_profile_shrinkage="eb_nt")
    with pytest.raises(ValueError, match="ops_profile_weighting must be"):
        _run(tmp_path, ops_profile_weighting="pca")
