"""Per-cell nuisance regression in the OPS profile path (ops_profile_nuisance).

A synthetic screen plants covariate slopes (cell cycle, size, density, a field
offset) into some features and a gene effect into another. The regression must
remove the planted slopes, keep the gene effect, leave the profiles untouched
when the setting is empty, and handle gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ops_tvn as T

GENES = [f"G{g}" for g in range(6)]
NAMES_OUT = ["cell_dapi_mean_intensity", "cell_a", "cell_b", "nucleus_c"]


def _well_table(well, rng, gene_effect, n_per=80):
    rows = []
    for gene in GENES + ["nontargeting"] * 3:
        for k in range(2):
            guide = f"{gene}_{k}" if gene != "nontargeting" else f"nt{len(rows)}_{k}"
            rows += [(gene, guide)] * n_per
    n = len(rows)
    p_s = rng.random(n)
    p_g2 = rng.random(n) * (1 - p_s)
    site = rng.integers(1, 4, n)
    frame = pd.DataFrame({
        "well": well, "sgRNA": [r[1] for r in rows], "gene": [r[0] for r in rows], "mitotic": False,
        "site": site, "p_G1": 1 - p_s - p_g2, "p_S": p_s, "p_G2": p_g2,
        "cell_area": np.exp(rng.normal(7, 0.3, n)), "nucleus_area": np.exp(rng.normal(5, 0.3, n)),
        "cell_neighbors_within_50": rng.poisson(4, n).astype(float),
        "cell_nearest_neighbor_distance": np.exp(rng.normal(3, 0.4, n)),
        "nucleus_centroid_y": rng.uniform(0, 1000, n), "nucleus_centroid_x": rng.uniform(0, 1000, n)})
    size = np.log(frame.cell_area.to_numpy())
    x = rng.standard_normal((n, 4)) * 0.5
    x[:, 0] += 4 * p_s + 1.5 * size + 0.5 * frame.cell_neighbors_within_50.to_numpy() + 2.0 * site
    x[:, 1] += 3 * size
    x[:, 2] += 2 * p_s
    x[:, 3] += gene_effect * (frame.gene.to_numpy() == "G1")
    for j, name in enumerate(NAMES_OUT):
        frame[name] = x[:, j]
    return frame


def _cells(folder, seed=0, gene_effect=3.0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for well in ("A1", "A2"):
        _well_table(well, rng, gene_effect).to_parquet(folder / f"raw_{well}.parquet", index=False)
    return folder


def _profile(cells, **extra):
    settings = {"plate": "P1", "ops_profile_cells": str(cells), "ops_tvn_whiten_epsilon": 0,
                "ops_profile_min_cells": 10, "ops_profile_recipe": {"nt": "barcode"}}
    settings.update(extra)
    return T.profile_cell_tables(settings)


def _prepared(cells, kinds):
    from spacr.ops_engine import _ph_feature_columns
    from spacr.tabular import table_columns

    paths = sorted(cells.glob("raw_*.parquet"))
    features = _ph_feature_columns(table_columns(str(paths[0]), canonicalise=False))
    recipe = T.profile_recipe({})
    metas, values = [], []
    for path in paths:
        meta, x, columns = T._profile_well(str(path), "P1", path.stem[4:], features, recipe, None,
                                           "nontargeting", nuisance=kinds)
        metas.append(meta)
        values.append(x)
    return metas, values, columns


def _corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])


def test_planted_slopes_are_removed_and_the_gene_effect_is_kept(tmp_path):
    cells = _cells(tmp_path / "cells")
    kinds = T.nuisance_kinds(["cell_cycle", "size", "density", "position", "field"])
    metas, values, columns = _prepared(cells, kinds)
    names = [c for c in metas[0].columns if c.startswith("nz_") and c != "nz_field"]
    control = [m.gene_symbol_0.str.startswith("nontargeting").to_numpy() for m in metas]
    covariates = [m[names].to_numpy(float) for m in metas]
    on = np.concatenate(control)
    before = np.vstack(values)[on]
    made = T.regress_nuisance(values, covariates, control, [m["nz_field"].to_numpy() for m in metas])
    n_nt = int(on.sum())
    assert made["n_fit"] == n_nt and made["coefficients"].shape == (len(names) + 1, len(columns))
    after = np.vstack(values)[on]
    z = np.vstack(covariates)[on]
    bound = 3 / np.sqrt(n_nt)
    dapi, planted, effect = (columns.index(c) for c in ("cell_dapi_mean_intensity", "cell_a", "nucleus_c"))
    p_s, area = names.index("nz_cc_p_S"), names.index("nz_size_cell_area")
    assert _corr(before[:, dapi], z[:, p_s]) > 0.3
    assert _corr(after[:, dapi], z[:, p_s]) < bound
    assert _corr(after[:, planted], z[:, area]) < bound
    out = np.vstack(values)
    gene_cells = np.concatenate([m.gene_symbol_0.eq("G1").to_numpy() for m in metas])
    assert out[gene_cells, effect].mean() - out[on, effect].mean() > 2.5
    means = np.array([x[(meta["nz_field"] == field).to_numpy()].mean(axis=0)
                      for x, meta in zip(values, metas) for field in meta["nz_field"].unique()])
    assert len(means) == 6 and np.allclose(means, means[0], atol=1e-3)


def test_slope_fitted_on_controls_ignores_the_gene_effect(tmp_path):
    kinds = T.nuisance_kinds(["cell_cycle", "size"])
    coefficients = []
    for effect in (0.0, 5.0):
        cells = _cells(tmp_path / f"e{int(effect)}", gene_effect=effect)
        metas, values, _ = _prepared(cells, kinds)
        names = [c for c in metas[0].columns if c.startswith("nz_")]
        control = [m.gene_symbol_0.str.startswith("nontargeting").to_numpy() for m in metas]
        coefficients.append(T.regress_nuisance(values, [m[names].to_numpy(float) for m in metas],
                                               control)["coefficients"])
    assert np.allclose(coefficients[0], coefficients[1])


def test_empty_setting_is_identical_to_no_setting(tmp_path):
    cells = _cells(tmp_path / "cells")
    base = _profile(cells)
    for off in ({"ops_profile_nuisance": []}, {"ops_profile_nuisance": None},
                {"ops_profile_nuisance": [], "ops_profile_nuisance_fit": "all"}):
        genes, guides, info = _profile(cells, **off)
        pd.testing.assert_frame_equal(genes, base[0])
        pd.testing.assert_frame_equal(guides, base[1])
        assert "nuisance" not in info
    genes, _, info = _profile(cells, ops_profile_nuisance=["size", "cell_cycle"])
    assert info["nuisance"]["sources"] == ["cell_cycle", "size"] and info["nuisance"]["n_fit"] > 0
    pd.testing.assert_frame_equal(genes, _profile(cells, ops_profile_nuisance=["size", "cell_cycle"])[0])
    assert not genes.equals(base[0])


def test_gaps_in_features_and_covariates_are_safe():
    rng = np.random.default_rng(3)
    n = 400
    z = rng.standard_normal((n, 2))
    y = (z @ rng.standard_normal((2, 4)) + 0.1 * rng.standard_normal((n, 4))).astype(np.float32)
    y[5, 1] = np.nan
    y[17, 3] = np.nan
    z[9, 0] = np.nan
    values = [y.copy()]
    made = T.regress_nuisance(values, [z], [np.ones(n, bool)])
    assert np.isnan(values[0][5, 1]) and np.isnan(values[0][17, 3])
    assert np.isfinite(np.delete(values[0], [5, 17], axis=0)).all()
    assert made["n_fit"] == n - 1
    good = np.delete(np.arange(n), [5, 9, 17])
    assert np.abs(values[0][good]).std(axis=0).max() < 0.3
    assert np.isfinite(made["coefficients"]).all()


def test_choices_are_checked_and_missing_columns_named(tmp_path):
    assert T.nuisance_kinds(None) == () and T.nuisance_kinds([]) == () and T.nuisance_kinds("") == ()
    assert T.nuisance_kinds("size") == ("size",)
    with pytest.raises(ValueError, match="unknown sources"):
        T.nuisance_kinds(["age"])
    cells = _cells(tmp_path / "cells")
    with pytest.raises(ValueError, match="ops_profile_nuisance_fit"):
        _profile(cells, ops_profile_nuisance=["size"], ops_profile_nuisance_fit="some")
    path = cells / "raw_A1.parquet"
    pd.read_parquet(path).drop(columns=["p_S", "p_G2", "p_G1"]).to_parquet(path)
    with pytest.raises(ValueError, match="cell-cycle"):
        _profile(cells, ops_profile_nuisance=["cell_cycle"])
