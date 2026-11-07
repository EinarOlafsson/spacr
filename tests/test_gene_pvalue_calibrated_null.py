"""The per-gene null of an optical pooled screen is calibrated.

The cell bootstrap ignored the variance between guides and called a third
of the non-targeting guides of Funk et al.'s plate A hits. The calibrated
null adds that variance. The earlier function is kept here verbatim, as it
was before spaCR 1.5.1.8, as the reference for the cell bootstrap, and the
plate A pilot's test (hits2.py) as the reference for the calibrated null.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import bootstrap_gene_pvalues


def _old_bootstrap(control_values, guides, features, *, n_boot=1000, seed=0,
                   max_cells=2000):
    """bootstrap_gene_pvalues before the calibrated null, verbatim."""
    from spacr.multiple_testing import _multipletests

    rng = np.random.default_rng(int(seed))
    pool = {feature: pd.to_numeric(control_values[feature], errors="coerce")
            .dropna().to_numpy() for feature in features}
    rows = []
    for gene, members in guides.groupby("gene", sort=True):
        sizes = np.minimum(members["n_cells"].to_numpy(int), int(max_cells))
        row: Dict[str, Any] = {"gene": gene}
        for feature in features:
            values = pool[feature]
            observed = float(np.nanmedian(members[feature]))
            if values.size == 0 or not np.isfinite(observed):
                row[f"{feature}_pvalue"] = np.nan
                continue
            centre = float(np.median(values))
            null = np.empty(int(n_boot))
            per_guide = np.empty((int(n_boot), sizes.size))
            for index, size in enumerate(sizes):
                draws = rng.choice(values, size=(int(n_boot), max(1, size)))
                per_guide[:, index] = np.median(draws, axis=1)
            null[:] = np.median(per_guide, axis=1)
            extreme = np.abs(null - centre) >= abs(observed - centre)
            row[f"{feature}_pvalue"] = float(
                (extreme.sum() + 1) / (int(n_boot) + 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    for feature in features:
        column = f"{feature}_pvalue"
        if column not in out:
            continue
        fdr = pd.Series(np.nan, index=out.index)
        finite = out[column].notna()
        if finite.any():
            fdr[finite] = _multipletests(out.loc[finite, column].to_numpy(),
                                         "fdr_bh")[1]
        out[f"{feature}_fdr"] = fdr
    return out


def _screen(seed=0, n_control_guides=200, between_sd=0.3, cells=200):
    """Guide medians whose guides vary beyond cell sampling, as in an OPS."""
    rng = np.random.default_rng(seed)
    rows, control_cells = [], []
    for g in range(n_control_guides):
        shift = rng.normal(0, between_sd)
        values = rng.normal(shift, 1.0, cells)
        control_cells.append(values)
        rows.append({"guide": f"nt{g}", "gene": "nontargeting",
                     "n_cells": cells, "x": float(np.median(values))})
    return pd.DataFrame(rows), pd.DataFrame({"x": np.concatenate(control_cells)})


def test_the_calibrated_null_holds_its_size_where_the_cell_bootstrap_does_not():
    controls, control_cells = _screen()
    held_out = controls.iloc[100:].copy()
    held_out["gene"] = [f"fake{i // 4}" for i in range(len(held_out))]
    reference = controls.iloc[:100]
    calibrated = bootstrap_gene_pvalues(
        control_cells, held_out, ["x"], n_boot=2000, control_guides=reference)
    legacy = bootstrap_gene_pvalues(
        control_cells, held_out, ["x"], n_boot=300, null="cell_bootstrap")
    size = (calibrated["x_pvalue"] < 0.05).mean()
    assert size <= 0.12
    assert (legacy["x_pvalue"] < 0.05).mean() > 2 * max(size, 0.05)
    fit = calibrated.attrs["null"]["x"]
    assert fit["between_variance"] == pytest.approx(0.09, rel=0.5)


def test_the_cell_bootstrap_is_the_earlier_test_value_for_value():
    controls, control_cells = _screen(seed=3, n_control_guides=40, cells=30)
    genes = controls.iloc[:20].assign(
        gene=[f"g{i // 2}" for i in range(20)])
    expected = _old_bootstrap(control_cells, genes, ["x"], n_boot=100, seed=5)
    actual = bootstrap_gene_pvalues(control_cells, genes, ["x"], n_boot=100,
                                    seed=5, null="cell_bootstrap")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def _pilot_null_p(rng, obs, sizes, m0, s2b, s2c, nsim):
    """The plate A pilot's calibrated test (hits2.py), verbatim."""
    sd = np.sqrt(s2b + np.pi / 2 * s2c / np.asarray(sizes, float))
    sims = np.median(m0 + rng.standard_normal((nsim, len(sd))) * sd, axis=1)
    return (np.sum(np.abs(sims - m0) >= abs(obs - m0)) + 1) / (nsim + 1)


def test_the_calibrated_null_reproduces_the_plate_a_pilot_exactly():
    controls, control_cells = _screen(seed=7, n_control_guides=60, cells=80)
    rng = np.random.default_rng(1)
    genes = pd.DataFrame({
        "guide": [f"t{i}" for i in range(40)],
        "gene": [f"G{i // 4:02d}" for i in range(40)],
        "n_cells": rng.integers(20, 300, 40),
        "x": rng.normal(0, 0.5, 40), "y": rng.normal(0, 0.5, 40)})
    controls["y"] = rng.normal(0, 0.2, len(controls))
    control_cells["y"] = rng.normal(0, 1, len(control_cells))
    out = bootstrap_gene_pvalues(control_cells, genes, ["x", "y"],
                                 n_boot=500, seed=0, control_guides=controls)
    pilot = np.random.default_rng(0)
    params = {}
    for f in ("x", "y"):
        v = control_cells[f].to_numpy(float)
        s2c = (1.4826 * np.median(np.abs(v - np.median(v)))) ** 2
        med = controls[f].to_numpy(float)
        vm = np.pi / 2 * s2c / controls.n_cells.to_numpy(float)
        params[f] = (float(np.median(med)),
                     max(0.0, float(np.var(med, ddof=1) - vm.mean())), s2c)
    expected = []
    for _gene, d in genes.groupby("gene"):
        expected.append([_pilot_null_p(pilot, float(np.median(d[f])),
                                       d.n_cells, *params[f], 500)
                         for f in ("x", "y")])
    expected = np.asarray(expected)
    assert np.array_equal(out[["x_pvalue", "y_pvalue"]].to_numpy(), expected)
    for f in ("x", "y"):
        m0, s2b, s2c = params[f]
        assert out.attrs["null"][f] == {"centre": m0, "cell_variance": s2c,
                                        "between_variance": s2b}


def test_the_calibrated_null_needs_the_control_guides():
    controls, control_cells = _screen(n_control_guides=5, cells=10)
    with pytest.raises(ValueError, match="control_guides"):
        bootstrap_gene_pvalues(control_cells, controls, ["x"])
    with pytest.raises(ValueError, match="null must be one of"):
        bootstrap_gene_pvalues(control_cells, controls, ["x"], null="exact")


def test_a_feature_the_controls_cannot_calibrate_has_no_pvalue():
    controls = pd.DataFrame({"guide": ["a"], "gene": ["nt"], "n_cells": [5],
                             "x": [0.0], "y": [0.0]})
    genes = pd.DataFrame({"guide": ["g"], "gene": ["G"], "n_cells": [5],
                          "x": [1.0], "y": [np.nan]})
    cells = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    out = bootstrap_gene_pvalues(cells, genes, ["x", "y"], n_boot=50,
                                 control_guides=pd.concat([controls] * 3))
    assert np.isfinite(out.loc[0, "x_pvalue"])
    assert np.isnan(out.loc[0, "y_pvalue"])
    one = bootstrap_gene_pvalues(cells, genes, ["x"], n_boot=50,
                                 control_guides=controls)
    assert np.isnan(one.loc[0, "x_pvalue"]) and one.attrs["null"]["x"] is None
