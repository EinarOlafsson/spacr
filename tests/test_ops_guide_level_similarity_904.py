"""Item 904: guide-level gene-gene similarity with calibrated pair P values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (guide_level_similarity,
                                            guide_mean_vectors)


def _guides(seed=0, genes=40, dims=8):
    """Gene pairs (G00, G01) and (G02, G03) share a direction; the rest are
    noise, and twelve control guides form three pseudo-genes."""
    rng = np.random.default_rng(seed)
    shared = {0: rng.normal(0, 1, dims), 2: rng.normal(0, 1, dims)}
    rows = []
    for g in range(genes):
        base = shared.get(g - g % 2) if g < 4 else None
        for k in range(4):
            v = rng.normal(0, 0.3 if base is not None else 1, dims)
            if base is not None:
                v = v + base
            rows.append({"guide": f"g{g}_{k}", "gene": f"G{g:02d}",
                         **{f"PC_{d}": v[d] for d in range(dims)}})
    for k in range(12):
        rows.append({"guide": f"nt{k}", "gene": f"NTpseudo_{k // 4:02d}",
                     **{f"PC_{d}": rng.normal(0, 1) for d in range(dims)}})
    return pd.DataFrame(rows), [f"PC_{d}" for d in range(dims)]


def test_genes_whose_guides_agree_get_small_pair_p_values():
    guides, columns = _guides()
    result = guide_level_similarity(guides, columns, null_size=2000,
                                    control_null_size=500)
    pairs = result["pairs"].set_index(["gene_a", "gene_b"])
    assert pairs.loc[("G00", "G01"), "p"] < 0.01
    assert pairs.loc[("G02", "G03"), "p"] < 0.01
    assert pairs["p"].median() > 0.2
    summary = result["summary"]
    assert summary["active_genes"] == 40 and summary["pairs_tested"] == 780
    assert summary["calibration_NT_pseudo_x_active (control)"]["pairs"] == 120
    assert list(result["names"][-3:]) == ["NTpseudo_00", "NTpseudo_01", "NTpseudo_02"]
    again = guide_level_similarity(guides, columns, null_size=2000,
                                   control_null_size=500)
    assert np.array_equal(again["pairs"]["p"], result["pairs"]["p"])


def test_only_the_named_genes_are_tested():
    guides, columns = _guides(seed=1)
    result = guide_level_similarity(guides, columns, genes=["G00", "G01", "G05", "nope"],
                                    null_size=500, control_null_size=100)
    assert result["summary"]["pairs_tested"] == 3


def test_normalised_means_keep_single_guide_rows_from_dominating():
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    names, raw, k = guide_mean_vectors(rows, ["A", "A", "B"], normalise=False)
    _n, unit, _k = guide_mean_vectors(rows, ["A", "A", "B"])
    assert list(names) == ["A", "B"] and list(k) == [2, 1]
    assert np.linalg.norm(raw[0]) == pytest.approx(np.sqrt(0.5))
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)


def test_the_aggregate_phase_writes_pair_p_values(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_nt_rows="pseudo", ops_profile_similarity="guide_level",
                  ops_guide_null_size=300)["profile_options"]
    pairs = pd.read_parquet(report["outputs"]["gene_pairs_parquet"])
    assert len(pairs) == 40 * 39 // 2
    assert pairs["p"].between(1 / 301, 1).all()
    assert report["guide_level"]["active_genes"] == 40
    with pytest.raises(ValueError, match="ops_profile_similarity must be"):
        _run(tmp_path, ops_profile_similarity="euclidean")
