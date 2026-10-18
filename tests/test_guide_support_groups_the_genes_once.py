"""``guide_support`` finds each gene's own row without a scan per gene.

It filtered the whole gene table once for every gene, a full boolean mask
per gene, and lesson 13's Regression results draw it on the GUI thread.
The gene rows are now grouped once; the table it returns is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr import guide_concordance as G


def _results(n_genes=40):
    rows = []
    rng = np.random.default_rng(0)
    for g in range(n_genes):
        rows.append({"feature": f"gene_fraction:gene[{g}]",
                     "coefficient": float(rng.normal()),
                     "p_value": float(rng.uniform())})
        for k in range(1 + g % 3):
            rows.append({"feature": f"fraction:grna[{g}_g{k}]",
                         "coefficient": float(rng.normal()),
                         "p_value": float(rng.uniform())})
    rows.append({"feature": "fraction:grna[orphan_g1]", "coefficient": 0.1,
                 "p_value": 0.5})
    rows.append({"feature": "gene_fraction:gene[3]", "coefficient": 9.0,
                 "p_value": 0.0001})
    return pd.DataFrame(rows)


def _scanned(results, alpha=0.05):
    """The previous per-gene scan, kept as the reference."""
    frame = results.copy()
    frame["feature"] = frame["feature"].astype(str)
    frame["_gene"] = frame["feature"].map(G._gene_of)
    frame["_p"] = pd.to_numeric(frame.get("p_value"), errors="coerce")
    frame["_effect"] = pd.to_numeric(frame["coefficient"], errors="coerce")
    guides = frame[frame["feature"].map(G._is_guide_term)
                   & frame["_gene"].notna()]
    genes = frame[frame["feature"].map(G._is_gene_term)
                  & frame["_gene"].notna()]
    out = {}
    for gene, _block in guides.groupby("_gene"):
        gene_row = genes[genes["_gene"] == gene]
        out[gene] = (float(gene_row["_p"].min()) if len(gene_row) else np.nan,
                     float(gene_row["_effect"].iloc[0])
                     if len(gene_row) else np.nan)
    return out


def test_the_gene_columns_match_the_scan():
    results = _results()
    support = G.guide_support(results)
    expected = _scanned(results)

    assert set(support.index) == set(expected)
    for gene, (gene_p, coefficient) in expected.items():
        row = support.loc[gene]
        assert (np.isnan(gene_p) and np.isnan(row["gene_p"])) or \
            row["gene_p"] == gene_p
        assert (np.isnan(coefficient) and np.isnan(row["gene_coefficient"])) \
            or row["gene_coefficient"] == coefficient


def test_the_gene_table_is_not_masked_once_per_gene(monkeypatch):
    results = _results(n_genes=60)
    masks = []
    real = pd.Series.__eq__

    def counted(self, other):
        if self.name == "_gene":
            masks.append(other)
        return real(self, other)

    monkeypatch.setattr(pd.Series, "__eq__", counted)
    G.guide_support(results)

    assert len(masks) <= 1, f"{len(masks)} per-gene masks"
