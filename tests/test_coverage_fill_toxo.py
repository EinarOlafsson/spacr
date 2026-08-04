"""Coverage-fill for spacr.toxo pure-logic + plotting helpers (Agg)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import toxo as T


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _normalize_y_lims — every branch
# ---------------------------------------------------------------------------

def test_normalize_y_lims_none():
    broken, lo, hi = T._normalize_y_lims(None, np.array([1.0, 2.0, np.inf]))
    assert broken is False and hi is None and lo[0] == 0.0


def test_normalize_y_lims_none_empty():
    broken, lo, hi = T._normalize_y_lims(None, np.array([np.inf, np.inf]))
    assert broken is False and lo == [0.0, 1.0]


def test_normalize_y_lims_single():
    broken, lo, hi = T._normalize_y_lims([0, 10], np.array([1.0]))
    assert broken is False and lo == [0, 10]


def test_normalize_y_lims_broken():
    broken, lo, hi = T._normalize_y_lims([[0, 5], [10, 20]], np.array([1.0]))
    assert broken is True and lo == [0, 5] and hi == [10, 20]


def test_normalize_y_lims_bad():
    with pytest.raises(ValueError):
        T._normalize_y_lims([1, 2, 3], np.array([1.0]))
    with pytest.raises(ValueError):
        T._normalize_y_lims(["a", "b"], np.array([1.0]))


# ---------------------------------------------------------------------------
# plot_gene_phenotypes / plot_gene_heatmaps
# ---------------------------------------------------------------------------

_MEAN_COL = "T.gondii GT1 CRISPR Phenotype - Mean Phenotype"
_SE_COL = "T.gondii GT1 CRISPR Phenotype - Standard Error"


def _gene_df(n=30):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Gene ID": [f"TGGT1_{200000 + i}" for i in range(n)],
        _MEAN_COL: rng.normal(0, 1, n),
        _SE_COL: rng.random(n) * 0.2,
        "extra_metric": rng.random(n),
    })


def test_plot_gene_phenotypes(tmp_path):
    """The rank curve covers every gene and each highlighted gene is scattered
    at *its own* (rank, phenotype).

    The broad ``except Exception: pytest.skip`` this test used to carry is
    gone: the call succeeds deterministically, and a skip would have hidden
    exactly the failure the assertions below are here to catch.
    """
    df = _gene_df()
    genes = ["TGGT1_200001", "TGGT1_200005"]
    out = tmp_path / "p.pdf"
    T.plot_gene_phenotypes(df, genes, save_path=str(out))

    assert out.is_file() and out.stat().st_size > 0

    fig = plt.gcf()
    assert len(fig.axes) == 1
    ax = fig.axes[0]

    # The ranked mean-phenotype line: one point per gene, y sorted ascending
    # because rank *is* the sort order.
    assert len(ax.lines) == 1
    x, y = (np.asarray(a) for a in (ax.lines[0].get_xdata(),
                                    ax.lines[0].get_ydata()))
    assert list(x) == list(range(1, len(df) + 1))
    assert np.all(np.diff(y) >= 0)
    assert y == pytest.approx(
        np.sort(df[_MEAN_COL].to_numpy()))

    # Independently work out where each highlighted gene must land.
    ranked = df.sort_values(_MEAN_COL).reset_index(drop=True)
    expected = {}
    for gene in genes:
        pos = int(ranked.index[ranked["Gene ID"] == gene][0])
        expected[gene] = (pos + 1, ranked.loc[pos, _MEAN_COL])
    # The two highlights must be distinguishable — same point twice would
    # pass a weaker check.
    assert len(set(expected.values())) == 2

    scatters = {c.get_label(): np.asarray(c.get_offsets())
                for c in ax.collections
                if c.get_label().startswith("Highlighted Gene: ")}
    assert set(scatters) == {f"Highlighted Gene: {g}" for g in genes}
    for gene, (rank, value) in expected.items():
        got = scatters[f"Highlighted Gene: {gene}"]
        assert got.shape == (1, 2)
        assert got[0] == pytest.approx([rank, value])

    # Each highlight is labelled with its gene name.
    assert sorted(t.get_text() for t in ax.texts) == sorted(genes)
    assert ax.get_xlabel() == "Rank"
    assert ax.get_ylabel() == "Mean Phenotype"


def test_plot_gene_heatmaps(tmp_path):
    df = _gene_df()
    # gene_list is matched against extract_gene_id(Gene ID) → the numeric part
    genes = ["200001", "200005", "200009"]
    cols = ["T.gondii GT1 CRISPR Phenotype - Mean Phenotype", "extra_metric"]
    T.plot_gene_heatmaps(df, genes, cols, normalize=True,
                         save_path=str(tmp_path / "h.pdf"))
    assert (tmp_path / "h.pdf").exists()


# ---------------------------------------------------------------------------
# custom_volcano_plot
# ---------------------------------------------------------------------------

def test_custom_volcano_plot(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    # feature -> variable -> gene_nr (split on '_' first token)
    features = [f"{220000 + i}_1" for i in range(n)]
    data = pd.DataFrame({
        "feature": features,
        "coefficient": rng.normal(0, 0.4, n),
        "p_value": np.clip(np.abs(rng.normal(0.05, 0.05, n)), 1e-8, 1),
    })
    metadata = pd.DataFrame({
        "gene_nr": [str(220000 + i) for i in range(n)],
        "tagm_location": rng.choice(["cytosol", "nucleus - chromatin",
                                     "dense granules", "unknown"], n),
    })
    meta_path = tmp_path / "meta.csv"
    metadata.to_csv(meta_path, index=False)
    hits = T.custom_volcano_plot(
        data, str(meta_path), point_size=50, figsize=6,
        save_path=str(tmp_path / "v.pdf"))
    assert (tmp_path / "v.pdf").exists()
    assert isinstance(hits, list)


def test_custom_volcano_plot_broken_axis(tmp_path):
    rng = np.random.default_rng(2)
    n = 40
    data = pd.DataFrame({
        "feature": [f"{220000 + i}_1" for i in range(n)],
        "coefficient": rng.normal(0, 0.4, n),
        "p_value": np.clip(np.abs(rng.normal(0.05, 0.05, n)), 1e-8, 1),
    })
    metadata = pd.DataFrame({
        "gene_nr": [str(220000 + i) for i in range(n)],
        "tagm_location": ["cytosol"] * n,
    })
    # broken y-axis path
    hits = T.custom_volcano_plot(
        data, metadata, figsize=6, y_lims=[[0, 3], [5, 12]])
    assert isinstance(hits, list)


# ---------------------------------------------------------------------------
# go_term_enrichment_by_column
# ---------------------------------------------------------------------------

def test_go_term_enrichment_by_column(tmp_path):
    rng = np.random.default_rng(3)
    n = 40
    gene_nrs = [str(220000 + i) for i in range(n)]
    go_terms = ["metabolism", "signaling", "transport", "binding"]
    metadata = pd.DataFrame({
        "Gene ID": [f"TGGT1_{g}" for g in gene_nrs],
        "GO": [";".join(rng.choice(go_terms, size=2)) for _ in range(n)],
    })
    meta_path = tmp_path / "go_meta.csv"
    metadata.to_csv(meta_path, index=False)
    # hits = first 10 genes
    significant_df = pd.DataFrame({"n_gene": gene_nrs[:10]})

    # Work out, from the same input and without touching the product code,
    # what enrichment score every GO term must get: (hits with term / hits) /
    # (genes with term / genes).
    all_counts = metadata["GO"].str.split(";").explode().value_counts()
    hit_counts = (metadata.iloc[:10]["GO"].str.split(";")
                  .explode().value_counts())
    expected = {
        term: (hit_counts.get(term, 0) / 10) / (all_counts[term] / n)
        for term in all_counts.index
    }
    assert len(expected) == len(go_terms)
    # The fixture is only a real test if the terms differ: metabolism is
    # over-represented among the hits and the other three are depleted, so a
    # plot drawing a constant (or nothing) cannot match.
    assert expected["metabolism"] > 1.0
    assert all(v < 1.0 for k, v in expected.items() if k != "metabolism")

    plt.close("all")
    # single GO column keeps it fast
    T.go_term_enrichment_by_column(
        significant_df, str(meta_path), go_term_columns=["GO"])

    # One per-column figure plus the combined figure.
    figs = [plt.figure(i) for i in plt.get_fignums()]
    assert len(figs) == 2
    per_column, combined = figs

    assert per_column.axes and combined.axes
    per_ax = per_column.axes[0]
    assert per_ax.get_title() == "GO Term Enrichment Analysis for GO"
    assert len(per_ax.collections) == 1
    points = np.asarray(per_ax.collections[0].get_offsets())
    # One scatter point per GO term, at (enrichment score, -log10 p).
    assert points.shape == (len(expected), 2)
    assert sorted(points[:, 0]) == pytest.approx(sorted(expected.values()))
    # Enrichment is plotted descending, so metabolism leads.
    assert points[0, 0] == pytest.approx(expected["metabolism"])
    # -log10(p) is finite and the enriched term is the most significant one.
    assert np.isfinite(points[:, 1]).all()
    assert points[0, 1] == max(points[:, 1])
    assert per_ax.get_xlabel() == "Enrichment Score"
    assert per_ax.get_ylabel() == "-log10(P-value)"
    # Every term is named in the legend.
    legend = {t.get_text() for t in per_ax.get_legend().get_texts()}
    assert set(expected) <= legend

    comb_ax = combined.axes[0]
    assert comb_ax.get_title() == "Combined GO Term Enrichment Analysis"
    assert len(comb_ax.collections) == 1
    comb_points = np.asarray(comb_ax.collections[0].get_offsets())
    assert sorted(comb_points[:, 0]) == pytest.approx(sorted(expected.values()))
    # Each point is annotated with its GO term.
    assert sorted(t.get_text() for t in comb_ax.texts) == sorted(expected)
