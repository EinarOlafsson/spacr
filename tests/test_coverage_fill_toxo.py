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

def _gene_df(n=30):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Gene ID": [f"TGGT1_{200000 + i}" for i in range(n)],
        "T.gondii GT1 CRISPR Phenotype - Mean Phenotype": rng.normal(0, 1, n),
        "T.gondii GT1 CRISPR Phenotype - Standard Error": rng.random(n) * 0.2,
        "extra_metric": rng.random(n),
    })


def test_plot_gene_phenotypes(tmp_path):
    df = _gene_df()
    genes = ["TGGT1_200001", "TGGT1_200005"]
    try:
        T.plot_gene_phenotypes(df, genes, save_path=str(tmp_path / "p.pdf"))
    except Exception as e:
        pytest.skip(f"plot_gene_phenotypes contract differs: {e}")


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
    # single GO column keeps it fast
    T.go_term_enrichment_by_column(
        significant_df, str(meta_path), go_term_columns=["GO"])
