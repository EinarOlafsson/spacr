"""Coverage-fill for spacr.sim analysis/plotting functions over a sweep DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import sim as SIM


@pytest.fixture(autouse=True)
def _noop_saves(monkeypatch):
    # save_plot / save_shap_plot write PDFs into ./figures — stub them out.
    monkeypatch.setattr(SIM, "save_plot", lambda *a, **k: None)
    monkeypatch.setattr(SIM, "save_shap_plot", lambda *a, **k: None)
    yield
    plt.close("all")


def _sweep_df(n=50):
    rng = np.random.default_rng(0)
    cols = {
        "number_of_active_genes": rng.integers(5, 50, n),
        "number_of_control_genes": rng.integers(5, 50, n),
        "avg_reads_per_gene": rng.integers(50, 500, n),
        "classifier_accuracy": rng.uniform(0.6, 0.99, n),
        "nr_plates": rng.integers(1, 5, n),
        "number_of_genes": rng.integers(50, 500, n),
        "avg_genes_per_well": rng.uniform(1, 10, n),
        "avg_cells_per_well": rng.uniform(5, 50, n),
        "sequencing_error": rng.uniform(0.001, 0.1, n),
        "well_ineq_coeff": rng.uniform(0.5, 2.0, n),
        "gene_ineq_coeff": rng.uniform(0.5, 2.0, n),
        "optimal_threshold": rng.uniform(0.2, 0.8, n),
        "accuracy": rng.uniform(0.5, 1.0, n),
        "genes_per_well_gini": rng.uniform(0, 1, n),
        "wells_per_gene_gini": rng.uniform(0, 1, n),
        "roc_auc": rng.uniform(0.5, 1.0, n),
    }
    # target correlated with a couple of features
    cols["prauc"] = (
        0.3 * cols["classifier_accuracy"]
        + 0.2 * (cols["avg_reads_per_gene"] / 500)
        + rng.normal(0, 0.05, n)
    )
    return pd.DataFrame(cols)


def test_plot_correlation_matrix():
    fig = SIM.plot_correlation_matrix(_sweep_df(), annot=False, clean=True)
    assert fig is not None


def test_plot_feature_importance():
    fig = SIM.plot_feature_importance(_sweep_df(), target="prauc", clean=True)
    assert fig is not None


def test_plot_feature_importance_exclude():
    fig = SIM.plot_feature_importance(
        _sweep_df(), target="prauc", exclude="nr_plates", clean=True)
    assert fig is not None


def test_calculate_permutation_importance():
    fig = SIM.calculate_permutation_importance(
        _sweep_df(), target="prauc", n_repeats=3, clean=True)
    assert fig is not None


def test_plot_partial_dependences():
    fig = SIM.plot_partial_dependences(_sweep_df(), target="prauc", clean=True)
    assert fig is not None
