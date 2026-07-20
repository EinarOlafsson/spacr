"""
Tests for spacr.sim — simulation and classifier/ROC math.

Most functions here are pure numpy/pandas → predictable to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.sim as S


# ---------------------------------------------------------------------------
# Gini coefficients
# ---------------------------------------------------------------------------

def test_gini_perfect_equality_is_zero():
    x = np.ones(100, dtype=float)
    assert abs(S.gini(x)) < 1e-9


def test_gini_extreme_inequality_approaches_one():
    x = np.zeros(1000, dtype=float)
    x[0] = 1_000_000.0
    g = S.gini(x)
    assert 0.9 < g < 1.0, f"gini for concentration should be near 1, got {g}"


def test_gini_matches_reference_formula(rng):
    """gini(x) should agree with textbook formula: sum(|xi-xj|)/(2n^2 mean)."""
    x = rng.uniform(1, 100, size=50)
    g = S.gini(x)
    n = len(x)
    ref = np.sum(np.abs(x.reshape(-1, 1) - x.reshape(1, -1))) / (2 * n * n * x.mean())
    assert abs(g - ref) < 1e-6


def test_gini_coefficient_matches_reference(rng):
    """The alternative gini_coefficient() uses the outer-diff form directly."""
    x = rng.uniform(1, 50, size=30)
    g = S.gini_coefficient(x)
    n = len(x)
    ref = np.sum(np.abs(np.subtract.outer(x, x))) / (2 * n * n * x.mean())
    assert abs(g - ref) < 1e-9


# ---------------------------------------------------------------------------
# normalize_array
# ---------------------------------------------------------------------------

def test_normalize_array_maps_to_unit_interval(rng):
    a = rng.uniform(-10, 10, size=100)
    n = S.normalize_array(a)
    assert n.min() == pytest.approx(0.0)
    assert n.max() == pytest.approx(1.0)


def test_normalize_array_is_monotonic(rng):
    a = rng.uniform(0, 1, size=30)
    idx = np.argsort(a)
    n = S.normalize_array(a)
    assert np.all(np.diff(n[idx]) >= -1e-12), "normalize_array must preserve order"


# ---------------------------------------------------------------------------
# power law distribution
# ---------------------------------------------------------------------------

def test_power_law_distribution_sums_to_one():
    d = S.generate_power_law_distribution(50, coeff=1.2)
    assert d.sum() == pytest.approx(1.0)
    assert (d > 0).all()


def test_power_law_distribution_is_monotonically_decreasing():
    d = S.generate_power_law_distribution(30, coeff=1.5)
    assert np.all(np.diff(d) <= 0), "power-law weights should decrease with rank"


# ---------------------------------------------------------------------------
# gene / plate helpers
# ---------------------------------------------------------------------------

def test_generate_gene_list_length_and_uniqueness():
    genes = S.generate_gene_list(number_of_genes=20, number_of_all_genes=1000)
    assert len(genes) == 20
    assert len(set(genes)) == 20  # unique
    assert all(1 <= g <= 1000 for g in genes)


def test_generate_plate_map_shape():
    pm = S.generate_plate_map(nr_plates=2)
    # 2 plates × 16 rows × 24 cols (384-well format)
    assert len(pm) == 2 * 16 * 24
    for col in ("plate_id", "row_id", "column_id"):
        assert col in pm.columns
    # plate_id / row_id / column_id are stored as strings, not ints.
    assert set(pm["plate_id"].unique()) == {"1", "2"}
    assert set(pm["row_id"].unique()) == {str(i) for i in range(1, 17)}
    assert set(pm["column_id"].unique()) == {str(i) for i in range(1, 25)}


# ---------------------------------------------------------------------------
# classifier + ROC
# ---------------------------------------------------------------------------

def _make_binary_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"is_active": rng.integers(0, 2, size=n)})


def test_classifier_produces_score_column():
    df = _make_binary_df(200)
    out = S.classifier(
        positive_mean=0.7, positive_variance=0.02,
        negative_mean=0.3, negative_variance=0.02,
        classifier_accuracy=0.9,
        df=df,
    )
    assert "score" in out.columns
    assert ((out["score"] >= 0) & (out["score"] <= 1)).all()


def test_classifier_high_accuracy_yields_high_auc():
    """A near-perfect classifier should produce AUC > 0.9 on well-separated distributions."""
    np.random.seed(0)
    df = _make_binary_df(400, seed=1)
    scored = S.classifier(
        positive_mean=0.8, positive_variance=0.01,
        negative_mean=0.2, negative_variance=0.01,
        classifier_accuracy=0.98,
        df=df,
    )
    roc = S.compute_roc_auc(scored)
    assert roc["roc_auc"] > 0.9, f"expected AUC > 0.9, got {roc['roc_auc']}"


def test_classifier_low_accuracy_yields_low_auc():
    """A worst-case classifier (always wrong) inverts the score distribution → low AUC."""
    np.random.seed(0)
    df = _make_binary_df(400, seed=2)
    scored = S.classifier(
        positive_mean=0.8, positive_variance=0.01,
        negative_mean=0.2, negative_variance=0.01,
        classifier_accuracy=0.02,  # nearly always misclassifies
        df=df,
    )
    roc = S.compute_roc_auc(scored)
    assert roc["roc_auc"] < 0.2, f"expected inverted AUC < 0.2, got {roc['roc_auc']}"


def test_classifier_rejects_invalid_mean_variance():
    df = _make_binary_df(50)
    with pytest.raises(ValueError):
        S.classifier(1.5, 0.02, 0.3, 0.02, 0.9, df)  # mean > 1


# ---------------------------------------------------------------------------
# compute_roc_auc / compute_precision_recall
# ---------------------------------------------------------------------------

def _perfect_scores(n=100):
    """is_active=1 gets score 0.9, is_active=0 gets 0.1 → perfect separation."""
    return pd.DataFrame({
        "is_active": [1] * n + [0] * n,
        "score": [0.9] * n + [0.1] * n,
    })


def test_compute_roc_auc_perfect_is_one():
    roc = S.compute_roc_auc(_perfect_scores())
    assert roc["roc_auc"] == pytest.approx(1.0)
    for key in ("threshold", "tpr", "fpr", "roc_auc"):
        assert key in roc


def test_compute_precision_recall_perfect_has_pr_auc_one():
    pr = S.compute_precision_recall(_perfect_scores())
    assert pr["pr_auc"] == pytest.approx(1.0)
    for key in ("threshold", "precision", "recall", "f1_score", "pr_auc"):
        assert key in pr


def test_get_optimum_threshold_returns_scalar():
    pr = S.compute_precision_recall(_perfect_scores())
    t = S.get_optimum_threshold(pr)
    assert isinstance(t, float)
    assert 0.0 <= t <= 1.0
