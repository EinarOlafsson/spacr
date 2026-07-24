"""Coverage-fill batch 2 for spacr.ml mid-size functions (real-data fixtures)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import ml as ML


# ---------------------------------------------------------------------------
# calculate_p_values
# ---------------------------------------------------------------------------

def test_calculate_p_values_normal():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 60)
    model = LinearRegression().fit(X, y)
    p = ML.calculate_p_values(X, y, model)
    assert len(p) == 3 and np.all(np.isfinite(p))


def test_calculate_p_values_underdetermined():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (3, 5))    # more features than obs → dof<=0 → NaN
    y = rng.normal(0, 1, 3)
    model = LinearRegression().fit(X, y)
    p = ML.calculate_p_values(X, y, model)
    assert len(p) == 5 and np.all(np.isnan(p))


# ---------------------------------------------------------------------------
# regression_model — several backends
# ---------------------------------------------------------------------------

def _Xy(n=80, p=3):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (n, p))
    y = X @ np.array([0.8, -0.5, 0.3]) + rng.normal(0, 0.2, n)
    return X, y


def test_regression_model_ols():
    X, y = _Xy()
    model = ML.regression_model(X, y, regression_type="ols")
    assert hasattr(model, "predict")


def test_regression_model_ols_cov_type():
    X, y = _Xy()
    model = ML.regression_model(X, y, regression_type="ols", cov_type="HC3")
    assert hasattr(model, "predict")


def test_regression_model_lasso_fixed_and_auto():
    X, y = _Xy()
    m_fixed = ML.regression_model(X, y, regression_type="lasso", alpha=0.1)
    assert hasattr(m_fixed, "coef_")
    m_auto = ML.regression_model(X, y, regression_type="lasso", alpha="auto")
    assert hasattr(m_auto, "coef_")


def test_regression_model_ridge():
    X, y = _Xy()
    model = ML.regression_model(X, y, regression_type="ridge", alpha=1.0)
    assert hasattr(model, "coef_")


def test_regression_model_glm():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (80, 2))
    y = np.abs(X @ np.array([0.5, 0.3]) + rng.normal(2, 0.3, 80))  # positive
    model = ML.regression_model(X, y, regression_type="glm")
    assert hasattr(model, "predict")


def test_regression_model_unsupported():
    X, y = _Xy()
    with pytest.raises(ValueError):
        ML.regression_model(X, y, regression_type="bogus")


# ---------------------------------------------------------------------------
# _calculate_similarity
# ---------------------------------------------------------------------------

def test_calculate_similarity():
    rng = np.random.default_rng(4)
    n = 40
    df = pd.DataFrame({
        "columnID": ["c1"] * 20 + ["c2"] * 20,
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })
    out = ML._calculate_similarity(
        df, ["f1", "f2", "f3"], "columnID", "c2", "c1")
    assert "similarity_to_pos_euclidean" in out.columns
    assert "similarity_to_neg_mahalanobis" in out.columns


def test_calculate_similarity_list_controls():
    rng = np.random.default_rng(5)
    n = 30
    df = pd.DataFrame({
        "columnID": (["c1", "c2", "c3"] * 10)[:n],
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
    })
    out = ML._calculate_similarity(
        df, ["f1", "f2"], "columnID", ["c2", "c3"], ["c1"])
    assert "similarity_to_pos_cosine" in out.columns


# ---------------------------------------------------------------------------
# check_and_clean_data
# ---------------------------------------------------------------------------

def test_check_and_clean_data():
    rng = np.random.default_rng(6)
    n = 40
    df = pd.DataFrame({
        "fraction": rng.uniform(0, 1, n),
        "prediction": rng.uniform(0, 1, n),
        "grna": rng.choice(["g1", "g2"], n),
        "gene": rng.choice(["geneA", "geneB"], n),
        "plateID": ["p1"] * n,
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2"], n),
        "prc": rng.choice(["p1_r1_c1", "p1_r2_c2"], n),
    })
    out = ML.check_and_clean_data(df, dependent_variable="prediction")
    assert "gene_fraction" in out.columns
    assert "gene" in out.columns


# ---------------------------------------------------------------------------
# process_scores
# ---------------------------------------------------------------------------

def _scores_df(n_wells=4, per_well=10):
    rng = np.random.default_rng(7)
    rows = []
    for w in range(n_wells):
        for o in range(per_well):
            rows.append({
                "prcfo": f"plate1_r{w+1}_c1_f1_o{o+1}",
                "pred": float(rng.uniform(0.1, 0.9)),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("agg", ["mean", "median", "quantile"])
def test_process_scores_agg_types(agg):
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cell_count=2, agg_type=agg)
    assert "cell_count" in out.columns and dv == "pred"
    assert (out["cell_count"] >= 2).all()


def test_process_scores_none_agg():
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cell_count=2, agg_type=None)
    assert "cell_count" in out.columns


def test_process_scores_poisson():
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cell_count=2,
        regression_type="poisson")
    assert "cell_count" in out.columns


def test_process_scores_invert_complement():
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cell_count=2,
        invert_dependent_variable=True)
    assert "cell_count" in out.columns


def test_process_scores_invert_reciprocal():
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cell_count=2,
        invert_dependent_variable=-1)
    assert "cell_count" in out.columns


def test_process_scores_transform():
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cell_count=2,
        agg_type="mean", transform="log")
    assert dv == "log_pred"


def test_process_scores_bad_agg():
    df = _scores_df()
    with pytest.raises(ValueError):
        ML.process_scores(df, "pred", plate="plate1",
                          min_cell_count=2, agg_type="bogus")


def test_process_scores_bad_invert():
    df = _scores_df()
    with pytest.raises(ValueError):
        ML.process_scores(df, "pred", plate="plate1",
                          min_cell_count=2, invert_dependent_variable=99)
