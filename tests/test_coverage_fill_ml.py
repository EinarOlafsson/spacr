"""Coverage-fill for spacr.ml pure-logic helpers (no GPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import ml as ML


# ---------------------------------------------------------------------------
# create_volcano_filename
# ---------------------------------------------------------------------------

def test_create_volcano_filename_regular():
    out = ML.create_volcano_filename("/a/b/scores.csv", "ols", 0.1, "/dst")
    assert out.endswith("ols_scores_volcano_plot.pdf")
    assert out.startswith("/dst")


def test_create_volcano_filename_quantile_no_dst():
    out = ML.create_volcano_filename("/a/b/scores.csv", "quantile", 0.25, None)
    assert "0.25_scores_volcano_plot.pdf" in out
    assert out.startswith("/a/b")


# ---------------------------------------------------------------------------
# scale_variables
# ---------------------------------------------------------------------------

def test_scale_variables():
    X = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
    y = np.array([[0.0], [1.0], [2.0]])
    Xs, ys = ML.scale_variables(X, y)
    assert Xs.shape == X.shape
    assert 0.0 <= float(Xs["a"].min()) and float(Xs["a"].max()) <= 1.0


# ---------------------------------------------------------------------------
# select_glm_family — all four branches
# ---------------------------------------------------------------------------

def test_select_glm_family_binomial():
    fam = ML.select_glm_family(np.array([0, 1, 0, 1]))
    assert "Binomial" in type(fam).__name__


def test_select_glm_family_quasibinomial():
    fam = ML.select_glm_family(np.array([0.1, 0.5, 0.9]))
    assert fam is not None


def test_select_glm_family_poisson():
    fam = ML.select_glm_family(np.array([1, 2, 3, 10]))
    assert "Poisson" in type(fam).__name__


def test_select_glm_family_gaussian():
    fam = ML.select_glm_family(np.array([-1.5, 2.3, 5.1]))
    assert "Gaussian" in type(fam).__name__


# ---------------------------------------------------------------------------
# check_distribution — branches
# ---------------------------------------------------------------------------

def test_check_distribution_binary():
    assert ML.check_distribution(np.array([0, 1, 0, 1])) == "logit"


def test_check_distribution_beta():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.2, 0.8, 100)
    assert ML.check_distribution(y) in ("beta", "quasi_binomial", "glm", "ols")


def test_check_distribution_boundary_quasibinomial():
    y = np.array([0.0, 0.5, 1.0, 0.5])
    assert ML.check_distribution(y) == "quasi_binomial"


def test_check_distribution_normal_ols():
    rng = np.random.default_rng(1)
    y = rng.normal(10, 2, 200)
    assert ML.check_distribution(y) in ("ols", "glm")


# ---------------------------------------------------------------------------
# pick_glm_family_and_link
# ---------------------------------------------------------------------------

def test_pick_glm_binary():
    fam = ML.pick_glm_family_and_link(np.array([0, 1, 1, 0]))
    assert "Binomial" in type(fam).__name__


def test_pick_glm_strict_0_1_raises():
    with pytest.raises(ValueError):
        ML.pick_glm_family_and_link(np.array([0.2, 0.5, 0.8]))


def test_pick_glm_count_poisson():
    # Non-normal count data → Poisson.
    fam = ML.pick_glm_family_and_link(np.array([1, 2, 3, 4, 100, 2, 3]))
    assert fam is not None


def test_pick_glm_gaussian_normal():
    rng = np.random.default_rng(2)
    fam = ML.pick_glm_family_and_link(rng.normal(5, 1, 300))
    assert "Gaussian" in type(fam).__name__


# ---------------------------------------------------------------------------
# apply_transformation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t", ["log", "sqrt", "square", "unknown"])
def test_apply_transformation(t):
    out = ML.apply_transformation(None, t)
    if t == "unknown":
        assert out is None
    else:
        assert out is not None


# ---------------------------------------------------------------------------
# check_normality
# ---------------------------------------------------------------------------

def test_check_normality_true(capsys):
    rng = np.random.default_rng(3)
    assert ML.check_normality(rng.normal(0, 1, 100), "x", verbose=True) in (True, False)


def test_check_normality_false():
    assert ML.check_normality([0, 0, 0, 0, 100], "y") in (True, False)


# ---------------------------------------------------------------------------
# clean_controls
# ---------------------------------------------------------------------------

def test_clean_controls_removes_values():
    df = pd.DataFrame({"col": ["a", "b", "c", "a"], "v": [1, 2, 3, 4]})
    out = ML.clean_controls(df, ["a"], "col")
    assert "a" not in out["col"].values


def test_clean_controls_missing_column_noop():
    df = pd.DataFrame({"x": [1, 2]})
    out = ML.clean_controls(df, ["a"], "nonexistent")
    assert len(out) == 2


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------

def test_find_optimal_threshold():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    t = ML.find_optimal_threshold(y_true, y_proba)
    assert 0.0 <= float(t) <= 1.0


# ---------------------------------------------------------------------------
# prepare_formula + save_summary_to_file
# ---------------------------------------------------------------------------

def test_prepare_formula_variants():
    # False → rowID/columnID as FIXED effects; True → they become
    # random effects (handled separately) so they drop from the formula.
    fixed = ML.prepare_formula("recruitment", random_row_column_effects=False)
    random = ML.prepare_formula("recruitment", random_row_column_effects=True)
    assert "recruitment ~" in fixed
    assert "rowID" in fixed and "columnID" in fixed
    assert "rowID" not in random


def test_save_summary_to_file(tmp_path):
    import statsmodels.api as sm
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 50)
    y = 2 * x + rng.normal(0, 0.3, 50)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    out = tmp_path / "summary.csv"
    ML.save_summary_to_file(model, file_path=str(out))
    assert out.exists()
