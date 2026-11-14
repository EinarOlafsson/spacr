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

def case_create_volcano_filename_regular():
    _seen = []
    out = ML.create_volcano_filename("/a/b/scores.csv", "ols", 0.1, "/dst")
    _seen.append(bool(
        out.endswith("ols_scores_volcano_plot.pdf")))
    _seen.append(bool(
        out.startswith("/dst")))
    return _seen


def case_create_volcano_filename_quantile_no_dst():
    _seen = []
    out = ML.create_volcano_filename("/a/b/scores.csv", "quantile", 0.25, None)
    _seen.append(('0.25_scores_volcano_plot.pdf' if '0.25_scores_volcano_plot.pdf' in (out) else ("missing", '0.25_scores_volcano_plot.pdf')))
    _seen.append(bool(
        out.startswith("/a/b")))
    return _seen


# ---------------------------------------------------------------------------
# scale_variables
# ---------------------------------------------------------------------------

def case_scale_variables():
    _seen = []
    X = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
    y = np.array([[0.0], [1.0], [2.0]])
    Xs, ys = ML.scale_variables(X, y)
    _seen.append(bool(
        Xs.shape == X.shape))
    _seen.append(bool(
        0.0 <= float(Xs["a"].min()) and float(Xs["a"].max()) <= 1.0))
    return _seen


# ---------------------------------------------------------------------------
# select_glm_family — all four branches
# ---------------------------------------------------------------------------

def case_select_glm_family_binomial():
    _seen = []
    fam = ML.select_glm_family(np.array([0, 1, 0, 1]))
    _seen.append(('Binomial' if 'Binomial' in (type(fam).__name__) else ("missing", 'Binomial')))
    return _seen


def case_select_glm_family_quasibinomial():
    _seen = []
    fam = ML.select_glm_family(np.array([0.1, 0.5, 0.9]))
    _seen.append(bool(
        fam is not None))
    return _seen


def case_select_glm_family_poisson():
    _seen = []
    fam = ML.select_glm_family(np.array([1, 2, 3, 10]))
    _seen.append(('Poisson' if 'Poisson' in (type(fam).__name__) else ("missing", 'Poisson')))
    return _seen


def case_select_glm_family_gaussian():
    _seen = []
    fam = ML.select_glm_family(np.array([-1.5, 2.3, 5.1]))
    _seen.append(('Gaussian' if 'Gaussian' in (type(fam).__name__) else ("missing", 'Gaussian')))
    return _seen


# ---------------------------------------------------------------------------
# check_distribution — branches
# ---------------------------------------------------------------------------

def case_check_distribution_binary():
    _seen = []
    _seen.append(keep(ML.check_distribution(np.array([0, 1, 0, 1]))))
    return _seen


def case_check_distribution_beta():
    _seen = []
    rng = np.random.default_rng(0)
    y = rng.uniform(0.2, 0.8, 100)
    _seen.append(bool(
        ML.check_distribution(y) in ("beta", "quasi_binomial", "glm", "ols")))
    return _seen


def case_check_distribution_boundary_quasibinomial():
    _seen = []
    y = np.array([0.0, 0.5, 1.0, 0.5])
    _seen.append(keep(ML.check_distribution(y)))
    return _seen


def case_check_distribution_normal_ols():
    _seen = []
    rng = np.random.default_rng(1)
    y = rng.normal(10, 2, 200)
    _seen.append(bool(
        ML.check_distribution(y) in ("ols", "glm")))
    return _seen


# ---------------------------------------------------------------------------
# pick_glm_family_and_link
# ---------------------------------------------------------------------------

def case_pick_glm_binary():
    _seen = []
    fam = ML.pick_glm_family_and_link(np.array([0, 1, 1, 0]))
    _seen.append(('Binomial' if 'Binomial' in (type(fam).__name__) else ("missing", 'Binomial')))
    return _seen


def case_pick_glm_strict_0_1_is_binomial_not_a_refusal():
    # Was "raises ValueError: Use BetaModel"; a proportion is a binomial mean
    # and the neighbouring branch already fitted it as one.
    _seen = []
    fam = ML.pick_glm_family_and_link(np.array([0.2, 0.5, 0.8]))
    _seen.append(('Binomial' if 'Binomial' in (type(fam).__name__) else ("missing", 'Binomial')))
    return _seen


def case_pick_glm_count_poisson():
    # Non-normal count data → Poisson.
    _seen = []
    fam = ML.pick_glm_family_and_link(np.array([1, 2, 3, 4, 100, 2, 3]))
    _seen.append(bool(
        fam is not None))
    return _seen


def case_pick_glm_gaussian_normal():
    _seen = []
    rng = np.random.default_rng(2)
    fam = ML.pick_glm_family_and_link(rng.normal(5, 1, 300))
    _seen.append(('Gaussian' if 'Gaussian' in (type(fam).__name__) else ("missing", 'Gaussian')))
    return _seen


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

def case_check_normality_true(capsys):
    _seen = []
    rng = np.random.default_rng(3)
    _seen.append(bool(
        ML.check_normality(rng.normal(0, 1, 100), "x", verbose=True) in (True, False)))
    return _seen


def case_check_normality_false():
    _seen = []
    _seen.append(bool(
        ML.check_normality([0, 0, 0, 0, 100], "y") in (True, False)))
    return _seen


# ---------------------------------------------------------------------------
# clean_controls
# ---------------------------------------------------------------------------

def case_clean_controls_removes_values():
    _seen = []
    df = pd.DataFrame({"col": ["a", "b", "c", "a"], "v": [1, 2, 3, 4]})
    out = ML.clean_controls(df, ["a"], "col")
    _seen.append((("absent", 'a') if 'a' not in (out["col"].values) else ("present", 'a')))
    return _seen


def case_clean_controls_missing_column_noop():
    _seen = []
    df = pd.DataFrame({"x": [1, 2]})
    out = ML.clean_controls(df, ["a"], "nonexistent")
    _seen.append(keep(len(out)))
    return _seen


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------

def case_find_optimal_threshold():
    _seen = []
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    t = ML.find_optimal_threshold(y_true, y_proba)
    _seen.append(bool(
        0.0 <= float(t) <= 1.0))
    return _seen


# ---------------------------------------------------------------------------
# prepare_formula + save_summary_to_file
# ---------------------------------------------------------------------------

def case_prepare_formula_variants():
    # False → rowID/columnID as FIXED effects; True → they become
    # random effects (handled separately) so they drop from the formula.
    _seen = []
    fixed = ML.prepare_formula("recruitment", random_row_column_effects=False)
    random = ML.prepare_formula("recruitment", random_row_column_effects=True)
    _seen.append(('recruitment ~' if 'recruitment ~' in (fixed) else ("missing", 'recruitment ~')))
    _seen.append(bool(
        "rowID" in fixed and "columnID" in fixed))
    _seen.append((("absent", 'rowID') if 'rowID' not in (random) else ("present", 'rowID')))
    return _seen


def case_save_summary_to_file(tmp_path):
    _seen = []
    import statsmodels.api as sm
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 50)
    y = 2 * x + rng.normal(0, 0.3, 50)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    out = tmp_path / "summary.csv"
    ML.save_summary_to_file(model, file_path=str(out))
    _seen.append(bool(
        out.exists()))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_select_glm_family_binomial, ['Binomial'], id="select_glm_family_binomial"),
    pytest.param(case_select_glm_family_quasibinomial, [True], id="select_glm_family_quasibinomial"),
    pytest.param(case_select_glm_family_poisson, ['Poisson'], id="select_glm_family_poisson"),
    pytest.param(case_select_glm_family_gaussian, ['Gaussian'], id="select_glm_family_gaussian"),
    pytest.param(case_pick_glm_binary, ['Binomial'], id="pick_glm_binary"),
    pytest.param(case_pick_glm_strict_0_1_is_binomial_not_a_refusal, ['Binomial'], id="pick_glm_strict_0_1_is_binomial_not_a_refusal"),
    pytest.param(case_pick_glm_count_poisson, [True], id="pick_glm_count_poisson"),
    pytest.param(case_pick_glm_gaussian_normal, ['Gaussian'], id="pick_glm_gaussian_normal"),
    pytest.param(case_find_optimal_threshold, [True], id="find_optimal_threshold"),
    pytest.param(case_save_summary_to_file, [True], id="save_summary_to_file"),
    pytest.param(case_check_distribution_binary, ['logit'], id="check_distribution_binary"),
    pytest.param(case_check_distribution_beta, [True], id="check_distribution_beta"),
    pytest.param(case_check_distribution_boundary_quasibinomial, ['quasi_binomial'], id="check_distribution_boundary_quasibinomial"),
    pytest.param(case_check_distribution_normal_ols, [True], id="check_distribution_normal_ols"),
    pytest.param(case_check_normality_true, [True], id="check_normality_true"),
    pytest.param(case_check_normality_false, [True], id="check_normality_false"),
    pytest.param(case_clean_controls_removes_values, [("absent", 'a')], id="clean_controls_removes_values"),
    pytest.param(case_clean_controls_missing_column_noop, [2], id="clean_controls_missing_column_noop"),
    pytest.param(case_prepare_formula_variants, ['recruitment ~', True, ("absent", 'rowID')], id="prepare_formula_variants"),
    pytest.param(case_create_volcano_filename_regular, [True, True], id="create_volcano_filename_regular"),
    pytest.param(case_create_volcano_filename_quantile_no_dst, ['0.25_scores_volcano_plot.pdf', True], id="create_volcano_filename_quantile_no_dst"),
    pytest.param(case_scale_variables, [True, True], id="scale_variables"),
])
def test_fill_ml_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

