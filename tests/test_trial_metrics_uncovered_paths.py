"""The paths a sweep row takes when the fit is not the one the metric expected.

Every statistic in :mod:`spacr.trial_metrics` is read off whatever a trial
happened to produce -- thirteen model families, a permutation test with no
model at all, and coefficient tables that are sometimes missing the column a
metric wants. The rule is the same everywhere: the statistic that cannot be
had is left out, the rest of the row is still written, and nothing raises.

These are the branches where that rule is exercised: a design matrix that does
not line up with its residuals, a fit that never reported its rank, a table
with no p-values, and the optional imports that a headless install can be
missing entirely.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from spacr.trial_metrics import (
    calibration,
    control_recovery,
    design_diagnostics,
    design_summary,
    guide_support_summary,
    hit_counts,
    qc_verdicts,
    residual_diagnostics,
    summarise_trial,
)


class _Model:
    """A stand-in fit whose attributes are set by the test that needs them."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def _residuals(n: int = 40, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


# ---------------------------------------------------------------------------
# control_recovery
# ---------------------------------------------------------------------------

def test_a_table_with_no_feature_column_recovers_no_controls():
    """Controls are found by name, and this table has no names to search.

    A coefficient table written by a family that labels its rows something
    other than ``feature`` still has p-values, so it is tempting to rank it
    anyway -- but a rank with no identifier attached cannot say which
    coefficient the positive control is, which is the only thing the column
    is for.
    """
    results = pd.DataFrame({"term": ["gene[239740]", "gene[233460]"],
                            "coefficient": [1.4, 0.05],
                            "p_value": [6.4e-08, 0.28]})

    assert control_recovery(results, {"positive_control": "239740"}) == {}


# ---------------------------------------------------------------------------
# residual_diagnostics
# ---------------------------------------------------------------------------

def test_normality_statistics_that_cannot_be_imported_cost_only_themselves(
        monkeypatch):
    """The residual trend survives statsmodels' stattools being unavailable.

    Durbin-Watson and Jarque-Bera come from an optional import; the trend
    slope is computed here with numpy alone and must still reach the row.
    """
    monkeypatch.setitem(sys.modules, "statsmodels.stats.stattools", None)

    out = residual_diagnostics(
        _Model(resid=_residuals(), fittedvalues=_residuals(seed=1)))

    assert "durbin_watson" not in out
    assert "jarque_bera_p" not in out
    assert out["residual_trend_slope"] == pytest.approx(
        np.polyfit(_residuals(seed=1), _residuals(), 1)[0])


def test_a_one_dimensional_design_costs_only_the_heteroscedasticity_tests():
    """Breusch-Pagan needs a matrix, and a single-regressor fit may store a
    vector. Losing its test must not lose the four residual statistics that
    were computed before it.
    """
    residuals = _residuals(20)
    fitted = _residuals(20, seed=1)

    out = residual_diagnostics(
        _Model(resid=residuals, fittedvalues=fitted,
               model=_Model(exog=_residuals(20, seed=2))))

    assert "breusch_pagan_p" not in out
    assert "white_p" not in out
    assert "durbin_watson" in out
    assert out["residual_trend_slope"] == pytest.approx(
        np.polyfit(fitted, residuals, 1)[0])


def test_a_design_that_does_not_match_the_residuals_is_not_tested_against_them():
    """A model whose exog has a different number of rows than its residuals
    cannot be asked about heteroscedasticity: the two are not the same fit.
    The residual behaviour that needs no design is still reported.
    """
    residuals = _residuals(20)
    fitted = _residuals(20, seed=1)

    out = residual_diagnostics(
        _Model(resid=residuals, fittedvalues=fitted,
               model=_Model(exog=np.random.default_rng(2).normal(size=(10, 3)))))

    assert "breusch_pagan_p" not in out
    assert "durbin_watson" in out
    assert out["residual_trend_slope"] == pytest.approx(
        np.polyfit(fitted, residuals, 1)[0])


def test_a_design_wider_than_thirty_columns_gets_breusch_pagan_but_not_white():
    """White's test squares every column pair, so it is skipped on a wide
    design rather than allowed to dominate the trial. The cheaper
    heteroscedasticity test still runs.
    """
    rng = np.random.default_rng(0)
    exog = np.column_stack([np.ones(600), rng.normal(size=(600, 30))])
    coefficients = np.linalg.lstsq(exog, rng.normal(size=600), rcond=None)[0]
    fitted = exog @ coefficients
    residuals = rng.normal(size=600)

    out = residual_diagnostics(
        _Model(resid=residuals, fittedvalues=fitted, model=_Model(exog=exog)))

    assert 0.0 <= out["breusch_pagan_p"] <= 1.0
    assert "white_p" not in out


def test_a_single_non_finite_residual_does_not_cost_the_trend_slope():
    """The slope is fitted over the residuals that exist.

    Dropping a real diagnostic because one well produced a NaN would hide a
    funnel in the other several hundred.
    """
    residuals = _residuals(20)
    fitted = _residuals(20, seed=1)
    with_gap = residuals.copy()
    with_gap[3] = np.nan

    out = residual_diagnostics(_Model(resid=with_gap, fittedvalues=fitted))

    finite = np.isfinite(with_gap)
    assert out["residual_trend_slope"] == pytest.approx(
        np.polyfit(fitted[finite], residuals[finite], 1)[0])


def test_residuals_and_fitted_values_of_different_lengths_do_not_raise():
    """Two vectors of different lengths are not a fit and its predictions.

    They cannot be regressed against each other, and the attempt must not cost
    the residual statistics that need no fitted values at all -- every caller
    guards this whole block with one try, so a raise here loses all of them.
    """
    residuals = np.concatenate([_residuals(12), [np.nan, np.inf]])

    out = residual_diagnostics(
        _Model(resid=residuals, fittedvalues=_residuals(12, seed=1)))

    assert "residual_trend_slope" not in out
    assert "durbin_watson" in out
    assert "jarque_bera_p" in out


def test_a_fit_with_no_fitted_values_reports_residual_behaviour_but_no_trend():
    """A trend needs something to trend against.

    Several families expose residuals without fitted values; the slope is the
    only statistic that requires both, so it is the only one omitted.
    """
    out = residual_diagnostics(_Model(resid=_residuals(20)))

    assert "residual_trend_slope" not in out
    assert "durbin_watson" in out
    assert "jarque_bera_p" in out


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def test_degenerate_expected_quantiles_cost_only_the_inflation_figure(
        monkeypatch):
    """Genomic inflation is a ratio, and a zero denominator is not a number.

    The other two calibration statistics are counted from the p-values
    directly and are unaffected, so the row keeps them.
    """
    p_values = np.concatenate([np.random.default_rng(0).uniform(size=100),
                               np.full(20, 1e-9)])
    monkeypatch.setattr(np, "median", lambda *args, **kwargs: 0.0)

    out = calibration(pd.DataFrame({"p_value": p_values}))

    assert "genomic_inflation" not in out
    assert out["n_tests"] == 120
    assert out["p_first_bin_excess"] > 0


# ---------------------------------------------------------------------------
# design_summary
# ---------------------------------------------------------------------------

def test_a_model_frame_without_well_or_guide_columns_reports_its_row_count():
    """The fitted row count is readable from any frame; the well and guide
    counts need the columns spaCR names them with, and a frame that lacks
    them gets no invented count.
    """
    out = design_summary({"model_data": pd.DataFrame({"value": [1.0, 2.0, 3.0]})})

    assert out == {"n_rows_fitted": 3}


def test_a_model_frame_with_only_wells_counts_wells_and_not_guides():
    """One of the two columns present is enough for one of the two counts."""
    out = design_summary(
        {"model_data": pd.DataFrame({"prc": ["p1_A_1", "p1_A_2", "p1_A_2"]})})

    assert out == {"n_rows_fitted": 3, "n_wells": 2}


# ---------------------------------------------------------------------------
# design_diagnostics
# ---------------------------------------------------------------------------

def test_a_fit_that_never_reported_its_rank_has_it_computed_from_the_design():
    """statsmodels keeps the rank on the model, but not every family here is
    statsmodels. The identifiability question is the one that decides whether
    the rest of the row means anything, so it is paid for rather than left
    unanswered.
    """
    exog = np.column_stack([np.ones(10), np.arange(10.0)])

    out = design_diagnostics(_Model(model=_Model(exog=exog)))

    assert out["design_rank"] == 2
    assert out["non_identifiable_directions"] == 0
    assert out["design_identifiable"] is True
    # No df_resid to read either, so it comes from rows minus rank.
    assert out["residual_degrees_of_freedom"] == 8


def test_a_rank_deficient_design_computed_here_is_reported_as_unidentifiable():
    """The computed rank is a real rank, not a column count: a duplicated
    predictor has to come back as a null direction.
    """
    column = np.arange(10.0)
    exog = np.column_stack([np.ones(10), column, column * 2.0])

    out = design_diagnostics(_Model(model=_Model(exog=exog)))

    assert out["design_rank"] == 2
    assert out["non_identifiable_directions"] == 1
    assert out["design_identifiable"] is False


def test_a_design_whose_rank_cannot_be_decomposed_yields_no_design_metrics():
    """A design holding a non-number has no rank at all.

    The decomposition fails rather than returning a wrong rank, and a wrong
    rank would be worse than none: it decides ``design_identifiable``, which
    is what says whether the coefficients are unique.
    """
    exog = np.array([[1.0, np.nan], [1.0, 2.0], [1.0, 3.0]])

    assert design_diagnostics(_Model(model=_Model(exog=exog))) == {}


def test_standard_errors_that_do_not_match_the_design_omit_the_vif():
    """VIF is read off the standard errors, one per parameter.

    A family reporting fewer standard errors than the design has parameters
    cannot be paired up with it, so the collinearity that does not need them
    is still reported and the VIF is not.
    """
    rng = np.random.default_rng(0)
    exog = np.column_stack([np.ones(30), rng.normal(size=(30, 3))])

    out = design_diagnostics(
        _Model(model=_Model(exog=exog, rank=4, k_constant=1), df_resid=26,
               bse=np.array([1.0, 2.0]), mse_resid=1.0))

    assert "max_vif" not in out
    assert "n_vif_above_10" not in out
    assert out["n_collinear_pairs"] == 0
    assert out["design_identifiable"] is True


def test_a_design_of_nothing_but_a_constant_has_no_vif_and_no_pairs():
    """An intercept has no variance, so it is not a predictor of anything.

    Its VIF would be zero and its correlation with itself one; both would be
    reported as if a real predictor had been measured.
    """
    out = design_diagnostics(
        _Model(model=_Model(exog=np.ones((12, 1)), rank=1, k_constant=1),
               df_resid=11, bse=np.array([0.5]), mse_resid=2.0))

    assert "max_vif" not in out
    assert "n_collinear_pairs" not in out
    assert "max_abs_predictor_correlation" not in out
    assert out["design_rank"] == 1
    assert out["wells_per_parameter"] == pytest.approx(12.0)


def test_a_single_varying_predictor_has_no_pair_to_correlate():
    """Pair statistics need two predictors; one predictor is not a pair, and
    a correlation of one column with itself is not a finding."""
    exog = np.column_stack([np.ones(12), np.arange(12.0)])

    out = design_diagnostics(
        _Model(model=_Model(exog=exog, rank=2), df_resid=10))

    assert "n_collinear_pairs" not in out
    assert "max_abs_predictor_correlation" not in out
    assert out["design_rank"] == 2


def test_a_design_too_wide_for_the_pairwise_scan_reports_no_pair_statistics():
    """The correlation matrix is quadratic in the predictor count.

    Above the cap it is the expense of the whole trial, so the pair counts are
    skipped and the identifiability fields -- which cost nothing, the fit
    already knows them -- are still reported.
    """
    from spacr.trial_metrics import _MAX_PREDICTORS_FOR_PAIRWISE

    width = _MAX_PREDICTORS_FOR_PAIRWISE + 1
    exog = np.random.default_rng(0).normal(size=(12, width))

    out = design_diagnostics(_Model(model=_Model(exog=exog, rank=12),
                                    df_resid=0))

    assert out["n_parameters"] == width
    assert "n_collinear_pairs" not in out
    assert "max_abs_predictor_correlation" not in out


def test_predictors_whose_correlation_is_not_a_number_yield_no_pair_counts():
    """A design whose scale overflows has no defined correlation.

    Counting the resulting NaN comparisons would report zero collinear pairs
    for a design nothing could be said about -- a clean bill of health from a
    test that did not run.
    """
    column = np.array([1e200, -1e200, 0.0, 5e199, -5e199, 1e199])
    exog = np.column_stack([column, column * 0.5 + 1e199])

    out = design_diagnostics(_Model(model=_Model(exog=exog, rank=2),
                                    df_resid=4))

    assert "n_collinear_pairs" not in out
    assert "max_abs_predictor_correlation" not in out
    assert out["n_parameters"] == 2
    assert out["design_rank"] == 2


# ---------------------------------------------------------------------------
# guide_support_summary
# ---------------------------------------------------------------------------

def _guide_results() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": ["gene_fraction:gene[239740]",
                    "fraction:grna[239740_1]",
                    "fraction:grna[239740_2]"],
        "coefficient": [1.4, 1.2, 1.5],
        "p_value": [6.4e-08, 1e-04, 2e-04],
    })


def test_gene_support_that_cannot_be_imported_leaves_the_rest_of_the_row(
        monkeypatch):
    """The gene-level counts are optional; the same table yields them when
    the module is importable and nothing at all when it is not.
    """
    assert guide_support_summary(_guide_results())["n_genes_tested"] == 1

    monkeypatch.setitem(sys.modules, "spacr.guide_concordance", None)

    assert guide_support_summary(_guide_results()) == {}


def test_a_results_object_that_is_not_a_table_yields_no_gene_counts():
    """Guide support is computed over a frame's columns, and a bundle that
    handed back a plain mapping has none to compute over."""
    assert guide_support_summary({"feature": ["gene_fraction:gene[239740]"]}) == {}


# ---------------------------------------------------------------------------
# hit_counts
# ---------------------------------------------------------------------------

def test_a_table_with_only_corrected_p_values_still_counts_them():
    """Some families report only the corrected column.

    The raw count is omitted rather than reported as zero, because a zero
    would read as "nothing passed" instead of "nothing was measured".
    """
    results = pd.DataFrame({"feature": ["gene[1]", "gene[2]"],
                            "q_value": [0.01, 0.2]})

    out = hit_counts({"results": results})

    assert out["n_results"] == 2
    assert out["n_below_alpha"] == 1
    assert "n_raw_below_alpha" not in out


# ---------------------------------------------------------------------------
# qc_verdicts
# ---------------------------------------------------------------------------

def test_a_row_whose_wells_per_parameter_is_not_a_number_keeps_its_other_verdict():
    """One unreadable design field costs the design panel and nothing else.

    The overall verdict is the worst AVAILABLE panel, so it falls back to the
    inference panel rather than disappearing.
    """
    row = {"n_wells": 96, "n_parameters": 12, "design_rank": 12,
           "wells_per_parameter": "unknown", "condition_number": 30.0,
           "non_identifiable_directions": 0,
           "n_results": 500, "genomic_inflation": 1.05}

    out = qc_verdicts(row)

    assert "qc_design" not in out
    assert out["qc_inference"]
    assert out["qc_verdict"] == out["qc_inference"]


# ---------------------------------------------------------------------------
# summarise_trial
# ---------------------------------------------------------------------------

def _trial_output() -> dict:
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    exog = sm.add_constant(rng.normal(size=(30, 2)))
    endog = exog @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=30)
    results = pd.DataFrame({
        "feature": ["Intercept"] + [f"gene_fraction:gene[{i}]"
                                    for i in range(29)],
        "coefficient": rng.normal(size=30),
        "p_value": np.linspace(1e-6, 0.99, 30),
        "q_value": np.linspace(1e-4, 0.99, 30),
    })
    model_data = pd.DataFrame({"prc": [f"p1_A_{i}" for i in range(30)],
                               "grna": [f"g{i % 5}" for i in range(30)]})
    return {"results": results,
            "model": sm.OLS(endog, exog).fit(),
            "model_data": model_data}


def test_a_model_that_raises_on_every_question_costs_only_the_model_metrics():
    """A trial that produced an unusable model object still contributes its
    counts and its control recovery.

    Each metric block is guarded on its own precisely so that one broken
    input is a few empty columns rather than a missing sweep row.
    """
    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError(f"this fit cannot answer {name!r}")

    results = pd.DataFrame({
        "feature": ["Intercept", "gene_fraction:gene[239740]",
                    "gene_fraction:gene[233460]"],
        "coefficient": [0.1, 1.4, 0.05],
        "p_value": [1e-30, 6.4e-08, 0.28],
    })

    row = summarise_trial({"results": results, "model": _Hostile()},
                          {"positive_control": "239740"})

    assert row["n_results"] == 3
    assert row["positive_control_rank"] == 1
    assert "design_rank" not in row
    assert "r_squared" not in row


def test_a_verdict_that_cannot_be_scored_leaves_every_measurement_standing(
        monkeypatch):
    """The verdicts are a judgement ON the row, so losing them must not lose
    the measurements the judgement was about.
    """
    output = _trial_output()
    settings = {"positive_control": "239740"}

    scored = summarise_trial(output, settings)
    assert scored["qc_design"]

    monkeypatch.setitem(sys.modules, "spacr.regression_diagnostics", None)

    row = summarise_trial(output, settings)

    assert "qc_design" not in row
    assert "qc_verdict" not in row
    assert row["n_results"] == scored["n_results"]
    assert row["design_rank"] == scored["design_rank"]
    assert row["n_wells"] == scored["n_wells"]
