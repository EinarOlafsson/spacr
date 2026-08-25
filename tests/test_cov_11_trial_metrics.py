"""A sweep row survives a model that answers nonsense.

Every function in :mod:`spacr.trial_metrics` runs over whatever a fit
happened to produce, across thirteen model families and a permutation test
that has no model at all. A diagnostic that raises would take down the whole
sweep row -- including the metrics that computed fine -- so each one omits
the statistic it could not get and returns the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.trial_metrics import (
    control_recovery,
    design_diagnostics,
    fit_quality,
    hit_counts,
    qc_verdicts,
    residual_diagnostics,
)


class _Model:
    """A stand-in fit whose attributes are set by the test that needs them."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class _Inner:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


# ---------------------------------------------------------------------------
# fit_quality
# ---------------------------------------------------------------------------

def test_a_statistic_that_is_not_a_number_is_dropped_not_reported():
    """A family reporting a string for R-squared contributes no R-squared.

    The alternative -- letting float() raise -- loses the AIC and BIC that
    the same model reported perfectly well.
    """
    model = _Model(rsquared="not a number", aic=12.5, resid=np.zeros(0))

    out = fit_quality(model)

    assert "r_squared" not in out
    assert out["aic"] == pytest.approx(12.5)


def test_residuals_that_are_not_numbers_leave_the_other_statistics_standing():
    """A residual vector of text still lets AIC through."""
    model = _Model(aic=3.0, resid=["a", "b", "c"])

    out = fit_quality(model)

    assert out["aic"] == pytest.approx(3.0)
    assert "residual_se" not in out
    assert "n_observations" not in out


# ---------------------------------------------------------------------------
# residual_diagnostics
# ---------------------------------------------------------------------------

def test_residuals_that_cannot_be_read_yield_no_diagnostics_and_no_error():
    """Nothing can be said about residuals that are not numeric."""
    model = _Model(resid=["a", "b"], fittedvalues=["c", "d"])

    assert residual_diagnostics(model) == {}


def test_a_trend_line_that_will_not_fit_is_omitted_not_fatal(monkeypatch):
    """The normality and heteroscedasticity checks survive a failed polyfit.

    The trend slope is one of five things this function reports; a singular
    least-squares problem must cost only that one.
    """
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=40)

    def refuse(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np, "polyfit", refuse)

    out = residual_diagnostics(
        _Model(resid=residuals, fittedvalues=rng.normal(size=40)))

    assert "residual_trend_slope" not in out
    assert "durbin_watson" in out


# ---------------------------------------------------------------------------
# control_recovery
# ---------------------------------------------------------------------------

def test_a_table_of_nothing_but_the_intercept_ranks_nobody():
    """Ranks are among real coefficients, and there are none here.

    Reporting rank 1 of 1 for a control that is not in the table would read
    as a perfectly recovered positive control.
    """
    results = pd.DataFrame({"feature": ["Intercept"], "coefficient": [0.3],
                            "p_value": [1e-9]})

    assert control_recovery(results, {"positive_control": "239740"}) == {}


def test_a_q_value_that_is_not_a_number_costs_only_the_q_value():
    """The control's rank and p-value still get reported.

    A q column holding something unconvertible is a broken table, but the
    rank is what says whether the assay worked, and it is still readable.
    """
    results = pd.DataFrame({
        "feature": ["Intercept", "gene[239740]", "gene[233460]"],
        "coefficient": [0.1, 1.4, 0.05],
        "p_value": [1e-30, 6.4e-08, 0.28],
        "q_value": [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
    })

    out = control_recovery(results, {"positive_control": "239740"})

    assert out["positive_control_found"] is True
    assert out["positive_control_rank"] == 1
    assert "positive_control_q" not in out


# ---------------------------------------------------------------------------
# design_diagnostics
# ---------------------------------------------------------------------------

def test_a_design_matrix_of_text_yields_no_design_metrics():
    """Rank and collinearity are undefined for something that is not numbers."""
    model = _Model(model=_Inner(exog=np.array([["a", "b"], ["c", "d"]])))

    assert design_diagnostics(model) == {}


@pytest.mark.parametrize("exog", [
    np.array([1.0, 2.0, 3.0]),          # one dimension: not a design matrix
    np.zeros((0, 3)),                   # no rows at all
])
def test_a_design_that_is_not_a_matrix_of_rows_yields_nothing(exog):
    """A design needs both dimensions before any of these questions apply."""
    assert design_diagnostics(_Model(model=_Inner(exog=exog))) == {}


def test_a_correlation_matrix_too_big_for_memory_costs_only_the_pair_counts(
        monkeypatch):
    """Rank and identifiability still come back when the pairs cannot.

    The pair statistics are quadratic in the predictor count; the rank is
    already known from the fit, and it is the field that decides whether the
    rest of the row means anything.
    """
    rng = np.random.default_rng(0)
    exog = np.column_stack([np.ones(30), rng.normal(size=(30, 3))])

    def refuse(*args, **kwargs):
        raise MemoryError("the correlation matrix does not fit")

    monkeypatch.setattr(np, "corrcoef", refuse)

    out = design_diagnostics(
        _Model(model=_Inner(exog=exog, rank=4, k_constant=1), df_resid=26))

    assert out["design_rank"] == 4
    assert out["design_identifiable"] is True
    assert "n_collinear_pairs" not in out


# ---------------------------------------------------------------------------
# hit_counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("output", [None, [], "results.csv", 7])
def test_counting_hits_in_something_that_is_not_a_result_bundle_is_empty(output):
    """A trial that returned nothing usable contributes no counts, not a crash."""
    assert hit_counts(output) == {}


# ---------------------------------------------------------------------------
# qc_verdicts
# ---------------------------------------------------------------------------

def _scorable_row():
    return {"n_wells": 96, "n_parameters": 12, "design_rank": 12,
            "wells_per_parameter": 8.0, "condition_number": 30.0,
            "non_identifiable_directions": 0,
            "n_results": 500, "genomic_inflation": 1.05}


def test_a_scorer_that_raises_costs_only_its_own_panel(monkeypatch):
    """The design verdict survives an inference scorer that blows up.

    The overall verdict is the worst AVAILABLE panel, so losing one panel
    must not lose the row -- it must lose that panel's opinion only.
    """
    from spacr import regression_diagnostics

    def refuse(_payload):
        raise RuntimeError("the inference scorer is unhappy")

    monkeypatch.setattr(regression_diagnostics, "score_inference", refuse)

    out = qc_verdicts(_scorable_row())

    assert "qc_inference" not in out
    assert out["qc_design"]
    assert out["qc_verdict"] == out["qc_design"]


def test_an_overall_verdict_that_cannot_be_combined_leaves_the_panels(
        monkeypatch):
    """Panel levels are still reported when the roll-up itself fails."""
    from spacr import regression_qc

    def refuse(_verdicts):
        raise RuntimeError("these verdicts cannot be compared")

    monkeypatch.setattr(regression_qc, "worst_verdict", refuse)

    out = qc_verdicts(_scorable_row())

    assert out["qc_design"]
    assert out["qc_inference"]
    assert "qc_verdict" not in out
