"""A dose-response fit reports what the experiment supports and nothing more.

The module's whole job is refusing to hand back a number the data does not
carry, so the interesting paths are the ones where something is missing:

* a series with one concentration, or no response span at all, has no curve
  in it -- the helpers still have to answer rather than divide by a span of
  zero or index past the end of a one-element array;
* an optimiser that never converges, or converges onto residuals that are not
  finite, is a refusal with a message, not a silent EC50;
* a covariance matrix scipy could not estimate means every Wald interval is
  ``None``, and the fit says so instead of printing a bound computed from
  ``inf``;
* a fit that lands exactly on the data has no residual scatter, so there is no
  profile interval to walk and the result records why.

Each of those has an ordinary experimental cause -- a plateau that was never
reached, a dilution series that stopped short -- and each of them used to be
reported as a confident midpoint.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import dose_response as dr


TRUE = (10.0, 90.0, 0.0, -1.0)


def _series(replicates=3, noise=0.0, seed=0):
    doses = np.repeat([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], replicates)
    response = dr.four_parameter_logistic(doses, *TRUE)
    if noise:
        response = response + np.random.default_rng(seed).normal(
            0.0, noise, response.size)
    return doses, response


@pytest.fixture(scope="module")
def fitted():
    doses, response = _series(noise=1.0)
    return dr.fit_dose_response(doses, response)


# -- monotonicity -----------------------------------------------------------

def test_a_flat_series_is_trivially_monotone():
    """No response span means no excursion to measure against."""
    check = dr.monotonicity([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])

    assert check.is_monotone is True
    assert check.span == 0.0
    assert check.reversal_fraction == 0.0
    assert check.turning_points == ()


def test_a_single_concentration_is_trivially_monotone():
    """One concentration is one point; there is no trend to reverse."""
    check = dr.monotonicity([1.0], [5.0])

    assert check.is_monotone is True
    assert check.medians.size == 1


def test_an_empty_series_summarises_to_nothing():
    """No observations gives three empty arrays, not an index error."""
    distinct, medians, counts = dr._per_dose(np.zeros(0), np.zeros(0))

    assert distinct.size == medians.size == counts.size == 0


# -- what the result says ---------------------------------------------------

def test_a_result_reports_the_window_its_curve_moves_through(fitted):
    """``span`` is top minus bottom, the window the curve actually covers."""
    assert fitted.span == pytest.approx(fitted.top - fitted.bottom)
    assert fitted.span > 0


def test_a_bounded_fit_makes_no_one_sided_statement(fitted):
    """A bounded EC50 has no bound sentence, so a caller can print it blind."""
    assert fitted.ec50_bounded is True
    assert fitted.bound_statement() == ""


def test_a_dose_that_is_not_a_number_prints_as_not_available(fitted):
    """A non-finite bound is written "n/a" rather than "nan"."""
    unbounded = dataclasses.replace(
        fitted, ec50_bounded=False, bound_direction=dr.BOUND_ABOVE,
        dose_max=float("nan"))

    assert "n/a" in unbounded.bound_statement()


def test_an_unestimable_covariance_is_a_stated_caveat(fitted):
    """The reader is told there is no Wald interval, and why."""
    no_cov = dataclasses.replace(fitted, covariance_ok=False)

    assert any("covariance matrix could not be estimated" in c
               for c in no_cov.caveats())


def test_too_few_concentrations_for_a_lack_of_fit_test_is_a_caveat(fitted):
    """Replicates without enough distinct concentrations leaves no df."""
    thin = dataclasses.replace(fitted, lack_of_fit_p=None, n_doses=4)

    assert any("leaves it no" in c for c in thin.caveats())


def test_few_residual_degrees_of_freedom_is_a_caveat(fitted):
    """Wide intervals on 2 df are explained rather than left looking wrong."""
    thin = dataclasses.replace(fitted, dof=2)

    assert any("residual degree(s) of freedom" in c for c in thin.caveats())


def test_notes_are_printed_under_the_caveats(fitted):
    """A note recorded during the fit reaches the report the user reads."""
    noted = dataclasses.replace(fitted, notes=("the top plateau is assumed",))

    assert "  · the top plateau is assumed" in noted.report()


def test_a_set_iterates_its_group_fits():
    """``for fit in results`` walks the levels in the order they were seen."""
    frame = pd.DataFrame({
        "conc": np.tile(_series()[0], 2),
        "resp": np.tile(_series(noise=1.0)[1], 2),
        "gene": ["a"] * 24 + ["b"] * 24,
    })
    spec = dr.DoseResponseSpec(concentration="conc", response="resp",
                               group="gene")

    results = dr.fit_frame(frame, spec)

    assert [fit.group for fit in results] == ["a", "b"]
    assert len(list(iter(results))) == len(results)


# -- direction and starting points ------------------------------------------

def test_a_pinned_activation_direction_is_taken_as_given():
    """A caller who knows the direction is not second-guessed by a rank test."""
    doses, response = _series()

    assert dr._direction_sign(doses, response, dr.DIRECTION_ACTIVATION) == 1.0


def test_a_rankless_series_falls_back_to_its_end_medians():
    """A response with no rank correlation is decided by first vs last median.

    Spearman returns ``nan`` on a constant response; treating that as "no
    direction" would leave the optimiser with a slope sign of zero.
    """
    doses = np.array([1.0, 2.0, 3.0, 4.0])

    assert dr._direction_sign(doses, np.array([5.0] * 4),
                              dr.DIRECTION_AUTO) == 1.0
    assert dr._direction_sign(doses, np.array([9.0, 9.0, 1.0, 1.0]),
                              dr.DIRECTION_AUTO) == -1.0


def test_a_median_sitting_exactly_on_the_midpoint_is_the_crossing():
    """A dose whose median IS the half-maximal response is the midpoint."""
    guesses = dr._initial_guesses(np.array([1.0, 2.0, 3.0, 4.0]),
                                  np.array([5.0, 8.0, 9.0, 5.0]), -1.0)

    assert guesses[0][2] == pytest.approx(0.0)


def test_a_single_concentration_still_yields_a_starting_midpoint():
    """One concentration has no pair to cross between, and still gets a guess.

    The upstream guard refuses such a series; the helper must not raise on the
    way there.
    """
    guesses = dr._initial_guesses(np.array([1.0, 1.0, 1.0]),
                                  np.array([1.0, 2.0, 3.0]), -1.0)

    assert guesses[0][2] == pytest.approx(0.0)


# -- the optimiser -----------------------------------------------------------

def test_an_optimiser_that_never_converges_is_a_refusal(monkeypatch):
    """Every starting point failing is reported as a shape problem.

    Returning the last non-converged parameters would give a confident EC50
    for a curve that was never fitted.
    """
    def _explode(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr(dr, "curve_fit", _explode)
    doses, response = _series(noise=1.0)

    with pytest.raises(dr.DoseResponseError, match="did not converge"):
        dr.fit_dose_response(doses, response)


def test_the_optimisers_own_message_is_kept(monkeypatch):
    """The exception text from scipy is captured as data, not swallowed."""
    def _explode(*args, **kwargs):
        raise ValueError("x and p0 have incompatible shapes")

    monkeypatch.setattr(dr, "curve_fit", _explode)
    doses, response = _series(noise=1.0)

    popt, pcov, ok, messages = dr._fit_once(doses, response, (0, 1, 0, -1))

    assert popt is None and pcov is None and ok is False
    assert any("incompatible shapes" in m for m in messages)


def test_a_fit_whose_residuals_are_not_finite_is_rejected(monkeypatch):
    """Parameters that produce an infinite residual are not a better fit.

    Without the finiteness check ``inf`` compares as "not less than" the best
    so far only by luck, and a NaN comparison would accept it.
    """
    monkeypatch.setattr(
        dr, "curve_fit",
        lambda *a, **k: (np.array([0.0, 1e308, 0.0, 1.0]), np.eye(4)))
    doses, response = _series(noise=1.0)

    # The overflow IS the condition under test; numpy's warning about it is
    # not a second finding.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(dr.DoseResponseError, match="did not converge"):
            dr.fit_dose_response(doses, response)


def test_a_curve_fitted_upside_down_is_turned_the_right_way_up():
    """``top`` below ``bottom`` is swapped, and the Hill slope with it.

    The plateaus and the slope sign are one statement about direction; leaving
    them inconsistent makes the sign of the Hill slope meaningless.
    """
    flipped, no_cov = dr._canonicalise(np.array([90.0, 10.0, 0.0, -1.0]), None)

    assert list(flipped) == [10.0, 90.0, 0.0, 1.0]
    assert no_cov is None

    _params, covariance = dr._canonicalise(
        np.array([90.0, 10.0, 0.0, -1.0]), np.diag([1.0, 2.0, 3.0, 4.0]))

    assert list(np.diag(covariance)) == [2.0, 1.0, 3.0, 4.0]


# -- what the fit refuses to quote -------------------------------------------

def test_no_within_dose_scatter_leaves_the_lack_of_fit_test_untestable():
    """Identical replicates give zero pure error, so there is nothing to test.

    Dividing by it would report an infinite F statistic as a certainty that
    the model is wrong.
    """
    dose = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    response = dose.copy()

    assert dr._lack_of_fit(dose, response, 0.0) == (None, None, None)


def test_a_fit_that_lands_exactly_on_the_data_reports_no_interval(monkeypatch):
    """Zero residual scatter has no profile interval and no Wald one either.

    Both facts are recorded as notes: an EC50 printed with no interval and no
    explanation reads as an interval that was forgotten.
    """
    doses, response = _series(replicates=2)
    monkeypatch.setattr(
        dr, "curve_fit",
        lambda *a, **k: (np.array(TRUE), np.full((4, 4), np.inf)))

    result = dr.fit_dose_response(doses, response)

    assert result.sse == 0.0
    assert result.covariance_ok is False
    assert result.hill_ci == (None, None)
    assert result.top_ci == (None, None)
    assert any("positive residual sum of squares" in n for n in result.notes)
    assert any("covariance matrix was not estimable" in n
               for n in result.notes)


def test_a_series_that_never_reaches_the_midpoint_is_bounded_by_direction():
    """When the responses never cross the half-maximum, the curve's own
    direction says which end the experiment stopped at."""
    responses = np.array([1.0, 5.0])

    assert dr._bound_direction(0.0, 90.0, 10.0, 1.0, responses,
                               -2.0, 2.0, False, False) == dr.BOUND_ABOVE
    assert dr._bound_direction(0.0, 90.0, 10.0, -1.0, responses,
                               -2.0, 2.0, False, False) == dr.BOUND_BELOW


def test_a_midpoint_outside_the_tested_range_is_bounded_by_the_range():
    """A fitted midpoint past the highest or lowest dose is one-sided."""
    spanning = np.array([5.0, 95.0])

    assert dr._bound_direction(3.0, 90.0, 10.0, -1.0, spanning,
                               -2.0, 2.0, False, False) == dr.BOUND_ABOVE
    assert dr._bound_direction(-3.0, 90.0, 10.0, -1.0, spanning,
                               -2.0, 2.0, False, False) == dr.BOUND_BELOW


def test_a_grouping_column_with_no_values_is_refused():
    """A group column with no levels has nothing to fit one curve per."""
    frame = pd.DataFrame({"conc": pd.Series(dtype=float),
                          "resp": pd.Series(dtype=float),
                          "gene": pd.Series(dtype=str)})
    spec = dr.DoseResponseSpec(concentration="conc", response="resp",
                               group="gene")

    with pytest.raises(dr.DoseResponseError, match="has no values"):
        dr.fit_frame(frame, spec)
