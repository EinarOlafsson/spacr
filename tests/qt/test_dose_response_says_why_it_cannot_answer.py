"""The dose-response engine on the inputs where the honest answer is "no".

Every case here is one where the engine could have produced a number and
must instead produce a verdict, a refusal or an open interval, and say why:

* a 5PL with a non-positive asymmetry;
* a hormesis check on too few points, on a flat series, on an exact fit,
  when the hormetic model does not converge or does not improve, when the
  hump does not pay for its parameter, and when the stimulation peaks at
  the lowest concentration tested;
* the 5PL and hormesis searches when the optimiser raises, warns, or
  returns parameters whose curve overflows;
* widening a covariance that is missing or of the wrong shape;
* selectivity when an interval is degenerate or an end of it is not a
  concentration;
* a synergy surface built from single-agent fits with no span;
* a checkerboard or plate table missing its columns, a plate spec with no
  control column or a non-finite Z' gate, and a plate with no controls.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import dose_response as dr

TRUE = (10.0, 90.0, 0.0, -1.0)


def _series(seed: int = 4):
    doses = np.repeat([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], 3)
    response = dr.four_parameter_logistic(doses, *TRUE)
    return doses, response + np.random.default_rng(seed).normal(
        0.0, 1.0, response.size)


@pytest.fixture(scope="module")
def bounded() -> dr.DoseResponseResult:
    return dr.fit_dose_response(*_series())


def _hormetic(start: float, stimulation: float):
    doses = start * 1.7 ** np.arange(10)
    dose = np.repeat(doses, 3)
    clean = dr.brain_cousens(dose, 0.0, 100.0, 0.0, -2.0, stimulation)
    return dose, clean + np.random.default_rng(1).normal(0.0, 2.0, dose.size)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("asymmetry", [0.0, -1.0, float("nan")])
def test_a_5pl_exponent_that_is_not_positive_is_refused(asymmetry):
    with pytest.raises(dr.DoseResponseError, match="strictly positive"):
        dr.five_parameter_logistic([1.0, 2.0], 0.0, 100.0, 0.0, -1.0,
                                   asymmetry)


def test_the_corrected_aic_has_no_meaning_without_residual_freedom():
    assert np.isnan(dr._aicc(1.0, 5, 4))
    assert np.isnan(dr._aicc(0.0, 30, 4))
    assert np.isfinite(dr._aicc(1.0, 30, 4))


def test_one_dose_or_a_flat_series_has_no_low_dose_excursion():
    assert dr._low_dose_excursion(np.array([1.0, 1.0]),
                                  np.array([3.0, 4.0]), -1.0) == (
        0.0, 0.0, None)
    assert dr._low_dose_excursion(np.array([1.0, 2.0, 3.0]),
                                  np.array([5.0, 5.0, 5.0]), -1.0) == (
        0.0, 0.0, None)


# ---------------------------------------------------------------------------
# hormesis verdicts
# ---------------------------------------------------------------------------

def test_a_verdict_of_no_hormesis_says_so_with_or_without_a_reason():
    assert dr.HormesisCheck(False).describe() == "no hormesis"
    assert dr.HormesisCheck(False, note="too few").describe() == (
        "no hormesis: too few")


def test_a_positive_verdict_without_two_ec50s_names_no_shift():
    text = dr.HormesisCheck(
        True, stimulation=12.0, max_stimulation_fraction=0.25,
        peak_dose=0.5, f_statistic=9.0, p_value=0.004, dof=20,
        delta_aic=6.0, ec50_monotone=None, ec50_hormetic=1.2).describe()
    assert text.startswith("hormesis: the response is stimulated by 25%")
    assert "monotone fit puts the EC50" not in text


def test_too_few_points_cannot_carry_a_hormesis_model():
    check = dr.hormesis([0.1, 1.0, 10.0], [90.0, 50.0, 10.0])
    assert check.is_hormetic is False
    assert check.n_obs == 3
    assert "cannot carry a five-parameter" in check.note


def test_a_flat_series_has_no_hump():
    dose = np.repeat([0.1, 0.3, 1.0, 3.0, 10.0], 3)
    check = dr.hormesis(dose, np.full(dose.size, 42.0))
    assert check.is_hormetic is False
    assert "every response is the same" in check.note


def test_the_verdict_on_too_few_observations_or_an_exact_fit():
    dose = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    popt = (0.0, 100.0, 0.0, -1.0)
    few = dr._hormesis_verdict(dose, dose, -1.0, 1.0, popt, 0.1, 0.05, 2.0)
    assert few.is_hormetic is False and "at least" in few.note
    dose, response = _series()
    exact = dr._hormesis_verdict(dose, response, -1.0, 0.0, TRUE,
                                 0.1, 0.05, 2.0)
    assert exact.is_hormetic is False
    assert "monotone fit is exact" in exact.note


def test_a_hormesis_model_that_does_not_converge_is_a_verdict(monkeypatch):
    monkeypatch.setattr(dr, "_fit_brain_cousens", lambda *a, **k: None)
    check = dr.hormesis(*_hormetic(0.8, 400.0))
    assert check.is_hormetic is False
    assert "did not converge" in check.note


def test_a_hormesis_model_that_does_not_improve_is_not_tested(monkeypatch):
    def _worse(dose, response, popt, start):
        fitted = dr.four_parameter_logistic(dose, *popt)
        sse = float(np.sum((response - fitted) ** 2))
        return sse * 2.0, np.array(list(popt) + [0.5])

    monkeypatch.setattr(dr, "_fit_brain_cousens", _worse)
    check = dr.hormesis(*_hormetic(0.8, 400.0))
    assert check.is_hormetic is False
    assert "did not improve the fit" in check.note
    assert check.stimulation == 0.5


def test_a_hump_that_does_not_pay_for_its_parameter_is_refused():
    check = dr.hormesis(*_hormetic(0.8, 400.0), min_delta_aic=1e6)
    assert check.is_hormetic is False
    assert "does not pay for its parameter" in check.note


def test_a_hump_at_the_lowest_dose_is_hormesis_that_is_not_located():
    dose, response = _hormetic(0.8, 400.0)
    check = dr.hormesis(dose, response)
    assert check.is_hormetic is True
    assert check.peak_dose == pytest.approx(float(dose.min()))
    assert "largest at the lowest concentration" in check.note


# ---------------------------------------------------------------------------
# the searches when the optimiser misbehaves
# ---------------------------------------------------------------------------

def _curve_fit_answers(monkeypatch, *answers):
    """Replace the optimiser with a queue of answers; an exception is raised,
    anything else returned, and a tuple ``("warn", value)`` warns first."""
    queue = list(answers)

    def _fake(*_args, **_kwargs):
        answer = queue.pop(0) if queue else queue_default
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, tuple) and isinstance(answer[0], str):
            warnings.warn("covariance could not be estimated",
                          RuntimeWarning, stacklevel=2)
            return answer[1]
        return answer

    queue_default = RuntimeError("no more answers")
    monkeypatch.setattr(dr, "curve_fit", _fake)


def test_the_5pl_search_reports_why_nothing_converged(monkeypatch):
    dose, response = _series()
    _curve_fit_answers(
        monkeypatch,
        RuntimeError("Optimal parameters not found"),
        ("warn", (np.array([np.nan, 1.0, 0.0, -1.0, 1.0]), None)),
        (np.array([1e308, 1e308, 0.0, -1.0, 1.0]), None))
    assert dr._fit_five(dose, response, TRUE) is None


def test_a_5pl_without_a_covariance_still_fits(monkeypatch):
    dose, response = _series()
    found = np.array([10.0, 80.0, 0.0, -1.0, 1.0])
    _curve_fit_answers(monkeypatch, ("warn", (found, None)), (found, None),
                       RuntimeError("stop"))
    sse, vector, matrix, notes = dr._fit_five(dose, response, TRUE)
    assert matrix is None
    assert vector.tolist() == [10.0, 90.0, 0.0, -1.0, 1.0]
    assert sse > 0
    assert any(note.startswith("RuntimeWarning") for note in notes)


def test_the_hormesis_search_skips_answers_it_cannot_use(monkeypatch):
    dose, response = _series()
    _curve_fit_answers(
        monkeypatch,
        ValueError("x0 is infeasible"),
        (np.array([np.inf, 1.0, 0.0, -1.0, 1.0]), None),
        (np.array([1e308, 1e308, 0.0, -1.0, 1.0]), None))
    assert dr._fit_brain_cousens(dose, response, TRUE, 3.0) is None


def test_a_covariance_is_widened_only_when_there_is_a_4x4_one():
    assert dr._pad_symmetric_covariance(None, 20) is None
    assert dr._pad_symmetric_covariance(np.eye(3), 20) is None
    padded = dr._pad_symmetric_covariance(np.eye(4), 5)
    assert padded.shape == (5, 5)
    assert padded[:4, :4].tolist() == np.eye(4).tolist(), (
        "no rescale without residual degrees of freedom to rescale to")
    assert padded[4].tolist() == [0.0] * 5


def test_a_5pl_profile_over_nothing_finite_is_infinite():
    log_dose = np.log10(np.array([0.1, 1.0, 10.0]))
    assert dr._profile_five_sse(log_dose, np.full(3, np.nan), 0.0,
                                -1.0) == float("inf")


# ---------------------------------------------------------------------------
# selectivity
# ---------------------------------------------------------------------------

def test_a_degenerate_interval_has_no_standard_error(bounded):
    assert dr._log10_standard_error(
        dataclasses.replace(bounded, confidence=0.0)) is None
    assert dr._log10_standard_error(
        dataclasses.replace(bounded, log10_ec50_ci=(0.2, 0.2))) is None
    assert dr._log10_standard_error(bounded) > 0


def test_an_interval_that_cannot_be_propagated_falls_back_to_its_ends(
        bounded):
    host = dataclasses.replace(bounded, confidence=0.0)
    index = dr.selectivity_index(bounded, host)
    assert index.status == dr.STATUS_UNBOUNDED
    assert index.index == pytest.approx(1.0)
    assert "interval arithmetic" in index.note
    assert index.index_low == pytest.approx(bounded.ec50_low
                                            / bounded.ec50_high)


def test_one_sided_ends_that_are_not_concentrations_stay_open(bounded):
    host = dataclasses.replace(bounded, ec50_bounded=False,
                               ec50_high=float("inf"), ec50_low=2.0)
    pathogen = dataclasses.replace(bounded, ec50_low=0.0, ec50_high=4.0)
    index = dr.selectivity_index(pathogen, host)
    assert index.status == dr.STATUS_UNBOUNDED
    assert index.index_low == pytest.approx(0.5)
    assert index.index_high is None, (
        "an infinite host bound over a zero parasite bound is no number")
    assert "host EC50 is not bounded" in index.note

    host = dataclasses.replace(host, ec50_high=8.0)
    index = dr.selectivity_index(pathogen, host)
    assert index.index_high is None, "nothing divides by a zero EC50"
    assert index.index_low == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# synergy, checkerboards and plates
# ---------------------------------------------------------------------------

def test_single_agent_fits_with_no_span_leave_the_surface_empty(bounded):
    flat = dataclasses.replace(bounded, bottom=50.0, top=50.0)
    dose_a = np.array([0.0, 0.0, 1.0, 1.0])
    dose_b = np.array([0.0, 1.0, 0.0, 1.0])
    surface = dr.bliss_surface(dose_a, dose_b, np.array([50.0] * 4),
                               fit_a=flat, fit_b=flat)
    assert np.all(surface.expected == 0.0)
    assert np.all(np.isnan(surface.observed))
    assert surface.n_cells == 0
    assert surface.summary() == {"model": dr.SYNERGY_BLISS, "n_cells": 0,
                                 "note": ""}


def test_a_checkerboard_missing_a_column_names_the_ones_it_has():
    frame = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    with pytest.raises(dr.DoseResponseError, match="'viability' is not"):
        dr.checkerboard_from_frame(frame, dose_a="a", dose_b="b",
                                   response="viability")


def test_a_plate_spec_needs_a_control_column_and_a_finite_gate():
    with pytest.raises(dr.DoseResponseError, match="control_column"):
        dr.PlateSpec(plate="plate", control=" ", positive=("pos",),
                     negative=("neg",))
    with pytest.raises(dr.DoseResponseError, match="finite number"):
        dr.PlateSpec(plate="plate", control="role", positive=("pos",),
                     negative=("neg",), min_zprime=float("nan"))


def test_a_plate_table_missing_a_column_is_refused():
    spec = dr.PlateSpec(plate="plate", control="role", positive=("pos",),
                        negative=("neg",))
    frame = pd.DataFrame({"plate": ["p1"], "role": ["pos"]})
    with pytest.raises(dr.DoseResponseError, match="'signal' is not"):
        dr.plate_reports(frame, spec, response="signal")


def test_a_plate_with_neither_control_says_both_are_missing():
    spec = dr.PlateSpec(plate="plate", control="role", positive=("pos",),
                        negative=("neg",))
    frame = pd.DataFrame({
        "plate": ["p1", "p1", "p1", "p2", "p2"],
        "role": ["pos", "neg", "sample", "sample", "sample"],
        "signal": [100.0, 0.0, 50.0, 40.0, 60.0],
    })
    reports = {report.plate: report
               for report in dr.plate_reports(frame, spec,
                                              response="signal")}
    assert reports["p2"].status == dr.STATUS_REFUSED
    assert "no positive and negative control" in reports["p2"].note
