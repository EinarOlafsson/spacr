"""Control-chart baselines and dose-response fits say what they rest on.

* :func:`control_chart` names a baseline it cannot find, says when the campaign
  is shorter than the baseline asked for, and leaves single-well plates out of
  S-bar with a note.
* :func:`fit_dose_response` refuses optimisers that never converge or converge
  onto non-finite residuals, records why an exact fit has no interval, makes no
  one-sided statement for a bounded EC50, and always carries the R² caveat.

One general test per unit, each walking a case table with ``subtests``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import dose_response as dr
from spacr.qt.widgets.control_chart import (
    ESTIMATOR_SUBGROUP_S, RULES_LIMITS_ONLY, ControlChartError, ControlChartSpec,
    control_chart,
)

TRUE = (10.0, 90.0, 0.0, -1.0)


def _series_frame(values):
    return pd.DataFrame({"plateID": [f"P{i + 1:02d}" for i in range(len(values))],
                         "signal": [float(v) for v in values],
                         "run_order": list(range(len(values)))})


def _spec(**extra):
    return ControlChartSpec(value="signal", plate="plateID", order="run_order",
                            rules=RULES_LIMITS_ONLY, **extra)


def test_a_control_chart_baseline_states_what_it_was_built_from(subtests):
    """Baseline refusals and notes for missing plates, short campaigns, lone wells."""
    with subtests.test(msg="a_baseline_named_by_absent_plates_is_refused"):
        with pytest.raises(ControlChartError, match="none of the 1 plate"):
            control_chart(_series_frame([10.0, 12.0] * 6),
                          _spec(baseline_plates=("NotHere",)))
    with subtests.test(msg="a_short_campaign_says_what_it_used"):
        result = control_chart(_series_frame([10.0, 12.0] * 5), _spec(baseline_n=20))
        assert any("asked for 20 plates" in note for note in result.notes)
        assert len(result.baseline) == 10
    with subtests.test(msg="a_single_well_plate_is_named_and_left_out"):
        rows = []
        for index in range(12):
            for offset in ([-1.0, 1.0] if index else [0.0]):
                rows.append({"plateID": f"P{index + 1:02d}", "run_order": index,
                             "signal": 100.0 + offset})
        result = control_chart(pd.DataFrame(rows),
                               _spec(estimator=ESTIMATOR_SUBGROUP_S, baseline_n=12))
        assert any("single control well" in note for note in result.notes)
        assert any("the mean over the 11 that do" in note for note in result.notes)
        assert math.isfinite(result.sigma) and result.sigma > 0


def _dose_series(replicates=3, noise=0.0, seed=0):
    doses = np.repeat([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], replicates)
    response = dr.four_parameter_logistic(doses, *TRUE)
    if noise:
        response = response + np.random.default_rng(seed).normal(0.0, noise, response.size)
    return doses, response


def _caveat_series(doses):
    dose = np.repeat(doses, 3)
    clean = dr.four_parameter_logistic(dose, 0.0, 100.0, 0.0, -1.0)
    return dose, clean + np.random.default_rng(1).normal(0.0, 1.0, dose.size)


def test_a_dose_response_fit_reports_only_what_the_data_carry(monkeypatch, subtests):
    """Refusals for failed optimisers; notes and caveats for what a fit lacks."""
    def _explode(*_a, **_k):
        raise RuntimeError("Optimal parameters not found")

    refusals = [
        ("an_optimiser_that_never_converges", _explode),
        ("residuals_that_are_not_finite",
         lambda *a, **k: (np.array([0.0, 1e308, 0.0, 1.0]), np.eye(4))),
    ]
    for case_id, fake in refusals:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                mp.setattr(dr, "curve_fit", fake)
                doses, response = _dose_series(noise=1.0)
                with np.errstate(over="ignore", invalid="ignore"):
                    with pytest.raises(dr.DoseResponseError, match="did not converge"):
                        dr.fit_dose_response(doses, response)
    with subtests.test(msg="an_exact_fit_has_no_interval_and_says_why"):
        with monkeypatch.context() as mp:
            mp.setattr(dr, "curve_fit",
                       lambda *a, **k: (np.array(TRUE), np.full((4, 4), np.inf)))
            result = dr.fit_dose_response(*_dose_series(replicates=2))
        assert result.sse == 0.0
        assert result.covariance_ok is False
        assert (result.hill_ci, result.top_ci) == ((None, None), (None, None))
        assert any("positive residual sum of squares" in n for n in result.notes)
        assert any("covariance matrix was not estimable" in n for n in result.notes)
    with subtests.test(msg="a_bounded_fit_makes_no_one_sided_statement"):
        result = dr.fit_dose_response(*_dose_series(noise=1.0))
        assert result.ec50_bounded is True
        assert result.bound_statement() == ""
    full = np.array([27.0, 9, 3, 1, 1 / 3, 1 / 9, 1 / 27, 1 / 81, 1 / 243, 1 / 729])
    for case_id, doses, starts in (
            ("a_clean_fit_still_warns_about_r_squared", full,
             ["R² is 0.9995", "Lack-of-fit F = 3.19"]),
            ("a_truncated_series_warns_about_that_too", full[4:],
             ["The point estimate is deliberately withheld", "R² is 0.9962"])):
        with subtests.test(msg=case_id):
            result = dr.fit_dose_response(*_caveat_series(doses), dr.DoseResponseSpec(unit="µM"))
            caveats = result.caveats()
            assert len(caveats) == len(starts)
            assert [c.startswith(s) for c, s in zip(caveats, starts)] == [True, True]
            assert "  ! " in result.report()
