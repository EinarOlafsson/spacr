"""A stored confidence interval whose lower bound is not a concentration.

The fold uncertainty is a ratio of the two ends of the interval, and it is
printed in the headline. A lower bound of zero -- what a rounded, exported or
hand-edited result can carry, where the fit itself cannot -- would make that
ratio infinite and put an infinite factor in front of the reader.

The profile walk is here too, at the one input where doubling the step sixty
times still does not reach the limit.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from spacr.qt.widgets import dose_response as dr


TRUE = (10.0, 90.0, 0.0, -1.0)          # bottom, top, log10(EC50), Hill


def a_bounded_fit() -> dr.DoseResponseResult:
    """A clean inhibition series whose midpoint the doses actually bracket."""
    doses = np.repeat([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], 3)
    response = dr.four_parameter_logistic(doses, *TRUE)
    response = response + np.random.default_rng(4).normal(0.0, 1.0,
                                                          response.size)
    return dr.fit_dose_response(doses, response)


def test_a_bounded_fit_states_its_interval_as_a_factor_either_way():
    result = a_bounded_fit()

    assert result.ec50_bounded is True
    assert result.ec50_low is not None and result.ec50_low > 0
    fold = result.ec50_fold_uncertainty
    assert fold == pytest.approx(
        float(np.sqrt(result.ec50_high / result.ec50_low)))
    assert fold > 1.0
    assert "a factor of" in result.headline()


@pytest.mark.parametrize("low", [0.0, -0.5])
def test_an_interval_whose_low_end_is_not_a_concentration_states_no_factor(
        low):
    """``sqrt(high / 0)`` is infinity and ``sqrt(high / -x)`` is not a number.

    Neither is a multiplicative uncertainty, so none is offered -- and the
    headline goes back to naming the two ends and nothing else.
    """
    result = dataclasses.replace(a_bounded_fit(), ec50_low=low)

    assert result.ec50_fold_uncertainty is None
    headline = result.headline()
    assert "a factor of" not in headline
    assert "CI" in headline


def test_an_open_interval_states_no_factor_either():
    """The same answer for the ordinary reason: one side never closed."""
    result = dataclasses.replace(a_bounded_fit(), ec50_high=None)

    assert result.ec50_fold_uncertainty is None


def test_a_profile_walk_whose_step_vanishes_into_the_centre_leaves_that_side_open():
    """A stalled walk is reported as an unbounded side, not as a bound.

    The walk steps outward as ``centre + step``, doubling ``step`` up to
    sixty times. Past about ``1e300`` the addition rounds straight back to
    ``centre``: the candidate never moves, so it never crosses the threshold
    and never reaches the limit either. Falling out of the loop with no
    outside point and bisecting anyway would take ``outside`` from whatever
    the previous call left behind.
    """
    doses = np.repeat([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], 3)
    response = dr.four_parameter_logistic(doses, *TRUE)
    log_dose = np.log10(doses)
    centre = 1e300
    limit = 1e301
    assert centre + 1.0 * 2.0 ** 59 == centre, "the step vanishes into it"
    target = dr._profile_sse(log_dose, response, centre, -1.0) + 1e6

    bound = dr._profile_bound(log_dose, response, centre, -1.0, target,
                              step=1.0, limit=limit, upward=True)

    assert bound is None
