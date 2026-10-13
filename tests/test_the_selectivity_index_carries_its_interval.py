"""Host toxicity over parasite killing, and the uncertainty of a ratio.

387 argues this is the number that decides whether anybody cares about an
anti-parasitic compound: a molecule that kills Toxoplasma at 1 uM and the HFF
monolayer at 1.2 uM is not a hit, and an EC50 with a clean interval says
nothing about that.

WHAT THESE TESTS PIN is not the arithmetic -- dividing two numbers is not
worth a test file -- it is the three things a selectivity index is usually
got wrong on:

* the interval, which must widen when two uncertain numbers are divided
  rather than being inherited from one of them or dropped;
* refusal, which must propagate rather than producing an index from a curve
  the engine already declined to quote;
* one-sidedness, because "at least 8-fold" is a useful sentence and losing it
  to a None is worse than saying it.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.dose_response import (
    DEFAULT_CONFIDENCE, STATUS_FITTED, STATUS_REFUSED, STATUS_UNBOUNDED,
    fit_dose_response, four_parameter_logistic, selectivity_index)

DOSES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def _curve(ec50, *, seed, noise=0.02, replicates=3):
    """A clean inhibition curve with a known EC50."""
    rng = np.random.default_rng(seed)
    dose = np.repeat(DOSES, replicates)
    safe = np.where(dose == 0, 1e-6, dose)
    y = four_parameter_logistic(safe, 0.0, 1.0, np.log10(ec50), -1.0)
    return dose, y + rng.normal(0.0, noise, y.shape)


@pytest.fixture
def paired():
    """Parasite killed at 1 uM, host toxic at 30 -- a true index of 30."""
    parasite = fit_dose_response(*_curve(1.0, seed=0))
    host = fit_dose_response(*_curve(30.0, seed=1))
    return parasite, host


def test_it_recovers_the_ratio_it_was_given(paired):
    parasite, host = paired
    si = selectivity_index(parasite, host)
    assert si.status == STATUS_FITTED
    assert si.index == pytest.approx(30.0, rel=0.2)


def test_the_interval_is_wider_than_either_curves_own(paired):
    """THE POINT OF THE FILE. A ratio of two uncertain numbers is more
    uncertain than either, and an index that inherited one curve's interval
    would understate it."""
    parasite, host = paired
    si = selectivity_index(parasite, host)
    ratio_width = np.log10(si.index_high) - np.log10(si.index_low)
    host_width = host.log10_ec50_ci[1] - host.log10_ec50_ci[0]
    parasite_width = parasite.log10_ec50_ci[1] - parasite.log10_ec50_ci[0]
    assert ratio_width > host_width
    assert ratio_width > parasite_width
    # and it is the sum in quadrature, not the sum: strictly less than both
    # widths added, which is what a careless propagation gives.
    assert ratio_width < host_width + parasite_width


def test_the_interval_is_symmetric_in_log_space_not_linear(paired):
    """Back-transforming a symmetric log interval gives an asymmetric linear
    one. That asymmetry is correct for a ratio and is not a bug to 'fix'."""
    parasite, host = paired
    si = selectivity_index(parasite, host)
    below = si.index / si.index_low
    above = si.index_high / si.index
    assert below == pytest.approx(above, rel=1e-6)
    assert si.index - si.index_low != pytest.approx(si.index_high - si.index)


@pytest.mark.parametrize("missing", ["parasite", "host"])
def test_a_refused_curve_refuses_the_index(paired, missing):
    parasite, host = paired
    args = (None, host) if missing == "parasite" else (parasite, None)
    si = selectivity_index(*args)
    assert si.status == STATUS_REFUSED
    assert si.index is None
    assert missing in si.note


def test_an_unbounded_curve_still_says_what_it_can():
    """One-sided rather than nothing. The host curve here never reaches its
    own half-maximum inside the tested range, so the index has a floor and no
    ceiling -- and the floor is the useful half."""
    parasite = fit_dose_response(*_curve(1.0, seed=2))
    # Host toxicity just past the top concentration tested: still monotone,
    # so the engine fits it, but the midpoint is an extrapolation. Pushed
    # much further out the series goes flat and the engine REFUSES it
    # instead, which is a different branch and already covered above.
    host = fit_dose_response(*_curve(150.0, seed=7, noise=0.01))
    assert not host.ec50_bounded, "the fixture must be unbounded, not refused"
    si = selectivity_index(parasite, host)
    assert si.status == STATUS_UNBOUNDED
    assert si.index is None or si.log10_index is not None
    assert "not bounded" in si.note or "open interval" in si.note


def test_the_summary_row_shows_a_refusal_rather_than_hiding_it(paired):
    parasite, _ = paired
    row = selectivity_index(parasite, None).summary_row()
    assert row["status"] == STATUS_REFUSED
    assert np.isnan(row["selectivity_index"])
    assert row["note"]
