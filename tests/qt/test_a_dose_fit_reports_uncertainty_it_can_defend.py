"""Two guards in the dose-response fit, driven rather than assumed.

Instruction 288.

``ec50_fold_uncertainty`` declines when the lower bound is not positive,
marked ``# pragma: no cover - a log-space CI is positive``. That reason
is right about how ``fit_dose_response`` builds a result -- the bounds
are back-transformed from log space, so they are positive by
construction -- but ``DoseResponseResult`` is a public frozen dataclass
with no validation, and ``dataclasses.replace`` is all it takes to build
one with a non-positive bound. Without the guard that is a divide that
returns ``inf``, and the panel would print "within a factor of inf".

``_profile_bound``'s ``for ... else`` returns ``None`` when 60 doublings
never crossed the threshold, marked "60 doublings passes any finite
limit". That is not quite true: the walk covers ``step * 2**59``, so a
step small enough relative to the limit exhausts the loop. It is also
reachable outright with an infinite limit.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.dose_response import (
    DoseResponseSpec, _profile_bound, fit_dose_response,
    four_parameter_logistic,
)

DOSES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _fitted():
    rng = np.random.default_rng(1)
    dose = np.repeat(np.asarray(DOSES, dtype=float), 3)
    clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(1.0), -1.0)
    response = clean + rng.normal(0.0, 1.0, dose.size)
    return fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))


# ---------------------------------------------------------------------------
# ec50_fold_uncertainty
# ---------------------------------------------------------------------------

def test_a_real_fit_reports_a_fold_uncertainty():
    """The normal path, so the refusals below mean something."""
    result = _fitted()
    assert result.ec50_low is not None and result.ec50_low > 0
    fold = result.ec50_fold_uncertainty
    assert fold is not None and fold > 1.0


def test_a_zero_lower_bound_is_declined_rather_than_divided_by():
    """THE ARM. A hand-built result is all it takes.

    ``DoseResponseResult`` is public and frozen and validates nothing, so
    this is a shape a caller can produce -- and the alternative to the
    guard is a division that yields ``inf`` and a panel that reports
    "within a factor of inf".
    """
    broken = replace(_fitted(), ec50_low=0.0)
    assert broken.ec50_fold_uncertainty is None


def test_a_negative_lower_bound_is_declined_too():
    """The other side of ``<= 0``; a bare ``== 0`` would miss this and
    return the square root of a negative number."""
    broken = replace(_fitted(), ec50_low=-1.0)
    assert broken.ec50_fold_uncertainty is None


def test_without_the_guard_it_would_not_merely_be_wrong_but_infinite():
    """WHY the guard is not cosmetic, asserted on the arithmetic itself."""
    with np.errstate(divide="ignore"):
        assert np.isinf(np.sqrt(np.float64(2.0) / np.float64(0.0)))


def test_an_open_interval_is_still_declined_by_the_earlier_check():
    """So the test above is not the only thing keeping this None."""
    assert replace(_fitted(), ec50_low=None).ec50_fold_uncertainty is None
    assert replace(_fitted(), ec50_high=None).ec50_fold_uncertainty is None


# ---------------------------------------------------------------------------
# _profile_bound's exhausted walk
# ---------------------------------------------------------------------------

def _flat():
    log_dose = np.array([-1.0, 0.0, 1.0, 2.0])
    response = np.array([0.1, 0.4, 0.7, 0.9])
    return log_dose, response


def test_an_unbounded_walk_gives_up_after_sixty_doublings():
    """THE ARM. With an infinite limit the walk can never arrive."""
    log_dose, response = _flat()
    assert _profile_bound(log_dose, response, 0.0, 1.0,
                          target=1e18, step=0.1,
                          limit=float("inf"), upward=True) is None


def test_a_step_too_small_for_its_limit_also_gives_up():
    """The part the pragma's reason got wrong.

    "60 doublings passes any finite limit" is false: the walk reaches
    ``step * 2**59``, so a small enough step against a large enough
    finite limit exhausts the loop rather than arriving.
    """
    log_dose, response = _flat()
    assert 1e-30 * 2.0 ** 59 < 1e9, "the arithmetic this test rests on"
    assert _profile_bound(log_dose, response, 0.0, 1.0,
                          target=1e18, step=1e-30,
                          limit=1e9, upward=True) is None


def test_a_walk_that_reaches_its_limit_returns_none_by_the_other_route():
    """Same answer, different line -- and the distinction is the point.

    Both mean "this experiment does not bound the EC50 on that side", so
    a test asserting only the return value cannot tell the two apart.
    """
    log_dose, response = _flat()
    assert _profile_bound(log_dose, response, 0.0, 1.0,
                          target=1e18, step=0.1,
                          limit=3.0, upward=True) is None


def test_a_walk_that_crosses_the_threshold_returns_a_bound():
    """The path that must keep working, or every test above passes
    against a function that returns None unconditionally."""
    log_dose, response = _flat()
    found = _profile_bound(log_dose, response, 0.0, 1.0,
                           target=1e-9, step=0.1,
                           limit=6.0, upward=True)
    assert found is not None
