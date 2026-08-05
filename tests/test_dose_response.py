"""Dose–response: known EC50s planted, and the refusals pinned harder.

Every dataset here is generated from
:func:`spacr.qt.widgets.dose_response.four_parameter_logistic` itself, so the
truth is a number written in this file rather than an eyeballed curve. That
makes two very different kinds of assertion possible, and both are here:

**Recovery** — a clean 10-point, 3-fold dilution series with 1% noise, whose
EC50 the fitter has to find. It does, to about 0.1–3%. The tolerance asserted
is 10% (``EC50 / true`` within 1.1 either way), which is roughly two standard
errors on this design: tight enough that a mis-parameterised model, a fit run
in the wrong direction or a lost log10 misses it by a factor of 3 or more,
loose enough that it does not fail on a different BLAS.

The two recovery cases also assert that the true EC50 lies **inside** the
reported interval, and their seeds are chosen so that it does so comfortably
rather than by a hair. That is not a thumb on the scale: a nominal 95%
interval misses one seed in twenty *by construction*, so a single-seed
containment assertion is a coin flip unless the seed is fixed to a case where
it is not marginal. The interval's coverage — the property that actually
matters — is tested statistically over 200 seeds in
:func:`test_the_interval_covers_the_truth_about_ninety_five_percent_of_the_time`.

**Refusal** — the half of this module that is the reason it exists. The single
most important test in the file is
:func:`test_a_truncated_series_reports_a_one_sided_bound_and_no_ec50`: the
same generator, with the dilution series cut off below the midpoint, must come
back with ``ec50 is None``, ``ec50_bounded is False``, a one-sided sentence
naming the highest concentration tested, and no interval at all — not a
plausible number with error bars. A fitter that passes every recovery test and
fails that one is worse than no fitter, because it is confidently wrong on the
experiment people actually run.

The dose series, once, so every number below can be checked by hand::

    27, 9, 3, 1, 1/3, 1/9, 1/27, 1/81, 1/243, 1/729     (µM)

and the truncated one is its lower six, every one of them below the planted
EC50 of 1 µM.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.dose_response import (
    BOUND_ABOVE, BOUND_BELOW, CI_PROFILE, CI_WALD, DIRECTION_ACTIVATION,
    DIRECTION_INHIBITION, MIN_DOSES, PLATEAU_SLACK, SHALLOW_HILL, STATUS_FITTED,
    STATUS_REFUSED, STATUS_UNBOUNDED, STEEP_HILL, DoseResponseError,
    DoseResponseSet, DoseResponseSpec, fit_dose_response, fit_frame,
    four_parameter_logistic, monotonicity,
)

#: A 10-point, 3-fold dilution series from 27 µM down to 1.37 nM.
DOSES = 27.0 / 3.0 ** np.arange(10)

#: The lower six of it — every dose below the planted EC50 of 1 µM.
TRUNCATED = DOSES[DOSES < 1.0][:6]


def series(*, ec50=1.0, top=100.0, bottom=0.0, hill=-1.0, doses=DOSES,
           replicates=3, noise=1.0, seed=0):
    """A 4PL with a known EC50 and seeded Gaussian noise.

    :returns: ``(dose, response)``, one entry per observation, replicates
        interleaved the way a plate reader writes them.
    """
    rng = np.random.default_rng(seed)
    dose = np.repeat(np.asarray(doses, dtype=float), replicates)
    clean = four_parameter_logistic(dose, bottom, top, np.log10(ec50), hill)
    return dose, clean + rng.normal(0.0, noise, dose.size)


def bell(*, seed=1):
    """An activation curve that a toxic top dose kills — the classic refusal.

    Peak response at about 1 µM, back to baseline by 27 µM.
    """
    dose = np.repeat(DOSES, 3)
    rise = four_parameter_logistic(dose, 0.0, 100.0, np.log10(0.3), 1.5)
    survival = four_parameter_logistic(dose, 0.0, 1.0, np.log10(3.0), -3.0)
    rng = np.random.default_rng(seed)
    return dose, rise * survival + rng.normal(0.0, 2.0, dose.size)


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------

def test_the_model_is_half_maximal_at_the_ec50_and_flat_at_the_ends():
    """The definition the whole module rests on, checked directly."""
    midpoint = four_parameter_logistic(2.5, 10.0, 90.0, math.log10(2.5), -1.7)
    assert midpoint == pytest.approx(50.0)
    # x -> 0 and x -> inf are the two plateaus, and which is which is set by
    # the sign of the slope.
    assert four_parameter_logistic(0.0, 10.0, 90.0, 0.0, -1.7) == \
        pytest.approx(90.0)
    assert four_parameter_logistic(1e12, 10.0, 90.0, 0.0, -1.7) == \
        pytest.approx(10.0)
    assert four_parameter_logistic(0.0, 10.0, 90.0, 0.0, +1.7) == \
        pytest.approx(10.0)
    assert four_parameter_logistic(1e12, 10.0, 90.0, 0.0, +1.7) == \
        pytest.approx(90.0)


def test_swapping_the_plateaus_and_the_slope_sign_is_the_same_curve():
    """The exact symmetry that makes canonicalisation necessary."""
    x = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    a = four_parameter_logistic(x, 5.0, 95.0, 0.3, -1.4)
    b = four_parameter_logistic(x, 95.0, 5.0, 0.3, +1.4)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_an_inhibition_curve_recovers_its_planted_ec50():
    """EC50 1.0 µM planted; 0.9989 recovered, CI 0.956 – 1.045."""
    dose, response = series(ec50=1.0, hill=-1.0, seed=1)
    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))

    assert result.ec50_bounded is True
    assert result.status == STATUS_FITTED
    assert result.ec50 == pytest.approx(1.0, rel=0.10)
    assert result.ec50 == pytest.approx(0.998912, rel=1e-3)
    assert result.ec50_low < 1.0 < result.ec50_high
    # The direction is inferred, not assumed.
    assert result.hill < 0
    assert result.direction == DIRECTION_INHIBITION
    assert result.hill == pytest.approx(-1.0, abs=0.1)
    assert result.hill_ci[0] < -1.0 < result.hill_ci[1]
    # Canonical form: `top` is the larger plateau whichever end it sits at.
    assert result.top > result.bottom
    assert result.top == pytest.approx(100.0, abs=2.0)
    assert result.bottom == pytest.approx(0.0, abs=2.0)
    assert result.n_obs == 30 and result.n_doses == 10 and result.dof == 26


def test_an_activation_curve_recovers_its_planted_ec50():
    """EC50 2.5 µM planted on a rising curve; 2.4996 recovered."""
    dose, response = series(ec50=2.5, top=80.0, bottom=10.0, hill=1.4, seed=34)
    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))

    assert result.ec50_bounded is True
    assert result.ec50 == pytest.approx(2.5, rel=0.10)
    assert result.ec50 == pytest.approx(2.499625, rel=1e-3)
    assert result.ec50_low < 2.5 < result.ec50_high
    assert result.hill > 0
    assert result.direction == DIRECTION_ACTIVATION
    assert result.hill_ci[0] < 1.4 < result.hill_ci[1]
    assert result.top == pytest.approx(80.0, abs=2.0)
    assert result.bottom == pytest.approx(10.0, abs=2.0)


def test_the_reported_interval_is_multiplicative_and_positive():
    """A log-space interval back-transforms to a factor, never below zero."""
    dose, response = series(ec50=1.0, seed=1)
    result = fit_dose_response(dose, response)
    assert result.ec50_low > 0.0
    fold = result.ec50_fold_uncertainty
    assert fold is not None and fold > 1.0
    # sqrt(high/low) is the factor, and the estimate sits inside it.
    assert result.ec50_low <= result.ec50 <= result.ec50_high
    assert result.ec50_high / result.ec50_low == pytest.approx(fold ** 2)


def test_the_wald_interval_is_available_and_close_on_a_well_determined_curve():
    """Both intervals agree when the surface really is quadratic — which is
    exactly the case in which the Wald interval is not the problem."""
    dose, response = series(ec50=1.0, seed=1)
    profile = fit_dose_response(dose, response,
                                DoseResponseSpec(ci_method=CI_PROFILE))
    wald = fit_dose_response(dose, response,
                             DoseResponseSpec(ci_method=CI_WALD))
    assert wald.ci_method == CI_WALD
    assert wald.ec50_low == pytest.approx(profile.ec50_low, rel=0.02)
    assert wald.ec50_high == pytest.approx(profile.ec50_high, rel=0.02)
    # The Wald interval is symmetric in log space by construction; the
    # profile's is not obliged to be.
    log_wald = np.log10([wald.ec50_low, wald.ec50, wald.ec50_high])
    assert (log_wald[1] - log_wald[0]) == pytest.approx(
        log_wald[2] - log_wald[1], abs=1e-6)


# ---------------------------------------------------------------------------
# The incomplete curve — the test this module exists for
# ---------------------------------------------------------------------------

def test_a_truncated_series_reports_a_one_sided_bound_and_no_ec50():
    """Every dose below the midpoint: no EC50, no interval, a ``>`` sentence.

    The generator is identical to the recovery case — same EC50, same
    plateaus, same slope, same noise — and only the dilution series is cut off
    at 0.333 µM, three dilutions short of the planted 1 µM midpoint. The lower
    plateau is therefore never reached: the observed responses run 74 to 101
    on a curve whose fitted half-maximum is around 50.
    """
    dose, response = series(ec50=1.0, hill=-1.0, doses=TRUNCATED, seed=20260804)
    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))

    # 1. The point estimate is not reachable under a name that reads as one.
    assert result.ec50 is None
    assert result.ec50_bounded is False
    assert result.status == STATUS_UNBOUNDED
    # 2. The number still exists, under a name that says what it is.
    assert result.ec50_unconstrained > 0.0
    assert np.isfinite(result.ec50_unconstrained)
    # 3. No interval at all — not a wide one.
    assert result.ec50_low is None and result.ec50_high is None
    assert result.log10_ec50_ci == (None, None)
    assert result.ec50_fold_uncertainty is None
    # 4. A one-sided statement, naming the concentration it is one side of.
    assert result.bound_direction == BOUND_ABOVE
    statement = result.bound_statement()
    assert statement.startswith("EC50 > ")
    assert "0.333" in statement
    assert "highest concentration tested" in statement
    assert result.dose_max == pytest.approx(TRUNCATED.max())

    # 5. And report() says all of it in words, not just in fields.
    text = result.report()
    assert "does not bound the EC50" in text
    assert "ec50_bounded = False" in text
    assert "EC50 > " in text
    assert "highest concentration tested" in text
    assert "must not be quoted as an EC50" in text.replace("\n", " ")

    # 6. The summary row a table renders carries the refusal, not a blank.
    row = result.summary_row()
    assert row["status"] == STATUS_UNBOUNDED
    assert math.isnan(row["ec50"])
    assert math.isnan(row["ec50_low"]) and math.isnan(row["ec50_high"])
    assert row["ec50_unconstrained"] == pytest.approx(
        result.ec50_unconstrained)
    assert "EC50 > " in row["note"]

    # 7. The exported parameter table agrees with the object.
    frame = result.parameter_frame()
    ec50_row = frame[frame["parameter"] == "ec50"].iloc[0]
    assert math.isnan(ec50_row["estimate"])
    assert math.isnan(ec50_row["ci_low"]) and math.isnan(ec50_row["ci_high"])


def test_a_truncated_series_is_unbounded_below_when_it_starts_past_the_midpoint():
    """The mirror case: every dose above the EC50, so it is a ``<`` bound."""
    doses = np.array([3.0, 9.0, 27.0, 81.0, 243.0])
    dose, response = series(ec50=1.0, hill=-1.0, doses=doses, seed=5)
    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))
    assert result.ec50 is None
    assert result.ec50_bounded is False
    assert result.bound_direction == BOUND_BELOW
    statement = result.bound_statement()
    assert statement.startswith("EC50 < ")
    assert "lowest concentration tested" in statement


def test_the_wald_interval_is_withheld_too_when_the_curve_is_incomplete():
    """The Wald machinery would happily have produced one. It is not used.

    This is the specific failure mode the profile interval was chosen over
    the Wald one to expose, so asking for the Wald interval must not become a
    way around the refusal.
    """
    dose, response = series(ec50=1.0, doses=TRUNCATED, seed=20260804)
    result = fit_dose_response(dose, response,
                               DoseResponseSpec(ci_method=CI_WALD))
    assert result.ec50_bounded is False
    assert result.ec50 is None
    assert result.ec50_low is None and result.ec50_high is None


def test_an_astronomically_wide_wald_bound_is_reported_as_open():
    """``L ± t·SE`` has no stopping rule, and 10**400 is not a concentration.

    A forced fit to a bell shape leaves the midpoint unidentified, and the
    Wald formula answers with an upper bound hundreds of decades past the
    highest dose tested — which used to raise ``OverflowError`` on the way
    back out of log space, and which is in any case a statement about the
    formula rather than the experiment. Both methods now stop at the same
    reach: past a factor of 10**PROFILE_REACH beyond the tested range, the
    side is open.
    """
    dose, response = bell()
    result = fit_dose_response(dose, response, DoseResponseSpec(
        ci_method=CI_WALD, allow_non_monotone=True))
    for bound in (result.ec50_low, result.ec50_high):
        assert bound is None or np.isfinite(bound)
    for bound in result.log10_ec50_ci:
        if bound is not None:
            assert (np.log10(result.dose_min) - 10.0 < bound
                    < np.log10(result.dose_max) + 10.0)


def test_an_extrapolated_plateau_alone_is_enough_to_withhold_the_ec50():
    """The plateau rule is not redundant with the in-range rule.

    Here the fitted midpoint lands inside the tested concentrations, and the
    curve is still refused, because a plateau more than
    :data:`PLATEAU_SLACK` of the observed span outside the observed responses
    is a claim about data nobody collected.
    """
    dose, response = series(ec50=1.0, doses=TRUNCATED, seed=20260804)
    result = fit_dose_response(dose, response)
    assert result.ec50_bounded is False
    span = float(response.max() - response.min())
    assert result.bottom < response.min() - PLATEAU_SLACK * span


# ---------------------------------------------------------------------------
# Non-monotone data
# ---------------------------------------------------------------------------

def test_a_bell_shaped_series_is_refused_rather_than_fitted():
    """Cytotoxicity at the top dose. The engine raises; it never returns."""
    dose, response = bell()
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))
    message = str(caught.value)
    assert "not monotone" in message
    assert "turns around at" in message
    assert "cytotoxicity" in message
    assert "biphasic" in message
    assert "allow_non_monotone" in message
    # The concentration where it turns is named, not just alluded to.
    assert "1" in message


def test_the_monotonicity_check_is_readable_on_its_own():
    """It is a public function, because the screen wants to show it."""
    dose, response = bell()
    check = monotonicity(dose, response)
    assert check.is_monotone is False
    assert check.reversal_fraction > 0.5
    assert check.sign_changes >= 1
    assert check.turning_points
    assert "not monotone" in check.describe()

    clean_dose, clean_response = series(ec50=1.0, seed=1)
    ok = monotonicity(clean_dose, clean_response)
    assert ok.is_monotone is True
    assert ok.reversal_fraction < 0.05
    assert ok.spearman_rho < -0.9
    assert "monotone" in ok.describe()


def test_a_non_monotone_fit_can_be_forced_and_says_so_afterwards():
    """The escape hatch exists, and using it does not hide the warning."""
    dose, response = bell()
    result = fit_dose_response(
        dose, response, DoseResponseSpec(allow_non_monotone=True))
    assert result.check.is_monotone is False
    assert any("monotonicity check, which failed" in c
               for c in result.caveats())


# ---------------------------------------------------------------------------
# Interval coverage
# ---------------------------------------------------------------------------

def test_the_interval_covers_the_truth_about_ninety_five_percent_of_the_time():
    """200 seeded noisy replicate designs; the nominal 95% interval must be
    roughly 95%.

    This is the loose-but-real check that the interval is an interval and not
    a decoration. It is deliberately not a tight one — 200 draws puts a ±3
    point binomial standard error on the estimate — but it separates a
    correct interval from every wrong one that matters: a normal quantile
    instead of ``t`` undercovers, a linear-space interval undercovers badly
    and asymmetrically, and a mis-scaled covariance is off by a factor.

    The design is 8 concentrations × 2 replicates with noise at 5% of the
    response span, which leaves 12 residual degrees of freedom — few enough
    that the ``t`` versus ``z`` distinction is doing visible work.
    """
    truth = 1.0
    doses = 30.0 / 3.0 ** np.arange(8)
    covered = withheld = 0
    trials = 200
    for seed in range(trials):
        dose, response = series(ec50=truth, doses=doses, replicates=2,
                                noise=5.0, seed=seed)
        result = fit_dose_response(dose, response)
        if not result.ec50_bounded:
            withheld += 1
            continue
        if result.ec50_low <= truth <= result.ec50_high:
            covered += 1
    bounded = trials - withheld
    # A design that spans the midpoint by two decades either way should
    # almost never be withheld; if it often were, the coverage below would be
    # measured on a self-selected subset and would mean nothing.
    assert withheld <= trials * 0.05, f"{withheld} of {trials} withheld"
    rate = covered / bounded
    assert 0.85 <= rate <= 1.0, f"coverage {rate:.3f} over {bounded} fits"


# ---------------------------------------------------------------------------
# Fit quality
# ---------------------------------------------------------------------------

def test_lack_of_fit_is_significant_for_data_that_is_not_a_sigmoid():
    """A response that ramps linearly in log dose between two plateaus.

    It is monotone, it has both plateaus, and R² comes out at 0.995 — which
    is the entire point. The lack-of-fit test against pure error returns
    p < 1e-10, because the replicates say the noise is small and the 4PL
    misses the corners of the ramp by far more than that.
    """
    doses = 30.0 / 3.0 ** np.arange(9)
    dose = np.repeat(doses, 4)
    rng = np.random.default_rng(11)
    ramp = np.clip((math.log10(10.0) - np.log10(dose))
                   / (math.log10(10.0) - math.log10(0.1)), 0.0, 1.0)
    response = 100.0 * ramp + rng.normal(0.0, 1.0, dose.size)

    result = fit_dose_response(dose, response)
    assert result.lack_of_fit_p is not None
    assert result.lack_of_fit_p < 1e-10
    assert result.lack_of_fit_df == (5, 27)
    # And the point of printing it next to R².
    assert result.r_squared > 0.98
    assert any("lack-of-fit" in c.lower() and "wrong shape" in c
               for c in result.caveats())


def test_lack_of_fit_is_not_significant_for_genuine_four_pl_data():
    """The same statistic on data that really is a 4PL: nothing to report.

    A different seed from the recovery tests, chosen because the test is
    *approximate* for a nonlinear model — over 120 seeded genuine-4PL datasets
    of this shape it rejects at nominal 5% about 11% of the time, which the
    engine's docstring states. Pinning a single seed that happens to reject
    would be pinning that approximation error, not the statistic.
    """
    dose, response = series(ec50=1.0, seed=0)
    result = fit_dose_response(dose, response)
    assert result.lack_of_fit_p is not None
    assert result.lack_of_fit_p > 0.05
    assert result.lack_of_fit_df == (6, 20)
    assert not any("wrong shape" in c for c in result.caveats())
    assert "consistent with a 4PL" in result.report()


def test_the_lack_of_fit_test_is_roughly_calibrated_on_genuine_data():
    """It over-rejects, but not wildly — and that is a documented fact.

    Forty seeded datasets drawn from a real 4PL. If the test were badly
    broken (a wrong degrees-of-freedom charge, pure error computed against
    the fitted values rather than the group means) this would reject almost
    every time, which is the failure worth catching.
    """
    rejections = 0
    for seed in range(40):
        dose, response = series(ec50=1.0, seed=seed)
        result = fit_dose_response(dose, response)
        assert result.lack_of_fit_p is not None
        rejections += result.lack_of_fit_p < 0.05
    assert rejections <= 8, f"{rejections}/40 false rejections"


def test_r_squared_is_reported_with_the_warning_attached():
    """It is never printed on its own."""
    dose, response = series(ec50=1.0, seed=1)
    result = fit_dose_response(dose, response)
    assert result.r_squared > 0.99
    warning = [c for c in result.caveats() if "R²" in c]
    assert warning and "nearly useless" in warning[0]
    assert "residual standard error" in warning[0]


def test_without_replicates_lack_of_fit_says_untested_rather_than_passed():
    """"Cannot be tested" and "passed" are different states."""
    dose, response = series(ec50=1.0, replicates=1, noise=1.0, seed=2)
    result = fit_dose_response(dose, response)
    assert result.has_replicates is False
    assert result.lack_of_fit_p is None
    assert result.lack_of_fit_df is None
    assert any("no pure-error estimate" in c for c in result.caveats())
    assert "not testable" in result.report()


def test_an_absurd_hill_slope_is_flagged():
    """Steep and shallow both get said out loud.

    The steep case needs a 1.5-fold series: on the 3-fold series above, a
    slope of -12 turns over entirely inside one dilution step, so the data
    cannot resolve it and the fit comes back at half of it — which is a fact
    about the design, not about the fitter.
    """
    fine = 27.0 / 1.5 ** np.arange(14)
    dose, steep = series(ec50=1.0, hill=-12.0, doses=fine, noise=0.3, seed=3)
    result = fit_dose_response(dose, steep)
    assert abs(result.hill) >= STEEP_HILL
    assert result.is_steep() is True
    assert any("all-or-nothing step" in c for c in result.caveats())

    dose, shallow = series(ec50=1.0, hill=-0.1, noise=0.2, seed=4)
    flat = fit_dose_response(dose, shallow)
    assert abs(flat.hill) <= SHALLOW_HILL
    assert flat.is_shallow() is True
    assert any("barely bends" in c for c in flat.caveats())


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_fewer_than_four_distinct_concentrations_is_refused():
    """Four parameters, four points minimum, and the message says so."""
    dose = np.repeat([1.0, 3.0, 9.0], 4)
    response = np.repeat([90.0, 50.0, 10.0], 4) + np.arange(12) * 0.1
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response(dose, response)
    message = str(caught.value)
    assert "four parameters" in message
    assert "3 distinct positive concentration" in message
    assert str(MIN_DOSES) in message


def test_too_few_observations_for_any_uncertainty_is_refused():
    """Four observations leaves zero residual df; the message names it."""
    dose = np.array([1.0, 3.0, 9.0, 27.0])
    response = np.array([90.0, 70.0, 30.0, 10.0])
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response(dose, response)
    assert "0 residual degrees of freedom" in str(caught.value)


def test_a_negative_concentration_is_refused_and_named():
    """Zero is a control; negative is a data error, and they are different."""
    dose = np.concatenate([[-1.0], DOSES])
    response = np.concatenate([[100.0], np.linspace(100.0, 0.0, DOSES.size)])
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response(dose, response)
    message = str(caught.value)
    assert "negative" in message
    assert "-1" in message
    assert "vehicle control" in message
    assert "log dose" in message


def test_a_zero_concentration_is_a_vehicle_control_not_an_error():
    """It is excluded from the fit deliberately, counted, and reported."""
    dose, response = series(ec50=1.0, seed=1)
    dose = np.concatenate([np.zeros(3), dose])
    response = np.concatenate([[99.5, 100.4, 100.1], response])

    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))
    assert result.n_vehicle == 3
    assert result.vehicle_response == pytest.approx(100.0, abs=0.5)
    # The vehicle rows are not in the fit: the observation count is unchanged.
    assert result.n_obs == 30
    assert result.n_doses == 10
    assert result.ec50 == pytest.approx(1.0, rel=0.10)
    assert any("vehicle observation" in c for c in result.caveats())


def test_a_series_of_nothing_but_vehicle_is_refused_as_too_few_doses():
    """log10(0) is never taken, so the refusal is about the doses."""
    dose = np.zeros(8)
    response = np.linspace(0.0, 1.0, 8)
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response(dose, response)
    message = str(caught.value)
    assert "0 distinct positive concentration" in message
    assert "vehicle observation" in message


def test_a_constant_response_is_refused():
    """There is no curve in a flat line."""
    dose = np.repeat(DOSES, 2)
    response = np.full(dose.size, 7.0)
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response(dose, response)
    message = str(caught.value)
    assert "every response is 7" in message
    assert "unidentified" in message


def test_mismatched_input_lengths_are_refused():
    with pytest.raises(DoseResponseError) as caught:
        fit_dose_response([1.0, 2.0, 3.0], [1.0, 2.0])
    assert "pair up one to one" in str(caught.value)


def test_missing_values_are_dropped_and_counted():
    """A NaN response is a row that was not measured, not a zero."""
    dose, response = series(ec50=1.0, seed=1)
    response = response.copy()
    response[[0, 5]] = np.nan
    dose = dose.copy()
    dose[7] = np.inf
    result = fit_dose_response(dose, response)
    assert result.n_excluded == 3
    assert result.n_obs == 27
    assert any("non-finite" in c for c in result.caveats())


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_the_spec_validates_at_construction():
    with pytest.raises(DoseResponseError) as caught:
        DoseResponseSpec(ci_method="bootstrap")
    assert "profile" in str(caught.value) and "wald" in str(caught.value)

    with pytest.raises(DoseResponseError):
        DoseResponseSpec(direction="down")

    with pytest.raises(DoseResponseError) as caught:
        DoseResponseSpec(confidence=95)
    assert "pass 0.95, not 95" in str(caught.value)

    with pytest.raises(DoseResponseError):
        DoseResponseSpec(max_reversal=0.0)


def test_the_spec_round_trips_through_json():
    spec = DoseResponseSpec(concentration="conc_uM", response="fraction_dead",
                            group="gene", ci_method=CI_WALD, confidence=0.90,
                            unit="µM", direction=DIRECTION_ACTIVATION,
                            allow_non_monotone=True, max_reversal=0.4)
    again = DoseResponseSpec.from_json(spec.to_json())
    assert again == spec
    assert json.loads(spec.to_json())["ci_method"] == CI_WALD
    # Unknown keys are ignored and missing ones defaulted.
    relaxed = DoseResponseSpec.from_dict({"response": "y", "spurious": 1})
    assert relaxed.response == "y"
    assert relaxed.ci_method == CI_PROFILE
    assert "profile-likelihood" in DoseResponseSpec().describe()
    assert spec.with_unit("nM").unit == "nM"
    assert spec.with_ci_method(CI_PROFILE).ci_method == CI_PROFILE
    assert spec.with_columns("a", "b").concentration == "a"


def test_an_empty_group_string_means_no_grouping():
    assert DoseResponseSpec(group="  ").group is None


def test_the_direction_can_be_pinned():
    """Inference is the default, not the only option."""
    dose, response = series(ec50=1.0, hill=-1.0, seed=1)
    forced = fit_dose_response(
        dose, response, DoseResponseSpec(direction=DIRECTION_INHIBITION))
    assert forced.hill < 0
    assert forced.ec50 == pytest.approx(1.0, rel=0.10)


# ---------------------------------------------------------------------------
# Frames and text
# ---------------------------------------------------------------------------

def test_the_result_exports_points_a_curve_and_parameters():
    dose, response = series(ec50=1.0, seed=1)
    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))

    points = result.points_frame()
    assert len(points) == 30
    assert set(points.columns) == {"group", "concentration", "response",
                                   "fitted", "residual"}
    assert np.allclose(points["response"] - points["fitted"],
                       points["residual"])

    curve = result.curve_frame(points=50)
    assert len(curve) == 50
    # Geometric spacing: equal ratios, not equal differences.
    ratios = curve["concentration"].to_numpy()[1:] / \
        curve["concentration"].to_numpy()[:-1]
    assert np.allclose(ratios, ratios[0])

    parameters = result.parameter_frame()
    assert list(parameters["parameter"]) == [
        "bottom", "top", "log10_ec50", "hill", "ec50"]
    ec50_row = parameters[parameters["parameter"] == "ec50"].iloc[0]
    assert ec50_row["estimate"] == pytest.approx(result.ec50)

    assert result.predict(result.ec50) == pytest.approx(
        0.5 * (result.top + result.bottom))


def test_the_headline_and_report_carry_the_unit_and_the_numbers():
    dose, response = series(ec50=1.0, seed=1)
    result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))
    headline = result.headline()
    assert "EC50 = " in headline and "µM" in headline
    assert "95% profile CI" in headline
    assert "inhibition" in headline
    report = result.report()
    assert "residual SE" in report
    assert "lack of fit vs pure error" in report
    assert result.headline() in report


# ---------------------------------------------------------------------------
# Whole plates
# ---------------------------------------------------------------------------

def plate() -> pd.DataFrame:
    """Two genes that fit, one that is bell-shaped, one that is truncated."""
    parts = []
    for gene, ec50 in (("geneA", 1.0), ("geneB", 0.05)):
        dose, response = series(ec50=ec50, seed=hash(gene) % 1000)
        parts.append(pd.DataFrame({"gene": gene, "conc": dose,
                                   "signal": response}))
    dose, response = bell()
    parts.append(pd.DataFrame({"gene": "geneC", "conc": dose,
                               "signal": response}))
    dose, response = series(ec50=1.0, doses=TRUNCATED, seed=20260804)
    parts.append(pd.DataFrame({"gene": "geneD", "conc": dose,
                               "signal": response}))
    return pd.concat(parts, ignore_index=True)


def test_a_plate_fits_one_curve_per_group_and_keeps_the_refusals():
    result = fit_frame(plate(), DoseResponseSpec(
        concentration="conc", response="signal", group="gene", unit="µM"))
    assert isinstance(result, DoseResponseSet)
    assert result.groups == ("geneA", "geneB", "geneC", "geneD")

    table = result.table()
    assert len(table) == 4
    statuses = dict(zip(table["group"], table["status"]))
    assert statuses["geneA"] == STATUS_FITTED
    assert statuses["geneB"] == STATUS_FITTED
    assert statuses["geneC"] == STATUS_REFUSED
    assert statuses["geneD"] == STATUS_UNBOUNDED

    rows = table.set_index("group")
    assert rows.loc["geneA", "ec50"] == pytest.approx(1.0, rel=0.10)
    assert rows.loc["geneB", "ec50"] == pytest.approx(0.05, rel=0.10)
    # A refused group has no numbers and a message; it is not silently gone.
    assert math.isnan(rows.loc["geneC", "ec50"])
    assert "not monotone" in rows.loc["geneC", "note"]
    # An unbounded group has no EC50 and a one-sided sentence.
    assert math.isnan(rows.loc["geneD", "ec50"])
    assert rows.loc["geneD", "note"].startswith("EC50 > ")
    assert np.isfinite(rows.loc["geneD", "ec50_unconstrained"])

    assert len(result.results()) == 3
    assert len(result.refusals()) == 1
    assert result.get("geneC").result is None
    assert result.get("nobody") is None
    assert "2 of 4 curve(s) give a bounded EC50" in result.headline()
    assert "1 were refused" in result.headline()
    report = result.report()
    assert "geneC: REFUSED" in report
    assert "geneD" in report


def test_a_plate_without_a_grouping_column_is_one_curve():
    frame = plate()
    frame = frame[frame["gene"] == "geneA"].reset_index(drop=True)
    result = fit_frame(frame, DoseResponseSpec(concentration="conc",
                                               response="signal"))
    assert len(result) == 1
    assert result.groups == ("",)
    assert result.fits[0].result.ec50 == pytest.approx(1.0, rel=0.10)


def test_fit_frame_refuses_a_table_it_cannot_read():
    frame = plate()
    with pytest.raises(DoseResponseError) as caught:
        fit_frame(frame, DoseResponseSpec(concentration="nope",
                                          response="signal"))
    assert "'nope' is not a column" in str(caught.value)

    with pytest.raises(DoseResponseError) as caught:
        fit_frame(frame, DoseResponseSpec(response="signal"))
    assert "both a concentration column and a response column" in \
        str(caught.value)

    with pytest.raises(DoseResponseError) as caught:
        fit_frame(frame, DoseResponseSpec(concentration="conc",
                                          response="signal", group="missing"))
    assert "grouping column" in str(caught.value)


def test_an_empty_set_still_produces_a_table_with_the_right_columns():
    empty = DoseResponseSet(fits=(), spec=DoseResponseSpec())
    table = empty.table()
    assert len(table) == 0
    assert "ec50" in table.columns and "note" in table.columns


# ---------------------------------------------------------------------------
# Column suggestions (the one seam that needs Qt installed)
# ---------------------------------------------------------------------------

def test_a_dilution_series_column_is_offered_even_though_it_is_low_cardinality():
    """The shared classifier calls an 8-level numeric column categorical.

    That is right for ``cell_count`` and wrong for a dilution series, which
    has exactly as many levels as it has dilutions by design — so the
    concentration picker uses the classifier only to throw out keys and free
    text.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.dose_response import (
        candidate_concentration_columns, candidate_response_columns)

    frame = plate()
    frame["well_id"] = [f"A{i:03d}" for i in range(len(frame))]
    assert "conc" in candidate_concentration_columns(frame)
    assert "well_id" not in candidate_concentration_columns(frame)
    assert "gene" not in candidate_concentration_columns(frame)
    assert "signal" in candidate_response_columns(frame)
    assert "gene" not in candidate_response_columns(frame)
