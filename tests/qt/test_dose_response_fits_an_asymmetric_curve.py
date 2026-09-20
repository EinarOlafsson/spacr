"""The five-parameter logistic, offered by the screen and never defaulted to.

A 4PL is symmetric about its midpoint. Real curves are not always, and the
failure is quiet: a symmetric fit to an asymmetric series does not diverge or
look wrong, it reports a *different EC50* with a clean interval and an R²
above 0.99. The series here is built from
``five_parameter_logistic(..., asymmetry=0.2)`` with a true EC50 of exactly
1, so the displacement is a measurement rather than an argument: the
symmetric fit to it reports about 0.5. The scatter is small on purpose --
half a unit on a span of a hundred -- because the question these tests ask is
about the model and not about the seed.

The other half of the item is the refusal to make this the default. Two of
these tests are about the 5PL declining to claim anything on a symmetric
series: the asymmetry it fits there is inside the noise, its F test says so,
and the fit says so in its own caveats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import spacr.qt.widgets.dose_response as engine
from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import (
    CI_PROFILE, CI_WALD, MODEL_4PL, MODEL_5PL, STATUS_FITTED,
    DoseResponseError, DoseResponseSpec, fit_dose_response,
    five_parameter_logistic, four_parameter_logistic,
)

pytestmark = pytest.mark.qt

DOSES = np.sort(81.0 / 3.0 ** np.arange(9))
REPLICATES = 3
TRUE_EC50 = 1.0
TRUE_ASYMMETRY = 0.20


def _dose():
    """The concentration column: a nine-point 3-fold series, in triplicate."""
    return np.repeat(DOSES, REPLICATES)


def _asymmetric(seed=20260919, noise=0.5):
    """An asymmetric inhibition series whose EC50 is exactly 1."""
    dose = _dose()
    clean = five_parameter_logistic(dose, 0.0, 100.0, np.log10(TRUE_EC50),
                                    -1.3, TRUE_ASYMMETRY)
    return dose, clean + np.random.default_rng(seed).normal(0.0, noise,
                                                            dose.size)


def _symmetric(seed=20260920, noise=0.5):
    """A plain 4PL series with the same EC50 and the same noise."""
    dose = _dose()
    clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(TRUE_EC50),
                                    -1.3)
    return dose, clean + np.random.default_rng(seed).normal(0.0, noise,
                                                            dose.size)


def test_the_fitted_midpoint_is_the_ec50_under_every_asymmetry():
    """``log10_ec50`` means the same thing in both models.

    Written the way the 5PL usually is, its third parameter is the
    inflection point and NOT the half-maximal concentration once the
    exponent leaves 1 -- the mistake this parameterisation exists to make
    impossible. At ``x = EC50`` the response has to be exactly halfway
    between the plateaus for every exponent.
    """
    half = []
    for asymmetry in (0.1, 0.25, 1.0, 4.0, 12.0):
        value = five_parameter_logistic(np.array([TRUE_EC50]), 0.0, 100.0,
                                        np.log10(TRUE_EC50), -1.3, asymmetry)
        half.append(float(value[0]))
    assert half == pytest.approx([50.0] * 5, abs=1e-9), half


def test_the_five_parameter_curve_is_the_four_parameter_one_at_asymmetry_one():
    """The models are nested, which is what the F test below rests on."""
    dose = _dose()
    four = four_parameter_logistic(dose, 3.0, 97.0, 0.2, -1.7)
    five = five_parameter_logistic(dose, 3.0, 97.0, 0.2, -1.7, 1.0)

    assert five == pytest.approx(four, abs=1e-12)


def test_a_symmetric_fit_moves_the_ec50_of_an_asymmetric_curve():
    """The 4PL does not fail on asymmetric data. It answers a different
    question, confidently."""
    dose, response = _asymmetric()

    symmetric = fit_dose_response(dose, response)
    asymmetric = fit_dose_response(dose, response,
                                   DoseResponseSpec(model=MODEL_5PL))

    assert symmetric.ec50 is not None and asymmetric.ec50 is not None
    assert symmetric.r_squared > 0.99, symmetric.r_squared
    assert symmetric.ec50 < 0.7 * TRUE_EC50, (
        f"the symmetric fit was expected to be displaced below the true "
        f"EC50 of {TRUE_EC50}; it reported {symmetric.ec50}")
    assert asymmetric.ec50 == pytest.approx(TRUE_EC50, rel=0.3), (
        f"the asymmetric fit reported {asymmetric.ec50}")
    assert abs(np.log10(asymmetric.ec50 / TRUE_EC50)) < \
        abs(np.log10(symmetric.ec50 / TRUE_EC50))
    assert asymmetric.asymmetry == pytest.approx(TRUE_ASYMMETRY, rel=0.6)
    assert asymmetric.asymmetry_p < 0.01, asymmetric.asymmetry_p
    assert any("asymmetry is supported" in caveat
               for caveat in asymmetric.caveats()), asymmetric.caveats()


def test_the_asymmetry_of_a_symmetric_series_is_reported_as_unsupported():
    """The case the default exists for: the fifth parameter finds nothing."""
    dose, response = _symmetric()

    symmetric = fit_dose_response(dose, response)
    asymmetric = fit_dose_response(dose, response,
                                   DoseResponseSpec(model=MODEL_5PL))

    assert asymmetric.asymmetry_p > 0.05, asymmetric.asymmetry_p
    assert asymmetric.ec50 == pytest.approx(symmetric.ec50, rel=0.15)
    assert any("did not earn itself" in caveat
               for caveat in asymmetric.caveats()), asymmetric.caveats()


def test_the_fifth_parameter_costs_a_degree_of_freedom_and_a_concentration():
    """It is a parameter, not a setting: the counts move with it."""
    dose, response = _asymmetric()

    four = fit_dose_response(dose, response)
    five = fit_dose_response(dose, response, DoseResponseSpec(model=MODEL_5PL))

    assert four.dof == dose.size - 4
    assert five.dof == dose.size - 5
    assert four.n_parameters == 4 and five.n_parameters == 5
    assert len(four.parameters) == 4 and len(five.parameters) == 5
    assert "asymmetry" in set(five.parameter_frame()["parameter"])
    assert "asymmetry" not in set(four.parameter_frame()["parameter"])

    four_doses = np.repeat(DOSES[:4], REPLICATES)
    short = four_parameter_logistic(four_doses, 0.0, 100.0, 0.0, -1.3)
    fit_dose_response(four_doses, short)
    with pytest.raises(DoseResponseError) as refusal:
        fit_dose_response(four_doses, short,
                          DoseResponseSpec(model=MODEL_5PL))
    assert "five-parameter logistic has five parameters" in \
        str(refusal.value)


def test_the_four_parameter_model_is_what_a_spec_is_unless_asked():
    """Nothing chooses the 5PL on a user's behalf, including a round trip."""
    assert DoseResponseSpec().model == MODEL_4PL
    assert DoseResponseSpec(model=MODEL_5PL).describe().startswith("5PL ·")
    assert DoseResponseSpec().describe().startswith("4PL ·")

    restored = DoseResponseSpec.from_json(
        DoseResponseSpec(model=MODEL_5PL).to_json())
    assert restored.model == MODEL_5PL
    with pytest.raises(DoseResponseError):
        DoseResponseSpec(model="6pl")


def test_the_asymmetric_profile_attains_its_own_minimum():
    """The invariant every profile interval rests on.

    ``_profile_sse`` is ``min SSE`` over every parameter but the midpoint.
    Evaluated AT the fitted midpoint it must return the fit's own residual
    sum of squares, because there the conditional minimum is the
    unconditional one. A profiler that stops short returns something larger,
    and since the threshold the walk is allowed to spend is only
    ``sse * (1 + q**2/dof)`` -- about 1.2 * sse here -- overshooting by a few
    percent at the centre eats the allowance before the walk has moved, and
    the interval comes back far too narrow. This is asserted directly rather
    than inferred from the intervals, because an interval that is 45% of its
    honest width still looks like an interval.
    """
    for asymmetry in (0.1, 0.2, 0.25, 0.5, 1.0, 3.0, 8.0):
        dose = _dose()
        clean = five_parameter_logistic(dose, 0.0, 100.0, np.log10(TRUE_EC50),
                                        -1.3, asymmetry)
        response = clean + np.random.default_rng(4242).normal(0.0, 0.5,
                                                              dose.size)
        result = fit_dose_response(dose, response,
                                   DoseResponseSpec(model=MODEL_5PL))
        conditional = engine._profile_sse(
            np.log10(dose), response, result.log10_ec50,
            -1.0 if result.hill < 0 else 1.0, MODEL_5PL)

        assert conditional <= result.sse * (1.0 + 1e-6), (
            f"at asymmetry {asymmetry} the profile's conditional minimum "
            f"{conditional} exceeds the fit's own sse {result.sse} by "
            f"{conditional / result.sse:.4f}x; every 5PL profile interval "
            f"built on it is too narrow")


def test_the_asymmetric_profile_interval_covers_the_ec50_it_is_built_for():
    """What the interval is for, counted.

    Ten draws from the same asymmetric truth, through the default interval
    method, asking how often the 95% interval contains the EC50 that
    generated the data. Ten is far too few to estimate coverage, and that is
    not what this counts: it is a floor that a badly under-covering interval
    cannot clear. The single sweep of coordinate descent this replaced
    covered 5 of 12.
    """
    seeds = (20260919, 20260930, 7, 11, 3, 5, 101, 202, 303, 404)
    covered = 0
    widths = []
    for seed in seeds:
        dose, response = _asymmetric(seed=seed)
        result = fit_dose_response(
            dose, response,
            DoseResponseSpec(model=MODEL_5PL, ci_method=CI_PROFILE))
        low, high = result.log10_ec50_ci
        assert low is not None and high is not None, (seed, result.status)
        assert low < result.log10_ec50 < high, (seed, low, high)
        widths.append(high - low)
        covered += bool(low <= np.log10(TRUE_EC50) <= high)

    assert covered >= 9, (
        f"the 95% profile interval covered the true EC50 in {covered} of "
        f"{len(seeds)} draws; widths (log10) {widths}")
    assert min(widths) > 0.02, (
        f"an interval this narrow is a collapsed one, not a measurement: "
        f"{widths}")


def test_an_asymmetry_that_finds_nothing_keeps_the_four_estimated_intervals(
        monkeypatch):
    """The fallback reports what was measured, not less.

    When the five-parameter search finds nothing better than the symmetric
    fit, the curve reported IS the 4PL with the exponent pinned at 1 -- and
    the four parameters the symmetric fit estimated were estimated. Throwing
    their covariance away because the vector grew a fifth entry told the
    reader the covariance was not estimable, which is false, and under a
    Wald interval it left the midpoint with no interval at all and reported
    a cleanly bounded fit as unbounded. The branch is reached here by making
    the five-parameter search return nothing, because no natural series was
    found that reaches it.
    """
    dose, response = _symmetric()
    honest = fit_dose_response(dose, response,
                               DoseResponseSpec(ci_method=CI_WALD))

    monkeypatch.setattr(engine, "_fit_five", lambda *a, **k: None)
    fallen = fit_dose_response(
        dose, response,
        DoseResponseSpec(model=MODEL_5PL, ci_method=CI_WALD))

    assert fallen.asymmetry == 1.0
    assert fallen.status == STATUS_FITTED, fallen.status
    assert fallen.ec50 is not None and fallen.ec50_low is not None \
        and fallen.ec50_high is not None
    assert fallen.ec50 == pytest.approx(honest.ec50, rel=1e-9)
    assert fallen.hill_ci[0] is not None and fallen.hill_ci[1] is not None
    assert fallen.top_ci[0] is not None and fallen.bottom_ci[0] is not None
    assert fallen.asymmetry_ci == (None, None), fallen.asymmetry_ci
    assert not any("not estimable" in caveat for caveat in fallen.caveats()), \
        fallen.caveats()
    assert any("had nothing to do" in note for note in fallen.notes), \
        fallen.notes

    honest_half = honest.hill_ci[1] - honest.hill_ci[0]
    fallen_half = fallen.hill_ci[1] - fallen.hill_ci[0]
    assert fallen_half >= honest_half, (
        f"pinning a parameter and spending a degree of freedom on it cannot "
        f"narrow the others: {fallen_half} vs {honest_half}")
    assert fallen_half == pytest.approx(honest_half, rel=0.05)


@pytest.fixture()
def screen(qtbot):
    """The screen, loaded with one asymmetric compound and nothing chosen."""
    dose, response = _asymmetric()
    frame = pd.DataFrame({"gene": "geneA", "conc_uM": dose,
                          "parasite_killed": response})
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="one asymmetric compound")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("parasite_killed")
    widget.group_picker.setCurrentText("gene")
    return widget


def test_the_curve_picker_offers_both_models_and_starts_symmetric(screen):
    """The default on screen is the default in the engine."""
    items = [screen.model_picker.itemText(i)
             for i in range(screen.model_picker.count())]
    assert items == ["Four-parameter logistic (symmetric)",
                     "Five-parameter logistic (asymmetric)"]
    assert screen.model_picker.currentData() == MODEL_4PL
    assert screen.spec().model == MODEL_4PL

    screen.fit()

    assert "4PL dose–response" in screen.report.toPlainText()


def test_choosing_the_asymmetric_curve_fits_it_and_shows_the_test(screen):
    """Driving the picker changes the model, the report and the verdict."""
    screen.model_picker.setCurrentIndex(1)
    assert screen.spec().model == MODEL_5PL

    screen.fit()

    text = screen.report.toPlainText()
    assert "5PL dose–response" in text
    assert "asymmetry" in text
    assert "asymmetry is supported" in text
    result = screen.result_set().fits[0].result
    assert result.model == MODEL_5PL
    assert result.asymmetry_p < 0.05
