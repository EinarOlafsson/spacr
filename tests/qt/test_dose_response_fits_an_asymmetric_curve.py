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

from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import (
    MODEL_4PL, MODEL_5PL, DoseResponseError, DoseResponseSpec,
    fit_dose_response, five_parameter_logistic, four_parameter_logistic,
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
