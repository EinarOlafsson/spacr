"""A curve that turns around at the BOTTOM is a finding, not a bad plate.

The engine has always refused a non-monotone series, and its message has
always said the same thing: a bell shape is usually cytotoxicity at the top
dose, so drop that dose. That is right for one of the two reversals a
concentration series produces and exactly wrong for the other. A low-dose
stimulation -- hormesis -- is the finding, and dropping the doses it sits on
throws it away.

Worse, the quiet case: a hump can be worth a fifth of the response span and
still sit inside :data:`MAX_REVERSAL`, in which case the old code fitted it,
reported an EC50 displaced by the hump, and said nothing at all. The
``f = 80`` series below is that case -- monotone by the check, hormetic by
the test, and its monotone EC50 is 2.1 where the underlying curve's is 1.

Every series here is built from `brain_cousens` with a known stimulation
coefficient, so "how big is the hump" is a number and not an impression::

    f =  40   6% of the span   real (p = 2e-05) and below the minimum effect
    f =  80  17% of the span   hormesis, and the monotonicity check passes it
    f = 200  66% of the span   hormesis, and the monotonicity check refuses it
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import (
    HORMESIS_MIN_EFFECT, STATUS_REFUSED, DoseResponseError, DoseResponseSpec,
    brain_cousens, four_parameter_logistic, fit_dose_response, hormesis,
    monotonicity,
)

pytestmark = pytest.mark.qt

DOSES = np.sort(81.0 / 3.0 ** np.arange(9))
REPLICATES = 3
SEED = 20260919


def _dose():
    """A nine-point 3-fold series in triplicate, EC50 in the middle of it."""
    return np.repeat(DOSES, REPLICATES)


def _hormetic(stimulation, seed=SEED, noise=2.0):
    """An inhibition series with a known low-dose hump on it."""
    dose = _dose()
    clean = brain_cousens(dose, 0.0, 100.0, 0.0, -2.0, stimulation)
    return dose, clean + np.random.default_rng(seed).normal(0.0, noise,
                                                            dose.size)


def _cytotoxic(seed=SEED, noise=2.0):
    """An activation series the top two doses kill: the OTHER reversal."""
    dose = _dose()
    clean = four_parameter_logistic(dose, 0.0, 100.0, 0.0, 1.4)
    collapsed = np.where(dose >= 27.0, 8.0, clean)
    return dose, collapsed + np.random.default_rng(seed).normal(0.0, noise,
                                                                dose.size)


def test_the_hormesis_model_is_the_plain_curve_when_nothing_is_stimulated():
    """Brain-Cousens nests the 4PL, which is what the F test rests on."""
    dose = _dose()
    plain = four_parameter_logistic(dose, 0.0, 100.0, 0.0, -2.0)
    hormetic = brain_cousens(dose, 0.0, 100.0, 0.0, -2.0, 0.0)

    assert hormetic == pytest.approx(plain, abs=1e-12)


def test_a_hump_the_monotonicity_check_passes_is_still_reported():
    """The quiet case, and the reason this is not only a refusal message."""
    dose, response = _hormetic(80.0)

    assert monotonicity(dose, response).is_monotone
    result = fit_dose_response(dose, response)

    assert result.hormesis is not None
    assert result.hormesis.is_hormetic, result.hormesis.note
    assert result.hormesis.max_stimulation_fraction > HORMESIS_MIN_EFFECT
    assert result.ec50 > 1.5, (
        "the monotone fit should be visibly displaced by the hump")
    assert any("hormetic" in caveat for caveat in result.caveats()), \
        result.caveats()
    assert "hormesis" in result.report()


def test_a_hormetic_series_is_refused_as_hormesis_and_not_as_cytotoxicity():
    """The two reversals get two different sentences."""
    dose, response = _hormetic(200.0)

    assert not monotonicity(dose, response).is_monotone
    with pytest.raises(DoseResponseError) as refusal:
        fit_dose_response(dose, response)

    message = str(refusal.value)
    assert "hormetic" in message
    assert "cytotoxicity at the top dose" not in message
    assert "Re-fit without the top dose" not in message
    assert "dropping the doses it sits on would throw it away" in message


def test_a_top_dose_collapse_is_still_refused_as_the_bell_it_is():
    """And says, in the same breath, that it was tested for the other one."""
    dose, response = _cytotoxic()

    verdict = hormesis(dose, response)
    with pytest.raises(DoseResponseError) as refusal:
        fit_dose_response(dose, response)

    assert not verdict.is_hormetic
    message = str(refusal.value)
    assert "cytotoxicity at the top dose" in message
    assert "tested for hormesis and it is not that" in message


def test_a_hump_below_the_minimum_effect_is_not_called_hormesis():
    """Significance is not size, and this one is significant and small."""
    dose, response = _hormetic(40.0)

    verdict = hormesis(dose, response)

    assert verdict.p_value < 0.01, verdict.p_value
    assert verdict.delta_aic > 2.0, verdict.delta_aic
    assert 0 < verdict.max_stimulation_fraction < HORMESIS_MIN_EFFECT
    assert not verdict.is_hormetic
    assert "minimum effect" in verdict.note
    assert fit_dose_response(dose, response).hormesis.is_hormetic is False


def test_the_minimum_effect_is_a_threshold_and_not_a_verdict():
    """Lowering it below the measured hump changes the same series' answer."""
    dose, response = _hormetic(40.0)

    default = hormesis(dose, response)
    lenient = hormesis(dose, response,
                       min_effect=default.max_stimulation_fraction / 2.0)

    assert not default.is_hormetic and lenient.is_hormetic
    assert lenient.max_stimulation_fraction == pytest.approx(
        default.max_stimulation_fraction)


def test_a_monotone_series_is_not_hormetic_and_is_not_even_tested():
    """Nothing is spent on a series with no low-dose excursion at all."""
    dose = _dose()
    clean = four_parameter_logistic(dose, 0.0, 100.0, 0.0, -1.5)
    response = clean + np.random.default_rng(5).normal(0.0, 2.0, dose.size)

    verdict = hormesis(dose, response)
    result = fit_dose_response(dose, response)

    assert not verdict.is_hormetic
    assert verdict.stimulation < 0
    assert result.hormesis is None
    assert "hormesis" not in result.report()


def test_hormesis_on_an_activation_series_is_a_dip_below_the_control():
    """The mirror case: stimulated means away from the main effect."""
    dose, hump = _hormetic(120.0)
    response = 100.0 - hump

    verdict = hormesis(dose, response)

    assert verdict.is_hormetic, verdict.note
    assert verdict.stimulation > 0
    assert verdict.peak_dose < 10.0


def test_the_verdict_carries_what_ignoring_the_hump_costs():
    """Both EC50s, so the displacement is a number in the report."""
    dose, response = _hormetic(80.0)

    verdict = hormesis(dose, response)

    assert verdict.ec50_monotone is not None
    assert verdict.ec50_hormetic is not None
    assert verdict.ec50_monotone > verdict.ec50_hormetic
    assert "the monotone fit puts the EC50 at" in verdict.describe()


@pytest.fixture()
def screen(qtbot):
    """One hormetic compound and one ordinary one, on the same plate."""
    dose, hormetic = _hormetic(200.0)
    ordinary = (four_parameter_logistic(dose, 0.0, 100.0, 0.0, -1.5)
                + np.random.default_rng(11).normal(0.0, 2.0, dose.size))
    frame = pd.concat([
        pd.DataFrame({"gene": "hormetic", "conc_uM": dose,
                      "parasite_killed": hormetic}),
        pd.DataFrame({"gene": "ordinary", "conc_uM": dose,
                      "parasite_killed": ordinary}),
    ], ignore_index=True)
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="two compounds")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("parasite_killed")
    widget.group_picker.setCurrentText("gene")
    return widget


def _cell(screen, row, key):
    from spacr.qt.screens.dose_response import TABLE_COLUMNS
    column = [k for k, _h in TABLE_COLUMNS].index(key)
    return screen.table.item(row, column)


def test_the_screen_shows_the_hormesis_refusal_in_the_row_and_the_report(
        screen):
    """The word a user has to see is in the cell, not only in the tooltip."""
    screen.fit()

    table = screen.result_set().table()
    row = int(table.index[table["group"] == "hormetic"][0])
    assert table.loc[row, "status"] == STATUS_REFUSED
    assert "hormetic" in _cell(screen, row, "note").text()
    assert "hormetic" in _cell(screen, row, "note").toolTip()

    screen.show_group(row)
    assert "hormetic" in screen.report.toPlainText()


def test_forcing_the_fit_keeps_the_finding_on_the_curve(screen):
    """Ticking the override fits it and does not lose the diagnosis."""
    screen.force_check.setChecked(True)

    screen.fit()

    fit = screen.result_set().get("hormetic")
    assert fit.result is not None
    assert fit.result.hormesis.is_hormetic
    screen.show_group(screen.result_set().groups.index("hormetic"))
    assert "hormetic" in screen.report.toPlainText()
