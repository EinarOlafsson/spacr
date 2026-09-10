"""Bliss and Loewe over a checkerboard, and why the answer is not one number.

387 asks for the interaction SURFACE rather than an index, in as many words:
"one number for a whole checkerboard hides exactly the concentration-dependent
structure that makes synergy interesting". A combination is routinely
synergistic in one corner of a grid and additive in another, and a mean over
the grid reports neither. So `InteractionSurface.summary()` deliberately has
no mean: it reports the strongest cell, where that cell is, and how many cells
fall each side of zero.

THE TWO MODELS ENCODE DIFFERENT NULL HYPOTHESES and are not interchangeable.
Bliss asks whether the agents act independently -- surviving fractions
multiply. Loewe asks whether they behave as dilutions of one another -- a
fixed effect costs a constant total dose. Agents of unequal potency that are
exactly Bliss-independent read as Loewe-synergistic, and that is a property of
the models rather than a defect in either. These tests pin each against the
board it is defined on rather than against each other.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.dose_response import (
    SYNERGY_BLISS, SYNERGY_LOEWE, bliss_surface, fit_dose_response,
    four_parameter_logistic, loewe_surface)

DOSES = np.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])


def _fit(ec50, seed, hill=-1.0):
    dose = np.repeat(DOSES, 3)
    safe = np.where(dose == 0, 1e-6, dose)
    y = four_parameter_logistic(safe, 0.0, 1.0, np.log10(ec50), hill)
    noise = np.random.default_rng(seed).normal(0.0, 0.002, y.shape)
    return fit_dose_response(dose, y + noise)


def _effect(dose, ec50, hill=-1.0):
    safe = np.where(dose <= 0, 1e-12, dose)
    return 1.0 - four_parameter_logistic(safe, 0.0, 1.0, np.log10(ec50), hill)


@pytest.fixture
def board():
    """A checkerboard whose combination is exactly Bliss-independent."""
    a, b = np.meshgrid(DOSES, DOSES, indexing="ij")
    ea, eb = _effect(a, 1.0), _effect(b, 3.0)
    combined = ea + eb - ea * eb
    return a, b, 1.0 - combined, _fit(1.0, 1), _fit(3.0, 2)


def test_bliss_reads_zero_on_an_independent_board(board):
    """THE CALIBRATION TEST. A board built to the model's own definition must
    come back at zero, or every number the model reports is offset."""
    a, b, response, fa, fb = board
    surface = bliss_surface(a.ravel(), b.ravel(), response.ravel(),
                            fit_a=fa, fit_b=fb)
    assert surface.model == SYNERGY_BLISS
    assert np.nanmax(np.abs(surface.excess)) < 0.05


def test_bliss_finds_synergy_that_was_put_there(board):
    """Fifteen percent of the remaining headroom, added to every cell."""
    a, b, response, fa, fb = board
    combined = 1.0 - response
    boosted = combined + 0.15 * (1.0 - combined)
    surface = bliss_surface(a.ravel(), b.ravel(), (1.0 - boosted).ravel(),
                            fit_a=fa, fit_b=fb)
    summary = surface.summary()
    assert summary["max_excess"] > 0.1
    assert summary["synergistic_cells"] > summary["n_cells"] * 0.9


def test_loewe_reads_zero_where_one_agent_is_absent(board):
    """LOEWE'S OWN CALIBRATION. With one dose at zero the combination index
    is a/Da, and the observed effect IS that agent's effect, so the index is
    exactly 1 and the excess exactly 0. Anything else means the curve is
    being inverted wrongly."""
    a, b, response, fa, fb = board
    surface = loewe_surface(a.ravel(), b.ravel(), response.ravel(),
                            fit_a=fa, fit_b=fb)
    axes = np.concatenate([surface.excess[1:, 0], surface.excess[0, 1:]])
    assert np.nanmax(np.abs(axes)) < 0.06


def test_the_untreated_well_is_not_infinitely_synergistic(board):
    """It read +1.0 -- the strongest possible synergy -- from the one well
    where nothing was combined, and was the maximum of the whole surface."""
    a, b, response, fa, fb = board
    surface = loewe_surface(a.ravel(), b.ravel(), response.ravel(),
                            fit_a=fa, fit_b=fb)
    assert np.isnan(surface.excess[0, 0])


def test_loewe_says_which_cells_it_cannot_answer(board):
    """A combination killing more than either agent alone has no Loewe dose,
    and the surface says so rather than extrapolating one."""
    a, b, response, fa, fb = board
    surface = loewe_surface(a.ravel(), b.ravel(), response.ravel(),
                            fit_a=fa, fit_b=fb)
    assert "plateaus" in surface.note
    assert surface.n_cells < surface.excess.size


def test_the_summary_reports_no_mean(board):
    """The number 387 asks not to produce is absent, not warned about."""
    a, b, response, fa, fb = board
    summary = bliss_surface(a.ravel(), b.ravel(), response.ravel(),
                            fit_a=fa, fit_b=fb).summary()
    assert "max_excess" in summary and "max_at_dose_a" in summary
    assert not any("mean" in k or "index" in k for k in summary)


def test_an_activator_is_not_reported_as_its_own_antagonist():
    """The effect conversion inverts on the Hill sign, and getting that
    backwards makes every activating compound look antagonistic."""
    from spacr.qt.widgets.dose_response import _effect_curve
    for hill in (-1.0, 1.0):
        curve = _effect_curve(_fit(1.0, 3, hill=hill))
        assert curve(np.array([0.01]))[0] < 0.1
        assert curve(np.array([100.0]))[0] > 0.9


def test_repeated_wells_are_averaged_not_dropped():
    a = np.array([1.0, 1.0, 3.0]); b = np.array([1.0, 1.0, 1.0])
    response = np.array([0.4, 0.6, 0.5])
    surface = bliss_surface(a, b, response, fit_a=_fit(1.0, 4), fit_b=_fit(3.0, 5))
    assert surface.observed.shape == (2, 1)
