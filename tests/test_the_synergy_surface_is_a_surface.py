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
import pandas as pd
import pytest

from spacr.qt.widgets.dose_response import (
    SYNERGY_BLISS, SYNERGY_LOEWE, DoseResponseError, bliss_surface,
    checkerboard_from_frame, fit_dose_response, four_parameter_logistic,
    loewe_surface)

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


# ---------------------------------------------------------------------------
# Reading a checkerboard off a well table
# ---------------------------------------------------------------------------

BOARD = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def _board_frame(a_doses=BOARD, b_doses=BOARD):
    """Every combination of two dose series, including both single-agent axes."""
    rows = []
    for a in a_doses:
        for b in b_doses:
            effect = 1.0 - (1.0 / (1.0 + a)) * (1.0 / (1.0 + b))
            rows.append({"tmp": a, "pyr": b, "burden": 100.0 * (1.0 - effect)})
    return pd.DataFrame(rows)


def test_the_module_exports_everything_it_names():
    """`__all__` is a promise, and a star import is how it gets tested.

    A name listed in `__all__` with no symbol behind it makes
    ``from ... import *`` raise AttributeError -- which no test that imports
    by name will ever notice. This one does.
    """
    import spacr.qt.widgets.dose_response as module

    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"named in __all__ but absent: {missing}"


def test_a_checkerboard_keeps_its_single_agent_axes():
    """Both surfaces are calibrated against those rows, so they must survive.

    A caller who filtered the single-agent wells out would get a surface with
    no reference -- the combination predicted from the combination.
    """
    board = checkerboard_from_frame(_board_frame(), dose_a="tmp",
                                    dose_b="pyr", response="burden")

    assert board.shape == (len(BOARD), len(BOARD))
    a_dose, a_response = board.a_alone
    b_dose, b_response = board.b_alone
    assert sorted(a_dose) == list(BOARD[1:])
    assert sorted(b_dose) == list(BOARD[1:])
    assert len(a_response) == len(a_dose)
    assert len(b_response) == len(b_dose)


def test_a_checkerboard_feeds_the_surfaces_it_was_read_for():
    """The join is only useful if what comes out goes straight in."""
    board = checkerboard_from_frame(_board_frame(), dose_a="tmp",
                                    dose_b="pyr", response="burden")
    fit_a = fit_dose_response(*board.a_alone, group="tmp")
    fit_b = fit_dose_response(*board.b_alone, group="pyr")

    surface = bliss_surface(board.dose_a, board.dose_b, board.response,
                            fit_a=fit_a, fit_b=fit_b)
    assert surface.excess.shape == board.shape


def test_a_pair_of_dose_series_is_not_a_checkerboard():
    """No well has both agents, so there is no interaction to measure."""
    board = _board_frame()
    frame = board.loc[(board["tmp"] == 0) | (board["pyr"] == 0)].reset_index(
        drop=True)
    with pytest.raises(DoseResponseError, match="two dose series"):
        checkerboard_from_frame(frame, dose_a="tmp", dose_b="pyr",
                                response="burden")


def test_a_board_with_no_single_agent_row_is_refused_by_name():
    """Without agent B alone there is no curve for B to predict from."""
    frame = _board_frame()
    frame = frame.loc[frame["tmp"] > 0].reset_index(drop=True)

    with pytest.raises(DoseResponseError, match="no well has pyr alone"):
        checkerboard_from_frame(frame, dose_a="tmp", dose_b="pyr",
                                response="burden")
