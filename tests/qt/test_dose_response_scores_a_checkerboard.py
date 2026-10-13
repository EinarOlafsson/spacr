"""The Dose-Response screen scores a two-compound checkerboard.

`bliss_surface`, `loewe_surface` and `checkerboard_from_frame` have existed in
the engine; the screen could not reach them. These tests are about the trip
from the Second compound picker to the engine and back, on the engine's own
calibration boards: a board built to be exactly Bliss-independent must read
near zero, and synergy put into a board on purpose must be found.

Effects are fractions affected; the readout is survival, falling with dose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import (
    SYNERGY_BLISS, SYNERGY_LOEWE, four_parameter_logistic,
)

pytestmark = pytest.mark.qt

DOSES = np.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
REPLICATES = 3


def _effect(dose, ec50, hill=-1.0):
    safe = np.where(dose <= 0, 1e-12, dose)
    return 1.0 - four_parameter_logistic(safe, 0.0, 1.0, np.log10(ec50), hill)


def _board(boost=0.0, seed=0, without_b_alone=False):
    """Every combination of the two dose series, three wells each.

    :param boost: the share of the remaining headroom added to every
        COMBINATION well -- synergy put there on purpose.
    :param without_b_alone: drop the second compound's single-agent axis.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in DOSES:
        for b in DOSES:
            if without_b_alone and a == 0 and b > 0:
                continue
            ea, eb = _effect(np.array(a), 1.0), _effect(np.array(b), 3.0)
            combined = ea + eb - ea * eb
            if a > 0 and b > 0:
                # SYNERGY LIVES ONLY WHERE BOTH COMPOUNDS ARE. Boosting the
                # single-agent axes too makes each compound look more potent
                # on its own, the single-agent fits absorb it, and Bliss
                # predicted from them matches the boosted board: an excess of
                # 0.006, measured, which is the right answer to that board.
                combined = combined + boost * (1.0 - combined)
            for _ in range(REPLICATES):
                rows.append({"cmpd_a_uM": a, "cmpd_b_uM": b,
                             "survival": float(1.0 - combined)
                             + rng.normal(0.0, 0.002)})
    return pd.DataFrame(rows)


def _screen(qtbot, frame):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="checkerboard")
    widget.concentration_picker.setCurrentText("cmpd_a_uM")
    widget.response_picker.setCurrentText("survival")
    return widget


@pytest.fixture()
def screen(qtbot):
    return _screen(qtbot, _board())


def test_the_second_compound_starts_unchosen_so_nothing_is_scored(screen):
    assert screen.second_dose_picker.currentText() == "(none)"
    items = [screen.second_dose_picker.itemText(i)
             for i in range(screen.second_dose_picker.count())]
    assert "cmpd_b_uM" in items

    screen.fit()

    assert screen._synergy == {}
    assert "excess over" not in screen.report.toPlainText()


def test_an_independent_board_reads_near_zero_against_bliss(screen):
    """THE CALIBRATION: built to Bliss's definition, it must score zero."""
    screen.second_dose_picker.setCurrentText("cmpd_b_uM")

    screen.fit()

    summary = screen._synergy[""].summary()
    assert summary["n_cells"] > 0
    assert abs(summary["max_excess"]) < 0.05, summary
    assert abs(summary["min_excess"]) < 0.05, summary
    assert "all rows: Bliss excess over " in screen.report.toPlainText()


def test_synergy_put_into_the_board_is_found(qtbot):
    widget = _screen(qtbot, _board(boost=0.15, seed=1))
    widget.second_dose_picker.setCurrentText("cmpd_b_uM")

    widget.fit()

    summary = widget._synergy[""].summary()
    assert summary["max_excess"] > 0.1, summary
    assert summary["synergistic_cells"] > summary["antagonistic_cells"]


def test_the_curve_is_the_first_compound_alone(screen):
    """Combination wells are not a dose series of either agent."""
    screen.second_dose_picker.setCurrentText("cmpd_b_uM")

    screen.fit()

    fit = screen.result_set().fits[0]
    positive_a_alone = (DOSES > 0).sum() * REPLICATES
    assert fit.result.n_obs == positive_a_alone


def test_a_board_without_the_second_compound_alone_says_why(qtbot):
    """Both models predict FROM the single agents; without one, no score."""
    widget = _screen(qtbot, _board(without_b_alone=True))
    widget.second_dose_picker.setCurrentText("cmpd_b_uM")

    widget.fit()

    refusal = widget._synergy[""]
    assert isinstance(refusal, str) and "alone" in refusal
    assert "all rows: no synergy surface — " in widget.report.toPlainText()


def test_loewe_can_be_chosen_instead(screen):
    screen.second_dose_picker.setCurrentText("cmpd_b_uM")
    screen.synergy_picker.setCurrentIndex(
        screen.synergy_picker.findData(SYNERGY_LOEWE))

    screen.fit()

    assert screen._synergy[""].model == SYNERGY_LOEWE
    assert "all rows: Loewe excess over " in screen.report.toPlainText()
