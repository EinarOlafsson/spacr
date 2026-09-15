"""The Dose-Response screen normalises each plate to its own controls.

The engine has done this since `normalise_to_controls` and `PlateSpec`; the
screen could not reach it. These tests are about the trip from the pickers to
the engine and back: which response column is fitted, what each plate's
controls said, and that a plate the engine refuses is left out of the curve
and named in the report rather than silently scaled by another plate's
controls.

Three plates of one compound, EC50 1 uM, each read at a different absolute
scale -- which is the whole reason to normalise per plate::

    P1   vehicle ~1000, positive ~100       -> usable
    P2   vehicle ~2000, positive ~200       -> usable, twice P1's scale
    P3   positive wells only, no vehicle    -> refused, and left out
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import (
    PERCENT_COLUMN, STATUS_FITTED, STATUS_REFUSED, four_parameter_logistic,
)

pytestmark = pytest.mark.qt

DOSES = 27.0 / 3.0 ** np.arange(10)
REPLICATES = 3
CONTROL_WELLS = 8


def _plate(name, negative, positive, rng, *, with_negative=True):
    """One plate: a dose series of geneA plus its control wells."""
    dose = np.repeat(DOSES, REPLICATES)
    inhibition = four_parameter_logistic(dose, 0.0, 100.0, np.log10(1.0), 1.0)
    window = negative - positive
    signal = negative - inhibition / 100.0 * window
    rows = [pd.DataFrame({
        "plate": name, "role": "sample", "gene": "geneA", "conc_uM": dose,
        "signal": signal + rng.normal(0.0, window * 0.01, dose.size)})]
    rows.append(pd.DataFrame({
        "plate": name, "role": "pos", "gene": "control", "conc_uM": 0.0,
        "signal": positive + rng.normal(0.0, window * 0.02, CONTROL_WELLS)}))
    if with_negative:
        rows.append(pd.DataFrame({
            "plate": name, "role": "neg", "gene": "control", "conc_uM": 0.0,
            "signal": negative + rng.normal(0.0, window * 0.02,
                                            CONTROL_WELLS)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture()
def frame():
    """P1 and P2 usable at different scales; P3 has no vehicle wells."""
    rng = np.random.default_rng(20260915)
    return pd.concat([
        _plate("P1", 1000.0, 100.0, rng),
        _plate("P2", 2000.0, 200.0, rng),
        _plate("P3", 1500.0, 150.0, rng, with_negative=False),
    ], ignore_index=True)


@pytest.fixture()
def screen(qtbot, frame):
    """The frame loaded and the dose, response and group columns chosen."""
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="three plates")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    return widget


def _items(picker):
    return [picker.itemText(i) for i in range(picker.count())]


def _gene_a(screen):
    return next(fit for fit in screen.result_set().fits if fit.group == "geneA")


def test_the_plate_row_starts_unchosen_so_the_default_fit_is_unchanged(screen):
    """Nothing is normalised until a plate AND a control column are chosen."""
    assert screen.plate_picker.currentText() == "(none)"
    assert screen.control_picker.currentText() == "(none)"
    assert "plate" in _items(screen.plate_picker)
    assert "role" in _items(screen.control_picker)
    assert _items(screen.positive_picker) == []

    screen.fit()

    assert screen.result_set().spec.response == "signal"
    assert screen._plate_reports == ()


def test_choosing_the_control_column_offers_its_levels_and_guesses_both(screen):
    """The two controls are levels of the chosen column, pre-picked by name."""
    screen.control_picker.setCurrentText("role")

    assert _items(screen.positive_picker) == ["neg", "pos", "sample"]
    assert screen.positive_picker.currentText() == "pos"
    assert screen.negative_picker.currentText() == "neg"


def test_normalising_per_plate_fits_percent_inhibition(screen):
    """Two plates at different scales become one curve on one scale."""
    screen.plate_picker.setCurrentText("plate")
    screen.control_picker.setCurrentText("role")

    screen.fit()

    assert screen.result_set().spec.response == PERCENT_COLUMN
    fit = _gene_a(screen)
    assert fit.status == STATUS_FITTED
    assert 0.5 < fit.result.ec50 < 2.0, fit.result.ec50
    report = screen.report.toPlainText()
    assert report.startswith("P1: usable, Z′ ")
    assert "P2: usable, Z′ " in report


def test_a_plate_without_a_vehicle_is_refused_named_and_left_out(screen):
    """A refusal is a line in the report and no wells in the curve."""
    screen.plate_picker.setCurrentText("plate")
    screen.control_picker.setCurrentText("role")

    screen.fit()

    by_plate = {report.plate: report for report in screen._plate_reports}
    assert by_plate["P3"].status == STATUS_REFUSED
    assert "P3: refused" in screen.report.toPlainText()
    assert "no negative control well" in screen.report.toPlainText()
    # P1 and P2 each carry the full series; P3's are NaN and dropped.
    assert _gene_a(screen).result.n_obs == 2 * DOSES.size * REPLICATES


def test_one_level_for_both_controls_fits_nothing_and_says_why(screen):
    """No assay window on any plate is the engine's refusal, shown whole."""
    screen.plate_picker.setCurrentText("plate")
    screen.control_picker.setCurrentText("role")
    screen.negative_picker.setCurrentText("pos")

    screen.fit()

    assert screen.result_set() is None
    assert "usable pair of controls" in screen.report.toPlainText()
