"""The Dose-Response screen pools each group's EC50 across replicate plates.

`pool_frame` combines per-plate fits on the log10 scale with plate as a random
effect; the screen could not reach it, so a user with three replicate plates
fitted three curves and averaged the numbers by hand. These tests are about
the trip from the Plate picker to the engine and back.

The plates are the ones the normalisation tests build -- one compound, EC50
1 uM, three plates at three absolute scales, P3 with no vehicle wells -- so
the two features are checked against the same data.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.dose_response import STATUS_FITTED

# The same plates and the same screen fixture as the normalisation tests.
from tests.qt.test_dose_response_normalises_to_each_plate import (  # noqa: F401
    frame,
    screen,
)

pytestmark = pytest.mark.qt


def test_without_a_plate_column_nothing_is_pooled(screen):
    """The default fit is the one it always was."""
    screen.fit()

    assert screen._pooled == {}
    assert "pooled EC50" not in screen.report.toPlainText()


def test_each_group_is_pooled_across_the_plates_it_fitted_on(screen):
    """Normalised, geneA fits on P1 and P2 and pools to one EC50 near 1."""
    screen.plate_picker.setCurrentText("plate")
    screen.control_picker.setCurrentText("role")

    screen.fit()

    pooled = screen._pooled["geneA"]
    assert pooled.status == STATUS_FITTED
    assert pooled.n_used == 2
    assert 0.5 < pooled.ec50 < 2.0, pooled.ec50
    assert "geneA: pooled EC50 " in screen.report.toPlainText()


def test_a_plate_that_could_not_be_fitted_is_named_in_the_pooled_line(screen):
    """P3's wells are NaN once normalised; the pooled line says so."""
    screen.plate_picker.setCurrentText("plate")
    screen.control_picker.setCurrentText("role")

    screen.fit()

    assert "P3" in screen._pooled["geneA"].note
    line = next(line for line in screen.report.toPlainText().splitlines()
                if line.startswith("geneA: pooled EC50 "))
    assert "P3" in line


def test_pooling_needs_only_the_plate_column(screen):
    """Each plate is fitted on its own scale, so raw plates pool too."""
    screen.plate_picker.setCurrentText("plate")

    screen.fit()

    assert screen.result_set().spec.response == "signal"
    pooled = screen._pooled["geneA"]
    assert pooled.status == STATUS_FITTED
    assert pooled.n_used == 3
    assert 0.5 < pooled.ec50 < 2.0, pooled.ec50


def test_a_group_seen_on_one_plate_is_not_pooled(qtbot, frame):
    """One replicate is one fit, already in the grid -- not a pooled claim."""
    from spacr.qt.screens.dose_response import DoseResponseScreen

    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame[frame["plate"] == "P1"].reset_index(drop=True))
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    widget.plate_picker.setCurrentText("plate")

    widget.fit()

    assert widget.result_set() is not None
    assert "geneA" not in widget._pooled
