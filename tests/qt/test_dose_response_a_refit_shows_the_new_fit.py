"""Fitting a second time shows the second fit.

After a fit the screen selects the grid's first row, and selecting a row is
what draws its curve and writes its report. A second fit that lands on a grid
whose first row was already selected changed nothing the selection model
could see, so no selection signal fired: the result set was the new one while
the report still read "fitting…" and the plot still drew the first fit's
curves. Measured on nightly 160380b8c, before any of the plate, pooling,
selectivity or checkerboard work, so it is older than those.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from tests.qt.test_dose_response_screen import frame, screen  # noqa: F401

pytestmark = pytest.mark.qt


def _labels(screen):
    return [line.get_label() for line in screen._figure.axes[0].get_lines()
            if not line.get_label().startswith("_")]


def test_a_second_fit_replaces_the_report_and_the_plot(screen):
    screen.fit()
    assert _labels(screen) == ["geneA", "geneB", "geneD"]

    screen.group_picker.setCurrentIndex(0)
    screen.fit()

    assert [fit.group for fit in screen.result_set().fits] == [""]
    assert screen.report.toPlainText() != "fitting…"
    assert "geneA" not in screen.report.toPlainText()
    assert _labels(screen) == ["all rows"]


def test_refitting_the_same_choice_redraws_rather_than_keeping_fitting(screen):
    screen.fit()
    first = screen.report.toPlainText()

    screen.fit()

    assert screen.report.toPlainText() == first
