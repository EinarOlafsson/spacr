"""The Dose-Response screen divides a host EC50 by a response EC50.

`selectivity_index` has existed in the engine; the screen could not reach it,
so the one number that decides whether an anti-parasitic compound is worth
anything had to be computed by hand from two fits. These tests are about the
trip from the Host response picker to the engine and back.

Two compounds, each killing the parasite at 1 uM::

    geneA   host EC50 10 uM         -> index near 10, bounded both ways
    geneB   the host never dies     -> no index, and the report says why
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import (
    STATUS_FITTED, STATUS_REFUSED, four_parameter_logistic,
)

pytestmark = pytest.mark.qt

DOSES = 243.0 / 3.0 ** np.arange(10)
REPLICATES = 3


def _kill(ec50, rng, dose):
    """Percent killed at each dose, a clean 4PL rising 0 -> 100 with noise."""
    if ec50 is None:
        return rng.normal(0.0, 1.0, dose.size)
    clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(ec50), 1.0)
    return clean + rng.normal(0.0, 1.0, dose.size)


@pytest.fixture()
def frame():
    rng = np.random.default_rng(20260915)
    dose = np.repeat(DOSES, REPLICATES)
    parts = []
    for gene, host_ec50 in (("geneA", 10.0), ("geneB", None)):
        parts.append(pd.DataFrame({
            "gene": gene, "conc_uM": dose,
            "parasite_killed": _kill(1.0, rng, dose),
            "host_killed": _kill(host_ec50, rng, dose)}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture()
def screen(qtbot, frame):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="two compounds")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("parasite_killed")
    widget.group_picker.setCurrentText("gene")
    return widget


def _items(picker):
    return [picker.itemText(i) for i in range(picker.count())]


def test_the_host_picker_starts_unchosen_and_offers_the_readouts(screen):
    """Nothing is computed until a host readout is chosen."""
    assert screen.host_picker.currentText() == "(none)"
    assert {"host_killed", "parasite_killed"} <= set(_items(screen.host_picker))

    screen.fit()

    assert screen._selectivity == {}
    assert "selectivity index" not in screen.report.toPlainText()


def test_the_host_ec50_over_the_response_ec50_is_the_index(screen):
    """Host 10 uM over parasite 1 uM is a selectivity index near 10."""
    screen.host_picker.setCurrentText("host_killed")

    screen.fit()

    index = screen._selectivity["geneA"]
    assert index.status == STATUS_FITTED, index.note
    assert 5.0 < index.index < 20.0, index.index
    assert index.index_low < index.index < index.index_high
    assert "geneA: selectivity index " in screen.report.toPlainText()


def test_a_host_that_never_dies_gives_no_index_and_says_why(screen):
    """No host curve means no ratio -- refused, not reported as infinite."""
    screen.host_picker.setCurrentText("host_killed")

    screen.fit()

    index = screen._selectivity["geneB"]
    assert index.status != STATUS_FITTED
    assert index.index is None
    line = next(line for line in screen.report.toPlainText().splitlines()
                if line.startswith("geneB: "))
    assert "host" in line
