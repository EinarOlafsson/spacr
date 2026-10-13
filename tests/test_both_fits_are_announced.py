"""A run that fits twice says so, on the plot and in the header.

Instruction 147 A, reported 2026-08-18 after a glm run: "i can only see guides
only and it only runs once. i thought there would be 2 runs one with gene and
one with guide if i choose level=both".

BOTH FITS HAD RUN. Driven end to end at the time: resolve_levels('glm','both')
returns ('grna','gene'), results_grna.csv had 15 rows and results_gene.csv 5.
What was wrong is that the panel opened on guides -- correct, so a gene is not
drawn once per guide -- and NOTHING SAID the other half existed. "It only runs
once" is the honest reading from the user's side of that screen.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _frame(guides=True, genes=True, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        gene = f"{200000 + index}"
        if guides:
            for guide in range(3):
                rows.append({"feature": f"fraction:grna[{gene}_{guide}]",
                             "coefficient": float(rng.normal()),
                             "p_value": float(rng.uniform())})
        if genes:
            rows.append({"feature": f"gene_fraction:gene[{gene}]",
                         "coefficient": float(rng.normal()),
                         "p_value": float(rng.uniform())})
    return pd.DataFrame(rows)


def _panel(qtbot, frame):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame, source="results.csv")
    return panel


def test_the_note_names_the_level_shown_and_the_one_that_is_not(qtbot):
    panel = _panel(qtbot, _frame())
    note = panel.both_levels_note()

    assert panel.level() == "grna"
    assert "guides only" in note
    assert "60 of 80" in note
    assert "gene fit is in this run too" in note


def test_it_says_the_other_way_round_too(qtbot):
    panel = _panel(qtbot, _frame())
    panel.set_level("gene")
    note = panel.both_levels_note()

    assert "genes only" in note
    assert "20 of 80" in note
    assert "guide fit is in this run too" in note


def test_the_note_is_on_the_plot_not_only_in_the_header(qtbot):
    """A status line at the top is read once, on load. "Am I looking at guides
    or genes" is asked every time the user comes back to the tab, so the
    answer belongs beside the marks.

    READ OFF THE PLOT, not off a chosen slot. These two assertions used to
    name `volcano._note`, which is the CLICK slot, and that was the bug: the
    sentence was in the right widget and in the wrong drawer, so the first
    click erased it. What the requirement is about is that a reader looking
    at the dots can see it, so that is what is asserted -- and it is now
    carried by `level_note()`, whose slot no click can reach.
    """
    panel = _panel(qtbot, _frame())

    assert "gene fit is in this run too" in panel.volcano.level_note()
    assert "gene fit is in this run too" in panel.volcano._status.text()


def test_it_follows_the_level_without_being_asked(qtbot):
    panel = _panel(qtbot, _frame())
    panel.set_level("gene")

    assert "guide fit is in this run too" in panel.volcano.level_note()
    assert "guide fit is in this run too" in panel.volcano._status.text()


def test_the_header_carries_it_on_load(qtbot):
    panel = _panel(qtbot, _frame())
    assert "gene fit is in this run too" in panel._status


def test_a_one_level_table_says_nothing(qtbot):
    """A note that fires every time is a note nobody reads.

    A guide-only fit -- level='grna' -- has no gene half to point at, and
    saying so would be noise on every screen that ran one model on purpose.
    """
    panel = _panel(qtbot, _frame(genes=False))
    assert panel.both_levels_note() == ""
    assert "is in this run too" not in panel.volcano.level_note()
    assert "is in this run too" not in panel.volcano._status.text()


def test_the_whole_fit_says_nothing_either(qtbot):
    """Nothing is hidden, so there is nothing to announce."""
    panel = _panel(qtbot, _frame())
    panel.set_level(None)
    assert panel.both_levels_note() == ""


def test_the_note_does_not_clobber_a_diagnostics_own_numbers(qtbot):
    """`set_status_note` is how the Q-Q and the control panel carry the
    figures they exist for. Writing the level sentence over those would trade
    a panel's whole content for something the header already says."""
    panel = _panel(qtbot, _frame())
    for plot in (panel.qq, panel.controls, panel.agreement):
        assert "is in this run too" not in getattr(plot, "_note", "")
