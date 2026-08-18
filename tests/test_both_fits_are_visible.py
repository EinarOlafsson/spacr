"""Half a run may not be invisible behind a right-click nobody performed.

Instruction 147 A.

    "i jsut ran glm and gota interactive colcano plot, i can only see guides
     only and it only runs once. i thought there would be 2 runs one with
     gene and one with guide if i choose level=both. fix this."

BOTH FITS DO RUN. Reproduced end to end on 2026-08-18: `resolve_levels('glm',
'both')` gives ('grna', 'gene'), and the run writes results_grna.csv with 15
rows, results_gene.csv with 5, and a combined results.csv with 22. The backend
is right and the volcano's guides-only default is right -- it is what stops a
gene being drawn once per guide, which was its own report.

WHAT WAS WRONG IS THAT NOTHING SAID SO. The level lived on a right-click menu
and nowhere else, so the panel opened on 16 of 22 coefficients with no mark on
screen that the other six existed. "It only runs once" is the honest reading
from the user's side, and the fix is discoverability rather than reverting the
default.

So the level control is ON the plot, above the axes, showing the current level
with its count -- and it carries the sentence naming the fit that is not being
drawn, which the host supplies because only the host knows the run.
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


def _frame(n: int = 22) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}_1]" for i in range(n)],
        "coefficient": rng.normal(0, 0.5, n),
        "p_value": rng.uniform(0.001, 0.99, n),
    })


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame())
    plot.set_status("22 coefficients.")
    return plot


def _levels(chosen: str = "grna"):
    """What a host offers: the three levels with their counts."""
    picked = []
    return picked, [
        ("genes and guides (22)", lambda: picked.append(None), chosen is None),
        ("guides only (16)", lambda: picked.append("grna"), chosen == "grna"),
        ("genes only (6)", lambda: picked.append("gene"), chosen == "gene"),
    ]


def _combo(plot):
    from PySide6.QtWidgets import QComboBox

    boxes = plot.findChildren(QComboBox)
    assert boxes, "the plot has no level control on it at all"
    return boxes[0]


# --------------------------------------------------------------------------- #
#  The control is on the plot, not three clicks into a menu
# --------------------------------------------------------------------------- #

def test_a_plot_offered_levels_shows_a_control_without_a_right_click(volcano):
    _picked, options = _levels()

    volcano.offer_levels(options)

    combo = _combo(volcano)
    assert combo.isVisibleTo(volcano)
    assert [combo.itemText(i) for i in range(combo.count())] == [
        label for label, _callback, _checked in options]


def test_the_control_shows_the_level_that_is_actually_drawn(volcano):
    """A filter the user cannot see the state of is one they read a subset
    through and believe is the whole screen."""
    _picked, options = _levels(chosen="gene")

    volcano.offer_levels(options)

    assert _combo(volcano).currentText() == "genes only (6)"


def test_the_counts_are_on_the_control_so_a_subset_is_visibly_a_subset(volcano):
    _picked, options = _levels()

    volcano.offer_levels(options)

    combo = _combo(volcano)
    assert "16" in combo.currentText(), combo.currentText()
    assert any("6" in combo.itemText(i) for i in range(combo.count()))


def test_choosing_a_level_on_the_control_calls_the_hosts_callback(volcano):
    """Driving the widget proves it is wired, not merely populated."""
    picked, options = _levels()
    volcano.offer_levels(options)
    combo = _combo(volcano)

    combo.setCurrentIndex(2)
    combo.activated.emit(2)

    assert picked == ["gene"]


def test_a_plot_nobody_offered_levels_to_shows_nothing(qtbot):
    """A Q-Q of a simulation has no gene/guide split to make, and a control
    that is always there is one the reader has to learn to ignore."""
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.widgets.fast_plots import QQPlot

    plot = QQPlot()
    qtbot.addWidget(plot)
    plot.set_p_values(np.random.default_rng(0).random(50))

    assert not [box for box in plot.findChildren(QComboBox)
                if box.isVisibleTo(plot)]


def test_offering_no_levels_again_takes_the_control_away(volcano):
    _picked, options = _levels()
    volcano.offer_levels(options)

    volcano.offer_levels([])

    assert not _combo(volcano).isVisibleTo(volcano)


# --------------------------------------------------------------------------- #
#  And it says what is NOT being drawn
# --------------------------------------------------------------------------- #

NOTE = ("guides: 16 of 22 coefficients. The gene fit is in this run too — "
        "switch with Level.")


def test_the_note_names_the_fit_that_is_not_on_screen(volcano):
    _picked, options = _levels()

    volcano.offer_levels(options, note=NOTE)

    assert volcano.level_note() == NOTE
    assert NOTE in volcano._status.text()


def test_the_note_is_beside_the_control_as_well_as_in_the_status_line(volcano):
    """"on the plot and in the status line". A status line is under the axes
    and is rewritten by whatever was last clicked."""
    from PySide6.QtWidgets import QLabel

    _picked, options = _levels()

    volcano.offer_levels(options, note=NOTE)

    said = [label.text() for label in volcano.findChildren(QLabel)
            if label.isVisibleTo(volcano)]
    assert NOTE in said, said


def test_the_note_survives_a_redraw(volcano):
    """`set_results` rewrites the headline. A sentence about the run that
    vanished the first time the plot redrew would be no notice at all."""
    _picked, options = _levels()
    volcano.offer_levels(options, note=NOTE)

    volcano.set_results(_frame())
    volcano.set_status("16 coefficients.")

    assert NOTE in volcano._status.text()


def test_the_note_survives_a_click(volcano):
    _picked, options = _levels()
    volcano.offer_levels(options, note=NOTE)

    volcano.set_status_note("g3   coefficient=0.4")

    said = volcano._status.text()
    assert NOTE in said, said
    assert "g3" in said, said


def test_a_run_with_one_level_says_nothing_extra(volcano):
    """A note that fires every time is a note nobody reads."""
    from PySide6.QtWidgets import QLabel

    _picked, options = _levels()

    volcano.offer_levels(options)

    assert volcano.level_note() == ""
    assert "" == "".join(label.text() for label in volcano.findChildren(QLabel)
                         if label.isVisibleTo(volcano)
                         and label.text().startswith("guides:"))


# --------------------------------------------------------------------------- #
#  Against the real panel
# --------------------------------------------------------------------------- #

def test_the_real_panel_puts_the_level_on_the_volcano(qtbot):
    """The panel offers three levels with their counts, so the control it
    grows is the one a user of the regression screen actually sees."""
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    frame = _frame(24)
    frame.loc[:5, "feature"] = [f"gene_fraction:gene[g{i}]" for i in range(6)]
    panel.set_frame(frame)

    boxes = [box for box in panel.volcano.findChildren(QComboBox)]
    assert boxes, "the volcano on the real panel has no level control"
    assert boxes[0].currentText().startswith("guides only"), (
        boxes[0].currentText())
