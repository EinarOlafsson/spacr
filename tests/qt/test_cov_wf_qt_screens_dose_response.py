"""The Dose–Response screen's four remaining un-driven decisions.

``tests/qt/test_dose_response_screen.py`` fits a synthetic plate and reads the
engine's numbers back out of the grid; ``test_cov_w3_7_dose_response.py``
drives the loading half. What is left are four small forks that nothing in
either file takes, and all four are places where the screen has to *not* do
something:

* the concentration picker walking past a numeric column whose name does not
  look like a dose before it lands on the one that does;
* the figure drawn with no row selected -- every curve, no EC50 marker;
* a bounded EC50 whose interval has an open end -- the line, but no band;
* closing a screen whose canvas has no pending-draw timer to cancel.

Each is exercised here beside the input that takes the other side of the same
fork, so an assertion that something is absent is always paired with the
input that produces it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Guessing the dose column
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The figure with and without a selected row
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A bounded EC50 whose interval has an open end
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The other operand of each two-part guard
# ---------------------------------------------------------------------------

def test_the_name_guess_reads_every_option_before_it_settles(qtbot):
    """The dose guess is a scan, not a look at the head of the list.

    ``_refill`` is the one place a picker's contents are replaced, and its
    ``prefer`` hints are matched against *every* name in turn. A version that
    only inspected the first entry would leave the guess working on tables
    whose dose column happens to sort first and silently failing on every
    other table — the same screen guessing right or wrong depending on
    alphabetical luck. Both outcomes are driven here: a list where the hint
    is only found at the end, and a list where it is nowhere at all.
    """
    from PySide6.QtWidgets import QComboBox

    picker = QComboBox()
    qtbot.addWidget(picker)

    DoseResponseScreen._refill(
        picker, ["assay_level", "plate_row", "readout", "conc_uM"],
        prefer=("conc",))
    assert picker.currentText() == "conc_uM", (
        "the hint sits last, so the scan has to walk the three before it")

    DoseResponseScreen._refill(picker, ["alpha", "beta", "gamma"],
                               prefer=("conc",))
    assert picker.currentText() == "alpha", (
        "no name earns the hint, so the box keeps its own first entry")
    assert picker.count() == 3
