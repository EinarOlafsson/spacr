"""The one guard in the results panel's wiring that its own plot list keeps shut.

The constructor walks :meth:`RegressionResultsPanel._keyed_plots` twice. The
second walk is the SELECT-MANY route -- a band or a modifier-click on a plot
selects every coefficient it encloses -- and it skips the two histograms,
because a bar stands for a hundred rows and "the rows behind this bar" is a
narrowing of the table, not a selection of a hundred coefficients. Without the
skip the gene tile would be showing one guide while the image tabs showed
another.

Nothing that ships can put the question to that skip: ``_keyed_plots()``
returns a literal tuple of the five point plots and no module in ``spacr``
subclasses the panel, so the guard is a promise the panel makes to itself.
This file makes the promise testable the only way it can be -- by handing the
constructor a plot list that breaks the rule -- and pins the difference the
guard makes: the same band of keys changes the selection when a point plot
emits it and leaves the selection alone when a histogram does.

The other nine lines and six arcs in this module's round-4 chunk are already
driven by ``tests/qt/test_cov_wf_qt_widgets_regression_results.py``; they are
not repeated here.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.regression_results import (                  # noqa: E402
    RegressionResultsPanel,
)

KEYS = ["fraction:grna[G0_1]", "fraction:grna[G0_2]",
        "fraction:grna[G0_3]", "fraction:grna[G0_4]"]


def _guides() -> pd.DataFrame:
    """Four guide coefficients -- enough rows that a narrowing shows."""
    return pd.DataFrame({"feature": KEYS,
                         "coefficient": [1.0, 2.0, 3.0, 4.0],
                         "p_value": [0.01, 0.02, 0.3, 0.4]})


def _visible_features(panel) -> list:
    """The feature on every row the coefficient table is still showing."""
    table = panel.table.table
    headers = [table.horizontalHeaderItem(column).text()
               for column in range(table.columnCount())]
    column = headers.index("feature")
    return [table.item(row, column).text()
            for row in range(table.rowCount())
            if not table.isRowHidden(row) and table.item(row, column)]


@pytest.mark.parametrize("histogram", ["p_values", "effect_distribution"])
def test_a_histogram_offered_as_a_keyed_plot_still_only_narrows(
        qtbot, histogram):
    """A band on a point plot selects the coefficients it encloses; the same
    band on a histogram narrows the table to them and touches nothing else.
    The constructor keeps those two routes apart by skipping the histograms
    while it wires the select-many route, and a panel that offers a histogram
    as a keyed plot is the only thing that can ask whether the skip works --
    it is also exactly the mistake the skip exists to survive, because a bar
    of forty rows announcing itself as a selection of forty coefficients is
    how the gene tile ends up showing a guide nobody clicked."""

    class _PanelThatListsAHistogram(RegressionResultsPanel):
        # `_keyed_plots` is private, and overriding it is the only way in:
        # the guard asks whether a plot THE PANEL ITSELF listed is one of the
        # two histograms, and the shipped `_keyed_plots` returns a literal
        # tuple holding neither, so no public call can make the question
        # interesting.
        def _keyed_plots(self):
            return super()._keyed_plots() + (getattr(self, histogram),)

    panel = _PanelThatListsAHistogram()
    qtbot.addWidget(panel)
    bars = getattr(panel, histogram)
    assert bars in panel._keyed_plots(), (
        "the panel under test is not actually offering the histogram as a "
        "keyed plot, so the guard was never asked anything")

    panel.set_frame(_guides(), "/runs/guides/results.csv")
    assert _visible_features(panel) == KEYS, "the run opens showing every row"

    # PRESENT: a point plot's band is a selection, and the panel reports it.
    panel.volcano.keys_selected.emit(KEYS[:2])
    assert panel.selected_keys() == KEYS[:2], (
        "a band on the volcano must select every coefficient it encloses")

    # ABSENT: the histogram's band reaches the table as a narrowing only. If
    # the constructor had wired it to the select-many route as well, this
    # emission would have replaced the selection with its own three keys.
    bars.keys_selected.emit(KEYS[:3])

    assert _visible_features(panel) == KEYS[:3], (
        "the bar's rows were never brought to the front, so the histogram "
        "reached neither route")
    assert panel.selected_keys() == KEYS[:2], (
        "the histogram's bar was taken as a selection of three coefficients, "
        "so a bar and a dot now mean the same thing to every linked view")
