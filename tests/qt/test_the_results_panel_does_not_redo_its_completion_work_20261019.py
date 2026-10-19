"""Lesson 13's results panel does its completion work once.

When the run ends the panel is handed the same summary text more than once,
and every hand-off rebuilt every folded section; the guide/gene split of the
coefficient table was re-derived for each of the seven notes and menus that
ask for its counts. Both now happen once per new input. Counted, not timed.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QTextBrowser  # noqa: E402

from spacr.qt.widgets import folding_summary  # noqa: E402
from spacr.qt.widgets.folding_summary import FoldingSummaryView  # noqa: E402

pytestmark = pytest.mark.qt

SUMMARY = """THE ANSWER
----------
  effect                 0.42 (95% CI 0.31 to 0.53)
  wells                  1536

DETAILS
-------
  type                   ols
"""


def test_the_same_summary_is_not_rebuilt(qtbot, monkeypatch):
    view = FoldingSummaryView()
    qtbot.addWidget(view)
    rebuilds = []
    real = FoldingSummaryView._rebuild
    monkeypatch.setattr(FoldingSummaryView, "_rebuild",
                        lambda self: (rebuilds.append(1), real(self)))

    view.setPlainText(SUMMARY)
    first = view.findChildren(QTextBrowser)
    view.setPlainText(SUMMARY)
    view.setPlainText(SUMMARY)

    assert rebuilds == [1]
    assert view.findChildren(QTextBrowser) == first
    assert view.toPlainText() == SUMMARY


def test_a_different_summary_is_rebuilt(qtbot):
    view = FoldingSummaryView()
    qtbot.addWidget(view)
    view.setPlainText(SUMMARY)
    view.set_section_expanded("DETAILS", True)
    changed = SUMMARY.replace("1536", "768")

    view.setPlainText(changed)

    assert view.toPlainText() == changed
    assert "768" in "".join(b.toHtml() for b in view.findChildren(QTextBrowser))
    view.setPlainText("")
    assert view.toPlainText() == ""
    assert view.section_titles() == ()


def test_the_level_split_is_derived_once_per_table(qtbot, monkeypatch):
    from spacr import hits
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    calls = []
    real = hits.coefficient_levels
    monkeypatch.setattr(hits, "coefficient_levels",
                        lambda frame: (calls.append(1), real(frame))[1])
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    frame = pd.DataFrame({
        "feature": ["gene_fraction:gene[1]", "grna_fraction:grna[a]",
                    "grna_fraction:grna[b]"],
        "coefficient": [0.1, 0.2, 0.3], "p_value": [0.01, 0.2, 0.3]})
    panel._frame = frame
    calls.clear()

    first = panel.level_counts()
    for _ in range(6):
        assert panel.level_counts() == first
    assert len(calls) == 1

    other = frame.iloc[:2].copy()
    panel._frame = other
    assert panel.level_counts()[None] == 2
    assert len(calls) == 2
