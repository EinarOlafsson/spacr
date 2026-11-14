"""The advisor window between its two pages, and before it has anything to say.

The proposal page is rendered again on every language change, which can happen
at any moment -- including before a proposal exists.
"""
from __future__ import annotations

from spacr.settings_advisor import Advice, Choice, Reading, Undecided
from spacr.qt.widgets.settings_advisor_dialog import ProposalPage


def test_a_language_change_before_any_proposal_changes_nothing(qtbot):
    """Retranslating an empty proposal page is a no-op, not a crash.

    The language can be changed while the questions are still on screen. The
    page has no advice to re-render then, and reaching for the row it has not
    built yet would take the window down.
    """
    page = ProposalPage()
    qtbot.addWidget(page)
    before = page.table.rowCount()

    page.retranslate_dynamic_content("fr")

    assert page.table.rowCount() == before == 0
    assert page.undecided.toPlainText() == ""


def test_a_language_change_rebuilds_the_rows_it_already_showed(qtbot):
    """Retranslating re-renders the table from the advice it is holding."""
    page = ProposalPage()
    qtbot.addWidget(page)
    advice = Advice(
        chosen=(Choice("regression_type", "beta", "measured"),),
        undecided=(Undecided("inference", "no control guides were named"),),
        reading=Reading(plates=2, wells=96, guides=10, genes=5,
                        n_response=500))
    page.show_the_proposal(advice, {})

    page.retranslate_dynamic_content("en")

    assert page.table.rowCount() == 1
    assert page.table.item(0, 0).text() == "regression_type"
    assert "inference" in page.undecided.toPlainText()
    assert "2 plate(s)" in page.summary.text()
