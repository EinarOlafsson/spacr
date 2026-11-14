"""Table-driven tests of the settings-advisor proposal page and dialog.

The proposal page turns an :class:`~spacr.settings_advisor.Advice` into a
summary of the :class:`~spacr.settings_advisor.Reading` behind it, a table of
chosen values and a list of what stayed undecided.  The dialog switches
between the question page and the proposal, and Apply must only be offered
while a proposal is on screen.  One test per unit, walking a case table with
pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.settings_advisor import Advice, Choice, Reading, Undecided  # noqa: E402
from spacr.qt.widgets.settings_advisor_dialog import (  # noqa: E402
    ProposalPage, SettingsAdvisorDialog)

_DECIDED = "Every setting this advisor can decide was decided."


def test_the_proposal_page_summarises_the_reading_and_lists_the_choices(
        qtbot, subtests):
    """Summary text, table rows and undecided list, per kind of proposal."""

    diameter = Choice(key="cell_diameter", value=30,
                      why="typical for this magnification")
    cases = [
        ("a reading is summarised with its counts",
         Advice(chosen=(diameter,),
                reading=Reading(plates=2, wells=384, guides=1200, genes=300,
                                n_response=50_000)),
         {"cell_diameter": 12},
         "Read 2 plate(s), 384 well(s), 1,200 guide(s), 300 gene(s) and "
         "50,000 object row(s).",
         [["cell_diameter", "12", "30", "typical for this magnification"]],
         _DECIDED),
        ("a capped reading says where the response sample stopped",
         Advice(reading=Reading(plates=1, wells=96, guides=10, genes=5,
                                n_response=400_000, capped=True)),
         {},
         "Read 1 plate(s), 96 well(s), 10 guide(s), 5 gene(s) and 400,000 "
         "object row(s). The response was read from the first 400,000 "
         "object row(s); the score table is larger than the 400,000-row "
         "sample this reads.",
         [], _DECIDED),
        ("no reading leaves the summary empty but fills the table",
         Advice(chosen=(diameter,), reading=None), {"cell_diameter": 12}, "",
         [["cell_diameter", "12", "30", "typical for this magnification"]],
         _DECIDED),
        ("an unchanged value is marked so",
         Advice(chosen=(Choice("regression_type", "beta", "a share"),),
                undecided=(Undecided("inference", "no control guides"),),
                reading=Reading(plates=2, wells=96, guides=10, genes=5,
                                n_response=500)),
         {"regression_type": "beta"},
         "Read 2 plate(s), 96 well(s), 10 guide(s), 5 gene(s) and 500 "
         "object row(s).",
         [["regression_type", "beta", "beta  (unchanged)", "a share"]],
         "Not decided — left unchanged:\n  • inference: no control guides"),
    ]
    for case_id, advice, current, summary, rows, undecided in cases:
        with subtests.test(msg=case_id):
            page = ProposalPage()
            qtbot.addWidget(page)
            page.show_the_proposal(advice, current)
            for _ in range(2):            # first render, then a retranslation
                assert page.summary.text() == summary
                assert [[page.table.item(r, c).text() for c in range(4)]
                        for r in range(page.table.rowCount())] == rows
                assert page.undecided.toPlainText() == undecided
                page.retranslate_dynamic_content("en")


@pytest.fixture
def dialog(qtbot):
    """The advisor on a reading rich enough to produce a real proposal."""
    reading = Reading(plates=4, wells=384, guides=60, genes=20,
                      guides_per_gene=3.0, rows=4, columns=6, n_response=1000,
                      low=0.02, high=0.97, inside_unit=True, on_unit=True,
                      normal_p=1e-30, skew=1.4, response="pred")
    widget = SettingsAdvisorDialog(reading, {"regression_type": "mixed"})
    qtbot.addWidget(widget)
    return widget


def test_the_dialog_offers_apply_only_beside_the_proposal_it_shows(dialog):
    """Next shows the proposal and Apply; Back hides Apply again."""

    def buttons():
        return (dialog.back.isVisible(), dialog.next.isVisible(),
                dialog.apply.isVisible())

    assert dialog.advice() is None
    assert dialog.accepted_settings() == {}
    assert dialog.pages.currentWidget() is dialog.questions
    dialog.show()
    assert buttons() == (False, True, False)

    shown = dialog.show_the_proposal()
    assert dialog.advice() is shown
    assert dialog.accepted_settings() == shown.as_settings()
    assert dialog.accepted_settings()["inference"] == "nonparametric"
    assert dialog.pages.currentWidget() is dialog.proposal
    assert buttons() == (True, False, True)

    dialog.show_the_questions()
    assert dialog.pages.currentWidget() is dialog.questions
    assert buttons() == (False, True, False)
    assert dialog.advice() is shown
