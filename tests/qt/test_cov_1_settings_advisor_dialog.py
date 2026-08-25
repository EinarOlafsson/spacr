"""The advisor window between its two pages, and before it has anything to say.

The proposal page is rendered again on every language change, which can happen
at any moment -- including before a proposal exists and after one that decided
everything. Going Back to the questions has to put the buttons back too: a
window offering Apply on the question page would write settings the user never
saw a proposal for.
"""
from __future__ import annotations

import pytest

from spacr.settings_advisor import Advice, Choice, Reading, Undecided
from spacr.qt.widgets.settings_advisor_dialog import (ProposalPage,
                                                      SettingsAdvisorDialog)


@pytest.fixture
def dialog(qtbot):
    """The advisor on a reading rich enough to produce a real proposal."""
    reading = Reading(plates=4, wells=384, guides=60, genes=20,
                      guides_per_gene=3.0, rows=4, columns=6,
                      n_response=1000, low=0.02, high=0.97,
                      inside_unit=True, on_unit=True, normal_p=1e-30,
                      skew=1.4, response="pred")
    widget = SettingsAdvisorDialog(reading, {"regression_type": "mixed"})
    qtbot.addWidget(widget)
    return widget


def test_going_back_takes_apply_away_again(dialog):
    """Back returns to the questions with Apply hidden and Next offered.

    Apply writes ``accepted_settings()``. Leaving it on the question page
    would let a user change an answer and then apply the proposal computed
    from the previous one.
    """
    dialog.show()
    dialog.show_the_proposal()
    assert dialog.apply.isVisible()

    dialog.show_the_questions()

    assert dialog.pages.currentWidget() is dialog.questions
    assert dialog.apply.isVisible() is False
    assert dialog.back.isVisible() is False
    assert dialog.next.isVisible() is True


def test_the_dialog_hands_back_the_proposal_it_showed(dialog):
    """``advice()`` is the same object the table was rendered from.

    The caller applies ``accepted_settings()`` but reports the reasons from
    ``advice()``; two different objects would let the window explain one
    proposal while writing another.
    """
    assert dialog.advice() is None

    shown = dialog.show_the_proposal()

    assert dialog.advice() is shown
    assert dialog.accepted_settings() == shown.as_settings()


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


def test_a_proposal_that_decided_everything_says_so(qtbot):
    """No undecided settings gets its own sentence, not an empty box.

    An empty panel under the table reads as a rendering failure. The advisor
    says it decided everything so the absence is an answer.
    """
    page = ProposalPage()
    qtbot.addWidget(page)
    advice = Advice(chosen=(Choice("regression_type", "beta",
                                   "the response is a proportion inside "
                                   "(0, 1) on every plate"),),
                    undecided=(),
                    reading=Reading(plates=1, wells=96))

    page.show_the_proposal(advice, {"regression_type": "mixed"})

    said = page.undecided.toPlainText()
    assert "decided" in said.lower()
    assert "•" not in said, "nothing should be listed as left unchanged"
    assert page.table.rowCount() == 1

    # And the same page, re-rendered in another language, keeps saying it.
    page.retranslate_dynamic_content("en")
    assert page.undecided.toPlainText() == said


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
