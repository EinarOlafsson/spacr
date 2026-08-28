"""The terms are long enough to be terms, and the acceptance is gated on them.

Two claims, and they are one mechanism:

* THE DOCUMENT IS LONG. A four-sentence summary is read without noticing,
  and the clauses that matter -- what the developers may do with what a
  user sends them -- are exactly the clauses a summary drops. They are
  written out, in sections, in the words the maintainer asked for;
* THE ACCEPTANCE IS DISABLED UNTIL THE END OF IT HAS BEEN ON SCREEN, and
  the text is greyed with it. The enabling IS the evidence that the text
  went past the reader: nothing else on the screen can tell whether it did.

A viewport tall enough to hold the whole document counts as read. The gate
asks "is the end on screen", not "did a scroll bar move", or the reader on
a large monitor is locked out of a licence they can see all of.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.setup_slides import (SLIDES, SetupSlides,  # noqa: E402
                                           TERMS_SLIDE)

pytestmark = pytest.mark.qt

TERMS_INDEX = [title for title, _b, _k in SLIDES].index(TERMS_SLIDE)


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A settings store of this test's own, so nothing is accepted for real."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences, terms

    importlib.reload(preferences)
    importlib.reload(terms)
    yield
    importlib.reload(preferences)
    importlib.reload(terms)


@pytest.fixture
def terms_module():
    from spacr.qt import terms

    return terms


@pytest.fixture
def slides(qtbot, qapp):
    """A setup screen showing the terms, LAID OUT.

    The gate measures a scroll area, and a scroll area that has never been
    given a size has no answer to give -- so this shows the window and lets
    the event loop run before any test looks at it.
    """
    made = SetupSlides()
    qtbot.addWidget(made)
    made.show()
    made._show_slide(TERMS_INDEX)
    qapp.processEvents()
    return made


def _scroll_to_the_end(made, qapp) -> None:
    bar = made._terms_scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    qapp.processEvents()


class TestTheTermsAreLongEnoughToBeTerms:

    def test_they_take_more_than_a_glance(self, terms_module):
        """"there should be much more text, so the user has to scroll down".

        The number is a floor, not a target: what it rules out is the
        four-paragraph summary this replaced, which fitted in the card with
        room to spare.
        """
        assert len(terms_module.terms_text()) > 3000

    def test_they_are_written_in_sections(self, terms_module):
        """A wall of prose is scrolled past; sections are navigated."""
        headings = [point for point in terms_module.TERMS
                    if point.strip() == point.strip().upper()]
        assert len(headings) >= 6

    def test_they_do_not_fit_the_card_they_are_shown_in(self, slides):
        """The scrolling is the point, so it is asserted on the real page."""
        bar = slides._terms_scroll.verticalScrollBar()

        assert bar.maximum() > 0


class TestTheTermsSayWhatTheMaintainerAsked:
    """The three data-use clauses survive the rewrite.

    These assert SUBSTANCE, not the sentences the first draft happened to
    use. That draft explained itself in an essayist's voice and was
    rewritten into the licence-agreement form a reader already knows how
    to read, so the phrasing moved; what may be done with what you send is
    exactly what must not move, which is why this class exists.
    """

    def test_the_developers_may_use_the_data_to_develop_spacr(
            self, terms_module):
        said = terms_module.terms_text().lower()

        assert "grant the licensor" in said
        assert "develop the software further" in said

    def test_uploaded_images_and_models_may_become_community_resources(
            self, terms_module):
        said = terms_module.terms_text().lower()

        assert "shared resource" in said
        assert "community resources" in said
        assert "object detection" in said

    def test_logs_and_metadata_may_be_used_to_make_it_better(
            self, terms_module):
        said = terms_module.terms_text().lower()

        assert "diagnostic data" in said
        assert "improving the software" in said
        # The grant has to be open-ended, not confined to fixing that bug.
        assert "any other purpose connected with its development" in said

    def test_the_permission_names_the_report_control_it_covers(
            self, terms_module):
        """"if the user dosnt unclick include logs in report, i can use those
        logs" -- a permission nobody was told about is not a permission, so
        the setting is named where the permission is granted."""
        assert "Include recent logs in a report" in terms_module.terms_text()

    def test_they_say_the_log_is_never_published(self, terms_module):
        """The issue can be public; the log never is. The terms grant use of
        logs that are SENT, and say so rather than leaving the reader to
        assume the tracker gets them."""
        said = terms_module.terms_text().lower()

        assert "diagnostic data is not published" in said
        assert "records the path rather than the contents" in said

    def test_they_read_as_an_agreement_rather_than_an_essay(self,
                                                            terms_module):
        """The form is the one Apple, Microsoft and Google all use.

        A document that talks ABOUT terms leaves the reader unsure they
        agreed to anything. This one opens by saying what accepting means,
        defines what it is talking about, and sets the two sections that
        are always set in capitals in capitals.
        """
        said = terms_module.terms_text()

        assert "END USER LICENCE AGREEMENT" in said
        assert "PLEASE READ THIS AGREEMENT CAREFULLY" in said
        assert "1. DEFINITIONS" in said
        assert "DISCLAIMER OF WARRANTIES" in said
        assert "LIMITATION OF LIABILITY" in said
        assert "PROVIDED \u201cAS IS\u201d" in said


class TestRewrittenTermsAskAgain:

    def test_the_version_moved_past_the_one_that_did_not_say_this(
            self, terms_module):
        """A profile that accepted 1.0 accepted a document with no data-use
        clause in it, so it has not accepted this one."""
        assert terms_module.TERMS_VERSION != "1.0"

    def test_a_profile_that_accepted_the_old_wording_is_asked_again(
            self, terms_module):
        terms_module.record_agreement("1.0")

        assert terms_module.needs_agreement() is True


class TestTheChromeIsTranslatedAndTheDocumentIsNot:

    def test_the_hint_is_in_every_catalog(self, terms_module):
        from spacr.qt import i18n

        terms_module.register_translations()

        assert i18n.has_translation(terms_module.SCROLL_HINT)
        assert i18n.tr(terms_module.SCROLL_HINT, "de") != \
            terms_module.SCROLL_HINT

    def test_the_slide_blurb_still_matches_the_catalogued_one(
            self, terms_module):
        """The blurb changed with the terms, and a row keyed on the old
        wording would translate nothing."""
        from spacr.qt import i18n

        terms_module.register_translations()
        blurb = SLIDES[TERMS_INDEX][1]

        assert i18n.has_translation(blurb)
        assert i18n.tr(blurb, "sv") != blurb

    def test_the_terms_themselves_are_not_catalogued(self, terms_module):
        """A translated licence summary is not the licence."""
        from spacr.qt import i18n

        assert not i18n.has_translation(terms_module.TERMS[1])


class TestTheGateIsOpen:
    """The acceptance is a choice, not a scrolling exercise.

    Asked for 2026-08-28: "change it so the terms of service dont need to be
    scrolled through in the startup window." These replace the class that
    asserted the opposite -- a scroll-gated switch, greyed text and a hint
    telling the reader to keep dragging.
    """

    def test_the_switch_is_live_from_the_moment_the_page_opens(self, slides):
        """Nothing has been scrolled, and the acceptance is still offered."""
        assert slides._agree.isEnabled() is True

    def test_the_text_is_not_greyed(self, slides):
        """A greyed page said "wait"; there is nothing to wait for now."""
        body = getattr(slides, "_terms_body", None)
        if body is not None:
            assert "color:" not in (body.styleSheet() or "")

    def test_the_scroll_hint_is_not_shown(self, slides):
        """"Scroll to the end" would now be an instruction to do nothing."""
        hint = getattr(slides, "_scroll_hint", None)
        if hint is not None:
            assert hint.isVisible() is False

    def test_a_page_that_was_never_shown_is_still_acceptable(self, slides):
        """The old gate answered False for an unlaid-out page."""
        assert slides.terms_were_read() is True

    def test_the_document_is_still_there_in_full(self, terms_module):
        """Removing the gate must not shorten what is being agreed to."""
        assert len(terms_module.terms_text()) > 2000

    def test_accepting_is_still_recorded(self, terms_module):
        """The agreement is unchanged: only the greying is gone."""
        terms_module.record_agreement(terms_module.TERMS_VERSION)
        assert terms_module.agreed_version() == terms_module.TERMS_VERSION
        assert terms_module.needs_agreement() is False
