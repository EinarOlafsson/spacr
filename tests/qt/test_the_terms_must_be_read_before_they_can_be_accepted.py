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
    """The three data-use clauses, in the maintainer's own words."""

    def test_the_developers_may_use_the_data_to_develop_spacr(
            self, terms_module):
        said = terms_module.terms_text().lower()

        assert "use the data you send them to develop spacr" in said

    def test_uploaded_images_and_models_may_become_community_resources(
            self, terms_module):
        said = terms_module.terms_text().lower()

        assert "upload to a shared resource" in said
        assert "community resources" in said
        assert "object detection" in said

    def test_logs_and_metadata_may_be_used_to_make_it_better(
            self, terms_module):
        said = terms_module.terms_text().lower()

        assert "logs and the metadata" in said
        assert "make the software better" in said

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

        assert "does not publish your log" in said
        assert "names that path" in said


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


class TestTheAcceptanceIsGatedOnHavingReachedTheEnd:

    def test_it_starts_disabled(self, slides):
        assert slides.terms_were_read() is False
        assert slides._agree.isEnabled() is False

    def test_the_text_is_greyed_too(self, slides):
        """"the text should also be grayed out" -- one state on both halves,
        so the page reads as one thing waiting."""
        assert slides._dim_ink() in slides._terms_body.styleSheet()

    def test_the_greyed_switch_says_why(self, slides, terms_module):
        assert slides._scroll_hint.isVisibleTo(slides)
        assert terms_module.SCROLL_HINT[:30] in slides._scroll_hint.text()

    def test_reaching_the_end_enables_it(self, slides, qapp):
        _scroll_to_the_end(slides, qapp)

        assert slides.terms_were_read() is True
        assert slides._agree.isEnabled() is True

    def test_reaching_the_end_ungreys_the_text(self, slides, qapp):
        _scroll_to_the_end(slides, qapp)

        assert slides._terms_body.styleSheet() == ""
        assert not slides._scroll_hint.isVisibleTo(slides)

    def test_scrolling_back_up_does_not_un_read_them(self, slides, qapp):
        """A gate that closed behind the reader would take the acceptance
        away from somebody who had already earned it."""
        _scroll_to_the_end(slides, qapp)
        bar = slides._terms_scroll.verticalScrollBar()
        bar.setValue(0)
        qapp.processEvents()

        assert slides._agree.isEnabled() is True

    def test_partway_down_is_not_the_end(self, slides, qapp):
        bar = slides._terms_scroll.verticalScrollBar()
        bar.setValue(bar.maximum() // 2)
        qapp.processEvents()

        assert slides._agree.isEnabled() is False


class TestAViewportTallEnoughCountsAsRead:
    """"or a large monitor becomes a trap"."""

    def test_a_document_that_fits_needs_no_scrolling(self, slides, qapp):
        """The gate is "the end is on screen", not "a scroll bar moved".

        Shrinking the document is the same measurement as growing the
        viewport -- the scroll range is the difference between the two --
        and it is the half a test can reach on a headless screen.
        """
        assert slides._agree.isEnabled() is False
        slides._terms_body.setText("These terms fit on one line.")
        qapp.processEvents()

        assert slides._terms_scroll.verticalScrollBar().maximum() == 0
        assert slides.terms_were_read() is True
        assert slides._agree.isEnabled() is True


class TestAnUnmeasuredGateIsAClosedGate:

    def test_a_page_that_was_never_shown_has_not_been_read(self, qtbot):
        """"The end is on screen" cannot be true of a page that is not.

        An unshown scroll area still answers questions about its range, off
        geometry nothing laid out; believing that answer would open the gate
        for everybody, so it is not believed.
        """
        made = SetupSlides()
        qtbot.addWidget(made)

        assert made._terms_scroll.isVisible() is False
        assert made.terms_were_read() is False
        assert made._agree.isEnabled() is False

    def test_a_page_with_no_scroll_area_is_not_gated(self, slides):
        """The gate exists to prove the text was seen. With no widget to
        scroll there is nothing to prove and nothing to refuse."""
        slides._terms_read = False
        slides._terms_scroll = None

        assert slides.terms_were_read() is True


class TestTheRefusalNamesTheRightObstacle:

    def test_an_unread_document_is_told_to_be_scrolled(self, slides,
                                                       terms_module):
        """"tick the box above" is not actionable advice about a box that
        will not take a tick, so the reason it is greyed is said as well."""
        slides.next()

        said = slides._agree_note.text()
        assert terms_module.SCROLL_HINT[:30] in said
        assert terms_module.WHY_NOT_YET[:30] in said

    def test_a_read_but_unticked_document_is_told_only_to_tick(
            self, slides, qapp, terms_module):
        _scroll_to_the_end(slides, qapp)
        slides.next()

        said = slides._agree_note.text()
        assert terms_module.SCROLL_HINT[:30] not in said
        assert terms_module.WHY_NOT_YET[:30] in said

    def test_the_keyboard_is_sent_to_the_terms_not_to_the_dead_switch(
            self, slides):
        """A disabled switch cannot take focus, so a Next pressed before the
        end would leave the caret nowhere and Page Down would do nothing."""
        slides.next()

        assert slides.focusWidget() is slides._terms_scroll

    def test_the_keyboard_is_sent_to_the_switch_once_it_is_live(
            self, slides, qapp):
        _scroll_to_the_end(slides, qapp)
        slides.next()

        assert slides.focusWidget() is slides._agree

    def test_it_still_will_not_leave_the_slide(self, slides, qapp):
        _scroll_to_the_end(slides, qapp)

        assert slides.next() == TERMS_INDEX

    def test_a_read_and_ticked_document_moves_on(self, slides, qapp):
        _scroll_to_the_end(slides, qapp)
        slides._agree.setChecked(True)

        assert slides.next() == TERMS_INDEX + 1


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
