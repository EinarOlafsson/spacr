"""Setup asks for the terms of use, and the answer is recorded with its version.

spaCR is distributed under a licence whose first clause is that the licence
exists only once its terms are agreed to, so acceptance is a condition of use
rather than a courtesy. That makes three things load-bearing:

* it is ASKED -- on the way in, on a slide of its own;
* it is RECORDED -- an agreement nobody can look up afterwards is not one,
  and it is kept beside the other setup answers;
* the VERSION is recorded with it, so terms rewritten later are a new
  agreement instead of one inherited silently.

And one thing that must not happen: the Next button on that slide is not
greyed with nothing beside it. A control that does nothing and says nothing
leaves the reader to guess which of the things on the page is stopping them.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox                    # noqa: E402

from spacr.qt.widgets.setup_slides import (SLIDES, SetupSlides,  # noqa: E402
                                           TERMS_SLIDE,
                                           open_setup_if_needed)

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A settings store of this test's own.

    Without it the test accepts the terms on the user's machine, and they are
    never asked -- which is both a wrong record and an untestable one.
    """
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
def slides(qtbot):
    made = SetupSlides()
    qtbot.addWidget(made)
    made._show_slide(TERMS_INDEX)
    return made


TERMS_INDEX = [title for title, _b, _k in SLIDES].index(TERMS_SLIDE)


class TestTheTermsAreAsked:

    def test_there_is_a_slide_for_them(self):
        assert TERMS_SLIDE in [title for title, _b, _k in SLIDES]

    def test_it_comes_before_the_closing_slide(self):
        """The last slide says the setup is done; the terms decide whether."""
        assert TERMS_INDEX == len(SLIDES) - 2

    def test_the_slide_carries_the_terms_themselves(self, slides, terms_module):
        page = slides._pages.widget(TERMS_INDEX)
        shown = " ".join(label.text() for label in page.findChildren(type(
            slides._blurb)))

        for point in terms_module.TERMS:
            assert point[:40] in shown

    def test_the_slide_names_the_licence_and_links_to_it(self, slides,
                                                         terms_module):
        page = slides._pages.widget(TERMS_INDEX)
        shown = " ".join(label.text() for label in page.findChildren(type(
            slides._blurb)))

        assert terms_module.LICENSE_NAME in shown
        assert terms_module.LICENSE_URL in shown

    def test_the_agreement_is_a_control_the_user_operates(self, slides):
        page = slides._pages.widget(TERMS_INDEX)

        assert page.findChildren(QCheckBox)


class TestTheNextButtonExplainsItself:

    def test_it_is_not_disabled(self, slides):
        """A dead control says neither what is missing nor where to look."""
        assert slides._next.isEnabled()

    def test_pressing_it_unagreed_stays_on_the_slide(self, slides):
        assert slides.next() == TERMS_INDEX
        assert slides.slide() == TERMS_INDEX

    def test_pressing_it_unagreed_says_what_is_missing(self, slides,
                                                       terms_module):
        slides.next()

        assert not slides._agree_note.isHidden()
        assert terms_module.WHY_NOT_YET[:40] in slides._agree_note.text()

    def test_agreeing_drops_the_complaint(self, slides):
        slides.next()
        slides._agree.setChecked(True)

        assert slides._agree_note.isHidden()

    def test_agreeing_lets_the_screen_finish(self, slides):
        slides._agree.setChecked(True)

        assert slides.next() == TERMS_INDEX + 1


class TestTheAgreementIsRecorded:

    def test_a_fresh_profile_has_agreed_to_nothing(self, terms_module):
        assert terms_module.agreed_version() == ""
        assert terms_module.needs_agreement() is True

    def test_finishing_the_screen_records_the_version(self, slides,
                                                      terms_module):
        slides._agree.setChecked(True)
        slides.accept()

        assert terms_module.agreed_version() == terms_module.TERMS_VERSION
        assert terms_module.needs_agreement() is False

    def test_the_record_says_when(self, slides, terms_module):
        slides._agree.setChecked(True)
        slides.accept()

        assert terms_module.agreed_at()
        assert terms_module.agreement_record()["accepted_at"] == \
            terms_module.agreed_at()

    def test_the_record_can_be_looked_up_whole(self, slides, terms_module):
        slides._agree.setChecked(True)
        slides.accept()
        record = terms_module.agreement_record()

        assert record["version"] == terms_module.TERMS_VERSION
        assert record["license"] == terms_module.LICENSE_NAME

    def test_closing_the_screen_records_nothing(self, slides, terms_module):
        """A dismissal chooses the defaults; it does not accept a licence."""
        slides.reject()

        assert terms_module.agreed_version() == ""

    def test_a_store_that_refuses_the_write_asks_again(self, slides,
                                                       terms_module,
                                                       monkeypatch):
        """An unwritten acceptance must never be treated as given."""
        def boom(*_args, **_kwargs):
            raise OSError("the settings file is read-only")

        monkeypatch.setattr(terms_module, "record_agreement", boom)
        slides._agree.setChecked(True)
        slides.accept()

        assert terms_module.needs_agreement() is True

    def test_an_unreadable_store_has_not_agreed(self, terms_module,
                                                monkeypatch):
        def boom():
            raise OSError("the settings file cannot be read")

        monkeypatch.setattr(terms_module, "_settings", boom)

        assert terms_module.agreed_version() == ""
        assert terms_module.agreed_at() == ""


class TestRewrittenTermsAreANewAgreement:

    def test_a_different_version_needs_agreeing_again(self, terms_module):
        terms_module.record_agreement("1.0")

        assert terms_module.needs_agreement("2.0") is True

    def test_the_same_version_is_not_asked_twice(self, terms_module):
        terms_module.record_agreement("1.0")

        assert terms_module.needs_agreement("1.0") is False

    def test_the_screen_reopens_while_the_terms_are_unaccepted(self,
                                                              monkeypatch):
        """Dismissal marks the questions answered; a licence has no default."""
        import spacr.qt.setup_screen as setup_screen

        monkeypatch.setattr(setup_screen, "should_open", lambda *a, **k: False)
        monkeypatch.setenv("SPACR_NO_SETUP", "0")
        monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")

        opened = {}

        class Fake:
            def __init__(self, parent=None):
                opened["built"] = True

            def exec(self):
                return 0

        monkeypatch.setattr(
            "spacr.qt.widgets.setup_slides.SetupSlides", Fake)

        open_setup_if_needed()

        assert opened.get("built") is True

    def test_an_accepted_profile_is_left_alone(self, terms_module,
                                               monkeypatch):
        import spacr.qt.setup_screen as setup_screen

        monkeypatch.setattr(setup_screen, "should_open", lambda *a, **k: False)
        monkeypatch.setenv("SPACR_NO_SETUP", "0")
        monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
        terms_module.record_agreement()

        assert open_setup_if_needed() is None

    def test_a_launch_that_declined_setup_is_never_asked(self, monkeypatch):
        """A batch job on a server can be due and have nobody to answer."""
        monkeypatch.setenv("SPACR_NO_SETUP", "1")

        assert open_setup_if_needed() is None
