"""The one screen whose first question is the language.

Reported on 2026-08-23: "language in the startup is also not implemented
(other than english for settings and the rest)". The setup screen had no
``tr()`` in it at all -- it asked which language to use and then went on
asking the other five questions in English, which leaves the user no way
to tell whether the setting took.

Two claims, and the second is the one that was missing:

1. every caption on the screen has a catalog entry, so a translation
   exists to apply; and
2. choosing a language APPLIES it, to the screen the user is looking at,
   as they pick it -- the way the theme and the greeting already did.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QAbstractButton, QLabel

from spacr.qt import i18n
from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides


@pytest.fixture
def slides(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    made = SetupSlides()
    qtbot.addWidget(made)
    made.show()
    return made


def test_every_slide_title_and_blurb_is_in_the_catalog():
    """Nothing on the screen may be untranslatable prose."""
    missing = [text for title, blurb, _keys in SLIDES
               for text in (title, blurb)
               if not i18n.has_translation(text)]
    assert not missing, f"no catalog entry for: {missing}"


def test_every_question_label_is_in_the_catalog():
    """The captions beside the controls, which come from setup_screen."""
    from spacr.qt.setup_screen import questions

    missing = [q[1] for q in questions() if not i18n.has_translation(q[1])]
    assert not missing, f"no catalog entry for: {missing}"


def test_choosing_a_language_redraws_the_screen_in_it(slides, qapp):
    """The proof the setting took, on the screen that asked for it."""
    box = slides._editors["language"]
    assert "Language" in slides._title.text()

    box.setCurrentIndex(box.findData("de"))
    qapp.processEvents()

    assert "Sprache" in slides._title.text()
    assert slides._next.text() == i18n.tr("Next ›", "de")

    # THE COUNTER IS BLANK ON SLIDE ONE, ON PURPOSE. This used to assert
    # "1 von 7" and went red without the translation getting worse:
    # `_show_slide` now writes "" at index 0, because "1 of 7" was landing on
    # top of the GPU note and says least there anyway -- nobody needs telling
    # they are at the beginning.
    assert slides._where.text() == ""

    # So the counter is proved translated on a slide that HAS one, rather
    # than the assertion being dropped. Asserting only the blank would pass
    # against a counter that had stopped being translated entirely.
    slides._show_slide(1)
    qapp.processEvents()
    assert slides._where.text() == i18n.tr("{n} of {total}", "de").format(
        n=2, total=len(SLIDES))
    assert slides._where.text() != "2 of 7"


def test_it_switches_between_two_translations_not_through_english(slides,
                                                                  qapp):
    """German then Korean. A screen that translated the RENDERED text
    would be looking up German in a Korean catalog and finding nothing."""
    box = slides._editors["language"]
    box.setCurrentIndex(box.findData("de"))
    qapp.processEvents()
    box.setCurrentIndex(box.findData("ko"))
    qapp.processEvents()

    assert slides._title.text() == f"<b>{i18n.tr('Language', 'ko')}</b>"
    assert i18n.tr("Language", "ko") not in ("Language", "Sprache")


def test_nothing_on_the_first_slide_is_left_in_english(slides, qapp):
    """The whole visible page, not just the parts this test names."""
    box = slides._editors["language"]
    box.setCurrentIndex(box.findData("sv"))
    qapp.processEvents()

    english = []
    for widget in list(slides.findChildren(QLabel)) + list(
            slides.findChildren(QAbstractButton)):
        if not widget.isVisibleTo(slides):
            continue
        text = (widget.text() or "").replace("<b>", "").replace("</b>", "")
        text = text.strip()
        if not text or not i18n.has_translation(text):
            continue
        # A caption still equal to its English source, when Swedish says
        # something else, is one the pass missed.
        if i18n.tr(text, "sv") != text:
            english.append(text)
    assert not english, f"still in English on the language slide: {english}"
