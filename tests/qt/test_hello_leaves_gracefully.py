"""Hello arrived well and was switched off.

"the transition to Hello is good, but the transition away from Hello is
abrupt and bad, make Hello appear somewhere where there will be nothing in
the next page and make it fade gracefully."

Two causes, both fixed together. The greeting was a ROW IN THE COLUMN, so
it took space on the language slide and the next slide needed that space
back -- which is why leaving had to be instant. It floats over the card now,
in a band the question rows never reach on any slide, so nothing waits on it
and it can be given the time to go.

It leaves more slowly than it arrives: 700ms out against 420ms in. A word
that leaves at the speed it came reads as being taken away.
"""
from __future__ import annotations

import time

import pytest
from PySide6.QtCore import Qt

from spacr.qt.widgets.setup_slides import (GREETING_BAND, GREETING_FADE_MS,
                                           GREETING_LEAVE_MS, SetupSlides)


HEIGHT = 620


@pytest.fixture
def slides(qapp):
    dialog = SetupSlides()
    dialog.resize(900, HEIGHT)
    dialog.show()
    qapp.processEvents()
    try:
        yield dialog
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_the_greeting_floats_over_the_card(slides):
    """In the layout it would take space the next slide needs back."""
    assert slides._greeting.parent() is slides.card
    assert slides._greeting.isVisible() is False


def test_it_sits_in_the_band_no_slide_uses(slides, qapp):
    slides._show_the_greeting()
    qapp.processEvents()

    top = slides._greeting.geometry().top()
    assert top == pytest.approx(HEIGHT * GREETING_BAND, abs=4)
    # Across the whole card, so it is centred whatever the word is.
    assert slides._greeting.width() == slides.card.width()


def test_it_takes_no_clicks(slides):
    """It lies over the card; a word that swallowed a click would be a trap."""
    assert slides._greeting.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_leaving_is_a_fade_not_a_switch(slides, qapp):
    slides._show_the_greeting()
    qapp.processEvents()
    assert slides._greeting.isVisible()

    slides._fade_the_greeting_away()

    assert slides._goodbye is not None, "it was switched off, not faded"
    assert slides._goodbye.duration() == GREETING_LEAVE_MS
    # STILL VISIBLE while it fades: hiding first is the abrupt cut with
    # extra steps, because a hidden widget does not animate.
    assert slides._greeting.isVisible()


def test_it_is_gone_when_the_fade_ends(slides, qapp):
    slides._show_the_greeting()
    qapp.processEvents()
    slides._fade_the_greeting_away()

    deadline = time.time() + (GREETING_LEAVE_MS / 1000.0) + 2.0
    while time.time() < deadline and slides._greeting.isVisible():
        qapp.processEvents()
        time.sleep(0.02)

    assert not slides._greeting.isVisible()


def test_it_leaves_more_slowly_than_it_arrives():
    assert GREETING_LEAVE_MS > GREETING_FADE_MS


def test_fading_something_already_gone_does_nothing(slides):
    """Leaving the language slide twice must not start a second animation."""
    assert not slides._greeting.isVisible()

    slides._fade_the_greeting_away()

    assert slides._goodbye is None


def test_the_next_slide_starts_the_fade(slides, qapp):
    """The real path, not just the helper."""
    slides._show_the_greeting()
    qapp.processEvents()

    slides._show_slide(1)

    assert slides._goodbye is not None
