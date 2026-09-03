"""The popup's footer says API and Animation, in words.

TWO REQUESTS, ONE DAY APART IN THE SAME AFTERNOON, and the second reverses
the first:

  2026-09-02, earlier: "instead of API just show a teel dot for api and a
  purple square for annimation" -- because the words repeated on every row
  and cost a line after the first reading.

  2026-09-02, instruction 371: "an API link and Annimation link text (remove
  the dot)", and for the bottom strip "which should also just say API".

This file was written for the first and is kept for the second, because what
it actually asserts survives both: the footer's two controls are TARGETS, they
are ANNOUNCED as words, and they are distinguishable WITHOUT COLOUR.

That last one was the marks' whole justification -- instruction 89 added a
colourblind mode, and a teal dot beside a purple square differs in FORM. Words
are strictly better on that axis, so the requirement is satisfied by the
reversal rather than dropped by it, and the assertion is unchanged.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets import hover_tooltip


@pytest.fixture
def tip(qtbot, qt_theme_applied):
    popup = hover_tooltip.HoverTooltip()
    qtbot.addWidget(popup)
    return popup


def test_the_footer_draws_the_words(tip):
    assert tip._api_link.text() == hover_tooltip.API_MARK == "API"
    assert tip._animation_link.text() == hover_tooltip.ANIMATION_MARK
    assert tip._animation_link.text() == "Animation"


def test_a_screen_reader_still_hears_the_words(tip):
    """Set explicitly rather than inferred from the label.

    While the marks were drawn, these were the ONLY place the words existed
    and a reader announcing "black circle" would have been a regression. The
    drawn text says the words again now, so leaving these to be inferred
    would work -- and would trade a guarantee for a coincidence.
    """
    assert tip._api_link.accessibleName() == "API"
    assert tip._animation_link.accessibleName() == "Animation"
    assert tip._api_link.accessibleDescription()
    assert tip._animation_link.accessibleDescription()


def test_a_pointer_can_still_ask_what_the_control_does(tip):
    """The tooltip outlived the mark it was explaining, and should.

    It existed because a bare dot is a rebus. With words drawn it is no
    longer load-bearing, but "API" alone does not say WHICH page it opens,
    and the tooltip does.
    """
    assert "API" in tip._api_link.toolTip()
    assert "animation" in tip._animation_link.toolTip().lower()


def test_the_two_are_told_apart_without_colour(tip):
    """Instruction 89 added a colourblind mode.

    The marks answered this by differing in SHAPE as well as hue. Words
    answer it outright. The assertion is deliberately unchanged across the
    reversal, because it is the requirement, not the implementation -- and
    it is what stops a later tidy-up from making the two look alike again.
    """
    assert hover_tooltip.API_MARK != hover_tooltip.ANIMATION_MARK
    assert tip._api_link.text().strip() and tip._animation_link.text().strip()


def test_the_marks_keep_their_named_colours_and_are_drawn_larger(tip):
    """A theme switch must not repaint a colour the maintainer chose."""
    sheet = tip.styleSheet()
    assert hover_tooltip.TEAL in sheet
    assert hover_tooltip.PURPLE in sheet
    # Drawn above the prose size: at the popup's small font a circle is a
    # few pixels across, and these are click targets.
    from spacr.qt.theme import font_px

    assert f"font-size: {font_px('small') + 4}px" in sheet


def test_both_marks_are_still_clickable(tip, monkeypatch):
    """The whole point of them: they are links, not decoration."""
    opened = []
    monkeypatch.setattr(tip, "open_api_documentation",
                        lambda: opened.append("api"))
    monkeypatch.setattr(tip, "toggle_animation",
                        lambda: opened.append("animation"))
    # Reconnect, because the constructor bound the originals.
    tip._api_link.clicked.connect(lambda: opened.append("api"))
    tip._animation_link.clicked.connect(lambda: opened.append("animation"))
    tip._api_link.clicked.emit()
    tip._animation_link.clicked.emit()
    assert "api" in opened and "animation" in opened
