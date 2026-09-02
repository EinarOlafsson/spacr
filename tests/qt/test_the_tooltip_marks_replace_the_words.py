"""The popup's footer is a teal dot and a purple square (instruction 347).

Asked for on 2026-09-02: "instead of API just show a teel dot for api and a
purple square for annimation". The words were repeated on every row that had
them, so after the first reading they cost a line and said nothing.

WHAT MUST SURVIVE THE CHANGE, and is what most of this file asserts: the marks
are still TARGETS, they are still ANNOUNCED as words to a screen reader, and
they are still distinguishable without colour.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets import hover_tooltip


@pytest.fixture
def tip(qtbot, qt_theme_applied):
    popup = hover_tooltip.HoverTooltip()
    qtbot.addWidget(popup)
    return popup


def test_the_footer_draws_marks_and_not_words(tip):
    assert tip._api_link.text() == hover_tooltip.API_MARK
    assert tip._animation_link.text() == hover_tooltip.ANIMATION_MARK
    for widget in (tip._api_link, tip._animation_link):
        assert "API" not in widget.text()
        assert "Animation" not in widget.text()


def test_a_screen_reader_still_hears_the_words(tip):
    """A reader that announced "black circle" would be a regression."""
    assert tip._api_link.accessibleName() == "API"
    assert tip._animation_link.accessibleName() == "Animation"
    assert tip._api_link.accessibleDescription()
    assert tip._animation_link.accessibleDescription()


def test_a_pointer_can_still_ask_what_the_mark_means(tip):
    """The word has to live somewhere reachable, or the mark is a rebus."""
    assert "API" in tip._api_link.toolTip()
    assert "animation" in tip._animation_link.toolTip().lower()


def test_the_two_marks_differ_in_shape_not_only_colour(tip):
    """Instruction 89 added a colourblind mode.

    Two dots in two colours would be one distinction; a dot and a square are
    two. This is the assertion that stops a later tidy-up from making them
    both circles.
    """
    assert hover_tooltip.API_MARK != hover_tooltip.ANIMATION_MARK


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
