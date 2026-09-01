"""The capability note stays out of the nav row, and off slide one's counter.

Reported 2026-09-01: "the new table in spacr setup needs to be centered,
now it overlaps with the back button, and dont show 1 of 7 (also because
it overlaps) just show 2 of 7".

The note is placed with ``setGeometry`` over the card rather than laid
out, so nothing else in the dialog pushes back when it grows. It grew a
capability table the day before, and the extra height put it straight
over Back and the step counter -- a label that is only decoration ending
up on top of the one control that leaves the slide.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import setup_slides as S


@pytest.fixture
def slides(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dialog = S.SetupSlides()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    return dialog


def test_the_note_stops_above_the_back_button(slides):
    """THE OVERLAP. Geometry, not appearance."""
    slides._show_slide(0)
    slides._place_the_gpu_note()

    note = slides._gpu_note
    card = slides.card
    back = slides._back

    assert note.isVisible(), "no note to place on the first slide"
    note_bottom = note.y() + note.height()
    buttons_top = back.mapTo(card, back.rect().topLeft()).y()

    assert note_bottom <= buttons_top, (
        f"the note runs to y={note_bottom} and the Back button starts at "
        f"y={buttons_top}; the note is sitting on the button")


def test_the_note_still_starts_below_the_greeting(slides):
    """So the clamp did not fix the overlap by moving the note off the
    top of the card instead."""
    slides._show_slide(0)
    slides._place_the_gpu_note()
    assert slides._gpu_note.y() > 0


def test_a_short_card_lifts_the_note_rather_than_losing_it(slides, qtbot):
    """The clamp's other end, which the nav-row floor must not break.

    The note is the answer to "can this machine run spaCR", so a small
    window must not be the reason it is missed.
    """
    slides.resize(720, 380)
    qtbot.wait(10)
    slides._show_slide(0)
    slides._place_the_gpu_note()

    note = slides._gpu_note
    assert note.y() >= 0
    assert note.y() + note.height() <= slides.card.height(), (
        "the note hangs off the bottom of the card")


def test_the_first_slide_shows_no_step_counter(slides):
    """"1 of 7" says least where there is least room for it."""
    slides._show_slide(0)
    assert slides._where.text() == ""


def test_every_later_slide_does_show_one(slides):
    """So the fix is 'not on slide one', not 'never'."""
    for index in range(1, len(S.SLIDES)):
        slides._show_slide(index)
        assert slides._where.text(), f"slide {index} has no counter"
        assert str(index + 1) in slides._where.text()


def test_going_back_to_the_first_slide_clears_it_again(slides):
    """The counter is set on every slide change, so a stale one would
    only show after navigating away and back."""
    slides._show_slide(2)
    assert slides._where.text()
    slides._show_slide(0)
    assert slides._where.text() == ""


def test_the_capability_table_is_centred(slides):
    """Qt rich text ignores `margin:auto`, so the centring has to come
    from a block that it does honour."""
    text = slides._gpu_note.text()
    if "<table" not in text:
        pytest.skip("no capability table on this machine")
    assert '<div align="center">' in text, (
        "the table is not wrapped in a centring block")
    assert text.index('<div align="center">') < text.index("<table")
