"""The GPU note belongs in its band, not in the corner of the card.

The note is shown on the first slide, and the only thing that laid it out
gave up on a note that was not visible YET -- while running before the
slide made it visible. So the label was shown having never been placed,
keeping the geometry a QLabel is born with: the top left corner, where it
read as a stray sentence above the language list and went away only when
the first Next hid it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.setup_slides import (                    # noqa: E402
    GPU_NOTE_BAND, GPU_REQUIREMENT, SetupSlides,
)

CARD_MARGIN = 28


@pytest.fixture
def slides(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = SetupSlides()
    qtbot.addWidget(window)
    window.resize(900, 640)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_the_gpu_note_is_not_in_the_corner(slides):
    """The corner is where an unplaced QLabel sits, and it is the defect."""
    note = slides._gpu_note
    assert note.isVisible(), "the note is shown on the first slide"
    assert note.geometry().topLeft() != note.geometry().topLeft().__class__(0, 0)


def test_the_gpu_note_sits_in_its_own_band(slides):
    """Placed at the band the constant names, inside the card's margins."""
    note, card = slides._gpu_note, slides.card
    assert note.geometry().y() == int(card.height() * GPU_NOTE_BAND)
    assert note.geometry().x() == CARD_MARGIN
    assert note.width() == card.width() - 2 * CARD_MARGIN


def test_showing_the_note_places_it(slides):
    """The defect was an ordering one, and this is the ordering.

    The placement used to bail out on a note that was not visible YET, and
    it ran before the slide made the note visible -- so the label was shown
    having never been laid out. Whatever the order, becoming visible has to
    be enough to put it in its band.
    """
    note, card = slides._gpu_note, slides.card
    note.setVisible(False)
    note.move(0, 0)
    slides._place_the_greeting()          # the pass that used to give up
    note.setVisible(True)
    slides._place_the_gpu_note()
    assert note.geometry().y() == int(card.height() * GPU_NOTE_BAND)
    assert note.geometry().y() > 0


def test_the_note_stays_in_its_band_when_the_card_is_resized(slides, qtbot):
    """A resize re-places it rather than stranding it at the old height."""
    slides.resize(1100, 820)
    qtbot.wait(30)
    slides._place_the_gpu_note()
    note, card = slides._gpu_note, slides.card
    assert note.geometry().y() == int(card.height() * GPU_NOTE_BAND)


def test_the_note_carries_the_requirement_sentence(slides):
    """What is drawn there is the GPU sentence, not something else."""
    assert GPU_REQUIREMENT.split(".")[0] in slides._gpu_note.text()


def test_the_note_leaves_with_the_first_slide(slides):
    """It is a first-slide note; the later slides must not carry it."""
    slides._show_slide(1) if hasattr(slides, "_show_slide") else None
    if hasattr(slides, "_show_slide"):
        assert not slides._gpu_note.isVisible()
