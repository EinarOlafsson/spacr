"""The first setup slide tells the user whether spaCR can use their GPU.

Reported as missing: "there used to be a line on the first screen
showing the user if they had access to the cuda gpu (access text was
green, no access was red)".

It had not been removed. It was written, coloured and placed OFF THE
BOTTOM OF THE CARD -- ``sizeHint().height()`` on a word-wrapped QLabel
is the height it would like if it could pick its own width, which for
two sentences is 323 px against a 700 px card. Placed at 0.78 of the
height and 323 tall, the text sat below the card's edge.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import setup_slides as S

pytestmark = pytest.mark.qt


@pytest.fixture
def slides(qtbot):
    dialog = S.SetupSlides()
    qtbot.addWidget(dialog)
    dialog.resize(900, 700)
    dialog.show()
    qtbot.waitExposed(dialog)
    return dialog


class TestTheNoteIsOnTheCard:

    @pytest.mark.parametrize("size", [(900, 700), (700, 520), (600, 420),
                                      (1400, 1000)])
    def test_it_fits_inside_the_card_at_every_size(self, qtbot, size):
        """THE DEFECT. A note drawn past the bottom edge is written,
        coloured, placed -- and not on screen."""
        dialog = S.SetupSlides()
        qtbot.addWidget(dialog)
        dialog.resize(*size)
        dialog.show()
        qtbot.waitExposed(dialog)

        note = dialog._gpu_note
        assert note.isVisible()
        assert note.geometry().bottom() <= dialog.card.height(), (
            f"at {size} the GPU note runs {note.geometry().bottom() - dialog.card.height()} "
            f"px past the bottom of the card, so the user never sees it")
        assert note.geometry().top() >= 0

    def test_its_height_is_the_wrapped_height(self, slides):
        """``heightForWidth`` is the right question for a word-wrapped
        label, and the height actually used has to be that.

        MEASURED, and it corrected what this test first claimed. Once the
        label is laid out, `sizeHint()` agrees with `heightForWidth()`
        -- the 323 px came from asking BEFORE that, while the dialog was
        still being built. So the two cannot be told apart here, and what
        is worth holding is that the box is the wrapped height rather
        than whatever a premature hint answered.
        """
        note = slides._gpu_note
        width = note.width()

        assert note.geometry().height() == note.heightForWidth(width)
        assert note.geometry().height() < slides.card.height() // 3, (
            "the note is a third of the card tall, which is the shape the "
            "unwrapped hint produced")

    def test_it_sits_in_its_band_when_there_is_room(self, slides):
        top = slides._gpu_note.geometry().top()

        assert top == int(slides.card.height() * S.GPU_NOTE_BAND)


class TestWhatItSays:

    def test_it_names_the_two_steps_that_need_a_card(self, slides):
        """NOT A WARNING AND NOT A GATE: everything else runs without
        one, so the note says which two steps are affected and leaves the
        decision to the reader."""
        text = slides._gpu_note.text()

        assert "Segmentation and object classification" in text
        assert "Everything else runs without one" in text

    def test_the_verdict_is_green_when_torch_can_reach_the_card(self, slides):
        """The colour is the whole signal at a glance."""
        usable, _name = S.graphics_card()
        if not usable:
            pytest.skip("no usable GPU on this machine; see the red case")

        text = slides._gpu_note.text()
        assert S.GPU_YES_INK in text
        assert S.GPU_NO_INK not in text
        assert "Compatible GPU" in text

    def test_the_card_is_named_either_way(self, slides):
        """"No compatible GPU" on its own leaves the reader wondering
        whether spaCR looked."""
        _usable, name = S.graphics_card()
        if not name:
            pytest.skip("nothing could be identified on this machine")

        assert name in slides._gpu_note.text()

    def test_red_and_green_are_the_two_inks_and_they_differ(self):
        assert S.GPU_YES_INK != S.GPU_NO_INK
        assert S.GPU_YES_INK.startswith("#") and S.GPU_NO_INK.startswith("#")

    def test_a_card_torch_cannot_use_is_told_apart_from_no_card(self):
        """A CPU-only torch build and a driver older than the CUDA runtime
        both present as "cuda not available", and that is a different
        problem from having no card -- with a different fix."""
        assert "spacr-doctor" in S.GPU_DOCTOR_HINT
        assert "The card is there but torch cannot use it" in S.GPU_DOCTOR_HINT


class TestWhereItAppears:

    def test_it_is_on_the_first_slide_only(self, slides, qtbot):
        """It answers "can this machine run spaCR", which the reader asks
        once. Carried down the rest it would be a banner that stopped
        being read on slide two and took the space anyway."""
        assert slides._gpu_note.isVisible()

        slides._show_slide(1)
        qtbot.wait(10)
        assert not slides._gpu_note.isVisible()

        slides._show_slide(0)
        qtbot.wait(10)
        assert slides._gpu_note.isVisible()

    def test_it_is_placed_the_moment_it_is_shown(self, slides, qtbot):
        """Nothing else lays this label out, so a note made visible and
        left unplaced sits in the corner it was born in."""
        slides._show_slide(1)
        qtbot.wait(10)
        slides._show_slide(0)
        qtbot.wait(10)

        assert slides._gpu_note.geometry().top() > 0
        assert slides._gpu_note.geometry().bottom() <= slides.card.height()
