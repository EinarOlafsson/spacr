"""Neither greeting method may assume the label already exists.

Instruction 310, A33. ``_place_the_greeting`` guarded its use of the
greeting label; ``_fade_the_greeting_away`` -- which ``_show_slide``
calls for every slide after the first -- dereferenced it directly.

The two disagreed about whether the label may be absent, and only one was
right. ``_place_the_greeting`` fetches the CARD with ``getattr`` for a
reason: a resize can arrive while the dialog is still being built. The
greeting is created AFTER the card, so there is a window in which the
card exists and the greeting does not -- and in that window the guarded
method survived and the unguarded one raised.

No user has hit it: the attribute is set in ``__init__`` and never
cleared, so in a finished dialog both paths agree. That is what makes it
worth a test rather than a bug report -- it is latent, and the next
refactor that clears the label or moves its creation makes it live.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.setup_slides import SetupSlides


class _HalfBuilt:
    """A dialog caught between its card and its greeting.

    Not a mock of the methods under test -- the real ones are called
    against it, unbound, which is how the window is reproduced without
    racing a real construction.
    """

    def __init__(self, card=None):
        self.card = card


def test_fading_a_greeting_that_does_not_exist_yet_is_a_no_op():
    """THE ASYMMETRY, driven.

    Before the fix this raised AttributeError, and ``_show_slide`` calls
    it on every slide change.
    """
    SetupSlides._fade_the_greeting_away(_HalfBuilt())      # must not raise


def test_placing_a_greeting_that_does_not_exist_yet_is_a_no_op():
    """The half that was already right, asserted so a tidy-up that
    'harmonises' the two cannot remove the wrong one."""
    SetupSlides._place_the_greeting(_HalfBuilt())          # must not raise


def test_neither_needs_a_card_either():
    """The card is fetched with getattr for the same reason.

    A resize before the card exists is the earlier half of the same
    window, and it reaches both methods.
    """
    empty = type("Empty", (), {})()
    SetupSlides._fade_the_greeting_away(empty)
    SetupSlides._place_the_greeting(empty)


def test_a_finished_dialog_still_fades_its_greeting(qtbot, qt_theme_applied,
                                                    tmp_path, monkeypatch):
    """THE BEHAVIOUR THE GUARD MUST NOT COST.

    A guard that returns early is only correct if the real case still
    happens -- otherwise the fade is silently dead and the greeting
    stays on screen over every later slide.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    dialog._greeting.setText("hello")
    dialog._greeting.setVisible(True)

    dialog._fade_the_greeting_away()

    assert dialog._goodbye is not None, (
        "the greeting was not given a fade; the guard is swallowing the "
        "real case")
