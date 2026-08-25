"""Styling degrades instead of failing, and a colour string is read strictly.

Two independent promises in :mod:`spacr.figure_style`. Applying a style must
never be what stops a run, so a missing or broken palette library falls back
to spaCR's own colours. And `to_rgb` is what the contrast check measures
against, so anything it cannot read has to come back as None -- a guessed
colour would produce a contrast number about a colour nobody chose.
"""
from __future__ import annotations

import pytest

from spacr.figure_style import (
    DARK_GRID,
    DARK_INK,
    PRINT_GRID,
    PRINT_INK,
    _apply_palette,
    theme_ink,
    to_rgb,
)


@pytest.fixture()
def restored_rc_params():
    """Give the colour cycle back afterwards; rcParams is process-global."""
    import matplotlib as mpl

    before = dict(mpl.rcParams)
    yield mpl
    mpl.rcParams.update(before)


# ---------------------------------------------------------------------------
# The palette falls back rather than raising
# ---------------------------------------------------------------------------

def test_a_palette_seaborn_cannot_give_falls_back_to_spacrs_own(
        monkeypatch, restored_rc_params):
    """An unknown palette name still leaves a usable colour cycle.

    seaborn raises on a name it does not know, and the figure still has to be
    drawn in something -- so the cycle becomes spaCR's own palette rather
    than whatever matplotlib's default happened to be.
    """
    import seaborn as sns

    from spacr.qt.widgets.fast_plots import PALETTE

    def refuse(*args, **kwargs):
        raise ValueError("no palette by that name")

    monkeypatch.setattr(sns, "color_palette", refuse)

    _apply_palette("a-palette-that-does-not-exist")

    cycle = restored_rc_params.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle == list(PALETTE)


# ---------------------------------------------------------------------------
# Reading a colour
# ---------------------------------------------------------------------------

def test_a_short_hex_colour_expands_to_the_same_colour_as_its_long_form():
    """`#0f8` and `#00ff88` are one colour, so they must read as one number."""
    assert to_rgb("#0f8") == to_rgb("#00ff88")
    assert to_rgb("#0f8f") == to_rgb("#00ff88")


def test_a_fully_transparent_colour_is_not_a_colour():
    """Alpha 00 means nothing is drawn, so there is no ink to measure.

    Returning the underlying RGB would let the contrast check pass a figure
    whose text is invisible.
    """
    assert to_rgb("#123456ff") == to_rgb("#123456")
    assert to_rgb("#12345600") is None


@pytest.mark.parametrize("text", ["#gggggg", "#zz00zz00"])
def test_hex_digits_that_are_not_digits_are_refused(text):
    """A malformed hex string is None, never a partially parsed colour."""
    assert to_rgb(text) is None


@pytest.mark.parametrize("text", ["#12345", "#1", "#", "#1234567"])
def test_a_hex_string_of_the_wrong_length_is_refused(text):
    """Only 3, 4, 6 and 8 digits name a colour; the rest are typos."""
    assert to_rgb(text) is None


# ---------------------------------------------------------------------------
# Theme colours when the preference store will not answer
# ---------------------------------------------------------------------------

def test_a_theme_lookup_that_raises_still_yields_print_colours(monkeypatch):
    """A broken preference store leaves a figure that prints, not a traceback.

    Print ink on a white page is the safe answer: it is legible on paper
    whatever the application theme turns out to be.
    """
    from spacr.qt import preferences

    def refuse():
        raise RuntimeError("the preference store is unreadable")

    monkeypatch.setattr(preferences, "resolve_effective_theme", refuse)

    assert theme_ink() == (PRINT_INK, PRINT_GRID)


def test_a_dark_theme_gets_light_ink(monkeypatch):
    """The fallback means something only if the real answer differs."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "space")

    assert theme_ink() == (DARK_INK, DARK_GRID)
