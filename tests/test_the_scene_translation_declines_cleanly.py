"""Small refusals inside the matplotlib-to-pyqtgraph translation.

The translation's contract, stated in ``SceneReport``, is that a translation
which dropped something is not a picture of the panel -- so the caller writes
the matplotlib page instead. Everything here is a piece of that: a size that
cannot be converted, a colour the appearance refuses, a line whose transform
cannot be read. Each returns a value the caller can act on rather than raising
into a figure that is already half drawn.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _Look.px — a size that is not a number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("points", [None, "thick", object(), [1.0]])
def test_a_size_that_is_not_a_number_falls_back_to_the_minimum(points):
    """Lines 307-308.

    matplotlib carries sizes that are sometimes None and sometimes a string
    -- ``linewidth=None`` means "use the rcParam" and never reaches a float.
    The minimum keeps the artist VISIBLE: returning 0.0 would translate the
    line into a zero-width one, which pyqtgraph draws as nothing at all, and
    the scene would be reported complete while missing a line.
    """
    from spacr.figures.scene import _Look

    look = _Look(dpi=100.0)
    assert look.px(points, minimum=0.5) == 0.5


def test_a_real_size_is_scaled_from_points_to_pixels():
    """The conversion the class exists for, at two different dpi.

    The comment beside ``scale`` records what happens without it: a 3 pt
    marker drawn 3 px across is 1.9 times too small at this suite's 140 dpi,
    and a panel of points becomes a panel of dust.
    """
    from spacr.figures.scene import _Look

    coarse = _Look(dpi=72.0).px(10.0)
    fine = _Look(dpi=144.0).px(10.0)

    assert fine == pytest.approx(coarse * 2.0)
    assert coarse == pytest.approx(10.0)


def test_a_size_below_the_minimum_is_raised_to_it():
    """The ``max`` beside the conversion, for a hairline width."""
    from spacr.figures.scene import _Look

    assert _Look(dpi=100.0).px(0.0, minimum=0.25) == 0.25


# ---------------------------------------------------------------------------
# _reference_line — a transform that cannot be read, and a real 2-D line
# ---------------------------------------------------------------------------

def test_an_artist_whose_transform_cannot_be_read_is_not_a_reference():
    """Lines 387-388: None, meaning "not a reference line", not a crash.

    The caller uses the answer to decide how to place the artist. A raise here
    would abandon a panel that was already partly translated, and the honest
    default for an artist that will not answer is that it is ordinary data.
    """
    from spacr.figures.scene import _reference_line

    class _RefusesItsTransform:
        def get_transform(self):
            raise RuntimeError("this artist has no usable transform")

    figure, axes = plt.subplots()
    try:
        assert _reference_line(_RefusesItsTransform(), axes) is None
    finally:
        plt.close(figure)


def test_a_line_that_varies_in_both_axes_is_not_a_reference():
    """Line 398: the fall-through for an ordinary plotted line.

    A reference line is one CONSTANT in x or in y -- an axhline or an axvline.
    A line that moves in both is data, and calling it a reference would place
    it with the chrome and paint it in the chrome colour.
    """
    from spacr.figures.scene import _reference_line

    figure, axes = plt.subplots()
    try:
        line, = axes.plot([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        assert _reference_line(line, axes) is None
    finally:
        plt.close(figure)


def test_a_horizontal_and_a_vertical_reference_are_recognised():
    """The two taken sides, so the None above is visibly a third outcome."""
    from spacr.figures.scene import _reference_line

    figure, axes = plt.subplots()
    try:
        horizontal = axes.axhline(0.5)
        vertical = axes.axvline(0.5)
        assert _reference_line(horizontal, axes) == "h"
        assert _reference_line(vertical, axes) == "v"
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# _pen — an appearance that refuses the colour
# ---------------------------------------------------------------------------

def test_a_colour_the_appearance_refuses_yields_no_pen(monkeypatch):
    """Line 363: None, which every caller checks before drawing.

    ``paint`` returns None when the saved appearance has no colour for that
    role -- a mode that draws no chrome, for instance. The pen callers all
    treat None as "do not draw this artist" and count it as missing, which is
    what makes the scene report say the translation was incomplete rather than
    drawing the artist in some default colour the user did not choose.
    """
    from spacr.figures import scene

    look = scene._Look(dpi=100.0)
    monkeypatch.setattr(look, "paint", lambda *_a, **_k: None)

    assert scene._pen("#ff0000", 1.0, "-", 1.0, look, "chrome") is None


def test_a_colour_the_appearance_accepts_yields_a_pen():
    """The taken side, so the refusal above is visibly a decision."""
    from spacr.figures import scene

    look = scene._Look(dpi=100.0)
    pen = scene._pen("#ff0000", 1.0, "-", 1.0, look, "chrome")

    assert pen is not None


def test_an_axes_fraction_line_that_varies_in_both_is_not_a_reference():
    """Line 398: the final fall-through.

    Reaching it needs an artist that is BOTH not in data coordinates -- an
    ordinary plotted line returns at the transData check above -- and not
    constant in either axis. A diagonal drawn in axes-fraction coordinates is
    exactly that, and it is a real thing: a 1:1 guide line drawn across a
    panel is usually placed that way.

    Calling it a reference would place it with the chrome and paint it in the
    chrome colour, so the None is what keeps it looking like the annotation it
    is.
    """
    from spacr.figures.scene import _reference_line

    figure, axes = plt.subplots()
    try:
        line, = axes.plot([0.0, 1.0], [0.0, 1.0],
                          transform=axes.transAxes)
        assert line.get_transform() is not axes.transData
        assert _reference_line(line, axes) is None
    finally:
        plt.close(figure)
