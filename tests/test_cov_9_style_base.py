"""Shared page settings that change what a figure MEANS, not just how it looks.

An axis limit, an inverted axis and a painted background are three settings a
reader cannot recover from the picture. If ``apply_page`` drops one of them,
the figure on screen answers a different question from the one the style
describes, and nothing in the file says so.
"""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

from spacr.style_base import FigureStyle, apply_page  # noqa: E402


@pytest.fixture
def canvas():
    figure, axes = plt.subplots()
    axes.plot([0, 1, 2], [0, 1, 4])
    yield figure, axes
    plt.close(figure)


def test_a_y_limit_in_the_style_reaches_the_axis(canvas):
    """A y limit crops what the reader is shown and must be applied.

    Two figures drawn from the same data with different y limits look like
    two different results; a limit the renderer ignores makes the style file
    and the figure disagree about which one was produced.
    """
    figure, axes = canvas
    apply_page(figure, axes, FigureStyle(y_lim=(-3.0, 7.0)))
    assert axes.get_ylim() == (-3.0, 7.0)


def test_an_inverted_x_axis_actually_reverses_the_direction(canvas):
    """Inverting x reverses which end of the axis is "more".

    This is the one page setting that can turn a figure into its own mirror
    image while every tick label stays correct, so it has to be applied
    rather than quietly dropped.
    """
    figure, axes = canvas
    low, high = axes.get_xlim()
    apply_page(figure, axes, FigureStyle(invert_x=True))
    inverted_low, inverted_high = axes.get_xlim()
    assert inverted_low > inverted_high
    assert (inverted_low, inverted_high) == (high, low)


def test_a_named_background_paints_both_the_page_and_the_axes(canvas):
    """A background colour must cover the page as well as the plotting area.

    Painting only the axes leaves a white border around a dark figure, which
    is exactly the artefact that makes an exported panel unusable in a dark
    document.
    """
    from matplotlib.colors import to_rgba

    figure, axes = canvas
    apply_page(figure, axes, FigureStyle(background_color="#101820"))
    expected = to_rgba("#101820")
    assert figure.patch.get_facecolor() == expected
    assert axes.get_facecolor() == expected


def test_the_default_background_leaves_the_page_alone(canvas):
    """``"none"`` means "do not paint", not "paint a colour called none".

    The default has to be distinguishable from a chosen colour, or every
    figure would be repainted whether the user asked for it or not.
    """
    figure, axes = canvas
    before_figure = figure.patch.get_facecolor()
    before_axes = axes.get_facecolor()
    apply_page(figure, axes, FigureStyle())
    assert figure.patch.get_facecolor() == before_figure
    assert axes.get_facecolor() == before_axes
