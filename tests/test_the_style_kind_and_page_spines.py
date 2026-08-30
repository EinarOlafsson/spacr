"""Naming a style class, and the page settings that can be turned off.

``style_kind`` strips a trailing "Style" so ``VolcanoStyle`` becomes
``volcano``, which is what keeps independently declared style names from
colliding. A class NOT named that way must keep its whole name -- stripping
nothing is what makes the key correct for a style declared outside the
convention, and this headless helper exists precisely so styles can be
declared without the Qt widgets.
"""
from __future__ import annotations

import dataclasses

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


class VolcanoStyle:
    pass


class Heatmap:
    pass


class Style:
    pass


def test_a_class_named_for_the_convention_loses_its_suffix():
    """The taken side: VolcanoStyle keys as 'volcano'."""
    from spacr.style_base import style_kind

    assert style_kind(VolcanoStyle()) == "volcano"


def test_a_class_not_named_for_the_convention_keeps_its_whole_name():
    """Arc 112 -> 114: nothing is stripped.

    Removing a suffix that is not there would turn 'Heatmap' into something
    shorter and key two different styles onto one name, which is the collision
    the function exists to prevent.
    """
    from spacr.style_base import style_kind

    assert style_kind(Heatmap()) == "heatmap"


def test_a_class_called_only_style_falls_back_to_figure():
    """The ``or "figure"`` beside it: stripping everything leaves nothing.

    An empty key would collide with every other empty key, which is worse than
    a shared generic one.
    """
    from spacr.style_base import style_kind

    assert style_kind(Style()) == "figure"


# ---------------------------------------------------------------------------
# apply_page — the spines a caller chose to keep
# ---------------------------------------------------------------------------

def _style(**changes):
    from spacr.style_base import FigureStyle

    return dataclasses.replace(FigureStyle(), **changes)


def test_the_top_and_right_spines_are_kept_when_asked():
    """The opt-out restores spines hidden by ambient Matplotlib settings.

    Applying a style that asks to keep the frame visible must show both spines
    even when they were initially hidden by global ``rcParams``.
    """
    from spacr.style_base import apply_page

    with matplotlib.rc_context(
        {"axes.spines.top": False, "axes.spines.right": False}
    ):
        figure, axes = plt.subplots()
        try:
            apply_page(figure, axes, _style(hide_top_right_spines=False))
            assert axes.spines["top"].get_visible()
            assert axes.spines["right"].get_visible()
        finally:
            plt.close(figure)


def test_the_top_and_right_spines_are_hidden_by_default():
    """The taken side, which is the house style."""
    from spacr.style_base import apply_page

    figure, axes = plt.subplots()
    try:
        apply_page(figure, axes, _style(hide_top_right_spines=True))
        assert not axes.spines["top"].get_visible()
        assert not axes.spines["right"].get_visible()
    finally:
        plt.close(figure)


def test_a_grid_that_was_asked_for_sits_below_the_data():
    """``set_axisbelow(True)`` beside it, which is why the grid is drawable."""
    from spacr.style_base import apply_page

    figure, axes = plt.subplots()
    try:
        apply_page(figure, axes, _style(grid=True))
        assert axes.get_axisbelow()
    finally:
        plt.close(figure)


def test_a_background_of_none_leaves_the_figure_transparent():
    """The ``!= "none"`` guard: 'none' is the documented default and a no-op."""
    from spacr.style_base import apply_page

    figure, axes = plt.subplots()
    try:
        before = figure.patch.get_facecolor()
        apply_page(figure, axes, _style(background_color="none"))
        assert figure.patch.get_facecolor() == before
    finally:
        plt.close(figure)


def test_a_real_background_colour_is_applied_to_both():
    """The taken side: the figure AND the axes, because either alone shows."""
    from spacr.style_base import apply_page

    figure, axes = plt.subplots()
    try:
        apply_page(figure, axes, _style(background_color="#123456"))
        assert figure.patch.get_facecolor()[:3] != (1.0, 1.0, 1.0)
        assert axes.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    finally:
        plt.close(figure)
