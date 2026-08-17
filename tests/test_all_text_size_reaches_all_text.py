""""All text size" reaches ALL of it, and opens at what the figure uses.

GitHub issue #108, 2026-08-17, macOS: "the font size in the figure previews
does not work as expected. Font size is by default to large to be visible.
Adjusting font size using the "Figure settings..." button from 10 to 2, does
not reduce the font size, in fact increases it, and when returning to the
"Figure settings..." button menu the font size has been returned to 10."

ALL THREE SYMPTOMS WERE ONE CONTROL, and the cause is what it did not reach.
Measured on a volcano-shaped figure: 23 text objects, 20 reached, and the
three missed were an ANNOTATION at 22pt (a gene name -- the largest text on
the plot), the SUPTITLE, and the LEGEND'S TITLE. Each lives in a different
container, which is how all three were missed at once.

Shrinking "all text" therefore shrank everything except the biggest thing on
the figure, which then dominated it -- and that reads exactly as "in fact
increases it". The volcano annotates its hits by name, so this is the common
case.
"""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.qt.widgets.figure_settings import (  # noqa: E402
    _current_text_size, _every_text)


@pytest.fixture
def volcano():
    """The shape that produced the report: labelled hits, suptitle, legend."""
    figure, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [1, 0], label="x")
    ax.set_xlabel("coefficient")
    ax.set_ylabel("-log10(p)")
    ax.set_title("volcano")
    figure.suptitle("a run")
    ax.annotate("EAF1", (0.5, 0.5), fontsize=22)
    ax.annotate("TSG101", (0.3, 0.7), fontsize=22)
    ax.legend(title="condition")
    yield figure
    plt.close(figure)


def _sizes(figure):
    return [t.get_fontsize() for t in _every_text(figure)
            if str(t.get_text()).strip()]


# --------------------------------------------------------------------------- #
#  What it reaches
# --------------------------------------------------------------------------- #

def test_an_annotation_is_reached(volcano):
    """THE ONE THAT CAUSED THE REPORT. On the volcano these are the gene
    names, and they are the largest text on the plot."""
    reached = {id(t) for t in _every_text(volcano)}

    for axis in volcano.axes:
        for note in axis.texts:
            assert id(note) in reached, note.get_text()


def test_the_suptitle_is_reached(volcano):
    reached = {str(t.get_text()) for t in _every_text(volcano)}

    assert "a run" in reached


def test_the_legend_title_is_reached(volcano):
    """It is not among the legend's own `get_texts()`, which is how it was
    missed while every other legend entry was found."""
    reached = {str(t.get_text()) for t in _every_text(volcano)}

    assert "condition" in reached


def test_nothing_on_the_figure_is_left_behind(volcano):
    """The whole claim, in one assertion: set every size and nothing is
    bigger than what was asked for."""
    for item in _every_text(volcano):
        item.set_fontsize(3)

    assert max(_sizes(volcano)) == 3.0


def test_shrinking_does_not_leave_a_giant_behind(volcano):
    """The reported symptom, reproduced as a property: after asking for a
    small font, the LARGEST text must not still be the annotation."""
    before = max(_sizes(volcano))
    for item in _every_text(volcano):
        item.set_fontsize(4)

    assert before == 22.0, "fixture no longer reproduces the report"
    assert max(_sizes(volcano)) == 4.0


# --------------------------------------------------------------------------- #
#  What it opens at
# --------------------------------------------------------------------------- #

def test_it_opens_at_the_figure_s_own_size(volcano):
    """It opened at a hardcoded 10 whatever the figure used -- the third
    symptom, "the font size has been returned to 10". It had never left 10;
    it had never read the figure."""
    for item in _every_text(volcano):
        item.set_fontsize(7)

    assert _current_text_size(volcano) == 7


def test_the_mode_wins_not_the_largest(volcano):
    """A figure has many tick labels at the body size and one or two headings
    above it. "The font size of this figure" means the body size to a reader,
    so a single 22pt heading must not set the control to 22."""
    assert _current_text_size(volcano) < 22


def test_a_figure_with_no_text_falls_back(volcano):
    figure = plt.figure()
    try:
        assert _current_text_size(figure, default=11) == 11
    finally:
        plt.close(figure)


def test_blank_text_does_not_count(volcano):
    """An empty title is a Text object at some size that nobody can see, and
    a figure whose control opened at the size of its invisible text would be
    wrong for every visible thing on it."""
    figure, ax = plt.subplots()
    try:
        ax.set_title("")
        ax.title.set_fontsize(40)
        for label in ax.get_xticklabels():
            label.set_fontsize(9)
        assert _current_text_size(figure) == 9
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The range
# --------------------------------------------------------------------------- #

def test_two_is_reachable():
    """2 is what the reporter typed. A 2pt font is unreadable and that is
    their business; a control that silently clamps is one that lies about
    what it did."""
    import inspect

    from spacr.qt.widgets import figure_settings

    source = inspect.getsource(figure_settings.FigureSettingsDialog)
    assert "all_text.setRange(2," in source, "2 is still not reachable"
