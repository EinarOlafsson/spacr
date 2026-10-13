"""Instruction 149 F.5 and F.6: the last two ways of showing the adjusted p.

    "id like the user to have access to visualizing adjusted P in all the ways
     that are acceptable to the field. showing as color, showing the descrete
     P on the axis buy showing the line where the adjusted p threshold lands,
     etc."

F.1 to F.4 shipped: the raw p on a continuous axis with a binary call in the
colour, the exact BH critical line, the stepped adjusted axis kept, and the
local FDR. The two that were left are the two that had to be reasoned about
rather than merely built:

F.5  A CONTINUOUS COLOUR RAMP OVER q. It CONFLICTS with the colouring already
     on the plot -- a dot cannot be coloured for its condition and for its q
     at once -- so whichever is chosen the other is not shown, and the caption
     says which is in force. That is the localisation rule: one sentence per
     figure.

F.6  SIZE OR OPACITY BY q. The one encoding that COMPOSES rather than
     competing, because size and opacity are channels the colour is not using.

Both are OFFERS, not defaults, and this file asserts that first: a volcano
opened cold is still F.1's, because F.1 is the field's own volcano and the
ramp answers a different question.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

from spacr.multiple_testing import adjust_p_values

ALPHA = 0.05


def _frame(n: int = 200, hits: int = 12, seed: int = 3,
           condition: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 1, n - hits),
                        10.0 ** (-rng.uniform(4, 8, hits))])
    q, _ = adjust_p_values(p, method="fdr_bh", alpha=ALPHA)
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{100000 + i}_{i % 4 + 1}]"
                    for i in range(p.size)],
        "coefficient": rng.normal(0, 1, p.size),
        "p_value": p, "q_value": q,
        "multiple_testing_method": "fdr_bh"})
    if condition:
        frame["condition"] = ["a", "b"] * (p.size // 2)
    return frame


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame())
    return plot


def _entries(plot) -> list:
    from spacr.qt.widgets.fast_plots import menu_entries

    return [action.text() for action in menu_entries(plot.build_style_menu())]


def _action(plot, fragment: str):
    from spacr.qt.widgets.fast_plots import menu_entries

    for action in menu_entries(plot.build_style_menu()):
        if fragment in action.text():
            return action
    raise AssertionError(f"no entry containing {fragment!r}: {_entries(plot)}")


def _brushes(plot) -> list:
    """Every point's brush, in drawn order."""
    out = []
    for item in plot._scatter_items():
        out += [point.brush() for point in item.points()]
    return out


def _sizes(plot) -> list:
    out = []
    for item in plot._scatter_items():
        out += [float(point.size()) for point in item.points()]
    return out


# --------------------------------------------------------------------------- #
#  Both are offers
# --------------------------------------------------------------------------- #

def test_a_volcano_opens_on_the_binary_call(volcano):
    """F.1 keeps the colour channel. Neither of these is a default."""
    assert volcano.q_colour() == "call"
    assert volcano.q_mark() == "none"


def test_both_are_on_the_plots_own_menu(volcano):
    entries = _entries(volcano)
    assert any("continuous ramp over q" in text for text in entries), entries
    assert any("size by q" in text for text in entries), entries
    assert any("opacity by q" in text for text in entries), entries


def test_an_unknown_encoding_is_refused_rather_than_ignored(volcano):
    with pytest.raises(ValueError):
        volcano.set_q_colour("rainbow")
    with pytest.raises(ValueError):
        volcano.set_q_mark("wobble")


# --------------------------------------------------------------------------- #
#  F.5 -- the ramp, and what it costs
# --------------------------------------------------------------------------- #

def test_the_ramp_gives_more_than_two_colours(volcano):
    """The binary call is two brushes by construction. A ramp that produced
    two would be the call wearing a colormap."""
    before = {brush.color().rgba() for brush in _brushes(volcano)}
    assert len(before) == 2, before

    volcano.set_q_colour("ramp")
    after = {brush.color().rgba() for brush in _brushes(volcano)}
    assert len(after) > 2, after


def test_the_ramp_is_ordered_by_the_evidence_not_by_the_row(volcano):
    """The brightest colour belongs to the smallest q. Checked through the
    lookup rather than by eye: a ramp that ran the other way would still
    "have many colours"."""
    volcano.set_q_colour("ramp")
    q = volcano._q_values
    brushes = _brushes(volcano)
    strongest = int(np.nanargmin(q))
    weakest = int(np.nanargmax(q))
    # viridis: bright end is high in green and red, dark end is not.
    strong = brushes[strongest].color()
    weak = brushes[weakest].color()
    assert strong.green() > weak.green()


def test_the_ramp_does_not_build_a_brush_per_point(volcano):
    """The whole reason this module exists. 1,215 QBrush objects cost 39.5 ms
    against 3.5 ms for a reused set, so the ramp is quantised to Q_STOPS."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    volcano.set_q_colour("ramp")
    distinct = {brush.color().rgba() for brush in _brushes(volcano)}
    assert len(distinct) <= VolcanoPlot.Q_STOPS, len(distinct)


def test_the_ramp_says_what_it_took_the_colour_from(volcano):
    """A dot cannot be coloured for two things, so the one that lost has to
    be named -- silently dropping it is how a q ramp is read as a condition."""
    volcano.set_q_colour("ramp")
    caption = volcano.caption()
    assert "continuous ramp over q" in caption
    assert "called/not-called colouring is off" in caption


def test_the_ramp_names_the_category_it_displaced(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(condition=True), category_column="condition")
    plot.set_q_colour("ramp")
    assert "the condition colouring is off" in plot.caption(), plot.caption()


def test_going_back_to_the_call_restores_the_two_brushes(volcano):
    volcano.set_q_colour("ramp")
    volcano.set_q_colour("call")
    assert len({brush.color().rgba() for brush in _brushes(volcano)}) == 2
    assert "continuous ramp" not in volcano.caption()


def test_the_ramps_legend_names_q_values_and_is_not_the_call(volcano):
    volcano.set_q_colour("ramp")
    labels = list(volcano._legend_colours)
    assert labels and all(label.startswith("q ") for label in labels), labels
    assert not any("called" in label for label in labels)


def test_one_q_for_the_whole_screen_greys_the_ramp(qtbot):
    """The staircase at its limit. A ramp over one number is one colour, and
    an entry that looks live and does nothing is what 106 forbids."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    frame = _frame(n=30, hits=0, seed=11)
    frame["p_value"] = 0.5
    frame["q_value"] = 0.5
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(frame)

    action = _action(plot, "continuous ramp over q")
    assert not action.isEnabled()
    assert "one colour" in action.text()


# --------------------------------------------------------------------------- #
#  F.6 -- the encoding that composes
# --------------------------------------------------------------------------- #

def test_size_by_q_gives_every_point_its_own_diameter(volcano):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    assert len(set(_sizes(volcano))) == 1, "a cold volcano is one size"
    volcano.set_q_mark("size")
    sizes = _sizes(volcano)
    assert len(set(sizes)) > 2
    smallest, largest = VolcanoPlot.Q_SIZE_RANGE
    assert min(sizes) >= smallest - 1e-9
    assert max(sizes) <= largest + 1e-9


def test_the_biggest_mark_is_the_smallest_q(volcano):
    volcano.set_q_mark("size")
    q = volcano._q_values
    sizes = np.asarray(_sizes(volcano))
    assert sizes[int(np.nanargmin(q))] > sizes[int(np.nanargmax(q))]


def test_size_composes_with_a_category_colouring(qtbot):
    """THE PROPERTY IT WAS SINGLED OUT FOR. The ramp cannot do this: colour
    keeps saying "condition" while the mark says q."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(condition=True), category_column="condition")
    coloured = {brush.color().rgba() for brush in _brushes(plot)}
    plot.set_q_mark("size")

    assert {brush.color().rgba() for brush in _brushes(plot)} == coloured
    assert len(set(_sizes(plot))) > 2
    # The legend still names the CONDITIONS, not q -- the colour channel
    # never changed hands.
    assert set(plot._legend_colours) == {"a (100)", "b (100)"}


def test_opacity_by_q_fades_the_weak_and_keeps_the_hue(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(condition=True), category_column="condition")
    plot.set_q_mark("opacity")
    q = plot._q_values
    brushes = _brushes(plot)
    strongest = brushes[int(np.nanargmin(q))].color()
    weakest = brushes[int(np.nanargmax(q))].color()

    assert strongest.alpha() > weakest.alpha()
    assert weakest.alpha() >= VolcanoPlot.Q_ALPHA_RANGE[0]
    # THE HUE IS UNTOUCHED: only the alpha was replaced, which is what makes
    # this compose. Both conditions are still on the plot as themselves.
    hues = {(brush.color().red(), brush.color().green(),
             brush.color().blue()) for brush in brushes}
    assert len(hues) == 2, hues


def test_a_faded_point_is_never_invisible(volcano):
    """A point faded to nothing is a point removed from the plot, and this
    instruction exists because a plot was showing something the data did not
    say."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    volcano.set_q_mark("opacity")
    assert min(brush.color().alpha() for brush in _brushes(volcano)) >= \
        VolcanoPlot.Q_ALPHA_RANGE[0]


def test_opacity_does_not_build_a_brush_per_point(volcano):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    volcano.set_q_mark("opacity")
    distinct = {brush.color().rgba() for brush in _brushes(volcano)}
    assert len(distinct) <= 2 * VolcanoPlot.Q_STOPS, len(distinct)


def test_the_mark_says_so_in_the_caption(volcano):
    volcano.set_q_mark("size")
    assert "Point SIZE" in volcano.caption()
    volcano.set_q_mark("opacity")
    assert "Point OPACITY" in volcano.caption()
    volcano.set_q_mark("none")
    assert "Point SIZE" not in volcano.caption()
    assert "Point OPACITY" not in volcano.caption()


def test_the_ramp_and_the_size_are_on_at_once(volcano):
    """The composition, driven rather than argued: both encodings live, and
    the caption carries both sentences."""
    volcano.set_q_colour("ramp")
    volcano.set_q_mark("size")
    assert len({brush.color().rgba() for brush in _brushes(volcano)}) > 2
    assert len(set(_sizes(volcano))) > 2
    caption = volcano.caption()
    assert "continuous ramp over q" in caption and "Point SIZE" in caption


# --------------------------------------------------------------------------- #
#  And the rule the whole instruction rests on
# --------------------------------------------------------------------------- #

def test_no_encoding_moves_a_point(volcano):
    """WHAT NOT TO OFFER: jitter on the y axis. Every one of these six ways
    changes how a mark is DRAWN and none of them changes where it is."""
    before = {}
    for item in volcano._scatter_items():
        for point in item.points():
            before[point.data()] = (point.pos().x(), point.pos().y())
    for colour in ("ramp", "call"):
        for mark in ("size", "opacity", "none"):
            volcano.set_q_colour(colour)
            volcano.set_q_mark(mark)
            after = {}
            for item in volcano._scatter_items():
                for point in item.points():
                    after[point.data()] = (point.pos().x(), point.pos().y())
            assert after == before, f"{colour}/{mark} moved a point"
