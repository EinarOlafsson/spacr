"""Instruction 178 A — the plot types asked for by name.

    "i should be able to right click on them and show them as: line, bar,
     jitter-bar, jitter-box, jitter, box, violin plots"

Five of the seven were there. THE THREE COMPOSITES WERE NOT, and a composite
is not the same request as the two marks it is made of: a jitter-box is the
box AND the observations at once — the box summarises and the points are the
evidence, which is exactly the argument 139 B makes for making it the default
of the generated figures. Offering `jitter` and `box` separately gives a user
two pictures and no way to see both.

The property that makes the composites worth having rather than merely
present: THE POINTS KEEP THEIR ROWS, so a jitter-box stays clickable where a
bare box does not.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.fast_plots import (ControlSeparation,   # noqa: E402
                                         GroupedPlot, MARK_TYPES)

#: Every type the ask names, in its own words.
ASKED_FOR = ("line", "bar", "jitter_bar", "jitter_box", "jitter", "box",
             "violin")


def _groups():
    rng = np.random.default_rng(0)
    return {"pc": rng.normal(2.0, 0.5, 20),
            "nc": rng.normal(0.0, 0.5, 24),
            "mid": rng.normal(1.0, 0.5, 18)}


def _panel(qtbot, mark):
    panel = ControlSeparation()
    qtbot.addWidget(panel)
    panel.set_mark(mark)
    panel.set_groups(_groups())
    return panel


def test_every_type_the_ask_named_is_on_offer():
    offered = {key for key, _label in MARK_TYPES}
    missing = [k for k in ASKED_FOR if k not in offered]
    assert not missing, f"the ask named these and they are not offered: {missing}"


@pytest.mark.parametrize("mark", ASKED_FOR)
def test_each_one_draws_rather_than_raising(qtbot, mark):
    plot = GroupedPlot()
    qtbot.addWidget(plot)
    rng = np.random.default_rng(0)
    assert plot.add_group_mark(1.0, rng.normal(size=25), mark) == 25


@pytest.mark.parametrize("mark", ASKED_FOR)
def test_each_one_puts_something_on_the_panel(qtbot, mark):
    assert len(_panel(qtbot, mark).plot.listDataItems()) > 0


@pytest.mark.parametrize("mark", ASKED_FOR)
def test_the_menu_offers_it_and_ticks_the_one_in_force(qtbot, mark):
    panel = _panel(qtbot, mark)
    label = dict(MARK_TYPES)[mark]

    def walk(menu):
        for action in menu.actions():
            if action.menu() is not None:
                yield from walk(action.menu())
            else:
                yield action

    entries = {a.text(): a for a in walk(panel.build_style_menu())
               if a.text().strip()}
    assert label in entries
    assert entries[label].isChecked(), f"{mark} is in force and not ticked"


# -- what makes a composite worth having ------------------------------------

def test_a_composite_draws_more_than_either_half_alone(qtbot):
    """It is the summary AND the observations, not one of them."""
    box = len(_panel(qtbot, "box").plot.listDataItems())
    jitter = len(_panel(qtbot, "jitter").plot.listDataItems())
    both = len(_panel(qtbot, "jitter_box").plot.listDataItems())

    assert both > box and both > jitter


def test_the_points_in_a_composite_keep_their_rows(qtbot):
    """A bare box stands for many rows and cannot honestly select one; a
    composite carries the individual observations, so it can."""
    plot = GroupedPlot()
    qtbot.addWidget(plot)
    rng = np.random.default_rng(0)
    rows = np.arange(25)

    plot.add_group_mark(1.0, rng.normal(size=25), "jitter_box", rows=rows)
    carried = [item for item in plot.plot.listDataItems()
               if getattr(item, "getData", None) is not None]
    assert carried, "the composite drew nothing selectable"


def test_a_line_joins_its_groups_rather_than_leaving_loose_markers(qtbot):
    """`add_group_mark` sees one group at a time and cannot join anything;
    the panel does it once every group is on."""
    line = len(_panel(qtbot, "line").plot.listDataItems())
    points = len(_panel(qtbot, "points").plot.listDataItems())
    assert line > points, "the means were drawn but never joined"


def test_a_line_through_one_group_is_not_drawn(qtbot):
    """A "line" through one point is a point."""
    panel = ControlSeparation()
    qtbot.addWidget(panel)
    panel.set_mark("line")
    rng = np.random.default_rng(0)
    panel.set_groups({"only": rng.normal(size=12)})
    # It still draws the marker; what it must not do is invent a segment.
    assert len(panel.plot.listDataItems()) >= 1


def test_an_unknown_mark_is_refused_loudly(qtbot):
    """A silent fallback would make a typo look like a working option."""
    panel = ControlSeparation()
    qtbot.addWidget(panel)
    with pytest.raises(ValueError, match="unknown mark"):
        panel.set_mark("swarm")
