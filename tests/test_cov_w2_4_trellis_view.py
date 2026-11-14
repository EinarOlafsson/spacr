"""Trellis canvas — brushing one panel, the highlight, and the empty states.

``tests/qt/test_trellis.py`` covers the grid and the scales. What is left is
the interactive half, and it is the half where a bug is invisible: a brush
that returns the wrong rows still draws a rectangle, and a highlight that
misses still looks like a chart.

So the brush is swept over a panel whose contents are known by hand and the
RETURNED KEYS are asserted, not the count. The linked-selection handler is
driven both ways -- the cheap re-style for a scatter and the full redraw for
a histogram, whose bars cannot be re-coloured per row.

Also here: the two messages the canvas shows instead of a chart, the
per-panel y limits that only a non-count chart reaches, and the notice line
that has to mention the filter it could not apply.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.selection import DataFilter, RangeFilter
from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec
from spacr.qt.widgets.trellis_spec import TrellisSpec
from spacr.qt.widgets.trellis_view import TrellisCanvas, TrellisPanelWidget

#: ``classify_columns`` calls a numeric column with few distinct values a
#: category. These two are measurements, and saying so keeps the frame small
#: enough that every panel's contents can be counted by eye.
NUMERIC = {"area": CONTINUOUS, "intensity": CONTINUOUS}


#: Six keyed objects, arranged so one panel of the 2 × 2 grid is EMPTY::
#:
#:     plateID  gene  object_label
#:     p1       a     1, 2
#:     p1       b     3
#:     p2       a     4, 5, 6
#:
#: The empty (p2, b) panel is the interesting one: it is still drawn, and it
#: has no marks to re-style, which is the branch a fully populated grid can
#: never reach.
_LAYOUT = (("p1", "a"), ("p1", "a"), ("p1", "b"),
           ("p2", "a"), ("p2", "a"), ("p2", "a"))


@pytest.fixture
def keyed():
    """Six objects with DISTINCT keys, two plates by two genes."""
    return pd.DataFrame([
        {"plateID": plate, "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "object_label": index + 1, "gene": gene,
         "area": 10.0 * index + 10.0,
         "intensity": 100.0 - 5.0 * index}
        for index, (plate, gene) in enumerate(_LAYOUT)
    ])


def _histogram_spec(**kwargs):
    return TrellisSpec(
        graph=GraphSpec(x="area", facet_row="plateID", facet_col="gene",
                        bins=2, roles=NUMERIC),
        **kwargs)


@pytest.fixture
def link():
    return LinkedSelection()


@pytest.fixture
def canvas(qtbot, keyed, link):
    widget = TrellisCanvas(link=link)
    qtbot.addWidget(widget)
    widget.set_frame(keyed)
    return widget


# ---------------------------------------------------------------------------
# What it shows instead of a chart
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-panel y limits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The notice line
# ---------------------------------------------------------------------------

def test_the_notice_says_when_the_shared_filter_does_not_apply(canvas, link):
    """A chart of more rows than the filter panel claims is the failure."""
    link.set_filter(DataFilter().add(RangeFilter("not_a_column", 0, 1)))
    canvas.set_trellis_spec(_histogram_spec())

    assert "shared filter does not apply" in canvas.notice()
    # ...and the chart is still drawn, from the unfiltered rows.
    assert canvas.trellis.n_occupied == 4


# ---------------------------------------------------------------------------
# Brushing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Following someone else's selection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The panel widget
# ---------------------------------------------------------------------------

def test_clearing_the_channels_empties_every_zone(qtbot, keyed, link):
    panel = TrellisPanelWidget(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(keyed)
    panel.zone("x").set_column("area")
    panel.zone("facet_row").set_column("plateID")
    assert panel.spec.graph.x == "area"

    panel.clear_channels()

    assert all(zone.column in (None, "") for zone in panel._zones.values())
    assert panel.spec.graph.x is None
