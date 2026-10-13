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

from spacr.selection import DataFilter, RangeFilter, Selection
from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec
from spacr.qt.widgets.trellis_spec import SCALE_FREE, TrellisSpec
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


def _scatter_spec(**kwargs):
    return TrellisSpec(
        graph=GraphSpec(x="area", y="intensity", facet_row="plateID",
                        facet_col="gene", roles=NUMERIC),
        **kwargs)


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

def test_a_canvas_with_no_table_asks_for_one(qtbot, link):
    widget = TrellisCanvas(link=link)
    qtbot.addWidget(widget)
    widget.set_trellis_spec(_histogram_spec())

    assert widget.trellis is None
    assert widget.panel_axes() == {}
    assert any("Load a table" in text.get_text()
               for text in widget.figure().axes[0].texts)


def test_an_empty_table_asks_for_one_too(qtbot, keyed, link):
    widget = TrellisCanvas(link=link)
    qtbot.addWidget(widget)
    widget.set_frame(keyed.iloc[0:0])
    widget.set_trellis_spec(_histogram_spec())

    assert widget.trellis is None
    assert any("Load a table" in text.get_text()
               for text in widget.figure().axes[0].texts)


def test_a_deferred_spec_change_waits_for_the_debounce(canvas):
    """Typing in a spin box must not redraw on every keystroke."""
    canvas.set_trellis_spec(_histogram_spec())
    drawn = canvas.trellis

    canvas.set_trellis_spec(_histogram_spec(wrap=1), immediate=False)

    assert canvas._debounce.isActive() is True
    # Nothing has been redrawn yet -- the previous grid is still on screen.
    assert canvas.trellis is drawn
    canvas.render_now()
    assert canvas._debounce.isActive() is False


# ---------------------------------------------------------------------------
# Per-panel y limits
# ---------------------------------------------------------------------------

def test_a_scatter_writes_its_own_y_limits_per_panel(canvas):
    """Only a non-count chart has a y scale to set; free mode differs it."""
    canvas.set_trellis_spec(_scatter_spec(scale_y=SCALE_FREE))
    limits = {ax.get_ylim() for ax in canvas.panel_axes().values()}
    assert len(limits) > 1


def test_a_shared_scatter_hands_every_panel_the_same_y_limits(canvas):
    canvas.set_trellis_spec(_scatter_spec())
    limits = {ax.get_ylim() for ax in canvas.panel_axes().values()}
    assert len(limits) == 1


def test_a_categorical_y_axis_gets_ticks_at_the_level_positions(canvas):
    """Levels, not numbers: a category axis with 0.0/0.5/1.0 says nothing."""
    canvas.set_trellis_spec(TrellisSpec(
        graph=GraphSpec(x="area", y="gene", facet_row="plateID",
                        roles=NUMERIC)))
    ax = canvas.panel_axes()[(0, 0)]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) >= {"a", "b"}
    assert ax.get_ylim() == pytest.approx((-0.6, len(labels) - 0.4))


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


def test_the_notice_counts_the_highlighted_rows(canvas, keyed, link):
    canvas.set_trellis_spec(_scatter_spec())
    link.set_selection(Selection.from_frame(keyed.iloc[:3], source="test"))
    canvas.render_now()

    assert "3 highlighted" in canvas.notice()


# ---------------------------------------------------------------------------
# Brushing
# ---------------------------------------------------------------------------

def test_brushing_a_panel_returns_that_panels_rows_only(canvas, keyed):
    canvas.set_trellis_spec(_scatter_spec())

    picked = canvas.brush(-1e9, -1e9, 1e9, 1e9, row=0, col=0, publish=False)

    assert picked is not None
    # Panel (p1, a) holds the first two objects and nothing else.
    expected = Selection.from_frame(keyed.iloc[:2], source=canvas.link_source)
    assert set(picked.keys) == set(expected.keys)


def test_brushing_publishes_to_every_linked_view_by_default(canvas, link):
    canvas.set_trellis_spec(_scatter_spec())
    seen = []
    link.selection_changed.connect(lambda: seen.append(link.selection))

    published = canvas.brush(-1e9, -1e9, 1e9, 1e9, row=1, col=0)

    assert published is link.selection
    # Panel (p2, a) holds three objects.
    assert seen and len(link.selection.keys) == 3


def test_brushing_a_panel_the_grid_does_not_have_selects_nothing(canvas):
    canvas.set_trellis_spec(_scatter_spec())
    assert canvas.brush(0, 0, 1e9, 1e9, row=9, col=9, publish=False) is None


def test_brushing_before_anything_is_drawn_selects_nothing(canvas):
    canvas.set_trellis_spec(TrellisSpec())
    assert canvas.trellis is None
    assert canvas.brush(0, 0, 1, 1, publish=False) is None


def test_a_table_with_no_object_keys_cannot_brush(qtbot, link):
    """Said out loud in the notice rather than publishing an empty selection."""
    frame = pd.DataFrame({"gene": ["a", "a", "b", "b"],
                          "area": [1.0, 2.0, 3.0, 4.0],
                          "intensity": [4.0, 3.0, 2.0, 1.0]})
    widget = TrellisCanvas(link=link)
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    widget.set_trellis_spec(TrellisSpec(
        graph=GraphSpec(x="area", y="intensity", facet_col="gene",
                        roles=NUMERIC)))

    assert widget.brush(-1e9, -1e9, 1e9, 1e9, publish=False) is None
    assert "no object keys" in widget.notice()


# ---------------------------------------------------------------------------
# Following someone else's selection
# ---------------------------------------------------------------------------

def test_a_scatter_restyles_its_marks_without_a_redraw(canvas, keyed, link):
    """The empty panel is in the grid too, and it has nothing to re-style."""
    canvas.set_trellis_spec(_scatter_spec())
    axes_before = canvas.panel_axes()
    assert canvas.trellis.n_empty == 1

    link.set_selection(Selection.from_frame(keyed.iloc[2:4], source="other"))

    # Same Axes objects: the grid was re-styled, not rebuilt.
    assert canvas.panel_axes() == axes_before
    assert canvas.selected_count() == 2
    assert "2 highlighted" in canvas.notice()


def test_a_histogram_redraws_because_bars_cannot_be_restyled(canvas, keyed,
                                                             link):
    canvas.set_trellis_spec(_histogram_spec())
    axes_before = dict(canvas.panel_axes())

    link.set_selection(Selection.from_frame(keyed.iloc[:2], source="other"))

    assert canvas.panel_axes() != axes_before
    assert canvas.trellis is not None


def test_a_selection_arriving_before_a_chart_is_ignored(canvas, keyed, link):
    canvas.set_trellis_spec(TrellisSpec())
    assert canvas.trellis is None

    link.set_selection(Selection.from_frame(keyed.iloc[:1], source="other"))

    assert canvas.trellis is None
    assert canvas.panel_axes() == {}


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
