"""What the graph specification does with an empty, unknown or oversized ask.

The spec turns dropped columns into a chart. Every branch here is one where
there is less to draw than the caller assumed -- no columns dropped, a facet
column that is not in the table, a grid larger than the page, an empty frame
under a brush -- and each has to answer with something a renderer can draw
rather than a plausible-looking chart of the wrong thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.graph_spec import (
    BAR, CHANNELS, HISTOGRAM, GraphSpec, Scales, SpecError, _category_levels,
    _count_limit, _levels, _numeric, brush_mask, column_kinds, facet_grid,
    scales_for)


def _frame(n=24):
    generator = np.random.default_rng(0)
    return pd.DataFrame({
        "area": generator.normal(100.0, 20.0, n),
        "intensity": generator.normal(500.0, 60.0, n),
        "plateID": [f"p{i % 4 + 1}" for i in range(n)],
        "wellID": [f"w{i % 6 + 1}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# The channels
# ---------------------------------------------------------------------------

def test_an_unknown_channel_is_named_along_with_the_real_ones():
    """A drop zone that does not exist cannot hold a column.

    Returning None instead would make a typo in a channel name read as "that
    zone is empty", and the chart would silently lose the column.
    """
    with pytest.raises(SpecError) as excinfo:
        GraphSpec(x="area").column_for("depth")
    message = str(excinfo.value)
    assert "depth" in message
    for channel in CHANNELS:
        assert channel in message


def test_clearing_a_columns_role_restores_the_rule():
    """``None`` removes the override rather than storing it as a role.

    A stored None would be an override that says "treat this column as
    nothing", and the automatic continuous/categorical rule would never run
    for it again.
    """
    spec = GraphSpec(x="plateID").with_role("plateID", "continuous")
    assert spec.roles == {"plateID": "continuous"}

    assert spec.with_role("plateID", None).roles == {}


def test_the_used_columns_are_listed_once_each_in_channel_order():
    """One column on two channels is still one column to read from the table.

    The list drives the column subset that is loaded; a duplicate would ask
    the frame for the same column twice, and channel order is what makes the
    subset deterministic.
    """
    spec = GraphSpec(x="area", y="area", colour="plateID")

    assert spec.used_columns() == ("area", "plateID")


def test_a_spec_with_nothing_dropped_describes_itself_as_empty():
    """The caption has to say there is no chart yet, not name a plot type.

    Inferring a kind from no columns would put "scatter" under a blank
    canvas.
    """
    assert GraphSpec().describe() == "nothing dropped yet"
    assert GraphSpec().is_empty


# ---------------------------------------------------------------------------
# Facets
# ---------------------------------------------------------------------------

def test_a_facet_column_that_is_not_in_the_table_is_refused():
    """Faceting by a column the frame has not got has no levels to split on.

    Falling back to one panel would draw a grid that silently ignored the
    channel the user dropped a column onto.
    """
    frame = _frame()

    with pytest.raises(SpecError) as excinfo:
        _levels(frame, "treatment", 12)
    message = str(excinfo.value)
    assert "treatment" in message
    assert str(len(frame.columns)) in message


def test_a_grid_larger_than_the_page_loses_columns_before_rows():
    """A grid is read down the page, so a lost column costs less than a row.

    The cap has to be enforced somewhere; taking it off the rows would make a
    tall grid lose the levels a reader scrolls to, which are the ones they
    were looking for.
    """
    frame = _frame(n=48)

    grid = facet_grid(frame, GraphSpec(x="area", facet_row="plateID",
                                       facet_col="wellID"),
                      max_panels=8)

    rows_lost = 4 - len(grid.row_levels)
    cols_lost = 6 - len(grid.col_levels)
    assert len(grid.row_levels) * len(grid.col_levels) <= 8
    assert cols_lost > rows_lost, "the columns axis absorbed most of the cap"
    assert "capped at 8 panels" in grid.notice
    assert grid.hidden_rows > 0, "the rows outside the drawn levels are counted"


# ---------------------------------------------------------------------------
# Shared scales
# ---------------------------------------------------------------------------

def test_a_channel_with_no_column_has_no_numbers_to_bound():
    """An empty channel is not a column of NaN; it has no values at all.

    Returning an empty array instead would make the limits ``(nan, nan)`` and
    every panel would be drawn on an axis nobody can read.
    """
    frame = _frame()

    assert _numeric(frame, None) is None
    assert _numeric(frame, "not_a_column") is None
    assert _category_levels(frame, None) is None
    assert _category_levels(frame, "not_a_column") is None


def test_a_categorical_axis_gets_the_same_tick_for_the_same_level():
    """Levels line up across panels, or a grid compares different things.

    Each panel would otherwise number its own levels from zero, so ``p1`` in
    one panel and ``p3`` in the next would sit on the same tick.
    """
    frame = _frame()
    scales = scales_for(frame, GraphSpec(x="plateID"), column_kinds(frame))

    positions = scales.x_positions()
    assert positions == {"p1": 0, "p2": 1, "p3": 2, "p4": 3}
    assert scales.y_positions() is None


def test_a_continuous_axis_has_no_level_positions():
    """A number is at its own value, not at a tick index."""
    frame = _frame()
    scales = scales_for(frame, GraphSpec(x="area", y="intensity"),
                        column_kinds(frame))

    assert scales.x_positions() is None
    assert scales.y_positions() is None


def test_a_count_axis_over_a_column_that_is_not_there_has_no_height():
    """Without the counted column there is no tallest bar to share.

    A zero limit would flatten every panel's count axis onto the baseline.
    """
    frame = _frame()
    spec = GraphSpec(x="not_a_column")
    grid = facet_grid(frame, spec)

    assert _count_limit(frame, spec, grid, BAR, None, None) is None


def test_panels_that_hold_no_rows_contribute_no_bars():
    """A combination of facet levels that drew nothing has no bar to measure.

    Such panels are drawn empty rather than closed up, so the shared count
    axis has to skip them; asking an empty panel for its tallest bar reduces
    over nothing and would poison the maximum with NaN.
    """
    frame = _frame(n=24)
    spec = GraphSpec(x="plateID", facet_row="wellID", facet_col="plateID")
    grid = facet_grid(frame, spec)
    assert any(panel.is_empty for panel in grid.panels), (
        "the fixture must produce at least one empty panel")

    tallest = _count_limit(frame, spec, grid, BAR, None, None)

    assert tallest is not None
    assert tallest > 1.0


def test_a_histogram_with_no_bins_has_no_shared_count():
    """Without bin edges there is nothing to count rows into.

    The edges are None exactly when the column held no finite value, so the
    honest answer is that this figure has no shared count axis -- a zero
    would draw every panel flat against the baseline.
    """
    frame = _frame()
    spec = GraphSpec(x="area")
    grid = facet_grid(frame, spec)

    assert _count_limit(frame, spec, grid, HISTOGRAM, None, None) is None


def test_a_categorical_y_axis_gets_its_own_tick_positions():
    """A level on y is at a tick, the same tick, in every panel.

    Numbering per panel would put two different groups on one row of a
    faceted box plot and make the panels uncomparable.
    """
    scales = Scales(y_levels=("p1", "p2", "p3"))

    assert scales.y_positions() == {"p1": 0, "p2": 1, "p3": 2}


def test_a_shared_count_axis_is_the_tallest_bar_anywhere():
    """The count axis must hold the biggest panel, with room above it.

    An axis fitted to one panel would clip the bars of another, and a grid
    whose panels are drawn to different scales cannot be compared at all.
    """
    frame = _frame(n=48)
    spec = GraphSpec(x="plateID", facet_row="wellID")
    kinds = column_kinds(frame)
    grid = facet_grid(frame, spec)

    scales = scales_for(frame, spec, kinds, grid=grid)

    assert scales.count_limit is not None
    assert scales.count_limit > 1.0


# ---------------------------------------------------------------------------
# Brushing
# ---------------------------------------------------------------------------

def test_brushing_an_empty_table_selects_nothing_without_indexing_it():
    """A drag over a panel with no rows has nothing to select.

    The mask still has to be a bool array of the frame's length, so the
    caller can index with it either way; computing the axes first would
    reduce over an empty column and produce NaN bounds.
    """
    frame = _frame().iloc[0:0]
    spec = GraphSpec(x="area", y="intensity")

    mask = brush_mask(frame, spec, column_kinds(_frame()),
                      0.0, 0.0, 1000.0, 1000.0)

    assert mask.dtype == bool
    assert mask.size == 0
