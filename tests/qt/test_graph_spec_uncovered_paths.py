"""Degenerate inputs to the spec: the grid, the levels and the count axis.

Four of them, each a shape the engine is handed rather than one it chooses:

* a panel ceiling so low that neither facet axis can give anything up. The
  trim is a loop over a condition the trim itself is supposed to relieve, and
  a grid is built while the canvas re-renders, on the GUI thread — a loop that
  never ends there freezes the whole application;
* a facet level spelled like a number that has no order (``NaN``), which would
  make the level sort arbitrary rather than wrong in one place;
* a facet panel with nothing finite in it, which contributes no bin to the
  shared count axis the other panels are read against;
* and a brush swept across a categorical axis, where the rectangle has to be
  matched against the levels under the ticks rather than against numbers.

:meth:`~spacr.qt.widgets.graph_spec.GraphSpec.describe` is here too: it is the
one line printed above the chart, and it has to say whether the plot kind was
inferred or pinned by the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.graph_spec import (
    BOX, CATEGORICAL, CONTINUOUS, HISTOGRAM, MAX_PANELS, SCATTER, GraphSpec,
    brush_mask, facet_grid, scales_for,
)


def _three_groups() -> pd.DataFrame:
    """Three genes and a continuous measurement, with rows enough to classify."""
    return pd.DataFrame({
        "gene": [f"g{i % 3}" for i in range(30)],
        "area": np.linspace(1.0, 30.0, 30),
    })


@pytest.fixture
def plate() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "area": rng.normal(100.0, 5.0, 24),
        "row": list("ABC") * 8,
        "column": [str(1 + i % 4) for i in range(24)],
    })


def test_a_ceiling_below_one_panel_still_returns_the_single_unfaceted_panel(
        plate):
    """Nothing to trim, so the trim stops and the whole table is one panel."""
    grid = facet_grid(plate, GraphSpec(x="area"), max_panels=0)

    assert grid.shape == (1, 1)
    assert grid.n_panels == 1
    assert grid.panel(0, 0).n == len(plate)
    assert grid.hidden_rows == 0
    assert not grid.is_faceted
    assert "capped" not in grid.notice


def test_a_ceiling_below_one_panel_collapses_a_faceted_grid_to_one_row(plate):
    """Both axes are trimmed to a single level, and then the loop lets go.

    The rows of the table that belonged to the levels that were cut are
    counted as hidden, which is what tells the reader the picture is not
    the whole table.
    """
    spec = GraphSpec(x="area", facet_row="row", facet_col="column")

    grid = facet_grid(plate, spec, max_panels=0)

    assert grid.shape == (1, 1)
    assert grid.row_levels == ("A",)
    assert grid.col_levels == ("1",)
    assert grid.is_faceted
    assert grid.hidden_rows == len(plate) - grid.panel(0, 0).n
    assert grid.hidden_rows > 0
    assert "capped at 0 panels" in grid.notice


def test_a_workable_ceiling_trims_the_columns_before_the_rows(plate):
    """The ordinary path the degenerate one is a floor under.

    A grid is read down the page, so a lost column costs the reader less
    than a lost row.
    """
    spec = GraphSpec(x="area", facet_row="row", facet_col="column")

    grid = facet_grid(plate, spec, max_panels=6)

    assert grid.shape == (3, 2)
    assert grid.row_levels == ("A", "B", "C")
    assert grid.col_levels == ("1", "2")
    assert "capped at 6 panels" in grid.notice


def test_an_unrestricted_grid_keeps_every_level_and_says_nothing_about_caps(
        plate):
    spec = GraphSpec(x="area", facet_row="row", facet_col="column")

    grid = facet_grid(plate, spec, max_panels=MAX_PANELS)

    assert grid.shape == (3, 4)
    assert grid.hidden_rows == 0
    assert grid.notice == ""
    assert sum(panel.n for panel in grid.panels) == len(plate)


def test_a_spec_describes_the_kind_it_will_draw_and_whether_it_was_pinned():
    """The one line the builder shows above the chart."""
    spec = GraphSpec(x="area", y="intensity", colour="gene", kind=SCATTER)
    kinds = {"area": CONTINUOUS, "intensity": CONTINUOUS,
             "gene": CATEGORICAL}

    assert spec.describe(kinds) == (
        "scatter (pinned) · x: area · y: intensity · colour: gene")


def test_an_inferred_kind_is_described_without_the_pinned_mark():
    """"Pinned" means the user overrode the inference, so it has to be true."""
    spec = GraphSpec(x="area")

    assert spec.describe({"area": CONTINUOUS}) == "histogram · x: area"
    assert GraphSpec().describe({}) == "nothing dropped yet"


def test_a_level_spelled_like_a_number_that_has_no_order_sorts_as_text():
    """``NaN`` parses as a float and cannot be compared, so it is text.

    A level that sorted by a NaN key would make the whole facet axis
    arbitrary — the panels would come back in a different order for the
    same table — rather than putting one panel in the wrong place.
    """
    frame = pd.DataFrame({
        "plate": ["P2", "P10", "nan", "alpha"],
        "area": [1.0, 2.0, 3.0, 4.0],
    })
    spec = GraphSpec(x="area", facet_col="plate")

    grid = facet_grid(frame, spec)

    # ``nan`` takes its place among the words, after ``alpha`` and behind the
    # prefixed plates. A numeric key would have pulled it ahead of all three,
    # because numbers sort before text whatever their value.
    assert grid.col_levels == ("P2", "P10", "alpha", "nan")
    assert facet_grid(frame, spec).col_levels == grid.col_levels


def test_an_infinite_level_still_sorts_as_the_number_it_parses_to():
    """The contrast: ``inf`` has an order, so it keeps the numeric key.

    The minus signs are what pin the whole-string comparison down. Split on
    runs of digits, ``-10`` reads as the text ``-`` followed by the number
    ``10`` and lands *after* ``-5``; compared as numbers it comes first, and
    the bare ``2`` comes after both rather than ahead of them.
    """
    frame = pd.DataFrame({
        "dose": ["2", "-5", "-10", "inf"],
        "area": [1.0, 2.0, 3.0, 4.0],
    })
    spec = GraphSpec(x="area", facet_col="dose")

    assert facet_grid(frame, spec).col_levels == ("-10", "-5", "2", "inf")


def test_a_facet_panel_with_nothing_finite_to_bin_does_not_lower_the_count_axis():
    """The shared count axis is the tallest bin anywhere, ignoring empty ones.

    A panel whose column is entirely missing has no histogram at all. Reading
    a zero out of it — or worse, letting it raise on ``counts.max()`` of an
    empty array — would flatten the axis the other panels are read against.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "area": [float("nan")] * 8 + list(rng.normal(100.0, 5.0, 40)),
        "plate": ["p1"] * 8 + ["p2"] * 40,
    })
    spec = GraphSpec(x="area", facet_col="plate", kind=HISTOGRAM)
    kinds = spec.kinds_for(frame)
    grid = facet_grid(frame, spec)

    scales = scales_for(frame, spec, kinds, grid)

    assert grid.shape == (1, 2)
    assert grid.panel(0, 0).n == 8
    assert scales.count_limit is not None
    assert scales.count_limit > 1.0


def test_a_brush_across_two_categories_selects_exactly_those_groups():
    """A categorical axis is matched by the level under the swept ticks.

    A box occupies the width of its tick, not the tick itself, so a level
    counts as swept once the rectangle reaches within half a tick of it. This
    sweep runs from 0.6 to 1.6 — between the ticks at both ends — and it has
    to take the two boxes it visibly crosses, ``g1`` and ``g2``. Matching the
    tick positions alone would drop ``g2``, whose box the drag ended inside.
    """
    frame = _three_groups()
    spec = GraphSpec(x="gene", y="area", kind=BOX)
    kinds = spec.kinds_for(frame)
    scales = scales_for(frame, spec, kinds, facet_grid(frame, spec))
    assert scales.x_levels == ("g0", "g1", "g2")

    swept = brush_mask(frame, spec, kinds, 0.6, 0.0, 1.6, 1e6, scales)

    assert set(frame["gene"][swept]) == {"g1", "g2"}
    assert list(swept) == [gene in ("g1", "g2") for gene in frame["gene"]]


def test_a_brush_with_no_scales_cannot_place_the_categories_and_keeps_them_all():
    """Without the level order there is no tick to match the sweep against.

    ``scales`` is optional, and the levels live on it. Guessing an order —
    the frame's, say — would select whichever groups happened to be sorted
    under the rectangle, which is worse than selecting none of them: the
    caller would get a plausible answer to a question it did not ask.
    """
    frame = _three_groups()
    spec = GraphSpec(x="gene", y="area", kind=BOX)
    kinds = spec.kinds_for(frame)
    assert kinds["gene"] == CATEGORICAL and kinds["area"] == CONTINUOUS

    swept = brush_mask(frame, spec, kinds, -0.4, 0.0, 1.4, 1e6)

    assert list(swept) == [True] * len(frame)


def test_the_numeric_axis_of_that_same_brush_still_narrows_it():
    """Only the axis it cannot place is left alone."""
    frame = _three_groups()
    spec = GraphSpec(x="gene", y="area", kind=BOX)
    kinds = spec.kinds_for(frame)

    swept = brush_mask(frame, spec, kinds, -0.4, 10.5, 1.4, 20.5)

    assert list(swept) == [10.5 <= v <= 20.5 for v in frame["area"]]
    assert 0 < int(swept.sum()) < len(frame)
