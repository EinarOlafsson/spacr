"""A grid of small multiples has to admit what it is not showing.

Panels side by side are read as comparable. That is only true when the
scales, the levels and the row count are what the reader assumes, so every
place the grid quietly drops something -- levels beyond the cap, rows outside
them, a sample instead of the whole table, a panel with no usable values --
has to reach the notice or the count axis. These drive those.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.graph_spec import (CATEGORICAL, CONTINUOUS, HISTOGRAM,
                                         SCATTER, GraphSpec)
from spacr.qt.widgets.trellis_spec import (SCALE_SHARED, TrellisSpec, trellis)


def test_a_spec_can_be_built_from_a_plain_dictionary():
    """A graph given as a dict is turned into a real GraphSpec.

    Specs arrive from saved figure settings as JSON. Keeping the raw dict
    would leave every property below reaching into a mapping that has no
    validation behind it, and an unknown kind would surface as a drawing
    failure rather than a message.
    """
    spec = TrellisSpec(graph={"x": "area", "y": "solidity",
                              "colour": "gene", "kind": SCATTER})

    assert isinstance(spec.graph, GraphSpec)
    assert spec.x == "area"
    assert spec.y == "solidity"
    assert spec.colour == "gene"


def test_changing_the_kind_leaves_everything_else_where_it_was():
    """``with_kind`` replaces only the kind, on a new spec.

    The spec is frozen so a menu can offer a change without committing it.
    Rebuilding it from scratch would lose the facets and the scale modes the
    user had already set.
    """
    spec = TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID",
                                       kind=HISTOGRAM,
                                       roles={"area": CONTINUOUS}),
                       scale_x=SCALE_SHARED, wrap=3)

    changed = spec.with_kind(SCATTER)

    assert changed.graph.kind == SCATTER
    assert changed.facet_row == "plateID"
    assert changed.wrap == 3
    assert spec.graph.kind == HISTOGRAM      # the original is untouched


def test_a_grid_says_which_levels_it_left_out():
    """Levels beyond the cap, and the rows in them, are both reported.

    Twenty plates drawn as twelve panels is not twenty plates. Without the
    notice, a reader compares twelve panels believing they are the screen.
    """
    frame = pd.DataFrame({
        "plateID": [f"p{i % 20:02d}" for i in range(60)],
        "area": np.arange(60, dtype=float),
    })
    spec = TrellisSpec(graph=GraphSpec(x="area", facet_col="plateID", bins=2,
                                       roles={"area": CONTINUOUS}))

    result = trellis(frame, spec)

    assert "20 levels" in result.grid.notice
    assert "the first 12 are drawn" in result.grid.notice
    assert result.grid.notice in result.notice
    assert "row(s) outside the drawn levels" in result.notice


def test_a_grid_drawn_from_a_subset_says_so():
    """A sampled or binned frame carries its notice onto the grid.

    The whole point of the strategy notice is that a subset can never be
    mistaken for the whole table. Dropping it at the trellis boundary would
    lose it exactly where the picture gets bigger and more convincing.
    """
    frame = pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0),
                          "g": ["a", "b"] * 10})
    spec = TrellisSpec(graph=GraphSpec(x="x", y="y", facet_col="g",
                                       kind=SCATTER, point_budget=4,
                                       roles={"x": CONTINUOUS,
                                              "y": CONTINUOUS}))

    result = trellis(frame, spec)

    assert result.data.strategy != "full"
    assert result.data.notice
    assert result.data.notice in result.notice


def test_a_panel_whose_values_are_all_missing_contributes_no_height():
    """A panel of rows with no finite value does not raise the count axis.

    The rows are there -- the panel is occupied and prints its n -- but they
    bin to nothing. Counting them would make the shared count axis taller
    than any bar in the grid, and every real bar would look shorter than it
    is.
    """
    frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2", "p2"],
        "area": [1.0, 2.0, 3.0, np.nan, np.nan],
    })
    spec = TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID", bins=2,
                                       kind=HISTOGRAM,
                                       roles={"area": CONTINUOUS}))

    result = trellis(frame, spec)

    assert result.n_at(1, 0) == 2                 # the rows are still counted
    top = result.scales_at(0, 0).count_limit
    assert top is not None
    assert top == pytest.approx(2 * 1.08)         # p1's tallest bin, not 2 + 2
    assert result.scales_at(1, 0).count_limit == top


def test_a_grid_with_nothing_to_bin_sets_no_count_ceiling():
    """When no panel has a usable value the count axis is left open.

    A ceiling of zero collapses the axis and draws a grid of flat panels,
    which reads as "every group was measured and every count was zero".
    """
    frame = pd.DataFrame({"plateID": ["p1", "p1", "p2", "p2"],
                          "area": [np.nan] * 4})
    spec = TrellisSpec(graph=GraphSpec(x="area", facet_row="plateID", bins=2,
                                       kind=HISTOGRAM,
                                       roles={"area": CONTINUOUS}))

    result = trellis(frame, spec)

    assert result.scales_at(0, 0).x_edges is None
    assert result.scales_at(0, 0).count_limit is None
    assert result.scales_at(1, 0).count_limit is None
