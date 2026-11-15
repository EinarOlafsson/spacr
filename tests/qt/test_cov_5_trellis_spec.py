"""A grid of small multiples has to admit what it is not showing.

Panels side by side are read as comparable. That is only true when the
scales, the levels and the row count are what the reader assumes, so every
place the grid quietly drops something -- levels beyond the cap, rows outside
them, a sample instead of the whole table, a panel with no usable values --
has to reach the notice or the count axis. These drive those.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.graph_spec import (CONTINUOUS, HISTOGRAM,
                                         SCATTER, GraphSpec)
from spacr.qt.widgets.trellis_spec import (SCALE_SHARED, TrellisSpec)


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
