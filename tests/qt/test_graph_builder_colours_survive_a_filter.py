"""A filter keeps every surviving level's colour in the Graph Builder.

The axes follow a filter (``RESCALE_ON_FILTER``), but the colour levels come
from the whole table: a level's position in that order picks its hue, so
filtering one gene out must not shift the hue of every gene after it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets.graph_builder import (  # noqa: E402
    GraphCanvas, categorical_colours,
)
from spacr.qt.widgets.graph_spec import GraphSpec  # noqa: E402
from spacr.selection import CategoryFilter, DataFilter  # noqa: E402

pytestmark = pytest.mark.qt


def _frame() -> pd.DataFrame:
    n = 60
    return pd.DataFrame({
        "plateID": ["p1"] * 30 + ["p2"] * 30,
        "rowID": ["r1"] * n,
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": range(1, n + 1),
        "area": np.r_[np.linspace(1.0, 10.0, 30), np.linspace(100.0, 200.0, 30)],
        "intensity": np.linspace(0.0, 1.0, n),
        "gene": [f"g{i % 3}" for i in range(n)],
    })


@pytest.fixture
def view(qtbot):
    link = LinkedSelection()
    canvas = GraphCanvas(link=link, source="colour_test")
    qtbot.addWidget(canvas)
    canvas.set_frame(_frame())
    canvas.set_spec(GraphSpec(x="area", y="intensity", colour="gene"))
    yield canvas, link
    canvas.unlink_selection()


def _drawn_colour_of(canvas, level):
    rows = canvas._visible
    colours, _levels = canvas._level_colours(rows["gene"])
    return set(colours[(rows["gene"] == level).to_numpy()])


def test_filtering_out_one_level_keeps_the_others_hues(view):
    canvas, link = view
    order = categorical_colours()
    assert canvas.scales.colour_levels == ("g0", "g1", "g2")
    assert _drawn_colour_of(canvas, "g2") == {order[2]}

    link.set_filter(DataFilter().add(CategoryFilter("gene", ("g1", "g2"))))
    canvas.render_now()

    assert set(canvas._visible["gene"]) == {"g1", "g2"}
    assert canvas.scales.colour_levels == ("g0", "g1", "g2")
    assert _drawn_colour_of(canvas, "g1") == {order[1]}
    assert _drawn_colour_of(canvas, "g2") == {order[2]}


def test_the_axes_still_follow_the_filter(view):
    canvas, link = view
    link.set_filter(DataFilter().add(CategoryFilter("plateID", ("p1",))))
    canvas.render_now()

    low, high = canvas.axes_at(0, 0).get_xlim()
    assert high < 100.0
    assert canvas.scales.colour_levels == ("g0", "g1", "g2")
