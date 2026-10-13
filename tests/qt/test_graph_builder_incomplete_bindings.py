"""Pinned chart kinds remain editable while their bindings are incomplete."""
import numpy as np
import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_builder import GraphBuilderPanel, GraphCanvas
from spacr.qt.widgets.graph_spec import (
    BAR, BAR_JITTER, BOX, CATEGORICAL, CHANNELS, CONTINUOUS, HEATMAP, HISTOGRAM,
    JITTER, LINE, SCATTER, VIOLIN, GraphSpec,
)

VALID = [
    (BAR, "group", None), (BAR, None, "group"),
    (BAR, "group", "value"), (BAR_JITTER, "group", "value"),
    (HISTOGRAM, "value", None), (HISTOGRAM, None, "value"),
    (SCATTER, "value", "other"), (SCATTER, "group", "value"),
    (LINE, "value", "other"), (BOX, "group", "value"),
    (BOX, "value", "group"), (VIOLIN, "group", "value"),
    (JITTER, "value", "group"), (HEATMAP, "group", "batch"),
]


@pytest.fixture
def frame():
    return pd.DataFrame({"value": np.linspace(1, 10, 40),
                         "other": np.linspace(20, 80, 40),
                         "group": ["a", "b"] * 20,
                         "batch": ["one", "two", "three", "four"] * 10})


@pytest.fixture
def canvas(qtbot, frame):
    view = GraphCanvas(link=LinkedSelection(), source="incomplete_graph")
    qtbot.addWidget(view)
    view.set_frame(frame)
    return view


def message(canvas):
    return " ".join(text.get_text() for ax in canvas.figure().axes for text in ax.texts)


@pytest.mark.parametrize("kind,x,y", VALID)
def test_clear_then_rebind_preserves_pinned_kind_and_marks(canvas, kind, x, y):
    spec = GraphSpec(kind=kind, x=x, y=y)
    canvas.set_spec(spec, immediate=True)
    assert canvas.render_data is not None
    ax = canvas.axes_at()
    before = (len(ax.patches), len(ax.lines), len(ax.collections), len(ax.images))
    assert any(before)
    canvas.set_channel("x", None)
    canvas.set_channel("y", None)
    canvas.render_now()
    assert canvas.spec.kind == kind
    assert canvas.render_data is None
    assert "X or Y" in message(canvas)
    assert canvas._brush_grid is None
    assert canvas._live_highlight is False
    assert canvas.selected_count() == 0
    canvas.set_spec(spec, immediate=True)
    ax = canvas.axes_at()
    assert (len(ax.patches), len(ax.lines), len(ax.collections), len(ax.images)) == before


@pytest.mark.parametrize("kind,x,y,expected", [
    (SCATTER, "value", None, "both X and Y"),
    (LINE, None, "value", "both X and Y"),
    (HISTOGRAM, "group", None, "continuous"),
    (BAR, "value", None, "categorical"),
    (BAR, "value", "group", "categorical"),
    (BAR_JITTER, "group", None, "continuous"),
    (BOX, "value", "other", "categorical"),
    (VIOLIN, "group", None, "continuous"),
    (JITTER, "group", "batch", "continuous"),
    (HEATMAP, "value", "group", "categorical"),
])
def test_incompatible_kind_explains_required_bindings(canvas, kind, x, y, expected):
    spec = GraphSpec(kind=kind, x=x, y=y)
    canvas.set_spec(spec, immediate=True)
    assert canvas.spec == spec
    assert canvas.render_data is None
    assert expected in message(canvas)


@pytest.mark.parametrize("channel", CHANNELS)
def test_missing_bound_columns_are_recoverable(canvas, channel):
    spec = GraphSpec(x="value", y="other", kind=SCATTER).with_channel(channel, "removed")
    canvas.set_spec(spec, immediate=True)
    assert canvas.render_data is None
    assert "removed" in message(canvas)
    assert "unavailable" in message(canvas)


def test_axis_role_change_recovers_histogram(canvas):
    spec = GraphSpec(x="value", kind=HISTOGRAM, roles={"value": CATEGORICAL})
    canvas.set_spec(spec, immediate=True)
    assert "continuous" in message(canvas)
    # The classification override is part of validation, not just inference.
    numeric = spec.with_role("value", CONTINUOUS)
    canvas.set_spec(numeric, immediate=True)
    assert canvas.render_data is not None


def test_real_drop_zone_clear_recovers_without_losing_bar_choice(qtbot, frame):
    panel = GraphBuilderPanel(link=LinkedSelection(), source="clear_bar")
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    panel.set_spec(GraphSpec(x="group", kind=BAR))
    panel.canvas.render_now()
    panel.clear_channels()
    panel.canvas.render_now()
    assert panel.spec.kind == BAR
    assert "X or Y" in message(panel.canvas)
    panel.zone("x").set_column("group")
    panel.canvas.render_now()
    bars = panel.canvas.axes_at().patches
    assert [bar.get_height() for bar in bars] == [20, 20]
