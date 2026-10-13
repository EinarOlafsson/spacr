"""The chart's quieter branches: nothing to jitter, nothing to highlight.

A chart that draws the wrong thing is loud. These are the paths where it
draws nothing, or redraws instead of restyling, or drops a gesture that never
landed on data — each one silent by nature, and each one a way for a brush to
publish a selection nobody made.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.qt.widgets import graph_builder as gb                 # noqa: E402
from spacr.qt.widgets.graph_spec import (BAR, HISTOGRAM,         # noqa: E402
                                          SCATTER, GraphSpec)
from spacr.selection import Selection, as_key_index              # noqa: E402

pytestmark = pytest.mark.qt


def _frame(rows=60):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
        "area": rng.normal(100, 10, rows),
        "intensity": rng.normal(size=rows),
        "gene": [f"g{i % 3}" for i in range(rows)],
    })


@pytest.fixture()
def canvas(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="cov5")
    qtbot.addWidget(view)
    view.set_frame(_frame())
    return view


@pytest.fixture()
def panel(qtbot):
    widget = gb.GraphBuilderPanel(link=LinkedSelection(), source="cov5panel")
    qtbot.addWidget(widget)
    widget.canvas.set_frame(_frame())
    return widget


# ---------------------------------------------------------------------------
# Marks with nothing to place
# ---------------------------------------------------------------------------

def test_jitter_over_a_chart_with_no_measurement_draws_nothing(canvas):
    """A category and no number is a count plot; there is nothing to spread."""
    from matplotlib.figure import Figure

    canvas.set_spec(GraphSpec(x="gene", y="", kind=BAR), immediate=True)
    ax = Figure().add_subplot(111)
    rows = canvas._render_data.frame

    canvas._draw_jitter(ax, rows, gb.active_palette(), over_bars=True)

    assert len(ax.collections) == 0
    assert len(ax.lines) == 0


# ---------------------------------------------------------------------------
# The notice line
# ---------------------------------------------------------------------------

def test_the_notice_carries_what_the_grid_had_to_say(canvas):
    """A trellis that dropped panels says so; the chart has to relay it."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    grid = types.SimpleNamespace(notice="only the first 6 panels are drawn")

    said = canvas._notice_text(canvas._render_data, grid)

    assert "only the first 6 panels are drawn" in said


# ---------------------------------------------------------------------------
# Answering a selection made elsewhere
# ---------------------------------------------------------------------------

def test_an_aggregate_chart_redraws_rather_than_restyling(canvas,
                                                           monkeypatch):
    """A bar's highlight is a recomputed reduction, not a re-coloured artist."""
    canvas.set_spec(GraphSpec(x="gene", y="area", kind=BAR), immediate=True)
    assert canvas._live_highlight is False
    drawn = []
    monkeypatch.setattr(canvas, "render_now", lambda: drawn.append(True))

    canvas.link.set_selection(
        Selection(keys=as_key_index(["p1_r1_c1_f1_1"]), source="elsewhere"))

    assert drawn == [True]


def test_a_panel_with_no_overlay_is_stepped_over(canvas):
    """Not every panel gets a highlight artist; the loop must survive that."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    assert canvas._live_highlight is True
    canvas._overlays = dict.fromkeys(canvas._overlays, None)
    picked = canvas._render_data.frame.head(3)

    canvas.link.set_selection(Selection.from_frame(picked, source="elsewhere"))

    assert canvas._selected_mask is not None
    assert int(canvas._selected_mask.sum()) == 3


def test_a_selection_arriving_before_the_first_draw_is_ignored(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="cov5empty")
    qtbot.addWidget(view)

    view.on_linked_selection_changed(
        Selection(keys=as_key_index(["p1_r1_c1_f1_1"]), source="elsewhere"))

    assert view._selected_mask is None


# ---------------------------------------------------------------------------
# Gestures that never landed on data
# ---------------------------------------------------------------------------

def test_a_drag_off_the_data_area_previews_nothing(canvas):
    """Outside the axes there is no data coordinate to sweep between."""
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    ax = canvas._axes_at and next(iter(canvas._axes_at.values())) is not None
    axes = canvas._figure.axes[0]
    canvas._on_press(types.SimpleNamespace(inaxes=axes, xdata=90.0, ydata=0.0))

    canvas._on_motion(types.SimpleNamespace(inaxes=axes, xdata=None,
                                            ydata=None))

    assert canvas._drag_patch is None


def test_a_release_off_the_data_area_publishes_nothing(canvas):
    canvas.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER),
                    immediate=True)
    axes = canvas._figure.axes[0]
    canvas._on_press(types.SimpleNamespace(inaxes=axes, xdata=90.0, ydata=0.0))
    before = canvas.link.selection.keys

    canvas._on_release(types.SimpleNamespace(inaxes=axes, xdata=None,
                                             ydata=None))

    assert canvas.link.selection.keys is before
    assert canvas._drag_origin is None


# ---------------------------------------------------------------------------
# The shelf's controls
# ---------------------------------------------------------------------------

def test_clearing_the_shelf_empties_every_channel(panel):
    panel.set_spec(GraphSpec(x="area", y="intensity", colour="gene",
                             kind=SCATTER))
    assert panel.spec.x == "area"

    panel.clear_channels()

    assert panel.spec.x is None and panel.spec.y is None
    assert panel.spec.colour is None
    assert all(zone.column is None for zone in panel._zones.values())


def test_turning_a_control_republishes_the_whole_spec(panel):
    """The spec travels with every change, so a saved chart restores exactly."""
    panel.set_spec(GraphSpec(x="area", kind=HISTOGRAM, bins=20))
    seen = []
    panel.spec_changed.connect(seen.append)

    panel._bins.setValue(37)

    assert seen, "changing the bin count published nothing"
    assert seen[-1].bins == 37
    assert panel.canvas.spec.bins == 37
    assert seen[-1].x == "area"


def test_a_control_moved_while_the_shelf_is_syncing_publishes_nothing(panel):
    """``set_spec`` moves every control; each move must not echo back."""
    seen = []
    panel.spec_changed.connect(seen.append)

    panel.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER, bins=44))

    assert seen == []
    assert panel._bins.value() == 44
