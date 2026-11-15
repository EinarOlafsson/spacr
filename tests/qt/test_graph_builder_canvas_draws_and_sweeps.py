"""Graph builder: the artists each chart kind draws, sweeps, and the shelf's spec.

The canvas is rendered for real (a matplotlib figure on the offscreen
platform), so every case asserts the artists that came out: exact box, jitter
and bar counts, a binned density of the requested shape, a hue legend that
gathers overflow into one "other" entry, and nothing at all for a chart with
nothing measurable.  A sweep previews a rectangle and publishes exactly the
keys it covers; a gesture off the data, a click, an unknown panel or a press
outside the axes publishes nothing.  The shelf republishes the whole spec on a
control change and stays silent while it syncs.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.backend_bases import MouseEvent  # noqa: E402

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets import graph_builder as gb  # noqa: E402
from spacr.qt.widgets.graph_spec import (  # noqa: E402
    CONTINUOUS, HISTOGRAM, SCATTER, GraphSpec,
)
from spacr.selection import Selection  # noqa: E402

pytestmark = pytest.mark.qt

_SCATTER = GraphSpec(x="area", y="intensity", kind=SCATTER)


def _frame(rows=60):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "area": rng.normal(100, 10, rows),
        "intensity": rng.normal(size=rows),
        "gene": [f"g{i % 3}" for i in range(rows)],
        "cell_count": [i % 5 for i in range(rows)],
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
    })


def _canvas(qtbot, frame=None, spec=None):
    canvas = gb.GraphCanvas(link=LinkedSelection(), source="draws")
    qtbot.addWidget(canvas)
    canvas.set_frame(_frame() if frame is None else frame)
    if spec is not None:
        canvas.set_spec(spec, immediate=True)
    return canvas


def _mouse(canvas, ax, xdata, ydata, name):
    x, y = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, canvas._canvas, x, y, 1)


def test_graph_canvas_draws_the_exact_artists_each_kind_calls_for(qtbot, monkeypatch, subtests):
    """Box, jitter, violin, bar, density and hue-legend draws produce exact artist counts and labels."""
    with subtests.test(msg="debounced_edit_then_drawn"):
        canvas = _canvas(qtbot)
        canvas._debounce.stop()
        canvas.set_spec(_SCATTER, immediate=False)
        assert canvas._debounce.isActive()
        canvas._debounce.stop()
        canvas.render_now()
        assert len(canvas.figure().axes) == 1
    with subtests.test(msg="empty_level_skipped"):
        frame = _frame()
        frame.loc[frame["gene"] == "g2", "area"] = np.nan
        canvas = _canvas(qtbot, frame, GraphSpec(x="gene", y="area", kind="box"))
        ax = canvas.figure().axes[0]
        assert (len(ax.lines), len(ax.patches)) == (12, 2)
        canvas.set_spec(GraphSpec(x="gene", y="area", kind="jitter"), immediate=True)
        assert [len(c.get_offsets()) for c in canvas.figure().axes[0].collections] == [20, 20]
    with subtests.test(msg="nothing_measurable_draws_nothing"):
        frame = _frame()
        frame["area"] = np.nan
        canvas = _canvas(qtbot, frame, GraphSpec(x="gene", y="area", kind="violin"))
        assert len(canvas.figure().axes[0].collections) == 0
        canvas.set_spec(GraphSpec(x="gene", y="area", kind="jitter"), immediate=True)
        assert len(canvas.figure().axes[0].collections) == 0
    with subtests.test(msg="jitter_needs_a_categorical_axis"):
        canvas = _canvas(qtbot, spec=GraphSpec(x="area", y="intensity", kind="jitter"))
        ax = canvas.figure().axes[0]
        canvas._draw_jitter(ax, canvas._frame, gb.active_palette(), over_bars=False)
        assert len(ax.collections) == 0
    with subtests.test(msg="selection_counted_on_its_bars"):
        canvas = _canvas(qtbot, spec=GraphSpec(x="gene", kind="bar"))
        ax = canvas.figure().axes[0]
        assert len(ax.patches) == 3
        mask = (canvas._frame["gene"] == "g1").to_numpy()
        canvas._draw_bar(ax, canvas._frame, mask, gb.active_palette())
        assert ax.containers[-1].get_label() == "selected"
        assert [p.get_height() for p in ax.containers[-1]] == [0.0, 20.0, 0.0]
    with subtests.test(msg="density_raster_weighted_by_third_column"):
        canvas = _canvas(qtbot, _frame(4000), GraphSpec(
            x="area", y="intensity", colour="cell_count",
            roles={"cell_count": CONTINUOUS}, kind=SCATTER,
            point_budget=100, bins=12))
        (image,) = canvas.figure().axes[0].images
        assert image.get_array().shape == (12, 12)
    with subtests.test(msg="hue_overflow_gathered_into_other"):
        frame = _frame()
        frame["gene"] = [f"g{i % 11}" for i in range(len(frame))]
        canvas = _canvas(qtbot, frame, GraphSpec(x="area", y="intensity",
                                                 colour="gene", kind=SCATTER))
        labels = [t.get_text() for t in canvas.figure().legends[0].get_texts()]
        assert labels == [f"g{i}" for i in range(8)] + [f"{gb.OTHER_LABEL} (3)"]
        assert len(gb.categorical_colours()) == 8
    with subtests.test(msg="unresolvable_selection_is_no_selection"):
        canvas = _canvas(qtbot, spec=_SCATTER)

        class Refusing:
            is_active = True

            def mask_for(self, _frame):
                raise KeyError("prcfo")

        monkeypatch.setattr(type(canvas.link), "selection",
                            property(lambda _self: Refusing()))
        canvas._keyed = True
        assert canvas._selection_mask(canvas._frame) is None
        canvas.render_now()
        assert len(canvas.figure().axes) == 1
        monkeypatch.undo()
    with subtests.test(msg="panel_without_overlay_stepped_over"):
        canvas = _canvas(qtbot, spec=_SCATTER)
        assert canvas._live_highlight is True
        canvas._overlays = dict.fromkeys(canvas._overlays, None)
        picked = canvas._render_data.frame.head(3)
        canvas.link.set_selection(Selection.from_frame(picked, source="elsewhere"))
        assert int(canvas._selected_mask.sum()) == 3


def test_graph_canvas_sweeps_publish_exactly_what_they_cover(qtbot, subtests):
    """A full sweep previews and publishes its covered keys; off-data, outside, click and unknown-panel gestures publish nothing."""
    with subtests.test(msg="sweep_previews_then_publishes"):
        canvas = _canvas(qtbot, spec=_SCATTER)
        ax = canvas.figure().axes[0]
        assert canvas._keyed is True
        (lx, hx), (ly, hy) = ax.get_xlim(), ax.get_ylim()
        x0, y0 = lx + 0.05 * (hx - lx), ly + 0.05 * (hy - ly)
        x1, y1 = hx - 0.05 * (hx - lx), hy - 0.05 * (hy - ly)
        canvas._on_press(_mouse(canvas, ax, x0, y0, "button_press_event"))
        canvas._on_motion(_mouse(canvas, ax, x1, y1, "motion_notify_event"))
        assert canvas._drag_patch.get_width() == pytest.approx(x1 - x0)
        canvas._on_release(_mouse(canvas, ax, x1, y1, "button_release_event"))
        assert (canvas._drag_origin, canvas._drag_patch) == (None, None)
        frame = canvas._frame
        covered = frame[frame.area.between(x0, x1) & frame.intensity.between(y0, y1)]
        assert set(canvas.link.selection.keys) == set(
            Selection.from_frame(covered, source="draws").keys)
    with subtests.test(msg="brush_without_publishing"):
        canvas = _canvas(qtbot, spec=_SCATTER)
        published = []
        canvas.link.selection_changed.connect(published.append)
        ax = canvas.figure().axes[0]
        (lx, hx), (ly, hy) = ax.get_xlim(), ax.get_ylim()
        selection = canvas.brush(lx, ly, hx, hy, publish=False)
        assert len(selection.keys) == 60
        assert published == []
    with subtests.test(msg="press_outside_axes"):
        canvas = _canvas(qtbot, spec=_SCATTER)
        canvas._on_press(MouseEvent("button_press_event", canvas._canvas, 1.0, 1.0, 1))
        assert canvas._drag_origin is None
        canvas._on_motion(MouseEvent("motion_notify_event", canvas._canvas, 1.0, 1.0, 1))
        assert canvas._drag_patch is None
    for case_id, handler in (("drag_off_data", "_on_motion"),
                             ("release_off_data", "_on_release")):
        with subtests.test(msg=case_id):
            canvas = _canvas(qtbot, spec=_SCATTER)
            axes = canvas._figure.axes[0]
            canvas._on_press(types.SimpleNamespace(inaxes=axes, xdata=90.0,
                                                   ydata=0.0, button=1))
            before = canvas.link.selection.keys
            getattr(canvas, handler)(types.SimpleNamespace(inaxes=axes, xdata=None,
                                                           ydata=None))
            assert canvas._drag_patch is None
            assert canvas.link.selection.keys is before
            if handler == "_on_release":
                assert canvas._drag_origin is None
    with subtests.test(msg="click_clears_the_selection"):
        canvas = _canvas(qtbot, spec=_SCATTER)
        ax = canvas.figure().axes[0]
        mx, my = sum(ax.get_xlim()) / 2.0, sum(ax.get_ylim()) / 2.0
        canvas._on_press(_mouse(canvas, ax, mx, my, "button_press_event"))
        cleared = []
        canvas.clear_linked_selection = lambda: cleared.append(True)
        canvas._on_release(_mouse(canvas, ax, mx, my, "button_release_event"))
        assert cleared == [True]
    with subtests.test(msg="release_on_unknown_panel"):
        canvas = _canvas(qtbot, spec=_SCATTER)
        ax = canvas.figure().axes[0]
        (lx, hx), (ly, hy) = ax.get_xlim(), ax.get_ylim()
        canvas._on_press(_mouse(canvas, ax, (lx + hx) / 2, (ly + hy) / 2,
                                "button_press_event"))
        canvas._axes_at = {}
        brushed = []
        canvas.brush = lambda *args, **kwargs: brushed.append(args)
        canvas._on_release(_mouse(canvas, ax, lx + 0.9 * (hx - lx),
                                  ly + 0.9 * (hy - ly), "button_release_event"))
        assert brushed == []


def test_graph_builder_panel_publishes_whole_specs_except_while_syncing(qtbot, subtests):
    """Clearing empties every zone, a control change republishes the whole spec, and set_spec echoes nothing."""
    def make():
        panel = gb.GraphBuilderPanel(link=LinkedSelection(), source="shelf")
        qtbot.addWidget(panel)
        panel.canvas.set_frame(_frame())
        return panel

    with subtests.test(msg="clear_channels"):
        panel = make()
        panel.set_spec(GraphSpec(x="area", y="intensity", colour="gene", kind=SCATTER))
        assert panel.spec.x == "area"
        panel.clear_channels()
        assert (panel.spec.x, panel.spec.y, panel.spec.colour) == (None, None, None)
        assert [zone.column for zone in panel._zones.values()] == [None] * len(panel._zones)
    with subtests.test(msg="control_republishes_whole_spec"):
        panel = make()
        panel.set_spec(GraphSpec(x="area", kind=HISTOGRAM, bins=20))
        seen = []
        panel.spec_changed.connect(seen.append)
        panel._bins.setValue(37)
        assert [(s.x, s.bins) for s in seen] == [("area", 37)]
        assert panel.canvas.spec.bins == 37
    with subtests.test(msg="sync_publishes_nothing"):
        panel = make()
        seen = []
        panel.spec_changed.connect(seen.append)
        panel.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER, bins=44))
        assert seen == []
        assert panel._bins.value() == 44
    with subtests.test(msg="closing_the_panel_closes_its_canvas"):
        panel = make()
        closed = []
        real_close = panel.canvas.close
        panel.canvas.close = lambda: (closed.append(True), real_close())[1]
        panel.show()
        panel.close()
        assert closed == [True]
        assert not panel.canvas._debounce.isActive()
