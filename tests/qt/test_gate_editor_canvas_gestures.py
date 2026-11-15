"""The Gate Editor canvas: clicking out polygons, sweeping shapes, the wand and the wheel.

Every gesture is driven through the handlers matplotlib calls, and the effect
is asserted on the canvas afterwards.  A polygon gets one vertex per click on
the data (also on a 3D view that fell back to the flat scatter) and closes on
its first vertex; a sweep previews exactly the shape it will make and makes
nothing when its ends are not both data positions on one plot;
``gate_from_drag`` builds the armed tool's gate or nothing; the wand names
what it is missing; the wheel zooms about the pointer and reset restores the
data limits.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture; every case builds its own canvas.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

from spacr.qt.widgets.gate_editor import GateCanvas  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    ELLIPSE, POLYGON, RECTANGLE, THRESHOLD, WAND, EllipseGate, PolygonGate,
    RectGate, ThresholdGate,
)
from spacr.qt.widgets.graph_builder import GraphSpec  # noqa: E402

pytestmark = pytest.mark.qt

_TRIANGLE = ((1.0, 1.0), (5.0, 1.0), (5.0, 5.0))
_WIDE = ((1.0, 1.0), (9.0, 1.0), (5.0, 9.0))


class _Mouse:
    """A press/motion/release carrying both data and pixel coordinates."""

    def __init__(self, ax, x_data, y_data, *, step=0):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = 1
        self.step = step


class _OffThePlot:
    """A gesture that landed outside every axes."""

    inaxes = None
    xdata = ydata = None
    x = y = 0.0
    button = 1
    step = 1


def _frame():
    return pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]})


def _canvas(qtbot, *, mode=None, tool=None):
    """A flat scatter of a against b (or a 3D request that fell back to flat), with ``tool`` armed."""
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(_frame())
    canvas.set_spec(GraphSpec(x="a", y="b"))
    if mode:
        canvas.set_mode(mode)
        assert not canvas._in_volume()
    if tool is not None:
        canvas.set_tool(tool)
    record = {"drawn": [], "counts": [], "failed": []}
    canvas.gate_drawn.connect(record["drawn"].append)
    canvas.polygon_changed.connect(record["counts"].append)
    canvas.wand_failed.connect(record["failed"].append)
    return canvas, canvas.axes_at(0, 0), record


def test_polygon_clicks_place_vertices_and_close_on_the_first_one(qtbot, subtests):
    """Clicks on the data add vertices without a rubber band; the first vertex closes; off-plot clicks do nothing."""
    for mode, vertices in ((None, _TRIANGLE), ("xD", _WIDE)):
        view = mode or "flat"
        with subtests.test(msg=f"{view}_click_back_on_first_closes"):
            canvas, ax, record = _canvas(qtbot, mode=mode, tool=POLYGON)
            for x, y in vertices:
                canvas._on_press(_Mouse(ax, x, y))
            canvas._on_press(_Mouse(ax, *vertices[0]))
            (gate,) = record["drawn"]
            assert isinstance(gate, PolygonGate)
            assert gate.vertices == vertices
            assert canvas.pending_vertices() == ()
            assert record["counts"] == [1, 2, 3, 0]
        with subtests.test(msg=f"{view}_close_polygon_gives_flat_gate"):
            canvas, ax, record = _canvas(qtbot, mode=mode, tool=POLYGON)
            for x, y in vertices:
                canvas._on_press(_Mouse(ax, x, y))
            assert canvas.pending_vertices() == vertices
            assert record["counts"] == [1, 2, 3]
            gate = canvas.close_polygon(name="p")
            assert isinstance(gate, PolygonGate)
            assert (gate.x_column, gate.y_column, gate.vertices) == ("a", "b", vertices)
    with subtests.test(msg="click_off_plot_places_nothing"):
        canvas, _ax, record = _canvas(qtbot, tool=POLYGON)
        canvas._on_press(_OffThePlot())
        assert canvas.pending_vertices() == ()
        assert record["counts"] == []
    with subtests.test(msg="no_axes_cannot_close"):
        canvas, ax, record = _canvas(qtbot, tool=POLYGON)
        for x, y in _TRIANGLE:
            canvas._on_press(_Mouse(ax, x, y))
        assert canvas._near_first_vertex(_OffThePlot(), 1.0, 1.0) is False
        assert canvas.pending_vertices() == _TRIANGLE
    with subtests.test(msg="no_rubber_band"):
        canvas, ax, _record = _canvas(qtbot, tool=POLYGON)
        canvas._on_press(_Mouse(ax, 1.0, 1.0))
        canvas._on_motion(_Mouse(ax, 4.0, 4.0))
        assert canvas._drag_patch is None


def test_sweeps_preview_the_shape_they_make_or_make_nothing(qtbot, subtests):
    """Box and oval sweeps preview their exact patch and emit their gate; broken releases emit nothing."""
    with subtests.test(msg="box_previewed_then_made"):
        canvas, ax, record = _canvas(qtbot, tool=RECTANGLE)
        canvas._on_press(_Mouse(ax, 1.0, 1.0))
        canvas._on_motion(_Mouse(ax, 6.0, 4.0))
        patch = canvas._drag_patch
        assert isinstance(patch, Rectangle)
        assert patch.get_xy() == pytest.approx((1.0, 1.0))
        assert (patch.get_width(), patch.get_height()) == pytest.approx((5.0, 3.0))
        canvas._on_release(_Mouse(ax, 6.0, 4.0))
        assert canvas._drag_patch is None
        (gate,) = record["drawn"]
        assert isinstance(gate, RectGate)
        assert (gate.x_low, gate.x_high, gate.y_low, gate.y_high) == (1.0, 6.0, 1.0, 4.0)
    with subtests.test(msg="oval_previewed_as_oval"):
        canvas, ax, record = _canvas(qtbot, tool=ELLIPSE)
        canvas._on_press(_Mouse(ax, 1.0, 1.0))
        canvas._on_motion(_Mouse(ax, 7.0, 5.0))
        oval = canvas._drag_patch
        assert isinstance(oval, Ellipse)
        assert oval.get_center() == pytest.approx((4.0, 3.0))
        assert (oval.get_width(), oval.get_height()) == pytest.approx((6.0, 4.0))
        canvas._on_release(_Mouse(ax, 7.0, 5.0))
        (gate,) = record["drawn"]
        assert isinstance(gate, EllipseGate)

    def other_plot(ax):
        event = _Mouse(ax, 6.0, 4.0)
        event.inaxes = object()
        return event

    def no_data(ax):
        event = _Mouse(ax, 6.0, 4.0)
        event.xdata = event.ydata = None
        return event

    for case_id, pressed, release in (("released_on_another_plot", True, other_plot),
                                      ("end_without_data_position", True, no_data),
                                      ("release_without_press", False,
                                       lambda ax: _Mouse(ax, 6.0, 4.0))):
        with subtests.test(msg=case_id):
            canvas, ax, record = _canvas(qtbot, tool=RECTANGLE)
            if pressed:
                canvas._on_press(_Mouse(ax, 1.0, 1.0))
            canvas._on_release(release(ax))
            assert record["drawn"] == []
            assert canvas._drag_origin is None
            assert canvas._drag_patch is None


def test_gate_from_drag_builds_the_armed_tools_gate_or_nothing(qtbot, subtests):
    """Cuts read only the horizontal sweep, ovals need area, click tools and missing columns make nothing."""
    cases = {
        "cut_reads_horizontal_only": (THRESHOLD, (1.0, 99.0, 6.0, -99.0),
                                      ThresholdGate, ("a", 1.0, 6.0)),
        "zero_width_oval": (ELLIPSE, (3.0, 1.0, 3.0, 4.0), None, None),
        "zero_height_oval": (ELLIPSE, (1.0, 3.0, 4.0, 3.0), None, None),
        "real_oval": (ELLIPSE, (1.0, 1.0, 7.0, 5.0), EllipseGate, (4.0, 3.0)),
        "polygon_does_not_sweep": (POLYGON, (1.0, 1.0, 6.0, 4.0), None, None),
        "wand_does_not_sweep": (WAND, (1.0, 1.0, 6.0, 4.0), None, None),
    }
    for case_id, (tool, drag, kind, fields) in cases.items():
        with subtests.test(msg=case_id):
            canvas, _ax, _record = _canvas(qtbot, tool=tool)
            gate = canvas.gate_from_drag(*drag)
            if kind is None:
                assert gate is None
            else:
                assert isinstance(gate, kind)
                if kind is ThresholdGate:
                    assert (gate.column, gate.low, gate.high) == fields
                else:
                    assert (gate.x_centre, gate.y_centre) == fields
    for tool in (THRESHOLD, RECTANGLE, ELLIPSE):
        with subtests.test(msg=f"no_measurement_on_screen_{tool}"):
            canvas, _ax, _record = _canvas(qtbot)
            canvas._spec = GraphSpec(x="", y="")
            canvas.set_tool(tool)
            assert canvas.gate_from_drag(1.0, 1.0, 6.0, 4.0) is None


def test_the_wand_names_what_it_lacks_and_the_wheel_zooms_about_the_pointer(qtbot, subtests):
    """Wand failures are exact sentences; a wheel notch zooms 0.8x about the pointer, off-plot does nothing, reset restores."""
    with subtests.test(msg="wand_needs_a_table"):
        canvas, ax, record = _canvas(qtbot, tool=WAND)
        canvas.set_frame(None)
        canvas._on_press(_Mouse(ax, 5.0, 5.0))
        assert record["failed"] == ["the wand needs a table and two measurements on screen"]
        assert record["drawn"] == []
    with subtests.test(msg="wand_names_missing_measurement"):
        canvas, ax, record = _canvas(qtbot, tool=WAND)
        canvas._spec = GraphSpec(x="a", y="not_a_column")
        canvas._on_press(_Mouse(ax, 5.0, 5.0))
        assert record["failed"] == [
            "wand names column 'not_a_column', which this table does not have. "
            "A gate drawn on one dataset only re-applies to a table that "
            "carries the same measurements"]
    with subtests.test(msg="wheel_zooms_about_pointer_then_reset"):
        canvas, ax, _record = _canvas(qtbot)
        before_x, before_y = ax.get_xlim(), ax.get_ylim()
        canvas._on_scroll(_Mouse(ax, 2.0, 2.0, step=1))
        zoomed = canvas.axes_at(0, 0)

        def about_pointer(limits):
            return (2.0 - (2.0 - limits[0]) * 0.8, 2.0 + (limits[1] - 2.0) * 0.8)

        assert zoomed.get_xlim() == pytest.approx(about_pointer(before_x))
        assert zoomed.get_ylim() == pytest.approx(about_pointer(before_y))
        zoom_x, zoom_y = canvas._zoom
        assert zoom_x == pytest.approx(about_pointer(before_x))
        assert zoom_y == pytest.approx(about_pointer(before_y))
        canvas.reset_view()
        assert canvas._zoom is None
        assert canvas._volume_zoom == 1.0
        assert canvas._view_angles is None
        assert canvas.axes_at(0, 0).get_xlim() == pytest.approx(before_x)
    with subtests.test(msg="wheel_off_plot_changes_nothing"):
        canvas, ax, _record = _canvas(qtbot)
        before = ax.get_xlim()
        canvas._on_scroll(_OffThePlot())
        assert canvas._zoom is None
        assert canvas.axes_at(0, 0).get_xlim() == before
