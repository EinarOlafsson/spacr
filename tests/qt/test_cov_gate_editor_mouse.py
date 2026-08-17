"""The mouse, on the Gate Editor's canvas.

Every gesture the 2D scatter and the 3D volume answer -- pressing, moving,
releasing, wheeling, clicking out a polygon and waving the wand -- driven
through the same handlers matplotlib calls, with the effect asserted on the
canvas afterwards rather than on the fact that the call returned.

The load-bearing property throughout is that a gesture means the same thing
to the user as to the gate it produces: the dashed placeholder shows what the
release will commit, a click that cannot become a gate says why, and a
gesture the view cannot read leaves everything as it was instead of inventing
a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import GateCanvas, _project
from spacr.qt.widgets.gate_spec import (
    BoxGate, CylinderGate, EllipseGate, GateSet, PolygonGate, PrismGate,
    RectGate, ThresholdGate,
    ELLIPSE, POLYGON, RECTANGLE, THRESHOLD, WAND,
)
from spacr.qt.widgets.graph_builder import GraphSpec


# ---------------------------------------------------------------------------
# The events matplotlib delivers
# ---------------------------------------------------------------------------

class _Mouse:
    """A press/motion/release, carrying both data and pixel coordinates.

    Handles are hit-tested in PIXELS and gates in data units, so an event
    that has only one of the two exercises half of what a real one does.
    """

    def __init__(self, ax, x_data, y_data, *, step=0):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = 1
        self.step = step


class _Pixels:
    """An event with pixel coordinates only -- what the volume reads."""

    def __init__(self, ax, x, y, *, step=0):
        self.inaxes = ax
        self.x, self.y = float(x), float(y)
        self.xdata = self.ydata = 0.0
        self.button = 1
        self.step = step


class _OffThePlot:
    """A gesture that landed outside every axes."""

    inaxes = None
    xdata = ydata = None
    x = y = 0.0
    button = 1
    step = 1


# ---------------------------------------------------------------------------
# Canvases
# ---------------------------------------------------------------------------

def _scatter(qtbot, gates=None, *, frame=None):
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    if frame is None:
        frame = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]})
    canvas.set_frame(frame)
    canvas.set_spec(GraphSpec(x="a", y="b"))
    if gates is not None:
        canvas.set_gates(gates)
    return canvas, canvas.axes_at(0, 0)


@pytest.fixture
def box_canvas(qtbot):
    """A scatter with one rectangle on it, at 2..8 in both measurements."""
    gate = RectGate(name="g", x_column="a", y_column="b",
                    x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0)
    canvas, ax = _scatter(qtbot, GateSet().add(gate))
    return canvas, ax


@pytest.fixture
def cloud(qtbot):
    """Two well-separated clouds -- what the wand is for."""
    rng = np.random.default_rng(0)
    frame = pd.concat([
        pd.DataFrame({"a": rng.normal(0.0, 0.3, 200),
                      "b": rng.normal(0.0, 0.3, 200)}),
        pd.DataFrame({"a": rng.normal(8.0, 0.3, 200),
                      "b": rng.normal(8.0, 0.3, 200)}),
    ], ignore_index=True)
    canvas, ax = _scatter(qtbot, frame=frame)
    return canvas, ax


@pytest.fixture
def volume(qtbot):
    """A real three-dimensional canvas, with a real Axes3D under it."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(0.0, 1.0, 200),
                          "b": rng.normal(0.0, 1.0, 200),
                          "c": rng.normal(0.0, 1.0, 200)})
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(frame)
    canvas.set_spec(GraphSpec(x="a", y="b"))
    canvas.set_mode("3D", z_column="c")
    ax = canvas.axes_at(0, 0)
    assert canvas._in_volume(), "the fixture did not get a volume to draw on"
    return canvas, ax


def _arm(canvas, tool):
    """Arm a tool and hand back the axes the redraw it triggers leaves up."""
    canvas.set_tool(tool)
    return canvas.axes_at(0, 0)


def _pixel_of_vertex(canvas, ax, vertex):
    """Where a pending volume vertex is on screen, right now."""
    columns = (canvas._spec.x, canvas._spec.y, canvas._z_column)
    first, second = canvas._pending_plane
    normal = next(c for c in columns if c not in (first, second))
    limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    point = [0.0, 0.0, 0.0]
    point[columns.index(first)] = float(vertex[0])
    point[columns.index(second)] = float(vertex[1])
    point[columns.index(normal)] = float(limits[columns.index(normal)][0])
    return ax.transData.transform(_project(ax, point))


# ---------------------------------------------------------------------------
# Grabbing an anchor point
# ---------------------------------------------------------------------------

def test_a_plot_whose_pixels_cannot_be_read_still_lets_the_gate_be_dragged(
        box_canvas, monkeypatch):
    """An anchor is hit-tested in pixels, so a transform that cannot answer
    means no anchor is grabbable. That must degrade to moving the gate, not
    to a traceback out of a mouse handler."""
    canvas, ax = box_canvas
    on_the_corner = _Mouse(ax, 8.0, 8.0)

    class _Unreadable:
        @staticmethod
        def transform(_point):
            raise RuntimeError("this axes cannot say where that is")

    monkeypatch.setattr(ax, "transData", _Unreadable())
    assert canvas.handle_at(on_the_corner) is None

    canvas._on_press(on_the_corner)
    assert canvas._resize is None, "an unreadable anchor was grabbed anyway"
    assert canvas._move_name == "g", (
        "the press was swallowed; a corner that cannot be measured should "
        "still be inside the gate, which is a move")


def test_a_gate_that_cannot_be_hit_tested_does_not_hide_the_one_beneath_it(
        qtbot):
    """A box is three measurements and this scatter is showing two, so it
    cannot be hit-tested here. Skipping it must not skip the rest: the
    rectangle underneath is still grabbable."""
    gates = GateSet()
    gates.add(BoxGate(name="volume-box", x_column="a", y_column="b",
                      z_column="c", x_low=0.0, x_high=10.0,
                      y_low=0.0, y_high=10.0, z_low=0.0, z_high=10.0))
    gates.add(RectGate(name="flat", x_column="a", y_column="b",
                       x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0))
    canvas, ax = _scatter(qtbot, gates)

    assert canvas.gate_at(5.0, 5.0) == "flat"


def test_a_gate_on_other_measurements_cannot_be_grabbed_invisibly(qtbot):
    """It is not drawn on these axes, so a press where it would be must not
    pick it up -- an outline in the wrong units is bad enough without being
    draggable."""
    gates = GateSet()
    gates.add(RectGate(name="elsewhere", x_column="p", y_column="q",
                       x_low=0.0, x_high=10.0, y_low=0.0, y_high=10.0))
    canvas, ax = _scatter(qtbot, gates)

    assert canvas.gate_at(5.0, 5.0) is None
    canvas._on_press(_Mouse(ax, 5.0, 5.0))
    assert canvas._move_name is None


# ---------------------------------------------------------------------------
# The placeholder that follows the mouse
# ---------------------------------------------------------------------------

def test_dragging_off_the_plot_takes_the_placeholder_with_it(box_canvas):
    """Off the axes there is no position to place the gate at, so there is
    nothing to promise: the dashed shape goes rather than hanging over the
    plot at the last place the mouse was."""
    canvas, ax = box_canvas
    canvas._on_press(_Mouse(ax, 5.0, 5.0))
    canvas._on_motion(_Mouse(ax, 6.0, 6.0))
    assert canvas._ghost, "nothing followed the mouse to begin with"

    canvas._on_motion(_OffThePlot())
    assert canvas._dragged_to(_OffThePlot()) is None
    assert canvas._ghost == [], "the placeholder outlived the plot"


def test_the_placeholder_for_a_threshold_is_the_two_lines_it_will_move_to(
        qtbot):
    """A cut has no outline to draw dashed -- it is its bounds. The
    placeholder has to be those bounds where the release will put them, or it
    is promising something the release does not do."""
    gates = GateSet().add(ThresholdGate(name="cut", column="a",
                                        low=2.0, high=6.0))
    canvas, ax = _scatter(qtbot, gates)

    canvas._on_press(_Mouse(ax, 4.0, 5.0))
    assert canvas._move_name == "cut"
    canvas._on_motion(_Mouse(ax, 5.0, 5.0))

    placed = sorted(float(line.get_xdata()[0]) for line in canvas._ghost)
    assert placed == [3.0, 7.0], (
        "the dashed cut is not where releasing here would put it")


def test_an_open_ended_cut_shows_only_the_bound_it_has(qtbot):
    """None means unbounded. Drawing a line for it would show the user a cut
    they never made."""
    gates = GateSet().add(ThresholdGate(name="cut", column="a",
                                        low=2.0, high=None))
    canvas, ax = _scatter(qtbot, gates)

    canvas._on_press(_Mouse(ax, 6.0, 5.0))
    assert canvas._move_name == "cut"
    canvas._on_motion(_Mouse(ax, 7.0, 5.0))

    assert [float(line.get_xdata()[0]) for line in canvas._ghost] == [3.0]


def test_a_shape_the_flat_view_cannot_draw_leaves_no_placeholder(box_canvas):
    """A cylinder is a shape in three measurements; on two axes there is no
    outline for it. Nothing dashed is the right answer -- an approximation
    would be a shape nobody drew."""
    canvas, ax = box_canvas
    cylinder = CylinderGate(name="tube", u_column="a", v_column="b",
                            axis_column="c", u_centre=5.0, v_centre=5.0,
                            u_radius=2.0, v_radius=2.0)

    canvas._show_ghost(cylinder)

    assert canvas._ghost == []


def test_a_placeholder_from_a_replotted_figure_is_still_forgotten(box_canvas):
    """The plot is rebuilt from scratch on every redraw -- another view
    changing the gates is enough -- and matplotlib refuses to remove an
    artist whose axes has gone with it. Keeping it would mean every later
    drag tried to remove it again, for ever."""
    canvas, ax = box_canvas
    canvas._on_press(_Mouse(ax, 5.0, 5.0))
    canvas._on_motion(_Mouse(ax, 6.0, 6.0))
    stale = list(canvas._ghost)
    assert stale, "nothing followed the mouse to begin with"

    canvas.render_now()
    with pytest.raises(NotImplementedError):
        stale[0].remove()

    canvas._clear_ghost()
    assert canvas._ghost == []


def test_nothing_picked_up_means_nothing_would_be_placed(box_canvas):
    """`_dragged_to` is the one answer to "what would releasing here do", and
    with nothing in flight the answer has to be "nothing" -- otherwise a
    stray motion would draw a placeholder for a gate nobody grabbed."""
    canvas, ax = box_canvas
    assert canvas._resize is None and canvas._move_name is None
    assert canvas._dragged_to(_Mouse(ax, 5.0, 5.0)) is None


# ---------------------------------------------------------------------------
# A gate removed while the mouse is still down
# ---------------------------------------------------------------------------

def test_a_gate_deleted_mid_resize_snaps_back_instead_of_being_edited(
        box_canvas):
    """Another view can remove a gate while a drag is in flight. The pull
    then has nothing to apply to: the plot is redrawn as it is, and no edit
    is reported for a gate that no longer exists."""
    canvas, ax = box_canvas
    edits = []
    canvas.gate_edited.connect(edits.append)

    canvas._on_press(_Mouse(ax, 8.0, 8.0))
    assert canvas._resize == ("g", "x_high,y_high")
    canvas.gates.remove("g")

    canvas._on_motion(_Mouse(ax, 9.0, 9.0))
    assert canvas._ghost == [], "a placeholder for a gate that is gone"

    canvas._on_release(_Mouse(ax, 9.0, 9.0))
    assert edits == []
    assert canvas._resize is None, "the drag state must always be cleared"


def test_a_gate_deleted_mid_move_is_not_moved_on_release(box_canvas):
    canvas, ax = box_canvas
    edits = []
    canvas.gate_edited.connect(edits.append)

    canvas._on_press(_Mouse(ax, 5.0, 5.0))
    assert canvas._move_name == "g"
    canvas.gates.remove("g")

    canvas._on_motion(_Mouse(ax, 6.0, 6.0))
    assert canvas._ghost == []

    canvas._on_release(_Mouse(ax, 6.0, 6.0))
    assert edits == []
    assert canvas._move_name is None


# ---------------------------------------------------------------------------
# Clicking out a polygon
# ---------------------------------------------------------------------------

def test_clicking_back_on_the_first_vertex_closes_the_polygon(qtbot):
    """What everyone tries. The Close button used to be the only way to
    finish, so a polygon looked impossible to complete."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, POLYGON)
    drawn = []
    canvas.gate_drawn.connect(drawn.append)
    counts = []
    canvas.polygon_changed.connect(counts.append)

    for x, y in ((1.0, 1.0), (5.0, 1.0), (5.0, 5.0)):
        canvas._on_press(_Mouse(ax, x, y))
    canvas._on_press(_Mouse(ax, 1.0, 1.0))

    assert len(drawn) == 1, "clicking the first vertex again did not close it"
    assert isinstance(drawn[0], PolygonGate)
    assert drawn[0].vertices == ((1.0, 1.0), (5.0, 1.0), (5.0, 5.0)), (
        "the closing click was kept as a fourth vertex")
    assert canvas.pending_vertices() == ()
    assert counts == [1, 2, 3, 0], "the vertex count was not reported"


def test_a_click_off_the_plot_places_no_vertex(qtbot):
    """There is no data coordinate for it, and a vertex at a made-up one is
    a corner of the gate nobody clicked."""
    canvas, _ax = _scatter(qtbot)
    canvas.set_tool(POLYGON)
    canvas._on_press(_OffThePlot())
    assert canvas.pending_vertices() == ()


def test_with_nothing_clicked_yet_there_is_no_first_vertex_to_close_on(qtbot):
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, POLYGON)
    assert canvas._near_first_vertex(_Mouse(ax, 1.0, 1.0), 1.0, 1.0) is False


def test_a_click_with_no_axes_under_it_cannot_close_the_polygon(qtbot):
    """Closing is measured in pixels, and without an axes there is nothing to
    measure against. Refusing beats closing the shape somewhere arbitrary."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, POLYGON)
    for x, y in ((1.0, 1.0), (5.0, 1.0), (5.0, 5.0)):
        canvas._on_press(_Mouse(ax, x, y))

    assert canvas._near_first_vertex(_OffThePlot(), 1.0, 1.0) is False
    assert len(canvas.pending_vertices()) == 3, "the shape was closed anyway"


def test_placing_a_vertex_draws_no_rubber_band(qtbot):
    """The polygon tool is click-per-vertex; a swept box following the mouse
    would be previewing a gate the tool does not make."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, POLYGON)
    canvas._on_press(_Mouse(ax, 1.0, 1.0))
    canvas._on_motion(_Mouse(ax, 4.0, 4.0))
    assert canvas._drag_patch is None


# ---------------------------------------------------------------------------
# The wand
# ---------------------------------------------------------------------------

def test_the_wand_turns_one_click_into_a_gate(cloud):
    """It emits `gate_drawn` like any other drawn shape, so it lands in the
    same naming and undo path -- it is a way of PRODUCING a polygon, not a
    fourth kind of gate."""
    canvas, _stale = cloud
    ax = _arm(canvas, WAND)
    drawn = []
    canvas.gate_drawn.connect(drawn.append)

    canvas._on_press(_Mouse(ax, 0.0, 0.0))

    assert len(drawn) == 1
    gate = drawn[0]
    assert isinstance(gate, PolygonGate)
    assert (gate.x_column, gate.y_column) == ("a", "b")
    inside = gate.mask(canvas.population())
    assert inside[:200].sum() > 150, "the clicked cloud was missed"
    assert not inside[200:].any(), "the wand reached the other population"


def test_a_wand_click_that_grows_nothing_says_what_to_change(cloud):
    """The two things that make it fail are both things the user fixes by
    changing a setting and clicking again, so it reports rather than
    raising out of a mouse handler."""
    canvas, _stale = cloud
    ax = _arm(canvas, WAND)
    drawn, failed = [], []
    canvas.gate_drawn.connect(drawn.append)
    canvas.wand_failed.connect(failed.append)

    canvas._on_press(_Mouse(ax, 4.0, 4.0))       # the empty gap between them

    assert drawn == []
    assert len(failed) == 1
    assert "maximum distance" in failed[0] or "tolerance" in failed[0], failed


def test_the_wand_needs_a_table_to_grow_a_gate_out_of(qtbot):
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, WAND)
    failed = []
    canvas.wand_failed.connect(failed.append)
    canvas.set_frame(None)

    canvas._on_press(_Mouse(ax, 5.0, 5.0))

    assert failed == ["the wand needs a table and two measurements on screen"]


def test_the_wand_names_a_measurement_the_table_does_not_have(qtbot):
    """A spec carried over from another table. The status line says which
    measurement is missing instead of the click doing nothing."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, WAND)
    failed = []
    canvas.wand_failed.connect(failed.append)
    canvas._spec = GraphSpec(x="a", y="not_a_column")

    canvas._on_press(_Mouse(ax, 5.0, 5.0))

    assert len(failed) == 1
    assert "not_a_column" in failed[0]


def test_a_wand_click_off_the_plot_is_ignored(cloud):
    """Nothing to grow from and nothing to complain about -- the pointer was
    never on the data."""
    canvas, _ax = cloud
    canvas.set_tool(WAND)
    drawn, failed = [], []
    canvas.gate_drawn.connect(drawn.append)
    canvas.wand_failed.connect(failed.append)

    canvas._on_press(_OffThePlot())

    assert drawn == [] and failed == []


# ---------------------------------------------------------------------------
# Sweeping a shape out
# ---------------------------------------------------------------------------

def test_sweeping_a_box_previews_it_and_then_makes_it(qtbot):
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, RECTANGLE)
    drawn = []
    canvas.gate_drawn.connect(drawn.append)

    canvas._on_press(_Mouse(ax, 1.0, 1.0))
    canvas._on_motion(_Mouse(ax, 6.0, 4.0))
    assert canvas._drag_patch is not None, "nothing previewed the sweep"
    assert canvas._drag_patch.get_width() == pytest.approx(5.0)
    assert canvas._drag_patch.get_height() == pytest.approx(3.0)

    canvas._on_release(_Mouse(ax, 6.0, 4.0))
    assert canvas._drag_patch is None, "the preview outlived the sweep"
    assert len(drawn) == 1
    gate = drawn[0]
    assert isinstance(gate, RectGate)
    assert (gate.x_low, gate.x_high, gate.y_low, gate.y_high) == (
        1.0, 6.0, 1.0, 4.0)


def test_a_sweep_released_on_another_plot_makes_nothing(qtbot):
    """The two ends would be in different measurements' units, so there is no
    rectangle to be had."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, RECTANGLE)
    drawn = []
    canvas.gate_drawn.connect(drawn.append)

    canvas._on_press(_Mouse(ax, 1.0, 1.0))
    elsewhere = _Mouse(ax, 6.0, 4.0)
    elsewhere.inaxes = object()
    canvas._on_release(elsewhere)

    assert drawn == []
    assert canvas._drag_origin is None, "the sweep was left half-finished"


def test_a_sweep_whose_end_has_no_data_position_makes_nothing(qtbot):
    """`inaxes` says the pointer was on the plot but the position could not
    be turned into measurements. A gate needs both ends in data units."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, RECTANGLE)
    drawn = []
    canvas.gate_drawn.connect(drawn.append)

    canvas._on_press(_Mouse(ax, 1.0, 1.0))
    nowhere = _Mouse(ax, 6.0, 4.0)
    nowhere.xdata = nowhere.ydata = None
    canvas._on_release(nowhere)

    assert drawn == []


def test_with_no_tool_armed_a_drag_selects_objects_instead_of_gating(qtbot):
    """Disarming every tool puts the canvas back to brushing, which is what
    the plot does everywhere else in the app: the swept objects become the
    shared selection the other views highlight."""
    from spacr.qt.linked_selection import linked_selection

    link = linked_selection()
    link.clear_selection()
    try:
        frame = pd.DataFrame([
            {"plateID": "p1", "rowID": "r1", "columnID": f"c{i}",
             "fieldID": "f1", "object_label": i, "a": float(i),
             "b": float(i)}
            for i in range(1, 11)])
        canvas, _stale = _scatter(qtbot, frame=frame)
        canvas.set_tool("")
        ax = canvas.axes_at(0, 0)   # disarming redraws; use the axes it made
        drawn = []
        canvas.gate_drawn.connect(drawn.append)

        canvas._on_press(_Mouse(ax, 1.0, 1.0))
        canvas._on_release(_Mouse(ax, 4.0, 4.0))

        assert drawn == [], "brushing made a gate"
        assert link.selection.is_active, "the sweep selected nothing"
        assert len(link.selection.keys) == 4, (
            "the objects under the sweep are not the ones that were shared")
    finally:
        link.clear_selection()


def test_a_release_with_no_sweep_behind_it_makes_nothing(qtbot):
    """The button came up without ever having gone down here -- switching
    tools mid-gesture does this."""
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, RECTANGLE)
    drawn = []
    canvas.gate_drawn.connect(drawn.append)

    canvas._on_release(_Mouse(ax, 6.0, 4.0))

    assert drawn == []


def test_an_oval_sweep_previews_an_oval_and_a_box_sweep_a_box(qtbot):
    """"the oval looks like a square when dragged but does in fact generate
    an oval gate" -- the preview has to be the shape being made."""
    from matplotlib.patches import Ellipse, Rectangle

    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, ELLIPSE)
    canvas._on_press(_Mouse(ax, 1.0, 1.0))
    canvas._on_motion(_Mouse(ax, 7.0, 5.0))
    oval = canvas._drag_patch
    assert isinstance(oval, Ellipse)
    assert oval.get_center() == pytest.approx((4.0, 3.0))
    assert (oval.get_width(), oval.get_height()) == pytest.approx((6.0, 4.0))
    canvas._on_release(_Mouse(ax, 7.0, 5.0))

    canvas.set_tool(RECTANGLE)
    canvas._on_press(_Mouse(ax, 1.0, 1.0))
    canvas._on_motion(_Mouse(ax, 7.0, 5.0))
    assert isinstance(canvas._drag_patch, Rectangle)
    assert canvas._drag_patch.get_xy() == pytest.approx((1.0, 1.0))


# ---------------------------------------------------------------------------
# What a sweep becomes
# ---------------------------------------------------------------------------

def test_a_sweep_with_no_measurement_on_screen_makes_no_gate(qtbot):
    """Every tool needs the columns it would name. Producing a gate on ""
    would be a gate that can never be re-applied."""
    canvas, _ax = _scatter(qtbot)
    canvas._spec = GraphSpec(x="", y="")

    canvas.set_tool(THRESHOLD)
    assert canvas.gate_from_drag(1.0, 1.0, 6.0, 4.0) is None
    canvas.set_tool(RECTANGLE)
    assert canvas.gate_from_drag(1.0, 1.0, 6.0, 4.0) is None
    canvas.set_tool(ELLIPSE)
    assert canvas.gate_from_drag(1.0, 1.0, 6.0, 4.0) is None


def test_a_cut_reads_only_the_horizontal_sweep(qtbot):
    """On a histogram the vertical axis is a count, and gating on a count is
    not a thing anyone means."""
    canvas, _ax = _scatter(qtbot)
    canvas.set_tool(THRESHOLD)

    gate = canvas.gate_from_drag(1.0, 99.0, 6.0, -99.0)

    assert isinstance(gate, ThresholdGate)
    assert (gate.column, gate.low, gate.high) == ("a", 1.0, 6.0)


def test_an_oval_dragged_to_nothing_is_not_a_gate(qtbot):
    """A zero-width sweep would be an ellipse with a zero radius, which
    EllipseGate refuses. Nothing drawn is the right answer to nothing
    dragged."""
    canvas, _ax = _scatter(qtbot)
    canvas.set_tool(ELLIPSE)

    assert canvas.gate_from_drag(3.0, 1.0, 3.0, 4.0) is None
    assert canvas.gate_from_drag(1.0, 3.0, 4.0, 3.0) is None
    real = canvas.gate_from_drag(1.0, 1.0, 7.0, 5.0)
    assert isinstance(real, EllipseGate)
    assert (real.x_centre, real.y_centre) == (4.0, 3.0)


def test_a_tool_that_does_not_sweep_makes_nothing_out_of_one(qtbot):
    """The polygon and the wand are click gestures. A drag with one of them
    armed is not half a gate."""
    canvas, _ax = _scatter(qtbot)
    canvas.set_tool(POLYGON)
    assert canvas.gate_from_drag(1.0, 1.0, 6.0, 4.0) is None
    canvas.set_tool(WAND)
    assert canvas.gate_from_drag(1.0, 1.0, 6.0, 4.0) is None


# ---------------------------------------------------------------------------
# The wheel
# ---------------------------------------------------------------------------

def test_the_wheel_zooms_the_scatter_about_the_pointer(qtbot):
    """Zooming toward what you are looking at is what every map does;
    centre-zoom means chasing a feature back into view after every notch."""
    canvas, ax = _scatter(qtbot)
    before = ax.get_xlim()

    canvas._on_scroll(_Mouse(ax, 2.0, 2.0, step=1))

    after = canvas.axes_at(0, 0).get_xlim()
    assert after != before, "the wheel did nothing"
    assert after[1] - after[0] < before[1] - before[0], "the wheel zoomed out"
    assert canvas._zoom is not None, (
        "the zoom was not remembered, so the next redraw undoes it")


def test_a_wheel_notch_off_the_plot_changes_nothing(qtbot):
    """There is no point to zoom about."""
    canvas, ax = _scatter(qtbot)
    before = ax.get_xlim()

    canvas._on_scroll(_OffThePlot())

    assert canvas._zoom is None
    assert canvas.axes_at(0, 0).get_xlim() == before


def test_reset_puts_the_limits_back_where_the_data_asks(qtbot):
    canvas, ax = _scatter(qtbot)
    before = ax.get_xlim()
    canvas._on_scroll(_Mouse(ax, 2.0, 2.0, step=1))
    assert canvas.axes_at(0, 0).get_xlim() != before

    canvas.reset_view()

    assert canvas._zoom is None
    assert canvas._volume_zoom == 1.0
    assert canvas._view_angles is None
    assert canvas.axes_at(0, 0).get_xlim() == pytest.approx(before)


def test_the_old_name_for_reset_is_still_the_same_reset(qtbot):
    """`reset_zoom` was the 2D-only version; callers kept the name."""
    assert GateCanvas.reset_zoom is GateCanvas.reset_view


# ---------------------------------------------------------------------------
# The volume
# ---------------------------------------------------------------------------

def test_a_drag_in_the_volume_turns_it_rather_than_drawing(volume):
    """Spin is the default, and the gate tools must not see the drag: "if i
    press pollygon and tried to draw a gate, then i could all of a suded
    spinn the graph" was the 2D press handler eating it."""
    canvas, _stale = volume
    canvas.set_tool(POLYGON)
    # Arming a tool redraws, and the volume is rebuilt from scratch when it
    # is: the axes the mouse lands on is the one that exists now.
    ax = canvas.axes_at(0, 0)
    azimuth = float(ax.azim)

    canvas._on_press(_Pixels(ax, 100, 100))
    assert canvas._spin_from == (100.0, 100.0)
    assert canvas.pending_vertices() == (), "a spin placed a polygon vertex"

    canvas._on_motion(_Pixels(ax, 160, 100))
    assert float(ax.azim) != azimuth, "the drag did not turn the volume"

    canvas._on_release(_Pixels(ax, 160, 100))
    assert canvas._spin_from is None, "the spin never ended"


def test_the_wheel_zooms_the_volume_and_not_the_flat_limits(volume):
    canvas, ax = volume
    before = ax.get_zlim3d()

    canvas._on_scroll(_Pixels(ax, 100, 100, step=1))

    assert ax.get_zlim3d() != before, "the wheel did nothing in 3D"
    assert canvas._volume_zoom > 1.0
    assert canvas._zoom is None, "the volume was zoomed as if it were flat"


def test_each_click_in_the_volume_places_one_vertex(volume):
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    counts = []
    canvas.polygon_changed.connect(counts.append)

    canvas._on_press(_Pixels(ax, 250, 220))
    canvas._on_press(_Pixels(ax, 330, 220))

    assert len(canvas.pending_vertices()) == 2
    assert counts == [1, 2]
    assert canvas._pending_plane == ("a", "b"), (
        "the vertices were not recorded against the plane they were "
        "clicked on")


def test_a_volume_whose_pixels_cannot_be_read_places_no_vertex(
        volume, monkeypatch):
    """A vertex in the volume is a click read back onto the chosen plane. If
    the axes cannot say where its own data is, there is no position to place
    one at -- and a made-up one is a corner of the gate nobody clicked."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    ax = canvas.axes_at(0, 0)

    class _Unreadable:
        @staticmethod
        def transform(_point):
            raise RuntimeError("this axes cannot say where that is")

        @staticmethod
        def inverted():
            raise RuntimeError("nor the other way round")

    monkeypatch.setattr(ax, "transData", _Unreadable())
    assert canvas.screen_to_volume(_Pixels(ax, 250, 220)) is None

    canvas._on_press(_Pixels(ax, 250, 220))

    assert canvas.pending_vertices() == ()
    assert canvas._pending_plane is None


def test_turning_the_view_mid_polygon_starts_the_shape_again(volume):
    """Vertices from two planes are not one shape, and mixing them quietly
    would produce a prism whose outline nobody drew."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    canvas._on_press(_Pixels(ax, 250, 220))
    canvas._on_press(_Pixels(ax, 330, 220))
    assert canvas._pending_plane == ("a", "b")

    canvas.set_anchor_axis("x")
    canvas._on_press(_Pixels(ax, 300, 260))

    assert canvas._pending_plane == ("b", "c")
    assert len(canvas.pending_vertices()) == 1, (
        "vertices clicked on the old plane were kept")


def test_clicking_the_first_vertex_closes_the_polygon_in_the_volume(volume):
    """The same gesture as on the flat scatter. It is measured by projecting
    the stored vertex onto the anchor face -- the two measurements it holds
    are not screen coordinates, and feeding them to the 2D helper made this
    work only by coincidence."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    drawn, asked = [], []
    canvas.gate_drawn.connect(drawn.append)
    canvas.depth_requested.connect(asked.append)
    for px, py in ((250, 220), (330, 220), (330, 300)):
        canvas._on_press(_Pixels(ax, px, py))
    vertices = canvas.pending_vertices()
    assert len(vertices) == 3

    on_the_first = _pixel_of_vertex(canvas, ax, vertices[0])
    assert canvas._near_first_volume_vertex(_Pixels(ax, *on_the_first)), (
        "a click on the first vertex was not recognised as one")
    far_away = _Pixels(ax, on_the_first[0] + 90.0, on_the_first[1] + 90.0)
    assert not canvas._near_first_volume_vertex(far_away), (
        "a click most of the plot away closed the shape")

    canvas._on_press(_Pixels(ax, *on_the_first))

    assert canvas.pending_vertices() == (), "the polygon was not closed"
    assert drawn == [], (
        "the footprint was emitted before its depth was asked for, which is "
        "what made one drawn polygon prompt for a name twice")
    assert canvas._pending_volume_gate is not None
    assert isinstance(canvas._pending_volume_gate, PrismGate)
    assert canvas._pending_volume_gate.vertices == vertices
    assert asked and "depth" in asked[0]


def test_closing_a_volume_polygon_of_two_vertices_makes_nothing(volume):
    """Two clicks and a change of mind. The canvas does not raise at the
    user for it, and it does not invent a third vertex either."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    drawn = []
    canvas.gate_drawn.connect(drawn.append)
    canvas._on_press(_Pixels(ax, 250, 220))
    canvas._on_press(_Pixels(ax, 330, 220))

    canvas.close_polygon_now()

    assert drawn == []
    assert canvas._pending_volume_gate is None
    assert len(canvas.pending_vertices()) == 2, (
        "the unfinished shape was thrown away")


def test_a_polygon_needs_a_third_measurement_to_become_a_prism(volume):
    """A prism is a footprint extended along the measurement the plane does
    not use. With the third axis repeating one already on screen there is no
    such measurement, so there is no prism to make."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    for px, py in ((250, 220), (330, 220), (330, 300)):
        canvas._on_press(_Pixels(ax, px, py))
    assert len(canvas.pending_vertices()) == 3

    canvas._z_column = "a"          # the third axis is not a third axis

    assert canvas.close_polygon() is None


def test_a_click_cannot_close_a_polygon_whose_plane_is_gone(volume):
    """The measurements on screen changed under the pending shape. Refusing
    to measure beats closing it on a plane that no longer exists."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    for px, py in ((250, 220), (330, 220), (330, 300)):
        canvas._on_press(_Pixels(ax, px, py))

    canvas._pending_plane = ("gone", "b")
    assert canvas._near_first_volume_vertex(_Pixels(ax, 250, 220)) is False


def test_nothing_pending_means_nothing_to_close_on_in_the_volume(volume):
    canvas, ax = volume
    assert canvas._near_first_volume_vertex(_Pixels(ax, 250, 220)) is False


def test_a_volume_polygon_needs_a_normal_to_close_against(volume):
    """The plane names two of the three measurements; without a third there
    is no face to project the first vertex onto."""
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    canvas._on_press(_Pixels(ax, 250, 220))
    canvas._z_column = "a"

    assert canvas._near_first_volume_vertex(_Pixels(ax, 250, 220)) is False


def test_a_click_with_no_axes_under_it_cannot_close_a_volume_polygon(volume):
    canvas, ax = volume
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    canvas._on_press(_Pixels(ax, 250, 220))

    assert canvas._near_first_volume_vertex(_OffThePlot()) is False


# ---------------------------------------------------------------------------
# A box gate seen in the flat editor
# ---------------------------------------------------------------------------

def _flat_box(qtbot):
    """A three-measurement BOX, on a scatter showing two of its three axes.

    The ordinary way to meet one: draw a gate in the 3D view, then go back to
    the flat scatter to adjust it.
    """
    gate = BoxGate(name="g", x_column="a", y_column="b", z_column="c",
                   x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0,
                   z_low=2.0, z_high=8.0)
    frame = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0],
                          "c": [0.0, 5.0, 10.0]})
    return _scatter(qtbot, GateSet().add(gate), frame=frame)


def test_a_box_gate_can_be_picked_up_in_the_flat_view(qtbot):
    """It is drawn and it offers handles, but it could not be grabbed.

    `_outline` and `_handles_for` both go through `_as_flat`, which shows a
    box as the rectangle it is from the front. `gate_at` did not -- it
    hit-tested the RAW box, whose `columns` include the depth measurement the
    flat probe has no column for, so `mask()` raised GateError, the except
    swallowed it, and the click found nothing. The gate was visible, its
    corners were visible, and clicking it did nothing at all.
    """
    canvas, _ax = _flat_box(qtbot)

    assert canvas.gate_at(5.0, 5.0) == "g", (
        "a box drawn in the volume is a rectangle from the front, and the "
        "front is what the user is clicking on")
    assert canvas.gate_at(9.5, 9.5) is None, "outside is still outside"


def test_dragging_a_box_gate_shows_the_ghost_it_would_become(qtbot):
    """`_show_ghost` had the same asymmetry as `gate_at`: it asked
    `_gate_points` for the raw box, which the flat axes cannot lay out, so
    the user pulled a corner with no preview of the result."""
    canvas, _ax = _flat_box(qtbot)
    gate = canvas.gates.get("g")

    canvas._show_ghost(gate)

    assert canvas._ghost, "pulling a box gate showed no preview at all"


def test_a_box_gate_can_be_resized_by_the_handles_it_offers(qtbot):
    """`_handles_for` shows a box AS FLAT, so it offers a rectangle's roles
    -- 'x_low,y_low' and the rest. `BoxGate` then refused every one of them
    with `GateError: BoxGate has no handle 'x_low'`, which `_dragged_to`
    swallowed. The corners were drawn, they could be grabbed, and pulling
    them did nothing.

    Pulling one keeps the box a BOX, and keeps the depth the flat view has
    no way to express -- replacing it with the rectangle drawn for it would
    silently drop the z range the user set in the volume.
    """
    canvas, ax = _flat_box(qtbot)
    canvas._resize = ("g", "x_low,y_low")

    pulled = canvas._dragged_to(_Mouse(ax, 4.0, 4.0))

    assert pulled is not None, "the handle it offered did nothing"
    assert isinstance(pulled, BoxGate), "a box stays a box"
    assert (pulled.x_low, pulled.y_low) == (4.0, 4.0)
    assert (pulled.z_low, pulled.z_high) == (2.0, 8.0), (
        "the depth the flat view cannot show must survive the drag")
