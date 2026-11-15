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
    BoxGate, GateSet, PolygonGate, PrismGate,
    RectGate, POLYGON, WAND,
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

    # THE ARTIST IS STALE, checked by asking the artist rather than by
    # catching what `remove()` raises. Matplotlib used to raise
    # NotImplementedError for an artist whose axes had gone; on 3.8 it
    # detaches the artist (`.axes` becomes None) and removing it is a no-op.
    # Pinning the exception pinned the library's version, not spaCR's
    # behaviour, and the test went red on an upgrade that broke nothing.
    assert stale[0].axes is None, "the redraw did not detach the placeholder"

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


def test_with_nothing_clicked_yet_there_is_no_first_vertex_to_close_on(qtbot):
    canvas, _stale = _scatter(qtbot)
    ax = _arm(canvas, POLYGON)
    assert canvas._near_first_vertex(_Mouse(ax, 1.0, 1.0), 1.0, 1.0) is False


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


# ---------------------------------------------------------------------------
# The wheel
# ---------------------------------------------------------------------------


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


def test_dragging_a_box_gate_shows_the_ghost_it_would_become(qtbot):
    """`_show_ghost` had the same asymmetry as `gate_at`: it asked
    `_gate_points` for the raw box, which the flat axes cannot lay out, so
    the user pulled a corner with no preview of the result."""
    canvas, _ax = _flat_box(qtbot)
    gate = canvas.gates.get("g")

    canvas._show_ghost(gate)

    assert canvas._ghost, "pulling a box gate showed no preview at all"
