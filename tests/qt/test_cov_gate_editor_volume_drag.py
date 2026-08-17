"""What the mouse does inside the 3D volume: spin, sweep, depth and wheel.

Instruction 52 again, from the complaint that opened it:

    "i cant zoom in or spin on any of the axees. if i press pollygon and tried
     to draw a gate, then i could all of a suded spinn the graph"

Those are four separate gestures sharing one mouse -- turning the volume,
sweeping out a footprint, dragging the slab that gives it depth, and the wheel
-- and every one of them was reachable only through code no test drove. These
pin the gestures themselves: what the user sees move, what they are told while
they hold the button, and what they get when they let go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.qt.widgets.gate_editor as editor
from spacr.qt.widgets.gate_editor import GateCanvas
from spacr.qt.widgets.gate_spec import (
    BoxGate, CylinderGate, ThresholdGate,
)


# ---------------------------------------------------------------------------
# A volume you can drive
# ---------------------------------------------------------------------------

class _Identity:
    """A transData that leaves projected coordinates where they are."""

    @staticmethod
    def transform(point):
        return (float(point[0]), float(point[1]))


class _Line:
    """One of the dashed outlines the sweep leaves behind."""

    def __init__(self, xs, ys, zs, style):
        self.xs, self.ys, self.zs = tuple(xs), tuple(ys), tuple(zs)
        self.style = style
        self.removed = False

    def remove(self):
        self.removed = True


class _Volume:
    """A stand-in for matplotlib's Axes3D.

    Only what the gesture handlers actually touch: the three limits, the
    camera angles, and enough of ``plot`` to see what was drawn. Real Axes3D
    needs a figure, a renderer and a live projection matrix before it will
    answer any of those, which puts the gestures out of reach of a test.
    """

    def __init__(self, elev=15.0, azim=40.0):
        self.elev, self.azim = elev, azim
        self.limits = {"x": (0.0, 10.0), "y": (0.0, 20.0), "z": (0.0, 30.0)}
        self.drawn: list = []
        self.added: list = []
        self.transData = _Identity()

    # -- what makes it a volume -------------------------------------------
    def get_zlim(self):
        return self.limits["z"]

    def get_xlim3d(self):
        return self.limits["x"]

    def get_ylim3d(self):
        return self.limits["y"]

    def get_zlim3d(self):
        return self.limits["z"]

    def set_xlim3d(self, low, high):
        self.limits["x"] = (float(low), float(high))

    def set_ylim3d(self, low, high):
        self.limits["y"] = (float(low), float(high))

    def set_zlim3d(self, low, high):
        self.limits["z"] = (float(low), float(high))

    # -- the camera --------------------------------------------------------
    def view_init(self, elev=None, azim=None):
        self.elev, self.azim = float(elev), float(azim)

    # -- what it can draw --------------------------------------------------
    def plot(self, xs, ys, zs, **kwargs):
        line = _Line(xs, ys, zs, kwargs.get("linestyle"))
        self.drawn.append(line)
        return [line]

    def add_collection3d(self, artist):
        self.added.append(artist)


class _NoCamera:
    """A depth-reporting axes that cannot be turned.

    matplotlib's own 3D axes always can; this exists to drive the guard that
    refuses rather than raising when it meets one that cannot.
    """

    def get_zlim(self):
        return (0.0, 30.0)


class _Event:
    """A matplotlib mouse event, reduced to what the handlers read."""

    def __init__(self, x=0.0, y=0.0, inaxes=True, **extra):
        self.x = float(x)
        self.y = float(y)
        self.inaxes = object() if inaxes else None
        for key, value in extra.items():
            setattr(self, key, value)


FRAME = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 20.0], "c": [0.0, 30.0]})


@pytest.fixture
def canvas(qtbot):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    from dataclasses import replace
    # GraphSpec is frozen -- a spec is a value, like a gate.
    widget._spec = replace(widget._spec, x="a", y="b")
    widget._z_column = "c"
    widget._mode = "3D"
    widget._frame = FRAME.copy()
    return widget


@pytest.fixture
def volume(canvas):
    axes = _Volume()
    canvas.axes_at = lambda *_a, **_k: axes
    return axes


@pytest.fixture
def said(canvas):
    """Everything the canvas told the user about the depth gesture."""
    messages: list = []
    canvas.depth_requested.connect(messages.append)
    return messages


def _sees(canvas, *corners):
    """Make the mouse read as the given points on the anchor plane."""
    points = iter(corners)
    canvas.screen_to_volume = lambda _event: next(points)


# ---------------------------------------------------------------------------
# Pressing: which gesture a press starts
# ---------------------------------------------------------------------------

def test_a_press_outside_the_plot_starts_nothing_but_is_still_swallowed(
        canvas, volume):
    """It has to be swallowed, or the 2D gate tools see the drag and the
    volume spins while a gate tool is armed -- "if i press pollygon and tried
    to draw a gate, then i could all of a suded spinn the graph"."""
    canvas.set_drag_mode("draw")
    _sees(canvas, ("a", 1.0, "b", 2.0))

    assert canvas._volume_press(_Event(5.0, 6.0, inaxes=False)) is True
    assert canvas._volume_drag is None
    assert canvas._spin_from is None


def test_the_polygon_tool_never_arms_a_drag(canvas, volume):
    """A polygon is click-per-vertex, so a drag must not start a sweep and
    must not start a spin either -- the spin is what the report describes."""
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("polygon")
    _sees(canvas, ("a", 1.0, "b", 2.0))

    assert canvas._volume_press(_Event(5.0, 6.0)) is False
    assert canvas._volume_drag is None
    assert canvas._spin_from is None


def test_a_press_in_2d_leaves_the_flat_gate_tools_alone(canvas):
    canvas._mode = "2D"
    assert canvas._volume_press(_Event(5.0, 6.0)) is False


# ---------------------------------------------------------------------------
# Spinning: one drag turns about one axis
# ---------------------------------------------------------------------------

def _spin(canvas, start, end):
    canvas._spin_from = start
    return canvas._volume_motion(_Event(end[0], end[1]))


def test_the_upright_lock_turns_the_volume_without_tipping_the_horizon(
        canvas, volume):
    """"say click the y axis, then i should be able to spin on the x axis."
    Locked to the upright, a sideways drag is a change of azimuth only, so the
    horizon stays level and every view stays readable."""
    canvas.set_spin_axis("z")

    assert _spin(canvas, (10.0, 20.0), (30.0, 90.0)) is True
    assert volume.azim == pytest.approx(50.0)      # 40 + 20 * 0.5
    assert volume.elev == pytest.approx(15.0)      # the vertical drag is ignored
    assert canvas._view_angles == pytest.approx((15.0, 50.0))


def test_locking_to_a_horizontal_axis_tips_the_volume_instead_of_turning_it(
        canvas, volume):
    canvas.set_spin_axis("x")

    _spin(canvas, (10.0, 20.0), (90.0, 60.0))
    assert volume.elev == pytest.approx(35.0)      # 15 + 40 * 0.5
    assert volume.azim == pytest.approx(40.0)      # unturned


def test_the_tip_stops_at_straight_down_instead_of_rolling_over(canvas,
                                                                volume):
    """Past 90 degrees the volume is upside down and nothing on it can be
    read, which is the state the axis lock exists to keep the user out of."""
    volume.elev = 80.0
    canvas.set_spin_axis("y")

    _spin(canvas, (0.0, 0.0), (0.0, 400.0))
    assert volume.elev == pytest.approx(90.0)


def test_unlocking_the_spin_lets_one_drag_do_both(canvas, volume):
    canvas.set_spin_axis("")

    _spin(canvas, (10.0, 20.0), (30.0, 60.0))
    assert volume.azim == pytest.approx(50.0)
    assert volume.elev == pytest.approx(35.0)


def test_an_axis_nobody_can_spin_about_falls_back_to_the_upright(canvas,
                                                                 volume):
    """Not to free rotation: free rotation reaches angles from which nothing
    can be read, which is what the lock replaced."""
    canvas.set_spin_axis("diagonal")

    _spin(canvas, (10.0, 20.0), (30.0, 60.0))
    assert volume.azim == pytest.approx(50.0)
    assert volume.elev == pytest.approx(15.0)


def test_each_step_of_a_spin_measures_from_the_last_one(canvas, volume):
    """Otherwise the second half of a long drag re-applies the first half's
    turn and the volume runs away from the pointer."""
    canvas.set_spin_axis("z")
    canvas._spin_from = (0.0, 0.0)

    canvas._volume_motion(_Event(20.0, 0.0))
    assert canvas._spin_from == (20.0, 0.0)
    canvas._volume_motion(_Event(40.0, 0.0))
    assert volume.azim == pytest.approx(60.0)      # 40 + 10 + 10, not 40 + 30


def test_moving_the_mouse_with_no_button_down_does_not_turn_the_volume(
        canvas, volume):
    assert canvas._volume_motion(_Event(90.0, 90.0)) is True
    assert (volume.elev, volume.azim) == (15.0, 40.0)


def test_a_spin_that_wanders_off_the_plot_holds_its_angle(canvas, volume):
    canvas.set_spin_axis("z")
    canvas._spin_from = (10.0, 20.0)

    assert canvas._volume_motion(_Event(90.0, 20.0, inaxes=False)) is True
    assert volume.azim == pytest.approx(40.0)


def test_an_axes_that_cannot_be_turned_refuses_the_spin_instead_of_raising(
        canvas):
    """A guard, and a redundant one: every axes with a depth limit is an
    Axes3D and every Axes3D can be turned. Driven so the refusal is a
    refusal and not a traceback out of a mouse handler."""
    canvas.axes_at = lambda *_a, **_k: _NoCamera()
    canvas._spin_from = (10.0, 20.0)

    assert canvas._volume_motion(_Event(90.0, 90.0)) is True
    assert canvas._spin_from == (10.0, 20.0)


def test_moving_the_mouse_over_a_flat_plot_is_not_a_spin(canvas):
    canvas._mode = "2D"
    assert canvas._volume_motion(_Event(90.0, 90.0)) is False


def test_letting_go_ends_the_spin(canvas, volume):
    """Left set, the next stray motion event would carry on turning the
    volume with no button held."""
    canvas._spin_from = (10.0, 20.0)

    assert canvas._volume_release(_Event(30.0, 40.0)) is True
    assert canvas._spin_from is None


def test_a_release_in_2d_is_left_to_the_flat_gate_tools(canvas):
    canvas._mode = "2D"
    assert canvas._volume_release(_Event(30.0, 40.0)) is False


# ---------------------------------------------------------------------------
# Sweeping: the footprint you can see while you hold the button
# ---------------------------------------------------------------------------

def test_the_swept_footprint_is_shown_at_both_ends_of_the_depth(canvas,
                                                                volume):
    """A volume gate IS a footprint extended through the volume, so drawing
    it on one face only would show the user half of what they are making."""
    canvas.set_drag_mode("draw")
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    _sees(canvas, ("a", 3.0, "b", 6.0))

    assert canvas._volume_motion(_Event(40.0, 50.0)) is True
    assert len(volume.drawn) == 2
    near, far = volume.drawn
    assert set(near.zs) == {0.0} and set(far.zs) == {30.0}
    assert near.xs == (1.0, 3.0, 3.0, 1.0, 1.0)
    assert near.ys == (2.0, 2.0, 6.0, 6.0, 2.0)
    assert near.style == "--"                      # a proposal, not a gate


def test_the_footprint_follows_the_mouse_instead_of_smearing(canvas, volume):
    """Every motion event replaces the outline. Left to accumulate, a single
    drag paints a solid block of dashes across the volume."""
    canvas.set_drag_mode("draw")
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    _sees(canvas, ("a", 3.0, "b", 6.0), ("a", 5.0, "b", 9.0))

    canvas._volume_motion(_Event(40.0, 50.0))
    first_pair = list(canvas._ghost)
    canvas._volume_motion(_Event(60.0, 70.0))

    assert len(canvas._ghost) == 2
    assert all(line.removed for line in first_pair)
    assert canvas._ghost[0].xs == (1.0, 5.0, 5.0, 1.0, 1.0)


def test_the_footprint_is_drawn_on_the_plane_the_user_picked(canvas, volume):
    """Pick X and the sweep must lie flat on B/C at the two ends of A -- not
    on X/Y, which is where the first rework drew it whatever was picked."""
    canvas.set_anchor_axis("x")
    canvas.set_drag_mode("draw")
    canvas._volume_drag = ("b", 4.0, "c", 5.0)
    _sees(canvas, ("b", 8.0, "c", 15.0))

    canvas._volume_motion(_Event(40.0, 50.0))
    near, far = volume.drawn
    assert set(near.xs) == {0.0} and set(far.xs) == {10.0}
    assert near.ys == (4.0, 8.0, 8.0, 4.0, 4.0)
    assert near.zs == (5.0, 5.0, 15.0, 15.0, 5.0)


def test_a_sweep_the_volume_cannot_read_shows_nothing(canvas, volume):
    """Edge-on there is no inverse, so there is no rectangle to show. An
    outline invented from a failed read would sit somewhere the gate will
    not be."""
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    canvas.screen_to_volume = lambda _event: None

    assert canvas._volume_motion(_Event(40.0, 50.0)) is True
    assert volume.drawn == []
    assert canvas._ghost == []


def test_a_sweep_with_no_extent_makes_no_gate(canvas, volume):
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    _sees(canvas, ("a", 1.0, "b", 2.0))
    assert canvas._gate_from_volume_drag(_Event(1.0, 1.0)) is None


def test_a_sweep_the_volume_cannot_read_makes_no_gate(canvas, volume):
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    canvas.screen_to_volume = lambda _event: None
    assert canvas._gate_from_volume_drag(_Event(1.0, 1.0)) is None


def test_a_sweep_that_makes_nothing_clears_the_outline_and_redraws(canvas,
                                                                   volume):
    """A dashed rectangle left hanging over the volume looks like a gate the
    user has, and the click that made it produced none."""
    drawn: list = []
    canvas.gate_drawn.connect(drawn.append)
    redrawn: list = []
    canvas.render_now = lambda: redrawn.append(True)
    canvas.set_drag_mode("draw")
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    _sees(canvas, ("a", 3.0, "b", 6.0), ("a", 1.0, "b", 2.0))
    canvas._volume_motion(_Event(40.0, 50.0))     # a real outline exists

    assert canvas._volume_release(_Event(40.0, 50.0)) is True
    assert canvas._volume_drag is None
    assert canvas._ghost == []
    assert drawn == []
    assert redrawn == [True]


# ---------------------------------------------------------------------------
# The second gesture: a slab you drag out
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_on(monkeypatch):
    """A camera looking straight at the A axis: depth C runs up the screen.

    ``_project`` is the one place matplotlib's moving 3D projection is read,
    so replacing it is how a test states a camera angle.
    """
    monkeypatch.setattr(editor, "_project",
                        lambda _ax, point: (point[0], point[2]))


def test_a_depth_drag_reads_as_a_fraction_of_the_visible_range(
        canvas, volume, straight_on):
    """Not as pixels: the same hand movement has to mean the same slab after
    zooming, and the measurements have no common unit."""
    canvas._pending_volume_axis = "c"

    bounds = canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 15.0))
    # C spans 0..30 on screen, so half the axis is half its range.
    assert bounds == pytest.approx((0.0, 15.0))


def test_dragging_the_depth_backwards_still_asks_for_the_same_slab(
        canvas, volume, straight_on):
    """The gesture is a distance. Punishing the user for dragging up rather
    than down would make the control depend on where they started."""
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag(
        (0.0, 0.0), _Event(0.0, -15.0)) == pytest.approx((0.0, 15.0))


def test_a_click_instead_of_a_drag_asks_for_full_depth(canvas, volume,
                                                       straight_on):
    """"click for full depth" -- and a hand is never perfectly still, so a
    couple of pixels of wobble is still a click."""
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag(
        (0.0, 0.0), _Event(2.0, 1.0)) == (None, None)


def test_dragging_the_whole_way_down_is_full_depth_not_a_slab(canvas, volume,
                                                              straight_on):
    """A slab of 99% of the volume is what the user meant by "all of it", and
    an unbounded gate says that exactly rather than to four digits."""
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag(
        (0.0, 0.0), _Event(0.0, 29.5)) == (None, None)


def test_a_depth_axis_pointing_at_the_viewer_cannot_be_read(canvas, volume,
                                                            monkeypatch):
    """Edge-on the whole axis lands on one screen point, so no drag length
    can be turned into a depth."""
    monkeypatch.setattr(editor, "_project",
                        lambda _ax, point: (point[0], point[1]))
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 40.0)) is None


def test_a_measurement_with_no_visible_range_has_no_depth_to_read(
        canvas, volume, straight_on):
    volume.limits["z"] = (7.0, 7.0)
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 40.0)) is None


def test_a_depth_on_a_measurement_no_longer_shown_is_refused(canvas, volume,
                                                             straight_on):
    """The user swapped an axis between the two gestures. Reading the drag
    against a measurement that is not on the plot would bound the gate on a
    number nobody could see."""
    canvas._pending_volume_axis = "gone"

    assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 15.0)) is None


def test_there_is_no_depth_to_read_off_a_flat_plot(canvas):
    canvas.axes_at = lambda *_a, **_k: object()
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 15.0)) is None


def test_a_projection_that_fails_costs_the_depth_and_not_the_session(
        canvas, volume, monkeypatch):
    """matplotlib has moved the 3D projection more than once; a version that
    cannot answer must not take a mouse handler down with it."""
    def boom(_ax, _point):
        raise RuntimeError("this matplotlib projects differently")

    monkeypatch.setattr(editor, "_project", boom)
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 15.0)) is None


# ---------------------------------------------------------------------------
# What the user is told while the depth is being dragged
# ---------------------------------------------------------------------------

def test_dragging_the_depth_names_the_slab_before_it_is_committed(
        canvas, volume, straight_on, said):
    """"Instruction 52 ... the depth is a second gesture after the shape is
    drawn" -- and a second gesture with no readout is a gate made blind."""
    canvas._depth_drag_from = (0.0, 0.0)
    canvas._pending_volume_axis = "c"

    assert canvas._volume_motion(_Event(0.0, 15.0)) is True
    assert said == ["Depth 0 to 15 — release to create the gate."]


def test_holding_still_says_the_gate_will_go_all_the_way_through(
        canvas, volume, straight_on, said):
    canvas._depth_drag_from = (0.0, 0.0)
    canvas._pending_volume_axis = "c"

    canvas._volume_motion(_Event(1.0, 1.0))
    assert said == ["Full depth selected — release to create the gate."]


def test_a_depth_that_cannot_be_read_yet_says_nothing_rather_than_guessing(
        canvas, volume, monkeypatch, said):
    monkeypatch.setattr(editor, "_project",
                        lambda _ax, point: (point[0], point[1]))
    canvas._depth_drag_from = (0.0, 0.0)
    canvas._pending_volume_axis = "c"

    assert canvas._volume_motion(_Event(0.0, 40.0)) is True
    assert said == []


def test_a_depth_the_view_cannot_read_says_how_to_get_it_back(canvas, volume,
                                                              monkeypatch,
                                                              said):
    """Naming the two controls that fix it. "That angle is no good" leaves
    the user holding a footprint with nothing to do about it."""
    monkeypatch.setattr(editor, "_project",
                        lambda _ax, point: (point[0], point[1]))
    canvas._depth_drag_from = (0.0, 0.0)
    canvas._pending_volume_gate = CylinderGate(
        name="(unnamed)", u_column="a", v_column="b", axis_column="c",
        u_centre=1.0, v_centre=1.0, u_radius=1.0, v_radius=1.0)
    canvas._pending_volume_axis = "c"

    assert canvas._volume_release(_Event(0.0, 40.0)) is True
    assert canvas._depth_drag_from is None
    assert "Spin" in said[0] and "Draw" in said[0]
    # The footprint is kept, so the fix is turn-and-drag rather than redraw.
    assert canvas._pending_volume_gate is not None


def test_a_dragged_depth_bounds_the_gate_it_was_dragged_for(canvas, volume,
                                                            straight_on):
    """The whole two-gesture flow, driven through the mouse handlers: sweep a
    footprint, then drag its depth."""
    drawn: list = []
    canvas.gate_drawn.connect(drawn.append)
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("box")
    _sees(canvas, ("a", 1.0, "b", 2.0), ("a", 5.0, "b", 8.0))

    canvas._volume_press(_Event(0.0, 0.0))
    canvas._volume_release(_Event(40.0, 50.0))
    assert isinstance(canvas._pending_volume_gate, BoxGate)
    assert drawn == []

    canvas._volume_press(_Event(0.0, 0.0))
    canvas._volume_release(_Event(0.0, 15.0))
    assert len(drawn) == 1
    assert (drawn[0].z_low, drawn[0].z_high) == pytest.approx((0.0, 15.0))
    assert canvas._pending_volume_gate is None
    assert canvas._pending_volume_axis == ""


def test_a_finished_depth_clears_the_message_so_it_does_not_linger(
        canvas, volume, straight_on, said):
    canvas._pending_volume_gate = CylinderGate(
        name="(unnamed)", u_column="a", v_column="b", axis_column="c",
        u_centre=1.0, v_centre=1.0, u_radius=1.0, v_radius=1.0)
    canvas._pending_volume_axis = "c"

    canvas._finish_volume_depth((0.0, 15.0))
    assert said[-1] == ""


# ---------------------------------------------------------------------------
# Committing the depth
# ---------------------------------------------------------------------------

def test_a_depth_with_no_footprint_waiting_makes_no_gate(canvas):
    drawn: list = []
    canvas.gate_drawn.connect(drawn.append)

    assert canvas._finish_volume_depth((1.0, 2.0)) is None
    assert drawn == []


def test_a_footprint_with_no_axis_named_makes_no_gate(canvas):
    canvas._pending_volume_gate = BoxGate(
        name="(unnamed)", x_column="a", y_column="b", z_column="c")
    canvas._pending_volume_axis = ""

    assert canvas._finish_volume_depth((1.0, 2.0)) is None


def test_a_depth_the_gate_has_no_side_for_is_explained_not_raised(canvas,
                                                                  said):
    """A guard the two-gesture flow does not currently trip: the footprint's
    own normal and the axis the depth is measured against are worked out
    separately, and nothing ties them together beyond both being read off the
    same anchor plane. Driven so a mismatch reads as a sentence naming the
    measurement rather than a traceback out of a mouse handler."""
    drawn: list = []
    canvas.gate_drawn.connect(drawn.append)
    canvas._pending_volume_gate = ThresholdGate(name="(unnamed)", column="a",
                                                low=0.0, high=1.0)
    canvas._pending_volume_axis = "c"

    assert canvas._finish_volume_depth((1.0, 2.0)) is None
    assert drawn == []
    assert "no bound on 'c'" in said[-1]


def test_a_flat_shape_needs_no_second_gesture(canvas):
    """In 2D there is no third measurement to extend along, so asking for a
    depth would be asking for something the plot cannot express -- the shape
    is finished on the spot instead of being held for a gesture that can
    never arrive. A guard today: a sweep can only start inside the volume."""
    canvas._mode = "2D"
    drawn: list = []
    canvas.gate_drawn.connect(drawn.append)
    gate = BoxGate(name="(unnamed)", x_column="a", y_column="b", z_column="c")

    canvas._begin_volume_depth(gate)
    assert drawn == [gate]
    assert canvas._pending_volume_gate is None


# ---------------------------------------------------------------------------
# The wheel
# ---------------------------------------------------------------------------

def test_the_wheel_zooms_the_volume_in_about_what_is_in_it(canvas, volume):
    """"i cant zoom in or spin on any of the axees." The limits move, not the
    camera: a gate is a statement in data units, and a camera trick would
    leave the outlines somewhere other than the objects they enclose."""
    canvas._apply_volume_zoom(volume)              # the resting view
    before = dict(volume.limits)

    assert canvas._volume_scroll(_Event(0.0, 0.0, step=1)) is True
    assert canvas._volume_zoom == pytest.approx(1.25)
    centres = {"x": 5.0, "y": 10.0, "z": 15.0}
    for axis in ("x", "y", "z"):
        low, high = volume.limits[axis]
        was_low, was_high = before[axis]
        assert high - low == pytest.approx((was_high - was_low) / 1.25)
        assert (low + high) / 2 == pytest.approx(centres[axis])


def test_a_wheel_with_no_step_still_zooms_the_way_the_button_says(canvas,
                                                                  volume):
    """Some backends report a button and no step. Ignoring those made the
    wheel dead on exactly the machines that report that way."""
    canvas._apply_volume_zoom(volume)              # the resting view
    resting = volume.limits["x"][1] - volume.limits["x"][0]

    assert canvas._volume_scroll(_Event(0.0, 0.0, step=0, button="down"))
    assert canvas._volume_zoom == pytest.approx(0.8)
    low, high = volume.limits["x"]
    assert high - low == pytest.approx(resting / 0.8)


def test_the_zoom_cannot_be_wound_past_useful(canvas, volume):
    """Unclamped, a few seconds on the wheel leaves the volume at a scale
    where the data is a single dot or entirely off the plot, and only
    Reset view gets out of it."""
    canvas._volume_zoom = 49.0
    canvas._volume_scroll(_Event(0.0, 0.0, step=1))
    assert canvas._volume_zoom == pytest.approx(50.0)

    canvas._volume_zoom = 0.06
    canvas._volume_scroll(_Event(0.0, 0.0, step=-1))
    assert canvas._volume_zoom == pytest.approx(0.05)


def test_the_wheel_on_a_flat_plot_is_left_to_the_2d_zoom(canvas):
    canvas._mode = "2D"
    assert canvas._volume_scroll(_Event(0.0, 0.0, step=1)) is False


def test_a_wheel_notch_with_no_axes_left_is_swallowed_not_raised(canvas,
                                                                 monkeypatch):
    """A guard, and a redundant one -- the volume check just above it asks
    the same question of the same axes. Driven so that if the two ever come
    apart the wheel refuses instead of raising."""
    monkeypatch.setattr(canvas, "_in_volume", lambda: True)
    canvas.axes_at = lambda *_a, **_k: None

    assert canvas._volume_scroll(_Event(0.0, 0.0, step=1)) is True
    assert canvas._volume_zoom == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The zoom itself
# ---------------------------------------------------------------------------

def test_zooming_in_halves_what_the_volume_shows(canvas, volume):
    canvas._volume_zoom = 2.0
    canvas._apply_volume_zoom(volume)

    # A spans 0..10: mean 5, three standard deviations is 15, which is the
    # half-width at zoom 1. At zoom 2 it is 7.5 either side of the mean.
    assert volume.limits["x"] == pytest.approx((-2.5, 12.5))


def test_a_measurement_that_never_varies_still_gets_room_to_be_seen(canvas,
                                                                    volume):
    """Its extent is zero, and an axis of zero height draws nothing at all."""
    canvas._frame = pd.DataFrame({"a": [7.0, 7.0], "b": [0.0, 20.0],
                                  "c": [0.0, 30.0]})
    canvas._apply_volume_zoom(volume)

    assert volume.limits["x"] == pytest.approx((6.0, 8.0))


def test_a_measurement_that_is_not_in_the_table_is_skipped(canvas, volume):
    canvas._z_column = "not_measured"
    canvas._apply_volume_zoom(volume)

    assert volume.limits["z"] == (0.0, 30.0)
    assert volume.limits["x"] != (0.0, 10.0)       # the others still zoomed


def test_a_measurement_that_is_all_blanks_is_skipped(canvas, volume):
    canvas._frame = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 20.0],
                                  "c": [np.nan, np.nan]})
    canvas._apply_volume_zoom(volume)

    assert volume.limits["z"] == (0.0, 30.0)


def test_zooming_with_nothing_loaded_leaves_the_axes_alone(canvas, volume):
    canvas._frame = None
    canvas._apply_volume_zoom(volume)

    assert volume.limits == {"x": (0.0, 10.0), "y": (0.0, 20.0),
                             "z": (0.0, 30.0)}


def test_the_same_measurement_on_two_axes_refuses_the_sweep(canvas, volume):
    """The X, Y and Z pickers are filled from ONE column list with nothing
    excluded (`_refill_axis_pickers`), so a user can put the same measurement
    on X and on Z. There is then no third axis to extend a footprint through.

    Both sweep handlers took the depth column with a bare `next(...)`, which
    raises StopIteration when nothing matches. `_show_volume_drag` runs on
    every motion event, so that was a traceback out of the mouse handler on
    every pixel of the drag, and no gate on release. `close_polygon` already
    refused this same situation politely; these two were missed.
    """
    from dataclasses import replace
    canvas._spec = replace(canvas._spec, x="a", y="b")
    canvas._z_column = "a"
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    _sees(canvas, ("a", 3.0, "b", 6.0), ("a", 3.0, "b", 6.0))

    assert canvas._volume_motion(_Event(40.0, 50.0)) is True
    assert volume.drawn == [], "nothing can be previewed through no depth"
    assert canvas._gate_from_volume_drag(_Event(40.0, 50.0)) is None


def test_the_same_measurement_on_two_axes_arms_no_plane_to_draw_on(canvas):
    """The other half of the duplicate-axis fault, and the root of both.

    A plane and the normal it is extended along are THREE measurements. When
    two of the pickers name the same one there is no such plane, and
    `anchor_plane` used to hand back a triple with a repeat in it --
    ('a', 'b', 'a'). Every caller then went wrong in its own way. The aura
    was the visible one: `order` is keyed by column name, so the repeat
    collapsed it to two entries and every corner of the quad kept a None in
    the slot nothing had filled --

        [[None, 0, 0], [None, 0, 0], [None, 20, 0], [None, 20, 0]]

    -- which is not a shape any renderer can draw. All three callers already
    handle None, which is what "there is no anchor plane" means.
    """
    from dataclasses import replace
    canvas.set_mode("3D", z_column="a")
    canvas._spec = replace(canvas._spec, x="a", y="b")
    canvas.set_anchor_axis("z")

    assert canvas.anchor_plane() is None, (
        "three axes showing two measurements do not make a plane")


def test_three_distinct_measurements_still_arm_their_plane(canvas):
    """The fix must not cost the ordinary case."""
    from dataclasses import replace
    canvas.set_mode("3D", z_column="c")
    canvas._spec = replace(canvas._spec, x="a", y="b")
    canvas.set_anchor_axis("z")

    assert canvas.anchor_plane() == ("a", "b", "c")
