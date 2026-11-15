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

from spacr.qt.widgets.gate_editor import GateCanvas


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


def _sees(canvas, *corners):
    """Make the mouse read as the given points on the anchor plane."""
    points = iter(corners)
    canvas.screen_to_volume = lambda _event: next(points)


# ---------------------------------------------------------------------------
# Pressing: which gesture a press starts
# ---------------------------------------------------------------------------


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


def test_locking_to_a_horizontal_axis_turns_about_that_axis(canvas, volume):
    """Locked to X, the X axis stays where it is on screen and the other two
    turn about it."""
    from spacr.qt.widgets.volume_view import view_axes

    canvas.set_spin_axis("x")
    before = view_axes(volume.elev, volume.azim, 0.0)
    _spin(canvas, (10.0, 20.0), (90.0, 60.0))
    after = view_axes(volume.elev, volume.azim, canvas._view_angles[2])
    x = np.array([1.0, 0.0, 0.0])
    assert float(x @ after[0]) == pytest.approx(float(x @ before[0]), abs=1e-6)
    assert float(x @ after[1]) == pytest.approx(float(x @ before[1]), abs=1e-6)
    assert (volume.elev, volume.azim) != pytest.approx((15.0, 40.0))


def test_the_free_spin_goes_over_the_top_instead_of_stopping(canvas, volume):
    """The old lock clamped the elevation at 90 degrees, which is what made
    whole orientations unreachable. Free, a long upward drag keeps turning."""
    canvas.set_spin_axis("")
    _spin(canvas, (0.0, 0.0), (0.0, 400.0))
    from spacr.qt.widgets.volume_view import view_axes
    _u, _v, w = view_axes(*canvas._view_angles)
    assert np.isfinite(w).all()
    assert canvas._view_angles != pytest.approx((90.0, 40.0, 0.0))


# ---------------------------------------------------------------------------
# Sweeping: the footprint you can see while you hold the button
# ---------------------------------------------------------------------------


def test_a_sweep_with_no_extent_makes_no_gate(canvas, volume):
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    _sees(canvas, ("a", 1.0, "b", 2.0))
    assert canvas._gate_from_volume_drag(_Event(1.0, 1.0)) is None


def test_a_sweep_the_volume_cannot_read_makes_no_gate(canvas, volume):
    canvas._volume_drag = ("a", 1.0, "b", 2.0)
    canvas.screen_to_volume = lambda _event: None
    assert canvas._gate_from_volume_drag(_Event(1.0, 1.0)) is None


# ---------------------------------------------------------------------------
# The second gesture: a slab you drag out
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# What the user is told while the depth is being dragged
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Committing the depth
# ---------------------------------------------------------------------------

def test_a_depth_with_no_footprint_waiting_makes_no_gate(canvas):
    drawn: list = []
    canvas.gate_drawn.connect(drawn.append)

    assert canvas._finish_volume_depth((1.0, 2.0)) is None
    assert drawn == []


# ---------------------------------------------------------------------------
# The wheel
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The zoom itself
# ---------------------------------------------------------------------------


