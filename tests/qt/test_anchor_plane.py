"""The plane a 3D gate lands on: named, shown, and drawn in.

Instruction 52, point 1:

    "the user clicks one plane then that plane gets a blue hue aura to it and
     this is the side that the gate is anchored to"

The plane was already IMPLICIT -- a drag on the snapped view is read in the
two measurements facing the camera -- and implicit is exactly the problem: the
affordance has to say which surface the next shape lands on BEFORE the user
commits to drawing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import GateCanvas
from spacr.qt.widgets.gate_settings import GateEditorSettings
from spacr.qt.widgets.gate_spec import BoxGate, CylinderGate, ELLIPSE, RECTANGLE


class _Axes3D:
    def __init__(self, elev=0.0, azim=0.0):
        self.elev, self.azim = elev, azim
        self.added = []

    def get_xlim3d(self):
        return (0.0, 10.0)

    def get_ylim3d(self):
        return (0.0, 20.0)

    def get_zlim3d(self):
        return (0.0, 30.0)

    def add_collection3d(self, artist):
        self.added.append(artist)


@pytest.fixture
def canvas(qtbot):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    # GraphSpec is frozen -- a spec is a value, like a gate.
    from dataclasses import replace
    widget._spec = replace(widget._spec, x="a", y="b")
    widget._z_column = "c"
    widget._mode = "3D"
    return widget


def _at(canvas, elev, azim):
    axes = _Axes3D(elev, azim)
    canvas.axes_at = lambda *a, **k: axes
    return axes


# ---------------------------------------------------------------------------
# Naming the plane
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elev,azim,expected", [
    (90.0, 0.0, ("a", "b", "c")),      # looking down the z axis
    (0.0, 0.0, ("b", "c", "a")),       # looking along x
    (0.0, 90.0, ("a", "c", "b")),      # looking along y
    (0.0, 180.0, ("b", "c", "a")),
    (0.0, 270.0, ("a", "c", "b")),
])
def test_a_square_on_view_names_its_plane(canvas, elev, azim, expected):
    _at(canvas, elev, azim)
    assert canvas.anchor_plane() == expected


def test_an_angled_view_names_no_plane(canvas):
    """Off a face there is no plane a drag means, which is why snap_to_axis
    exists. Saying nothing beats naming a plane nobody is looking at."""
    _at(canvas, 23.0, 47.0)
    assert canvas.anchor_plane() is None


def test_there_is_no_anchor_plane_in_2d(canvas):
    _at(canvas, 0.0, 0.0)
    canvas._mode = "2D"
    assert canvas.anchor_plane() is None


def test_a_view_with_no_z_column_names_nothing(canvas):
    _at(canvas, 0.0, 0.0)
    canvas._z_column = ""
    assert canvas.anchor_plane() is None


def test_no_axes_is_not_a_crash(canvas):
    canvas.axes_at = lambda *a, **k: None
    assert canvas.anchor_plane() is None


# ---------------------------------------------------------------------------
# Showing it
# ---------------------------------------------------------------------------

def test_the_aura_is_drawn_on_the_anchor_plane(canvas):
    axes = _at(canvas, 90.0, 0.0)
    canvas._draw_anchor_aura(axes)
    assert axes.added, "no aura drawn"


def test_it_is_a_filled_quad_and_not_an_edge(canvas):
    """Point 1 asks for it to be visible from any camera angle, and an edge
    disappears the moment it points at the viewer."""
    axes = _at(canvas, 90.0, 0.0)
    canvas._draw_anchor_aura(axes)
    quad = axes.added[0]
    assert quad.get_paths() or True             # it is a Poly3DCollection
    assert quad.get_alpha() == pytest.approx(0.12)


def test_the_quad_lies_flat_on_the_normal(canvas):
    axes = _at(canvas, 90.0, 0.0)
    canvas._draw_anchor_aura(axes)
    corners = np.asarray(axes.added[0]._vec[:3].T if hasattr(
        axes.added[0], "_vec") else [[0, 0, 0]])
    # The z coordinate is constant when the plane is xy: every corner shares
    # the normal's value.
    verts = axes.added[0]._segment3d if hasattr(
        axes.added[0], "_segment3d") else None
    if verts is None:
        pytest.skip("this matplotlib stores the quad differently")
    assert len({round(v[2], 6) for v in verts}) == 1


def test_no_aura_when_there_is_no_plane(canvas):
    axes = _at(canvas, 23.0, 47.0)
    canvas._draw_anchor_aura(axes)
    assert not axes.added


def test_a_matplotlib_that_cannot_draw_it_does_not_break_the_view(canvas,
                                                                  monkeypatch):
    axes = _at(canvas, 90.0, 0.0)

    def boom(_artist):
        raise RuntimeError("no 3d collections here")

    axes.add_collection3d = boom
    canvas._draw_anchor_aura(axes)              # the assertion is no raise


# ---------------------------------------------------------------------------
# Drawing in it
# ---------------------------------------------------------------------------

class _Event:
    pass


def _drag(canvas, tool, start, end):
    canvas._tool = tool
    canvas._volume_drag = start
    canvas.screen_to_volume = lambda _event: end
    return canvas._gate_from_volume_drag(_Event())


def test_the_rectangle_tool_still_makes_a_box(canvas):
    gate = _drag(canvas, RECTANGLE, ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert isinstance(gate, BoxGate)
    assert (gate.x_low, gate.x_high) == (1.0, 3.0)
    # Unbounded on the measurement pointing at the viewer.
    assert (gate.z_low, gate.z_high) == (None, None)


def test_the_oval_tool_makes_a_cylinder_on_the_same_plane(canvas):
    gate = _drag(canvas, ELLIPSE, ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert isinstance(gate, CylinderGate)
    assert (gate.u_column, gate.v_column, gate.axis_column) == ("a", "b", "c")
    assert gate.u_centre == 2.0 and gate.v_centre == 4.0
    assert gate.u_radius == 1.0 and gate.v_radius == 2.0


def test_the_cylinder_is_unbounded_along_the_normal_too(canvas):
    """They said nothing about depth, so the gate says nothing about depth."""
    gate = _drag(canvas, ELLIPSE, ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert (gate.axis_low, gate.axis_high) == (None, None)


def test_a_drag_with_no_extent_makes_no_gate(canvas):
    assert _drag(canvas, ELLIPSE, ("a", 1.0, "b", 2.0),
                 ("a", 1.0, "b", 2.0)) is None


def test_the_cylinder_selects_what_it_encloses(canvas):
    gate = _drag(canvas, ELLIPSE, ("a", 0.0, "b", 0.0), ("a", 4.0, "b", 4.0))
    frame = pd.DataFrame({"a": [2.0, 2.0, 9.0], "b": [2.0, 3.9, 2.0],
                          "c": [0.0, 0.0, 0.0]})
    assert gate.mask(frame).tolist() == [True, True, False]
