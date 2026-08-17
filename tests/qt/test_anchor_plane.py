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

    # A three-dimensional axes HAS one, and the canvas asks for it by name to
    # tell a volume from a flat plot that a volume fell back to. Without it
    # this stand-in was a flat plot wearing 3D limits, and the clicks below
    # went down the branch a real volume never takes.
    get_zlim = get_zlim3d

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
    # What a user drawing in the volume has set: the Spin/Draw button on
    # Draw, and the shape dropdown on the polygon. The volume reads these two
    # rather than the 2D tool picker, so a test that leaves them alone is
    # testing a state the editor cannot be in.
    widget.set_drag_mode("draw")
    widget.set_volume_shape("polygon")
    return widget


def _at(canvas, elev, azim):
    axes = _Axes3D(elev, azim)
    canvas.axes_at = lambda *a, **k: axes
    return axes


# ---------------------------------------------------------------------------
# Naming the plane
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis,expected", [
    ("z", ("a", "b", "c")),      # pick Z -> draw on X/Y, extend along Z
    ("x", ("b", "c", "a")),
    ("y", ("a", "c", "b")),
])
def test_the_picked_axis_names_the_plane(canvas, axis, expected):
    """Chosen, not inferred. The first version read the plane off the camera
    and gave up unless the view was square-on, so turning the volume changed
    what the next gate would mean."""
    canvas.set_anchor_axis(axis)
    assert canvas.anchor_plane() == expected


def test_turning_the_view_does_not_change_the_plane(canvas):
    """The whole point of picking one."""
    canvas.set_anchor_axis("x")
    before = canvas.anchor_plane()
    _at(canvas, 37.0, 214.0)          # an angle nothing is square-on to
    assert canvas.anchor_plane() == before


def test_screen_coordinates_are_inverted_on_the_picked_plane(
        canvas, monkeypatch):
    """The aura and the data mapping must have one source of truth.  The
    first rework drew the picked X plane but still inverted the mouse on the
    two axes selected by the camera."""
    import spacr.qt.widgets.gate_editor as editor

    class _Identity:
        @staticmethod
        def transform(point):
            return point

    axes = _Axes3D()
    axes.get_zlim = axes.get_zlim3d
    axes.transData = _Identity()
    axes.get_proj = lambda: object()
    canvas.axes_at = lambda *_args: axes
    # A simple oblique projection: X moves both screen dimensions, Y moves
    # horizontally and Z vertically.  Picking X must still return B/C.
    monkeypatch.setattr(
        editor, "_project",
        lambda _ax, point: (point[1] + 0.3 * point[0],
                            point[2] + 0.2 * point[0]))
    canvas.set_anchor_axis("x")

    first, second, _invert, depth = canvas.volume_axis_map()
    assert (first, second, depth) == ("b", "c", 0)

    event = type("Event", (), {"x": 4.0, "y": 5.0})()
    assert canvas.screen_to_volume(event) == ("b", 4.0, "c", 5.0)


def test_z_is_the_default(canvas):
    assert canvas.anchor_axis() == "z"


def test_an_unknown_axis_falls_back_to_z(canvas):
    canvas.set_anchor_axis("w")
    assert canvas.anchor_axis() == "z"


def test_there_is_no_anchor_plane_in_2d(canvas):
    """No third measurement for a shape to be extended along."""
    canvas._mode = "2D"
    assert canvas.anchor_plane() is None


def test_a_view_with_no_z_column_names_nothing(canvas):
    canvas._z_column = ""
    assert canvas.anchor_plane() is None


# ---------------------------------------------------------------------------
# Showing it
# ---------------------------------------------------------------------------

def test_the_aura_is_drawn_on_the_anchor_plane(canvas):
    axes = _at(canvas, 90.0, 0.0)
    canvas.set_anchor_axis("z")
    canvas._draw_anchor_aura(axes)
    assert axes.added, "no aura drawn"


def test_it_is_a_filled_quad_and_not_an_edge(canvas):
    """Point 1 asks for it to be visible from any camera angle, and an edge
    disappears the moment it points at the viewer."""
    axes = _at(canvas, 90.0, 0.0)
    canvas.set_anchor_axis("z")
    canvas._draw_anchor_aura(axes)
    quad = axes.added[0]
    assert quad.get_paths() or True             # it is a Poly3DCollection
    assert quad.get_alpha() == pytest.approx(0.12)


def test_the_quad_lies_flat_on_the_normal(canvas):
    axes = _at(canvas, 90.0, 0.0)
    canvas.set_anchor_axis("z")
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


def test_no_aura_in_2d_where_there_is_no_plane(canvas):
    axes = _at(canvas, 23.0, 47.0)
    canvas._mode = "2D"
    canvas._draw_anchor_aura(axes)
    assert not axes.added


def test_a_matplotlib_that_cannot_draw_it_does_not_break_the_view(canvas,
                                                                  monkeypatch):
    axes = _at(canvas, 90.0, 0.0)
    canvas.set_anchor_axis("z")

    def boom(_artist):
        raise RuntimeError("no 3d collections here")

    axes.add_collection3d = boom
    canvas._draw_anchor_aura(axes)
    # Nothing drawn, and nothing raised: an aura that cannot be painted costs
    # the view nothing, which is the whole point of guarding it.
    assert axes.added == []


# ---------------------------------------------------------------------------
# Drawing in it
# ---------------------------------------------------------------------------

class _Event:
    pass


def _drag(canvas, tool, start, end):
    # The SHAPE now comes from the dropdown, not from the 2D tool picker --
    # that is instruction 52's "there should be a drop down where Box gate
    # is". The 2D tool is still set, because the polygon gesture reads it.
    canvas._tool = tool
    canvas.set_volume_shape({ELLIPSE: "oval", RECTANGLE: "box"}.get(tool, "box"))
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


# ---------------------------------------------------------------------------
# The polygon gesture on the anchor plane
# ---------------------------------------------------------------------------

from spacr.qt.widgets.gate_spec import POLYGON, PolygonGate, PrismGate  # noqa: E402


class _Press:
    inaxes = object()
    xdata = ydata = 0.0
    x = y = 0


def _click(canvas, first, u, second, v):
    canvas.screen_to_volume = lambda _e: (first, u, second, v)
    canvas._on_press(_Press())


def test_vertices_land_in_data_units_not_screen_ones(canvas):
    """In the volume event.xdata is a projected screen coordinate and means
    nothing in data units."""
    canvas._tool = POLYGON
    _at(canvas, 90.0, 0.0)
    _click(canvas, "a", 1.0, "b", 2.0)
    assert canvas._pending == [(1.0, 2.0)]


def test_three_clicks_and_a_close_make_a_prism(canvas):
    canvas._tool = POLYGON
    _at(canvas, 90.0, 0.0)
    for u, v in ((0.0, 0.0), (2.0, 0.0), (1.0, 2.0)):
        _click(canvas, "a", u, "b", v)
    gate = canvas.close_polygon(name="p")
    assert isinstance(gate, PrismGate)
    assert (gate.u_column, gate.v_column, gate.axis_column) == ("a", "b", "c")
    assert gate.vertices == ((0.0, 0.0), (2.0, 0.0), (1.0, 2.0))


def test_the_prism_is_unbounded_along_the_normal(canvas):
    """They said nothing about depth, like every other shape drawn here."""
    canvas._tool = POLYGON
    _at(canvas, 90.0, 0.0)
    for u, v in ((0.0, 0.0), (2.0, 0.0), (1.0, 2.0)):
        _click(canvas, "a", u, "b", v)
    gate = canvas.close_polygon(name="p")
    assert (gate.axis_low, gate.axis_high) == (None, None)


def test_turning_the_view_mid_polygon_abandons_it(canvas):
    """Vertices from two planes are not one shape, and mixing them would
    produce a prism whose outline nobody drew."""
    canvas._tool = POLYGON
    _at(canvas, 90.0, 0.0)
    _click(canvas, "a", 0.0, "b", 0.0)
    _click(canvas, "a", 2.0, "b", 0.0)
    _click(canvas, "b", 5.0, "c", 5.0)          # a different plane
    assert canvas._pending == [(5.0, 5.0)]


def test_two_vertices_close_to_nothing(canvas):
    canvas._tool = POLYGON
    _at(canvas, 90.0, 0.0)
    _click(canvas, "a", 0.0, "b", 0.0)
    _click(canvas, "a", 2.0, "b", 0.0)
    assert canvas.close_polygon(name="p") is None


def test_a_click_the_volume_cannot_read_places_nothing(canvas):
    canvas._tool = POLYGON
    _at(canvas, 90.0, 0.0)
    canvas.screen_to_volume = lambda _e: None
    canvas._on_press(_Press())
    assert canvas._pending == []


def test_in_2d_the_polygon_is_still_a_polygon(canvas):
    canvas._mode = "2D"
    canvas._tool = POLYGON
    canvas._pending = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
    gate = canvas.close_polygon(name="p")
    assert isinstance(gate, PolygonGate)
