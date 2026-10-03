"""The gate editor's 3D mode orbits freely and draws gates at any angle.

The maintainer, 2026-10-03: "the gating in the gate editor in 3D is not
usable. And the axes still snap to one centered and one set at no angle in
3D mode."

The free spin is an orbit: the vertical axis stays upright, the camera never
rolls and the view stays wherever a drag leaves it. Gates are drawn on the
screen at whatever angle the volume shows -- a clicked polygon swept along
the line of sight, or a box or ellipsoid fitted to a framed population and
refined with handles -- and they feed the same gate set as 2D.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.gate_editor import _screen_points
from spacr.qt.widgets.gate_spec import (
    BoxGate, GateSet, ViewGate, _EllipsoidGate, gate_from_dict,
)


def _table():
    rng = np.random.default_rng(7)
    near = rng.normal(0.0, 0.4, (300, 3))
    far = rng.normal(4.0, 0.4, (300, 3))
    both = np.vstack([near, far])
    return pd.DataFrame({"a": both[:, 0], "b": both[:, 1], "c": both[:, 2]})


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1100, 800)
    widget.set_frame(_table())
    widget._x.setCurrentText("a")
    widget._y.setCurrentText("b")
    widget.gates.mode_requested.emit("3D")
    widget._z.setCurrentText("c")
    names = iter(f"g{i}" for i in range(1, 100))
    widget.gates.set_namer(lambda: next(names))
    return widget


class _Mouse:
    def __init__(self, ax, x, y, button=1, dblclick=False):
        self.inaxes, self.x, self.y = ax, float(x), float(y)
        self.xdata = self.ydata = 0.0
        self.button = button
        self.step = 0
        self.dblclick = dblclick


def _drag(canvas, start, end, steps=8):
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, *start))
    for i in range(1, steps + 1):
        t = i / steps
        canvas._on_motion(_Mouse(ax, start[0] + (end[0] - start[0]) * t,
                                 start[1] + (end[1] - start[1]) * t))
    release = _Mouse(ax, *end)
    canvas._on_release(release)
    canvas._on_button_release(release)


def _angles(ax):
    return (float(ax.elev), float(ax.azim), float(ax.roll))


def _pick_shape(screen, key):
    picker = screen.gates._volume_shape
    picker.setCurrentIndex(picker.findData(key))
    screen.gates._drag_buttons["draw"].click()


def _centre_px(canvas, point):
    return _screen_points(canvas.axes_at(0, 0), [point])[0]


def test_the_default_view_is_tilted_not_square_on(screen):
    elev, azim, roll = _angles(screen.gates.canvas.axes_at(0, 0))
    assert 10.0 <= elev <= 60.0
    assert azim % 90.0 != pytest.approx(0.0)
    assert roll == pytest.approx(0.0)


def test_an_orbit_keeps_its_angle_and_never_rolls_or_snaps(screen, qtbot):
    canvas = screen.gates.canvas
    _drag(canvas, (300, 300), (357, 331))
    released = _angles(canvas.axes_at(0, 0))
    assert released[2] == pytest.approx(0.0)
    assert released[0] not in (0.0, 90.0, -90.0)
    assert released[1] % 90.0 != pytest.approx(0.0)
    for _ in range(10):
        QApplication.processEvents()
        qtbot.wait(5)
    canvas.render_now()
    assert _angles(canvas.axes_at(0, 0)) == pytest.approx(released)
    _drag(canvas, (300, 300), (330, 300))
    turned = _angles(canvas.axes_at(0, 0))
    assert turned[0] == pytest.approx(released[0])
    assert turned[1] != pytest.approx(released[1])


def test_drawing_and_zooming_leave_the_angle_alone(screen):
    canvas = screen.gates.canvas
    _drag(canvas, (300, 300), (340, 350))
    turned = _angles(canvas.axes_at(0, 0))
    ax = canvas.axes_at(0, 0)
    canvas._on_scroll(_Mouse(ax, 300, 300))
    _pick_shape(screen, "lasso")
    _drag(canvas, (100, 100), (140, 140))
    assert _angles(canvas.axes_at(0, 0)) == pytest.approx(turned)


def _click_polygon(canvas, centre, half=70.0):
    ax = canvas.axes_at(0, 0)
    corners = [(centre[0] - half, centre[1] - half),
               (centre[0] + half, centre[1] - half),
               (centre[0] + half, centre[1] + half),
               (centre[0] - half, centre[1] + half)]
    for corner in corners:
        canvas._on_press(_Mouse(ax, *corner))
        canvas._on_release(_Mouse(ax, *corner))
    canvas._on_motion(_Mouse(ax, centre[0], centre[1]))
    return corners


def test_a_clicked_polygon_selects_the_population_it_surrounds(screen):
    canvas = screen.gates.canvas
    _drag(canvas, (300, 300), (350, 330))
    _pick_shape(screen, "view_polygon")
    centre = _centre_px(canvas, [4.0, 4.0, 4.0])
    corners = _click_polygon(canvas, centre)
    assert canvas._live is not None
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, *corners[0]))
    canvas._on_release(_Mouse(ax, *corners[0]))
    gate = canvas.gates.get("g1")
    assert isinstance(gate, ViewGate)
    inside = canvas.gates.mask(canvas.population(), "g1")
    assert inside[300:].mean() > 0.9
    assert not inside[:300].any()


def test_a_double_click_also_closes_the_polygon(screen):
    canvas = screen.gates.canvas
    _pick_shape(screen, "view_polygon")
    centre = _centre_px(canvas, [0.0, 0.0, 0.0])
    corners = _click_polygon(canvas, centre, half=60.0)
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, corners[-1][0] + 1, corners[-1][1],
                            dblclick=True))
    assert "g1" in canvas.gates.names
    inside = canvas.gates.mask(canvas.population(), "g1")
    assert inside[:300].mean() > 0.9 and not inside[300:].any()


def test_the_3d_gate_lands_in_the_same_gate_set_and_columns_as_2d(screen):
    canvas = screen.gates.canvas
    _pick_shape(screen, "view_polygon")
    centre = _centre_px(canvas, [4.0, 4.0, 4.0])
    corners = _click_polygon(canvas, centre)
    canvas.close_polygon_now()
    gates = screen.gates.gates
    assert "g1" in gates.names
    assert set(gates.get("g1").columns) == {"a", "b", "c"}
    again = gate_from_dict(json.loads(json.dumps(gates.get("g1").to_dict())))
    frame = canvas.population()
    assert np.array_equal(again.mask(frame), gates.mask(frame, "g1"))


@pytest.mark.parametrize("shape, kind", [("ellipsoid", _EllipsoidGate),
                                         ("box_handles", BoxGate)])
def test_a_framed_population_becomes_a_fitted_solid(screen, shape, kind):
    canvas = screen.gates.canvas
    _pick_shape(screen, shape)
    centre = _centre_px(canvas, [0.0, 0.0, 0.0])
    _drag(canvas, (centre[0] - 70, centre[1] - 70),
          (centre[0] + 70, centre[1] + 70))
    gate = canvas.gates.get("g1")
    assert isinstance(gate, kind)
    inside = canvas.gates.mask(canvas.population(), "g1")
    assert inside[:300].mean() > 0.6 and not inside[300:].any()


def test_pulling_an_ellipsoid_handle_resizes_it(screen):
    canvas = screen.gates.canvas
    _pick_shape(screen, "ellipsoid")
    centre = _centre_px(canvas, [0.0, 0.0, 0.0])
    _drag(canvas, (centre[0] - 70, centre[1] - 70),
          (centre[0] + 70, centre[1] + 70))
    before = canvas.gates.get("g1")
    assert canvas.active_gate == "g1"
    handles = dict(canvas._volume_handles(canvas._handle_gate()))
    tip = _screen_points(canvas.axes_at(0, 0), [handles["x_high"]])[0]
    axis = _screen_points(canvas.axes_at(0, 0),
                          [handles["centre"]])[0] - tip
    away = tip - axis / np.hypot(*axis) * 40.0
    _drag(canvas, tuple(tip), tuple(away))
    after = canvas.gates.get("g1")
    assert after.x_radius > before.x_radius
    assert after.y_radius == pytest.approx(before.y_radius)


def test_the_centre_handle_moves_a_box_through_the_volume(screen):
    canvas = screen.gates.canvas
    _pick_shape(screen, "box_handles")
    centre = _centre_px(canvas, [0.0, 0.0, 0.0])
    _drag(canvas, (centre[0] - 70, centre[1] - 70),
          (centre[0] + 70, centre[1] + 70))
    before = canvas.gates.get("g1")
    handles = dict(canvas._volume_handles(canvas._handle_gate()))
    grip = _screen_points(canvas.axes_at(0, 0), [handles["centre"]])[0]
    _drag(canvas, tuple(grip), (grip[0] + 60, grip[1] + 30))
    after = canvas.gates.get("g1")
    moved = np.array([after.x_low - before.x_low, after.y_low - before.y_low,
                      after.z_low - before.z_low])
    assert np.abs(moved).max() > 0.1
    assert after.x_high - after.x_low == pytest.approx(
        before.x_high - before.x_low)


def test_a_gate_is_renamed_with_its_children(screen):
    tree = screen.gates.tree
    gates = screen.gates.gates
    gates.add(BoxGate.from_limits("outer", ("a", "b", "c"),
                                  [(-1, 1), (-1, 1), (-1, 1)]))
    gates.add(BoxGate.from_limits("inner", ("a", "b", "c"),
                                  [(-.5, .5), (-.5, .5), (-.5, .5)],
                                  parent="outer"))
    tree.set_gates(gates, screen.gates.canvas.population())
    assert tree._rename_gate("outer", "cells")
    assert "cells" in gates.names and "outer" not in gates.names
    assert gates.get("inner").parent == "cells"
    assert not tree._rename_gate("inner", "cells")
    tree.select("inner")
    tree.remove_selected()
    assert "inner" not in gates.names


def test_the_ellipsoid_round_trips_and_counts_by_distance():
    gate = _EllipsoidGate(name="e", x_column="a", y_column="b", z_column="c",
                          x_radius=1.0, y_radius=2.0, z_radius=3.0)
    frame = pd.DataFrame({"a": [0.0, 0.9, 1.1, 0.0], "b": [0.0, 0, 0, 1.9],
                          "c": [0.0, 0, 0, 0.5]})
    assert gate.mask(frame).tolist() == [True, True, False, True]
    again = gate_from_dict(json.loads(json.dumps(gate.to_dict())))
    assert again == gate
    assert GateSet([gate]).mask(frame, "e").sum() == 3
