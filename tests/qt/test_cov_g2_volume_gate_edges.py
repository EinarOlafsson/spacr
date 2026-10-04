"""The 3-D gate canvas when there is nothing to pick, grab or move."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import gate_editor as ge  # noqa: E402
from spacr.qt.widgets.gate_spec import BoxGate, ViewGate, _EllipsoidGate  # noqa: E402
from tests.qt.test_gate_editor_3d_mode_is_usable import _Mouse, screen  # noqa: E402,F401


@pytest.fixture
def canvas(screen):  # noqa: F811
    return screen.gates.canvas


def _ellipsoid():
    return _EllipsoidGate(name="e", x_column="a", y_column="b", z_column="c",
                          x_centre=0.0, y_centre=0.0, z_centre=0.0,
                          x_radius=1.0, y_radius=1.0, z_radius=1.0)


def test_screen_points_of_unprojectable_data_are_none():
    class _Broken:
        @property
        def transData(self):
            raise RuntimeError("no transform")

    assert ge._screen_points(_Broken(), [(0, 0, 0)]) is None


def test_outlines_and_shapes_that_select_nothing(canvas, monkeypatch):
    assert canvas._inside_outline([(0, 0), (1, 1)]) is None
    monkeypatch.setattr(ge, "_screen_points", lambda ax, points: None)
    assert canvas._inside_outline([(0, 0), (1, 0), (1, 1)]) is None
    assert canvas._fitted_gate_from_outline([(0, 0), (1, 0), (1, 1)]) is None
    monkeypatch.undo()
    far = [(-900, -900), (-899, -900), (-899, -899)]
    assert canvas._fitted_gate_from_outline(far) is None


def test_an_empty_polygon_draws_nothing(canvas):
    canvas._view_polygon = []
    canvas._show_view_polygon(None)
    assert canvas._ghost == []


def test_an_outline_around_nothing_asks_again(canvas, monkeypatch):
    asked = []
    canvas.depth_requested.connect(asked.append)
    monkeypatch.setattr(canvas, "gate_from_screen_outline", lambda outline: None)
    canvas._view_polygon = [(1, 1), (2, 1), (2, 2)]
    canvas._finish_view_polygon()
    assert asked and "encloses nothing" in asked[-1]


def test_the_volume_needs_three_plotted_columns(canvas, monkeypatch):
    monkeypatch.setattr(canvas, "_plot_columns", lambda: ("a", "b", None))
    assert canvas._volume_xyz() is None


def test_no_handles_means_nothing_to_grab(canvas, monkeypatch):
    ax = canvas.axes_at(0, 0)
    monkeypatch.setattr(canvas, "_handle_gate", lambda: _ellipsoid())
    monkeypatch.setattr(canvas, "_volume_handles", lambda gate: [])
    assert canvas._grab_volume_handle(_Mouse(ax, 10, 10)) is False
    monkeypatch.setattr(canvas, "_volume_handles",
                        lambda gate: [("x_high", np.zeros(3))])
    monkeypatch.setattr(ge, "_screen_points", lambda ax, points: None)
    assert canvas._grab_volume_handle(_Mouse(ax, 10, 10)) is False
    monkeypatch.setattr(ge, "_screen_points",
                        lambda ax, points: np.array([[500.0, 500.0]]))
    assert canvas._grab_volume_handle(_Mouse(ax, 10, 10)) is False


def test_a_view_gate_without_a_population_has_no_handles(canvas, monkeypatch):
    gate = ViewGate.__new__(ViewGate)
    monkeypatch.setattr(canvas, "_volume_xyz", lambda frame=None: None)
    assert canvas._volume_handles(gate) == []


def test_moving_a_handle_needs_a_drag_an_axis_and_a_projection(canvas,
                                                              monkeypatch):
    ax = canvas.axes_at(0, 0)
    canvas._handle_drag = None
    assert canvas._handle_moved(_Mouse(ax, 1, 1)) is None
    canvas._handle_drag = ("e", "x_high", (0.0, 0.0), _ellipsoid(),
                           np.zeros(3))
    monkeypatch.setattr(canvas, "_pixels_per_unit", lambda ax, point: None)
    assert canvas._handle_moved(_Mouse(ax, 1, 1)) is None
    monkeypatch.setattr(canvas, "axes_at", lambda r, c: None)
    assert canvas._handle_moved(_Mouse(ax, 1, 1)) is None
    canvas._handle_drag = None


def test_pixels_per_unit_is_none_when_projection_fails(canvas, monkeypatch):
    monkeypatch.setattr(ge, "_screen_points", lambda ax, points: None)
    assert canvas._pixels_per_unit(canvas.axes_at(0, 0), np.zeros(3)) is None


def test_moving_an_ellipsoid_shifts_its_centre(canvas):
    moved = canvas._moved_gate(_ellipsoid(), {"a": 1.0, "b": 2.0, "c": 3.0},
                               np.zeros(3), np.zeros(3))
    assert (moved.x_centre, moved.y_centre, moved.z_centre) == (1.0, 2.0, 3.0)


def test_handles_need_an_enabled_closed_gate_of_a_known_kind(canvas,
                                                           monkeypatch):
    from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate

    box = BoxGate.from_limits("b", ("a", "b", "c"),
                              [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    flat = ThresholdGate(name="t", column="a", low=0.0)
    canvas._gates = GateSet([box, flat])
    monkeypatch.setattr(canvas, "_shows_columns", lambda gate: True)
    canvas._active = "b"
    monkeypatch.setattr(canvas, "is_gate_enabled", lambda name: False)
    assert canvas._handle_gate() is None
    monkeypatch.setattr(canvas, "is_gate_enabled", lambda name: True)
    canvas._volume_preview = replace(box, x_low=None)
    assert canvas._handle_gate() is None
    canvas._volume_preview = None
    assert canvas._handle_gate() == box
    canvas._active = "t"
    assert canvas._handle_gate() is None
