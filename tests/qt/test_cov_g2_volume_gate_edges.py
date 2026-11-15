"""The 3-D gate canvas when there is nothing to pick, grab or move."""
from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import gate_editor as ge  # noqa: E402
from spacr.qt.widgets.gate_spec import ViewGate  # noqa: E402
from tests.qt.test_gate_editor_3d_mode_is_usable import _Mouse, screen  # noqa: E402,F401


@pytest.fixture
def canvas(screen):  # noqa: F811
    return screen.gates.canvas


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


def test_a_view_gate_without_a_population_has_no_handles(canvas, monkeypatch):
    gate = ViewGate.__new__(ViewGate)
    monkeypatch.setattr(canvas, "_volume_xyz", lambda frame=None: None)
    assert canvas._volume_handles(gate) == []


def test_pixels_per_unit_is_none_when_projection_fails(canvas, monkeypatch):
    monkeypatch.setattr(ge, "_screen_points", lambda ax, points: None)
    assert canvas._pixels_per_unit(canvas.axes_at(0, 0), np.zeros(3)) is None


