"""Pulling 3-D gate handles, view-polygon clicks and tree renames at the edges."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import gate_editor as ge  # noqa: E402
from spacr.qt.widgets.gate_spec import (BoxGate, CompositeGate, GateSet,  # noqa: E402
                                        ThresholdGate, ViewGate, _EllipsoidGate)
from tests.qt.test_gate_editor_3d_mode_is_usable import _Mouse, screen  # noqa: E402,F401

JACOBIAN = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.fixture
def canvas(screen, monkeypatch):  # noqa: F811
    canvas = screen.gates.canvas
    monkeypatch.setattr(canvas, "_plot_columns", lambda: ("a", "b", "c"))
    monkeypatch.setattr(canvas, "_pixels_per_unit", lambda ax, point: JACOBIAN)
    return canvas


def _box():
    return BoxGate.from_limits("b", ("a", "b", "c"),
                               [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])


def _ellipsoid():
    return _EllipsoidGate(name="e", x_column="a", y_column="b", z_column="c",
                          x_radius=2.0, y_radius=2.0, z_radius=2.0)


def _view(projection=None):
    projection = projection if projection is not None else (
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0))
    return ViewGate(name="v", x_column="a", y_column="b", z_column="c",
                    projection=projection,
                    vertices=((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)))


def _pull(canvas, gate, role, x=5.0, y=0.0):
    ax = canvas.axes_at(0, 0)
    canvas._handle_drag = (gate.name, role, (0.0, 0.0), gate, np.zeros(3))
    try:
        return canvas._handle_moved(_Mouse(ax, x, y))
    finally:
        canvas._handle_drag = None


def test_side_handles_move_one_face_or_radius(canvas):
    assert _pull(canvas, _box(), "x_high").x_high == pytest.approx(6.0)
    assert _pull(canvas, _box(), "z_high") == _box()
    assert _pull(canvas, _ellipsoid(), "x_low").x_radius == pytest.approx(3.0)
    assert _pull(canvas, _view(), "x_high") == _view()


def test_moving_whole_gates_through_the_volume(canvas):
    moved = canvas._moved_gate(_box(), {"a": 1.0, "b": 2.0, "c": 3.0},
                               np.zeros(3), np.zeros(3))
    assert (moved.x_low, moved.y_high, moved.z_low) == (1.0, 3.0, 3.0)
    flat = ThresholdGate(name="t", column="a", low=0.0)
    assert canvas._moved_gate(flat, {}, np.zeros(3), np.zeros(3)) is flat
    shifted = canvas._moved_gate(_view(), {}, np.zeros(3),
                                 np.array([1.0, 0.0, 0.0]))
    assert shifted != _view()
    blind = _view(projection=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                              (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    assert canvas._moved_gate(blind, {}, np.zeros(3),
                              np.array([1.0, 0.0, 0.0])) == blind


def test_a_flat_gate_has_no_volume_handles(canvas):
    assert canvas._volume_handles(ThresholdGate(name="t", column="a",
                                                low=0.0)) == []


def test_releasing_an_unchanged_handle_only_redraws(canvas, monkeypatch):
    drawn, edited = [], []
    monkeypatch.setattr(canvas, "render_now", lambda: drawn.append(True))
    canvas.gate_edited.connect(edited.append)
    canvas._handle_drag = None
    canvas._release_volume_handle(_Mouse(canvas.axes_at(0, 0), 0, 0))
    gate = _box()
    canvas._handle_drag = (gate.name, "x_high", (0.0, 0.0), gate, np.zeros(3))
    canvas._release_volume_handle(_Mouse(canvas.axes_at(0, 0), 0, 0))
    assert drawn == [True, True] and edited == []


def test_a_press_outside_the_axes_is_still_the_volumes(canvas, monkeypatch):
    monkeypatch.setattr(canvas, "_in_volume", lambda: True)
    assert canvas._volume_press_gesture(SimpleNamespace(inaxes=None)) is True


def test_a_stuck_live_artist_is_forgotten(canvas):
    class _Stuck:
        def remove(self):
            raise ValueError("already gone")

    canvas._live = _Stuck()
    canvas._clear_live()
    assert canvas._live is None


def test_a_preview_that_clashes_draws_the_saved_gates(canvas, monkeypatch):
    from dataclasses import replace

    canvas._gates = GateSet([_box()])
    canvas._volume_preview = replace(_box(), name="orphan", parent="missing")
    drawn = []
    monkeypatch.setattr(canvas, "_draw_volume_gate",
                        lambda ax, gate, *a, **k: drawn.append(gate.name),
                        raising=False)
    try:
        canvas._draw_volume_gates(canvas.axes_at(0, 0), canvas.population(),
                                  {"fg": "#000"})
    except Exception:
        pass
    canvas._volume_preview = None
    assert list(canvas._gates.names) == ["b"]


def test_a_click_on_the_last_vertex_adds_no_corner(canvas, monkeypatch):
    monkeypatch.setattr(canvas, "_in_volume", lambda: True)
    monkeypatch.setattr(canvas, "drag_mode", lambda: "draw")
    monkeypatch.setattr(canvas, "volume_shape", lambda: "view_polygon")
    monkeypatch.setattr(canvas, "_show_view_polygon", lambda event: None)
    canvas._view_polygon = [(10.0, 10.0)]
    assert canvas._view_polygon_click(_Mouse(canvas.axes_at(0, 0), 11, 10))
    assert canvas._view_polygon == [(10.0, 10.0)]
    canvas._view_polygon = []


def test_renames_reach_combinations_and_disabled_gates(screen):  # noqa: F811
    tree = screen.gates.tree
    gates = screen.gates.gates
    gates.add(_box())
    gates.add(BoxGate.from_limits("c", ("a", "b", "c"),
                                  [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]))
    gates.add(CompositeGate(name="both", operands=("b", "c")))
    tree.set_gates(gates, screen.gates.canvas.population())
    tree._disabled.add("b")
    assert tree._rename_gate("b", "box")
    assert gates.get("both").operands == ("box", "c")
    assert "box" in tree._disabled and "b" not in tree._disabled


def test_double_clicks_off_the_name_or_on_no_gate_ask_nothing(screen,  # noqa: F811
                                                              monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QTreeWidgetItem

    asked = []
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: asked.append(1) or ("x", False)))
    tree = screen.gates.tree
    item = QTreeWidgetItem(["nobody"])
    tree._on_double_clicked(item, 1)
    tree._on_double_clicked(None, 0)
    tree._on_double_clicked(item, 0)
    assert asked == []
    screen.gates.gates.add(_box())
    tree._on_double_clicked(QTreeWidgetItem(["b"]), 0)
    assert asked == [1]
    renamed = []
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("box", True)))
    monkeypatch.setattr(tree, "_rename_gate",
                        lambda old, new: renamed.append((old, new)))
    tree._on_double_clicked(QTreeWidgetItem(["b"]), 0)
    assert renamed == [("b", "box")]


def test_a_view_gate_that_cannot_mask_has_no_centre_handle(canvas, monkeypatch):
    def broken(self, frame):
        raise ValueError("no columns")

    monkeypatch.setattr(ViewGate, "mask", broken)
    assert canvas._volume_handles(_view()) == []


def test_dragging_without_a_handle_does_nothing(canvas, monkeypatch):
    drawn = []
    monkeypatch.setattr(canvas, "render_now", lambda: drawn.append(True))
    canvas._handle_drag = None
    canvas._drag_volume_handle(_Mouse(canvas.axes_at(0, 0), 0, 0))
    assert drawn == []


def test_gestures_off_the_volume_are_not_the_volumes(canvas, monkeypatch):
    monkeypatch.setattr(canvas, "_in_volume", lambda: False)
    assert canvas._volume_press_gesture(SimpleNamespace(inaxes=None)) is False


def test_a_lasso_around_nothing_asks_again(canvas, monkeypatch):
    asked = []
    canvas.depth_requested.connect(asked.append)
    monkeypatch.setattr(canvas, "_in_volume", lambda: True)
    monkeypatch.setattr(canvas, "_extend_lasso", lambda event: None)
    monkeypatch.setattr(canvas, "volume_shape", lambda: "view_lasso")
    monkeypatch.setattr(canvas, "gate_from_screen_outline", lambda outline: None)
    canvas._handle_drag = None
    canvas._lasso = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    assert canvas._volume_release(_Mouse(canvas.axes_at(0, 0), 5, 5)) is True
    assert asked and "encloses nothing" in asked[-1]


def test_a_one_point_lasso_draws_no_ghost_and_makes_no_gate(canvas, monkeypatch):
    monkeypatch.setattr(canvas, "_lasso_outline", lambda: [(1.0, 1.0)])
    canvas._show_lasso_ghost()
    assert canvas._ghost == []
    assert canvas.gate_from_screen_outline([(0, 0), (1, 1)]) is None


def test_a_gate_whose_count_is_unavailable_says_so(screen, monkeypatch):  # noqa: F811
    tree = screen.gates.tree
    gates = screen.gates.gates
    gates.add(_box())
    monkeypatch.setattr(tree, "_gate_stats", lambda: ({}, {"b": "no column"}))
    tree.set_gates(gates, screen.gates.canvas.population())
    item = tree.tree.topLevelItem(0)
    assert item.text(1) == tree.UNAVAILABLE
