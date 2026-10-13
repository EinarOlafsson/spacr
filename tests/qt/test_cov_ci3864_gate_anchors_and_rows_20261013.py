"""Grabbing a gate by its anchor, a selected surface, and a row with no count.

* A press on a rectangle's corner anchor grabs that anchor for a resize; it
  does not start moving the whole gate, which is what a press inside the
  gate but away from every anchor does.
* A plain click on a volume surface that is already selected keeps that
  selection and starts dragging it, rather than narrowing it to the same
  surface again.
* A gate the counter answered neither a number nor a reason for shows an
  empty row, not ``n/a``, which would claim the table cannot answer it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication                           # noqa: E402

from spacr.qt.linked_selection import LinkedSelection                # noqa: E402
from spacr.qt.screens.gate_editor import GateEditorScreen            # noqa: E402
from spacr.qt.widgets.gate_editor import (                           # noqa: E402
    GateCanvas, GateTree, _screen_points)
from spacr.qt.widgets.gate_spec import BoxGate, GateSet, RectGate    # noqa: E402
from spacr.qt.widgets.graph_builder import GraphSpec                 # noqa: E402

pytestmark = pytest.mark.qt


class _Mouse:
    """A press carrying both data and pixel coordinates."""

    def __init__(self, ax, x_data, y_data):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = 1
        self.step = 0
        self.key = None
        self.dblclick = False


@pytest.fixture
def rect_canvas(qtbot):
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(pd.DataFrame({"a": [0.0, 5.0, 10.0],
                                   "b": [0.0, 5.0, 10.0]}))
    canvas.set_spec(GraphSpec(x="a", y="b"))
    canvas.set_gates(GateSet().add(RectGate(
        name="g", x_column="a", y_column="b",
        x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0)))
    return canvas, canvas.axes_at(0, 0)


def _just_outside(ax, x_data, y_data, pixels):
    """A press ``pixels`` up and right of a data point, in both units."""
    event = _Mouse(ax, x_data, y_data)
    event.x, event.y = event.x + pixels, event.y + pixels
    event.xdata, event.ydata = ax.transData.inverted().transform(
        (event.x, event.y))
    return event


def test_a_press_on_a_corner_anchor_resizes_instead_of_moving(rect_canvas):
    """Just outside the corner only the anchor is under the pointer, so no
    overlap list opens and the press goes straight to the anchor."""
    canvas, ax = rect_canvas
    corner = _just_outside(ax, 8.0, 8.0, 4.0)
    assert canvas.gate_at(float(corner.xdata), float(corner.ydata)) is None
    grabbed = canvas.handle_at(corner)
    assert grabbed is not None and grabbed[0] == "g"

    canvas._on_press(corner)
    assert canvas._resize == grabbed
    assert canvas._move_name is None


def test_a_press_away_from_every_anchor_grabs_none(rect_canvas):
    canvas, ax = rect_canvas
    middle = _Mouse(ax, 5.0, 5.0)
    assert canvas.handle_at(middle) is None
    canvas._on_press(middle)
    assert canvas._resize is None
    assert canvas._move_name == "g"


@pytest.fixture
def volume_screen(qtbot):
    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1100, 800)
    values = np.random.default_rng(41).uniform(-1.5, 1.5, (200, 3))
    widget.set_frame(pd.DataFrame(values, columns=["a", "b", "c"]))
    widget._x.setCurrentText("a")
    widget._y.setCurrentText("b")
    widget.gates.mode_requested.emit("3D")
    widget._z.setCurrentText("c")
    widget.show()
    QApplication.processEvents()
    return widget


def _box():
    return BoxGate.from_limits("box", ("a", "b", "c"), [(-1, 1)] * 3)


def _event(canvas, point):
    return SimpleNamespace(inaxes=canvas.axes_at(0, 0), x=float(point[0]),
                           y=float(point[1]), xdata=0.0, ydata=0.0, button=1,
                           step=0, key=None, dblclick=False)


def test_a_click_on_a_selected_surface_keeps_the_selection_and_drags_it(
        volume_screen):
    from dataclasses import replace

    screen = volume_screen
    screen.gates.set_gates(GateSet([_box()]))
    screen.gates.tree.select("box")
    screen.gates._drag_buttons["edit"].click()
    for _ in range(3):
        QApplication.processEvents()
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    canvas = screen.gates.canvas
    canvas.render_now()

    mesh = canvas._editable_boundary(_box())
    group = 1
    indices = {i for face, value in zip(mesh.faces, mesh.surface_groups)
               if value == group for i in face}
    point = np.asarray(mesh.vertices)[sorted(indices)].mean(axis=0)
    pixel = _screen_points(canvas.axes_at(0, 0), [point])[0]
    found = canvas.volume_surface_at(_event(canvas, pixel))
    assert found is not None
    canvas._surface_gate = "box"
    canvas._selected_surfaces = {found[1], 0 if found[1] else 2}
    selected = set(canvas._selected_surfaces)
    history = screen.gates.undo_stack.index()

    assert canvas._grab_volume_surface(_event(canvas, pixel)) is True
    assert canvas._selected_surfaces == selected, (
        "a plain click on a selected surface narrowed the selection")
    assert canvas._surface_drag is not None
    canvas._on_release(_event(canvas, pixel))
    assert screen.gates.gates.get("box") == _box()
    assert screen.gates.undo_stack.index() == history


def test_a_gate_with_neither_a_count_nor_a_reason_shows_an_empty_row(
        qtbot, monkeypatch):
    tree = GateTree()
    qtbot.addWidget(tree)
    monkeypatch.setattr(tree, "_gate_stats", lambda: ({}, {}))
    gates = GateSet().add(RectGate(name="g", x_column="a", y_column="b",
                                   x_low=0.0, x_high=1.0,
                                   y_low=0.0, y_high=1.0))
    tree.set_gates(gates, pd.DataFrame({"a": [0.5], "b": [0.5]}))

    item = tree.tree.topLevelItem(0)
    assert item.text(0) == "g"
    assert [item.text(column) for column in (1, 2, 3)] == ["", "", ""]
    assert GateTree.UNAVAILABLE not in [item.text(1)]
