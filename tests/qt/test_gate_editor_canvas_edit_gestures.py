"""Edit-mode gestures on the Gate Editor canvas commit only real, valid boundaries (item 672).

Driven through the screen the way a user drives it: a plain click selects one
surface and commits nothing; a surface drag whose edit axis changed mid
gesture, or that would fold the volume, leaves the stored gate and the undo
history untouched (the fold with a reason); a stale outline anchor drags
nothing; and dragging a box's outline anchor turns it into a prism that keeps
its depth, undoably.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_editor import _screen_points
from spacr.qt.widgets.gate_spec import (
    BoxGate, CylinderGate, GateSet, PrismGate, RectGate,
)

pytestmark = pytest.mark.qt


@pytest.fixture
def make(qtbot):
    """Build a fresh 3-D screen over 200 uniform points in a, b, c."""
    return lambda: _screen(qtbot)


def _screen(qtbot):
    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1100, 800)
    values = np.random.default_rng(41).uniform(-1.5, 1.5, (200, 3))
    widget.set_frame(pd.DataFrame(values, columns=['a', 'b', 'c']))
    widget._x.setCurrentText('a')
    widget._y.setCurrentText('b')
    widget.gates.mode_requested.emit('3D')
    widget._z.setCurrentText('c')
    widget.show()
    QApplication.processEvents()
    return widget


def _box():
    return BoxGate.from_limits('box', ('a', 'b', 'c'), [(-1, 1)] * 3)


def _cylinder():
    return CylinderGate(name='cyl', u_column='a', v_column='b', axis_column='c',
                        u_radius=1, v_radius=1, axis_low=-1, axis_high=1)


def _install(screen, gate):
    screen.gates.set_gates(GateSet([gate]))
    screen.gates.tree.select(gate.name)
    screen.gates._drag_buttons['edit'].click()
    for _ in range(3):
        QApplication.processEvents()
    screen.gates.canvas.render_now()
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    return screen.gates.canvas


def _event(canvas, point, **values):
    event = dict(inaxes=canvas.axes_at(0, 0), x=float(point[0]),
                 y=float(point[1]), xdata=0., ydata=0., button=1, step=0,
                 key=None, dblclick=False)
    event.update(values)
    return SimpleNamespace(**event)


def _surface_pixel(canvas, gate, group):
    mesh = canvas._editable_boundary(gate)
    indices = {i for face, value in zip(mesh.faces, mesh.surface_groups)
               if value == group for i in face}
    point = np.asarray(mesh.vertices)[sorted(indices)].mean(axis=0)
    return _screen_points(canvas.axes_at(0, 0), [point])[0]


def test_surface_gestures_commit_nothing_unless_the_drag_is_valid(make, subtests):
    """A click selects one surface; a drag whose axis changed, or that folds the volume, is dropped without an undo step."""
    with subtests.test(msg="a plain click selects one surface"):
        screen = make()
        canvas = _install(screen, _box())
        canvas._surface_gate = 'box'
        canvas._selected_surfaces = {0, 2}
        pixel = _surface_pixel(canvas, _box(), 1)
        found = canvas.volume_surface_at(_event(canvas, pixel))
        assert found[0] == 'box' and found[1] not in {0, 2}
        history = screen.gates.undo_stack.index()
        canvas._on_press(_event(canvas, pixel))
        assert canvas._selected_surfaces == {found[1]}
        assert canvas._surface_drag is not None
        canvas._on_release(_event(canvas, pixel))
        assert canvas._surface_drag is None
        assert canvas._volume_preview is None
        assert screen.gates.gates.get('box') == _box()
        assert screen.gates.undo_stack.index() == history

    with subtests.test(msg="the edit axis changed mid gesture"):
        screen = make()
        canvas = _install(screen, _box())
        canvas._edit_axis = 'z'
        pixel = _surface_pixel(canvas, _box(), 1)
        history = screen.gates.undo_stack.index()
        canvas._on_press(_event(canvas, pixel))
        assert canvas._surface_drag is not None
        canvas._edit_axis = 'x'
        canvas._on_release(_event(canvas, pixel + [0, 40]))
        assert screen.gates.gates.get('box') == _box()
        assert screen.gates.undo_stack.index() == history
        assert canvas._surface_drag is None
        assert canvas._edit_drag_projection is None

    with subtests.test(msg="a drag that folds the volume is refused"):
        screen = make()
        gate = _cylinder()
        canvas = _install(screen, gate)
        canvas._edit_axis = 'z'
        mesh = canvas._editable_boundary(gate)
        top = max(mesh.surface_groups)
        pixel = _surface_pixel(canvas, gate, top)
        assert canvas.volume_surface_at(_event(canvas, pixel)) == ('cyl', top)
        messages = []
        canvas.depth_requested.connect(messages.append)
        history = screen.gates.undo_stack.index()
        canvas._on_press(_event(canvas, pixel))
        assert canvas._surface_drag[2] == (top,)
        point = canvas._surface_drag[4]
        direction = canvas._pixels_per_unit(canvas.axes_at(0, 0), point)[:, 2]
        end = pixel - direction * 1.2
        canvas._on_motion(_event(canvas, end))
        assert canvas._volume_preview is None
        assert any('intersect' in message for message in messages)
        canvas._on_release(_event(canvas, end))
        assert screen.gates.gates.get('cyl') == gate
        assert screen.gates.undo_stack.index() == history


def test_outline_anchors_drag_only_live_gates_and_turn_a_box_into_a_prism(make, subtests):
    """A stale anchor drags nothing silently; a box anchor edit commits a depth-keeping prism that undo restores."""
    with subtests.test(msg="a stale outline anchor"):
        screen = make()
        screen.gates.mode_requested.emit('2D')
        gate = RectGate(name='flat', x_column='a', y_column='b',
                        x_low=-1, x_high=1, y_low=-1, y_high=1)
        screen.gates.set_gates(GateSet([gate]))
        canvas = screen.gates.canvas
        failures = []
        canvas.wand_failed.connect(failures.append)
        canvas._resize = ('flat', 'outline:40:0.5')
        assert canvas._dragged_to(
            _event(canvas, [0, 0], xdata=.3, ydata=.2)) is None
        assert failures == []
        assert screen.gates.gates.get('flat') is gate

    with subtests.test(msg="a box outline anchor becomes a prism"):
        screen = make()
        screen.gates.mode_requested.emit('2D')
        gate = BoxGate.from_limits('box', ('a', 'b', 'c'),
                                   [(-1, 1), (-1, 1), (-.5, .75)])
        screen.gates.set_gates(GateSet([gate]))
        screen.gates.tree.select('box')
        canvas = screen.gates.canvas
        ax = canvas.axes_at(0, 0)
        handle = next(h for h in canvas._handles_for(ax, gate)
                      if h.role.startswith('outline:')
                      and h.x == pytest.approx(1) and 0 < h.y < .9)
        pixel = ax.transData.transform([handle.x, handle.y])
        messages = []
        canvas.depth_requested.connect(messages.append)
        canvas._on_press(_event(canvas, pixel, xdata=handle.x, ydata=handle.y))
        assert canvas._resize == ('box', handle.role)
        canvas._on_release(_event(canvas, pixel + [20, 0],
                                  xdata=handle.x + .5, ydata=handle.y))
        edited = screen.gates.gates.get('box')
        assert isinstance(edited, PrismGate)
        assert (edited.u_column, edited.v_column, edited.axis_column) == (
            'a', 'b', 'c')
        assert (edited.axis_low, edited.axis_high) == (-.5, .75)
        assert (handle.x + .5, handle.y) in edited.vertices
        assert messages == [
            "Editing an outline anchor commits a piecewise-linear boundary; "
            "counts and exports use that same boundary."]
        rows = pd.DataFrame({'a': [1.2, 1.2, 0], 'b': [handle.y, handle.y, 0],
                             'c': [0, 1, 0]})
        assert gate.mask(rows).tolist() == [False, False, True]
        assert edited.mask(rows).tolist() == [True, False, True]
        screen.gates.undo_stack.undo()
        assert screen.gates.gates.get('box') == gate
