"""Dragging a volume anchor that no longer exists says so and moves nothing.

A handle is captured at press time. If the boundary it belonged to is
replaced before the pointer moves (an undo, a reload of the gate set), its
anchor role can name a vertex the current mesh does not have. The drag
must then be refused with a message, not raise out of a motion event or
move some other vertex.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_editor import _screen_points
from spacr.qt.widgets.gate_spec import BoxGate, GateSet

pytestmark = pytest.mark.qt


@pytest.fixture
def canvas(qtbot):
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
    gate = BoxGate.from_limits('box', ('a', 'b', 'c'), [(-1, 1)] * 3)
    widget.gates.set_gates(GateSet([gate]))
    widget.gates.tree.select(gate.name)
    widget.gates._drag_buttons['edit'].click()
    for _ in range(3):
        QApplication.processEvents()
    widget.gates.canvas.render_now()
    return widget.gates.canvas, gate


def test_a_stale_anchor_role_is_refused_with_a_message(canvas):
    canvas, gate = canvas
    ax = canvas.axes_at(0, 0)
    ax.view_init(elev=31, azim=-47)
    canvas._view_angles = 31., -47., 0.
    canvas._canvas.draw()
    canvas._edit_axis = 'x'
    point = np.array([1., 1., 1.])
    start = _screen_points(ax, [point])[0]
    direction = canvas._pixels_per_unit(ax, point)[:, 0]
    end = start + direction * .12
    said = []
    canvas.depth_requested.connect(said.append)
    canvas._handle_drag = (gate.name, 'anchor:99:1', tuple(start), gate, point)

    moved = canvas._handle_moved(SimpleNamespace(x=float(end[0]), y=float(end[1])))

    assert moved is None
    assert said == ['The selected volume anchor does not exist.']
