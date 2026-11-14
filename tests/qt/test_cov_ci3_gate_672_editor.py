"""Edit-mode gestures commit only real, valid boundaries (item 672).

Each test drives the gate editor's canvas the way the existing 672 tests do
and checks what the user would see: the stored gate, the undo history, the
messages, and the anchors and surfaces offered.
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
from spacr.qt.widgets.gate_spec import (
    BoxGate, GateSet, ViewGate,
)

pytestmark = pytest.mark.qt

BOWTIE = ((-1., -1.), (1., 1.), (1., -1.), (-1., .5))


@pytest.fixture
def screen(qtbot):
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


def _install(screen, gate):
    screen.gates.set_gates(GateSet([gate]))
    screen.gates.tree.select(gate.name)
    screen.gates._drag_buttons['edit'].click()
    for _ in range(3):
        QApplication.processEvents()
    screen.gates.canvas.render_now()
    return screen.gates.canvas


def _event(canvas, point, **values):
    event = dict(inaxes=canvas.axes_at(0, 0), x=float(point[0]), y=float(point[1]),
                 xdata=0., ydata=0., button=1, step=0, key=None, dblclick=False)
    event.update(values)
    return SimpleNamespace(**event)


def _crossed_view():
    return ViewGate(name='view', x_column='a', y_column='b', z_column='c',
                    projection=tuple(map(tuple, np.eye(4))), vertices=BOWTIE)


def test_a_self_crossing_view_lasso_offers_no_editable_surfaces(screen):
    gate = _crossed_view()
    canvas = _install(screen, gate)
    assert canvas.drag_mode() == 'edit'
    assert canvas._editable_boundary(gate) is None
    handles = canvas._volume_handles(gate)
    assert [role for role, _point in handles] == ['centre']
    kept = canvas._volume_xyz()[gate.mask(canvas.population())]
    assert handles[0][1] == pytest.approx(kept.mean(axis=0))
    centre = _screen_points(canvas.axes_at(0, 0), [[0, 0, 0]])[0]
    assert canvas.volume_surface_at(_event(canvas, centre)) is None
    canvas._surface_gate = 'view'
    canvas._selected_surfaces = {0}
    assert not canvas._edit_surface_wheel(_event(canvas, centre, step=1, button='up'))
    canvas._surface_gate = 'another'
    assert not canvas._edit_surface_wheel(_event(canvas, centre, step=1, button='up'))
    assert screen.gates.gates.get('view') is gate


def test_a_surface_too_small_to_see_cannot_be_picked(screen):
    tiny = BoxGate.from_limits('box', ('a', 'b', 'c'), [(0, 1e-7)] * 3)
    canvas = _install(screen, tiny)
    pixel = _screen_points(canvas.axes_at(0, 0), [[5e-8] * 3])[0]
    assert canvas.volume_surface_at(_event(canvas, pixel)) is None
    canvas = _install(screen, _box())
    pixel = _screen_points(canvas.axes_at(0, 0), [[0, 0, 0]])[0]
    assert canvas.volume_surface_at(_event(canvas, pixel)) is not None
