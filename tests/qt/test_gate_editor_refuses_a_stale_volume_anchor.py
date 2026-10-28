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


@pytest.mark.parametrize('unavailable', ['no-handle', 'different-role', 'no-artist', 'no-axes'])
def test_an_unavailable_anchor_preview_leaves_gate_membership_and_artists_alone(
        canvas, monkeypatch, unavailable):
    canvas, gate = canvas
    mesh = canvas._editable_boundary(gate)
    vertices = np.asarray(mesh.vertices).copy()
    before = gate.mask(canvas.population()).copy()
    canvas._handle_drag = (gate.name, 'anchor:0:1', (0., 0.), gate, np.zeros(3))
    if unavailable == 'no-handle':
        canvas._handle_drag = None
    elif unavailable == 'different-role':
        canvas._handle_drag = (gate.name, 'centre', (0., 0.), gate, np.zeros(3))
    elif unavailable == 'no-artist':
        canvas._mesh_artists.pop(mesh.name)
    else:
        monkeypatch.setattr(canvas, 'axes_at', lambda *args: None)
    assert canvas._preview_anchor_drag(SimpleNamespace(x=10., y=10.)) is False
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(gate.mask(canvas.population()), before)
    assert canvas._volume_preview is None


@pytest.mark.parametrize('gesture', ['anchor', 'surface'])
def test_a_changed_projection_consumes_motion_without_moving_the_saved_volume(canvas, gesture):
    canvas, gate = canvas
    mesh = canvas._editable_boundary(gate)
    original = np.asarray(mesh.vertices).copy()
    canvas._edit_drag_projection = (np.eye(2, 3),
                                    (canvas._canvas.width() + 1, canvas._canvas.height()),
                                    canvas._edit_axis)
    if gesture == 'anchor':
        canvas._handle_drag = (gate.name, 'anchor:0:1', (0., 0.), gate, np.zeros(3))
        handled = canvas._preview_anchor_drag(SimpleNamespace(x=10., y=10.))
    else:
        canvas._surface_drag = gate, mesh, {mesh.surface_groups[0]}, (0., 0.), np.zeros(3)
        handled = canvas._preview_surface_drag(SimpleNamespace(x=10., y=10.))
    assert handled is True
    np.testing.assert_array_equal(mesh.vertices, original)
    assert canvas._volume_preview is None


@pytest.mark.parametrize('gesture, message', [
    ('anchor', 'The selected volume anchor does not exist.'),
    ('surface', 'A selected volume surface does not exist.'),
])
def test_stale_preview_targets_report_once_and_preserve_the_original_volume(canvas, gesture, message):
    canvas, gate = canvas
    ax = canvas.axes_at(0, 0)
    point = np.array([1., 1., 1.])
    start = _screen_points(ax, [point])[0]
    canvas._edit_axis = 'x'
    delta = canvas._pixels_per_unit(ax, point)[:, 0] * .12
    event = SimpleNamespace(x=float(start[0] + delta[0]), y=float(start[1] + delta[1]))
    mesh = canvas._editable_boundary(gate)
    original = np.asarray(mesh.vertices).copy()
    said = []
    canvas.depth_requested.connect(said.append)
    if gesture == 'anchor':
        canvas._handle_drag = (gate.name, 'anchor:99999:1', tuple(start), gate, point)
        handled = canvas._preview_anchor_drag(event)
    else:
        canvas._surface_drag = gate, mesh, {max(mesh.surface_groups) + 1}, tuple(start), point
        handled = canvas._preview_surface_drag(event)
    assert handled is True
    assert said == [message]
    np.testing.assert_array_equal(mesh.vertices, original)
    assert canvas._volume_preview is None


@pytest.mark.parametrize('missing', ['axes', 'artist'])
def test_a_surface_without_a_drawn_boundary_requests_the_regular_redraw(canvas, monkeypatch, missing):
    canvas, gate = canvas
    mesh = canvas._editable_boundary(gate)
    canvas._surface_drag = gate, mesh, {mesh.surface_groups[0]}, (0., 0.), np.zeros(3)
    if missing == 'axes':
        monkeypatch.setattr(canvas, 'axes_at', lambda *args: None)
        assert canvas._preview_surface_drag(SimpleNamespace(x=1., y=1.)) is False
        assert canvas._show_volume_drag_preview(mesh, mesh.vertices, mesh.faces,
                                                mesh.surface_groups) is False
    else:
        canvas._mesh_artists.pop(mesh.name)
        assert canvas._preview_surface_drag(SimpleNamespace(x=1., y=1.)) is False
        canvas._edit_axis = 'x'
        canvas._volume_motion(SimpleNamespace(x=0., y=0.))
        assert canvas._volume_preview == gate
