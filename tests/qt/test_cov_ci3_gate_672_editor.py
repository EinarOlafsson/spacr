"""Edit-mode gestures commit only real, valid boundaries (item 672).

Each test drives the gate editor's canvas the way the existing 672 tests do
and checks what the user would see: the stored gate, the undo history, the
messages, and the anchors and surfaces offered.
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
    BoxGate, CylinderGate, EditableVolumeGate, GateSet, PolygonGate,
    PrismGate, RectGate, ViewGate,
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
    return screen.gates.canvas


def _event(canvas, point, **values):
    event = dict(inaxes=canvas.axes_at(0, 0), x=float(point[0]), y=float(point[1]),
                 xdata=0., ydata=0., button=1, step=0, key=None, dblclick=False)
    event.update(values)
    return SimpleNamespace(**event)


def _surface_pixel(canvas, gate, group):
    mesh = canvas._editable_boundary(gate)
    indices = {i for face, value in zip(mesh.faces, mesh.surface_groups) if value == group for i in face}
    point = np.asarray(mesh.vertices)[sorted(indices)].mean(axis=0)
    return _screen_points(canvas.axes_at(0, 0), [point])[0]


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
    kept = canvas._volume_xyz()[gate.mask(screen._frame)]
    assert handles[0][1] == pytest.approx(kept.mean(axis=0))
    centre = _screen_points(canvas.axes_at(0, 0), [[0, 0, 0]])[0]
    assert canvas.volume_surface_at(_event(canvas, centre)) is None
    canvas._surface_gate = 'view'
    canvas._selected_surfaces = {0}
    assert not canvas._edit_surface_wheel(_event(canvas, centre, step=1, button='up'))
    canvas._surface_gate = 'another'
    assert not canvas._edit_surface_wheel(_event(canvas, centre, step=1, button='up'))
    assert screen.gates.gates.get('view') is gate


def test_edit_anchors_are_not_offered_for_a_gate_on_other_measurements(screen):
    canvas = _install(screen, _box())
    elsewhere = ViewGate(name='elsewhere', x_column='a', y_column='b', z_column='d',
                         projection=tuple(map(tuple, np.eye(4))),
                         vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
    assert canvas._editable_boundary(elsewhere) is None
    assert canvas._volume_handles(elsewhere) == []
    assert len(canvas._volume_handles(_box())) == 26


def test_a_surface_too_small_to_see_cannot_be_picked(screen):
    tiny = BoxGate.from_limits('box', ('a', 'b', 'c'), [(0, 1e-7)] * 3)
    canvas = _install(screen, tiny)
    pixel = _screen_points(canvas.axes_at(0, 0), [[5e-8] * 3])[0]
    assert canvas.volume_surface_at(_event(canvas, pixel)) is None
    canvas = _install(screen, _box())
    pixel = _screen_points(canvas.axes_at(0, 0), [[0, 0, 0]])[0]
    assert canvas.volume_surface_at(_event(canvas, pixel)) is not None


def test_an_unprojectable_point_gives_no_edit_displacement(screen):
    canvas = _install(screen, _box())
    canvas._edit_drag_projection = None
    ax = canvas.axes_at(0, 0)
    assert canvas._volume_edit_delta(ax, np.full(3, np.nan), np.asarray([10., 0.])) is None
    assert canvas._surface_moved(_event(canvas, [0, 0])) is None


def test_a_plain_click_selects_one_surface_and_commits_nothing(screen):
    canvas = _install(screen, _box())
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    canvas._surface_gate = 'box'
    canvas._selected_surfaces = {0, 2}
    pixel = _surface_pixel(canvas, _box(), 1)
    found = canvas.volume_surface_at(_event(canvas, pixel))
    assert found is not None and found[1] not in {0, 2}
    history = screen.gates.undo_stack.index()
    canvas._on_press(_event(canvas, pixel))
    assert canvas._selected_surfaces == {found[1]}
    assert canvas._surface_drag is not None
    canvas._on_release(_event(canvas, pixel))
    assert canvas._surface_drag is None and canvas._volume_preview is None
    assert screen.gates.gates.get('box') == _box()
    assert screen.gates.undo_stack.index() == history


def test_a_surface_drag_whose_edit_axis_changed_mid_gesture_is_dropped(screen):
    canvas = _install(screen, _box())
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    canvas._edit_axis = 'z'
    pixel = _surface_pixel(canvas, _box(), 1)
    history = screen.gates.undo_stack.index()
    canvas._on_press(_event(canvas, pixel))
    assert canvas._surface_drag is not None
    canvas._edit_axis = 'x'
    canvas._on_release(_event(canvas, pixel + [0, 40]))
    assert screen.gates.gates.get('box') == _box()
    assert screen.gates.undo_stack.index() == history
    assert canvas._surface_drag is None and canvas._edit_drag_projection is None


def test_a_surface_drag_that_folds_the_volume_is_refused_with_a_reason(screen):
    gate = _cylinder()
    canvas = _install(screen, gate)
    screen.apply_settings(replace(screen._settings, show_anchors=False))
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


def test_a_non_finite_pointer_position_drags_nothing(screen):
    screen.gates.mode_requested.emit('2D')
    gate = RectGate(name='flat', x_column='a', y_column='b', x_low=-1, x_high=1, y_low=-1, y_high=1)
    screen.gates.set_gates(GateSet([gate]))
    canvas = screen.gates.canvas
    canvas._resize = ('flat', 'x_high')
    assert canvas._dragged_to(_event(canvas, [0, 0], xdata=float('nan'), ydata=0.)) is None
    assert canvas._dragged_to(_event(canvas, [0, 0], xdata=2., ydata=0.)).x_high == 2.


def test_a_stale_outline_anchor_drags_nothing_and_reports_nothing(screen):
    screen.gates.mode_requested.emit('2D')
    gate = RectGate(name='flat', x_column='a', y_column='b', x_low=-1, x_high=1, y_low=-1, y_high=1)
    screen.gates.set_gates(GateSet([gate]))
    canvas = screen.gates.canvas
    failures = []
    canvas.wand_failed.connect(failures.append)
    canvas._resize = ('flat', 'outline:40:0.5')
    assert canvas._dragged_to(_event(canvas, [0, 0], xdata=.3, ydata=.2)) is None
    assert failures == []
    assert screen.gates.gates.get('flat') is gate


def test_a_box_outline_anchor_edit_becomes_a_prism_keeping_its_depth(screen):
    screen.gates.mode_requested.emit('2D')
    gate = BoxGate.from_limits('box', ('a', 'b', 'c'), [(-1, 1), (-1, 1), (-.5, .75)])
    screen.gates.set_gates(GateSet([gate]))
    screen.gates.tree.select('box')
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)
    handle = next(h for h in canvas._handles_for(ax, gate)
                  if h.role.startswith('outline:') and h.x == pytest.approx(1) and 0 < h.y < .9)
    pixel = ax.transData.transform([handle.x, handle.y])
    messages = []
    canvas.depth_requested.connect(messages.append)
    canvas._on_press(_event(canvas, pixel, xdata=handle.x, ydata=handle.y))
    assert canvas._resize == ('box', handle.role)
    end = _event(canvas, pixel + [20, 0], xdata=handle.x + .5, ydata=handle.y)
    canvas._on_release(end)
    edited = screen.gates.gates.get('box')
    assert isinstance(edited, PrismGate)
    assert (edited.u_column, edited.v_column, edited.axis_column) == ('a', 'b', 'c')
    assert (edited.axis_low, edited.axis_high) == (-.5, .75)
    assert (handle.x + .5, handle.y) in edited.vertices
    assert any('piecewise-linear' in message for message in messages)
    rows = pd.DataFrame({'a': [1.2, 1.2, 0], 'b': [handle.y, handle.y, 0], 'c': [0, 1, 0]})
    assert gate.mask(rows).tolist() == [False, False, True]
    assert edited.mask(rows).tolist() == [True, False, True]
    screen.gates.undo_stack.undo()
    assert screen.gates.gates.get('box') == gate


def test_a_footprint_outline_edit_of_a_cylinder_keeps_its_axis_range(screen):
    screen.gates.mode_requested.emit('2D')
    gate = CylinderGate(name='cyl', u_column='a', v_column='b', axis_column='c',
                        u_radius=1, v_radius=1, axis_low=-.25, axis_high=.5)
    screen.gates.set_gates(GateSet([gate]))
    canvas = screen.gates.canvas
    footprint = PolygonGate(name='cyl', x_column='a', y_column='b',
                            vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
    canvas._as_flat = lambda shown: footprint if shown is gate else shown
    canvas._resize = ('cyl', 'outline:1:0.5')
    edited = canvas._dragged_to(_event(canvas, [0, 0], xdata=1.5, ydata=0.))
    assert isinstance(edited, PrismGate)
    assert edited.vertices == ((-1, -1), (1, -1), (1.5, 0), (1, 1), (-1, 1))
    assert (edited.axis_column, edited.axis_low, edited.axis_high) == ('c', -.25, .5)
