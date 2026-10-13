"""Real rotated gate edits share their rendered boundary with scientific counts."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_editor import _screen_points
from spacr.qt.widgets.gate_settings import GateEditorSettings
from spacr.qt.widgets.gate_spec import BoxGate, CylinderGate, EditableVolumeGate, EllipseGate, GateSet, PolygonGate, RectGate


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


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_real_rotated_anchor_drag_changes_only_the_selected_measurement(screen, axis):
    canvas = _install(screen, _box())
    ax = canvas.axes_at(0, 0)
    ax.view_init(elev=31, azim=-47)
    canvas._view_angles = 31., -47., 0.
    canvas._canvas.draw()
    canvas._edit_axis = axis
    role, point = next((role, point) for role, point in canvas._volume_handles(_box())
                       if np.array_equal(point, [1., 1., 1.]))
    start = _screen_points(ax, [point])[0]
    direction = canvas._pixels_per_unit(ax, point)[:, ('x', 'y', 'z').index(axis)]
    end = start + direction * .12
    canvas._on_press(_event(canvas, start))
    assert canvas._handle_drag[1] == role
    canvas._on_motion(_event(canvas, end))
    assert screen.gates.gates.get('box') == _box()
    canvas._on_release(_event(canvas, end))
    edited = screen.gates.gates.get('box')
    assert isinstance(edited, EditableVolumeGate)
    index = int(role.split(':')[1])
    expected = np.asarray(EditableVolumeGate.from_gate(_box()).vertices)
    expected[index, ('x', 'y', 'z').index(axis)] += .12
    assert np.asarray(edited.vertices) == pytest.approx(expected)
    assert screen.gates.undo_stack.canUndo()
    screen.gates.undo_stack.undo()
    assert screen.gates.gates.get('box') == _box()
    screen.gates.undo_stack.redo()
    assert screen.gates.gates.get('box') == edited


def test_preview_density_visibility_and_rotation_preserve_exact_analytic_membership(screen):
    gate = CylinderGate(name='cylinder', u_column='a', v_column='b', axis_column='c',
                        u_radius=1, v_radius=1, axis_low=-1, axis_high=1)
    angle = np.pi / 32
    witnesses = pd.DataFrame({'a': [.999 * np.cos(angle), 1.01, 0],
                              'b': [.999 * np.sin(angle), 0, 0], 'c': [0, 0, 1.1]})
    expected = gate.mask(witnesses)
    assert expected.tolist() == [True, False, False]
    canvas = _install(screen, gate)
    snapshot = json.dumps(screen.gates.gates.to_dict(), sort_keys=True)
    for density, visible in [(1, False), (3, True), (2, False)]:
        screen.apply_settings(replace(screen._settings, anchor_density=density, show_anchors=visible))
        canvas.axes_at(0, 0).view_init(23 + density, -60 + density * 10)
        canvas.render_now()
        assert np.array_equal(screen.gates.gates.mask(witnesses, 'cylinder'), expected)
        assert json.dumps(screen.gates.gates.to_dict(), sort_keys=True) == snapshot
    screen.apply_settings(replace(screen._settings, show_anchors=True))
    role, point = canvas._volume_handles(gate)[0]
    pixel = _screen_points(canvas.axes_at(0, 0), [point])[0]
    canvas._on_press(_event(canvas, pixel))
    canvas._on_release(_event(canvas, pixel))
    assert screen.gates.gates.get('cylinder') is gate


def test_selected_adjacent_surfaces_wheel_moves_shared_vertices_once_and_updates_counts(screen):
    canvas = _install(screen, _box())
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    canvas._surface_gate = 'box'
    canvas._selected_surfaces = {1, 5}
    canvas._edit_axis = 'z'
    ax = canvas.axes_at(0, 0)
    before_limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    original = EditableVolumeGate.from_gate(_box())
    chosen = {i for face, group in zip(original.faces, original.surface_groups) if group in {1, 5} for i in face}
    assert canvas._volume_scroll(_event(canvas, [300, 300], step=1, button='up'))
    edited = screen.gates.gates.get('box')
    displacement = np.asarray(edited.vertices) - original.vertices
    expected = np.zeros_like(displacement)
    expected[list(chosen), 2] = .05
    assert displacement == pytest.approx(expected)
    after_ax = canvas.axes_at(0, 0)
    assert (after_ax.get_xlim3d(), after_ax.get_ylim3d(), after_ax.get_zlim3d()) == before_limits
    rows = pd.DataFrame({'a': [0, 0, 1.1], 'b': [0, 0, 0], 'c': [1.025, 1.1, 0]})
    assert edited.mask(rows).tolist() == [True, False, False]
    screen.gates.tree.set_gates(screen.gates.gates, rows)
    assert screen.gates.tree.tree.topLevelItem(0).text(1) == '1'


@pytest.mark.parametrize('angles', [(25, -60), (41, 125), (-30, -140)])
def test_surface_pick_tracks_the_camera_and_multiselection_is_real(screen, angles):
    canvas = _install(screen, _box())
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    ax = canvas.axes_at(0, 0)
    ax.view_init(*angles)
    canvas._view_angles = *angles, 0.
    canvas._canvas.draw()
    mesh = canvas._editable_boundary(_box())
    picked = {}
    for group in sorted(set(mesh.surface_groups)):
        indices = {i for face, value in zip(mesh.faces, mesh.surface_groups) if value == group for i in face}
        point = np.asarray(mesh.vertices)[sorted(indices)].mean(axis=0)
        pixel = _screen_points(ax, [point])[0]
        event = _event(canvas, pixel, key='control')
        actual = canvas.volume_surface_at(event)
        if actual == ('box', group):
            picked[group] = point
    assert len(picked) >= 2
    groups = list(picked)[:2]
    for group in groups:
        pixel = _screen_points(canvas.axes_at(0, 0), [picked[group]])[0]
        event = _event(canvas, pixel, key='control')
        canvas._on_press(event)
        canvas._on_release(_event(canvas, pixel, key='control'))
    assert canvas._selected_surfaces == set(groups)
    assert canvas._surface_drag is None
    assert screen.gates.gates.get('box') == _box()


def test_saved_mesh_and_editor_controls_restore_without_changing_the_boundary(screen, tmp_path):
    canvas = _install(screen, _box())
    mesh = EditableVolumeGate.from_gate(_box()).moved_vertices([7], [0, 0, .3])
    screen.gates._on_gate_edited(mesh)
    screen.apply_settings(replace(screen._settings, anchor_density=4, show_anchors=False))
    path = tmp_path / 'strategy.json'
    screen.save_gates(str(path))
    payload = json.loads(path.read_text())
    assert payload['editor_controls'] == {'anchor_density': 4, 'show_anchors': False}
    screen.gates.set_gates(GateSet())
    screen.apply_settings(GateEditorSettings())
    screen.open_settings()
    assert screen.load_gates(str(path))
    assert screen.gates.gates.get('box') == mesh
    assert canvas._anchor_density == 4 and not canvas._show_anchors
    assert screen._settings_dialog._anchor_density.value() == 4
    assert not screen._settings_dialog._show_anchors.isChecked()
    assert np.array_equal(mesh.mask(screen._frame), screen.gates.gates.mask(screen._frame, 'box'))
    payload['editor_controls']['anchor_density'] = 0
    path.write_text(json.dumps(payload))
    assert not screen.load_gates(str(path))
    assert screen.gates.gates.get('box') == mesh


def test_multiselected_surface_drag_commits_one_connected_boundary(screen):
    canvas = _install(screen, _box())
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    mesh = canvas._editable_boundary(_box())
    selected = {}
    for group in sorted(set(mesh.surface_groups)):
        indices = {i for face, value in zip(mesh.faces, mesh.surface_groups) if value == group for i in face}
        point = np.asarray(mesh.vertices)[sorted(indices)].mean(axis=0)
        pixel = _screen_points(canvas.axes_at(0, 0), [point])[0]
        if canvas.volume_surface_at(_event(canvas, pixel)) == ('box', group):
            selected[group] = point
    assert len(selected) >= 2
    groups = list(selected)[:2]
    for group in groups:
        pixel = _screen_points(canvas.axes_at(0, 0), [selected[group]])[0]
        canvas._on_press(_event(canvas, pixel, key='control'))
        canvas._on_release(_event(canvas, pixel, key='control'))
    canvas._edit_axis = 'z'
    start = _screen_points(canvas.axes_at(0, 0), [selected[groups[0]]])[0]
    canvas._on_press(_event(canvas, start))
    assert set(canvas._surface_drag[2]) == set(groups)
    point = canvas._surface_drag[4]
    direction = canvas._pixels_per_unit(canvas.axes_at(0, 0), point)[:, 2]
    end = start + direction * .08
    canvas._on_motion(_event(canvas, end))
    assert screen.gates.gates.get('box') == _box()
    canvas._on_release(_event(canvas, end))
    expected = mesh.moved_surfaces(groups, [0, 0, .08])
    assert np.asarray(screen.gates.gates.get('box').vertices) == pytest.approx(np.asarray(expected.vertices))
    assert screen.gates.undo_stack.canUndo()


@pytest.mark.parametrize('cancel', ['mode', 'dimension', 'settings', 'load', 'resize', 'axis'])
def test_cancelled_or_resized_anchor_gesture_cannot_commit_stale_geometry(screen, cancel):
    canvas = _install(screen, _box())
    point = np.asarray([1., 1., 1.])
    pixel = _screen_points(canvas.axes_at(0, 0), [point])[0]
    canvas._on_press(_event(canvas, pixel))
    assert canvas._handle_drag is not None
    if cancel == 'mode':
        canvas.set_drag_mode('spin')
    elif cancel == 'dimension':
        screen.gates.mode_requested.emit('2D')
    elif cancel == 'settings':
        screen.apply_settings(replace(screen._settings, anchor_density=3))
    elif cancel == 'load':
        screen.gates.set_gates(GateSet([_box()]))
    elif cancel == 'resize':
        canvas._canvas.resize(canvas._canvas.width() + 30, canvas._canvas.height())
    else:
        canvas._edit_axis = 'z'
    canvas._on_release(_event(canvas, pixel + [20, 10]))
    assert screen.gates.gates.get('box') == _box()
    assert canvas._surface_drag is None and canvas._handle_drag is None


@pytest.mark.parametrize('gate', [RectGate(name='flat', x_column='a', y_column='b', x_low=-1, x_high=1, y_low=-1, y_high=1),
                                EllipseGate(name='flat', x_column='a', y_column='b', x_radius=1, y_radius=1)])
def test_actual_2d_outline_anchor_edit_updates_membership_and_undo(screen, gate):
    screen.gates.mode_requested.emit('2D')
    screen.gates.set_gates(GateSet([gate]))
    screen.gates.tree.select('flat')
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)
    handles = canvas._handles_for(ax, gate)
    handle = next(h for h in handles if h.role.startswith('outline:') and h.x > .5)
    pixel = ax.transData.transform([handle.x, handle.y])
    original = _event(canvas, pixel, xdata=handle.x, ydata=handle.y)
    canvas._on_press(original)
    assert canvas._resize == ('flat', handle.role)
    assert canvas._dragged_to(original) is gate
    end = _event(canvas, pixel + [5, 0], xdata=handle.x + .2, ydata=handle.y)
    before = screen.gates.gates.mask(screen._frame, 'flat')
    canvas._on_motion(end)
    assert np.array_equal(screen.gates.gates.mask(screen._frame, 'flat'), before)
    canvas._on_release(end)
    edited = screen.gates.gates.get('flat')
    assert isinstance(edited, PolygonGate)
    points = canvas._gate_points(canvas.axes_at(0, 0), edited)
    assert tuple(points) == edited.vertices
    assert (handle.x + .2, handle.y) in edited.vertices
    screen.gates.undo_stack.undo()
    assert screen.gates.gates.get('flat') == gate


def test_crossing_2d_anchor_move_is_refused_without_changing_counts(screen):
    screen.gates.mode_requested.emit('2D')
    gate = PolygonGate(name='flat', x_column='a', y_column='b', vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
    screen.gates.set_gates(GateSet([gate]))
    canvas = screen.gates.canvas
    canvas._resize = ('flat', 'vertex:0')
    failures = []
    canvas.wand_failed.connect(failures.append)
    point = _event(canvas, [0, 0], xdata=1.5, ydata=0)
    assert canvas._dragged_to(point) is None
    assert failures and 'cross' in failures[0]
    assert screen.gates.gates.get('flat') == gate


def test_anchor_density_visibility_controls_change_handles_not_scientific_geometry(screen):
    canvas = _install(screen, _box())
    screen.open_settings()
    dialog = screen._settings_dialog
    before = screen.gates.gates.to_dict()
    counts = []
    for density in (1, 2, 3, 4):
        dialog._anchor_density.setValue(density)
        anchors = canvas._volume_handles(_box())
        counts.append(len(anchors))
        for axis in range(3):
            values = sorted(set(point[axis] for _, point in anchors))
            assert values == pytest.approx(np.linspace(-1, 1, density + 1))
    assert counts == [8, 26, 56, 98]
    dialog._show_anchors.setChecked(False)
    role, point = canvas._volume_handles(_box())[0]
    pixel = _screen_points(canvas.axes_at(0, 0), [point])[0]
    assert not canvas._grab_volume_handle(_event(canvas, pixel))
    assert screen.gates.gates.to_dict() == before
    screen.gates.mode_requested.emit('2D')
    ax = canvas.axes_at(0, 0)
    assert canvas.handle_at(_event(canvas, ax.transData.transform([-1, -1]))) is None


def test_curved_gate_depth_anchor_creates_a_nonuniform_volume_and_updates_membership(screen):
    gate = CylinderGate(name='depth', u_column='a', v_column='b', axis_column='c',
                        u_radius=1, v_radius=1, axis_low=-1, axis_high=1)
    canvas = _install(screen, gate)
    canvas._edit_axis = 'z'
    mesh = canvas._editable_boundary(gate)
    candidates = [(role, point) for role, point in canvas._volume_handles(gate) if np.allclose(point, [1, 0, 1])]
    assert len(candidates) == 1
    role, point = candidates[0]
    start = _screen_points(canvas.axes_at(0, 0), [point])[0]
    canvas._on_press(_event(canvas, start))
    assert canvas._handle_drag[1] == role
    end = start + canvas._pixels_per_unit(canvas.axes_at(0, 0), point)[:, 2] * .4
    canvas._on_motion(_event(canvas, end))
    canvas._on_release(_event(canvas, end))
    edited = screen.gates.gates.get('depth')
    assert isinstance(edited, EditableVolumeGate)
    roof = np.asarray(edited.vertices)[np.asarray(mesh.vertices)[:, 2] == 1, 2]
    assert np.ptp(roof) == pytest.approx(.4)
    neighbor = np.cos(np.pi / 16)
    assert .4 * (.995 - neighbor) / (1 - neighbor) > .2
    witnesses = pd.DataFrame({'a': [.995, -.995, .995], 'b': [0, 0, 0], 'c': [1.2, 1.2, 1.6]})
    assert gate.mask(witnesses).tolist() == [False, False, False]
    assert edited.mask(witnesses).tolist() == [True, False, False]


def test_rendered_view_lasso_can_be_edited_after_another_camera_rotation(screen):
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)
    center = _screen_points(ax, [[0, 0, 0]])[0]
    outline = [center + offset for offset in ((-35, -35), (35, -35), (35, 35), (-35, 35))]
    gate = replace(canvas.gate_from_screen_outline(outline), name='view')
    canvas = _install(screen, gate)
    original = gate.to_dict()
    canvas._view_angles = 37., 105., 0.
    canvas.render_now()
    mesh = canvas._editable_boundary(gate)
    assert isinstance(mesh, EditableVolumeGate)
    assert len(canvas._volume_handles(gate)) > len(mesh.vertices)
    assert screen.gates.gates.get('view').to_dict() == original
    canvas._surface_gate = 'view'
    canvas._selected_surfaces = {max(mesh.surface_groups)}
    canvas._edit_axis = 'z'
    assert canvas._volume_scroll(_event(canvas, [300, 300], step=1, button='up'))
    edited = screen.gates.gates.get('view')
    assert isinstance(edited, EditableVolumeGate)
    assert edited != mesh
    assert np.array_equal(screen.gates.gates.mask(screen._frame, 'view'), edited.mask(screen._frame))


@pytest.mark.parametrize('values', [{'anchor_density': True}, {'anchor_density': 2.5}, {'show_anchors': 'false'}])
def test_malformed_anchor_settings_are_refused(values):
    with pytest.raises(ValueError):
        GateEditorSettings(**values)


@pytest.mark.parametrize('controls', [None, [], {'unknown_control': True}, {'anchor_density': None}, {'anchor_density': float('inf')}])
def test_invalid_saved_controls_cannot_replace_the_live_strategy(screen, tmp_path, controls):
    canvas = _install(screen, _box())
    path = tmp_path / 'invalid.json'
    data = GateSet([_box().rename('other')]).to_dict()
    data['editor_controls'] = controls
    path.write_text(json.dumps(data))
    assert not screen.load_gates(str(path))
    assert screen.gates.gates.get('box') == _box()
    assert canvas.active_gate == 'box'


def test_a_reloaded_strategy_cannot_reuse_another_mesh_surface_selection(screen):
    canvas = _install(screen, _box())
    canvas._surface_gate = 'box'
    canvas._selected_surfaces = {1, 5}
    replacement = EditableVolumeGate.from_gate(_box()).moved_vertices([7], [0, 0, .1])
    screen.gates.set_gates(GateSet([replacement]))
    assert canvas._selected_surfaces == set()
    assert canvas._surface_gate is None


def test_end_on_and_off_axes_surface_picks_do_not_invent_a_hit(screen):
    canvas = _install(screen, _box())
    canvas._view_angles = 0., 0., 0.
    canvas.render_now()
    point = _screen_points(canvas.axes_at(0, 0), [[0, 0, 0]])[0]
    assert canvas.volume_surface_at(_event(canvas, point)) is not None
    assert canvas.volume_surface_at(_event(canvas, point, inaxes=None)) is None
    assert canvas.volume_surface_at(_event(canvas, [-1e6, -1e6])) is None


def test_surface_reselection_and_toggle_off_are_explicit(screen):
    canvas = _install(screen, _box())
    screen.apply_settings(replace(screen._settings, show_anchors=False))
    ax = canvas.axes_at(0, 0)
    point = _screen_points(ax, [[0, 0, 1]])[0]
    selected = canvas.volume_surface_at(_event(canvas, point))
    assert selected is not None
    canvas._surface_gate = 'another'
    canvas._selected_surfaces = {999}
    canvas._on_press(_event(canvas, point, key='control'))
    assert canvas._selected_surfaces == {selected[1]}
    point = _screen_points(canvas.axes_at(0, 0), [[0, 0, 1]])[0]
    canvas._on_press(_event(canvas, point, key='control'))
    assert canvas._selected_surfaces == set()
    canvas._on_press(_event(canvas, [-1e6, -1e6]))
    assert canvas._surface_drag is None


def test_surface_wheel_follows_outward_normal_for_both_orientation_encodings(screen):
    base = EditableVolumeGate.from_gate(_box())
    for mesh in (base, replace(base, faces=tuple(tuple(reversed(face)) for face in base.faces))):
        canvas = _install(screen, mesh)
        canvas._selected_surfaces = {1}
        canvas._surface_gate = 'box'
        canvas._edit_axis = ''
        canvas._volume_scroll(_event(canvas, [300, 300], step=1, button='up'))
        edited = screen.gates.gates.get('box')
        expected = np.asarray(mesh.vertices).copy()
        expected[4:, 2] += .05
        assert np.asarray(edited.vertices) == pytest.approx(expected)


def test_opposite_selected_surfaces_have_no_arbitrary_wheel_direction(screen):
    canvas = _install(screen, _box())
    canvas._selected_surfaces = {0, 1}
    canvas._surface_gate = 'box'
    assert canvas._edit_surface_wheel(_event(canvas, [300, 300], step=1, button='up'))
    assert screen.gates.gates.get('box') == _box()


def test_a_collapsing_surface_and_unprojectable_axis_are_refused_without_a_commit(screen):
    canvas = _install(screen, _box())
    canvas._selected_surfaces = {1}
    canvas._surface_gate = 'box'
    canvas._edit_axis = 'z'
    messages = []
    canvas.depth_requested.connect(messages.append)
    assert canvas._edit_surface_wheel(_event(canvas, [300, 300], step=-40, button='down'))
    assert messages
    assert screen.gates.gates.get('box') == _box()
    canvas._view_angles = 90., 0., 0.
    canvas.render_now()
    assert np.array_equal(canvas._volume_edit_delta(canvas.axes_at(0, 0), np.zeros(3), np.asarray([20., 0.])), np.zeros(3))


def test_end_on_real_anchor_drag_preserves_geometry_and_undo_until_the_axis_is_exposed(screen):
    canvas = _install(screen, _box())
    canvas._edit_axis = 'z'
    canvas._view_angles = 90., 0., 0.
    canvas.render_now()
    handles = canvas._volume_handles(_box())
    role, point = next((role, point) for role, point in handles if np.array_equal(point, [0, 0, 1]))
    start = _screen_points(canvas.axes_at(0, 0), [point])[0]
    before = screen.gates.gates.to_dict()
    history = screen.gates.undo_stack.index()
    messages = []
    canvas.depth_requested.connect(messages.append)
    canvas._on_press(_event(canvas, start))
    assert canvas._handle_drag[1] == role
    canvas._on_motion(_event(canvas, start + [20, 0]))
    canvas._on_release(_event(canvas, start + [20, 0]))
    assert screen.gates.gates.to_dict() == before
    assert screen.gates.undo_stack.index() == history
    assert any('Rotate the graph' in message for message in messages)
    canvas._view_angles = 31., -47., 0.
    canvas.render_now()
    start = _screen_points(canvas.axes_at(0, 0), [point])[0]
    delta = canvas._pixels_per_unit(canvas.axes_at(0, 0), point)[:, 2] * .12
    canvas._on_press(_event(canvas, start))
    canvas._on_release(_event(canvas, start + delta))
    edited = screen.gates.gates.get('box')
    assert isinstance(edited, EditableVolumeGate)
    assert len(edited.vertices) == 9
    assert edited.vertices[-1] == pytest.approx((0, 0, 1.12))
    assert screen.gates.undo_stack.index() == history + 1


def test_surface_selection_survives_repeated_wheel_events_and_count_refresh(screen):
    canvas = _install(screen, _box())
    canvas._selected_surfaces = {1, 5}
    canvas._surface_gate = 'box'
    canvas._edit_axis = 'z'
    for _ in range(3):
        canvas._volume_scroll(_event(canvas, [300, 300], step=1, button='up'))
        QApplication.processEvents()
        assert canvas._selected_surfaces == {1, 5}
    edited = screen.gates.gates.get('box')
    original = EditableVolumeGate.from_gate(_box())
    selected = {i for face, group in zip(original.faces, original.surface_groups) if group in {1, 5} for i in face}
    expected = np.asarray(original.vertices).copy()
    expected[list(selected), 2] += 2 * (1.025 ** 3 - 1)
    assert np.asarray(edited.vertices) == pytest.approx(expected)


def test_real_event_turns_between_press_and_release_do_not_cancel_a_valid_anchor_drag(screen):
    canvas = _install(screen, _box())
    canvas._edit_axis = 'z'
    point = np.asarray([1., 1., 1.])
    start = _screen_points(canvas.axes_at(0, 0), [point])[0]
    end = start + canvas._pixels_per_unit(canvas.axes_at(0, 0), point)[:, 2] * .12
    canvas._on_press(_event(canvas, start))
    captured = canvas._edit_drag_projection
    QApplication.processEvents()
    assert canvas._edit_drag_projection is not None
    assert tuple(canvas._canvas.size().toTuple()) == captured[1]
    canvas._on_motion(_event(canvas, end))
    QApplication.processEvents()
    canvas._on_release(_event(canvas, end))
    edited = screen.gates.gates.get('box')
    assert isinstance(edited, EditableVolumeGate)
    assert edited.vertices[7][2] == pytest.approx(1.12)


def test_export_uses_the_actual_edited_depth_boundary_and_preserves_parent_membership(screen, tmp_path):
    original = _box()
    mesh = EditableVolumeGate.from_gate(original).moved_vertices([7], [0, 0, .4])
    rows = pd.DataFrame({'a': [.9, -.9, 1.1, .9], 'b': [.9, -.9, 0, .9], 'c': [1.2, 1.2, 0, .5],
                         'plateID': ['p1'] * 4, 'rowID': ['r1'] * 4, 'columnID': ['c1'] * 4,
                         'fieldID': ['f1'] * 4, 'object_label': [1, 2, 3, 4]})
    mesh = replace(mesh, name='edited', parent='parent')
    parent = BoxGate.from_limits('parent', ('a', 'b', 'c'), [(-1, 1), (-1, 1), (1, 2)])
    gates = GateSet([parent, mesh])
    assert mesh.mask(rows).tolist() == [True, False, False, True]
    assert gates.mask(rows, 'edited').tolist() == [True, False, False, False]
    path = tmp_path / 'measurement.db'
    with sqlite3.connect(path) as database:
        rows.to_sql('cell', database, index=False)
    written, failed = screen._write_gates(str(path), 'cell', gates)
    assert not failed
    assert ('edited', 1) in written
    with sqlite3.connect(path) as database:
        exported = pd.read_sql_query('SELECT object_label,edited FROM filters ORDER BY object_label', database)
    assert exported.edited.tolist() == [1, 0, 0, 0]


def test_edit_mode_disambiguates_anchor_drag_from_rotation_and_keeps_help_inline(screen):
    canvas = _install(screen, EditableVolumeGate.from_gate(_box()))
    panel = screen.gates
    edit = panel._drag_buttons['edit']
    assert not edit.toolTip()
    assert 'piecewise' in panel._action_help.explains(edit)
    QApplication.sendEvent(edit, QEvent(QEvent.Type.Enter))
    assert 'piecewise' in panel._action_help.text()
    before = panel._action_help.height()
    for button in panel._drag_buttons.values():
        QApplication.sendEvent(button, QEvent(QEvent.Type.Enter))
        assert not button.toolTip()
        assert panel._action_help.height() == before
    role, point = canvas._volume_handles(screen.gates.gates.get('box'))[0]
    pixel = _screen_points(canvas.axes_at(0, 0), [point])[0]
    canvas.set_drag_mode('spin')
    assert not canvas._grab_volume_handle(_event(canvas, pixel))
    canvas._on_press(_event(canvas, pixel, button=3))
    assert canvas._handle_drag is None and canvas._surface_drag is None
    canvas._on_motion(_event(canvas, pixel + [25, 10], button=3))
    canvas._on_release(_event(canvas, pixel + [25, 10], button=3))
    assert canvas._view_angles is not None


def test_2d_density_controls_each_edge_without_converting_a_stored_gate(screen):
    screen.gates.mode_requested.emit('2D')
    gate = RectGate(name='flat', x_column='a', y_column='b', x_low=-1, x_high=1, y_low=-1, y_high=1)
    screen.gates.set_gates(GateSet([gate]))
    canvas = screen.gates.canvas
    counts = []
    for density in (1, 2, 4):
        screen.apply_settings(replace(screen._settings, anchor_density=density))
        handles = canvas._handles_for(canvas.axes_at(0, 0), gate)
        counts.append(len(handles))
        for edge in (-1., 1.):
            assert sorted(set(h.x for h in handles if h.y == edge)) == pytest.approx(np.linspace(-1, 1, 2 * density + 1))
            assert sorted(set(h.y for h in handles if h.x == edge)) == pytest.approx(np.linspace(-1, 1, 2 * density + 1))
        assert screen.gates.gates.get('flat') is gate
    assert counts[0] < counts[1] < counts[2]
