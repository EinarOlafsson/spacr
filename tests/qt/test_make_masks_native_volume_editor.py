"""Native shared volume editing updates full mask voxels, not preview dots.

2026-10-12: the anchor editing that lived on the "3D anchors" tab of a
separate window is now the True 3D view inside Make Masks itself; these
tests drive that view through the screen.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen
from spacr.qt.widgets.gate_editor import _screen_points
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate
from spacr.tiff_io import write_tiff


@pytest.fixture
def screen(qtbot, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1300, 900)
    widget.show()
    yield widget
    stage = widget._mask_stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle(), timeout=5000)
    stage.dirty = False


@pytest.fixture
def editor(screen, tmp_path):
    image = np.arange(8 ** 3, dtype=np.uint16).reshape(8, 8, 8)
    write_tiff(tmp_path / 'volume.tif', image, metadata={'axes': 'ZYX', 'voxel_spacing': [1., .5, .25], 'unit': 'um'})
    assert screen.open_volume(str(tmp_path / 'volume.tif')) is screen._mask_stage
    stage = screen._mask_stage
    stage.set_view('true3d')
    for _ in range(4):
        QApplication.processEvents()
    return stage


def _box(name='1'):
    return BoxGate.from_limits(name, ('X', 'Y', 'Z'), [(0, 1), (0, 2), (0, 4)])


def _draw(editor, qtbot, gate=None):
    editor._panel.canvas.gate_drawn.emit(gate or _box('(unnamed)'))
    qtbot.waitUntil(lambda: not editor.busy, timeout=5000)
    assert editor.session.mask.any()
    return editor.session.geometries[1]


def _event(canvas, point, **extra):
    values = dict(inaxes=canvas.axes_at(0, 0), x=float(point[0]), y=float(point[1]), xdata=0., ydata=0.,
                  button=1, step=0, key=None, dblclick=False)
    values.update(extra)
    return SimpleNamespace(**values)


def test_actual_native_view_draw_save_reopen_matches_independent_voxels(editor, qtbot):
    _draw(editor, qtbot)
    expected = np.zeros((8, 8, 8), dtype=np.uint16)
    expected[:4, :4, :4] = 1
    assert np.array_equal(editor.session.mask, expected)
    assert '64 voxels' in editor._count.text()
    assert editor._panel.tree_section.isHidden()
    assert editor._panel.canvas._mode == '3D'
    assert editor.save()
    session = engine.load_volume_image_and_mask(str(__import__('pathlib').Path(editor.session.source_path).parent), 'volume.tif')
    assert np.array_equal(session.mask, expected)
    assert session.geometries == editor.session.geometries
    assert session.axes == 'ZYX' and session.spacing == (1., .5, .25)


def test_the_preview_notice_stays_and_the_graph_notice_stays_hidden(editor, screen):
    notice = editor.findChild(type(editor._count), 'VolumeMaskPreviewNotice')
    assert notice.isVisible()
    # 2026-10-15: the notice now describes the downsampled volume rendering.
    assert 'downsampled' in notice.text() and 'full image' in notice.text()
    assert editor._panel.canvas._notice.isHidden()
    editor._panel.canvas.render_now()
    QApplication.processEvents()
    assert editor._panel.canvas._notice.isHidden()
    screen._status_label.setText('Updated 32 voxels.')
    QApplication.processEvents()
    assert notice.isVisible()


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_rotated_anchor_drag_edits_real_physical_axis_and_mask_history(editor, qtbot, axis):
    gate = _draw(editor, qtbot)
    canvas = editor._panel.canvas
    canvas._view_angles = (31., -47., 0.)
    canvas.render_now()
    for _ in range(3):
        QApplication.processEvents()
    editor._panel._edit_axis_picker.setCurrentIndex(editor._panel._edit_axis_picker.findData(axis))
    role, point = next((role, point) for role, point in canvas._volume_handles(gate)
                       if np.array_equal(point, [1., 2., 4.]))
    ax = canvas.axes_at(0, 0)
    start = _screen_points(ax, [point])[0]
    movement = canvas._pixels_per_unit(ax, point)[:, 'xyz'.index(axis)] * .8
    old = editor.session.mask.copy()
    canvas._on_press(_event(canvas, start))
    canvas._on_motion(_event(canvas, start + movement))
    canvas._on_release(_event(canvas, start + movement))
    qtbot.waitUntil(lambda: not editor.busy, timeout=5000)
    edited = editor.session.geometries[1]
    difference = np.asarray(edited.vertices) - gate.vertices
    assert np.allclose(difference[np.arange(len(difference)) != int(role.split(':')[1])], 0)
    assert difference[int(role.split(':')[1]), 'xyz'.index(axis)] == pytest.approx(.8)
    new = editor.session.mask.copy()
    assert not np.array_equal(old, new)
    editor._screen._act_undo.trigger()
    assert np.array_equal(editor.session.mask, old) and editor.session.geometries[1] == gate
    editor._screen._act_redo.trigger()
    assert np.array_equal(editor.session.mask, new) and editor.session.geometries[1] == edited


def test_density_visibility_only_affect_handles_and_multisurface_wheel_changes_voxels(editor, qtbot):
    gate = _draw(editor, qtbot)
    canvas = editor._panel.canvas
    snapshot = editor.session.mask.copy()
    before_history = len(editor.session._undo)
    for density in (1, 4, 2):
        editor._density.setValue(density)
        editor._anchors.setChecked(False)
        assert np.array_equal(editor.session.mask, snapshot)
        assert editor.session.geometries[1] == gate
        assert len(editor.session._undo) == before_history
        editor._anchors.setChecked(True)
    editor._anchors.setChecked(False)
    canvas._surface_gate = '1'
    canvas._selected_surfaces = {1, 5}
    canvas._edit_axis = 'z'
    for _ in range(6):
        assert canvas._volume_scroll(_event(canvas, [200, 200], step=1, button='up'))
        qtbot.waitUntil(lambda: not editor.busy, timeout=5000)
        assert canvas._surface_gate == '1' and canvas._selected_surfaces == {1, 5}
    edited = editor.session.geometries[1]
    selected = {i for face, group in zip(gate.faces, gate.surface_groups) if group in {1, 5} for i in face}
    expected = np.zeros((len(gate.vertices), 3))
    expected[list(selected), 2] = 4 * (1.025 ** 6 - 1)
    assert np.asarray(edited.vertices) - gate.vertices == pytest.approx(expected)
    assert not np.array_equal(snapshot, editor.session.mask)


def test_a_colliding_edited_volume_is_restored_and_other_label_is_unchanged(editor, qtbot):
    _draw(editor, qtbot)
    editor.session.mask[6, 6, 6] = 17
    original = editor.session.mask.copy()
    geometry = editor.session.geometries[1]
    editor._panel.canvas.gate_edited.emit(EditableVolumeGate.from_gate(BoxGate.from_limits('1', ('X', 'Y', 'Z'), [(0, 2), (0, 4), (0, 8)])))
    qtbot.waitUntil(lambda: not editor.busy, timeout=5000)
    assert 'overlaps' in editor._screen._status_label.text()
    assert np.array_equal(editor.session.mask, original)
    assert editor.session.geometries[1] == geometry
    assert editor._panel.gates.get('1') == geometry


def test_end_on_axis_refusal_preserves_voxels_geometry_and_history(editor, qtbot):
    gate = _draw(editor, qtbot)
    canvas = editor._panel.canvas
    canvas._view_angles = (0., 0., 0.)
    canvas.render_now()
    canvas._edit_axis = 'x'
    role, point = canvas._volume_handles(gate)[-1]
    start = _screen_points(canvas.axes_at(0, 0), [point])[0]
    snapshot = editor.session.mask.copy()
    history = len(editor.session._undo)
    canvas._on_press(_event(canvas, start))
    canvas._on_motion(_event(canvas, start + [30., 0.]))
    canvas._on_release(_event(canvas, start + [30., 0.]))
    assert not editor.busy
    assert np.array_equal(editor.session.mask, snapshot)
    assert editor.session.geometries[1] == gate and len(editor.session._undo) == history


def test_make_masks_opens_a_volume_in_its_own_view_and_back_to_2d(qtbot, tmp_path):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    volume = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    write_tiff(tmp_path / 'volume.tif', volume, metadata={'axes': 'ZYX'})
    stage = screen.open_volume(str(tmp_path / 'volume.tif'))
    assert stage is screen._mask_stage and stage.session.image.shape == (3, 4, 5)
    assert screen._canvas.image.shape == (4, 5)
    assert not hasattr(screen, '_btn_volume')
    assert 'draw 3D masks slice by slice' in screen._btn_open.toolTip()
    assert screen._on_pick_file(str(tmp_path / 'volume.tif')) is True
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    assert screen._mask_stage is stage
    np.testing.assert_array_equal(stage.session.image, volume)
    folder = tmp_path / 'fields'
    folder.mkdir()
    write_tiff(folder / 'field.tif', np.arange(25, dtype=np.uint16).reshape(5, 5))
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    assert stage.session is None and screen._canvas.image.shape == (5, 5)


def test_volume_open_help_uses_footer_when_inside_an_action_scope(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.tooltip_policy import OPT_OUT_PROPERTY

    host = AppScreen('mask')
    qtbot.addWidget(host)
    screen = MakeMasksScreen(host)
    command = screen._btn_open
    host._install_action_hints()
    assert command in host._action_hints
    assert command.property(OPT_OUT_PROPERTY)
    assert command.toolTip() == ''
    assert host.eventFilter(command, QEvent(QEvent.ToolTip)) is True
    host._show_hover_hint(command)
    assert 'draw 3D masks slice by slice' in host._hint_strip.toolTip()


def test_real_rotated_rectangle_through_view_draws_full_native_volume(editor, qtbot):
    canvas = editor._panel.canvas
    editor._panel._volume_shape.setCurrentIndex(editor._panel._volume_shape.findData('view_rect'))
    editor._panel._drag_buttons['draw'].click()
    canvas._view_angles = (31., -47., 0.)
    canvas.render_now()
    for _ in range(3):
        QApplication.processEvents()
    ax = canvas.axes_at(0, 0)
    points = _screen_points(ax, editor._panel._frame[['X', 'Y', 'Z']].to_numpy())
    low, high = points.min(axis=0) - 4, points.max(axis=0) + 4
    canvas._on_press(_event(canvas, low))
    canvas._on_motion(_event(canvas, high))
    canvas._on_release(_event(canvas, high))
    qtbot.waitUntil(lambda: not editor.busy, timeout=5000)
    assert np.all(editor.session.mask == 1)
    assert isinstance(editor.session.geometries[1], EditableVolumeGate)


def test_pending_edit_cancellation_on_leaving_publishes_nothing_late(editor, qtbot, monkeypatch):
    import threading
    started, release = threading.Event(), threading.Event()
    original = editor.session._paint_geometry
    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(editor.session, '_paint_geometry', blocked)
    editor._panel.canvas.gate_drawn.emit(_box('(unnamed)'))
    qtbot.waitUntil(started.is_set, timeout=5000)
    assert editor.busy and not editor._strip.isEnabled()
    session, worker = editor.session, editor._worker
    assert editor._screen._volume_leave_ok()
    editor._screen._leave_volume()
    release.set()
    qtbot.waitUntil(worker.idle, timeout=5000)
    QApplication.processEvents()
    assert not session.mask.any() and not session._undo and not session.geometries
    assert editor.session is None


def test_unsaved_leave_cancel_keeps_the_volume_and_its_geometry(editor, qtbot, monkeypatch):
    _draw(editor, qtbot)
    original = editor.session.mask.copy()
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Cancel)
    assert not editor._screen._volume_leave_ok()
    assert editor.session is not None and editor.dirty
    assert np.array_equal(editor.session.mask, original)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Discard)


def test_saved_control_changes_reopen_without_changing_voxels(editor, qtbot):
    _draw(editor, qtbot)
    original = editor.session.mask.copy()
    editor._density.setValue(4)
    editor._anchors.setChecked(False)
    assert np.array_equal(editor.session.mask, original)
    assert editor.save()
    path = editor.session.source_path
    screen = editor._screen
    assert screen.open_volume(path) is editor
    editor.set_view('true3d')
    assert editor._density.value() == 4 and not editor._anchors.isChecked()
    assert np.array_equal(editor.session.mask, original)


def test_dirty_busy_leave_requires_explicit_discard_and_cancel_preserves_completed_edit(editor, qtbot, monkeypatch):
    import threading
    gate = _draw(editor, qtbot)
    session = editor.session
    completed = session.mask.copy()
    geometry = dict(session.geometries)
    history = len(session._undo)
    started, release = threading.Event(), threading.Event()
    original = session._paint_geometry
    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(session, '_paint_geometry', blocked)
    editor._panel.canvas.gate_edited.emit(gate.moved_vertices([7], (.2, .2, .2)))
    qtbot.waitUntil(started.is_set, timeout=5000)
    questions = []
    def refuse(*args, **kwargs):
        questions.append(args)
        return QMessageBox.Cancel
    monkeypatch.setattr(QMessageBox, 'question', refuse)
    screen = editor._screen
    screen.show()
    assert not screen.close()
    assert questions and questions[-1][3] == QMessageBox.Discard | QMessageBox.Cancel
    assert editor.busy and not editor._cancel.is_set()
    assert np.array_equal(session.mask, completed) and session.geometries == geometry
    assert len(session._undo) == history
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Discard)
    worker = editor._worker
    assert screen.close()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    release.set()
    qtbot.waitUntil(worker.idle, timeout=5000)
    QApplication.processEvents()
    assert np.array_equal(session.mask, completed) and session.geometries == geometry
    assert len(session._undo) == history
