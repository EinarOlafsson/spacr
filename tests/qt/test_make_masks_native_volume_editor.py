"""Native shared volume editing updates full mask voxels, not preview dots."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from shiboken6 import isValid

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen, _VolumeMaskEditor
from spacr.qt.widgets.gate_editor import _screen_points
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate
from spacr.tiff_io import write_tiff


@pytest.fixture
def editor(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    image = np.arange(8 ** 3, dtype=np.uint16).reshape(8, 8, 8)
    write_tiff(tmp_path / 'volume.tif', image, metadata={'axes': 'ZYX', 'voxel_spacing': [1., .5, .25], 'unit': 'um'})
    session = engine.load_volume_image_and_mask(str(tmp_path), 'volume.tif')
    widget = _VolumeMaskEditor(session)
    qtbot.addWidget(widget)
    widget.show()
    widget._tabs.setCurrentWidget(widget._panel)
    for _ in range(4):
        QApplication.processEvents()
    yield widget
    if isValid(widget):
        widget._dirty = False
        widget.close()
    qtbot.waitUntil(widget._worker.idle, timeout=5000)


def _box(name='1'):
    return BoxGate.from_limits(name, ('X', 'Y', 'Z'), [(0, 1), (0, 2), (0, 4)])


def _draw(editor, qtbot, gate=None):
    editor._panel.canvas.gate_drawn.emit(gate or _box('(unnamed)'))
    qtbot.waitUntil(lambda: not editor._busy, timeout=5000)
    assert editor._session.mask.any()
    return editor._session.geometries[1]


def _event(canvas, point, **extra):
    values = dict(inaxes=canvas.axes_at(0, 0), x=float(point[0]), y=float(point[1]), xdata=0., ydata=0.,
                  button=1, step=0, key=None, dblclick=False)
    values.update(extra)
    return SimpleNamespace(**values)


def test_actual_native_view_draw_save_reopen_matches_independent_voxels(editor, qtbot):
    _draw(editor, qtbot)
    expected = np.zeros((8, 8, 8), dtype=np.uint16)
    expected[:4, :4, :4] = 1
    assert np.array_equal(editor._session.mask, expected)
    assert '64 voxels' in editor._count.text()
    assert editor._panel.tree_section.isHidden()
    assert editor._panel.canvas._mode == '3D'
    assert editor._save_mask()
    session = engine.load_volume_image_and_mask(str(__import__('pathlib').Path(editor._session.source_path).parent), 'volume.tif')
    assert np.array_equal(session.mask, expected)
    assert session.geometries == editor._session.geometries
    assert session.axes == 'ZYX' and session.spacing == (1., .5, .25)


def test_long_volume_status_is_readable_without_moving_graph_or_actions(editor, qtbot):
    editor._notice.setText('Ready.')
    editor._count.setText('One object.')
    QApplication.processEvents()
    positions = (editor._panel.geometry(), editor._slice_view.geometry(),
                 editor._save.geometry())
    long_notice = 'A spatial volume error with complete physical coordinates. ' * 24
    long_count = 'Exact voxel count and spacing for the selected object. ' * 24
    editor._notice.setText(long_notice)
    editor._count.setText(long_count)
    QApplication.processEvents()
    assert (editor._panel.geometry(), editor._slice_view.geometry(),
            editor._save.geometry()) == positions
    assert editor._notice.text() == long_notice
    assert editor._count.text() == long_count
    for status in (editor._notice, editor._count):
        view = status._view
        assert view.toPlainText() == status.text()
        assert view.verticalScrollBar().maximum() > 0
        view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())
        assert view.verticalScrollBar().value() == view.verticalScrollBar().maximum()
    assert editor._preview_notice.isVisible()
    assert 'Sampled' in editor._preview_notice.text()
    assert 'full image' in editor._preview_notice.text()
    assert editor._panel.canvas._notice.isHidden()
    editor._panel.canvas.render_now()
    QApplication.processEvents()
    assert editor._panel.canvas._notice.isHidden()
    editor._notice.setText('Updated 32 voxels.')
    QApplication.processEvents()
    assert editor._preview_notice.isVisible()
    assert 'Sampled' in editor._preview_notice.text()
    assert (editor._panel.geometry(), editor._slice_view.geometry(),
            editor._save.geometry()) == positions


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
    old = editor._session.mask.copy()
    canvas._on_press(_event(canvas, start))
    canvas._on_motion(_event(canvas, start + movement))
    canvas._on_release(_event(canvas, start + movement))
    qtbot.waitUntil(lambda: not editor._busy, timeout=5000)
    edited = editor._session.geometries[1]
    difference = np.asarray(edited.vertices) - gate.vertices
    assert np.allclose(difference[np.arange(len(difference)) != int(role.split(':')[1])], 0)
    assert difference[int(role.split(':')[1]), 'xyz'.index(axis)] == pytest.approx(.8)
    new = editor._session.mask.copy()
    assert not np.array_equal(old, new)
    editor._undo.click()
    assert np.array_equal(editor._session.mask, old) and editor._session.geometries[1] == gate
    editor._redo.click()
    assert np.array_equal(editor._session.mask, new) and editor._session.geometries[1] == edited


def test_density_visibility_only_affect_handles_and_multisurface_wheel_changes_voxels(editor, qtbot):
    gate = _draw(editor, qtbot)
    canvas = editor._panel.canvas
    snapshot = editor._session.mask.copy()
    before_history = len(editor._session._undo)
    for density in (1, 4, 2):
        editor._density.setValue(density)
        editor._anchors.setChecked(False)
        assert np.array_equal(editor._session.mask, snapshot)
        assert editor._session.geometries[1] == gate
        assert len(editor._session._undo) == before_history
        editor._anchors.setChecked(True)
    editor._anchors.setChecked(False)
    canvas._surface_gate = '1'
    canvas._selected_surfaces = {1, 5}
    canvas._edit_axis = 'z'
    for _ in range(6):
        assert canvas._volume_scroll(_event(canvas, [200, 200], step=1, button='up'))
        qtbot.waitUntil(lambda: not editor._busy, timeout=5000)
        assert canvas._surface_gate == '1' and canvas._selected_surfaces == {1, 5}
    edited = editor._session.geometries[1]
    selected = {i for face, group in zip(gate.faces, gate.surface_groups) if group in {1, 5} for i in face}
    expected = np.zeros((len(gate.vertices), 3))
    expected[list(selected), 2] = 4 * (1.025 ** 6 - 1)
    assert np.asarray(edited.vertices) - gate.vertices == pytest.approx(expected)
    assert not np.array_equal(snapshot, editor._session.mask)


def test_a_colliding_edited_volume_is_restored_and_other_label_is_unchanged(editor, qtbot):
    _draw(editor, qtbot)
    editor._session.mask[6, 6, 6] = 17
    original = editor._session.mask.copy()
    geometry = editor._session.geometries[1]
    editor._panel.canvas.gate_edited.emit(EditableVolumeGate.from_gate(BoxGate.from_limits('1', ('X', 'Y', 'Z'), [(0, 2), (0, 4), (0, 8)])))
    qtbot.waitUntil(lambda: not editor._busy, timeout=5000)
    assert 'overlaps' in editor._notice.text()
    assert np.array_equal(editor._session.mask, original)
    assert editor._session.geometries[1] == geometry
    assert editor._panel.gates.get('1') == geometry


def test_end_on_axis_refusal_preserves_voxels_geometry_and_history(editor, qtbot):
    gate = _draw(editor, qtbot)
    canvas = editor._panel.canvas
    canvas._view_angles = (0., 0., 0.)
    canvas.render_now()
    canvas._edit_axis = 'x'
    role, point = canvas._volume_handles(gate)[-1]
    start = _screen_points(canvas.axes_at(0, 0), [point])[0]
    snapshot = editor._session.mask.copy()
    history = len(editor._session._undo)
    canvas._on_press(_event(canvas, start))
    canvas._on_motion(_event(canvas, start + [30., 0.]))
    canvas._on_release(_event(canvas, start + [30., 0.]))
    assert not editor._busy
    assert np.array_equal(editor._session.mask, snapshot)
    assert editor._session.geometries[1] == gate and len(editor._session._undo) == history


def test_make_masks_opens_volume_without_replacing_2d_labels_or_boxes(qtbot, tmp_path):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    image = np.arange(25, dtype=np.uint16).reshape(5, 5)
    labels = np.zeros_like(image)
    screen._canvas.set_image_and_mask(image, labels)
    screen._canvas.boxes = [(3, 1., 1., 4., 4.)]
    volume = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    write_tiff(tmp_path / 'volume.tif', volume, metadata={'axes': 'ZYX'})
    dialog = screen.open_volume(str(tmp_path / 'volume.tif'))
    assert dialog is not None
    assert screen._canvas.image is image and screen._canvas.mask is labels
    assert screen._canvas.boxes == [(3, 1., 1., 4., 4.)]
    assert screen._btn_volume.objectName() == 'OpenVolumeMask'
    assert 'draw 3D masks slice by slice' in screen._btn_volume.property('apiTooltipDescriptionSource')
    assert 'draw 3D masks slice by slice' in screen._btn_volume.property('_spacr_i18n_tooltip')
    assert screen._btn_volume.accessibleDescription()
    dialog.close()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(dialog) and screen._volume_dialog is None


def test_volume_open_help_uses_footer_when_inside_an_action_scope(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.tooltip_policy import OPT_OUT_PROPERTY

    host = AppScreen('mask')
    qtbot.addWidget(host)
    screen = MakeMasksScreen(host)
    command = screen._btn_volume
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
    qtbot.waitUntil(lambda: not editor._busy, timeout=5000)
    assert np.all(editor._session.mask == 1)
    assert isinstance(editor._session.geometries[1], EditableVolumeGate)


def test_pending_edit_cancellation_destroys_dialog_without_late_gui_publication(editor, qtbot, monkeypatch):
    import threading
    started, release = threading.Event(), threading.Event()
    original = editor._session._paint_geometry
    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(editor._session, '_paint_geometry', blocked)
    editor._panel.canvas.gate_drawn.emit(_box('(unnamed)'))
    qtbot.waitUntil(started.is_set, timeout=5000)
    assert editor._busy and not editor._save.isEnabled()
    session, worker = editor._session, editor._worker
    assert editor.close()
    editor.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(editor)
    release.set()
    qtbot.waitUntil(worker.idle, timeout=5000)
    assert not session.mask.any() and not session._undo and not session.geometries


def test_unsaved_close_cancel_preserves_native_dialog_and_geometry(editor, qtbot, monkeypatch):
    _draw(editor, qtbot)
    original = editor._session.mask.copy()
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Cancel)
    assert not editor.close()
    assert isValid(editor) and not editor._closing
    assert np.array_equal(editor._session.mask, original)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Discard)


def test_saved_control_changes_reopen_without_changing_voxels(editor, qtbot):
    _draw(editor, qtbot)
    original = editor._session.mask.copy()
    editor._density.setValue(4)
    editor._anchors.setChecked(False)
    assert np.array_equal(editor._session.mask, original)
    assert editor._save_mask()
    session = engine.load_volume_image_and_mask(str(__import__('pathlib').Path(editor._session.source_path).parent), 'volume.tif')
    reopened = _VolumeMaskEditor(session)
    qtbot.addWidget(reopened)
    assert reopened._density.value() == 4 and not reopened._anchors.isChecked()
    assert np.array_equal(reopened._session.mask, original)


def test_dirty_busy_close_requires_explicit_discard_and_cancel_preserves_completed_edit(editor, qtbot, monkeypatch):
    import threading
    gate = _draw(editor, qtbot)
    session = editor._session
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
    assert not editor.close()
    assert questions and questions[-1][3] == QMessageBox.Discard | QMessageBox.Cancel
    assert isValid(editor) and editor._busy and not editor._closing and not editor._cancel.is_set()
    assert np.array_equal(session.mask, completed) and session.geometries == geometry
    assert len(session._undo) == history
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Discard)
    worker = editor._worker
    assert editor.close()
    editor.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(editor)
    release.set()
    qtbot.waitUntil(worker.idle, timeout=5000)
    assert np.array_equal(session.mask, completed) and session.geometries == geometry
    assert len(session._undo) == history
