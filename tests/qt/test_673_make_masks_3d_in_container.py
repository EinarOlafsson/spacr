"""N673/N692 follow-up (2026-10-12): 3D images open inside the Make Masks view.

The maintainer's verdict on the separate "3D mask editor" window: a volume
opens in the same container as the 2-D test data, with a z-slice slider on
the left, a Z button that makes the wheel change the slice, and a View list
left of the Mask tab offering Default, Orthogonal and True 3D. Every edit,
from any view, lands in the one label volume.

No GPU and no network: the volumes are small synthetic stacks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (MODE_BOX, MODE_BRUSH, MODE_ERASE, MODE_NONE,
                                         MODE_RECROP, MODE_ZOOM, MakeMasksScreen)
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate
from spacr.tiff_io import write_tiff

WAIT = 60000


def _settle(rounds=4):
    for _ in range(rounds):
        QApplication.processEvents()


def _volume(folder, name="volume.tif", shape=(10, 24, 32), seed=4):
    image = np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)
    tifffile.imwrite(Path(folder) / name, image)
    return Path(folder) / name


def _spaced(folder, name="spaced.tif"):
    image = np.arange(8 ** 3, dtype=np.uint16).reshape(8, 8, 8)
    write_tiff(Path(folder) / name, image,
               metadata={"axes": "ZYX", "voxel_spacing": [1., .5, .25], "unit": "um"})
    return Path(folder) / name


@pytest.fixture
def discard(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)


@pytest.fixture
def screen(qtbot, discard):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1500, 950)
    widget.show()
    qtbot.waitExposed(widget)
    _settle()
    yield widget
    stage = widget._mask_stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle(), timeout=WAIT)
    stage.dirty = False


@pytest.fixture
def stage(screen, tmp_path):
    assert screen.open_volume(str(_volume(tmp_path))) is screen._mask_stage
    _settle()
    return screen._mask_stage


@pytest.fixture
def true3d(screen, tmp_path):
    assert screen.open_volume(str(_spaced(tmp_path))) is screen._mask_stage
    stage = screen._mask_stage
    assert stage.set_view("true3d")
    _settle()
    return stage


def _wait(stage, qtbot):
    qtbot.waitUntil(lambda: not stage.busy, timeout=WAIT)
    _settle()


def _draw_box(stage, qtbot, limits=((0, 1), (0, 2), (0, 4))):
    stage._panel.canvas.gate_drawn.emit(BoxGate.from_limits("(unnamed)", ("X", "Y", "Z"), list(limits)))
    _wait(stage, qtbot)


def _shown(stage, image):
    return stage._display(image)


def _mouse(kind, position, button, buttons=None):
    return QMouseEvent(kind, QPointF(position), QPointF(position), button,
                       button if buttons is None else buttons, Qt.NoModifier)


def _wheel(canvas, up=True):
    centre = QPointF(canvas.rect().center())
    return QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, 120 if up else -120),
                       Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)


def _press_key(screen, key):
    screen._spacr_screen_holders[key].activated.emit()
    _settle(2)


def _stroke(widget, qtbot, start, end):
    qtbot.mousePress(widget, Qt.LeftButton, pos=start)
    qtbot.mouseMove(widget, end)
    qtbot.mouseRelease(widget, Qt.LeftButton, pos=end)
    _settle(2)


def test_a_volume_opens_in_the_mask_tab_and_never_in_a_window(screen, tmp_path):
    stage = screen._mask_stage
    assert not stage.view_picker.isVisible() and not stage._rail.isVisible()
    assert screen.open_volume(str(_volume(tmp_path))) is stage
    _settle()
    assert not [widget for widget in QApplication.topLevelWidgets()
                if isinstance(widget, QDialog) and widget.isVisible()]
    assert screen._view_tabs.widget(0) is stage and stage.isAncestorOf(screen._canvas)
    assert screen._view_tabs.currentIndex() == 0
    assert screen._body_stack.currentWidget() is screen._image_page
    assert stage.view_picker.isVisible() and stage._rail.isVisible() and stage._strip.isVisible()
    assert screen._canvas.image.shape == (24, 32) and stage._cursor == [16, 12, 5]
    assert stage._slider.value() == 5 and stage._slice_label.text() == "6 / 10"
    assert screen._canvas.mode == MODE_BRUSH
    assert not screen._mode_buttons[MODE_BOX].isEnabled() and not screen._btn_recrop.isEnabled()
    assert "10 slices" in screen._status_label.text()
    assert isinstance(screen._history, mm._VolumeSliceHistory)
    assert screen._objects_now() == 0 and stage._count.text() == "0 objects in the volume"


def test_the_view_list_sits_left_of_the_mask_tab_and_only_in_3d(stage, screen, tmp_path):
    tabs = screen._view_tabs
    picker = stage.view_picker
    assert tabs.cornerWidget(Qt.TopLeftCorner) is picker
    assert [picker.itemText(index) for index in range(picker.count())] == [
        "Default", "Orthogonal", "True 3D"]
    right_of_picker = picker.mapTo(screen, picker.rect().topRight()).x()
    mask_tab = tabs.tabBar().tabRect(0)
    left_of_mask = tabs.tabBar().mapTo(screen, mask_tab.topLeft()).x()
    assert right_of_picker <= left_of_mask
    assert tabs.tabText(0) == "Mask"
    assert not stage.set_view("sideways")
    picker.setCurrentIndex(1)
    assert stage.view == "orthogonal"
    tifffile.imwrite(tmp_path / "flat.tif", np.zeros((12, 16), dtype=np.uint16))
    flat = tmp_path / "flat_folder"
    flat.mkdir()
    tifffile.imwrite(flat / "field.tif", np.zeros((12, 16), dtype=np.uint16))
    assert screen._open_folder(str(flat))
    _settle()
    assert stage.session is None and not picker.isVisible() and not stage._rail.isVisible()
    assert not stage._xz.isVisible() and stage.view == "default"
    assert isinstance(screen._history, engine.MaskHistory)
    assert screen._canvas.volume_label is None and screen._canvas.crosshair is None
    assert not stage.set_view("true3d") and not stage.set_slice(3) and not stage.step_slice(1)
    assert screen._mode_buttons[MODE_BOX].isEnabled() and screen._btn_recrop.isEnabled()


def test_the_slider_and_page_keys_change_the_slice(stage, screen):
    session = stage.session
    stage._slider.setValue(2)
    assert stage._cursor[2] == 2 and stage._slice_label.text() == "3 / 10"
    assert np.array_equal(screen._canvas.image, _shown(stage, session.plane("Z", 2)[0]))
    _press_key(screen, "PgDown")
    assert stage._cursor[2] == 3 and stage._slider.value() == 3
    _press_key(screen, "PgUp")
    _press_key(screen, "PgUp")
    assert stage._cursor[2] == 1
    assert screen._spacr_screen_holders["PgUp"].key() == QKeySequence("PgUp")
    assert stage.set_slice(99) and stage._cursor[2] == 9
    assert not stage.set_slice(9) and not stage.step_slice(1)
    assert stage.set_slice(-4) and stage._cursor[2] == 0


def test_the_z_button_switches_the_wheel_between_slice_and_zoom(stage, screen):
    canvas = screen._canvas
    assert not stage._z_wheel.isChecked() and canvas.slice_wheel is None
    canvas.wheelEvent(_wheel(canvas))
    assert canvas.is_zoomed() and stage._cursor[2] == 5
    canvas.reset_zoom()
    stage._z_wheel.click()
    assert stage._z_wheel.isChecked() and canvas.slice_wheel is not None
    assert "QPushButton#MakeMasksSliceWheel:checked" in mm._make_masks_qss(
        mm.active_palette(), 1.0)
    canvas.wheelEvent(_wheel(canvas, up=False))
    assert stage._cursor[2] == 6 and not canvas.is_zoomed()
    canvas.wheelEvent(_wheel(canvas, up=True))
    canvas.wheelEvent(_wheel(canvas, up=True))
    assert stage._cursor[2] == 4
    canvas._stroke_in_progress = True
    canvas.wheelEvent(_wheel(canvas, up=True))
    assert stage._cursor[2] == 4
    canvas._stroke_in_progress = False
    stage._z_wheel.click()
    assert canvas.slice_wheel is None
    stage.leave()
    stage._z_wheel.click()
    assert canvas.slice_wheel is None
    stage._z_wheel.click()


def test_brush_strokes_on_a_slice_draw_one_object_and_undo_through_the_volume(stage, screen, qtbot):
    canvas = screen._canvas
    session = stage.session
    centre = canvas.rect().center()
    _stroke(canvas, qtbot, centre - QPoint(30, 0), centre + QPoint(30, 0))
    assert session.mask[5].any() and not session.mask[:5].any() and not session.mask[6:].any()
    assert set(np.unique(session.mask)) == {0, 1}
    assert stage._selected == 1 and stage._labels.currentData() == 1 and stage.dirty
    assert screen._act_undo.isEnabled() and screen._objects_now() == 1
    assert stage._count.text().startswith("Object 1:")
    stage.set_slice(7)
    qtbot.mousePress(canvas, Qt.LeftButton, Qt.ShiftModifier, centre - QPoint(30, 0))
    qtbot.mouseMove(canvas, centre + QPoint(30, 0))
    qtbot.mouseRelease(canvas, Qt.LeftButton, Qt.ShiftModifier, centre + QPoint(30, 0))
    _settle(2)
    assert np.unique(session.mask[7]).tolist() == [0, 1], "Shift keeps the chosen object"
    stage._labels.setCurrentIndex(stage._labels.findData(0))
    assert stage.draw_label() == 2
    _stroke(canvas, qtbot, canvas._image_to_canvas(4.5, 21.5),
            canvas._image_to_canvas(10.5, 21.5))
    assert 2 in session.mask[7] and stage._selected == 2, (
        "a stroke clear of object 1 is a new object")
    ones = np.count_nonzero(session.mask[7] == 1)
    screen._set_mode(MODE_ERASE)
    _stroke(canvas, qtbot, centre - QPoint(80, 0), centre + QPoint(80, 0))
    assert np.count_nonzero(session.mask[7] == 1) < ones
    after_erase = session.mask.copy()
    screen._act_undo.trigger()
    assert not np.array_equal(session.mask, after_erase)
    screen._act_redo.trigger()
    assert np.array_equal(session.mask, after_erase)
    assert np.array_equal(canvas.mask, session.plane("Z", 7)[1])


def test_two_d_ids_new_to_a_slice_never_join_a_distant_object(stage, screen):
    session = stage.session
    session.paint_plane("Z", 0, 1, paint=np.ones((24, 32), dtype=bool))
    stage.refresh()
    canvas = screen._canvas
    edited = canvas.mask.copy()
    edited[2:6, 2:6] = 1
    canvas.mask = edited
    canvas.stroke_finished.emit()
    assert np.unique(session.mask[5]).tolist() == [0, 2]
    assert np.all(session.mask[0] == 1)


def test_undo_and_redo_with_nothing_or_while_busy_change_nothing(stage, screen):
    assert stage.step_history(False) is None and stage.step_history(True) is None
    assert not stage.dirty
    stage.session.paint_plane("Z", 5, 1, paint=np.ones((24, 32), dtype=bool))
    stage._set_busy(True)
    assert stage.step_history(False) is None and stage.session.mask.any()
    assert not stage._strip.isEnabled() and not stage.view_picker.isEnabled()
    stage._set_busy(False)
    assert stage.step_history(False) is not None and not stage.session.mask.any()


def test_orthogonal_panes_match_the_slices_through_the_crosshair(stage, screen, qtbot):
    session = stage.session
    session.paint_plane("Z", 5, 3, paint=np.ones((24, 32), dtype=bool))
    stage.refresh()
    stage.view_picker.setCurrentIndex(stage.view_picker.findData("orthogonal"))
    _settle()
    x, y, z = stage._cursor
    assert stage._xz.isVisible() and stage._yz.isVisible()
    image, labels = session.plane("Y", y)
    assert np.array_equal(stage._xz.image, _shown(stage, image)) and np.array_equal(stage._xz.mask, labels)
    image, labels = session.plane("X", x)
    assert np.array_equal(stage._yz.image, _shown(stage, image).T)
    assert np.array_equal(stage._yz.mask, labels.T)
    assert stage._xz.image.shape == (10, 32) and stage._yz.image.shape == (24, 10)
    assert screen._canvas.crosshair == (x, y)
    assert stage._xz.crosshair == (x, z) and stage._yz.crosshair == (z, y)
    stage._xz.grab()
    canvas = screen._canvas
    corner = canvas._image_to_canvas(3.5, 20.5)
    assert stage.eventFilter(canvas, _mouse(QEvent.MouseButtonPress, corner, Qt.MiddleButton))
    assert stage._cursor[:2] == [3, 20]
    assert np.array_equal(stage._yz.mask, session.plane("X", 3)[1].T)
    assert stage.eventFilter(canvas, _mouse(QEvent.MouseButtonRelease, corner, Qt.MiddleButton,
                                            Qt.NoButton))
    spot = stage._xz._image_to_canvas(10.5, 2.5)
    assert stage.eventFilter(stage._xz, _mouse(QEvent.MouseMove, spot, Qt.NoButton, Qt.MiddleButton))
    assert stage._cursor == [10, 20, 2] and stage._slider.value() == 2
    spot = stage._yz._image_to_canvas(7.5, 4.5)
    stage.eventFilter(stage._yz, _mouse(QEvent.MouseButtonPress, spot, Qt.MiddleButton))
    assert stage._cursor == [10, 4, 7]
    assert stage.eventFilter(canvas, _mouse(QEvent.MouseButtonPress, QPointF(-9, -9), Qt.MiddleButton))
    assert stage._cursor == [10, 4, 7]
    assert not stage.eventFilter(canvas, QEvent(QEvent.Enter))
    stage.view_picker.setCurrentIndex(0)
    assert not stage._xz.isVisible() and canvas.crosshair is None


def test_side_views_draw_with_the_brush_and_follow_the_main_tool(stage, screen, qtbot):
    session = stage.session
    stage.set_view("orthogonal")
    _settle()
    x, y, z = stage._cursor
    centre = stage._xz.rect().center()
    _stroke(stage._xz, qtbot, centre - QPoint(15, 0), centre + QPoint(15, 0))
    assert stage._xz.mode == MODE_BRUSH and stage._xz.volume_label is not None
    section = session.mask[:, y, :]
    assert section.any() and np.count_nonzero(session.mask) == np.count_nonzero(section)
    assert stage._selected == 1 and stage.dirty
    before = session.mask.copy()
    centre = stage._yz.rect().center()
    _stroke(stage._yz, qtbot, centre - QPoint(0, 10), centre + QPoint(0, 10))
    changed = session.mask != before
    assert changed.any() and np.all(np.argwhere(changed)[:, 2] == x)
    screen._set_mode(MODE_ZOOM)
    before = session.mask.copy()
    _stroke(stage._xz, qtbot, centre, centre + QPoint(10, 0))
    assert stage._xz.mode == MODE_NONE and np.array_equal(session.mask, before)
    assert stage.commit_plane("X", np.zeros((3, 3), dtype=np.uint16)) == 0
    assert "does not match" in screen._status_label.text()


def test_the_view_list_appears_only_while_a_volume_is_open(screen, tmp_path):
    stage = screen._mask_stage
    flat = tmp_path / "flat"
    flat.mkdir()
    tifffile.imwrite(flat / "field.tif", np.zeros((12, 16), dtype=np.uint16))
    _volume(flat, "zstack.tif")
    assert screen._open_folder(str(flat))
    _settle()
    assert not stage.view_picker.isVisible()
    screen._on_next()
    _settle()
    assert stage.session is not None and stage.view_picker.isVisible()
    assert Path(stage.session.source_path).name == "zstack.tif"
    screen._on_prev()
    _settle()
    assert stage.session is None and not stage.view_picker.isVisible()
    assert screen._canvas.image.shape == (12, 16)


def test_a_zstack_in_a_queue_asks_before_unsaved_edits_are_left(screen, tmp_path, monkeypatch):
    folder = tmp_path / "queue"
    folder.mkdir()
    _volume(folder, "a.tif")
    _volume(folder, "b.tif", seed=9)
    assert screen._open_folder(str(folder))
    _settle()
    stage = screen._mask_stage
    assert Path(stage.session.source_path).name == "a.tif"
    stage.session.paint_plane("Z", 5, 1, paint=np.ones((24, 32), dtype=bool))
    stage.dirty = True
    questions = []

    def answer(choice):
        def ask(*args, **kwargs):
            questions.append(args)
            return choice
        return ask

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.Cancel))
    screen._on_next()
    assert Path(stage.session.source_path).name == "a.tif" and stage.dirty
    assert questions[-1][3] & QMessageBox.Save
    assert not screen._open_folder(str(folder))
    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.Save))
    screen._on_next()
    assert Path(stage.session.source_path).name == "b.tif"
    assert np.all(tifffile.imread(folder / "masks" / "a.tif")[5] == 1)
    stage.session.paint_plane("Z", 1, 4, paint=np.ones((24, 32), dtype=bool))
    stage.dirty = True
    screen._btn_save_on_navigation.setChecked(True)
    screen._on_prev()
    assert Path(stage.session.source_path).name == "a.tif"
    assert np.all(tifffile.imread(folder / "masks" / "b.tif")[1] == 4)
    assert stage.session.mask[5].all()
    screen._btn_save_on_navigation.setChecked(False)


def test_a_running_3d_edit_can_only_be_discarded_or_waited_for(stage, screen, monkeypatch):
    stage.dirty = True
    stage.busy = True
    asked = []

    def ask(*args, **kwargs):
        asked.append(args)
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", ask)
    screen._btn_save_on_navigation.setChecked(True)
    assert not screen._volume_leave_ok()
    assert asked[-1][3] == QMessageBox.Discard | QMessageBox.Cancel
    assert "still running" in asked[-1][2]
    assert stage.save() is False
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    assert screen._volume_leave_ok() and not stage.dirty
    stage.busy = False
    screen._btn_save_on_navigation.setChecked(False)


def test_an_edited_volume_that_will_not_be_left_keeps_the_screen_open(stage, screen, tmp_path, monkeypatch):
    stage.session.paint_plane("Z", 0, 1, paint=np.ones((24, 32), dtype=bool))
    stage.dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Cancel)
    other = _volume(tmp_path, "other.tif")
    assert screen.open_volume(str(other)) is None
    assert Path(stage.session.source_path).name == "volume.tif"
    assert screen.close() is False
    assert stage.session is not None
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)


def test_open_volume_respects_unsaved_boxes_and_the_open_folders_masks(screen, tmp_path, monkeypatch):
    path = _volume(tmp_path)
    monkeypatch.setattr(screen, "_save_boxes_if_needed", lambda: False)
    assert screen.open_volume(str(path)) is None and screen._mask_stage.session is None
    monkeypatch.undo()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    sibling = tmp_path / "labels"
    sibling.mkdir()
    screen._folder, screen._masks_dir = str(tmp_path), str(sibling)
    assert screen.open_volume(str(path)) is screen._mask_stage
    assert Path(screen._mask_stage.session.mask_path).parent == sibling
    assert screen._image_files == ["volume.tif"] and screen._src_label.text() == str(path)


def test_a_volume_that_cannot_load_from_the_queue_is_reported(screen, tmp_path, monkeypatch):
    folder = tmp_path / "queue"
    folder.mkdir()
    _volume(folder, "a.tif")
    (folder / "masks").mkdir()
    tifffile.imwrite(folder / "masks" / "a.tif", np.zeros((2, 4, 5), dtype=np.uint16))
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append(text))
    assert screen._open_folder(str(folder))
    assert screen._mask_stage.session is None
    assert warned and "does not match this volume" in warned[-1]


def test_save_keeps_shape_axes_and_voxel_size_and_ctrl_s_saves_the_volume(screen, tmp_path, qtbot):
    path = _spaced(tmp_path)
    assert screen.open_volume(str(path))
    stage = screen._mask_stage
    canvas = screen._canvas
    centre = canvas.rect().center()
    _stroke(canvas, qtbot, centre - QPoint(40, 0), centre + QPoint(40, 0))
    drawn = stage.session.mask.copy()
    assert drawn.any()
    screen._act_undo.trigger()
    assert not stage.session.mask.any()
    screen._act_redo.trigger()
    assert np.array_equal(stage.session.mask, drawn)
    screen._btn_save.click()
    assert not stage.dirty and "Saved 3D mask" in screen._status_label.text()
    reopened = engine.load_volume_image_and_mask(str(tmp_path), path.name)
    assert reopened.mask.shape == (8, 8, 8) and reopened.axes == "ZYX"
    assert reopened.spacing == (1., .5, .25) and reopened.unit == "um"
    assert np.array_equal(reopened.mask, drawn)
    stage.session.paint_plane("Z", 0, 9, paint=np.ones((8, 8), dtype=bool))
    stage.dirty = True
    _press_key(screen, "Ctrl+S")
    assert not stage.dirty
    tifffile.imwrite(path, np.zeros((8, 8, 8), dtype=np.uint16))
    stage.dirty = True
    assert not screen._on_save() and stage.dirty
    assert "source volume changed" in screen._status_label.text()


def test_box_and_recrop_stay_off_while_a_volume_is_open(stage, screen):
    screen._set_mode(MODE_BOX)
    screen._set_mode(MODE_RECROP)
    assert screen._canvas.mode == MODE_BRUSH


def test_interpolate_assign_and_delete_work_on_the_whole_volume(stage, screen, monkeypatch):
    session = stage.session
    stage._interpolate_button.click()
    assert "at least two slices" in screen._status_label.text()
    stage._delete_button.click()
    assert "Object ID" in screen._status_label.text()
    for index in (1, 6):
        session.paint_plane("Z", index, 1, paint=np.pad(np.ones((4, 4), dtype=bool), ((10, 10), (14, 14))))
    stage.refresh()
    stage._labels.setCurrentIndex(stage._labels.findData(1))
    stage._interpolate_button.click()
    assert all(np.count_nonzero(session.mask[index] == 1) == 16 for index in range(1, 7))
    assert "Interpolated" in screen._status_label.text() and stage.dirty
    session.paint_plane("Z", 9, 1, paint=np.pad(np.ones((2, 2), dtype=bool), ((0, 22), (0, 30))))
    stage._assign_button.click()
    assert set(np.unique(session.mask)) == {0, 1, 2}
    stage._labels.setCurrentIndex(stage._labels.findData(2))
    stage._delete_button.click()
    assert set(np.unique(session.mask)) == {0, 1} and stage._selected == 0
    assert "Deleted object 2" in screen._status_label.text()
    monkeypatch.setattr(session, "assign_objects", lambda: (_ for _ in ()).throw(ValueError("full")))
    stage._assign_button.click()
    assert screen._status_label.text() == "full"


def test_the_object_list_offers_the_first_unused_id(stage):
    session = stage.session
    session.paint_plane("Z", 0, 2, paint=np.ones((24, 32), dtype=bool))
    stage.refresh()
    assert stage.draw_label() == 1
    session.paint_plane("Z", 1, 1, paint=np.ones((24, 32), dtype=bool))
    stage.refresh()
    assert stage.draw_label() == 3
    stage._selected = 7
    stage.refresh()
    assert stage._selected == 0


def test_true_3d_draws_reshapes_and_deletes_objects_into_voxels(true3d, screen, qtbot):
    stage = true3d
    panel = stage._panel
    assert stage._pages.currentWidget() is stage._true3d and panel.isVisible()
    assert panel.canvas._mode == "3D" and panel.tree_section.isHidden()
    assert stage.findChild(QDialog) is None
    _draw_box(stage, qtbot)
    expected = np.zeros((8, 8, 8), dtype=np.uint16)
    expected[:4, :4, :4] = 1
    assert np.array_equal(stage.session.mask, expected)
    assert stage._selected == 1 and "64 voxels" in stage._count.text()
    assert "Updated 64 voxels" in screen._status_label.text()
    # 2026-10-15: the object is drawn by the layer rendering, not a point cloud.
    qtbot.waitUntil(stage.render_idle, timeout=WAIT)
    _settle()
    assert stage._render_state.picked.max() == 1
    axes = panel.canvas._axes[(0, 0)]
    assert len(axes.collections) >= 2
    stage.set_view("default")
    stage.set_slice(0)
    assert screen._canvas.mask[:4, :4].all() and not screen._canvas.mask[4:].any()
    stage.set_view("true3d")
    _settle()
    panel.canvas.gate_edited.emit(EditableVolumeGate.from_gate(
        BoxGate.from_limits("1", ("X", "Y", "Z"), [(0, 2), (0, 4), (0, 8)])))
    _wait(stage, qtbot)
    assert np.all(stage.session.mask == 1)
    stage._delete_button.click()
    assert not stage.session.mask.any() and stage._selected == 0
    screen._act_undo.trigger()
    assert np.all(stage.session.mask == 1)
    screen._act_undo.trigger()
    assert np.array_equal(stage.session.mask, expected)


def test_true_3d_keys_mirror_the_gate_editor(true3d, screen, monkeypatch):
    stage = true3d
    panel = stage._panel
    for key, mode in (("Shift+E", "edit"), ("Shift+D", "draw"), ("Shift+S", "spin")):
        assert screen._spacr_screen_holders[key].key() == QKeySequence(key)
        _press_key(screen, key)
        assert panel.drag_mode() == mode
    assert panel._spin_buttons[""].isChecked()
    _press_key(screen, "Shift+Z")
    assert panel._spin_buttons["z"].isChecked() and panel.canvas._spin_axis == "z"
    monkeypatch.setattr(mm.time, "monotonic", lambda: 1e9)
    _press_key(screen, "Shift+X")
    assert panel.anchor_axis() == "x" and panel.drag_mode() == "draw"
    stage.busy = True
    assert not stage.drag_key("edit") and not stage.axis_key("y")
    stage.busy = False
    stage.set_view("default")
    assert not stage.drag_key("edit") and not stage.axis_key("y")


@pytest.mark.parametrize("case", ["repeat", "label", "version", "session", "return"])
def test_selected_label_mesh_requests_deduplicate_and_keep_newest(qtbot, case):
    import threading
    from types import SimpleNamespace

    started, release = threading.Event(), threading.Event()
    calls = []
    first = SimpleNamespace(geometries={}, geometry_for_label=lambda label: label)
    other = SimpleNamespace(geometries={}, geometry_for_label=lambda label: label)
    panel = SimpleNamespace(canvas=SimpleNamespace(_surface_gate=None, _selected_surfaces=set()),
                            set_gates=lambda gates: None)
    stage = SimpleNamespace(session=first, _selected=1, _version=1, _meshes={},
                            _panel=panel, _say=lambda message: None, _restoring=False)

    def work(request):
        calls.append((id(request[0]), request[1], request[2]))
        started.set()
        assert release.wait(5), "test-owned blocked mesh was not released"
        return mm._volume_mesh(request)

    worker = mm._NewestRequestWorker(work, lambda *args: None, name="test-selected-mesh")
    stage._mesher = worker
    select = lambda: mm._VolumeStage._select_label(stage)
    try:
        select()
        assert started.wait(5)
        select()
        select()
        expected = [(id(first), 1, 1)]
        if case in ("label", "return"):
            stage._selected = 2
            select()
            stage._selected = 3 if case == "label" else 1
            select()
            if case == "label":
                expected.append((id(first), 3, 1))
        elif case == "version":
            stage._version = 2
            select()
            expected.append((id(first), 1, 2))
        elif case == "session":
            stage.session = other
            select()
            expected.append((id(other), 1, 1))
        release.set()
        qtbot.waitUntil(worker.idle, timeout=5000)
        assert calls == expected
    finally:
        release.set()
        assert worker.close(5)


def test_a_large_object_gets_its_anchors_off_the_gui_thread(true3d, screen, qtbot):
    stage = true3d
    session = stage.session
    session.paint_plane("Z", 2, 1, paint=np.pad(np.ones((3, 3), dtype=bool), ((1, 4), (1, 4))))
    stage.refresh()
    stage._labels.setCurrentIndex(stage._labels.findData(1))
    assert "Preparing the anchors of object 1" in screen._status_label.text()
    qtbot.waitUntil(lambda: 1 in stage._meshes, timeout=WAIT)
    _settle()
    assert "1" in stage._panel.gates.names
    assert "drag its anchors" in screen._status_label.text()
    assert stage._panel.drag_mode() == "edit"


def test_an_object_without_a_boundary_says_why_and_is_not_asked_twice(true3d, screen, qtbot):
    stage = true3d
    session = stage.session
    corner = np.zeros((8, 8), dtype=bool)
    corner[0, 0] = corner[1, 1] = True
    session.paint_plane("Z", 2, 1, paint=corner)
    stage.refresh()
    stage._labels.setCurrentIndex(stage._labels.findData(1))
    qtbot.waitUntil(lambda: 1 in stage._meshes, timeout=WAIT)
    _settle()
    assert stage._meshes[1] is None and "1" not in stage._panel.gates.names
    message = screen._status_label.text()
    submitted = []
    stage._mesher.submit = lambda request: submitted.append(request)
    stage._select_label()
    assert not submitted and screen._status_label.text() == message


def test_late_or_stale_results_are_not_published(true3d, screen):
    stage = true3d
    message = screen._status_label.text()
    stage._take_mesh(((object(), 1, stage._version), None, None))
    stage._take_mesh(((stage.session, 1, stage._version - 1), None, None))
    stage._take_edit(((object(), 1, None, stage._cancel), 64, None))
    assert screen._status_label.text() == message and not stage._meshes
    stage._take_mesh(((stage.session, 5, stage._version), None, ValueError("no boundary")))
    assert screen._status_label.text() == "no boundary" and stage._meshes == {5: None}
    stage._selected = 5
    stage.busy = True
    stage._take_mesh(((stage.session, 5, stage._version), None, None))
    stage.busy = False


def test_anchor_controls_change_only_the_handles(true3d, qtbot):
    stage = true3d
    _draw_box(stage, qtbot)
    snapshot = stage.session.mask.copy()
    history = len(stage.session._undo)
    stage._density.setValue(4)
    stage._anchors.setChecked(False)
    assert stage.session._editor_controls == {"anchor_density": 4, "show_anchors": False}
    assert np.array_equal(stage.session.mask, snapshot) and len(stage.session._undo) == history
    stage.refresh()
    assert stage._density.value() == 4 and not stage._anchors.isChecked()


def test_shapes_that_cannot_be_voxelized_or_did_not_change_queue_nothing(true3d, screen, qtbot):
    stage = true3d
    stage._geometry_changed()
    assert not stage.busy
    stage._panel.canvas.gate_drawn.emit(
        BoxGate.from_limits("(unnamed)", ("X", "Y", "Intensity"), [(0, 1), (0, 2), (0, 100)]))
    _wait(stage, qtbot)
    assert "valid bounded spatial volume" in screen._status_label.text()
    assert not stage.session.mask.any()
    stage._panel.canvas.gate_drawn.emit(
        BoxGate.from_limits("(unnamed)", ("Z", "Y", "X"), [(0, 4), (0, 2), (0, 1)]))
    _wait(stage, qtbot)
    expected = np.zeros((8, 8, 8), dtype=np.uint16)
    expected[:4, :4, :4] = 1
    assert np.array_equal(stage.session.mask, expected)
    history = len(stage.session._undo)
    stage._geometry_changed()
    assert not stage.busy and len(stage.session._undo) == history
    stage._restoring = True
    stage._geometry_changed()
    stage._restoring = False
    assert not stage.busy


def test_the_volume_layers_are_drawn_under_the_3d_view(true3d, qtbot):
    """2026-10-15: the sampled object cloud is replaced by the napari-style
    volume rendering (test_673_napari_true3d.py covers it in full); the
    wrapper still passes a refused render through untouched."""
    stage = true3d
    _draw_box(stage, qtbot)
    assert stage._with_layers(lambda: False)() is False
    assert stage._with_layers(lambda: True)() is True
    qtbot.waitUntil(stage.render_idle, timeout=WAIT)
    _settle()
    assert stage._render_state is not None
    assert stage._render_state.picked.max() == 1


def test_leaving_a_volume_mid_edit_cancels_it_without_a_late_result(true3d, screen, qtbot, monkeypatch):
    import threading

    stage = true3d
    session = stage.session
    started, release = threading.Event(), threading.Event()
    original = session._paint_geometry

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(session, "_paint_geometry", blocked)
    stage._panel.canvas.gate_drawn.emit(BoxGate.from_limits("(unnamed)", ("X", "Y", "Z"),
                                                            [(0, 1), (0, 2), (0, 4)]))
    qtbot.waitUntil(started.is_set, timeout=WAIT)
    assert stage.busy and not stage._true3d.isEnabled()
    cancel = stage._cancel
    stage.leave()
    assert cancel.is_set() and not stage.busy and stage.session is None
    release.set()
    qtbot.waitUntil(stage._worker.idle, timeout=WAIT)
    _settle()
    assert not session.mask.any() and not session._undo and stage.session is None


def test_the_crosshair_needs_a_picture_to_be_drawn_on(qtbot):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.crosshair = (3, 4)
    canvas.grab()
    canvas.set_image_and_mask(np.zeros((10, 10), dtype=np.uint16), np.zeros((10, 10), dtype=np.uint16))
    canvas.grab()
    assert canvas.crosshair == (3, 4)


def test_replace_plane_keeps_ids_and_refuses_bad_sections():
    session = engine.VolumeMaskSession(np.zeros((3, 4, 5), dtype=np.uint8),
                                       np.zeros((3, 4, 5), dtype=np.uint8))
    session.mask[0, 0, 0] = 1
    session.mask[0, 0, 1] = 2
    edited = np.zeros((4, 5), dtype=np.uint16)
    edited[1, :2] = 1
    edited[2, :2] = 2
    edited[3, :2] = 7
    edited[3, 3:] = 3
    assert session.replace_plane("Z", 2, edited, keep=(2,)) == 8
    assert session.mask[2, 1, 0] == 4 and session.mask[2, 2, 0] == 2
    assert session.mask[2, 3, 0] == 7 and session.mask[2, 3, 4] == 3
    for bad in (np.zeros((2, 2), dtype=np.uint16), np.zeros((4, 5), dtype=float),
                np.full((4, 5), 300, dtype=np.uint16), np.full((4, 5), -1, dtype=np.int16)):
        with pytest.raises(ValueError):
            session.replace_plane("Z", 1, bad)
    full = engine.VolumeMaskSession(np.zeros((2, 1, 128), dtype=np.uint8),
                                    np.arange(256, dtype=np.uint8).reshape(2, 1, 128))
    with pytest.raises(ValueError):
        full.replace_plane("Z", 0, np.full((1, 128), 200, dtype=np.uint8))
    assert session.replace_plane("Z", 1, np.full((4, 5), 9, dtype=np.uint16)) == 20
    assert session.delete_label(2) == 3 and 2 not in session.mask
    assert session.undo() and session.mask[2, 2, 0] == 2


def test_volume_files_are_recognised_from_their_headers(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    _volume(tmp_path, "stack.tif")
    tifffile.imwrite(tmp_path / "flat.tif", np.zeros((5, 6), dtype=np.uint8))
    (tmp_path / "broken.tif").write_bytes(b"not a tiff")
    with mrcfile.new(tmp_path / "volume.mrc") as handle:
        handle.set_data(np.zeros((3, 4, 5), dtype=np.float32))
    with mrcfile.new(tmp_path / "section.mrc") as handle:
        handle.set_data(np.zeros((4, 5), dtype=np.float32))
    assert engine._is_volume_file(tmp_path / "stack.tif")
    assert engine._is_volume_file(tmp_path / "volume.mrc")
    for name in ("flat.tif", "broken.tif", "section.mrc", "field.png"):
        assert not engine._is_volume_file(tmp_path / name)
