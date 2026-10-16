"""Manual labels and serial wheel traversal of generic synthetic frames."""
from __future__ import annotations

import threading
from pathlib import Path
from uuid import UUID

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QDialog, QLabel, QWidget

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "masks").mkdir()
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[5:15, 5:15] = 7
    mask[25:35, 25:35] = 11
    for index in range(4):
        imageio.imwrite(folder / f"frame_{index}.tif", np.full(mask.shape, index + 1, np.uint8))
        imageio.imwrite(folder / "masks" / f"frame_{index}.tif", mask)
    widget = mm.MakeMasksScreen()
    monkeypatch.setattr(widget, "_confirm", lambda *args: True)
    qtbot.addWidget(widget)
    widget._set_mode(mm.MODE_EDIT_LABEL)
    assert widget._open_folder(str(folder))
    widget._canvas.resize(400, 400)
    widget._canvas.refresh()
    widget._btn_save_on_navigation.setChecked(False)
    yield widget
    widget.close()


def wheel(screen, steps, point=(10, 10)):
    canvas = screen._canvas
    pos = QPointF(canvas._image_to_canvas(point[0] + .5, point[1] + .5))
    event = QWheelEvent(pos, pos, QPoint(), QPoint(0, -120 * steps),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    canvas.wheelEvent(event)


def wait_idle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen._loading and not screen._label_time_queue
                    and screen._label_time_pending is None, timeout=10000)


def enable_time(screen):
    screen._btn_label_time.setChecked(True)
    assert screen._time_label_enabled
    assert screen._canvas.mode == mm.MODE_LABEL_TIME


def test_label_controls_edit_zero_collision_and_undo(screen, monkeypatch):
    assert screen._canvas.selected_label == 1
    menu_row = screen._label_value.parentWidget().layout()
    assert menu_row.itemAt(0).widget() is screen._label_value
    assert isinstance(menu_row.itemAt(1).widget(), QLabel)
    screen._step_selected_label(1)
    assert screen._label_value.text() == "2"
    screen._step_selected_label(-1)
    before = screen._canvas.mask.copy()
    monkeypatch.setattr(mm.QInputDialog, "getText", lambda *args, **kw: ("321", True))
    screen._edit_label_at(10, 10, 7)
    assert int(screen._canvas.mask[10, 10]) == 321
    assert screen._canvas.manual_ids
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    screen._on_redo()
    assert int(screen._canvas.mask[10, 10]) == 321
    assert not screen._canvas.assign_label_at((10, 10), 11)
    assert int(screen._canvas.mask[10, 10]) == 321
    assert screen._canvas.assign_label_at((10, 10), 0)
    assert int(screen._canvas.mask[10, 10]) == 0
    screen._on_undo()
    assert int(screen._canvas.mask[10, 10]) == 321


def test_hover_and_wheel_visit_both_endpoints_and_save(qtbot, screen):
    enable_time(screen)
    screen._label_value.setText("321")
    canvas = screen._canvas
    pos = QPointF(canvas._image_to_canvas(10.5, 10.5))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, pos,
                                      Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    assert int(canvas.mask[10, 10]) == 321
    wheel(screen, 3)
    wait_idle(qtbot, screen)
    assert screen._current_index == 3
    assert int(canvas.mask[10, 10]) == 321
    screen._label_value.setText("654")
    wheel(screen, -3)
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    assert int(canvas.mask[10, 10]) == 654
    wheel(screen, -1)
    wait_idle(qtbot, screen)
    for filename in screen._image_files:
        saved = imageio.imread(engine.mask_save_path(screen._folder, filename))
        assert int(saved[10, 10]) == 654
        assert int(saved[30, 30]) == 11
    assert not screen._btn_save_on_navigation.isChecked()


def test_delayed_steps_keep_order_and_identity_snapshots(qtbot, screen, monkeypatch):
    real_load = engine.load_image_and_mask
    started, release = threading.Event(), threading.Event()
    visited = []

    def delayed(folder, filename, **kwargs):
        visited.append(filename)
        if filename == "frame_1.tif":
            started.set()
            assert release.wait(5)
        return real_load(folder, filename, **kwargs)

    monkeypatch.setattr(engine, "load_image_and_mask", delayed)
    enable_time(screen)
    screen._label_value.setText("321")
    try:
        wheel(screen, 1)
        qtbot.waitUntil(started.is_set, timeout=2000)
        assert screen._current_index == 0
        screen._label_value.setText("654")
        wheel(screen, 2)
        screen._on_undo()
        assert int(screen._canvas.mask[10, 10]) == 321
    finally:
        release.set()
    wait_idle(qtbot, screen)
    assert visited == ["frame_1.tif", "frame_2.tif", "frame_3.tif"]
    assert screen._current_index == 3
    assert int(screen._canvas.mask[10, 10]) == 654
    first = imageio.imread(engine.mask_save_path(screen._folder, "frame_0.tif"))
    assert int(first[10, 10]) == 321


def test_load_failure_retains_previous_frame_and_saved_edit(qtbot, screen, monkeypatch):
    before_image = screen._canvas.image.copy()
    enable_time(screen)
    screen._label_value.setText("321")

    def broken(*args, **kwargs):
        raise OSError("synthetic unreadable frame")

    monkeypatch.setattr(engine, "load_image_and_mask", broken)
    wheel(screen, 3)
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    np.testing.assert_array_equal(screen._canvas.image, before_image)
    assert int(screen._canvas.mask[10, 10]) == 321
    assert "synthetic unreadable frame" in screen._status_label.text()
    assert not screen._canvas.label_time_busy


def test_tool_change_cancels_inflight_frame_without_losing_edits(qtbot, screen, monkeypatch):
    real_load = engine.load_image_and_mask
    started, release = threading.Event(), threading.Event()

    def delayed(folder, filename, **kwargs):
        started.set()
        assert release.wait(5)
        return real_load(folder, filename, **kwargs)

    monkeypatch.setattr(engine, "load_image_and_mask", delayed)
    enable_time(screen)
    screen._label_value.setText("321")
    try:
        wheel(screen, 3)
        qtbot.waitUntil(started.is_set, timeout=2000)
        screen._set_mode(mm.MODE_BRUSH)
    finally:
        release.set()
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    assert int(screen._canvas.mask[10, 10]) == 321
    assert screen._canvas.mode == mm.MODE_BRUSH
    assert screen._mode_buttons[mm.MODE_BRUSH].isEnabled()


def test_save_failure_refuses_navigation_and_keeps_edits(screen, monkeypatch):
    enable_time(screen)
    screen._label_value.setText("321")
    with monkeypatch.context() as patch:
        patch.setattr(screen, "_save_mask", lambda: None)
        wheel(screen, 2)
        assert screen._current_index == 0
        assert int(screen._canvas.mask[10, 10]) == 321
        assert screen._label_time_edited
        assert not screen._label_time_queue
        assert screen._load_worker is None


def test_dialog_result_cannot_edit_a_different_frame(screen, monkeypatch):
    def navigate(*args, **kwargs):
        screen._on_next()
        return "321", True

    monkeypatch.setattr(mm.QInputDialog, "getText", navigate)
    screen._edit_label_at(10, 10, 7)
    assert screen._current_index == 1
    assert int(screen._canvas.mask[10, 10]) == 7


def test_ordinary_wheel_keeps_zoom_and_does_not_navigate(screen):
    screen._set_mode(mm.MODE_BRUSH)
    assert not screen._canvas.is_zoomed()
    wheel(screen, -1)
    assert screen._current_index == 0
    assert screen._canvas.is_zoomed()


def test_time_confirmation_cancel_blue_state_and_off_gates(screen, monkeypatch):
    path = engine.mask_save_path(screen._folder, screen._image_files[0])
    before = open(path, "rb").read()
    prompts = []
    monkeypatch.setattr(screen, "_confirm", lambda title, text: prompts.append((title, text)) or False)
    screen._btn_label_time.click()
    assert not screen._time_label_enabled
    assert not screen._btn_label_time.isChecked()
    assert not screen._lineage_controls.isEnabled()
    assert "32-bit TIFF" in prompts[0][1]
    screen._add_lineage_cell()
    assert screen._lineage is None
    monkeypatch.setattr(screen, "_confirm", lambda *args: True)
    screen._btn_label_time.click()
    assert screen._btn_label_time.isChecked()
    assert screen._lineage_controls.isEnabled()
    qss = mm._make_masks_qss(mm.active_palette(), 1)
    assert "QPushButton#MakeMasksLabelTime:checked" in qss
    assert mm.active_palette()["button_accent"] in qss
    assert open(path, "rb").read() == before
    screen._btn_label_time.click()
    assert screen._canvas.mode == mm.MODE_EDIT_LABEL
    assert not screen._lineage_controls.isEnabled()
    screen._add_lineage_cell()
    screen._divide_lineage_cell()
    assert screen._lineage is None
    wheel(screen, -1)
    assert screen._current_index == 0 and screen._canvas.is_zoomed()


def test_time_off_during_load_discards_future_steps(qtbot, screen, monkeypatch):
    real_load = engine.load_image_and_mask
    started, release = threading.Event(), threading.Event()

    def delayed(folder, filename, **kwargs):
        started.set()
        assert release.wait(5)
        return real_load(folder, filename, **kwargs)

    monkeypatch.setattr(engine, "load_image_and_mask", delayed)
    enable_time(screen)
    try:
        wheel(screen, 3)
        qtbot.waitUntil(started.is_set, timeout=2000)
        assert screen._btn_label_time.isEnabled()
        screen._btn_label_time.click()
    finally:
        release.set()
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    assert screen._canvas.mode == mm.MODE_EDIT_LABEL
    assert screen._lineage["cells"]
    assert not screen._canvas.label_time_busy


def test_uint32_label_and_embedded_identity_reopen_without_sidecar(qtbot, screen):
    top = engine.MAX_MASK_LABEL
    screen._label_value.setText(str(top))
    screen._step_selected_label(1)
    assert screen._canvas.selected_label == top
    assert screen._canvas.assign_label_at((10, 10), top)
    path = screen._save_mask()
    assert imageio.imread(path).dtype == np.uint32
    for sidecar in Path(screen._folder).rglob("*.json"):
        sidecar.unlink()
    fresh = mm.MakeMasksScreen()
    qtbot.addWidget(fresh)
    assert fresh._open_folder(screen._folder)
    assert int(fresh._canvas.mask[10, 10]) == top
    assert fresh._canvas.manual_ids
    fresh.close()


@pytest.mark.parametrize("background", [True, False])
def test_sparse_uint32_canvas_colours_match_regular_overlay(background):
    labels = np.full((40, 40), engine.MAX_MASK_LABEL, np.uint32)
    if background:
        labels[:20] = 0
    image = np.full(labels.shape, 20000, np.uint16)
    grey = mm._rgb32(np.repeat((image >> 8).astype(np.uint8)[..., None], 3, axis=-1))
    actual = mm._blend_labels(grey, labels)
    expected = mm._rgb32(engine.overlay_mask(image, labels))
    np.testing.assert_array_equal(actual, expected)
    extents = mm._object_extents(labels)
    assert engine.MAX_MASK_LABEL in extents
    mixed = labels.copy()
    mixed[30, 30] = 7
    mixed_colours = mm._blend_labels(grey.copy(), mixed)
    mixed[mixed == engine.MAX_MASK_LABEL] = 0
    low_only = mm._blend_labels(grey.copy(), mixed)
    assert mixed_colours[30, 30] == low_only[30, 30]


def test_named_division_undo_save_and_reopen_without_sidecar(qtbot, screen, monkeypatch):
    enable_time(screen)
    screen._label_name.setText("1")
    screen._add_lineage_cell()
    parent = screen._selected_cell
    assert screen._canvas.assign_label_at((10, 10), screen._canvas.selected_label)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)
    screen._divide_lineage_cell()
    graph = screen._lineage
    children = graph["cells"][parent]["children"]
    assert len(children) == 2 and parent not in children
    assert [graph["cells"][key]["name"] for key in children] == ["1", "1a"]
    assert all(graph["cells"][key]["parent_uuid"] == parent for key in children)
    first, second = children
    assert screen._canvas.assign_label_at((10, 10), screen._canvas.selected_label)
    screen._select_lineage_cell(screen._lineage_model.index(screen._lineage_model.keys.index(second), 0))
    assert screen._canvas.assign_label_at((30, 30), screen._canvas.selected_label)
    screen._on_undo()
    screen._on_undo()
    screen._on_undo()
    assert not screen._lineage["cells"][parent]["children"]
    screen._on_redo()
    screen._on_redo()
    screen._on_redo()
    assert screen._lineage["cells"][parent]["children"] == children
    screen._select_lineage_cell(screen._lineage_model.index(screen._lineage_model.keys.index(first), 0))
    screen._divide_lineage_cell()
    repeated = screen._lineage["cells"][first]["children"]
    assert [screen._lineage["cells"][key]["name"] for key in repeated] == ["1", "1b"]
    assert screen._daughter_name("1a") == "1aa"
    assert screen._rename_lineage_cell(second, "arbitrary branch name")
    screen._save_mask()
    for sidecar in Path(screen._folder).rglob("*.json"):
        sidecar.unlink()
    fresh = mm.MakeMasksScreen()
    qtbot.addWidget(fresh)
    assert fresh._open_folder(screen._folder)
    assert fresh._lineage["cells"][parent]["children"] == children
    assert fresh._lineage["cells"][second]["name"] == "arbitrary branch name"
    assert fresh._lineage["frames"][fresh._lineage_frame()][str(int(fresh._canvas.mask[30, 30]))] == second
    assert not fresh._time_label_enabled
    fresh.close()


def test_time_is_unavailable_for_volume_sessions(screen, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(screen._mask_stage, "session", object())
        screen._toggle_label_time(True)
        assert not screen._time_label_enabled
        assert screen._canvas.mode == mm.MODE_EDIT_LABEL
        assert "3D volumes" in screen._status_label.text()
        screen._sync_button_states()
        assert not screen._btn_label_time.isEnabled()


def test_virtual_cell_table_handles_ten_thousand_rows(screen):
    enable_time(screen)
    frame = screen._lineage_frame()
    state = engine.validate_lineage()
    for number in range(1, 10001):
        state["cells"][str(UUID(int=number))] = {
            "name": str(number), "parent_uuid": None, "born_frame": frame,
            "division_frame": None, "children": [], "label": number, "revision": 0}
    before = len(screen._lineage_table.findChildren(QWidget))
    screen._lineage = engine.validate_lineage(state)
    screen._refresh_lineage_rows()
    assert screen._lineage_model.rowCount() == 10000
    assert screen._lineage_model.data(screen._lineage_model.index(9999, 0)) == "10000"
    assert len(screen._lineage_table.findChildren(QWidget)) == before
    screen._select_lineage_cell(screen._lineage_model.index(9999, 0))
    assert screen._canvas.selected_label == 10000


def test_permuted_untracked_labels_keep_objects_and_persistent_identity(qtbot, screen):
    for index, filename in enumerate(screen._image_files):
        mask = np.zeros((40, 40), np.uint32)
        mask[5:15, 5:15] = 1 + index % 2
        mask[25:35, 25:35] = 2 - index % 2
        imageio.imwrite(engine.mask_save_path(screen._folder, filename), mask)
    screen._load_current()
    enable_time(screen)
    wheel(screen, 3)
    wait_idle(qtbot, screen)
    uuid = screen._selected_cell
    assert screen._save_mask()
    for index, filename in enumerate(screen._image_files):
        mask = imageio.imread(engine.mask_save_path(screen._folder, filename))
        assert np.count_nonzero(mask) == 200
        assert len(np.unique(mask)) == 3
        assert int(mask[30, 30]) == 2 - index % 2
        frame = str(Path(screen._folder, filename).absolute())
        assert screen._lineage["frames"][frame][str(int(mask[10, 10]))] == uuid
    before = screen._canvas.mask.copy()
    assert not screen._canvas.assign_label_at((30, 30), screen._canvas.selected_label)
    np.testing.assert_array_equal(screen._canvas.mask, before)


def click_pixel(screen, point):
    canvas = screen._canvas
    pos = QPointF(canvas._image_to_canvas(point[0] + .5, point[1] + .5))
    canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, pos,
                                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, pos, pos,
                                         Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


def test_brush_correction_with_max_id_saves_on_time_navigation(qtbot, screen):
    enable_time(screen)
    screen._label_value.setText(str(engine.MAX_MASK_LABEL))
    assert screen._canvas.assign_label_at((10, 10), engine.MAX_MASK_LABEL)
    screen._set_mode(mm.MODE_BRUSH)
    screen._canvas.brush_radius = 1
    click_pixel(screen, (20, 20))
    assert int(screen._canvas.mask[20, 20]) == 1
    assert int(screen._canvas.mask[10, 10]) == engine.MAX_MASK_LABEL
    screen._set_mode(mm.MODE_LABEL_TIME)
    wheel(screen, 1)
    wait_idle(qtbot, screen)
    saved = imageio.imread(engine.mask_save_path(screen._folder, "frame_0.tif"))
    assert int(saved[20, 20]) == 1
    assert int(saved[10, 10]) == engine.MAX_MASK_LABEL


def test_erasing_a_tracked_object_prunes_mapping_with_undo(screen):
    enable_time(screen)
    assert screen._canvas.assign_label_at((10, 10), 1)
    frame, uuid = screen._lineage_frame(), screen._selected_cell
    screen._set_mode(mm.MODE_ERASE_OBJECT)
    click_pixel(screen, (10, 10))
    assert int(screen._canvas.mask[10, 10]) == 0
    assert "1" not in screen._lineage["frames"][frame]
    screen._on_undo()
    assert int(screen._canvas.mask[10, 10]) == 1
    assert screen._lineage["frames"][frame]["1"] == uuid
    screen._on_redo()
    assert "1" not in screen._lineage["frames"][frame]
    assert uuid in screen._lineage["cells"]


def test_time_save_preserves_disconnected_numeric_identity_and_count(screen, monkeypatch):
    screen._canvas.mask[25:35, 25:35] = 7
    screen._canvas.manual_ids = False
    screen._set_mode(mm.MODE_BRUSH)
    enable_time(screen)
    assert screen._canvas.manual_ids
    counts = []
    monkeypatch.setattr(screen, "_note_curated", lambda _filename, n_objects: counts.append(n_objects))
    path = screen._save_mask()
    saved = imageio.imread(path)
    assert saved.dtype == np.uint32
    assert set(np.unique(saved)) == {0, 7}
    assert counts == [1]
