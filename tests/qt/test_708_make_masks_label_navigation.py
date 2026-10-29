"""Manual labels and serial wheel traversal of generic synthetic frames."""
from __future__ import annotations

import copy
import threading
from pathlib import Path
from uuid import UUID, uuid4

import imageio.v2 as imageio
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent, QModelIndex, QPoint, QPointF, Qt
from PySide6.QtGui import QCloseEvent, QMouseEvent, QValidator, QWheelEvent
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QWidget

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
    qtbot.waitUntil(lambda: not widget._loading and widget._reader_worker is None
                   and widget._load_worker is None and widget._canvas.mask is not None)
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


def _edit_snapshot(screen):
    """Capture the current editable transaction and the TIFFs it may save."""
    canvas, history = screen._canvas, screen._history
    return {
        "pixels": (canvas.mask.dtype.str, canvas.mask.shape, canvas.mask.tobytes()),
        "image": canvas.image.tobytes(),
        "lineage": copy.deepcopy(screen._lineage),
        "selection": (screen._selected_cell, canvas.selected_label),
        "dirty": (screen._lineage_dirty, screen._label_time_edited, canvas.manual_ids),
        "frame": (screen._folder, screen._current_index),
        "undo": [(mask.dtype.str, mask.tobytes()) for mask in history._undo],
        "redo": [(mask.dtype.str, mask.tobytes()) for mask in history._redo],
        "lineage_undo": copy.deepcopy(list(history._lineage_undo)),
        "lineage_redo": copy.deepcopy(list(history._lineage_redo)),
        "files": {str(path): path.read_bytes() for path in Path(screen._folder).rglob("*.tif")},
    }


def test_blank_label_entry_keeps_the_previous_identity(screen):
    screen._label_value.setText("321")
    before = _edit_snapshot(screen)
    screen._label_value.clear()
    assert screen._label_value.validator().validate("", 0)[0] == QValidator.Intermediate
    assert not screen._label_value.hasAcceptableInput()
    assert screen._canvas.selected_label == 321
    screen._label_value.editingFinished.emit()
    assert screen._label_value.text() == "321"
    assert _edit_snapshot(screen) == before


def test_lineage_table_rejects_non_name_edits_and_invalid_selection(screen):
    enable_time(screen)
    screen._add_lineage_cell()
    identity = screen._selected_cell
    model = screen._lineage_model
    before = _edit_snapshot(screen)
    assert model.data(model.index(0, 0), Qt.ToolTipRole) == identity
    assert model.data(model.index(0, 0), Qt.DecorationRole) is None
    assert model.data(QModelIndex()) is None
    assert not model.setData(QModelIndex(), "replacement")
    assert not model.setData(model.index(0, 1), "42")
    assert not model.setData(model.index(0, 0), "replacement", Qt.DisplayRole)
    screen._select_lineage_cell(QModelIndex())
    assert _edit_snapshot(screen) == before


def test_lineage_rename_refusal_and_same_name_preserve_history(screen):
    enable_time(screen)
    assert not screen._rename_lineage_cell(None, "orphan")
    screen._add_lineage_cell()
    identity = screen._selected_cell
    before = _edit_snapshot(screen)
    model = screen._lineage_model
    assert model.setData(model.index(0, 0), "1")
    assert not model.setData(model.index(0, 0), "   ")
    assert "nonempty name" in screen._status_label.text()
    assert _edit_snapshot(screen) == before
    screen._btn_label_time.click()
    assert not (model.flags(model.index(0, 0)) & Qt.ItemIsEditable)
    assert not screen._rename_lineage_cell(identity, "disabled edit")
    assert _edit_snapshot(screen) == before


@pytest.mark.parametrize("response", ["cancel", "blank_name", "blank_label", "same_label"])
def test_refused_division_keeps_the_parent_and_every_saved_pixel(screen, monkeypatch, response):
    enable_time(screen)
    screen._add_lineage_cell()
    parent = screen._selected_cell
    assert screen._canvas.assign_label_at((10, 10), 1)
    before = _edit_snapshot(screen)

    def answer(dialog):
        fields = dialog.findChildren(QLineEdit)
        assert len(fields) == 4
        if response == "cancel":
            return QDialog.Rejected
        if response == "blank_name":
            fields[0].clear()
        elif response == "blank_label":
            fields[1].clear()
        else:
            fields[3].setText(fields[1].text())
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", answer)
    screen._divide_lineage_cell()
    assert _edit_snapshot(screen) == before
    assert screen._lineage["cells"][parent]["children"] == []
    if response == "blank_label":
        assert "invalid literal" in screen._status_label.text()
    elif response != "cancel":
        assert "distinct nonzero labels" in screen._status_label.text()


def test_division_requires_a_selected_undivided_cell(screen, monkeypatch):
    enable_time(screen)
    before = _edit_snapshot(screen)
    screen._divide_lineage_cell()
    assert _edit_snapshot(screen) == before
    screen._add_lineage_cell()
    parent = screen._selected_cell
    monkeypatch.setattr(QDialog, "exec", lambda dialog: QDialog.Accepted)
    screen._divide_lineage_cell()
    select(screen, parent)
    before = _edit_snapshot(screen)
    screen._divide_lineage_cell()
    assert "has not already divided" in screen._status_label.text()
    assert _edit_snapshot(screen) == before


@pytest.mark.parametrize("point", [None, (-1, 10), (10, 40), (0, 0)])
def test_empty_or_outside_label_assignments_preserve_the_transaction(screen, point):
    enable_time(screen)
    before = _edit_snapshot(screen)
    assert screen._canvas.assign_label_at(point, 321)
    assert _edit_snapshot(screen) == before


@pytest.mark.parametrize("answer", [("-1", True), ("", True), ("321", False)])
def test_numeric_label_dialog_rejects_invalid_or_cancelled_input(screen, monkeypatch, answer):
    before = _edit_snapshot(screen)
    monkeypatch.setattr(mm.QInputDialog, "getText", lambda *args, **kwargs: answer)
    click_pixel(screen, (10, 10))
    assert _edit_snapshot(screen) == before
    if answer[1]:
        assert "non-negative whole-number" in screen._status_label.text()


def test_label_clicks_outside_or_on_background_never_open_the_dialog(screen, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("an empty location opened the numeric label dialog")

    monkeypatch.setattr(mm.QInputDialog, "getText", unexpected)
    before = _edit_snapshot(screen)
    click_pixel(screen, (0, 0))
    canvas = screen._canvas
    position = QPointF(-20, -20)
    event = QMouseEvent(QEvent.MouseButtonPress, position, position,
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    canvas.mousePressEvent(event)
    canvas.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, position, position,
                                        Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    assert event.isAccepted()
    assert _edit_snapshot(screen) == before


def test_selected_maximum_label_wraps_to_a_free_local_id(screen):
    assert screen._canvas.assign_label_at((10, 10), engine.MAX_MASK_LABEL)
    assert screen._canvas.assign_label_at((30, 30), 1)
    enable_time(screen)
    screen._label_value.setText(str(engine.MAX_MASK_LABEL))
    before_mask = screen._canvas.mask.copy()
    screen._add_lineage_cell()
    assert screen._canvas.selected_label == 2
    assert screen._lineage["cells"][screen._selected_cell]["label"] == 2
    np.testing.assert_array_equal(screen._canvas.mask, before_mask)
    assert not screen._lineage["frames"].get(screen._lineage_frame())


def test_time_click_can_remove_a_track_without_deleting_its_lineage(screen):
    enable_time(screen)
    click_pixel(screen, (10, 10))
    identity, frame = screen._selected_cell, screen._lineage_frame()
    assert screen._lineage["frames"][frame]["1"] == identity
    before = screen._canvas.mask.copy()
    screen._label_value.setText("0")
    click_pixel(screen, (10, 10))
    assert screen._canvas.mask[10, 10] == 0
    assert "1" not in screen._lineage["frames"][frame]
    assert identity in screen._lineage["cells"]
    assert screen._label_value.text() == "0"
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._lineage["frames"][frame]["1"] == identity


def test_conflicting_track_stops_the_wheel_queue_without_saving(screen):
    enable_time(screen)
    click_pixel(screen, (10, 10))
    screen._add_lineage_cell()
    click_pixel(screen, (30, 30))
    before = _edit_snapshot(screen)
    wheel(screen, 2, point=(10, 10))
    assert not screen._label_time_queue
    assert screen._load_worker is None
    assert "already labels another object in this frame" in screen._status_label.text()
    assert _edit_snapshot(screen) == before


def test_time_only_navigation_and_dialogs_are_inert_in_ordinary_mode(screen, monkeypatch):
    before = _edit_snapshot(screen)
    screen._set_mode(mm.MODE_LABEL_TIME)
    assert screen._canvas.mode == mm.MODE_EDIT_LABEL
    screen._queue_label_time_step(2, (10, 10), 321)
    assert not screen._label_time_queue and screen._load_worker is None
    screen._set_mode(mm.MODE_BRUSH)
    monkeypatch.setattr(mm.QInputDialog, "getText", lambda *args, **kwargs: pytest.fail("wrong tool opened dialog"))
    screen._edit_label_at(10, 10, 7)
    assert _edit_snapshot(screen) == before


def test_failed_time_save_refuses_next_and_close_without_losing_the_edit(screen, monkeypatch):
    enable_time(screen)
    click_pixel(screen, (10, 10))
    screen._set_mode(mm.MODE_EDIT_LABEL)
    before = _edit_snapshot(screen)
    warnings = []

    def refuse(*args, **kwargs):
        raise OSError("synthetic read-only mask folder")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "save_mask", refuse)
        patch.setattr(screen, "_warn", lambda *args: warnings.append(args))
        screen._on_next()
        event = QCloseEvent()
        screen.closeEvent(event)
        assert not event.isAccepted()
        assert _edit_snapshot(screen) == before
    assert len(warnings) == 2
    assert all("read-only mask folder" in str(warning) for warning in warnings)


def test_time_enable_merges_later_lineage_and_skips_corrupt_metadata(screen):
    folder = screen._folder
    later_frame = str(Path(folder) / "frame_1.tif")
    state, identity = engine.new_lineage_cell(None, "later cell", 41, later_frame)
    later_mask = screen._canvas.mask.copy()
    later_mask[later_mask == 7] = 41
    engine.save_mask(folder, "frame_1.tif", later_mask, preserve_ids=True, lineage=state)
    tifffile.imwrite(engine.mask_save_path(folder, "frame_2.tif"),
                     screen._canvas.mask, description='{"spacr_mask":')
    before = _edit_snapshot(screen)
    enable_time(screen)
    assert screen._lineage["cells"][identity]["name"] == "later cell"
    assert screen._lineage_model.rowCount() == 1
    assert screen._lineage_dirty
    assert "1 mask file(s)" in screen._status_label.text()
    assert "was skipped" in screen._status_label.text()
    after = _edit_snapshot(screen)
    for key in ("pixels", "image", "undo", "redo", "lineage_undo", "lineage_redo", "files"):
        assert after[key] == before[key]
    screen._btn_label_time.click()
    screen._btn_label_time.click()
    assert screen._lineage_model.rowCount() == 1
    assert _edit_snapshot(screen) == after


def test_conflicting_next_frame_metadata_keeps_the_current_transaction(qtbot, screen):
    enable_time(screen)
    click_pixel(screen, (10, 10))
    assert screen._save_label_time_if_needed()
    conflict = copy.deepcopy(screen._lineage)
    frame = screen._lineage_frame()
    other_identity = str(uuid4())
    conflict["frame_ids"][frame] = other_identity
    for cell in conflict["cells"].values():
        if cell["born_frame"] == frame:
            cell["born_frame_id"] = other_identity
    engine.save_mask(screen._folder, "frame_1.tif", screen._canvas.mask,
                     preserve_ids=True, lineage=conflict)
    before = _edit_snapshot(screen)
    wheel(screen, 2)
    wait_idle(qtbot, screen)
    assert "frame's identity" in screen._status_label.text()
    assert "current edits are kept" in screen._status_label.text()
    assert not screen._canvas.label_time_busy
    assert _edit_snapshot(screen) == before


def test_time_wheel_refuses_a_volume_without_changing_the_2d_frame(screen):
    tifffile.imwrite(Path(screen._folder) / "frame_1.tif",
                     np.ones((3, 40, 40), np.uint8), photometric="minisblack",
                     metadata={"axes": "ZYX"})
    enable_time(screen)
    before = _edit_snapshot(screen)
    wheel(screen, 2, point=(0, 0))
    assert "requires a queue of 2D frames" in screen._status_label.text()
    assert not screen._label_time_queue
    assert screen._load_worker is None
    assert _edit_snapshot(screen) == before


def test_loading_time_frame_ignores_click_hover_and_redo(qtbot, screen, monkeypatch):
    real_load = engine.load_image_and_mask
    started, release = threading.Event(), threading.Event()

    def delayed(folder, filename, **kwargs):
        started.set()
        assert release.wait(10)
        return real_load(folder, filename, **kwargs)

    monkeypatch.setattr(engine, "load_image_and_mask", delayed)
    enable_time(screen)
    try:
        wheel(screen, 1)
        qtbot.waitUntil(started.is_set, timeout=3000)
        assert screen._canvas.label_time_busy
        before = _edit_snapshot(screen)
        click_pixel(screen, (30, 30))
        hover(screen, (30, 30))
        screen._on_redo()
        assert _edit_snapshot(screen) == before
    finally:
        release.set()
        wait_idle(qtbot, screen)
    assert screen._current_index == 1
    assert screen._canvas.mask[30, 30] == 11


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
    assert int(canvas.mask[10, 10]) == 7
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


def test_separate_label_tools_switch_without_disabling_time(qtbot, screen, monkeypatch):
    screen.resize(1366, 768)
    screen.show()
    qtbot.waitExposed(screen)
    direct = screen._label_tool_button
    tracking = screen._label_time_tool_button
    toggle = screen._btn_label_time
    qtbot.waitUntil(lambda: all(button.visibleRegion().boundingRect() == button.rect()
                               for button in (direct, tracking, toggle)))
    assert direct is not tracking
    assert direct.text() == "Edit Label"
    assert tracking.text() == "Edit Label / Time"
    assert direct.isEnabled() and not tracking.isEnabled()
    row = screen._tool_row_layout
    assert row.indexOf(direct) + 1 == row.indexOf(screen._label_tool_group)
    cluster = screen._label_tool_group.layout()
    assert cluster.itemAt(cluster.count() - 1).widget() is toggle
    qtbot.mouseClick(toggle, Qt.LeftButton)
    assert screen._time_label_enabled and toggle.isChecked()
    for button in (direct, tracking, toggle):
        assert button.isVisible() and button.isEnabled()
        assert button.visibleRegion().boundingRect() == button.rect()
    assert tracking.isChecked() and not direct.isChecked()
    screen._label_value.setText("321")
    canvas = screen._canvas
    pos = QPointF(canvas._image_to_canvas(10.5, 10.5))
    click_pixel(screen, (10, 10))
    identity = screen._selected_cell
    frame = screen._lineage_frame()
    assert screen._lineage["frames"][frame]["321"] == identity

    qtbot.mouseClick(direct, Qt.LeftButton)
    assert toggle.isChecked() and screen._time_label_enabled
    assert direct.isChecked() and not tracking.isChecked()
    assert canvas.mode == mm.MODE_EDIT_LABEL
    screen._label_value.setText("654")
    canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, pos,
                                      Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    assert int(canvas.mask[10, 10]) == 321
    dialogs = []

    def choose_label(*args, **kwargs):
        dialogs.append(kwargs["text"])
        return "654", True

    monkeypatch.setattr(mm.QInputDialog, "getText", choose_label)
    qtbot.mouseClick(canvas, Qt.LeftButton, pos=pos.toPoint())
    assert dialogs == ["321"]
    assert int(canvas.mask[10, 10]) == 654
    assert screen._lineage["frames"][frame] == {"654": identity}
    screen._on_undo()
    assert int(canvas.mask[10, 10]) == 321
    assert screen._lineage["frames"][frame] == {"321": identity}
    screen._on_redo()
    assert int(canvas.mask[10, 10]) == 654
    assert screen._selected_cell == identity
    wheel(screen, -1)
    assert screen._current_index == 0 and canvas.is_zoomed()

    qtbot.mouseClick(tracking, Qt.LeftButton, pos=QPoint(5, tracking.height() // 2))
    assert toggle.isChecked() and screen._time_label_enabled
    assert tracking.isChecked() and not direct.isChecked()
    assert canvas.mode == mm.MODE_LABEL_TIME
    assert screen._canvas.selected_label == 654
    wheel(screen, 1)
    wait_idle(qtbot, screen)
    assert screen._current_index == 1
    assert int(canvas.mask[10, 10]) == 654
    assert screen._lineage["frames"][screen._lineage_frame()]["654"] == identity
    assert direct.text() == "Edit Label" and tracking.text() == "Edit Label / Time"
    qtbot.mouseClick(direct, Qt.LeftButton)
    qtbot.mouseClick(toggle, Qt.LeftButton)
    assert direct.isChecked() and direct.isEnabled()
    assert not tracking.isChecked() and not tracking.isEnabled()
    assert not screen._lineage_controls.isEnabled()
    assert screen._selected_cell == identity and screen._lineage["cells"]


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


def test_sorted_lineage_selection_and_refresh_keep_the_cell_identity(screen):
    enable_time(screen)
    frame = screen._lineage_frame()
    state = engine.validate_lineage()
    identities = [str(UUID(int=number)) for number in (1, 2, 3)]
    for identity, name, label in zip(identities, ("Zulu", "Alpha", "Mike"),
                                   (101, 303, 202)):
        state["cells"][identity] = {
            "name": name, "parent_uuid": None, "born_frame": frame,
            "division_frame": None, "children": [], "label": label, "revision": 0}
    screen._lineage = engine.validate_lineage(state)
    screen._selected_cell = identities[0]
    screen._refresh_lineage_rows()
    table = screen._lineage_table
    proxy = table.model()
    assert proxy.sourceModel() is screen._lineage_model
    table.sortByColumn(1, Qt.DescendingOrder)
    assert [proxy.index(row, 1).data() for row in range(3)] == [303, 202, 101]
    table.selectRow(1)
    assert screen._selected_cell == identities[2]
    assert screen._canvas.selected_label == 202
    table.sortByColumn(0, Qt.AscendingOrder)
    assert [proxy.index(row, 0).data() for row in range(3)] == ["Alpha", "Mike", "Zulu"]
    table.selectRow(0)
    selected = identities[1]
    assert screen._selected_cell == selected
    assert screen._label_name.text() == "Alpha"
    assert screen._canvas.selected_label == 303
    assert screen._canvas.assign_label_at((10, 10), screen._canvas.selected_label)
    assert int(screen._canvas.mask[10, 10]) == 303
    assert int(screen._canvas.mask[30, 30]) == 11
    assert proxy.setData(proxy.index(0, 0), "Zzz", Qt.EditRole)
    screen._refresh_lineage_rows()
    assert [proxy.index(row, 0).data() for row in range(3)] == ["Mike", "Zulu", "Zzz"]
    assert table.currentIndex().row() == 2
    assert screen._selected_cell == selected
    assert screen._label_name.text() == "Zzz"
    assert screen._canvas.selected_label == 303
    assert screen._lineage_model.keys == identities
    assert [screen._lineage["cells"][key]["label"] for key in identities] == [101, 303, 202]
    assert int(screen._canvas.mask[10, 10]) == 303


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


@pytest.fixture
def dividing(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = tmp_path / "dividing"
    (folder / "masks").mkdir(parents=True)
    for index in range(5):
        mask = np.zeros((40, 40), dtype=np.uint16)
        if index < 2:
            mask[5:15, 5:35] = 5
        else:
            mask[5:15, 5:15] = 8
            mask[5:15, 25:35] = 3
        imageio.imwrite(folder / f"frame_{index}.tif", np.full(mask.shape, index + 1, np.uint8))
        imageio.imwrite(folder / "masks" / f"frame_{index}.tif", mask)
    widget = mm.MakeMasksScreen()
    monkeypatch.setattr(widget, "_confirm", lambda *args: True)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)
    qtbot.addWidget(widget)
    widget._set_mode(mm.MODE_EDIT_LABEL)
    assert widget._open_folder(str(folder))
    qtbot.waitUntil(lambda: not widget._loading and widget._reader_worker is None
                    and widget._load_worker is None)
    widget._canvas.resize(400, 400)
    widget._canvas.refresh()
    widget._btn_save_on_navigation.setChecked(False)
    yield widget
    widget.close()


def hover(screen, point):
    canvas = screen._canvas
    pos = QPointF(canvas._image_to_canvas(point[0] + .5, point[1] + .5))
    canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, pos,
                                      Qt.NoButton, Qt.NoButton, Qt.NoModifier))


def select(screen, identity):
    model = screen._lineage_model
    screen._select_lineage_cell(model.index(model.keys.index(identity), 0))


@pytest.mark.parametrize("pixel_label", [1, 70000], ids=["ordinary-labels", "beyond-16-bit"])
def test_division_branches_trace_both_ways_and_reopen_from_tiffs_only(qtbot, dividing, pixel_label):
    screen = dividing
    sources = {path: path.read_bytes() for path in Path(screen._folder).glob("*.tif")}
    source_pixels = {path: imageio.imread(path).copy() for path in sources}
    enable_time(screen)
    screen._label_value.setText(str(pixel_label))
    screen._label_name.setText("1")
    screen._add_lineage_cell()
    mother = screen._selected_cell
    hover(screen, (10, 10))
    wheel(screen, 2)
    wait_idle(qtbot, screen)
    assert screen._current_index == 2
    screen._divide_lineage_cell()
    first, second = screen._lineage["cells"][mother]["children"]
    assert screen._selected_cell == first
    hover(screen, (10, 10))
    wheel(screen, 2)
    wait_idle(qtbot, screen)
    assert screen._current_index == 4
    select(screen, second)
    hover(screen, (30, 10))
    wheel(screen, -4, point=(30, 10))
    wait_idle(qtbot, screen)
    assert screen._current_index == 1
    assert "already continues through its other daughter" in screen._status_label.text()
    mask = screen._canvas.mask
    assert int(mask[10, 30]) == int(mask[10, 10]) != 0
    assert screen._lineage["frames"][screen._lineage_frame()][str(int(mask[10, 30]))] == mother
    screen._save_mask()
    for sidecar in Path(screen._folder).rglob("*.json"):
        sidecar.unlink()
    folder = screen._folder
    owners = {}
    for index in range(5):
        frame = str(Path(folder) / f"frame_{index}.tif")
        saved = imageio.imread(engine.mask_save_path(folder, f"frame_{index}.tif"))
        assert saved.dtype == np.uint32
        assert np.count_nonzero(saved) == (300 if index < 2 else 200)
        assert int(saved[10, 10]) >= pixel_label and int(saved[10, 30]) >= pixel_label
        metadata = engine.load_mask_metadata(folder, f"frame_{index}.tif")
        assert metadata["lineage"]["cells"][mother]["label"] == pixel_label
        if index >= 2:
            assert metadata["lineage"]["cells"][mother]["children"] == [first, second]
        mapping = metadata["lineage"]["frames"][frame]
        expected = (mother, mother) if index < 2 else (first, second)
        assert (mapping[str(int(saved[10, 10]))], mapping[str(int(saved[10, 30]))]) == expected
        owners[index] = (int(saved[10, 10]), int(saved[10, 30]), frame)
    assert owners[0][0] == owners[0][1] and owners[1][0] == owners[1][1]
    assert all(owners[index][0] != owners[index][1] for index in (2, 3, 4))
    fresh = mm.MakeMasksScreen()
    qtbot.addWidget(fresh)
    assert fresh._open_folder(folder)
    qtbot.waitUntil(lambda: not fresh._loading and fresh._reader_worker is None
                    and fresh._load_worker is None)
    graph = fresh._lineage
    assert graph["cells"][mother]["children"] == [first, second]
    assert [graph["cells"][key]["name"] for key in (mother, first, second)] == ["1", "1", "1a"]
    assert graph["cells"][first]["parent_uuid"] == graph["cells"][second]["parent_uuid"] == mother
    assert UUID(mother).int and UUID(first).int != UUID(second).int
    for index, (left, right, frame) in owners.items():
        mapping = graph["frames"][frame]
        expected = (mother, mother) if index < 2 else (first, second)
        assert (mapping[str(left)], mapping[str(right)]) == expected
    assert not fresh._time_label_enabled
    np.testing.assert_array_equal(fresh._canvas.mask, imageio.imread(engine.mask_save_path(folder, "frame_0.tif")))
    for path, original in sources.items():
        assert path.read_bytes() == original
        np.testing.assert_array_equal(imageio.imread(path), source_pixels[path])
    fresh.close()


@pytest.fixture
def relabelled(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = tmp_path / "relabelled"
    (folder / "masks").mkdir(parents=True)
    frames = [str(folder / f"frame_{index}.tif") for index in range(4)]
    state, early = engine.new_lineage_cell(None, "107", 107, frames[0])
    state = engine.assign_lineage_label(state, frames[0], 107, early)
    state, mother = engine.new_lineage_cell(state, "61", 61, frames[1])
    state = engine.assign_lineage_label(state, frames[1], 61, mother)
    state, (kept, sister) = engine.divide_lineage_cell(
        state, mother, frames[2], ["89", "90"], [89, 90])
    for frame in frames[2:]:
        state = engine.assign_lineage_label(state, frame, 89, kept)
        state = engine.assign_lineage_label(state, frame, 90, sister)
    for index, local in enumerate(((107,), (61,), (89, 90), (89, 90))):
        mask = np.zeros((40, 40), np.uint32)
        mask[5:15, 5:15] = local[0]
        if len(local) > 1:
            mask[25:35, 25:35] = local[1]
        imageio.imwrite(frames[index], np.full(mask.shape, index + 1, np.uint8))
        engine.save_mask(str(folder), f"frame_{index}.tif", mask,
                         preserve_ids=True, lineage=state)
    widget = mm.MakeMasksScreen()
    monkeypatch.setattr(widget, "_confirm", lambda *args: True)
    qtbot.addWidget(widget)
    widget._set_mode(mm.MODE_EDIT_LABEL)
    assert widget._open_folder(str(folder))
    qtbot.waitUntil(lambda: not widget._loading and widget._reader_worker is None
                    and widget._load_worker is None)
    widget._canvas.resize(400, 400)
    widget._canvas.refresh()
    widget._btn_save_on_navigation.setChecked(False)
    yield widget, (early, mother, kept, sister)
    widget.close()


def _graph(lineage):
    return ({key: (cell["name"], cell["label"], cell["parent_uuid"], cell["children"])
             for key, cell in lineage["cells"].items()},
            {Path(frame).name: mapping for frame, mapping in lineage["frames"].items()})


def test_selected_mother_continues_through_its_daughter_with_undo_and_reopen(qtbot, relabelled):
    screen, (early, mother, kept, sister) = relabelled
    screen._on_next()
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    screen._on_next()
    qtbot.waitUntil(lambda: not screen._loading and screen._current_index == 2, timeout=10000)
    enable_time(screen)
    select(screen, mother)
    before_pixels, before_graph = screen._canvas.mask.copy(), _graph(screen._lineage)
    hover(screen, (10, 10))
    np.testing.assert_array_equal(screen._canvas.mask, before_pixels)
    click_pixel(screen, (10, 10))
    assert int(screen._canvas.mask[10, 10]) == 61
    assert screen._selected_cell == kept
    assert "89 now carries 61 through the division" in screen._status_label.text()
    after_graph = _graph(screen._lineage)
    assert after_graph[0][kept][:2] == ("61", 61)
    assert after_graph[0][mother][3] == [kept, sister]
    assert after_graph[0][sister][:3] == ("90", 90, mother)
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before_pixels)
    assert _graph(screen._lineage) == before_graph
    screen._on_redo()
    assert _graph(screen._lineage) == after_graph
    select(screen, kept)
    wheel(screen, 1)
    wait_idle(qtbot, screen)
    assert screen._current_index == 3 and int(screen._canvas.mask[10, 10]) == 61
    click_pixel(screen, (30, 30))
    assert "61 already labels another object in this frame" in screen._status_label.text()
    assert int(screen._canvas.mask[30, 30]) == 90
    wheel(screen, -3)
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    assert screen._selected_cell == mother
    assert "107 was merged into 61" in screen._status_label.text()
    screen._save_mask()
    for sidecar in Path(screen._folder).rglob("*.json"):
        sidecar.unlink()
    fresh = mm.MakeMasksScreen()
    qtbot.addWidget(fresh)
    assert fresh._open_folder(screen._folder)
    qtbot.waitUntil(lambda: not fresh._loading and fresh._reader_worker is None
                    and fresh._load_worker is None)
    cells, frames = _graph(fresh._lineage)
    assert early not in cells
    assert cells[mother][:2] == ("61", 61) and cells[mother][3] == [kept, sister]
    assert cells[kept][:3] == ("61", 61, mother) and cells[sister][:3] == ("90", 90, mother)
    assert frames["frame_0.tif"] == {"61": mother} and frames["frame_1.tif"] == {"61": mother}
    assert frames["frame_2.tif"] == frames["frame_3.tif"] == {"61": kept, "90": sister}
    for index in range(4):
        saved = imageio.imread(engine.mask_save_path(screen._folder, f"frame_{index}.tif"))
        assert saved.dtype == np.uint32 and int(saved[10, 10]) == 61
    fresh.close()

