"""Item 708: Time labelling refuses what it cannot do and keeps every edit.

Covers the cell table's read-only cells, renaming and adding cells that the
lineage refuses, running out of labels, a cancelled or invalid division,
assignment outside the image or on the background, the edit-label dialog,
the busy canvas while a frame loads, and wheel steps that cannot continue.

Real screen on synthetic 2-D frames (``tests/qt/test_708_make_masks_label_navigation``).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QModelIndex, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QValidator
from PySide6.QtWidgets import QDialog

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

from .test_708_make_masks_label_navigation import enable_time, screen, wait_idle

__all__ = ["screen"]


def test_the_label_field_accepts_only_uint32_decimal_text():
    validator = mm._MaskLabelValidator()
    assert validator.validate("", 0)[0] == QValidator.Intermediate
    assert validator.validate("4294967295", 3)[0] == QValidator.Acceptable
    assert validator.validate("4294967296", 3)[0] == QValidator.Invalid
    assert validator.validate("١٢", 1)[0] == QValidator.Invalid


def _with_cell(screen):
    enable_time(screen)
    screen._label_name.setText("founder")
    screen._add_lineage_cell()
    uuid = screen._selected_cell
    assert uuid in screen._lineage["cells"]
    return uuid


def test_the_cell_table_answers_only_its_own_rows_and_renames_only_names(screen):
    uuid = _with_cell(screen)
    model = screen._lineage_model
    assert model.data(QModelIndex()) is None
    assert model.data(model.index(0, 0), Qt.ToolTipRole) == uuid
    assert model.data(model.index(0, 0), Qt.DecorationRole) is None
    assert model.data(model.index(0, 0)) == "founder"
    assert model.setData(model.index(0, 1), "9") is False
    assert model.setData(model.index(0, 0), "x", Qt.DisplayRole) is False
    assert model.setData(model.index(0, 0), "renamed") is True
    assert screen._lineage["cells"][uuid]["name"] == "renamed"


def test_renaming_is_refused_when_off_unchanged_or_empty(screen):
    uuid = _with_cell(screen)
    assert screen._rename_lineage_cell(uuid, "founder") is True
    assert screen._rename_lineage_cell(uuid, "   ") is False
    assert "nonempty name" in screen._status_label.text()
    assert screen._lineage["cells"][uuid]["name"] == "founder"
    assert screen._rename_lineage_cell(None, "x") is False
    screen._btn_label_time.setChecked(False)
    assert screen._rename_lineage_cell(uuid, "later") is False
    assert screen._lineage["cells"][uuid]["name"] == "founder"


def test_an_invalid_table_index_selects_nothing(screen):
    uuid = _with_cell(screen)
    screen._select_lineage_cell(QModelIndex())
    assert screen._selected_cell == uuid


def test_labels_wrap_to_the_lowest_free_id_and_then_run_out(screen, monkeypatch):
    _with_cell(screen)
    monkeypatch.setattr(engine, "MAX_MASK_LABEL", 12)
    assert screen._free_lineage_label(12, excluded=(12,)) == 2
    used = set(range(1, 13))
    monkeypatch.setattr(screen._canvas, "mask", np.array([sorted(used)], np.uint16))
    with pytest.raises(ValueError, match="No unused 32-bit label remains"):
        screen._free_lineage_label(5)
    cells = len(screen._lineage["cells"])
    screen._add_lineage_cell()
    assert "No unused 32-bit label remains" in screen._status_label.text()
    assert len(screen._lineage["cells"]) == cells
    screen._divide_lineage_cell()
    assert "No unused 32-bit label remains" in screen._status_label.text()
    assert len(screen._lineage["cells"]) == cells


def test_a_cancelled_or_invalid_division_records_nothing(screen, monkeypatch):
    uuid = _with_cell(screen)
    cells = dict(screen._lineage["cells"])
    monkeypatch.setattr(QDialog, "exec", lambda dialog: QDialog.Rejected)
    screen._divide_lineage_cell()
    assert screen._lineage["cells"] == cells

    def accept_with_same_labels(dialog):
        from PySide6.QtWidgets import QLineEdit
        fields = dialog.findChildren(QLineEdit)
        for field in fields:
            if field.text().isdecimal():
                field.setText("5")
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", accept_with_same_labels)
    screen._divide_lineage_cell()
    assert "distinct nonzero labels" in screen._status_label.text()
    assert not screen._lineage["cells"][uuid].get("children")


def test_assignment_outside_the_image_or_on_background_changes_nothing(screen):
    enable_time(screen)
    mask = screen._canvas.mask
    assert screen._prepare_label_assignment((400, 2), 5) == (mask, 0, 5, None)
    assert screen._prepare_label_assignment((1, 1), 5) == (mask, 0, 5, None)
    canvas = screen._canvas
    assert canvas.assign_label_at(None, 5) is True
    assert np.array_equal(canvas.mask, mask)


def test_erasing_a_tracked_object_resets_the_label_field_to_zero(screen):
    screen._set_mode(mm.MODE_EDIT_LABEL)
    uuid = _with_cell(screen)
    screen._set_mode(mm.MODE_EDIT_LABEL)
    frame = screen._lineage_frame()
    state = engine.assign_lineage_label(screen._lineage, frame, 7, uuid)
    screen._lineage = state
    screen._label_value.setText("3")
    assert screen._canvas.assign_label_at((10, 10), 0)
    assert int(screen._canvas.mask[10, 10]) == 0
    assert screen._label_value.text() == "0"


def test_the_canvas_without_an_editor_hook_assigns_directly(screen):
    canvas = screen._canvas
    canvas.prepare_label_assignment = None
    assert canvas.assign_label_at((10, 10), 9)
    assert int(canvas.mask[10, 10]) == 9


def test_the_edit_label_dialog_needs_its_mode_and_a_number(screen, monkeypatch):
    asked = []
    monkeypatch.setattr(mm.QInputDialog, "getText",
                        lambda *a, **k: asked.append(1) or ("seven", True))
    screen._set_mode(mm.MODE_BRUSH)
    screen._edit_label_at(10, 10, 7)
    assert asked == []
    screen._set_mode(mm.MODE_EDIT_LABEL)
    screen._edit_label_at(10, 10, 7)
    assert asked == [1]
    assert "non-negative whole-number label" in screen._status_label.text()
    assert int(screen._canvas.mask[10, 10]) == 7


def _press(canvas, x, y, kind=QEvent.MouseButtonPress, buttons=Qt.LeftButton):
    pos = QPointF(canvas._image_to_canvas(x + .5, y + .5))
    return QMouseEvent(kind, pos, pos, Qt.LeftButton, buttons, Qt.NoModifier)


def test_a_busy_canvas_swallows_clicks_and_moves(screen):
    enable_time(screen)
    canvas = screen._canvas
    before = canvas.mask.copy()
    canvas.label_time_busy = True
    event = _press(canvas, 10, 10)
    canvas.mousePressEvent(event)
    assert event.isAccepted() and Qt.LeftButton in canvas._swallowed
    canvas.mouseMoveEvent(_press(canvas, 11, 11, QEvent.MouseMove, Qt.NoButton))
    assert np.array_equal(canvas.mask, before)
    canvas.label_time_busy = False


def test_a_click_in_time_mode_assigns_the_selected_label(screen):
    enable_time(screen)
    screen._label_value.setText("77")
    canvas = screen._canvas
    canvas.mousePressEvent(_press(canvas, 10, 10))
    assert int(canvas.mask[10, 10]) == 77


def test_redo_waits_while_a_frame_is_loading(screen):
    screen._set_mode(mm.MODE_EDIT_LABEL)
    assert screen._canvas.assign_label_at((10, 10), 9)
    screen._on_undo()
    screen._canvas.label_time_busy = True
    screen._on_redo()
    assert int(screen._canvas.mask[10, 10]) == 7
    screen._canvas.label_time_busy = False
    screen._on_redo()
    assert int(screen._canvas.mask[10, 10]) == 9


def test_wheel_steps_outside_time_mode_or_after_a_failed_assignment_stop(screen, monkeypatch):
    screen._set_mode(mm.MODE_EDIT_LABEL)
    screen._queue_label_time_step(2, (10, 10), 5)
    assert not screen._label_time_queue
    enable_time(screen)
    monkeypatch.setattr(screen._canvas, "assign_label_at", lambda point, label: False)
    screen._queue_label_time_step(3, (10, 10), 5)
    assert not screen._label_time_queue and screen._current_index == 0


def test_time_steps_refuse_to_enter_a_volume(screen, monkeypatch):
    enable_time(screen)
    monkeypatch.setattr(engine, "_is_volume_file", lambda path: True)
    screen._queue_label_time_step(1, (10, 10), 5)
    assert "requires a queue of 2D frames" in screen._status_label.text()
    assert screen._current_index == 0 and not screen._label_time_queue


def test_a_loaded_frame_with_unreadable_lineage_is_not_entered(qtbot, screen, monkeypatch):
    enable_time(screen)

    def broken(*args, **kwargs):
        raise ValueError("lineage disagrees")

    monkeypatch.setattr(screen, "_lineage_for_loaded_frame", broken)
    screen._queue_label_time_step(1, (10, 10), 5)
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    assert "lineage disagrees" in screen._status_label.text()


def test_unsaved_time_edits_hold_navigation_and_close_while_loading(screen, monkeypatch):
    enable_time(screen)
    screen._label_time_edited = True
    monkeypatch.setattr(screen, "_loading", True)
    assert screen._save_label_time_if_needed() is False
    assert screen._save_mask_if_needed() is False
    event = SimpleCloseEvent()
    screen.closeEvent(event)
    assert event.ignored
    monkeypatch.setattr(screen, "_loading", False)


class SimpleCloseEvent:
    def __init__(self):
        self.ignored = False

    def ignore(self):
        self.ignored = True

    def accept(self):
        self.ignored = False


def test_time_mode_is_not_entered_while_off_or_with_a_volume(screen, monkeypatch):
    screen._set_mode(mm.MODE_EDIT_LABEL)
    screen._set_mode(mm.MODE_LABEL_TIME)
    assert screen._canvas.mode == mm.MODE_EDIT_LABEL
    monkeypatch.setattr(screen._mask_stage, "session", object())
    screen._set_mode(mm.MODE_LABEL_TIME)
    screen._set_mode(mm.MODE_EDIT_LABEL)
    assert screen._canvas.mode == mm.MODE_EDIT_LABEL


def test_enabling_time_merges_lineage_found_in_the_other_frames(qtbot, screen):
    other = screen._folder
    frame = os.path.join(other, "frame_1.tif")
    state, uuid = engine.new_lineage_cell(None, "elsewhere", 7, frame)
    state = engine.assign_lineage_label(state, frame, 7, uuid)
    engine.save_mask(other, "frame_1.tif", screen._canvas.mask.astype(np.uint32),
                     preserve_ids=True, lineage=state)
    screen._lineage_scanned = None
    enable_time(screen)
    assert uuid in screen._lineage["cells"] and screen._lineage_dirty
    assert screen._lineage_model.keys == [uuid]


def test_unreadable_lineage_in_another_frame_is_skipped_and_counted(screen, monkeypatch):
    real = engine.load_mask_metadata

    def unreadable(folder, name, **kwargs):
        if name.endswith("_2.tif"):
            raise ValueError("bad metadata")
        return real(folder, name, **kwargs)

    monkeypatch.setattr(engine, "load_mask_metadata", unreadable)
    result = screen._merge_sequence_lineage(engine.validate_lineage(None), screen._folder)
    assert result is not None
    assert "1 mask file(s) could not be read" in screen._status_label.text()


def test_a_loaded_pair_that_cannot_be_applied_is_reported_as_a_load_failure(screen,
                                                                            monkeypatch):
    from types import SimpleNamespace

    failures = []
    retired = []
    worker = SimpleNamespace(token=screen._load_token, error=None, result=("image", "mask"),
                             filename="frame_1.tif", metadata={})
    screen._load_worker = worker
    screen._label_time_pending = None

    def refuse(*args, **kwargs):
        raise ValueError("the mask has another shape")

    monkeypatch.setattr(screen, "_apply_loaded_pair", refuse)
    monkeypatch.setattr(screen, "_handle_load_failure", failures.append)
    monkeypatch.setattr(mm, "_retire_worker", retired.append)
    screen._on_background_load_finished()
    assert [str(error) for error in failures] == ["the mask has another shape"]
    assert retired == [worker] and screen._load_worker is None and not screen._loading
