"""Box editing refuses failed reads and writes without losing an annotation."""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QInputDialog

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BOX, MODE_DRAW, MakeMasksScreen


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    """Open two genuine TIFF fields and close the editor's helper windows."""
    folder = tmp_path / "fields"
    folder.mkdir()
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    imageio.imwrite(folder / "field_0.tif", image)
    imageio.imwrite(folder / "field_1.tif", image + 1)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    yield widget, folder
    widget._magnifier.close()
    widget.close_folded()


def _edit_box(widget):
    """Use the same signal as a completed canvas gesture."""
    widget._set_mode(MODE_BOX)
    widget._canvas.boxes = [(0, 12, 12, 32, 32)]
    widget._canvas.boxes_changed.emit()
    assert widget._boxes_dirty


def test_corrupt_project_refuses_box_mode_save_and_export(
        qtbot, qt_theme_applied, tmp_path):
    """A damaged ledger must never be overwritten with empty annotations."""
    folder = tmp_path / "fields"
    folder.mkdir()
    imageio.imwrite(folder / "field.tif", np.zeros((64, 64), dtype=np.uint16))
    project = folder / engine.YOLO_ANNOTATIONS_NAME
    damaged = b"{not valid JSON"
    project.write_bytes(damaged)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    try:
        assert widget._open_folder(str(folder))
        assert widget._box_load_error is not None
        assert not widget._canvas.boxes_editable
        assert not widget._mode_buttons[MODE_BOX].isEnabled()
        widget._set_mode(MODE_BOX)
        assert widget._canvas.mode != MODE_BOX
        assert widget._on_save_boxes() is None
        assert widget._on_export_yolo_boxes(str(tmp_path / "labels.txt")) is None
        assert project.read_bytes() == damaged
        assert not (tmp_path / "labels.txt").exists()
    finally:
        widget._magnifier.close()
        widget.close_folded()


def test_cancelled_class_dialog_keeps_classes_and_history(opened, monkeypatch):
    """Dismissing the class picker does not create a phantom class or edit."""
    widget, _folder = opened
    widget._set_mode(MODE_BOX)
    before = list(widget._box_classes)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *_args: ("discard me", False)))

    assert widget._on_add_box_class() is None
    widget._on_box_class_changed(-1)
    widget._on_undo()
    widget._on_redo()

    assert widget._box_classes == before
    assert widget._canvas.boxes == []
    assert not widget._boxes_dirty
    assert not widget._box_history.can_undo()


def test_accepted_class_dialog_and_reselecting_same_class_are_stable(
        opened, monkeypatch):
    """A chosen class is persistent, and choosing it again makes no new edit."""
    widget, _folder = opened
    widget._set_mode(MODE_BOX)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *_args: ("infected", True)))

    assert widget._on_add_box_class() == 1
    assert widget._box_classes == ["object", "infected"]
    widget._canvas.boxes = [(1, 12, 12, 32, 32)]
    widget._canvas.boxes_changed.emit()
    assert widget._on_save_boxes() is not None
    assert not widget._boxes_dirty

    widget._canvas.selected_box = 0
    widget._on_box_class_changed(widget._box_class_combo.findData(1))
    assert widget._canvas.boxes == [(1, 12, 12, 32, 32)]
    assert not widget._boxes_dirty


def test_cancelled_export_dialog_preserves_unsaved_box(opened, monkeypatch,
                                                       tmp_path):
    """Cancel leaves the edit in memory without writing project or labels."""
    widget, folder = opened
    _edit_box(widget)
    calls = []
    def cancel_picker(*_args):
        calls.append("picker")
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(cancel_picker))

    assert widget._on_export_yolo_boxes() is None

    assert calls == ["picker"]
    assert widget._boxes_dirty
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()
    assert not list(tmp_path.glob("*.txt"))


def test_export_picker_choice_writes_labels_and_commits_project(
        opened, monkeypatch, tmp_path):
    """The actual picker path writes labels only after saving the project."""
    widget, folder = opened
    _edit_box(widget)
    target = tmp_path / "field_0.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *_args: (str(target), "YOLO labels (*.txt)")))

    assert widget._on_export_yolo_boxes() == str(target)

    assert (folder / engine.YOLO_ANNOTATIONS_NAME).is_file()
    assert target.read_text(encoding="utf-8").startswith("0 ")
    assert not widget._boxes_dirty


def test_visible_save_button_commits_boxes_and_failed_export_keeps_them(
        opened, monkeypatch, tmp_path):
    """The visible save action commits first; a later export error cannot erase it."""
    widget, folder = opened
    _edit_box(widget)
    assert widget._btn_save_boxes.isEnabled()
    widget._btn_save_boxes.click()
    project = folder / engine.YOLO_ANNOTATIONS_NAME
    assert project.exists()
    assert not widget._boxes_dirty
    saved = project.read_bytes()

    def fail_export(*_args, **_kwargs):
        raise OSError("labels disk is read-only")

    monkeypatch.setattr(engine, "export_yolo_boxes", fail_export)
    target = tmp_path / "field_0.txt"
    assert widget._on_export_yolo_boxes(str(target)) is None
    assert "labels disk is read-only" in widget._status_label.text()
    assert project.read_bytes() == saved
    assert not target.exists()


def test_export_refuses_failed_project_save_before_writing_label(
        opened, monkeypatch, tmp_path):
    """YOLO labels cannot claim a box edit that the project failed to save."""
    widget, folder = opened
    _edit_box(widget)

    def fail_save(*_args, **_kwargs):
        raise OSError("project disk is read-only")

    monkeypatch.setattr(engine, "save_yolo_boxes", fail_save)
    target = tmp_path / "field_0.txt"
    assert widget._on_export_yolo_boxes(str(target)) is None
    assert widget._boxes_dirty
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()
    assert not target.exists()


def test_save_and_export_without_an_open_field_never_show_a_picker(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """Detached controls cannot create a label that claims an unknown image."""
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *_args: calls.append("picker")))
    try:
        original_mode = widget._canvas.mode
        widget._set_mode(MODE_BOX)
        assert widget._canvas.mode == original_mode
        assert not widget._box_controls.isEnabled()
        assert widget._on_add_box_class("orphan") is None
        assert widget._on_save_boxes() is None
        assert widget._on_export_yolo_boxes(str(tmp_path / "orphan.txt")) is None
        assert widget._on_export_yolo_boxes() is None
        assert calls == []
        assert not (tmp_path / "orphan.txt").exists()
    finally:
        widget._magnifier.close()
        widget.close_folded()


def test_failed_box_save_refuses_opening_another_folder(opened, monkeypatch,
                                                        tmp_path):
    """A source switch cannot strand an unsaved annotation on the old field."""
    widget, folder = opened
    _edit_box(widget)
    other = tmp_path / "other"
    other.mkdir()
    imageio.imwrite(other / "replacement.tif", np.zeros((64, 64), dtype=np.uint16))

    def fail_save(*_args, **_kwargs):
        raise OSError("project disk is read-only")

    monkeypatch.setattr(engine, "save_yolo_boxes", fail_save)
    assert not widget._open_folder(str(other))
    assert widget._folder == str(folder)
    assert widget._box_field == (str(folder), "field_0.tif")
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert widget._boxes_dirty
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_unreadable_next_image_clears_the_previous_fields_boxes(opened):
    """A failed TIFF decode cannot leave the prior field's boxes on screen."""
    widget, folder = opened
    _edit_box(widget)
    second = folder / "field_1.tif"
    second.write_bytes(b"not a TIFF")

    widget._on_next()

    assert (folder / engine.YOLO_ANNOTATIONS_NAME).exists()
    assert widget._canvas.image is None
    assert widget._canvas.boxes == []
    assert widget._box_field is None
    assert not widget._boxes_dirty
    assert not widget._box_history.can_undo()
    assert not widget._box_controls.isEnabled()
    assert widget._on_add_box_class("orphan") is None
    assert not widget._boxes_dirty
    widget._set_mode(MODE_DRAW)
    widget._set_mode(MODE_BOX)
    assert widget._canvas.mode == MODE_DRAW
    assert widget._on_save_boxes() is None
