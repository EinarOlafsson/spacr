"""Box editing refuses failed reads and writes without losing an annotation."""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QInputDialog

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BOX, MakeMasksScreen


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
