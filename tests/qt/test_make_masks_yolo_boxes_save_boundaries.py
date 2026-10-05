"""Failed Box saves keep the field, queue and source files together."""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtGui import QImage

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BOX, MODE_RECROP, MakeMasksScreen, _MaskCanvas


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    """Own a two-image editor with visibly distinct source intensities."""
    folder = tmp_path / "fields"
    folder.mkdir()
    image = np.zeros((64, 64), dtype=np.uint16)
    image[12:32, 12:32] = 30000
    for index in range(2):
        imageio.imwrite(folder / f"field_{index}.tif", image + index)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    yield widget, folder
    widget._magnifier.close()
    widget.close_folded()


def _edited_box(widget):
    """Finish the same screen notification that a dragged box emits."""
    widget._set_mode(MODE_BOX)
    widget._canvas.boxes = [(0, 12, 12, 32, 32)]
    widget._canvas.boxes_changed.emit()
    assert widget._boxes_dirty


def _unwritable_boxes(monkeypatch):
    """Make the project annotation write fail without touching source files."""
    def refuse(*_args, **_kwargs):
        raise OSError("box project is read-only")

    monkeypatch.setattr(engine, "save_yolo_boxes", refuse)


def test_previous_stays_on_second_field_when_dirty_boxes_cannot_be_saved(
        opened, monkeypatch):
    """Previous cannot show field zero while field one's edit is unsaved."""
    widget, folder = opened
    widget._on_next()
    assert widget._current_index == 1
    _edited_box(widget)
    before_image = widget._canvas.image.copy()
    before_source = (folder / "field_1.tif").read_bytes()
    _unwritable_boxes(monkeypatch)

    widget._on_prev()

    assert widget._current_index == 1
    assert widget._box_field == (str(folder), "field_1.tif")
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert np.array_equal(widget._canvas.image, before_image)
    assert widget._boxes_dirty
    assert (folder / "field_1.tif").read_bytes() == before_source
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


@pytest.mark.parametrize("keep", [True, False], ids=["keep", "discard"])
def test_verdict_does_not_record_or_advance_after_failed_box_save(
        opened, monkeypatch, keep):
    """Neither curation verdict can outrun an unsaved annotation edit."""
    widget, folder = opened
    _edited_box(widget)
    before_image = widget._canvas.image.copy()
    before_source = (folder / "field_0.tif").read_bytes()
    _unwritable_boxes(monkeypatch)
    recorded = []
    monkeypatch.setattr(engine, "record_curation",
                        lambda *args, **kwargs: recorded.append((args, kwargs)))

    assert widget._on_curate(keep) is None

    assert recorded == []
    assert widget._current_index == 0
    assert widget._box_field == (str(folder), "field_0.tif")
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert np.array_equal(widget._canvas.image, before_image)
    assert widget._boxes_dirty
    assert (folder / "field_0.tif").read_bytes() == before_source
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_direct_load_rolls_back_index_to_unsaved_box_owner(opened, monkeypatch):
    """A caller changing the index cannot strand boxes on another image."""
    widget, folder = opened
    _edited_box(widget)
    original_image = widget._canvas.image.copy()
    original_boxes = list(widget._canvas.boxes)
    _unwritable_boxes(monkeypatch)

    widget._current_index = 1
    widget._load_current()

    assert widget._current_index == 0
    assert widget._box_field == (str(folder), "field_0.tif")
    assert widget._canvas.boxes == original_boxes
    assert np.array_equal(widget._canvas.image, original_image)
    assert widget._boxes_dirty
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_failed_class_save_refuses_recrop_retirement_and_keeps_child(
        opened, monkeypatch):
    """A new class with no boxes must be saved before archiving its parent."""
    widget, folder = opened
    parent = folder / "field_0.tif"
    parent_bytes = parent.read_bytes()
    widget._set_mode(MODE_RECROP)
    child_name = widget.recrop(0, 0, 48, 48)
    assert child_name is not None
    child = folder / child_name
    child_bytes = child.read_bytes()
    queued = list(widget._image_files)
    assert widget._on_add_box_class("rare") == 1
    assert widget._canvas.boxes == [] and widget._boxes_dirty
    _unwritable_boxes(monkeypatch)

    assert widget.finish_recrop() is False

    assert widget._image_files == queued
    assert widget._current_index == 0
    assert widget._recrop_children == [child_name]
    assert widget._box_classes == ["object", "rare"]
    assert widget._boxes_dirty
    assert parent.read_bytes() == parent_bytes
    assert child.read_bytes() == child_bytes
    assert not (folder / engine.RECROP_ARCHIVE_DIRNAME / parent.name).exists()
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_box_paint_ignores_mapping_during_a_field_transition(
        qtbot, qt_theme_applied):
    """An old pixmap with no current mask cannot produce stray box pixels."""
    canvas = _MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    image = np.zeros((64, 64), dtype=np.uint16)
    mask = np.zeros_like(image)
    canvas.set_image_and_mask(image, mask)
    canvas.mode = MODE_BOX
    canvas.boxes = [(0, 12, 12, 32, 32)]
    assert canvas.pixmap() is not None
    canvas.mask = None
    frame = QImage(canvas.size(), QImage.Format_RGB32)

    canvas.render(frame)

    assert canvas.boxes == [(0, 12, 12, 32, 32)]
    assert canvas._image_to_canvas(12, 12) is None
