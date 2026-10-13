"""A Box edit stays with its source when host controls or queues change."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BOX, MODE_DRAW, MakeMasksScreen


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    """Open two distinct standalone fields in different source folders."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    imageio.imwrite(first / "a.tif", image)
    imageio.imwrite(second / "b.tif", image + 1)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget.open_paths([str(first / "a.tif"), str(second / "b.tif")])
    yield widget, first, second
    widget._magnifier.close()
    widget.close_folded()


def _dirty_box(widget):
    """Make the current field's in-memory annotation need a project save."""
    widget._set_mode(MODE_BOX)
    widget._canvas.boxes = [(0, 12, 12, 32, 32)]
    widget._canvas.boxes_changed.emit()
    assert widget._boxes_dirty


def _refuse_save(monkeypatch):
    """Report an actual save error without creating an annotation ledger."""
    def fail(*_args, **_kwargs):
        raise OSError("box project is read-only")

    monkeypatch.setattr(engine, "save_yolo_boxes", fail)


@pytest.mark.parametrize("shortcut_rows", ["absent", "empty"])
def test_box_tools_work_fullscreen_without_optional_shortcut_labels(
        opened, monkeypatch, shortcut_rows):
    """Missing hints cannot disable classes, editing, or a durable save."""
    widget, first, _second = opened
    widget.showFullScreen()
    if shortcut_rows == "absent":
        monkeypatch.delattr(widget, "_shortcut_rows")
    else:
        monkeypatch.setattr(widget, "_shortcut_rows", {})

    widget._set_mode(MODE_BOX)
    assert widget._canvas.mode == MODE_BOX
    assert widget._box_controls.isEnabled()
    assert widget._on_add_box_class("rare") == 1
    widget._canvas.boxes = [(1, 12, 12, 32, 32)]
    widget._canvas.boxes_changed.emit()
    widget._set_mode(MODE_DRAW)
    widget._set_mode(MODE_BOX)
    assert widget._on_save_boxes() == first / engine.YOLO_ANNOTATIONS_NAME

    project = json.loads((first / engine.YOLO_ANNOTATIONS_NAME).read_text())
    assert project["classes"] == ["object", "rare"]
    assert project["images"]["a.tif"]["boxes"] == [[1, 12.0, 12.0, 32.0, 32.0]]
    assert not widget._boxes_dirty


def test_unbound_dirty_boxes_refuse_new_field_load_without_writing(
        opened, monkeypatch):
    """A cleared source binding cannot assign an old edit to the next image."""
    widget, first, second = opened
    _dirty_box(widget)
    prior = widget._canvas.image.copy()
    widget._box_field = None
    decoded = []
    monkeypatch.setattr(widget, "_load_pair",
                        lambda *_args: decoded.append("decoded"))
    monkeypatch.setattr(engine, "save_yolo_boxes",
                        lambda *_args, **_kwargs: decoded.append("saved"))

    widget._current_index = 1
    widget._load_current()

    assert decoded == []
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert np.array_equal(widget._canvas.image, prior)
    assert widget._boxes_dirty
    assert not (first / engine.YOLO_ANNOTATIONS_NAME).exists()
    assert not (second / engine.YOLO_ANNOTATIONS_NAME).exists()


@pytest.mark.parametrize("retain_owner", [True, False],
                         ids=["owner-still-queued", "owner-no-longer-queued"])
def test_changed_multifolder_queue_cannot_load_another_image_after_save_fails(
        opened, monkeypatch, retain_owner):
    """Rollback finds the old owner if present and never decodes a new field."""
    widget, first, second = opened
    _dirty_box(widget)
    prior = widget._canvas.image.copy()
    _refuse_save(monkeypatch)
    decoded = []
    monkeypatch.setattr(widget, "_load_pair",
                        lambda *_args: decoded.append("decoded"))

    widget._image_files = ["b.tif"] + (["a.tif"] if retain_owner else [])
    widget._field_folders = [str(second)] + ([str(first)] if retain_owner else [])
    widget._current_index = 0
    widget._load_current()

    assert decoded == []
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert np.array_equal(widget._canvas.image, prior)
    assert widget._box_field == (str(first), "a.tif")
    assert widget._boxes_dirty
    if retain_owner:
        assert widget._current_index == 1
        assert widget._folder == str(first)
    else:
        assert widget._current_index == 0
    assert not (first / engine.YOLO_ANNOTATIONS_NAME).exists()
    assert not (second / engine.YOLO_ANNOTATIONS_NAME).exists()
