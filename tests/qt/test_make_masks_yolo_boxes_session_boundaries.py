"""Box annotations cannot be lost or misattributed across session boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import (
    MODE_BOX,
    MODE_RECROP,
    MakeMasksScreen,
)


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    """Own an editor with two distinct standalone source images."""
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


def _dirty_box(widget):
    """Record the edit a completed box gesture sends through the screen."""
    widget._set_mode(MODE_BOX)
    widget._canvas.boxes = [(0, 12, 12, 32, 32)]
    widget._canvas.boxes_changed.emit()
    assert widget._boxes_dirty


def _fail_box_save(monkeypatch):
    """Make a disk failure leave the edited annotation unsaved."""
    def fail(*_args, **_kwargs):
        raise OSError("box disk is read-only")

    monkeypatch.setattr(engine, "save_yolo_boxes", fail)


def test_failed_box_save_refuses_queue_skip_without_recording_a_verdict(
        opened, monkeypatch):
    """Skip cannot mark a field done while its boxes are only in memory."""
    from spacr import curation_queue

    widget, folder = opened
    _dirty_box(widget)
    _fail_box_save(monkeypatch)
    marked = []
    monkeypatch.setattr(curation_queue, "mark_state",
                        lambda *args: marked.append(args))
    widget._queue = SimpleNamespace(folder=str(folder))

    widget._on_skip()

    assert marked == []
    assert widget._current_index == 0
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert widget._boxes_dirty
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_failed_box_save_refuses_starting_or_ending_blind_review(
        opened, monkeypatch):
    """A failed write cannot shuffle or reveal fields out from under an edit."""
    from spacr import run_journal

    widget, folder = opened
    _dirty_box(widget)
    _fail_box_save(monkeypatch)
    calls = []
    monkeypatch.setattr(run_journal, "start_blinding",
                        lambda *args, **kwargs: calls.append("start"))
    monkeypatch.setattr(run_journal, "unblind",
                        lambda *args, **kwargs: calls.append("end"))
    order = widget._field_pairs()

    assert widget._start_blind() is False
    assert widget._blind is None
    assert widget._field_pairs() == order
    assert calls == []

    widget._blind = {"key_id": "unused", "codes": {}, "original": []}
    try:
        assert widget._end_blind(ask=lambda: True) is False
        assert widget._blind is not None
        assert widget._field_pairs() == order
        assert calls == []
        assert widget._current_index == 0
        assert widget._boxes_dirty
        assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()
    finally:
        widget._blind = None


def test_failed_box_save_refuses_uncertainty_reorder(opened, monkeypatch):
    """Ranking may write scores but cannot change field order before boxes save."""
    widget, folder = opened
    _dirty_box(widget)
    _fail_box_save(monkeypatch)
    pairs = widget._field_pairs()
    scores = {
        pairs[0]: {"field": 0.1, "n_objects": 1, "n_passes": 1},
        pairs[1]: {"field": 0.9, "n_objects": 1, "n_passes": 1},
    }

    widget._apply_uncertainty_ranking(pairs, scores, model="cpsam")

    assert widget._field_pairs() == pairs
    assert widget._current_index == 0
    assert widget._canvas.boxes == [(0, 12, 12, 32, 32)]
    assert widget._boxes_dirty
    assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_annotated_source_cannot_be_recropped_or_changed(opened):
    """Recrop must not make a child whose boxes still name the parent pixels."""
    widget, folder = opened
    source = folder / "field_0.tif"
    original = source.read_bytes()
    _dirty_box(widget)
    names = list(widget._image_files)

    widget._set_mode(MODE_RECROP)
    assert widget._canvas.mode == MODE_BOX
    assert widget.recrop(0, 0, 48, 48) is None
    assert widget._image_files == names
    assert widget._recrop_children == []
    assert source.read_bytes() == original
    assert not list(folder.glob("*__r*"))


def test_a_pending_recrop_child_prevents_switching_to_box(opened):
    """The parent cannot acquire boxes after a child has already been cut."""
    widget, folder = opened
    source = folder / "field_0.tif"
    original = source.read_bytes()
    widget._set_mode(MODE_RECROP)
    child = widget.recrop(0, 0, 48, 48)
    assert child is not None
    assert widget._recrop_children == [child]

    widget._set_mode(MODE_BOX)

    assert widget._canvas.mode == MODE_RECROP
    assert widget._canvas.boxes == []
    assert widget._recrop_children == [child]
    assert source.read_bytes() == original


def test_segmentation_bundle_explains_why_box_editing_is_unavailable(
        qtbot, qt_theme_applied, tmp_path):
    """A Cellpose bundle stays read-only for boxes with a useful instruction."""
    folder = tmp_path / "bundle"
    folder.mkdir()
    source = folder / "field_seg.npy"
    np.save(source, {"img": np.zeros((64, 64), dtype=np.uint8),
                     "masks": np.zeros((64, 64), dtype=np.uint16)})
    original = source.read_bytes()
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    try:
        assert widget._open_folder(str(folder), files=[source.name])
        assert widget._box_load_error is not None
        assert not widget._mode_buttons[MODE_BOX].isEnabled()
        assert not widget._canvas.boxes_editable

        widget._set_mode(MODE_BOX)

        assert widget._canvas.mode != MODE_BOX
        assert "Export the image" in widget._status_label.text()
        assert widget._on_save_boxes() is None
        assert source.read_bytes() == original
        assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()
    finally:
        widget._magnifier.close()
        widget.close_folded()


def test_blind_yolo_export_reveals_no_source_name_or_label_file(
        opened, monkeypatch, tmp_path):
    """Neither the default picker nor an explicit label path bypasses blinding."""
    from PySide6.QtWidgets import QFileDialog

    widget, folder = opened
    _dirty_box(widget)
    source = folder / "field_0.tif"
    widget._blind = {"key_id": "unused", "codes": {str(source): "Q17"},
                     "original": [str(source)]}
    calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *args: calls.append("picker")))
    monkeypatch.setattr(widget, "_warn",
                        lambda title, message: calls.append((title, message)))
    target = tmp_path / "field_0.txt"
    try:
        assert widget._on_export_yolo_boxes() is None
        assert widget._on_export_yolo_boxes(str(target)) is None
        assert "picker" not in calls
        assert len(calls) == 2
        assert all("field_0" not in " ".join(value) for value in calls)
        assert not target.exists()
        assert not (tmp_path / ".classes.json").exists()
        assert not (folder / engine.YOLO_ANNOTATIONS_NAME).exists()
        assert widget._boxes_dirty
    finally:
        widget._blind = None
