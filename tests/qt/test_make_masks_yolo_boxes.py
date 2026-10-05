"""Real mouse interactions for Make Masks' YOLO bounding-box tool.

Boxes describe image pixels and classes; they are not segmentation labels.
The canvas in these tests shows a 64-pixel field as a centred 400-pixel
picture in a 600-by-400 widget, so a drag can check the coordinate transform
as well as the resulting annotation.
"""
from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import (
    MODE_BOX,
    MODE_DRAW,
    MODE_NONE,
    MODE_ZOOM,
    MakeMasksScreen,
    _MaskCanvas,
)
from spacr.qt.theme import active_palette

IMAGE_SIZE = 64
CANVAS_SIZE = (600, 400)
PICTURE_SIZE = 400
LEFT_MARGIN = 100


def _image() -> np.ndarray:
    """Give the box and existing mask object a real field to sit on."""
    return np.arange(IMAGE_SIZE ** 2, dtype=np.uint16).reshape(
        IMAGE_SIZE, IMAGE_SIZE)


def _mask() -> np.ndarray:
    """Keep an existing segmentation object under the boxes."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    mask[20:36, 20:36] = 7
    return mask


def _point(x: float, y: float) -> tuple[float, float]:
    """Map a known whole-field image pixel to its canvas-local centre."""
    scale = PICTURE_SIZE / IMAGE_SIZE
    return LEFT_MARGIN + (x + 0.25) * scale, (y + 0.25) * scale


def _event(kind, x, y, *, button=Qt.LeftButton, buttons=Qt.LeftButton,
           modifiers=Qt.NoModifier):
    """Build the Qt event that a physical mouse gesture would deliver."""
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def _press(widget, point, *, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    """Press a mouse button at a canvas-local point."""
    widget.mousePressEvent(_event(QEvent.Type.MouseButtonPress, *point,
                                  button=button, buttons=button,
                                  modifiers=modifiers))


def _move(widget, point, *, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    """Move while the given mouse button is held."""
    widget.mouseMoveEvent(_event(QEvent.Type.MouseMove, *point,
                                 button=Qt.NoButton, buttons=button,
                                 modifiers=modifiers))


def _release(widget, point, *, button=Qt.LeftButton,
             modifiers=Qt.NoModifier):
    """Release a mouse button at a canvas-local point."""
    widget.mouseReleaseEvent(_event(QEvent.Type.MouseButtonRelease, *point,
                                    button=button, buttons=Qt.NoButton,
                                    modifiers=modifiers))


def _drag_canvas(widget, start, end, *, modifiers=Qt.NoModifier):
    """Finish one two-corner drag in canvas-local coordinates."""
    _press(widget, start, modifiers=modifiers)
    _move(widget, end, modifiers=modifiers)
    _release(widget, end, modifiers=modifiers)


def _drag_pixels(widget, start, end, *, modifiers=Qt.NoModifier):
    """Finish a whole-field drag at two independently known image pixels."""
    _drag_canvas(widget, _point(*start), _point(*end), modifiers=modifiers)


def _accent_pixels(widget) -> int:
    """Count rendered accent pixels in the in-progress box outline."""
    image = QImage(widget.size(), QImage.Format_RGB32)
    image.fill(QColor("black"))
    widget.render(image)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint32).reshape(
        image.height(), image.bytesPerLine() // 4)
    accent = np.uint32(QColor(active_palette()["accent"]).rgb())
    return int((pixels[:, :image.width()] == accent).sum())


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    """Own a canvas with one segmentation object and no YOLO boxes."""
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(*CANVAS_SIZE)
    widget.set_image_and_mask(_image(), _mask())
    assert widget.pixmap().width() == PICTURE_SIZE
    assert widget.boxes == []
    widget.mode = MODE_BOX
    return widget


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path):
    """Own a two-field Make Masks screen for history and persistence tests."""
    folder = tmp_path / "fields"
    folder.mkdir()
    imageio.imwrite(folder / "field_00.tif", _image())
    imageio.imwrite(folder / "field_01.tif", _image() + 1)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    widget._canvas.resize(*CANVAS_SIZE)
    widget._canvas.set_image_and_mask(_image(), _mask())
    assert widget._canvas.pixmap().width() == PICTURE_SIZE
    return widget, folder


@pytest.mark.parametrize(
    ("start", "end"),
    [((12, 44), (24, 56)), ((24, 56), (12, 44))],
)
def test_a_box_drag_normalizes_corners_without_painting_the_mask(
        canvas, start, end):
    """Either drag direction gives one half-open box and leaves labels alone."""
    before = canvas.mask.copy()
    changes = []
    canvas.boxes_changed.connect(lambda *_args: changes.append(list(canvas.boxes)))

    _drag_pixels(canvas, start, end)

    assert canvas.boxes == [(0, 12, 44, 25, 57)]
    assert changes == [[(0, 12, 44, 25, 57)]]
    assert np.array_equal(canvas.mask, before)


def test_a_click_or_letterbox_drag_does_not_create_a_box(canvas):
    """The Box tool does not turn a click or canvas margin into an object."""
    point = _point(12, 44)
    _press(canvas, point)
    _release(canvas, point)
    _drag_canvas(canvas, (20, 20), (50, 70))

    assert canvas.boxes == []


def test_box_preview_is_visible_before_any_annotation_is_committed(canvas):
    """The drag shows its rectangle while the saved box list stays empty."""
    quiet = _accent_pixels(canvas)
    _press(canvas, _point(12, 44))
    _move(canvas, _point(24, 56))

    assert canvas.boxes == []
    assert _accent_pixels(canvas) > quiet

    _release(canvas, _point(24, 56))
    assert canvas.boxes == [(0, 12, 44, 25, 57)]


def test_a_box_drag_does_not_follow_navigation_to_a_new_field(canvas):
    """Releasing after the field changes cannot label the next image."""
    _press(canvas, _point(12, 44))
    _move(canvas, _point(24, 56))

    new_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    canvas.set_image_and_mask(_image() + 1, new_mask)
    _release(canvas, _point(24, 56))

    assert canvas.boxes == []
    assert np.array_equal(canvas.mask, new_mask)


def test_a_box_can_reach_the_last_image_pixel_without_exceeding_it(canvas):
    """The far edge is exclusive and clipped to the image's real bounds."""
    _drag_pixels(canvas, (60, 61), (63, 63))

    assert canvas.boxes == [(0, 60, 61, 64, 64)]


def test_zoom_and_pan_keep_boxes_in_whole_image_coordinates(canvas):
    """A box stays on its pixels while the viewport moves around it."""
    _drag_pixels(canvas, (45, 45), (50, 50))
    outside = canvas.boxes[0]

    canvas.mode = MODE_ZOOM
    _drag_pixels(canvas, (8, 8), (40, 40))
    assert canvas.boxes == [outside]
    assert canvas._viewport_bounds() == (8, 8, 41, 41)

    canvas.mode = MODE_BOX
    start = canvas._image_to_canvas(12.25, 12.25)
    end = canvas._image_to_canvas(20.25, 20.25)
    _drag_canvas(canvas, (start.x(), start.y()), (end.x(), end.y()))
    assert canvas.boxes == [outside, (0, 12, 12, 21, 21)]

    assert canvas.pan_by(2, 1)
    assert canvas.boxes == [outside, (0, 12, 12, 21, 21)]


def test_right_click_deletes_the_top_box_without_erasing_the_mask(canvas):
    """A Box-mode right click consumes the overlapping segmentation object."""
    before = canvas.mask.copy()
    _drag_pixels(canvas, (12, 12), (26, 26))
    _drag_pixels(canvas, (30, 30), (20, 20))
    first, top = canvas.boxes
    assert first != top

    hit = _point(23, 23)
    _press(canvas, hit, button=Qt.RightButton)
    _release(canvas, hit, button=Qt.RightButton)

    assert canvas.boxes == [first]
    assert np.array_equal(canvas.mask, before)


def test_ctrl_drag_creates_nested_box_without_triggering_mask_ctrl_edit(canvas):
    """The Box tool owns Ctrl+left even over a segmentation object."""
    before_mask = canvas.mask.copy()
    _drag_pixels(canvas, (12, 12), (32, 32))
    outer = canvas.boxes[0]

    _drag_pixels(canvas, (22, 22), (28, 28),
                 modifiers=Qt.ControlModifier)

    assert canvas.boxes == [outer, (0, 22, 22, 29, 29)]
    assert np.array_equal(canvas.mask, before_mask)


def test_drag_inside_moves_and_corner_drag_resizes_only_the_box(canvas):
    """A box can be corrected in place without painting its mask underneath."""
    before_mask = canvas.mask.copy()
    _drag_pixels(canvas, (12, 12), (24, 24))

    _drag_pixels(canvas, (18, 18), (22, 20))
    moved = canvas.boxes[0]
    assert moved == (0, 16, 14, 29, 27)

    _drag_pixels(canvas, (28, 26), (31, 30))
    resized = canvas.boxes[0]
    assert resized[:3] == moved[:3]
    assert resized[3] > moved[3]
    assert resized[4] > moved[4]
    assert np.array_equal(canvas.mask, before_mask)


def test_switching_draw_and_box_abandons_the_unfinished_gesture(screen):
    """A mouse release after changing tools cannot commit an old gesture."""
    widget, _folder = screen
    canvas = widget._canvas
    before = canvas.mask.copy()

    widget._set_mode(MODE_DRAW)
    _press(canvas, _point(12, 44))
    _move(canvas, _point(24, 44))
    widget._set_mode(MODE_BOX)
    _release(canvas, _point(24, 56))
    assert canvas.boxes == []
    assert np.array_equal(canvas.mask, before)

    _press(canvas, _point(12, 44))
    _move(canvas, _point(24, 56))
    widget._set_mode(MODE_DRAW)
    _release(canvas, _point(24, 56))
    assert canvas.boxes == []
    assert np.array_equal(canvas.mask, before)

    widget._set_mode(MODE_BOX)
    _drag_pixels(canvas, (12, 44), (24, 56))
    assert canvas.boxes == [(0, 12, 44, 25, 57)]


def test_prompt_and_magnifier_leave_box_mode_before_segmenting(screen,
                                                               monkeypatch):
    """The segmentation toggles do not quietly reuse the Box mouse gesture."""
    widget, _folder = screen
    canvas = widget._canvas
    before_mask = canvas.mask.copy()

    widget._set_mode(MODE_BOX)
    _press(canvas, _point(12, 44))
    _move(canvas, _point(24, 56))
    widget._btn_magnifier.setChecked(True)
    assert canvas.mode == MODE_NONE
    _release(canvas, _point(24, 56))
    assert canvas.boxes == []
    assert np.array_equal(canvas.mask, before_mask)
    widget._btn_magnifier.setChecked(False)

    monkeypatch.setattr(type(widget), "_prompt_ready", lambda self: True)
    widget._set_mode(MODE_BOX)
    _press(canvas, _point(12, 44))
    _move(canvas, _point(24, 56))
    widget._btn_prompt.setChecked(True)
    assert canvas.mode == MODE_NONE
    _release(canvas, _point(24, 56))
    assert canvas.boxes == []
    assert np.array_equal(canvas.mask, before_mask)
    widget._btn_prompt.setChecked(False)


def test_class_choice_undo_and_redo_apply_to_boxes_without_touching_masks(
        screen):
    """One named class and one box stay distinct from segmentation history."""
    widget, _folder = screen
    canvas = widget._canvas
    original_mask = canvas.mask.copy()
    widget._set_mode(MODE_BOX)

    assert widget._on_add_box_class(" infected ") == 1
    assert widget._box_classes == ["object", "infected"]
    assert widget._box_class_combo.currentData() == 1
    assert widget._on_add_box_class("infected") is None
    assert widget._on_add_box_class("INFECTED") is None
    assert widget._on_add_box_class("bad/name") is None
    assert widget._on_add_box_class("  ") is None
    assert widget._box_classes == ["object", "infected"]

    _drag_pixels(canvas, (12, 44), (24, 56))
    assert canvas.boxes == [(1, 12, 44, 25, 57)]
    widget._on_undo()
    assert canvas.boxes == []
    widget._on_redo()
    assert canvas.boxes == [(1, 12, 44, 25, 57)]

    hit = _point(18, 50)
    _press(canvas, hit)
    _release(canvas, hit)
    assert canvas.selected_box == 0
    widget._box_class_combo.setCurrentIndex(
        widget._box_class_combo.findData(0))
    assert canvas.boxes == [(0, 12, 44, 25, 57)]
    widget._on_undo()
    assert canvas.boxes == [(1, 12, 44, 25, 57)]
    assert np.array_equal(canvas.mask, original_mask)


def test_saved_boxes_and_classes_follow_their_field_on_reopen(screen, qtbot):
    """Two fields keep independent boxes when navigation and reload occur."""
    widget, folder = screen
    widget._set_mode(MODE_BOX)
    assert widget._on_add_box_class("infected") == 1
    _drag_pixels(widget._canvas, (12, 44), (24, 56))
    widget._btn_save.click()

    widget._on_next()
    assert widget._current_index == 1
    assert widget._canvas.boxes == []
    widget._box_class_combo.setCurrentIndex(
        widget._box_class_combo.findData(0))
    _drag_pixels(widget._canvas, (4, 4), (12, 12))
    second_path = widget._on_save_boxes()
    assert second_path is not None and second_path.is_file()

    widget._on_prev()
    assert widget._canvas.boxes == [(1, 12, 44, 25, 57)]
    widget._on_next()
    assert widget._canvas.boxes == [(0, 4, 4, 13, 13)]

    reopened = MakeMasksScreen()
    qtbot.addWidget(reopened)
    assert reopened._open_folder(str(folder))
    assert reopened._box_classes == ["object", "infected"]
    assert reopened._canvas.boxes == [(1, 12, 44, 25, 57)]
    reopened._on_next()
    assert reopened._canvas.boxes == [(0, 4, 4, 13, 13)]


def test_changed_source_refuses_autosave_and_keeps_the_current_field(screen):
    """Navigation cannot silently bind drawn boxes to different image bytes."""
    widget, folder = screen
    widget._set_mode(MODE_BOX)
    _drag_pixels(widget._canvas, (12, 44), (24, 56))
    expected = list(widget._canvas.boxes)
    source = folder / "field_00.tif"
    original_bytes = source.read_bytes()
    project = folder / engine.YOLO_ANNOTATIONS_NAME

    try:
        imageio.imwrite(source, np.full((IMAGE_SIZE, IMAGE_SIZE), 123,
                                        dtype=np.uint16))
        widget._on_next()

        assert widget._current_index == 0
        assert widget._canvas.boxes == expected
        assert not project.exists()
    finally:
        source.write_bytes(original_bytes)
        widget._boxes_dirty = False


def test_yolo_export_uses_normalized_xywh_and_keeps_source_image(screen,
                                                                  tmp_path):
    """A named class exports its exact box, never a converted source image."""
    widget, folder = screen
    source = folder / "field_00.tif"
    source_bytes = source.read_bytes()
    before_mask = widget._canvas.mask.copy()
    widget._set_mode(MODE_BOX)
    assert widget._on_add_box_class("infected") == 1
    _drag_pixels(widget._canvas, (12, 44), (24, 56))

    target = tmp_path / "labels.txt"
    assert widget._on_export_yolo_boxes(str(target)) == str(target)
    fields = target.read_text(encoding="utf-8").strip().split()
    assert len(fields) == 5
    assert int(fields[0]) == 1
    assert [float(value) for value in fields[1:]] == pytest.approx(
        [18.5 / 64, 50.5 / 64, 13 / 64, 13 / 64], abs=1e-6)
    metadata = json.loads((target.parent / ".classes.json").read_text(
        encoding="utf-8"))
    assert "infected" in json.dumps(metadata)
    assert source.read_bytes() == source_bytes
    assert np.array_equal(widget._canvas.mask, before_mask)


def test_exporting_a_negative_field_writes_an_empty_label_file(screen,
                                                                tmp_path):
    """A verified field with no boxes is a usable YOLO negative example."""
    widget, folder = screen
    source = folder / "field_00.tif"
    source_bytes = source.read_bytes()
    before_mask = widget._canvas.mask.copy()
    widget._set_mode(MODE_BOX)

    target = tmp_path / "negative.txt"
    assert widget._on_export_yolo_boxes(str(target)) == str(target)
    assert target.read_text(encoding="utf-8") == ""
    metadata = json.loads((target.parent / ".classes.json").read_text(
        encoding="utf-8"))
    assert "object" in json.dumps(metadata)
    assert source.read_bytes() == source_bytes
    assert np.array_equal(widget._canvas.mask, before_mask)
