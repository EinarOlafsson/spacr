"""Canvas geometry and gesture boundaries for Make Masks' Box tool."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent

from spacr.qt.screens.make_masks import MODE_BOX, MODE_BRUSH, _MaskCanvas


def _event(kind, point, *, button=Qt.LeftButton, buttons=Qt.LeftButton,
           modifiers=Qt.NoModifier):
    """Deliver a local-position mouse event without relying on screen scale."""
    pos = QPointF(*point)
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def _press(canvas, point, *, modifiers=Qt.NoModifier, button=Qt.LeftButton,
           buttons=None):
    """Start an actual canvas gesture."""
    if buttons is None:
        buttons = button
    canvas.mousePressEvent(_event(QEvent.Type.MouseButtonPress, point,
                                  button=button, buttons=buttons,
                                  modifiers=modifiers))


def _move(canvas, point, *, modifiers=Qt.NoModifier):
    """Move the held left button."""
    canvas.mouseMoveEvent(_event(QEvent.Type.MouseMove, point,
                                 button=Qt.NoButton, modifiers=modifiers))


def _release(canvas, point, *, modifiers=Qt.NoModifier):
    """Finish the held left-button gesture."""
    canvas.mouseReleaseEvent(_event(QEvent.Type.MouseButtonRelease, point,
                                    buttons=Qt.NoButton, modifiers=modifiers))


def _drag(canvas, start, end, *, modifiers=Qt.NoModifier):
    """Run one mouse drag with the same keyboard modifiers throughout."""
    _press(canvas, start, modifiers=modifiers)
    _move(canvas, end, modifiers=modifiers)
    _release(canvas, end, modifiers=modifiers)


def _pixel(canvas, x, y):
    """Choose a point inside a known whole-image pixel."""
    point = canvas._image_to_canvas(x + 0.25, y + 0.25)
    return point.x(), point.y()


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    """Own a 64-pixel field letterboxed inside a wider Qt canvas."""
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[20:36, 20:36] = 7
    widget.set_image_and_mask(image, mask)
    widget.mode = MODE_BOX
    assert widget.pixmap().width() == 400
    return widget


def test_readonly_canvas_rejects_drag_and_delete_without_touching_mask(canvas):
    """A read-only field cannot gain or lose boxes through mouse events."""
    canvas.boxes = [(0, 12, 12, 25, 25)]
    canvas.boxes_editable = False
    before_mask = canvas.mask.copy()
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _drag(canvas, _pixel(canvas, 30, 30), _pixel(canvas, 40, 40))
    _press(canvas, _pixel(canvas, 18, 18), button=Qt.RightButton)

    assert canvas.boxes == [(0, 12, 12, 25, 25)]
    assert edits == []
    assert np.array_equal(canvas.mask, before_mask)


@pytest.mark.parametrize("modifier", [Qt.ShiftModifier, Qt.AltModifier])
def test_modified_box_drag_pans_the_view_without_editing_boxes(canvas,
                                                                modifier):
    """Shift and Alt move a zoomed viewport, not an annotation corner."""
    canvas.boxes = [(0, 25, 25, 36, 36)]
    canvas.zoom_at(32, 32, 4.0)
    before_view = canvas._viewport_bounds()
    before_mask = canvas.mask.copy()
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _drag(canvas, (300, 200), (390, 260), modifiers=modifier)

    assert canvas._viewport_bounds() != before_view
    assert canvas.boxes == [(0, 25, 25, 36, 36)]
    assert edits == []
    assert canvas._pan_from is None
    assert np.array_equal(canvas.mask, before_mask)


def test_drag_from_image_into_letterbox_has_no_annotation(canvas):
    """Leaving the drawn picture cannot turn the margin into image pixels."""
    before_mask = canvas.mask.copy()
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _drag(canvas, _pixel(canvas, 12, 44), (50, 300))

    assert canvas.boxes == []
    assert edits == []
    assert canvas._box_drag is None
    assert np.array_equal(canvas.mask, before_mask)


def test_competing_right_press_cannot_delete_before_left_release(canvas):
    """A chorded right press leaves the selected box's index intact."""
    first = (0, 5, 5, 15, 15)
    second = (0, 25, 25, 35, 35)
    canvas.boxes = [first, second]
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(list(canvas.boxes)))

    _press(canvas, _pixel(canvas, 30, 30))
    _move(canvas, _pixel(canvas, 32, 31))
    _press(canvas, _pixel(canvas, 10, 10), button=Qt.RightButton,
           buttons=Qt.LeftButton | Qt.RightButton)
    _release(canvas, _pixel(canvas, 32, 31))

    moved = (0, 27, 26, 37, 36)
    assert canvas.boxes == [first, moved]
    assert edits == [[first, moved]]
    assert canvas._box_drag is None


def test_standalone_right_press_clears_a_lost_left_drag(canvas):
    """A new right gesture cancels a stale preview before deleting its hit."""
    canvas.boxes = [(0, 5, 5, 15, 15)]
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(list(canvas.boxes)))

    _press(canvas, _pixel(canvas, 40, 40))
    _move(canvas, _pixel(canvas, 50, 50))
    assert canvas._box_preview is not None
    _press(canvas, _pixel(canvas, 10, 10), button=Qt.RightButton)
    _release(canvas, _pixel(canvas, 50, 50))

    assert canvas.boxes == []
    assert edits == [[]]
    assert canvas._box_drag is None
    assert canvas._box_preview is None


def test_right_click_on_empty_image_cancels_preview_without_erasing(canvas):
    """An empty right-click dismisses a stale draft and preserves labels."""
    original = (0, 5, 5, 15, 15)
    canvas.boxes = [original]
    before_mask = canvas.mask.copy()
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _press(canvas, _pixel(canvas, 40, 40))
    _move(canvas, _pixel(canvas, 50, 50))
    assert canvas._box_preview is not None
    _press(canvas, _pixel(canvas, 40, 10), button=Qt.RightButton)
    _release(canvas, _pixel(canvas, 50, 50))

    assert canvas.boxes == [original]
    assert edits == []
    assert canvas._box_drag is None
    assert canvas._box_preview is None
    assert np.array_equal(canvas.mask, before_mask)


def test_box_pan_ignores_tiny_motion_and_a_blocked_image_edge(canvas):
    """A pan at the viewport limit cannot move boxes or create an edit."""
    original = (0, 25, 25, 36, 36)
    canvas.boxes = [original]
    canvas.zoom_at(32, 32, 4.0)
    assert canvas.pan_by(-10_000, -10_000)
    before_view = canvas._viewport_bounds()
    before_mask = canvas.mask.copy()
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _press(canvas, (300, 200), modifiers=Qt.ShiftModifier)
    _move(canvas, (300, 200), modifiers=Qt.ShiftModifier)
    assert canvas._pan_from is not None
    _move(canvas, (390, 290), modifiers=Qt.ShiftModifier)
    _release(canvas, (390, 290), modifiers=Qt.ShiftModifier)

    assert canvas._viewport_bounds() == before_view
    assert canvas.boxes == [original]
    assert edits == []
    assert canvas._pan_from is None
    assert np.array_equal(canvas.mask, before_mask)


def test_valid_preview_released_in_letterbox_is_discarded(canvas):
    """The last valid hover point cannot be mistaken for the release point."""
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _press(canvas, _pixel(canvas, 12, 44))
    _move(canvas, _pixel(canvas, 24, 56))
    assert canvas._box_preview == (0, 12, 44, 25, 57)
    _release(canvas, (50, 300))

    assert canvas.boxes == []
    assert edits == []
    assert canvas._box_drag is None
    assert canvas._box_preview is None


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((12, 12), (8, 10), (0, 8, 10, 25, 25)),
        ((24, 12), (28, 10), (0, 12, 10, 29, 25)),
        ((12, 24), (8, 28), (0, 8, 12, 25, 29)),
        ((12, 12), (30, 30), (0, 24, 24, 25, 25)),
    ],
)
def test_other_corner_resizes_keep_half_open_valid_bounds(
        canvas, start, end, expected):
    """Top and left handles resize in image pixels, even across the far edge."""
    canvas.boxes = [(0, 12, 12, 25, 25)]
    before_mask = canvas.mask.copy()
    edits = []
    canvas.boxes_changed.connect(lambda: edits.append(True))

    _drag(canvas, _pixel(canvas, *start), _pixel(canvas, *end))

    assert canvas.boxes == [expected]
    assert edits == [True]
    assert expected[1] < expected[3] and expected[2] < expected[4]
    assert np.array_equal(canvas.mask, before_mask)


def test_canvas_can_paint_before_a_field_is_loaded(qtbot, qt_theme_applied):
    """A blank screen with no pixmap ignores pending box geometry safely."""
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    widget.mode = MODE_BOX
    widget.boxes = [(0, 1, 1, 3, 3)]
    image = QImage(widget.size(), QImage.Format_RGB32)

    widget.render(image)

    assert widget._box_hit(None) is None
    assert widget._image_to_canvas(1, 1) is None


def test_cancelling_a_brush_stroke_closes_its_single_edit(canvas):
    """Switching tools mid-stroke records the painted pixels just once."""
    canvas.mode = MODE_BRUSH
    before = canvas.mask.copy()
    finished = []
    canvas.stroke_finished.connect(lambda: finished.append(canvas.last_edit))
    start = _pixel(canvas, 45, 45)
    end = _pixel(canvas, 48, 45)

    _press(canvas, start)
    _move(canvas, end)
    assert canvas._stroke_in_progress
    canvas.cancel_gesture()
    _release(canvas, end)

    assert len(finished) == 1
    assert finished[0]["kind"] == "paint"
    assert not canvas._stroke_in_progress
    assert not np.array_equal(canvas.mask, before)
