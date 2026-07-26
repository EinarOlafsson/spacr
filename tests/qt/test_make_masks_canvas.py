"""Behavioural tests for ``spacr.qt.screens.make_masks``.

Covers the canvas interaction layer (coordinate mapping, brush / erase /
erase-object / magic-wand / zoom-rectangle mouse handling, the zoom
overlay paint, resize refit) and the screen layer (folder navigation,
undo/redo wiring, object ops, save, and the *headless-safe* error
reporting added after a real hang was reproduced — see
``test_message_box_would_block_headless``).

Everything runs under QT_QPA_PLATFORM=offscreen with no modal dialogs.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (
    MODE_BRUSH,
    MODE_ERASE,
    MODE_ERASE_OBJECT,
    MODE_NONE,
    MODE_WAND_ADD,
    MODE_WAND_ERASE,
    MODE_ZOOM,
    MakeMasksScreen,
    _MaskCanvas,
    is_headless,
)
from spacr.qt.theme import PALETTE


# ---------------------------------------------------------------------------
# Geometry helpers
#
# The canvas is pinned to 600x400 and fed a 64x64 image. refresh() scales
# the composited pixmap to fit while keeping the aspect ratio, so the
# pixmap is 400x400, horizontally centred with a 100 px margin either
# side. That makes the canvas->image mapping exactly:
#     img_x = (canvas_x - 100) * 64/400 ,  img_y = canvas_y * 64/400
# ---------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2      # 100


def canvas_xy(img_x: float, img_y: float) -> tuple:
    """Canvas-local point that maps onto full-image pixel (img_x, img_y)."""
    return (MARGIN_X + img_x * PIXMAP_N / IMG_N, img_y * PIXMAP_N / IMG_N)


def _evt(kind, x, y, buttons=Qt.LeftButton, button=Qt.LeftButton):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def press(x, y):
    return _evt(QEvent.Type.MouseButtonPress, x, y)


def move(x, y, buttons=Qt.LeftButton):
    return _evt(QEvent.Type.MouseMove, x, y, buttons=buttons,
                button=Qt.NoButton)


def release(x, y):
    return _evt(QEvent.Type.MouseButtonRelease, x, y,
                buttons=Qt.NoButton, button=Qt.LeftButton)


def accent_pixels(qimg) -> int:
    """Count pixels painted in the theme accent colour (the zoom rect)."""
    im = qimg.convertToFormat(QImage.Format_RGB32)
    arr = np.frombuffer(im.constBits(), dtype=np.uint32).reshape(
        im.height(), im.bytesPerLine() // 4)
    return int((arr == np.uint32(QColor(PALETTE["accent"]).rgb())).sum())


def block_image() -> np.ndarray:
    """64x64 uint16 image: black background, one uniform bright square."""
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[20:40, 20:40] = 30000
    return img


def ramp_image() -> np.ndarray:
    """64x64 uint16 linear ramp — has real dynamic range for normalize."""
    col = np.linspace(0, 65535, IMG_N, dtype=np.float64)
    return np.tile(col, (IMG_N, 1)).astype(np.uint16)


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    """A _MaskCanvas at a known size holding a known image + blank mask."""
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.set_image_and_mask(block_image(), np.zeros((IMG_N, IMG_N), np.uint8))
    assert c.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    return c


# ---------------------------------------------------------------------------
# Image folder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def folder_3(tmp_path: Path) -> Path:
    folder = tmp_path / "three"
    folder.mkdir()
    rng = np.random.default_rng(7)
    for i in range(3):
        imageio.imwrite(folder / f"img_{i:02d}.tif",
                        rng.integers(0, 65535, (IMG_N, IMG_N), dtype=np.uint16))
    return folder


@pytest.fixture
def folder_corrupt_second(tmp_path: Path) -> Path:
    """img_00.tif is a real image; img_01.tif is unreadable garbage."""
    folder = tmp_path / "corrupt"
    folder.mkdir()
    imageio.imwrite(folder / "img_00.tif", block_image())
    (folder / "img_01.tif").write_bytes(b"this is definitely not a TIFF")
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder_3: Path):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._open_folder(str(folder_3))
    # Pin the canvas geometry so canvas_xy() is valid for this screen too.
    s._canvas.resize(CANVAS_W, CANVAS_H)
    s._canvas.refresh()
    assert s._canvas.pixmap().width() == PIXMAP_N
    return s


# ===========================================================================
# The bug that motivated the headless-safe messaging
# ===========================================================================

def test_message_box_would_block_headless(qt_theme_applied):
    """Documents *why* make_masks must not call QMessageBox directly.

    Under the offscreen platform plugin nobody can dismiss a modal box, so
    ``QMessageBox.exec()`` never returns. We assert the platform really is
    headless rather than actually hanging the suite to prove it.
    """
    from PySide6.QtWidgets import QApplication
    assert QApplication.instance().platformName() == "offscreen"
    assert is_headless() is True


@pytest.mark.parametrize("platform_name, expected", [
    ("offscreen", True),
    ("minimal", True),
    ("vnc", True),
    ("", True),
    ("xcb", False),
    ("wayland", False),
    ("cocoa", False),
])
def test_is_headless_classifies_platform(monkeypatch, platform_name, expected):
    class _App:
        @staticmethod
        def platformName():
            return platform_name

    monkeypatch.setattr(mm, "QApplication",
                        type("Q", (), {"instance": staticmethod(lambda: _App)}))
    assert is_headless() is expected


def test_is_headless_true_without_qapplication(monkeypatch):
    monkeypatch.setattr(mm, "QApplication",
                        type("Q", (), {"instance": staticmethod(lambda: None)}))
    assert is_headless() is True


def test_is_headless_true_when_platform_name_raises(monkeypatch):
    class _App:
        @staticmethod
        def platformName():
            raise RuntimeError("platform plugin gone")

    monkeypatch.setattr(mm, "QApplication",
                        type("Q", (), {"instance": staticmethod(lambda: _App)}))
    assert is_headless() is True


# ===========================================================================
# Canvas — viewport / refresh / coordinate mapping
# ===========================================================================

def test_viewport_bounds_without_mask_is_degenerate(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    assert c._viewport_bounds() == (0, 0, 0, 0)
    assert c.is_zoomed() is False


def test_viewport_bounds_full_then_zoomed(canvas):
    assert canvas._viewport_bounds() == (0, 0, IMG_N, IMG_N)
    canvas._zoom_x0, canvas._zoom_y0 = 16, 12
    canvas._zoom_x1, canvas._zoom_y1 = 48, 44
    assert canvas._viewport_bounds() == (16, 12, 48, 44)
    assert canvas.is_zoomed() is True


def test_refresh_without_data_leaves_pixmap_unset(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.refresh()
    assert c.pixmap().isNull()
    # image alone is not enough — the mask is still missing
    c.image = block_image()
    c.refresh()
    assert c.pixmap().isNull()


def test_refresh_with_empty_zoom_region_keeps_previous_pixmap(canvas):
    before = canvas.pixmap().toImage()
    canvas._zoom_x0 = canvas._zoom_x1 = 10   # zero-width viewport
    canvas._zoom_y0, canvas._zoom_y1 = 10, 20
    canvas.refresh()
    assert canvas.pixmap().toImage() == before


def test_refresh_renders_mask_colour_over_image(canvas):
    plain = canvas.pixmap().toImage()
    canvas.mask[20:40, 20:40] = 3
    canvas.refresh()
    tinted = canvas.pixmap().toImage()
    assert tinted != plain, "painting the mask must change the composite"
    # A pixel inside the labelled square is no longer pure grey.
    cx, cy = canvas_xy(30, 30)
    px = QColor(tinted.pixel(int(cx) - MARGIN_X, int(cy)))
    assert not (px.red() == px.green() == px.blue()), (
        f"expected a coloured overlay, got grey {px.getRgb()}"
    )


def test_canvas_to_image_maps_and_rejects_margins(canvas):
    assert canvas._canvas_to_image(*canvas_xy(32, 32)) == (32, 32)
    assert canvas._canvas_to_image(*canvas_xy(0, 0)) == (0, 0)
    assert canvas._canvas_to_image(*canvas_xy(63, 63)) == (63, 63)
    # Left / right letterbox margins and below the pixmap are all outside.
    assert canvas._canvas_to_image(MARGIN_X - 1, 200) is None
    assert canvas._canvas_to_image(MARGIN_X + PIXMAP_N, 200) is None
    assert canvas._canvas_to_image(300, PIXMAP_N) is None


def test_canvas_to_image_none_without_mask_or_pixmap(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    # No mask, no pixmap.
    assert c._canvas_to_image(300, 200) is None
    # Mask present but nothing rendered yet -> QLabel.pixmap() is a *null*
    # QPixmap (never None); the guard has to catch that.
    c.mask = np.zeros((IMG_N, IMG_N), np.uint8)
    assert c.pixmap() is not None and c.pixmap().isNull()
    assert c._canvas_to_image(300, 200) is None


def test_canvas_to_image_honours_zoom(canvas):
    canvas._zoom_x0, canvas._zoom_y0 = 16, 16
    canvas._zoom_x1, canvas._zoom_y1 = 48, 48
    canvas.refresh()
    # Centre of the zoomed 32x32 viewport is full-image pixel 32.
    assert canvas._canvas_to_image(*canvas_xy(32, 32)) == (32, 32)
    # Top-left of the viewport now maps to image pixel 16, not 0.
    assert canvas._canvas_to_image(MARGIN_X, 0) == (16, 16)


def test_mask_radius_scales_with_zoom(canvas):
    canvas.brush_radius = 100
    # Unzoomed: 100 screen px over a 400 px pixmap showing 64 image px.
    assert canvas._mask_radius_for_brush() == 16
    canvas._zoom_x0, canvas._zoom_y0 = 16, 16
    canvas._zoom_x1, canvas._zoom_y1 = 48, 48
    canvas.refresh()
    assert canvas._mask_radius_for_brush() == 8
    # Never smaller than a single pixel.
    canvas.brush_radius = 1
    assert canvas._mask_radius_for_brush() == 1


def test_mask_radius_without_pixmap_returns_raw_radius(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.brush_radius = 17
    assert c._mask_radius_for_brush() == 17          # no mask
    c.mask = np.zeros((IMG_N, IMG_N), np.uint8)
    assert c._mask_radius_for_brush() == 17          # null pixmap


# ===========================================================================
# Canvas — brush / erase
# ===========================================================================

def test_brush_press_paints_and_signals(canvas):
    started, finished = [], []
    canvas.stroke_started.connect(lambda: started.append(1))
    canvas.stroke_finished.connect(lambda: finished.append(1))
    canvas.mode = MODE_BRUSH
    canvas.brush_radius = 40           # -> 6 image px half-width

    canvas.mousePressEvent(press(*canvas_xy(32, 32)))
    assert started == [1] and finished == []
    radius = 40 * IMG_N // PIXMAP_N
    assert canvas.mask[32, 32] == 255
    assert canvas.mask[32 - radius, 32 - radius] == 255
    assert canvas.mask[0, 0] == 0
    painted = int((canvas.mask > 0).sum())
    assert painted == (2 * radius) ** 2, f"got {painted}px"

    canvas.mouseReleaseEvent(release(*canvas_xy(32, 32)))
    assert finished == [1]
    assert canvas._last_pt is None
    # A second release must not double-fire.
    canvas.mouseReleaseEvent(release(*canvas_xy(32, 32)))
    assert finished == [1]


def test_brush_drag_paints_a_connected_line(canvas):
    canvas.mode = MODE_BRUSH
    canvas.brush_radius = 10           # -> 1 image px
    canvas.mousePressEvent(press(*canvas_xy(10, 32)))
    canvas.mouseMoveEvent(move(*canvas_xy(50, 32)))
    canvas.mouseReleaseEvent(release(*canvas_xy(50, 32)))

    row = canvas.mask[32, :]
    assert row[10] == 255 and row[50] == 255
    # Bresenham stamped every pixel in between — no gaps.
    assert (row[10:51] > 0).all(), f"gap in stroke: {np.where(row[10:51] == 0)}"
    assert row[5] == 0 and row[60] == 0
    # Exactly one connected component.
    _, n = engine.label(canvas.mask > 0)
    assert n == 1


def test_erase_mode_zeroes_painted_pixels(canvas):
    canvas.mask[25:35, 25:35] = 200
    canvas.mode = MODE_ERASE
    canvas.brush_radius = 20           # -> 3 image px
    canvas.mousePressEvent(press(*canvas_xy(30, 30)))
    canvas.mouseReleaseEvent(release(*canvas_xy(30, 30)))
    assert canvas.mask[30, 30] == 0
    assert canvas.mask[25, 25] == 200, "erase must not wipe the whole object"


def test_brush_move_without_button_does_nothing(canvas):
    canvas.mode = MODE_BRUSH
    before = canvas.mask.copy()
    canvas.mouseMoveEvent(move(*canvas_xy(32, 32), buttons=Qt.NoButton))
    assert (canvas.mask == before).all()


def test_mouse_events_are_inert_without_a_mask(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.mode = MODE_BRUSH
    started = []
    c.stroke_started.connect(lambda: started.append(1))
    c.mousePressEvent(press(300, 200))
    c.mouseMoveEvent(move(300, 200))
    assert started == []
    assert c.mask is None


def test_mode_none_press_is_ignored(canvas):
    started = []
    canvas.stroke_started.connect(lambda: started.append(1))
    canvas.mode = MODE_NONE
    canvas.mousePressEvent(press(*canvas_xy(32, 32)))
    assert started == []
    assert not canvas.mask.any()


def test_press_outside_pixmap_does_not_start_a_stroke(canvas):
    started = []
    canvas.stroke_started.connect(lambda: started.append(1))
    canvas.mode = MODE_BRUSH
    canvas.mousePressEvent(press(MARGIN_X - 20, 200))   # in the letterbox
    assert started == []
    assert not canvas.mask.any()


def test_drag_entering_the_canvas_still_opens_a_stroke(canvas):
    """Regression: a drag that *starts* in the letterbox margin used to
    paint without ever emitting stroke_started, so the resulting edit
    was never pushed onto the undo history."""
    started, finished = [], []
    canvas.stroke_started.connect(lambda: started.append(1))
    canvas.stroke_finished.connect(lambda: finished.append(1))
    canvas.mode = MODE_BRUSH
    canvas.brush_radius = 10

    canvas.mousePressEvent(press(MARGIN_X - 20, 200))    # outside -> no start
    assert started == []
    canvas.mouseMoveEvent(move(*canvas_xy(32, 32)))      # now inside
    assert started == [1], "moving into the image must open the stroke"
    canvas.mouseReleaseEvent(release(*canvas_xy(32, 32)))
    assert finished == [1], "the stroke must close so history records it"
    assert canvas.mask[32, 32] == 255


def test_drag_leaving_the_canvas_keeps_the_stroke_open(canvas):
    canvas.mode = MODE_BRUSH
    canvas.brush_radius = 10
    canvas.mousePressEvent(press(*canvas_xy(32, 32)))
    canvas.mouseMoveEvent(move(MARGIN_X - 30, 200))      # out into the margin
    assert canvas._last_pt == QPoint(32, 32), "last point must be preserved"
    assert canvas._stroke_in_progress is True


# ===========================================================================
# Canvas — erase-object and magic wand
# ===========================================================================

def test_erase_object_removes_only_the_clicked_label(canvas):
    canvas.mask[20:40, 20:40] = 4
    canvas.mask[5:10, 5:10] = 9
    finished = []
    canvas.stroke_finished.connect(lambda: finished.append(1))
    canvas.mode = MODE_ERASE_OBJECT
    canvas.mousePressEvent(press(*canvas_xy(30, 30)))
    assert not (canvas.mask == 4).any()
    assert int((canvas.mask == 9).sum()) == 25
    assert finished == [1], "object erase is a complete stroke on press"


def test_erase_object_on_background_is_a_noop(canvas):
    canvas.mask[20:40, 20:40] = 4
    before = canvas.mask.copy()
    canvas.mode = MODE_ERASE_OBJECT
    canvas.mousePressEvent(press(*canvas_xy(2, 2)))     # background
    assert (canvas.mask == before).all()


def test_wand_add_fills_the_uniform_square(canvas):
    canvas.mode = MODE_WAND_ADD
    canvas.wand_tolerance = 100.0
    canvas.wand_max_pixels = 10_000
    finished = []
    canvas.stroke_finished.connect(lambda: finished.append(1))
    canvas.mousePressEvent(press(*canvas_xy(30, 30)))
    assert finished == [1]
    assert canvas.mask[30, 30] == 255
    # The bright square is 20x20 and the surrounding black is far outside
    # the tolerance, so exactly 400 px get filled.
    assert int((canvas.mask > 0).sum()) == 400
    assert canvas.mask[19, 19] == 0


def test_wand_erase_clears_the_uniform_square(canvas):
    canvas.mask[:] = 77
    canvas.mode = MODE_WAND_ERASE
    canvas.wand_tolerance = 100.0
    canvas.wand_max_pixels = 10_000
    canvas.mousePressEvent(press(*canvas_xy(30, 30)))
    assert canvas.mask[30, 30] == 0
    assert int((canvas.mask == 0).sum()) == 400
    assert canvas.mask[0, 0] == 77


def test_wand_respects_max_pixels(canvas):
    canvas.mode = MODE_WAND_ADD
    canvas.wand_tolerance = 100.0
    canvas.wand_max_pixels = 25
    canvas.mousePressEvent(press(*canvas_xy(30, 30)))
    filled = int((canvas.mask > 0).sum())
    assert 0 < filled <= 25, f"budget of 25 px exceeded: {filled}"


# ===========================================================================
# Canvas — zoom rectangle + overlay paint
# ===========================================================================

def test_zoom_drag_commits_viewport_and_emits(canvas):
    seen = []
    canvas.zoom_changed.connect(seen.append)
    canvas.mode = MODE_ZOOM
    canvas.mousePressEvent(press(*canvas_xy(8, 8)))
    canvas.mouseMoveEvent(move(*canvas_xy(56, 56)))
    assert canvas._zoom_drag_end is not None
    canvas.mouseReleaseEvent(release(*canvas_xy(56, 56)))

    assert seen == [True]
    assert canvas.is_zoomed() is True
    assert canvas._viewport_bounds() == (8, 8, 57, 57)
    assert canvas._zoom_drag_start is None
    # The rendered pixmap now shows only the sub-region.
    assert canvas.pixmap().width() == PIXMAP_N


def test_zoom_drag_smaller_than_five_pixels_is_ignored(canvas):
    seen = []
    canvas.zoom_changed.connect(seen.append)
    canvas.mode = MODE_ZOOM
    canvas.mousePressEvent(press(*canvas_xy(32, 32)))
    canvas.mouseMoveEvent(move(*canvas_xy(33, 33)))
    canvas.mouseReleaseEvent(release(*canvas_xy(33, 33)))
    assert seen == []
    assert canvas.is_zoomed() is False


def test_zoom_drag_released_off_canvas_is_discarded(canvas):
    seen = []
    canvas.zoom_changed.connect(seen.append)
    canvas.mode = MODE_ZOOM
    canvas.mousePressEvent(press(*canvas_xy(8, 8)))
    canvas._zoom_drag_end = QPoint(5, 200)        # inside the letterbox
    canvas.mouseReleaseEvent(release(5, 200))
    assert seen == []
    assert canvas.is_zoomed() is False
    assert canvas._zoom_drag_start is None


def test_zoom_move_without_button_does_not_extend_rect(canvas):
    canvas.mode = MODE_ZOOM
    canvas.mousePressEvent(press(*canvas_xy(8, 8)))
    start_end = canvas._zoom_drag_end
    canvas.mouseMoveEvent(move(*canvas_xy(56, 56), buttons=Qt.NoButton))
    assert canvas._zoom_drag_end == start_end


def test_reset_zoom_emits_once_and_only_when_zoomed(canvas):
    seen = []
    canvas.zoom_changed.connect(seen.append)
    canvas._zoom_x0, canvas._zoom_y0 = 8, 8
    canvas._zoom_x1, canvas._zoom_y1 = 40, 40
    canvas.reset_zoom()
    assert seen == [False]
    assert canvas._viewport_bounds() == (0, 0, IMG_N, IMG_N)
    # Already un-zoomed -> no further signal.
    canvas.reset_zoom()
    assert seen == [False]
    # silent=True suppresses the signal entirely.
    canvas._zoom_x0, canvas._zoom_y0 = 8, 8
    canvas._zoom_x1, canvas._zoom_y1 = 40, 40
    canvas.reset_zoom(silent=True)
    assert seen == [False]
    assert canvas.is_zoomed() is False


def test_paint_event_draws_the_dashed_zoom_rectangle(canvas):
    canvas.mode = MODE_ZOOM
    plain = canvas.grab().toImage()
    assert accent_pixels(plain) == 0

    canvas._zoom_drag_start = QPoint(150, 50)
    canvas._zoom_drag_end = QPoint(450, 350)
    with_rect = canvas.grab().toImage()
    # A dashed 2px border on a 300x300 rect — a few thousand accent px.
    assert accent_pixels(with_rect) > 500, "no zoom rectangle was drawn"
    assert with_rect != plain


def test_paint_event_skips_rectangle_outside_zoom_mode(canvas):
    canvas._zoom_drag_start = QPoint(150, 50)
    canvas._zoom_drag_end = QPoint(450, 350)
    canvas.mode = MODE_BRUSH
    assert accent_pixels(canvas.grab().toImage()) == 0
    # …and in zoom mode with no drag in flight.
    canvas.mode = MODE_ZOOM
    canvas._zoom_drag_start = None
    assert accent_pixels(canvas.grab().toImage()) == 0
    canvas._zoom_drag_start = QPoint(150, 50)
    canvas._zoom_drag_end = None
    assert accent_pixels(canvas.grab().toImage()) == 0


def test_resize_refits_the_pixmap(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.set_image_and_mask(block_image(), np.zeros((IMG_N, IMG_N), np.uint8))
    c.show()
    qtbot.wait(20)
    assert c.pixmap().width() == PIXMAP_N
    c.resize(900, 700)
    qtbot.waitUntil(lambda: c.pixmap().width() == 700, timeout=3000)
    assert c.pixmap().height() == 700
    c.hide()


# ===========================================================================
# Screen — messaging seams (no modal ever reached under offscreen)
# ===========================================================================

class _FakeBox:
    """Stand-in for QMessageBox used to assert the *display* code path."""
    Yes = "YES"
    No = "NO"
    answer = "YES"
    calls: list = []

    @staticmethod
    def warning(parent, title, text):
        _FakeBox.calls.append(("warning", title, text))

    @staticmethod
    def question(parent, title, text):
        _FakeBox.calls.append(("question", title, text))
        return _FakeBox.answer


@pytest.fixture
def fake_box(monkeypatch):
    _FakeBox.calls = []
    _FakeBox.answer = _FakeBox.Yes
    monkeypatch.setattr(mm, "QMessageBox", _FakeBox)
    monkeypatch.setattr(mm, "is_headless", lambda: False)
    return _FakeBox


def test_warn_falls_back_to_status_line_when_headless(screen, caplog):
    with caplog.at_level("WARNING", logger="spacr.qt.make_masks"):
        screen._warn("Load failed", "boom")
    assert screen._status_label.text() == "Load failed: boom"
    assert "Load failed: boom" in caplog.text


def test_warn_uses_message_box_when_a_display_exists(screen, fake_box):
    screen._warn("Save failed", "disk on fire")
    assert fake_box.calls == [("warning", "Save failed", "disk on fire")]
    assert screen._status_label.text() == "Save failed: disk on fire"


def test_confirm_refuses_when_headless(screen):
    assert screen._confirm("Clear mask", "sure?") is False
    assert "no display" in screen._status_label.text()


def test_confirm_returns_user_answer_when_a_display_exists(screen, fake_box):
    assert screen._confirm("Clear mask", "sure?") is True
    fake_box.answer = fake_box.No
    assert screen._confirm("Clear mask", "sure?") is False
    assert [c[0] for c in fake_box.calls] == ["question", "question"]


# ===========================================================================
# Screen — clear mask
# ===========================================================================

def test_clear_mask_is_not_performed_without_confirmation(screen):
    screen._canvas.mask[10:20, 10:20] = 255
    before = screen._canvas.mask.copy()
    screen._on_clear_mask()                        # headless -> declined
    assert (screen._canvas.mask == before).all()
    assert int((screen._canvas.mask > 0).sum()) == 100


def test_clear_mask_runs_when_the_user_says_yes(screen, fake_box):
    screen._canvas.mask[10:20, 10:20] = 255
    screen._on_clear_mask()
    assert not screen._canvas.mask.any()
    assert fake_box.calls[0][1] == "Clear mask"


def test_clear_mask_declined_by_the_user(screen, fake_box):
    fake_box.answer = fake_box.No
    screen._canvas.mask[10:20, 10:20] = 255
    screen._on_clear_mask()
    assert int((screen._canvas.mask > 0).sum()) == 100


def test_clear_mask_public_api_zeroes_and_is_undoable(screen):
    screen._canvas.mask[10:20, 10:20] = 255
    screen._history.push(screen._canvas.mask)
    screen.clear_mask()
    assert not screen._canvas.mask.any()
    assert screen._btn_undo.isEnabled()
    screen._on_undo()
    assert int((screen._canvas.mask > 0).sum()) == 100


def test_clear_mask_without_an_image_is_a_noop(qtbot, qt_theme_applied):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._on_clear_mask()
    assert s._canvas.mask is None
    assert s._status_label.text() == "Ready."


# ===========================================================================
# Screen — load failure must not leave a stale mask behind
# ===========================================================================

def test_load_failure_clears_canvas_and_blocks_a_stale_save(
        qtbot, qt_theme_applied, folder_corrupt_second: Path):
    """Regression: on a read error the canvas kept the *previous* field's
    mask while _current_index already pointed at the failed file, so
    Save wrote the wrong mask out under the new filename."""
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._open_folder(str(folder_corrupt_second))
    assert s._image_files == ["img_00.tif", "img_01.tif"]
    s._canvas.mask[10:30, 10:30] = 255
    stale = s._canvas.mask.copy()
    assert stale.any()

    s._on_next()                                   # img_01.tif is garbage

    assert s._current_index == 1
    assert s._canvas.mask is None, "stale mask left on the canvas"
    assert s._canvas.image is None
    assert s._status_label.text().startswith("Load failed:")
    assert not s._history.can_undo()
    assert not s._btn_reset_zoom.isEnabled()

    s._on_save()
    assert not (folder_corrupt_second / "masks" / "img_01.tif").exists(), (
        "the previous field's mask was written under the failed filename"
    )


def test_save_failure_is_reported_not_raised(qtbot, qt_theme_applied,
                                             folder_3: Path):
    # A regular file where the masks/ directory needs to go.
    (folder_3 / "masks").write_text("not a directory")
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._open_folder(str(folder_3))
    s._canvas.mask[5:15, 5:15] = 255
    s._on_save()
    assert s._status_label.text().startswith("Save failed:")
    assert (folder_3 / "masks").is_file()


def test_open_folder_without_images_reports_and_keeps_empty_state(
        qtbot, qt_theme_applied, tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "readme.txt").write_text("no pictures here")
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._open_folder(str(empty))
    assert s._image_files == []
    assert s._folder == ""
    assert s._status_label.text().startswith("No images:")
    assert s._body_stack.currentWidget() is s._empty_state
    assert not s._btn_save.isEnabled()


def test_load_current_without_files_is_a_noop(qtbot, qt_theme_applied):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._load_current()
    assert s._canvas.image is None
    assert s._status_label.text() == "Ready."


# ===========================================================================
# Screen — folder picker
# ===========================================================================

def test_pick_folder_opens_the_chosen_directory(qtbot, qt_theme_applied,
                                                monkeypatch, folder_3: Path):
    seen = {}

    class _Dlg:
        @staticmethod
        def getExistingDirectory(parent, caption, start):
            seen["caption"] = caption
            seen["start"] = start
            return str(folder_3)

    monkeypatch.setattr(mm, "QFileDialog", _Dlg)
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._on_pick_folder()
    assert seen["caption"] == "Pick images folder"
    assert seen["start"] == os.getcwd()          # no folder open yet
    assert s._folder == str(folder_3)
    assert len(s._image_files) == 3
    assert s._body_stack.currentWidget() is s._body_splitter


def test_pick_folder_cancelled_changes_nothing(qtbot, qt_theme_applied,
                                               monkeypatch):
    monkeypatch.setattr(
        mm, "QFileDialog",
        type("D", (), {"getExistingDirectory": staticmethod(
            lambda *a, **k: "")}))
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._on_pick_folder()
    assert s._folder == ""
    assert s._image_files == []
    assert s._body_stack.currentWidget() is s._empty_state


# ===========================================================================
# Screen — navigation, tool plumbing, history
# ===========================================================================

def test_navigation_updates_status_and_clamps_at_both_ends(screen):
    assert screen._status_label.text() == "img_00.tif  (1/3)"
    screen._on_prev()                             # already first
    assert screen._current_index == 0
    screen._on_next()
    assert screen._status_label.text() == "img_01.tif  (2/3)"
    screen._on_next()
    screen._on_next()                             # already last
    assert screen._current_index == 2
    assert screen._status_label.text() == "img_02.tif  (3/3)"
    screen._on_prev()
    assert screen._current_index == 1


def test_navigation_is_inert_before_a_folder_is_open(qtbot, qt_theme_applied):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._on_next()
    s._on_prev()
    s._on_save()
    assert s._current_index == 0
    assert s._status_label.text() == "Ready."


def test_brush_slider_syncs_canvas_and_label(screen):
    screen._brush_slider.setValue(42)
    assert screen._canvas.brush_radius == 42
    assert screen._brush_size_label.text() == "42 px"


def test_wand_spinboxes_sync_to_canvas(screen):
    screen._wand_tol.setValue(250.0)
    screen._wand_max.setValue(4321)
    assert screen._canvas.wand_tolerance == 250.0
    assert screen._canvas.wand_max_pixels == 4321


def test_normalize_spinboxes_change_the_rendered_image(qtbot, qt_theme_applied,
                                                       tmp_path: Path):
    folder = tmp_path / "ramp"
    folder.mkdir()
    imageio.imwrite(folder / "ramp.tif", ramp_image())
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._open_folder(str(folder))
    s._canvas.resize(CANVAS_W, CANVAS_H)
    s._canvas.refresh()
    before = s._canvas.pixmap().toImage()

    s._norm_lo.setValue(40.0)
    assert s._canvas.norm_lo == 40.0
    after = s._canvas.pixmap().toImage()
    assert after != before, "clipping the low percentile must change output"

    s._norm_hi.setValue(60.0)
    assert s._canvas.norm_hi == 60.0
    assert s._canvas.pixmap().toImage() != after


def test_zoom_changed_toggles_button_and_status(screen):
    screen._on_zoom_changed(True)
    assert screen._btn_reset_zoom.isEnabled()
    assert screen._status_label.text() == "Zoomed — press Esc to reset"
    screen._on_zoom_changed(False)
    assert not screen._btn_reset_zoom.isEnabled()
    assert screen._status_label.text() == "Zoom reset"


def test_reset_zoom_button_clears_canvas_viewport(screen):
    screen._canvas._zoom_x0, screen._canvas._zoom_y0 = 4, 4
    screen._canvas._zoom_x1, screen._canvas._zoom_y1 = 40, 40
    screen._on_reset_zoom()
    assert not screen._canvas.is_zoomed()
    assert screen._status_label.text() == "Zoom reset"


def test_undo_and_redo_are_noops_with_a_single_snapshot(screen):
    before = screen._canvas.mask.copy()
    assert not screen._history.can_undo()
    screen._on_undo()
    screen._on_redo()
    assert (screen._canvas.mask == before).all()
    assert not screen._btn_undo.isEnabled()
    assert not screen._btn_redo.isEnabled()


def test_stroke_finished_pushes_history_and_enables_undo(screen):
    assert not screen._btn_undo.isEnabled()
    screen._canvas.mask[10:20, 10:20] = 255
    screen._on_stroke_started()                    # deliberately a no-op
    screen._canvas.stroke_finished.emit()
    assert screen._btn_undo.isEnabled()
    screen._on_undo()
    assert not screen._canvas.mask.any(), "undo must return to the loaded mask"
    assert screen._btn_redo.isEnabled()
    screen._on_redo()
    assert int((screen._canvas.mask > 0).sum()) == 100


def test_brush_stroke_through_the_canvas_lands_in_history(screen):
    """Full path: set the brush tool, drag on the canvas, undo the result."""
    screen._set_mode(MODE_BRUSH)
    screen._brush_slider.setValue(10)
    c = screen._canvas
    c.mousePressEvent(press(*canvas_xy(20, 32)))
    c.mouseMoveEvent(move(*canvas_xy(44, 32)))
    c.mouseReleaseEvent(release(*canvas_xy(44, 32)))
    assert (c.mask[32, 20:45] > 0).all()
    assert screen._btn_undo.isEnabled()
    screen._on_undo()
    assert not c.mask.any()


def test_drag_from_the_margin_is_undoable(screen):
    """Companion to the canvas-level regression: an edit begun off the
    pixmap must still be recoverable with undo."""
    screen._set_mode(MODE_BRUSH)
    screen._brush_slider.setValue(10)
    c = screen._canvas
    c.mousePressEvent(press(MARGIN_X - 25, 200))
    c.mouseMoveEvent(move(*canvas_xy(32, 32)))
    c.mouseReleaseEvent(release(*canvas_xy(32, 32)))
    assert c.mask[32, 32] == 255
    assert screen._btn_undo.isEnabled(), "off-canvas drag was never recorded"
    screen._on_undo()
    assert not c.mask.any()


def test_stroke_finished_without_a_mask_pushes_nothing(qtbot,
                                                       qt_theme_applied):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    assert s._canvas.mask is None
    s._canvas.stroke_finished.emit()
    assert not s._history.can_undo()
    assert not s._history.can_redo()
    assert not s._btn_undo.isEnabled()


def _shortcut(widget, sequence: str):
    from PySide6.QtGui import QKeySequence, QShortcut
    want = QKeySequence(sequence)
    found = [sc for sc in widget.findChildren(QShortcut) if sc.key() == want]
    assert len(found) == 1, f"expected exactly one {sequence!r} shortcut"
    return found[0]


@pytest.mark.parametrize("sequence, mode", [
    ("B", MODE_BRUSH),
    ("E", MODE_ERASE),
    ("W", MODE_WAND_ADD),
    ("Z", MODE_ZOOM),
])
def test_tool_shortcuts_select_the_mode(screen, sequence, mode):
    _shortcut(screen, sequence).activated.emit()
    assert screen._canvas.mode == mode
    assert screen._mode_buttons[mode].isChecked()


def test_navigation_and_save_shortcuts(screen, folder_3: Path):
    _shortcut(screen, "Right").activated.emit()
    assert screen._current_index == 1
    _shortcut(screen, "Left").activated.emit()
    assert screen._current_index == 0

    screen._canvas.mask[4:14, 4:14] = 255
    _shortcut(screen, "Ctrl+S").activated.emit()
    saved = folder_3 / "masks" / "img_00.tif"
    assert saved.is_file()
    assert int((imageio.imread(saved) > 0).sum()) == 100


def test_undo_redo_and_escape_shortcuts(screen):
    screen._canvas.mask[6:16, 6:16] = 255
    screen._canvas.stroke_finished.emit()
    _shortcut(screen, "Ctrl+Z").activated.emit()
    assert not screen._canvas.mask.any()
    _shortcut(screen, "Ctrl+Y").activated.emit()
    assert int((screen._canvas.mask > 0).sum()) == 100
    _shortcut(screen, "Ctrl+Shift+Z").activated.emit()   # redo alias, empty now
    assert int((screen._canvas.mask > 0).sum()) == 100

    screen._canvas._zoom_x0, screen._canvas._zoom_y0 = 4, 4
    screen._canvas._zoom_x1, screen._canvas._zoom_y1 = 40, 40
    _shortcut(screen, "Escape").activated.emit()
    assert not screen._canvas.is_zoomed()


def test_set_mode_checks_exactly_one_button(screen):
    for mode in (MODE_BRUSH, MODE_ERASE, MODE_ERASE_OBJECT,
                 MODE_WAND_ADD, MODE_WAND_ERASE, MODE_ZOOM):
        screen._set_mode(mode)
        assert screen._canvas.mode == mode
        checked = [m for m, b in screen._mode_buttons.items() if b.isChecked()]
        assert checked == [mode]


# ===========================================================================
# Screen — object operations
# ===========================================================================

def test_invert_mask_swaps_foreground_and_background(screen):
    m = screen._canvas.mask
    m[:] = 0
    m[10:20, 10:20] = 255
    screen._canvas.stroke_finished.emit()      # close the "stroke" -> history
    screen._on_invert()
    out = screen._canvas.mask
    assert out[15, 15] == 0, "the object became background"
    assert out[0, 0] > 0, "the background became an object"
    assert int((out > 0).sum()) == IMG_N * IMG_N - 100
    assert screen._btn_undo.isEnabled()
    screen._on_undo()
    assert int((screen._canvas.mask > 0).sum()) == 100


def test_fill_holes_closes_a_ring_and_relabels(screen):
    m = screen._canvas.mask
    m[:] = 0
    m[10:20, 10:12] = 255
    m[10:20, 18:20] = 255
    m[10:12, 10:20] = 255
    m[18:20, 10:20] = 255
    screen._on_fill_holes()
    out = screen._canvas.mask
    assert out[15, 15] > 0, "the ring interior was not filled"
    assert int((out > 0).sum()) == 100
    assert sorted(int(v) for v in np.unique(out) if v > 0) == [1]


def test_remove_small_then_relabel(screen):
    m = screen._canvas.mask
    m[:] = 0
    m[2:4, 2:4] = 255          # 4 px
    m[30:40, 30:40] = 255      # 100 px
    screen._min_area.setValue(10)
    screen._on_remove_small()
    out = screen._canvas.mask
    assert int((out > 0).sum()) == 100
    assert not (out[2:4, 2:4] > 0).any()
    screen._on_relabel()
    assert sorted(int(v) for v in np.unique(screen._canvas.mask) if v > 0) == [1]


def test_object_ops_are_noops_without_an_image(qtbot, qt_theme_applied):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    for fn in (s._on_fill_holes, s._on_relabel, s._on_invert,
               s._on_remove_small):
        fn()
    assert s._canvas.mask is None
    assert not s._btn_undo.isEnabled()


# ===========================================================================
# Screen — save round-trip
# ===========================================================================

def test_save_writes_labelled_uint16_and_reports_the_path(screen,
                                                          folder_3: Path):
    screen._canvas.mask[:] = 0
    screen._canvas.mask[5:15, 5:15] = 255
    screen._canvas.mask[40:50, 40:50] = 255
    screen._on_save()
    out = folder_3 / "masks" / "img_00.tif"
    assert out.is_file()
    assert screen._status_label.text() == f"Saved → {out}"
    disk = imageio.imread(out)
    assert disk.dtype == np.uint16
    assert disk.shape == (IMG_N, IMG_N)
    assert sorted(int(v) for v in np.unique(disk) if v > 0) == [1, 2]
    assert int((disk > 0).sum()) == 200


def test_saved_mask_is_reloaded_on_the_next_visit(screen, folder_3: Path):
    screen._canvas.mask[:] = 0
    screen._canvas.mask[8:18, 8:18] = 255
    screen._on_save()
    screen._on_next()
    assert not screen._canvas.mask.any(), "img_01 has no mask yet"
    screen._on_prev()
    assert int((screen._canvas.mask > 0).sum()) == 100
    assert not screen._history.can_undo(), "history resets per field"


# ===========================================================================
# Construction / drag-and-drop degradation
# ===========================================================================

def test_screen_survives_a_missing_dnd_module(qtbot, qt_theme_applied,
                                              monkeypatch, folder_3: Path):
    import sys
    monkeypatch.setitem(sys.modules, "spacr.qt.dnd_handlers", None)
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    assert s.acceptDrops() is False, "dropzone should not have installed"
    # …and the screen is still fully functional.
    s._open_folder(str(folder_3))
    assert len(s._image_files) == 3
    assert s._canvas.mask is not None


def test_screen_installs_a_dropzone_when_dnd_is_available(qtbot,
                                                          qt_theme_applied):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    assert s.acceptDrops() is True
    from spacr.qt.dnd_handlers import MakeMasksDropHandler
    assert isinstance(s._dnd_handler, MakeMasksDropHandler)
    assert s._dnd_screen is s
