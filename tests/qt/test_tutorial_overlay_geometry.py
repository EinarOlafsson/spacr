"""Pixel-level tests for the tutorial overlay painters and the Recorder.

The cursor arrow and the highlight ring are the only things the viewer
sees that are *not* the app itself, so they are checked the way a viewer
would check them: paint onto a known solid background, read the pixels
back, and assert *where* the marks landed and *what colour* they are.

Everything here runs offscreen against real QPixmaps — no mocking.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

BG_RED = (255, 0, 0)
BG_WHITE = (255, 255, 255)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_array(pixmap_or_image):
    """RGB uint8 array (h, w, 3) for a QPixmap or QImage."""
    from PySide6.QtGui import QImage, QPixmap
    img = (pixmap_or_image.toImage()
             if isinstance(pixmap_or_image, QPixmap) else pixmap_or_image)
    img = img.convertToFormat(QImage.Format_RGB888)
    raw = np.frombuffer(img.constBits(), dtype=np.uint8)
    raw = raw.reshape(img.height(), img.bytesPerLine())
    # .copy() is mandatory: the buffer belongs to `img`, which dies when
    # this function returns.
    return raw[:, :img.width() * 3].reshape(
        img.height(), img.width(), 3).copy()


def _changed_bbox(arr, background):
    """(x0, y0, x1, y1) of every pixel differing from ``background``."""
    changed = np.any(arr != np.array(background, dtype=np.uint8), axis=-1)
    ys, xs = np.nonzero(changed)
    assert xs.size, "nothing was painted"
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _filled(width, height, colour):
    from PySide6.QtGui import QColor, QPixmap
    pm = QPixmap(width, height)
    pm.fill(QColor(*colour))
    return pm


# ---------------------------------------------------------------------------
# _draw_cursor_on
# ---------------------------------------------------------------------------

def test_cursor_arrow_lands_at_the_requested_point(qt_theme_applied):
    """The arrow's tip is the point the caller asked for, and the whole
    mark (arrow + its 2 px drop shadow) fits in a ~21x27 box hanging
    down-right from it."""
    from spacr.qt.tutorial.engine import _draw_cursor_on
    pm = _filled(200, 200, BG_RED)
    _draw_cursor_on(pm, (50, 60))
    arr = _to_array(pm)

    x0, y0, x1, y1 = _changed_bbox(arr, BG_RED)
    # Tip is at (50, 60); antialiasing + the 1.5 px pen bleed one pixel up
    # and to the left, never more.
    assert 49 <= x0 <= 50
    assert 59 <= y0 <= 60
    # Arrow spans +18 x / +23 y, shadow adds +2, pen adds ~1.
    assert x1 <= 50 + 21
    assert y1 <= 60 + 26
    # …and it is a *white* arrow, not just some smudge.
    assert tuple(arr[70, 55]) == (255, 255, 255)
    assert int(np.all(arr == 255, axis=-1).sum()) > 50
    # Nothing painted anywhere else on the canvas.
    assert tuple(arr[10, 10]) == BG_RED
    assert tuple(arr[190, 190]) == BG_RED


def test_cursor_arrow_translates_with_the_position(qt_theme_applied):
    """Moving the cursor moves the mark by exactly the same delta — the
    painter must not be drawing at a fixed spot."""
    from spacr.qt.tutorial.engine import _draw_cursor_on
    a = _filled(300, 300, BG_RED)
    b = _filled(300, 300, BG_RED)
    _draw_cursor_on(a, (30, 40))
    _draw_cursor_on(b, (130, 190))

    ax0, ay0, ax1, ay1 = _changed_bbox(_to_array(a), BG_RED)
    bx0, by0, bx1, by1 = _changed_bbox(_to_array(b), BG_RED)
    assert (bx0 - ax0, by0 - ay0) == (100, 150)
    assert (bx1 - ax1, by1 - ay1) == (100, 150)


def test_cursor_draws_a_shadow_under_the_arrow(qt_theme_applied):
    """The shadow is a 40 %-black copy offset by 2 px, so on a white
    canvas the pixels just outside the arrow's lower-left edge are grey
    rather than white or untouched."""
    from spacr.qt.tutorial.engine import _draw_cursor_on
    pm = _filled(120, 120, BG_WHITE)
    _draw_cursor_on(pm, (40, 40))
    arr = _to_array(pm).astype(int)
    # Row through the arrow's tail: the shadow extends 2 px right of the
    # arrow's bottom-right vertex at (40+15, 40+22).
    strip = arr[40 + 23, 40 + 15:40 + 18]
    grey = [p for p in strip if 0 < p[0] < 255 and p[0] == p[1] == p[2]]
    assert grey, f"expected a grey shadow band, got {strip.tolist()}"


# ---------------------------------------------------------------------------
# _draw_highlight_on
# ---------------------------------------------------------------------------

def test_highlight_ring_is_hollow_and_outset_by_four_px(qt_theme_applied):
    """The ring is drawn 4 px outside the widget rect with a 4 px pen and
    NO fill — a filled highlight would hide the very widget it points
    at."""
    from spacr.qt.tutorial.engine import _draw_highlight_on
    pm = _filled(200, 200, BG_WHITE)
    _draw_highlight_on(pm, (40, 50, 60, 30))
    arr = _to_array(pm)

    x0, y0, x1, y1 = _changed_bbox(arr, BG_WHITE)
    # rect inset by -4 => (36, 46, 68, 38); a 4 px pen straddles the path
    # by 2 px each way.
    assert (x0, y0) == (34, 44)
    assert (x1, y1) == (105, 85)

    # Interior of the widget rect is completely untouched.
    interior = arr[55:75, 45:95]
    assert np.all(interior == 255), "highlight must not fill the widget"

    # The stroke is the spaCR accent blue (74, 158, 255): blue dominates.
    r, g, b = arr[65, 36]
    assert b == 255 and b > g > r


def test_highlight_ring_tracks_the_rect(qt_theme_applied):
    from spacr.qt.tutorial.engine import _draw_highlight_on
    a = _filled(300, 300, BG_WHITE)
    b = _filled(300, 300, BG_WHITE)
    _draw_highlight_on(a, (20, 20, 50, 40))
    _draw_highlight_on(b, (20, 20, 100, 80))
    ax0, ay0, ax1, ay1 = _changed_bbox(_to_array(a), BG_WHITE)
    bx0, by0, bx1, by1 = _changed_bbox(_to_array(b), BG_WHITE)
    assert (ax0, ay0) == (bx0, by0)          # same top-left
    assert bx1 - ax1 == 50 and by1 - ay1 == 40   # grows with w/h


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

def _red_widget(qtbot, w, h):
    """A widget that paints itself solid red, so every pixel the Recorder
    grabs has a known value the app stylesheet cannot repaint."""
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtWidgets import QWidget

    class _Red(QWidget):
        def paintEvent(self, event):          # noqa: N802 (Qt naming)
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(255, 0, 0))
            painter.end()

    widget = _Red()
    widget.resize(w, h)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_recorder_writes_sequentially_numbered_frames(qtbot,
                                                       qt_theme_applied,
                                                       tmp_path):
    from spacr.qt.tutorial.engine import Recorder
    widget = _red_widget(qtbot, 160, 120)
    rec = Recorder(widget, tmp_path / "frames", fps=5, size=(160, 120))

    assert rec.frame_idx == 0
    assert rec.cursor_pos == (80.0, 60.0)      # centre of the frame

    p0 = rec.snap()
    p1 = rec.snap()
    assert p0.name == "frame_000000.png"
    assert p1.name == "frame_000001.png"
    assert rec.frame_idx == 2
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == [
        "frame_000000.png", "frame_000001.png"]

    from PySide6.QtGui import QImage
    img = QImage(str(p0))
    assert (img.width(), img.height()) == (160, 120)


def test_recorder_letterboxes_a_window_smaller_than_the_frame(
        qtbot, qt_theme_applied, tmp_path):
    """A 100x100 widget in a 400x200 frame scales to 200x200 (aspect
    kept) and gets centred on black — so the outer thirds are black and
    the middle is the widget."""
    from PySide6.QtGui import QImage
    from spacr.qt.tutorial.engine import Recorder
    widget = _red_widget(qtbot, 100, 100)
    rec = Recorder(widget, tmp_path / "f", size=(400, 200))
    path = rec.snap(cursor_pos=(5, 5))

    arr = _to_array(QImage(str(path)))
    assert arr.shape == (200, 400, 3)
    assert tuple(arr[100, 10]) == (0, 0, 0)      # left letterbox
    assert tuple(arr[100, 390]) == (0, 0, 0)     # right letterbox
    assert tuple(arr[100, 200]) == (255, 0, 0)   # centred widget
    # Widget occupies x in [100, 300)
    assert tuple(arr[100, 101]) == (255, 0, 0)
    assert tuple(arr[100, 98]) == (0, 0, 0)


def test_recorder_remembers_and_reuses_the_cursor_position(
        qtbot, qt_theme_applied, tmp_path):
    """snap(cursor_pos=...) both draws there and becomes the new resting
    position, so a later snap() with no argument repeats it."""
    from PySide6.QtGui import QImage
    from spacr.qt.tutorial.engine import Recorder
    widget = _red_widget(qtbot, 200, 200)
    rec = Recorder(widget, tmp_path / "f", size=(200, 200))

    moved = rec.snap(cursor_pos=(120.7, 60.2))
    assert rec.cursor_pos == (120.7, 60.2)
    repeat = rec.snap()
    assert rec.cursor_pos == (120.7, 60.2)

    # int() truncation, not rounding: the arrow tip sits at (120, 60).
    bbox_a = _changed_bbox(_to_array(QImage(str(moved))), BG_RED)
    bbox_b = _changed_bbox(_to_array(QImage(str(repeat))), BG_RED)
    assert bbox_a == bbox_b
    assert bbox_a[0] in (119, 120)
    assert bbox_a[1] in (59, 60)


def test_recorder_composites_the_highlight_ring_into_the_frame(
        qtbot, qt_theme_applied, tmp_path):
    from PySide6.QtGui import QImage
    from spacr.qt.tutorial.engine import Recorder
    widget = _red_widget(qtbot, 200, 200)
    rec = Recorder(widget, tmp_path / "f", size=(200, 200))

    plain = _to_array(QImage(str(rec.snap(cursor_pos=(180, 180)))))
    ringed = _to_array(QImage(str(rec.snap(cursor_pos=(180, 180),
                                             highlight_rect=(20, 20, 60, 40)))))
    # Ring pixels only appear in the second frame.
    assert tuple(plain[40, 16]) == (255, 0, 0)
    r, g, b = ringed[40, 16]
    assert b > r and b > 200
    # …and the widget behind the ring is still visible.
    assert tuple(ringed[40, 50]) == (255, 0, 0)
