"""Pixel-level tests for the tutorial overlay painters and the Recorder.

The cursor arrow, highlight ring, and dimmed spotlight are the only things
the viewer sees that are *not* the app itself, so they are checked the way
a viewer would check them: paint onto a known solid background, read the
pixels back, and assert *where* the marks landed and *what colour* they are.

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

def test_cursor_dot_is_centred_at_the_requested_point(qt_theme_applied):
    """The requested point is the centre of a tiny solid magenta dot."""
    from spacr.qt.tutorial.engine import _draw_cursor_on
    pm = _filled(200, 200, BG_RED)
    _draw_cursor_on(pm, (50, 60))
    arr = _to_array(pm)

    x0, y0, x1, y1 = _changed_bbox(arr, BG_RED)
    assert 45 <= x0 <= 46 and 55 <= y0 <= 56
    assert 54 <= x1 <= 55 and 64 <= y1 <= 65
    assert tuple(arr[60, 50]) == (255, 0, 153)
    # Nothing painted anywhere else on the canvas.
    assert tuple(arr[10, 10]) == BG_RED
    assert tuple(arr[190, 190]) == BG_RED


def test_cursor_dot_translates_with_the_position(qt_theme_applied):
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


def test_cursor_dot_has_no_shadow_or_second_colour(qt_theme_applied):
    """The click point is one magenta colour, with no hollow centre."""
    from spacr.qt.tutorial.engine import _draw_cursor_on
    pm = _filled(120, 120, BG_WHITE)
    _draw_cursor_on(pm, (40, 40))
    arr = _to_array(pm)
    assert tuple(arr[40, 40]) == (255, 0, 153)
    assert tuple(arr[47, 47]) == BG_WHITE


# ---------------------------------------------------------------------------
# _draw_highlight_on
# ---------------------------------------------------------------------------

def test_highlight_ring_is_hollow_and_outset_by_four_px(qt_theme_applied):
    """The ring is drawn 4 px outside the widget rect with a 1 px pen and
    NO fill — a filled highlight would hide the very widget it points
    at."""
    from spacr.qt.tutorial.engine import _draw_highlight_on
    pm = _filled(200, 200, BG_WHITE)
    _draw_highlight_on(pm, (40, 50, 60, 30))
    arr = _to_array(pm)

    x0, y0, x1, y1 = _changed_bbox(arr, BG_WHITE)
    # The one-pixel anti-aliased stroke stays close to the outset path.
    assert 35 <= x0 <= 36 and 45 <= y0 <= 46
    assert 103 <= x1 <= 104 and 83 <= y1 <= 84

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
# _draw_spotlight_on
# ---------------------------------------------------------------------------

def test_spotlight_dims_the_frame_but_preserves_the_focus(
        qt_theme_applied):
    from spacr.qt.tutorial.engine import _draw_spotlight_on
    pm = _filled(200, 160, BG_WHITE)
    _draw_spotlight_on(pm, (60, 50, 50, 30), padding=10, opacity=150)
    arr = _to_array(pm)

    # The app outside the focus is visibly greyed and darkened.
    outside = arr[20, 20]
    assert outside[0] == outside[1] == outside[2]
    assert 80 < int(outside[0]) < 180

    # The selected widget and its ten-pixel breathing room remain unchanged.
    assert tuple(arr[65, 85]) == BG_WHITE
    assert tuple(arr[65, 52]) == BG_WHITE

    # Rounded spotlight corners fall back into the dimmed mask.
    assert tuple(arr[40, 50]) != BG_WHITE


def test_spotlight_tracks_the_requested_focus_rect(qt_theme_applied):
    from spacr.qt.tutorial.engine import _draw_spotlight_on
    a = _filled(240, 180, BG_WHITE)
    b = _filled(240, 180, BG_WHITE)
    _draw_spotlight_on(a, (20, 30, 40, 30), padding=0)
    _draw_spotlight_on(b, (140, 90, 40, 30), padding=0)
    aa = _to_array(a)
    bb = _to_array(b)

    assert tuple(aa[45, 40]) == BG_WHITE
    assert tuple(aa[105, 160]) != BG_WHITE
    assert tuple(bb[45, 40]) != BG_WHITE
    assert tuple(bb[105, 160]) == BG_WHITE


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

    moved = rec.snap(cursor_pos=(120.7, 60.2), show_pointer=True)
    assert rec.cursor_pos == (120.7, 60.2)
    repeat = rec.snap(show_pointer=True)
    assert rec.cursor_pos == (120.7, 60.2)

    # int() truncation, not rounding: the dot is centred at (120, 60).
    bbox_a = _changed_bbox(_to_array(QImage(str(moved))), BG_RED)
    bbox_b = _changed_bbox(_to_array(QImage(str(repeat))), BG_RED)
    assert bbox_a == bbox_b
    assert bbox_a[0] in (119, 120)
    assert bbox_a[1] in (59, 60)


def test_recorder_hides_pointer_unless_the_step_is_a_click(
        qtbot, qt_theme_applied, tmp_path):
    from PySide6.QtGui import QImage
    from spacr.qt.tutorial.engine import Recorder
    widget = _red_widget(qtbot, 200, 200)
    rec = Recorder(widget, tmp_path / "f", size=(200, 200))

    passive = _to_array(QImage(str(rec.snap(cursor_pos=(100, 100))))
    clicked = _to_array(QImage(str(rec.snap(
        cursor_pos=(100, 100), show_pointer=True))))

    assert tuple(passive[100, 100]) == BG_RED
    assert tuple(clicked[100, 100]) == (255, 0, 153)


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
    # The rest of the app is subdued.
    assert tuple(ringed[150, 150]) != (255, 0, 0)


def test_recorder_can_keep_the_background_bright_for_overview_steps(
        qtbot, qt_theme_applied, tmp_path):
    from PySide6.QtGui import QImage
    from spacr.qt.tutorial.engine import Recorder
    widget = _red_widget(qtbot, 200, 200)
    rec = Recorder(widget, tmp_path / "f", size=(200, 200))

    frame = _to_array(QImage(str(rec.snap(
        cursor_pos=(180, 180),
        highlight_rect=(20, 20, 60, 40),
        dim_background=False,
    ))))

    # The blue ring is still present while pixels away from it are untouched.
    r, g, b = frame[40, 16]
    assert b > r and b > 200
    assert tuple(frame[150, 150]) == (255, 0, 0)
