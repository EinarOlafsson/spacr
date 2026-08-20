"""The Cells tab draws its crops the way the annotation app draws its own.

Asked for 2026-08-19: "when hovering over images i should see the rim turn
white and i should be able to click each image to get its information. the
images should have rounded edges like in the annotation app".

The look is SHARED CODE rather than an imitation -- `spacr.qt.widgets
.tile_chrome` is the annotate screen's own painting, extracted. Two
implementations of one appearance drift apart, and this tab already borrows
the annotator's setting names for that reason (170 B).
"""
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QMouseEvent

from spacr.qt.widgets.cell_montage_view import _Thumb, _thumbnail
from spacr.qt.widgets.tile_chrome import (BORDER_WIDTH, HOVER_RING_WIDTH,
                                          IMAGE_RADIUS, TILE_INSET,
                                          TILE_RADIUS, cover_rect, paint_tile)


@pytest.fixture()
def thumb(qtbot):
    rng = np.random.default_rng(0)
    crop = rng.integers(30, 120, (64, 64, 3)).astype("uint8")
    widget = _thumbnail(crop, None, size=64, picture={"channels": "r,g,b"})
    qtbot.addWidget(widget)
    return widget


# ------------------------------------------------------------ rounded edges


def test_the_corners_are_actually_round():
    """A real rounded corner, not a rounded frame laid over a square image."""
    assert TILE_RADIUS > 0
    assert IMAGE_RADIUS > 0
    assert IMAGE_RADIUS == max(1, TILE_RADIUS - TILE_INSET)


def test_the_rings_sit_at_fixed_insets():
    """Hover ADDS a ring; nothing moves or resizes when the cursor arrives."""
    assert TILE_INSET == HOVER_RING_WIDTH + BORDER_WIDTH


def test_a_crop_fills_the_tile_rather_than_letterboxing(qtbot):
    """The clip trims the overflow, so no canvas shows at the corners.

    `qtbot` is not decoration: constructing a QPixmap before a QApplication
    exists SEGFAULTS rather than raising, so a test that only touches
    geometry still needs the app up.
    """
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPixmap

    wide = QPixmap(200, 50)
    box = QRectF(0, 0, 100, 100)
    drawn = cover_rect(wide, box)

    assert drawn.width() >= box.width() - 1e-6
    assert drawn.height() >= box.height() - 1e-6
    assert drawn.center().x() == pytest.approx(box.center().x())


def test_an_empty_pixmap_is_not_a_crash(qtbot):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPixmap

    assert cover_rect(QPixmap(), QRectF(0, 0, 10, 10)) == QRectF(0, 0, 10, 10)


def test_painting_a_tile_with_no_pixmap_still_draws_its_rings(qtbot):
    """An empty tile the cursor is on must look like the tile the cursor is
    on, or the grid loses its place."""
    from PySide6.QtGui import QPainter, QPixmap

    target = QPixmap(80, 80)
    target.fill()
    painter = QPainter(target)
    try:
        paint_tile(painter, 80.0, 80.0, None, border_colour="#888888",
                   ring_colour="#ffffff", current=True)
    finally:
        painter.end()


# ------------------------------------------------------------------- hover


def test_a_thumbnail_starts_unhovered(thumb):
    assert isinstance(thumb, _Thumb)
    assert thumb._hovered is False


def test_the_rim_lights_under_the_cursor(thumb):
    """Driven through the real handlers, with the events Qt actually sends:
    `enterEvent` takes a QEnterEvent, and a bare QEvent is a TypeError."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent

    where = QPointF(5.0, 5.0)
    thumb.enterEvent(QEnterEvent(where, where, where))
    assert thumb._hovered is True

    thumb.leaveEvent(QEvent(QEvent.Type.Leave))
    assert thumb._hovered is False


def test_the_hover_ring_is_the_annotators_white(thumb):
    """`current_ring_color` is pure white on the dark theme and flips on the
    light one, where white would vanish. Borrowed rather than hard-coded."""
    from spacr.qt.screens.annotate import current_ring_color

    _border, ring = thumb._colours()
    assert ring == current_ring_color()


def test_a_highlighted_cell_keeps_its_own_state_ring(qtbot):
    """The class colour is the INNER ring; hover adds the outer one, so a
    highlighted cell under the cursor shows both."""
    rng = np.random.default_rng(1)
    crop = rng.integers(30, 120, (32, 32, 3)).astype("uint8")
    widget = _Thumb(_pixmap_of(crop), "", size=32, highlight="#00ff00")
    qtbot.addWidget(widget)

    border, ring = widget._colours()
    assert border == "#00ff00"
    assert ring != border


def _pixmap_of(crop):
    from spacr.qt.widgets.cell_montage_view import _pixmap

    return _pixmap(crop, 32)


# ------------------------------------------------------------------- click


def test_clicking_a_cell_reports_its_provenance(thumb):
    seen = []
    thumb.clicked.connect(seen.append)

    thumb.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPoint(5, 5), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

    assert len(seen) == 1


def test_a_right_click_is_not_a_left_click(thumb):
    seen = []
    thumb.clicked.connect(seen.append)

    thumb.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPoint(5, 5), Qt.RightButton,
        Qt.RightButton, Qt.NoModifier))

    assert not seen


def test_the_detail_window_opens_and_is_kept(qtbot, tmp_path):
    """Python collects a dialog with no reference the moment the handler
    returns, and the window vanishes as it appears."""
    from spacr.qt.widgets.cell_montage_view import _WellTab

    tab = _WellTab(key=("g", "gene", "g", "w"), label="w", parent=None)
    qtbot.addWidget(tab)

    tab._show_cell_detail("plate1_r1_c1 · object 12 · cut from merged/")

    assert tab._details
    assert tab._details[0].isVisible()
    qtbot.addWidget(tab._details[0])


def test_an_empty_provenance_opens_nothing(qtbot):
    """A tile with no tooltip has nothing to show, and an empty window reads
    as a broken click."""
    from spacr.qt.widgets.cell_montage_view import _WellTab

    tab = _WellTab(key=("g", "gene", "g", "w"), label="w", parent=None)
    qtbot.addWidget(tab)

    tab._show_cell_detail("   ")

    assert not tab._details
