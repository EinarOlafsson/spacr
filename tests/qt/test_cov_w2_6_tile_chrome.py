"""Crop tiles are painted, then the pixels are read back.

A tile is three layers over each other, and the only honest check of "the
crop is clipped to a rounded corner" or "the current ring sits outside the
state ring" is what ends up in the image. Every case here paints onto a real
QImage and inspects it.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

from spacr.qt.widgets import tile_chrome as tc


@pytest.fixture
def canvas(qapp):
    """A 60x60 white image and a painter on it."""
    image = QImage(60, 60, QImage.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    yield image, painter
    if painter.isActive():
        painter.end()


def _solid(qapp, colour, w=40, h=40):
    pixmap = QPixmap(w, h)
    pixmap.fill(QColor(colour))
    return pixmap


# --------------------------------------------------------------------------
# cover_rect
# --------------------------------------------------------------------------

def test_a_wide_crop_is_scaled_to_fill_and_overflows_the_sides(qapp):
    """Crops rather than letterboxes: a tile is always a complete rounded
    square with no canvas showing through at the corners."""
    box = QRectF(0, 0, 100, 100)
    out = tc.cover_rect(_solid(qapp, "red", 200, 100), box)
    assert out.height() == pytest.approx(100)
    assert out.width() == pytest.approx(200)
    assert out.center() == box.center()
    assert out.left() < box.left() and out.right() > box.right()


def test_a_tall_crop_overflows_the_top_and_bottom(qapp):
    box = QRectF(0, 0, 100, 100)
    out = tc.cover_rect(_solid(qapp, "red", 50, 200), box)
    assert out.width() == pytest.approx(100)
    assert out.top() < 0 and out.bottom() > 100


def test_a_crop_that_already_matches_the_box_comes_back_unchanged(qapp):
    box = QRectF(4, 4, 40, 40)
    out = tc.cover_rect(_solid(qapp, "red", 40, 40), box)
    assert (out.x(), out.y(), out.width(), out.height()) == (4, 4, 40, 40)


def test_a_crop_with_no_pixels_leaves_the_box_alone(qapp):
    box = QRectF(0, 0, 20, 20)
    assert tc.cover_rect(QPixmap(), box) == box


# --------------------------------------------------------------------------
# stroke_ring
# --------------------------------------------------------------------------

def test_a_ring_with_no_room_left_is_not_drawn(canvas):
    """Insetting past the middle would make Qt draw an inside-out rect."""
    image, painter = canvas
    tc.stroke_ring(painter, "#ff0000", 2, 40.0, 60.0, 60.0)
    painter.end()
    assert image.pixelColor(30, 30) == QColor("white")
    assert image.pixelColor(2, 30) == QColor("white")


def test_a_ring_is_stroked_where_its_inset_puts_it(canvas):
    image, painter = canvas
    tc.stroke_ring(painter, "#ff0000", 2, 5.0, 60.0, 60.0)
    painter.end()
    assert image.pixelColor(5, 30).red() > 200
    assert image.pixelColor(30, 30) == QColor("white")   # nothing filled in


# --------------------------------------------------------------------------
# paint_tile
# --------------------------------------------------------------------------

def test_the_crop_is_clipped_to_a_rounded_corner_not_framed_over_a_square(
        canvas, qapp):
    """A rounded frame laid over a square image leaves the crop's own corner
    showing outside it. The corner has to be cut out of the crop."""
    image, painter = canvas
    tc.paint_tile(painter, 60, 60, _solid(qapp, "#0000ff", 60, 60),
                  border_colour="")
    painter.end()
    middle = image.pixelColor(30, 30)
    assert middle.blue() > 200 and middle.red() < 60
    # The extreme corner of the crop area is outside the rounded clip, so
    # the canvas still shows there rather than the crop's own square corner.
    corner = image.pixelColor(tc.TILE_INSET, tc.TILE_INSET)
    assert corner == QColor("white")


def test_an_empty_slot_the_cursor_is_on_still_looks_like_it(canvas):
    """An empty tile the cursor is on has to look like the tile the cursor
    is on, so it gets its rings with no crop underneath."""
    image, painter = canvas
    tc.paint_tile(painter, 60, 60, None, border_colour="#00ff00",
                  ring_colour="#ffffff", current=True)
    painter.end()
    assert image.pixelColor(30, 30) == QColor("white")      # no crop drawn
    state = image.pixelColor(tc.HOVER_RING_WIDTH + 1, 30)
    assert state.green() > 150 and state.red() < 120


def test_a_null_pixmap_is_treated_as_an_empty_slot(canvas):
    image, painter = canvas
    tc.paint_tile(painter, 60, 60, QPixmap(), border_colour="#00ff00")
    painter.end()
    assert image.pixelColor(30, 30) == QColor("white")


def test_a_tile_too_small_for_its_chrome_draws_no_crop(canvas, qapp):
    image, painter = canvas
    tc.paint_tile(painter, 6, 6, _solid(qapp, "#0000ff"),
                  border_colour="#00ff00")
    painter.end()
    assert image.pixelColor(3, 3) == QColor("white")


def test_the_current_ring_sits_outside_the_state_ring(canvas, qapp):
    """Hover adds the outer ring without moving the crop or replacing the
    crop's class-coloured state ring."""
    image, painter = canvas
    tc.paint_tile(painter, 60, 60, _solid(qapp, "#000000", 60, 60),
                  border_colour="#00ff00", ring_colour="#ff0000",
                  current=True)
    painter.end()
    outer = image.pixelColor(1, 30)
    inner = image.pixelColor(tc.HOVER_RING_WIDTH + 1, 30)
    assert outer.red() > 150 and outer.green() < 120
    assert inner.green() > 150 and inner.red() < 120


def test_a_tile_that_is_not_current_gets_no_outer_ring(canvas, qapp):
    image, painter = canvas
    tc.paint_tile(painter, 60, 60, _solid(qapp, "#000000", 60, 60),
                  border_colour="#00ff00", ring_colour="#ff0000",
                  current=False)
    painter.end()
    assert image.pixelColor(1, 30) == QColor("white")


def test_the_image_corner_radius_stays_inside_the_outer_one():
    """Otherwise the crop's corner bulges through the ring drawn around it."""
    assert tc.IMAGE_RADIUS == tc.TILE_RADIUS - tc.TILE_INSET
    assert tc.TILE_INSET == tc.HOVER_RING_WIDTH + tc.BORDER_WIDTH
    assert tc.IMAGE_RADIUS >= 1
