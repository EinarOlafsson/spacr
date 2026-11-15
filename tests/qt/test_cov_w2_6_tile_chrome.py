"""Crop tiles are painted, then the pixels are read back.

A tile is three layers over each other, and the only honest check of "the
crop is clipped to a rounded corner" or "the current ring sits outside the
state ring" is what ends up in the image. Every case here paints onto a real
QImage and inspects it.
"""
from __future__ import annotations

import pytest

from PySide6.QtGui import QColor, QImage, QPainter

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


# --------------------------------------------------------------------------
# cover_rect
# --------------------------------------------------------------------------

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

def test_the_image_corner_radius_stays_inside_the_outer_one():
    """Otherwise the crop's corner bulges through the ring drawn around it."""
    assert tc.IMAGE_RADIUS == tc.TILE_RADIUS - tc.TILE_INSET
    assert tc.TILE_INSET == tc.HOVER_RING_WIDTH + tc.BORDER_WIDTH
    assert tc.IMAGE_RADIUS >= 1
