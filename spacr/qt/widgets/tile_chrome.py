"""Draw rounded crop tiles with shared annotation-screen chrome.

The annotation and montage views use this module so their clipping and
selection states stay visually consistent. A tile composes three layers:

    ┌── current ring  (white) — the tile the next action targets
    │ ┌── state ring          — resting gray, or the crop's class colour
    │ │ ┌── the crop itself, clipped to a rounded rect (a real round
    │ │ │   corner, not a rounded frame laid over a square image)

The rings use fixed insets, so hover adds the outer current ring without
moving the crop or replacing its class-colored state ring.
"""
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap

#: Width of the thin state border around every crop, in pixels.
BORDER_WIDTH = 2
#: The current-tile ring, drawn OUTSIDE the state ring.
HOVER_RING_WIDTH = 3
#: Chrome per side, in pixels.
TILE_INSET = HOVER_RING_WIDTH + BORDER_WIDTH
#: Outer corner radius of the rounded square.
TILE_RADIUS = 10
#: Corner radius of the image inside the chrome.
IMAGE_RADIUS = max(1, TILE_RADIUS - TILE_INSET)

__all__ = ["BORDER_WIDTH", "HOVER_RING_WIDTH", "TILE_INSET", "TILE_RADIUS",
           "IMAGE_RADIUS", "cover_rect", "stroke_ring", "paint_tile"]


def cover_rect(pixmap: QPixmap, box: QRectF) -> QRectF:
    """Rect to draw ``pixmap`` into so it fills ``box`` at its own aspect.

    Crops rather than letterboxes -- the clip path trims the overflow -- so a
    tile is always a complete rounded square with no canvas showing through
    at the corners. A crop that already matches the box comes back unchanged.
    """
    width = float(pixmap.width())
    height = float(pixmap.height())
    if width <= 0 or height <= 0:
        return box
    scale = max(box.width() / width, box.height() / height)
    drawn_w = width * scale
    drawn_h = height * scale
    return QRectF(box.center().x() - drawn_w / 2.0,
                  box.center().y() - drawn_h / 2.0, drawn_w, drawn_h)


def stroke_ring(painter: QPainter, colour: str, width: int, inset: float,
                w: float, h: float) -> None:
    """Stroke one rounded rect inset by ``inset`` from the widget edge."""
    if w - 2 * inset <= 0 or h - 2 * inset <= 0:
        return
    pen = QPen(QColor(colour))
    pen.setWidth(int(width))
    painter.setPen(pen)
    radius = max(1.0, TILE_RADIUS - inset)
    painter.drawRoundedRect(
        QRectF(inset, inset, w - 2 * inset, h - 2 * inset), radius, radius)


def paint_tile(painter: QPainter, w: float, h: float, pixmap, *,
               border_colour: str, ring_colour: str = "",
               current: bool = False) -> None:
    """Draw one tile: the clipped crop, its state ring, and the hover ring.

    :param pixmap: the crop, or ``None`` for an empty slot -- which still
        gets its rings, because an empty tile the cursor is on has to look
        like the tile the cursor is on.
    :param current: whether the cursor (or the keyboard) is on this tile.
    """
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(painter.brush().__class__())    # Qt.NoBrush equivalent
    inner = QRectF(TILE_INSET, TILE_INSET,
                   max(0.0, w - 2 * TILE_INSET), max(0.0, h - 2 * TILE_INSET))
    if (pixmap is not None and not pixmap.isNull()
            and inner.width() > 0 and inner.height() > 0):
        clip = QPainterPath()
        clip.addRoundedRect(inner, IMAGE_RADIUS, IMAGE_RADIUS)
        painter.save()
        painter.setClipPath(clip)
        painter.drawPixmap(cover_rect(pixmap, inner), pixmap,
                           QRectF(pixmap.rect()))
        painter.restore()
    if border_colour:
        stroke_ring(painter, border_colour, BORDER_WIDTH,
                    HOVER_RING_WIDTH + BORDER_WIDTH / 2.0, w, h)
    if current and ring_colour:
        stroke_ring(painter, ring_colour, HOVER_RING_WIDTH,
                    HOVER_RING_WIDTH / 2.0, w, h)
