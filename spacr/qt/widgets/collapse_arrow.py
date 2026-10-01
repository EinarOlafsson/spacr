"""Consistent collapse glyphs with unchanged pointer targets."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon

#: Requested size relative to the original splitter glyph, never compounded.
ARROW_SCALE = 0.75


def paint_collapse_arrow(painter, rect, orientation, towards_start, palette,
                         hovered=False):
    """Draw the existing tab with its arrow at 75 percent of baseline.

    :param painter: active painter on the handle.
    :param rect: unchanged clickable handle rectangle.
    :param orientation: splitter orientation (horizontal means a vertical bar).
    :param towards_start: point left/up when true, right/down otherwise.
    :param palette: active theme colors.
    :param hovered: whether to accent the glyph and tab border.
    """
    painter.setRenderHint(QPainter.Antialiasing, True)
    tab = QColor(palette.get("surface_alt", palette.get("surface",
                                                        "#2a2e37")))
    tab = tab.lighter(165) if tab.lightness() < 128 else tab.darker(108)
    accent = QColor(palette.get("accent", "#4c8dff"))
    edge = accent if hovered else QColor(palette.get("border", "#3a3f4b"))
    ink = accent if hovered else QColor(palette.get("text_muted",
                                                    palette.get("text",
                                                                "#b9bfca")))
    if orientation == Qt.Horizontal:
        w = rect.width()
        h = max(24, w * 3)
        top = rect.center().y() - h // 2
        painter.setPen(edge)
        painter.setBrush(tab)
        painter.drawRoundedRect(0, top, w - 1, h, 3, 3)
        cx, cy, s = rect.center().x(), rect.center().y(), max(2, w // 4)
        points = ([QPoint(cx + s, cy - 2 * s), QPoint(cx - s, cy),
                   QPoint(cx + s, cy + 2 * s)] if towards_start else
                  [QPoint(cx - s, cy - 2 * s), QPoint(cx + s, cy),
                   QPoint(cx - s, cy + 2 * s)])
        thickness = max(1, w // 6)
    else:
        h = rect.height()
        w = max(24, h * 3)
        left = rect.center().x() - w // 2
        painter.setPen(edge)
        painter.setBrush(tab)
        painter.drawRoundedRect(left, 0, w, h - 1, 3, 3)
        cx, cy, s = rect.center().x(), rect.center().y(), max(2, h // 4)
        points = ([QPoint(cx - 2 * s, cy + s), QPoint(cx, cy - s),
                   QPoint(cx + 2 * s, cy + s)] if towards_start else
                  [QPoint(cx - 2 * s, cy - s), QPoint(cx, cy + s),
                   QPoint(cx + 2 * s, cy - s)])
        thickness = max(1, h // 6)
    # Scale the baseline glyph only; the painted tab and hit target stay put.
    painter.save()
    painter.translate(rect.center())
    painter.scale(ARROW_SCALE, ARROW_SCALE)
    painter.translate(-rect.center())
    stroke = QPen(ink, thickness)
    stroke.setCapStyle(Qt.RoundCap)
    stroke.setJoinStyle(Qt.RoundJoin)
    painter.setPen(stroke)
    painter.setBrush(Qt.NoBrush)
    painter.drawPolyline(QPolygon(points))
    painter.restore()
