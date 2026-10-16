"""Consistent collapse glyphs with unchanged pointer targets."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

#: Requested size of the chevron relative to its tab, never compounded.
ARROW_SCALE = 0.75

#: Size of the whole painted handle (tab and chevron) relative to the
#: 2026-10-01 baseline. The maintainer asked on 2026-10-16 for the little
#: collapse arrow at 75 percent of the size it had then, centred on its handle.
HANDLE_SCALE = 0.75


def _collapse_arrow_geometry(rect, orientation, towards_start):
    """Where the tab and the chevron of a collapse handle are drawn.

    Everything is computed in floating point around the exact centre of
    ``rect`` (``QRectF(rect).center()``, not the integer ``QRect.center()``,
    which sits half a pixel up and left on an even width), so the tab and the
    chevron are centred on the handle to the sub-pixel.

    :param rect: the clickable handle rectangle; it is never changed.
    :param orientation: splitter orientation (horizontal means a vertical bar).
    :param towards_start: point left/up when true, right/down otherwise.
    :returns: ``(tab, points, thickness)``: the tab as a ``QRectF``, the
        chevron's three ``QPointF`` and its stroke width.
    """
    frame = QRectF(rect)
    centre = frame.center()
    cx, cy = centre.x(), centre.y()
    if orientation == Qt.Horizontal:
        across = frame.width()
        along = max(24.0, across * 3)
    else:
        across = frame.height()
        along = max(24.0, across * 3)
    tab_across = across * HANDLE_SCALE
    tab_along = along * HANDLE_SCALE
    s = max(2.0, float(int(across) // 4)) * HANDLE_SCALE * ARROW_SCALE
    thickness = (max(1.0, float(int(across) // 6)) * HANDLE_SCALE
                 * ARROW_SCALE)
    if orientation == Qt.Horizontal:
        tab = QRectF(cx - tab_across / 2, cy - tab_along / 2,
                     tab_across, tab_along)
        points = ([QPointF(cx + s, cy - 2 * s), QPointF(cx - s, cy),
                   QPointF(cx + s, cy + 2 * s)] if towards_start else
                  [QPointF(cx - s, cy - 2 * s), QPointF(cx + s, cy),
                   QPointF(cx - s, cy + 2 * s)])
    else:
        tab = QRectF(cx - tab_along / 2, cy - tab_across / 2,
                     tab_along, tab_across)
        points = ([QPointF(cx - 2 * s, cy + s), QPointF(cx, cy - s),
                   QPointF(cx + 2 * s, cy + s)] if towards_start else
                  [QPointF(cx - 2 * s, cy - s), QPointF(cx, cy + s),
                   QPointF(cx + 2 * s, cy - s)])
    return tab, points, thickness


def paint_collapse_arrow(painter, rect, orientation, towards_start, palette,
                         hovered=False):
    """Draw the collapse tab and its chevron, centred on the handle.

    :param painter: active painter on the handle.
    :param rect: unchanged clickable handle rectangle.
    :param orientation: splitter orientation (horizontal means a vertical bar).
    :param towards_start: point left/up when true, right/down otherwise.
    :param palette: active theme colors.
    :param hovered: whether to accent the glyph and tab border.

    The tab and the chevron are drawn at :data:`HANDLE_SCALE` of the
    2026-10-01 baseline, the chevron at a further :data:`ARROW_SCALE` of its
    tab, both centred on the handle (:func:`_collapse_arrow_geometry`). The
    hit target, the whole handle, stays put.
    """
    painter.setRenderHint(QPainter.Antialiasing, True)
    tab_color = QColor(palette.get("surface_alt", palette.get("surface",
                                                              "#2a2e37")))
    tab_color = (tab_color.lighter(165) if tab_color.lightness() < 128
                 else tab_color.darker(108))
    accent = QColor(palette.get("accent", "#4c8dff"))
    edge = accent if hovered else QColor(palette.get("border", "#3a3f4b"))
    ink = accent if hovered else QColor(palette.get("text_muted",
                                                    palette.get("text",
                                                                "#b9bfca")))
    tab, points, thickness = _collapse_arrow_geometry(rect, orientation,
                                                     towards_start)
    painter.save()
    painter.setPen(QPen(edge, 1.0))
    painter.setBrush(tab_color)
    painter.drawRoundedRect(tab.adjusted(0.5, 0.5, -0.5, -0.5), 2.25, 2.25)
    stroke = QPen(ink, thickness)
    stroke.setCapStyle(Qt.RoundCap)
    stroke.setJoinStyle(Qt.RoundJoin)
    painter.setPen(stroke)
    painter.setBrush(Qt.NoBrush)
    painter.drawPolyline(QPolygonF(points))
    painter.restore()
