"""The rounded card the first-run questions sit on.

Instruction 221: "a box with round corners and a periphery with a blue line
that follows the corner of the box the mouse is closest to ... i basically
want my version of the sleek intro that apple has."

THE CORNER LINE IS A POINTER READOUT, NOT A GLOW, and stating that is the
whole of the design: the accent tracks WHICH of the four corners the pointer
is nearest and moves between them. Built as a glow it would light on hover
and say nothing about where the mouse is.

INVARIANTS 10: DECORATION IS NEVER LOAD-BEARING. If the blur, the ambient
background or the corner accent cannot be drawn, the card still shows its
questions and still saves the answers. Every paint step here is inside a
guard for that reason.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

#: Corner order: top-left, top-right, bottom-right, bottom-left.
CORNERS = ("topLeft", "topRight", "bottomRight", "bottomLeft")


class SetupCard(QWidget):
    """A rounded, translucent card whose border lights at the near corner.

    :param radius: corner radius in pixels.
    :param arc: how far along each edge the accent runs, in pixels.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 radius: int = 18, arc: int = 90):
        super().__init__(parent)
        self._radius = int(radius)
        self._arc = int(arc)
        self._corner = 0
        # MOUSE TRACKING, or `mouseMoveEvent` only fires while a button is
        # held -- which is never, on a card the user is only reading.
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)

    # ------------------------------------------------------------------
    def nearest_corner(self, point: QPointF) -> int:
        """Index into :data:`CORNERS` of the corner nearest ``point``.

        Ties go to the earlier corner, which only happens at the exact
        centre and is therefore never seen.
        """
        rect = QRectF(self.rect())
        points = (rect.topLeft(), rect.topRight(),
                  rect.bottomRight(), rect.bottomLeft())
        best, chosen = None, 0
        for index, corner in enumerate(points):
            dx = point.x() - corner.x()
            dy = point.y() - corner.y()
            distance = dx * dx + dy * dy
            if best is None or distance < best:
                best, chosen = distance, index
        return chosen

    def corner(self) -> str:
        """The corner the accent is currently on."""
        return CORNERS[self._corner]

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event):            # noqa: N802 - Qt naming
        self._follow(event.position())
        super().mouseMoveEvent(event)

    def event(self, event):
        # A hover move arrives even when the widget has no mouse grab, which
        # is the ordinary case here.
        try:
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.HoverMove:
                self._follow(event.position())
        except Exception:                        # pragma: no cover
            pass
        return super().event(event)

    def _follow(self, position) -> None:
        corner = self.nearest_corner(QPointF(position))
        if corner != self._corner:
            self._corner = corner
            # ONLY THE BORDER REPAINTS. The card is translucent over a
            # blurred backdrop, and repainting that on every mouse move is
            # the difference between a smooth accent and a stuttering one.
            self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event):                # noqa: N802 - Qt naming
        try:
            self._paint()
        except Exception:                        # pragma: no cover
            # Decoration is never load-bearing: an unpainted card is still a
            # card with working controls on it.
            pass
        super().paintEvent(event)

    def _paint(self) -> None:
        from ..theme import active_palette

        palette = active_palette()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

            body = QColor(palette.get("surface", palette["bg"]))
            body.setAlpha(216)          # translucent: the blobs show through
            painter.setPen(Qt.NoPen)
            painter.setBrush(body)
            painter.drawRoundedRect(rect, self._radius, self._radius)

            edge = QColor(palette["fg"])
            edge.setAlpha(38)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(edge, 1.0))
            painter.drawRoundedRect(rect, self._radius, self._radius)

            # THE ACCENT, on the near corner only.
            painter.setPen(QPen(QColor(palette["accent"]), 2.0,
                                Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(self._corner_path(rect))
        finally:
            painter.end()

    def _corner_path(self, rect: QRectF) -> QPainterPath:
        """The two edge runs meeting at the current corner, with its arc."""
        radius, arc = float(self._radius), float(self._arc)
        path = QPainterPath()
        name = CORNERS[self._corner]
        if name == "topLeft":
            box = QRectF(rect.left(), rect.top(), radius * 2, radius * 2)
            path.moveTo(rect.left(), rect.top() + radius + arc)
            path.lineTo(rect.left(), rect.top() + radius)
            path.arcTo(box, 180, -90)
            path.lineTo(rect.left() + radius + arc, rect.top())
        elif name == "topRight":
            box = QRectF(rect.right() - radius * 2, rect.top(),
                         radius * 2, radius * 2)
            path.moveTo(rect.right() - radius - arc, rect.top())
            path.lineTo(rect.right() - radius, rect.top())
            path.arcTo(box, 90, -90)
            path.lineTo(rect.right(), rect.top() + radius + arc)
        elif name == "bottomRight":
            box = QRectF(rect.right() - radius * 2, rect.bottom() - radius * 2,
                         radius * 2, radius * 2)
            path.moveTo(rect.right(), rect.bottom() - radius - arc)
            path.lineTo(rect.right(), rect.bottom() - radius)
            path.arcTo(box, 0, -90)
            path.lineTo(rect.right() - radius - arc, rect.bottom())
        else:
            box = QRectF(rect.left(), rect.bottom() - radius * 2,
                         radius * 2, radius * 2)
            path.moveTo(rect.left() + radius + arc, rect.bottom())
            path.lineTo(rect.left() + radius, rect.bottom())
            path.arcTo(box, 270, -90)
            path.lineTo(rect.left(), rect.bottom() - radius - arc)
        return path
