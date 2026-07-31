"""Shared custom-painted documentation-dot button."""
from __future__ import annotations

from typing import Tuple

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QToolButton


class DotLink(QToolButton):
    """Small circular link with a generous, accessible hit area."""

    def __init__(
        self,
        *,
        tooltip: str,
        colours: Tuple[str, str, str, str],
        accessible_description: str,
        parent=None,
    ):
        super().__init__(parent)
        self._colours = colours
        self._dot_diameter = 7.0
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setAccessibleDescription(accessible_description)
        self.setText("")
        # The visible mark is deliberately smaller than its hit target.
        self.setFixedSize(14, 14)

    def paintEvent(self, _event) -> None:
        """Paint only the state-coloured dot, never a platform icon."""
        normal, hover, pressed, disabled = self._colours
        if not self.isEnabled():
            colour = QColor(disabled)
        elif self.isDown():
            colour = QColor(pressed)
        elif self.underMouse():
            colour = QColor(hover)
        else:
            colour = QColor(normal)
        side = self._dot_diameter + (1.0 if self.underMouse() else 0.0)
        left = (self.width() - side) / 2.0
        top = (self.height() - side) / 2.0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(QRectF(left, top, side, side))


__all__ = ["DotLink"]
