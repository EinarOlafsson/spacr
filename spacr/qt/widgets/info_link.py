"""Compact teal documentation dot used for spaCR API links."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter
from PySide6.QtWidgets import QToolButton


class InfoLink(QToolButton):
    """A small teal dot that opens an API-reference URL when pressed."""

    def __init__(
        self,
        url: str,
        *,
        tooltip: str = "Open API reference",
        parent=None,
    ):
        super().__init__(parent)
        self._url = str(url)
        self.setObjectName("InfoLink")
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setAccessibleDescription(
            "Opens the relevant spaCR API reference in a web browser."
        )
        self.setText("")
        self._dot_diameter = 7.0
        # The hit area is deliberately larger than the mark itself.
        self.setFixedSize(14, 14)
        self.clicked.connect(self.open_documentation)

    def paintEvent(self, _event) -> None:
        """Paint only the teal dot; avoid the platform's bulky info icon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if not self.isEnabled():
            colour = QColor("#6A8F8D")
        elif self.isDown():
            colour = QColor("#118D88")
        elif self.underMouse():
            colour = QColor("#48D8D0")
        else:
            colour = QColor("#20B8B0")
        side = self._dot_diameter + (1.0 if self.underMouse() else 0.0)
        left = (self.width() - side) / 2.0
        top = (self.height() - side) / 2.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(QRectF(left, top, side, side))

    def url(self) -> str:
        """Return the documentation URL opened by this button."""
        return self._url

    def open_documentation(self) -> None:
        """Open the configured documentation page in the system browser."""
        QDesktopServices.openUrl(QUrl(self._url))
