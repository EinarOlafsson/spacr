"""Compact information-icon link used for spaCR documentation."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QStyle, QToolButton


class InfoLink(QToolButton):
    """An icon-only, accessible button that opens a documentation URL."""

    def __init__(
        self,
        url: str,
        *,
        tooltip: str = "Open documentation",
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
            "Opens the relevant spaCR documentation in a web browser."
        )
        icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxInformation
        )
        if icon.isNull():
            self.setText("i")
        else:
            self.setIcon(icon)
            self.setIconSize(QSize(14, 14))
        self.setFixedSize(20, 20)
        self.clicked.connect(self.open_documentation)

    def url(self) -> str:
        """Return the documentation URL opened by this button."""
        return self._url

    def open_documentation(self) -> None:
        """Open the configured documentation page in the system browser."""
        QDesktopServices.openUrl(QUrl(self._url))
