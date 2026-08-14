"""Compact teal documentation dot used for spaCR API links."""
from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from .dot_link import DotLink


class InfoLink(DotLink):
    """A small teal dot that opens an API-reference URL when pressed."""

    def __init__(
        self,
        url: str,
        *,
        tooltip: str = "Open API reference",
        parent=None,
    ):
        super().__init__(
            tooltip=tooltip,
            colours=("#20B8B0", "#48D8D0", "#118D88", "#6A8F8D"),
            accessible_description=(
                "Opens the relevant spaCR API reference in a web browser."
            ),
            parent=parent,
        )
        self._url = str(url)
        self.setObjectName("InfoLink")
        self.clicked.connect(self.open_documentation)

    def url(self) -> str:
        """Return the documentation URL opened by this button."""
        return self._url

    def set_url(self, url: str) -> None:
        """Change the destination without rebuilding the compact link."""
        self._url = str(url)

    def open_documentation(self) -> None:
        """Open the configured documentation page in the system browser."""
        QDesktopServices.openUrl(QUrl(self._url))
