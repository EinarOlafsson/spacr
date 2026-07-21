"""
Tile — a large square button + caption pair used on the startup screen.

Signals:
    clicked() — emitted when the tile is pressed.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class Tile(QWidget):
    clicked = Signal()

    def __init__(
        self,
        text: str,
        icon: Optional[QIcon] = None,
        icon_size: int = 64,
        tile_size: int = 120,
        caption: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._text = text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)   # room between tile and caption
        layout.setAlignment(Qt.AlignHCenter)

        self._button = QPushButton()
        self._button.setObjectName("Tile")
        self._button.setFixedSize(tile_size, tile_size)
        self._button.setCursor(Qt.PointingHandCursor)
        if icon is not None:
            self._button.setIcon(icon)
            self._button.setIconSize(QSize(icon_size, icon_size))
        else:
            initials = "".join(w[0].upper() for w in text.split()[:2])[:2]
            self._button.setText(initials or text[:2].upper())
        self._button.clicked.connect(self.clicked.emit)
        self._button.setToolTip(caption or text)
        layout.addWidget(self._button, alignment=Qt.AlignHCenter)

        self._caption = QLabel(caption or text)
        self._caption.setObjectName("TileCaption")
        self._caption.setAlignment(Qt.AlignHCenter)
        self._caption.setWordWrap(True)
        self._caption.setMaximumWidth(tile_size + 40)
        layout.addWidget(self._caption)

    @property
    def text(self) -> str:
        return self._text
