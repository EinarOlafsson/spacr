"""Tile — a large square button + caption used on the startup screen.

Hovering the tile animates its icon so the glyph appears to grow
inside the button frame — the button itself stays the same size.

Signals:
    clicked() — emitted when the tile is pressed.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class _TileButton(QPushButton):
    """QPushButton subclass with an animated `iconPixels` property so
    the icon can be tweened on hover without touching the button's
    outer geometry."""

    def __init__(self, base_size: int, parent=None):
        super().__init__(parent)
        self._base_size = int(base_size)
        self._icon_pixels = int(base_size)
        self._anim = QPropertyAnimation(self, b"iconPixels", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _get_icon_pixels(self) -> int:
        return self._icon_pixels

    def _set_icon_pixels(self, v: int) -> None:
        self._icon_pixels = int(v)
        self.setIconSize(QSize(self._icon_pixels, self._icon_pixels))

    iconPixels = Property(int, _get_icon_pixels, _set_icon_pixels)

    def enterEvent(self, event: QEvent) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._icon_pixels)
        self._anim.setEndValue(int(self._base_size * 1.18))   # 18% zoom
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._icon_pixels)
        self._anim.setEndValue(self._base_size)
        self._anim.start()
        super().leaveEvent(event)


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

        self._button = _TileButton(icon_size)
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
