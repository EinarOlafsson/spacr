"""Home-screen tile widgets.

Two variants:

:class:`Tile`
    Classic square tile with an icon centred above a caption. Kept
    for compatibility with older screens that consume it.

:class:`HTile`
    Horizontal card — icon on the LEFT, name on top, one-line
    description underneath. Minimalist look inspired by iOS Settings
    and the VS Code command palette. This is what the startup screen
    uses by default.

Both emit ``clicked()`` when pressed.
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
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Classic square Tile (kept for backwards compatibility)
# ---------------------------------------------------------------------------

class _TileButton(QPushButton):
    """Button with an animated ``iconPixels`` property so the icon
    tweens on hover without changing the button's outer geometry."""

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
        """Animate the icon toward its hover-zoomed size on cursor enter."""
        self._anim.stop()
        self._anim.setStartValue(self._icon_pixels)
        self._anim.setEndValue(int(self._base_size * 1.18))
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Animate the icon back to its base size on cursor leave."""
        self._anim.stop()
        self._anim.setStartValue(self._icon_pixels)
        self._anim.setEndValue(self._base_size)
        self._anim.start()
        super().leaveEvent(event)


class Tile(QWidget):
    """Large square tile with an icon and a caption underneath.

    Kept for older screens. New home-screen code uses :class:`HTile`.

    :param text: fallback label (also used to derive initials if no icon).
    :param icon: optional QIcon to render inside the tile.
    :param icon_size: base icon side length in px; animates on hover.
    :param tile_size: fixed side length of the tile button in px.
    :param caption: caption shown under the tile; defaults to ``text``.
    :ivar clicked: emitted when the tile button is pressed.
    """

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
        layout.setSpacing(12)
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
        """The tile's text label as passed to the constructor."""
        return self._text


# ---------------------------------------------------------------------------
# Horizontal Tile — the new minimalist home-screen card
# ---------------------------------------------------------------------------

class HTile(QPushButton):
    """Horizontal card: icon left, name + description right.

    Renders as a full-width row inside a two- or three-column grid.
    Uses ``font-family: "Open Sans"`` (Regular for the name, Light
    for the description) and a subtle background that only appears
    on hover — nothing to distract until you know what you want.

    :param text: primary label (e.g. app name).
    :param description: single-line subtitle (e.g. app tagline).
    :param icon: QIcon rendered on the left.
    :param icon_size: icon side length in px.
    :ivar clicked: emitted when the tile is pressed.
    """

    def __init__(
        self,
        text: str,
        description: str = "",
        icon: Optional[QIcon] = None,
        icon_size: int = 40,
        parent=None,
    ):
        super().__init__(parent)
        self._text = text
        self._icon_size = int(icon_size)

        self.setObjectName("HTile")
        self.setCursor(Qt.PointingHandCursor)
        # Custom text layout — clear the button's default text so it
        # doesn't render alongside our QLabels.
        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(icon_size, icon_size))

        self.setSizePolicy(self.sizePolicy().horizontalPolicy(),
                           self.sizePolicy().verticalPolicy())
        self.setMinimumHeight(72)
        self.setToolTip(description or text)

        # Two-line label stack next to the icon.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(icon_size + 24, 8, 16, 8)   # left padding leaves room for QIcon
        layout.setSpacing(0)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        name_lbl = QLabel(text)
        name_lbl.setObjectName("HTileName")
        text_col.addWidget(name_lbl)

        if description:
            desc_lbl = QLabel(description)
            desc_lbl.setObjectName("HTileDesc")
            desc_lbl.setWordWrap(True)
            text_col.addWidget(desc_lbl)
        text_col.addStretch(1)

        layout.addLayout(text_col, 1)

    @property
    def text_label(self) -> str:
        """The tile's primary label as passed to the constructor."""
        return self._text
