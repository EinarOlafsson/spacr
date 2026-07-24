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
        icon_size: int = 52,
        parent=None,
    ):
        super().__init__(parent)
        self._text = text

        # Icon size and all icon-adjacent geometry track the user's font-size
        # preference so the tile grows with the text and nothing clips when
        # the font is bumped up. ``icon_size`` is the base (100 %) side length.
        from ..preferences import scaled_px
        self._base_icon = scaled_px(int(icon_size))
        self._icon_pixels = self._base_icon

        # Hover zoom animation on the icon (slightly enlarges on cursor enter).
        self._icon_anim = QPropertyAnimation(self, b"iconPixels", self)
        self._icon_anim.setDuration(140)
        self._icon_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.setObjectName("HTile")
        self.setCursor(Qt.PointingHandCursor)
        # Accessibility: screen readers announce the app name + one-line
        # description as the button's role. Tooltip stays for sighted
        # hover; the accessible bits are what NVDA / VoiceOver read.
        self.setAccessibleName(text)
        if description:
            self.setAccessibleDescription(description)
        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(self._base_icon, self._base_icon))

        # Height tracks the font scale AND the icon so tall/large fonts don't
        # clip the two-line label stack. Grow vertically to fit content.
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setMinimumHeight(max(scaled_px(72), self._base_icon + scaled_px(24)))
        self.setToolTip(description or text)

        # Two-line label stack next to the icon. Left padding (scaled) leaves
        # room for the QIcon the button paints on the left edge.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._base_icon + scaled_px(24),
                                  scaled_px(8), scaled_px(16), scaled_px(8))
        layout.setSpacing(0)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        name_lbl = QLabel(text)
        name_lbl.setObjectName("HTileName")
        # Don't clip — the tile stretches to accommodate the label
        # when longer app names appear. Explicit minimum width so
        # short names still look proportionate.
        name_lbl.setMinimumWidth(0)
        from PySide6.QtWidgets import QSizePolicy
        name_lbl.setSizePolicy(QSizePolicy.Expanding,
                                 QSizePolicy.Preferred)
        if description:
            # Description shown BELOW the name (two-line tile).
            text_col.addStretch(1)
            text_col.addWidget(name_lbl)
            desc_lbl = QLabel(description)
            desc_lbl.setObjectName("HTileDesc")
            desc_lbl.setWordWrap(True)
            text_col.addWidget(desc_lbl)
            text_col.addStretch(1)
        else:
            # Name-only tile: vertically centre the label so it sits
            # in the middle rather than pinned to the top-left.
            text_col.addStretch(1)
            text_col.addWidget(name_lbl)
            text_col.addStretch(1)

        layout.addLayout(text_col, 1)

    # -- hover-zoom icon animation ------------------------------------------
    def _get_icon_pixels(self) -> int:
        return self._icon_pixels

    def _set_icon_pixels(self, v: int) -> None:
        self._icon_pixels = int(v)
        self.setIconSize(QSize(self._icon_pixels, self._icon_pixels))

    iconPixels = Property(int, _get_icon_pixels, _set_icon_pixels)

    def enterEvent(self, event: QEvent) -> None:
        """Slightly enlarge the icon on cursor enter."""
        self._icon_anim.stop()
        self._icon_anim.setStartValue(self._icon_pixels)
        self._icon_anim.setEndValue(int(self._base_icon * 1.15))
        self._icon_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Return the icon to its base size on cursor leave."""
        self._icon_anim.stop()
        self._icon_anim.setStartValue(self._icon_pixels)
        self._icon_anim.setEndValue(self._base_icon)
        self._icon_anim.start()
        super().leaveEvent(event)

    @property
    def text_label(self) -> str:
        """The tile's primary label as passed to the constructor."""
        return self._text
