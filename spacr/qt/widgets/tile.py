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
from PySide6.QtGui import QFontMetrics, QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .eliding import ElidingLabel


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
        # Set before any layout work: Qt can ask for sizeHint() while the
        # widget is still being built, and the hint reads this attribute.
        self._name_lbl = None

        # Icon size and all icon-adjacent geometry track the user's font-size
        # preference so the tile grows with the text and nothing clips when
        # the font is bumped up. ``icon_size`` is the base (100 %) side length.
        # (No hover zoom — the icon/text stay a fixed size on hover.)
        from ..preferences import scaled_px
        self._base_icon = scaled_px(int(icon_size))

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

        # Height tracks the font scale (scaled_px) but keeps the original
        # proportions — the earlier icon-driven height made the tiles too
        # tall. Width is handled by the caller (also via scaled_px).
        self.setMinimumHeight(scaled_px(72))
        # Tooltip leads with the NAME even when there's a description, so
        # a tile too narrow for its label is still identifiable on hover.
        self.setToolTip(f"{text} — {description}" if description else text)

        # Two-line label stack next to the icon. Left padding (scaled) leaves
        # room for the QIcon the button paints on the left edge.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._base_icon + scaled_px(24),
                                  scaled_px(8), scaled_px(16), scaled_px(8))
        layout.setSpacing(0)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        # The name is an ElidingLabel, not a plain QLabel: a plain one
        # silently clips ("Annotator Agreeme") when the tile is narrower
        # than the name, which is unreadable and unclickable. This one
        # shortens with an ellipsis and moves the full name into the
        # tooltip — and :meth:`sizeHint` below makes sure the tile is
        # usually wide enough that it never has to.
        name_lbl = ElidingLabel(text)
        name_lbl.setObjectName("HTileName")
        # Don't clip — the tile stretches to accommodate the label
        # when longer app names appear. Explicit minimum width so
        # short names still look proportionate.
        name_lbl.setMinimumWidth(0)
        from PySide6.QtWidgets import QSizePolicy
        name_lbl.setSizePolicy(QSizePolicy.Expanding,
                                 QSizePolicy.Preferred)
        self._name_lbl = name_lbl
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

    # -- geometry ------------------------------------------------------
    #
    # HTile draws its name in a CHILD QLabel, not in the button's own
    # text. QPushButton.sizeHint()/minimumSizeHint() only measure the
    # button's own text + icon, so without these overrides the label's
    # width requirement never reaches the layout: every tile reported
    # the same ~92 px hint no matter how long the app name was, callers
    # that did `max(floor, tile.sizeHint().width())` always got the
    # floor, and anything longer than the floor left over got clipped.

    def required_width(self) -> int:
        """Width in px at which this tile shows its whole name.

        Layout margins (which already reserve room for the icon) plus
        the advance of the full name, with a 2 px guard for the
        sub-pixel rounding QLabel does when it lays the text out.
        """
        layout = self.layout()
        if self._name_lbl is None or layout is None:
            # Asked mid-construction — fall back to the plain button hint.
            return QPushButton.sizeHint(self).width()
        self.ensurePolished()
        self._name_lbl.ensurePolished()
        margins = layout.contentsMargins()
        advance = QFontMetrics(self._name_lbl.font()).horizontalAdvance(
            self._text)
        return margins.left() + margins.right() + advance + 2

    def sizeHint(self) -> QSize:               # noqa: N802 (Qt casing)
        """Report the width the name actually needs, not just the icon's."""
        base = super().sizeHint()
        return QSize(max(base.width(), self.required_width()),
                     max(base.height(), self.minimumHeight()))

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        """Stay shrinkable — the name elides when a caller caps the width."""
        base = QPushButton.sizeHint(self)
        return QSize(min(base.width(), self.required_width()),
                     max(base.height(), self.minimumHeight()))

    def is_name_elided(self) -> bool:
        """True when the tile is too narrow to show the whole name."""
        return self._name_lbl is not None and self._name_lbl.is_elided()

    @property
    def name_label(self) -> ElidingLabel:
        """The label that renders the tile's name."""
        return self._name_lbl

    @property
    def text_label(self) -> str:
        """The tile's primary label as passed to the constructor."""
        return self._text
