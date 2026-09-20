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

#: How much wider than its tile a :class:`Tile` caption may run, in logical
#: pixels before the interface scale. Named because it is scaled now: the
#: cap and the tile edge it is derived from have to move together, and a
#: literal added to an already-scaled edge would shrink as a share of it.
CAPTION_SLACK_PX = 40

#: Shortest an :class:`HTile` may be, in logical pixels before scaling.
HTILE_MIN_HEIGHT_PX = 72

#: Gap between an :class:`HTile`'s icon and its text, before scaling. The
#: left margin is this plus the icon's own width, which is how the icon is
#: given its column without being in the layout.
HTILE_ICON_GAP_PX = 24

#: The :class:`HTile` margins that are not the icon's column: right, and
#: top and bottom, in logical pixels before scaling.
HTILE_MARGIN_PX = 16
HTILE_MARGIN_Y_PX = 8


class _TileButton(QPushButton):
    """Button with an animated ``iconPixels`` property so the icon
    tweens on hover without changing the button's outer geometry.

    :param base_size: the icon's resting side length in px. The animation
        returns to it on leave, so it is the size the tile reads as -- the
        BUTTON's geometry is set by the caller and does not follow it.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, base_size: int, parent=None):
        """Build the button with its icon at the resting size.

        ``base_size`` is the size at 100 %: the interface scale is applied
        here rather than by the caller, so a tile built after a zoom comes
        up at the same size as the tiles already on screen, and so
        :meth:`_apply_icon_scale` has an unscaled number to work from.
        """
        super().__init__(parent)
        from ..preferences import scaled_px
        self._icon_base_px = int(base_size)
        self._base_size = scaled_px(self._icon_base_px)
        self._icon_pixels = self._base_size
        self._anim = QPropertyAnimation(self, b"iconPixels", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.setIconSize(QSize(self._icon_pixels, self._icon_pixels))

    def _get_icon_pixels(self) -> int:
        """The animated icon size, in px. Read by the property."""
        return self._icon_pixels

    def _set_icon_pixels(self, v: int) -> None:
        """Set the icon size and apply it. Written by the animation."""
        self._icon_pixels = int(v)
        self.setIconSize(QSize(self._icon_pixels, self._icon_pixels))

    iconPixels = Property(int, _get_icon_pixels, _set_icon_pixels)

    def _apply_icon_scale(self, scale=None) -> None:
        """Move the resting size, and the icon with it, to a new scale.

        THE RESTING SIZE IS THE ONE THAT MATTERS. Setting the icon size
        alone would look right until the pointer next crossed the tile:
        the leave animation returns the icon to ``_base_size``, so a stale
        resting size undoes the zoom on the first hover. The animation is
        stopped for the same reason -- a running tween would land on the
        size it was aimed at before the scale moved.

        Both numbers come from ``_icon_base_px``, the size at 100 %, so
        zooming out returns the tile to the pixel it started at rather
        than to whatever twenty roundings left behind.

        :param scale: the interface scale; the stored preference when None.
        """
        from ..preferences import _scaled_side, get_font_scale

        if scale is None:
            scale = get_font_scale()
        self._anim.stop()
        self._base_size = _scaled_side(self._icon_base_px, scale)
        self._set_icon_pixels(self._base_size)

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
    :param parent: parent widget; ownership only.
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
        """Build a square tile: an icon over its label.

        :param text: the label.
        :param icon: the picture; ``None`` leaves the tile text-only.
        :param icon_size: the icon's edge, in pixels.
        :param tile_size: the tile's edge, in pixels.
        :param caption: a second line under the label.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        from ..preferences import scaled_px
        self._text = text
        self._tile_base_px = int(tile_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignHCenter)

        self._button = _TileButton(icon_size)
        self._button.setObjectName("Tile")
        side = scaled_px(self._tile_base_px)
        self._button.setFixedSize(side, side)
        self._button.setCursor(Qt.PointingHandCursor)
        if icon is not None:
            self._button.setIcon(icon)
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
        self._caption.setMaximumWidth(scaled_px(self._tile_base_px
                                                + CAPTION_SLACK_PX))
        layout.addWidget(self._caption)

    def _apply_icon_scale(self, scale=None) -> None:
        """Re-square the tile and re-cap its caption at a new scale.

        THE FRAME HAS TO MOVE WITH THE MARK. The button's edge is fixed,
        so an icon that grew inside it would be cropped by its own tile
        long before it reached the size the wheel asked for. The caption's
        width cap is derived from the same edge and moves with it, or a
        caption that fitted on one line at 100 % wraps to three at 200 %.

        The button re-sizes its own icon: it carries its own copy of this
        method, and :func:`spacr.qt.preferences._rescale_icon_sizes` visits
        every widget, so both are reached without either calling the
        other.

        :param scale: the interface scale; the stored preference when None.
        """
        from ..preferences import _scaled_side, get_font_scale

        if scale is None:
            scale = get_font_scale()
        side = _scaled_side(self._tile_base_px, scale)
        self._button.setFixedSize(side, side)
        self._caption.setMaximumWidth(
            _scaled_side(self._tile_base_px + CAPTION_SLACK_PX, scale))

    @property
    def text(self) -> str:
        """The tile's text label as passed to the constructor."""
        return self._text



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
    :param parent: parent widget; ownership only.
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
        """Build a wide tile: an icon beside its label and description.

        :param text: the label.
        :param description: the sentence beside it.
        :param icon: the picture; ``None`` leaves the tile text-only.
        :param icon_size: the icon's edge, in pixels.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self._text = text
        self._name_lbl = None

        from ..preferences import scaled_px
        self._icon_base_px = int(icon_size)
        self._base_icon = scaled_px(self._icon_base_px)

        self.setObjectName("HTile")
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(text)
        if description:
            self.setAccessibleDescription(description)
        if icon is not None:
            self.setIcon(icon)

        self.setToolTip(f"{text} — {description}" if description else text)

        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        self._apply_icon_scale(None)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        name_lbl = ElidingLabel(text)
        name_lbl.setObjectName("HTileName")
        name_lbl.setMinimumWidth(0)
        from PySide6.QtWidgets import QSizePolicy
        name_lbl.setSizePolicy(QSizePolicy.Expanding,
                                 QSizePolicy.Preferred)
        self._name_lbl = name_lbl
        if description:
            text_col.addStretch(1)
            text_col.addWidget(name_lbl)
            desc_lbl = QLabel(description)
            desc_lbl.setObjectName("HTileDesc")
            desc_lbl.setWordWrap(True)
            text_col.addWidget(desc_lbl)
            text_col.addStretch(1)
        else:
            text_col.addStretch(1)
            text_col.addWidget(name_lbl)
            text_col.addStretch(1)

        layout.addLayout(text_col, 1)

    def _apply_icon_scale(self, scale=None) -> None:
        """Re-size the icon, and the column the layout keeps clear for it.

        THE ICON IS NOT IN THE LAYOUT. It is painted by the button, and
        the text is kept out of its way by a left margin wide enough to
        hold it -- so an icon that grew on its own would grow straight
        under the name. Every one of these numbers moves together or the
        card comes apart: at 200 % a card sized for a 52 px mark has a
        104 px mark drawn across its first word.

        Called at construction, so a tile built after a zoom comes up at
        the scale already on screen, and again from
        :func:`spacr.qt.preferences._rescale_icon_sizes` when the scale
        moves. Every size is derived from ``_icon_base_px`` and the module
        constants -- the sizes at 100 % -- so the same scale always gives
        the same pixel, whichever direction the wheel reached it from.

        :param scale: the interface scale; the stored preference when None.
        """
        from ..preferences import (_scaled_side, _set_scaled_icon_size,
                                   get_font_scale)

        if scale is None:
            scale = get_font_scale()
        self._base_icon = _scaled_side(self._icon_base_px, scale)
        if not self.icon().isNull():
            _set_scaled_icon_size(self, self._icon_base_px, scale=scale)
        self.setMinimumHeight(_scaled_side(HTILE_MIN_HEIGHT_PX, scale))
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(
                self._base_icon + _scaled_side(HTILE_ICON_GAP_PX, scale),
                _scaled_side(HTILE_MARGIN_Y_PX, scale),
                _scaled_side(HTILE_MARGIN_PX, scale),
                _scaled_side(HTILE_MARGIN_Y_PX, scale))
        self.updateGeometry()

    def required_width(self) -> int:
        """Width in px at which this tile shows its whole name.

        Layout margins (which already reserve room for the icon) plus
        the advance of the full name, with a 2 px guard for the
        sub-pixel rounding QLabel does when it lays the text out.
        """
        layout = self.layout()
        if self._name_lbl is None or layout is None:
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
