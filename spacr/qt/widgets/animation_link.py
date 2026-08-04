"""Purple setting-animation link and its click-triggered animation popup."""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication, QHideEvent, QKeyEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from spacr.setting_animations import SettingAnimation

from ..theme import SPACING, active_palette
from .dot_link import DotLink
# The same player the hover tooltip uses. Imported for its zoom and its
# rounded corners: a `QMovie` can only scale a GIF, so the dot used to open a
# 300-pixel window showing a smaller illustration than a 220-pixel hover did.
from .hover_tooltip import AnimationView


LOGGER = logging.getLogger(__name__)


class AnimationPopup(QFrame):
    """Process-wide popup that plays one packaged setting animation."""

    _INSTANCE: Optional["AnimationPopup"] = None
    DISPLAY_SIZE = 300

    def __init__(self):
        super().__init__(
            None,
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setObjectName("SettingAnimationPopup")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._animation: Optional[SettingAnimation] = None

        self._image = AnimationView(self.DISPLAY_SIZE, self)
        self._image.setObjectName("SettingAnimationImage")

        self._error = QLabel(self)
        self._error.setObjectName("SettingAnimationError")
        self._error.setAlignment(Qt.AlignCenter)
        self._error.setWordWrap(True)
        self._error.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self._error.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            SPACING["xs"], SPACING["xs"], SPACING["xs"], SPACING["xs"]
        )
        layout.setSpacing(0)
        layout.addWidget(self._image)
        layout.addWidget(self._error)
        self._apply_theme()

    @classmethod
    def instance(cls) -> "AnimationPopup":
        """Return the lazily created process-wide popup."""
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def _apply_theme(self) -> None:
        """Refresh top-level popup colours after a live theme change."""
        palette = active_palette()
        self.setStyleSheet(
            "QFrame#SettingAnimationPopup {"
            f"background-color: {palette['surface_alt']};"
            f"border: 1px solid {palette['border']};"
            "border-radius: 8px;"
            "}"
            # Transparent, not black: the rounded corners are cut into the
            # frames themselves and a square background would fill them in.
            "QLabel#SettingAnimationImage { background: transparent; }"
            "QLabel#SettingAnimationError {"
            f"background: #000000; color: {palette['error']};"
            "padding: 12px;"
            "}"
        )

    def show_animation(
        self,
        anchor: QWidget,
        animation: SettingAnimation,
    ) -> None:
        """Load ``animation`` and show it above ``anchor`` when possible."""
        from .hover_tooltip import HoverTooltip

        tooltip = HoverTooltip._INSTANCE
        if tooltip is not None:
            tooltip.cancel_hide()
            tooltip.hide()

        self._apply_theme()
        self._animation = animation
        self.setWindowTitle(animation.title)
        self.setAccessibleName(animation.title)
        self.setAccessibleDescription(
            "Animated explanation of the selected spaCR setting."
        )
        self._error.hide()
        self._image.show()

        # Zoomed by `animation_zoom`, exactly as the hover tooltip zooms it,
        # so the same illustration is not smaller in the bigger window.
        if self._image.load(animation):
            self._error.hide()
        else:
            LOGGER.error(
                "Could not load setting animation %s from %s",
                animation.slug,
                animation.path,
            )
            self._image.hide()
            self._error.setText(
                "This setting animation could not be loaded. "
                "See the debug log for its asset path."
            )
            self._error.show()

        self.adjustSize()
        self._position_near(anchor)
        self.show()
        self._image.play()

    def animation_view(self) -> AnimationView:
        """The square player — exposed for layout and zoom tests."""
        return self._image

    def animation(self) -> Optional[SettingAnimation]:
        """The registry entry currently loaded, or ``None``."""
        return self._animation

    def _position_near(self, anchor: QWidget) -> None:
        """Centre over the setting row, preferring the space above it."""
        row_anchor = anchor
        parent = anchor.parentWidget()
        while parent is not None:
            if parent.objectName() in {
                "SettingLabelWithInfo",
                "SettingControlWithInfo",
            }:
                row_anchor = parent
                break
            parent = parent.parentWidget()

        try:
            top_left = row_anchor.mapToGlobal(row_anchor.rect().topLeft())
            bottom_left = row_anchor.mapToGlobal(row_anchor.rect().bottomLeft())
            centre_x = top_left.x() + row_anchor.width() // 2
        except RuntimeError:
            top_left = QPoint(0, 0)
            bottom_left = QPoint(0, 0)
            centre_x = 0

        screen = (
            QGuiApplication.screenAt(top_left)
            or QGuiApplication.primaryScreen()
        )
        x = centre_x - self.width() // 2
        y = top_left.y() - self.height() - 4
        if screen is not None:
            geometry = screen.availableGeometry()
            x = min(max(geometry.left(), x), geometry.right() - self.width() + 1)
            if y < geometry.top():
                y = bottom_left.y() + 4
            y = min(max(geometry.top(), y), geometry.bottom() - self.height() + 1)
        self.move(x, y)

    def hideEvent(self, event: QHideEvent) -> None:
        """Stop swapping frames whenever the popup is no longer visible."""
        self._image.stop()
        super().hideEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Close immediately on Escape; defer other keys to Qt."""
        if event.key() == Qt.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)


class AnimationLink(DotLink):
    """Purple dot that opens one explanatory setting animation."""

    def __init__(
        self,
        animation: SettingAnimation,
        *,
        tooltip: str = "Show setting animation",
        parent=None,
    ):
        super().__init__(
            tooltip=tooltip,
            colours=("#9B009B", "#D14AD1", "#700070", "#765A76"),
            accessible_description=(
                "Shows a short animated explanation of this spaCR setting."
            ),
            parent=parent,
        )
        self._animation = animation
        self.setObjectName("SettingAnimationLink")
        self.clicked.connect(self.open_animation)

    def animation(self) -> SettingAnimation:
        """Return the immutable registry entry opened by this link."""
        return self._animation

    def open_animation(self) -> None:
        """Open this link's animation in the shared popup."""
        AnimationPopup.instance().show_animation(self, self._animation)


class SettingLinkStack(QWidget):
    """Vertically stack animation and API dots around a shared midpoint."""

    def __init__(
        self,
        animation_link: AnimationLink,
        api_link: QWidget,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("SettingLinkStack")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(animation_link)
        layout.addWidget(api_link)
        self.setFixedSize(14, 28)
        self.animation_link = animation_link
        self.api_link = api_link


__all__ = ["AnimationLink", "AnimationPopup", "SettingLinkStack"]
