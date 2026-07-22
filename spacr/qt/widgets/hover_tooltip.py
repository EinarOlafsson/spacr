"""
HoverTooltip — a QFrame-based popup that stays visible when the mouse
enters it. Unlike QToolTip, users can move their cursor into the popup
to click links inside.

Usage:
    tip = HoverTooltip.instance()
    tip.show_for(some_widget, "some html")   # on hover-enter
    tip.start_hide()                          # on hover-leave
The popup cancels its own hide timer if the mouse enters it, and only
actually hides when neither the anchor nor the popup itself is under
the cursor.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from ..theme import PALETTE, SPACING


class HoverTooltip(QFrame):
    """Sticky QFrame popup that survives cursor entry so users can click links.

    Access via :meth:`instance` — the popup is a process-wide singleton.
    """

    _INSTANCE: Optional["HoverTooltip"] = None

    def __init__(self):
        # Popup window with tool-tip semantics but our own paint control.
        super().__init__(
            None,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setObjectName("HoverTooltip")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        # Inline QSS — this widget lives above the app stylesheet as a
        # separate top-level window, so app-level QSS doesn't reach it.
        self.setStyleSheet(
            f"QFrame#HoverTooltip {{"
            f"  background-color: {PALETTE['surface_alt']};"
            f"  border: 1px solid {PALETTE['border']};"
            f"  border-radius: 6px;"
            f"}}"
            f"QLabel {{"
            f"  color: {PALETTE['fg']};"
            f"  font-size: 12px;"
            f"  background: transparent;"
            f"}}"
        )
        self._label = QLabel(self)
        self._label.setTextFormat(Qt.RichText)
        self._label.setOpenExternalLinks(True)
        self._label.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.LinksAccessibleByMouse
        )
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(380)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACING["sm"], SPACING["xs"],
                                SPACING["sm"], SPACING["xs"])
        lay.addWidget(self._label)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._maybe_hide)
        self._anchor: Optional[QWidget] = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "HoverTooltip":
        """Return the process-wide singleton, creating it on first access."""
        if cls._INSTANCE is None:
            cls._INSTANCE = HoverTooltip()
        return cls._INSTANCE

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def show_for(self, anchor: QWidget, html: str) -> None:
        """Show the tooltip beneath ``anchor`` with body ``html``.

        :param anchor: widget the popup docks to (clamped to its screen).
        :param html: rich-text body; empty strings are ignored.
        """
        if not html:
            return
        self._anchor = anchor
        self._label.setText(html)
        self.adjustSize()
        # Position: just below the anchor, left-aligned to its left edge,
        # clamped to the screen so we never overflow.
        try:
            below_left = anchor.mapToGlobal(anchor.rect().bottomLeft())
        except Exception:
            below_left = QPoint(0, 0)
        screen = QGuiApplication.screenAt(below_left) \
                 or QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = min(max(geo.left(), below_left.x()),
                    geo.right() - self.width())
            y = below_left.y() + 4
            if y + self.height() > geo.bottom():
                # Not enough space below — flip above
                y = anchor.mapToGlobal(anchor.rect().topLeft()).y() \
                    - self.height() - 4
            self.move(x, y)
        else:
            self.move(below_left)
        self.show()

    def start_hide(self, delay_ms: int = 250) -> None:
        """Schedule a hide after ``delay_ms`` unless the cursor re-enters."""
        self._hide_timer.start(delay_ms)

    def cancel_hide(self) -> None:
        """Cancel any pending hide timer (called on cursor re-entry)."""
        self._hide_timer.stop()

    # ------------------------------------------------------------------
    def _maybe_hide(self) -> None:
        if self.underMouse():
            return
        if self._anchor is not None and self._anchor.underMouse():
            return
        self.hide()

    def enterEvent(self, event):
        """Cancel the hide timer when the cursor enters the popup."""
        self.cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Restart the hide timer with a short delay when the cursor leaves."""
        self.start_hide(delay_ms=100)
        super().leaveEvent(event)
