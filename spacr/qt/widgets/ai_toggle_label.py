"""
AiToggleLabel — a clickable text label used in place of a QCheckBox
for the "AI" switch that sits at the bottom-right of every AppScreen.

* Reads "AI" in white when off.
* Reads "AI" in the accent blue when on.
* Emits `toggled(bool)` on click; also exposes a QCheckBox-compatible
  `isChecked()` / `setChecked()` API so the AppScreen doesn't care
  which widget it's talking to.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel

from ..theme import FONT_SIZE, PALETTE


class AiToggleLabel(QLabel):
    """Clickable "AI" text label that behaves like a QCheckBox toggle.

    :ivar toggled: emitted with the new on/off state whenever the user
        clicks or :meth:`setChecked` flips the state.
    """

    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__("AI", parent)
        self.setObjectName("AiToggleLabel")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Click to toggle AI. When ON (blue), pressing Enter in "
            "the console routes your message through your chat "
            "subscription via the selected provider."
        )
        self._on = False
        self._refresh_style()

    # -- QCheckBox-compat API -----------------------------------------
    def isChecked(self) -> bool:
        """Return True when the AI toggle is currently ON."""
        return self._on

    def setChecked(self, on: bool) -> None:
        """Set the toggle state; emits ``toggled`` only on a real change."""
        on = bool(on)
        if on == self._on:
            return
        self._on = on
        self._refresh_style()
        self.toggled.emit(self._on)

    # -- click ---------------------------------------------------------
    def mousePressEvent(self, event):
        """Flip the toggle on left-click; forward other buttons to Qt."""
        if event.button() == Qt.LeftButton:
            self._on = not self._on
            self._refresh_style()
            self.toggled.emit(self._on)
            return
        super().mousePressEvent(event)

    # -- style ---------------------------------------------------------
    def _refresh_style(self) -> None:
        color = PALETTE["accent"] if self._on else PALETTE["fg"]
        self.setStyleSheet(
            f"QLabel#AiToggleLabel {{"
            f"  color: {color};"
            f"  font-size: {FONT_SIZE['body']}px;"
            f"  font-weight: 600;"
            f"  padding: 4px 10px;"
            f"  background: transparent;"
            f"}}"
        )
