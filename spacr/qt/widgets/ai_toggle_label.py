"""
AiToggleLabel — a clickable text label used in place of a QCheckBox
for the "AI" switch that sits at the bottom-right of every AppScreen.

* Reads "AI" in the current theme's foreground colour when off — white
  on the dark themes, near-black on light.
* Reads "AI" in the accent blue when on.
* Emits `toggled(bool)` on click; also exposes a QCheckBox-compatible
  `isChecked()` / `setChecked()` API so the AppScreen doesn't care
  which widget it's talking to.

.. note::

   The OFF colour is resolved through :func:`spacr.qt.theme.active_palette`
   (i.e. ``palette_for(resolve_effective_theme())``) every time the style
   is rebuilt, never imported from ``theme.PALETTE``. That module-level
   name is the *dark* palette and nothing updates it, so importing it
   painted ``#ffffff`` "AI" text onto the light theme's ``#fafafa`` page:
   **1.04:1** measured, white on near-white, invisible. It is 18.50:1
   now. See :mod:`tests.qt.test_theme_blind_console_widgets`.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel

from ..theme import FONT_SIZE, active_palette


class AiToggleLabel(QLabel):
    """Clickable text label that behaves like a QCheckBox toggle.

    Originally the "AI" switch, now also used for the "LP" (live
    preview) toggle. Every consumer gets the same on-blue / off-white
    visual so the row of toggles reads consistently.

    :param text: label text (default ``"AI"`` for back-compat).
    :param tooltip: hover tooltip; falls back to a sensible AI-flavoured
        message when omitted.
    :ivar toggled: emitted with the new on/off state whenever the user
        clicks or :meth:`setChecked` flips the state.
    """

    toggled = Signal(bool)

    def __init__(self, parent=None, text: str = "AI",
                     tooltip: str | None = None):
        super().__init__(text, parent)
        self.setObjectName("AiToggleLabel")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip if tooltip is not None else (
            "Click to toggle AI. When ON (blue), pressing Enter in "
            "the console routes your message through your chat "
            "subscription via the selected provider."
        ))
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
        # Use the theme-invariant ``button_accent`` for the ON colour so
        # the toggle looks identical in every theme. The OFF colour is
        # the theme's own ``fg``, resolved HERE rather than imported:
        # `active_palette()` reads the preference that is in force right
        # now, so the label inks white on dark and near-black on light.
        # It used to come from `theme.PALETTE`, which is frozen dark —
        # white "AI" on the light theme's #fafafa page.
        palette = active_palette()
        on_color = palette["button_accent"]
        color = on_color if self._on else palette["fg"]
        self.setStyleSheet(
            f"QLabel#AiToggleLabel {{"
            f"  color: {color};"
            f"  font-size: {FONT_SIZE['body']}px;"
            f"  font-weight: 600;"
            f"  padding: 4px 10px;"
            f"  background: transparent;"
            f"}}"
        )
