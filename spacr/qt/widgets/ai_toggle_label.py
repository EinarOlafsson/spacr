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

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtWidgets import QLabel

from ..i18n import tr
from ..theme import active_palette, font_px

#: The widest a toggle may force the row it sits in to be.
#:
#: These toggles live in the action row, and that row's minimum width is
#: the minimum width of the whole screen -- a QHBoxLayout cannot go below
#: the sum of its children. "Live" is 83px and never caused anyone
#: trouble, which is why this went unnoticed for as long as it did.
#: "Hyperparameter search" is 281px, and it held the action row at 1109px
#: on every module with a hyperparameter panel (activation, classify,
#: classify_merged, ml_analyze). At a 1200px window the body splitter was
#: then left with 60px for the entire settings column -- a 290px settings
#: card inside a 50px viewport, with the labels hanging out of the right
#: of it.
#:
#: Only the MINIMUM is capped. `sizeHint` still asks for the full text, so
#: a layout with room to spare shows all of it; a squeezed one elides
#: instead of forcing the window wider. A secondary control must not be
#: able to starve the primary one -- see INVARIANTS 10.
ELIDE_ABOVE_PX = 120


class AiToggleLabel(QLabel):
    """Clickable text label that behaves like a QCheckBox toggle.

    Originally the "AI" switch, now also used for the "Live" preview
    toggle. Every consumer gets the same on-blue / off-white
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
        source_text = str(text)
        source_tooltip = tooltip if tooltip is not None else (
            "Click to toggle AI. When ON (blue), pressing Enter in "
            "the console routes your message through your chat "
            "subscription via the selected provider."
        )
        super().__init__(tr(source_text), parent)
        # Retain canonical English sources so a runtime language switch never
        # translates a translation and never loses the toggle's current state.
        self.setProperty("_spacr_i18n_text", source_text)
        self.setObjectName("AiToggleLabel")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("_spacr_i18n_tooltip", source_tooltip)
        self.setToolTip(tr(source_tooltip))
        self._on = False
        self._restyling = False
        # The logical text, always. `QLabel.text()` holds whatever fits
        # right now, which may be elided; every caller that asks this
        # widget what it says wants the full thing.
        self._full_text = tr(source_text)
        self._eliding = False
        self._refresh_style()

    # -- live preference changes ---------------------------------------
    def changeEvent(self, event):
        """Re-style when the application sheet or palette is replaced.

        Saving Preferences calls ``app.setStyleSheet(...)``, which sends
        every widget a ``StyleChange``. Widgets styled by the application
        sheet pick the new Zoom up for free; this one is not, so without
        this hook "Live" and "AI" would keep the size and the colour they
        were built with until the app was restarted.
        """
        try:
            kind = event.type()
        except Exception:              # pragma: no cover - defensive
            kind = None
        super().changeEvent(event)
        if kind in (QEvent.StyleChange, QEvent.PaletteChange,
                    QEvent.ApplicationPaletteChange,
                    QEvent.ApplicationFontChange):
            self._refresh_style()

    # -- width -----------------------------------------------------------
    def minimumSizeHint(self) -> QSize:      # noqa: N802 (Qt naming)
        """Cap how much width this toggle can demand of its row.

        QLabel reports the full width of its text here, which makes the
        text a hard floor for every ancestor layout. See
        :data:`ELIDE_ABOVE_PX` for what that cost.
        """
        hint = super().minimumSizeHint()
        if hint.width() <= ELIDE_ABOVE_PX:
            return hint
        return QSize(ELIDE_ABOVE_PX, hint.height())

    def setText(self, text) -> None:         # noqa: N802 (Qt naming)
        """Remember the full text, then show as much of it as fits.

        The language switch calls this with a fresh translation, so the
        stored text has to follow it rather than be captured once.
        """
        if not self._eliding:
            self._full_text = str(text)
        super().setText(text)
        if not self._eliding:
            self._apply_elision()

    def text(self) -> str:
        """The full logical text, even when the label is showing less."""
        return getattr(self, "_full_text", None) or super().text()

    def displayed_text(self) -> str:
        """What the label is actually painting, elided or not.

        Distinct from :meth:`text`, which is the logical label. Tests need
        both to tell "the toggle says X" from "the toggle currently fits
        this much of X".
        """
        return QLabel.text(self)

    def resizeEvent(self, event):            # noqa: N802 (Qt naming)
        """Re-elide for the width just granted."""
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self) -> None:
        """Show the full text when it fits, an elided one when it does not."""
        full = getattr(self, "_full_text", "")
        if not full:
            return
        inner = self.contentsRect().width()
        if inner <= 0:
            return
        metrics = self.fontMetrics()
        shown = (full if metrics.horizontalAdvance(full) <= inner
                 else metrics.elidedText(full, Qt.ElideRight, inner))
        # An elision that keeps no character of the label -- "" when the width
        # cannot fit even the ellipsis, "…" when it barely can -- paints a
        # blank toggle. The full text drawn slightly clipped is strictly
        # better: it still says which control this is. Zoom lands here because
        # the enlarged font gets measured against the width the layout granted
        # the smaller one, one relayout behind. The long labels this eliding
        # exists for keep plenty of characters at ELIDE_ABOVE_PX and are
        # untouched by this guard.
        if not shown.strip("…. \t"):
            shown = full
        if shown == QLabel.text(self):
            return
        # `setText` re-enters through the override above; the flag keeps it
        # from mistaking the elided text for a new logical text and
        # truncating the stored copy one character at a time.
        self._eliding = True
        try:
            QLabel.setText(self, shown)
        finally:
            self._eliding = False

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
        # Zoom reaches this through `font_px`, not through the application
        # sheet: a per-widget `setStyleSheet` outranks it, so the literal
        # `FONT_SIZE['body']` that used to be here pinned "Live" and "AI"
        # at 13 px whatever the preference said. Padding scales with it or
        # the hit target stops matching the glyphs.
        size = font_px("body")
        sheet = (
            f"QLabel#AiToggleLabel {{"
            f"  color: {color};"
            f"  font-size: {size}px;"
            f"  font-weight: 600;"
            f"  padding: {max(2, round(size * 4 / 13))}px"
            f" {max(4, round(size * 10 / 13))}px;"
            f"  background: transparent;"
            f"}}"
        )
        # `setStyleSheet` itself posts a StyleChange back to this widget, so
        # `changeEvent` would call straight back in. Both guards matter: the
        # flag stops the immediate recursion, the comparison stops a
        # StyleChange storm when nothing about the answer has changed.
        if self._restyling or sheet == self.styleSheet():
            return
        self._restyling = True
        try:
            self.setStyleSheet(sheet)
        finally:
            self._restyling = False
        # The new sheet moves both the font size and the padding, so the
        # `sizeHint` the layout is holding is stale. Without this the widget
        # keeps its old width and the elision below measures the bigger glyphs
        # against it, hiding the text that the zoom just enlarged.
        self.updateGeometry()
        self._apply_elision()
