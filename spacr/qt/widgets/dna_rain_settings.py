"""The DNA button and the popover it opens.

The DNA rain's controls used to be a bar pinned to the bottom of the
sequencing screen: four sliders and a swatch, permanently on show,
under a settings form the user is actually there to fill in. A backdrop
is not worth a strip of chrome — so the controls moved behind a button.

The button is an :class:`~spacr.qt.widgets.ai_toggle_label.AiToggleLabel`
reading ``DNA``, sitting immediately left of the ``AI`` toggle in the
actions row and built from the same class, so it inks the same white
when off and the same accent blue when on, in every theme, without a
second stylesheet to keep in step.

Clicking it opens :class:`DnaRainSettingsPopover` — a frameless
``Qt.Popup`` holding the same
:class:`~spacr.qt.widgets.dna_rain.DnaRainSettingsBar`, laid out as a
grid instead of a row. ``Qt.Popup`` is what makes it behave like a
menu: it closes on a click anywhere else, on Escape, and when the
screen it belongs to goes away.

The one subtlety is the click that lands *on the button* while the
popover is open. Qt sends that press to the popup, closes the popup
because the press was outside it, and then replays the press at the
widget underneath — which is the button, which toggles straight back
on. One click, open-close-open, and the popover looks like it will not
close. :data:`REOPEN_GUARD_MS` is the window in which the replayed half
of that click is ignored.

The guard is armed by that press and by nothing else: not by Escape,
not by a click somewhere else on the screen, not by a programmatic
close. Anything broader would swallow a genuine click that happened to
follow a close too quickly.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QElapsedTimer, QPoint, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from ..theme import RADIUS, SPACING, active_palette
from .ai_toggle_label import AiToggleLabel
from .dna_rain import DnaRainSettingsBar

#: How long a re-open is ignored after a click on the button closed
#: the popover. Long enough to swallow the replayed half of that one
#: click, short enough that a user who genuinely clicks twice in a row
#: gets their popover back.
REOPEN_GUARD_MS = 250

#: Gap in pixels between the button and the popover.
ANCHOR_GAP = 6


class DnaRainSettingsPopover(QFrame):
    """Frameless popup holding one :class:`DnaRainSettingsBar`.

    :param bar: the settings bar to show. Owned by the popover once
        handed over — it is reparented into it.
    :ivar closed: emitted whenever the popover stops being visible,
        however that happened, so the button can un-toggle itself.
    """

    closed = Signal()

    def __init__(self, bar: DnaRainSettingsBar,
                 parent: Optional[QWidget] = None):
        # Parented, but still a window: a Qt.Popup with a parent is
        # destroyed with it and lands on the parent's screen, which is
        # what a per-screen popover wants. It is never laid out inside
        # the parent.
        super().__init__(
            parent,
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setObjectName("DnaRainPopover")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._bar = bar
        self._hidden_at = QElapsedTimer()
        self._replay_expected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING["xs"], SPACING["xs"],
                                  SPACING["xs"], SPACING["xs"])
        layout.setSpacing(0)
        # addWidget does the reparenting. An explicit setParent() first
        # would mark the bar hidden and it would stay blank inside a
        # shown popover.
        layout.addWidget(bar)
        self.apply_theme()

    @property
    def bar(self) -> DnaRainSettingsBar:
        """The settings bar inside this popover."""
        return self._bar

    def apply_theme(self) -> None:
        """Re-take the popup's colours, and the bar's, from the palette.

        A top-level window gets the application stylesheet but not the
        screen's ``clear_container_surfaces`` sweep, so it has to state
        its own surface — otherwise it is the blanket ``QWidget`` window
        fill, which is flat black under every dark theme.

        The bar has the same problem for the same reason, and neither is
        reached by the re-apply that re-styles the rest of a screen on a
        theme switch. Both are refreshed on every open, which is the
        only moment either is on screen.
        """
        palette = active_palette()
        self.setStyleSheet(
            "QFrame#DnaRainPopover {"
            f" background: {palette['surface_alt']};"
            f" border: 1px solid {palette['border']};"
            f" border-radius: {RADIUS['md']}px; }}"
        )
        self._bar.restyle_for_theme()

    # -- open / close ---------------------------------------------------
    def open_near(self, anchor: QWidget) -> None:
        """Show the popover just above (or below) ``anchor``."""
        self.apply_theme()
        self.adjustSize()
        self._position_near(anchor)
        self._replay_expected = False
        self.show()
        self.raise_()

    def just_closed(self) -> bool:
        """True if a click on the button closed it a moment ago.

        Which means the second half of that same click is on its way to
        the button and must not be allowed to re-open this.
        """
        return (self._hidden_at.isValid()
                and self._hidden_at.elapsed() < REOPEN_GUARD_MS)

    def _position_near(self, anchor: QWidget) -> None:
        """Above the anchor when it fits, below it when it does not.

        The actions row lives at the bottom of the screen, so above is
        almost always the right answer; the fallback is what keeps the
        popover on screen when the window is short or the row has been
        dragged up against the top of the display.
        """
        try:
            top_left = anchor.mapToGlobal(anchor.rect().topLeft())
            bottom_left = anchor.mapToGlobal(anchor.rect().bottomLeft())
            centre_x = top_left.x() + anchor.width() // 2
        except RuntimeError:      # anchor's C++ side is gone
            top_left = bottom_left = QPoint(0, 0)
            centre_x = 0

        x = centre_x - self.width() // 2
        y = top_left.y() - self.height() - ANCHOR_GAP
        screen = (QGuiApplication.screenAt(top_left)
                  or QGuiApplication.primaryScreen())
        if screen is not None:
            area = screen.availableGeometry()
            x = min(max(area.left(), x), area.right() - self.width() + 1)
            if y < area.top():
                y = bottom_left.y() + ANCHOR_GAP
            y = min(max(area.top(), y), area.bottom() - self.height() + 1)
        self.move(x, y)

    # -- Qt events ------------------------------------------------------
    def mousePressEvent(self, event):   # noqa: N802 (Qt override)
        """Notice the press that is about to close us *and* be replayed.

        A ``Qt.Popup`` receives every press while it is open, including
        the ones outside itself; Qt closes it after this handler returns
        and replays the press at the widget underneath. Only the presses
        that land on the button matter — that is the one that would
        toggle it straight back on.
        """
        anchor = self.parentWidget()
        local = event.position().toPoint()
        if anchor is not None and not self.rect().contains(local):
            over_button = anchor.rect().contains(
                anchor.mapFromGlobal(event.globalPosition().toPoint()))
            self._replay_expected = bool(over_button)
        super().mousePressEvent(event)

    def hideEvent(self, event):     # noqa: N802 (Qt override)
        """Arm the guard if a click on the button is closing us."""
        if self._replay_expected:
            self._hidden_at.restart()
            self._replay_expected = False
        super().hideEvent(event)
        self.closed.emit()

    def keyPressEvent(self, event):     # noqa: N802 (Qt override)
        """Escape closes, like every other popup in the app."""
        if event.key() == Qt.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)


class DnaSettingsButton(AiToggleLabel):
    """The ``DNA`` toggle that opens the rain's settings.

    Built from :class:`AiToggleLabel` rather than styled to look like
    one, so it is the AI toggle's twin by construction: same object
    name, same QSS, same off-white/on-accent behaviour under a live
    theme switch.

    :param bar: the bound settings bar to put in the popover.
    """

    def __init__(self, bar: DnaRainSettingsBar,
                 parent: Optional[QWidget] = None):
        super().__init__(
            parent,
            text="DNA",
            tooltip=("Click to show the DNA rain settings — colour "
                     "(including a random colour per falling string), "
                     "speed, visibility and font size. They apply live; "
                     "nothing restarts."),
        )
        self.setObjectName("AiToggleLabel")
        self._popover = DnaRainSettingsPopover(bar, self)
        self._popover.closed.connect(self._on_popover_closed)
        self.toggled.connect(self._on_toggled)

    @property
    def popover(self) -> DnaRainSettingsPopover:
        """The popover this button opens."""
        return self._popover

    @property
    def settings_bar(self) -> DnaRainSettingsBar:
        """The settings bar inside the popover."""
        return self._popover.bar

    def is_open(self) -> bool:
        """True while the settings are on screen."""
        return self._popover.isVisible()

    def _on_toggled(self, on: bool) -> None:
        if not on:
            self._popover.hide()
            return
        if self._popover.just_closed():
            # The click that closed the popover reached us as well.
            # Stay closed rather than flickering straight back open.
            self.setChecked(False)
            return
        self._popover.open_near(self)

    def _on_popover_closed(self) -> None:
        """Un-toggle when it closed on its own (Escape, a click away)."""
        self.setChecked(False)

    def hideEvent(self, event):     # noqa: N802 (Qt override)
        """Take the popover down with the screen it belongs to.

        Module screens are built once and kept in a stack, so switching
        tabs hides this button rather than destroying it. Without this
        the popover would be left floating over whatever screen the user
        moved to.
        """
        self._popover.hide()
        super().hideEvent(event)


__all__ = ["DnaRainSettingsPopover", "DnaSettingsButton", "REOPEN_GUARD_MS"]
