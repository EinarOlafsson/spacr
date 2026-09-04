"""Explain unavailable options and offer an applicable next step.

A control whose entry is unavailable has three jobs it
cannot do while it is honestly dead: say WHY, link the documentation, and --
where it is possible in this environment -- offer to make itself available.
This module is where all three live, once, because the regression backend
picker and the Image UMAP's GPU acceleration ask exactly the same question
and a second interactive tooltip is a second set of hover, focus and
dismissal bugs.

Why this is not a ``QToolTip``: PySide6 reports::

    QToolTip has a linkActivated signal:  False
    QToolTip is a QWidget:                False

``QToolTip`` is a STATIC PAINTER. It renders rich text and nothing inside it
can be clicked, hovered or focused, so an "install link in the tooltip" drawn
with it is a link that looks pressable and is not. What works is a frameless
tooltip-styled ``QLabel`` with ``Qt.TextBrowserInteraction``: its
``linkActivated`` carries the href, so ``api`` opens the documentation and
``install`` opens the offer.

THE ROW ITSELF STAYS DEAD. ``QStandardItem.setEnabled(False)`` leaves
``ItemIsSelectable`` SET -- Qt will not ACTIVATE a disabled row, but a
model-level selection can still land on it -- so a caller must clear that
flag too. :func:`disable_combo_row` does both and is what the callers use.

THE THREE THINGS AN INTERACTIVE TOOLTIP HAS TO GET RIGHT, and how each is
answered here:

* IT MUST NOT VANISH WHILE THE POINTER MOVES TOWARD IT. The pointer has to
  cross the gap between the row and the panel, and a plain leave-event
  dismissal closes it mid-journey, which makes the Install link unreachable.
  The hide is a timer, the timer is cancelled on entry, and while it is
  pending the cursor is checked against the CORRIDOR -- the union of the
  anchor's rectangle and the panel's -- so travelling through the gap
  re-arms the timer instead of firing it. See :meth:`AvailabilityPanel.
  _maybe_hide`.
* IT MUST BE DISMISSABLE. Escape, a click anywhere outside it, and moving
  well clear of the corridor all close it.
* IT MUST BE REACHABLE BY KEYBOARD, and that cannot be inherited from the row
  because the row is disabled. :meth:`AvailabilityPanel.open_for` is the
  explicit keyboard route: it shows the panel ACTIVATED with the first link
  focused, Up/Down move between the unavailable entries, Escape closes it and
  returns focus to wherever it came from. A keyboard-opened panel is pinned
  and never closes on a hover timer, because a reader who is not holding the
  mouse must not have the panel taken away from them.

WHAT PRESSING INSTALL DOES is :func:`run_install_offer`, and it has THREE
answers rather than two -- see :class:`spacr.updater.InstallOffer`. The
dry-run report is shown BEFORE anything is installed, and a plan that would
move numpy, torch, pandas or scikit-learn is refused by default and needs a
second confirmation naming what moves.
"""
from __future__ import annotations

import logging
import time
from html import escape
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QEvent, QPoint, QRect, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QMessageBox, QVBoxLayout, QWidget)

from ..theme import SPACING, active_palette, font_px

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AvailabilityPanel", "disable_combo_row", "run_install_offer",
    "install_word_for", "explain",
]


#: The word in the slot to the RIGHT of the API link. "Install" is only
#: written where pressing it can install something here; the other two say
#: what the press will actually do, because a button labelled Install that
#: cannot install is the inert control this whole design exists to avoid.
_INSTALL_WORDS = {
    "install": "Install",
    "elsewhere": "How to get it",
    "impossible": "What it needs",
    "ready": "",
}


def install_word_for(action) -> str:
    """The link word for an offer's ``action``; ``''`` when there is none."""
    return _INSTALL_WORDS.get(str(action), "What it needs")


def disable_combo_row(combo, index: int, *, tooltip: str = "") -> None:
    """Disable a combo row and remove it from keyboard selection.

    ``QStandardItem.setEnabled(False)`` leaves ``Qt.ItemIsSelectable`` set in
    the item's flags. Qt refuses to activate a disabled row from the popup,
    so the mouse route is closed, but a
    model-level selection (``setCurrentIndex``, a settings CSV round-trip, a
    view's own selection model) can still land on it. An unavailable entry
    must not become the selected value, so this function clears both flags.

    A disabled item retains its ``Qt.ToolTipRole``, so the explanatory tooltip
    remains available on hover.

    :param combo: ``QComboBox`` backed by a ``QStandardItemModel``.
    :param index: Row to disable.
    :param tooltip: Explanatory tooltip; an empty string leaves it unchanged.
    """
    model = combo.model()
    item = model.item(index) if hasattr(model, "item") else None
    if item is None:
        return
    item.setEnabled(False)
    item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
    if tooltip:
        item.setToolTip(str(tooltip))


class AvailabilityPanel(QFrame):
    """The hover panel a greyed-out option opens: reason, API, Install.

    Access through :meth:`instance` -- it is a process-wide singleton, so one
    panel serves every control that has an unavailable option and there is
    only ever one of them on screen.

    :signal api_requested: the API link was pressed; carries the URL.
    :signal install_requested: the Install link was pressed; carries the
        :class:`spacr.updater.InstallOffer` of the entry on screen.
    :signal dismissed: the panel closed by Escape, by a click away, or by the
        pointer leaving the corridor.
    """

    api_requested = Signal(str)
    install_requested = Signal(object)
    dismissed = Signal()

    _INSTANCE: Optional["AvailabilityPanel"] = None

    #: Width of the prose column. Wide enough for the cuML paragraph without
    #: turning into a document; the recipes go in the dialog, not here.
    TEXT_WIDTH = 420

    #: How long the panel waits before closing once the pointer leaves. The
    #: gap between a combo popup row and the panel is a few pixels of travel;
    #: 250 ms is the same delay `HoverTooltip` settled on.
    HIDE_DELAY_MS = 250

    #: How long the corridor keeps re-arming the timer. A pointer that is
    #: inside the union of anchor and panel is assumed to be travelling
    #: TOWARD the panel -- but only for this long, so a pointer that stops
    #: dead inside the corridor still lets the panel go.
    CORRIDOR_GRACE_MS = 2500

    def __init__(self) -> None:
        # Qt.Tool rather than Qt.ToolTip: a ToolTip window can never take
        # focus, and the keyboard route in `open_for` needs it to.
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint
                         | Qt.NoDropShadowWindowHint)
        self.setObjectName("AvailabilityPanel")
        self.setFocusPolicy(Qt.StrongFocus)

        self._title = QLabel(self)
        self._title.setObjectName("AvailabilityPanelTitle")
        self._title.setWordWrap(True)
        self._title.setMaximumWidth(self.TEXT_WIDTH)

        self._body = QLabel(self)
        self._body.setObjectName("AvailabilityPanelBody")
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.RichText)
        self._body.setMaximumWidth(self.TEXT_WIDTH)
        self._body.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # THE TWO LINK WORDS. Separate labels rather than one, so "INSTALL to
        # the right of the API link" is a fact about geometry a test can
        # measure rather than a claim about a string.
        self._links = QWidget(self)
        self._links.setObjectName("AvailabilityPanelLinks")
        row = QHBoxLayout(self._links)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["md"])
        self._api_link = self._make_link("AvailabilityPanelApiLink")
        self._install_link = self._make_link("AvailabilityPanelInstallLink")
        row.addWidget(self._api_link)
        row.addWidget(self._install_link)
        row.addStretch(1)

        column = QVBoxLayout(self)
        column.setContentsMargins(SPACING["sm"], SPACING["xs"],
                                  SPACING["sm"], SPACING["xs"])
        column.setSpacing(SPACING["xs"])
        column.addWidget(self._title)
        column.addWidget(self._body)
        column.addWidget(self._links)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._maybe_hide)
        self._hide_since = 0.0

        self._anchor: Optional[QWidget] = None
        self._anchor_rect: Optional[QRect] = None
        self._entries: List[Dict[str, Any]] = []
        self._index = 0
        self._pinned = False
        self._return_focus: Optional[QWidget] = None
        self._filtering = False
        self._install_handler: Optional[Callable[[Any], None]] = None
        self._apply_theme()

    def set_install_handler(self, slot) -> None:
        """Make ``slot`` the ONLY receiver of :attr:`install_requested`.

        The panel is a process-wide singleton with two callers. A plain
        ``connect`` on every show accumulates receivers, so one press would
        eventually run the regression picker's install AND the Image UMAP's;
        a plain ``disconnect()`` warns when nothing is connected yet. Holding
        the current slot and replacing it does neither.
        """
        previous = self._install_handler
        if previous is not None:
            try:
                self.install_requested.disconnect(previous)
            except (RuntimeError, TypeError):
                pass
        self._install_handler = slot
        if slot is not None:
            self.install_requested.connect(slot)

    def _make_link(self, name: str) -> QLabel:
        label = QLabel(self)
        label.setObjectName(name)
        label.setTextFormat(Qt.RichText)
        # The measured route: `linkActivated` carries the href, and
        # TextBrowserInteraction includes LinksAccessibleByKeyboard, which is
        # what puts the word in the tab order once the panel has focus.
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.linkActivated.connect(self._on_link)
        return label

    # -- singleton ----------------------------------------------------------

    @classmethod
    def instance(cls) -> "AvailabilityPanel":
        """The process-wide panel, created on first use."""
        if cls._INSTANCE is None:
            cls._INSTANCE = AvailabilityPanel()
        return cls._INSTANCE

    # -- what is on screen --------------------------------------------------

    def entries(self) -> List[Dict[str, Any]]:
        """The entries this panel is currently cycling through."""
        return list(self._entries)

    def current_entry(self) -> Optional[Dict[str, Any]]:
        """The entry on screen, or ``None`` when nothing is shown."""
        if not self._entries:
            return None
        return self._entries[self._index]

    def current_offer(self):
        """The :class:`spacr.updater.InstallOffer` on screen, or ``None``."""
        entry = self.current_entry()
        return None if entry is None else entry.get('offer')

    def api_link(self) -> QLabel:
        """The **API** word -- exposed so a test can measure where it sits."""
        return self._api_link

    def install_link(self) -> QLabel:
        """The **Install** word -- exposed for the same reason."""
        return self._install_link

    def body_label(self) -> QLabel:
        """The explanation label."""
        return self._body

    def is_pinned(self) -> bool:
        """Was this opened by keyboard? A pinned panel ignores hover timers."""
        return self._pinned

    # -- showing ------------------------------------------------------------

    def show_for(self, anchor: QWidget, entries, index: int = 0, *,
                 anchor_rect: Optional[QRect] = None,
                 pinned: bool = False) -> None:
        """Show the panel for ``entries[index]``, docked under ``anchor``.

        :param anchor: the widget the panel belongs to. Used for placement
            and as one half of the corridor the pointer may cross.
        :param entries: mappings as
            :func:`spacr.regression_backends.availability_entry` returns --
            ``{title, reason, url, offer}``. More than one makes the panel
            cyclable with Up/Down when it is pinned.
        :param index: which of them to show.
        :param anchor_rect: the anchor's rectangle in GLOBAL coordinates,
            when the thing being explained is smaller than the widget -- a
            single row of an open combo popup, for instance.
        :param pinned: opened by keyboard. See :meth:`open_for`.
        """
        items = [dict(entry) for entry in (entries or [])]
        if not items:
            return
        self._entries = items
        self._index = max(0, min(int(index), len(items) - 1))
        self._anchor = anchor
        self._anchor_rect = QRect(anchor_rect) if anchor_rect else None
        self._pinned = bool(pinned)
        self._hide_timer.stop()
        self._apply_theme()
        self._render()
        self.adjustSize()
        self._position()
        self.setAttribute(Qt.WA_ShowWithoutActivating, not self._pinned)
        self.show()
        self._install_filter()

    def open_for(self, anchor: QWidget, entries, index: int = 0, *,
                 anchor_rect: Optional[QRect] = None) -> None:
        """The KEYBOARD route: show the panel pinned, activated and focused.

        The row that would normally carry this help is disabled, so it cannot
        be tabbed to and nothing can be inherited from it. A caller wires this
        to a key on the control that IS focusable -- the combo itself -- and
        the panel takes it from there: Tab moves between **API** and
        **Install**, Enter presses one, Up/Down move to the next unavailable
        entry, Escape closes and hands focus back.
        """
        self._return_focus = QApplication.focusWidget()
        self.show_for(anchor, entries, index, anchor_rect=anchor_rect,
                      pinned=True)
        self.raise_()
        self.activateWindow()
        self._api_link.setFocus(Qt.TabFocusReason)

    def show_entry(self, index: int) -> None:
        """Move to another of the entries without moving the panel."""
        if not self._entries:
            return
        self._index = int(index) % len(self._entries)
        self._render()
        self.adjustSize()
        self._position()

    def _render(self) -> None:
        entry = self.current_entry() or {}
        offer = entry.get('offer')
        action = getattr(offer, 'action', 'impossible')
        title = str(entry.get('title') or "")
        reason = str(entry.get('reason') or "")
        message = str(getattr(offer, 'message', "") or "")
        self._title.setText(f"<b>{escape(title)}</b>")
        # The refusal first, then what would fix it. Two sentences that say
        # the same thing are collapsed to one -- `backend_status` and
        # `backend_install_offer` genuinely agree on some entries.
        parts = [reason]
        if message and message.strip() != reason.strip():
            parts.append(message)
        self._body.setText("<br><br>".join(
            escape(part).replace("\n", "<br>") for part in parts if part))
        url = str(entry.get('url') or "")
        self._api_link.setVisible(bool(url))
        self._api_link.setText('<a href="api">API</a>')
        word = install_word_for(action)
        self._install_link.setVisible(bool(word))
        if word:
            self._install_link.setText(f'<a href="install">{escape(word)}</a>')
        counter = ""
        if self._pinned and len(self._entries) > 1:
            counter = (f"  <span>{self._index + 1}/{len(self._entries)}"
                       f" &middot; Up/Down</span>")
        if counter:
            self._title.setText(f"<b>{escape(title)}</b>{counter}")

    def _position(self) -> None:
        """Dock under the anchor, clamped to the screen it is on."""
        rect = self._anchor_global_rect()
        if rect is None:
            return
        point = QPoint(rect.left(), rect.bottom() + 2)
        screen = QGuiApplication.screenAt(rect.center())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            size = self.sizeHint()
            x = min(max(point.x(), available.left()),
                    max(available.right() - size.width(), available.left()))
            y = point.y()
            if y + size.height() > available.bottom():
                y = max(available.top(), rect.top() - size.height() - 2)
            point = QPoint(x, y)
        self.move(point)

    def _anchor_global_rect(self) -> Optional[QRect]:
        if self._anchor_rect is not None:
            return QRect(self._anchor_rect)
        anchor = self._anchor
        if anchor is None:
            return None
        try:
            top_left = anchor.mapToGlobal(QPoint(0, 0))
        except RuntimeError:
            self._anchor = None
            return None
        return QRect(top_left, anchor.size())

    # -- staying alive across the gap --------------------------------------

    def start_hide(self, delay_ms: Optional[int] = None) -> None:
        """Schedule the hide the pointer is allowed to interrupt."""
        if self._pinned:
            return
        if not self._hide_timer.isActive():
            self._hide_since = time.monotonic()
        self._hide_timer.start(int(self.HIDE_DELAY_MS if delay_ms is None
                                   else delay_ms))

    def cancel_hide(self) -> None:
        """Stop a pending hide (the pointer came back)."""
        self._hide_timer.stop()

    def corridor(self) -> Optional[QRect]:
        """The rectangle the pointer is allowed to be inside while travelling.

        The union of the anchor's rectangle and the panel's, which is exactly
        the region a pointer moving from one to the other passes through. A
        naive implementation hides on the anchor's leave event and the pointer
        never arrives.
        """
        rect = self._anchor_global_rect()
        if rect is None:
            return self.geometry() if self.isVisible() else None
        return rect.united(self.geometry())

    def _cursor_pos(self) -> QPoint:
        """Where the pointer is, in global coordinates.

        A method rather than a bare :func:`QCursor.pos` call so the corridor
        can be driven in a test: the offscreen platform plugin has no real
        pointer to move, and the gap-crossing rule is precisely the one that
        has to be checked rather than assumed.
        """
        return QCursor.pos()

    def _maybe_hide(self) -> None:
        if self._pinned:
            return
        if self.underMouse():
            return
        anchor = self._anchor
        if anchor is not None:
            try:
                if anchor.underMouse():
                    return
            except RuntimeError:
                self._anchor = None
        corridor = self.corridor()
        elapsed_ms = (time.monotonic() - self._hide_since) * 1000.0
        if (corridor is not None and elapsed_ms < self.CORRIDOR_GRACE_MS
                and corridor.contains(self._cursor_pos())):
            # Still travelling. Re-arm rather than close, or the Install link
            # can never be reached.
            self._hide_timer.start(int(self.HIDE_DELAY_MS))
            return
        self.dismiss()

    def dismiss(self) -> None:
        """Close the panel and hand focus back where it came from."""
        self._hide_timer.stop()
        self._pinned = False
        was_visible = self.isVisible()
        self.hide()
        self._remove_filter()
        target, self._return_focus = self._return_focus, None
        if target is not None:
            try:
                target.setFocus(Qt.OtherFocusReason)
            except RuntimeError:
                pass
        if was_visible:
            self.dismissed.emit()

    # -- events -------------------------------------------------------------

    def enterEvent(self, event):
        """The pointer arrived: the panel stays."""
        self.cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """The pointer left: start the interruptible hide."""
        self.start_hide(100)
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        """Escape closes; Up/Down move between the entries."""
        key = event.key()
        if key == Qt.Key_Escape:
            self.dismiss()
            event.accept()
            return
        if key in (Qt.Key_Down, Qt.Key_Right) and len(self._entries) > 1:
            self.show_entry(self._index + 1)
            event.accept()
            return
        if key in (Qt.Key_Up, Qt.Key_Left) and len(self._entries) > 1:
            self.show_entry(self._index - 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _install_filter(self) -> None:
        app = QApplication.instance()
        if app is not None and not self._filtering:
            app.installEventFilter(self)
            self._filtering = True

    def _remove_filter(self) -> None:
        app = QApplication.instance()
        if app is not None and self._filtering:
            app.removeEventFilter(self)
        self._filtering = False

    def eventFilter(self, obj, event):
        """A press anywhere outside the panel dismisses it."""
        if event.type() == QEvent.MouseButtonPress and self.isVisible():
            try:
                inside = self.geometry().contains(event.globalPosition()
                                                  .toPoint())
            except AttributeError:
                inside = self.geometry().contains(self._cursor_pos())
            if not inside:
                self.dismiss()
        return False

    def hideEvent(self, event):
        """Drop the app filter whenever the panel leaves the screen."""
        self._remove_filter()
        super().hideEvent(event)

    # -- routing ------------------------------------------------------------

    def _on_link(self, href: str) -> None:
        """``api`` and ``install`` -- the two hrefs this panel understands."""
        entry = self.current_entry() or {}
        target = str(href or "").strip().lower()
        if target == "api":
            url = str(entry.get('url') or "")
            if url:
                self.api_requested.emit(url)
                QDesktopServices.openUrl(QUrl(url))
            return
        if target == "install":
            offer = entry.get('offer')
            self._pinned = True     # the dialog steals the pointer; stay put
            self.cancel_hide()
            self.install_requested.emit(offer)
            return
        LOGGER.debug("AvailabilityPanel ignored href %r", href)

    # -- looks --------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Tooltip styling, inline, because this is a separate top-level."""
        palette = active_palette()
        self.setStyleSheet(
            f"QFrame#AvailabilityPanel {{"
            f"  background-color: {palette['surface_alt']};"
            f"  border: 1px solid {palette['border']};"
            f"  border-radius: 6px;"
            f"}}"
            f"QWidget#AvailabilityPanelLinks {{ background: transparent; }}"
            f"QLabel {{"
            f"  color: {palette['fg']};"
            f"  font-size: {font_px('small')}px;"
            f"  background: transparent;"
            f"}}"
            f"QLabel#AvailabilityPanelApiLink,"
            f"QLabel#AvailabilityPanelInstallLink {{"
            f"  color: {palette['accent']};"
            f"  text-decoration: none;"
            f"}}"
        )


# ---------------------------------------------------------------------------
# What pressing the word actually does
# ---------------------------------------------------------------------------

def run_install_offer(parent, offer, *, confirm=None, inform=None,
                      dry_run=None, install=None) -> str:
    """Handle the three outcomes of an install offer in one place.

    1. INSTALLABLE HERE -- the dry-run report is shown FIRST, in full, and
       the install runs only on confirmation. A plan that would move numpy,
       torch, pandas or scikit-learn is refused by default and needs a
       second confirmation naming what moves.
    2. NOT HERE, BUT POSSIBLE ELSEWHERE -- the environment that would take it
       is described and NOTHING IS RUN.
    3. NOT POSSIBLE -- said plainly, with the recipe, and nothing is run.

    Every side effect is an injected callable so the whole flow can be driven
    without a screen: ``tests/qt/conftest.py`` makes a static modal raise on
    purpose, and a flow that could only be tested through one could not be
    tested at all.

    :param parent: the widget dialogs are parented to.
    :param offer: a :class:`spacr.updater.InstallOffer`.
    :param confirm: ``(title, text) -> bool``. Defaults to a
        ``QMessageBox.question`` whose default button is **No**.
    :param inform: ``(title, text) -> None``. Defaults to
        ``QMessageBox.information``.
    :param dry_run: ``(requirement) -> DryRun``. Defaults to
        :func:`spacr.updater.dry_run_install` behind a progress dialog --
        the resolver talks to the network and has been measured taking
        minutes, and a frozen window is indistinguishable from a crash.
    :param install: ``(command) -> (code, output)``. Defaults to
        :func:`spacr.updater.run_install_command`.
    :returns: one of ``'ready'``, ``'explained'``, ``'declined'``,
        ``'refused'``, ``'installed'``, ``'failed'``.
    """
    from ...updater import dry_run_install, install_decision, \
        run_install_command

    confirm = confirm or _default_confirm(parent)
    inform = inform or _default_inform(parent)
    dry_run = dry_run or _default_dry_run(parent)
    install = install or run_install_command

    action = str(getattr(offer, 'action', 'impossible'))
    title = str(getattr(offer, 'title', 'spaCR'))

    if action == "ready":
        inform(title, getattr(offer, 'message', "This is already available."))
        return "ready"

    if action in ("elsewhere", "impossible"):
        # RUNS NOTHING. This is the branch instruction 158 B exists for: a
        # prompt that runs pip here either fails, or succeeds at breaking the
        # install, and the second is worse.
        inform(title, offer.as_text())
        return "explained"

    requirement = str(getattr(offer, 'requirement', "") or "")
    if not requirement:
        inform(title, offer.as_text())
        return "explained"

    report = dry_run(requirement)
    decision = install_decision(report)
    if not decision['allowed']:
        inform(f"{title} -- not installed",
               decision['headline']
               + "\n\nNothing has been installed. Run the command yourself "
                 "to see the full output:\n    "
               + " ".join(str(part) for part in (offer.command or [])))
        return "refused"

    command = " ".join(str(part) for part in (offer.command or []))
    if not confirm(f"Install {requirement}?",
                   decision['report'] + "\n\n" + command):
        return "declined"

    if decision['needs_second_confirmation']:
        # THE SECOND CONFIRMATION NAMES WHAT MOVES. Not "are you sure" -- the
        # packages, with their versions, in the sentence the user has to
        # agree to.
        if not confirm("This moves packages spaCR depends on",
                       decision['headline'] + "\n\n" + command):
            return "refused"

    code, output = install(offer.command)
    if int(code or 0) != 0:
        inform(f"{title} -- install failed",
               (output or "").strip()[-4000:]
               or "The packaging tool failed and said nothing.")
        return "failed"
    inform(f"{title} -- restart spaCR",
           f"{requirement} is installed. RESTART spaCR before using it: pip "
           f"can upgrade packages underneath a process that has already "
           f"imported them, and this one has.")
    return "installed"


def _default_confirm(parent) -> Callable[[str, str], bool]:
    def _confirm(title: str, text: str) -> bool:
        """Ask a yes/no question in a modal box."""
        answer = QMessageBox.question(
            parent, str(title), str(text),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes
    return _confirm


def _default_inform(parent) -> Callable[[str, str], None]:
    def _inform(title: str, text: str) -> None:
        """Say something in a modal box."""
        QMessageBox.information(parent, str(title), str(text))
    return _inform


def _default_dry_run(parent) -> Callable[[str], Any]:
    """:func:`spacr.updater.dry_run_install`, off the GUI thread.

    The resolver downloads metadata for every candidate; against ``cuml-cu12``
    it took minutes on this machine. Run inline it freezes the window, and a
    frozen window during "I pressed Install" is indistinguishable from a
    crash -- so it runs on a worker while a cancellable progress dialog pumps
    the event loop. Cancelling returns a :class:`spacr.updater.DryRun` that
    did not answer, which :func:`spacr.updater.install_decision` refuses,
    because an install whose consequences are unknown is not offered.
    """
    import threading

    from PySide6.QtWidgets import QProgressDialog

    from ...updater import DryRun, dry_run_install

    def _run(requirement):
        """Dry-run an install and hand the outcome back."""
        outcome: Dict[str, Any] = {}

        def _work():
            """Do the dry run. Called off the GUI thread."""
            outcome['value'] = dry_run_install(requirement)

        worker = threading.Thread(target=_work, daemon=True)
        dialog = QProgressDialog(
            f"Working out what installing {requirement} would change...",
            "Cancel", 0, 0, parent)
        dialog.setWindowTitle("Checking")
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker.start()
        dialog.show()
        while worker.is_alive():
            QApplication.processEvents()
            worker.join(0.05)
            if dialog.wasCanceled():
                break
        dialog.close()
        if 'value' not in outcome:
            return DryRun(str(requirement), False,
                          error="Cancelled before the resolver answered. "
                                "Nothing has been installed.")
        return outcome['value']

    return _run


def explain(anchor, entries, index: int = 0, *, pinned: bool = False,
            parent=None, anchor_rect=None, on_installed=None):
    """Open the shared panel on ``entries[index]`` with Install already wired.

    The two-line form for a caller that has one control and one entry -- the
    Image UMAP's GPU acceleration is exactly that::

        from spacr.gpu_reduce import availability_entry
        explain(self._gpu_button, [availability_entry()], parent=self)

    The panel is a process-wide singleton, so the Install connection is
    replaced on every call rather than accumulated; two callers connected at
    once would both answer one press.

    :param anchor: the widget the panel docks under.
    :param entries: mappings from ``availability_entry`` -- see
        :meth:`AvailabilityPanel.show_for`.
    :param pinned: open it by the keyboard route (activated and focused).
    :param parent: what the install dialogs are parented to; defaults to
        ``anchor``.
    :param on_installed: called with the offer after a successful install,
        for a caller that has to re-probe.
    :returns: the panel, or ``None`` when ``entries`` is empty.
    """
    if not entries:
        return None
    panel = AvailabilityPanel.instance()

    def _install(offer):
        """Run one install offer and report what happened."""
        outcome = run_install_offer(parent if parent is not None else anchor,
                                    offer)
        if outcome == "installed" and on_installed is not None:
            on_installed(offer)

    panel.set_install_handler(_install)
    if pinned:
        panel.open_for(anchor, entries, index, anchor_rect=anchor_rect)
    else:
        panel.show_for(anchor, entries, index, anchor_rect=anchor_rect)
    return panel
