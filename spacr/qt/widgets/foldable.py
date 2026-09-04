"""Shared click-to-fold behavior for titled Qt panels.

Folding hides the panel body so it no longer contributes to the layout size
hint; adjacent stretchable content can then occupy the released space. The
heading remains visible and provides the control for restoring the body.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QLabel, QWidget

from ..i18n import tr

LOG = logging.getLogger("spacr.qt.foldable")

#: What a folded heading shows, and what an open one does. The arrow is part
#: of the label text rather than a second widget: a heading whose affordance
#: is a sibling can be laid out apart from it, and then the arrow points at
#: nothing.
OPEN_MARK = "▾"
SHUT_MARK = "▸"


class _ClickToFold(QObject):
    """Convert a heading click into a fold-state change.

    The heading owns the event filter so Qt retains it for the lifetime of
    the interactive label.

    :param label: the heading to watch. ALSO THE QOBJECT PARENT, which is
        what the note above means by the heading owning the filter.
    :param toggle: called on a left-button release over the heading. Takes
        no arguments and returns nothing: this decides WHEN to fold, never
        what the folded state should be.
    """

    def __init__(self, label: QLabel, toggle: Callable[[], None]):
        super().__init__(label)
        self._toggle = toggle

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt name
        if event.type() == QEvent.MouseButtonRelease and \
                getattr(event, "button", lambda: None)() == Qt.LeftButton:
            try:
                self._toggle()
            except Exception:                                # noqa: BLE001
                LOG.debug("folding failed", exc_info=True)
            return True
        return False


class Folder:
    """The fold state of one panel, and the two widgets it moves.

    Not a QWidget. The panels this serves are already built and already in
    their layouts, and a wrapper would mean re-parenting them -- which
    changes what their stylesheets match and what their splitters remember.

    :param heading: the label that folds the panel when clicked. It is
        rewritten to carry the arrow, so it must be a label this owns.
    :param body: the widget shown and hidden.
    :param name: what the panel is called, for the tooltip and for anything
        remembering fold state. Defaults to the heading's own text.
    :param on_change: called with the new shut/open state after each fold.
    """

    def __init__(self, heading: QLabel, body: QWidget, name: str = "",
                 on_change: Optional[Callable[[bool], None]] = None):
        self.heading = heading
        self.body = body
        self.name = name or heading.text().strip()
        self._on_change = on_change
        self._shut = False
        self._alert = ""
        heading.setCursor(Qt.PointingHandCursor)
        self._refresh_tooltip()
        self._filter = _ClickToFold(heading, self.toggle)
        heading.installEventFilter(self._filter)
        # A HEADING IS COMPOSED, NOT WRITTEN. The line reads as an arrow, the
        # panel name and sometimes an alert, so asking the catalog for the
        # finished line asks for a key that cannot exist. Keep the generic
        # language pass off it and rebuild it from the translated parts.
        heading.setProperty("i18nSkipText", True)
        heading.retranslate_dynamic_content = self._retranslate
        self._repaint()

    # ------------------------------------------------------------- state

    @property
    def shut(self) -> bool:
        return self._shut

    def toggle(self) -> bool:
        """Fold if open, unfold if shut. Returns the new shut state."""
        return self.set_shut(not self._shut)

    def set_shut(self, shut: bool) -> bool:
        shut = bool(shut)
        if shut == self._shut:
            return shut
        self._shut = shut
        self.body.setVisible(not shut)
        if not shut:
            # ARRIVING IS SEEING. An alert that survived the unfold would
            # keep claiming there is something to look at after the user has
            # looked at it.
            self._alert = ""
        self._repaint()
        if self._on_change is not None:
            try:
                self._on_change(shut)
            except Exception:                                # noqa: BLE001
                LOG.debug("fold callback failed", exc_info=True)
        return shut

    def alert(self, note: str = "!") -> None:
        """Mark the folded strip as having something to say.

        A FOLDED CONSOLE THAT RECEIVES AN ERROR SAYS SO. Silence from a panel
        the user folded is indistinguishable from silence from a panel with
        nothing in it, and the first is the one that matters.
        """
        if not self._shut:
            return
        self._alert = str(note or "!")
        self._repaint()

    def _retranslate(self, language: Optional[str] = None) -> None:
        """Rebuild the heading and its tooltip in ``language``."""
        self._refresh_tooltip(language)
        self._repaint(language)

    def _refresh_tooltip(self, language: Optional[str] = None) -> None:
        # THE NAME LOOKS CLICKABLE BEFORE IT IS CLICKED. A gesture nobody
        # knows about is not a feature, and the pointer is the only hint a
        # heading can carry without a second widget beside it.
        self.heading.setToolTip(tr(
            "Click to fold {name} away, and click again to bring it back. "
            "The panel above takes the space.",
            language, name=tr(self.name, language)))

    def _repaint(self, language: Optional[str] = None) -> None:
        mark = SHUT_MARK if self._shut else OPEN_MARK
        text = f"{mark} {tr(self.name, language)}"
        if self._shut and self._alert:
            text = f"{text}  {self._alert}"
        self.heading.setText(text)


def make_foldable(heading: QLabel, body: QWidget, name: str = "",
                  on_change: Optional[Callable[[bool], None]] = None,
                  *, persist_key: str = "") -> Folder:
    """Make clicking ``heading`` fold ``body`` away. Returns the Folder.

    The Folder is returned so the caller can hold it: it owns the event
    filter, and a Folder nobody keeps stops working silently.

    :param heading: label that receives the click event filter and displays
        the open or shut marker.
    :param body: panel whose visibility the heading toggles.
    :param persist_key: ``"<module>/<panel>"``. Given, the fold survives a
        restart. Empty means it does not, which is what a bare panel in a
        test wants -- a test that wrote to the real preferences would fold a
        panel on the user's next launch.
    """
    key = str(persist_key or "").strip()
    shut_at_start = False
    if key:
        try:
            from ..preferences import get_folded_panels

            shut_at_start = bool(get_folded_panels().get(key))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the folded panels", exc_info=True)

    def remember(shut: bool) -> None:
        if key:
            try:
                from ..preferences import set_folded_panel

                set_folded_panel(key, shut)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not store the fold", exc_info=True)
        if on_change is not None:
            on_change(shut)

    folder = Folder(heading, body, name=name,
                    on_change=remember if key else on_change)
    if shut_at_start:
        folder.set_shut(True)
    return folder
