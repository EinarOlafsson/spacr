"""Fold a panel away by clicking its name (instruction 228).

THE SPACE IS THE POINT, NOT THE FOLDING. A panel that collapses and leaves a
gap has cost the user a click and given them nothing: whatever is above has
to actually grow into the room it released. That is why the body is HIDDEN
rather than resized -- a hidden widget contributes nothing to its layout's
size hint, so the stretch above takes the space without anybody computing
it.

ONE FUNCTION, WHEREVER THE THREE ARE ASSEMBLED. Two panels that fold and one
that does not is worse than none folding, because the user learns the
gesture and then meets the exception. Applying this by hand per screen is
exactly how the plate-map button, the tooltip rule and the movable-dialog
rule each came to be missing from one module and had to be fixed centrally
afterwards.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QLabel, QWidget

LOG = logging.getLogger("spacr.qt.foldable")

#: What a folded heading shows, and what an open one does. The arrow is part
#: of the label text rather than a second widget: a heading whose affordance
#: is a sibling can be laid out apart from it, and then the arrow points at
#: nothing.
OPEN_MARK = "▾"
SHUT_MARK = "▸"


class _ClickToFold(QObject):
    """Turns a press on the heading into a fold. Held by the heading.

    HELD, because an event filter is owned by nobody: Qt keeps a bare
    pointer, and a filter that only the constructor referenced is collected
    at the end of the call that installed it -- after which the heading
    stops responding and nothing says why.
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
        # THE NAME LOOKS CLICKABLE BEFORE IT IS CLICKED. A gesture nobody
        # knows about is not a feature, and the pointer is the only hint a
        # heading can carry without a second widget beside it.
        heading.setToolTip(f"Click to fold {self.name} away, and click again "
                           f"to bring it back. The panel above takes the "
                           f"space.")
        self._filter = _ClickToFold(heading, self.toggle)
        heading.installEventFilter(self._filter)
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

    def _repaint(self) -> None:
        mark = SHUT_MARK if self._shut else OPEN_MARK
        text = f"{mark} {self.name}"
        if self._shut and self._alert:
            text = f"{text}  {self._alert}"
        self.heading.setText(text)


def make_foldable(heading: QLabel, body: QWidget, name: str = "",
                  on_change: Optional[Callable[[bool], None]] = None
                  ) -> Folder:
    """Make clicking ``heading`` fold ``body`` away. Returns the Folder.

    The Folder is returned so the caller can hold it: it owns the event
    filter, and a Folder nobody keeps stops working silently.
    """
    return Folder(heading, body, name=name, on_change=on_change)
