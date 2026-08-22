"""Give every popup the translucent card and the travelling rim.

    "i want every settings pop up throughout the entire spacr program
     (preferences, hyperparamiters, settings, live settings, AI settings,
     figure settings, etc.) to have the same transparent background with the
     new rim"

ONE INSTALL POINT, NOT THIRTY-NINE EDITS. There are thirty-nine ``QDialog``
subclasses in this package and there will be more next week; a look applied
by hand in each of them is a look that is missing from the fortieth. This
installs an application event filter instead, so a dialog gets the treatment
the first time it is shown, whoever wrote it and whenever it was added.

WHAT THE TREATMENT IS, and each part is here for a reason the setup screen
found the hard way:

* a :class:`~spacr.qt.widgets.setup_card.SetupCard` is put BEHIND the
  dialog's own contents and kept at its size. It paints the translucent body
  and runs the rim; it holds no layout, so it cannot disturb one;
* the dialog and its layout CONTAINERS are made transparent. This palette's
  ``bg`` is literally ``#000000``, so any untagged container between the card
  and the eye paints a black rectangle over it -- which is what "black boxes"
  meant on the setup screen, and is invisible to a code reading;
* the controls are left alone. A control you can see through is a control you
  cannot read, so combos, edits, buttons and tables keep their own surface.

A DIALOG CAN SAY NO by carrying the ``spacrNoGlass`` property. Nothing in
spaCR sets it today; it is there because the next thing somebody embeds may
be a native colour picker or a video surface that must own its own painting.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QComboBox,
                               QDialog, QLineEdit, QPushButton, QTextEdit,
                               QWidget)

LOG = logging.getLogger("spacr.qt.glass")

#: Set on a dialog that has already been treated, so a second show is cheap
#: and a re-show never stacks two cards.
GLASSED = "spacrGlassed"

#: Set on a dialog that must keep its own painting.
NO_GLASS = "spacrNoGlass"

#: Pixels between the dialog's edge and the card's.
#:
#: SMALL, because a dialog is sized to its contents and every pixel here is
#: taken off what the contents were given. The setup screen can afford 44
#: because it chose its own size; a settings dialog cannot.
INSET = 8

#: Extra margin given to the dialog's own layout, so the rim has room.
#:
#: WITHOUT IT THE RIM IS DRAWN AND NEVER SEEN. A dialog's contents run to
#: its edges, so the card's border sits underneath a tab bar or a button
#: and the light travels behind them. This pushes the contents in far
#: enough to leave the band the rim runs along clear.
RIM_ROOM = 10

#: Widget types that keep their own background.
#:
#: THE CONTROLS, not the containers. A combo you can see through is a combo
#: whose current value is competing with a moving backdrop, and the value is
#: the thing the user came to read.
OPAQUE = (QComboBox, QLineEdit, QTextEdit, QAbstractSpinBox, QPushButton,
          QAbstractItemView)


def wants_glass(widget: QWidget) -> bool:
    """Whether ``widget`` should be given the card and the rim."""
    if not isinstance(widget, QDialog):
        return False
    if widget.property(NO_GLASS):
        return False
    return not widget.property(GLASSED)


def clear_the_containers(dialog: QWidget) -> int:
    """Stop the layout containers painting over the card. Returns how many.

    Walks the whole tree, so a dialog whose settings live on the pages of a
    tab widget is covered as well: "every tab of every popup panel" is a
    page that is itself a plain QWidget, and one of those is enough to bury
    the card under a black rectangle.
    """
    try:
        from ..theme import make_transparent
    except Exception:                                        # noqa: BLE001
        LOG.debug("no theme helper for transparency", exc_info=True)
        return 0
    holders = []
    for child in dialog.findChildren(QWidget):
        if isinstance(child, OPAQUE):
            continue
        # A control's internals are not containers either: a combo's popup
        # view and a spin box's line edit are children of an opaque thing.
        if any(isinstance(parent, OPAQUE)
               for parent in _ancestors(child, dialog)):
            continue
        holders.append(child)
    if not holders:
        return 0
    try:
        make_transparent(*holders)
    except Exception:                                        # noqa: BLE001
        LOG.debug("a container would not go transparent", exc_info=True)
        return 0
    return len(holders)


def _ancestors(widget: QWidget, stop: QWidget):
    """Every parent of ``widget`` up to but not including ``stop``."""
    parent = widget.parentWidget()
    while parent is not None and parent is not stop:
        yield parent
        parent = parent.parentWidget()


class _Backdrop(QObject):
    """Keeps one card at the size of the dialog it sits behind.

    An event filter rather than a resizeEvent override, because the dialog
    is somebody else's class and this must not require subclassing it.
    """

    def __init__(self, dialog: QDialog, card: QWidget):
        super().__init__(dialog)
        self._dialog = dialog
        self._card = card
        dialog.installEventFilter(self)
        self._fit()

    def _fit(self) -> None:
        try:
            self._card.setGeometry(self._dialog.rect().adjusted(
                INSET, INSET, -INSET, -INSET))
            self._card.lower()
        except Exception:                                    # noqa: BLE001
            LOG.debug("the backdrop would not fit", exc_info=True)

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        if watched is self._dialog and event.type() in (
                QEvent.Type.Resize, QEvent.Type.Show):
            self._fit()
        return False


def glass(dialog: QDialog) -> bool:
    """Give one dialog the card and the rim. True if it was applied.

    Idempotent: a dialog that already carries :data:`GLASSED` is left alone,
    so a dialog shown, closed and shown again does not accumulate cards.
    """
    if not wants_glass(dialog):
        return False
    try:
        from .setup_card import SetupCard
    except Exception:                                        # noqa: BLE001
        LOG.debug("no card to put behind this dialog", exc_info=True)
        return False
    try:
        # THE DRIFTING BACKDROP FIRST, so the card has something to be
        # translucent OVER. A translucent panel on an opaque dialog is just
        # a slightly different opaque panel -- what makes the setup screen
        # read as glass is the moving strata showing through it, and that
        # is what "the same transparent background" means here.
        backdrop = _install_the_backdrop(dialog)

        card = SetupCard(dialog)
        # BEHIND THE CONTENTS, IN FRONT OF THE BACKDROP, and never in the
        # layout: it is not added to one at all, because a backdrop that
        # took part in a layout would push the dialog's own contents
        # around, and the contents are the point.
        card.lower()
        card.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # SHOWN EXPLICITLY. This runs on the dialog's Show event, so the
        # parent is already visible and a child made now stays hidden until
        # it is told otherwise -- which is a card that is there, sized, and
        # painting nothing.
        card.show()
        # AND THE BACKDROP GOES UNDER THE CARD. `install_ambient` lowers
        # itself to the bottom of the sibling order and so does the card,
        # so whichever is lowered LAST wins the bottom -- and a card under
        # the strata is a card nobody sees, rim and all. One more `lower`
        # on the backdrop puts them in the order the look needs: strata,
        # then the translucent body, then the dialog's own contents.
        if backdrop is not None:
            backdrop.lower()
        _make_room_for_the_rim(dialog)
        _Backdrop(dialog, card)
        clear_the_containers(dialog)
        dialog.setProperty(GLASSED, True)
        return True
    except Exception:                                        # noqa: BLE001
        # DECORATION IS NEVER LOAD-BEARING. A dialog that cannot be glassed
        # is a dialog that opens looking as it always did.
        LOG.debug("could not glass a dialog", exc_info=True)
        return False


def _make_room_for_the_rim(dialog: QDialog) -> bool:
    """Widen the dialog's own margins so the rim is not painted over.

    ONCE, and additively: the dialog keeps whatever margins it chose and
    gains the band. Re-running would push the contents in again, which is
    why `glass` is idempotent and this is only called from it.
    """
    layout = dialog.layout()
    if layout is None:
        return False
    try:
        left, top, right, bottom = layout.getContentsMargins()
        layout.setContentsMargins(left + RIM_ROOM, top + RIM_ROOM,
                                  right + RIM_ROOM, bottom + RIM_ROOM)
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not make room for the rim", exc_info=True)
        return False


def _install_the_backdrop(dialog: QDialog) -> Optional[QWidget]:
    """Put the drifting strata behind ``dialog``, or None if unavailable.

    The same engine and theme the setup screen uses, so a popup and the
    first-run screen are recognisably the same surface rather than two
    takes on one idea.
    """
    try:
        from ..preferences import get_ambient_enabled

        # THE USER'S OWN ANSWER ABOUT ANIMATED BACKGROUNDS. Somebody who
        # turned the backdrop off on the module screens has not asked for
        # it back in every popup; the card and the rim still apply, so the
        # look is the same one, just still.
        if not get_ambient_enabled():
            return None
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the ambient preference", exc_info=True)
    try:
        from .ambient import install_ambient
        from .setup_slides import BACKDROP_SPEED, BACKDROP_THEME

        return install_ambient(dialog, theme=BACKDROP_THEME,
                               speed=BACKDROP_SPEED)
    except Exception:                                        # noqa: BLE001
        # INVARIANTS 10: with no ambient engine the card is still a card,
        # and the dialog still works.
        LOG.debug("no ambient backdrop for this dialog", exc_info=True)
        return None


class _GlassInstaller(QObject):
    """Applies :func:`glass` to every dialog the first time it is shown."""

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        try:
            if event.type() == QEvent.Type.Show and wants_glass(watched):
                glass(watched)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the glass filter tripped", exc_info=True)
        return False


#: The one installed filter. Held so it is not collected, and so a second
#: call is a no-op rather than a second filter on every event in the app.
_INSTALLED: Optional[_GlassInstaller] = None


def install_glass_everywhere(application=None) -> bool:
    """Install the filter. True when it was installed by this call.

    Called once at startup. Every dialog opened afterwards -- Preferences,
    the hyperparameter search, live settings, the AI providers, the figure
    settings, and the thirty-odd others -- is treated on its first show
    without knowing anything about this module.
    """
    global _INSTALLED

    if _INSTALLED is not None:
        return False
    try:
        from PySide6.QtWidgets import QApplication

        application = application or QApplication.instance()
        if application is None:
            return False
        _INSTALLED = _GlassInstaller(application)
        application.installEventFilter(_INSTALLED)
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("the glass filter would not install", exc_info=True)
        _INSTALLED = None
        return False


def uninstall_glass_everywhere(application=None) -> bool:
    """Take the filter back off. True when there was one to remove.

    THE APPLICATION OUTLIVES ONE SCREEN'S WORTH OF INTENT. A filter that
    can only be installed changes every dialog for the rest of the
    process, which is right for a running spaCR and wrong for anything
    that wants to look at a dialog as its author wrote it -- a test being
    the obvious case, since one file installing this would otherwise
    decide the look of every dialog examined after it.

    Dialogs already treated keep their card: this removes the filter, not
    the effect.
    """
    global _INSTALLED

    if _INSTALLED is None:
        return False
    try:
        from PySide6.QtWidgets import QApplication

        application = application or QApplication.instance()
        if application is not None:
            application.removeEventFilter(_INSTALLED)
    except Exception:                                        # noqa: BLE001
        LOG.debug("the glass filter would not come off", exc_info=True)
    finally:
        _INSTALLED = None
    return True
