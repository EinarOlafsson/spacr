"""Give every popup the translucent card and the travelling rim.

Every settings popup in the program -- preferences, the hyperparameter
search, live settings, AI settings, figure settings and the rest -- gets
the same translucent background and the same rim.

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
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView,
                               QAbstractSpinBox, QComboBox, QDialog,
                               QDialogButtonBox, QLineEdit, QPushButton,
                               QTextEdit, QWidget)

LOG = logging.getLogger("spacr.qt.glass")

#: Set on a dialog that has already been treated, so a second show is cheap
#: and a re-show never stacks two cards.
#: Set on a button already wired to send the rim round, so a second pass
#: over the same dialog does not connect it twice.
SPINS = "spacrSpinsTheRim"
GLASSED = "spacrGlassed"

#: Set on a dialog that must keep its own painting.
NO_GLASS = "spacrNoGlass"

#: Set on a dialog whose window flags this module has already rewritten.
#:
#: `spacr.qt.dialogs._DetachEveryDialog` reads it and leaves such a dialog
#: alone: a second `setWindowFlags` would recreate the native window a
#: second time, and the surface that comes back is not translucent on
#: every window manager.
DETACHED = "spacrDetached"

#: Pixels between the dialog's edge and the card's.
#:
#: ZERO. It was 8, and those eight pixels read as a box with square edges
#: behind the box with rounded ones: a band of the dialog's own background
#: running all the way round the rounded card, in the one place a square
#: corner is most visible. The card IS the window
#: now, so there is no band to see -- and nothing depends on the
#: compositor except the four corner arcs themselves, rather than a full
#: frame.
#:
#: The rim still has room: `_make_room_for_the_rim` widens the DIALOG's
#: layout margins, which is a different thing from insetting the card.
INSET = 0

#: Extra margin given to the dialog's own layout, so the rim has room.
#:
#: WITHOUT IT THE RIM IS DRAWN AND NEVER SEEN. A dialog's contents run to
#: its edges, so the card's border sits underneath a tab bar or a button
#: and the light travels behind them. This pushes the contents in far
#: enough to leave the band the rim runs along clear.
RIM_ROOM = 10

#: Corner radius of a glassed dialog, shared by the card and the backdrop.
#:
#: ONE NUMBER FOR TWO SURFACES. They are the same rectangle, so they must
#: round by the same amount or the backdrop's corners show past the card's.
#: This is `SetupCard`'s own default, named here so the backdrop can be told
#: the same thing.
CARD_RADIUS = 18

#: Marks a dialog that has already been told how to close itself.
CLOSE_HINT = "spacrSaysPressEscape"

#: Widget types that keep their own background.
#:
#: THE CONTROLS, not the containers. A combo you can see through is a combo
#: whose current value is competing with a moving backdrop, and the value is
#: the thing the user came to read.
OPAQUE = (QComboBox, QLineEdit, QTextEdit, QAbstractSpinBox, QPushButton,
          QAbstractItemView)


#: Button texts that mean "go on" and "go back", for a dialog whose
#: buttons carry no QDialogButtonBox role.
#:
#: LOWER CASE AND SUBSTRING-MATCHED, because a button may read "Next ›",
#: "&Save" or "Start spaCR". Anything unmatched spins nothing rather than
#: guessing -- a wrong direction is worse than none, since the direction is
#: the whole message.
FORWARD_WORDS = ("next", "ok", "save", "apply", "yes", "start", "continue",
                 "run", "accept", "done", "finish", "install")
BACKWARD_WORDS = ("cancel", "close", "back", "previous", "no", "discard",
                  "reset", "abort", "quit")


def button_direction(button: QAbstractButton) -> Optional[bool]:
    """True for a forward button, False for a back one, None if unclear.

    THE ROLE FIRST. A `QDialogButtonBox` already knows which of its buttons
    accepts and which rejects, and that answer is better than any reading
    of the label -- it survives translation, which the words below do not.
    """
    box = button.parentWidget()
    while box is not None and not isinstance(box, QDialogButtonBox):
        box = box.parentWidget()
    if isinstance(box, QDialogButtonBox):
        role = box.buttonRole(button)
        if role in (QDialogButtonBox.AcceptRole, QDialogButtonBox.ApplyRole,
                    QDialogButtonBox.YesRole):
            return True
        if role in (QDialogButtonBox.RejectRole, QDialogButtonBox.NoRole,
                    QDialogButtonBox.DestructiveRole):
            return False
    said = button.text().replace("&", "").strip().lower()
    if any(word in said for word in FORWARD_WORDS):
        return True
    if any(word in said for word in BACKWARD_WORDS):
        return False
    return None


def spin_on_every_button(dialog: QDialog, card) -> int:
    """Send the rim round on each button. Returns how many were wired.

    A POSITIVE CLICK GOES CLOCKWISE AND A NEGATIVE ONE BACK. The direction
    is the message -- it says which way through the dialog the click just
    took you -- which is why a button nobody can classify spins nothing at
    all rather than being guessed at.

    ONCE PER BUTTON. A dialog reaches the installer on Polish and again on
    Show, and a second connection would send the light round twice on one
    click -- which, since the two laps run down together, reads as a rim
    moving at double speed rather than as a bug.
    """
    wired = 0
    for button in dialog.findChildren(QAbstractButton):
        if button.property(SPINS):
            continue
        forward = button_direction(button)
        if forward is None:
            continue
        button.clicked.connect(
            lambda _checked=False, c=card, f=forward: c.circuit(clockwise=f))
        button.setProperty(SPINS, True)
        wired += 1
    return wired


def wants_glass(widget: QWidget) -> bool:
    """Whether ``widget`` should be given the card and the rim.

    A DIALOG THAT BROUGHT ITS OWN CARD IS LEFT ALONE. The setup screen
    builds one and lays its slides out inside it; glassing it added a
    SECOND card, and the second one covered the first one's contents --
    `childAt` over the GitHub button returned the card, so the click never
    reached it.

    Checked by looking rather than by asking, so anything else that builds
    its own card is covered without having to remember to say so.
    """
    if not isinstance(widget, QDialog):
        return False
    if widget.property(NO_GLASS):
        return False
    if widget.property(GLASSED):
        return False
    try:
        from .setup_card import SetupCard

        return not widget.findChildren(SetupCard)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not look for a card", exc_info=True)
        return True


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


class _DragByBackground(QObject):
    """Move a frameless dialog by dragging its empty background.

    THE TITLE BAR WAS WHERE A WINDOW WAS DRAGGED FROM, so taking it away
    takes that with it, and every popup has to stay a window the user can
    move. A press that lands on a control is left entirely alone; only one
    on the dialog itself starts a drag.
    """

    def __init__(self, dialog: QDialog):
        super().__init__(dialog)
        self._dialog = dialog
        self._grab = None
        dialog.installEventFilter(self)

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        # `getattr`, NOT `self._dialog`. The C++ QObject outlives this
        # Python object's attributes: during teardown Qt goes on delivering
        # events to a filter whose `__dict__` has already been cleared, and
        # the AttributeError is printed by Qt on every one of them --
        # "Error calling Python override of QObject::eventFilter" -- which
        # is noise nobody can act on in a test log.
        dialog = getattr(self, "_dialog", None)
        if dialog is None or watched is not dialog:
            return False
        try:
            kind = event.type()
            if kind == QEvent.Type.MouseButtonPress and \
                    event.button() == Qt.LeftButton:
                # ONLY ON THE BACKGROUND. `childAt` answers None when the
                # press is on the dialog itself rather than on something
                # in it, which is exactly the empty space a title bar used
                # to be.
                where = event.position().toPoint()
                if dialog.childAt(where) is None:
                    self._grab = (event.globalPosition().toPoint()
                                  - dialog.frameGeometry().topLeft())
            elif kind == QEvent.Type.MouseMove and self._grab is not None:
                dialog.move(event.globalPosition().toPoint() - self._grab)
            elif kind == QEvent.Type.MouseButtonRelease:
                self._grab = None
        except Exception:                                    # noqa: BLE001
            LOG.debug("a drag went wrong", exc_info=True)
            self._grab = None
        return False


#: What the dialog's own body paints once the card is behind it: nothing.
#: `background: transparent` rather than a colour, because any colour is a
#: square of it in the eight-pixel band around the rounded card.
NO_BACKGROUND = "QDialog { background: transparent; border: none; }"


def _paint_nothing_behind_the_card(dialog: QDialog) -> bool:
    """Stop the dialog painting its own square background. True if applied.

    ADDITIVE, because a dialog may carry a stylesheet of its own and this
    must not replace it. Appended, so it wins over an earlier `QDialog`
    rule in the same sheet, and the application-wide sheet loses to a
    widget sheet by Qt's own precedence.
    """
    try:
        existing = dialog.styleSheet() or ""
        if NO_BACKGROUND in existing:
            return False
        dialog.setStyleSheet(f"{existing}\n{NO_BACKGROUND}".strip())
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("a dialog would not drop its background", exc_info=True)
        return False


def make_frameless(dialog: QDialog) -> bool:
    """Drop the title bar and let the card's rounded corners show.

    "they also dont need the x and minus at the top make the edges rounded
    on all" -- a settings window is dismissed by its own Cancel or by
    Escape, so the close and minimise buttons were chrome around chrome.

    TRANSLUCENT, or the corners are not round: the card paints a rounded
    body, and without this the square window behind it fills the four
    corners with the theme's background and the shape is lost.

    AND STILL MOVABLE. See `_DragByBackground` -- the title bar was where a
    window was dragged from.

    IT PUTS BACK A DIALOG IT HAD TO HIDE. `setWindowFlags` on a VISIBLE
    widget hides it, and Qt requires `show()` to bring it back. This runs
    from the filter below, which fires while a dialog is being shown -- so
    without the restore, opening Preferences hid Preferences, and an
    `exec()` sat on an invisible modal window with no way to dismiss it.
    """
    try:
        # `isVisible()` is not the test: a child of a parent that has never
        # been shown reports False while still being marked visible itself,
        # and hiding one of those would leave it hidden when its parent
        # finally opened.
        was_showing = not dialog.isHidden()
        # THE ATTRIBUTE BEFORE THE FLAGS. `setWindowFlags` RECREATES the
        # native window, and a translucency asked for afterwards applies to
        # a window that already exists -- which on X11 means it does not
        # apply at all. Reported 2026-08-22: "there is a box with square
        # edges behind the box with rounded edges."
        dialog.setAttribute(Qt.WA_TranslucentBackground, True)
        # ONE FLAGS CHANGE, NOT TWO. `spacr.qt.dialogs._DetachEveryDialog`
        # also rewrites the flags on Polish, to turn Qt.Dialog into
        # Qt.Window so a window manager cannot glue the popup to its
        # parent. Each `setWindowFlags` RECREATES the native window, and a
        # window recreated after the translucent one was made is where the
        # square corners came back -- reported as "the rectangular
        # non-rounded black corners are still visible around the
        # preferences and settings windows". So the detach happens here,
        # in the same call, and the marker below tells that filter this
        # dialog is already done.
        dialog.setWindowFlags((dialog.windowFlags()
                               & ~Qt.WindowType.Dialog)
                              | Qt.WindowType.Window
                              | Qt.FramelessWindowHint)
        dialog.setProperty(DETACHED, True)
        # AND THE STYLESHEET HAS TO AGREE. WA_TranslucentBackground stops Qt
        # filling the window with the palette's base; it does NOT stop the
        # application stylesheet's `QDialog { background: ... }` rule, which
        # paints the square box this is about. Scoped to the dialog itself
        # with `#objectName`-free `QDialog` -- a bare `QDialog` selector in
        # a widget stylesheet applies to that widget and inherits to its
        # QDialog children, of which a settings popup has none.
        _paint_nothing_behind_the_card(dialog)
        _DragByBackground(dialog)
        if was_showing and dialog.isHidden():
            dialog.show()
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not make a dialog frameless", exc_info=True)
        return False


def round_the_corners(dialog: QWidget, radius: int = CARD_RADIUS) -> bool:
    """Cut the window itself to the card's rounded shape. True if applied.

    TRANSLUCENCY IS NOT ENOUGH, AND THAT IS THE WHOLE POINT OF THIS.
    `WA_TranslucentBackground` asks the window manager to composite the
    corner pixels away; a mask REMOVES them from the window's shape, so
    the corners are gone whether or not anything is compositing, and
    whether or not the surface came back with an alpha channel after its
    flags were rewritten. It is the one way to be sure no square is left
    round a rounded card, which is what kept coming back.

    The mask is rebuilt on every resize -- see :class:`_Backdrop` -- and
    it follows the same radius the card paints, so the cut edge sits
    under the rim rather than beside it.
    """
    try:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainterPath, QRegion

        from PySide6.QtGui import QTransform

        rect = dialog.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        # BUILT AT FOUR TIMES THE SIZE AND SCALED BACK. `toFillPolygon`
        # flattens the arcs at a fixed tolerance, and at real size that
        # polygon keeps pixels just outside the curve the card paints --
        # so a sliver of whatever drifts behind the card showed along
        # each rounded corner. Flattening a four-times path puts the
        # polygon's error below one pixel once it is scaled down.
        # NOT ERODED. A mask a pixel inside the edge cuts the outermost
        # row all the way round, which takes the rim with it -- the card
        # paints the full rect, so the mask covers the full rect too.
        step = 4.0
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(0.0, 0.0, rect.width() * step, rect.height() * step),
            float(radius) * step, float(radius) * step)
        polygon = QTransform().scale(1.0 / step, 1.0 / step).map(
            path.toFillPolygon())
        dialog.setMask(QRegion(polygon.toPolygon()))
        return True
    except Exception:                                        # noqa: BLE001
        # A window that cannot be masked is a window with square corners,
        # which is worse-looking and still perfectly usable.
        LOG.debug("could not round the window corners", exc_info=True)
        return False


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
            dialog = getattr(self, "_dialog", None)
            card = getattr(self, "_card", None)
            if dialog is None or card is None:
                return
            card.setGeometry(dialog.rect().adjusted(
                INSET, INSET, -INSET, -INSET))
            card.lower()
            round_the_corners(dialog)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the backdrop would not fit", exc_info=True)

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        # `getattr`, for the reason spelled out on `_DragByBackground`: the
        # C++ QObject outlives this Python object's `__dict__`, so a filter
        # still installed during teardown is asked about events after its
        # attributes are gone. Reading `self._dialog` directly raised
        # AttributeError on every one of them, and Qt printed the whole
        # traceback -- "Error calling Python override of
        # QObject::eventFilter" -- at spaCR startup.
        dialog = getattr(self, "_dialog", None)
        if dialog is None or watched is not dialog:
            return False
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
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

        # THE DIALOG PAINTS NOTHING OF ITS OWN. Without this its square
        # background shows around the card wherever the card does not
        # reach -- the black box behind the periphery.
        _paint_nothing_behind_the_card(dialog)

        card = SetupCard(dialog, radius=CARD_RADIUS)
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
        make_frameless(dialog)
        round_the_corners(dialog)
        spin_on_every_button(dialog, card)
        # AND THE DIALOG'S OWN VERDICT, for the paths that never touch a
        # button -- Escape rejects, and code can accept directly.
        try:
            dialog.accepted.connect(lambda c=card: c.circuit(clockwise=True))
            dialog.rejected.connect(lambda c=card: c.circuit(clockwise=False))
        except Exception:                                    # noqa: BLE001
            LOG.debug("a dialog had no verdict to follow", exc_info=True)
        _Backdrop(dialog, card)
        _say_how_to_close_it(dialog)
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


def _say_how_to_close_it(dialog: QDialog) -> bool:
    """Add "press Escape to close" to a dialog that has no way to be closed.

    Glassing makes a dialog FRAMELESS, so its title bar goes and with it the
    x. A dialog with an OK or a Close button is fine -- that button is the
    way out and always was. One WITHOUT any button is not: "if i press about
    spacr now i cannot close the window because there is no close button".

    Only added where it is needed, and only once. A hint under a form that
    already has a Cancel button is noise, and noise on every dialog is how a
    hint stops being read.

    :returns: whether a hint was added.
    """
    from PySide6.QtWidgets import (QAbstractButton, QDialogButtonBox, QLabel,
                                   QVBoxLayout)

    if dialog.property(CLOSE_HINT):
        return False
    # A BUTTON THAT CLOSES IT is any button at all: every dialog button box
    # rejects or accepts, and a bare button in a dialog with no box is what
    # a hand-built OK looks like.
    if dialog.findChildren(QDialogButtonBox) or dialog.findChildren(
            QAbstractButton):
        return False
    layout = dialog.layout()
    if not isinstance(layout, QVBoxLayout):
        return False
    try:
        from ..i18n import tr

        hint = QLabel(tr("press Escape to close"), dialog)
        hint.setObjectName("Muted")
        hint.setAlignment(Qt.AlignHCenter)
        # SMALL, as asked. It is a reminder, not a control.
        hint.setStyleSheet("font-size: 10px;")
        hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(hint)
        dialog.setProperty(CLOSE_HINT, True)
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not add the close hint", exc_info=True)
        return False


def _install_the_backdrop(dialog: QDialog) -> Optional[QWidget]:
    """Put the drifting strata behind ``dialog``, or None if unavailable.

    The same engine and theme the setup screen uses, so a popup and the
    first-run screen are recognisably the same surface rather than two
    takes on one idea.
    """
    theme = "aurora"
    try:
        from ..preferences import get_ambient_enabled, get_popup_backdrop

        # THE USER'S OWN ANSWER ABOUT ANIMATED BACKGROUNDS. Somebody who
        # turned the backdrop off on the module screens has not asked for
        # it back in every popup; the card and the rim still apply, so the
        # look is the same one, just still.
        if not get_ambient_enabled():
            return None
        theme = get_popup_backdrop()
        # AND THEIR ANSWER FOR POPUPS IN PARTICULAR. What belongs behind a
        # screen full of figures is not necessarily what belongs behind a
        # form somebody is reading, so `off` drops the movement and keeps
        # the card and the rim.
        if theme == "off":
            return None
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the ambient preference", exc_info=True)
    try:
        from .ambient import install_ambient
        from .setup_slides import BACKDROP_SPEED

        # ROUNDED TO THE CARD'S RADIUS, for the reason the setup window
        # needed it: a SQUARE backdrop behind a rounded card is a second
        # surface, and it is exactly the "non rounded edge black box behind
        # the periphery" reported on About spaCR and the live mask settings.
        return install_ambient(dialog, theme=theme, speed=BACKDROP_SPEED,
                               corner_radius=CARD_RADIUS)
    except Exception:                                        # noqa: BLE001
        # INVARIANTS 10: with no ambient engine the card is still a card,
        # and the dialog still works.
        LOG.debug("no ambient backdrop for this dialog", exc_info=True)
        return None


class _GlassInstaller(QObject):
    """Applies :func:`glass` to every dialog the first time it is shown."""

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        try:
            # POLISH FIRST, SHOW AS THE FALLBACK. Polish arrives before a
            # widget is visible, which is when the window flags can be
            # changed without hiding it. Not every dialog is polished
            # before its first show -- one built and exec'd in a single
            # expression may not be -- so Show still catches it, and
            # `make_frameless` puts back what it had to hide.
            if (event.type() in (QEvent.Type.Polish, QEvent.Type.Show)
                    and wants_glass(watched)):
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
