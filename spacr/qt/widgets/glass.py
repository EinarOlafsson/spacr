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

from PySide6.QtCore import QEvent, QObject, QRect, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView,
                               QAbstractSpinBox, QAbstractSlider, QGraphicsView, QPlainTextEdit, QApplication,
                               QComboBox, QDialog, QSplitterHandle, QTabBar,
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

    :param button: the button to classify: by its role in an enclosing
        ``QDialogButtonBox`` when it has one, otherwise by the words of its
        text.
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

    :param dialog: the dialog whose descendant buttons are wired; a button
        already wired, or whose direction :func:`button_direction` cannot tell,
        is skipped.
    :param card: the card whose ``circuit(clockwise=...)`` runs on each click,
        clockwise for a forward button.
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

    :param widget: the widget to test; only a ``QDialog`` without the opt-out
        or already-glassed property and without its own ``SetupCard``
        qualifies.
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

    :param dialog: the dialog whose descendant widgets, other than opaque
        controls and their children, are made transparent.
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


#: How far inside an edge a press still counts as a grab, in pixels.
#:
#: Frameless windows have no resize handles: the ones a user reaches for
#: belong to the title bar and the frame, and taking those away took the
#: resize with them. This band is what puts it back, and it is wide enough
#: to hit without aiming and narrow enough not to swallow a click on a
#: control sitting near the edge.
RESIZE_BAND = 6


def _edges_at(widget, point):
    """Which window edges ``point`` is on, as Qt edge flags (0 for none)."""
    edges = Qt.Edge(0)
    if widget.isMaximized() or widget.isFullScreen() or not widget.rect().contains(point):
        return edges
    if point.x() <= RESIZE_BAND:
        edges |= Qt.Edge.LeftEdge
    elif point.x() >= widget.width() - RESIZE_BAND:
        edges |= Qt.Edge.RightEdge
    if point.y() <= RESIZE_BAND:
        edges |= Qt.Edge.TopEdge
    elif point.y() >= widget.height() - RESIZE_BAND:
        edges |= Qt.Edge.BottomEdge
    if widget.minimumWidth() >= widget.maximumWidth():
        edges &= ~(Qt.LeftEdge | Qt.RightEdge)
    if widget.minimumHeight() >= widget.maximumHeight():
        edges &= ~(Qt.TopEdge | Qt.BottomEdge)
    return edges


def _cursor_for(edges):
    """The pointer shape that says which way an edge will move."""
    left = bool(edges & Qt.Edge.LeftEdge)
    right = bool(edges & Qt.Edge.RightEdge)
    top = bool(edges & Qt.Edge.TopEdge)
    bottom = bool(edges & Qt.Edge.BottomEdge)
    if (left and top) or (right and bottom):
        return Qt.CursorShape.SizeFDiagCursor
    if (right and top) or (left and bottom):
        return Qt.CursorShape.SizeBDiagCursor
    if left or right:
        return Qt.CursorShape.SizeHorCursor
    if top or bottom:
        return Qt.CursorShape.SizeVerCursor
    return None


def _blue_resize_cursor(edges):
    """Compatibility helper: resizing uses the unchanged native OS arrow."""
    from .cursor_policy import arrow_cursor

    return arrow_cursor()


class _ResizeEdgeHint(QWidget):
    """Paint a one-pixel blue line on the edges available for dragging."""

    def __init__(self, window):
        """Create an initially hidden edge overlay that never intercepts mouse input."""
        super().__init__(window)
        self.edges = Qt.Edge(0)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.hide()

    def show_edges(self, edges):
        """Match the window bounds and reveal only its currently resizable edges."""
        self.edges = edges
        self.setGeometry(self.parentWidget().rect())
        self.setVisible(bool(edges))
        if edges:
            self.raise_()
            self.update()

    def paintEvent(self, event):
        """Paint thin blue guides along active edges while leaving the corners unobscured."""
        painter = QPainter(self)
        painter.setPen(QPen(QColor('#168cff'), 1))
        left, top, right, bottom = 1, 1, self.width() - 2, self.height() - 2
        for edge, line in (
            (Qt.LeftEdge, (left, 8, left, bottom - 7)),
            (Qt.RightEdge, (right, 8, right, bottom - 7)),
            (Qt.TopEdge, (8, top, right - 7, top)),
            (Qt.BottomEdge, (8, bottom, right - 7, bottom)),
        ):
            if self.edges & edge:
                painter.drawLine(*line)


def _owns_mouse_gesture(widget, window):
    """Keep controls, selectable text and viewport gestures with their owner."""
    controls = (QAbstractButton, QAbstractSlider, QAbstractSpinBox,
                QComboBox, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractItemView,
                QGraphicsView, QSplitterHandle, QTabBar)
    while widget is not None and widget is not window:
        if isinstance(widget, controls):
            return True
        if hasattr(widget, "textInteractionFlags") and widget.textInteractionFlags() & Qt.TextSelectableByMouse:
            return True
        if getattr(widget, "hasSelectedText", lambda: False)():
            return True
        if widget.property("spacrOwnsMouseGesture"):
            return True
        widget = widget.parentWidget()
    return False


class _ResizeByEdge(QObject):
    """Resize from the original press geometry while keeping the arrow cursor.

    Absolute press coordinates avoid accumulated movement errors. Wayland
    requires compositor-owned moves and resizes; other platforms keep the
    interaction in Qt so the compositor cannot substitute another cursor.
    """

    def __init__(self, window):
        """Resize a frameless window by dragging its edges.

        :param window: the window to resize. INSTALLS ITSELF on it and turns
            on mouse tracking, so the caller only has to keep the object
            alive; it is also the QObject parent, which is how that happens
            by default.
        """
        super().__init__(window)
        self._window = window
        self._grab = None
        self._hint = _ResizeEdgeHint(window)
        window.setMouseTracking(True)
        for child in window.findChildren(QWidget):
            child.setMouseTracking(True)
        window.installEventFilter(self)

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        """Resize from a corner or edge and highlight that edge in blue.

        :param watched: the window receiving the pointer event.
        :param event: mouse press, move, release or leave event.
        :returns: whether a resize gesture consumed the event. Wayland uses
            its required compositor operation; other platforms use anchored
            geometry while respecting minimum and maximum window sizes.
        """
        window = getattr(self, "_window", None)
        if window is None or watched is not window:
            return False
        try:
            kind = event.type()
            if kind == QEvent.Resize:
                self._hint.setGeometry(window.rect())
            if kind in (QEvent.Hide, QEvent.WindowDeactivate, QEvent.WindowStateChange):
                self._grab = None
                self._hint.show_edges(Qt.Edge(0))
            if kind == QEvent.Type.MouseMove and self._grab is not None:
                if not event.buttons() & Qt.LeftButton:
                    self._grab = None
                    return False
                edges, origin, rectangle = self._grab
                delta = event.globalPosition().toPoint() - origin
                minimum = window.minimumSize().expandedTo(window.minimumSizeHint())
                maximum = window.maximumSize()
                x, y, width, height = rectangle.getRect()
                if edges & (Qt.LeftEdge | Qt.RightEdge):
                    proposed = width + (-delta.x() if edges & Qt.LeftEdge else delta.x())
                    width = min(max(proposed, max(1, minimum.width())), maximum.width())
                    if edges & Qt.LeftEdge:
                        x = rectangle.right() - width + 1
                if edges & (Qt.TopEdge | Qt.BottomEdge):
                    proposed = height + (-delta.y() if edges & Qt.TopEdge else delta.y())
                    height = min(max(proposed, max(1, minimum.height())), maximum.height())
                    if edges & Qt.TopEdge:
                        y = rectangle.bottom() - height + 1
                window.setGeometry(QRect(x, y, width, height))
                return True
            if kind == QEvent.Type.MouseButtonRelease:
                self._grab = None
                self._hint.show_edges(_edges_at(window, event.position().toPoint()))
            if kind == QEvent.Type.MouseMove and not event.buttons():
                edges = _edges_at(window, event.position().toPoint())
                self._hint.show_edges(edges)
                return False
            if (kind == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                edges = _edges_at(window, event.position().toPoint())
                if not edges:
                    return False
                handle = window.windowHandle()
                if handle is None:
                    return False
                self._hint.show_edges(edges)
                if QApplication.platformName().lower().startswith("wayland"):
                    handle.startSystemResize(edges)
                else:
                    self._grab = (edges, event.globalPosition().toPoint(), window.geometry())
                    window.setCursor(_blue_resize_cursor(edges))
                return True
            if kind == QEvent.Type.Leave:
                if self._grab is None:
                    self._hint.show_edges(Qt.Edge(0))
        except Exception:                                    # noqa: BLE001
            LOG.debug("the resize filter tripped", exc_info=True)
        return False


def let_the_user_resize(window) -> bool:
    """Give ``window`` edge-drag resizing. True when it was installed.

    Idempotent: a window that already carries the filter keeps the one it
    has, so a dialog shown, closed and shown again does not collect two.

    :param window: the top-level widget that gets an edge-drag resize filter;
        ``None`` or a window that already has one returns ``False``.
    """
    if window is None:
        return False
    if getattr(window, "_spacr_resizer", None) is not None:
        return False
    try:
        window._spacr_resizer = _ResizeByEdge(window)
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not make this window resizable", exc_info=True)
        return False


class _DragByBackground(QObject):
    """Move a frameless dialog by dragging its empty background.

    THE TITLE BAR WAS WHERE A WINDOW WAS DRAGGED FROM, so taking it away
    takes that with it, and every popup has to stay a window the user can
    move. A press that lands on a control is left entirely alone; only one
    on the dialog itself starts a drag.
    """

    def __init__(self, dialog: QDialog):
        """Drag a frameless dialog by any part of its background.

        :param dialog: the dialog to move. Installs itself on it and takes
            it as the QObject parent; see :meth:`eventFilter` for why every
            later read of it goes through ``getattr``.
        """
        super().__init__(dialog)
        self._dialog = dialog
        self._grab = None
        dialog.installEventFilter(self)

    def eventFilter(self, watched, event):
        """Drag passive child surfaces while preserving each control's gestures."""
        dialog = getattr(self, "_dialog", None)
        if dialog is None or not isinstance(watched, QWidget):
            return False
        try:
            kind = event.type()
            if watched.window() is not dialog:
                return False
            if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                point = event.globalPosition().toPoint()
                local = event.position().toPoint()
                target = watched.childAt(local) or watched
                if _edges_at(dialog, watched.mapTo(dialog, local)) or _owns_mouse_gesture(target, dialog):
                    self._grab = None
                    return False
                self._grab = (point, dialog.pos())
                return False
            if kind == QEvent.MouseMove and self._grab is not None:
                if not event.buttons() & Qt.LeftButton:
                    self._grab = None
                    return False
                origin, position = self._grab
                delta = event.globalPosition().toPoint() - origin
                if delta.manhattanLength() < QApplication.startDragDistance():
                    return False
                handle = dialog.windowHandle()
                if (QApplication.platformName().lower().startswith("wayland")
                        and handle is not None and handle.startSystemMove()):
                    self._grab = None
                else:
                    dialog.move(position + delta)
                return True
            if kind in (QEvent.MouseButtonRelease, QEvent.Hide, QEvent.WindowDeactivate):
                self._grab = None
        except RuntimeError:
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

    :param dialog: the dialog made translucent, frameless, draggable by its
        background and resizable by its edges; it is shown again if changing
        its flags hid it.
    """
    try:
        was_showing = not dialog.isHidden()
        dialog.setAttribute(Qt.WA_TranslucentBackground, True)
        dialog.setWindowFlags((dialog.windowFlags()
                               & ~Qt.WindowType.Dialog)
                              | Qt.WindowType.Window
                              | Qt.FramelessWindowHint)
        dialog.setProperty(DETACHED, True)
        _paint_nothing_behind_the_card(dialog)
        if getattr(dialog, "_spacr_background_drag", None) is None:
            dialog._spacr_background_drag = _DragByBackground(dialog)
        let_the_user_resize(dialog)
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

    :param dialog: the widget whose window mask is cut to a rounded rectangle
        of its current size; an empty size returns ``False``.
    """
    try:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainterPath, QRegion

        from PySide6.QtGui import QTransform

        rect = dialog.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
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
        LOG.debug("could not round the window corners", exc_info=True)
        return False


class _Backdrop(QObject):
    """Keeps one card at the size of the dialog it sits behind.

    An event filter rather than a resizeEvent override, because the dialog
    is somebody else's class and this must not require subclassing it.
    """

    def __init__(self, dialog: QDialog, card: QWidget):
        """Keep a backdrop sized to the dialog behind its card.

        :param dialog: the dialog to follow; also the QObject parent.
        :param card: the widget the backdrop is drawn behind. Held
            separately because it is not the dialog and not a fixed child
            of it -- the backdrop tracks the dialog's geometry but paints
            around this one.
        """
        super().__init__(dialog)
        self._dialog = dialog
        self._card = card
        dialog.installEventFilter(self)
        self._fit()

    def _fit(self) -> None:
        """Resize the backdrop to the dialog, if both are still alive.

        Every read goes through ``getattr``: Qt delivers to a filter whose Python
        attributes have already been cleared during teardown.
        """
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
        """Refit the backdrop when the dialog is resized or shown.

        :param watched: the dialog.
        :param event: the event.
        :returns: ``False`` -- both events are observed, never consumed.
        """
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

    :param dialog: the dialog to decorate; it is left alone unless
        :func:`wants_glass` accepts it.
    """
    if not wants_glass(dialog):
        return False
    try:
        from .setup_card import SetupCard
    except Exception:                                        # noqa: BLE001
        LOG.debug("no card to put behind this dialog", exc_info=True)
        return False
    try:
        backdrop = _install_the_backdrop(dialog)

        _paint_nothing_behind_the_card(dialog)

        card = SetupCard(dialog, radius=CARD_RADIUS)
        card.lower()
        card.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        card.show()
        if backdrop is not None:
            backdrop.lower()
        _make_room_for_the_rim(dialog)
        make_frameless(dialog)
        round_the_corners(dialog)
        spin_on_every_button(dialog, card)
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
        from ..theme import font_px
        hint.setStyleSheet(f"font-size: {font_px(10)}px;")
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

        if not get_ambient_enabled():
            return None
        theme = get_popup_backdrop()
        if theme == "off":
            return None
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the ambient preference", exc_info=True)
    try:
        from .ambient import install_ambient
        from .setup_slides import BACKDROP_SPEED

        return install_ambient(dialog, theme=theme, speed=BACKDROP_SPEED,
                               corner_radius=CARD_RADIUS)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no ambient backdrop for this dialog", exc_info=True)
        return None


#: The two moments this filter acts on, as a set of the enum members built
#: once. The filter is on the QApplication, so the membership test is the
#: first thing every event in the process pays for: as a tuple rebuilt per
#: event it cost two global lookups, four attribute lookups and a tuple
#: build for each of the 94,431 events one module open delivers.
_DRAG_MOMENTS = frozenset({QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease})

_GLASS_MOMENTS = frozenset({QEvent.Type.Polish, QEvent.Type.Show})


class _GlassInstaller(QObject):
    """Applies :func:`glass` to every dialog the first time it is shown."""

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        """Apply the glass treatment to a dialog as it appears.

        POLISH FIRST, SHOW AS THE FALLBACK. Polish arrives before the widget is
        visible, which is when the window flags can be changed without hiding
        it -- but not every dialog is polished before its first show, since one
        built and exec'd in a single expression may not be, so Show catches the
        rest.

        :param watched: the widget being polished or shown.
        :param event: the event.
        :returns: ``False`` -- never consumed.
        """
        try:
            if event.type() in _GLASS_MOMENTS and isinstance(watched, QWidget):
                if getattr(watched.window(), '_spacr_resizer', None) is not None:
                    watched.setMouseTracking(True)
            if event.type() in _DRAG_MOMENTS and isinstance(watched, QWidget):
                window = watched.window()
                resizer = getattr(window, '_spacr_resizer', None)
                if resizer is not None and watched is not window:
                    point = window.mapFromGlobal(event.globalPosition().toPoint())
                    mapped = QMouseEvent(event.type(), point.toPointF(), event.globalPosition(),
                                         event.button(), event.buttons(), event.modifiers())
                    if resizer.eventFilter(window, mapped):
                        return True
                drag = getattr(window, "_spacr_background_drag", None)
                if drag is not None and watched is not window:
                    return drag.eventFilter(watched, event)
            if event.type() in _GLASS_MOMENTS and wants_glass(watched):
                glass(watched)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the glass filter tripped", exc_info=True)
        return False


#: The one installed filter. Held so it is not collected, and so a second
#: call on the same application is a no-op rather than a second filter on
#: every event in that application.
_INSTALLED: Optional[_GlassInstaller] = None

#: The application that owns :data:`_INSTALLED`.
#:
#: IDEMPOTENCE IS PER APPLICATION, not per Python process.  A test harness
#: can launch spaCR against a stand-in application and then return to its real
#: QApplication; an embedded host can likewise tear one application down and
#: build another.  The old filter cannot serve the new application, so the
#: owner has to be remembered alongside it.
_INSTALLED_APP = None


def install_glass_everywhere(application=None) -> bool:
    """Install the filter. True when it was installed by this call.

    Called once at startup. Every dialog opened afterwards -- Preferences,
    the hyperparameter search, live settings, the AI providers, the figure
    settings, and the thirty-odd others -- is treated on its first show
    without knowing anything about this module.
    """
    global _INSTALLED, _INSTALLED_APP

    try:
        from PySide6.QtWidgets import QApplication

        application = application or QApplication.instance()
        from .cursor_policy import install_cursor_policy

        install_cursor_policy(application)
        if application is None:
            return False
        if _INSTALLED is not None and _INSTALLED_APP is application:
            return False

        if _INSTALLED is not None:
            try:
                if _INSTALLED_APP is not None:
                    _INSTALLED_APP.removeEventFilter(_INSTALLED)
            except Exception:                                # noqa: BLE001
                LOG.debug("the old glass filter would not come off",
                          exc_info=True)

        installed = _GlassInstaller(application)
        application.installEventFilter(installed)
        _INSTALLED = installed
        _INSTALLED_APP = application
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("the glass filter would not install", exc_info=True)
        _INSTALLED = None
        _INSTALLED_APP = None
        return False


def uninstall_glass_everywhere(application=None) -> bool:
    """Remove the application-wide glass event filter.

    When :func:`install_glass_everywhere` has registered a filter, remove it
    from ``application`` or, when omitted, from ``QApplication.instance()``.
    The module's installation state is cleared even if no application instance
    exists or Qt raises while removing the filter. Styling already applied to
    dialogs is not reverted.

    :param application: Qt application from which to remove the filter. If
        ``None``, use the current ``QApplication`` instance.
    :returns: ``True`` if an installed filter was registered when the call
        began; ``False`` if no filter was installed.
    """
    global _INSTALLED, _INSTALLED_APP

    if _INSTALLED is None:
        return False
    try:
        from PySide6.QtWidgets import QApplication

        application = _INSTALLED_APP or application or QApplication.instance()
        if application is not None:
            application.removeEventFilter(_INSTALLED)
    except Exception:                                        # noqa: BLE001
        LOG.debug("the glass filter would not come off", exc_info=True)
    finally:
        _INSTALLED = None
        _INSTALLED_APP = None
    return True
