"""EdgeDrawer — a panel that slides in from the left edge on hover.

The full app list used to be a permanent 220 px column. It is now a
*reveal*: a 6 px strip along the left edge that, when the pointer rests
against it, slides the whole panel in over the page and slides it out
again when the pointer leaves.

Three things make the difference between a reveal that feels deliberate
and one that fires while you are aiming at something else:

* **Dwell, not touch.** Entering the strip starts
  :data:`EdgeDrawer.OPEN_DELAY_MS`; a pointer that crosses the strip on
  its way somewhere else has left again before the timer fires. This is
  the whole reason a hot corner is usable at all.
* **A generous close.** Leaving does *not* close immediately —
  :data:`EdgeDrawer.CLOSE_DELAY_MS` covers the gap between the panel and
  wherever the pointer wobbled to, and any re-entry cancels it. Panels
  that snap shut the instant the pointer strays are unusable with a
  trackpad.
* **A pin.** Clicking any row while the drawer is open would otherwise
  race the close animation. :meth:`hold` keeps it open until something
  explicitly releases it, which is also how the keyboard path works.

**It is not mouse-only.** :meth:`open_for_keyboard` opens the drawer,
pins it, and focuses the first row; Escape (or focus leaving the panel)
closes it again. The window installs a shortcut for this — a reveal you
can only reach by hovering is a reveal a keyboard user cannot reach at
all.
"""
from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve, QEvent, QPoint, QPropertyAnimation, Qt, QTimer, Signal,
)
from PySide6.QtWidgets import QWidget


class EdgeDrawer(QWidget):
    """Hosts ``panel`` as a slide-in overlay on the left of ``host``.

    The drawer is a child of ``host`` (not a top-level window), so it
    clips to the page, scrolls with nothing, and needs no platform
    window flags. It is raised above its siblings whenever it opens.

    :param host: the widget the drawer overlays. Must outlive the drawer.
    :param panel: the content — typically the app ``Sidebar``.
    :param width: drawn width in px; defaults to the panel's own.
    """

    #: Width in px of the invisible strip along the left edge that arms
    #: the reveal. Wide enough to hit with a trackpad flick, narrow
    #: enough that it is not in the way of page content.
    TRIGGER_W = 6

    #: How long the pointer must rest in the strip before the drawer
    #: opens. Below ~120 ms a pointer merely travelling left-to-right
    #: triggers it; above ~400 ms it feels broken.
    OPEN_DELAY_MS = 220

    #: Grace period after the pointer leaves before the drawer closes.
    CLOSE_DELAY_MS = 400

    #: Slide duration. Long enough to read as motion, short enough not
    #: to be something you wait for.
    SLIDE_MS = 170

    opened = Signal()
    closed = Signal()

    def __init__(self, host: QWidget, panel: QWidget, width: int = 0,
                 parent=None):
        super().__init__(parent or host)
        self.setObjectName("EdgeDrawer")
        self._host = host
        self._panel = panel
        self._held = False
        #: Whether the drawer is *meant* to be on screen. Not derived
        #: from ``x()``: the slide takes 170 ms, and for those 170 ms a
        #: position-derived answer would say "closed" about a drawer
        #: that is opening — which is how a caller ends up racing the
        #: animation.
        self._open_state = False

        panel.setParent(self)
        panel.move(0, 0)
        self._width = int(width or panel.width() or panel.sizeHint().width())
        self.resize(self._width, host.height())
        panel.resize(self._width, self.height())

        # Start fully off-screen to the left. Not hidden: a hidden widget
        # reports no geometry, and the tutorial overlay (which highlights
        # the sidebar) needs a rectangle to point at.
        self.move(-self._width, 0)
        self.hide()

        self._anim = QPropertyAnimation(self, b"pos", self)
        self._anim.setDuration(self.SLIDE_MS)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        self._open_timer = QTimer(self)
        self._open_timer.setSingleShot(True)
        self._open_timer.setInterval(self.OPEN_DELAY_MS)
        self._open_timer.timeout.connect(self.open)

        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(self.CLOSE_DELAY_MS)
        self._close_timer.timeout.connect(self._close_unless_held)

        # The hot strip. A separate zero-chrome child so the drawer's own
        # geometry can stay off-screen while something on-screen still
        # receives the hover.
        self._trigger = _EdgeTrigger(host, self)
        self._trigger.show()

        host.installEventFilter(self)
        self.setAttribute(Qt.WA_Hover, True)

    # -- geometry ------------------------------------------------------
    def relayout(self) -> None:
        """Re-fit to the host. Called on every host resize."""
        h = self._host.height()
        self.resize(self._width, h)
        self._panel.resize(self._width, h)
        self._trigger.setGeometry(0, 0, self.TRIGGER_W, h)
        self._trigger.raise_()
        self.move(0 if self.is_open() else -self._width, 0)

    def is_open(self) -> bool:
        """True while the panel is on screen, or sliding on."""
        return self._open_state

    def is_fully_open(self) -> bool:
        """True only once the slide has finished."""
        return self._open_state and self.x() >= 0

    # -- open / close --------------------------------------------------
    def arm(self) -> None:
        """Pointer entered the hot strip — start the dwell timer."""
        self._close_timer.stop()
        if not self.is_open():
            self._open_timer.start()

    def disarm(self) -> None:
        """Pointer left the hot strip before the dwell elapsed."""
        self._open_timer.stop()

    def open(self) -> None:
        """Slide the panel in."""
        self._open_timer.stop()
        self._close_timer.stop()
        self._open_state = True
        self.relayout_for_open()
        self.show()
        self.raise_()
        self._animate_to(0)
        self.opened.emit()

    def relayout_for_open(self) -> None:
        h = self._host.height()
        self.resize(self._width, h)
        self._panel.resize(self._width, h)

    def close(self) -> None:
        """Slide the panel back out and unpin it."""
        self._open_timer.stop()
        self._close_timer.stop()
        self._held = False
        if not self._open_state:
            return
        self._open_state = False
        self._animate_to(-self._width)
        self.closed.emit()

    def schedule_close(self) -> None:
        """Close after the grace period, unless re-entered or held."""
        self._open_timer.stop()
        if not self._held:
            self._close_timer.start()

    def hold(self, held: bool = True) -> None:
        """Pin the drawer open (a click landed in it, or the keyboard did)."""
        self._held = bool(held)
        if self._held:
            self._close_timer.stop()

    def is_held(self) -> bool:
        return self._held

    def toggle(self) -> None:
        """Keyboard entry point: open+pin, or close if already open."""
        if self.is_open():
            self.close()
        else:
            self.open_for_keyboard()

    def open_for_keyboard(self) -> None:
        """Open, pin, and move focus into the panel.

        Without this the reveal is mouse-only, which makes every app
        except the nine on the Home tabs unreachable without a pointer.
        """
        self.open()
        self.hold(True)
        target = self._first_focusable()
        if target is not None:
            target.setFocus(Qt.TabFocusReason)

    def _first_focusable(self):
        for child in self._panel.findChildren(QWidget):
            if child.focusPolicy() != Qt.NoFocus and child.isVisibleTo(
                    self._panel):
                return child
        return None

    # -- animation -----------------------------------------------------
    def _animate_to(self, x: int) -> None:
        self._anim.stop()
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(QPoint(x, 0))
        self._anim.start()

    def _on_anim_finished(self) -> None:
        if not self._open_state:
            self.hide()

    def _close_unless_held(self) -> None:
        if not self._held:
            self.close()

    # -- events --------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self._host and event.type() == QEvent.Resize:
            self.relayout()
        return super().eventFilter(obj, event)

    def enterEvent(self, event):
        self._close_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.schedule_close()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            self._host.setFocus(Qt.OtherFocusReason)
            return
        super().keyPressEvent(event)


class _EdgeTrigger(QWidget):
    """The invisible strip along the left edge that arms the drawer.

    Transparent to paint but *not* to mouse events — it has to receive
    the hover. It never takes focus and never accepts a click, so a
    press near the edge still reaches whatever is underneath.
    """

    def __init__(self, host: QWidget, drawer: EdgeDrawer):
        super().__init__(host)
        self.setObjectName("EdgeDrawerTrigger")
        self._drawer = drawer
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("All apps — hover to reveal (Ctrl+B)")
        self.setGeometry(0, 0, EdgeDrawer.TRIGGER_W, host.height())

    def enterEvent(self, event):
        self._drawer.arm()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._drawer.disarm()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """A deliberate click on the strip opens immediately and pins."""
        self._drawer.open()
        self._drawer.hold(True)
        event.accept()
