"""Hold Z and turn the wheel to resize the interface's text, live.

INSTRUCTION 378, and the measurement that shaped it.

The request came with a condition -- "only if possible to do fast without
lag" -- so the first question was what a notch costs. On one 1440x900
MainWindow with 847 widgets:

    stylesheet() rebuild                          1.4 ms
    app.setStyleSheet(...) + repolish           587.5 ms   <- Preferences
    repolish the visible screen only            107   ms
    QApplication.setFont(...)                    14   ms

Only the last one is live, and it moves nothing: the application sheet
carries 49 hardcoded ``font-size: <N>px`` declarations, and a QSS font-size
beats the inherited application font.

THE ESCAPE THE INSTRUCTION PROPOSED DOES NOT EXIST. Part 2 of 378 assumed
the 49 declarations could be rewritten in ``em`` or ``%`` so that one
``setFont`` moved everything. Qt does not implement either for
``font-size``: ``QCss`` accepts only ``pt``, ``px`` and the CSS size
keywords, and silently drops any other unit, so ``font-size: 2em`` leaves
the widget at the inherited size. Measured 2026-09-05 on this PySide6, and
pinned by ``test_qt_still_has_no_relative_font_size``, which fails the day
Qt gains the unit and makes the simpler design available.

WHAT IS LIVE INSTEAD. An explicit ``QWidget.setFont`` *does* beat the
application sheet's ``font-size`` -- it is the same escape hatch that lets a
per-widget sheet win -- and it costs one font assignment and one relayout
rather than a global unpolish/repolish. So a notch snapshots each visible
widget's font once, then re-scales every widget from that snapshot by the
ratio the wheel has travelled. The role hierarchy survives because each
widget is scaled from its own baseline: a 22 px title and a 13 px caption
stay in proportion without this module knowing which is which.

WHAT IS NOT LIVE, AND WHY THAT WAS ACCEPTED. Every size the font scale pins
from Python -- row heights, column widths, icon sizes, tile geometry, all of
:func:`spacr.qt.preferences.scaled_px` -- moves only when the stylesheet is
rebuilt, and that is the 587 ms number. The maintainer chose, 2026-09-05:
"Text live, spacing on release." Text follows the wheel; the spacing around
it catches up in one step when the wheel stops or Z comes up. It is a
deliberate compromise, not an oversight, and it is why the settle exists.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

LOG = logging.getLogger(__name__)

#: What one wheel notch is worth, as a fraction of 100 %.
#:
#: 5 % is the slider's own granularity in Preferences, so the gesture and
#: the control cannot disagree about which sizes are reachable.
FONT_SCALE_STEP = 0.05

#: A wheel notch, in eighths of a degree. Qt's own unit for one detent.
_NOTCH = 120.0

#: How long the wheel has to be still before the spacing catches up, in ms.
#:
#: The settle is the expensive half of this gesture (see the module
#: docstring), so it may not run between two notches of one flick. 220 ms is
#: comfortably longer than the ~50 ms gap of a fast scroll and short enough
#: that a user who has stopped does not think the gesture is unfinished.
_SETTLE_MS = 220

#: The smallest text this may produce, in px. Matches
#: :func:`spacr.qt.theme.font_px`, so the live pass cannot render text the
#: settled stylesheet would refuse to.
_MIN_PX = 6

#: The point-size floor for the few widgets whose font is not set in pixels.
_MIN_PT = 4.5

_FILTER_ATTRIBUTE = "_spacr_live_zoom_filter"


def _alive(widget) -> bool:
    """Is this widget's C++ side still there?

    A gesture holds a list of widgets across event-loop turns, and anything
    on it can be deleted while the wheel is still turning -- a tooltip, a
    dialog the user closed, a screen that rebuilt itself. Touching the
    Python wrapper afterwards raises ``RuntimeError`` from shiboken.
    """
    try:
        from shiboken6 import isValid
        return bool(isValid(widget))
    except Exception:                                        # noqa: BLE001
        try:
            widget.objectName()
            return True
        except RuntimeError:
            return False


def _scaled_font(base: QFont, ratio: float,
                 current: QFont) -> Optional[QFont]:
    """Return ``base`` grown by ``ratio``, or None when nothing would move.

    Sized from ``base`` and compared against ``current``, which are not the
    same question. Sizing from the font now on the widget compounds the
    rounding -- twenty notches of ``round(px * 1.05)`` drift far enough that
    letting go visibly jumps -- while comparing against ``base`` would skip
    the notch that puts a widget BACK to the size it started at, and leave
    it stranded one step above every neighbour on the way down.

    The comparison earns its place: a 5 % step on an 11 px caption rounds to
    the same pixel about half the time, and assigning a font that resolves
    to the size the widget already has still invalidates its layout.

    :param base: the widget's font when the gesture began.
    :param ratio: how far the wheel has travelled, as a multiplier.
    :param current: the font on the widget right now.
    """
    px = base.pixelSize()
    if px > 0:
        target = max(_MIN_PX, int(round(px * ratio)))
        if target == current.pixelSize():
            return None
        font = QFont(base)
        font.setPixelSize(target)
        return font
    points = base.pointSizeF()
    if points > 0:
        target = max(_MIN_PT, points * ratio)
        if abs(target - current.pointSizeF()) < 0.01:
            return None
        font = QFont(base)
        font.setPointSizeF(target)
        return font
    # A font with neither a pixel nor a point size is unresolved; leaving it
    # alone lets it keep inheriting rather than pinning it at a guess.
    return None


class LiveZoomFilter(QObject):
    """Application-wide filter that turns Z + wheel into a live font scale.

    Installed on the QApplication because the gesture has to work on every
    screen -- a per-screen handler works in some places and not others,
    which is the complaint that produced 315's warning about application
    filters. That warning is about COST, so the body is two integer
    comparisons for an event it does not want, and the wheel branch is not
    even reached unless Z is down. Measured at 1.2 us per uninteresting
    event on this machine, which is the Python call itself rather than
    anything this does inside it.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._held = False
        #: The scale the gesture started from, and the one on screen now.
        self._base_scale = 1.0
        self._live_scale = 1.0
        #: (widget, the font it had, whether that font was its OWN) for
        #: everything the gesture may move, plus the subset it did move.
        self._baseline: list[tuple[object, QFont, bool]] = []
        self._touched: set = set()
        #: The window the wheel was over, for the settle's rebuild.
        self._window = None
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(_SETTLE_MS)
        # NOT `self.settle` directly: the wheel going quiet ends the
        # gesture but does NOT end the hold, and the two are different
        # answers to "is Z still down?".
        self._settle_timer.timeout.connect(
            lambda: self.settle(released=False))

    # -- the filter ------------------------------------------------------

    def eventFilter(self, watched, event):    # noqa: N802 - Qt naming
        """Watch the widgets this filter is installed on.

        :param watched: the object the event is for.
        :param event: the event.
        :returns: True to stop the event going further.
        """
        kind = event.type()
        if kind == QEvent.KeyPress:
            if self._is_the_key(event) and not event.isAutoRepeat():
                self._arm()
        elif kind == QEvent.KeyRelease:
            # X11 sends a release/press PAIR for every auto-repeat tick, so a
            # filter that trusts KeyRelease disarms itself a few hundred
            # milliseconds into the hold and the rest of the gesture scrolls
            # the list instead.
            if self._held and self._is_the_key(event) \
                    and not event.isAutoRepeat():
                self.settle()
        elif self._held:
            if kind == QEvent.Wheel:
                return self._wheeled(watched, event)
            if kind == QEvent.WindowDeactivate:
                # Alt-tabbing away while Z is down means the KeyRelease is
                # delivered to another application and never arrives here.
                self.settle()
        return False

    @staticmethod
    def _is_the_key(event) -> bool:
        """Is this the Z the gesture is armed by?

        Z, not Ctrl: Ctrl+wheel is already canvas zoom on the mask editor
        and the fractal, and a gesture that means two things on one screen
        means neither. Ctrl+Z is undo everywhere, so a Z carrying a
        command modifier is somebody else's shortcut.
        """
        if event.key() != Qt.Key_Z:
            return False
        blocking = (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
        return not (event.modifiers() & blocking)

    def _arm(self) -> None:
        """Start listening for the wheel.

        NO EXEMPTION FOR A FOCUSED TEXT FIELD, though the first version had
        one -- somebody typing a plate name is pressing the same key. It was
        removed because it disabled the gesture on exactly the screens that
        want it: a settings form is mostly fields, one of them always has
        the focus, and a gesture that silently does nothing there is a worse
        failure than the one it prevented. Nothing happens on the key alone
        -- it is never consumed, so the letter is still typed -- so a misfire
        needs the user to hold Z down AND turn the wheel, which typing does
        not do.

        MAKE MASKS TAKES Z AND WINS THERE. That screen binds a `QShortcut`
        for its Zoom tool, and Qt dispatches shortcuts in `processKeyEvent`
        BEFORE application event filters, so this filter never sees the key
        press on Make Masks at all. Measured 2026-09-05: armed without that
        shortcut, not armed with it. So it is mutual exclusion rather than
        coexistence, and the gesture is simply unavailable on that one
        screen -- which is the right way round, because Zoom is the tool
        somebody opened Make Masks to use.
        """
        self._held = True

    def _wheeled(self, watched, event) -> bool:
        """Take one wheel event for the gesture. Always consumes it.

        Consumed even when the scale is already at a bound, and even for a
        purely horizontal wheel: while Z is held the wheel belongs to this
        gesture, and a list that scrolled under the pointer at 200 % would
        look like the gesture had leaked.
        """
        from .preferences import FONT_SCALE_MAX, FONT_SCALE_MIN

        event.accept()
        notches = event.angleDelta().y() / _NOTCH
        if notches:
            if not self._baseline:
                self._begin(watched)
            target = self._live_scale + FONT_SCALE_STEP * notches
            target = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, target))
            # ROUNDED, because this number is read back with `int(x * 100)`.
            # Four notches down from 1.0 in binary floating point is
            # 0.7999999999999998, which the Preferences slider truncates to
            # 79 % -- the gesture and the control disagreeing by a percent
            # for no reason a user could ever discover.
            target = round(target, 4)
            if target != self._live_scale:
                self._live_scale = target
                self._apply()
        self._settle_timer.start()
        return True

    # -- the live half ---------------------------------------------------

    def _begin(self, watched) -> None:
        """Snapshot the font of everything on screen.

        Visible widgets only. A hidden stack page is most of a mature
        session's widget count, nothing there can be seen mid-gesture, and
        the settle rebuilds the stylesheet for all of them anyway -- so
        scaling them live would be paying the whole cost of the gesture for
        none of its benefit.
        """
        from .preferences import get_font_scale

        app = QApplication.instance()
        if app is None:
            return
        self._base_scale = get_font_scale()
        self._live_scale = self._base_scale
        # WA_SetFont says whether the font on the widget is the widget's
        # own or one QSS resolved onto it -- Qt sets the attribute in
        # `QWidget::setFont` and NOT in the style sheet's font pass. The
        # settle needs the difference: putting a QSS-derived font back with
        # `setFont` would pin it, and a pinned font outlives the sheet that
        # was supposed to own it.
        self._baseline = [(w, QFont(w.font()), w.testAttribute(Qt.WA_SetFont))
                          for w in app.allWidgets() if w.isVisible()]
        self._touched = set()
        try:
            self._window = watched.window() if hasattr(watched, "window") \
                else None
        except Exception:                                    # noqa: BLE001
            self._window = None

    def _apply(self) -> None:
        """Re-scale every snapshotted widget from its own baseline font.

        The whole live half of the gesture is this loop: ~10 ms for the 204
        visible widgets of a MainWindow on this machine, against 1285 ms to
        set the stylesheet on the same window (measured 2026-09-05,
        offscreen; the maintainer's numbers on real hardware are 587 ms for
        the sheet). See :func:`_scaled_font` for why each widget is sized
        from its baseline rather than from what it is wearing.
        """
        if not self._base_scale:
            return
        ratio = self._live_scale / self._base_scale
        for widget, base, _own in self._baseline:
            try:
                font = _scaled_font(base, ratio, widget.font())
                if font is None:
                    continue
                widget.setFont(font)
                self._touched.add(widget)
            except RuntimeError:
                # Deleted mid-gesture -- a dialog the user closed, a screen
                # that rebuilt itself. Ordinary, not exceptional.
                continue

    # NO PERCENTAGE IN THE STATUS BAR, though it was written and removed.
    # The wheel has no detents a user can count, so a readout is genuinely
    # useful -- but it is one more English sentence, and every user-facing
    # sentence in this package is a row in ten translation catalogs that a
    # ratchet test audits. Adding it here would have meant regenerating
    # catalogs that other work is editing at the same time. The live text is
    # the feedback for now, and Preferences still shows the number.

    # -- the settle ------------------------------------------------------

    def settle(self, released: bool = True) -> None:
        """End the gesture: persist the scale and let the spacing catch up.

        THE EXPENSIVE HALF, ON PURPOSE. Everything :func:`scaled_px` pins is
        rebuilt here, in one 587 ms step, rather than twenty times a second
        while the wheel turns -- which is the compromise the maintainer
        chose over a gesture that stutters. Called when the wheel has been
        still for :data:`_SETTLE_MS`, when Z comes up, and when the window
        loses focus with Z still down.

        The QSettings write happens here too, for the same reason: twenty
        writes a second to a settings file is not free.

        :param released: whether Z is now up. False when the wheel merely
            went quiet: the user is still holding the key and may keep
            scrolling, and disarming under them would make the second half
            of one gesture scroll the list. The next notch simply starts a
            fresh gesture from the scale this one just saved.
        """
        self._settle_timer.stop()
        if released:
            self._held = False
        if not self._baseline:
            self._window = None
            return

        # PUT EACH WIDGET BACK THE WAY IT WAS HELD, not merely back at the
        # old size. A widget whose font came from the sheet has to end this
        # with no font of its own again: `setFont` sets WA_SetFont, and the
        # only rule that then still reaches it is one the sheet names
        # explicitly -- so a widget outside the blanket QWidget rule would
        # keep this gesture's size for the rest of the session. A widget
        # that really did own its font (the console's monospace, the AI
        # toggle's size) gets that font back verbatim.
        for widget, base, own in self._baseline:
            try:
                if widget in self._touched:
                    widget.setFont(base if own else QFont())
            except RuntimeError:
                continue
        self._baseline = []
        self._touched = set()

        scale, self._live_scale = self._live_scale, 1.0
        window, self._window = self._window, None

        # RE-STYLE EVEN WHEN THE SCALE CAME BACK TO WHERE IT STARTED, and
        # this is a fix rather than tidiness. Clearing a QSS-dressed widget
        # with `setFont(QFont())` above leaves it INHERITING rather than
        # styled -- the sheet's font-size does not come back until something
        # re-polishes it. Returning here because the number happened to be
        # unchanged left every widget the gesture touched with no styled
        # font at all: measured, Z + one notch up + one notch down +
        # release took the visible widgets from 13 px to unset, with
        # nothing scheduled to repair them.
        #
        # `set_font_scale` is still skipped in that case -- there is nothing
        # to persist -- but the re-polish is not optional.
        from .preferences import apply_preferences_to_app, set_font_scale
        if scale != self._base_scale:
            set_font_scale(scale)
        try:
            apply_preferences_to_app(QApplication.instance())
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not apply the font scale the wheel chose")
        # Icons, tile geometry and the dock are rebuilt from Python rather
        # than from QSS; only the window knows how. Just the one window the
        # wheel was over -- walking topLevelWidgets() reaches windows whose
        # C++ side is already being torn down, and rebuilding one of those
        # segfaults rather than raising (see `_refresh_owner_window`).
        if window is not None and _alive(window):
            refresh = getattr(window, "refresh_theme", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:                            # noqa: BLE001
                    LOG.debug("a window would not rebuild after a live zoom",
                              exc_info=True)


def install_live_zoom(app=None) -> Optional[LiveZoomFilter]:
    """Install the Z + wheel font gesture on a running application.

    Idempotent: the filter is retained on the application object, so a
    second call returns the one already installed rather than stacking a
    second filter onto every event in the process.

    :param app: optional QApplication; falls back to the running instance.
    :returns: the filter, or None when there is no application to hold it.
    """
    app = app or QApplication.instance()
    if app is None:
        return None
    existing = getattr(app, _FILTER_ATTRIBUTE, None)
    if existing is not None:
        return existing
    parent = app if isinstance(app, QObject) else None
    live_zoom = LiveZoomFilter(parent)
    app.installEventFilter(live_zoom)
    setattr(app, _FILTER_ATTRIBUTE, live_zoom)
    return live_zoom
