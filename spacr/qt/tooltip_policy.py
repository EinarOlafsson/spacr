"""One rule for every tooltip in spaCR: when it appears and when it goes.

Qt's own answer is a style hint, ``SH_ToolTip_WakeUpDelay``, which most
styles set to about 700 ms and which nothing in spaCR was choosing. Tooltips
therefore arrived while the pointer was still travelling, and left the
instant it moved off the widget -- too fast to read, and impossible to reach
if the text ran long.

This module installs ONE event filter on ``QApplication`` and takes the
decision away from the style:

* a tooltip appears after :data:`SHOW_DELAY_MS` of hovering, not before;
* it stays while the pointer is on the widget OR on the tooltip itself;
* it leaves :data:`LINGER_MS` after the pointer leaves both.

Because the filter sits on the application object, a widget written later
obeys the rule without anyone remembering to ask for it: its ``ToolTip``
event travels to the application like every other one.

The filter shows the text itself, with ``QToolTip.showText`` and no owning
widget, rather than letting Qt show it. Handing Qt the widget hands Qt the
hiding as well -- Qt hides on the widget's ``Leave`` event, immediately,
which is the one behaviour this module exists to change.

Two things follow from taking the event, and both are handled rather than
accepted:

* Qt PROPAGATES a tooltip event to the parent widget when the widget under
  the pointer has no tooltip of its own, which is how a card explains
  itself while the pointer is on the label written on it. The text is
  therefore resolved with :func:`tooltip_text_for`, up the parent chain,
  exactly as Qt would have.
* A table, a header or a list answers from ``Qt::ToolTipRole`` inside its
  own ``event``, not from a ``toolTip()`` any filter can read. When
  nothing in the parent chain has a tooltip, the event is SENT AGAIN after
  the wait, with the filter standing aside, so those still appear -- and
  appear on the same two-second rule as everything else.

:func:`tooltips_enabled` is the preference switch, on by default. Cleared,
the ``ToolTip`` event is swallowed and no tooltip is shown anywhere.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QToolTip

LOG = logging.getLogger(__name__)

#: How long the pointer must rest before a tooltip appears, in milliseconds.
#: The maintainer asked for two seconds (2026-09-24): long enough that
#: crossing a toolbar never raises one, short enough to be an answer to a
#: question rather than a wait.
SHOW_DELAY_MS = 2000

#: How long a shown tooltip stays after the pointer has left both the widget
#: and the tooltip, in milliseconds.
LINGER_MS = 1000

#: How long ``QToolTip`` is told to keep the text up. This module decides
#: when the tooltip goes, so Qt's own expiry must not get there first; a
#: tooltip the pointer never leaves stays for an hour and then gives up,
#: which is the same as forever and cannot leak a stuck window.
HOLD_MS = 3600 * 1000

#: Dynamic property a widget can set to ``True`` to keep Qt's own instant
#: tooltip. Nothing in spaCR sets it; it exists so a widget with a reason
#: has a way out that is not "edit this module".
OPT_OUT_PROPERTY = "spacrNoTooltipPolicy"

_filter: Optional["_TooltipFilter"] = None
_enabled: Optional[bool] = None


def tooltips_enabled() -> bool:
    """Whether tooltips are shown at all. Cached; ``True`` by default.

    Read on every ``ToolTip`` event, which is why the answer is cached
    rather than re-read from ``QSettings`` each time.
    :func:`invalidate_tooltip_policy` drops the cache, and
    :func:`spacr.qt.preferences.set_tooltips_enabled` calls it, so the
    switch and the screen can never disagree.
    """
    global _enabled
    if _enabled is None:
        try:
            from .preferences import get_tooltips_enabled
            _enabled = bool(get_tooltips_enabled())
        except Exception:                                   # noqa: BLE001
            LOG.debug("could not read the tooltip preference", exc_info=True)
            _enabled = True
    return _enabled


def invalidate_tooltip_policy() -> None:
    """Forget the cached preference, and hide anything already up."""
    global _enabled
    _enabled = None
    if _filter is not None:
        _filter.hide_now()


def tooltip_text_for(widget) -> str:
    """The tooltip a hover on ``widget`` would raise, parents included.

    This is not a convenience: it is the behaviour being preserved. Qt
    PROPAGATES a tooltip event up the parent chain, so a label with no
    tooltip of its own inside a card that has one shows the card's. A
    filter that read ``widget.toolTip()`` alone and then swallowed the
    event would break every one of those -- a card explains itself until
    the pointer lands on the text written on it, and then it stops.

    :returns: the first non-empty tooltip from the widget outwards, or
        ``""`` if neither it nor any parent up to its window has one.
    """
    seen = 0
    while widget is not None and seen < 64:
        seen += 1
        try:
            text = str(widget.toolTip() or "")
        except Exception:                                    # noqa: BLE001
            return ""
        if text:
            return text
        try:
            if widget.isWindow():
                return ""
            widget = widget.parentWidget()
        except Exception:                                    # noqa: BLE001
            return ""
    return ""


class _TooltipFilter(QObject):
    """The application-wide filter. One instance, installed once."""

    def __init__(self, show_delay_ms: int = SHOW_DELAY_MS,
                 linger_ms: int = LINGER_MS) -> None:
        super().__init__()
        self.show_delay_ms = int(show_delay_ms)
        self.linger_ms = int(linger_ms)
        self._widget: Optional[object] = None
        self._text = ""
        self._pos = QPoint()
        self._showing = False
        self._replaying = False
        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.timeout.connect(self._show_now)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_if_the_pointer_left)

    def eventFilter(self, obj, event) -> bool:               # noqa: N802
        try:
            kind = event.type()
        except Exception:                                    # noqa: BLE001
            return False
        if kind == QEvent.Type.ToolTip:
            return self._on_tooltip(obj, event)
        if kind in (QEvent.Type.Leave, QEvent.Type.Hide,
                    QEvent.Type.WindowDeactivate):
            if obj is self._widget:
                self._start_the_linger()
        elif kind in (QEvent.Type.MouseButtonPress,
                      QEvent.Type.Wheel,
                      QEvent.Type.KeyPress):
            self.hide_now()
        return False

    def _on_tooltip(self, obj, event) -> bool:
        if self._replaying:
            return False
        if not tooltips_enabled():
            self.hide_now()
            return True
        widget = obj if hasattr(obj, "toolTip") else None
        if widget is None:
            return False
        try:
            if bool(widget.property(OPT_OUT_PROPERTY)):
                return False
        except Exception:                                    # noqa: BLE001
            pass
        try:
            self._pos = QPoint(event.globalPos())
        except Exception:                                    # noqa: BLE001
            self._pos = QCursor.pos()
        text = tooltip_text_for(widget)
        self._hide_timer.stop()
        if widget is self._widget and self._showing and text == self._text:
            return True
        if widget is not self._widget:
            if self._showing:
                self._hide_text()
            self._show_timer.stop()
        self._widget = widget
        self._text = text
        if not self._show_timer.isActive():
            self._show_timer.start(self.remaining_delay_ms())
        return True

    def remaining_delay_ms(self) -> int:
        """How much longer to wait, given the wait Qt has already served.

        Qt does not deliver the ``ToolTip`` event the moment the pointer
        stops: the style's ``SH_ToolTip_WakeUpDelay`` -- around 700 ms with
        Fusion -- has already gone by. Adding the full two seconds on top
        of that would make the total 2.7 s, which is not the number the
        maintainer asked for. So the style's delay is subtracted, and what
        the reader experiences is two seconds from resting to reading.
        """
        already = 0
        try:
            from PySide6.QtWidgets import QStyle
            style = QApplication.style()
            if style is not None:
                already = int(style.styleHint(
                    QStyle.StyleHint.SH_ToolTip_WakeUpDelay))
        except Exception:                                    # noqa: BLE001
            already = 0
        return max(0, self.show_delay_ms - max(0, already))

    def _show_now(self) -> None:
        """The pointer rested long enough. Put the text on screen."""
        self._show_timer.stop()
        if not tooltips_enabled():
            return
        widget = self._widget
        if widget is None:
            return
        try:
            if not widget.isVisible():
                return
        except Exception:                                    # noqa: BLE001
            return
        if not self._text:
            self._replay(widget)
            return
        try:
            QToolTip.showText(self._pos, self._text, None)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not show a tooltip", exc_info=True)
            return
        self._showing = True

    def _replay(self, widget) -> None:
        """Hand the widget back the event, now that the wait is over.

        Nothing in the widget's own chain of parents has a tooltip, so the
        text -- if there is any -- belongs to something INSIDE the widget:
        a table cell, a header section, a list row, all of which Qt answers
        from ``Qt::ToolTipRole`` in the widget's own ``event``. Swallowing
        that would silently take those tooltips away, and showing it at the
        moment of the hover would leave them the only fast ones left. So
        the event is sent again, after the wait, with the filter standing
        aside for exactly that one delivery.
        """
        from PySide6.QtGui import QHelpEvent

        self._widget = None
        self._text = ""
        try:
            local = widget.mapFromGlobal(self._pos)
            again = QHelpEvent(QEvent.Type.ToolTip, local, self._pos)
        except Exception:                                    # noqa: BLE001
            return
        self._replaying = True
        try:
            QApplication.sendEvent(widget, again)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not replay a tooltip event", exc_info=True)
        finally:
            self._replaying = False

    def _start_the_linger(self) -> None:
        """The pointer left the widget; give it :data:`LINGER_MS` to return."""
        self._show_timer.stop()
        if not self._showing:
            self._widget = None
            self._text = ""
            return
        self._hide_timer.start(self.linger_ms)

    def _hide_if_the_pointer_left(self) -> None:
        """Hide, unless the pointer is resting on the tooltip itself.

        A tooltip the reader has moved onto is a tooltip they are reading.
        Qt gives the text its own top-level widget, so asking which widget
        is under the pointer is enough to tell the two cases apart.
        """
        if self._pointer_is_on_the_tooltip():
            self._hide_timer.start(self.linger_ms)
            return
        self.hide_now()

    def _pointer_is_on_the_tooltip(self) -> bool:
        """Whether the window under the pointer is a tooltip window.

        The mask is not decoration. ``Qt::ToolTip`` is ``Popup | Sheet``,
        and both of those carry the ``Window`` bit, so a plain ``flags &
        ToolTip`` test is true of EVERY ordinary window -- which made the
        tooltip refuse to hide at all while any window sat under the
        pointer. Only the masked comparison asks the intended question.
        """
        try:
            under = QApplication.widgetAt(QCursor.pos())
        except Exception:                                    # noqa: BLE001
            return False
        if under is None:
            return False
        window = under.window()
        try:
            flags = window.windowFlags()
            kind = flags & Qt.WindowType.WindowType_Mask
        except Exception:                                    # noqa: BLE001
            return False
        return kind == Qt.WindowType.ToolTip

    def _hide_text(self) -> None:
        try:
            QToolTip.hideText()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not hide a tooltip", exc_info=True)
        self._showing = False

    def hide_now(self) -> None:
        """Take any tooltip away at once and forget what was hovered."""
        self._show_timer.stop()
        self._hide_timer.stop()
        if self._showing:
            self._hide_text()
        self._widget = None
        self._text = ""


def install_tooltip_policy(app=None) -> bool:
    """Install the application-wide tooltip filter. Idempotent.

    :param app: the QApplication; defaults to the running instance.
    :returns: ``True`` if a filter was installed by this call.
    """
    global _filter
    app = app or QApplication.instance()
    if app is None:
        return False
    if _filter is not None:
        app.removeEventFilter(_filter)
        app.installEventFilter(_filter)
        return False
    _filter = _TooltipFilter()
    app.installEventFilter(_filter)
    return True


def uninstall_tooltip_policy(app=None) -> bool:
    """Remove the filter. ``True`` if there was one."""
    global _filter
    if _filter is None:
        return False
    _filter.hide_now()
    app = app or QApplication.instance()
    if app is not None:
        app.removeEventFilter(_filter)
    _filter = None
    return True


def tooltip_policy() -> Optional["_TooltipFilter"]:
    """The installed filter, or ``None``. For tests and for diagnostics."""
    return _filter
