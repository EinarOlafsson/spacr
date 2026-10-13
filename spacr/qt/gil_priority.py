"""Keep the Qt interface responsive during Python-bound background work.

Python code that does not release the global interpreter lock can delay the
GUI thread. :func:`claim` temporarily lowers the interpreter thread-switch
interval to :data:`BUSY_INTERVAL`; :func:`release` restores the previous value
after the last active worker finishes. Prefer the balanced
:func:`responsive_gui` context manager for pipeline work.

The setting is process-wide, so it is applied only while a Qt worker is
active. Importing this module does not change the switch interval and headless
runs incur no cost.

Notes
-----
A shorter interval creates more context switches and can make pure-Python
workers marginally slower. NumPy and similar compiled operations generally
release the interpreter lock and are less affected.
"""
from __future__ import annotations

import logging
import sys
import threading
import weakref
from contextlib import contextmanager

LOG = logging.getLogger("spacr.qt.gil_priority")

#: What the interval becomes while a worker is running, in seconds. 1 ms:
#: measured at 17.74 ms median against 16.00 idle and 42.42 unaided. Lower
#: buys little (the GUI thread only needs waking every 16 ms) and costs the
#: worker more switching.
BUSY_INTERVAL = 0.001

_LOCK = threading.RLock()
_DEPTH = 0
_RESTORE = None


def claim() -> None:
    """Request the responsive-GUI switch interval for one active worker.

    Calls are reference-counted and thread-safe. Each call should be paired
    with :func:`release`; the original interval is restored only after the
    final claim is released.
    """
    global _DEPTH, _RESTORE
    with _LOCK:
        if _DEPTH == 0:
            try:
                _RESTORE = sys.getswitchinterval()
                sys.setswitchinterval(BUSY_INTERVAL)
            except Exception:                         # noqa: BLE001
                LOG.debug("could not lower the switch interval", exc_info=True)
                _RESTORE = None
        _DEPTH += 1


def release() -> None:
    """Release one worker's claim and restore the interval when none remain.

    Extra calls after the count reaches zero have no effect.
    """
    global _DEPTH, _RESTORE
    with _LOCK:
        _DEPTH = max(0, _DEPTH - 1)
        if _DEPTH == 0 and _RESTORE is not None:
            try:
                sys.setswitchinterval(_RESTORE)
            except Exception:                         # noqa: BLE001
                LOG.debug("could not restore the switch interval",
                          exc_info=True)
            _RESTORE = None


def active() -> bool:
    """Return whether at least one worker holds a responsiveness claim."""
    with _LOCK:
        return _DEPTH > 0


@contextmanager
def responsive_gui():
    """Apply the responsive-GUI interval for the duration of a context.

    The claim is released when the block exits, including when it raises an
    exception. Nested and concurrent contexts are supported.

    Yields
    ------
    None
        Control returns to the context body while the claim is active.
    """
    claim()
    try:
        yield
    finally:
        release()


_HUB_ATTRIBUTE = "_spacr_application_event_hub"
_HUB_CLASS = None


def _application_event_hub_class():
    """The hub's class, defined on first use so importing this stays Qt-free.

    ONE PYTHON CALL PER EVENT INSTEAD OF ONE PER FILTER. Every
    application-wide event filter spaCR installs is a Python object, and Qt
    calls each of them for every event in the process; each call crosses
    from C++ into Python, takes the interpreter lock and builds wrappers for
    the object and the event before the filter's first line can say the
    event is not its business. Putting a screen's first stylesheet on
    delivers about seven events per widget -- 6,288 on Make Masks, most of
    them PaletteChange, FontChange and StyleChange, which no filter reads --
    and with thirteen filters installed the crossings were most of that
    call. Thread CPU time of the one setStyleSheet, same screen, same
    process, load 90-100, three repeats each:

        thirteen filters installed on the application   650-700 ms
        the same thirteen behind this hub               200-290 ms

    Each watcher names the event types it acts on, and the hub reads the
    type once and calls only the watchers that asked for it.

    QT'S OWN RULES ARE KEPT: the watcher registered last is asked first,
    registering again moves it to the front, the first ``True`` ends the
    event, a watcher removed while an event is being delivered is not asked
    about it, and a watcher whose object has been destroyed is skipped and
    then dropped, as Qt drops a destroyed filter. An
    exception in one watcher goes to ``sys.excepthook``, as PySide sends an
    exception raised in an event filter, and the rest are still asked.

    A FRAME IS PAINTED WITHOUT IT. Over the animated field every widget on
    screen is sent a paint event on every frame -- 163 on Home and 179 on
    Make Masks at 2560x1440 -- and an application filter is asked about
    each, though no watcher reads a paint. So when the update request that
    paints a widget's window reaches the hub, and no watcher acts on an
    event a repaint sends (a paint, a move, a resize or an update request),
    the hub steps off the application, hands the request on to the window
    itself, and steps back on when the window has painted: every filter on
    the window and every other application filter still hears the request,
    once, and nothing the watchers asked for passes while the hub is off.
    A request for anything that is not a widget goes the usual way.
    Measured offscreen, cheapest GUI-thread CPU of a synchronous field frame
    (median of six interleaved rounds): 0.65 ms of 4.8 on Home and 0.67 of
    6.8 on Make Masks.
    """
    global _HUB_CLASS
    if _HUB_CLASS is not None:
        return _HUB_CLASS
    from PySide6.QtCore import QCoreApplication, QEvent, QObject
    from shiboken6 import isValid

    update_request = QEvent.Type.UpdateRequest
    paint_pass_kinds = frozenset({
        update_request, QEvent.Type.Paint, QEvent.Type.Move,
        QEvent.Type.Resize, QEvent.Type.UpdateLater})

    class _ApplicationEventHub(QObject):
        """The one application event filter the watchers share."""

        def __init__(self, parent=None) -> None:
            """Start with no watchers."""
            super().__init__(parent)
            self._entries = []
            self._by_kind = {}
            self._paints_unheard = True

        def _rebuild(self) -> None:
            """Index the live watchers by event type, newest first."""
            by_kind = {}
            for entry in self._entries:
                for kind in entry[1]:
                    by_kind.setdefault(kind, []).append(entry)
            self._by_kind = {kind: tuple(chain)
                             for kind, chain in by_kind.items()}
            self._paints_unheard = not paint_pass_kinds.intersection(by_kind)

        def add(self, watcher, kinds) -> None:
            """Ask ``watcher`` first about every event of ``kinds``."""
            self.discard(watcher)
            self._entries.insert(0, [weakref.ref(watcher), frozenset(kinds),
                                     True])
            self._rebuild()

        def discard(self, watcher) -> bool:
            """Stop asking ``watcher``; ``True`` when it was being asked."""
            found = False
            kept = []
            for entry in self._entries:
                if entry[0]() is watcher:
                    entry[2] = False
                    found = True
                else:
                    kept.append(entry)
            if found:
                self._entries = kept
                self._rebuild()
            return found

        def watchers(self) -> tuple:
            """The live watchers, in the order they are asked."""
            return tuple(watcher for watcher in
                         (entry[0]() for entry in self._entries)
                         if watcher is not None and isValid(watcher))

        def eventFilter(self, watched, event):
            """Hand ``event`` to the watchers that asked for its type."""
            try:
                kind = event.type()
                chain = self._by_kind.get(kind)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
            if kind == update_request and self._paints_unheard:
                app = self.parent()
                if app is not None and watched.isWidgetType():
                    app.removeEventFilter(self)
                    try:
                        QCoreApplication.sendEvent(watched, event)
                    finally:
                        app.installEventFilter(self)
                    return True
            if not chain:
                return False
            dead = False
            try:
                for entry in chain:
                    if not entry[2]:
                        continue
                    watcher = entry[0]()
                    if watcher is None or not isValid(watcher):
                        dead = True
                        continue
                    try:
                        if watcher.eventFilter(watched, event):
                            return True
                    except Exception:
                        sys.excepthook(*sys.exc_info())
                return False
            finally:
                if dead:
                    self._forget_the_dead()

        def _forget_the_dead(self) -> None:
            """Drop every watcher whose object is gone.

            A watcher Qt held directly came off the application when its
            object was destroyed; behind the hub it would stay in the chain,
            skipped but still looked at on every event of its kinds -- one
            per module screen ever built, for the life of the process.
            """
            kept = []
            for entry in self._entries:
                watcher = entry[0]()
                if watcher is None or not isValid(watcher):
                    entry[2] = False
                else:
                    kept.append(entry)
            if len(kept) != len(self._entries):
                self._entries = kept
                self._rebuild()

    _HUB_CLASS = _ApplicationEventHub
    return _HUB_CLASS


def _application_event_hub(app, create: bool = True):
    """The hub on ``app``, installed on first use; ``None`` if it cannot be.

    :param app: the application. Anything that is not a live
        ``QCoreApplication`` -- a stand-in a test passes -- has no hub, and
        its watchers are installed on it directly instead.
    :param create: install a hub when there is none yet.
    """
    if app is None:
        return None
    try:
        from PySide6.QtCore import QCoreApplication
        from shiboken6 import isValid
    except ImportError:
        return None
    if not isinstance(app, QCoreApplication):
        return None
    hub = getattr(app, _HUB_ATTRIBUTE, None)
    if hub is not None and not isValid(hub):
        hub = None
    if hub is not None or not create:
        return hub
    if not isValid(app):
        return None
    hub = _application_event_hub_class()(app)
    app.installEventFilter(hub)
    setattr(app, _HUB_ATTRIBUTE, hub)
    return hub


def _watch_application_events(app, watcher, kinds) -> bool:
    """Call ``watcher.eventFilter`` for ``app``'s events of ``kinds``.

    The replacement for ``app.installEventFilter(watcher)`` on an
    application-wide filter; see :func:`_application_event_hub_class` for
    why. ``kinds`` must name every event type the filter can act on: an
    event of any other type is never shown to it.

    :param app: the application; ``None`` does nothing.
    :param watcher: a ``QObject`` with an ``eventFilter``.
    :param kinds: the ``QEvent.Type`` values it acts on.
    :returns: ``True`` when the watcher is now watching.
    """
    if app is None or watcher is None:
        return False
    hub = _application_event_hub(app)
    if hub is None:
        app.installEventFilter(watcher)
        return True
    hub.add(watcher, kinds)
    return True


def _stop_watching_application_events(app, watcher) -> bool:
    """Undo :func:`_watch_application_events`; ``True`` if it was watching.

    :param app: the application; ``None`` does nothing.
    :param watcher: the watcher to stop asking.
    """
    if app is None or watcher is None:
        return False
    hub = _application_event_hub(app, create=False)
    if hub is None:
        app.removeEventFilter(watcher)
        return True
    try:
        return hub.discard(watcher)
    except RuntimeError:
        return False


def _application_watchers(app) -> tuple:
    """The watchers on ``app``'s hub, the one asked first first.

    :param app: the application.
    :returns: an empty tuple when it has no hub.
    """
    hub = _application_event_hub(app, create=False)
    if hub is None:
        return ()
    try:
        return hub.watchers()
    except RuntimeError:
        return ()


def _widget_state(widget) -> tuple:
    """What a watcher could have heard change about ``widget``."""
    from PySide6.QtCore import Qt

    return (widget.isVisible(), widget.testAttribute(Qt.WA_WState_Polished),
            widget.pos(), widget.size(), widget.isEnabled(),
            widget.parent())


def _what_was_unheard(before, after) -> list:
    """The events a watcher missed about a widget, as fresh events.

    The style, palette and font changes a style sheet sends are told about
    every widget under it, whether its own changed or not, because a sheet
    sends a style change to each of them; everything else only when the
    widget's state shows it happened.

    :param before: the widget's :func:`_widget_state` before, or ``None``
        when it was built under the sheet.
    :param after: its :func:`_widget_state` now.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHideEvent, QMoveEvent, QResizeEvent, QShowEvent

    events = [QEvent(QEvent.StyleChange), QEvent(QEvent.PaletteChange),
              QEvent(QEvent.FontChange)]
    was = before or (False, False, after[2], after[3], after[4], None)
    if after[1] and not was[1]:
        events.append(QEvent(QEvent.Polish))
    if after[0] != was[0]:
        events.append(QShowEvent() if after[0] else QHideEvent())
    if after[2] != was[2]:
        events.append(QMoveEvent(after[2], was[2]))
    if after[3] != was[3]:
        events.append(QResizeEvent(after[3], was[3]))
    if after[4] != was[4]:
        events.append(QEvent(QEvent.EnabledChange))
    if after[5] is not was[5]:
        events.append(QEvent(QEvent.ParentChange))
    return events


@contextmanager
def _restyle_unheard(app, root):
    """Put a style sheet on ``root`` without the hub hearing each event.

    A sheet set on a container sends every widget under it a style change,
    a palette change and a font change, and the hub is asked about each:
    about 1,000 crossings into Python, 7 ms of the 25 ms a sheet on the
    Mask column takes, though no watcher acts on most of them. So the hub
    steps off the application while the block runs, as it does for a
    repaint, and when it is back the watchers are told what they would have
    heard about the widgets under ``root``: every style, palette and font
    change, and each show, hide, move, resize, polish, enable or re-parent
    the widgets' state shows happened. Each watcher still hears each such
    event once; it hears them after the sheet is on rather than during it.
    Without a hub this only runs the block.

    :param app: the application.
    :param root: the widget the sheet goes on.
    :yields: nothing.
    """
    hub = _application_event_hub(app, create=False)
    if hub is None or root is None:
        yield
        return
    from PySide6.QtWidgets import QWidget

    widgets = [root] + list(root.findChildren(QWidget))
    before = {id(widget): _widget_state(widget) for widget in widgets}
    app.removeEventFilter(hub)
    try:
        yield
    finally:
        app.installEventFilter(hub)
        _tell_the_watchers(hub, root, before)


def _tell_the_watchers(hub, root, before) -> None:
    """Hand ``hub`` what it missed under ``root``; see
    :func:`_restyle_unheard`. Only events some watcher asked for are made.
    """
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    if not isValid(root):
        return
    wanted = hub._by_kind
    if not wanted:
        return
    for widget in [root] + list(root.findChildren(QWidget)):
        events = _what_was_unheard(before.get(id(widget)),
                                   _widget_state(widget))
        for event in events:
            if event.type() in wanted:
                hub.eventFilter(widget, event)
