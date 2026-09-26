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
    """
    global _HUB_CLASS
    if _HUB_CLASS is not None:
        return _HUB_CLASS
    from PySide6.QtCore import QObject
    from shiboken6 import isValid

    class _ApplicationEventHub(QObject):
        """The one application event filter the watchers share."""

        def __init__(self, parent=None) -> None:
            """Start with no watchers."""
            super().__init__(parent)
            self._entries = []
            self._by_kind = {}

        def _rebuild(self) -> None:
            """Index the live watchers by event type, newest first."""
            by_kind = {}
            for entry in self._entries:
                for kind in entry[1]:
                    by_kind.setdefault(kind, []).append(entry)
            self._by_kind = {kind: tuple(chain)
                             for kind, chain in by_kind.items()}

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
                chain = self._by_kind.get(event.type())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
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
