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
from typing import Optional

LOG = logging.getLogger("spacr.qt.gil_priority")

#: What the interval becomes while a worker is running, in seconds. 1 ms:
#: measured at 17.74 ms median against 16.00 idle and 42.42 unaided. Lower
#: buys little (the GUI thread only needs waking every 16 ms) and costs the
#: worker more switching.
BUSY_INTERVAL = 0.001

_LOCK = threading.RLock()
_DEPTH = 0
_RESTORE = None


_HOLDS_LOCK = threading.Lock()
_HOLDS: dict = {}

#: Longest a hold keeps background imports waiting, in seconds, so a view
#: that never reports itself ready cannot keep them out for good.
HOLD_LIMIT_S = 60.0


def hold_background_imports(owner, busy=None) -> None:
    """Ask the optional background imports to wait while ``owner`` loads.

    The window's warm imports (the data libraries and a module's pipeline
    entry, torch included) compete with the GUI thread for the interpreter
    lock; on Windows that starved an Overview's first browser open by
    several seconds. A hold only postpones their start, never cancels it,
    and lapses after :data:`HOLD_LIMIT_S` or when ``owner`` is deleted.

    :param owner: the object loading; held weakly.
    :param busy: optional ``busy(owner) -> bool``; the hold counts only while
        it answers True, so a view holds while it loads and not while it idles.
    """
    import time

    try:
        reference = weakref.ref(owner)
    except TypeError:
        return
    with _HOLDS_LOCK:
        _HOLDS.setdefault(id(owner), (reference, time.monotonic(), busy))


def release_background_imports(owner) -> None:
    """End ``owner``'s hold; the waiting imports start at their next check.

    :param owner: the object that called :func:`hold_background_imports`.
    """
    with _HOLDS_LOCK:
        entry = _HOLDS.get(id(owner))
        if entry is not None and entry[0]() is owner:
            del _HOLDS[id(owner)]


def background_imports_held() -> bool:
    """Whether a live, recent hold asks the background imports to wait."""
    import time

    now = time.monotonic()
    with _HOLDS_LOCK:
        held = []
        for key, (reference, started, busy) in list(_HOLDS.items()):
            owner = reference()
            if owner is None or now - started >= HOLD_LIMIT_S:
                del _HOLDS[key]
            else:
                held.append((owner, busy))
    for owner, busy in held:
        try:
            if busy is None or busy(owner):
                return True
        except RuntimeError:
            continue
    return False


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


#: PyTorch's shared libraries :func:`_open_torch_libraries` maps, in the order
#: ``torch/__init__.py`` loads them, as ``(file in torch/lib, global)``.
_TORCH_LIBRARIES = (("libtorch_global_deps.so", True),
                    ("libtorch_cpu.so", False),
                    ("libtorch_cuda.so", False))

#: The CUDA libraries ``torch/__init__.py`` loads after its global library
#: when its CUDA runtime came from a wheel, as ``(folder under nvidia,
#: pattern)``.
_TORCH_WHEEL_CUDA_LIBRARIES = (("cuda_nvrtc", "libnvrtc.so.*[0-9]"),
                               ("nvjitlink", "libnvJitLink.so.*[0-9]"))

_TORCH_LIBRARIES_TRIED = False


def _dlopen_without_the_lock():
    """The C library's ``dlopen``, called through ctypes.

    A ctypes call to a foreign function releases the interpreter lock for
    the length of the call, which neither ``import`` nor ``ctypes.CDLL``
    does: both hold it while the dynamic linker maps and relocates.
    """
    import ctypes

    function = ctypes.CDLL(None).dlopen
    function.restype = ctypes.c_void_p
    function.argtypes = (ctypes.c_char_p, ctypes.c_int)
    return function


def _wheel_cuda_library(folder: str, pattern: str) -> Optional[str]:
    """The first ``nvidia/<folder>/lib/<pattern>`` on ``sys.path``.

    The search ``torch._preload_cuda_deps`` makes, so the file is the one
    PyTorch would load.

    :param folder: the package folder under ``nvidia``.
    :param pattern: the library's glob pattern.
    """
    import glob
    import os

    for entry in sys.path:
        root = os.path.join(entry, "nvidia")
        if not os.path.exists(root):
            continue
        found = (glob.glob(os.path.join(root, folder, "lib", pattern))
                 or glob.glob(os.path.join(entry, folder, "lib", pattern)))
        if found:
            return found[0]
    return None


def _open_wheel_cuda_libraries(dlopen, maps_path: str = "/proc/self/maps") -> bool:
    """Map NVRTC and nvJitLink as PyTorch does after its global library.

    Only when the CUDA runtime that library brought in came from a wheel,
    which is PyTorch's own condition; otherwise there is nothing to do.

    :param dlopen: from :func:`_dlopen_without_the_lock`.
    :param maps_path: the process's memory map, read for that condition.
    :returns: False when a library PyTorch would load could not be found.
    """
    import os

    try:
        with open(maps_path, encoding="utf-8") as maps:
            if "nvidia/cuda_runtime/lib/libcudart.so" not in maps.read():
                return True
    except OSError:
        return False
    for folder, pattern in _TORCH_WHEEL_CUDA_LIBRARIES:
        path = _wheel_cuda_library(folder, pattern)
        if path is None or not dlopen(os.fsencode(path), os.RTLD_NOW):
            return False
    return True


def _open_torch_libraries() -> bool:
    """Map PyTorch's shared libraries with the interpreter lock released.

    IMPORTING TORCH HOLDS THE LOCK FOR A THIRD OF A SECOND IN ONE C CALL.
    ``import torch`` maps ``libtorch_cpu`` and ``libtorch_cuda`` (1.3 GB)
    while it holds the interpreter lock: 300-340 ms in which no other thread
    runs Python. The GUI thread needs the lock for every event a Python
    object handles, so lesson 08's Measure run froze the window for 326 ms
    when its worker imported torch half a second after Run.

    This maps the same files first, in the order and with the flags
    ``torch/__init__.py`` uses (``libtorch_global_deps`` global; the
    wheel's NVRTC and nvJitLink when the CUDA runtime came from a wheel, as
    PyTorch does next; then ``libtorch_cpu`` and ``libtorch_cuda``), through
    ctypes calls to ``dlopen``, which release the lock. The import that
    follows finds them mapped. Torch itself is not imported, and the process
    ends up with the libraries PyTorch would have loaded, from the same
    files.

    Linux only, once per process, and only while torch is installed and not
    yet imported. A build without ``libtorch_cuda`` maps the rest. Anything
    unexpected stops it and leaves the rest to ``import torch``. Call it on
    a worker thread.

    Returns
    -------
    bool
        True when every library the build has was mapped here.
    """
    global _TORCH_LIBRARIES_TRIED
    with _LOCK:
        if _TORCH_LIBRARIES_TRIED or "torch" in sys.modules:
            return False
        _TORCH_LIBRARIES_TRIED = True
    if not sys.platform.startswith("linux"):
        return False
    try:
        import importlib.util
        import os

        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.origin:
            return False
        lib = os.path.join(os.path.dirname(spec.origin), "lib")
        dlopen = _dlopen_without_the_lock()
        for index, (name, is_global) in enumerate(_TORCH_LIBRARIES):
            path = os.path.join(lib, name)
            if not os.path.isfile(path):
                if index == 0:
                    return False
                continue
            mode = os.RTLD_NOW | (os.RTLD_GLOBAL if is_global else 0)
            if not dlopen(os.fsencode(path), mode):
                return False
            if index == 0 and not _open_wheel_cuda_libraries(dlopen):
                return False
        return True
    except Exception:                                 # noqa: BLE001
        LOG.debug("could not map the torch libraries early", exc_info=True)
        return False


def _hub_geometry_kinds() -> frozenset:
    """The event types that make a watch a geometry watch."""
    from PySide6.QtCore import QEvent

    return frozenset({QEvent.Type.Move, QEvent.Type.Resize,
                      QEvent.Type.Show, QEvent.Type.Hide})


def _hub_rebuild_objects(hub) -> None:
    """Index ``hub``'s object watchers' kinds and their geometry watches.

    The whole index, from every chain. Adding or dropping one watch goes
    through :func:`_hub_reindex_object` instead, which touches one object's
    chain: rebuilding from every chain on each change made a screen that
    registers a few hundred watches pay for the square of that, about
    0.28 s of building Make Masks on an Intel Mac.
    """
    geometry_kinds = _hub_geometry_kinds()
    counts = {}
    geometry = {}
    for key, chain in hub._objects.items():
        for entry in chain:
            for kind in entry[1]:
                counts[kind] = counts.get(kind, 0) + 1
            if entry[1] & geometry_kinds and (len(entry) < 5 or entry[4]):
                geometry[key] = entry[3]
    hub._kind_counts = counts
    hub._object_kinds = frozenset(counts)
    hub._geometry_watched = geometry


def _hub_reindex_object(hub, key, old_chain) -> None:
    """Update the index for one object whose chain changed from ``old_chain``.

    :param hub: the event hub.
    :param key: the object's pointer key in ``hub._objects``.
    :param old_chain: that object's chain before the change, or ``None``.
    """
    counts = getattr(hub, "_kind_counts", None)
    if counts is None:
        _hub_rebuild_objects(hub)
        return
    for entry in old_chain or ():
        for kind in entry[1]:
            left = counts.get(kind, 0) - 1
            if left > 0:
                counts[kind] = left
            else:
                counts.pop(kind, None)
    geometry_kinds = _hub_geometry_kinds()
    hub._geometry_watched.pop(key, None)
    for entry in hub._objects.get(key, ()):
        for kind in entry[1]:
            counts[kind] = counts.get(kind, 0) + 1
        if entry[1] & geometry_kinds and (len(entry) < 5 or entry[4]):
            hub._geometry_watched[key] = entry[3]
    hub._object_kinds = frozenset(counts)


def _hub_add_object(hub, watched, watcher, kinds, replay=True) -> None:
    """Ask ``watcher`` first about ``watched``'s events of ``kinds``."""
    from shiboken6 import getCppPointer

    key = getCppPointer(watched)[0]
    old_chain = hub._objects.get(key)
    chain = [entry for entry in old_chain or []
             if entry[0]() is not watcher]
    if key not in hub._objects:
        watched.destroyed.connect(
            lambda *_args, key=key: _hub_drop_object(hub, key))
    chain.insert(0, [weakref.ref(watcher), frozenset(kinds), True,
                     weakref.ref(watched), bool(replay)])
    hub._objects[key] = chain
    _hub_reindex_object(hub, key, old_chain)


def _hub_discard_object(hub, watched, watcher) -> bool:
    """Stop ``hub`` asking ``watcher`` about ``watched``."""
    from shiboken6 import getCppPointer

    try:
        key = getCppPointer(watched)[0]
    except (RuntimeError, TypeError):
        return False
    chain = hub._objects.get(key)
    if not chain:
        return False
    kept = []
    found = False
    for entry in chain:
        if entry[0]() is watcher:
            entry[2] = False
            found = True
        else:
            kept.append(entry)
    if kept:
        hub._objects[key] = kept
    else:
        hub._objects.pop(key, None)
    _hub_reindex_object(hub, key, chain)
    return found


def _hub_drop_object(hub, key) -> None:
    """Forget every watch on an object that has been destroyed."""
    chain = hub._objects.pop(key, None)
    for entry in chain or ():
        entry[2] = False
    if chain is not None:
        _hub_reindex_object(hub, key, chain)


def _hub_object_watchers(hub, watched) -> tuple:
    """The live watchers of ``watched``, in the order they are asked."""
    from shiboken6 import getCppPointer, isValid

    try:
        chain = hub._objects.get(getCppPointer(watched)[0], ())
    except (RuntimeError, TypeError):
        return ()
    return tuple(watcher for watcher in
                 (entry[0]() for entry in chain if entry[2])
                 if watcher is not None and isValid(watcher))


def _hub_ask_object_watchers(hub, watched, event, kind) -> bool:
    """Hand ``event`` to ``watched``'s watchers of ``kind``."""
    from shiboken6 import getCppPointer, isValid

    try:
        chain = hub._objects.get(getCppPointer(watched)[0])
    except (RuntimeError, TypeError):
        return False
    if not chain:
        return False
    for entry in tuple(chain):
        if not entry[2] or kind not in entry[1]:
            continue
        watcher = entry[0]()
        if watcher is None or not isValid(watcher):
            continue
        try:
            if watcher.eventFilter(watched, event):
                return True
        except Exception:
            sys.excepthook(*sys.exc_info())
    return False


_WINDOW_PLACES = weakref.WeakKeyDictionary()


def _hub_widget_place(widget) -> tuple:
    """``widget``'s geometry as plain ints: x, y, width and height.

    Read through ``QWidget`` itself, so a control that hides one of those
    names on its instance is still measured. A frame reads this for every
    geometry-watched widget, and each ``QRect`` PySide builds -- returned
    by ``geometry()`` or made in Python -- gives the interpreter lock up and
    waits to take it back. While a pipeline or a search thread runs Python
    that wait is a switch interval each time: the 42 widgets Map Barcodes
    watches cost 27-160 ms a frame, at worst 1.9 s, where ints cost 0.2 ms.

    A window's ``x()`` and ``y()`` place its frame, not its client area, so
    its ``geometry()`` is read again only when one of those ints changes.

    :param widget: a live widget.
    :returns: ``(x, y, width, height)`` as ``QWidget.geometry`` gives them.
    """
    from PySide6.QtWidgets import QWidget

    ints = (QWidget.x(widget), QWidget.y(widget), QWidget.width(widget),
            QWidget.height(widget))
    if not QWidget.isWindow(widget):
        return ints
    seen = _WINDOW_PLACES.get(widget)
    if seen is None or seen[0] != ints:
        rect = QWidget.geometry(widget)
        seen = (ints, (rect.x(), rect.y(), rect.width(), rect.height()))
        _WINDOW_PLACES[widget] = seen
    return seen[1]


def _hub_place_of(geometry) -> tuple:
    """A recorded place as ``(x, y, width, height)``, from a tuple or a rect."""
    if isinstance(geometry, tuple):
        return geometry
    return (geometry.x(), geometry.y(), geometry.width(), geometry.height())


def _hub_geometry_before(hub) -> list:
    """Where each geometry-watched widget is, before a paint pass.

    Five integer reads a widget and no Qt value objects, because this runs
    on every frame of the field (:func:`_hub_widget_place`).
    """
    from PySide6.QtWidgets import QWidget

    hidden_of = QWidget.isHidden
    states = []
    for ref in tuple(hub._geometry_watched.values()):
        widget = ref()
        if widget is None:
            continue
        try:
            states.append((widget, hidden_of(widget),
                           _hub_widget_place(widget)))
        except (AttributeError, RuntimeError, TypeError):
            continue
    return states


def _hub_geometry_after(hub, states) -> None:
    """Tell the watchers what moved during a paint pass they missed.

    A paint pass runs with the hub off the application, so a move, resize,
    show or hide sent from inside a paint is not heard as it happens. Each
    is made again from the widget's state and handed to the watchers that
    asked for it, once. Nothing but ints is read before a widget has moved.
    """
    from PySide6.QtWidgets import QWidget

    hidden_of = QWidget.isHidden
    for widget, hidden, geometry in states:
        try:
            now_hidden, now = hidden_of(widget), _hub_widget_place(widget)
        except (RuntimeError, TypeError):
            continue
        was = _hub_place_of(geometry)
        if now_hidden == hidden and now == was:
            continue
        from PySide6.QtCore import QPoint, QSize
        from PySide6.QtGui import (QHideEvent, QMoveEvent, QResizeEvent,
                                   QShowEvent)

        events = []
        if now_hidden != hidden:
            events.append(QHideEvent() if now_hidden else QShowEvent())
        if now[:2] != was[:2]:
            events.append(QMoveEvent(QPoint(*now[:2]), QPoint(*was[:2])))
        if now[2:] != was[2:]:
            events.append(QResizeEvent(QSize(*now[2:]), QSize(*was[2:])))
        for event in events:
            kind = event.type()
            if kind in hub._object_kinds:
                _hub_ask_object_watchers(hub, widget, event, kind)


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
            self._objects = {}
            self._object_kinds = frozenset()
            self._geometry_watched = {}

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
                    if (kind in self._object_kinds
                            and _hub_ask_object_watchers(self, watched, event, kind)):
                        return True
                    moved = (_hub_geometry_before(self)
                             if self._geometry_watched else None)
                    app.removeEventFilter(self)
                    try:
                        QCoreApplication.sendEvent(watched, event)
                    finally:
                        app.installEventFilter(self)
                        if moved:
                            _hub_geometry_after(self, moved)
                    return True
            if not chain:
                return (kind in self._object_kinds
                        and _hub_ask_object_watchers(self, watched, event, kind))
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
                return (kind in self._object_kinds
                        and _hub_ask_object_watchers(self, watched, event, kind))
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


def _watch_object_events(watched, watcher, kinds, replay=True) -> bool:
    """Call ``watcher.eventFilter`` for ``watched``'s events of ``kinds``.

    The replacement for ``watched.installEventFilter(watcher)`` with a
    Python filter that never acts on a paint. Over the animated field every
    widget on screen is painted on every frame, and Qt asks a widget's own
    filters about each paint: lesson 13's Regression screen re-entered
    Python 233 times per 3840x2160 frame, and during a run each re-entry
    waits behind the pipeline's hold on the interpreter lock. Behind the
    application hub the watcher hears only the kinds it names, in the same
    call that already asks the application's watchers, and a paint pass
    never reaches it; a move, resize, show or hide sent from inside a paint
    is told afterwards. It is asked after the application's own watchers
    and before the object's remaining Qt filters, which is where Qt asks an
    application filter.

    ``kinds`` must name every event type the filter can act on and must
    not include ``Paint`` or ``UpdateRequest``: such a filter is installed
    on ``watched`` directly, as before.

    :param watched: the object to watch.
    :param watcher: a ``QObject`` with an ``eventFilter``.
    :param kinds: the ``QEvent.Type`` values it acts on.
    :param replay: whether a move, resize, show or hide made inside a paint
        is told afterwards. Every frame of the field checks each widget
        watched this way, so a watcher that is told the next change anyway
        passes ``False``.
    :returns: ``True`` when the watcher is now watching.
    """
    if watched is None or watcher is None:
        return False
    from PySide6.QtCore import QCoreApplication, QEvent

    kinds = frozenset(kinds)
    hub = None
    if not kinds & {QEvent.Type.Paint, QEvent.Type.UpdateRequest}:
        hub = _application_event_hub(QCoreApplication.instance())
    if hub is None:
        watched.installEventFilter(watcher)
        return True
    _hub_add_object(hub, watched, watcher, kinds, replay)
    return True


def _show_paint_layer(owner, attribute: str, pixmap) -> None:
    """Show ``pixmap`` over ``owner`` in a plain child label, or hide it.

    A widget that drew a decoration in a Python ``paintEvent`` was asked
    for it on every frame of the window's field, a re-entry into Python
    each time. Drawn once into a pixmap and shown by a ``QLabel`` with no
    Python methods, it is painted by Qt alone, after the owner's own paint
    as the ``paintEvent`` drew it, and with the same pixels: an opaque
    colour drawn with coverage onto a transparent pixmap and then blended
    onto the owner is the same 8-bit sum as drawing it on the owner.

    :param owner: the widget the decoration belongs to.
    :param attribute: where on ``owner`` the label is kept.
    :param pixmap: the decoration at ``owner``'s size and pixel ratio, or
        ``None`` to hide it.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    layer = owner.__dict__.get(attribute)
    if pixmap is None or pixmap.isNull():
        if layer is not None:
            layer.hide()
        return
    if layer is None:
        layer = QLabel(owner)
        layer.setObjectName("spacrPaintLayer")
        layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layer.setFocusPolicy(Qt.NoFocus)
        layer.setStyleSheet("QLabel#spacrPaintLayer { background: transparent;"
                            " border: none; padding: 0px; margin: 0px; }")
        layer.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layer.setContentsMargins(0, 0, 0, 0)
        layer.setAccessibleName("")
        owner.__dict__[attribute] = layer
    layer.setGeometry(owner.rect())
    layer.setPixmap(pixmap)
    layer.raise_()
    layer.show()


def _blank_layer_pixmap(owner):
    """A transparent pixmap at ``owner``'s size and pixel ratio."""
    import math

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    ratio = owner.devicePixelRatioF()
    pixmap = QPixmap(max(1, math.ceil(owner.width() * ratio)),
                     max(1, math.ceil(owner.height() * ratio)))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    return pixmap


def _stop_watching_object_events(watched, watcher) -> bool:
    """Undo :func:`_watch_object_events`; ``True`` if it was watching."""
    if watched is None or watcher is None:
        return False
    from PySide6.QtCore import QCoreApplication

    hub = _application_event_hub(QCoreApplication.instance(), create=False)
    found = False
    if hub is not None:
        try:
            found = _hub_discard_object(hub, watched, watcher)
        except RuntimeError:
            found = False
    try:
        watched.removeEventFilter(watcher)
    except RuntimeError:
        pass
    return found


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

    before = {}
    for widget in [root] + list(root.findChildren(QWidget)):
        address = _cpp_address(widget)
        if address is not None:
            before[address] = (widget, _widget_state(widget))
    app.removeEventFilter(hub)
    try:
        yield
    finally:
        app.installEventFilter(hub)
        _tell_the_watchers(hub, root, before)


def _cpp_address(widget):
    """The C++ address behind ``widget``, or ``None`` when it has none.

    What the replay keys a widget by. A wrapper's ``id`` is no key:
    PySide on Windows answered ``findChildren`` after a sheet with new
    ``QWidget`` wrappers for ``QLabel``s Python still held, and the id of a
    wrapper made for one call can be handed to another object once it is
    collected, pairing one widget's state with another's.
    """
    from shiboken6 import getCppPointer

    try:
        return int(getCppPointer(widget)[0])
    except (RuntimeError, TypeError, ValueError, IndexError):
        return None


def _tell_the_watchers(hub, root, before) -> None:
    """Hand ``hub`` what it missed under ``root``; see
    :func:`_restyle_unheard`. Only events some watcher asked for are made.

    Each widget seen before the sheet is told about as the wrapper seen
    then, which is the one its owner holds, so a watcher that compares
    widgets by identity or keys them by wrapper finds it.
    """
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    if not isValid(root):
        return
    wanted = set(hub._by_kind) | set(getattr(hub, "_object_kinds", ()))
    if not wanted:
        return
    for widget in [root] + list(root.findChildren(QWidget)):
        seen = before.get(_cpp_address(widget))
        state = None
        if seen is not None:
            if isValid(seen[0]):
                widget = seen[0]
            state = seen[1]
        events = _what_was_unheard(state, _widget_state(widget))
        for event in events:
            if event.type() in wanted:
                hub.eventFilter(widget, event)
