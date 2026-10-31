"""Collect cyclic garbage on the GUI thread only.

Background: pressing Run in the live preview with ``cpsam`` produced, on
2026-09-01::

    QObject::killTimer: Timers cannot be stopped from another thread
    QObject::~QObject: Timers cannot be stopped from another thread
    Segmentation fault (core dumped)

Those two lines, in that order, are reproduced exactly by
``tests/qt/test_gc_runs_on_the_gui_thread.py`` with no spaCR code involved at
all: build a QObject that owns a *running* QTimer, drop it into a reference
CYCLE so only the collector can free it, then call ``gc.collect()`` from a
worker thread. The collector runs the destructor **on whichever thread it
happens to be running on**, and Qt cannot stop a timer from there.

The defect is not specific to the preview. CPython runs an automatic
collection whenever an allocation pushes a generation past its threshold, so
the collection runs on the thread that allocated most recently. A Cellpose
pass allocates heavily in a worker thread, so that worker is the likely
thread to run the collection and therefore to destroy unrelated widgets the
GUI thread released earlier. The resulting crash is timing-dependent, occurs
away from the code that created the cycle, and names no file.

Automatic collection is switched off and driven from a QTimer instead. That
timer lives on the GUI thread, so every destructor the collector runs is run
there -- which is the thread that owns the widgets.

Explicit pipeline cleanup uses :func:`spacr._gc.collect`, which requests a
full sweep on the next GUI timer tick. ``gc.disable()`` stops only automatic
collection, so third-party code calling the standard-library ``gc.collect``
directly can still collect on a worker. :func:`spacr.qt.thread_guard` reports
wrong-thread widget destruction if it recurs.

Memory is still collected: each tick applies CPython's own generational
policy against the same thresholds, so collections happen about as often as
they otherwise would, checked every ``INTERVAL_MS`` milliseconds on the
GUI thread.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import sys
import threading
import time
from typing import Optional

from .._gc import _requested

LOG = logging.getLogger("spacr.qt.gc_policy")

__all__ = ["install", "uninstall", "collect_once", "is_installed"]

#: How often the GUI thread checks whether a collection is due. Short enough
#: that the counts never run far past their thresholds, long enough to be
#: invisible next to the event loop's own work.
INTERVAL_MS = 1000

_timer = None
_saved_thresholds: Optional[tuple] = None
_was_enabled: bool = True


def is_installed() -> bool:
    """Whether the GUI-thread collection policy is currently in force."""
    return _timer is not None


def _a_heavy_import_is_running() -> bool:
    """Whether a background import holds :data:`spacr.qt.app.HEAVY_IMPORT_LOCK`.

    Every heavy background import freezes what it built when it finishes
    (:func:`spacr.qt.bridge._import_the_entry_module`,
    :func:`spacr.qt.app._freeze_what_survived`), so a generational sweep in
    the middle of one walks hundreds of thousands of new module objects for
    nothing, on the GUI thread. Opening a Make Masks folder starts the
    ``spacr.io`` import (torch and the rest); the timer's sweeps during it
    took 330, 83, 421 and 101 ms on a 1994 x 1994 field, the paint drops of
    lesson 14's restoration row. The automatic sweep waits until the lock is
    free; an explicit request is still serviced.
    """
    app = sys.modules.get("spacr.qt.app")
    lock = getattr(app, "HEAVY_IMPORT_LOCK", None)
    try:
        return bool(lock is not None and lock.locked())
    except Exception:
        return False


_BUSY_DEFER_LIMIT_SECONDS = 60.0
_busy_since: list = [None]


def _defer_for_busy_workers() -> bool:
    """Whether to skip this automatic sweep because a worker is busy.

    While a pipeline or render worker holds a
    :func:`spacr.qt.gil_priority.claim`, the GUI thread already competes
    with it for the interpreter lock, and a generational sweep of what the
    worker allocated lands as one long frame: 0.36 s in lesson 26 while the
    worker was compiling with numba. The sweep waits until the last claim
    is released and runs on the next tick. A run longer than
    :data:`_BUSY_DEFER_LIMIT_SECONDS` still gets one, so a long run cannot
    grow memory without bound.
    """
    try:
        from .gil_priority import active
        busy = bool(active())
    except Exception:
        busy = False
    if not busy:
        _busy_since[0] = None
        return False
    now = time.monotonic()
    if _busy_since[0] is None:
        _busy_since[0] = now
    if now - _busy_since[0] < _BUSY_DEFER_LIMIT_SECONDS:
        return True
    _busy_since[0] = now
    return False


_PAUSE_LOCK = threading.RLock()
_PAUSE = {"depth": 0, "resume": False}

_FROZEN_WINDOWS: list = []
_FREEZE_DUE = threading.Event()


def _freeze_survivors() -> None:
    """Have the GUI-thread collector collect, then freeze what survives.

    ``gc.freeze()`` moves every tracked object into the permanent
    generation, which the collector never examines again. That is right
    for imported modules and for the screens the main window keeps, and it
    keeps later sweeps to what was made since. It is wrong for anything
    else caught by it: a dialog closed just before the freeze, whose cycles
    no tick had collected yet, or one that was open and is closed after it.
    Either kept its wrapper and every object in its reference cycles for
    the rest of the session. Measured with the main window and six module
    screens open, all fifteen Preferences dialogs opened and closed between
    two freezes stayed alive, about 1.7 MB each.

    Callable from any thread: it only marks a freeze as due. The next
    GUI-thread tick of :func:`collect_once` collects the generations not yet
    frozen -- what was made since the last freeze, not the whole heap -- then
    freezes and records the top-level windows alive at that moment. Once one
    of those windows has been deleted, a later tick returns the permanent
    generation to the collector (:func:`_reclaim_frozen`). Reclaiming the
    whole heap at every freeze instead cost a full sweep per first module
    open: offscreen, opening every module in turn took 36.2 s instead of
    25.6 s, with 39 instead of 13 event-loop stalls over 250 ms.
    """
    _FREEZE_DUE.set()


def _remember_frozen_windows() -> None:
    """Keep weak references to the top-level windows alive at a freeze.

    Popups and tooltips are left out: a menu or tooltip opened for one
    gesture would otherwise make every deletion of one a full sweep.

    Runs on the GUI thread only, from :func:`collect_once`.
    """
    import weakref

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    windows = app.topLevelWidgets() if app is not None else []
    _FROZEN_WINDOWS[:] = [weakref.ref(window) for window in windows
                          if window.windowType() not in (Qt.Popup, Qt.ToolTip)]


def _a_frozen_window_was_deleted() -> bool:
    """Whether a window that was alive at the last freeze has been deleted since.

    A deleted window's wrapper, and everything in its reference cycles, is
    frozen garbage that only :func:`_reclaim_frozen` can free. A window
    that is merely hidden is still alive and needs nothing.
    """
    if not _FROZEN_WINDOWS:
        return False
    from shiboken6 import isValid

    for reference in _FROZEN_WINDOWS:
        window = reference()
        if window is None or not isValid(window):
            return True
    return False


def _collect_and_freeze() -> None:
    """Collect the unfrozen generations, then freeze the survivors.

    Runs on the GUI thread only, from :func:`collect_once`.
    """
    _FREEZE_DUE.clear()
    gc.collect()
    gc.freeze()
    _remember_frozen_windows()


def _reclaim_frozen() -> None:
    """Collect what earlier freezes kept, then freeze the survivors again.

    Runs on the GUI thread only, from :func:`collect_once`.
    """
    gc.unfreeze()
    _collect_and_freeze()


@contextlib.contextmanager
def _automatic_collection_paused():
    """Hold automatic cyclic collection off for a block, from any thread.

    Replaces the save-and-restore of ``gc.isenabled()`` that heavy imports
    did on their own: two such blocks on two threads, or one racing
    :func:`install`, could restore "enabled" after the policy had switched
    automatic collection off, and the collector then ran on whichever thread
    allocated next, the field-animation thread included. Pauses nest across
    threads; the last one out puts collection back as it was before the
    first, unless :func:`install` ran during the pause and took it over.

    :returns: a context manager; the body runs with automatic collection off.
    """
    with _PAUSE_LOCK:
        if _PAUSE["depth"] == 0:
            _PAUSE["resume"] = gc.isenabled()
            gc.disable()
        _PAUSE["depth"] += 1
    try:
        yield
    finally:
        with _PAUSE_LOCK:
            _PAUSE["depth"] -= 1
            if _PAUSE["depth"] == 0 and _PAUSE["resume"]:
                gc.enable()


def collect_once() -> int:
    """Collect the oldest due generation, or service an explicit full-cleanup
    request.

    Without an explicit cleanup request, collection is deferred during a heavy
    import or busy-worker deferral. Otherwise, generations are checked in
    order 2, 1, 0, and ``gc.collect(generation)`` is called for the first
    whose count exceeds its saved threshold (or its current threshold when
    none was saved). No collection is requested when none is due.

    Explicit worker cleanup requests take precedence and collect all
    generations, even below their normal thresholds. Next, once nothing
    defers them, come the freezes :func:`_freeze_survivors` asked for:
    if a window frozen alive has been deleted since, the frozen objects are
    returned to the collector first; then the unfrozen generations are
    collected and the survivors frozen.

    :returns: the generation collected, or ``-1`` when none is due or ordinary
        collection is deferred.
    """
    if _requested.is_set():
        _requested.clear()
        if _a_frozen_window_was_deleted():
            _reclaim_frozen()
        elif _FREEZE_DUE.is_set():
            _collect_and_freeze()
        else:
            gc.collect()
        return 2
    if _a_heavy_import_is_running() or _defer_for_busy_workers():
        return -1
    if _a_frozen_window_was_deleted():
        _reclaim_frozen()
        return 2
    if _FREEZE_DUE.is_set():
        _collect_and_freeze()
        return 2
    thresholds = _saved_thresholds or gc.get_threshold()
    counts = gc.get_count()
    for generation in (2, 1, 0):
        try:
            due = counts[generation] > thresholds[generation]
        except IndexError:                                   # pragma: no cover
            continue
        if due:
            gc.collect(generation)
            return generation
    return -1


def install(parent=None) -> bool:
    """Take cyclic collection off the worker threads.

    :param parent: a QObject to own the timer, normally the QApplication.
    :returns: whether the policy was installed. ``False`` when it already was,
        or when Qt is unavailable. Never raises; when installation fails the
        interpreter's own collection policy stays in place.
    """
    global _timer, _saved_thresholds, _was_enabled
    if _timer is not None:
        return False
    try:
        from PySide6.QtCore import QTimer
    except Exception:                                        # noqa: BLE001
        return False
    try:
        _saved_thresholds = gc.get_threshold()
        with _PAUSE_LOCK:
            _was_enabled = gc.isenabled() or (
                _PAUSE["depth"] > 0 and _PAUSE["resume"])
            _PAUSE["resume"] = False
            gc.disable()
        timer = QTimer(parent)
        timer.setInterval(INTERVAL_MS)
        timer.timeout.connect(collect_once)
        timer.start()
        _timer = timer
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not install the GUI-thread GC policy", exc_info=True)
        if _was_enabled:
            gc.enable()
        _timer = None
        return False
    LOG.info("cyclic GC now runs on the GUI thread every %d ms "
             "(thresholds %r)", INTERVAL_MS, _saved_thresholds)
    return True


def uninstall() -> bool:
    """Restore the interpreter's own collection policy.

    Present for tests and for a clean shutdown: leaving automatic collection
    off in a process that has stopped pumping the event loop would mean cycles
    are never collected at all.
    """
    global _timer, _saved_thresholds
    if _timer is None:
        return False
    try:
        _timer.stop()
        _timer.setParent(None)
    except Exception:                                        # noqa: BLE001
        pass
    _timer = None
    _saved_thresholds = None
    with _PAUSE_LOCK:
        if _PAUSE["depth"] > 0:
            _PAUSE["resume"] = _was_enabled
        elif _was_enabled:
            gc.enable()
    return True
