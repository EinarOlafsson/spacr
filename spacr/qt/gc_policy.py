"""Collect cyclic garbage on the GUI thread only.

WHY THIS EXISTS, MEASURED. Pressing Run in the live preview with ``cpsam``
produced, on 2026-09-01::

    QObject::killTimer: Timers cannot be stopped from another thread
    QObject::~QObject: Timers cannot be stopped from another thread
    Segmentation fault (core dumped)

Those two lines, in that order, are reproduced exactly by
``tests/qt/test_gc_runs_on_the_gui_thread.py`` with no spaCR code involved at
all: build a QObject that owns a *running* QTimer, drop it into a reference
CYCLE so only the collector can free it, then call ``gc.collect()`` from a
worker thread. The collector runs the destructor **on whichever thread it
happens to be running on**, and Qt cannot stop a timer from there.

That is the whole bug, and nothing about it is specific to the preview. CPython
runs an automatic collection whenever an allocation pushes a generation past
its threshold -- so the thread that pays for a collection is simply the thread
that allocated most recently. A Cellpose pass allocates enormously in a worker
thread, which makes that worker overwhelmingly likely to be the one that
inherits the sweep, and therefore the one that destroys some unrelated widget
the GUI thread abandoned earlier. The crash lands nowhere near the code that
caused it, is timing-dependent, and names no file: it is the shape of defect
that survives a long time.

WHAT THIS DOES. Automatic collection is switched off and driven from a QTimer
instead. That timer lives on the GUI thread, so every destructor the collector
runs is run there -- which is the thread that owns the widgets.

WHAT IT IS NOT. ``gc.disable()`` stops only *automatic* collection; an explicit
``gc.collect()`` from a worker thread still collects on that worker. This
module therefore removes the common cause rather than making the failure
impossible, and :func:`spacr.qt.thread_guard` still reports it if it recurs.

Memory is NOT left to grow: the tick below reproduces CPython's own
generational policy against the same thresholds, so collection happens at the
same frequency it otherwise would, on a different thread.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

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


def collect_once() -> int:
    """Run at most one generation's collection, CPython's own policy.

    The interpreter collects generation *n* when that generation's count
    passes its threshold, youngest first, and collecting a generation also
    clears the younger ones. Reproducing that here -- rather than calling a
    full ``gc.collect()`` on every tick -- is what keeps the cost the same as
    it was before this module existed. A full sweep every second would walk
    every live numpy array in the process.

    :returns: the generation collected, or ``-1`` when nothing was due.
    """
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
        or when Qt is unavailable -- never raises, because failing to install a
        mitigation must not be worse than the defect it mitigates.
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
        _was_enabled = gc.isenabled()
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
    if _was_enabled:
        gc.enable()
    return True
