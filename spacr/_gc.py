"""Explicit pipeline cleanup without destroying GUI objects on workers."""

import gc
import sys
import threading

_requested = threading.Event()


def collect() -> int:
    """Collect now, or request a full sweep on the GUI's next timer tick.

    Worker sweeps can destroy unrelated Qt widgets on the wrong thread.
    A running GUI therefore services worker requests through
    :func:`spacr.qt.gc_policy.collect_once`. Repeated requests coalesce.
    Headless processes and the GUI thread retain synchronous collection;
    checking for an application does not import Qt into headless jobs.

    :returns: unreachable objects collected, or zero for a deferred request.
    """
    core = sys.modules.get("PySide6.QtCore")
    if core is not None:
        app = core.QCoreApplication.instance()
        if app is not None and core.QThread.currentThread() != app.thread():
            _requested.set()
            return 0
    return gc.collect()
