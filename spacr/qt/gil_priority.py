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
                # An interpreter without the knob is not a reason to fail a
                # run; it is a reason for the window to be less smooth.
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
