"""
Keep the interface responsive while a Python-bound worker runs.

Instruction 126: "when something is running after hitting run, the theme
starts lagging."

MEASURED FIRST, because the instruction says to and because three plausible
causes want three different fixes. Offscreen, 1280x800, `blobs` backdrop with
its shading already on `ambient._FrameProducer`, timer interval 16 ms, two
worker threads:

    idle                                 median 16.00 ms   p95  16.04 ms
    numpy worker (what a run mostly is)  median 16.00 ms   p95  20.05 ms
    pure-Python worker                   median 42.42 ms   p95 118.63 ms
    pure-Python, switchinterval 0.001    median 17.74 ms   p95  48.46 ms

Three things follow, and only the third needed writing.

THE PRODUCER THREAD ALREADY FIXED THE COMMON CASE. A numpy worker costs the
animation NOTHING -- 16.00 ms against 16.00 ms idle -- because numpy releases
the interpreter lock and the GUI thread's frame is now one `drawImage`.

MOVING THE SHADING DOES NOT HELP A PURE-PYTHON WORKER, exactly as the
instruction predicted: it is the same lock, and the GUI thread cannot even
wake up to blit. 42 ms a frame is the lag that was reported.

WHAT DOES HELP IS ASKING FOR THE LOCK MORE OFTEN. Python hands the GIL over
every :func:`sys.getswitchinterval` seconds, 5 ms by default; a 60 Hz timer
needs it every 16 ms and is competing with a thread that never blocks. At
1 ms the GUI thread gets five times the chances and the median comes back to
within 11% of idle.

WHAT IT COSTS, stated because it is not free: more context switches, so a
pure-Python worker runs marginally slower. That is the right trade for an
interactive application -- the run is already the long pole, and a user
watching a frozen window cannot tell it from a hang.

SCOPED TO THE RUN, and restored after. A process-wide switch interval left at
1 ms would slow every headless `spacr-run` for a GUI that is not there.

Usage::

    from spacr.qt.gil_priority import responsive_gui

    with responsive_gui():
        payload = pipeline(settings)
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
    """Ask for the GUI's share of the interpreter lock. Idempotent, nesting.

    COUNTED, because two modules can run at once -- Mask and Measure, or a
    parameter search beside a fit -- and the first to finish must not hand
    the interval back while the second is still going.
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
    """Give it back when the last worker finishes."""
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
    """Whether any worker currently holds the claim."""
    with _LOCK:
        return _DEPTH > 0


@contextmanager
def responsive_gui():
    """Hold the claim for the length of one run, whatever it ends as.

    `finally`, so a run that raises does not leave the process at 1 ms for as
    long as it lives -- which would be a permanent tax on every later
    headless call in the same interpreter.
    """
    claim()
    try:
        yield
    finally:
        release()
