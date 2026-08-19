"""Say WHERE a timer was started off the GUI thread, not just that it was.

Qt prints

    QBasicTimer::start: Timers cannot be started from another thread

and stops there. The warning names no file, no function and no thread, so three
attempts at finding the caller by reading code have missed it -- and it matters
more than a warning usually would, for two reasons:

  * A TIMER STARTED OFF ITS OWN THREAD DOES NOT START. Whatever it was
    debouncing silently never happens, so the warning reports a dead feature.
  * IT ARRIVES IMMEDIATELY BEFORE A CRASH, every time, on the maintainer's
    machine. Touching QObject state from the wrong thread is exactly the kind
    of thing that corrupts Qt's internals and aborts a moment later, so the
    two are almost certainly one bug.

So this installs a wrapper that logs a STACK when it happens. It is not a
diagnostic to be enabled when someone remembers: the event is rare, arrives
during a real run, and is followed by the process dying -- there is no second
chance to turn instrumentation on.

THE COST IS ONE THREAD COMPARISON per timer start, which is a pointer compare.
Nothing is wrapped when the guard cannot be installed.
"""

from __future__ import annotations

import logging
import threading
import traceback

LOG = logging.getLogger("spacr.qt.thread_guard")

__all__ = ["install", "offences"]

#: Every off-thread start seen this session, as formatted stacks. Kept so a
#: crash report can carry them even if the log rotated.
_OFFENCES: list = []

_INSTALLED = False


def offences() -> list:
    """Copies of the stacks recorded so far."""
    return list(_OFFENCES)


def install() -> bool:
    """Wrap the timer entry points. Returns whether it took.

    Idempotent: called twice, the second call does nothing rather than
    wrapping the wrapper, which would double every log line and make the
    stacks harder to read rather than easier.
    """
    global _INSTALLED
    if _INSTALLED:
        return False
    try:
        from PySide6.QtCore import QObject, QThread, QTimer
    except Exception:                                            # noqa: BLE001
        return False

    gui_thread = QThread.currentThread()

    def _report(what: str) -> None:
        stack = "".join(traceback.format_stack()[:-2])
        _OFFENCES.append(stack)
        LOG.warning(
            "%s started from %r, not the GUI thread. The timer WILL NOT "
            "START, so whatever it drives is now dead. Stack:\n%s",
            what, threading.current_thread().name, stack)

    real_timer_start = QTimer.start
    real_object_start = QObject.startTimer

    def guarded_timer_start(self, *args, **kwargs):
        if QThread.currentThread() is not gui_thread:
            _report("QTimer.start")
        return real_timer_start(self, *args, **kwargs)

    def guarded_object_start(self, *args, **kwargs):
        if QThread.currentThread() is not gui_thread:
            _report("QObject.startTimer")
        return real_object_start(self, *args, **kwargs)

    QTimer.start = guarded_timer_start
    QObject.startTimer = guarded_object_start
    _INSTALLED = True
    LOG.info("thread guard installed on the GUI thread")
    return True
