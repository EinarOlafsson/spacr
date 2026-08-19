"""Report timer starts attempted from the wrong Qt thread.

Qt normally reports only that a timer cannot start from another thread, without
naming the responsible object or call site. This module wraps the timer entry
points and logs a Python stack when that condition occurs. The recorded stacks
can also be attached to a crash report if the process exits before the log is
reviewed.

Each timer start adds one thread-affinity comparison. If the Qt entry points
cannot be wrapped safely, the guard leaves them unchanged.
"""

from __future__ import annotations

import logging
import threading
import traceback

LOG = logging.getLogger("spacr.qt.thread_guard")

__all__ = ["install", "offences", "born_off_thread"]

#: Every off-thread start seen this session, as formatted stacks. Kept so a
#: crash report can carry them even if the log rotated.
_OFFENCES: list = []

_INSTALLED = False

#: Stacks of QObjects constructed off the GUI thread, capped.
_BORN_OFF_THREAD: list = []
_BORN_LIMIT = 20


def offences() -> list:
    """Copies of the stacks recorded so far."""
    return list(_OFFENCES)


def born_off_thread() -> list:
    """Stacks where a QObject was constructed away from the GUI thread."""
    return list(_BORN_OFF_THREAD)


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

    def _report(what: str, why: str) -> None:
        stack = "".join(traceback.format_stack()[:-2])
        _OFFENCES.append(stack)
        LOG.warning(
            "%s is illegal here: %s (python thread %r). The timer WILL NOT "
            "START, so whatever it drives is now dead. Stack:\n%s",
            what, why, threading.current_thread().name, stack)

    real_timer_start = QTimer.start
    real_object_start = QObject.startTimer

    def _wrong_thread(obj) -> str:
        """Why this start is illegal, or "" when it is fine.

        ASKS EXACTLY WHAT Qt ASKS: is the object's own thread the thread
        calling? Qt refuses whenever they differ, which happens both ways
        round -- a worker touching a GUI object, and the GUI thread touching a
        WORKER-AFFINE object, the second being far easier to write by accident
        because the code reads as ordinary GUI-thread code.

        COMPARED WITH `==`, NOT `is`. `QThread.currentThread()` hands back a
        fresh Python wrapper around the same underlying QThread on each call,
        so an identity test reports every ordinary start as illegal -- which is
        precisely what the first version of this did, flagging a plain
        GUI-thread `timer.start()` as "the caller is Qt mainThread, not the GUI
        thread". A guard that cries wolf on the common path is worse than none,
        because the one line that matters is then buried.

        There is deliberately no "is the caller the GUI thread" fallback. It
        added nothing the affinity test does not already cover, and it was the
        half that misfired.
        """
        try:
            owner = obj.thread()
        except Exception:                                        # noqa: BLE001
            return ""
        if owner is None:
            return ""
        current = QThread.currentThread()
        if owner == current:
            return ""
        return f"the object lives on {owner!r} and the caller is {current!r}"

    def guarded_timer_start(self, *args, **kwargs):
        why = _wrong_thread(self)
        if why:
            _report("QTimer.start", why)
        return real_timer_start(self, *args, **kwargs)

    def guarded_object_start(self, *args, **kwargs):
        why = _wrong_thread(self)
        if why:
            _report("QObject.startTimer", why)
        return real_object_start(self, *args, **kwargs)

    # AND WHERE A QObject IS BORN, because the timer warning names neither
    # the object nor the code that made it, and the Python-level wrappers
    # above never fired for the crash being chased: the start was in C++.
    #
    # A QObject constructed on a worker LIVES on that worker. Every later
    # touch from the GUI thread is then illegal -- Qt says so about the one
    # case it can detect (a timer) and says nothing about the rest, which is
    # how the process comes to segfault somewhere entirely unrelated, in an
    # event filter one time and inside pandas' CSV parser the next.
    #
    # Constructing one off-thread is not always wrong, so this REPORTS and
    # never refuses, and it stops after a handful so a legitimate producer
    # cannot flood the log.
    real_object_init = QObject.__init__

    def guarded_object_init(self, *args, **kwargs):
        result = real_object_init(self, *args, **kwargs)
        try:
            if (threading.current_thread() is not threading.main_thread()
                    and len(_BORN_OFF_THREAD) < _BORN_LIMIT):
                stack = "".join(traceback.format_stack()[:-1])
                _BORN_OFF_THREAD.append(stack)
                LOG.warning(
                    "%s was CONSTRUCTED on %r, so it lives there and every "
                    "later touch from the GUI thread is illegal. Stack:\n%s",
                    type(self).__name__, threading.current_thread().name,
                    stack)
        except Exception:                                    # noqa: BLE001
            pass
        return result

    QObject.__init__ = guarded_object_init
    QTimer.start = guarded_timer_start
    QObject.startTimer = guarded_object_start
    _INSTALLED = True
    LOG.info("thread guard installed on the GUI thread")
    return True
