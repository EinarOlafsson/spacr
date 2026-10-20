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
        from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer
    except Exception:                                            # noqa: BLE001
        return False

    def _report(what: str, why: str) -> None:
        """Record and log one cross-thread offence, with its stack.

        The stack is trimmed of this function and its caller, so the top frame
        is the code that started the timer rather than the guard.
        """
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

        Makes the same test Qt makes: is the object's own thread the calling
        thread? Qt refuses whenever they differ, which happens both ways
        round -- a worker touching a GUI object, and the GUI thread touching a
        worker-affine object; the second is easier to write by accident
        because the code reads as ordinary GUI-thread code.

        The threads are compared with `==`, not `is`. `QThread.currentThread()`
        returns a new Python wrapper around the same underlying QThread on each
        call, so an identity test would report every ordinary start, such as a
        plain GUI-thread `timer.start()`, as illegal and bury the real reports.

        There is deliberately no "is the caller the GUI thread" fallback. It
        added nothing the affinity test does not already cover, and it was the
        half that misfired.

        PySide6 can attach the returned QThread wrapper to the queried object.
        Restore the application's adopted thread to its application after the
        read, so collecting a temporary timer cannot delete the GUI thread.
        """
        try:
            owner = obj.thread()
        except Exception:                                        # noqa: BLE001
            return ""
        finally:
            try:
                application = QCoreApplication.instance()
                if application is not None:
                    application.thread()
            except Exception:
                pass
        if owner is None:
            return ""
        current = QThread.currentThread()
        if owner == current:
            return ""
        return f"the object lives on {owner!r} and the caller is {current!r}"

    def guarded_timer_start(self, *args, **kwargs):
        """Report a QTimer started off its own thread, then start it.

        The start is reported and then performed. The guard is a diagnostic:
        refusing the start would change behaviour under the guard and hide the
        bug it reports.
        """
        why = _wrong_thread(self)
        if why:
            _report("QTimer.start", why)
        return real_timer_start(self, *args, **kwargs)

    def guarded_object_start(self, *args, **kwargs):
        """Report a QObject timer started off its own thread, then start it."""
        why = _wrong_thread(self)
        if why:
            _report("QObject.startTimer", why)
        return real_object_start(self, *args, **kwargs)

    real_object_init = QObject.__init__

    def guarded_object_init(self, *args, **kwargs):
        """Note a QObject built off the main thread, after building it."""
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
