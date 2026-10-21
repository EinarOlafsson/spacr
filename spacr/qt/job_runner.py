"""Run a callable off the GUI thread and deliver its result on the GUI thread.

Screens such as ``DbBrowserScreen``, ``PlateViewScreen`` and ``ReportScreen``
previously each had their own copy of this logic, and the copies differed: two
connected ``thread.finished`` to a *closure*, which never retires the job (see
:meth:`_retire_finished_jobs`). This module implements the rules once, for any
screen that moves work off the GUI thread.

The rules:

* ``PipelineWorker.finished`` is emitted **on the worker thread**. A closure
  connected to it is invoked there and must not touch widgets. Connecting
  the signal to a bound method of a GUI-thread QObject lets Qt queue the
  call onto the GUI thread; this runner uses ``_worker_finished`` for that
  delivery. A worker-thread slot can also relay a ``Signal`` to such a
  receiver without touching widgets itself.
* ``QThread.finished`` must be connected to a **bound method of a GUI-thread
  QObject**, never a closure. PySide6 makes the QThread itself the receiver
  for a closure, and :func:`spacr.qt.bridge.make_thread` connects
  ``thread.finished -> thread.deleteLater`` first. Slots run in connection
  order, so the DeferredDelete is posted ahead of the closure's metacall and
  Qt discards queued events for a destroyed receiver: the job is never
  retired and ``active_jobs()`` never returns to zero.
* A strong reference to both the QThread and the worker must be held until
  ``thread.finished``. A QThread garbage-collected while running aborts the
  process.
* Result delivery and thread retirement are separate. :meth:`cancel`
  clears pending deliveries immediately; a late completion then emits no
  ``job_finished`` signal. The finished-thread sweep still releases the
  thread references independently.

Threaded submissions go through ``make_thread``, so they appear in the
process-wide :func:`spacr.qt.bridge.registry` and turn the background
activity spinner (:mod:`spacr.qt.widgets.activity_spinner`). Inline mode
bypasses this registration and does not change the runner's busy state.
"""
from __future__ import annotations

import logging
import time
import weakref
from typing import Any, Callable, Dict, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from .bridge import make_thread, thread_has_stopped

__all__ = ["JobRunner"]

LOG = logging.getLogger(__name__)


def _capture(fn: Callable[[], Any], payload: Dict[str, Any]) -> None:
    """Run ``fn`` and leave its return value in ``payload``.

    ``PipelineWorker`` calls its function as ``fn(settings)`` and its
    ``finished`` signal carries only a success flag, so a job's actual result
    travels in the settings dict it was handed. Runs on the worker thread and
    touches nothing but ``payload``.
    """
    payload["result"] = fn()


#: Every JobRunner alive in this process. Weak, so a runner whose widget has
#: gone leaves no entry behind.
#:
#: WHY A REGISTRY. Qt ABORTS THE PROCESS if a running QThread is destroyed --
#: no Python exception, no traceback, nothing in the log. Each runner shuts
#: down in its own widget's `closeEvent`, which covers a widget being closed
#: and does NOT cover the application quitting with a job still in flight: the
#: widget is never closed, it is destroyed. The registry lets application
#: shutdown retire those threads before their widgets disappear.
_LIVE_RUNNERS: "weakref.WeakSet" = weakref.WeakSet()


#: The whole budget :func:`shutdown_all` may spend waiting, in milliseconds.
SHUTDOWN_BUDGET_MS = 2000


def shutdown_all(timeout_ms: int = SHUTDOWN_BUDGET_MS) -> int:
    """Cancel live runners; return how many shutdown calls completed normally.

    Call before Qt starts destroying widgets. Cancellation is requested for
    all runners before any wait, then they share one ``timeout_ms`` wait deadline. This
    bounds thread waiting, not the total application exit time. Threads
    that outlast the budget are parked by
    :func:`spacr.qt.bridge.drain_thread`, not terminated. Failed shutdown
    calls are logged, excluded from the return count, and do not stop the
    remaining runners from being processed.

    :param timeout_ms: the total wait budget across every runner.
    """
    deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
    runners = list(_LIVE_RUNNERS)
    for runner in runners:
        try:
            runner.cancel()
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not cancel a job runner", exc_info=True)
    asked = 0
    for runner in runners:
        remaining = max(0, int((deadline - time.monotonic()) * 1000))
        try:
            runner.shutdown(remaining)
            asked += 1
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not shut down a job runner", exc_info=True)
    return asked


def _forget_deliveries(runner_ref) -> None:
    """Drop a destroyed runner's pending deliveries and its busy flag.

    A worker's completion is a bound slot of the runner, which Qt drops once
    the runner is destroyed; without this the runner's Python half would keep
    reporting a pending, busy job that can never be delivered. Only Python
    state is touched, so it is safe while the C++ object is being destroyed.

    :param runner_ref: a weak reference to the runner.
    """
    runner = runner_ref()
    if runner is None:
        return
    runner._pending.clear()
    runner._busy = False


class JobRunner(QObject):
    """Run callables off the GUI thread; deliver their results on it.

    :param parent: the widget that owns the work. Kept as the runner's Qt
        parent so the runner dies with it.
    :param threaded: ``False`` runs work and its handler inline on the
        calling thread. Failure and completion signals are emitted before
        return; no worker is registered and busy state does not change.
    :param app_key: the name jobs appear under in the run registry, and so in
        the activity spinner's tooltip.
    :param user_visible: ``False`` for housekeeping the user did not start.
        Such a job still turns the activity spinner -- something IS running --
        but never claims a run banner. The usage poller submits every two
        seconds; without this Home flashes "<module> usage - running" on and
        off for as long as a module screen is open.
    """

    #: One job finished. ``True`` when it ran and its handler ran cleanly.
    job_finished = Signal(bool)
    #: A job raised. Carries a one-line message fit for a status bar.
    job_failed = Signal(str)
    #: Emitted whenever :meth:`is_busy` may have changed.
    busy_changed = Signal(bool)

    #: Internal relay: (job id, ok). Emitted from the worker thread,
    #: received on the GUI thread.
    _settled = Signal(int, bool)

    def __init__(self, parent: Optional[QObject] = None, *,
                 threaded: bool = True, app_key: str = "",
                 user_visible: bool = True) -> None:
        """Create a runner for one widget's background work.

        It registers itself the moment it exists, so a runner cannot be created
        and then missed by the quit-time drain; the registration is weak, so it
        holds nothing alive that Qt would otherwise collect.

        :param parent: the owning object, or ``None``.
        :param threaded: run jobs on a worker thread. ``False`` runs work and
            its handler on the calling thread without registering a worker
            or changing busy state.
        :param app_key: how this runner's work is named in the run registry.
        :param user_visible: whether these jobs count as runs the user started.
            Set ``False`` for housekeeping, so Home's run banner does not show
            it as a run.
        """
        super().__init__(parent)
        _LIVE_RUNNERS.add(self)
        self._threaded = bool(threaded)
        self._app_key = app_key or "loading"
        self._user_visible = bool(user_visible)
        self._jobs: Dict[int, Tuple[Any, Any]] = {}
        self._pending: Dict[int, Tuple[Dict[str, Any], Callable, int]] = {}
        self._next_id = 0
        #: Bumped by :meth:`cancel`. A result whose generation is stale is
        #: retired but never handed to its handler.
        self._generation = 0
        self._busy = False
        self._settled.connect(self._on_settled)
        self.destroyed.connect(
            lambda *_args, runner=weakref.ref(self): _forget_deliveries(runner))


    def submit(self, fn: Callable[[], Any],
               on_done: Optional[Callable[[Any], None]] = None) -> bool:
        """Submit ``fn()`` and deliver its successful result to ``on_done``.

        :param fn: a zero-argument callable. In threaded mode it runs off
            the GUI thread and must not touch widgets. Inline mode runs it
            on the calling thread.
        :param on_done: optional handler for ``fn``'s return value, called
            on the GUI thread in threaded mode or inline otherwise.
            Cancellation discards pending threaded deliveries.
        :returns: ``True`` after starting a threaded job, before its outcome
            is known. Inline mode returns ``False`` if either callable raises
            an exception; otherwise it returns ``True``.
        """
        if not self._threaded:
            ok = True
            try:
                result = fn()
            except Exception as exc:
                self._fail(exc)
                ok = False
            else:
                if on_done is not None:
                    try:
                        on_done(result)
                    except Exception as exc:
                        self._fail(exc)
                        ok = False
            self.job_finished.emit(ok)
            return ok

        self._next_id += 1
        job_id = self._next_id
        box: Dict[str, Any] = {}
        thread, worker = make_thread(
            lambda payload, _fn=fn: _capture(_fn, payload), box,
            app_key=self._app_key, journal=False,
            user_visible=self._user_visible, capture_figures=False)
        self._jobs[job_id] = (thread, worker)
        self._pending[job_id] = (box, on_done, self._generation)
        worker.error.connect(self._on_worker_error_text)
        worker._spacr_job_id = job_id
        worker.finished.connect(self._worker_finished)
        thread.finished.connect(self._retire_finished_jobs)
        self._set_busy(True)
        thread.start()
        return True


    def _worker_finished(self, ok: bool) -> None:
        """Settle the job whose worker just finished. Always on the GUI thread.

        A bound method of this runner rather than a lambda: Qt then queues
        the call onto the runner's thread and drops the connection when the
        runner is destroyed, so a parked worker that finishes after its
        screen has gone reaches nothing, instead of a callable that calls
        into a deleted runner from the worker thread.

        :param ok: whether the job's function returned normally.
        """
        worker = self.sender()
        job_id = getattr(worker, "_spacr_job_id", None)
        if job_id is None:
            return
        self._on_settled(job_id, ok)

    def _relay(self, job_id: int, ok: bool) -> None:
        """Re-emit a worker-thread completion as a GUI-thread call.

        Runs **on the worker thread**. Emitting a Signal is the only thing
        it may safely do there.

        The guard is not defensive noise. A screen can be closed while a
        worker is still running: ``shutdown`` asks the thread to stop and
        waits a bounded time, and a job that outlasts the budget is parked
        (see :func:`spacr.qt.bridge.drain_thread`) rather than terminated
        mid-write. That parked worker will finish eventually, and by then
        this runner's C++ half has gone with its parent widget -- PySide6
        raises ``RuntimeError: Signal source has been deleted``. Unguarded
        it surfaces as an unhandled exception in the Qt event loop, which
        pytest-qt turns into a failure in whatever test runs next.
        """
        try:
            self._settled.emit(job_id, bool(ok))
        except RuntimeError:
            self._pending.pop(job_id, None)
            self._busy = bool(self._pending)

    def _on_settled(self, job_id: int, ok: bool) -> None:
        """Finish one job by id. Always on the GUI thread."""
        entry = self._pending.pop(job_id, None)
        if entry is None:
            return
        box, on_done, generation = entry
        ok = bool(ok)
        if ok and on_done is not None and generation == self._generation:
            try:
                on_done(box.get("result"))
            except Exception as exc:
                self._fail(exc)
                ok = False
        self._set_busy(bool(self._pending))
        self.job_finished.emit(ok)

    def _retire_finished_jobs(self) -> None:
        """Release the refs of every job whose QThread has stopped.

        A sweep rather than "retire the sender", because the sender is
        exactly what may already be gone: ``make_thread`` queues
        ``thread.deleteLater`` off the same signal, and ``QObject.sender()``
        is null for a queued call whose emitter was destroyed.
        """
        for job_id, entry in list(self._jobs.items()):
            if thread_has_stopped(entry[0]):
                self._jobs.pop(job_id, None)

    def _on_worker_error_text(self, text: str) -> None:
        """Report a worker failure given as raw text.

        :param text: the worker's error output. The last non-blank line is what
            is shown -- for a traceback that is the exception itself -- and an
            entirely blank one still reports something rather than nothing.
        """
        line = ""
        for candidate in reversed(str(text).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        self.job_failed.emit(line or "unknown error")

    def _fail(self, exc: Exception) -> None:
        """Log and report a failed job.

        :param exc: the exception raised; its class name is used when it carries
            no message, so a bare ``KeyError`` still says something.
        """
        LOG.info("background job failed", exc_info=True)
        self.job_failed.emit(str(exc) or exc.__class__.__name__)


    def _set_busy(self, busy: bool) -> None:
        """Announce a change in whether work is in flight.

        :param busy: the new state; announced only when it actually changed, so
            a burst of jobs does not emit per job.
        """
        busy = bool(busy)
        if busy != self._busy:
            self._busy = busy
            try:
                self.busy_changed.emit(busy)
            except RuntimeError:
                return

    def is_busy(self) -> bool:
        """Whether threaded completions are pending; cancel clears this immediately.

        Inline work does not set this flag. A cancelled worker may still be
        running after this returns ``False``.
        """
        return self._busy

    def active_jobs(self) -> int:
        """Count tracked worker references, including running or not-yet-retired jobs.

        Finished threads remain counted until the retirement sweep. Shutdown
        clears this count even when a thread has been parked elsewhere.
        """
        return len(self._jobs)

    def pending_jobs(self) -> int:
        """Count pending threaded completion records, cleared immediately by cancel."""
        return len(self._pending)


    def cancel(self) -> None:
        """Abandon the results of everything in flight.

        Request thread interruption without waiting, then clear pending
        deliveries and busy state immediately. Late completions are ignored
        and emit no ``job_finished`` signal. Worker references remain until
        retirement or shutdown; interruption does not forcibly stop work.
        A worker's error signal can still emit ``job_failed``.
        """
        self._generation += 1
        for thread, _worker in list(self._jobs.values()):
            try:
                thread.requestInterruption()
            except RuntimeError:
                pass
        self._pending.clear()
        self._set_busy(False)

    def shutdown(self, timeout_ms: int = 3000) -> None:
        """Cancel deliveries, then wait within a shared budget or park live threads.

        Call from ``closeEvent``. Qt aborts the process if a running QThread is
        destroyed, so a bounded wait is needed when a screen closes mid-load.
        ``timeout_ms`` is shared by every thread of this runner. Threads that
        outlast it are parked by
        :func:`spacr.qt.bridge.drain_thread` rather than terminated.
        """
        from .bridge import drain_thread

        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        self.cancel()
        for thread, worker in list(self._jobs.values()):
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            try:
                drain_thread(thread, worker, timeout_ms=remaining)
            except RuntimeError:
                pass
        self._jobs.clear()
