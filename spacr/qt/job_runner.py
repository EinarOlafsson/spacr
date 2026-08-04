"""One correct way to run a callable off the GUI thread.

The Qt layer had eleven copies of this object -- ``DbBrowserScreen._run_job``,
``PlateViewScreen._run_job``, ``ReportScreen._start_job`` and so on -- and the
copies did not agree. Two of them wired ``thread.finished`` to a *closure*,
which silently never retires the job (see :meth:`_retire_finished_jobs` for
why), and that bug reached production twice. This module is the shape those
screens converged on, written down once, so a screen that needs to stop
blocking the GUI thread does not have to re-derive the rules.

The rules, which are not obvious and are all load-bearing:

* ``PipelineWorker.finished`` is emitted **on the worker thread**. A closure
  connected to it is invoked there. Touching a widget from it is undefined
  behaviour. The only safe thing a slot on that signal may do is re-emit a
  ``Signal`` whose receiver is a bound method of a GUI-thread object -- Qt
  then queues the call onto the GUI thread.
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
* The completion handler must run for **every** job, including one whose
  result is no longer wanted, or the bookkeeping leaks and the screen is
  permanently "busy". Whether the *result* is used is a separate decision,
  taken from the generation counter -- see :meth:`cancel`.

Everything submitted here goes through ``make_thread``, so it appears in the
process-wide :func:`spacr.qt.bridge.registry` and turns the background
activity spinner (:mod:`spacr.qt.widgets.activity_spinner`) without the
caller doing anything.
"""
from __future__ import annotations

import logging
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


class JobRunner(QObject):
    """Run callables off the GUI thread; deliver their results on it.

    :param parent: the widget that owns the work. Kept as the runner's Qt
        parent so the runner dies with it.
    :param threaded: ``False`` runs every job inline, emitting the same
        signals in the same order, so a test can drive a screen
        synchronously without the behaviour diverging.
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
        super().__init__(parent)
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

    # -- submitting -------------------------------------------------------

    def submit(self, fn: Callable[[], Any],
               on_done: Optional[Callable[[Any], None]] = None) -> bool:
        """Run ``fn()`` off the GUI thread, then ``on_done(result)`` on it.

        :param fn: a zero-argument callable. It runs on a worker thread and
            **must not touch any widget** -- return the data instead.
        :param on_done: called on the GUI thread with ``fn``'s return value.
        :returns: True when the job was started (or, unthreaded, ran).
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
        # journal=False: this is read-only UI housekeeping, not an analysis
        # run. A reproducibility record for "the user opened a table" is
        # noise, and `RunRegistry.cancel_all` treats journalled jobs as a
        # reason to refuse to close the application.
        thread, worker = make_thread(
            lambda payload, _fn=fn: _capture(_fn, payload), box,
            app_key=self._app_key, journal=False,
            user_visible=self._user_visible)
        # Strong references. PySide6 does not keep the worker alive through
        # the started->run connection alone, and a collected worker means the
        # thread spins forever without ever calling run().
        self._jobs[job_id] = (thread, worker)
        self._pending[job_id] = (box, on_done, self._generation)
        worker.error.connect(self._on_worker_error_text)
        # A closure -- deliberately. `worker.finished` is emitted on the
        # WORKER thread, and all this one does is call `_relay`, which
        # re-emits a Signal; emitting is safe from any thread. The Signal's
        # receiver (`_on_settled`) is a bound method of this GUI-thread
        # object, so Qt queues the real work back onto the GUI thread.
        worker.finished.connect(
            lambda ok, jid=job_id: self._relay(jid, ok))
        # A BOUND METHOD -- and the contrast with the line above is the whole
        # point. See the module docstring.
        thread.finished.connect(self._retire_finished_jobs)
        self._set_busy(True)
        thread.start()
        return True

    # -- completion -------------------------------------------------------

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
            pass

    def _on_settled(self, job_id: int, ok: bool) -> None:
        """Finish one job by id. Always on the GUI thread."""
        entry = self._pending.pop(job_id, None)
        if entry is None:
            return
        box, on_done, generation = entry
        ok = bool(ok)
        # Bookkeeping happens for every job; only *use* of the result is
        # conditional. A cancelled load that skipped this would leave the
        # runner permanently busy.
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
        line = ""
        for candidate in reversed(str(text).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        self.job_failed.emit(line or "unknown error")

    def _fail(self, exc: Exception) -> None:
        LOG.info("background job failed", exc_info=True)
        self.job_failed.emit(str(exc) or exc.__class__.__name__)

    # -- state ------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        busy = bool(busy)
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def is_busy(self) -> bool:
        """True while a submitted job has not yet delivered its result."""
        return self._busy

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def pending_jobs(self) -> int:
        """How many results have not been delivered yet."""
        return len(self._pending)

    # -- cancelling -------------------------------------------------------

    def cancel(self) -> None:
        """Abandon the results of everything in flight.

        The threads are asked to stop and are then left to retire
        themselves; they are *not* joined, because joining on the GUI thread
        is the freeze this class exists to remove. Their results are dropped
        on arrival by the generation check, so nothing reaches a handler that
        may be about to be destroyed.
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
        """Cancel, then wait briefly so no QThread outlives the widget.

        Call from ``closeEvent``. Qt aborts the process if a running QThread
        is destroyed, so a bounded wait here is the price of leaving a screen
        mid-load. Threads that outlast the budget are parked by
        :func:`spacr.qt.bridge.drain_thread` rather than terminated.
        """
        from .bridge import drain_thread

        self.cancel()
        for thread, worker in list(self._jobs.values()):
            try:
                drain_thread(thread, worker, timeout_ms=timeout_ms)
            except RuntimeError:
                pass
        self._jobs.clear()
