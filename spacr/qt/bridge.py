"""Run spaCR pipelines in Qt workers and relay their progress to the GUI.

:class:`PipelineWorker` executes a pipeline call in a ``QThread`` and routes
captured standard output and errors through ``line_ready``. The process-wide
:data:`registry` tracks active jobs for shared surfaces such as Home, while
``make_thread`` centralizes worker registration and lifecycle setup.

Stopping and pausing are cooperative. Every worker receives a
:class:`spacr.cancellation.CancellationToken`, which long-running pipelines
poll at safe boundaries. :class:`PauseGate` and :func:`checkpoint` provide the
pause protocol, but pipeline entries that do not call ``checkpoint`` report
``supports_pause=False`` and must expose Pause as unavailable.
"""
from __future__ import annotations

import atexit
import contextlib
import gc
import importlib
import io
import logging
import os
import re
import sys
import threading
import time
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal

from spacr.qt.gil_priority import _open_torch_libraries, responsive_gui
from spacr.cancellation import (
    CancellationToken,
    PipelineCancelled,
    checkpoint as cancellation_checkpoint,
    installed_token,
)

from ..figures.style import figure_style, theme_target

LOG = logging.getLogger(__name__)

#: Carries the settings problems a run was started with to whoever watches
#: (AppScreen, so a refused run can list every problem, not only the one
#: that stopped it). Never propagates, so nothing reaches the console twice.
SETTINGS_PROBLEMS_LOG = logging.getLogger("spacr.qt.settings_problems")
SETTINGS_PROBLEMS_LOG.propagate = False
SETTINGS_PROBLEMS_LOG.setLevel(logging.INFO)
SETTINGS_PROBLEMS_LOG.addHandler(logging.NullHandler())


class _StreamRedirector(io.TextIOBase):
    """A file-like object that emits every write to a queue for the UI.

    Pipeline libraries (cellpose especially) print progress WITHOUT a
    trailing newline while they set up — model download, warmup, etc.
    A pure "emit only on \\n" redirector holds those bytes hostage in
    the buffer, and to the user it looks like the app hung after
    "Starting mask…". Two mitigations here:

    1. **Chunk cap** — if the buffer grows past ``_MAX_BUF_CHARS``
       we emit it regardless of newline, then keep buffering. This
       makes long dependency-import chatter visible instead of silent.
    2. **Idle flush** — the caller can call :meth:`idle_flush` from a
       QTimer to emit whatever partial line has been sitting quiet for
       a while, so short-but-newline-less progress lines still surface.
    """

    _MAX_BUF_CHARS = 1024

    def __init__(self, on_write: Callable[[str], None]):
        """Buffer writes into lines and hand each one to ``on_write``.

        :param on_write: called with each completed line. CALLED ON WHATEVER
            THREAD WROTE -- the worker for a print, the pump thread for an
            idle flush -- so anything touching Qt widgets from here has to
            get itself onto the GUI thread.
        """
        super().__init__()
        self._buf = ""
        self._on_write = on_write
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        """Buffer written text and emit it a line at a time.

        Emitting happens OUTSIDE the lock, so a slow slot cannot block the
        thread that is writing. A buffer that grows past the cap without a
        newline is flushed anyway -- a progress bar that never emits one would
        otherwise be held until the run ended.

        :param s: the text written; coerced with :func:`str`, because a
            non-string reaching a redirected stream is a caller's bug and losing
            the output would hide it.
        :returns: how many characters were accepted, as a stream must.
        """
        if not isinstance(s, str):
            s = str(s)
        with self._lock:
            self._buf += s
            emits = []
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                emits.append(line + "\n")
            if len(self._buf) >= self._MAX_BUF_CHARS:
                emits.append(self._buf)
                self._buf = ""
        for chunk in emits:
            self._safe_emit(chunk)
        return len(s)

    def flush(self) -> None:
        """Emit whatever is buffered, newline or not."""
        with self._lock:
            pending, self._buf = self._buf, ""
        if pending:
            self._safe_emit(pending)

    def idle_flush(self) -> None:
        """Emit any pending partial line — safe to call from the pump."""
        with self._lock:
            pending, self._buf = self._buf, ""
        if pending:
            self._safe_emit(pending)

    def _safe_emit(self, s: str) -> None:
        """Hand one line to the callback, swallowing anything it raises.

        A print must never fail because something downstream of the console did.
        """
        try:
            from ..crashreport import _capture_error_line
            _capture_error_line(s, origin="bridge")
        except Exception:
            pass
        try:
            self._on_write(s)
        except Exception:
            pass


class _ThreadStreamRouter(io.TextIOBase):
    """Route writes from each worker thread to that worker's console.

    Replacing ``sys.stdout`` independently in overlapping workers is unsafe:
    the second worker saves the first worker's redirector, and whichever one
    finishes first restores a stream belonging to the other run. This single
    process-wide proxy keeps the public stream stable while selecting the
    destination by thread identity.
    """

    def __init__(self, original):
        """Wrap the real stream and route writes by thread identity.

        :param original: the stream this proxy replaces, kept so a write
            from a thread with no registered console still reaches the
            terminal. It is the FALLBACK, not a tee: a write that finds a
            target goes there instead, not as well.
        """
        super().__init__()
        self.original = original
        self._targets: Dict[int, List[_StreamRedirector]] = {}
        self._lock = threading.RLock()

    def register(self, target: _StreamRedirector) -> None:
        """Route this thread's writes to a redirector.

        Registered as a STACK per thread, so a nested run restores the outer
        one's target when it unregisters rather than clearing the route.

        :param target: the redirector to send this thread's output to.
        """
        ident = threading.get_ident()
        with self._lock:
            self._targets.setdefault(ident, []).append(target)

    def unregister(self, target: _StreamRedirector) -> None:
        """Stop routing this thread's writes to a redirector.

        :param target: the redirector to remove; one already gone is ignored,
            since teardown order is not this object's to guarantee.
        """
        ident = threading.get_ident()
        with self._lock:
            stack = self._targets.get(ident, [])
            if target in stack:
                stack.remove(target)
            if not stack:
                self._targets.pop(ident, None)

    def has_targets(self) -> bool:
        """Report whether any thread is currently routed.

        :returns: ``True`` while at least one redirector is registered.
        """
        with self._lock:
            return bool(self._targets)

    def _target(self):
        """The stream for the calling thread, or the original.

        Reads the TOP of that thread's stack, so nested redirections unwind in
        the order they were made rather than the last one winning for good.
        """
        with self._lock:
            stack = self._targets.get(threading.get_ident(), [])
            return stack[-1] if stack else self.original

    def write(self, value: str) -> int:
        """Write to whichever redirector this thread is routed to.

        :param value: the text.
        :returns: how many characters were accepted.
        """
        try:
            return self._target().write(value)
        except ValueError:
            return 0

    def flush(self) -> None:
        """Flush this thread's redirector, tolerating one that has gone away."""
        try:
            self._target().flush()
        except Exception:
            pass

    @property
    def encoding(self):
        """The wrapped stream's encoding, or ``None``.

        Delegated rather than declared: code that inspects a stream's encoding
        is asking about the real one underneath.
        """
        return getattr(self.original, "encoding", None)

    def isatty(self) -> bool:
        """Whether the wrapped stream is a terminal.

        :returns: the real stream's answer, and ``False`` for a stream that
            does not implement it -- a router is never itself a terminal.
        """
        try:
            return bool(getattr(self.original, "isatty", lambda: False)())
        except ValueError:
            return False


_STREAM_ROUTER_LOCK = threading.RLock()
_STDOUT_ROUTER: Optional[_ThreadStreamRouter] = None
_STDERR_ROUTER: Optional[_ThreadStreamRouter] = None
_KEPT_ROUTERS: Dict[str, List[_ThreadStreamRouter]] = {
    "stdout": [], "stderr": []}


def _install_stream_router(name: str) -> _ThreadStreamRouter:
    """Install the router for ``sys.<name>``, reusing one this process kept.

    Every router that has been ``sys.stdout`` or ``sys.stderr`` is kept for
    the life of the process. CPython's ``print`` holds the stream it read
    from :mod:`sys` as a borrowed reference while it writes each argument and
    separator, so a router that a finishing worker uninstalls while another
    thread is still inside ``print`` would otherwise be freed under that
    print, which then writes through freed memory. A kept router that wraps
    the current stream is put back instead of making a new one, so at most
    one is kept per wrapped stream.

    Called with :data:`_STREAM_ROUTER_LOCK` held.

    :param name: ``"stdout"`` or ``"stderr"``.
    :returns: the router now installed as that stream.
    """
    stream = getattr(sys, name)
    if isinstance(stream, _ThreadStreamRouter):
        return stream
    kept = _KEPT_ROUTERS[name]
    router = next((r for r in kept if r.original is stream), None)
    if router is None:
        router = _ThreadStreamRouter(stream)
        kept.append(router)
    setattr(sys, name, router)
    return router


def _register_worker_streams(
    target: _StreamRedirector,
) -> tuple[_ThreadStreamRouter, _ThreadStreamRouter]:
    """Install/reuse the process routers and register the calling thread."""
    global _STDOUT_ROUTER, _STDERR_ROUTER
    with _STREAM_ROUTER_LOCK:
        _STDOUT_ROUTER = _install_stream_router("stdout")
        _STDERR_ROUTER = _install_stream_router("stderr")
        _STDOUT_ROUTER.register(target)
        _STDERR_ROUTER.register(target)
        return _STDOUT_ROUTER, _STDERR_ROUTER


def _unregister_worker_streams(
    target: _StreamRedirector,
    stdout_router: _ThreadStreamRouter,
    stderr_router: _ThreadStreamRouter,
) -> None:
    """Remove the calling worker and restore original streams when idle."""
    global _STDOUT_ROUTER, _STDERR_ROUTER
    with _STREAM_ROUTER_LOCK:
        stdout_router.unregister(target)
        stderr_router.unregister(target)
        if not stdout_router.has_targets() and sys.stdout is stdout_router:
            sys.stdout = stdout_router.original
            _STDOUT_ROUTER = None
        if not stderr_router.has_targets() and sys.stderr is stderr_router:
            sys.stderr = stderr_router.original
            _STDERR_ROUTER = None


_MPL_SHOW_LOCK = threading.RLock()
_MPL_SHOW_TARGETS: Dict[int, List[Callable[..., Any]]] = {}
_MPL_ORIGINAL_SHOW: Optional[Callable[..., Any]] = None
_MPL_MODULE = None


def _matplotlib_show_router(*args, **kwargs):
    """Route ``plt.show()`` to the current thread's figure capture.

    Calls without a registered capture are discarded on worker threads so
    Matplotlib cannot start a Qt event loop outside the GUI thread. Calls on
    the main thread fall back to the original ``show`` implementation.
    """
    with _MPL_SHOW_LOCK:
        stack = _MPL_SHOW_TARGETS.get(threading.get_ident(), [])
        target = stack[-1] if stack else None
        original = _MPL_ORIGINAL_SHOW
    if target is not None:
        return target(*args, **kwargs)
    if threading.current_thread() is not threading.main_thread():
        LOG.debug("plt.show() on %s with no capture: dropped rather than "
                  "entering a Qt event loop off the GUI thread",
                  threading.current_thread().name)
        return None
    return original(*args, **kwargs) if original is not None else None


def _register_matplotlib_show(plt, target: Callable[..., Any]) -> None:
    """Route ``plt.show`` by worker thread without cross-run restoration."""
    global _MPL_ORIGINAL_SHOW, _MPL_MODULE
    with _MPL_SHOW_LOCK:
        if not _MPL_SHOW_TARGETS:
            _MPL_ORIGINAL_SHOW = plt.show
            _MPL_MODULE = plt
            plt.show = _matplotlib_show_router
        _MPL_SHOW_TARGETS.setdefault(threading.get_ident(), []).append(target)


def _unregister_matplotlib_show(target: Callable[..., Any]) -> None:
    """Remove one thread's ``show`` target, restoring the original when the last goes.

    The module-level patch is undone only when no thread is routed any
    more, and only if it is still this router that is installed -- something
    else having patched ``show`` in the meantime is not ours to revert.

    :param target: the show callable to unregister.
    """
    global _MPL_ORIGINAL_SHOW, _MPL_MODULE
    with _MPL_SHOW_LOCK:
        ident = threading.get_ident()
        stack = _MPL_SHOW_TARGETS.get(ident, [])
        if target in stack:
            stack.remove(target)
        if not stack:
            _MPL_SHOW_TARGETS.pop(ident, None)
        if not _MPL_SHOW_TARGETS and _MPL_MODULE is not None:
            if _MPL_MODULE.show is _matplotlib_show_router:
                _MPL_MODULE.show = _MPL_ORIGINAL_SHOW
            _MPL_ORIGINAL_SHOW = None
            _MPL_MODULE = None



#: Attribute a pipeline entry point sets on itself to declare that it
#: polls :func:`checkpoint` often enough for a pause to mean something.
#: See :func:`pausable`.
PAUSABLE_ATTR = "__spacr_pausable__"

#: Attribute carrying the app key an entry point belongs to, stamped by
#: :func:`resolve_pipeline_entry` so :func:`make_thread` can name the job
#: without every caller having to pass it.
APP_KEY_ATTR = "__spacr_app_key__"


class PauseGate:
    """A latch a worker thread waits on at a checkpoint while paused.

    **Why this is not connected to every pipeline.** Pausing a
    running job can only mean one of two things:

    * stop the thread wherever it happens to be — which, for spaCR,
      means possibly mid-``np.save`` of a mask (``spacr.object`` writes
      ``.npy`` without a tmp+rename) or between the several ``INSERT``\\ s
      that make up one measured field. Both leave exactly the
      half-written artefact ``spacr.resume`` exists to clean up, so this
      is not a safe pause.
    * let the pipeline reach a boundary where nothing is half-written,
      and hold it there. That requires the pipeline to *ask*, which is
      what :func:`checkpoint` is for.

    Only the second leaves no half-written output, and it needs the
    pipeline's cooperation: a gate can be *set* from the GUI thread, but
    if nothing ever *reads* it the button does nothing. Pause therefore remains disabled until an
    entry point opts in via :func:`pausable`. Stop is a separate contract:
    :mod:`spacr.cancellation` is polled by shipped workflows at durable
    boundaries and never claims to suspend an in-progress write.

    Thread-safety: :meth:`pause` / :meth:`resume` are called from the GUI
    thread; :meth:`wait_if_paused` blocks the worker thread. A
    :class:`threading.Event` holds the paused state.
    """

    def __init__(self) -> None:
        """Create the gate open, with nothing paused."""
        self._running = threading.Event()
        self._running.set()
        self._paused_since: Optional[float] = None

    def pause(self) -> None:
        """Ask the worker to stop at its next checkpoint."""
        if self._running.is_set():
            self._paused_since = time.time()
        self._running.clear()

    def resume(self) -> None:
        """Release a paused worker."""
        self._paused_since = None
        self._running.set()

    def is_paused(self) -> bool:
        """True once :meth:`pause` has been called and not yet released."""
        return not self._running.is_set()

    def paused_for(self) -> float:
        """Seconds spent waiting, or ``0.0`` when not paused."""
        since = self._paused_since
        return 0.0 if since is None else max(0.0, time.time() - since)

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """Block while paused. Returns True once running again.

        Called by pipeline code through :func:`checkpoint`; safe to call
        when not paused, where it returns immediately.
        """
        return self._running.wait(timeout)


#: Per-thread gate, installed by :meth:`PipelineWorker.run` for the
#: duration of the call. Thread-local rather than global so two
#: overlapping workers cannot pause each other.
_LOCAL = threading.local()


def current_gate() -> Optional[PauseGate]:
    """The :class:`PauseGate` for the calling thread, or ``None``."""
    return getattr(_LOCAL, "gate", None)


def checkpoint() -> None:
    """Pipeline-side cancellation/pause point. Call only where stopping is safe.

    "Safe" means: no file is half-written, no multi-table insert is
    half-done, and no child process is still working. The top of a
    per-field or per-batch loop, before that unit's first write,
    qualifies; anywhere inside the unit does not.

    A no-op when the calling thread has no gate (i.e. outside a
    :class:`PipelineWorker`), so pipeline code can call it
    unconditionally and stay importable from a plain script.
    """
    cancellation_checkpoint()
    gate = current_gate()
    if gate is not None:
        gate.wait_if_paused()


def pausable(fn: Callable) -> Callable:
    """Mark an entry point as honouring :func:`checkpoint`.

    Setting this on a function that does *not* call :func:`checkpoint`
    enables a Pause button that has no effect, so the marker is set
    explicitly rather than inferred.

    :param fn: the entry-point callable to mark; it is returned, and an object
        that cannot take attributes is returned unmarked.
    """
    try:
        setattr(fn, PAUSABLE_ATTR, True)
    except (AttributeError, TypeError):
        pass
    return fn



#: ``spacr.utils.print_progress`` writes exactly this shape, and it is
#: the only progress signal the pipelines emit. Parsing it off stdout is
#: read-only observation — it gives the Home screen a real "field 41 of
#: 96" without anything having to be threaded through the pipeline.
_PROGRESS_RE = re.compile(r"\bProgress:\s*(\d+)\s*/\s*(\d+)")

_MASK_GPU_RE = re.compile(
    r"\[mask GPUs\] (\w+) Progress: (\d+)/(\d+) archives, failed (\d+)"
    r"((?: \| GPU \S+ \w+ \d+/\d+)*)")
_MASK_GPU_WORKER_RE = re.compile(r"GPU (\S+) (\w+) (\d+)/(\d+)")
_WATCH_FOLDER_RE = re.compile(
    r"watch_folder: (\d+) analysed, (\d+) waiting, (\d+) failed")


def _watch_folder_progress(text: str) -> Optional[dict]:
    """Read the newest folder-watch count line in ``text``, if any.

    The folder watch of Make Masks prints the line whenever a count changes.

    :param text: worker output.
    :returns: ``{'done', 'waiting', 'failed'}`` counts, or None.
    """
    matches = list(_WATCH_FOLDER_RE.finditer(text or ""))
    if not matches:
        return None
    done, waiting, failed = (int(value) for value in matches[-1].groups())
    return {"done": done, "waiting": waiting, "failed": failed}


def _mask_gpu_progress(text: str) -> Optional[dict]:
    """Read the newest parallel mask progress line in ``text``, if any.

    ``spacr._mask_workers._progress_line`` writes the line; this returns the
    object role, overall ``done``/``total``/``failed`` archive counts and a
    ``workers`` list of ``(device, state, done, total)`` tuples.
    """
    matches = list(_MASK_GPU_RE.finditer(text or ""))
    if not matches:
        return None
    match = matches[-1]
    workers = [(device, state, int(done), int(total))
               for device, state, done, total
               in _MASK_GPU_WORKER_RE.findall(match.group(5))]
    return {"object_type": match.group(1), "done": int(match.group(2)),
            "total": int(match.group(3)), "failed": int(match.group(4)),
            "workers": workers}

WORKER_SETTING_KEYS = (
    "n_jobs",
    "n_workers",
    "ultrack_n_workers",
    "infection_xgb_n_jobs",
)


def worker_capacity(total: Optional[int] = None) -> int:
    """Logical CPU capacity used by the cooperative run allocator."""
    value = os.cpu_count() if total is None else total
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def available_worker_count(total: Optional[int] = None) -> int:
    """Workers a newly-started run may use, never fewer than one."""
    capacity = worker_capacity(total)
    reserved = sum(
        max(0, int(getattr(handle, "worker_count", 1)) - 1)
        for handle in registry().active()
    )
    return max(1, capacity - reserved)


def apply_worker_budget(
    settings: Dict[str, Any],
    total: Optional[int] = None,
) -> int:
    """Cap pool settings in-place and return this run's worker allocation.

    ``-1`` and ``None`` mean "all available" in the libraries used by spaCR,
    so they resolve to the current remaining budget. Explicit smaller values
    are preserved. The return value is stored on the run handle so the next
    run sees what this one actually reserved.

    :param settings: the run settings dict, modified in place; each
        ``WORKER_SETTING_KEYS`` entry present is clamped to between 1 and the
        available workers, with ``None``, ``-1`` and other non-positive values
        taking all of them and unparseable values left alone.
    """
    available = available_worker_count(total)
    allocations: List[int] = []
    for key in WORKER_SETTING_KEYS:
        if key not in settings:
            continue
        raw = settings.get(key)
        try:
            requested = int(raw) if raw is not None else available
        except (TypeError, ValueError):
            continue
        if requested <= 0:
            requested = available
        resolved = max(1, min(requested, available))
        settings[key] = resolved
        allocations.append(resolved)
    return max(allocations, default=1)


class RunHandle(QObject):
    """One in-flight job: what it is, how far along, and its pause gate.

    Lives on the GUI thread. ``progress`` is scraped from the worker's
    stdout (see :data:`_PROGRESS_RE`) rather than reported by the
    pipeline, because the pipelines have no reporting channel.

    :param app_key: which module is running. Falls back to ``"job"`` so a
        handle always has a name to show.
    :param worker: the :class:`PipelineWorker` doing the work. Its
        ``worker_count`` is read once here rather than on every update.
    :param thread: the thread the worker was moved to. Held so the handle
        can wait on it, not so it can be restarted.
    :param parent: parent object.
    """

    changed = Signal()

    def __init__(self, app_key: str, worker: "PipelineWorker",
                 thread: QThread, parent=None):
        """Wrap one running job for the run registry.

        ``blocks_shutdown`` and ``user_visible`` are read from the worker at
        construction rather than on demand: :meth:`retire` drops the worker
        reference, and the answers are still needed after that.

        :param app_key: what the job is called.
        :param worker: the pipeline worker doing the work.
        :param thread: the thread it runs on.
        :param parent: parent object, or ``None``.
        """
        super().__init__(parent)
        self.app_key = app_key or "job"
        self.worker = worker
        self.thread = thread
        self.worker_count = max(1, int(getattr(worker, "worker_count", 1)))
        self.started_at = time.time()
        #: ``(done, total)`` from the last progress line, or ``None``.
        self.progress: Optional[tuple] = None
        #: Last non-blank line the job printed — what it is doing now.
        self.last_line = ""
        #: Whether a still-running instance of this job is a reason to
        #: refuse to close the application. Cached at construction because
        #: :meth:`retire` drops ``self.worker``. See
        #: :attr:`PipelineWorker.blocks_shutdown`.
        self.blocks_shutdown = bool(
            getattr(worker, "blocks_shutdown", True))
        #: False for housekeeping the user did not start and cannot act on.
        #: Home's run banner and anything else that reports "a run is in
        #: progress" must skip these. The usage poller submits every two
        #: seconds, so without this Home flashes "<module> usage - running"
        #: on and off for as long as a module screen is open.
        self.user_visible = bool(getattr(worker, "user_visible", True))
        #: "analysis" for a run that writes results (see
        #: :attr:`blocks_shutdown`), "housekeeping" for read-only UI work.
        #: The update check refuses only while an analysis is registered.
        self.kind = "analysis" if self.blocks_shutdown else "housekeeping"
        worker.line_ready.connect(self._on_line)

    @property
    def supports_pause(self) -> bool:
        """Whether pausing this job would actually pause it."""
        return self.worker.supports_pause

    @property
    def gate(self) -> PauseGate:
        """The pause gate this run checks between steps.

        :returns: the gate.
        """
        return self.worker.gate

    def elapsed(self) -> float:
        """How long this run has been going, in seconds.

        Floored at zero: a clock adjusted backwards mid-run would otherwise
        report a negative age, which reads as a bug in the run rather than in
        the clock.

        :returns: the elapsed seconds.
        """
        return max(0.0, time.time() - self.started_at)

    def is_running(self) -> bool:
        """Whether this handle still owns a live QThread."""
        thread = self.thread
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            return False

    def request_cancel(self, reason: str = "cancelled by the user") -> None:
        """Cooperatively cancel this job and ask its thread to retire."""
        worker = self.worker
        thread = self.thread
        if worker is not None:
            worker.request_cancel(reason)
        if thread is not None:
            try:
                thread.requestInterruption()
            except RuntimeError:
                pass

    def fraction(self) -> Optional[float]:
        """Completed fraction in ``0..1``, or ``None`` when unknown."""
        if not self.progress:
            return None
        done, total = self.progress
        return None if total <= 0 else max(0.0, min(1.0, done / total))

    def retire(self) -> None:
        """The job's thread has stopped — drop out of the registry.

        Wired to ``QThread.finished`` rather than ``worker.finished``:
        the latter is emitted inside ``run()``, before the thread's event
        loop has actually stopped, and releasing the last reference at
        that moment is how this module segfaulted once already.
        """
        try:
            self.worker.line_ready.disconnect(self._on_line)
        except (RuntimeError, TypeError):
            pass
        self.worker = None
        self.thread = None
        registry().unregister(self)

    def _on_line(self, chunk: str) -> None:
        """Record the newest output line, and any progress it carries.

        :param chunk: whatever the worker just printed; blank output is ignored,
            and the line is truncated so one enormous line cannot become the
            status bar.
        """
        text = chunk.strip()
        if not text:
            return
        self.last_line = text.splitlines()[-1][:200]
        matches = list(_PROGRESS_RE.finditer(text))
        if matches:
            match = matches[-1]
            self.progress = (int(match.group(1)), int(match.group(2)))
        self.changed.emit()


class RunRegistry(QObject):
    """Track jobs created by :func:`make_thread` until their handles retire.

    Deliberately tiny: a list plus a ``changed`` signal. It exists so
    that a surface which is *not* the screen that started the job — the
    Home page — can show what spaCR is doing, without teaching every
    screen to report in.

    :param parent: parent widget.
    """

    changed = Signal()

    def __init__(self, parent=None):
        """Create the empty run registry.

        :param parent: parent object, or ``None``.
        """
        super().__init__(parent)
        self._handles: List[RunHandle] = []

    def register(self, handle: RunHandle) -> RunHandle:
        """Take ownership of a run and start reporting it.

        :param handle: the run to track.
        :returns: the same handle, for chaining.
        """
        self._handles.append(handle)
        handle.changed.connect(self.changed)
        self.changed.emit()
        return handle

    def unregister(self, handle: RunHandle) -> None:
        """Stop tracking a run and hand ownership back to Python.

        :param handle: the run to drop.
        """
        if handle in self._handles:
            self._handles.remove(handle)
            handle.setParent(None)
            self.changed.emit()

    def active(self) -> List[RunHandle]:
        """Return registered handles in registration order.

        The list includes unstarted jobs and stopped threads awaiting
        retirement; it does not filter by whether a thread is running.
        """
        return list(self._handles)

    def _of_kind(self, *kinds: str) -> List[RunHandle]:
        """Registered handles of the given kinds, in registration order.

        A handle that names no kind counts as an analysis, so an unknown job
        is never mistaken for housekeeping.

        :param kinds: "analysis", "housekeeping", "package_install",
            "model_download" or "external".
        :returns: the matching handles, started or not.
        """
        return [handle for handle in self._handles
                if getattr(handle, "kind", "analysis") in kinds]

    def is_busy(self) -> bool:
        """Whether any run is still registered.

        What the window asks before quitting.

        :returns: True while a run is tracked.
        """
        return bool(self._handles)

    def cancel_all(
        self,
        timeout_ms: int = 5000,
        reason: str = "application shutdown",
    ) -> List[RunHandle]:
        """Cancel every active job and wait up to one shared deadline.

        The return value contains workers that are still live **and whose
        survival is a reason not to close** — see
        :attr:`RunHandle.blocks_shutdown`. They are kept registered and
        strongly referenced; callers must refuse to destroy the GUI rather
        than use ``QThread.terminate()`` mid-write.

        Read-only UI housekeeping (``journal=False`` — a history refresh, a
        model scan) is cancelled and waited for exactly like everything
        else, but it never appears in the return value. It writes nothing,
        so there is no half-written artefact to protect, and a caller that
        treats it as a veto turns "the run-history list is still loading"
        into "the application cannot be closed". ``MainWindow.closeEvent``
        answers a non-empty return by refusing to close *and showing a
        modal*: on a headless run — CI, a test shard — nothing can ever
        dismiss that modal, and the process spins in its nested event loop
        with no forward progress. A housekeeping job that outlived the
        screen which started it used to be enough to trigger it.

        A handle whose ``stop_on_quit`` is false (an install that must not
        be killed half way) is left running and is not waited for.

        :param timeout_ms: total wait budget across all active threads.
        :param reason: cancellation reason recorded by each worker.
        :returns: handles that block shutdown and did not stop in the budget.
        """
        handles = self.active()
        for handle in handles:
            if getattr(handle, "stop_on_quit", True):
                handle.request_cancel(reason)
        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        for handle in handles:
            thread = handle.thread
            if thread is None or not handle.is_running():
                continue
            remaining_ms = max(
                0, int((deadline - time.monotonic()) * 1000))
            if remaining_ms <= 0:
                break
            _wait_serving_the_interface(thread, remaining_ms)
        return [
            handle for handle in handles
            if handle.is_running() and handle.blocks_shutdown
        ]

    def clear(self) -> None:
        """Drop every handle. For tests — never call this on a live app."""
        if self._handles:
            self._handles.clear()
            self.changed.emit()


_REGISTRY: Optional[RunRegistry] = None


def registry() -> RunRegistry:
    """The process-wide :class:`RunRegistry` (created on first use)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = RunRegistry()
    return _REGISTRY


class _ExternalJobHandle(QObject):
    """A job spaCR runs outside a :class:`PipelineWorker`, in the run registry.

    A child process (a model install, a queued job that is its own
    process) has no worker thread, yet the user started it and may want to
    watch or stop it. This handle carries what the Jobs window, Home and
    the quit check read from a :class:`RunHandle`: a name, progress, the
    last output line, elapsed time and a Cancel that calls ``cancel``.

    :param app_key: what the job is called.
    :param cancel: called with the reason when the user cancels, or ``None``
        when the job cannot be stopped.
    :param blocks_shutdown: whether a still-running job is a reason to
        refuse to close the window.
    :param user_visible: whether the job is shown to the user at all.
    :param stop_on_quit: whether closing spaCR cancels the job.
    :param kind: "package_install", "model_download" or "external".
    :param parent: parent object, or ``None``.
    """

    changed = Signal()

    def __init__(self, app_key: str, cancel: Optional[Callable[[str], Any]] = None,
                 *, blocks_shutdown: bool = False, user_visible: bool = True,
                 stop_on_quit: bool = True, kind: str = "external", parent=None):
        """Describe one running external job.

        :param app_key: what the job is called.
        :param cancel: the stop request, or ``None``.
        :param blocks_shutdown: whether it vetoes closing the window.
        :param user_visible: whether it is listed.
        :param stop_on_quit: whether closing spaCR cancels it.
        :param kind: what the job is, for the update check.
        :param parent: parent object, or ``None``.
        """
        super().__init__(parent)
        self.app_key = app_key or "job"
        self.kind = str(kind or "external")
        self.worker = None
        self.thread = None
        self.worker_count = 1
        self.started_at = time.time()
        self.progress: Optional[tuple] = None
        self.last_line = ""
        self.blocks_shutdown = bool(blocks_shutdown)
        self.user_visible = bool(user_visible)
        self.supports_pause = False
        self.stop_on_quit = bool(stop_on_quit)
        self.gate = PauseGate()
        self._cancel = cancel
        self._live = True

    def elapsed(self) -> float:
        """Seconds since the job started, never negative."""
        return max(0.0, time.time() - self.started_at)

    def is_running(self) -> bool:
        """Whether the job has not yet been retired."""
        return self._live

    def fraction(self) -> Optional[float]:
        """Completed fraction in ``0..1``, or ``None`` when unknown."""
        if not self.progress:
            return None
        done, total = self.progress
        return None if total <= 0 else max(0.0, min(1.0, done / total))

    def request_cancel(self, reason: str = "cancelled by the user") -> None:
        """Ask the job to stop; nothing happens when it cannot be stopped.

        :param reason: why, passed on to the stop request.
        """
        if self._cancel is not None and self._live:
            self._cancel(reason)

    def report(self, done: Optional[int] = None, total: Optional[int] = None,
               line: str = "") -> None:
        """Record progress and the newest output line, and tell the listeners.

        :param done: steps finished, with ``total``; ``None`` leaves it.
        :param total: steps in all.
        :param line: what the job is doing now; blank leaves it.
        """
        if done is not None and total:
            self.progress = (int(done), int(total))
        text = str(line or "").strip()
        if text:
            self.last_line = text.splitlines()[-1][:200]
        self.changed.emit()

    def retire(self) -> None:
        """The job has ended: drop out of the registry."""
        if not self._live:
            return
        self._live = False
        registry().unregister(self)


def _track_external_job(app_key: str,
                        cancel: Optional[Callable[[str], Any]] = None,
                        **options: Any) -> _ExternalJobHandle:
    """Register a job that runs outside :func:`make_thread` until it retires.

    The caller reports progress through :meth:`_ExternalJobHandle.report`
    and must call :meth:`_ExternalJobHandle.retire` when the job ends.

    :param app_key: what the job is called in the Jobs window.
    :param cancel: called with the reason on Cancel, or ``None``.
    :param options: ``blocks_shutdown``, ``user_visible``,
        ``stop_on_quit`` and ``kind`` for the handle.
    :returns: the registered handle.
    """
    reg = registry()
    handle = _ExternalJobHandle(app_key, cancel, parent=reg, **options)
    reg.register(handle)
    return handle


def _registered_handle(worker: Any) -> Optional["RunHandle"]:
    """The registry handle of a :func:`make_thread` worker, or ``None``.

    :param worker: the worker :func:`make_thread` returned.
    :returns: its handle while the job is registered.
    """
    if worker is None:
        return None
    for handle in registry().active():
        if getattr(handle, "worker", None) is worker:
            return handle
    return None



#: ``(thread, worker)`` pairs that outlived the widget which owned them.
#: Parking a stubborn thread here is not a leak-with-a-nice-name: it is the
#: only way to satisfy Qt's rule that a QThread must not be destroyed while
#: it is running, once the owner has decided to go away anyway. Entries are
#: released by :func:`prune_parked_threads`, which every :func:`drain_thread`
#: call runs first.
_PARKED_THREADS: List[tuple] = []
_PARKED_LOCK = threading.Lock()

#: How long the exit hook below waits for the parked threads, in milliseconds.
PARKED_EXIT_WAIT_MS = 20_000

_PARKED_EXIT_HOOK_INSTALLED = False


def wait_for_parked_threads(timeout_ms: int = PARKED_EXIT_WAIT_MS) -> int:
    """Wait for parked Qt threads within a shared time limit.

    Parked threads remain referenced until native work finishes, preventing Qt
    from destroying a running ``QThread`` during application shutdown.

    :param timeout_ms: Total waiting time, in milliseconds, shared by all
        parked threads.
    :returns: Number of threads still running when the time limit expires.
    """
    deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
    with _PARKED_LOCK:
        pairs = list(_PARKED_THREADS)
    for thread, _worker in pairs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            thread.wait(int(remaining * 1000))
        except RuntimeError:
            continue
    return prune_parked_threads()


def _drain_parked_threads_at_exit() -> None:
    """``atexit`` hook: give parked threads their last chance to finish."""
    try:
        still_running = wait_for_parked_threads()
    except KeyboardInterrupt:
        _record_shutdown_workers("wait interrupted")
        raise
    if still_running:
        detail = _record_shutdown_workers("wait expired")
        LOG.error(
            "%d worker thread(s) are still running as the process exits. "
            "Their QThread wrappers are about to be destroyed, which Qt "
            "treats as fatal. Whatever they are inside did not answer a "
            "stop request. %s", still_running, detail)


def _record_shutdown_workers(phase):
    """Record only worker state and installed public source locations, never payloads."""
    from ..crashreport import _native_context_values, _note_native_context

    with _PARKED_LOCK:
        pairs = list(_PARKED_THREADS)
    _note_native_context(action="exit waits for parked workers", shutdown_phase=phase,
                         shutdown_workers=len(pairs), shutdown_wait_ms=PARKED_EXIT_WAIT_MS)
    frames = sys._current_frames()
    package = os.path.realpath(os.path.join(os.path.dirname(__file__), "..")) + os.sep
    summaries = []
    for number, (thread, worker) in enumerate(pairs[:8], 1):
        values = {"shutdown_worker": number}
        try:
            values["shutdown_interruption"] = thread.isInterruptionRequested()
        except (AttributeError, RuntimeError):
            pass
        token = getattr(worker, "cancel_token", None)
        if token is not None:
            values["shutdown_cancelled"] = bool(token.cancelled)
        frame = frames.get(getattr(worker, "_shutdown_thread_id", None))
        while frame is not None:
            code = frame.f_code
            module = frame.f_globals.get("__name__", "")
            if (os.path.realpath(code.co_filename).startswith(package)
                    and isinstance(module, str) and module.startswith("spacr.")):
                values.update(shutdown_module=module, shutdown_function="function " + code.co_name,
                              shutdown_line=frame.f_lineno)
                break
            frame = frame.f_back
        clean = _native_context_values(values)
        _note_native_context(**clean)
        summaries.append(" ".join(f"{key}={value}" for key, value in clean.items()))
    return "Shutdown phase: " + phase + ". Public worker state: " + "; ".join(summaries)


#: The total budget for asking in-flight work to stop when the process ends,
#: in milliseconds. Threads still running after it are parked, and the parked
#: drain gives them :data:`PARKED_EXIT_WAIT_MS` more.
EXIT_STOP_BUDGET_MS = 2000


def _park_still_running(pairs) -> int:
    """Park every ``(thread, worker)`` pair whose thread is still running.

    :param pairs: ``(thread, worker)`` pairs to check.
    :returns: how many were parked.
    """
    parked = 0
    for thread, worker in pairs:
        if thread_has_stopped(thread):
            continue
        with _PARKED_LOCK:
            if not any(entry[0] is thread for entry in _PARKED_THREADS):
                _PARKED_THREADS.append((thread, worker))
                parked += 1
    return parked


def retire_live_work(timeout_ms: int = EXIT_STOP_BUDGET_MS) -> int:
    """Stop every job runner and registered run within one shared budget.

    The application's ``aboutToQuit`` drain and the interpreter's exit hook
    both end here. ``aboutToQuit`` only fires when an event loop quits, so a
    process that builds windows and then simply ends (a script, a test
    subprocess, a capture run that returns early) never sees it, and the
    QThreads its widgets started are destroyed during interpreter teardown
    while they still run. Qt treats that as fatal. Whatever cannot stop in
    the budget is parked, never terminated.

    :param timeout_ms: total wait budget across every runner and run.
    :returns: how many threads are still running afterwards.
    """
    deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0

    def remaining() -> int:
        """What is left of the shared budget, in milliseconds."""
        return max(0, int((deadline - time.monotonic()) * 1000))

    runners = sys.modules.get(__package__ + ".job_runner")
    if runners is not None:
        try:
            runners.shutdown_all(remaining())
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not shut down the job runners", exc_info=True)
    if _REGISTRY is not None:
        try:
            handles = _REGISTRY.active()
            _REGISTRY.cancel_all(timeout_ms=remaining(),
                                 reason="application exit")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not cancel the registered runs", exc_info=True)
        else:
            _park_still_running(
                (getattr(handle, "thread", None),
                 getattr(handle, "worker", None))
                for handle in handles if isinstance(handle, RunHandle))
    _stop_loose_qthreads(remaining())
    return prune_parked_threads()


def _loose_qthreads() -> List[Any]:
    """Every running QThread this process created and nothing has parked.

    A screen that subclasses QThread itself (the updater's news fetch, a
    model download, a folder scan) is neither a job runner nor a registered
    run, so the two drains above never see it. It is found here instead,
    among the Python-created QThread wrappers that are still alive. The
    GUI thread and threads Qt adopted are skipped: neither is ours to stop.
    ``type()`` rather than ``isinstance``: the heap holds lazy module proxies
    whose ``__class__`` warns or imports when asked, and this runs while the
    interpreter is on its way out.
    """
    import gc

    try:
        from shiboken6 import Shiboken
    except ImportError:
        return []
    current = QThread.currentThread()
    with _PARKED_LOCK:
        parked = [entry[0] for entry in _PARKED_THREADS]
    found: List[Any] = []
    for candidate in gc.get_objects():
        if not issubclass(type(candidate), QThread):
            continue
        try:
            if (candidate is current or candidate == current
                    or not Shiboken.isValid(candidate)
                    or not Shiboken.createdByPython(candidate)
                    or not candidate.isRunning()):
                continue
        except RuntimeError:
            continue
        if any(candidate is entry for entry in parked + found):
            continue
        found.append(candidate)
    return found


def _stop_loose_qthreads(timeout_ms: int) -> int:
    """Ask every loose QThread to stop, wait within ``timeout_ms``, park the rest.

    A thread that outlasts the budget is taken off its parent before it is
    parked: a parent widget destroyed at exit takes its QThread children
    with it whatever Python still holds, and that is the abort.

    :param timeout_ms: the shared wait budget, in milliseconds.
    :returns: how many were parked.
    """
    threads = _loose_qthreads()
    for thread in threads:
        try:
            thread.requestInterruption()
            thread.quit()
        except RuntimeError:
            pass
    deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
    stubborn = []
    for thread in threads:
        remaining = max(0, int((deadline - time.monotonic()) * 1000))
        try:
            if thread.wait(remaining):
                continue
            thread.setParent(None)
        except RuntimeError:
            continue
        stubborn.append((thread, None))
    return _park_still_running(stubborn)


def _retire_live_work_at_exit() -> None:
    """``atexit`` hook: stop live work, then wait for whatever was parked.

    Registered when this module is imported, which is after PySide6 is, so
    it runs before PySide6's own teardown destroys the Qt objects.
    """
    try:
        retire_live_work()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not retire the live work at exit", exc_info=True)
    if parked_thread_count():
        _drain_parked_threads_at_exit()


def _install_parked_exit_hook() -> None:
    """Register one retirement hook, including any later parked-thread drain."""
    global _PARKED_EXIT_HOOK_INSTALLED
    if _PARKED_EXIT_HOOK_INSTALLED:
        return
    _PARKED_EXIT_HOOK_INSTALLED = True
    atexit.register(_retire_live_work_at_exit)


def prune_parked_threads() -> int:
    """Release parked ``(thread, worker)`` pairs whose thread has exited.

    :returns: how many pairs are still parked afterwards.
    """
    with _PARKED_LOCK:
        alive = []
        for pair in _PARKED_THREADS:
            try:
                if pair[0].isRunning():
                    alive.append(pair)
            except RuntimeError:
                pass
        _PARKED_THREADS[:] = alive
        return len(_PARKED_THREADS)


def parked_thread_count() -> int:
    """How many stubborn threads are currently parked. For diagnostics."""
    with _PARKED_LOCK:
        return len(_PARKED_THREADS)


def thread_has_stopped(thread) -> bool:
    """True when ``thread`` is finished, never started, or already deleted.

    ``None`` counts as stopped, and so does a QThread whose C++ half PySide6
    has already taken away: asking it anything raises ``RuntimeError``, and
    an object that no longer exists is certainly not still running. That
    case is not exotic — ``make_thread`` wires ``thread.finished ->
    thread.deleteLater``, so any GUI pump that delivers a retirement slot
    has usually flushed the deferred delete on the way.

    :param thread: a QThread, or None.
    """
    if thread is None:
        return True
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


def prune_job_pairs(pairs, finished=None) -> List[tuple]:
    """Return the ``(thread, worker)`` pairs a screen must still hold.

    The idiom this replaces — ``[p for p in jobs if p[0].isRunning()]``
    wired to ``thread.finished`` — is wrong twice over, and the two errors
    hid each other:

    * The slot is queued onto the GUI thread, so it runs *after* the OS
      thread has gone. ``make_thread`` also wires ``thread.finished ->
      thread.deleteLater``, and the same pump that delivers this call
      flushes that deferred delete first — so ``isRunning()`` raises
      ``RuntimeError: Internal C++ object already deleted`` from inside a Qt
      slot. The list comprehension never completes and **every** pair is
      retained, forever, along with its worker.
    * ``self.sender()`` is not a way out: Qt nulls the sender of a queued
      call whose emitter has since been destroyed, which is exactly what
      ``deleteLater`` just did. A screen keyed on it silently retires
      nothing.

    So: a wrapper whose C++ half is gone is *proof* of retirement (
    ``deleteLater`` is only ever posted from ``thread.finished``), and a
    ``finished`` sender is used when Qt still offers one.

    :param pairs: the screen's ``(thread, worker)`` ownership list.
    :param finished: the QThread that just finished, when known.
    :returns: the pairs still worth holding a reference to.
    """
    kept: List[tuple] = []
    for pair in pairs:
        thread = pair[0] if pair else None
        if finished is not None and thread is finished:
            continue
        if not thread_has_stopped(thread):
            kept.append(pair)
    return kept


def emit_safely(signal, *args) -> bool:
    """Emit a Qt signal unless its C++ receiver has been destroyed.

    Parameters
    ----------
    signal : PySide6.QtCore.SignalInstance
        Bound signal to emit.
    *args
        Values passed to ``signal.emit``.

    Returns
    -------
    bool
        ``True`` when emission succeeds; ``False`` when PySide raises
        ``RuntimeError`` because the receiving object no longer exists.
    """
    try:
        signal.emit(*args)
        return True
    except RuntimeError:
        return False


def drain_thread(thread, worker=None, timeout_ms: int = 3000) -> bool:
    """Ask ``thread`` to stop, wait for it, and **never** terminate it.

    ``QThread.terminate()`` is ``pthread_cancel`` under the covers, and
    every thread in this application runs Python. A cancelled thread that
    happened to hold the GIL never releases it and the whole process stops
    making progress; one cancelled inside a Qt or PySide internal leaves a
    corrupt heap and the process dies later, somewhere unrelated. Neither
    failure names the line that caused it, which is why
    :meth:`RunRegistry.cancel_all` documents "refuse to destroy the GUI
    rather than use ``QThread.terminate()``" — this function is how a
    caller does that.

    A thread that will not stop is *parked* (see :data:`_PARKED_THREADS`)
    so that nothing drops the last reference to a running QThread — which
    is the abort this whole module is arranged around.

    :param thread: the QThread to drain. ``None`` is accepted and is a no-op.
    :param worker: object to keep alive alongside a parked thread.
    :param timeout_ms: how long to wait for the thread to exit.
    :returns: True when the thread has stopped (or was already gone).
    """
    prune_parked_threads()
    if thread is None:
        return True
    try:
        if not thread.isRunning():
            return True
    except RuntimeError:
        return True
    try:
        thread.quit()
        stopped = bool(thread.wait(max(0, int(timeout_ms))))
    except RuntimeError:
        return True
    if stopped:
        return True
    with _PARKED_LOCK:
        _PARKED_THREADS.append((thread, worker))
    _install_parked_exit_hook()
    LOG.warning(
        "A worker thread did not stop within %d ms; it is parked rather "
        "than terminated so the process is not left with a corrupt heap.",
        timeout_ms,
    )
    return False


class _SkipFigureCapture(Exception):
    """Select the no-Matplotlib path for read-only background jobs."""


class _OutputDrain(QObject):
    """Sends a worker's waiting output from the thread that built the worker.

    Lives in that thread (the GUI thread for every screen), so the chunk
    is emitted where its receivers live. A worker built where no event
    loop runs still loses nothing: the run drains its output before it
    reports that it finished.

    :param worker: the :class:`PipelineWorker` whose output this sends.
    """

    def __init__(self, worker: "PipelineWorker"):
        """Connect to ``worker``'s pending-output signal."""
        super().__init__()
        import weakref

        self._worker = weakref.ref(worker)
        worker._output_pending.connect(self._schedule, Qt.QueuedConnection)

    def _schedule(self) -> None:
        """Drain the worker one interval from now."""
        from PySide6.QtCore import QTimer

        QTimer.singleShot(
            int(PipelineWorker._OUTPUT_INTERVAL_S * 1000), self._drain)

    def _drain(self) -> None:
        """Emit what the worker has waiting, if it still exists."""
        worker = self._worker()
        if worker is not None:
            worker._drain_output()


class PipelineWorker(QObject):
    """Runs one pipeline function in its own thread.

    Signals:

    * ``line_ready(str)`` — a chunk of stdout/stderr text
    * ``finished(bool)`` — True if the function returned without an
      unhandled exception
    * ``error(str)`` — traceback string on failure
    * ``figure_ready(object)`` — a matplotlib Figure that the pipeline
      asked to show(); emitted from the worker thread so the UI slot can
      attach it.

    :param fn: the pipeline function to run. It is called on the worker
        thread, so anything it constructs belongs to that thread.
    :param settings: the settings dict handed to ``fn``.
    :param worker_count: how many workers the run is allowed, passed through
        to the pipeline rather than used here.
    :param app_key: which module is running, for the journal and for routing
        output back to the right screen.
    :param journal: whether to record the run in the journal.
    :param capture_figures: whether a figure the pipeline shows is captured
        and emitted through ``figure_ready``. False leaves matplotlib alone,
        which is what a run wants when nobody is watching it.
    """

    line_ready = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    _error_detail = Signal(str, str)
    figure_ready = Signal(object, str)
    #: Whatever the pipeline function RETURNED. For the regression that is
    #: {'results': coef_df, 'res_folder': ..., 'model': ...} -- the table the
    #: run just fitted, in memory.
    #:
    #: It used to be thrown away, so the only way to see a finished run was to
    #: guess where it had written and re-read the CSV. Guessing a path is how
    #: a screen ends up showing last month's results, or none at all.
    result_ready = Signal(object)
    _output_pending = Signal()
    _OUTPUT_INTERVAL_S = 0.05

    def __init__(
        self,
        fn: Callable[..., Any],
        settings: Dict[str, Any],
        worker_count: int = 1,
        app_key: str = "",
        journal: bool = True,
        capture_figures: bool = True,
    ):
        """Prepare to run ``fn(settings)`` in a worker thread.

        :param fn: pipeline entry point (see :func:`resolve_pipeline_entry`).
        :param settings: keyword-style dict passed as the sole argument.
        :param worker_count: worker allocation reserved in the run registry.
        :param app_key: optional explicit module name for run-history records.
        :param journal: create a reproducibility manifest. Set false only for
            read-only background UI maintenance such as refreshing history.
        :param capture_figures: intercept Matplotlib output for an analysis
            run. Read-only UI jobs disable this so polling cannot load the
            plotting stack merely by opening a screen.
        """
        super().__init__()
        self._fn = fn
        self._spacr_output_captured = True
        self._spacr_error_occurrence = ""
        self._settings = settings
        self._app_key_override = str(app_key or "")
        self._journal_enabled = bool(journal)
        self._capture_figures = bool(capture_figures)
        self.worker_count = max(1, int(worker_count))
        self.cancel_token = CancellationToken()
        self.was_cancelled = False
        #: Latch this worker waits on when the pipeline calls
        #: :func:`checkpoint`. Always present; only *effective* when
        #: :attr:`supports_pause` is True.
        self.gate = PauseGate()
        self._out_lock = threading.RLock()
        self._out_parts: List[str] = []
        self._out_last = 0.0
        self._out_scheduled = False
        self._out_drain = _OutputDrain(self)

    def _push_output(self, text: str, *, captured=False) -> None:
        """Hand worker output to ``line_ready``, joining bursts.

        Emits at once when nothing is waiting and the last emission is
        older than 50 ms; otherwise the text waits and the drain on the
        thread that built this worker sends it within one interval, so a
        run printing thousands of lines a second costs the GUI thread twenty
        appends a second rather than one per line. Every other emission of
        this worker drains first, so the order of the output is kept. The
        lock is held only to move text, never across an emission, so a
        process forked mid-run (a DataLoader worker) cannot inherit it held.

        :param text: one chunk of the run's stdout or stderr.
        """
        if not captured:
            try:
                from ..crashreport import _capture_error_line
                _capture_error_line(text, origin="bridge")
            except Exception:
                pass
        with self._out_lock:
            now = time.monotonic()
            immediate = (not self._out_parts
                         and now - self._out_last >= self._OUTPUT_INTERVAL_S)
            if immediate:
                self._out_last = now
            else:
                self._out_parts.append(text)
                if self._out_scheduled:
                    return
                self._out_scheduled = True
        if immediate:
            self.line_ready.emit(text)
        else:
            self._output_pending.emit()

    def _drain_output(self) -> None:
        """Emit whatever output is waiting, as one chunk. Any thread."""
        with self._out_lock:
            self._out_scheduled = False
            if not self._out_parts:
                return
            text = "".join(self._out_parts)
            self._out_parts.clear()
            self._out_last = time.monotonic()
        self.line_ready.emit(text)

    def _say(self, text: str) -> None:
        """Emit ``text`` after any output still waiting."""
        try:
            from ..crashreport import _capture_error_line
            _capture_error_line(text, origin="bridge")
        except Exception:
            pass
        self._drain_output()
        with self._out_lock:
            self._out_last = time.monotonic()
        self.line_ready.emit(text)

    def _say_error(self, tb: str, *, exception=None) -> None:
        """Forward a classified failure and its identity after pending output."""
        self._spacr_error_occurrence = ""
        try:
            from ..crashreport import _dispatch_error, _new_error_occurrence
            self._spacr_error_occurrence = _new_error_occurrence(exception)
            _dispatch_error(tb, exception=exception, traceback_text=tb,
                            module=self.app_key, event_id=self._spacr_error_occurrence)
        except Exception:
            pass
        self._drain_output()
        self._error_detail.emit(tb, self._spacr_error_occurrence)
        self.error.emit(tb)

    def request_cancel(self, reason: str = "cancelled by the user") -> bool:
        """Request a stop at the pipeline's next declared safe boundary."""
        first = self.cancel_token.cancel(reason)
        self.gate.resume()
        return first

    @property
    def app_key(self) -> str:
        """App key this job belongs to, or ``""`` for an ad-hoc job."""
        return self._app_key_override or str(
            getattr(self._fn, APP_KEY_ATTR, "") or ""
        )

    @property
    def supports_pause(self) -> bool:
        """True only when the entry point declares :func:`pausable`.

        No shipped pipeline does, so this is False everywhere —
        see :class:`PauseGate` for why Pause stays disabled until a
        pipeline calls :func:`checkpoint`.
        """
        return bool(getattr(self._fn, PAUSABLE_ATTR, False))

    @property
    def blocks_shutdown(self) -> bool:
        """Whether this job still running is a reason not to close the app.

        True for analysis runs. They write masks, measurements and model
        files, so stopping one anywhere other than a declared safe boundary
        risks the half-written artefact :mod:`spacr.resume` exists to clean
        up, so closing waits for them.

        False for the read-only background jobs the UI runs on its own
        behalf — refreshing run history, scanning a model folder, polling a
        job queue. ``journal=False`` already means exactly that (see
        :meth:`__init__`): nothing is produced, so nothing can be left
        half-produced, and a caller that lets one veto shutdown converts a
        list refresh into an application that will not quit.
        """
        return self._journal_enabled

    def run(self) -> None:
        """Invoked by QThread.started; runs the pipeline function to completion."""
        self._shutdown_thread_id = threading.get_ident()
        if self.cancel_token.cancelled:
            self.was_cancelled = True
            self._say(
                f"Cancelled before start: {self.cancel_token.reason}\n")
            self.gate.resume()
            self.finished.emit(False)
            return
        _LOCAL.gate = self.gate
        journal_holder = [None]

        def _forward_output(text: str) -> None:
            """Emit worker text and retain warning lines in its manifest."""
            self._push_output(text, captured=True)
            run = journal_holder[0]
            if run is not None and re.search(
                r"\b(?:warning|warn)\b", text, flags=re.IGNORECASE,
            ):
                run.record_warning(text)

        redirect = _StreamRedirector(_forward_output)
        stdout_router, stderr_router = _register_worker_streams(redirect)


        capture_show = None
        try:
            if not self._capture_figures:
                raise _SkipFigureCapture
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from matplotlib._pylab_helpers import Gcf
            worker = self
            emitted_ids = set()
            fig_counter = [0]

            def _registered_figures():
                """Yield existing pyplot figures without creating one.

                ``pyplot.figure(number)`` is both a lookup and a constructor.
                The numbers below come from pyplot's process-wide registry,
                but another thread can close one between the two calls.  Read
                its manager directly so capture never recreates a figure and
                accidentally assigns it to this run.
                """
                for number in list(plt.get_fignums()):
                    manager = Gcf.get_fig_manager(number)
                    if manager is not None:
                        yield manager.canvas.figure

            preexisting_figures = tuple(_registered_figures())
            preexisting_ids = {id(fig) for fig in preexisting_figures}

            def _already_emitted(fig):
                """Return whether this figure was emitted during this run.

                Both the object identity recorded for the run and the figure
                marker must match. Requiring both avoids false matches when
                Python reuses an object ID or a figure persists across runs.
                """
                return (id(fig) in emitted_ids
                        and getattr(fig, "_spacr_emitted", False))

            def _mark_emitted(fig):
                """Record that a figure has been emitted, by id and on the figure.

                The attribute is the cross-route half of the guard and a figure that
                refuses one still gets its tile -- it just loses that half.
                """
                emitted_ids.add(id(fig))
                try:
                    fig._spacr_emitted = True
                except Exception:                              # noqa: BLE001
                    pass

            def _capture_show(*args, **kwargs):
                """Emit each new figure once, rendering it HERE on the worker thread.

                Agg's savefig touches no Qt, so the expensive part happens off the GUI
                thread, and the PNG is decoded here too
                (:func:`spacr.qt.widgets.figure_queue._decode_display_png`): the GUI
                only moves a file and wraps a picture, which is what stops it hanging
                while figures stream in.

                Figures marked ``_spacr_live_update`` are re-emitted in place instead,
                which is how the training monitor refreshes without filling the gallery
                with one snapshot per epoch.
                """
                for fig in _registered_figures():
                    if id(fig) in preexisting_ids:
                        continue
                    with figure_style(theme_target()):
                        already_emitted = _already_emitted(fig)
                        if already_emitted and not getattr(
                                fig, "_spacr_live_update", False):
                            continue
                        _mark_emitted(fig)
                        png_path = ""
                        try:
                            import tempfile
                            from .widgets.figure_queue import (
                                _decode_display_png, render_figure_to_png)
                            fig_counter[0] += 1
                            tmp = os.path.join(
                                tempfile.gettempdir(),
                                f"spacr_fig_{os.getpid()}_{fig_counter[0]}.png")
                            if render_figure_to_png(fig, tmp):
                                png_path = tmp
                                _decode_display_png(tmp)
                        except Exception:
                            png_path = ""
                        worker.figure_ready.emit(fig, png_path)
                return None

            capture_show = _capture_show
            _register_matplotlib_show(plt, capture_show)

            def _publish_figure(fig, path=""):
                """Publish one figure, unless it has already been shown.

                A picture that was saved and then plotted is ONE picture: without this
                the gallery held two tiles for one file, which `generate_ml_scores`
                produces as an ordinary sequence.
                """
                if _already_emitted(fig) and not getattr(
                        fig, "_spacr_live_update", False):
                    return
                _mark_emitted(fig)
                png_path = ""
                try:
                    import tempfile
                    from .widgets.figure_queue import (
                        _decode_display_png, render_figure_to_png)
                    fig_counter[0] += 1
                    tmp = os.path.join(
                        tempfile.gettempdir(),
                        f"spacr_fig_{os.getpid()}_{fig_counter[0]}.png")
                    if render_figure_to_png(fig, tmp):
                        png_path = tmp
                        _decode_display_png(tmp)
                except Exception:
                    png_path = ""
                worker.figure_ready.emit(fig, png_path)

            try:
                from spacr.figure_sink import set_sink
                set_sink(_publish_figure)
            except Exception:
                LOG.debug("could not install the figure sink", exc_info=True)
        except _SkipFigureCapture:
            plt = None
        except Exception:
            plt = None

        journal_context = None
        journal_run = None
        try:
            from spacr.run_journal import open_run
            if self._journal_enabled:
                if self._settings.get("hash_inputs", False):
                    self._say(
                        "Recording reproducibility input hashes…\n"
                    )
                journal_context = open_run(
                    self.app_key or getattr(self._fn, "__name__", "job"),
                    self._settings,
                )
                journal_run = journal_context.__enter__()
                journal_holder[0] = journal_run
                self._say(
                    f"Reproducibility manifest: {journal_run.dir}\n"
                )
                lock = getattr(journal_run, "_analysis_lock", None)
                if lock:
                    self._say(f"{lock.get('summary')}\n")
        except Exception as exc:
            journal_context = None
            journal_run = None
            message = (
                "WARNING: could not open reproducibility manifest: "
                f"{type(exc).__name__}: {exc}\n"
            )
            LOG.exception("Could not open run journal")
            self._say(message)

        ok = False
        try:
            with installed_token(self.cancel_token), responsive_gui():
                self.cancel_token.checkpoint()
                _open_torch_libraries()
                payload = self._fn(self._settings)
            if journal_run is not None:
                from spacr.run_journal import _raise_if_incomplete
                _raise_if_incomplete(journal_run)
            ok = True
            if payload is not None:
                try:
                    self.result_ready.emit(payload)
                except Exception:      # pragma: no cover - never fail a run
                    LOG.debug("could not deliver the pipeline result",
                              exc_info=True)
        except PipelineCancelled as exc:
            self.was_cancelled = True
            message = f"Cancelled safely: {exc}\n"
            LOG.info("Pipeline %s cancelled: %s", self.app_key or self._fn,
                     str(exc))
            if journal_run is not None:
                journal_run.set_status("cancelled")
                journal_run.record_warning(message.strip())
            self._say(message)
        except SystemExit as exc:
            ok = exc.code in (None, 0)
            if not ok:
                tb = traceback.format_exc()
                LOG.error("Pipeline exited with status %r", exc.code, exc_info=True)
                if journal_run is not None:
                    journal_run.set_status("failed")
                    journal_run.error_traceback = tb
                self._say_error(tb, exception=exc)
        except Exception as exc:
            tb = traceback.format_exc()
            LOG.exception("Pipeline worker failed")
            if journal_run is not None:
                journal_run.set_status("failed")
                journal_run.error_traceback = tb
            self._say_error(tb, exception=exc)
        except BaseException as exc:
            tb = traceback.format_exc()
            LOG.exception("Pipeline worker aborted")
            if journal_run is not None:
                journal_run.set_status("failed")
                journal_run.error_traceback = tb
            self._say_error(tb, exception=exc)
        finally:
            if journal_run is not None and ok:
                try:
                    from spacr.run_journal import _raise_if_incomplete
                    _raise_if_incomplete(journal_run)
                except Exception as exc:
                    ok = False
                    journal_run.set_status("failed")
                    journal_run.error_traceback = traceback.format_exc()
                    self._say_error(journal_run.error_traceback, exception=exc)
                else:
                    journal_run.set_status("success")
            if journal_context is not None:
                try:
                    journal_context.__exit__(None, None, None)
                except Exception as exc:
                    LOG.exception("Could not close run journal")
                    self._say(
                        "WARNING: could not finalize reproducibility "
                        f"manifest: {type(exc).__name__}: {exc}\n"
                    )
            try:
                redirect.flush()
            except Exception:
                pass
            _unregister_worker_streams(
                redirect, stdout_router, stderr_router
            )
            try:
                from spacr.figure_sink import clear_sink
                clear_sink()
            except Exception:
                LOG.debug("could not clear the figure sink", exc_info=True)
            if capture_show is not None and plt is not None:
                try:
                    _unregister_matplotlib_show(capture_show)
                except Exception:
                    pass
            self.gate.resume()
            journal_holder[0] = None
            try:
                delattr(_LOCAL, "gate")
            except AttributeError:
                pass
            self._drain_output()
            try:
                from spacr.logging_util import _drain_package_logs
                _drain_package_logs()
            except Exception:
                if sys.__stderr__ is not None:
                    try:
                        traceback.print_exc(file=sys.__stderr__)
                    except (OSError, ValueError):
                        pass
            self.finished.emit(ok)



def _tag(app_key: str, fn: Optional[Callable]) -> Optional[Callable]:
    """Stamp ``fn`` with the app key it belongs to and return it.

    :func:`make_thread` reads this back so the run registry can name the
    job. Stamping beats adding an ``app_key`` argument to
    ``make_thread`` because every screen in the package already calls
    ``make_thread`` and none of them would have been updated.

    The mapping is 1:1 — no two app keys resolve to the same callable —
    so writing the attribute onto a shared module-level function is
    unambiguous. Silently skipped for callables that reject attributes
    (builtins, ``functools.partial``), which simply leaves the job
    unnamed.
    """
    if fn is None:
        return None
    try:
        setattr(fn, APP_KEY_ATTR, app_key)
    except (AttributeError, TypeError):
        pass
    return fn


def _say_what_is_wrong_with_the_settings(app_key, fn):
    """Wrap a pipeline entry so it reports settings problems before it runs.

    ``spacr.validate.validate_settings`` knows that ``n_job`` is not a
    setting and that ``n_jobs`` is. The CLI and the batch runner both ask it.
    The GUI did not: this bridge hands the settings dict straight to the
    pipeline, so a settings CSV loaded into a screen ran with its typos
    intact and the key did nothing. That is not hypothetical -- the
    real `crop_measure_settings.csv` in use here asks for `n_job`, and a
    measure run started thirty workers while the file said four.

    It REPORTS and runs anyway rather than refusing. The panel itself cannot
    produce an unknown key, so the only source is a file the user chose to
    load, and a screen that silently declines to start would be a worse
    failure than one that says what it ignored. The CLI still refuses.

    Validation never breaks a run: anything it raises is swallowed, because
    a broken checker must not stop work the checker was only advising on.
    """
    if fn is None:
        return None

    @functools.wraps(fn)
    def run(settings=None, *args, **kwargs):
        """Run the entry point, turning a settings problem into a message."""
        errors = []
        if isinstance(settings, dict):
            try:
                from spacr.validate import (ERROR, coerce_expected_types,
                                            validate_settings)

                settings = coerce_expected_types(settings, app_key)
                found = list(validate_settings(dict(settings), app_key))
            except Exception:                                    # noqa: BLE001
                found = []
            errors = [p for p in found
                      if getattr(p, "severity", "") == "error"]
            if errors:
                SETTINGS_PROBLEMS_LOG.info(
                    "settings problems", extra={"spacr_problems": errors})
            for problem in found:
                mark = "ERROR" if problem.severity == ERROR else "WARNING"
                where = f" [{problem.setting}]" if problem.setting else ""
                print(f"[settings] {mark}{where}: {problem.message}")
                if problem.fix:
                    print(f"[settings]     {problem.fix}")
        from spacr.resource_log import _ram_guard_scope
        from spacr.crashreport import _validation_error_context
        with _ram_guard_scope(settings), _validation_error_context(app_key, errors):
            if settings is None:
                return fn(*args, **kwargs)
            return fn(settings, *args, **kwargs)

    return run


def resolve_pipeline_entry(app_key: str) -> Callable[[Dict[str, Any]], Any] | None:
    """Return the pipeline function that runs a given app, or None if the
    app is interactive-only (annotate / make_masks) or unknown.

    Each returned entry point is wrapped with
    :func:`spacr.qt.verbose_logger.log_call` so that when the user has
    "Verbose logging" enabled, every pipeline invocation emits an
    entry-and-return trace in the console. Zero cost when verbose is
    off (the wrapper is a single attribute check).

    The result is also stamped with the app key (:func:`_tag`) so the
    run registry can say *which module* is running. Note that none of
    these are stamped :func:`pausable` — see :class:`PauseGate`.

    :param app_key: the app key, e.g. ``'mask'`` or ``'measure'``; keys outside
        the built-in chain fall back to the ``entry=`` an app registered, then
        to a plugin app of that key.
    """
    from .verbose_logger import log_call

    _ENTRY_IMPORT_ERRORS.pop(app_key, None)

    def _ret(fn):
        """Tag the entry point with its app key and its settings check."""
        return _tag(app_key, _say_what_is_wrong_with_the_settings(app_key, fn))

    try:
        if app_key == "mask":
            from spacr.core import preprocess_generate_masks
            return _ret(log_call(preprocess_generate_masks))
        if app_key == "timelapse":
            from spacr.core import preprocess_generate_masks_timelapse
            return _ret(log_call(preprocess_generate_masks_timelapse))
        if app_key == "motility":
            from spacr.timelapse import automated_motility_assay
            return _ret(log_call(automated_motility_assay))
        if app_key == "measure":
            from spacr.measure import measure_crop
            return _ret(log_call(measure_crop))
        if app_key == "external_masks":
            from spacr.external_masks import prepare_external_masks
            return _ret(log_call(prepare_external_masks))
        if app_key == "illumination":
            from spacr.illumination import prepare_illumination_correction
            return _ret(log_call(prepare_illumination_correction))
        if app_key == "classify_merged":
            from spacr.classify import classify
            return _ret(log_call(classify))
        if app_key == "classify":
            from spacr.deep_spacr import deep_spacr
            return _ret(log_call(deep_spacr))
        if app_key == "umap":
            from spacr.core import generate_image_umap
            return _ret(log_call(generate_image_umap))
        if app_key == "train_cellpose":
            from spacr.submodules import train_cellpose
            return _ret(log_call(train_cellpose))
        if app_key == "cellpose_masks":
            from spacr.spacr_cellpose import identify_masks_finetune
            return _ret(log_call(identify_masks_finetune))
        if app_key == "map_barcodes":
            from spacr.sequencing import generate_barecode_mapping
            return _ret(log_call(generate_barecode_mapping))
        if app_key == "ml_analyze":
            from spacr.ml import generate_ml_scores
            return _ret(log_call(generate_ml_scores))
        if app_key == "regression":
            from spacr.ml import perform_regression
            return _ret(log_call(perform_regression))
        if app_key == "recruitment":
            from spacr.submodules import analyze_recruitment
            return _ret(log_call(analyze_recruitment))
        if app_key == "activation":
            from spacr.deep_spacr import generate_activation_map
            return _ret(log_call(generate_activation_map))
        if app_key == "foreign":
            from spacr.foreign import import_project
            return _ret(log_call(import_project))
        if app_key == "align":
            from spacr.align import align_folder
            return _ret(log_call(align_folder))
        if app_key == "ops":
            from spacr.ops_engine import run_ops
            return _ret(log_call(run_ops))
        if app_key == "spatial_masks":
            from spacr.ops_engine import run_spatial_masks
            return _ret(log_call(run_spatial_masks))
        if app_key == "spatial_map":
            from spacr.ops_engine import run_spatial_map
            return _ret(log_call(run_spatial_map))
        if app_key == "convert":
            from spacr.convert import convert_folder
            return _ret(log_call(convert_folder))
        if app_key == "invasion":
            from spacr.submodules import analyze_invasion
            return _ret(log_call(analyze_invasion))
        if app_key == "replication":
            from spacr.submodules import analyze_replication
            return _ret(log_call(analyze_replication))
        if app_key == "analyze_plaques":
            from spacr.submodules import analyze_plaques
            return _ret(log_call(analyze_plaques))
        if app_key == "barcode_qc":
            from spacr.sequencing_qc import barcode_qc
            return _ret(log_call(barcode_qc))
        if app_key == "explain_cv":
            from spacr.surrogate import run_explain_cv
            return _ret(log_call(run_explain_cv))
        if app_key == "anndata_export":
            from spacr.anndata_export import run_anndata_export
            return _ret(log_call(run_anndata_export))
        from .app import registered_entry
        registered = registered_entry(app_key)
        if registered is not None:
            return _ret(log_call(registered))
        from spacr.plugins import get_app, load_object
        plugin_app = get_app(app_key)
        if plugin_app is not None:
            entry = load_object(plugin_app.entrypoint)
            if not callable(entry):
                raise TypeError(
                    f"Plugin entry point {plugin_app.entrypoint!r} is not callable"
                )
            return _ret(log_call(entry))
    except Exception as error:
        LOG.exception("Could not resolve pipeline entry for %s", app_key)
        _ENTRY_IMPORT_ERRORS[app_key] = error
        return None
    return None


_ENTRY_IMPORT_ERRORS: Dict[str, BaseException] = {}


def _entry_import_error(app_key: str) -> Optional[str]:
    """The first line of the error that stopped ``app_key``'s pipeline from loading.

    :func:`resolve_pipeline_entry` returns ``None`` both for an app with no
    pipeline and for one whose module failed to import. The second used to
    read as "interactive-only in this Qt build" (Regression on a Mac whose
    XGBoost could not load, 2026-10-20); the Run button now shows this.

    :param app_key: the app key handed to :func:`resolve_pipeline_entry`.
    :returns: the error's first non-empty line, or ``None`` when the last
        resolution of ``app_key`` raised nothing.
    """
    error = _ENTRY_IMPORT_ERRORS.get(app_key)
    if error is None:
        return None
    return next((line.strip() for line in str(error).splitlines() if line.strip()),
                type(error).__name__)


#: The module each built-in app's entry point is imported from, as
#: :func:`resolve_pipeline_entry` imports it. Kept beside it, and checked
#: against its source by a test, because the two must name the same module.
_ENTRY_MODULES: Dict[str, str] = {
    "mask": "spacr.core",
    "timelapse": "spacr.core",
    "motility": "spacr.timelapse",
    "measure": "spacr.measure",
    "external_masks": "spacr.external_masks",
    "illumination": "spacr.illumination",
    "classify_merged": "spacr.classify",
    "classify": "spacr.deep_spacr",
    "umap": "spacr.core",
    "train_cellpose": "spacr.submodules",
    "cellpose_masks": "spacr.spacr_cellpose",
    "map_barcodes": "spacr.sequencing",
    "ml_analyze": "spacr.ml",
    "regression": "spacr.ml",
    "recruitment": "spacr.submodules",
    "activation": "spacr.deep_spacr",
    "foreign": "spacr.foreign",
    "align": "spacr.align",
    "ops": "spacr.ops_engine",
    "spatial_masks": "spacr.ops_engine",
    "spatial_map": "spacr.ops_engine",
    "convert": "spacr.convert",
    "invasion": "spacr.submodules",
    "replication": "spacr.submodules",
    "analyze_plaques": "spacr.submodules",
    "barcode_qc": "spacr.sequencing_qc",
    "explain_cv": "spacr.surrogate",
    "anndata_export": "spacr.anndata_export",
}

#: One import thread per module, kept after it ends so a module whose import
#: failed is tried off the GUI thread once and then left to
#: :func:`resolve_pipeline_entry` to fail exactly as it always has.
_ENTRY_IMPORTS: Dict[str, threading.Thread] = {}


def _module_is_imported(name: str) -> bool:
    """Whether ``name`` is in :data:`sys.modules` and finished initialising.

    A module another thread is still importing is already in
    :data:`sys.modules`; importing it from the GUI thread would wait on its
    import lock for the rest of that import.
    """
    module = sys.modules.get(name)
    if module is None:
        return False
    spec = getattr(module, "__spec__", None)
    return not getattr(spec, "_initializing", False)


#: Top-level packages no two threads may start importing at once. torchvision
#: is circular (``torchvision.ops`` and ``torchvision.models`` import each
#: other through the package), and a second thread entering it while the
#: first is still inside gets a partially initialised module: ``cannot import
#: name 'batched_nms'`` (coverage shard 1, 2026-10-20). torch, Cellpose and
#: segment-anything are the packages that bring it in.
_GATED_IMPORT_ROOTS = frozenset({"torch", "torchvision", "cellpose",
                                 "segment_anything"})
#: spaCR modules that import a gated package at module level, directly or
#: through one another. A thread is held at these too, before it holds the
#: lock of anything the background import may need next: arriving at
#: torchvision while another thread is inside it is what breaks, and waiting
#: at ``spacr.object`` is waiting before that.
_GATED_SPACR_MODULES = frozenset({
    "spacr.attribution", "spacr.core", "spacr.deep_spacr", "spacr.io",
    "spacr.ml", "spacr.object", "spacr.plot", "spacr.sequencing", "spacr.sim",
    "spacr.spacr_cellpose", "spacr.submodules", "spacr.timelapse",
    "spacr.torch_artifacts", "spacr.torch_loader", "spacr.toxo", "spacr.utils",
})
#: The longest a thread waits for a background import before importing
#: anyway, the behaviour before the gate existed.
_GATE_WAIT_S = 60.0
_BACKGROUND_IMPORTERS: Dict[int, threading.Event] = {}
_BACKGROUND_IMPORTERS_LOCK = threading.Lock()


def _held_module_lock_blocks(waiters, tid: int, blocking=None) -> bool:
    """Whether a thread in ``waiters`` waits, directly or not, on ``tid``.

    Read from the import system's own record of which thread is blocked on
    which module lock, the record it uses to detect its own deadlocks. When
    that record cannot be read the answer is ``True``, so the caller stops
    waiting rather than risk a wait that never ends.

    :param waiters: identifiers of the threads that might be waiting.
    :param tid: the identifier of the thread that would wait for them.
    :param blocking: the thread-to-module-locks record; the import system's
        own by default.
    :returns: True when waiting for ``waiters`` could deadlock.
    """
    try:
        if blocking is None:
            from importlib import _bootstrap

            blocking = _bootstrap._blocking_on
        frontier, seen = list(waiters), set()
        while frontier:
            waiter = frontier.pop()
            if waiter in seen:
                continue
            seen.add(waiter)
            entry = blocking.get(waiter)
            if entry is None:
                continue
            for lock in (list(entry) if isinstance(entry, list) else [entry]):
                owner = getattr(lock, "owner", None)
                if owner == tid:
                    return True
                if owner is not None:
                    frontier.append(owner)
    except Exception:                                        # noqa: BLE001
        return True
    return False


def _wait_for_background_imports() -> bool:
    """Wait until no other thread is in a registered background import.

    Called by :class:`_WaitingLoader` before this thread runs a gated
    module. The wait ends early when the background thread is blocked
    on a module lock this thread holds, which is the one way waiting could
    deadlock, and after :data:`_GATE_WAIT_S` in any case.

    :returns: True when every background import had finished.
    """
    tid = threading.get_ident()
    deadline = time.monotonic() + _GATE_WAIT_S
    while True:
        with _BACKGROUND_IMPORTERS_LOCK:
            others = {other: done for other, done in _BACKGROUND_IMPORTERS.items()
                      if other != tid}
        if not others:
            return True
        if _held_module_lock_blocks(others, tid):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            LOG.warning("a background import is still running after %.0f s; "
                        "importing beside it", _GATE_WAIT_S)
            return False
        next(iter(others.values())).wait(min(0.02, remaining))


class _WaitingLoader:
    """Stands in for a gated module's loader until the module is created.

    The wait cannot happen in ``find_spec``: the import system holds its
    global lock around every finder, so a finder that waits stops every
    other thread's imports, the background one included. ``create_module``
    runs under this module's own lock only, which the deadlock check in
    :func:`_wait_for_background_imports` can see. The real loader is put back
    on the spec before the module is created, so the module never sees this
    one.
    """

    def __init__(self, loader):
        """:param loader: the loader the ordinary finders found."""
        self._loader = loader

    def create_module(self, spec):
        """Wait for background imports, then create the module as usual.

        :param spec: the module's spec; its loader is restored here.
        :returns: what the real loader creates, or ``None`` for the default.
        """
        _wait_for_background_imports()
        spec.loader = self._loader
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        """Run the module with the real loader.

        :param module: the module to execute.
        """
        self._loader.exec_module(module)


class _HeavyImportGate:
    """A ``sys.meta_path`` entry that keeps a thread out of a busy import.

    While another thread is in a registered background import, a module
    under :data:`_GATED_IMPORT_ROOTS` or in :data:`_GATED_SPACR_MODULES` is
    found by the ordinary finders and handed back with a
    :class:`_WaitingLoader`, so this thread waits before it runs the module.
    With no background import running, or for any other module, it finds
    nothing and costs nothing.
    """

    @staticmethod
    def find_spec(name, path=None, target=None):
        """The ordinary spec with a waiting loader, or ``None``.

        :param name: the module being imported.
        :param path: the parent package's search path.
        :param target: the module being reloaded, if any.
        :returns: a spec whose loader waits, or ``None`` when nothing waits.
        """
        if not (name.partition(".")[0] in _GATED_IMPORT_ROOTS
                or name in _GATED_SPACR_MODULES):
            return None
        tid = threading.get_ident()
        with _BACKGROUND_IMPORTERS_LOCK:
            busy = any(other != tid for other in _BACKGROUND_IMPORTERS)
        if not busy:
            return None
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if finder is _HeavyImportGate or find is None:
                continue
            spec = find(name, path, target)
            if spec is None:
                continue
            loader = getattr(spec, "loader", None)
            if loader is not None and hasattr(loader, "exec_module"):
                spec.loader = _WaitingLoader(loader)
            return spec
        return None


def _install_the_import_gate() -> None:
    """Put :class:`_HeavyImportGate` first in ``sys.meta_path``, once."""
    if not any(finder is _HeavyImportGate for finder in sys.meta_path):
        sys.meta_path.insert(0, _HeavyImportGate)


@contextlib.contextmanager
def _importing_in_the_background():
    """Register the calling thread as a background importer while it runs.

    Taken inside :data:`spacr.qt.app.HEAVY_IMPORT_LOCK` by every background
    import of the deep-learning stack. Any other thread that starts importing
    a gated package meanwhile waits for it in :class:`_HeavyImportGate`, so
    two threads never initialise the same package at once.
    """
    _install_the_import_gate()
    tid = threading.get_ident()
    done = threading.Event()
    with _BACKGROUND_IMPORTERS_LOCK:
        _BACKGROUND_IMPORTERS[tid] = done
    try:
        yield
    finally:
        with _BACKGROUND_IMPORTERS_LOCK:
            if _BACKGROUND_IMPORTERS.get(tid) is done:
                del _BACKGROUND_IMPORTERS[tid]
        done.set()


def _import_the_entry_module(name: str) -> None:
    """Import ``name`` on the calling thread, never raising.

    Holds the application's heavy-import lock when the application is loaded,
    as the module preloader does, so the import and an OpenGL context are
    never brought up together, and shortens the interpreter's switch interval
    so the GUI thread keeps its turns. A failure is only logged:
    :func:`resolve_pipeline_entry` repeats the import and reports it.

    When the import brought torch in, its GPU driver is brought up here too.
    The first ``torch.cuda.is_available()`` of a process initialises the
    driver, and the run's start asks it on the GUI thread (the animation
    policy's GPU check); every later call answers at once.

    The cyclic collector is off for the import, and what it built is frozen
    after it, as :func:`spacr.qt.app._freeze_what_survived` does after the
    other background imports: a full collection over the 2,900 modules torch
    and the rest bring in otherwise lands on whichever thread allocates next,
    and that was the GUI thread starting the run, 100-150 ms. The freeze is
    made by the GUI-thread collector after one collection of what is not yet
    frozen, deferred while a worker is busy
    (:func:`spacr.qt.gc_policy._freeze_survivors`).

    PyTorch's libraries are mapped first with the interpreter lock released
    (:func:`spacr.qt.gil_priority._open_torch_libraries`), whether or not
    this module imports torch: the run that follows does.
    """
    app_module = sys.modules.get("spacr.qt.app")
    lock = getattr(app_module, "HEAVY_IMPORT_LOCK", None)
    from .gc_policy import _automatic_collection_paused, _freeze_survivors

    with responsive_gui(), _automatic_collection_paused():
        if lock is not None:
            lock.acquire()
        try:
            with _importing_in_the_background():
                _open_torch_libraries()
                importlib.import_module(name)
                torch = sys.modules.get("torch")
                if torch is not None:
                    torch.cuda.is_available()
        except Exception:
            LOG.debug("could not import %s off the GUI thread", name,
                      exc_info=True)
        finally:
            _freeze_survivors()
            if lock is not None:
                lock.release()


def _import_the_entry_off_the_gui_thread(
        app_key: str) -> Optional[threading.Thread]:
    """Import the module ``app_key``'s entry point comes from, on a worker.

    THE FIRST RUN OF A SESSION USED TO FREEZE THE WINDOW. Pressing Run asked
    :func:`resolve_pipeline_entry` for the entry point, and for Mask that is
    ``from spacr.core import ...``: torch, cellpose, pandas, Matplotlib and
    about 2,900 modules, 3.6 s on the GUI thread the first time and nothing
    after. A screen calls this first and resolves the entry once the thread
    has finished, so the import runs while the window keeps painting.

    :param app_key: the app being run.
    :returns: the importing thread while it is still running, or ``None``
        when there is nothing to wait for: an app with no built-in entry, a
        module already imported, or one whose import was tried before.
    """
    name = _ENTRY_MODULES.get(app_key)
    if name is None:
        return None
    thread = _ENTRY_IMPORTS.get(name)
    if thread is None:
        if _module_is_imported(name):
            return None
        thread = threading.Thread(
            target=_import_the_entry_module, args=(name,),
            name=f"spacr-entry-import-{name}", daemon=False)
        _ENTRY_IMPORTS[name] = thread
        thread.start()
    return thread if thread.is_alive() else None


def _entry_import_pending(app_key: str) -> bool:
    """Whether resolving ``app_key``'s entry point would still import.

    True for a built-in app whose module is not imported yet and has not
    been tried, and while :func:`_import_the_entry_off_the_gui_thread` is
    still importing it.

    :param app_key: the app being run.
    """
    name = _ENTRY_MODULES.get(app_key)
    if name is None:
        return False
    thread = _ENTRY_IMPORTS.get(name)
    if thread is not None:
        return thread.is_alive()
    return not _module_is_imported(name)


resolve_pipeline_entry.entry_import_pending = _entry_import_pending
resolve_pipeline_entry.import_off_the_gui_thread = (
    _import_the_entry_off_the_gui_thread)


#: Stack for a pipeline worker thread, in bytes.
#:
#: A pthread on macOS gets 512 KB by default and Qt does not raise it, so a
#: QThread starts with ~1/16th of the main thread's 8 MB. The pipeline is the
#: same code either way, and parts of it want a *lot* of stack: a classify run
#: died with SIGBUS, "Thread stack size exceeded due to excessive recursion",
#: in ``___chkstk_darwin`` under OpenBLAS's ``dgetrf_parallel``, reached from
#: ``np.linalg.inv`` at ``spacr/ml.py`` (the Mahalanobis inverse covariance).
#: The crash report put that thread's stack at 544 KB. Nothing was recursing —
#: ``chkstk`` is the probe that discovers the guard page, and the report's
#: "excessive recursion" wording is a guess macOS makes about any stack
#: overflow.
#:
#: This is address space, not memory: pages commit as they are touched, so a
#: generous number costs nothing until it is used. 64 MB is the main thread's
#: 8 MB with room for LAPACK's blocked kernels on a wide feature matrix.
#: ``SPACR_WORKER_STACK_MB`` overrides it for anyone who needs to.
WORKER_STACK_BYTES = 64 * 1024 * 1024


def _widen_worker_stack(thread: "QThread") -> None:
    """Give a worker thread a stack the pipeline can actually run in.

    Must be called before ``start()`` — Qt ignores ``setStackSize`` on a
    running thread. Wrapped: if the platform refuses the size, the thread
    keeps the default and the run proceeds exactly as it did before, which is
    the behaviour this replaces (INVARIANTS §10).
    """
    megabytes = os.environ.get("SPACR_WORKER_STACK_MB", "").strip()
    size = WORKER_STACK_BYTES
    if megabytes:
        try:
            requested = int(megabytes)
        except (TypeError, ValueError):
            requested = 0
        if requested > 0:
            size = requested * 1024 * 1024
    try:
        thread.setStackSize(size)
    except Exception:
        pass


_GUI_CALL_TIMEOUT_S = 10.0

_GUI_CALLS = []


class _GuiThreadCalls(QObject):
    """Runs a callable handed to it from any thread on the thread it lives on.

    Made on the GUI thread by :func:`_install_the_workspace_dispatcher`;
    :func:`_call_on_the_gui_thread` posts work to it through a queued signal.
    """

    _asked = Signal(object)

    def __init__(self) -> None:
        """Connect the queued delivery to :meth:`_answer`."""
        super().__init__()
        self._asked.connect(self._answer, Qt.QueuedConnection)

    def _answer(self, job) -> None:
        """Run one posted job; the job reports its own outcome."""
        job()


def _call_on_the_gui_thread(fn, timeout_s: Optional[float] = None):
    """Return ``fn()``, computed on the GUI thread.

    On the GUI thread, or with no application or relay, ``fn`` runs in
    place. From another thread it is posted to the GUI thread and this
    waits at most ``timeout_s`` (default :data:`_GUI_CALL_TIMEOUT_S`): a
    GUI thread that is itself waiting on the caller then costs a
    ``TimeoutError`` rather than a deadlock.

    :raises TimeoutError: when the GUI thread does not answer in time.
    """
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    relay = _GUI_CALLS[0] if _GUI_CALLS else None
    if (app is None or relay is None
            or QThread.currentThread() is app.thread()):
        return fn()
    box = {}
    done = threading.Event()
    try:
        relay._asked.emit(functools.partial(_run_the_posted, fn, box, done))
    except RuntimeError:
        return fn()
    wait = _GUI_CALL_TIMEOUT_S if timeout_s is None else float(timeout_s)
    if not done.wait(wait):
        raise TimeoutError("the interface did not answer in time")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _wait_serving_the_interface(thread, timeout_ms: int = 5000) -> bool:
    """Wait on the GUI thread for ``thread`` to stop, answering it meanwhile.

    A worker can need the GUI thread before it can finish: closing a run
    journal reads the workspace sections through
    :func:`_call_on_the_gui_thread`. A ``closeEvent`` that waited with
    ``thread.wait(5000)`` held the GUI thread for the whole five seconds
    while the worker waited on it, then gave up and left the thread
    running behind a widget about to be destroyed. This waits in short
    slices and runs the calls posted to the GUI thread between them, and
    nothing else, so no other event is delivered inside the close.

    :param thread: a ``QThread``; ``None`` or a deleted one counts as stopped.
    :param timeout_ms: how long to wait in all, in milliseconds.
    :returns: ``True`` when the thread has stopped.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
    while True:
        remaining = int((deadline - time.monotonic()) * 1000)
        try:
            if thread is None or thread.wait(max(0, min(20, remaining))):
                return True
        except RuntimeError:
            return True
        relay = _GUI_CALLS[0] if _GUI_CALLS else None
        if relay is not None:
            try:
                QCoreApplication.sendPostedEvents(relay, QEvent.MetaCall)
            except RuntimeError:
                pass
        if remaining <= 0:
            return thread_has_stopped(thread)


def _run_the_posted(fn, box, done) -> None:
    """Run ``fn`` on the GUI thread, record its outcome in ``box``, signal."""
    try:
        box["value"] = fn()
    except BaseException as error:                       # noqa: BLE001
        box["error"] = error
    finally:
        done.set()


def _install_the_workspace_dispatcher() -> bool:
    """Read workspace sections on the GUI thread, whoever saves them.

    Call on the GUI thread. Idempotent; returns whether this call installed
    it.
    """
    from spacr import workspace

    if not _GUI_CALLS:
        _GUI_CALLS.append(_GuiThreadCalls())
    if workspace._STATE_DISPATCHER is _call_on_the_gui_thread:
        return False
    workspace._STATE_DISPATCHER = _call_on_the_gui_thread
    return True


def _import_pyplot_without_pinning_callers() -> None:
    """Import ``matplotlib.pyplot`` and free the cycles the import leaves.

    Building pyplot's mathtext grammar raises and catches hundreds of
    pyparsing exceptions. Each traceback is a reference cycle whose frames
    reach, through ``f_back``, every frame above the import: the screen
    method that started a job and :func:`make_thread` itself, with the
    worker, thread and run handle in their locals. Left alone, those
    objects outlive their owners and are destroyed from inside a later
    garbage collection. One sweep here, through :func:`spacr._gc.collect`
    on the GUI thread and before any worker exists, frees the cycles while
    nothing Qt is held (well under a second, once per process).
    """
    import matplotlib.pyplot  # noqa: F401

    from .. import _gc

    _gc.collect()


def make_thread(
    fn: Callable[[Dict[str, Any]], Any],
    settings: Dict[str, Any],
    app_key: str = "",
    *,
    journal: bool = True,
    user_visible: bool = True,
    capture_figures: bool = True,
) -> tuple["QThread", PipelineWorker]:
    """Return ``(thread, worker)`` — the caller connects the worker's signals
    and calls ``thread.start()``.

    The worker remains Python-owned and is not connected to ``deleteLater``;
    its last strong reference releases it. The QThread itself keeps
    ``deleteLater`` because it belongs to the caller's running event loop.

    A caller must hold a strong reference to both until ``thread.finished``:
    a QThread garbage-collected while running takes the process down.

    A handle is added to :func:`registry` before this function returns,
    while the thread is still unstarted. It remains registered until
    ``thread.finished`` invokes the handle's retirement method. Registered
    handles therefore include jobs awaiting start or retirement; they are not
    a count of threads currently executing. Surfaces such as Home can use
    these handles to show jobs they did not start. Retirement is connected to
    ``thread.finished`` rather than ``worker.finished``, which is emitted
    before the thread stops.

    :param fn: callable invoked as ``fn(settings)`` on the worker thread.
    :param settings: the single argument handed to ``fn``.
    :param app_key: overrides the key stamped on ``fn`` by
        :func:`resolve_pipeline_entry`. Only needed for ad-hoc jobs that
        want to show up on Home under a name of their own.
    :param journal: create a reproducibility record. Disable only for
        read-only UI housekeeping that is not an analysis run.
    :param user_visible: mark the handle for user-facing job displays.
        False hides it from displays that filter this flag; the handle is
        still registered and this option does not cancel or stop the job.
    :param capture_figures: prepare Matplotlib figure interception. Keep true
        for analysis runs; read-only UI jobs which cannot emit figures set it
        false to preserve the operation import boundary.
    :returns: an unstarted ``(QThread, PipelineWorker)`` pair.
    """
    if capture_figures and "matplotlib.pyplot" not in sys.modules:
        from .gc_policy import _automatic_collection_paused

        try:
            with _automatic_collection_paused():
                _import_pyplot_without_pinning_callers()
        except Exception:              # noqa: BLE001
            LOG.debug("matplotlib.pyplot could not be pre-imported",
                      exc_info=True)
    thread = QThread()
    _widen_worker_stack(thread)
    allocation = apply_worker_budget(settings)
    worker = PipelineWorker(
        fn, settings, worker_count=allocation, app_key=app_key,
        journal=journal, capture_figures=capture_figures,
    )
    worker.user_visible = bool(user_visible)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    thread.finished.connect(thread.deleteLater)

    reg = registry()
    handle = RunHandle(app_key or worker.app_key, worker, thread, parent=reg)
    reg.register(handle)
    thread.finished.connect(handle.retire)
    return thread, worker


_install_parked_exit_hook()
