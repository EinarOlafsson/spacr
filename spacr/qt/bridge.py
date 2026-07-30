"""
Background execution + progress bridge between the Qt UI and the pipeline
functions in spacr.core / spacr.deep_spacr / spacr.submodules / etc.

Runs each pipeline call in a QThread so the UI stays responsive. The
worker installs stdout/stderr shims that emit `line_ready(str)` on every
print, so the caller can pipe them into a QPlainTextEdit console.

Three things live here beyond "run a function on a thread":

* :class:`PipelineWorker` — the thread body, plus the stdout capture.
* :data:`registry` — a process-wide list of the jobs that are running
  right now, so a surface like the Home screen can say what spaCR is
  doing without every screen having to report in. ``make_thread`` is the
  single choke point every screen already goes through, so registration
  happens there and nothing else has to change.
* :class:`PauseGate` / :func:`checkpoint` — **cooperative** pause. Read
  :class:`PauseGate`'s docstring before wiring a Pause button to
  anything: as of today no shipped pipeline calls :func:`checkpoint`, so
  :attr:`PipelineWorker.supports_pause` is ``False`` for every entry in
  :func:`resolve_pipeline_entry` and a Pause control must render itself
  disabled. That is deliberate, and it is asserted by the test suite.
"""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

LOG = logging.getLogger(__name__)


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
        super().__init__()
        self._buf = ""
        self._on_write = on_write
        # The worker thread writes via print(); the idle-flush pump
        # thread calls idle_flush() concurrently. Both mutate _buf, so
        # every access is guarded — an unlocked race corrupted the
        # buffer and could crash the interpreter.
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
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
        # Emit OUTSIDE the lock so a slow slot can't block the writer.
        for chunk in emits:
            self._safe_emit(chunk)
        return len(s)

    def flush(self) -> None:
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
        super().__init__()
        self.original = original
        self._targets: Dict[int, List[_StreamRedirector]] = {}
        self._lock = threading.RLock()

    def register(self, target: _StreamRedirector) -> None:
        ident = threading.get_ident()
        with self._lock:
            self._targets.setdefault(ident, []).append(target)

    def unregister(self, target: _StreamRedirector) -> None:
        ident = threading.get_ident()
        with self._lock:
            stack = self._targets.get(ident, [])
            if target in stack:
                stack.remove(target)
            if not stack:
                self._targets.pop(ident, None)

    def has_targets(self) -> bool:
        with self._lock:
            return bool(self._targets)

    def _target(self):
        with self._lock:
            stack = self._targets.get(threading.get_ident(), [])
            return stack[-1] if stack else self.original

    def write(self, value: str) -> int:
        return self._target().write(value)

    def flush(self) -> None:
        try:
            self._target().flush()
        except Exception:
            pass

    @property
    def encoding(self):
        return getattr(self.original, "encoding", None)

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())


_STREAM_ROUTER_LOCK = threading.RLock()
_STDOUT_ROUTER: Optional[_ThreadStreamRouter] = None
_STDERR_ROUTER: Optional[_ThreadStreamRouter] = None


def _register_worker_streams(
    target: _StreamRedirector,
) -> tuple[_ThreadStreamRouter, _ThreadStreamRouter]:
    """Install/reuse the process routers and register the calling thread."""
    global _STDOUT_ROUTER, _STDERR_ROUTER
    with _STREAM_ROUTER_LOCK:
        if not isinstance(sys.stdout, _ThreadStreamRouter):
            _STDOUT_ROUTER = _ThreadStreamRouter(sys.stdout)
            sys.stdout = _STDOUT_ROUTER
        else:
            _STDOUT_ROUTER = sys.stdout
        if not isinstance(sys.stderr, _ThreadStreamRouter):
            _STDERR_ROUTER = _ThreadStreamRouter(sys.stderr)
            sys.stderr = _STDERR_ROUTER
        else:
            _STDERR_ROUTER = sys.stderr
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
    with _MPL_SHOW_LOCK:
        stack = _MPL_SHOW_TARGETS.get(threading.get_ident(), [])
        target = stack[-1] if stack else _MPL_ORIGINAL_SHOW
    return target(*args, **kwargs) if target is not None else None


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


# ---------------------------------------------------------------------------
# Cooperative pause
# ---------------------------------------------------------------------------

#: Attribute a pipeline entry point sets on itself to declare that it
#: polls :func:`checkpoint` often enough for a pause to mean something.
#: See :func:`pausable`.
PAUSABLE_ATTR = "__spacr_pausable__"

#: Attribute carrying the app key an entry point belongs to, stamped by
#: :func:`resolve_pipeline_entry` so :func:`make_thread` can name the job
#: without every caller having to pass it.
APP_KEY_ATTR = "__spacr_app_key__"


class PauseGate:
    """A latch a worker thread waits on, so a pause is a *pause*.

    **Why this is not simply hooked up to every pipeline.** Pausing a
    running job can only mean one of two things:

    * stop the thread wherever it happens to be — which, for spaCR,
      means possibly mid-``np.save`` of a mask (``spacr.object`` writes
      ``.npy`` without a tmp+rename) or between the several ``INSERT``\\ s
      that make up one measured field. Both leave exactly the
      half-written artefact ``spacr.resume`` exists to clean up. That is
      not a pause, it is corruption with a friendly label.
    * let the pipeline reach a boundary where nothing is half-written,
      and hold it there. That requires the pipeline to *ask*, which is
      what :func:`checkpoint` is for.

    Only the second is honest, and it cannot be bolted on from outside:
    a gate can be *set* from the GUI thread, but if nothing ever *reads*
    it the button does nothing. spaCR's pipelines currently read nothing
    — ``isInterruptionRequested`` has no call sites in the package
    either, which is why the existing Stop button is best-effort. So the
    machinery is here, it is tested, and the UI is required to disable
    the control until a pipeline opts in via :func:`pausable`.

    Thread-safety: :meth:`pause` / :meth:`resume` are called from the GUI
    thread; :meth:`wait_if_paused` blocks the worker thread. That is the
    whole contract — a :class:`threading.Event` does the rest.
    """

    def __init__(self) -> None:
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
    """Pipeline-side pause point. **Call only where stopping is safe.**

    "Safe" means: no file is half-written, no multi-table insert is
    half-done, and no child process is still working. The top of a
    per-field or per-batch loop, before that unit's first write,
    qualifies; anywhere inside the unit does not.

    A no-op when the calling thread has no gate (i.e. outside a
    :class:`PipelineWorker`), so pipeline code can call it
    unconditionally and stay importable from a plain script.
    """
    gate = current_gate()
    if gate is not None:
        gate.wait_if_paused()


def pausable(fn: Callable) -> Callable:
    """Mark an entry point as honouring :func:`checkpoint`.

    Setting this on a function that does *not* actually call
    :func:`checkpoint` is how you ship a Pause button that lies, so the
    marker is deliberately explicit rather than inferred.
    """
    try:
        setattr(fn, PAUSABLE_ATTR, True)
    except (AttributeError, TypeError):
        pass
    return fn


# ---------------------------------------------------------------------------
# Which jobs are running right now
# ---------------------------------------------------------------------------

#: ``spacr.utils.print_progress`` writes exactly this shape, and it is
#: the only progress signal the pipelines emit. Parsing it off stdout is
#: read-only observation — it gives the Home screen a real "field 41 of
#: 96" without anything having to be threaded through the pipeline.
_PROGRESS_RE = re.compile(r"\bProgress:\s*(\d+)\s*/\s*(\d+)")

# Settings that directly control process/thread pools in shipped pipelines.
# Each is capped to the budget remaining after older active runs. A run with
# N workers consumes N-1 slots because its own QThread is already one of the
# concurrently executing units; this gives the requested sequence:
# second = total - first + 1, then each later run subtracts the extra workers
# reserved by every older run.
WORKER_SETTING_KEYS = (
    "n_jobs",
    "n_workers",
    "n_workers_features",
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
    """

    changed = Signal()

    def __init__(self, app_key: str, worker: "PipelineWorker",
                 thread: QThread, parent=None):
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
        worker.line_ready.connect(self._on_line)

    # -- state ---------------------------------------------------------
    @property
    def supports_pause(self) -> bool:
        """Whether pausing this job would actually pause it."""
        return self.worker.supports_pause

    @property
    def gate(self) -> PauseGate:
        return self.worker.gate

    def elapsed(self) -> float:
        return max(0.0, time.time() - self.started_at)

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
        text = chunk.strip()
        if not text:
            return
        self.last_line = text.splitlines()[-1][:200]
        match = _PROGRESS_RE.search(text)
        if match:
            self.progress = (int(match.group(1)), int(match.group(2)))
        self.changed.emit()


class RunRegistry(QObject):
    """Every job started through :func:`make_thread`, while it runs.

    Deliberately tiny: a list plus a ``changed`` signal. It exists so
    that a surface which is *not* the screen that started the job — the
    Home page — can show what spaCR is doing, without teaching every
    screen to report in.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._handles: List[RunHandle] = []

    def register(self, handle: RunHandle) -> RunHandle:
        self._handles.append(handle)
        handle.changed.connect(self.changed)
        self.changed.emit()
        return handle

    def unregister(self, handle: RunHandle) -> None:
        if handle in self._handles:
            self._handles.remove(handle)
            # Hand ownership back to Python. Left parented, the handle
            # (and the worker it references) would live until the
            # registry did, i.e. until the process exited.
            handle.setParent(None)
            self.changed.emit()

    def active(self) -> List[RunHandle]:
        """Handles for the jobs running right now, oldest first."""
        return list(self._handles)

    def is_busy(self) -> bool:
        return bool(self._handles)

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


class PipelineWorker(QObject):
    """Runs one pipeline function in its own thread.

    Signals:
        line_ready(str)  — a chunk of stdout/stderr text
        finished(bool)   — True if the function returned without an
                            unhandled exception
        error(str)       — traceback string on failure
        figure_ready(object)
                         — a matplotlib Figure that the pipeline
                            asked to show(); emitted from the worker
                            thread so the UI slot can attach it.
    """

    line_ready = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    figure_ready = Signal(object, str)   # (figure, prerendered_png_path or "")

    def __init__(self, fn: Callable[..., Any], settings: Dict[str, Any],
                 worker_count: int = 1):
        """Prepare to run ``fn(settings)`` in a worker thread.

        :param fn: pipeline entry point (see :func:`resolve_pipeline_entry`).
        :param settings: keyword-style dict passed as the sole argument.
        """
        super().__init__()
        self._fn = fn
        self._settings = settings
        self.worker_count = max(1, int(worker_count))
        #: Latch this worker waits on when the pipeline calls
        #: :func:`checkpoint`. Always present; only *effective* when
        #: :attr:`supports_pause` is True.
        self.gate = PauseGate()

    @property
    def app_key(self) -> str:
        """App key this job belongs to, or ``""`` for an ad-hoc job."""
        return str(getattr(self._fn, APP_KEY_ATTR, "") or "")

    @property
    def supports_pause(self) -> bool:
        """True only when the entry point declares :func:`pausable`.

        No shipped pipeline does, so this is False everywhere today —
        see :class:`PauseGate` for why that is the honest answer rather
        than a missing feature.
        """
        return bool(getattr(self._fn, PAUSABLE_ATTR, False))

    def run(self) -> None:
        """Invoked by QThread.started; runs the pipeline function to completion."""
        _LOCAL.gate = self.gate
        redirect = _StreamRedirector(self.line_ready.emit)
        stdout_router, stderr_router = _register_worker_streams(redirect)

        # NOTE: there used to be a background "idle-flush pump" daemon
        # thread here that emitted line_ready periodically. It was
        # removed — emitting a Qt signal from a non-Qt-affinity Python
        # thread delivered as a DirectConnection and mutated console
        # widgets off the GUI thread, aborting the process. The real
        # cause of "pressed Run, nothing happens" was a garbage-collected
        # worker (see AppScreen._on_run keeping self._worker), not
        # missing flushes; the redirector's 1024-char chunk-cap already
        # surfaces long newline-less bursts.

        # Intercept matplotlib show() so figures land in the UI instead
        # of a blocking Tk window. `plt.show` gets restored in `finally`.
        capture_show = None
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            worker = self
            emitted_ids = set()
            fig_counter = [0]

            def _capture_show(*args, **kwargs):
                # Emit ordinary figures only once. Figures explicitly marked
                # ``_spacr_live_update`` are re-rendered and emitted in place;
                # this is how the training monitor refreshes without filling
                # the gallery with one snapshot per epoch.
                # Render each figure to a PNG HERE, in the worker thread (Agg
                # savefig touches no Qt) — the expensive part — so the GUI
                # thread only does a cheap file-move + pixmap load and never
                # hangs while figures stream in.
                for num in list(plt.get_fignums()):
                    fig = plt.figure(num)
                    already_emitted = id(fig) in emitted_ids
                    if already_emitted and not getattr(
                            fig, "_spacr_live_update", False):
                        continue
                    emitted_ids.add(id(fig))
                    png_path = ""
                    try:
                        import tempfile
                        from .widgets.figure_queue import render_figure_to_png
                        fig_counter[0] += 1
                        tmp = os.path.join(
                            tempfile.gettempdir(),
                            f"spacr_fig_{os.getpid()}_{fig_counter[0]}.png")
                        if render_figure_to_png(fig, tmp):
                            png_path = tmp
                    except Exception:
                        png_path = ""
                    worker.figure_ready.emit(fig, png_path)
                return None

            capture_show = _capture_show
            _register_matplotlib_show(plt, capture_show)
        except Exception:
            plt = None

        ok = False
        try:
            self._fn(self._settings)
            ok = True
        except SystemExit as exc:
            # ``sys.exit()`` and ``sys.exit(0)`` are successful early exits.
            # Any other code is a failure and must reach the same error path as
            # an exception; treating ``sys.exit(1)`` as success made CLI-style
            # pipeline failures appear as a green, completed GUI run.
            ok = exc.code in (None, 0)
            if not ok:
                tb = traceback.format_exc()
                LOG.error("Pipeline exited with status %r", exc.code)
                self.error.emit(tb)
        except Exception:
            tb = traceback.format_exc()
            LOG.exception("Pipeline worker failed")
            self.error.emit(tb)
        finally:
            try:
                redirect.flush()
            except Exception:
                pass
            _unregister_worker_streams(
                redirect, stdout_router, stderr_router
            )
            if capture_show is not None and plt is not None:
                try:
                    _unregister_matplotlib_show(capture_show)
                except Exception:
                    pass
            # Release the gate before announcing completion: a worker
            # left paused would otherwise strand anything that later
            # waits on it, and the job is over either way.
            self.gate.resume()
            _LOCAL.gate = None
            self.finished.emit(ok)


# ---------------------------------------------------------------------------
# Dispatch: app_key -> function to run
# ---------------------------------------------------------------------------

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
    """
    from .verbose_logger import log_call

    def _ret(fn):
        return _tag(app_key, fn)

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
        if app_key == "classify":
            # deep_spacr, not train_test_model. The Classify screen builds its
            # panel from deep_spacr_defaults, so it SHOWS generate_training_
            # dataset, apply_model_to_dataset, n_top_examples and tar_path --
            # every one of which train_test_model ignores. Running the training
            # stage alone meant those switches were settable and silently did
            # nothing. Tk (gui_utils.run_function_gui) and validate.APP_FUNCTIONS
            # both map classify -> deep_spacr; Qt was the odd one out.
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
        if app_key == "cellpose_all":
            from spacr.spacr_cellpose import check_cellpose_models
            return _ret(log_call(check_cellpose_models))
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
    except Exception:
        LOG.exception("Could not resolve pipeline entry for %s", app_key)
        return None
    return None


def make_thread(
    fn: Callable[[Dict[str, Any]], Any],
    settings: Dict[str, Any],
    app_key: str = "",
) -> tuple["QThread", PipelineWorker]:
    """Return ``(thread, worker)`` — the caller connects the worker's signals
    and calls ``thread.start()``.

    **The worker's C++ deletion is left to Python, deliberately.** There used
    to be a ``worker.finished.connect(worker.deleteLater)`` here, and it
    segfaulted the process: ``finished`` is emitted inside ``run()``, so the
    deletion was posted to the WORKER thread's own event loop and executed
    during that thread's deferred-delete flush, at the same moment the GUI
    thread dropped the object's last Python reference in its completion
    handler. Two owners, one object. gdb put the crash in
    ``QThread -> sendPostedEvents -> ~QObject -> Sbk_GetPyOverride``, and a
    stress harness over this function alone reproduced it 3 runs in 8.

    Re-chaining it off ``thread.finished`` is NOT a fix and was measured: the
    worker's thread affinity is still the worker thread, so ``deleteLater``
    still defers into a loop that has stopped, and the same race survives
    (2 crashes in 20 at 800 jobs). A PySide6 object constructed in Python is
    already owned by Python; adding ``deleteLater`` on top is the second owner.
    So the worker is simply not scheduled for deletion — the caller's last
    reference frees it, on the thread that holds it.

    The QThread itself keeps ``deleteLater``: it is created on, and has the
    affinity of, the thread that calls this function, so its deferred delete is
    flushed by that thread's own running loop.

    A caller MUST hold a strong reference to both until ``thread.finished``:
    a QThread garbage-collected while running takes the process down.

    The job is also added to :func:`registry` for as long as it runs, so
    surfaces that did not start it (the Home screen) can show what spaCR
    is doing. Deregistration hangs off ``thread.finished`` — which has
    the *caller's* affinity — rather than ``worker.finished``, which is
    emitted on the worker thread and would take the registry (a GUI-thread
    QObject) with it.

    :param fn: callable invoked as ``fn(settings)`` on the worker thread.
    :param settings: the single argument handed to ``fn``.
    :param app_key: overrides the key stamped on ``fn`` by
        :func:`resolve_pipeline_entry`. Only needed for ad-hoc jobs that
        want to show up on Home under a name of their own.
    :returns: an unstarted ``(QThread, PipelineWorker)`` pair.
    """
    thread = QThread()
    allocation = apply_worker_budget(settings)
    worker = PipelineWorker(fn, settings, worker_count=allocation)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)

    reg = registry()
    handle = RunHandle(app_key or worker.app_key, worker, thread, parent=reg)
    reg.register(handle)
    # NOTE the absence of `handle.deleteLater` here. The handle is
    # parented to the registry, so C++ already owns it; adding a deferred
    # delete on top is the second owner, which is precisely the mistake
    # documented above for the worker. `RunRegistry.unregister` reparents
    # it to nothing instead, and Python frees it when the last reference
    # goes — on the thread that holds it.
    #
    # The slot is a bound method of a GUI-thread QObject, not a closure:
    # `thread.finished` is delivered across a thread boundary, and a
    # closure would both capture the handle and run with the emitting
    # thread's affinity.
    thread.finished.connect(handle.retire)
    return thread, worker
