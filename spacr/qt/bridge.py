"""
Background execution + progress bridge between the Qt UI and the pipeline
functions in spacr.core / spacr.deep_spacr / spacr.submodules / etc.

Runs each pipeline call in a QThread so the UI stays responsive. The
worker installs stdout/stderr shims that emit `line_ready(str)` on every
print, so the caller can pipe them into a QPlainTextEdit console.
"""
from __future__ import annotations

import io
import sys
import traceback
from typing import Any, Callable, Dict

from PySide6.QtCore import QObject, QThread, Signal


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

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        # Emit whole lines eagerly so the UI updates smoothly.
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._safe_emit(line + "\n")
        # Chunk-cap: whatever is left after full-line drain, if it's
        # grown suspiciously large, flush it verbatim.
        if len(self._buf) >= self._MAX_BUF_CHARS:
            self._safe_emit(self._buf)
            self._buf = ""
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._safe_emit(self._buf)
            self._buf = ""

    def idle_flush(self) -> None:
        """Emit any pending partial line — safe to call from a QTimer."""
        if self._buf:
            self._safe_emit(self._buf)
            self._buf = ""

    def _safe_emit(self, s: str) -> None:
        try:
            self._on_write(s)
        except Exception:
            pass


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
    figure_ready = Signal(object)

    def __init__(self, fn: Callable[..., Any], settings: Dict[str, Any]):
        """Prepare to run ``fn(settings)`` in a worker thread.

        :param fn: pipeline entry point (see :func:`resolve_pipeline_entry`).
        :param settings: keyword-style dict passed as the sole argument.
        """
        super().__init__()
        self._fn = fn
        self._settings = settings

    def run(self) -> None:
        """Invoked by QThread.started; runs the pipeline function to completion."""
        import threading
        import time as _time
        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirect = _StreamRedirector(self.line_ready.emit)
        sys.stdout = redirect
        sys.stderr = redirect

        # Idle-flush pump — a daemon thread that flushes the buffered
        # partial line every 500 ms so the console never sits silent
        # for more than half a second while the pipeline is chatty. It
        # also emits a "keepalive" ping every 10 s of complete silence
        # so the user sees the run is still going. Purely diagnostic —
        # if this thread dies for any reason the pipeline still runs.
        stop_pump = threading.Event()
        last_output = [_time.time()]
        # Wrap the emit signal so we can tick "last_output" on every
        # real write. This lets keepalive skip when the pipeline is
        # already talking on its own.
        original_write = redirect._safe_emit
        def _tick_emit(text: str):
            last_output[0] = _time.time()
            original_write(text)
        redirect._safe_emit = _tick_emit  # type: ignore[assignment]

        def _pump():
            while not stop_pump.is_set():
                stop_pump.wait(0.5)
                if stop_pump.is_set():
                    break
                try:
                    redirect.idle_flush()
                except Exception:
                    pass
                now = _time.time()
                if now - last_output[0] >= 10.0:
                    stamp = _time.strftime("%H:%M:%S")
                    _tick_emit(
                        f"[{stamp}] …still running (no output for 10s)…\n")
        pump = threading.Thread(
            target=_pump, name="spacr-worker-pump", daemon=True)
        pump.start()

        # Intercept matplotlib show() so figures land in the UI instead
        # of a blocking Tk window. `plt.show` gets restored in `finally`.
        old_show = None
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            old_show = plt.show
            worker = self

            def _capture_show(*args, **kwargs):
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    worker.figure_ready.emit(fig)
                return None

            plt.show = _capture_show
        except Exception:
            plt = None

        ok = False
        try:
            self._fn(self._settings)
            ok = True
        except SystemExit:
            ok = True
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)
        finally:
            stop_pump.set()
            try:
                redirect.flush()
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if old_show is not None and plt is not None:
                try:
                    plt.show = old_show
                except Exception:
                    pass
            self.finished.emit(ok)


# ---------------------------------------------------------------------------
# Dispatch: app_key -> function to run
# ---------------------------------------------------------------------------

def resolve_pipeline_entry(app_key: str) -> Callable[[Dict[str, Any]], Any] | None:
    """Return the pipeline function that runs a given app, or None if the
    app is interactive-only (annotate / make_masks) or unknown.

    Each returned entry point is wrapped with
    :func:`spacr.qt.verbose_logger.log_call` so that when the user has
    "Verbose logging" enabled, every pipeline invocation emits an
    entry-and-return trace in the console. Zero cost when verbose is
    off (the wrapper is a single attribute check).
    """
    from .verbose_logger import log_call
    try:
        if app_key == "mask":
            from spacr.core import preprocess_generate_masks
            return log_call(preprocess_generate_masks)
        if app_key == "measure":
            from spacr.measure import measure_crop
            return log_call(measure_crop)
        if app_key == "classify":
            from spacr.deep_spacr import train_test_model
            return log_call(train_test_model)
        if app_key == "umap":
            from spacr.core import generate_image_umap
            return log_call(generate_image_umap)
        if app_key == "train_cellpose":
            from spacr.submodules import train_cellpose
            return train_cellpose
        if app_key == "cellpose_masks":
            from spacr.spacr_cellpose import identify_masks_finetune
            return identify_masks_finetune
        if app_key == "cellpose_all":
            from spacr.spacr_cellpose import check_cellpose_models
            return check_cellpose_models
        if app_key == "map_barcodes":
            from spacr.sequencing import generate_barecode_mapping
            return generate_barecode_mapping
        if app_key == "ml_analyze":
            from spacr.ml import generate_ml_scores
            return generate_ml_scores
        if app_key == "regression":
            from spacr.ml import perform_regression
            return perform_regression
        if app_key == "recruitment":
            from spacr.submodules import analyze_recruitment
            return analyze_recruitment
        if app_key == "activation":
            from spacr.deep_spacr import generate_activation_map
            return generate_activation_map
        if app_key == "analyze_plaques":
            from spacr.submodules import analyze_plaques
            return analyze_plaques
    except Exception:
        return None
    return None


def make_thread(
    fn: Callable[[Dict[str, Any]], Any],
    settings: Dict[str, Any],
) -> tuple["QThread", PipelineWorker]:
    """Return (thread, worker) — caller connects worker signals and calls
    thread.start()."""
    thread = QThread()
    worker = PipelineWorker(fn, settings)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker
