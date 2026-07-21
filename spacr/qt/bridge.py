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
    """A file-like object that emits every write to a queue for the UI."""

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
            try:
                self._on_write(line + "\n")
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        if self._buf:
            try:
                self._on_write(self._buf)
            except Exception:
                pass
            self._buf = ""


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
        super().__init__()
        self._fn = fn
        self._settings = settings

    def run(self):
        """Invoked by QThread.started."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirect = _StreamRedirector(self.line_ready.emit)
        sys.stdout = redirect
        sys.stderr = redirect

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
    app is interactive-only (annotate / make_masks) or unknown."""
    try:
        if app_key == "mask":
            from spacr.core import preprocess_generate_masks
            return preprocess_generate_masks
        if app_key == "measure":
            from spacr.measure import measure_crop
            return measure_crop
        if app_key == "classify":
            from spacr.deep_spacr import train_test_model
            return train_test_model
        if app_key == "umap":
            from spacr.core import generate_image_umap
            return generate_image_umap
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


def make_thread(fn, settings):
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
