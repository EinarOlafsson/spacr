"""
Live-preview segmentation widget.

Lets a user tune Cellpose parameters interactively against a single
sample tile before firing off a full plate run. The panel sits inside
the Mask app screen and does three things:

1. **Pick a sample.** Any ``.tif``/``.png``/``.tiff`` (dropped or via
   the file dialog) becomes the preview target. If the src field on
   the Mask app already points to a folder, the first image found
   there is loaded automatically.
2. **Tune knobs.** ``diameter``, ``flow_threshold``, ``cellprob``,
   ``channel`` — the values populate live from the Mask app settings
   so the preview matches the eventual run.
3. **Segment.** "Run preview" spawns a QThread that calls
   :meth:`cellpose.models.CellposeModel.eval` on the tile and hands
   back a mask. The panel renders original + mask overlay
   side-by-side.

Everything about Cellpose is **lazy-imported** in the worker thread —
this file is safe to import in an environment without a CUDA-capable
torch install or with Cellpose uninstalled. In that case the "Run
preview" button just surfaces a helpful error in the status area
instead of crashing at import time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

LOG = logging.getLogger("spacr.qt.live_preview")


SUPPORTED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Utility helpers — pure numpy, no Qt
# ---------------------------------------------------------------------------

def load_preview_image(path: Path) -> np.ndarray:
    """Read *path* into an (H, W) or (H, W, C) uint8/uint16 array.

    Tifffile is used for TIFFs to preserve bit-depth; other formats fall
    back to PIL. Raises :class:`FileNotFoundError` if the path is bad.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Return a viewable uint8 version of *img*, per-channel scaled."""
    if img.ndim == 3 and img.shape[-1] in (2, 3, 4):
        out = np.zeros(img.shape[:2] + (3,), dtype=np.uint8)
        for c in range(min(3, img.shape[-1])):
            slice_ = img[..., c].astype(np.float32)
            lo, hi = np.percentile(slice_, (1, 99))
            if hi <= lo:
                continue
            out[..., c] = np.clip(255 * (slice_ - lo) / (hi - lo),
                                   0, 255).astype(np.uint8)
        return out
    arr = img.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(255 * (arr - lo) / (hi - lo), 0, 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 overlay of ``mask`` boundaries over ``image``.

    Boundaries are drawn in green; interiors are dimmed 20% so contours
    read at a glance. Handles greyscale + multi-channel inputs.
    """
    base = _to_uint8(image)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base[..., :3].copy()
    if mask is None or not mask.any():
        return rgb
    # Boundary detection: pixels where any of the 4-neighbours differ
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :]   |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :]  |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:]   |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1]  |= mask[:, :-1] != mask[:, 1:]
    # Dim the interior of every labelled object
    inside = (mask > 0) & ~boundary
    rgb[inside] = (rgb[inside].astype(np.uint16) * 8 // 10).astype(np.uint8)
    rgb[boundary] = np.array([32, 220, 32], dtype=np.uint8)
    return rgb


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert an (H, W) or (H, W, 3) uint8 array to a :class:`QPixmap`."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr)
    h, w, _ = arr.shape
    img = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    # Deep-copy so pixmap outlives the numpy buffer.
    return QPixmap.fromImage(img.copy())


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _PreviewWorker(QThread):
    """QThread that runs one Cellpose segmentation and returns the mask."""

    finished_mask = Signal(object, str)   # (mask ndarray or None, error str)

    def __init__(self, image: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._image = image
        self._params = dict(params)

    def run(self):
        try:
            mask = _segment_once(self._image, self._params)
            self.finished_mask.emit(mask, "")
        except Exception as e:
            LOG.info("preview segmentation failed: %s", e, exc_info=True)
            self.finished_mask.emit(None, str(e))


def _segment_once(image: np.ndarray, params: dict) -> np.ndarray:
    """Run one CellposeModel.eval on ``image`` and return an (H, W) mask.

    ``params["model"]`` selects the backbone. ``"cpsam"`` (the default)
    loads Cellpose-SAM by passing ``pretrained_model="cpsam"`` per the
    spaCR pipeline_v2 convention; every other value is routed through
    the classic ``model_type=`` kwarg so legacy ``cyto2`` / ``cyto3`` /
    ``nuclei`` selections keep working.

    Cellpose is lazy-imported here so unit tests can exercise the pure
    helpers (``overlay_mask`` etc.) in an environment where Cellpose
    isn't installed.
    """
    from cellpose import models as cp_models
    model_name = params.get("model", "cpsam")
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    if model_name == "cpsam":
        model = cp_models.CellposeModel(
            gpu=gpu, pretrained_model="cpsam", device=None,
        )
    else:
        model = cp_models.CellposeModel(
            gpu=gpu, model_type=model_name, device=None,
        )
    if image.ndim == 3 and image.shape[-1] > 1:
        ch = int(params.get("channel", 0)) % image.shape[-1]
        img2d = image[..., ch]
    else:
        img2d = image
    result = model.eval(
        img2d,
        diameter=float(params.get("diameter", 30.0)) or None,
        flow_threshold=float(params.get("flow_threshold", 0.4)),
        cellprob_threshold=float(params.get("cellprob", 0.0)),
    )
    masks = result[0]
    if isinstance(masks, list):
        masks = masks[0]
    return np.asarray(masks)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class LivePreviewPanel(QWidget):
    """Interactive segmentation preview for the Mask app screen."""

    # Fired when a preview finishes — carries the mask so tests + hosts
    # can observe the result without introspecting private state.
    preview_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._worker: Optional[_PreviewWorker] = None
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # File picker row
        pick_row = QHBoxLayout()
        self._path_label = QLabel("No preview image loaded", self)
        self._path_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)
        pick_btn = QPushButton("Choose image…", self)
        pick_btn.clicked.connect(self._pick_file)
        pick_row.addWidget(self._path_label, 1)
        pick_row.addWidget(pick_btn)
        root.addLayout(pick_row)

        # Params row
        params = QHBoxLayout()
        self._model_box = QComboBox(self)
        # Cellpose-SAM is the default; the older cpu/cyto models stay as
        # legacy fallbacks so users with saved settings from older
        # versions still work. First entry is the default.
        self._model_box.addItems([
            "cpsam",        # Cellpose-SAM (default)
            "cyto3",        # legacy
            "cyto2",        # legacy
            "nuclei",       # legacy nuclear model
        ])
        self._diameter = QDoubleSpinBox(self)
        self._diameter.setRange(0, 400); self._diameter.setValue(30.0)
        self._diameter.setSuffix(" px")
        self._flow = QDoubleSpinBox(self)
        self._flow.setRange(-1, 3); self._flow.setSingleStep(0.05)
        self._flow.setValue(0.4)
        self._prob = QDoubleSpinBox(self)
        self._prob.setRange(-6, 6); self._prob.setSingleStep(0.1)
        self._prob.setValue(0.0)
        self._channel = QSpinBox(self)
        self._channel.setRange(0, 8)

        for label, w in (("model", self._model_box),
                          ("diameter", self._diameter),
                          ("flow_threshold", self._flow),
                          ("cellprob", self._prob),
                          ("channel", self._channel)):
            col = QVBoxLayout()
            col.addWidget(QLabel(label, self))
            col.addWidget(w)
            params.addLayout(col)
        root.addLayout(params)

        # Action row
        act = QHBoxLayout()
        self._run_btn = QPushButton("Run preview", self)
        self._run_btn.clicked.connect(self.run_preview)
        self._status = QLabel("", self)
        act.addWidget(self._run_btn)
        act.addWidget(self._status, 1)
        root.addLayout(act)

        # Canvas row
        canvas = QHBoxLayout()
        self._src_view = QLabel("original", self)
        self._src_view.setAlignment(Qt.AlignCenter)
        self._src_view.setMinimumHeight(256)
        self._src_view.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);")
        self._mask_view = QLabel("mask overlay", self)
        self._mask_view.setAlignment(Qt.AlignCenter)
        self._mask_view.setMinimumHeight(256)
        self._mask_view.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);")
        canvas.addWidget(self._src_view, 1)
        canvas.addWidget(self._mask_view, 1)
        root.addLayout(canvas, 1)

    # -- public API --------------------------------------------------------

    def load_image(self, path):
        """Load ``path`` into the preview pane. Silent on failure —
        the panel just keeps whatever was previously loaded."""
        try:
            arr = load_preview_image(Path(path))
        except Exception as e:
            self._status.setText(f"Load failed: {e}")
            return False
        self._image = arr
        self._image_path = Path(path)
        self._path_label.setText(str(path))
        self._src_view.setPixmap(numpy_to_qpixmap(_to_uint8(arr))
                                   .scaled(self._src_view.width(),
                                           self._src_view.height(),
                                           Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))
        self._mask_view.setPixmap(QPixmap())
        self._mask_view.setText("(run preview)")
        self._status.setText(f"Loaded {arr.shape} {arr.dtype}")
        return True

    def apply_settings(self, settings: dict):
        """Copy relevant values from a Mask-app ``settings`` dict."""
        try:
            if "diameter" in settings:
                self._diameter.setValue(float(settings["diameter"]))
            if "flow_threshold" in settings:
                self._flow.setValue(float(settings["flow_threshold"]))
            if "CP_prob" in settings:
                self._prob.setValue(float(settings["CP_prob"]))
            if "cell_channel" in settings and settings["cell_channel"] is not None:
                self._channel.setValue(int(settings["cell_channel"]))
            if "model_name" in settings:
                idx = self._model_box.findText(str(settings["model_name"]))
                if idx >= 0:
                    self._model_box.setCurrentIndex(idx)
        except Exception:
            # Bad settings shouldn't break the panel.
            LOG.debug("apply_settings failed", exc_info=True)

    def run_preview(self):
        """Kick off a background Cellpose segmentation on the loaded image."""
        if self._image is None:
            self._status.setText("Load an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Preview already running.")
            return
        params = self.current_params()
        self._run_btn.setEnabled(False)
        self._status.setText("Running preview…")
        worker = _PreviewWorker(self._image, params, self)
        worker.finished_mask.connect(self._on_worker_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def current_params(self) -> dict:
        """Snapshot the parameter widgets as a plain dict."""
        return {
            "model": self._model_box.currentText(),
            "diameter": self._diameter.value(),
            "flow_threshold": self._flow.value(),
            "cellprob": self._prob.value(),
            "channel": self._channel.value(),
        }

    # -- events ------------------------------------------------------------

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose preview image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if path:
            self.load_image(path)

    def _on_worker_done(self, mask, err):
        self._run_btn.setEnabled(True)
        self._worker = None
        if err:
            self._status.setText(f"Preview failed: {err}")
            self._mask_view.setText(f"Error: {err}")
            self.preview_ready.emit(None)
            return
        if mask is None or self._image is None:
            self._status.setText("Preview returned no mask.")
            return
        overlaid = overlay_mask(self._image, mask)
        n = int(mask.max()) if mask.size else 0
        self._status.setText(f"Found {n} object(s).")
        self._mask_view.setPixmap(numpy_to_qpixmap(overlaid)
                                     .scaled(self._mask_view.width(),
                                             self._mask_view.height(),
                                             Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))
        self.preview_ready.emit(mask)
