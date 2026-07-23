"""
Live-preview segmentation widget — v2.

Interactive Cellpose tuning surface for the Mask app screen. Compared
to v1, this rewrite adds every enhancement the user requested after
their first run through the panel:

* **Zoomable canvases (Ctrl+scroll, in sync).** Both the original and
  the mask overlay live in a shared :class:`QGraphicsView` pair — pan
  and zoom on one and the other tracks pixel-for-pixel.
* **Hover tooltip.** Move the cursor over the original and a pinned
  status line shows the pixel intensity for every channel plus, when
  present, the object label at that position from the last segmenta-
  tion. Same tooltip regardless of which view holds the cursor.
* **Normalise toggle.** Optional 2–98 % percentile stretch (per channel
  for RGB) so raw low-contrast tiles are legible.
* **Model-aware options.** Selecting ``cpsam`` hides the parameters
  Cellpose-SAM ignores (``flow_threshold``, ``cellprob``, ``diameter``)
  and shows only the ones it actually uses. Legacy models get the full
  set.
* **Outline colour + thickness.** Chosen from the toolbar; effect is
  live once a mask exists.
* **Multi-object segmentation.** An "object type" combo picks between
  ``cell``, ``nucleus``, and ``cell + nucleus``. In cell+nucleus mode
  the panel runs two Cellpose passes and overlays both masks in
  distinct colours.
* **Pre / Post filters.** When the object type is ``cell`` (or the
  combined mode) the panel routes pre / post-processing settings from
  the Mask app (``cell_min_size``, ``cell_max_size``,
  ``remove_background_cell``, background intensity, ...) through the
  segmentation. Users toggle these on/off with dedicated "Pre" / "Post"
  clickable labels sitting next to "Run preview" in the same visual
  style as the LP / AI toggles.

The whole file stays safe to import without cellpose — every cellpose
call is lazy-imported inside the worker thread.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (
    QBrush, QColor, QImage, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

LOG = logging.getLogger("spacr.qt.live_preview")

SUPPORTED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Object types the panel understands. Order matters — it drives the
# order of the combo.
OBJECT_TYPES = ("cell", "nucleus", "cell + nucleus")

# Overlay colours for individual object types. Cell = green (matches
# the classic v1 boundary colour), nucleus = magenta, and when both
# are shown together those colours read cleanly on top of most stains.
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cell":    (32, 220, 32),
    "nucleus": (222, 82, 200),
}


# ---------------------------------------------------------------------------
# Pure numpy helpers — no Qt, safe to unit-test without a display
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


def _to_uint8(img: np.ndarray, normalise: bool = True,
                lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Return a viewable uint8 version of *img*.

    :param normalise: when True (default) apply a per-channel percentile
        stretch. When False the array is clipped to [0, 255] without
        any rescaling — useful when the user has already normalised
        upstream and wants pixel values verbatim.
    :param lo_pct: lower percentile for the stretch (default 2 %).
    :param hi_pct: upper percentile for the stretch (default 98 %).
    """
    if img.ndim == 3 and img.shape[-1] in (2, 3, 4):
        out = np.zeros(img.shape[:2] + (3,), dtype=np.uint8)
        for c in range(min(3, img.shape[-1])):
            slice_ = img[..., c].astype(np.float32)
            if normalise:
                lo, hi = np.percentile(slice_, (lo_pct, hi_pct))
                if hi <= lo:
                    continue
                out[..., c] = np.clip(
                    255 * (slice_ - lo) / (hi - lo), 0, 255,
                ).astype(np.uint8)
            else:
                out[..., c] = np.clip(slice_, 0, 255).astype(np.uint8)
        return out
    arr = img.astype(np.float32)
    if normalise:
        lo, hi = np.percentile(arr, (lo_pct, hi_pct))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip(
            255 * (arr - lo) / (hi - lo), 0, 255,
        ).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a bool array marking the 4-connected boundary of ``mask``."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :]  |= mask[1:, :]  != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:]  |= mask[:, 1:]  != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def overlay_masks(image: np.ndarray,
                    masks: Dict[str, np.ndarray],
                    outline_rgb: Optional[Tuple[int, int, int]] = None,
                    outline_thickness: int = 1,
                    normalise: bool = True,
                    lo_pct: float = 2.0,
                    hi_pct: float = 98.0) -> np.ndarray:
    """Return an RGB uint8 view of ``image`` with every mask's boundary
    drawn in the object's colour (or ``outline_rgb`` when supplied).

    :param image: (H, W) or (H, W, C) source image.
    :param masks: ``{object_type: label_array}`` — one entry per object
        type currently visible on the panel.
    :param outline_rgb: overrides the per-object colour when the user
        picks a global outline colour from the toolbar.
    :param outline_thickness: number of pixels the boundary is dilated
        by (1 = crisp, 3 = highlighter). Tops out at 5.
    :param normalise: forwarded to :func:`_to_uint8`.
    """
    base = _to_uint8(image, normalise=normalise,
                        lo_pct=lo_pct, hi_pct=hi_pct)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base[..., :3].copy()
    outline_thickness = max(1, min(5, int(outline_thickness)))
    for obj_type, mask in masks.items():
        if mask is None or not mask.any():
            continue
        boundary = _boundary_mask(mask.astype(np.int32))
        for _ in range(outline_thickness - 1):
            # Dilate by one pixel: OR-shift in each cardinal direction
            b2 = boundary.copy()
            b2[1:, :]  |= boundary[:-1, :]
            b2[:-1, :] |= boundary[1:, :]
            b2[:, 1:]  |= boundary[:, :-1]
            b2[:, :-1] |= boundary[:, 1:]
            boundary = b2
        colour = outline_rgb if outline_rgb is not None else \
            OBJECT_COLORS.get(obj_type, (32, 220, 32))
        rgb[boundary] = np.array(colour, dtype=np.uint8)
    return rgb


def numpy_to_qpixmap(arr: np.ndarray, normalise: bool = True,
                        lo_pct: float = 2.0,
                        hi_pct: float = 98.0) -> QPixmap:
    """Convert an (H, W) or (H, W, 3) array to a :class:`QPixmap`."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr, normalise=normalise,
                          lo_pct=lo_pct, hi_pct=hi_pct)
    h, w, _ = arr.shape
    img = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# ---------------------------------------------------------------------------
# Segmentation worker
# ---------------------------------------------------------------------------

@dataclass
class PreviewRequest:
    """Everything the worker needs to run one segmentation pass.

    Kept as a plain dataclass so tests can construct it directly; the
    panel builds one from its widget state on each Run.
    """
    image:               np.ndarray
    model:               str = "cpsam"
    diameter:            float = 30.0
    flow_threshold:      float = 0.4
    cellprob:            float = 0.0
    channels:            Dict[str, int] = field(default_factory=dict)
    object_types:        Tuple[str, ...] = ("cell",)
    preprocess_settings: Dict[str, Any] = field(default_factory=dict)
    postprocess_settings: Dict[str, Any] = field(default_factory=dict)


class _PreviewWorker(QThread):
    """Runs one (or two) Cellpose passes in the background."""

    finished_masks = Signal(object, str)   # ({obj: mask, ...} or None, err)

    def __init__(self, request: PreviewRequest, parent=None):
        super().__init__(parent)
        self._request = request

    def run(self):
        try:
            masks = _segment_multi(self._request)
            self.finished_masks.emit(masks, "")
        except Exception as e:
            LOG.info("live-preview segmentation failed: %s", e,
                       exc_info=True)
            self.finished_masks.emit(None, str(e))


def _segment_multi(req: PreviewRequest) -> Dict[str, np.ndarray]:
    """Run one Cellpose pass per requested object type.

    Cellpose is lazy-imported here so importing this file cold — as
    unit tests do — does not require a CUDA-capable stack.

    Post-processing (min/max size filter, background removal) is
    applied per-object-type after the model returns, using the
    ``postprocess_settings`` dict on the request.
    """
    from cellpose import models as cp_models
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False

    if req.model == "cpsam":
        model = cp_models.CellposeModel(
            gpu=gpu, pretrained_model="cpsam", device=None)
    else:
        model = cp_models.CellposeModel(
            gpu=gpu, model_type=req.model, device=None)

    out: Dict[str, np.ndarray] = {}
    for obj in req.object_types:
        ch_idx = req.channels.get(obj, 0)
        image_2d = _select_channel(req.image, ch_idx)

        # Preprocess — background subtraction if the user opted in.
        if req.preprocess_settings.get(f"remove_background_{obj}"):
            bg = float(req.preprocess_settings.get("background", 100.0))
            image_2d = np.clip(image_2d.astype(np.float32) - bg,
                                0, None).astype(image_2d.dtype)

        result = model.eval(
            image_2d,
            diameter=float(req.diameter) or None,
            flow_threshold=float(req.flow_threshold),
            cellprob_threshold=float(req.cellprob),
        )
        mask = result[0]
        if isinstance(mask, list):
            mask = mask[0]
        mask = np.asarray(mask).astype(np.int32)

        # Post-processing — size filter by object type.
        mask = _apply_size_filter(mask, req.postprocess_settings, obj)
        out[obj] = mask
    return out


def _select_channel(image: np.ndarray, ch: int) -> np.ndarray:
    """Return a 2-D slice from ``image`` at channel index ``ch``."""
    if image.ndim == 3 and image.shape[-1] > 1:
        return image[..., int(ch) % image.shape[-1]]
    return image.squeeze()


def _apply_size_filter(mask: np.ndarray,
                          settings: Dict[str, Any],
                          obj: str) -> np.ndarray:
    """Zero out labels smaller than ``{obj}_min_size`` or larger than
    ``{obj}_max_size`` (both optional). Silently no-op when either
    setting is missing / non-numeric."""
    if not settings:
        return mask
    min_key = f"{obj}_min_size"
    max_key = f"{obj}_max_size"
    min_size = settings.get(min_key)
    max_size = settings.get(max_key)
    if min_size is None and max_size is None:
        return mask
    try:
        counts = np.bincount(mask.ravel())
    except TypeError:
        return mask
    keep = np.ones_like(counts, dtype=bool)
    if min_size is not None:
        keep &= counts >= float(min_size)
    if max_size is not None:
        keep &= counts <= float(max_size)
    keep[0] = True   # never drop background
    remap = np.where(keep, np.arange(len(counts)), 0)
    return remap[mask].astype(mask.dtype)


# ---------------------------------------------------------------------------
# Twin zoomable views with a shared transform
# ---------------------------------------------------------------------------

class _ZoomView(QGraphicsView):
    """A :class:`QGraphicsView` that emits pixel-hover events + supports
    plain wheel-scroll zoom.

    Two big usability upgrades over the first-cut version:

    * **Wheel = zoom (no Ctrl needed).** Turning the wheel zooms
      centred on the cursor. Shift+wheel scrolls the viewport
      vertically if the user wants scroll behaviour.
    * **Fit-to-height on load + resize.** The image always fills the
      canvas at 100 % zoom until the user starts scrolling, so a small
      preview panel doesn't leave the tile 1-cm tall in the corner.
      Every ``resizeEvent`` re-fits — as the splitter is dragged, the
      image grows to match.

    Zoom is broadcast to a peer view via :meth:`set_peer` so the mask
    canvas mirrors what the original canvas is doing (and vice versa).
    """

    hover_pixel = Signal(int, int)   # (x, y) in image coords
    zoom_changed = Signal(float)     # new scale factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._peer: Optional["_ZoomView"] = None
        self._syncing = False
        self._scale = 1.0
        self._user_zoomed = False
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFrameShape(QGraphicsView.NoFrame)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        # Fit-in-view on load, and forget any previous user zoom so the
        # new image starts at 100 % of the canvas.
        self._user_zoomed = False
        self._scale = 1.0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_peer(self, peer: "_ZoomView") -> None:
        self._peer = peer

    def scale_factor(self) -> float:
        return self._scale

    def reset_zoom(self) -> None:
        """Snap back to fit-in-view (100 % of the container)."""
        self._user_zoomed = False
        self._scale = 1.0
        self.resetTransform()
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # -- events ------------------------------------------------------------

    def wheelEvent(self, event):
        """Plain wheel = zoom around cursor. Shift+wheel = scroll."""
        if event.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        factor = 1.20 if event.angleDelta().y() > 0 else 0.833
        self._apply_zoom(factor, broadcast=True)
        event.accept()

    def resizeEvent(self, event):
        """Refit the tile whenever the container size changes, unless
        the user has manually zoomed in / out."""
        super().resizeEvent(event)
        if not self._user_zoomed and self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _apply_zoom(self, factor: float, broadcast: bool = False) -> None:
        if self._syncing:
            return
        self.scale(factor, factor)
        self._scale *= factor
        self._user_zoomed = True
        self.zoom_changed.emit(self._scale)
        if broadcast and self._peer is not None:
            self._peer._syncing = True
            self._peer._apply_zoom(factor, broadcast=False)
            self._peer._syncing = False

    def mouseMoveEvent(self, event):
        if self._pixmap_item is not None:
            scene_pt = self.mapToScene(event.position().toPoint())
            x = int(scene_pt.x())
            y = int(scene_pt.y())
            self.hover_pixel.emit(x, y)
        super().mouseMoveEvent(event)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class LivePreviewPanel(QWidget):
    """Interactive segmentation preview — Mask app only."""

    preview_ready = Signal(object)   # {object_type: mask}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._masks: Dict[str, np.ndarray] = {}
        self._settings: Dict[str, Any] = {}
        self._worker: Optional[_PreviewWorker] = None
        self._pre_on = False
        self._post_on = False
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # -- HIDDEN state widgets ----------------------------------------
        # Every parameter widget lives here even though only a subset
        # appears in the collapsed layout. The Live Settings dialog
        # re-parents them into its own form when it opens, then hands
        # them back on close so their values persist across opens.
        # All widgets are children of `self` so they're never
        # garbage-collected while re-parented.
        self._model_box = QComboBox(self)
        self._model_box.addItems(["cpsam", "cyto3", "cyto2", "nuclei"])
        self._model_box.currentIndexChanged.connect(
            self._on_model_or_object_changed)

        self._object_box = QComboBox(self)
        self._object_box.addItems(list(OBJECT_TYPES))
        self._object_box.currentIndexChanged.connect(
            self._on_model_or_object_changed)

        self._cell_channel = QSpinBox(self); self._cell_channel.setRange(0, 8)
        self._nucleus_channel = QSpinBox(self); self._nucleus_channel.setRange(0, 8)
        self._nucleus_channel.setValue(1)

        self._diameter = QDoubleSpinBox(self)
        self._diameter.setRange(0, 400); self._diameter.setValue(30.0)
        self._diameter.setSuffix(" px")
        self._flow = QDoubleSpinBox(self)
        self._flow.setRange(-1, 3); self._flow.setSingleStep(0.05)
        self._flow.setValue(0.4)
        self._prob = QDoubleSpinBox(self)
        self._prob.setRange(-6, 6); self._prob.setSingleStep(0.1)
        self._prob.setValue(0.0)

        # Two-field percentile stretch — user asked for this shape
        # explicitly (was a single toggle before).
        self._normalise_check = QCheckBox("Normalise", self)
        self._normalise_check.setChecked(True)
        self._normalise_check.toggled.connect(self._refresh_canvases)
        self._lo_pct = QDoubleSpinBox(self)
        self._lo_pct.setRange(0, 50); self._lo_pct.setValue(2.0)
        self._lo_pct.setSuffix(" %")
        self._lo_pct.valueChanged.connect(self._refresh_canvases)
        self._hi_pct = QDoubleSpinBox(self)
        self._hi_pct.setRange(50, 100); self._hi_pct.setValue(98.0)
        self._hi_pct.setSuffix(" %")
        self._hi_pct.valueChanged.connect(self._refresh_canvases)

        # Outline appearance
        self._outline_colour = QComboBox(self)
        for name in ("auto", "green", "magenta", "yellow", "cyan",
                       "white", "red"):
            self._outline_colour.addItem(name)
        self._outline_colour.currentIndexChanged.connect(
            self._refresh_canvases)
        self._outline_thickness = QSpinBox(self)
        self._outline_thickness.setRange(1, 5)
        self._outline_thickness.setValue(1)
        self._outline_thickness.valueChanged.connect(
            self._refresh_canvases)

        # Pre / Post checkboxes live inside the settings dialog too.
        self._pre_check = QCheckBox("Pre (cell background / min-max)",
                                       self)
        self._pre_check.toggled.connect(self._on_pre_toggle)
        self._post_check = QCheckBox(
            "Post (size filter after Cellpose)", self)
        self._post_check.toggled.connect(self._on_post_toggle)

        # Keep every hidden helper widget parented but invisible so it
        # doesn't render in the compact layout.
        for w in (self._model_box, self._object_box,
                    self._cell_channel, self._nucleus_channel,
                    self._diameter, self._flow, self._prob,
                    self._normalise_check, self._lo_pct, self._hi_pct,
                    self._outline_colour, self._outline_thickness,
                    self._pre_check, self._post_check):
            w.hide()

        # -- VISIBLE compact layout --------------------------------------

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

        # Action row — Run + Live settings + status
        act = QHBoxLayout()
        self._run_btn = QPushButton("Run preview", self)
        self._run_btn.clicked.connect(self.run_preview)
        self._live_settings_btn = QPushButton("Live settings…", self)
        self._live_settings_btn.clicked.connect(self.open_live_settings)
        self._status = QLabel("", self)
        act.addWidget(self._run_btn)
        act.addWidget(self._live_settings_btn)
        act.addWidget(self._status, 1)
        root.addLayout(act)

        # Twin zoomable canvases in a synchronised pair.
        canvas = QHBoxLayout()
        self._src_view = _ZoomView(self)
        self._src_view.setMinimumHeight(160)
        self._mask_view = _ZoomView(self)
        self._mask_view.setMinimumHeight(160)
        self._src_view.set_peer(self._mask_view)
        self._mask_view.set_peer(self._src_view)
        self._src_view.hover_pixel.connect(self._on_hover)
        self._mask_view.hover_pixel.connect(self._on_hover)
        canvas.addWidget(self._src_view, 1)
        canvas.addWidget(self._mask_view, 1)
        root.addLayout(canvas, 1)

        # Pinned hover info line
        self._hover_label = QLabel("Hover over the image to inspect pixels.",
                                     self)
        self._hover_label.setStyleSheet("color: palette(mid); "
                                            "font-family: monospace;")
        root.addWidget(self._hover_label)

        # Book-keeping for the dialog-based settings surface. Kept as
        # a member so tests + external hooks can introspect / drive it.
        self._live_settings_dialog: Optional["LiveSettingsDialog"] = None
        self._on_model_or_object_changed()

    # -- public API --------------------------------------------------------

    def load_image(self, path):
        try:
            arr = load_preview_image(Path(path))
        except Exception as e:
            self._status.setText(f"Load failed: {e}")
            return False
        self._image = arr
        self._image_path = Path(path)
        self._masks = {}
        self._path_label.setText(str(path))
        self._status.setText(f"Loaded {arr.shape} {arr.dtype}")
        self._refresh_canvases()
        return True

    def apply_settings(self, settings: dict):
        """Copy relevant values from a Mask-app ``settings`` dict, and
        cache the whole dict for the Pre / Post routes to read from."""
        self._settings = dict(settings)
        try:
            if "diameter" in settings:
                self._diameter.setValue(float(settings["diameter"]))
            if "flow_threshold" in settings:
                self._flow.setValue(float(settings["flow_threshold"]))
            if "CP_prob" in settings:
                self._prob.setValue(float(settings["CP_prob"]))
            if "cell_channel" in settings and settings["cell_channel"] is not None:
                self._cell_channel.setValue(int(settings["cell_channel"]))
            if "nucleus_channel" in settings and settings["nucleus_channel"] is not None:
                self._nucleus_channel.setValue(int(settings["nucleus_channel"]))
            if "model_name" in settings:
                idx = self._model_box.findText(str(settings["model_name"]))
                if idx >= 0:
                    self._model_box.setCurrentIndex(idx)
        except Exception:
            LOG.debug("apply_settings failed", exc_info=True)

    def current_params(self) -> dict:
        """Snapshot for tests + external callers."""
        return {
            "model": self._model_box.currentText(),
            "diameter": self._diameter.value(),
            "flow_threshold": self._flow.value(),
            "cellprob": self._prob.value(),
            "object_types": self._selected_object_types(),
            "cell_channel": self._cell_channel.value(),
            "nucleus_channel": self._nucleus_channel.value(),
            "normalise": self._normalise_check.isChecked(),
            "lo_pct": float(self._lo_pct.value()),
            "hi_pct": float(self._hi_pct.value()),
            "pre": self._pre_on,
            "post": self._post_on,
            "outline_thickness": self._outline_thickness.value(),
            "outline_colour": self._outline_colour.currentText(),
        }

    def run_preview(self):
        if self._image is None:
            self._status.setText("Load an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Preview already running.")
            return
        req = self._build_request()
        self._run_btn.setEnabled(False)
        self._status.setText("Running preview…")
        worker = _PreviewWorker(req, self)
        worker.finished_masks.connect(self._on_worker_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    # -- internals ---------------------------------------------------------

    def _selected_object_types(self) -> Tuple[str, ...]:
        current = self._object_box.currentText()
        if current == "cell + nucleus":
            return ("cell", "nucleus")
        return (current,)

    def _build_request(self) -> PreviewRequest:
        obj_types = self._selected_object_types()
        channels = {
            "cell":    self._cell_channel.value(),
            "nucleus": self._nucleus_channel.value(),
        }
        pre = self._settings if self._pre_on else {}
        post = self._settings if self._post_on else {}
        return PreviewRequest(
            image=self._image,
            model=self._model_box.currentText(),
            diameter=self._diameter.value(),
            flow_threshold=self._flow.value(),
            cellprob=self._prob.value(),
            channels=channels,
            object_types=obj_types,
            preprocess_settings=pre,
            postprocess_settings=post,
        )

    def _outline_rgb(self) -> Optional[Tuple[int, int, int]]:
        """Translate the outline-colour combo choice into an RGB tuple,
        or ``None`` when the user picked 'auto' (per-object colour)."""
        choice = self._outline_colour.currentText()
        mapping = {
            "green":   (32, 220, 32),
            "magenta": (222, 82, 200),
            "yellow":  (255, 220, 32),
            "cyan":    (32, 200, 220),
            "white":   (240, 240, 240),
            "red":     (240, 60, 60),
        }
        return mapping.get(choice)   # None on "auto"

    def _refresh_canvases(self):
        """Re-render both views from the current image + masks."""
        if self._image is None:
            return
        norm = self._normalise_check.isChecked()
        lo = float(self._lo_pct.value())
        hi = float(self._hi_pct.value())
        src_pix = numpy_to_qpixmap(
            _to_uint8(self._image, normalise=norm, lo_pct=lo, hi_pct=hi))
        self._src_view.set_pixmap(src_pix)
        if self._masks:
            overlay = overlay_masks(
                self._image, self._masks,
                outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi)
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)

    def _on_model_or_object_changed(self, *_):
        """Refresh visibility state — no visible-widget mutation on
        the compact layout anymore (options are hidden by default and
        only shown inside the Live Settings dialog when it's open).
        The dialog re-reads visibility rules on open, so nothing to
        do here at rest."""
        # Kept as a hook so any observers subscribed to model/object
        # combo changes still fire.
        dlg = self._live_settings_dialog
        if dlg is not None:
            try:
                dlg.refresh_visibility()
            except Exception:
                pass

    def _on_pre_toggle(self, on: bool):
        self._pre_on = on

    def _on_post_toggle(self, on: bool):
        self._post_on = on

    def open_live_settings(self):
        """Open (or focus) the Live Settings modal.

        The dialog rehomes every hidden state widget into its form so
        the user's edits go straight into `self._*` — nothing to sync.
        On close, widgets are re-parented back to `self` (hidden again)
        so state persists across opens.
        """
        if (self._live_settings_dialog is not None
                and self._live_settings_dialog.isVisible()):
            self._live_settings_dialog.raise_()
            self._live_settings_dialog.activateWindow()
            return
        self._live_settings_dialog = LiveSettingsDialog(self)
        self._live_settings_dialog.finished.connect(self._on_settings_closed)
        self._live_settings_dialog.show()

    def _on_settings_closed(self, *_):
        # Refresh canvases in case a visual-only setting changed (e.g.
        # outline colour) while the dialog was open.
        self._refresh_canvases()
        self._live_settings_dialog = None

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose preview image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if path:
            self.load_image(path)

    def _on_hover(self, x: int, y: int):
        """Render the pinned hover-info line for the pixel under the cursor."""
        if self._image is None:
            return
        h = self._image.shape[0]
        w = self._image.shape[1] if self._image.ndim >= 2 else 0
        if not (0 <= y < h and 0 <= x < w):
            self._hover_label.setText(
                "Hover over the image to inspect pixels.")
            return
        # Intensities across every channel
        if self._image.ndim == 3:
            vals = tuple(int(v) for v in self._image[y, x])
            i_str = f"channels={vals}"
        else:
            i_str = f"intensity={int(self._image[y, x])}"
        # Mask hit-tests
        hits = []
        for obj, mask in self._masks.items():
            if mask is None or mask.size == 0:
                continue
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                lbl = int(mask[y, x])
                if lbl > 0:
                    area = int((mask == lbl).sum())
                    hits.append(f"{obj}=#{lbl} area={area}px")
        obj_str = f"  {'  '.join(hits)}" if hits else ""
        self._hover_label.setText(f"(x={x:>4d}, y={y:>4d})  {i_str}{obj_str}")

    def _on_worker_done(self, masks, err):
        self._run_btn.setEnabled(True)
        self._worker = None
        if err:
            self._status.setText(f"Preview failed: {err}")
            self.preview_ready.emit(None)
            return
        if masks is None or not masks or self._image is None:
            self._status.setText("Preview returned no masks.")
            return
        self._masks = masks
        counts = [f"{k}={int(v.max() if v.size else 0)}"
                    for k, v in masks.items()]
        self._status.setText(f"Found {', '.join(counts)}.")
        self._refresh_canvases()
        self.preview_ready.emit(masks)


# ---------------------------------------------------------------------------
# Live Settings dialog
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout,
)


class LiveSettingsDialog(QDialog):
    """Modal dialog that surfaces every live-preview setting.

    Re-parents the panel's hidden state widgets into a QFormLayout so
    edits go straight into the panel's canonical fields — nothing to
    sync manually. On close, widgets are returned to the panel hidden
    so their values persist across opens.

    Rows shown (per the user's spec):
      * Normalisation upper + lower percentile
      * Outline colour
      * Outline thickness
      * Model
      * Flow threshold
      * Cell probability
      * Object type
      * Object channel (cell / nucleus depending on selection)
      * Pre  (bool)
      * Post (bool)
    """

    def __init__(self, panel: "LivePreviewPanel"):
        super().__init__(panel)
        self._panel = panel
        self.setWindowTitle("Live settings")
        self.setMinimumWidth(360)
        outer = QVBoxLayout(self)
        form = QFormLayout()

        # Show the widgets we'll be adding, then re-hide them on close.
        for w in self._managed_widgets():
            w.show()

        form.addRow("Model", panel._model_box)
        form.addRow("Object", panel._object_box)
        form.addRow("Cell channel", panel._cell_channel)
        form.addRow("Nucleus channel", panel._nucleus_channel)
        form.addRow("Diameter", panel._diameter)
        form.addRow("Flow threshold", panel._flow)
        form.addRow("Cell probability", panel._prob)
        form.addRow(panel._normalise_check)
        form.addRow("Lower percentile", panel._lo_pct)
        form.addRow("Upper percentile", panel._hi_pct)
        form.addRow("Outline colour", panel._outline_colour)
        form.addRow("Outline thickness", panel._outline_thickness)
        form.addRow(panel._pre_check)
        form.addRow(panel._post_check)

        outer.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)

        self.refresh_visibility()

    def _managed_widgets(self):
        p = self._panel
        return [p._model_box, p._object_box, p._cell_channel,
                p._nucleus_channel, p._diameter, p._flow, p._prob,
                p._normalise_check, p._lo_pct, p._hi_pct,
                p._outline_colour, p._outline_thickness,
                p._pre_check, p._post_check]

    def refresh_visibility(self):
        """Hide model-ignored knobs when Cellpose-SAM is selected and
        the unused channel spin depending on object type."""
        p = self._panel
        is_sam = p._model_box.currentText() == "cpsam"
        for w in (p._diameter, p._flow, p._prob):
            w.setEnabled(not is_sam)
            w.setToolTip("Ignored by Cellpose-SAM" if is_sam else "")
        obj = p._object_box.currentText()
        p._cell_channel.setEnabled(obj != "nucleus")
        p._nucleus_channel.setEnabled(obj != "cell")

    def closeEvent(self, event):
        """Re-hide the state widgets so the compact layout stays clean."""
        for w in self._managed_widgets():
            w.hide()
            # Re-parent back to the panel so the widget survives dialog
            # deletion (Qt would otherwise destroy children).
            w.setParent(self._panel)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Back-compat shims for callers that predate the multi-object rewrite
# ---------------------------------------------------------------------------

def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Legacy single-mask overlay retained for older imports."""
    return overlay_masks(image, {"cell": mask})
