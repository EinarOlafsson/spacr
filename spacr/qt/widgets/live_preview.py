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
# order of the combo. cell/nucleus can be previewed together; pathogen and
# organelle are single-compartment selections whose settings panels light up
# when chosen.
OBJECT_TYPES = ("cell", "nucleus", "cell + nucleus", "pathogen", "organelle")

# The four segmentation compartments, in the left→right order their settings
# panels appear in the Live settings dialog.
COMPARTMENTS = ("cell", "nucleus", "pathogen", "organelle")

# Overlay colours for individual object types. Cell = green (matches
# the classic v1 boundary colour), nucleus = magenta, and when both
# are shown together those colours read cleanly on top of most stains.
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cell":      (32, 220, 32),
    "nucleus":   (222, 82, 200),
    "pathogen":  (32, 200, 220),
    "organelle": (255, 220, 32),
}

# Per-compartment tuning settings, shown (greyed unless the compartment is the
# chosen object) in that compartment's panel. Each entry is
# ``(key_suffix, label, kind, spin_args)`` where the real setting key is
# ``f"{compartment}_{key_suffix}"`` and kind is one of int/float/bool/method.
# spin_args = (min, max, default) for int/float; ignored otherwise.
COMPARTMENT_FIELDS = (
    ("min_area",                   "Min area (px²)",        "int",   (0, 100_000_000, 0)),
    ("max_area",                   "Max area (px²)",        "int",   (0, 100_000_000, 0)),
    ("min_object_area",            "Min object area",       "int",   (0, 100_000_000, 100)),
    ("min_distance",               "Min distance",          "int",   (0, 100_000, 10)),
    ("area_multiplier",            "Area multiplier",       "float", (0.0, 1000.0, 2.0)),
    ("perimeter_fraction",         "Perimeter fraction",    "float", (0.0, 1.0, 0.5)),
    ("min_intensity_percentile",   "Min intensity pct",     "int",   (0, 100, 1)),
    ("max_intensity_percentile",   "Max intensity pct",     "int",   (0, 100, 99)),
    ("intensity_percentile",       "Intensity percentile",  "int",   (0, 100, 50)),
    ("intensity_threshold_method", "Intensity threshold",   "method", None),
    ("intensity_merge",            "Intensity merge",       "bool",  None),
    ("intensity_split",            "Intensity split",       "bool",  None),
    ("remove_border_objects",      "Remove border objects", "bool",  None),
)

# Threshold-method choices (see spacr.utils intensity-merge logic).
INTENSITY_THRESHOLD_METHODS = ("mean", "percentile")


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


def _full_range_max(img: np.ndarray) -> float:
    """Return the value that maps to white for a *raw* (un-normalised) view.

    For integer images this is the dtype maximum (e.g. 65535 for uint16), so
    a 16-bit image whose real values are small reads dark — the true raw
    view. For float images we assume a [0, 1] range unless the data clearly
    exceeds it, in which case we use the data max.
    """
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    m = float(np.nanmax(img)) if img.size else 1.0
    return 1.0 if m <= 1.0 else m


def _to_uint8(img: np.ndarray, normalise: bool = True,
                lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Return a viewable uint8 version of *img*.

    :param normalise: when True apply a per-channel percentile stretch. When
        False, map the *full bit-depth range* (0 → dtype max) to 0–255, i.e.
        the raw view — a 16-bit image with small values reads dark/black,
        not blown out. (Previously this clipped to [0, 255], which turned a
        16-bit image mostly white.)
    :param lo_pct: lower percentile for the stretch (default 2 %).
    :param hi_pct: upper percentile for the stretch (default 98 %).
    """
    full_max = _full_range_max(img) or 1.0
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
                out[..., c] = np.clip(
                    255 * slice_ / full_max, 0, 255).astype(np.uint8)
        return out
    arr = img.astype(np.float32)
    if normalise:
        lo, hi = np.percentile(arr, (lo_pct, hi_pct))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip(
            255 * (arr - lo) / (hi - lo), 0, 255,
        ).astype(np.uint8)
    return np.clip(255 * arr / full_max, 0, 255).astype(np.uint8)


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
    flows_ready = Signal(object)           # {obj: flow_rgb} (may be empty)

    def __init__(self, request: PreviewRequest, parent=None):
        super().__init__(parent)
        self._request = request

    def run(self):
        try:
            res = _segment_multi(self._request)
            # _segment_multi may return masks only (the stubbed test path) or
            # (masks, flows). Handle both.
            if isinstance(res, tuple):
                masks, flows = res
            else:
                masks, flows = res, {}
            self.finished_masks.emit(masks, "")
            self.flows_ready.emit(flows or {})
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
    flows_out: Dict[str, np.ndarray] = {}
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

        # Capture the RGB flow visualisation (flows[0]) if Cellpose returned
        # one, so the panel can show a Flows view alongside the masks.
        try:
            flows = result[1]
            flow_rgb = flows[0] if isinstance(flows, (list, tuple)) else flows
            if isinstance(flow_rgb, list):
                flow_rgb = flow_rgb[0]
            flows_out[obj] = np.asarray(flow_rgb)
        except Exception:
            pass

        # Return the RAW (unfiltered) mask — the panel applies the per-
        # compartment filters afterwards so the user can re-tune filters
        # without re-running Cellpose (see LivePreviewPanel._recompute_masks).
        out[obj] = mask
    return out, flows_out


def _select_channel(image: np.ndarray, ch: int) -> np.ndarray:
    """Return a 2-D slice from ``image`` at channel index ``ch``."""
    if image.ndim == 3 and image.shape[-1] > 1:
        return image[..., int(ch) % image.shape[-1]]
    return image.squeeze()


def _apply_size_filter(mask: np.ndarray,
                          settings: Dict[str, Any],
                          obj: str,
                          intensity_img: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the *same* post-segmentation filters the pipeline uses, so the
    live preview matches a real run.

    Reads the per-compartment knobs (``{obj}_min_area``, ``{obj}_max_area``,
    ``{obj}_remove_border_objects``, ``{obj}_min_intensity_percentile``,
    ``{obj}_max_intensity_percentile``) — the exact keys the compartment
    panels write — and runs them through :func:`spacr.utils._filter_objects`.
    Legacy ``{obj}_min_size``/``{obj}_max_size`` are honoured as a fallback.
    No-ops when nothing is set."""
    if not settings or mask is None:
        return mask

    def _num(key, default):
        v = settings.get(key, default)
        try:
            return type(default)(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    min_area = _num(f"{obj}_min_area", _num(f"{obj}_min_size", 0))
    max_area = _num(f"{obj}_max_area", _num(f"{obj}_max_size", 0))
    remove_border = bool(settings.get(f"{obj}_remove_border_objects", False))
    min_ip = _num(f"{obj}_min_intensity_percentile", 0)
    max_ip = _num(f"{obj}_max_intensity_percentile", 100)

    if not (min_area > 0 or max_area > 0 or remove_border
            or min_ip > 0 or max_ip < 100):
        return mask

    try:
        from spacr.utils import _filter_objects
        return _filter_objects(
            mask.astype(np.uint16).copy(),
            intensity_img=intensity_img,
            min_area=int(min_area), max_area=int(max_area),
            remove_border=remove_border,
            min_intensity_percentile=float(min_ip),
            max_intensity_percentile=float(max_ip),
        ).astype(mask.dtype)
    except Exception:
        LOG.debug("size filter failed", exc_info=True)
        return mask


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
        self._raw_masks: Dict[str, np.ndarray] = {}
        self._flows: Dict[str, np.ndarray] = {}
        self._settings: Dict[str, Any] = {}
        self._worker: Optional[_PreviewWorker] = None
        # Callback(dict) that pushes tuned live settings into the main panel.
        self._propagate_cb = None
        self._build_ui()
        self._build_compartment_widgets()
        # Accept image files dropped anywhere on the panel. QGraphicsView
        # enables acceptDrops by default and would otherwise swallow drops
        # over the image canvases; turning it off on the views lets the drag
        # events propagate up to this panel's handlers.
        self.setAcceptDrops(True)
        for _v in (getattr(self, "_src_view", None),
                   getattr(self, "_mask_view", None)):
            if _v is not None:
                _v.setAcceptDrops(False)

    # -- drag & drop -------------------------------------------------------

    _DND_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
                 ".gif", ".webp")

    def _dropped_image_path(self, event) -> Optional[str]:
        """Return the first dropped local image path, or None."""
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if Path(p).suffix.lower() in self._DND_EXTS:
                return p
        return None

    def dragEnterEvent(self, event):    # noqa: N802 (Qt naming)
        """Accept the drag only if it carries a supported image file."""
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):     # noqa: N802
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):         # noqa: N802
        """Load the dropped image into the preview."""
        path = self._dropped_image_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.load_image(path)

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

        # Tooltips for the segmentation controls (type + what they do).
        self._model_box.setToolTip(
            "(str) Cellpose model: cpsam (SAM, auto-diameter), cyto3/cyto2 "
            "(cytoplasm), nuclei.")
        self._object_box.setToolTip(
            "(str) Object(s) to segment. 'cell + nucleus' runs both passes.")
        self._cell_channel.setToolTip(
            "(int) Image channel index used for cell segmentation.")
        self._nucleus_channel.setToolTip(
            "(int) Image channel index used for nucleus segmentation.")
        self._diameter.setToolTip(
            "(float, px) Expected object diameter. Ignored by Cellpose-SAM "
            "(cpsam), which estimates it automatically.")
        self._flow.setToolTip(
            "(float) Cellpose flow threshold — higher keeps more masks.")
        self._prob.setToolTip(
            "(float) Cellpose cell-probability threshold — lower keeps more "
            "(dimmer) objects.")
        self._normalise_check.setToolTip(
            "(bool) Percentile-normalise the image for display + segmentation.")
        self._lo_pct.setToolTip(
            "(float, %) Lower percentile for normalisation.")
        self._hi_pct.setToolTip(
            "(float, %) Upper percentile for normalisation.")
        self._outline_colour.setToolTip(
            "(str) Overlay outline colour ('auto' = per-object colour).")
        self._outline_thickness.setToolTip(
            "(int, px) Overlay outline thickness.")

        # Keep every hidden helper widget parented but invisible so it
        # doesn't render in the compact layout.
        for w in (self._model_box, self._object_box,
                    self._cell_channel, self._nucleus_channel,
                    self._diameter, self._flow, self._prob,
                    self._normalise_check, self._lo_pct, self._hi_pct,
                    self._outline_colour, self._outline_thickness):
            w.hide()

        # -- VISIBLE compact layout --------------------------------------

        # File picker row
        pick_row = QHBoxLayout()
        self._path_label = QLabel(
            "No preview image loaded — drag & drop an image here to load it",
            self)
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
        # What the right-hand canvas shows: outline overlay, the raw label
        # mask, or the Cellpose flow field.
        self._view_mode = QComboBox(self)
        self._view_mode.addItems(["Overlay", "Masks", "Flows"])
        self._view_mode.setToolTip(
            "Right canvas: outline overlay · label masks · Cellpose flows")
        self._view_mode.currentTextChanged.connect(
            lambda *_: self._refresh_canvases())
        self._status = QLabel("", self)
        act.addWidget(self._run_btn)
        act.addWidget(self._live_settings_btn)
        act.addWidget(QLabel("View:", self))
        act.addWidget(self._view_mode)
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
        self._hover_label.setStyleSheet("color: #ffffff; "
                                            "font-family: monospace;")
        root.addWidget(self._hover_label)

        # Comparison scrubber — scrub back/forth through previous preview runs
        # to compare how different settings changed the segmentation. Hidden
        # until at least two runs exist.
        from PySide6.QtWidgets import QSlider
        self._history: list = []
        self._compare_row = QWidget(self)
        comp = QHBoxLayout(self._compare_row)
        comp.setContentsMargins(0, 0, 0, 0)
        comp.addWidget(QLabel("Compare runs", self))
        self._compare_slider = QSlider(Qt.Horizontal, self)
        self._compare_slider.setMinimum(0)
        self._compare_slider.setMaximum(0)
        self._compare_slider.valueChanged.connect(self._on_compare_scrub)
        comp.addWidget(self._compare_slider, 1)
        self._compare_label = QLabel("", self)
        self._compare_label.setStyleSheet("color: #ffffff; font-family: monospace;")
        comp.addWidget(self._compare_label)
        self._compare_row.setVisible(False)
        root.addWidget(self._compare_row)

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

    def set_propagate_callback(self, cb) -> None:
        """Register a callback(dict) used to push tuned live settings back to
        the main settings panel (wired by the AppScreen)."""
        self._propagate_cb = cb

    def settings_for_propagation(self) -> dict:
        """Map the live-preview widget values to main-panel settings keys."""
        model = self._model_box.currentText()
        out = {
            "model_name": model,
            "cell_channel": int(self._cell_channel.value()),
            "nucleus_channel": int(self._nucleus_channel.value()),
            "cell_diameter": float(self._diameter.value()),
            "cell_FT": float(self._flow.value()),
            "cell_CP_prob": float(self._prob.value()),
            "normalize": bool(self._normalise_check.isChecked()),
            "lower_percentile": float(self._lo_pct.value()),
        }
        # Per-compartment + common tuning settings (only present once the
        # compartment widgets have been built).
        if hasattr(self, "_compartment_widgets"):
            out.update(self._compartment_settings())
        return out

    def propagate_settings(self) -> None:
        """Send the current live settings to the main panel (if a callback is
        registered). Called on any live-settings change while the dialog's
        Propagate toggle is on."""
        if self._propagate_cb is not None:
            try:
                self._propagate_cb(self.settings_for_propagation())
            except Exception:
                LOG.debug("propagate_settings failed", exc_info=True)

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
        worker.flows_ready.connect(self._on_flows_ready)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    # -- internals ---------------------------------------------------------

    # -- per-compartment tuning widgets -----------------------------------

    def _build_compartment_widgets(self) -> None:
        """Create the common + per-compartment tuning widgets.

        They live on the panel (hidden) so their values persist across opens
        of the Live settings dialog, which re-parents them into its panels and
        hands them back on close — the same pattern the segmentation widgets
        use. Nothing is added to the compact panel layout.

        Populates:
          * ``self._common_widgets`` — signal-to-noise / remove-background /
            background controls that apply to whichever object is chosen.
          * ``self._compartment_widgets[compartment][suffix]`` — the per-
            compartment tuning spinners/checks/combos.
          * ``self._adjust_cells`` — the cell-only "adjust cells" toggle.
        """
        def _spin(kind, spin_args):
            if kind == "float":
                w = QDoubleSpinBox(self)
                lo, hi, dv = spin_args
                w.setRange(float(lo), float(hi)); w.setValue(float(dv))
                w.setDecimals(3)
            elif kind == "int":
                w = QSpinBox(self)
                lo, hi, dv = spin_args
                w.setRange(int(lo), int(hi)); w.setValue(int(dv))
            elif kind == "bool":
                w = QCheckBox(self)
            elif kind == "method":
                w = QComboBox(self)
                w.addItems(list(INTENSITY_THRESHOLD_METHODS))
            else:
                raise ValueError(kind)
            w.hide()
            return w

        # Pull the informative spaCR setting descriptions for tooltips.
        try:
            from spacr.settings import descriptions as _spacr_desc
        except Exception:
            _spacr_desc = {}

        # Common controls — one widget each, retargeted to the chosen object
        # at propagation time (see settings_for_propagation).
        self._common_widgets: Dict[str, QWidget] = {
            "signal_to_noise": _spin("int", (0, 100_000, 10)),
            "remove_background": _spin("bool", None),
            "background": _spin("int", (0, 100_000, 100)),
        }
        self._common_widgets["signal_to_noise"].setToolTip(
            "(int) Signal-to-noise ratio used to set the normalisation "
            "intensity range for the chosen object's channel.")
        self._common_widgets["remove_background"].setToolTip(
            "(bool) Subtract the background intensity from the chosen object's "
            "channel before segmentation.")
        self._common_widgets["background"].setToolTip(
            "(int) Background intensity subtracted from the chosen object's "
            "channel when 'Remove background' is on.")
        # Cell-only extra.
        self._adjust_cells = _spin("bool", None)
        self._adjust_cells.setToolTip(
            "(bool) Adjust cell masks using the nucleus/pathogen masks.")

        self._compartment_widgets: Dict[str, Dict[str, QWidget]] = {}
        for comp in COMPARTMENTS:
            group: Dict[str, QWidget] = {}
            for suffix, label, kind, spin_args in COMPARTMENT_FIELDS:
                w = _spin(kind, spin_args)
                key = f"{comp}_{suffix}"
                desc = _spacr_desc.get(key) or _spacr_desc.get(suffix)
                w.setToolTip(desc if desc else f"{label} for {comp} objects.")
                group[suffix] = w
            self._compartment_widgets[comp] = group

        # Re-filter the cached masks live whenever any filter widget changes,
        # so tuning updates the preview instantly (no Cellpose re-run).
        for w in self._all_compartment_widgets():
            for sig_name in ("valueChanged", "currentTextChanged", "toggled"):
                sig = getattr(w, sig_name, None)
                if sig is not None:
                    try:
                        sig.connect(lambda *_: self._recompute_masks())
                    except (TypeError, RuntimeError):
                        pass

    def _all_compartment_widgets(self) -> List[QWidget]:
        ws: List[QWidget] = list(self._common_widgets.values())
        ws.append(self._adjust_cells)
        for group in self._compartment_widgets.values():
            ws.extend(group.values())
        return ws

    def _primary_object(self) -> str:
        """The compartment the common controls target — the first selected."""
        return self._selected_object_types()[0]

    @staticmethod
    def _widget_value(w):
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.value()

    def _compartment_settings(self) -> dict:
        """Map every compartment + common tuning widget to its setting key."""
        out: dict = {}
        for comp, group in self._compartment_widgets.items():
            for suffix, w in group.items():
                out[f"{comp}_{suffix}"] = self._widget_value(w)
        obj = self._primary_object()
        out[f"{obj}_Signal_to_noise"] = self._widget_value(
            self._common_widgets["signal_to_noise"])
        out[f"remove_background_{obj}"] = self._widget_value(
            self._common_widgets["remove_background"])
        out[f"{obj}_background"] = self._widget_value(
            self._common_widgets["background"])
        out["adjust_cells"] = self._widget_value(self._adjust_cells)
        return out

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
        # One unified settings dict drives both background subtraction
        # (pre) and filtering (post): the common "remove background" +
        # "background" controls and the per-compartment filter values. No more
        # Pre/Post checkboxes — the settings apply whenever they're set.
        merged = dict(self._settings)
        if hasattr(self, "_compartment_widgets"):
            merged.update(self._compartment_settings())
        pre = merged
        post = merged
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

        mode = self._view_mode.currentText() if hasattr(self, "_view_mode") else "Overlay"
        if mode == "Flows" and self._flows:
            self._mask_view.set_pixmap(numpy_to_qpixmap(
                self._flows_rgb()))
        elif mode == "Masks" and self._masks:
            self._mask_view.set_pixmap(numpy_to_qpixmap(
                self._label_rgb()))
        elif self._masks:   # Overlay (default)
            overlay = overlay_masks(
                self._image, self._masks,
                outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi)
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)

    def _on_flows_ready(self, flows) -> None:
        """Store the per-object Cellpose flow RGB images from a preview run."""
        self._flows = flows or {}
        if hasattr(self, "_view_mode") and self._view_mode.currentText() == "Flows":
            self._refresh_canvases()

    def _label_rgb(self) -> np.ndarray:
        """Render the current label masks as a distinct-colour image (0 = black)."""
        h, w = self._image.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for obj, mask in self._masks.items():
            if mask is None or mask.shape[:2] != (h, w):
                continue
            base = np.array(OBJECT_COLORS.get(obj, (200, 200, 200)), dtype=np.uint8)
            # Vary brightness a little per label so neighbours are separable.
            labels = mask.astype(np.int64)
            present = labels > 0
            if not present.any():
                continue
            shade = (0.5 + 0.5 * ((labels % 7) / 6.0)).astype(np.float32)
            for c in range(3):
                out[..., c] = np.where(
                    present,
                    np.clip(base[c] * shade, 0, 255).astype(np.uint8),
                    out[..., c])
        return out

    def _flows_rgb(self) -> np.ndarray:
        """Combine per-object flow RGB images (first available / max-blend)."""
        imgs = [np.asarray(f) for f in self._flows.values()
                if f is not None and np.asarray(f).ndim == 3]
        if not imgs:
            h, w = self._image.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        out = imgs[0].astype(np.uint8)
        for f in imgs[1:]:
            if f.shape == out.shape:
                out = np.maximum(out, f.astype(np.uint8))
        return out[..., :3]

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
        # Cache the raw masks so filters can be re-applied live, then filter.
        self._raw_masks = masks
        self._recompute_masks(snapshot=True)

    def _obj_channel(self, obj: str) -> int:
        """Intensity channel index used for a given compartment."""
        if obj == "cell":
            return int(self._cell_channel.value())
        if obj == "nucleus":
            return int(self._nucleus_channel.value())
        return 0

    def _recompute_masks(self, snapshot: bool = False) -> None:
        """Re-apply the current per-compartment filters to the cached raw
        masks and refresh the views — no Cellpose re-run. Called both after a
        preview and whenever a filter widget changes."""
        raw = getattr(self, "_raw_masks", None)
        if not raw or self._image is None:
            return
        post = dict(self._settings)
        if hasattr(self, "_compartment_widgets"):
            post.update(self._compartment_settings())
        out = {}
        for obj, raw_mask in raw.items():
            intensity = _select_channel(self._image, self._obj_channel(obj))
            out[obj] = _apply_size_filter(raw_mask, post, obj,
                                          intensity_img=intensity)
        self._masks = out
        counts = [f"{k}={int(v.max() if v.size else 0)}"
                    for k, v in out.items()]
        self._status.setText(f"Found {', '.join(counts)}.")
        self._refresh_canvases()
        if snapshot:
            self._snapshot_run(out, counts)
        self.preview_ready.emit(out)

    # -- comparison scrubber ----------------------------------------------

    def _snapshot_run(self, masks, counts) -> None:
        """Record a preview run (image + masks + display params) so the user
        can scrub back to compare it against later runs."""
        if self._image is None:
            return
        snap = {
            "image": self._image,
            "masks": {k: v for k, v in masks.items()},
            "norm": self._normalise_check.isChecked(),
            "lo": float(self._lo_pct.value()),
            "hi": float(self._hi_pct.value()),
            "model": self._model_box.currentText(),
            "object": self._object_box.currentText(),
            "summary": ", ".join(counts),
        }
        self._history.append(snap)
        # Cap the history so memory stays bounded on long tuning sessions.
        if len(self._history) > 50:
            self._history = self._history[-50:]
        n = len(self._history)
        self._compare_row.setVisible(n >= 2)
        self._compare_slider.blockSignals(True)
        self._compare_slider.setMaximum(n - 1)
        self._compare_slider.setValue(n - 1)      # newest
        self._compare_slider.blockSignals(False)
        self._compare_label.setText(f"{n}/{n}")

    def _on_compare_scrub(self, idx: int) -> None:
        """Render the historical run at ``idx`` into the two canvases."""
        if not (0 <= idx < len(self._history)):
            return
        snap = self._history[idx]
        img = snap["image"]
        norm, lo, hi = snap["norm"], snap["lo"], snap["hi"]
        src_pix = numpy_to_qpixmap(
            _to_uint8(img, normalise=norm, lo_pct=lo, hi_pct=hi))
        self._src_view.set_pixmap(src_pix)
        if snap["masks"]:
            overlay = overlay_masks(
                img, snap["masks"], outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi)
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)
        self._compare_label.setText(
            f"{idx + 1}/{len(self._history)}  "
            f"{snap['model']}/{snap['object']}  {snap['summary']}")


# ---------------------------------------------------------------------------
# Live Settings dialog
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QScrollArea,
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
        outer = QVBoxLayout(self)

        # Show the widgets we'll be adding, then re-hide them on close.
        for w in self._managed_widgets():
            w.show()

        # Row of side-by-side panels: the segmentation + common controls on the
        # left, then one greyed-until-chosen panel per compartment to the right.
        panels_row = QHBoxLayout()
        panels_row.setSpacing(12)

        seg_group = QGroupBox("Segmentation")
        form = QFormLayout(seg_group)
        form.addRow("Model", panel._model_box)
        form.addRow("Primary object", panel._object_box)
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
        # Common controls — apply to whichever object is chosen.
        panel._common_widgets["signal_to_noise"].show()
        panel._common_widgets["remove_background"].show()
        panel._common_widgets["background"].show()
        form.addRow("Signal to noise", panel._common_widgets["signal_to_noise"])
        form.addRow("Remove background", panel._common_widgets["remove_background"])
        form.addRow("Background", panel._common_widgets["background"])
        panels_row.addWidget(seg_group)

        # One panel per compartment, greyed unless it's the chosen object.
        self._compartment_groupboxes: Dict[str, QGroupBox] = {}
        for comp in COMPARTMENTS:
            box = QGroupBox(comp.capitalize())
            cform = QFormLayout(box)
            for suffix, label, _kind, _args in COMPARTMENT_FIELDS:
                w = panel._compartment_widgets[comp][suffix]
                w.show()
                cform.addRow(label, w)
            if comp == "cell":
                panel._adjust_cells.show()
                cform.addRow("Adjust cells", panel._adjust_cells)
            self._compartment_groupboxes[comp] = box
            panels_row.addWidget(box)

        # Wrap the (wide) panel row in a horizontal scroll area so it fits on
        # screen no matter how many compartments are shown.
        row_host = QWidget()
        row_host.setLayout(panels_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(row_host)
        outer.addWidget(scroll, 1)

        # Run button lives in the dialog so settings can be iterated without
        # closing it — edit a value, hit Run, see the result, repeat.
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._run_btn = QPushButton("Run preview")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._panel.run_preview)
        buttons.addButton(self._run_btn, QDialogButtonBox.ActionRole)
        # Propagate toggle — when on (blue, like the AI / LP toggles), edits
        # here are pushed into the main settings panel so tuning in the live
        # preview updates the run configuration.
        self._propagate_btn = QPushButton("Propagate settings")
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, changes made here are copied into the main settings "
            "panel.")
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        buttons.addButton(self._propagate_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)

        # Re-gate the form whenever the object type or model changes, so
        # irrelevant settings grey out live.
        panel._object_box.currentTextChanged.connect(self.refresh_visibility)
        panel._model_box.currentTextChanged.connect(self.refresh_visibility)
        panel._normalise_check.toggled.connect(self.refresh_visibility)

        # Widgets whose changes propagate to the main panel while the toggle
        # is on — the segmentation controls plus every compartment/common knob.
        self._propagate_sources = [
            panel._model_box, panel._object_box, panel._cell_channel,
            panel._nucleus_channel, panel._diameter, panel._flow,
            panel._prob, panel._normalise_check, panel._lo_pct, panel._hi_pct,
        ] + panel._all_compartment_widgets()

        self.refresh_visibility()

        # Open wide enough to show the Segmentation panel + all four compartment
        # panels without the user having to drag the window wider. Clamp to the
        # available screen so it still fits on small displays (the horizontal
        # scroll area handles any remaining overflow).
        try:
            avail = self.screen().availableGeometry()
            want = row_host.sizeHint().width() + 48
            self.resize(min(want, avail.width() - 80), min(760, avail.height() - 80))
        except Exception:
            self.resize(1400, 720)

    def _on_propagate_toggled(self, on: bool) -> None:
        """Connect/disconnect live→main propagation and do an initial push."""
        for w in self._propagate_sources:
            for sig_name in ("valueChanged", "currentTextChanged", "toggled"):
                sig = getattr(w, sig_name, None)
                if sig is None:
                    continue
                try:
                    if on:
                        sig.connect(self._panel.propagate_settings)
                    else:
                        sig.disconnect(self._panel.propagate_settings)
                except (TypeError, RuntimeError):
                    pass
        if on:
            self._panel.propagate_settings()   # push current values now

    def _managed_widgets(self):
        p = self._panel
        return [p._model_box, p._object_box, p._cell_channel,
                p._nucleus_channel, p._diameter, p._flow, p._prob,
                p._normalise_check, p._lo_pct, p._hi_pct,
                p._outline_colour, p._outline_thickness,
                ] + p._all_compartment_widgets()

    def refresh_visibility(self):
        """Grey out settings that don't apply to the current selection.

        Rules (mirroring the pipeline's own relevance):
          * Segmentation knobs (diameter / flow / cell-prob) are ignored by
            Cellpose-SAM, so they grey out when the SAM model is picked.
          * The object type decides which channel spinners are live: the cell
            channel greys out for a nucleus-only object and vice-versa.
          * Pre-processing knobs (normalise + its two percentiles) are only
            relevant when the *Pre* step is enabled.
          * Overlay / post knobs (outline colour + thickness) are only
            relevant when the *Post* step is enabled.
        """
        p = self._panel

        # -- model: SAM still uses flow threshold + cell probability; only the
        #    diameter is ignored (SAM auto-estimates object size) --
        is_sam = p._model_box.currentText() == "cpsam"
        p._diameter.setEnabled(not is_sam)
        p._diameter.setToolTip("Ignored by Cellpose-SAM" if is_sam else "")
        p._flow.setEnabled(True)
        p._prob.setEnabled(True)
        p._flow.setToolTip("")
        p._prob.setToolTip("")

        # -- object: which channel spinners apply --
        selected = set(p._selected_object_types())
        p._cell_channel.setEnabled("cell" in selected)
        p._nucleus_channel.setEnabled("nucleus" in selected)

        # -- compartment panels: show only the primary object's panel plus,
        #    for 'cell + nucleus', a secondary Nucleus panel. The other
        #    compartments' panels are hidden entirely (their settings are the
        #    same shape and only the chosen object's are relevant). --
        ordered = list(p._selected_object_types())
        primary = ordered[0] if ordered else "cell"
        for comp, box in self._compartment_groupboxes.items():
            is_primary = (comp == primary)
            is_secondary = (comp == "nucleus" and "nucleus" in selected
                            and not is_primary)
            box.setVisible(is_primary or is_secondary)
            box.setEnabled(True)
            if is_primary:
                box.setTitle(f"{comp.capitalize()} (primary object)")
            elif is_secondary:
                box.setTitle("Nucleus (secondary object)")

        # -- Normalisation is always available (independent of the Pre step
        #    and of the model, incl. cpsam). The percentile bounds only apply
        #    while normalisation is on. --
        p._normalise_check.setEnabled(True)
        p._normalise_check.setToolTip("")
        norm_on = p._normalise_check.isChecked()
        for w in (p._lo_pct, p._hi_pct):
            w.setEnabled(norm_on)
            w.setToolTip("" if norm_on
                         else "Enable 'Normalise' to set percentile bounds")

        # Overlay / outline knobs are always relevant (they style the overlay
        # view), so they stay enabled.
        for w in (p._outline_colour, p._outline_thickness):
            w.setEnabled(True)

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
