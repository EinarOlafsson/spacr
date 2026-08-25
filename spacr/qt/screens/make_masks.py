"""
MakeMasksScreen — hand-correct a segmentation, on the record.

Load a folder of images and their masks (in ``<folder>/masks/``), draw
brush/erase strokes, run object-level operations (fill, relabel, invert,
remove small, Otsu detect), zoom and pan into a region for detailed
edits, flood-fill by intensity with the magic wand, undo/redo, and save
the edited mask back to ``<folder>/masks/<name>.tif`` as labelled uint16.

Three things here are less obvious than the tool buttons:

**Every edit is recorded.** The screen keeps a
:class:`spacr.curation.CurationLog` per field, seeded from any ledger
already beside the mask, and :func:`spacr.qt.mask_engine.save_mask`
writes it back. So :func:`spacr.curation.is_curated` can tell a mask that
was corrected by hand from one the pipeline produced, and the ledger says
what was done to it. One gesture is one entry: a right-button sweep over
six objects records a single ``sweep_delete`` of six, not six deletes,
for the same reason it is a single undo step.

**The magic-wand tolerance is a percentage by default.** An absolute
tolerance means two different things on an 8-bit and a 16-bit image; a
percentage of the image's own intensity range means one. See
:func:`spacr.qt.mask_engine.relative_tolerance`.

**The display percentiles carry six decimals.** On a 4-megapixel 16-bit
field the difference between 99.9 and 99.9999 is the difference between
clipping four thousand pixels and clipping four, and a few hot pixels are
often the entire reason a field looks black.

Two tools from the standalone curation tool are still absent: the
dividing line that splits one merged object into two, and the free-form
polygon that fills an outline into one object. Neither has a control on
this panel, so nothing here claims they exist.

THE SEGMENTATION WORKBENCH
--------------------------

Everything a person does to a segmentation happens on one screen, because
they are one job done in a loop: segment the folder, look at the masks,
correct what came out wrong, train on the corrections, segment again. The
modules that used to be rows of their own are buttons on this screen's
masthead — :data:`FOLD_ORDER` — each drawn as its own icon by
:class:`~spacr.qt.widgets.fold_strip.FoldStrip`.

Two of those buttons carry something the folded module did not have:

* **Mask the whole folder** runs the applying half of the Cellpose
  workbench over every image in the folder that is open here, rather than
  asking for the path a second time.
* **Save mask**, on the Curate window, writes the corrected labels.
  Curate paints and records and never wrote a pixel of its own: its
  ledger asserted corrections beside a file the pipeline had produced,
  and :func:`spacr.curation.is_curated` answered ``True`` for that
  untouched file. :meth:`spacr.curation.MaskCuration.save_mask` writes
  the labels and the ledger together, and this button is what presses it.

A folded module is opened as the widget it always was, in a window of its
own (:class:`FoldedModuleDialog`), so nothing it could do is lost on the
way in.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, QThread, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...curation import CurationLog
from .. import iconset
from .. import mask_engine as engine
from .. import prefs
from ..theme import SPACING, active_palette
from ..widgets import Card, Divider, EmptyState
from ..widgets.fold_strip import FoldStrip
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.make_masks")

#: The registry key this screen answers to.
APP_KEY = "make_masks"

#: The masthead, matching the registry row so the page and the tile that
#: opens it say the same thing.
HEADER_TITLE = "Make Masks"
HEADER_DESCRIPTION = (
    "Correct a mask by hand: brush, flood fill, relabel, fill, remove small")
HEADER_INSTRUCTION = (
    "Open a folder of images, correct each mask, and save it back.")

#: The applying half of the Cellpose loop, as a key. It has never had a tile
#: of its own — :class:`~spacr.qt.screens.train_cellpose.CellposeWorkbenchScreen`
#: carries it as a tab — but it has artwork and a settings form under this
#: name, and it is what "mask the whole folder" runs.
MASK_FOLDER_KEY = "cellpose_all"

#: The modules that fold into this screen, in the order their buttons appear
#: on the masthead. The button IS the module, so this is also the list of
#: keys :meth:`MakeMasksScreen.folded_screen` knows how to build.
FOLD_ORDER = (
    "train_cellpose",
    MASK_FOLDER_KEY,
    "model_compare",
    "model_zoo",
    "curate",
    "napari_bridge",
    "timelapse",
    "motility",
)

#: ``key -> (name, description, stage)`` for a folded module whose registry
#: row has gone.
#:
#: :class:`~spacr.qt.widgets.fold_strip.FoldStrip` reads a button's name, its
#: tooltip and its hover colour out of the app registry, which is right while
#: the module still has a row and answers nothing once it is folded and the
#: row is dropped: the tooltip empties and the stage falls back to stable, so
#: an alpha module's button would light blue where its tile lit green-cyan.
#: This is what the tile said, kept so the button can go on saying it.
#:
#: The registry still wins whenever it has the row, and
#: ``test_the_fold_fallback_agrees_with_the_registry`` asserts the two agree
#: for every key that has one — so the pair cannot drift apart while both
#: exist, and what is left after the row goes is what was last true.
FOLD_FALLBACK = {
    "train_cellpose": (
        "Cellpose Workbench",
        "Fine-tune a Cellpose model on your own labelled fields, then "
        "segment a folder of images with it or with a stock model",
        "beta"),
    MASK_FOLDER_KEY: (
        "Mask the whole folder",
        "Run the segmentation model over every image in the open folder",
        "beta"),
    "model_compare": (
        "Model Compare",
        "Two Cellpose models on the same fields: masks side by side, "
        "object-count and ARI deltas",
        "alpha"),
    "model_zoo": (
        "Model Zoo",
        "Browse, verify, download and bench Cellpose + classifier models on "
        "three of your fields",
        "alpha"),
    "curate": (
        "Curate",
        "Paint a mask right, and fix tracks by hand — on the record",
        "alpha"),
    "napari_bridge": (
        "Napari Bridge",
        "Correct a mask in napari and bring the corrected labels back",
        "alpha"),
    "timelapse": (
        "Timelapse",
        "Segment and track objects across the frames of a time series",
        "beta"),
    "motility": (
        "Motility Assay",
        "Automated motility assay: track velocity + infection QC",
        "beta"),
}

#: A folded key that shares another key's screen. The two halves of the
#: Cellpose loop are two tabs of one workbench, so pressing either button has
#: to reach the same widget: a checkpoint trained on one tab is what the
#: other tab segments with, and a second copy of the screen would not have it.
FOLD_HOSTS = {MASK_FOLDER_KEY: "train_cellpose"}

# Qt platform plugins that have no way for a human to click a dialog button.
_HEADLESS_PLATFORMS = ("offscreen", "minimal", "minimalegl", "vnc")


def is_headless() -> bool:
    """Return True when no interactive display is attached to this process.

    A modal ``QMessageBox`` runs its own event loop and only returns once
    somebody clicks a button. Under the ``offscreen`` / ``minimal`` Qt
    platform plugins — CI, a headless server, an SSH session with no X —
    nobody can, so the call never returns and the whole app hangs. Any
    message triggered by *data* rather than by a user gesture therefore
    has to degrade to the status line instead.

    Sibling screens (align / batch / convert / report / plate_view / …)
    solve this by never opening a modal at all — see their ``_set_status``
    docstrings, which cite this screen as the case that actually hung.
    That is not sufficient here because "Clear mask" genuinely needs a
    yes/no answer, so this screen keeps the modal when — and only when —
    there is somebody able to answer it.
    """
    app = QApplication.instance()
    if app is None:
        return True
    try:
        name = str(app.platformName()).strip().lower()
    except Exception:
        return True
    return (not name) or name in _HEADLESS_PLATFORMS


# ---------------------------------------------------------------------------
# Canvas — image + mask overlay with brush/erase mouse handling
# ---------------------------------------------------------------------------

MODE_NONE = "none"
MODE_BRUSH = "brush"
MODE_ERASE = "erase"
MODE_ERASE_OBJECT = "erase_object"
MODE_WAND_ADD = "wand_add"
MODE_WAND_ERASE = "wand_erase"
MODE_ZOOM = "zoom"

# Held with the left button, these pan from ANY tool. Two of them because
# window managers eat one or the other: Alt+drag moves the window on most
# Linux desktops, and Shift+drag is taken by some tablet drivers. Whichever
# one survives on this machine, panning still works without putting the
# brush down.
PAN_MODIFIERS = Qt.ShiftModifier | Qt.AltModifier

#: Smallest zoom viewport, in image pixels. Below a handful of pixels the
#: view is all interpolation and the wheel has nothing left to magnify.
MIN_VIEWPORT = 8

#: Decimals on the display-percentile boxes. Six, because the interesting
#: end of a 16-bit histogram is the last few pixels: on a 2048x2048 field,
#: 99.9999 clips four pixels and 99.9 clips four thousand, and the hot ones
#: are usually the entire reason the field looks black.
PERCENTILE_DECIMALS = 6


class _MaskLoadWorker(QThread):
    """Decode one image/mask pair without blocking Qt's main thread."""

    def __init__(self, folder: str, filename: str, token: int, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.filename = filename
        self.token = token
        self.result = None
        self.error: Optional[Exception] = None

    def run(self) -> None:
        """Load the pair, retaining either the result or original exception."""
        try:
            self.result = engine.load_image_and_mask(
                self.folder, self.filename
            )
        except Exception as exc:
            self.error = exc
            LOG.exception(
                "Failed to load mask source %s from %s",
                self.filename,
                self.folder,
            )


class _MaskCanvas(QLabel):
    """QLabel that displays the composited image+mask (optionally zoomed
    into a sub-region) and captures mouse events for brush / erase /
    magic-wand / erase-object / zoom-rectangle interactions.

    All coordinate math is done against the *full* image; the "zoom
    view" is just a crop of the composited pixmap. Mask edits go
    directly into `self.mask` (with the correct zoom offset applied).
    """

    stroke_started = Signal()      # emitted just before self.mask is mutated
    stroke_finished = Signal()     # emitted after a stroke completes
    zoom_changed = Signal(bool)    # emitted with True when zoom entered / False on reset

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None       # uint16 grayscale
        self.mask: Optional[np.ndarray] = None        # uint8 labels
        self.mode: str = MODE_NONE
        self.brush_radius: int = 10
        self.norm_lo: float = 1.0
        self.norm_hi: float = 99.9
        self.wand_tolerance: float = 1000.0
        self.wand_relative: bool = True
        self.wand_tol_pct: float = 5.0
        self.wand_max_pixels: int = 100_000
        self.zoom_speed: float = 1.15

        #: What the stroke that just finished did — ``{"kind", "target",
        #: "detail"}`` — for the screen to put in the curation ledger. Set
        #: by :meth:`_emit_stroke_end` immediately before ``stroke_finished``
        #: so a handler reads the edit it was told about, not the one before.
        self.last_edit: Optional[dict] = None

        # Zoom viewport in image coords; None = full-image view.
        self._zoom_x0: Optional[int] = None
        self._zoom_y0: Optional[int] = None
        self._zoom_x1: Optional[int] = None
        self._zoom_y1: Optional[int] = None

        # Zoom-rectangle drag state (widget-local pixel coords)
        self._zoom_drag_start: Optional[QPoint] = None
        self._zoom_drag_end: Optional[QPoint] = None

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {active_palette()['bg']};")
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)
        self._last_pt: Optional[QPoint] = None
        self._stroke_in_progress = False

        # Right-button sweep-delete: one gesture, one undo step, one ledger
        # entry naming every object it took out.
        self._sweeping = False
        self._sweep_labels: List[int] = []

        # Shift/Alt + left-drag pan, in widget coords.
        self._pan_from: Optional[QPoint] = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def set_image_and_mask(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Load a new image + mask pair and rerender at full-image zoom.

        :param image: uint16 grayscale array to display underneath.
        :param mask: uint8/uint16 label array painted on top.
        """
        self.image = image
        self.mask = mask
        self.reset_zoom(silent=True)
        self.refresh()

    def _viewport_bounds(self):
        """Return (x0, y0, x1, y1) — inclusive-of-x0, exclusive-of-x1."""
        if self.mask is None:
            return (0, 0, 0, 0)
        if self._zoom_x0 is not None:
            return (self._zoom_x0, self._zoom_y0, self._zoom_x1, self._zoom_y1)
        h, w = self.mask.shape[:2]
        return (0, 0, w, h)

    def is_zoomed(self) -> bool:
        """Return True when the canvas is viewing a zoomed sub-region."""
        return self._zoom_x0 is not None

    def reset_zoom(self, silent: bool = False) -> None:
        """Clear the zoom viewport and rerender the full image.

        :param silent: suppress the ``zoom_changed`` signal when True
            (used on image load so no callback fires spuriously).
        """
        was_zoomed = self.is_zoomed()
        self._zoom_x0 = self._zoom_y0 = self._zoom_x1 = self._zoom_y1 = None
        self._zoom_drag_start = self._zoom_drag_end = None
        if was_zoomed and not silent:
            self.zoom_changed.emit(False)
        self.refresh()

    def refresh(self) -> None:
        """Recompose image + mask overlay and repaint the canvas pixmap."""
        if self.image is None or self.mask is None:
            return
        img = engine.normalize_uint16(self.image, self.norm_lo, self.norm_hi)
        x0, y0, x1, y1 = self._viewport_bounds()
        sub_img = img[y0:y1, x0:x1]
        sub_mask = self.mask[y0:y1, x0:x1]
        composed = engine.overlay_mask(sub_img, sub_mask, alpha=0.5)
        h, w = composed.shape[:2]
        if w <= 0 or h <= 0:
            return
        qimg = QImage(composed.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        avail_w = max(200, self.width())
        avail_h = max(200, self.height())
        pixmap = pixmap.scaled(avail_w, avail_h,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
        self.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Coordinate mapping (widget-local px  ↔  full image px)
    # ------------------------------------------------------------------
    def _canvas_to_image(self, x: float, y: float) -> Optional[tuple]:
        # NB: QLabel.pixmap() returns a *null* QPixmap (never None) when no
        # pixmap is set, so the emptiness test has to be isNull().
        p = self.pixmap()
        if self.mask is None or p is None or p.isNull():
            return None
        pw, ph = p.width(), p.height()
        w, h = self.width(), self.height()
        ox = (w - pw) // 2
        oy = (h - ph) // 2
        cx, cy = float(x) - ox, float(y) - oy
        if not (0 <= cx < pw and 0 <= cy < ph):
            return None
        x0, y0, x1, y1 = self._viewport_bounds()
        sub_w = max(1, x1 - x0)
        sub_h = max(1, y1 - y0)
        img_x = int(x0 + cx * sub_w / pw)
        img_y = int(y0 + cy * sub_h / ph)
        # Clamp to image bounds
        img_x = max(0, min(self.mask.shape[1] - 1, img_x))
        img_y = max(0, min(self.mask.shape[0] - 1, img_y))
        return img_x, img_y

    def _image_delta(self, dx_px: float, dy_px: float) -> tuple:
        """Widget-pixel drag -> the image-pixel shift the viewport must take.

        Negated, because a pan moves the *view*, not the picture: dragging
        the content to the right has to slide the window left over the
        image for the pixel under the cursor to stay under the cursor.
        """
        p = self.pixmap()
        if self.mask is None or p is None or p.isNull() or not p.width():
            return (0, 0)
        x0, y0, x1, y1 = self._viewport_bounds()
        return (int(round(-dx_px * (x1 - x0) / p.width())),
                int(round(-dy_px * (y1 - y0) / p.height())))

    def pan_by(self, dx: int, dy: int) -> bool:
        """Slide the zoom viewport by (dx, dy) image px; True if it moved.

        Clamped to the image, so a pan cannot walk the view off the edge and
        leave the user looking at nothing with no way back but Reset zoom.
        Panning an unzoomed canvas does nothing and says so by returning
        False: the whole image is already on screen, there is nowhere to go.
        """
        if self.mask is None or not self.is_zoomed():
            return False
        h, w = self.mask.shape[:2]
        view_w = self._zoom_x1 - self._zoom_x0
        view_h = self._zoom_y1 - self._zoom_y0
        new_x0 = max(0, min(w - view_w, self._zoom_x0 + int(dx)))
        new_y0 = max(0, min(h - view_h, self._zoom_y0 + int(dy)))
        if new_x0 == self._zoom_x0 and new_y0 == self._zoom_y0:
            return False
        self._zoom_x0, self._zoom_x1 = new_x0, new_x0 + view_w
        self._zoom_y0, self._zoom_y1 = new_y0, new_y0 + view_h
        self.refresh()
        return True

    def zoom_at(self, img_x: int, img_y: int, factor: float) -> None:
        """Scale the viewport by ``factor`` about the image point given.

        ``factor > 1`` magnifies. The point under the cursor keeps its place
        in the view, which is what makes wheel-zoom feel like it is aimed at
        something rather than at the middle of the window. Zooming back out
        past the whole image resets to the full-image view instead of
        letting the viewport grow beyond the data.
        """
        if self.mask is None or factor <= 0:
            return
        h, w = self.mask.shape[:2]
        x0, y0, x1, y1 = self._viewport_bounds()
        view_w, view_h = max(1, x1 - x0), max(1, y1 - y0)
        new_w = min(w, max(min(MIN_VIEWPORT, w), int(round(view_w / factor))))
        new_h = min(h, max(min(MIN_VIEWPORT, h), int(round(view_h / factor))))
        if new_w >= w and new_h >= h:
            self.reset_zoom()
            return
        frac_x = (img_x - x0) / view_w
        frac_y = (img_y - y0) / view_h
        new_x0 = max(0, min(w - new_w, int(round(img_x - frac_x * new_w))))
        new_y0 = max(0, min(h - new_h, int(round(img_y - frac_y * new_h))))
        was_zoomed = self.is_zoomed()
        self._zoom_x0, self._zoom_x1 = new_x0, new_x0 + new_w
        self._zoom_y0, self._zoom_y1 = new_y0, new_y0 + new_h
        if not was_zoomed:
            self.zoom_changed.emit(True)
        self.refresh()

    def wheelEvent(self, event):
        """Zoom about the cursor, ``zoom_speed`` per notch, from any tool.

        The speed is adjustable because one step size does not suit both
        jobs: finding a cell in a 4k field wants big jumps, trimming its
        boundary wants a step small enough that the next notch does not
        overshoot the object.
        """
        if self.mask is None:
            return super().wheelEvent(event)
        notches = event.angleDelta().y()
        if not notches:
            return super().wheelEvent(event)
        speed = max(1.001, float(self.zoom_speed))
        factor = speed if notches > 0 else 1.0 / speed
        anchor = self._canvas_to_image(event.position().x(),
                                        event.position().y())
        if anchor is None:
            x0, y0, x1, y1 = self._viewport_bounds()
            anchor = ((x0 + x1) // 2, (y0 + y1) // 2)
        self.zoom_at(anchor[0], anchor[1], factor)
        event.accept()

    def effective_wand_tolerance(self) -> float:
        """The tolerance the wand will actually flood with, right now.

        Relative by default: a percentage of this image's own intensity
        range, so the same setting behaves the same on 8-bit and 16-bit
        data. Switching ``wand_relative`` off restores a plain absolute
        value for the case where somebody knows the exact grey-level
        distance they want.
        """
        if self.wand_relative and self.image is not None:
            return engine.relative_tolerance(self.image, self.wand_tol_pct)
        return float(self.wand_tolerance)

    def _mask_radius_for_brush(self) -> int:
        """Scale the brush radius (in screen px) to full-image px, taking
        the current zoom into account."""
        p = self.pixmap()
        if self.mask is None or p is None or p.isNull():
            return self.brush_radius
        x0, _, x1, _ = self._viewport_bounds()
        sub_w = max(1, x1 - x0)
        return max(1, int(self.brush_radius * sub_w / p.width()))

    # ------------------------------------------------------------------
    # Painting (adds a zoom-rectangle overlay while dragging)
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        """Draw the base pixmap plus a dashed zoom-rectangle when dragging."""
        super().paintEvent(event)
        if self.mode != MODE_ZOOM:
            return
        if self._zoom_drag_start is None or self._zoom_drag_end is None:
            return
        painter = QPainter(self)
        pen = QPen(QColor(active_palette()["accent"]))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        rect = QRect(self._zoom_drag_start, self._zoom_drag_end).normalized()
        painter.drawRect(rect)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def _emit_stroke_start(self):
        if not self._stroke_in_progress:
            self._stroke_in_progress = True
            self.stroke_started.emit()

    def _emit_stroke_end(self, kind: str = "paint", target=None, **detail):
        """Close the open stroke, naming what it did for the ledger.

        A no-op when no stroke is open, so the release handler can call it
        unconditionally after a tool (erase-object, wand) that already
        closed its own stroke on press — without that guard the release
        would overwrite :attr:`last_edit` with a second, empty description
        of an edit that has already been recorded.
        """
        if not self._stroke_in_progress:
            return
        self._stroke_in_progress = False
        self.last_edit = {"kind": str(kind), "target": target,
                           "detail": dict(detail)}
        self.stroke_finished.emit()

    # ------------------------------------------------------------------
    # Right-button sweep-delete
    # ------------------------------------------------------------------
    def _sweep_delete_at(self, pt) -> bool:
        """Delete the object under ``pt`` as part of the open sweep.

        The stroke is opened here, on the first object actually hit, rather
        than on the button press: a right-click that lands on background has
        then changed nothing and leaves no undo step and no ledger entry to
        step back through.
        """
        if self.mask is None or pt is None:
            return False
        x, y = pt
        h, w = self.mask.shape[:2]
        if not (0 <= y < h and 0 <= x < w) or int(self.mask[y, x]) <= 0:
            return False
        self._emit_stroke_start()
        removed = engine.erase_object_in_place(self.mask, x, y)
        if removed and removed not in self._sweep_labels:
            self._sweep_labels.append(removed)
        self.refresh()
        return True

    def mousePressEvent(self, event):
        """Dispatch a click to the current tool (brush/erase/wand/zoom/…).

        Two gestures are checked before the tool, because they work from
        *any* tool: the right button sweep-deletes, and Shift/Alt + left
        pans. Both are things you want mid-edit without putting the brush
        down and picking it up again.
        """
        if self.mask is None:
            return super().mousePressEvent(event)

        if event.button() == Qt.RightButton:
            self._sweeping = True
            self._sweep_labels = []
            self._sweep_delete_at(
                self._canvas_to_image(event.position().x(),
                                       event.position().y()))
            return

        if event.button() == Qt.LeftButton and (
                event.modifiers() & PAN_MODIFIERS):
            self._pan_from = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self.mode == MODE_NONE:
            return super().mousePressEvent(event)

        if self.mode == MODE_ZOOM:
            self._zoom_drag_start = event.position().toPoint()
            self._zoom_drag_end = event.position().toPoint()
            self.update()
            return

        pt = self._canvas_to_image(event.position().x(), event.position().y())
        if pt is None:
            return
        self._emit_stroke_start()

        if self.mode == MODE_ERASE_OBJECT:
            removed = int(self.mask[pt[1], pt[0]])
            self.mask = engine.erase_object_at(self.mask, *pt)
            self.refresh()
            self._emit_stroke_end(kind="delete", target=removed)
            return

        if self.mode in (MODE_WAND_ADD, MODE_WAND_ERASE):
            action = "add" if self.mode == MODE_WAND_ADD else "erase"
            tolerance = self.effective_wand_tolerance()
            self.mask = engine.magic_wand(
                self.image, self.mask, pt[0], pt[1],
                tolerance, self.wand_max_pixels, action=action,
            )
            self.refresh()
            self._emit_stroke_end(
                kind="wand", target=(255 if action == "add" else 0),
                action=action, tolerance=round(float(tolerance), 3),
                relative=bool(self.wand_relative),
            )
            return

        # Brush / erase strokes
        radius = self._mask_radius_for_brush()
        value = 255 if self.mode == MODE_BRUSH else 0
        engine.paint_disk(self.mask, pt[0], pt[1], radius, value)
        self._last_pt = QPoint(*pt)
        self.refresh()

    def mouseMoveEvent(self, event):
        """Extend a sweep, a pan, a brush/erase stroke, or a zoom drag."""
        if self.mask is None:
            return
        if self._sweeping and event.buttons() & Qt.RightButton:
            self._sweep_delete_at(
                self._canvas_to_image(event.position().x(),
                                       event.position().y()))
            return
        if self._pan_from is not None and event.buttons() & Qt.LeftButton:
            now = event.position().toPoint()
            dx, dy = self._image_delta(now.x() - self._pan_from.x(),
                                        now.y() - self._pan_from.y())
            # Only re-anchor once the drag has actually moved the view:
            # discarding sub-pixel drags instead of accumulating them is
            # what makes a slow pan at high zoom stall completely.
            if (dx or dy) and self.pan_by(dx, dy):
                self._pan_from = now
            return
        if self.mode == MODE_ZOOM and self._zoom_drag_start is not None \
                and event.buttons() & Qt.LeftButton:
            self._zoom_drag_end = event.position().toPoint()
            self.update()
            return
        if self.mode in (MODE_BRUSH, MODE_ERASE) and event.buttons() & Qt.LeftButton:
            pt = self._canvas_to_image(event.position().x(), event.position().y())
            if pt is None:
                return
            # A drag that *began* outside the pixmap never fired
            # stroke_started, so without this the resulting edit would
            # never be pushed onto the undo history. Idempotent.
            self._emit_stroke_start()
            radius = self._mask_radius_for_brush()
            value = 255 if self.mode == MODE_BRUSH else 0
            if self._last_pt is not None:
                engine.paint_line(self.mask,
                                    self._last_pt.x(), self._last_pt.y(),
                                    pt[0], pt[1], radius, value)
            else:
                engine.paint_disk(self.mask, pt[0], pt[1], radius, value)
            self._last_pt = QPoint(*pt)
            self.refresh()

    def mouseReleaseEvent(self, event):
        """Close a sweep or pan, commit a zoom rect, or finalize a stroke."""
        if event.button() == Qt.RightButton and self._sweeping:
            self._sweeping = False
            labels, self._sweep_labels = self._sweep_labels, []
            # ONE entry for the whole sweep. Six deletes in the ledger would
            # say six decisions were made; the user made one.
            self._emit_stroke_end(kind="sweep_delete", target=list(labels),
                                   n_objects=len(labels))
            return
        if self._pan_from is not None and event.button() == Qt.LeftButton:
            self._pan_from = None
            self.unsetCursor()
            return
        if self.mode == MODE_ZOOM and self._zoom_drag_start is not None \
                and self._zoom_drag_end is not None:
            # Convert both endpoints to image coords and commit
            p0 = self._canvas_to_image(self._zoom_drag_start.x(),
                                        self._zoom_drag_start.y())
            p1 = self._canvas_to_image(self._zoom_drag_end.x(),
                                        self._zoom_drag_end.y())
            self._zoom_drag_start = None
            self._zoom_drag_end = None
            if p0 is not None and p1 is not None:
                x0, x1 = sorted((p0[0], p1[0]))
                y0, y1 = sorted((p0[1], p1[1]))
                if x1 - x0 > 4 and y1 - y0 > 4:
                    self._zoom_x0, self._zoom_y0 = x0, y0
                    self._zoom_x1, self._zoom_y1 = x1 + 1, y1 + 1
                    self.zoom_changed.emit(True)
            self.refresh()
            return
        if self._last_pt is not None:
            self._last_pt = None
        self._emit_stroke_end(
            kind="erase" if self.mode == MODE_ERASE else "paint",
            target=(0 if self.mode == MODE_ERASE else 255),
            radius=int(self.brush_radius),
        )

    def resizeEvent(self, event):
        """Refit the composited pixmap to the new canvas size."""
        super().resizeEvent(event)
        self.refresh()


# ---------------------------------------------------------------------------
# Folded modules
# ---------------------------------------------------------------------------

def fold_description(key: str) -> tuple:
    """``(name, description, stage)`` for a folded module.

    The app registry answers while it still holds the module's row; once the
    row has been dropped — which is what folding a module ends in — the
    answer comes from :data:`FOLD_FALLBACK`, so the button goes on carrying
    the name, the sentence and the maturity colour its tile had.
    """
    from .. import app as app_module

    name = description = stage = ""
    for row in getattr(app_module, "APPS", ()):
        if row and row[0] == key:
            name, description = row[1] or "", row[2] or ""
            stage = app_module.app_stage(key)
            break
    fallback = FOLD_FALLBACK.get(key, ("", "", ""))
    return (name or fallback[0], description or fallback[1],
            stage or fallback[2])


class FoldedModuleDialog(QDialog):
    """One folded module, opened over its host as the whole screen it was.

    A fold that reimplemented the module it replaced would keep whatever the
    person doing the folding happened to think of and quietly drop the rest.
    So the button opens the module's OWN widget: every control, every worker,
    every drop target it had as a tile is what arrives, and the only thing
    that changed is where it is opened from.

    The window is not modal. The reason to fold Curate or the Model Zoo into
    the mask editor is to use them ON the field that is open behind them, and
    a modal window is one that cannot be looked past.

    :param key: the folded module's registry key.
    :param screen: the module's own widget, already built.
    :param title: the window title — the module's name.
    :param actions: extra buttons for the button box, each
        ``(label, tooltip, callback)``. This is where a capability the folded
        module lacks and its host has arrives.
    """

    def __init__(self, key: str, screen: QWidget, title: str,
                 parent: Optional[QWidget] = None, actions=()):
        super().__init__(parent)
        self.app_key = key
        self.screen = screen
        self.setObjectName("FoldedModuleDialog")
        self.setWindowTitle(title)
        self.setModal(False)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, SPACING["sm"])
        column.setSpacing(SPACING["sm"])
        column.addWidget(screen, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        #: Label -> button, for the extra actions.
        self.actions: dict = {}
        for label, tooltip, callback in actions:
            button = self.buttons.addButton(label,
                                            QDialogButtonBox.ActionRole)
            button.setToolTip(tooltip)
            # The bool ``clicked`` emits is swallowed here rather than in
            # every callback: these are the host's own methods, and one that
            # took a stray positional would fail only when pressed.
            button.clicked.connect(
                lambda _checked=False, cb=callback: cb())
            self.actions[label] = button
        self.buttons.rejected.connect(self.close)
        column.addWidget(self.buttons)
        self.resize(1120, 780)


# ---------------------------------------------------------------------------
# MakeMasksScreen
# ---------------------------------------------------------------------------

class MakeMasksScreen(QWidget):
    """Qt widget for the Make Masks app — the successor to Tk ModifyMaskApp.

    Owns the canvas, the tools panel, and the file-navigation state; see
    the module docstring for the full feature list.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._folder: str = ""
        self._image_files: List[str] = []
        self._current_index: int = 0
        self._history = engine.MaskHistory(capacity=25)
        #: The ledger for the field on screen, seeded from any sidecar
        #: already beside its mask so a second editing session appends to
        #: the first one's record instead of replacing it.
        self._log: Optional[CurationLog] = None
        self._load_token = 0
        self._load_worker: Optional[_MaskLoadWorker] = None
        self._pending_load = None
        self._loading = False
        #: Folded module key -> the module's own screen, built the first time
        #: its button is pressed and kept afterwards so a second press finds
        #: the paths, models and results the first one left.
        self._fold_screens: dict[str, QWidget] = {}
        #: Folded module key -> the window that screen lives in.
        self._fold_dialogs: dict[str, FoldedModuleDialog] = {}
        self._build_ui()
        self._install_shortcuts()
        self._sync_button_states()

        # Drag & drop — accepts a folder of images to fine-tune against.
        try:
            from ..dnd import install_dropzone
            from ..dnd_handlers import MakeMasksDropHandler
            install_dropzone(self, MakeMasksDropHandler(), self)
        except Exception:
            pass
        # HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box
        # you type in. Built here on the field, it is moved onto the
        # label as the last step, so every panel in the application
        # explains itself the same way.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # Masthead — the module's own name and blurb, the folder in force,
        # and the strip of modules that fold into this one.
        self._header = ModuleHeader(
            HEADER_TITLE,
            description=HEADER_DESCRIPTION,
            instruction=HEADER_INSTRUCTION,
            app_key=APP_KEY,
        )
        self._src_label = QLabel("No folder selected — click Open folder…")
        self._src_label.setObjectName("SubtitleSmall")
        # A deep folder path must never widen the window or push the fold
        # buttons off the end of the row: the label may shrink below its
        # ideal width, and the tooltip carries what is cut off.
        self._src_label.setSizePolicy(QSizePolicy.Maximum,
                                      QSizePolicy.Preferred)
        self._src_label.setMinimumWidth(0)
        self._header.add_trailing(self._src_label)
        self._folds = self._build_fold_strip()
        self._header.add_trailing(self._folds)
        outer.addWidget(self._header)
        outer.addWidget(Divider())

        # Body — a stack: EmptyState until a folder is opened, then splitter
        self._body_stack = QStackedWidget()

        self._empty_state = EmptyState(
            title="Open a folder of images to edit masks",
            subtitle=(
                "Pick a folder that contains microscopy images "
                "(.tif / .png / .jpg). Any existing masks in a `masks/` "
                "subfolder are loaded; new masks save back there as "
                "labeled uint16 TIFFs."
            ),
            icon=iconset.accent_icon("brush"),
            cta_label="Open folder…",
            on_action=self._on_pick_folder,
        )
        self._body_stack.addWidget(self._empty_state)

        self._body_splitter = QSplitter(Qt.Horizontal)
        self._body_splitter.setChildrenCollapsible(False)
        self._canvas = _MaskCanvas()
        self._canvas.stroke_started.connect(self._on_stroke_started)
        self._canvas.stroke_finished.connect(self._on_stroke_finished)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        self._body_splitter.addWidget(self._canvas)

        tools_scroll = QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setFrameShape(QScrollArea.NoFrame)
        tools_scroll.setWidget(self._build_tools_panel())
        self._body_splitter.addWidget(tools_scroll)
        self._body_splitter.setStretchFactor(0, 3)
        self._body_splitter.setStretchFactor(1, 1)
        self._body_splitter.setSizes([900, 380])
        self._body_stack.addWidget(self._body_splitter)
        self._body_stack.setCurrentWidget(self._empty_state)

        outer.addWidget(self._body_stack, 1)

        # Bottom nav bar
        nav = QWidget()
        nav_row = QHBoxLayout(nav)
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(SPACING["sm"])
        self._btn_open = QPushButton("Open folder…")
        self._btn_open.setObjectName("PrimaryButton")
        self._btn_open.setIcon(iconset.contrast_icon("open"))
        self._btn_open.setCursor(Qt.PointingHandCursor)
        self._btn_open.clicked.connect(self._on_pick_folder)
        nav_row.addWidget(self._btn_open)

        self._btn_prev = QPushButton("Prev image")
        self._btn_prev.setIcon(iconset.icon("prev"))
        self._btn_prev.setCursor(Qt.PointingHandCursor)
        self._btn_prev.clicked.connect(self._on_prev)
        nav_row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next image")
        self._btn_next.setIcon(iconset.icon("next"))
        self._btn_next.setLayoutDirection(Qt.RightToLeft)
        self._btn_next.setCursor(Qt.PointingHandCursor)
        self._btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(self._btn_next)

        self._btn_save = QPushButton("Save mask")
        self._btn_save.setObjectName("PrimaryButton")
        self._btn_save.setIcon(iconset.contrast_icon("save"))
        self._btn_save.setCursor(Qt.PointingHandCursor)
        self._btn_save.clicked.connect(self._on_save)
        nav_row.addWidget(self._btn_save)

        nav_row.addStretch(1)
        self._status_label = QLabel("Ready.")
        self._status_label.setObjectName("SubtitleSmall")
        nav_row.addWidget(self._status_label)
        outer.addWidget(nav)

    # ------------------------------------------------------------------
    # The folded modules
    # ------------------------------------------------------------------
    def _build_fold_strip(self) -> FoldStrip:
        """The masthead's strip of folded modules.

        Built through :class:`~spacr.qt.widgets.fold_strip.FoldStrip` so each
        button is the module's own icon, tooltipped with its own sentence and
        lit on hover in its own maturity colour, read from the tables the
        tiles read rather than from a second one here.
        """
        entries = []
        for key in FOLD_ORDER:
            if key == MASK_FOLDER_KEY:
                entries.append((key, self.mask_whole_folder))
            else:
                entries.append((key, partial(self.open_folded, key)))
        strip = FoldStrip(entries, parent=self)
        for key in FOLD_ORDER:
            self._restate_fold_button(strip.button_for(key), key)
        return strip

    @staticmethod
    def _restate_fold_button(button, key: str) -> None:
        """Give ``button`` the name, sentence and stage its tile carried.

        A no-op while the registry still holds the row — the strip has
        already read the same three things from the same place. It is what
        keeps the button honest afterwards, when the row is gone and the
        registry would report no description and a stable-blue hover for a
        module that is neither.
        """
        if button is None:
            return
        name, description, stage = fold_description(key)
        button.setToolTip(f"{name}\n{description}".strip())
        button.setAccessibleName(name)
        if button.property("stage") != stage:
            button.setProperty("stage", stage)
            # A property the stylesheet selects on is only read at polish, so
            # a button already on screen keeps the old colour until it is
            # polished again.
            button.style().unpolish(button)
            button.style().polish(button)

    def folded_screen(self, key: str) -> Optional[QWidget]:
        """The folded module's own screen, built on first use and kept.

        :param key: one of :data:`FOLD_ORDER`. Keys that share a screen —
            see :data:`FOLD_HOSTS` — resolve to the one widget that hosts
            them both.
        :returns: the screen, or ``None`` for a key this screen does not
            fold.
        """
        host = FOLD_HOSTS.get(key, key)
        if host not in FOLD_ORDER:
            return None
        screen = self._fold_screens.get(host)
        if screen is None:
            screen = self._build_folded_screen(host)
            self._fold_screens[host] = screen
        return screen

    def _build_folded_screen(self, key: str) -> QWidget:
        """Construct one folded module's widget.

        Each branch builds the module's real screen class, and the generic
        settings page is what a module with no screen of its own gets — the
        same page its tile opened.
        """
        if key == "train_cellpose":
            from .train_cellpose import CellposeWorkbenchScreen
            return CellposeWorkbenchScreen()
        if key == "model_compare":
            from .model_compare import ModelCompareScreen
            return ModelCompareScreen()
        if key == "model_zoo":
            from .model_zoo import ModelZooScreen
            screen = ModelZooScreen()
            # The zoo's "compare these two" hand-off is wired by whoever
            # hosts it. Folded, that is this screen, or the button would
            # select two models and open nothing.
            screen.compare_requested.connect(self._on_zoo_compare_requested)
            return screen
        if key == "curate":
            from .curate import CurateScreen
            return CurateScreen()
        if key == "napari_bridge":
            from .napari_bridge import NapariBridgeScreen
            return NapariBridgeScreen()
        from .app_screen import AppScreen
        return AppScreen(app_key=key)

    def _fold_actions(self, key: str) -> tuple:
        """Extra buttons for a folded module's window.

        Where a capability the folded module never had and this screen does
        arrives with the fold.
        """
        if key == "curate":
            return (("Save mask",
                     "Write the corrected labels back to the mask file, with "
                     "the correction ledger beside them",
                     self.save_curated_mask),)
        return ()

    def open_folded(self, key: str) -> Optional[FoldedModuleDialog]:
        """Open a folded module over this screen, pointed at the open field.

        :param key: one of :data:`FOLD_ORDER`.
        :returns: the module's window, or ``None`` for a key this screen does
            not fold. Pressing the same button again raises the window that
            is already open rather than building a second one.
        """
        host = FOLD_HOSTS.get(key, key)
        screen = self.folded_screen(host)
        if screen is None:
            return None
        dialog = self._fold_dialogs.get(host)
        if dialog is None:
            dialog = FoldedModuleDialog(
                host, screen, fold_description(host)[0], parent=self,
                actions=self._fold_actions(host))
            self._fold_dialogs[host] = dialog
        self.seed_folded(key)
        dialog.show()
        dialog.raise_()
        return dialog

    def seed_folded(self, key: str) -> dict:
        """Point a folded module at the field this screen has open.

        The whole reason these are buttons on this masthead rather than rows
        of their own is that the folder is already chosen here; a folded
        module that opened on an empty path would have folded the file dialog
        in with it.

        :param key: one of :data:`FOLD_ORDER`. Note that this is the button's
            key, not its host's: the two Cellpose halves share a screen and
            seed different halves of it.
        :returns: what was seeded, as ``{name: value}``. Empty when no folder
            is open, which is not a failure — the module opens on its own
            file picker exactly as its tile did.
        """
        if not self._folder:
            return {}
        screen = self.folded_screen(key)
        if screen is None:
            return {}
        if key in ("train_cellpose", MASK_FOLDER_KEY):
            return self._seed_cellpose(screen, key)
        if key == "model_compare":
            screen.set_source(self._folder)
            return {"folder": self._folder}
        if key == "model_zoo":
            screen.set_fields_source(self._folder)
            return {"folder": self._folder}
        if key in ("curate", "napari_bridge"):
            return self._seed_mask_editor(screen, key)
        # A module with no screen of its own: a settings page, whose one
        # path is the folder this screen already has open.
        screen.apply_settings_dict({"src": self._folder})
        return {"src": self._folder}

    def _seed_cellpose(self, workbench: QWidget, key: str) -> dict:
        """Open the Cellpose workbench on the half the button names.

        Training and applying read ``src`` differently — training wants the
        parent of ``train/images``, applying wants the folder of fields — so
        only the applying half is given the folder this screen has open.
        Training is opened on its own tab with its own path untouched, which
        is the one thing that must not be guessed at.
        """
        target = (workbench.train_screen if key == "train_cellpose"
                  else workbench.apply_screen)
        tabs = workbench.findChild(QTabWidget)
        if tabs is not None:
            tabs.setCurrentWidget(target)
        if key == "train_cellpose":
            return {}
        target.apply_settings_dict({"src": self._folder})
        return {"src": self._folder}

    def _seed_mask_editor(self, screen: QWidget, key: str) -> dict:
        """Hand a mask editor the field on screen, image and mask both.

        Nothing is opened for the user: both screens report on whether the
        file they were given has been curated before, and reading a mask off
        disk while the folder is being edited here would answer that question
        about the wrong copy.
        """
        filename = self._image_files[self._current_index]
        mask_path = engine.mask_save_path(self._folder, filename)
        seeded = {"mask": mask_path}
        screen._mask_edit.setText(mask_path)
        if key == "napari_bridge":
            image_path = os.path.join(self._folder, filename)
            screen._image_edit.setText(image_path)
            seeded["image"] = image_path
        return seeded

    def _on_zoo_compare_requested(self, request: dict) -> None:
        """Open the folded Model Compare on the two models the zoo picked."""
        dialog = self.open_folded("model_compare")
        if dialog is None:
            return
        dialog.screen.configure(
            model_a=request.get("model_a", ""),
            model_b=request.get("model_b", ""),
            folder=request.get("folder", ""),
            n_fields=int(request.get("n_fields", 0) or 0),
        )

    def mask_whole_folder(self) -> bool:
        """Segment every image in the open folder with the current model.

        The one folded button that does something rather than opening
        something: it points the applying half of the Cellpose workbench at
        the folder already open here and starts it. "The current model" is
        whatever that tab holds — the checkpoint the Train tab produced if
        there is one, and the stock model otherwise.

        :returns: whether a run was started. A folder that is not open, and a
            confirmation that is declined, both answer ``False``.
        """
        if not self._folder or not self._image_files:
            self._status_label.setText(
                "Open a folder of images before masking it.")
            return False
        count = len(self._image_files)
        if not self._confirm(
                "Mask the whole folder",
                f"Segment all {count} images in {self._folder}?\n\n"
                f"Images that already have a mask are left alone."):
            return False
        dialog = self.open_folded(MASK_FOLDER_KEY)
        if dialog is None:
            return False
        self._status_label.setText(
            f"Masking {count} images in {self._folder}…")
        self._start_folded_run(dialog.screen.apply_screen)
        return True

    @staticmethod
    def _start_folded_run(screen) -> None:
        """Press a folded module page's Run.

        One line, named, because it is the seam between this screen and a job
        that wants a GPU: a test drives everything up to it without starting
        Cellpose.
        """
        screen._on_run()

    def save_curated_mask(self) -> str:
        """Write the labels Curate corrected back to the mask file.

        The Curate screen paints and records and never wrote a pixel: its
        ledger asserted corrections beside a file the pipeline had produced,
        and :func:`spacr.curation.is_curated` answered ``True`` for that
        untouched file. The write lives here because this is the screen that
        writes masks.

        :returns: the path written, or ``""`` when there is nothing to write.
        """
        screen = self._fold_screens.get("curate")
        brush = getattr(screen, "brush", None)
        if brush is None:
            self._status_label.setText(
                "Open a mask in Curate before saving it.")
            return ""
        try:
            written = brush.session.save_mask()
        except Exception as exc:
            self._warn("Save failed", str(exc))
            return ""
        self._status_label.setText(f"Saved → {written}")
        return written

    def close_folded(self) -> None:
        """Close every folded module window, and everything it started.

        Closing the window is not enough. A module page polls the machine's
        RAM and GPU on a worker thread while it is visible, and Qt answers a
        running QThread being destroyed by aborting the process — so the
        page's own close handler, which drains that worker, has to run. A
        page nested inside another module's tabs never gets one from Qt,
        which is why they are closed by hand here.
        """
        from .app_screen import AppScreen

        for dialog in list(self._fold_dialogs.values()):
            screen = dialog.screen
            for page in screen.findChildren(AppScreen):
                page.close()
            screen.close()
            dialog.close()

    def _build_tools_panel(self) -> QWidget:
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])

        # Mode buttons — arranged as a 2×3 grid so buttons keep their labels
        mode_card = Card(title="Tools")
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setSpacing(SPACING["sm"])
        self._mode_buttons: dict[str, QPushButton] = {}
        modes = [
            (MODE_BRUSH,        "Brush",        "brush"),
            (MODE_ERASE,        "Erase",        "erase"),
            (MODE_ERASE_OBJECT, "Erase object", "erase_object"),
            (MODE_WAND_ADD,     "Wand +",       "wand_add"),
            (MODE_WAND_ERASE,   "Wand −",       "wand_erase"),
            (MODE_ZOOM,         "Zoom",         "zoom"),
        ]
        for i, (m, label, icon_key) in enumerate(modes):
            btn = QPushButton(label)
            btn.setIcon(iconset.icon(icon_key))
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, key=m: self._set_mode(key))
            grid.addWidget(btn, i // 3, i % 3)
            self._mode_buttons[m] = btn
        self._btn_brush = self._mode_buttons[MODE_BRUSH]
        self._btn_erase = self._mode_buttons[MODE_ERASE]
        self._btn_del_obj = self._mode_buttons[MODE_ERASE_OBJECT]
        self._btn_wand_add = self._mode_buttons[MODE_WAND_ADD]
        self._btn_wand_erase = self._mode_buttons[MODE_WAND_ERASE]
        self._btn_zoom = self._mode_buttons[MODE_ZOOM]
        mode_wrap = QWidget(); mode_wrap.setLayout(grid)
        mode_card.body_layout.addWidget(mode_wrap)

        # Reset zoom / undo redo row
        history_row = QHBoxLayout()
        history_row.setSpacing(SPACING["sm"])
        self._btn_reset_zoom = QPushButton("Reset zoom")
        self._btn_reset_zoom.setIcon(iconset.icon("zoom_reset"))
        self._btn_reset_zoom.setCursor(Qt.PointingHandCursor)
        self._btn_reset_zoom.setEnabled(False)
        self._btn_reset_zoom.clicked.connect(self._on_reset_zoom)
        history_row.addWidget(self._btn_reset_zoom)
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setIcon(iconset.icon("undo"))
        self._btn_undo.setCursor(Qt.PointingHandCursor)
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_undo)
        history_row.addWidget(self._btn_undo)
        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setIcon(iconset.icon("redo"))
        self._btn_redo.setCursor(Qt.PointingHandCursor)
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._on_redo)
        history_row.addWidget(self._btn_redo)
        hist_wrap = QWidget(); hist_wrap.setLayout(history_row)
        mode_card.body_layout.addWidget(hist_wrap)
        col.addWidget(mode_card)

        # Brush size slider
        brush_card = Card(title="Brush")
        brush_form = QFormLayout()
        self._brush_slider = QSlider(Qt.Horizontal)
        self._brush_slider.setRange(1, 100)
        self._brush_slider.setValue(10)
        self._brush_slider.valueChanged.connect(self._on_brush_size_changed)
        self._brush_size_label = QLabel("10 px")
        self._brush_size_label.setObjectName("Muted")
        brush_row = QHBoxLayout()
        brush_row.addWidget(self._brush_slider, 1)
        brush_row.addWidget(self._brush_size_label)
        brush_wrap = QWidget(); brush_wrap.setLayout(brush_row)
        brush_form.addRow("Radius", brush_wrap)
        brush_card.body_layout.addLayout(brush_form)
        col.addWidget(brush_card)

        # Magic wand card
        wand_card = Card(title="Magic wand")
        wand_form = QFormLayout()
        self._wand_relative = QCheckBox("Tolerance is % of image range")
        self._wand_relative.setChecked(True)
        self._wand_relative.setToolTip(
            "On: the tolerance below is a percentage of THIS image's own "
            "intensity range, so one setting behaves the same on 8-bit and "
            "16-bit data. Off: a fixed grey-level distance, which selects "
            "nothing on a 16-bit image at a value tuned for 8-bit and "
            "floods the whole frame the other way round."
        )
        self._wand_relative.toggled.connect(self._on_wand_relative_changed)
        wand_card.body_layout.addWidget(self._wand_relative)
        self._wand_pct = QDoubleSpinBox()
        self._wand_pct.setDecimals(3)
        self._wand_pct.setRange(0.001, 100.0)
        self._wand_pct.setSingleStep(0.5)
        self._wand_pct.setValue(5.0)
        self._wand_pct.valueChanged.connect(self._on_wand_pct_changed)
        wand_form.addRow("Tolerance %", self._wand_pct)
        self._wand_tol = QDoubleSpinBox()
        self._wand_tol.setRange(0.0, 1_000_000.0)
        self._wand_tol.setSingleStep(50.0)
        self._wand_tol.setValue(1000.0)
        self._wand_tol.setEnabled(False)
        self._wand_tol.valueChanged.connect(self._on_wand_tolerance_changed)
        wand_form.addRow("Tolerance (absolute)", self._wand_tol)
        self._wand_max = QSpinBox()
        self._wand_max.setRange(1, 10_000_000)
        self._wand_max.setSingleStep(1000)
        self._wand_max.setValue(100_000)
        self._wand_max.valueChanged.connect(self._on_wand_max_changed)
        wand_form.addRow("Max pixels", self._wand_max)
        wand_card.body_layout.addLayout(wand_form)
        col.addWidget(wand_card)

        # Display card — contrast percentiles and wheel-zoom speed.
        norm_card = Card(title="Display")
        norm_form = QFormLayout()
        self._norm_lo = QDoubleSpinBox()
        # setDecimals BEFORE setRange/setValue: a QDoubleSpinBox rounds
        # both to the precision it has at the time, so setting 99.9999
        # against the default two decimals stores 100.0 and the control
        # looks broken rather than imprecise.
        self._norm_lo.setDecimals(PERCENTILE_DECIMALS)
        self._norm_lo.setRange(0.0, 100.0)
        self._norm_lo.setSingleStep(0.01)
        self._norm_lo.setValue(1.0)
        self._norm_lo.setToolTip(
            "Percentile mapped to black. Raise it to sink background "
            "speckle. Six decimals, so 0.0001 clips only the darkest few "
            "pixels of a megapixel field."
        )
        self._norm_lo.valueChanged.connect(self._on_normalize_changed)
        self._norm_hi = QDoubleSpinBox()
        self._norm_hi.setDecimals(PERCENTILE_DECIMALS)
        self._norm_hi.setRange(0.0, 100.0)
        self._norm_hi.setSingleStep(0.01)
        self._norm_hi.setValue(99.9)
        self._norm_hi.setToolTip(
            "Percentile mapped to white. Lower it to lift faint objects. "
            "On a 16-bit field a handful of hot pixels hold the top of the "
            "range on their own, so the useful setting is 99.9999 — the "
            "top four pixels of four million — which needs six decimals."
        )
        self._norm_hi.valueChanged.connect(self._on_normalize_changed)
        norm_form.addRow("Lower %", self._norm_lo)
        norm_form.addRow("Upper %", self._norm_hi)
        self._zoom_speed = QDoubleSpinBox()
        self._zoom_speed.setDecimals(2)
        self._zoom_speed.setRange(1.01, 3.0)
        self._zoom_speed.setSingleStep(0.05)
        self._zoom_speed.setValue(1.15)
        self._zoom_speed.setToolTip(
            "How far one wheel notch zooms. Higher jumps across a large "
            "field faster; lower gives the fine steps that trimming an "
            "object boundary needs. Shift or Alt + drag pans, from any tool."
        )
        self._zoom_speed.valueChanged.connect(self._on_zoom_speed_changed)
        norm_form.addRow("Zoom per notch", self._zoom_speed)
        norm_card.body_layout.addLayout(norm_form)
        col.addWidget(norm_card)

        # Auto-filter card — size/intensity bounds applied on load.
        filter_card = Card(
            title="Auto-filter objects",
            subtitle="Applied when a field loads. 0 switches a bound off.",
        )
        filter_form = QFormLayout()
        self._filter_min_area = QSpinBox()
        self._filter_min_area.setRange(0, 100_000_000)
        self._filter_min_area.setToolTip(
            "Drop objects smaller than this many pixels. 0 = no minimum.")
        self._filter_max_area = QSpinBox()
        self._filter_max_area.setRange(0, 100_000_000)
        self._filter_max_area.setToolTip(
            "Drop objects larger than this many pixels. 0 = no maximum.")
        self._filter_min_int = QDoubleSpinBox()
        self._filter_min_int.setDecimals(2)
        self._filter_min_int.setRange(0.0, 65535.0)
        self._filter_min_int.setSingleStep(1.0)
        self._filter_min_int.setToolTip(
            "Drop objects whose MEAN value on the raw image is below this. "
            "Measured on the raw data, not on the contrast-stretched "
            "display, so changing the percentiles above cannot move it. "
            "0 = no minimum.")
        self._filter_max_int = QDoubleSpinBox()
        self._filter_max_int.setDecimals(2)
        self._filter_max_int.setRange(0.0, 65535.0)
        self._filter_max_int.setSingleStep(1.0)
        self._filter_max_int.setToolTip(
            "Drop objects whose MEAN raw value is above this. 0 = no maximum.")
        filter_form.addRow("Min area (px)", self._filter_min_area)
        filter_form.addRow("Max area (px)", self._filter_max_area)
        filter_form.addRow("Min mean intensity", self._filter_min_int)
        filter_form.addRow("Max mean intensity", self._filter_max_int)
        filter_card.body_layout.addLayout(filter_form)
        self._btn_filter = QPushButton("Apply filter now")
        self._btn_filter.setCursor(Qt.PointingHandCursor)
        self._btn_filter.clicked.connect(self._on_apply_filter)
        filter_card.body_layout.addWidget(self._btn_filter)
        col.addWidget(filter_card)

        # Object ops card
        obj_card = Card(title="Object operations")
        ops_col = QVBoxLayout()
        ops_col.setSpacing(SPACING["xs"])
        for label, cb in (
            ("Fill holes", self._on_fill_holes),
            ("Relabel", self._on_relabel),
            ("Invert mask", self._on_invert),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(cb)
            ops_col.addWidget(btn)
        remove_row = QHBoxLayout()
        remove_row.setSpacing(SPACING["sm"])
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 1_000_000)
        self._min_area.setValue(100)
        remove_row.addWidget(QLabel("Min area:"))
        remove_row.addWidget(self._min_area, 1)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_small)
        remove_row.addWidget(remove_btn)
        remove_wrap = QWidget(); remove_wrap.setLayout(remove_row)
        ops_col.addWidget(remove_wrap)
        detect_row = QHBoxLayout()
        detect_row.setSpacing(SPACING["sm"])
        self._btn_otsu = QPushButton("Otsu detect")
        self._btn_otsu.setCursor(Qt.PointingHandCursor)
        self._btn_otsu.setToolTip(
            "Threshold the image at Otsu's level and label what is left, "
            "honouring the minimum area above.")
        self._btn_otsu.clicked.connect(self._on_detect_otsu)
        detect_row.addWidget(self._btn_otsu)
        self._otsu_bright = QCheckBox("Bright")
        self._otsu_bright.setChecked(True)
        self._otsu_bright.setToolTip(
            "On: objects are brighter than background, as in fluorescence. "
            "Off: take the dark side instead, for brightfield or stain.")
        detect_row.addWidget(self._otsu_bright)
        self._combine_mode = QComboBox()
        # THE MODE IS THE ITEM'S DATA, NOT ITS LABEL. `replace` and `merge`
        # are shown to the user and a language switch rewrites the item text
        # in place; reading the mode back off that text would hand
        # `engine.combine_masks` a translated word it has never heard of, so
        # the untranslated key travels with the item instead.
        for _mode in ("replace", "merge"):
            self._combine_mode.addItem(_mode, _mode)
        self._combine_mode.setToolTip(
            "replace: the detection becomes the mask and what was there is "
            "gone. merge: keep every existing object and add detected ones "
            "only where nothing is labelled, so a detection run halfway "
            "through cannot undo the editing done so far.")
        detect_row.addWidget(self._combine_mode, 1)
        detect_wrap = QWidget(); detect_wrap.setLayout(detect_row)
        ops_col.addWidget(detect_wrap)
        clear_btn = QPushButton("Clear mask")
        clear_btn.setObjectName("DangerButton")
        clear_btn.clicked.connect(self._on_clear_mask)
        ops_col.addWidget(clear_btn)
        obj_ops_wrap = QWidget(); obj_ops_wrap.setLayout(ops_col)
        obj_card.body_layout.addWidget(obj_ops_wrap)
        col.addWidget(obj_card)

        col.addStretch(1)
        return wrap

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_next)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save)
        QShortcut(QKeySequence("B"), self, lambda: self._set_mode(MODE_BRUSH))
        QShortcut(QKeySequence("E"), self, lambda: self._set_mode(MODE_ERASE))
        QShortcut(QKeySequence("W"), self, lambda: self._set_mode(MODE_WAND_ADD))
        QShortcut(QKeySequence("Z"), self, lambda: self._set_mode(MODE_ZOOM))
        QShortcut(QKeySequence("Escape"), self, self._on_reset_zoom)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    # ------------------------------------------------------------------
    # Mode / brush plumbing
    # ------------------------------------------------------------------
    def _set_mode(self, mode: str):
        self._canvas.mode = mode
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

    def _on_brush_size_changed(self, v: int):
        self._canvas.brush_radius = int(v)
        self._brush_size_label.setText(f"{v} px")

    def _on_normalize_changed(self, _v: float):
        self._canvas.norm_lo = float(self._norm_lo.value())
        self._canvas.norm_hi = float(self._norm_hi.value())
        self._canvas.refresh()

    def _on_wand_tolerance_changed(self, v: float):
        self._canvas.wand_tolerance = float(v)

    def _on_wand_pct_changed(self, v: float):
        self._canvas.wand_tol_pct = float(v)

    def _on_wand_relative_changed(self, on: bool):
        """Switch the wand between a percentage and a fixed grey distance.

        Only the box that is in force stays enabled, so the panel cannot
        show two tolerances and leave which one the wand uses to be guessed
        from a checkbox three rows up.
        """
        self._canvas.wand_relative = bool(on)
        self._wand_pct.setEnabled(bool(on))
        self._wand_tol.setEnabled(not on)

    def _on_wand_max_changed(self, v: int):
        self._canvas.wand_max_pixels = int(v)

    def _on_zoom_speed_changed(self, v: float):
        self._canvas.zoom_speed = float(v)

    def _on_reset_zoom(self):
        self._canvas.reset_zoom()

    def _on_zoom_changed(self, zoomed: bool):
        self._btn_reset_zoom.setEnabled(zoomed)
        self._status_label.setText("Zoomed — press Esc to reset" if zoomed
                                     else "Zoom reset")

    def _on_undo(self):
        """Step the mask back one edit, and record that as an edit itself.

        The ledger is append-only: taking a stroke back adds an ``undo``
        entry rather than removing the entry for the stroke. That something
        was painted and then reconsidered is part of what happened to the
        data, and a history that can be quietly tidied is not evidence of
        anything.
        """
        prev = self._history.undo()
        if prev is None or self._canvas.mask is None:
            return
        # Diffed against what is ON the canvas, not against the history
        # head: undo() has already popped, so the head IS `prev` by now and
        # comparing the two would measure every undo as having changed
        # nothing — which is exactly how they went unrecorded.
        changed = self._diff(self._canvas.mask, prev)
        self._canvas.mask = prev
        self._canvas.refresh()
        self._record("undo", None, changed)
        self._refresh_history_buttons()

    def _on_redo(self):
        """Restore the most recently undone edit, recorded as a ``redo``."""
        nxt = self._history.redo()
        if nxt is None or self._canvas.mask is None:
            return
        changed = self._diff(self._canvas.mask, nxt)
        self._canvas.mask = nxt
        self._canvas.refresh()
        self._record("redo", None, changed)
        self._refresh_history_buttons()

    def _refresh_history_buttons(self):
        self._btn_undo.setEnabled(self._history.can_undo())
        self._btn_redo.setEnabled(self._history.can_redo())

    # ------------------------------------------------------------------
    # The curation ledger
    # ------------------------------------------------------------------
    def _record(self, kind: str, target=None, n_changed: int = 0, **detail):
        """Append one edit to this field's ledger, if it changed anything.

        An edit that moved no pixels is not recorded, for the same reason
        :mod:`spacr.napari_bridge` does not record one: a ledger padded with
        entries for clicks that landed on background is a ledger nobody
        reads, and ``is_curated`` would then answer True for every mask
        anyone ever opened the editor on.
        """
        if self._log is None or int(n_changed) <= 0:
            return None
        return self._log.append(kind, target, n_changed=int(n_changed),
                                 **detail)

    @staticmethod
    def _diff(before, after) -> int:
        """How many pixels two masks disagree on.

        A pair with no common shape is counted as everything ``after`` has
        labelled, which is the honest answer when there is nothing to
        compare against rather than a silent zero.
        """
        if after is None:
            return 0
        if before is None or before.shape != after.shape:
            return int(np.count_nonzero(after))
        return int(np.count_nonzero(before != after))

    def _pixels_changed(self, after) -> int:
        """How many pixels ``after`` differs from the last history snapshot.

        The snapshot is the state the edit in progress started from, so this
        is the size of that one edit — and it is what tells a stroke that
        repainted a third of the field from one that was a stray click.
        """
        return self._diff(self._history.head(), after)

    # ------------------------------------------------------------------
    # Size / intensity auto-filter
    # ------------------------------------------------------------------
    def _filter_bounds(self) -> dict:
        """The four filter bounds as :func:`mask_engine.filter_objects` wants."""
        return {
            "min_area": int(self._filter_min_area.value()),
            "max_area": int(self._filter_max_area.value()),
            "min_intensity": float(self._filter_min_int.value()),
            "max_intensity": float(self._filter_max_int.value()),
        }

    def apply_object_filter(self, *, on_load: bool = False) -> int:
        """Drop objects outside the size/intensity bounds; return how many.

        Runs itself when a field loads — a draft segmentation usually
        arrives with the same class of junk in every field, and clearing it
        by hand once per field is the work this exists to remove — and again
        whenever the user asks, since the bounds are tuned by looking at
        what the last run left behind.

        The result is one undo step and one ledger entry naming every object
        it removed, so an automatic edit is as traceable and as reversible
        as a click.

        :param on_load: True when this is the automatic run. It only changes
            what the status line says and what the ledger entry records; a
            filter that removed nothing stays quiet on load rather than
            reporting a non-event over the name of the field just opened.
        """
        if self._canvas.mask is None or self._canvas.image is None:
            return 0
        out, dropped = engine.filter_objects(
            self._canvas.mask, self._canvas.image, **self._filter_bounds())
        if not dropped:
            if not on_load:
                self._status_label.setText(
                    "Size/intensity filter: nothing outside the bounds.")
            return 0
        changed = int(np.count_nonzero(self._canvas.mask != out))
        self._canvas.mask = out
        self._canvas.refresh()
        self._record("filter", dropped, changed, n_objects=len(dropped),
                      automatic=bool(on_load), **self._filter_bounds())
        self._history.push(out)
        self._refresh_history_buttons()
        self._status_label.setText(
            f"Size/intensity filter removed {len(dropped)} object(s) — "
            "Ctrl+Z to undo"
        )
        return len(dropped)

    def _on_apply_filter(self):
        self.apply_object_filter(on_load=False)

    def _on_detect_otsu(self):
        """Threshold the image and fold the result in per replace/merge."""
        if self._canvas.image is None or self._canvas.mask is None:
            return
        mode = self._combine_mode.currentData()
        try:
            detected = engine.otsu_instances(
                self._canvas.image,
                bright=self._otsu_bright.isChecked(),
                min_area=int(self._min_area.value()),
            )
        except Exception as exc:
            self._warn("Otsu detect failed", str(exc))
            return
        found = int(detected.max())
        if not found:
            # Replacing with nothing would silently wipe the mask on a flat
            # field, or on one where the minimum area rejected everything.
            # Clearing a mask is what the Clear button is for, and it asks.
            self._status_label.setText(
                "Otsu found no objects — the mask is unchanged. Lower the "
                "minimum area, or try the other side."
            )
            return
        try:
            out = engine.combine_masks(self._canvas.mask, detected, mode)
        except Exception as exc:
            self._warn("Otsu detect failed", str(exc))
            return
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        self._record("detect", mode, changed, method="otsu", n_objects=found,
                      bright=bool(self._otsu_bright.isChecked()),
                      min_area=int(self._min_area.value()))
        self._history.push(out)
        self._refresh_history_buttons()
        side = "bright" if self._otsu_bright.isChecked() else "dark"
        self._status_label.setText(
            f"Otsu ({side}) found {found} object(s) — {mode}d into the mask"
        )


    # ------------------------------------------------------------------
    # User messaging (headless-safe — see :func:`is_headless`)
    # ------------------------------------------------------------------
    def _warn(self, title: str, text: str) -> None:
        """Report a non-fatal failure to the user.

        Shows a modal warning when a display is attached; otherwise the
        message goes to the status line and the log, because a modal box
        under the offscreen/minimal platform plugin never returns.
        """
        self._status_label.setText(f"{title}: {text}")
        if is_headless():
            LOG.warning("%s: %s", title, text)
            return
        QMessageBox.warning(self, title, text)

    def _confirm(self, title: str, text: str) -> bool:
        """Ask the user to approve a destructive action.

        Returns False when headless: with nobody to answer, the safe
        answer for an irreversible operation is "no".
        """
        if is_headless():
            LOG.warning("%s: no display to confirm on — not proceeding", title)
            self._status_label.setText(
                f"{title} cancelled — no display to confirm on"
            )
            return False
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pick images folder",
                                              self._folder or os.getcwd())
        if not d:
            return
        self._open_folder(d)

    def _open_folder(self, folder: str):
        files = engine.list_images(folder)
        if not files:
            self._warn("No images", f"Found no image files in: {folder}")
            return
        self._folder = folder
        self._image_files = files
        self._current_index = 0
        self._src_label.setText(f"{folder}  —  {len(files)} images")
        self._load_current()
        self._sync_button_states()
        prefs.push_recent_source("make_masks", folder)
        self._body_stack.setCurrentWidget(self._body_splitter)

    def _load_current(self):
        if not self._image_files:
            return
        self._load_token += 1
        token = self._load_token
        filename = self._image_files[self._current_index]
        image_path = os.path.join(self._folder, filename)
        if self._should_background_load(image_path):
            request = (self._folder, filename, token)
            if self._load_worker is not None:
                self._pending_load = request
                self._status_label.setText(f"Waiting to load {filename}…")
                return
            self._start_background_load(*request)
            return
        self._load_pair(self._folder, filename, token)

    @staticmethod
    def _should_background_load(path: str) -> bool:
        """Return True when decoding ``path`` is large enough to stall Qt.

        File size catches ordinary uncompressed microscopy TIFFs. A quick PIL
        header read also catches highly compressed large images without
        decoding their pixels.
        """
        threshold = 8 * 1024 * 1024
        try:
            if os.path.getsize(path) >= threshold:
                return True
            from PIL import Image
            with Image.open(path) as probe:
                bands = max(1, len(probe.getbands()))
                bytes_per_sample = 2 if "16" in probe.mode else 1
                return (
                    probe.width * probe.height * bands * bytes_per_sample
                    >= threshold
                )
        except (OSError, ValueError):
            # Let the real loader report corrupt/unreadable inputs.
            return False

    def _start_background_load(
        self, folder: str, filename: str, token: int
    ) -> None:
        """Start one retained image loader and disable edit controls."""
        self._loading = True
        self._status_label.setText(f"Loading {filename}…")
        self._sync_button_states()
        worker = _MaskLoadWorker(folder, filename, token, self)
        self._load_worker = worker
        worker.finished.connect(self._on_background_load_finished)
        worker.start()

    def _on_background_load_finished(self) -> None:
        """Apply the newest background result and start any pending request."""
        worker = self._load_worker
        if worker is None:
            return
        self._load_worker = None
        self._loading = False
        if worker.token == self._load_token:
            if worker.error is not None:
                self._handle_load_failure(worker.error)
            elif worker.result is not None:
                self._apply_loaded_pair(
                    worker.filename, worker.token, *worker.result
                )
        worker.deleteLater()
        pending, self._pending_load = self._pending_load, None
        if pending is not None:
            self._start_background_load(*pending)
        else:
            self._sync_button_states()

    def closeEvent(self, event):
        """Drain the background image loader before Qt destroys this screen.

        ``_MaskLoadWorker`` is parented to this widget, so without this the
        screen's destructor deletes a QThread that is still decoding a large
        TIFF, and Qt answers that with ``qFatal("QThread: Destroyed while
        thread is still running")`` — a core dump, not an exception. The
        window is exactly as wide as one image decode, which is why it shows
        up in a loaded test shard and almost never by hand.

        Any folded module still open goes with it: each one is a window of
        its own, and several of them own worker threads and viewers that must
        be told to stop rather than be collected out from under Qt.
        """
        from ..bridge import drain_thread

        self.close_folded()
        self._pending_load = None
        worker, self._load_worker = self._load_worker, None
        if worker is not None:
            try:
                worker.requestInterruption()
            except (AttributeError, RuntimeError):
                pass
            drain_thread(worker, timeout_ms=5000)
        self._loading = False
        super().closeEvent(event)

    def _load_pair(self, folder: str, filename: str, token: int) -> None:
        """Decode and apply a small pair synchronously."""
        try:
            image, mask = engine.load_image_and_mask(folder, filename)
        except Exception as exc:
            self._handle_load_failure(exc)
            return
        self._apply_loaded_pair(filename, token, image, mask)

    def _handle_load_failure(self, error: Exception) -> None:
        """Clear stale canvas state and visibly report an image-load error."""
        # Leaving the previous field visible while _current_index names the
        # failed file would let Save write the old mask under a new filename.
        self._canvas.image = None
        self._canvas.mask = None
        self._canvas.reset_zoom(silent=True)
        self._canvas.clear()
        self._history.clear()
        self._log = None
        self._refresh_history_buttons()
        self._btn_reset_zoom.setEnabled(False)
        self._warn("Load failed", str(error))

    def _apply_loaded_pair(
        self,
        filename: str,
        token: int,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Install a decoded pair if it still represents the selected field."""
        if token != self._load_token:
            return
        self._canvas.set_image_and_mask(image, mask)
        # Reset undo history for the new image and seed with the loaded mask
        self._history.clear()
        self._history.push(mask)
        self._refresh_history_buttons()
        self._btn_reset_zoom.setEnabled(False)
        self._log = self._open_ledger(filename)
        self._status_label.setText(
            f"{filename}  "
            f"({self._current_index + 1}/{len(self._image_files)})"
        )
        # Last, so its status message and its undo step sit on top of the
        # freshly seeded history rather than being wiped by it.
        self.apply_object_filter(on_load=True)

    def _open_ledger(self, filename: str) -> CurationLog:
        """The ledger for one field, ready to be appended to.

        Read from beside the mask so this session continues the record
        rather than starting a new one — the log is written back whole, and
        a fresh one would erase what an earlier session, or the napari
        round-trip, recorded about the same mask. A ledger that already
        names a source keeps it: the tool that made each edit is recorded on
        the edit, not on the file.
        """
        artifact = engine.mask_save_path(self._folder, filename)
        try:
            log = CurationLog.read_beside(artifact)
        except Exception as exc:
            # A damaged sidecar must not cost the user the edit they are
            # about to make. Start a fresh log, and say so rather than
            # quietly overwriting a record nobody can read.
            LOG.warning("Unreadable curation ledger beside %s: %s",
                        artifact, exc)
            log = CurationLog()
        if not log.artifact:
            log.artifact = artifact
            log.source = engine.CURATION_SOURCE
        return log

    def _on_prev(self):
        if not self._image_files or self._current_index <= 0:
            return
        self._current_index -= 1
        self._load_current()

    def _on_next(self):
        if not self._image_files or self._current_index >= len(self._image_files) - 1:
            return
        self._current_index += 1
        self._load_current()

    def _on_save(self):
        if not self._image_files or self._canvas.mask is None:
            return
        try:
            path = engine.save_mask(
                self._folder,
                self._image_files[self._current_index],
                self._canvas.mask,
                log=self._log,
            )
        except Exception as e:
            self._warn("Save failed", str(e))
            return
        edits = len(self._log) if self._log is not None else 0
        note = f"  ({edits} edit(s) recorded)" if edits else ""
        self._status_label.setText(f"Saved → {path}{note}")

    def _apply_op(self, op, kind: str = "edit", **detail):
        """Run a mask -> mask function, refresh, record it, push to history.

        :param kind: the verb this operation goes into the ledger under.
            One word per button, so the ledger's own summary counts the
            buttons the user pressed.
        :param detail: anything about the operation worth keeping with the
            entry, such as the threshold a removal used.
        """
        if self._canvas.mask is None:
            return
        out = op(self._canvas.mask)
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        self._record(kind, None, changed, **detail)
        self._history.push(self._canvas.mask)
        self._refresh_history_buttons()

    def _on_fill_holes(self):
        self._apply_op(engine.fill_holes, "fill_holes")

    def _on_relabel(self):
        self._apply_op(engine.relabel_objects, "relabel")

    def _on_invert(self):
        self._apply_op(engine.invert_mask, "invert")

    def _on_remove_small(self):
        area = int(self._min_area.value())
        self._apply_op(lambda m: engine.remove_small_objects(m, area),
                        "remove_small", min_area=area)

    def _on_clear_mask(self):
        if self._canvas.mask is None:
            return
        if not self._confirm("Clear mask", "Zero out the current mask?"):
            return
        self.clear_mask()

    def clear_mask(self) -> None:
        """Zero the current mask *without* asking, recording it in history.

        The confirmation lives in :meth:`_on_clear_mask`; this is the
        scriptable entry point (and what the undo stack sees).
        """
        self._apply_op(engine.clear_mask, "clear")

    def _on_stroke_started(self):
        # Brush/erase strokes mutate the mask in place; nothing to record
        # until the stroke ends. History already has the pre-stroke mask
        # from the previous op/load.
        pass

    def _on_stroke_finished(self):
        """Record and commit the gesture the canvas just finished.

        Recorded BEFORE the history push, because the entry's size is
        measured against the snapshot the gesture started from and pushing
        would make that snapshot the gesture's own result — every edit would
        then be recorded as having changed nothing.
        """
        if self._canvas.mask is None:
            return
        edit = self._canvas.last_edit or {}
        self._record(str(edit.get("kind") or "paint"), edit.get("target"),
                      self._pixels_changed(self._canvas.mask),
                      **dict(edit.get("detail") or {}))
        self._history.push(self._canvas.mask)
        self._refresh_history_buttons()

    # ------------------------------------------------------------------
    def _sync_button_states(self):
        has_files = bool(self._image_files)
        editable = has_files and not self._loading
        for b in (self._btn_prev, self._btn_next, self._btn_save,
                   self._btn_brush, self._btn_erase, self._btn_del_obj,
                   self._btn_wand_add, self._btn_wand_erase, self._btn_zoom,
                   self._btn_filter, self._btn_otsu):
            b.setEnabled(editable)
