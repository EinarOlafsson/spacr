"""Interactive editing and curation of segmentation masks.

Make Masks is filed under **Tools** on the home screen and is the manual
half of segmentation: it corrects masks a model got wrong, one field at a
time, and its masthead opens the Cellpose workflows that produced them.
:class:`MakeMasksScreen` loads images and labelled masks from
``<folder>/masks`` and saves edited labels as ``uint16`` TIFF files.

THE NINE TOOLS, in :data:`TOOL_MODES` order, because this vocabulary is what
a reader needs before opening the screen:

**Brush** and **Erase** paint and unpaint the active label a pixel at a time.
**Erase object** removes a whole label in one click. **Wand +** and
**Wand −** grow a region from the pixel clicked and add it to the label or
take it out of it; the tolerance is relative to the image's intensity range
by default, see :func:`spacr.qt.mask_engine.relative_tolerance`. **Draw**
traces a free-form outline that closes and fills as ONE object -- the tool a
brush is not, because a brush stamps disks along the path, so tracing a rim
with it labels the rim and leaves the middle background. **Divide** drags a
line across a merged object and makes it two, keeping the original identifier
on the larger component and giving the smaller a new one, with every other
object untouched; :data:`spacr.qt.mask_engine.DIVIDE_CUT_WIDTH` is the cut
width. **Zoom** rectangle-drags the view and changes no labels at all.

**Recrop** is the ninth and is the only one that changes WHICH field is on
screen rather than what is painted on it. A staged crop holding several cells
is not one training example, and curating it as one teaches a network that
two objects are one picture -- so a box round an object writes that region of
both the image and the mask as a field of its own
(:func:`spacr.qt.mask_engine.write_recrop`), queued straight after the current
field, and the multi-object original is retired into
``recropped_originals/`` rather than curated
(:func:`spacr.qt.mask_engine.retire_recropped_original`). A box shorter than
:data:`spacr.qt.mask_engine.RECROP_MIN_SIDE` on a side, or repeating a cut
already made past :data:`spacr.qt.mask_engine.RECROP_MAX_OVERLAP`, is refused;
objects the box cuts through are dropped, because an object whose boundary is
where the mouse was released is not that object; and the labels that survive
are renumbered from one.

Each field has a :class:`spacr.curation.CurationLog`, initialized from any
existing sidecar. :func:`spacr.qt.mask_engine.save_mask` writes the labels and
ledger together, allowing :func:`spacr.curation.is_curated` to distinguish
manually corrected masks from pipeline output. A single gesture produces one
ledger entry and one undo step.

:meth:`MakeMasksScreen.run_cellpose` applies the pipeline's resolved
Cellpose-SAM model to the open field and displays the mask, cell-probability
map, and flow field. The intermediate outputs support evaluation of
:data:`CELLPROB_THRESHOLD` when detections are missing or incomplete.

Editor modes are assembled from :func:`tool_row_entries` and
:data:`TOOL_MODES`. :meth:`MakeMasksScreen.add_toolbar_action` inserts
non-mode actions into the same toolbar. The settings panel carries the
operations that are not gestures -- object filling and relabeling, inversion,
size filtering and Otsu detection, with undo and redo over all of them --
alongside the brush, wand and display controls, and can be hidden to return
its width to the canvas.

Additional segmentation tools are opened from the masthead in
:data:`FOLD_ORDER` through
:class:`~spacr.qt.widgets.fold_strip.FoldStrip`. **Mask the whole folder**
applies Cellpose to every image in the selected folder, while **Save mask** on
the Curate page calls :meth:`spacr.curation.MaskCuration.save_mask`. Folded
modules retain their existing widgets inside :class:`FoldedModulePanel`.

Time-series tracking is integrated with Mask Generation as the Timelapse
settings category; see :mod:`spacr.qt.screens.mask`. Motility Assay instead
belongs to Measure because it consumes existing masks and writes measurement
tables rather than generating masks.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import deque
from functools import partial
from importlib.util import find_spec
from typing import Any, List, NamedTuple, Optional

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QThread,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
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
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
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
from .. import wand_rescue
from ..hidpi import follow_device_ratio, logical_size, scaled_for
from ..theme import SPACING, active_palette, mark_surface
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
#:
#: TIMELAPSE AND MOTILITY ARE NOT HERE. They were, and they were in the
#: wrong home: this screen is hand-curation of masks that already exist,
#: and both of those are things mask GENERATION does over a series --
#: their settings overlap that module's, not this one's tools. They fold
#: into Mask Generation instead, as switches that reveal their own
#: settings categories on its form; see :mod:`spacr.qt.screens.mask`.
FOLD_ORDER = (
    "train_cellpose",
    MASK_FOLDER_KEY,
    "model_compare",
    "model_zoo",
    "curate",
    "napari_bridge",
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
        "Fine-tune a Cellpose model on labelled fields, then apply the "
        "trained model or a stock model to an image folder.",
        "beta"),
    MASK_FOLDER_KEY: (
        "Mask the whole folder",
        "Apply the selected segmentation model to every image in the open "
        "folder.",
        "beta"),
    "model_compare": (
        "Model Compare",
        "Compare two Cellpose models on the same fields using side-by-side "
        "masks, object-count differences and adjusted Rand index (ARI).",
        "stable"),
    "model_zoo": (
        "Model Zoo",
        "Browse, verify, download and benchmark Cellpose and classifier "
        "models on selected fields.",
        "stable"),
    "curate": (
        "Curate",
        "Correct segmentation masks and tracking assignments manually while "
        "recording each edit in the curation log.",
        "alpha"),
    "napari_bridge": (
        "Napari Bridge",
        "Correct a segmentation mask in napari and import the revised labels "
        "into spaCR.",
        "alpha"),
}

#: A folded key that shares another key's screen. The two halves of the
#: Cellpose loop are two tabs of one workbench, so pressing either button has
#: to reach the same widget: a checkpoint trained on one tab is what the
#: other tab segments with, and a second copy of the screen would not have it.
FOLD_HOSTS = {MASK_FOLDER_KEY: "train_cellpose"}

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



MODE_NONE = "none"
MODE_BRUSH = "brush"
MODE_ERASE = "erase"
MODE_ERASE_OBJECT = "erase_object"
MODE_WAND_ADD = "wand_add"
MODE_WAND_ERASE = "wand_erase"
#: Trace a free-form outline; it closes and fills as ONE object. The tool a
#: brush is not: a brush stamps disks along the path, so tracing a rim with
#: it labels the rim and leaves the middle background.
MODE_DRAW = "draw"
#: Drag a line across a merged object and it becomes two, with every other
#: object's id untouched. The commonest correction a segmentation needs.
MODE_DIVIDE = "divide"
MODE_ZOOM = "zoom"
#: Drag a box round one object and that region of BOTH the image and the
#: mask becomes a field of its own, queued straight after this one. The only
#: tool here that changes WHICH field is on screen rather than what is
#: painted on it — see :func:`spacr.qt.mask_engine.write_recrop` for what it
#: writes and :func:`spacr.qt.mask_engine.retire_recropped_original` for
#: what happens to the field it was cut out of.
MODE_RECROP = "recrop"

#: The tools that fill the toolbar row, in the order they appear there:
#: ``(mode, label, icon key)``. THE ROW IS BUILT FROM THIS TABLE and not
#: from a list of literals at the layout site, so adding a tool to the row
#: is adding a line here. A ``MODE_*`` constant that nobody gave a line to
#: still reaches the row — :func:`tool_row_entries` names it after itself
#: — so a tool cannot be invisible because its author did not know this
#: table existed.
TOOL_MODES: List[tuple] = [
    (MODE_BRUSH,        "Brush",        "brush"),
    (MODE_ERASE,        "Erase",        "erase"),
    (MODE_ERASE_OBJECT, "Erase object", "erase_object"),
    (MODE_WAND_ADD,     "Wand +",       "wand_add"),
    (MODE_WAND_ERASE,   "Wand −",       "wand_erase"),
    (MODE_DRAW,         "Draw",         "draw"),
    (MODE_DIVIDE,       "Divide",       "divide"),
    (MODE_ZOOM,         "Zoom",         "zoom"),
    (MODE_RECROP,       "Recrop",       "recrop"),
]


def tool_row_entries() -> List[tuple]:
    """Every canvas tool the toolbar row should hold.

    :data:`TOOL_MODES` first, in its own order, then any other ``MODE_*``
    constant in this module the table does not mention — labelled from
    its own value and drawn with whatever :func:`spacr.qt.iconset.icon`
    has for that name, which is a fallback glyph when it has nothing.
    Alphabetical among themselves, so the row is the same on every run.

    ``MODE_NONE`` is excluded because it is not a tool: it is the canvas
    with no tool held, which is what the row shows when nothing is
    checked.
    """
    entries = list(TOOL_MODES)
    seen = {mode for mode, _label, _icon in entries}
    for name, value in sorted(globals().items()):
        if not name.startswith("MODE_") or name == "MODE_NONE":
            continue
        if not isinstance(value, str) or value in seen:
            continue
        entries.append((value, value.replace("_", " ").capitalize(), value))
        seen.add(value)
    return entries


PAN_MODIFIERS = Qt.ShiftModifier | Qt.AltModifier

#: Smallest zoom viewport, in image pixels. Below a handful of pixels the
#: view is all interpolation and the wheel has nothing left to magnify.
MIN_VIEWPORT = 8

#: Decimals on the display-percentile boxes. Six, because the interesting
#: end of a 16-bit histogram is the last few pixels: on a 2048x2048 field,
#: 99.9999 clips four pixels and 99.9 clips four thousand, and the hot ones
#: are usually the entire reason the field looks black.
PERCENTILE_DECIMALS = 6

#: Starting width of the settings pane, in pixels, and the width it is
#: put back at when the settings button turns it on again after a session
#: that never dragged the splitter.
SETTINGS_WIDTH = 380


class _MaskLoadWorker(QThread):
    """Decode one image/mask pair without blocking Qt's main thread."""

    def __init__(self, folder: str, filename: str, token: int, parent=None):
        """Load one image and its mask off the GUI thread.

        :param folder: the folder holding the pair.
        :param filename: the image to load; the mask is found beside it.
        :param token: identifies the request this worker answers. The screen
            compares it on completion, so a load the user has already
            navigated away from is discarded rather than drawn over the
            image now on screen.
        :param parent: parent object.

        The result and the original exception are both kept as attributes
        rather than raised, because a thread that raises loses the traceback
        the screen needs to say what failed.
        """
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
    magic-wand / erase-object / draw / divide / zoom-rectangle / recrop
    interactions.

    All coordinate math is done against the *full* image; the "zoom
    view" is just a crop of the composited pixmap. Mask edits go
    directly into `self.mask` (with the correct zoom offset applied).

    Recrop is the one gesture here that changes nothing the canvas owns:
    it is a rectangle dragged the way a zoom rectangle is, handed on
    through :attr:`recrop_requested` in full-image pixels for the screen
    to accept or refuse. What comes back is a mark in :attr:`recrop_boxes`
    saying that region has been cut out, which is the only thing on screen
    that distinguishes a box that was written from one that was not.

    :param parent: parent widget; ownership only.
    """

    stroke_started = Signal()
    stroke_finished = Signal()
    zoom_changed = Signal(bool)
    #: A recrop box was dragged, in FULL-image pixels: (x0, y0, x1, y1).
    #: The canvas neither writes it nor judges it — the box may be too
    #: small, or a re-draw of one already cut — because what it becomes is
    #: two files and a queue position, and none of that is a canvas's job.
    recrop_requested = Signal(int, int, int, int)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build an empty canvas: no image, no mask, no stroke in progress."""
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.mode: str = MODE_NONE
        self.brush_radius: int = 10
        self.norm_lo: float = 1.0
        self.norm_hi: float = 99.9
        self.wand_tolerance: float = 1000.0
        self.wand_relative: bool = True
        self.wand_tol_pct: float = 5.0
        self.wand_max_pixels: int = 100_000
        #: The wand's rescues — see :mod:`spacr.qt.wand_rescue` for what
        #: each one catches. They start at the values that tool shipped,
        #: which are tuned to be inert on a flood that did not run away.
        self.wand_trim_runaway: bool = True
        self.wand_runaway_ratio: float = 2.0
        self.wand_runaway_warmup: int = 12
        self.wand_runaway_min_base: int = 8
        self.wand_runaway_confirm: int = 2
        self.wand_intensity_border: bool = True
        self.wand_intensity_steps: int = 8
        self.wand_gradient_taper: bool = True
        self.wand_gradient_sigma: float = 2.0
        self.wand_gradient_margin: int = 8
        self.wand_gradient_erode: int = 3
        self.wand_salvage_over_cap: bool = True
        self.zoom_speed: float = 1.15
        follow_device_ratio(self, self.refresh)

        #: What the stroke that just finished did — ``{"kind", "target",
        #: "detail"}`` — for the screen to put in the curation ledger. Set
        #: by :meth:`_emit_stroke_end` immediately before ``stroke_finished``
        #: so a handler reads the edit it was told about, not the one before.
        self.last_edit: Optional[dict] = None

        self._zoom_x0: Optional[int] = None
        self._zoom_y0: Optional[int] = None
        self._zoom_x1: Optional[int] = None
        self._zoom_y1: Optional[int] = None

        self._zoom_drag_start: Optional[QPoint] = None
        self._zoom_drag_end: Optional[QPoint] = None

        #: Boxes already cut out of THIS field, as
        #: ``(x0, y0, x1, y1, name)`` in image pixels. Drawn on the canvas
        #: and kept there: without them a box that was written and a box
        #: that was refused look identical the moment the mouse comes up,
        #: which is how one object reached disk as three crops.
        self.recrop_boxes: List[tuple] = []

        self._gesture_points: List[QPoint] = []

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {active_palette()['bg']};")
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)
        self._last_pt: Optional[QPoint] = None
        self._stroke_in_progress = False

        self._sweeping = False
        self._sweep_labels: List[int] = []

        self._pan_from: Optional[QPoint] = None

        #: The live magnifier the screen gives this canvas, or None. While it
        #: is on, a left click commits the objects its box outlines and the
        #: wheel zooms the box instead of the view; the right-button sweep
        #: and Shift/Alt pan work as they do from any tool.
        self.magnifier: Optional["_LiveMagnifier"] = None

    def set_image_and_mask(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Load a new image + mask pair and rerender at full-image zoom.

        :param image: uint16 grayscale array to display underneath.
        :param mask: uint8/uint16 label array painted on top.
        """
        self.image = image
        self.mask = mask
        self._gesture_points = []
        self.recrop_boxes = []
        if self.magnifier is not None:
            self.magnifier.forget()
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
        pixmap = scaled_for(pixmap, self, avail_w, avail_h)
        self.setPixmap(pixmap)

    def _canvas_to_image(self, x: float, y: float) -> Optional[tuple]:
        """Widget coordinates to IMAGE pixel coordinates, or ``None``.

        ``None`` means the point is outside the drawn pixmap -- in the letterbox
        either side of it, or before an image is set -- and a caller must not
        treat that as pixel 0, which is what an unchecked conversion gives.

        Accounts for the centring offset and for the zoom viewport, then clamps:
        a click on the last row must land on the last row rather than one past it,
        which is a rounding error away.
        """
        p = self.pixmap()
        if self.mask is None or p is None or p.isNull():
            return None
        shown = logical_size(p)
        pw, ph = shown.width(), shown.height()
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
        img_x = max(0, min(self.mask.shape[1] - 1, img_x))
        img_y = max(0, min(self.mask.shape[0] - 1, img_y))
        return img_x, img_y

    def _image_to_canvas(self, img_x: float, img_y: float) -> Optional[QPoint]:
        """Where an image pixel lands on the widget, or ``None``.

        The inverse of :meth:`_canvas_to_image`, and the reason a recrop box
        stays on the object it was drawn round while the view is zoomed and
        panned: the boxes are kept in image pixels and mapped here on every
        repaint, rather than being remembered as the widget coordinates the
        mouse happened to be at.
        """
        p = self.pixmap()
        if self.mask is None or p is None or p.isNull():
            return None
        shown = logical_size(p)
        pw, ph = shown.width(), shown.height()
        ox = (self.width() - pw) // 2
        oy = (self.height() - ph) // 2
        x0, y0, x1, y1 = self._viewport_bounds()
        sub_w = max(1, x1 - x0)
        sub_h = max(1, y1 - y0)
        return QPoint(int(round(ox + (float(img_x) - x0) * pw / sub_w)),
                      int(round(oy + (float(img_y) - y0) * ph / sub_h)))

    def _image_delta(self, dx_px: float, dy_px: float) -> tuple:
        """Widget-pixel drag -> the image-pixel shift the viewport must take.

        Negated, because a pan moves the *view*, not the picture: dragging
        the content to the right has to slide the window left over the
        image for the pixel under the cursor to stay under the cursor.
        """
        p = self.pixmap()
        shown = logical_size(p)
        if self.mask is None or not shown.width():
            return (0, 0)
        x0, y0, x1, y1 = self._viewport_bounds()
        return (int(round(-dx_px * (x1 - x0) / shown.width())),
                int(round(-dy_px * (y1 - y0) / shown.height())))

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
        if self.magnifier is not None and self.magnifier.enabled:
            # The magnifier's zoom, not the view's: one wheel, one meaning.
            self.magnifier.wheel(notches > 0)
            self.update()
            event.accept()
            return
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

    def wand_rescue_settings(self) -> dict:
        """The rescue settings, keyed as :mod:`spacr.qt.wand_rescue` wants.

        One place builds this dict, so a control added to the panel reaches
        the flood by being read here rather than by being threaded through
        the click handler as well.
        """
        return {
            "trim_runaway": bool(self.wand_trim_runaway),
            "runaway_ratio": float(self.wand_runaway_ratio),
            "runaway_warmup": int(self.wand_runaway_warmup),
            "runaway_min_base": int(self.wand_runaway_min_base),
            "runaway_confirm": int(self.wand_runaway_confirm),
            "intensity_border": bool(self.wand_intensity_border),
            "intensity_steps": int(self.wand_intensity_steps),
            "gradient_taper": bool(self.wand_gradient_taper),
            "gradient_sigma": float(self.wand_gradient_sigma),
            "gradient_margin": int(self.wand_gradient_margin),
            "gradient_erode": int(self.wand_gradient_erode),
            "salvage_over_cap": bool(self.wand_salvage_over_cap),
        }

    def _mask_radius_for_brush(self) -> int:
        """Scale the brush radius (in screen px) to full-image px, taking
        the current zoom into account."""
        p = self.pixmap()
        shown = logical_size(p)
        if self.mask is None or not shown.width():
            return self.brush_radius
        x0, _, x1, _ = self._viewport_bounds()
        sub_w = max(1, x1 - x0)
        return max(1, int(self.brush_radius * sub_w / shown.width()))

    def paintEvent(self, event):
        """Draw the base pixmap plus whichever gesture is in flight.

        A draw or divide only reaches the mask on release, so until then the
        outline being traced and the cut being aimed exist nowhere but here:
        without the preview the user is dragging an invisible line.
        """
        super().paintEvent(event)
        self._paint_recrop_boxes()
        self._paint_magnifier()
        if self.mode in (MODE_DRAW, MODE_DIVIDE):
            self._paint_gesture()
            return
        if self.mode not in (MODE_ZOOM, MODE_RECROP):
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

    def _paint_magnifier(self) -> None:
        """Draw the live magnifier's box, when there is one to draw."""
        magnifier = self.magnifier
        if magnifier is None or not magnifier.enabled:
            return
        painter = QPainter(self)
        try:
            magnifier.paint(painter)
        finally:
            painter.end()

    def _paint_recrop_boxes(self) -> None:
        """Mark every region already cut out of this field, with its name.

        A recrop writes two files somewhere else and leaves the field on
        screen untouched, so without this the canvas looks exactly the same
        whether the box was written or refused. That is not a cosmetic
        difference: the standalone this came from put one object on disk
        three times as three near-identical crops, because the user could
        only tell a box had worked by drawing it again.
        """
        rendered = self.pixmap()
        if not self.recrop_boxes or self.mask is None \
                or rendered is None or rendered.isNull():
            return
        painter = QPainter(self)
        accent = QColor(active_palette()["accent"])
        fill = QColor(accent)
        fill.setAlpha(55)
        for box in self.recrop_boxes:
            x0, y0, x1, y1 = (int(v) for v in box[:4])
            rect = QRect(self._image_to_canvas(x0, y0),
                          self._image_to_canvas(x1, y1)).normalized()
            painter.fillRect(rect, fill)
            pen = QPen(accent)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
            name = str(box[4]) if len(box) > 4 else ""
            if name:
                painter.setPen(QPen(QColor(active_palette()["fg"])))
                painter.drawText(rect.adjusted(6, 4, 0, 0),
                                 Qt.AlignLeft | Qt.AlignTop, name)

    def _paint_gesture(self) -> None:
        """Draw the outline being traced, or the cut being aimed.

        The draw preview shows the segment that will close the loop as a
        dashed line back to the first point, because that segment is part of
        what gets filled and is the one part of the outline the user did not
        trace.
        """
        if len(self._gesture_points) < 2:
            return
        painter = QPainter(self)
        colour = QColor(active_palette()["accent"])
        if self.mode == MODE_DIVIDE:
            pen = QPen(colour)
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(self._gesture_points[0], self._gesture_points[-1])
            return
        pen = QPen(colour)
        pen.setWidth(2)
        painter.setPen(pen)
        for start, end in zip(self._gesture_points, self._gesture_points[1:]):
            painter.drawLine(start, end)
        closing = QPen(colour)
        closing.setWidth(1)
        closing.setStyle(Qt.DashLine)
        painter.setPen(closing)
        painter.drawLine(self._gesture_points[-1], self._gesture_points[0])

    def _emit_stroke_start(self):
        """Open a stroke, once. A stroke already open is not reopened.

        The ledger records one edit per stroke, so a second start would split a
        single drag into two entries.
        """
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
        down and picking it up again. With the magnifier on in whole-image
        scope the right button removes the one object under it instead of
        sweeping.
        """
        if self.mask is None:
            return super().mousePressEvent(event)

        if (event.button() == Qt.RightButton and self.magnifier is not None
                and self.magnifier.enabled
                and self.magnifier.scope == "image"):
            self.magnifier.hover(event.position())
            self.magnifier.remove()
            self.update()
            return

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

        if (event.button() == Qt.LeftButton and self.magnifier is not None
                and self.magnifier.enabled):
            self.magnifier.hover(event.position())
            self.magnifier.press()
            self.update()
            return

        if self.mode == MODE_NONE:
            return super().mousePressEvent(event)

        if self.mode in (MODE_ZOOM, MODE_RECROP):
            self._zoom_drag_start = event.position().toPoint()
            self._zoom_drag_end = event.position().toPoint()
            self.update()
            return

        pt = self._canvas_to_image(event.position().x(), event.position().y())
        if pt is None:
            return

        if self.mode in (MODE_DRAW, MODE_DIVIDE):
            self._gesture_points = [event.position().toPoint()]
            self.update()
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
            self.mask, report = wand_rescue.magic_wand(
                self.image, self.mask, pt[0], pt[1],
                tolerance, self.wand_max_pixels, action=action,
                **self.wand_rescue_settings(),
            )
            self.refresh()
            self._emit_stroke_end(
                kind="wand", target=(255 if action == "add" else 0),
                action=action, tolerance=round(float(tolerance), 3),
                relative=bool(self.wand_relative), **report,
            )
            return

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
            if (dx or dy) and self.pan_by(dx, dy):
                self._pan_from = now
            return
        if self.magnifier is not None and self.magnifier.enabled:
            self.magnifier.hover(event.position())
            if event.buttons() & Qt.LeftButton:
                self.magnifier.drag()
            self.update()
            return
        if self.mode in (MODE_ZOOM, MODE_RECROP) \
                and self._zoom_drag_start is not None \
                and event.buttons() & Qt.LeftButton:
            self._zoom_drag_end = event.position().toPoint()
            self.update()
            return
        if self.mode in (MODE_DRAW, MODE_DIVIDE) and self._gesture_points \
                and event.buttons() & Qt.LeftButton:
            now = event.position().toPoint()
            if self.mode == MODE_DIVIDE:
                self._gesture_points = [self._gesture_points[0], now]
            else:
                self._gesture_points.append(now)
            self.update()
            return
        if self.mode in (MODE_BRUSH, MODE_ERASE) and event.buttons() & Qt.LeftButton:
            pt = self._canvas_to_image(event.position().x(), event.position().y())
            if pt is None:
                return
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
            self._emit_stroke_end(kind="sweep_delete", target=list(labels),
                                   n_objects=len(labels))
            return
        if self._pan_from is not None and event.button() == Qt.LeftButton:
            self._pan_from = None
            self.unsetCursor()
            return
        if (event.button() == Qt.LeftButton and self.magnifier is not None
                and (self.magnifier.enabled
                     or self.magnifier._stroke is not None)):
            self.magnifier.release()
            self.update()
            return
        if (event.button() == Qt.RightButton and self.magnifier is not None
                and self.magnifier.enabled
                and self.magnifier.scope == "image"):
            return
        if self.mode in (MODE_ZOOM, MODE_RECROP) \
                and self._zoom_drag_start is not None \
                and self._zoom_drag_end is not None:
            p0 = self._canvas_to_image(self._zoom_drag_start.x(),
                                        self._zoom_drag_start.y())
            p1 = self._canvas_to_image(self._zoom_drag_end.x(),
                                        self._zoom_drag_end.y())
            self._zoom_drag_start = None
            self._zoom_drag_end = None
            if self.mode == MODE_RECROP:
                if p0 is not None and p1 is not None:
                    self.recrop_requested.emit(int(p0[0]), int(p0[1]),
                                                int(p1[0]) + 1,
                                                int(p1[1]) + 1)
                self.update()
                return
            if p0 is not None and p1 is not None:
                x0, x1 = sorted((p0[0], p1[0]))
                y0, y1 = sorted((p0[1], p1[1]))
                if x1 - x0 > 4 and y1 - y0 > 4:
                    self._zoom_x0, self._zoom_y0 = x0, y0
                    self._zoom_x1, self._zoom_y1 = x1 + 1, y1 + 1
                    self.zoom_changed.emit(True)
            self.refresh()
            return
        if self.mode in (MODE_DRAW, MODE_DIVIDE) and self._gesture_points:
            points, self._gesture_points = self._gesture_points, []
            self._finish_region_gesture(points)
            self.update()
            return
        if self._last_pt is not None:
            self._last_pt = None
        self._emit_stroke_end(
            kind="erase" if self.mode == MODE_ERASE else "paint",
            target=(0 if self.mode == MODE_ERASE else 255),
            radius=int(self.brush_radius),
        )

    def _finish_region_gesture(self, points) -> None:
        """Commit a finished draw / divide gesture, or drop it.

        The mask is touched here and nowhere else for these two tools, and
        only when the gesture did something: a traced outline that enclosed
        nothing and a line that separated nothing both leave the mask, the
        undo history and the ledger exactly as they were.

        The path is converted to image pixels here rather than as it is
        drawn, so that a gesture whose points fall outside the pixmap loses
        those points instead of the whole edit.
        """
        if self.mask is None:
            return
        image_points = [p for p in
                        (self._canvas_to_image(q.x(), q.y()) for q in points)
                        if p is not None]

        if self.mode == MODE_DIVIDE:
            if len(image_points) < 2:
                return
            divided, splits = engine.divide_object(
                self.mask, image_points[0], image_points[-1])
            if not splits:
                return
            self._emit_stroke_start()
            self.mask = divided
            self.refresh()
            self._emit_stroke_end(
                kind="divide",
                target=[int(source) for source, _ in splits],
                new_labels=[int(made) for _, made in splits],
                n_objects=len(splits),
            )
            return

        filled, new_label = engine.fill_polygon(self.mask, image_points)
        if not new_label:
            return
        self._emit_stroke_start()
        self.mask = filled
        self.refresh()
        self._emit_stroke_end(kind="draw", target=int(new_label),
                               n_points=len(image_points))

    def resizeEvent(self, event):
        """Refit the composited pixmap to the new canvas size."""
        super().resizeEvent(event)
        self.refresh()

    def leaveEvent(self, event):
        """Put the magnifier's box away when the mouse leaves the canvas."""
        if self.magnifier is not None and self.magnifier.enabled:
            self.magnifier.hover(None)
            self.update()
        super().leaveEvent(event)




#: Cellpose's own default for the threshold on the cell-probability map.
#: Lowering it keeps dimmer pixels, raising it keeps only confident ones.
CELLPROB_THRESHOLD = 0.0

#: Cellpose's own default for the flow-error threshold. A candidate mask
#: whose flows disagree with the ones the network predicted by more than
#: this is thrown away, so lowering it is stricter, not looser.
FLOW_THRESHOLD = 0.4

#: What the Recrop button says about itself. The only tool in the row
#: whose result is not a change to the picture under the cursor, so it is
#: the one that cannot be understood by pressing it and looking.
RECROP_TOOLTIP = (
    "Drag a box round one object and that region of the image and the mask "
    "becomes a field of its own, queued straight after this one. Objects "
    "the box cuts through are dropped. The field you cut them from is "
    "moved into recropped_originals/ when you move on, not deleted."
)

#: What the probability and flow panes say before Cellpose has run. They
#: are empty for a reason and the reason is worth a sentence: a blank
#: black pane reads as a broken view.
FLOW_RESTING_TEXT = (
    "Run Cellpose-SAM to see the cell-probability map\n"
    "and the flow field for this field."
)


def stretch_to_uint8(array: np.ndarray,
                     lower_pct: float = 1.0,
                     upper_pct: float = 99.0) -> np.ndarray:
    """Percentile-stretch any float array to 0-255 uint8.

    The probability map runs roughly -12..+8 and the flow components
    ±5; the intensity image is 16-bit counts. Shown raw beside each
    other they are three different scales and none of them is readable.
    Stretching each one by its own percentiles is what puts them on the
    same footing as the contrast-stretched intensity image the canvas
    already draws, which is the only way the panes can be compared with
    it by eye.
    """
    values = np.asarray(array, dtype=np.float32)
    if not values.size:
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(values, lower_pct))
    hi = float(np.percentile(values, upper_pct))
    span = max(hi - lo, 1e-6)
    scaled = np.clip((values - lo) / span, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def cellprob_heatmap(cellprob: np.ndarray) -> np.ndarray:
    """Cellpose's probability map as an RGB heatmap, ``(H, W, 3)`` uint8.

    A greyscale probability map beside a greyscale image is two pictures
    that look alike and mean different things. A colour ramp says at a
    glance which pixels the network was confident about, which is the
    whole reason to look at this map before moving a threshold.

    Matplotlib's ``magma`` is used where it is importable, and a plain
    black-to-white ramp stands in where it is not, so the pane is never
    the thing that fails.
    """
    scaled = stretch_to_uint8(cellprob).astype(np.float32) / 255.0
    try:
        import matplotlib
        cmap = matplotlib.colormaps["magma"]
    except Exception:
        return np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=2)
    return (np.asarray(cmap(scaled))[..., :3] * 255).astype(np.uint8)


def flow_rgb(flow: np.ndarray) -> Optional[np.ndarray]:
    """Cellpose's flow field as ``(H, W, 3)`` uint8, or None if it is not one.

    ``eval`` hands back the flow field twice over and the two entries are
    not the same thing: ``flows[0]`` is already an RGB *picture* of the
    field (hue = direction), while ``flows[1]`` is the raw ``(2, H, W)``
    vector field. This takes the picture where it is given one and builds
    an equivalent from the vectors otherwise, so the pane fills whichever
    entry a caller passes.
    """
    if flow is None:
        return None
    array = np.asarray(flow)
    if array.ndim == 3 and array.shape[0] == 2:
        dy, dx = stretch_to_uint8(array[0]), stretch_to_uint8(array[1])
        mag = stretch_to_uint8(np.hypot(array[0], array[1]))
        return np.ascontiguousarray(np.stack([dx, dy, mag], axis=-1))
    if array.ndim == 3 and array.shape[2] >= 3:
        return np.ascontiguousarray(array[..., :3].astype(np.uint8))
    return None


def cellpose_intermediates(flows) -> tuple:
    """Pull ``(cellprob, flow_rgb)`` out of one image's ``flows`` entry.

    Measured against cellpose 4.2.1.1: ``CellposeModel.eval`` returns
    ``(masks, flows, styles)``, and for one 2-D image ``flows`` is a list
    of three arrays that are three different things —
    ``flows[0]`` an ``(H, W, 3)`` uint8 RGB rendering of the field,
    ``flows[1]`` the ``(2, H, W)`` float32 vectors, and
    ``flows[2]`` the ``(H, W)`` float32 cell-probability map. Indexing it
    as though the members were interchangeable is how a flow pane ends up
    showing the probability map.

    :param flows: one image's flows list, as
        :func:`spacr.spacr_cellpose.parse_cellpose4_output` hands it over
        per image, or the raw list from a single-image ``eval``.
    :returns: ``(cellprob, rgb)``, either of which may be None when this
        Cellpose did not produce it.
    """
    if flows is None:
        return None, None
    members = list(flows) if isinstance(flows, (list, tuple)) else [flows]
    cellprob = None
    if len(members) > 2 and members[2] is not None:
        cellprob = np.asarray(members[2], dtype=np.float32)
    rgb = flow_rgb(members[0]) if members else None
    if rgb is None and len(members) > 1:
        rgb = flow_rgb(members[1])
    return cellprob, rgb


def load_cellpose_model(model_name: str):
    """Load a Cellpose model through spaCR's own resolver.

    :func:`spacr.utils._resolve_cellpose_pretrained` is what the pipeline
    itself calls: it maps every pre-SAM name onto ``cpsam``, keeps a
    fine-tuned checkpoint path as itself, and raises rather than quietly
    substituting stock weights for a checkpoint that is not there. Going
    around it with a second, simpler call would give this screen a
    different answer from the run it is meant to be correcting.
    """
    import torch
    from cellpose import models as cp_models

    from ...utils import _resolve_cellpose_pretrained

    pretrained = _resolve_cellpose_pretrained(model_name)
    from ...accelerator import cellpose_kwargs

    return cp_models.CellposeModel(pretrained_model=pretrained,
                                    **cellpose_kwargs())


def cellpose_detect(image: np.ndarray, model, *,
                    diameter: int = 0,
                    normalize: bool = True,
                    flow_threshold: float = FLOW_THRESHOLD,
                    cellprob_threshold: float = CELLPROB_THRESHOLD,
                    min_size: int = 0) -> tuple:
    """Segment one field with ``model``; return labels and both intermediates.

    The image goes in as a **batch of one**, which is what
    :func:`spacr.spacr_cellpose.parse_cellpose4_output` — the repository's
    own reader of this return value — is written for. Handed a bare 2-D
    array instead, ``eval`` returns a flat three-member flows list and
    that function reads ``len(masks)`` as the number of images and finds
    the image height, so the parse fails on an image that segmented
    perfectly well.

    ``diameter`` is passed as None when it is 0, which is Cellpose's
    "work it out from the image"; it is the one pre-SAM sizing argument
    Cellpose 4 still honours, since ``eval`` rescales by ``30/diameter``.

    :param image: one 2-D field, as the canvas holds it.
    :param model: a loaded ``CellposeModel`` (see
        :func:`load_cellpose_model`), or anything with the same ``eval``.
    :returns: ``(labels, cellprob, flow_rgb)`` — an int32 label image, and
        the two maps as :func:`cellpose_intermediates` reads them.
    """
    import inspect

    from ...spacr_cellpose import cellpose_channel_axis, parse_cellpose4_output

    field = np.asarray(image)
    kwargs = dict(
        batch_size=1,
        normalize=bool(normalize),
        channel_axis=cellpose_channel_axis(field),
        diameter=(int(diameter) or None),
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=int(min_size),
    )
    try:
        params = inspect.signature(model.eval).parameters
    except (TypeError, ValueError):
        params = None
    if params is not None and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    output = model.eval([field], **kwargs)
    masks, flows0, flows1, flows2, _flows3 = parse_cellpose4_output(output)
    labels = np.asarray(masks[0], dtype=np.int32)
    cellprob, rgb = cellpose_intermediates(
        [flows0[0] if flows0 else None,
         flows1[0] if flows1 else None,
         flows2[0] if flows2 else None])
    return labels, cellprob, rgb


# ---------------------------------------------------------------------------
# The live magnifier
# ---------------------------------------------------------------------------

#: The magnifier's starting settings and their ranges. Size is the side of the
#: square region the model segments, in IMAGE pixels; zoom is how many times
#: larger than the canvas draws that region the box draws it; a sensitivity
#: of 0 is each model's own default cut.
_MAGNIFIER_SIZE = 128
_MAGNIFIER_SIZE_RANGE = (32, 512)
_MAGNIFIER_ZOOM = 2.0
_MAGNIFIER_ZOOM_RANGE = (1.0, 8.0)
_MAGNIFIER_SENSITIVITY = 0.0
_MAGNIFIER_SENSITIVITY_RANGE = (-6.0, 6.0)

#: What the magnifier segments, in the order the Segment box offers them.
#: ``region`` runs the model on the box as the mouse moves; ``image`` runs it
#: once on the whole field in the background, and the box then reads those
#: objects instead of asking the model again.
_MAGNIFIER_SCOPES = ("region", "image")

#: Serialises every use of a Cellpose model between the magnifier's worker
#: thread and the screen's own detect button, which share the loaded models.
#: Re-entrant, because the detect button loads a model inside the same hold.
_CELLPOSE_LOCK = threading.RLock()


class _MagnifierRequest(NamedTuple):
    """One region to segment, with everything the model reads copied in.

    Copied rather than referenced: the worker reads it on another thread
    while the GUI thread goes on editing the mask and moving the box.
    ``key`` is what makes two requests the same request -- the field, the box
    (or ``"image"`` for the whole field), every setting a model reads and,
    for a region, whether cut objects are left out -- and it is what a click
    is matched to its result by.
    """

    key: tuple
    crop: np.ndarray
    box: tuple
    shape: tuple
    mode: str
    sensitivity: float
    bright: bool
    min_area: int
    model_name: str
    diameter: int
    colour: tuple
    #: Whether the objects the box's own edge cuts are left out of the answer.
    exclude_border: bool = True
    #: ``region`` for the box under the mouse, ``image`` for the whole field.
    scope: str = "region"


class _MagnifierResult(NamedTuple):
    """The objects found for one request, in that request's CROP coordinates.

    ``labels`` is exactly what the box outlines and exactly what a click
    commits: unless the request keeps them, it has already lost the objects
    the box edge cut (:func:`spacr.qt.mask_engine._drop_cut_objects`). A
    whole-image request's crop is the whole field, so nothing in it is cut.
    ``mode`` is the mode that actually ran and ``note`` says why when that is
    not the one asked for; ``overlay`` is the RGBA picture of the outlines,
    or None for a whole-image result, whose box draws its own slice of it;
    ``count`` is how many objects ``labels`` holds.
    """

    request: _MagnifierRequest
    labels: np.ndarray
    mode: str
    note: str
    overlay: Optional[np.ndarray]
    count: int = 0


def _classical_segmenter(request: _MagnifierRequest, load_model=None):
    """Threshold and watershed the region; needs nothing installed."""
    return engine._classical_region_labels(
        request.crop, sensitivity=request.sensitivity,
        bright=request.bright, min_area=request.min_area)


def _cellpose_segmenter(request: _MagnifierRequest, load_model=None):
    """Segment the region with the Cellpose model the screen has chosen.

    Sensitivity is Cellpose's cell-probability threshold with the sign
    reversed, so raising the magnifier's sensitivity keeps MORE objects, as
    the word says, where raising the threshold keeps fewer.
    """
    loader = load_model or load_cellpose_model
    with _CELLPOSE_LOCK:
        model = loader(request.model_name)
        labels, _cellprob, _flow = cellpose_detect(
            request.crop, model,
            diameter=int(request.diameter),
            normalize=True,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=-float(request.sensitivity),
            min_size=int(request.min_area),
        )
    return labels


#: ``mode -> segmenter``, in the order the Mode box offers them. A segmenter
#: takes ``(request, load_model)`` and returns labels shaped like
#: ``request.crop``. ADDING A MODEL IS ONE FUNCTION AND ONE LINE HERE --
#: DINOCell or SAMCell would be ``"dinocell": _dinocell_segmenter`` -- and
#: :func:`_segment_region` gives it the classical fallback for nothing.
_MAGNIFIER_SEGMENTERS = {
    "classical": _classical_segmenter,
    "cellpose": _cellpose_segmenter,
}


def _segment_region(request: _MagnifierRequest, load_model=None) -> tuple:
    """Run the request's mode, falling back to classical when it cannot run.

    The fallback is why the magnifier never simply does nothing: a model
    whose import or weights fail answers with the classical mode's objects
    and a note saying why, rather than with an empty box.

    :param request: the region and its settings.
    :param load_model: ``name -> model`` for modes that load one; called on
        the worker thread.
    :returns: ``(labels, mode_used, note)``; ``note`` is empty unless the mode
        asked for could not run.
    """
    segmenter = _MAGNIFIER_SEGMENTERS.get(request.mode)
    note = ""
    if segmenter is None:
        note = f"no magnifier mode is called {request.mode!r}"
    elif segmenter is not _classical_segmenter:
        try:
            return segmenter(request, load_model), request.mode, ""
        except Exception as exc:                            # noqa: BLE001
            LOG.warning("magnifier mode %s could not run; using classical",
                        request.mode, exc_info=True)
            note = f"{type(exc).__name__}: {exc}"
    return _classical_segmenter(request, load_model), "classical", note


def _candidate_overlay(labels: np.ndarray, colour) -> np.ndarray:
    """The objects found, as RGBA: a faint fill inside a solid outline."""
    from skimage.segmentation import find_boundaries

    lab = np.asarray(labels)
    rgba = np.zeros(lab.shape + (4,), dtype=np.uint8)
    inside = lab > 0
    if inside.any():
        red, green, blue = (int(c) for c in tuple(colour)[:3])
        rgba[inside] = (red, green, blue, 60)
        rgba[find_boundaries(lab, mode="inner") & inside] = (
            red, green, blue, 255)
    return rgba


def _rgba_qimage(rgba: np.ndarray) -> QImage:
    """An owned QImage of an ``(H, W, 4)`` uint8 array."""
    data = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = data.shape[:2]
    return QImage(data.data, width, height, 4 * width,
                  QImage.Format_RGBA8888).copy()


def _object_count(labels: np.ndarray) -> int:
    """How many distinct objects a label image holds."""
    return int(np.count_nonzero(np.unique(np.asarray(labels))))


def _single_object(result: _MagnifierResult, label: int) -> _MagnifierResult:
    """One object of a whole-image result, as a result of its own.

    Cut to the object's bounding box, so a commit pastes that small window at
    its corner rather than the whole field. Every pixel of the object goes
    in, including any part outside the box on screen: the whole image was
    segmented, so nothing cut it.

    :param result: a whole-image result.
    :param label: an id present in ``result.labels``.
    """
    body = np.asarray(result.labels) == int(label)
    rows = np.flatnonzero(body.any(axis=1))
    cols = np.flatnonzero(body.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    labels = np.where(body[y0:y1, x0:x1], int(label), 0).astype(np.int32)
    request = result.request._replace(box=(x0, y0, x1, y1))
    return result._replace(request=request, labels=labels, overlay=None,
                           count=1)


class _NewestRequestWorker:
    """Runs requests one at a time on a background thread, newest first.

    A mouse that has moved on makes every region still waiting stale, so a
    request superseded before it starts is DROPPED rather than run:
    :meth:`submit` replaces the one request waiting instead of queueing
    behind it. The pace is the worker's own -- the next request is taken when
    the last one finishes, not on a timer -- so a slow model is asked less
    often rather than falling further and further behind the mouse.

    A request submitted with ``pin=True`` is one a click is waiting on.
    Pinned requests run before the newest unpinned one, in the order they
    were clicked, and nothing supersedes them.

    A plain Python thread rather than a QThread: it owns no Qt object, starts
    when there is work and exits as soon as there is none, so an idle
    magnifier holds no thread at all and a closing screen has nothing to
    destroy out from under a running one.

    :param work: ``request -> result``, run on the worker thread.
    :param deliver: ``(request, result, error) -> None``, run on the worker
        thread after each request, ``error`` being the exception ``work``
        raised or None. It must hand over to the GUI thread itself.
    :param name: the thread's name, as a stack dump shows it.
    """

    def __init__(self, work, deliver, name: str = "spacr-magnifier"):
        """Build a worker with no thread; the first request starts one."""
        self._work = work
        self._deliver = deliver
        self._name = str(name)
        self._lock = threading.Lock()
        self._pending = None
        self._pinned: deque = deque()
        self._running = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        #: How many requests were replaced before they started.
        self.superseded = 0

    def submit(self, request, *, pin: bool = False) -> bool:
        """Ask for ``request``; return False once the worker is closed.

        :param request: anything with a ``key``; two requests with one key
            are the same request, and the second is not run again while the
            first is waiting or running.
        :param pin: True for a request a click is waiting on.
        """
        key = getattr(request, "key", None)
        with self._lock:
            if self._closed:
                return False
            running = getattr(self._running, "key", None)
            if pin:
                already = key is not None and (
                    running == key
                    or any(getattr(p, "key", None) == key for p in self._pinned))
                if not already:
                    if (self._pending is not None and key is not None
                            and getattr(self._pending, "key", None) == key):
                        self._pending = None
                    self._pinned.append(request)
            elif key is not None and running == key:
                if self._pending is not None:
                    self.superseded += 1
                self._pending = None
            else:
                if self._pending is not None:
                    self.superseded += 1
                self._pending = request
            if self._thread is None and (self._pinned or self._pending is not None):
                self._thread = threading.Thread(
                    target=self._loop, name=self._name, daemon=True)
                self._thread.start()
        return True

    def idle(self) -> bool:
        """Whether nothing is running or waiting."""
        with self._lock:
            return self._thread is None

    def drop_waiting(self) -> int:
        """Drop every request not yet started; one already running finishes.

        :returns: how many requests were dropped.
        """
        with self._lock:
            dropped = len(self._pinned) + (self._pending is not None)
            self._pending = None
            self._pinned.clear()
        return int(dropped)

    def close(self, timeout: float = 2.0) -> bool:
        """Drop what is waiting and wait up to ``timeout`` for what is running.

        :returns: True when no worker thread is left running.
        """
        with self._lock:
            self._closed = True
            self._pending = None
            self._pinned.clear()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(max(0.0, float(timeout)))
        return not thread.is_alive()

    def _loop(self) -> None:
        """Take requests until there are none; the thread then ends."""
        me = threading.current_thread()
        try:
            while True:
                with self._lock:
                    if self._closed or (self._pending is None
                                        and not self._pinned):
                        self._thread = None
                        return
                    if self._pinned:
                        request = self._pinned.popleft()
                    else:
                        request, self._pending = self._pending, None
                    self._running = request
                result, error = None, None
                try:
                    result = self._work(request)
                except Exception as exc:                    # noqa: BLE001
                    error = exc
                with self._lock:
                    self._running = None
                    closed = self._closed
                if closed:
                    continue
                try:
                    self._deliver(request, result, error)
                except Exception:                           # noqa: BLE001
                    LOG.exception("a magnifier result could not be delivered")
        finally:
            with self._lock:
                self._running = None
                if self._thread is me:
                    self._thread = None


class _LiveMagnifier(QObject):
    """A box under the mouse that shows its region magnified and segmented.

    FOUR COORDINATE SYSTEMS MEET HERE, and every conversion between them is in
    this class or in :func:`spacr.qt.mask_engine._magnifier_box`:

    * **widget** -- where the mouse is on the canvas, in logical pixels;
    * **image** -- the field's own pixels, which the mask shares;
    * **crop** -- the region the model sees, ``image[y0:y1, x0:x1]``, clipped
      at the image border and never padded;
    * **lens** -- the crop drawn ``zoom`` times larger than the canvas draws
      it, placed so the image pixel under the cursor stays under the cursor.

    The model answers in crop coordinates and a commit pastes that answer at
    ``(x0, y0)``, so nothing a model returns is ever in lens or widget pixels
    and zoom cannot move a committed object.

    Inference runs on a :class:`_NewestRequestWorker`. This object builds
    requests on the GUI thread, draws the box, and hands the result a click
    asked for to the screen, which owns the mask, the undo history and the
    ledger.

    A CLICK COMMITS THE RESULT FOR THE REGION CLICKED. If that result is
    already on screen it goes in at once; if the box is still updating, the
    click's request is pinned on the worker -- later mouse movement cannot
    supersede it -- and it goes in when it arrives.

    WHOLE-IMAGE SCOPE runs the model ONCE on the entire field, on a second
    worker, and the box then draws its slice of those objects with no model
    call per move. A click commits only the object under it, and a
    right-click asks the screen to remove the mask object under it. A change
    to any setting the model reads discards those objects and starts a new
    run; a run that was cancelled or failed is not started again until a
    click, or a setting, asks for it.

    :param canvas: the canvas the box is drawn over.
    :param parent: owning QObject.
    :param load_model: ``name -> model`` for modes that need one; called on
        the worker thread.
    :param context: ``() -> dict`` of the screen's own settings a model reads
        -- ``model_name``, ``diameter``, ``bright``, ``min_area`` -- read on
        the GUI thread whenever a request is built.
    """

    _delivered = Signal(object)
    #: The result a click was waiting on; the screen commits it.
    commit_ready = Signal(object)
    #: The wheel moved the zoom; carries the new value.
    zoom_changed = Signal(float)
    #: A sentence for the status line.
    status = Signal(str)
    #: The mask object under the mouse should go; carries image ``(x, y)``.
    remove_requested = Signal(int, int)
    #: A whole-image run started (True) or ended (False).
    busy_changed = Signal(bool)
    #: A press-and-drag's objects (item 417) as ``(outcome, final)``: shown
    #: while the button is down, committed once when ``final``.
    drag_ready = Signal(object)

    def __init__(self, canvas, parent=None, *, load_model=None, context=None):
        """Build a magnifier that is off and holds no thread."""
        super().__init__(parent)
        from ..bridge import emit_safely

        self._emit_safely = emit_safely
        self.canvas = canvas
        self.enabled = False
        self.mode = "classical"
        self.size = _MAGNIFIER_SIZE
        self.zoom = _MAGNIFIER_ZOOM
        self.sensitivity = _MAGNIFIER_SENSITIVITY
        self.scope = "region"
        self.exclude_border = True
        #: ``request -> labels`` or ``(labels, mode_used, note)``, run on the
        #: worker thread. Replaceable, which is how a test puts a stub in.
        self.segment = partial(_segment_region, load_model=load_model)
        self._context = context
        self._field = 0
        self._cursor: Optional[tuple] = None
        self._anchor: Optional[QPointF] = None
        self._requested_key: Optional[tuple] = None
        self._shown: Optional[_MagnifierResult] = None
        self._shown_image: Optional[QImage] = None
        self._waiting: set = set()
        self._unavailable: dict = {}
        #: ``(model key, labels, mode, note)`` for the last region the model
        #: answered. Read and written only on the region worker's thread, so
        #: a setting that only filters the answer does not ask again.
        self._raw: Optional[tuple] = None
        #: The key of the whole-image run on its way, or None.
        self._image_key: Optional[tuple] = None
        #: The objects of the last whole-image run, whole-field labels.
        self._image_result: Optional[_MagnifierResult] = None
        #: The key of a run cancelled or failed, not started again by itself.
        self._image_halted: Optional[tuple] = None
        #: ``(result, (box, object under the mouse), picture)`` last drawn.
        self._image_view: Optional[tuple] = None
        self._busy = False
        self._worker = _NewestRequestWorker(self._run, self._hand_over)
        self._image_worker = _NewestRequestWorker(
            self._run, self._hand_over, name="spacr-magnifier-image")
        self._delivered.connect(self._on_delivered, Qt.QueuedConnection)
        self._init_stroke()

    # -- settings ---------------------------------------------------------

    def set_enabled(self, on: bool) -> None:
        """Turn the box on or off; objects already committed are untouched.

        Turning it on in whole-image scope starts a run for the field on
        screen, unless its objects are already found. Turning it off leaves
        a run in progress going, so its objects are ready when it comes back.
        """
        self.enabled = bool(on)
        if self.enabled:
            self._image_halted = None
            self.refresh()
        else:
            self._cursor = None
            self._anchor = None
        self.canvas.update()

    def set_mode(self, mode: str) -> None:
        """Choose the model, and forget which models failed to run."""
        self.mode = str(mode or "classical")
        self._unavailable.clear()
        self.refresh()
        self.canvas.update()

    def set_scope(self, scope: str) -> None:
        """Choose between segmenting the box's region and the whole image.

        Leaving whole-image scope drops a run still waiting and ignores one
        still running. Objects already found are kept, and are offered again
        on coming back if no setting the model reads has changed since.
        """
        scope = str(scope or "region")
        if scope not in _MAGNIFIER_SCOPES:
            scope = "region"
        if scope == self.scope:
            return
        self.scope = scope
        self._image_halted = None
        self._image_view = None
        if scope == "region":
            self._stop_image()
        self.refresh()
        self.canvas.update()

    def set_exclude_border(self, on: bool) -> None:
        """Choose whether objects the box's own edge cuts are offered.

        Only the answer is filtered again: the model's answer for the region
        is kept on the worker, so the model is not asked a second time.
        """
        self.exclude_border = bool(on)
        self.refresh()
        self.canvas.update()

    def set_size(self, size: int) -> None:
        """Set the region's side in image pixels, within its range."""
        low, high = _MAGNIFIER_SIZE_RANGE
        self.size = max(low, min(high, int(size)))
        self.refresh()
        self.canvas.update()

    def set_zoom(self, zoom: float) -> None:
        """Set the magnification, within its range. The model is not asked."""
        low, high = _MAGNIFIER_ZOOM_RANGE
        self.zoom = max(low, min(high, float(zoom)))
        self.canvas.update()

    def set_sensitivity(self, value: float) -> None:
        """Set how readily an object is accepted, within its range."""
        low, high = _MAGNIFIER_SENSITIVITY_RANGE
        self.sensitivity = max(low, min(high, float(value)))
        self.refresh()
        self.canvas.update()

    def forget(self) -> None:
        """Drop everything tied to the field on screen; another is loading."""
        self._field += 1
        self._shown = None
        self._shown_image = None
        self._waiting.clear()
        self._requested_key = None
        self._stop_image()
        self._image_result = None
        self._image_view = None
        self._image_halted = None

    def close(self) -> bool:
        """Stop both workers, waiting briefly for a model call in flight."""
        region = self._worker.close()
        image = self._image_worker.close()
        return region and image

    # -- the mouse ----------------------------------------------------------

    def updating(self) -> bool:
        """Whether the objects on screen are not yet for the region asked for.

        In whole-image scope: whether the objects for the field and the
        settings now are not found yet.
        """
        if self.scope == "image":
            result = self._image_result
            return (result is None
                    or result.request.key != self._image_key_now())
        return (self._shown is None
                or self._shown.request.key != self._requested_key)

    def hover(self, pos) -> None:
        """Follow the mouse to widget point ``pos``; None puts the box away."""
        point = (None if pos is None
                 else self.canvas._canvas_to_image(pos.x(), pos.y()))
        if point is None:
            self._cursor = None
            self._anchor = None
            return
        self._anchor = QPointF(float(pos.x()), float(pos.y()))
        if point != self._cursor:
            self._cursor = point
            self.refresh()

    def refresh(self) -> None:
        """Ask for the region under the mouse, unless it is already shown.

        In whole-image scope, ask for the whole field instead -- and only
        when its objects for the settings now are neither found, on their
        way, nor cancelled.
        """
        if self.scope == "image":
            self._refresh_image()
            return
        request = self.build_request()
        if request is None:
            return
        self._requested_key = request.key
        if self._shown is not None and self._shown.request.key == request.key:
            return
        self._worker.submit(request)

    def _model_settings(self) -> tuple:
        """``(mode, sensitivity, bright, min_area, model_name, diameter)``.

        Everything a model reads, as it would read it now: the screen's own
        settings through ``context``, and classical in place of a mode that
        has already failed to load.
        """
        context = {"model_name": "cpsam", "diameter": 0, "bright": True,
                   "min_area": 0}
        if self._context is not None:
            context.update(self._context())
        model_name = str(context["model_name"])
        mode = self.mode
        if (mode, model_name) in self._unavailable:
            mode = "classical"
        return (mode, round(float(self.sensitivity), 4),
                bool(context["bright"]), int(context["min_area"]),
                model_name, int(context["diameter"]))

    @staticmethod
    def _accent() -> tuple:
        """The theme's accent colour as ``(red, green, blue)``."""
        accent = QColor(active_palette()["accent"])
        return (accent.red(), accent.green(), accent.blue())

    def build_request(self) -> Optional[_MagnifierRequest]:
        """The request for the region under the mouse now, or None."""
        canvas = self.canvas
        image = canvas.image
        if (not self.enabled or self._cursor is None or image is None
                or canvas.mask is None):
            return None
        settings = self._model_settings()
        mode, sensitivity, bright, min_area, model_name, diameter = settings
        box = engine._magnifier_box(image.shape, self._cursor[0],
                                    self._cursor[1], self.size)
        x0, y0, x1, y1 = box
        exclude = bool(self.exclude_border)
        return _MagnifierRequest(
            key=(self._field, box) + settings + (exclude,),
            crop=np.array(image[y0:y1, x0:x1], copy=True),
            box=box,
            shape=tuple(int(v) for v in image.shape[:2]),
            mode=mode,
            sensitivity=sensitivity,
            bright=bright,
            min_area=min_area,
            model_name=model_name,
            diameter=diameter,
            colour=self._accent(),
            exclude_border=exclude,
        )

    def click(self) -> bool:
        """Commit the objects for the region under the mouse.

        In whole-image scope, commit only the object under the mouse; see
        :meth:`_pick`.

        :returns: False when there is no region to commit -- the magnifier is
            off, the mouse is off the image, or no field is open.
        """
        if self.scope == "image":
            return self._pick()
        request = self.build_request()
        if request is None:
            return False
        self._requested_key = request.key
        if self._shown is not None and self._shown.request.key == request.key:
            self.commit_ready.emit(self._shown)
            return True
        self._waiting.add(request.key)
        self._worker.submit(request, pin=True)
        self.status.emit(
            "Magnifier: segmenting this region — its objects are added as "
            "soon as the box is up to date.")
        return True

    def remove(self) -> bool:
        """Ask the screen to remove the mask object under the mouse.

        Whole-image scope's right click. Any object in the mask qualifies,
        whatever put it there; the screen owns the mask, the undo history and
        the ledger, and says so when there is nothing under the mouse.

        :returns: False when the magnifier is off or the mouse is off the
            image.
        """
        if (not self.enabled or self._cursor is None
                or self.canvas.mask is None):
            return False
        self.remove_requested.emit(int(self._cursor[0]),
                                   int(self._cursor[1]))
        return True

    def cancel_image(self) -> bool:
        """Stop waiting for the whole-image run and throw its answer away.

        A model call already running cannot be interrupted: it finishes on
        its worker and its objects are discarded when they arrive. Moving the
        mouse does not start the run again; a click, or a change to a setting
        the model reads, does.

        :returns: False when no run was on its way.
        """
        from ..i18n import tr

        key = self._image_key
        if key is None:
            return False
        self._stop_image()
        self._image_halted = key
        self.status.emit(tr(
            "Magnifier: whole-image segmentation cancelled. Click the image "
            "to start it again."))
        self.canvas.update()
        return True

    def _image_key_now(self) -> Optional[tuple]:
        """The key a whole-image run for the field and settings now has."""
        canvas = self.canvas
        if canvas.image is None or canvas.mask is None:
            return None
        return (self._field, "image") + self._model_settings()

    def _refresh_image(self) -> None:
        """Start a whole-image run, unless one for the settings now is found,
        on its way, or was cancelled."""
        from ..i18n import tr

        if not self.enabled:
            return
        key = self._image_key_now()
        if key is None:
            return
        result = self._image_result
        if result is not None and result.request.key == key:
            return
        if key in (self._image_key, self._image_halted):
            return
        replaced = result is not None or self._image_key is not None
        self._image_result = None
        self._image_view = None
        self._start_image(key)
        if replaced:
            self.status.emit(tr(
                "Magnifier: the settings changed, so the whole-image objects "
                "were discarded. Segmenting the whole image again…"))
        else:
            self.status.emit(tr("Magnifier: segmenting the whole image…"))
        self.canvas.update()

    def _start_image(self, key: tuple) -> None:
        """Hand a copy of the whole field to the image worker under ``key``."""
        image = self.canvas.image
        mode, sensitivity, bright, min_area, model_name, diameter = key[2:]
        height, width = (int(v) for v in image.shape[:2])
        self._image_key = key
        self._image_halted = None
        self._image_worker.submit(_MagnifierRequest(
            key=key, crop=np.array(image, copy=True),
            box=(0, 0, width, height), shape=(height, width),
            mode=mode, sensitivity=sensitivity, bright=bright,
            min_area=min_area, model_name=model_name, diameter=diameter,
            colour=self._accent(), exclude_border=False, scope="image"))
        self._set_busy(True)

    def _stop_image(self) -> None:
        """Forget the whole-image run on its way, dropping it if not started."""
        self._image_key = None
        self._image_worker.drop_waiting()
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """Say that a whole-image run started or ended, once each."""
        busy = bool(busy)
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def _pick(self) -> bool:
        """Commit the whole-image object under the mouse, if there is one.

        Before the objects for the settings now are found nothing is
        committed and nothing is pinned: a run can take minutes, and an
        object going in long after the click that asked for it would be a
        surprise. A click while no run is on its way starts one.
        """
        from ..i18n import tr

        canvas = self.canvas
        if (not self.enabled or self._cursor is None or canvas.image is None
                or canvas.mask is None):
            return False
        key = self._image_key_now()
        result = self._image_result
        if result is None or result.request.key != key:
            if key is not None and key == self._image_key:
                self.status.emit(tr(
                    "Magnifier: the whole image is still being segmented — "
                    "nothing was added."))
            else:
                self._image_halted = None
                self._refresh_image()
            return True
        x, y = self._cursor
        label = int(result.labels[y, x])
        if label <= 0:
            self.status.emit(tr(
                "Magnifier: there is no object under the click — nothing was "
                "added."))
            return True
        self.commit_ready.emit(_single_object(result, label))
        return True

    def wheel(self, up: bool) -> float:
        """Step the zoom one wheel notch, at the canvas's own zoom per notch."""
        speed = max(1.001, float(getattr(self.canvas, "zoom_speed", 1.15)))
        self.set_zoom(self.zoom * speed if up else self.zoom / speed)
        self.zoom_changed.emit(self.zoom)
        return self.zoom

    # -- press and drag (item 417, parts 5 and 6) ---------------------------

    def _init_stroke(self) -> None:
        """Hold no press, and add every object in the box as item 407 did."""
        from PySide6.QtCore import QTimer

        from .._magnifier_drag import _PREVIEW_MS

        #: Which objects a click or a drag adds: ``zoom``, every object in the
        #: box, or ``touching``, only the objects under the mouse.
        self.save_mode = "zoom"
        #: The press in progress -- a ``_DragStroke`` -- with where and on
        #: which field it started, and whether it has moved far enough to be
        #: a drag.
        self._stroke = None
        self._stroke_from: Optional[tuple] = None
        self._stroke_moved = False
        self._stroke_timer = QTimer(self)
        self._stroke_timer.setSingleShot(True)
        self._stroke_timer.setInterval(_PREVIEW_MS)
        self._stroke_timer.timeout.connect(self._stroke_show)
        self._delivered.connect(self._stroke_delivered, Qt.QueuedConnection)

    def press(self) -> bool:
        """Start a stroke under the mouse: what a left press does while on.

        Nothing is added on the press. A release that has not moved is a
        click -- :meth:`click`, as item 407 built it -- except that with only
        objects touching the mouse it adds just the object under the cursor.
        A press that pulls is a drag; what a drag adds is
        :mod:`spacr.qt._magnifier_drag`'s to say. Under Whole image a drag
        reads the objects already found and asks no model; before they are
        found a press starts nothing, and its release is a click that says so.

        :returns: False when no stroke started.
        """
        from .._magnifier_drag import _DragStroke, _frame_step

        self._stroke = None
        canvas = self.canvas
        if (not self.enabled or self._cursor is None or canvas.image is None
                or canvas.mask is None):
            return False
        whole = self.scope == "image"
        found = self._image_result
        if whole and (found is None
                      or found.request.key != self._image_key_now()):
            return False
        stroke = _DragStroke(
            canvas.image.shape, self._cursor,
            step=0 if whole else _frame_step(self.size),
            keep_untouched=not whole and self.save_mode != "touching")
        self._stroke = stroke
        self._stroke_from = (QPointF(self._anchor), self._field)
        self._stroke_moved = False
        if whole:
            stroke.expect("image")
            stroke.deliver("image", found.labels, found.request.box)
        else:
            self._stroke_frame(self._cursor)
        return True

    def drag(self) -> None:
        """Follow the pressed mouse: extend the stroke and ask for its frames.

        Called on every move with the button down, and cheap. The press must
        first travel the platform's drag distance, so a hand that shakes
        during a click still clicks; after that a move adds a line of pixels
        and, every quarter box, one pinned request on the region worker. The
        model never runs here, and the mask is repainted at most every
        ``_PREVIEW_MS``.
        """
        stroke = self._stroke
        if stroke is None or self._cursor is None:
            return
        if not self._stroke_moved:
            travel = (self._anchor - self._stroke_from[0]).manhattanLength()
            if travel < QApplication.startDragDistance():
                return
            self._stroke_moved = True
        for centre in stroke.extend(self._cursor):
            self._stroke_frame(centre)
        self._stroke_dirty()

    def release(self) -> bool:
        """End a press: a click if it never moved, otherwise commit the stroke.

        The commit waits for any frame still on the worker, then reaches the
        screen once through :attr:`drag_ready` -- one edit, one undo step.
        """
        from ..i18n import tr

        stroke = self._stroke
        touching_click = bool(stroke is not None and stroke.step
                              and not stroke.keep_untouched)
        if stroke is None or not (self._stroke_moved or touching_click):
            self._stroke = None
            return self.click()
        stroke.release()
        if stroke.waiting():
            self.status.emit(tr(
                "Magnifier: segmenting the last regions — the objects are "
                "added as soon as they are done."))
        self._stroke_finish()
        return True

    def _stroke_frame(self, centre) -> None:
        """Ask for the box centred on ``centre`` as one of the stroke's frames.

        The box on screen, when it is that box, is taken at once; any other is
        pinned on the region worker, so later moves cannot supersede it.
        """
        cursor, self._cursor = self._cursor, centre
        request = self.build_request()
        self._cursor = cursor
        self._stroke.expect(request.key)
        shown = self._shown
        if shown is not None and shown.request.key == request.key:
            self._stroke.deliver(request.key, shown.labels, request.box)
        else:
            self._worker.submit(request, pin=True)

    def _stroke_delivered(self, payload) -> None:
        """Give the stroke a box it waits for; commit it if that was the last."""
        request, result, error = payload
        stroke = self._stroke
        if stroke is None:
            return
        if error is not None:
            stroke.drop(request.key)
        elif stroke.deliver(request.key, result.labels, request.box):
            self._stroke_dirty()
        self._stroke_finish()

    def _stroke_dirty(self) -> None:
        """Show what the stroke adds soon, unless a showing is already due."""
        if self._stroke.dirty and not self._stroke_timer.isActive():
            self._stroke_timer.start()

    def _stroke_finish(self) -> None:
        """Commit the stroke once the button is up and its last frame is in."""
        if self._stroke is not None and self._stroke.ready():
            self._stroke_show(final=True)

    def _stroke_show(self, final: bool = False) -> None:
        """Hand what the stroke adds to the screen: to show, or to commit.

        A stroke whose field has gone -- the user moved on while it waited --
        is dropped without a word: its objects were for a mask no longer on
        screen.
        """
        stroke = self._stroke
        if stroke is None:
            return
        gone = self._stroke_from[1] != self._field
        if final or gone:
            self._stroke = None
            self._stroke_timer.stop()
        if not gone:
            self.drag_ready.emit((stroke.outcome(), bool(final)))

    # -- the worker ---------------------------------------------------------

    def _run(self, request: _MagnifierRequest) -> _MagnifierResult:
        """Segment one request. Runs on a worker thread, never the GUI's.

        A region asked for again with only the border option changed reuses
        the model's last answer, which is kept here on the region worker.
        """
        if request.scope == "image":
            labels, used, note = self._segment_now(request)
            return _MagnifierResult(request, labels, used, note, None,
                                    _object_count(labels))
        model_key = request.key[:-1]
        cached = self._raw
        if cached is not None and cached[0] == model_key:
            _key, labels, used, note = cached
        else:
            labels, used, note = self._segment_now(request)
            self._raw = (model_key, labels, used, note)
        if request.exclude_border:
            labels = engine._drop_cut_objects(labels, request.box,
                                              request.shape)
        return _MagnifierResult(request, labels, used, note,
                                _candidate_overlay(labels, request.colour),
                                _object_count(labels))

    def _segment_now(self, request: _MagnifierRequest) -> tuple:
        """Ask the model: ``(int32 labels shaped like the crop, mode, note)``."""
        outcome = self.segment(request)
        if isinstance(outcome, tuple):
            labels, used, note = outcome
        else:
            labels, used, note = outcome, request.mode, ""
        labels = np.asarray(labels)
        if labels.shape != request.crop.shape[:2]:
            raise ValueError(
                f"{used} returned labels shaped {labels.shape} for a region "
                f"shaped {request.crop.shape[:2]}")
        return labels.astype(np.int32, copy=False), str(used), str(note)

    def _hand_over(self, request, result, error) -> None:
        """Pass a finished request to the GUI thread. Runs on the worker."""
        self._emit_safely(self._delivered, (request, result, error))

    def _on_delivered(self, payload) -> None:
        """Show a finished result, and commit it if a click was waiting."""
        request, result, error = payload
        if request.key[0] != self._field:
            return
        if request.scope == "image":
            self._on_image_delivered(request, result, error)
            return
        if error is not None:
            self._waiting.discard(request.key)
            LOG.warning("magnifier could not segment %s: %s",
                        request.box, error)
            self.status.emit(
                f"Magnifier could not segment this region: {error}")
            return
        self._note_fallback(request, result)
        self._shown = result
        self._shown_image = _rgba_qimage(result.overlay)
        if request.key in self._waiting:
            self._waiting.discard(request.key)
            self.commit_ready.emit(result)
        self.canvas.update()

    def _note_fallback(self, request, result) -> bool:
        """Say, once per model, that it could not run and classical stood in.

        :returns: True when this call said it.
        """
        if result.mode == request.mode:
            return False
        marker = (request.mode, request.model_name)
        if marker in self._unavailable:
            return False
        self._unavailable[marker] = result.note
        self.status.emit(
            f"Magnifier: {request.mode} could not run "
            f"({result.note}); the classical mode is segmenting "
            f"instead.")
        return True

    def _on_image_delivered(self, request, result, error) -> None:
        """Keep a finished whole-image run, unless it is no longer wanted."""
        from ..i18n import tr

        if request.key != self._image_key:
            return
        self._image_key = None
        self._set_busy(False)
        if error is not None:
            self._image_halted = request.key
            LOG.warning("magnifier could not segment the whole image: %s",
                        error)
            self.status.emit(tr(
                "Magnifier could not segment the whole image: {error}",
                error=error))
            return
        noted = self._note_fallback(request, result)
        stored = request._replace(crop=None)
        key = self._image_key_now()
        if (key is not None and key[2] == result.mode
                and key[:2] + key[3:] == request.key[:2] + request.key[3:]):
            # A model that could not load was just marked so, and the key the
            # settings now give names the classical mode that stood in.
            stored = stored._replace(key=key)
        self._image_result = result._replace(request=stored)
        self._image_view = None
        if not noted:
            self.status.emit(tr(
                "Magnifier: {n} object(s) found in the whole image. Click one "
                "to add it; right-click an object in the mask to remove it.",
                n=result.count))
        self.canvas.update()

    # -- drawing ------------------------------------------------------------

    def lens_geometry(self) -> Optional[tuple]:
        """``(box, lens, scale)`` for the box under the mouse, or None.

        ``box`` is the crop in image pixels, ``lens`` the widget rectangle it
        is drawn into and ``scale`` widget pixels per image pixel inside the
        lens: the canvas's own display scale times ``zoom``. The lens is
        placed so the centre of the image pixel under the cursor is exactly
        under the cursor, which at the image border leaves the clipped side
        visibly short rather than shifting the picture.
        """
        canvas = self.canvas
        if (not self.enabled or self._cursor is None or self._anchor is None
                or canvas.image is None or canvas.mask is None):
            return None
        rendered = canvas.pixmap()
        if rendered is None or rendered.isNull():
            return None
        shown = logical_size(rendered)
        vx0, _vy0, vx1, _vy1 = canvas._viewport_bounds()
        scale = shown.width() / max(1, vx1 - vx0) * float(self.zoom)
        cx, cy = self._cursor
        box = engine._magnifier_box(canvas.image.shape, cx, cy, self.size)
        x0, y0, x1, y1 = box
        lens = QRectF(self._anchor.x() - (cx + 0.5 - x0) * scale,
                      self._anchor.y() - (cy + 0.5 - y0) * scale,
                      (x1 - x0) * scale, (y1 - y0) * scale)
        return box, lens, scale

    def paint(self, painter: QPainter) -> None:
        """Draw the box: the region magnified, its objects, and its state.

        The region is contrast-stretched on its own, which is what makes the
        box an enhanced view rather than a bigger copy of the canvas. The
        last completed result is drawn where ITS region lies, so a result
        that is behind the mouse is offset rather than wrong, and the frame
        turns dashed with an "Updating…" mark until the result for this
        region arrives -- the box never blanks while it waits.

        In whole-image scope the box draws its slice of the whole-image
        objects instead, with the object a click would add filled more
        strongly, and the frame is dashed while a run is on its way.
        """
        geometry = self.lens_geometry()
        if geometry is None:
            return
        box, lens, scale = geometry
        x0, y0, x1, y1 = box
        canvas = self.canvas
        stretched = engine.normalize_uint16(
            np.ascontiguousarray(canvas.image[y0:y1, x0:x1]),
            canvas.norm_lo, canvas.norm_hi)
        rgb = np.ascontiguousarray(engine.overlay_mask(
            stretched, canvas.mask[y0:y1, x0:x1], alpha=0.5))
        height, width = rgb.shape[:2]
        picture = QImage(rgb.data, width, height, 3 * width,
                         QImage.Format_RGB888)
        palette = active_palette()
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.setClipRect(lens)
        painter.drawImage(lens, picture)
        if self.scope == "image":
            view = self._image_slice(box)
            if view is not None:
                painter.drawImage(lens, view)
            updating = self._image_key is not None
        else:
            shown = self._shown
            if shown is not None and self._shown_image is not None:
                sx0, sy0, sx1, sy1 = shown.request.box
                painter.drawImage(
                    QRectF(lens.left() + (sx0 - x0) * scale,
                           lens.top() + (sy0 - y0) * scale,
                           (sx1 - sx0) * scale, (sy1 - sy0) * scale),
                    self._shown_image)
            updating = self.updating()
        painter.setClipping(False)
        pen = QPen(QColor(palette["accent"]))
        pen.setWidth(2)
        if updating:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(lens)
        if updating:
            from ..i18n import tr

            caption = tr("Updating…")
            metrics = painter.fontMetrics()
            badge = QRectF(lens.left() + 4, lens.top() + 4,
                           metrics.horizontalAdvance(caption) + 10,
                           metrics.height() + 4)
            backing = QColor(palette["bg"])
            backing.setAlpha(200)
            painter.fillRect(badge, backing)
            painter.setPen(QPen(QColor(palette["fg"])))
            painter.drawText(badge, Qt.AlignCenter, caption)
        painter.restore()

    def _image_slice(self, box) -> Optional[QImage]:
        """The whole-image objects inside ``box``, outlined, as a picture.

        The object under the mouse is filled more strongly than the rest. The
        picture is kept until the box, that object or the objects themselves
        change, so repainting without moving recomputes nothing -- and
        nothing here calls a model.
        """
        result = self._image_result
        if result is None or self._cursor is None:
            return None
        x0, y0, x1, y1 = (int(v) for v in box)
        cx, cy = self._cursor
        under = int(result.labels[cy, cx])
        where = ((x0, y0, x1, y1), under)
        cached = self._image_view
        if cached is not None and cached[0] is result and cached[1] == where:
            return cached[2]
        crop = result.labels[y0:y1, x0:x1]
        rgba = _candidate_overlay(crop, result.request.colour)
        if under > 0:
            focus = (crop == under) & (rgba[..., 3] < 140)
            rgba[focus, 3] = 140
        picture = _rgba_qimage(rgba)
        self._image_view = (result, where, picture)
        return picture


class _FlowPane(QLabel):
    """Read-only pane for one Cellpose intermediate, scaled to fit its tab.

    Ported from the standalone curation tool's ``FlowView``, which solved
    the same problem: an intermediate is a picture to *look* at while
    deciding where a threshold goes, and it has to stay legible when the
    tab is resized. It keeps the full-resolution pixmap and rescales a
    copy, so repeated resizing never compounds interpolation error the
    way rescaling the displayed pixmap would.

    It is deliberately not editable. The mask lives on the canvas next
    door, and a second surface that could also be painted would mean two
    places to look for the same object.

    :param parent: parent widget; ownership only.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Build an empty flow pane, centred and with a floor on its size."""
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet(f"background: {active_palette()['bg']};")
        self.setWordWrap(True)
        self._pixmap: Optional[QPixmap] = None
        follow_device_ratio(self, self._rescale)
        self.clear_view()

    def show_rgb(self, rgb: np.ndarray) -> None:
        """Display one ``(H, W, 3)`` uint8 array."""
        data = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        height, width = data.shape[:2]
        image = QImage(data.data, width, height, 3 * width,
                       QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.setText("")
        self._rescale()

    def clear_view(self) -> None:
        """Drop the picture and say why the pane is empty."""
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(FLOW_RESTING_TEXT)

    def has_image(self) -> bool:
        """Whether a Cellpose run has filled this pane."""
        return self._pixmap is not None

    def _rescale(self) -> None:
        """Redraw the pixmap at the pane's current size.

        A no-op before a pixmap is set, so a resize during construction is
        harmless.
        """
        if self._pixmap is None:
            return
        self.setPixmap(scaled_for(self._pixmap, self, self.size()))

    def resizeEvent(self, event):
        """Refit the picture whenever the tab changes size."""
        super().resizeEvent(event)
        self._rescale()


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


class FoldedModulePanel(QWidget):
    """One folded module, as the whole screen it was, plus what the host adds.

    A fold that reimplemented the module it replaced would keep whatever the
    person doing the folding happened to think of and quietly drop the rest.
    So the button opens the module's OWN widget: every control, every worker,
    every drop target it had as a tile is what arrives, and the only thing
    that changed is where it is opened from.

    IT IS A PAGE ON THIS SCREEN, not a window over it. A window is the last
    resort for a fold, and it is what this becomes only when the host has no
    body to make pages out of — see
    :func:`spacr.qt.screens.map_barcodes.show_as_page`. As a page it is
    closed by the tab's own close button, so the standard Close button
    below belongs to the window shape alone and is added with it.

    :param key: the folded module's registry key.
    :param screen: the module's own widget, already built.
    :param title: the page's caption and the window title — the module's
        name.
    :param actions: extra buttons for the button row, each
        ``(label, tooltip, callback)``. This is where a capability the folded
        module lacks and its host has arrives.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, key: str, screen: QWidget, title: str,
                 parent: Optional[QWidget] = None, actions=()):
        """Wrap one folded module's screen with a title and its actions.

        :param key: the module's registry key.
        :param screen: the screen to wrap.
        :param title: the caption over it.
        :param parent: parent widget.
        :param actions: extra buttons for the panel's own row.
        """
        super().__init__(parent)
        self.app_key = key
        self.screen = screen
        self.setObjectName("FoldedModulePanel")
        self.setWindowTitle(title)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, SPACING["sm"])
        column.setSpacing(SPACING["sm"])
        column.addWidget(screen, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.NoButton, self)
        #: Label -> button, for the extra actions.
        self.actions: dict = {}
        for label, tooltip, callback in actions:
            button = self.buttons.addButton(label,
                                            QDialogButtonBox.ActionRole)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, cb=callback: cb())
            self.actions[label] = button
        self.buttons.rejected.connect(self.close)
        self.buttons.setVisible(bool(self.actions))
        column.addWidget(self.buttons)

    def add_close_button(self) -> None:
        """Give this panel the Close button a window needs.

        A page is closed by its tab. A window has no tab, so the row that
        carries the host's extra actions carries a Close beside them —
        added when the panel becomes a window rather than always, so a
        page never shows a button that would hide it inside its own tab.
        """
        if "Close" in self.actions:
            return
        button = self.buttons.addButton(QDialogButtonBox.Close)
        self.actions["Close"] = button
        self.buttons.setVisible(True)
        self.resize(1120, 780)



#: The file dialogs the bridge opens. A mask is a label image and an image is
#: whatever the microscope wrote, so the two filters are not the same one.
_MASK_FILTER = "Masks (*.tif *.tiff *.npy *.png);;All files (*)"
_IMAGE_FILTER = "Images (*.tif *.tiff *.npy *.png *.jpg);;All files (*)"


class NapariBridgeScreen(QWidget):
    """Exchange an image and label mask with an interactive napari viewer.

    napari is imported only when the viewer is opened, and spaCR's existing
    Qt event loop remains active. Corrected labels are validated and recorded
    in the same curation ledger used by the Curate screen. The corresponding
    headless operations are available from :mod:`spacr.napari_bridge`.

    :param parent: parent widget.
    """

    #: A field was opened in napari. Carries the mask path.
    opened = Signal(str)
    #: A correction came back and was written. Carries the mask path.
    corrected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the bridge's two path rows and its launch button.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self.setObjectName("NapariBridge")
        self._viewer: Any = None
        self._handoff: Any = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        title = QLabel("Napari Bridge", self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(
            "Open an image and label mask in napari for manual correction. "
            "When you import the corrected labels, spaCR validates the mask "
            "and records the change in the curation ledger.", self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._mask_edit = QLineEdit(self)
        self._mask_edit.setPlaceholderText("Label mask (.tif, .npy)")
        outer.addLayout(self._path_row("Mask", self._mask_edit,
                                       self._choose_mask))
        self._image_edit = QLineEdit(self)
        self._image_edit.setPlaceholderText(
            "Image to show underneath (optional)")
        outer.addLayout(self._path_row("Image", self._image_edit,
                                       self._choose_image))

        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING["sm"])
        self.open_button = QPushButton("Open in napari", self)
        self.open_button.setToolTip(
            "Open the field in a napari window. spaCR stays running.")
        self.open_button.clicked.connect(self.open_in_napari)
        buttons.addWidget(self.open_button)
        self.take_button = QPushButton("Take the mask back", self)
        self.take_button.setToolTip(
            "Read the corrected labels out of napari, write them back and "
            "record the correction")
        self.take_button.setEnabled(False)
        self.take_button.clicked.connect(self.take_mask_back)
        buttons.addWidget(self.take_button)
        self.close_button = QPushButton("Close viewer", self)
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close_viewer)
        buttons.addWidget(self.close_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.status = QPlainTextEdit(self)
        self.status.setObjectName("NapariBridgeStatus")
        self.status.setReadOnly(True)
        self.status.setPlaceholderText(
            "Choose a mask and press Open in napari.")
        mark_surface(self.status)
        outer.addWidget(self.status, 1)
        from ..dnd import install_for
        install_for(self, "napari_bridge")

    def _path_row(self, label: str, edit: QLineEdit, chooser) -> QHBoxLayout:
        """One labelled path field with a browse button beside it.

        :param label: the caption.
        :param edit: the field itself.
        :param chooser: what the browse button runs.
        :returns: the assembled row.
        """
        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        caption = QLabel(label, self)
        caption.setMinimumWidth(56)
        row.addWidget(caption)
        row.addWidget(edit, 1)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(chooser)
        row.addWidget(browse)
        return row

    def _choose_mask(self) -> None:
        """Ask for a label mask and put it in the field."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a label mask", self._mask_edit.text().strip(),
            _MASK_FILTER)
        if path:
            self._mask_edit.setText(path)
            self.describe_mask(path)

    def _choose_image(self) -> None:
        """Ask for an image and put it in the field."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open the image underneath",
            self._image_edit.text().strip(), _IMAGE_FILTER)
        if path:
            self._image_edit.setText(path)

    def set_paths(self, mask: str = "", image: str = "") -> None:
        """Set the mask and optional source-image paths."""
        if mask:
            self._mask_edit.setText(str(mask))
        if image:
            self._image_edit.setText(str(image))

    def mask_path(self) -> str:
        """Return the mask path entered in the form."""
        return self._mask_edit.text().strip()

    def image_path(self) -> str:
        """Return the optional source-image path entered in the form."""
        return self._image_edit.text().strip()

    def say(self, text: str, *, append: bool = False) -> str:
        """Display a status message and return the complete displayed text."""
        text = str(text)
        if append and self.status.toPlainText():
            self.status.setPlainText(
                f"{self.status.toPlainText()}\n\n{text}")
        else:
            self.status.setPlainText(text)
        return self.status.toPlainText()

    def describe_mask(self, path: str = "") -> str:
        """Describe a mask file and its recorded curation state."""
        path = path or self.mask_path()
        if not path or not os.path.isfile(path):
            return self.say("Choose a mask file first.")
        try:
            from ...napari_bridge import load_handoff
            handoff = load_handoff(path, self.image_path())
        except Exception as exc:
            return self.say(f"Could not read {os.path.basename(path)}: {exc}")
        return self.say(handoff.describe())

    def open_in_napari(self) -> Any:
        """Open the selected image and mask in napari.

        :returns: The napari viewer, or ``None`` if validation or startup
            fails. Missing optional dependencies are reported in the status
            pane.
        """
        path = self.mask_path()
        if not path or not os.path.isfile(path):
            self.say("Choose a mask file first.")
            return None
        try:
            from ...napari_bridge import (NapariExtraMissing, load_handoff,
                                          open_in_napari)
        except ImportError as exc:            # pragma: no cover - broken tree
            self.say(f"Could not load the napari bridge: {exc}")
            return None
        try:
            handoff = load_handoff(path, self.image_path())
        except Exception as exc:
            self.say(f"Could not read {os.path.basename(path)}: {exc}")
            return None
        try:
            viewer = open_in_napari(handoff)
        except NapariExtraMissing as exc:
            self.say(str(exc))
            return None
        except Exception as exc:
            LOG.exception("could not open napari")
            self.say(f"napari could not open this field: {exc}")
            return None
        self._viewer = viewer
        self._handoff = handoff
        self.take_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.say(f"{handoff.describe()}\n\nThe field is open in napari. "
                 f"After correcting the labels, return to spaCR and select "
                 f"Take the mask back; nothing is written until you do.")
        self.opened.emit(handoff.mask_path)
        return viewer

    def take_mask_back(self):
        """Import corrected labels and record the mask correction.

        :returns: :class:`spacr.napari_bridge.CorrectionResult`, or ``None``
            if no active handoff exists or validation fails.
        """
        if self._viewer is None or self._handoff is None:
            self.say("Open a field in napari first.")
            return None
        from ...napari_bridge import labels_from_viewer, write_back

        try:
            corrected = labels_from_viewer(self._viewer,
                                           name=self._handoff.name)
        except Exception as exc:
            self.say(str(exc))
            return None
        try:
            result = write_back(self._handoff.mask_path, corrected,
                                original=self._handoff.mask)
        except Exception as exc:
            self.say(str(exc))
            return None
        self.say(result.describe(), append=False)
        if result.written:
            self._handoff = self._reloaded(result)
            self.corrected.emit(result.mask_path)
        return result

    def _reloaded(self, result) -> Any:
        """The handoff, with the mask that was just written."""
        import dataclasses

        return dataclasses.replace(self._handoff, mask=result.mask)

    def close_viewer(self) -> None:
        """Close the active napari viewer, if present."""
        viewer = self._viewer
        self._viewer = None
        self._handoff = None
        self.take_button.setEnabled(False)
        self.close_button.setEnabled(False)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                LOG.debug("napari viewer would not close", exc_info=True)

    def viewer(self) -> Any:
        """Return the active napari viewer, or ``None``."""
        return self._viewer

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self.close_viewer()
        super().closeEvent(event)



class MakeMasksScreen(QWidget):
    """Qt widget for the Make Masks app — the successor to Tk ModifyMaskApp.

    Owns the canvas, the tools panel, and the file-navigation state; see
    the module docstring for the full feature list.

    :param parent: parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the editor, its canvas and its tool panel.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self._folder: str = ""
        self._image_files: List[str] = []
        #: The terminal-built session this screen is working through, or
        #: ``None`` when the folder was opened from the file dialog. Set by
        #: :meth:`open_queue`; what makes a save reach
        #: ``curate_status.csv``.
        self._queue = None
        self._current_index: int = 0
        self._history = engine.MaskHistory(capacity=25)
        #: The ledger for the field on screen, seeded from any sidecar
        #: already beside its mask so a second editing session appends to
        #: the first one's record instead of replacing it.
        self._log: Optional[CurationLog] = None
        self._load_token = 0
        #: Fields cut out of the one on screen this visit, in the order
        #: they were cut. Non-empty means the field on screen is a parent:
        #: it is retired the moment the user leaves it — see
        #: :meth:`finish_recrop`.
        self._recrop_children: List[str] = []
        self._load_worker: Optional[_MaskLoadWorker] = None
        self._pending_load = None
        self._loading = False
        #: Folded module key -> the module's own screen, built the first time
        #: its button is pressed and kept afterwards so a second press finds
        #: the paths, models and results the first one left.
        self._fold_screens: dict[str, QWidget] = {}
        #: Folded module key -> the window that screen lives in.
        #: Folded module key -> the panel that screen lives in,
        #: which is a page on this screen wherever it can be one.
        self._fold_dialogs: dict[str, FoldedModulePanel] = {}
        #: What this screen's own page is called once a folded module
        #: puts a page beside it. Named here because this screen is not
        #: the generic settings form and carries no registry key to be
        #: looked up by.
        self._fold_page_title = HEADER_TITLE
        self._build_ui()
        self._install_shortcuts()
        self._sync_button_states()

        try:
            from ..dnd import install_dropzone
            from ..dnd_handlers import MakeMasksDropHandler
            install_dropzone(self, MakeMasksDropHandler(), self)
        except Exception:
            pass
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)
        self._take_any_terminal_queue()

    def _take_any_terminal_queue(self) -> bool:
        """Open the queue ``spacr-make-masks`` handed over, if there is one.

        The terminal half of ledger item 396 ends here. ``spacr-make-masks``
        reads the folder, builds the session and leaves it in
        :mod:`spacr.cli_make_masks`; the first screen built in that process
        takes it. The import is of a CLI module that pulls argparse and
        :mod:`spacr.curation_queue` and no Qt, and it is done here rather
        than at module scope so that a screen opened the ordinary way pays
        for nothing.

        :returns: whether a handed-over queue was opened.
        """
        try:
            from ...cli_make_masks import take_handover
        except Exception:                                    # noqa: BLE001
            LOG.debug("no terminal queue handover available", exc_info=True)
            return False
        queue = take_handover()
        if queue is None:
            return False
        return self.open_queue(queue)

    def open_queue(self, queue) -> bool:
        """Open the fields a curation session offers, in the order it offers.

        The session decides WHICH fields and in WHAT ORDER -- reviewed ones
        already dropped, ``--limit`` already applied -- and this screen shows
        them. Only the nested layout is opened: this editor reads a draft
        from ``<folder>/masks/<stem>.tif`` and saves back to the same place,
        so a sibling or ``_seg.npy`` set would be read and written somewhere
        other than where its masks are. ``spacr-make-masks`` refuses those
        before Qt is imported; this is the second half of the same refusal,
        for anything that reaches the screen another way.

        :param queue: a :class:`spacr.curation_queue.CurationQueue`.
        :returns: whether the editor is now on that session.
        """
        from ...curation_queue import LAYOUT_NESTED

        if getattr(queue.layout, "kind", None) != LAYOUT_NESTED:
            LOG.warning("Make Masks edits the nested layout; %s is %s",
                        queue.folder, getattr(queue.layout, "kind", "unknown"))
            return False
        files = [item.image.name for item in queue.items
                 if item.image is not None]
        if not files:
            LOG.info("%s has nothing left to curate", queue.folder)
            return False
        if not self._open_folder(str(queue.folder), files=files):
            return False
        self._queue = queue
        self._src_label.setText(
            f"{queue.folder}  --  {len(files)} to curate this session")
        self._status_label.setText(queue.describe())
        return True

    def _note_curated(self, filename: str,
                      n_objects: Optional[int] = None) -> None:
        """Record a saved field as done in the session's resume record.

        Only when the folder came from :meth:`open_queue`: a folder opened
        from the file dialog is not a queue and must not grow a status file
        it was never asked for. A record that cannot be written is logged
        and swallowed, because the mask itself is already safely on disk and
        losing the session's place is the smaller failure of the two.

        :param filename: the image file that was just saved.
        :param n_objects: how many objects the saved mask had, if known.
        """
        if self._queue is None:
            return
        from ...curation_queue import mark_state

        stem = os.path.splitext(filename)[0]
        # READ THE FOLDER OUTSIDE THE TRY. What is meant to be swallowed here
        # is a FAILURE TO WRITE THE RECORD -- a full disk, a read-only sync
        # folder -- because the mask is already safe and losing the session's
        # place is the smaller loss. `self._queue.folder` is not that: if the
        # guard above ever stops running, it raises AttributeError, and inside
        # the try that error is logged as "could not record" and the screen
        # carries on as though a queue it does not have had been updated.
        folder = self._queue.folder
        try:
            mark_state(folder, stem, "done", n_objects=n_objects)
        except Exception:                                    # noqa: BLE001
            LOG.warning("could not record %s as done in the curation queue",
                        stem, exc_info=True)

    def _build_ui(self):
        """Lay out the canvas, the tool panel and the navigation row."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        self._header = ModuleHeader(
            HEADER_TITLE,
            description=HEADER_DESCRIPTION,
            instruction=HEADER_INSTRUCTION,
            app_key=APP_KEY,
        )
        self._src_label = QLabel("No folder selected — click Open folder…")
        self._src_label.setObjectName("SubtitleSmall")
        self._src_label.setSizePolicy(QSizePolicy.Maximum,
                                      QSizePolicy.Preferred)
        self._src_label.setMinimumWidth(0)
        self._header.add_trailing(self._src_label)
        self._folds = self._build_fold_strip()
        self._header.add_trailing(self._folds)
        outer.addWidget(self._header)
        outer.addWidget(Divider())

        self._tool_row = self._build_tool_row()
        outer.addWidget(self._tool_row)

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
        self._canvas.recrop_requested.connect(self._on_recrop_requested)
        self._magnifier = _LiveMagnifier(
            self._canvas, self, load_model=self._cellpose_model,
            context=self._magnifier_context)
        self._canvas.magnifier = self._magnifier
        self._magnifier.commit_ready.connect(self._commit_magnifier_result)
        self._magnifier.remove_requested.connect(self._remove_magnifier_object)
        self._magnifier.drag_ready.connect(self._apply_magnifier_drag)
        #: The mask a magnifier drag pastes onto, and the last one it showed.
        self._drag_base = self._drag_shown = None
        self._magnifier.status.connect(
            lambda text: self._status_label.setText(text))
        self._body_splitter.addWidget(self._build_view_tabs())

        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QScrollArea.NoFrame)
        self._settings_scroll.setWidget(self._build_tools_panel())
        for changed in (self._cp_model.currentIndexChanged,
                        self._cp_diameter.valueChanged,
                        self._otsu_bright.toggled,
                        self._min_area.valueChanged):
            changed.connect(self._on_magnifier_context_changed)
        self._body_splitter.addWidget(self._settings_scroll)
        self._body_splitter.setStretchFactor(0, 3)
        self._body_splitter.setStretchFactor(1, 1)
        self._body_splitter.setSizes([900, SETTINGS_WIDTH])
        self._body_stack.addWidget(self._body_splitter)
        self._body_stack.setCurrentWidget(self._empty_state)
        self._body_stack.currentChanged.connect(self._sync_tool_row_visibility)
        self._sync_tool_row_visibility()

        outer.addWidget(self._body_stack, 1)

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
        from ..make_masks_demo import install_test_data_button
        nav_row.addWidget(install_test_data_button(self))

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
            screen.compare_requested.connect(self._on_zoo_compare_requested)
            return screen
        if key == "curate":
            from .curate import CurateScreen
            return CurateScreen()
        if key == "napari_bridge":
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

    def open_folded(self, key: str) -> Optional[FoldedModulePanel]:
        """Open a folded module on this screen, pointed at the open field.

        The module arrives as a PAGE beside the editor, which is where a
        fold belongs; it becomes a window only if this screen has no body
        to make pages out of.

        :param key: one of :data:`FOLD_ORDER`.
        :returns: the module's panel, or ``None`` for a key this screen does
            not fold. Pressing the same button again raises the page that is
            already there rather than building a second one.
        """
        from .map_barcodes import show_as_page, show_as_window

        host = FOLD_HOSTS.get(key, key)
        screen = self.folded_screen(host)
        if screen is None:
            return None
        title = fold_description(host)[0]
        panel = self._fold_dialogs.get(host)
        if panel is None:
            panel = FoldedModulePanel(
                host, screen, title, parent=self,
                actions=self._fold_actions(host))
            self._fold_dialogs[host] = panel
        self.seed_folded(key)
        if show_as_page(panel, self, title) is None:
            panel.add_close_button()
            show_as_window(panel, self, title)
        panel.show()
        panel.raise_()
        return panel

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

        The Curate page retains corrected labels in its curation session. This
        method calls :meth:`spacr.curation.MaskCuration.save_mask`, which
        writes the labels and, when edits exist, writes the corresponding
        curation ledger beside the output file.

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
        """Close every folded module, and everything it started.

        Closing the panel is not enough. A module polls the machine's RAM
        and GPU on a worker thread, and Qt answers a running QThread being
        destroyed by aborting the process — so the module's own close
        handler, which drains that worker, has to run. A page nested
        inside another module's tabs never gets one from Qt, which is why
        they are closed by hand here.

        EVERY MODULE THAT WAS BUILT, not every module that was opened. A
        module is built the moment something asks this screen to point it
        at the open folder — :meth:`seed_folded` does, and so does any
        test or caller reaching for :meth:`folded_screen` — and pointing
        Model Compare at a folder starts a load thread before its panel
        has ever been on screen. Walking the panels alone left those
        threads running with nothing holding them, and the process died
        of it several actions later, in whatever happened to be running
        when the memory behind them was touched.
        """
        from .app_screen import AppScreen

        for screen in list(self._fold_screens.values()):
            for page in screen.findChildren(AppScreen):
                page.close()
            screen.close()
        for panel in list(self._fold_dialogs.values()):
            panel.close()

    def _build_tool_row(self) -> QWidget:
        """The one row that holds every tool, along the top of the screen.

        THE WHOLE SET IS VISIBLE AT ONCE. The tools used to be a 2x3 grid
        inside a card in the side panel, where finding a tool meant
        reading a block; in one row they are read left to right and the
        one you want is where you last saw it.

        The row is built from :func:`tool_row_entries`, so a tool added to
        :data:`TOOL_MODES` — or a ``MODE_*`` constant added with no table
        entry at all — appears here without its author editing this
        method. Actions that are not modes come in through
        :meth:`add_toolbar_action` and land in the same row.

        The row ends with the settings toggle, which is checkable because
        it reports a state rather than firing an action: it stays lit for
        as long as the settings are on screen.
        """
        bar = QWidget()
        bar.setObjectName("MakeMasksToolRow")
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        #: The row itself, kept so a tool added after this screen was built
        #: has somewhere to go.
        self._tool_row_layout = row
        #: Width the settings pane goes back to when it is shown again.
        self._settings_width = SETTINGS_WIDTH

        self._mode_buttons: dict[str, QPushButton] = {}
        for mode, label, icon_key in tool_row_entries():
            btn = QPushButton(label)
            btn.setIcon(iconset.icon(icon_key))
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, key=mode: self._set_mode(key))
            row.addWidget(btn)
            self._mode_buttons[mode] = btn
        self._btn_brush = self._mode_buttons[MODE_BRUSH]
        self._btn_erase = self._mode_buttons[MODE_ERASE]
        self._btn_del_obj = self._mode_buttons[MODE_ERASE_OBJECT]
        self._btn_wand_add = self._mode_buttons[MODE_WAND_ADD]
        self._btn_wand_erase = self._mode_buttons[MODE_WAND_ERASE]
        self._btn_zoom = self._mode_buttons[MODE_ZOOM]
        self._btn_recrop = self._mode_buttons[MODE_RECROP]
        self._btn_recrop.setToolTip(RECROP_TOOLTIP)

        row.addWidget(Divider(Qt.Vertical))
        self._btn_reset_zoom = QPushButton("Reset zoom")
        self._btn_reset_zoom.setIcon(iconset.icon("zoom_reset"))
        self._btn_reset_zoom.setCursor(Qt.PointingHandCursor)
        self._btn_reset_zoom.setEnabled(False)
        self._btn_reset_zoom.clicked.connect(self._on_reset_zoom)
        row.addWidget(self._btn_reset_zoom)
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setIcon(iconset.icon("undo"))
        self._btn_undo.setCursor(Qt.PointingHandCursor)
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_undo)
        row.addWidget(self._btn_undo)
        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setIcon(iconset.icon("redo"))
        self._btn_redo.setCursor(Qt.PointingHandCursor)
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._on_redo)
        row.addWidget(self._btn_redo)

        row.addStretch(1)

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setIcon(iconset.icon("settings"))
        self._btn_settings.setCheckable(True)
        self._btn_settings.setMinimumHeight(32)
        self._btn_settings.setCursor(Qt.PointingHandCursor)
        self._btn_settings.setToolTip(
            "Show or hide the settings — brush, wand, display, auto-filter "
            "and object operations, as one group. The canvas takes the "
            "width they give up.")
        self._btn_settings.setChecked(True)
        self._btn_settings.toggled.connect(self._on_toggle_settings)
        row.addWidget(self._btn_settings)

        scroller = QScrollArea()
        scroller.setObjectName("MakeMasksToolScroll")
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.NoFrame)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroller.setWidget(bar)
        scroller.setFixedHeight(
            bar.sizeHint().height()
            + scroller.horizontalScrollBar().sizeHint().height())
        scroller.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return scroller

    def add_toolbar_action(self, button: QPushButton) -> QPushButton:
        """Insert a non-mode action into the editor toolbar.

        The button is placed with the other actions, immediately before the
        stretch that keeps the Settings toggle aligned to the far edge.

        :param button: Action button to insert.
        :returns: The same button.
        """
        row = self._tool_row_layout
        row.insertWidget(max(row.indexOf(self._btn_settings) - 1, 0), button)
        return button

    def _sync_tool_row_visibility(self, *_args) -> None:
        """Show the tool row only while the editor is the body.

        There is nothing to brush, undo or configure until a folder is
        open, and a row of dead buttons over the empty state reads as a
        broken screen rather than an empty one.
        """
        self._tool_row.setVisible(
            self._body_stack.currentWidget() is self._body_splitter)

    def settings_shown(self) -> bool:
        """Whether the settings group is on screen."""
        return self._btn_settings.isChecked()

    def _on_toggle_settings(self, shown: bool) -> None:
        """Hide or show the settings as one group.

        THE CANVAS KEEPS THE SPACE. Hiding a splitter child gives its
        width to the sibling, so the image grows into the panel's place
        rather than leaving a gap where the panel was. The width the
        panel had is remembered while it is away, so a second press puts
        it back where the user last dragged it instead of at the default.
        """
        splitter = self._body_splitter
        if not shown:
            sizes = splitter.sizes()
            if len(sizes) > 1 and sizes[1] > 0:
                self._settings_width = sizes[1]
        self._settings_scroll.setVisible(shown)
        if shown:
            sizes = splitter.sizes()
            total = sum(sizes) or (900 + SETTINGS_WIDTH)
            side = max(min(self._settings_width, total - 1), 1)
            splitter.setSizes([total - side, side])

    def _build_tools_panel(self) -> QWidget:
        """Build the tool column: mode, brush, wand and mask operations.

        :returns: the assembled panel.
        """
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])

        brush_card = Card(title="Brush")
        brush_form = QFormLayout()
        self._brush_slider = QSlider(Qt.Horizontal)
        self._brush_slider.setRange(1, 100)
        self._brush_slider.setValue(10)
        self._brush_slider.setToolTip(
            "Radius of the brush and the eraser, in pixels on screen. It is "
            "scaled into image pixels at the current zoom, so the disk under "
            "the cursor stays the same size and zooming in paints a finer "
            "stroke on the data."
        )
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
        self._wand_pct.setToolTip(
            "How far the flood may stray from the value under the click, as "
            "a percentage of this image's own intensity range. Raise it to "
            "take in more of a dim object; lower it when the flood spills "
            "into the background."
        )
        self._wand_pct.valueChanged.connect(self._on_wand_pct_changed)
        wand_form.addRow("Tolerance %", self._wand_pct)
        self._wand_tol = QDoubleSpinBox()
        self._wand_tol.setRange(0.0, 1_000_000.0)
        self._wand_tol.setSingleStep(50.0)
        self._wand_tol.setValue(1000.0)
        self._wand_tol.setEnabled(False)
        self._wand_tol.setToolTip(
            "The same distance in raw grey levels, read only while the "
            "percentage above is switched off. A setting tuned on 8-bit data "
            "selects almost nothing on a 16-bit image, which is why the "
            "percentage is the default."
        )
        self._wand_tol.valueChanged.connect(self._on_wand_tolerance_changed)
        wand_form.addRow("Tolerance (absolute)", self._wand_tol)
        self._wand_max = QSpinBox()
        self._wand_max.setRange(1, 10_000_000)
        self._wand_max.setSingleStep(1000)
        self._wand_max.setValue(100_000)
        self._wand_max.setToolTip(
            "The most pixels one wand click may flood, which is also what "
            "bounds how long a click can take. What happens when a flood "
            "reaches the cap is the switch below."
        )
        self._wand_max.valueChanged.connect(self._on_wand_max_changed)
        wand_form.addRow("Max pixels", self._wand_max)
        self._wand_salvage = QCheckBox("Keep the nearest pixels at the cap")
        self._wand_salvage.setChecked(True)
        self._wand_salvage.setToolTip(
            "On: a flood over the budget is trimmed back to the pixels "
            "reachable nearest the click, which leaves a bounded object you "
            "can edit. Off: an over-budget flood is refused outright and the "
            "mask is untouched, so a tolerance that is far too high says so "
            "instead of handing back a piece of the field."
        )
        self._wand_salvage.toggled.connect(self._on_wand_salvage_changed)
        wand_form.addRow("", self._wand_salvage)
        wand_card.body_layout.addLayout(wand_form)

        runaway = QGroupBox("Trim a runaway flood")
        runaway.setCheckable(True)
        runaway.setChecked(True)
        runaway.setToolTip(
            "A flood that reaches a bright seam — debris, a saturated "
            "membrane, a well rim — walks out along it and takes the field. "
            "This reads the flood's width outward from the click and cuts it "
            "where it suddenly and persistently widens. It does nothing to a "
            "flood that did not run away."
        )
        runaway.toggled.connect(self._on_wand_trim_runaway_changed)
        runaway_form = QFormLayout(runaway)
        self._wand_runaway_ratio = QDoubleSpinBox()
        self._wand_runaway_ratio.setDecimals(2)
        self._wand_runaway_ratio.setRange(1.2, 10.0)
        self._wand_runaway_ratio.setSingleStep(0.1)
        self._wand_runaway_ratio.setValue(2.0)
        self._wand_runaway_ratio.setToolTip(
            "How much wider than the object a scanline must be to count as a "
            "leak. Lower it if leaks are getting through; raise it if a "
            "genuinely lobed object is being cut."
        )
        self._wand_runaway_ratio.valueChanged.connect(
            self._on_wand_runaway_ratio_changed)
        runaway_form.addRow("Growth ratio", self._wand_runaway_ratio)
        self._wand_runaway_warmup = QSpinBox()
        self._wand_runaway_warmup.setRange(1, 500)
        self._wand_runaway_warmup.setValue(12)
        self._wand_runaway_warmup.setToolTip(
            "Pixels nearest the click that are not judged. One pixel widening "
            "to two is a ratio of 2 and means nothing, so the first rows out "
            "of the seed are skipped."
        )
        self._wand_runaway_warmup.valueChanged.connect(
            self._on_wand_runaway_warmup_changed)
        runaway_form.addRow("Warm-up (px)", self._wand_runaway_warmup)
        self._wand_runaway_min_base = QSpinBox()
        self._wand_runaway_min_base.setRange(1, 1000)
        self._wand_runaway_min_base.setValue(8)
        self._wand_runaway_min_base.setToolTip(
            "The width the object must reach before a leak can be called at "
            "all. Raise it for large objects, lower it if the wand is used on "
            "something only a few pixels across."
        )
        self._wand_runaway_min_base.valueChanged.connect(
            self._on_wand_runaway_min_base_changed)
        runaway_form.addRow("Min baseline (px)", self._wand_runaway_min_base)
        self._wand_runaway_confirm = QSpinBox()
        self._wand_runaway_confirm.setRange(1, 20)
        self._wand_runaway_confirm.setValue(2)
        self._wand_runaway_confirm.setToolTip(
            "Consecutive widened scanlines required before cutting, so one "
            "noisy row cannot take half the object off."
        )
        self._wand_runaway_confirm.valueChanged.connect(
            self._on_wand_runaway_confirm_changed)
        runaway_form.addRow("Confirmation (px)", self._wand_runaway_confirm)
        wand_card.body_layout.addWidget(runaway)
        self._wand_runaway_group = runaway

        edge = QGroupBox("Shape the cut edge")
        edge.setToolTip(
            "A trimmed runaway ends in a straight line, which no cell has. "
            "These two put the boundary back on the image: one re-floods at "
            "a tolerance that does not escape, the other lets the edge settle "
            "onto the nearest intensity gradient."
        )
        edge_form = QFormLayout(edge)
        self._wand_intensity_border = QCheckBox("Re-flood below the escape")
        self._wand_intensity_border.setChecked(True)
        self._wand_intensity_border.setToolTip(
            "When a leak is found, search for the highest tolerance whose "
            "flood stays put and take that instead of the straight cut. The "
            "boundary is then drawn by the image's own intensities. Only "
            "helps when the seam is dimmer than the object; when it is "
            "exactly as bright, no tolerance separates them and the cut "
            "stands."
        )
        self._wand_intensity_border.toggled.connect(
            self._on_wand_intensity_border_changed)
        edge_form.addRow("", self._wand_intensity_border)
        self._wand_intensity_steps = QSpinBox()
        self._wand_intensity_steps.setRange(3, 14)
        self._wand_intensity_steps.setValue(8)
        self._wand_intensity_steps.setToolTip(
            "Halvings used to find that tolerance. Each step is one more "
            "flood, so this is precision against click latency; eight is "
            "finer than one grey level on most images."
        )
        self._wand_intensity_steps.valueChanged.connect(
            self._on_wand_intensity_steps_changed)
        edge_form.addRow("Search steps", self._wand_intensity_steps)
        self._wand_gradient_taper = QCheckBox("Taper onto the gradient")
        self._wand_gradient_taper.setChecked(True)
        self._wand_gradient_taper.setToolTip(
            "Let the provisional edge move onto the nearest real intensity "
            "change, inside the band below. This is what removes the last "
            "straight lines and circular arcs left by a cut or a budget."
        )
        self._wand_gradient_taper.toggled.connect(
            self._on_wand_gradient_taper_changed)
        edge_form.addRow("", self._wand_gradient_taper)
        self._wand_gradient_sigma = QDoubleSpinBox()
        self._wand_gradient_sigma.setDecimals(1)
        self._wand_gradient_sigma.setRange(0.0, 10.0)
        self._wand_gradient_sigma.setSingleStep(0.5)
        self._wand_gradient_sigma.setValue(2.0)
        self._wand_gradient_sigma.setToolTip(
            "Blur applied before looking for the edge. Raise it on speckled "
            "fields so noise is not mistaken for a boundary; lower it for "
            "small, sharply bounded objects."
        )
        self._wand_gradient_sigma.valueChanged.connect(
            self._on_wand_gradient_sigma_changed)
        edge_form.addRow("Smoothing (sigma)", self._wand_gradient_sigma)
        self._wand_gradient_margin = QSpinBox()
        self._wand_gradient_margin.setRange(1, 100)
        self._wand_gradient_margin.setValue(8)
        self._wand_gradient_margin.setToolTip(
            "How far either side of the cut the edge is free to move. Wider "
            "lets it find a boundary further away; too wide and it can reach "
            "the seam the cut was made to escape."
        )
        self._wand_gradient_margin.valueChanged.connect(
            self._on_wand_gradient_margin_changed)
        edge_form.addRow("Transition band (px)", self._wand_gradient_margin)
        self._wand_gradient_erode = QSpinBox()
        self._wand_gradient_erode.setRange(0, 50)
        self._wand_gradient_erode.setValue(3)
        self._wand_gradient_erode.setToolTip(
            "How far inside the kept region counts as certainly the object. "
            "Everything between that inset and the discarded part is what the "
            "taper is allowed to decide."
        )
        self._wand_gradient_erode.valueChanged.connect(
            self._on_wand_gradient_erode_changed)
        edge_form.addRow("Foreground inset (px)", self._wand_gradient_erode)
        wand_card.body_layout.addWidget(edge)
        self._wand_edge_group = edge

        col.addWidget(wand_card)

        norm_card = Card(title="Display")
        norm_form = QFormLayout()
        self._norm_lo = QDoubleSpinBox()
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
        self._min_area.setToolTip(
            "Smallest object worth keeping, in pixels. Removing small "
            "objects drops everything under it, and neither Otsu nor "
            "Cellpose-SAM will produce an object below it, so one judgement "
            "about debris is made in one box."
        )
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

        col.addWidget(self._build_cellpose_card())
        col.addWidget(self._build_magnifier_card())

        col.addStretch(1)
        return wrap

    def _install_shortcuts(self):
        """Bind the keys that move through fields and undo edits.

        ARROWS AND UNDO ARE THE WHOLE POINT of a curation screen: the work is
        hundreds of small corrections, and a hand that has to find the mouse
        between each one does a fraction as many.
        """
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_next)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save)
        QShortcut(QKeySequence("B"), self, lambda: self._set_mode(MODE_BRUSH))
        QShortcut(QKeySequence("E"), self, lambda: self._set_mode(MODE_ERASE))
        QShortcut(QKeySequence("W"), self, lambda: self._set_mode(MODE_WAND_ADD))
        QShortcut(QKeySequence("D"), self, lambda: self._set_mode(MODE_DRAW))
        QShortcut(QKeySequence("V"), self, lambda: self._set_mode(MODE_DIVIDE))
        QShortcut(QKeySequence("Z"), self, lambda: self._set_mode(MODE_ZOOM))
        QShortcut(QKeySequence("R"), self, lambda: self._set_mode(MODE_RECROP))
        QShortcut(QKeySequence("Escape"), self, self._on_reset_zoom)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    def _set_mode(self, mode: str):
        """Switch the canvas between draw, erase and wand.

        :param mode: the mode's name.
        """
        self._canvas.mode = mode
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

    def _on_brush_size_changed(self, v: int):
        """Resize the brush.

        :param v: the new radius in pixels.
        """
        self._canvas.brush_radius = int(v)
        self._brush_size_label.setText(f"{v} px")

    def _on_normalize_changed(self, _v: float):
        """Re-stretch the displayed intensity range.

        DISPLAY ONLY. The mask is drawn against what the user can see, but the
        pixels underneath are untouched -- a normalisation that changed the
        data would make every mask depend on the contrast it was drawn at.

        :param _v: the changed value; both ends are re-read from the widgets.
        """
        self._canvas.norm_lo = float(self._norm_lo.value())
        self._canvas.norm_hi = float(self._norm_hi.value())
        self._canvas.refresh()

    def _on_wand_tolerance_changed(self, v: float):
        """Set how far the wand will grow in intensity.

        :param v: the tolerance.
        """
        self._canvas.wand_tolerance = float(v)

    def _on_wand_pct_changed(self, v: float):
        """Set the wand's tolerance as a percentile instead of an absolute.

        :param v: the percentile.
        """
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
        """Cap how many pixels one wand fill may claim.

        :param v: the pixel cap.
        """
        self._canvas.wand_max_pixels = int(v)

    def _on_wand_salvage_changed(self, on: bool):
        """Keep or discard a fill that hit the cap.

        :param on: True to keep the truncated fill.
        """
        self._canvas.wand_salvage_over_cap = bool(on)

    def _on_wand_trim_runaway_changed(self, on: bool):
        """Turn runaway trimming on or off.

        :param on: True to trim.
        """
        self._canvas.wand_trim_runaway = bool(on)

    def _on_wand_runaway_ratio_changed(self, v: float):
        """Set the growth ratio that counts as a runaway.

        :param v: the ratio.
        """
        self._canvas.wand_runaway_ratio = float(v)

    def _on_wand_runaway_warmup_changed(self, v: int):
        """Set how many steps run before runaway detection starts.

        A WARMUP IS NEEDED because every fill grows fast at first: judging the
        ratio from step one would call every fill a runaway.

        :param v: the step count.
        """
        self._canvas.wand_runaway_warmup = int(v)

    def _on_wand_runaway_min_base_changed(self, v: int):
        """Set the smallest area a runaway judgement will be made against.

        :param v: the pixel count.
        """
        self._canvas.wand_runaway_min_base = int(v)

    def _on_wand_runaway_confirm_changed(self, v: int):
        """Set how many consecutive steps confirm a runaway.

        :param v: the step count.
        """
        self._canvas.wand_runaway_confirm = int(v)

    def _on_wand_intensity_border_changed(self, on: bool):
        """Enable the re-flood, and its step count with it.

        The step count is the precision of a search that is not running
        when the re-flood is off, so leaving it live would offer a setting
        that changes nothing.
        """
        self._canvas.wand_intensity_border = bool(on)
        self._wand_intensity_steps.setEnabled(bool(on))

    def _on_wand_intensity_steps_changed(self, v: int):
        """Set how many intensity steps the wand grows through.

        :param v: the step count.
        """
        self._canvas.wand_intensity_steps = int(v)

    def _on_wand_gradient_taper_changed(self, on: bool):
        """Enable the taper, and the three numbers that shape it."""
        self._canvas.wand_gradient_taper = bool(on)
        for w in (self._wand_gradient_sigma, self._wand_gradient_margin,
                  self._wand_gradient_erode):
            w.setEnabled(bool(on))

    def _on_wand_gradient_sigma_changed(self, v: float):
        """Set the blur applied before the gradient is measured.

        :param v: the sigma.
        """
        self._canvas.wand_gradient_sigma = float(v)

    def _on_wand_gradient_margin_changed(self, v: int):
        """Set how far past the gradient edge the fill may reach.

        :param v: the margin in pixels.
        """
        self._canvas.wand_gradient_margin = int(v)

    def _on_wand_gradient_erode_changed(self, v: int):
        """Set how much the gradient mask is eroded before use.

        :param v: the erosion in pixels.
        """
        self._canvas.wand_gradient_erode = int(v)

    def _on_zoom_speed_changed(self, v: float):
        """Set how fast the wheel zooms.

        :param v: the speed multiplier.
        """
        self._canvas.zoom_speed = float(v)

    def _on_reset_zoom(self):
        """Put the view back to the whole field."""
        self._canvas.reset_zoom()

    def _on_zoom_changed(self, zoomed: bool):
        """Enable the reset button only while the view is zoomed.

        :param zoomed: True when the view is not showing the whole field.
        """
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
        """Enable undo and redo from what the history actually holds."""
        self._btn_undo.setEnabled(self._history.can_undo())
        self._btn_redo.setEnabled(self._history.can_redo())

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
        """Apply the object filter to the mask on screen."""
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


    def _build_view_tabs(self) -> QTabWidget:
        """The canvas and Cellpose's two intermediates, as tabs.

        THE PANES ARE NOT DECORATION. The probability map and the flow
        field are what say *why* a mask came out the way it did: seeing
        which pixels the network was confident about, beside the objects
        it drew from them, is the difference between moving a threshold
        with a reason and moving it by guessing. They sit on tabs of the
        same pane as the mask so they are at the same size and the same
        zoom-to-fit as the image being judged.

        Both tabs stay ENABLED before Cellpose has run, unlike the
        standalone tool's, because a disabled tab cannot be opened to
        read the one sentence that explains why it is empty.
        """
        tabs = QTabWidget()
        tabs.setObjectName("MakeMasksViewTabs")
        tabs.addTab(self._canvas, "Mask")
        self._prob_pane = _FlowPane()
        self._flow_pane = _FlowPane()
        self._tab_prob = tabs.addTab(self._prob_pane, "Cell probability")
        self._tab_flow = tabs.addTab(self._flow_pane, "Flows")
        self._view_tabs = tabs
        return tabs

    def _reset_flow_panes(self) -> None:
        """Empty both intermediates and put the view back on the mask.

        They belong to ONE Cellpose run on ONE field. Carried over to the
        next field they would be a picture of the wrong image, read as a
        picture of this one — the worst shape this could take, because
        nothing on screen would say so.
        """
        self._prob_pane.clear_view()
        self._flow_pane.clear_view()
        self._view_tabs.setCurrentIndex(0)

    def _build_cellpose_card(self) -> Card:
        """The Cellpose-SAM settings, and the detect button they drive.

        The settings are ON THE PANEL rather than assumed. Both
        thresholds start at Cellpose's own defaults —
        :data:`CELLPROB_THRESHOLD` and :data:`FLOW_THRESHOLD` — so a run
        made without touching anything is the run Cellpose would have
        made, and a changed number is visibly a departure from it.

        The button itself goes in the one tool row rather than in this
        card: it is an action, and it has to stay reachable when the
        settings are toggled away.
        """
        from ...settings import cellpose_model_choices

        #: Loaded models, by the name that was asked for. Loading cpsam
        #: costs seconds and hundreds of megabytes, and a segmentation
        #: session runs it once per field.
        self._cp_loaded: dict = {}

        card = Card(
            title="Cellpose-SAM",
            subtitle="Segments the open field. Both thresholds start at "
                     "Cellpose's own defaults.",
        )
        form = QFormLayout()

        self._cp_model = QComboBox()
        for name in cellpose_model_choices():
            self._cp_model.addItem(name, name)
        self._cp_model.setToolTip(
            "Which weights segment this field. The list is read from the "
            "Cellpose installed on this machine rather than hard-coded, so "
            "a version that ships more models offers them here. A "
            "fine-tuned checkpoint trained by Train Cellpose is applied by "
            "running that module against the folder.")
        form.addRow("Model", self._cp_model)

        self._cp_cellprob = QDoubleSpinBox()
        self._cp_cellprob.setDecimals(2)
        self._cp_cellprob.setRange(-12.0, 12.0)
        self._cp_cellprob.setSingleStep(0.1)
        self._cp_cellprob.setValue(CELLPROB_THRESHOLD)
        self._cp_cellprob.setToolTip(
            "Where the cell-probability map is cut. Lower it to keep dimmer "
            "objects the network was unsure about; raise it to keep only "
            "confident ones. Open the Cell probability tab after a run and "
            "the number has something to be judged against.")
        form.addRow("Cell probability", self._cp_cellprob)

        self._cp_flow = QDoubleSpinBox()
        self._cp_flow.setDecimals(2)
        self._cp_flow.setRange(0.0, 10.0)
        self._cp_flow.setSingleStep(0.1)
        self._cp_flow.setValue(FLOW_THRESHOLD)
        self._cp_flow.setToolTip(
            "How far a candidate object's flows may disagree with the ones "
            "the network predicted before it is thrown away. LOWER IS "
            "STRICTER, which is the opposite of the way it reads. 0 turns "
            "the check off entirely.")
        form.addRow("Flow threshold", self._cp_flow)

        self._cp_diameter = QSpinBox()
        self._cp_diameter.setRange(0, 10_000)
        self._cp_diameter.setSingleStep(5)
        self._cp_diameter.setValue(0)
        self._cp_diameter.setToolTip(
            "Expected object diameter in pixels; 0 lets Cellpose work it "
            "out. It is the one pre-SAM sizing setting Cellpose 4 still "
            "honours — it rescales the image by 30/diameter — so it is the "
            "control to reach for when objects come out split or fused.")
        form.addRow("Diameter (px)", self._cp_diameter)

        card.body_layout.addLayout(form)

        self._cp_normalize = QCheckBox("Normalize each field")
        self._cp_normalize.setChecked(True)
        self._cp_normalize.setToolTip(
            "Percentile-normalize the field before segmenting it, which is "
            "what Cellpose expects. Turn it off only for data already "
            "normalized upstream, where doing it twice changes the result.")
        card.body_layout.addWidget(self._cp_normalize)

        self._btn_cellpose = QPushButton("Cellpose-SAM detect")
        self._btn_cellpose.setIcon(iconset.icon("run"))
        self._btn_cellpose.setCursor(Qt.PointingHandCursor)
        self._btn_cellpose.setToolTip(
            "Segment the open field with Cellpose-SAM and fold the result "
            "in as the replace/merge setting says. Fills the Cell "
            "probability and Flows tabs with what the run was thinking.")
        self._btn_cellpose.clicked.connect(self._on_detect_cellpose)
        self.add_toolbar_action(self._btn_cellpose)
        return card

    def _detect_min_area(self) -> int:
        """Smallest object a detection may keep, in pixels.

        The same box the Remove-small button reads, because they are the
        same judgement: an object this size is debris either way, and
        having Cellpose keep what the next button would delete would be
        two answers to one question.
        """
        return int(self._min_area.value())

    def _cellpose_model(self, model_name: str):
        """Load ``model_name`` once and keep it for the rest of the session."""
        with _CELLPOSE_LOCK:
            if model_name not in self._cp_loaded:
                self._cp_loaded[model_name] = load_cellpose_model(model_name)
            return self._cp_loaded[model_name]

    def _sync_model_choices(self) -> None:
        """Add any model the live Cellpose reports that the combo has not.

        The combo is built while the screen is, and importing Cellpose
        costs about two and a half seconds because it pulls in torch — so
        :func:`spacr.settings.cellpose_model_choices` answers from its
        fallback list until something has actually imported it. The first
        detect run is that something, and it is the first moment the live
        list can be had for free.
        """
        from ...settings import cellpose_model_choices

        for name in cellpose_model_choices():
            if self._cp_model.findData(name) < 0:
                self._cp_model.addItem(name, name)

    def _show_intermediates(self, cellprob, flow) -> None:
        """Put one run's probability map and flow field on their tabs."""
        if cellprob is None:
            self._prob_pane.clear_view()
        else:
            self._prob_pane.show_rgb(cellprob_heatmap(cellprob))
        if flow is None:
            self._flow_pane.clear_view()
        else:
            self._flow_pane.show_rgb(flow)

    def run_cellpose(self) -> int:
        """Segment the open field with Cellpose-SAM; return objects found.

        The two panes are filled BEFORE the mask is touched, and they are
        filled even when the run found nothing at all. A run that returns
        an empty mask is exactly the run whose probability map you need to
        see: it says whether the network found nothing, or found plenty
        and the threshold threw it away.

        The run blocks this screen while it is going. Cellpose on a GPU
        answers in about a second on one field, and moving it to a thread
        would mean a second worker on a screen that already drains one on
        close; the button is disabled and the cursor says wait instead.
        """
        if self._canvas.image is None or self._canvas.mask is None:
            self._status_label.setText(
                "Open a folder before running Cellpose-SAM.")
            return 0

        model_name = self._cp_model.currentData() or "cpsam"
        app = QApplication.instance()
        self._btn_cellpose.setEnabled(False)
        self._status_label.setText(f"Cellpose-SAM ({model_name}) running…")
        if app is not None:
            app.setOverrideCursor(Qt.WaitCursor)
            app.processEvents()
        try:
            with _CELLPOSE_LOCK:
                labels, cellprob, flow = cellpose_detect(
                    self._canvas.image,
                    self._cellpose_model(model_name),
                    diameter=int(self._cp_diameter.value()),
                    normalize=bool(self._cp_normalize.isChecked()),
                    flow_threshold=float(self._cp_flow.value()),
                    cellprob_threshold=float(self._cp_cellprob.value()),
                    min_size=self._detect_min_area(),
                )
        except Exception as exc:
            LOG.exception("Cellpose-SAM detect failed")
            self._warn("Cellpose-SAM detect failed", str(exc))
            return 0
        finally:
            if app is not None:
                app.restoreOverrideCursor()
            self._btn_cellpose.setEnabled(True)

        self._show_intermediates(cellprob, flow)
        self._sync_model_choices()

        found = int(labels.max()) if labels.size else 0
        if not found:
            self._status_label.setText(
                "Cellpose-SAM found no objects — the mask is unchanged. The "
                "Cell probability tab shows what it had to work with.")
            return 0

        mode = self._combine_mode.currentData()
        try:
            out = engine.combine_masks(self._canvas.mask, labels, mode)
        except Exception as exc:
            self._warn("Cellpose-SAM detect failed", str(exc))
            return 0
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        self._record("detect", mode, changed, method="cellpose",
                      model=model_name, n_objects=found,
                      cellprob_threshold=float(self._cp_cellprob.value()),
                      flow_threshold=float(self._cp_flow.value()),
                      diameter=int(self._cp_diameter.value()),
                      min_size=self._detect_min_area())
        self._history.push(out)
        self._refresh_history_buttons()
        self._status_label.setText(
            f"Cellpose-SAM ({model_name}) found {found} object(s) — "
            f"{mode}d into the mask. See the Cell probability and Flows tabs."
        )
        return found

    def _on_detect_cellpose(self):
        """Toolbar handler for the Cellpose-SAM detect button."""
        self.run_cellpose()

    def _build_magnifier_card(self) -> Card:
        """The live magnifier's settings, and the toggle that turns it on.

        The toggle goes in the tool row beside Cellpose-SAM detect, because it
        has to stay reachable with the settings hidden; the four settings the
        request named -- mode, size, zoom, sensitivity -- and the overlap rule
        go here, with what is segmented (the region under the mouse or the
        whole image once), whether objects cut by the box are offered, and
        the progress and Cancel of a whole-image run. Cellpose mode reads its
        model and diameter from the Cellpose-SAM card and classical mode
        reads Bright and Min area from Object operations, so each of those
        judgements is still made in one box. Nothing here persists between
        sessions, like every other setting on this panel.
        """
        magnifier = self._magnifier
        card = Card(
            title="Live magnifier",
            subtitle="Segments the region under the mouse, or the whole image "
                     "once, and shows its objects magnified. A click adds "
                     "objects to the mask.",
        )
        form = QFormLayout()

        self._mag_mode = QComboBox()
        self._mag_mode.addItem("Classical", "classical")
        try:
            has_cellpose = find_spec("cellpose") is not None
        except (ImportError, ValueError):
            has_cellpose = False
        if has_cellpose:
            self._mag_mode.addItem("Cellpose", "cellpose")
        self._mag_mode.setToolTip(
            "Which model segments the region in the box. Classical thresholds "
            "the region at Otsu's level and splits touching objects with a "
            "watershed; it needs nothing installed, follows the Bright switch "
            "and the Min area box under Object operations, and runs whenever "
            "a model cannot be loaded. Cellpose uses the model and diameter "
            "set in the Cellpose-SAM settings and is slow without a GPU.")
        self._mag_mode.currentIndexChanged.connect(
            lambda _index: magnifier.set_mode(self._mag_mode.currentData()))
        form.addRow("Mode", self._mag_mode)

        self._mag_scope = QComboBox()
        self._mag_scope.addItem("Region under the mouse", "region")
        self._mag_scope.addItem("Whole image", "image")
        self._mag_scope.setToolTip(
            "What the model segments. Region under the mouse runs it on the "
            "box as the mouse moves, and a click adds every object the box "
            "outlines. Whole image runs it once on the entire image in the "
            "background; the box then shows those objects, a click adds the "
            "object under it, and a right-click removes the mask object under "
            "it. Changing a setting the model reads discards the whole-image "
            "objects and segments the image again.")
        self._mag_scope.currentIndexChanged.connect(
            lambda _index: self._on_magnifier_scope(
                self._mag_scope.currentData()))
        form.addRow(QLabel("Segment"), self._mag_scope)

        self._mag_size = QSpinBox()
        self._mag_size.setRange(*_MAGNIFIER_SIZE_RANGE)
        self._mag_size.setSingleStep(16)
        self._mag_size.setValue(_MAGNIFIER_SIZE)
        self._mag_size.setToolTip(
            "Side of the square box, in image pixels. Under Region under the "
            "mouse it is also the region the model segments, so make it wider "
            "than the largest object you want to add.")
        self._mag_size.valueChanged.connect(magnifier.set_size)
        form.addRow("Size (px)", self._mag_size)

        self._mag_exclude_border = QCheckBox(
            "Exclude objects touching the box border")
        self._mag_exclude_border.setChecked(True)
        self._mag_exclude_border.setToolTip(
            "When ticked, an object that touches an edge of the box inside the "
            "image is not offered, because the box may have cut it off. Untick "
            "it to offer and add such objects too, as far as the box sees "
            "them. Objects at the image border are offered either way. It "
            "applies to Region under the mouse; Whole image never cuts an "
            "object.")
        self._mag_exclude_border.toggled.connect(magnifier.set_exclude_border)
        form.addRow(self._mag_exclude_border)
        self._build_magnifier_save_mode(form)

        self._mag_zoom = QDoubleSpinBox()
        self._mag_zoom.setDecimals(2)
        self._mag_zoom.setRange(*_MAGNIFIER_ZOOM_RANGE)
        self._mag_zoom.setSingleStep(0.25)
        self._mag_zoom.setValue(_MAGNIFIER_ZOOM)
        self._mag_zoom.setToolTip(
            "How many times larger than the canvas the box draws its region. "
            "The mouse wheel changes it while the magnifier is on. It changes "
            "only what you see: the model always segments the region at the "
            "image's own resolution.")
        self._mag_zoom.valueChanged.connect(magnifier.set_zoom)
        magnifier.zoom_changed.connect(self._mag_zoom.setValue)
        form.addRow("Zoom", self._mag_zoom)

        self._mag_sensitivity = QDoubleSpinBox()
        self._mag_sensitivity.setDecimals(2)
        self._mag_sensitivity.setRange(*_MAGNIFIER_SENSITIVITY_RANGE)
        self._mag_sensitivity.setSingleStep(0.25)
        self._mag_sensitivity.setValue(_MAGNIFIER_SENSITIVITY)
        self._mag_sensitivity.setToolTip(
            "How readily an object is accepted. Raise it to take in dimmer or "
            "less certain objects, lower it to keep only clear ones; 0 is each "
            "model's own default. For Cellpose it is the cell probability "
            "threshold with the sign reversed, so a sensitivity of 1.5 is a "
            "threshold of -1.5.")
        self._mag_sensitivity.valueChanged.connect(magnifier.set_sensitivity)
        form.addRow("Sensitivity", self._mag_sensitivity)

        self._mag_overlap = QComboBox()
        self._mag_overlap.addItem("Clip", "clip")
        self._mag_overlap.addItem("Skip", "skip")
        self._mag_overlap.addItem("Replace", "replace")
        self._mag_overlap.setToolTip(
            "What a new object does where the mask already has an object. "
            "Clip keeps only its unlabelled pixels, so no existing object "
            "loses a pixel. Skip leaves out any object that touches an "
            "existing one. Replace lets the new object take every pixel it "
            "covers.")
        form.addRow("Overlap", self._mag_overlap)
        card.body_layout.addLayout(form)

        progress = QHBoxLayout()
        self._mag_progress = QProgressBar()
        self._mag_progress.setRange(0, 0)
        self._mag_progress.setTextVisible(False)
        self._mag_progress.hide()
        self._mag_cancel = QPushButton("Cancel")
        self._mag_cancel.hide()
        self._mag_cancel.clicked.connect(
            lambda _checked=False: magnifier.cancel_image())
        progress.addWidget(self._mag_progress, 1)
        progress.addWidget(self._mag_cancel)
        card.body_layout.addLayout(progress)
        magnifier.busy_changed.connect(self._on_magnifier_busy)

        self._btn_magnifier = QPushButton("Magnifier")
        self._btn_magnifier.setIcon(iconset.icon("search"))
        self._btn_magnifier.setCheckable(True)
        self._btn_magnifier.setMinimumHeight(32)
        self._btn_magnifier.setCursor(Qt.PointingHandCursor)
        self._btn_magnifier.setToolTip(
            "Show a box under the mouse with the region around it magnified "
            "and the objects a model finds in it outlined. A click adds them "
            "to the mask as new objects — under Whole image, only the object "
            "clicked — one undo step per click. While it is on, the mouse "
            "wheel changes the box's zoom rather than the view's.")
        self._btn_magnifier.toggled.connect(self._on_toggle_magnifier)
        self.add_toolbar_action(self._btn_magnifier)
        return card

    def _build_magnifier_save_mode(self, form: QFormLayout) -> None:
        """Item 417, part 6: which objects a click or a drag adds.

        A method of its own with one call from the Live magnifier card, so the
        card can be rearranged without rewriting this row. The choice is read
        when the button goes down, so changing it mid-drag applies to the next
        press.
        """
        box = self._mag_save = QComboBox()
        box.addItem("All objects in the zoom area", "zoom")
        box.addItem("Only objects touching the mouse", "touching")
        box.setToolTip(
            "Which objects a click or a drag adds. All objects in the zoom "
            "area adds every object the box outlines. Only objects touching "
            "the mouse adds just the object under the cursor and leaves the "
            "rest of the box out. Press and drag to keep adding along the "
            "path: the objects the cursor passes over become one object, "
            "joined from the pieces found in each box where they lie in the "
            "image. Whole image always adds only the objects under the mouse.")
        box.currentIndexChanged.connect(
            lambda _index: setattr(self._magnifier, "save_mode",
                                   box.currentData()))
        form.addRow(QLabel("Objects added"), box)

    def _magnifier_context(self) -> dict:
        """The settings the magnifier's models read from elsewhere on the panel."""
        return {
            "model_name": self._cp_model.currentData() or "cpsam",
            "diameter": int(self._cp_diameter.value()),
            "bright": bool(self._otsu_bright.isChecked()),
            "min_area": self._detect_min_area(),
        }

    def _on_magnifier_scope(self, scope) -> None:
        """Segment the region under the mouse or the whole image.

        The border option is greyed out for the whole image, whose objects
        the box never cuts.
        """
        self._mag_exclude_border.setEnabled(scope != "image")
        self._magnifier.set_scope(scope)

    def _on_magnifier_busy(self, busy: bool) -> None:
        """Show the whole-image run's progress and Cancel while it runs."""
        self._mag_progress.setVisible(bool(busy))
        self._mag_cancel.setVisible(bool(busy))

    def _on_magnifier_context_changed(self, *_args) -> None:
        """A setting a magnifier model reads changed elsewhere on the panel.

        Whole-image objects found under the old value are discarded at once,
        and the status line says so, rather than on the next mouse move.
        """
        self._magnifier.refresh()

    def _on_toggle_magnifier(self, on: bool) -> None:
        """Turn the live magnifier on or off from the tool row.

        The status line is written before the magnifier is switched, so a
        whole-image run the switch starts has the last word on it.
        """
        from ..i18n import tr

        if on and self._magnifier.scope == "image":
            self._status_label.setText(tr(
                "Magnifier on: a click adds the object under it and a "
                "right-click removes the mask object under it; the mouse "
                "wheel changes its zoom."))
        elif on:
            self._status_label.setText(
                "Magnifier on: a click adds the objects outlined in the box; "
                "the mouse wheel changes its zoom.")
        else:
            self._status_label.setText(
                "Magnifier off. The objects it added stay in the mask.")
        self._magnifier.set_enabled(on)
        if on and self._canvas.underMouse():
            where = self._canvas.mapFromGlobal(QCursor.pos())
            self._magnifier.hover(QPointF(where))
        self._canvas.update()

    def _commit_magnifier_result(self, result) -> List[int]:
        """Paste the objects a magnifier click asked for into the mask.

        One click is one edit: one ledger entry naming the new ids and one
        undo step. The overlap rule is read now, from the Overlap box; new ids
        start one past the mask's top id, and an object left smaller than
        Min area by the rule is not added -- see
        :func:`spacr.qt.mask_engine._paste_region_objects`.

        A whole-image click arrives as that one object, cut to its bounding
        box, and is recorded with ``scope="image"``.

        :returns: the ids added; empty when nothing was.
        """
        from ..i18n import tr

        mask = self._canvas.mask
        request = result.request
        if mask is None or tuple(mask.shape[:2]) != tuple(request.shape):
            return []
        overlap = self._mag_overlap.currentData() or "clip"
        try:
            out, added = engine._paste_region_objects(
                mask, result.labels, request.box[:2], overlap=overlap,
                min_area=self._detect_min_area())
        except ValueError as exc:
            self._status_label.setText(
                f"Magnifier could not add objects: {exc}")
            return []
        if not added and request.scope == "image":
            self._status_label.setText(tr(
                "Magnifier: nothing was added — the Overlap rule or Min area "
                "leaves nothing of the object under the click."))
            return []
        if not added:
            self._status_label.setText(
                "Magnifier: nothing to add — the box outlines no object, or "
                "every object it outlines overlaps one already in the mask.")
            return []
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        self._record("magnifier", list(added), changed,
                      mode=result.mode, overlap=overlap,
                      box=[int(v) for v in request.box],
                      sensitivity=float(request.sensitivity),
                      n_objects=len(added), scope=request.scope)
        self._history.push(out)
        self._refresh_history_buttons()
        self._status_label.setText(tr(
            "Magnifier added {n} object(s) — Ctrl+Z to undo", n=len(added)))
        return added

    def _remove_magnifier_object(self, x: int, y: int) -> int:
        """Remove the mask object at image ``(x, y)``: a whole-image right click.

        Any object qualifies, whatever put it in the mask. One removal is one
        edit: a ``delete`` ledger entry naming the id, with
        ``tool="magnifier"``, and one undo step. A click on background
        changes nothing and records nothing.

        :returns: the id removed, or 0.
        """
        from ..i18n import tr

        mask = self._canvas.mask
        if mask is None:
            return 0
        height, width = mask.shape[:2]
        label = (int(mask[y, x]) if 0 <= y < height and 0 <= x < width
                 else 0)
        if label <= 0:
            self._status_label.setText(tr(
                "Magnifier: there is no mask object under the click — nothing "
                "was removed."))
            return 0
        out = engine.erase_object_at(mask, x, y)
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        self._record("delete", label, changed, tool="magnifier")
        self._history.push(out)
        self._refresh_history_buttons()
        self._status_label.setText(tr(
            "Magnifier removed object {label} — Ctrl+Z to undo", label=label))
        return label

    def _apply_magnifier_drag(self, payload) -> List[int]:
        """Show a magnifier drag's objects in the mask, or commit them (417).

        While the button is down the mask shows the drag's objects pasted onto
        the mask the drag started from, and nothing is recorded. The final
        paste is ONE edit -- one ``magnifier`` ledger entry marked
        ``drag=True`` and one undo step -- through the Overlap rule and Min
        area, as a click's objects go in. A mask another edit put on screen
        during the drag becomes the mask it pastes onto.

        :param payload: ``(outcome, final)`` from
            :attr:`_LiveMagnifier.drag_ready`.
        :returns: the ids the final paste added; empty for a preview.
        """
        from ..i18n import tr

        found, final = payload
        canvas = self._canvas
        if canvas.mask is not self._drag_shown:
            self._drag_base = canvas.mask
        base = out = self._drag_base
        overlap = self._mag_overlap.currentData() or "clip"
        added: List[int] = []
        if base is not None and found is not None and found.objects:
            pasted, added = engine._paste_region_objects(
                base, found.labels, found.origin, overlap=overlap,
                min_area=self._detect_min_area())
            out = pasted if added else base
        canvas.mask = out
        canvas.refresh()
        self._drag_shown = None if final else out
        if not final:
            return []
        if not added:
            self._status_label.setText(tr(
                "Magnifier: nothing was added — there was no object under the "
                "mouse, or the Overlap rule or Min area left nothing of it."))
            return []
        height, width = found.labels.shape[:2]
        x0, y0 = found.origin
        self._record("magnifier", list(added), self._pixels_changed(out),
                     mode=self._magnifier.mode, overlap=overlap,
                     box=[x0, y0, x0 + width, y0 + height],
                     sensitivity=float(self._magnifier.sensitivity),
                     n_objects=len(added), scope=self._magnifier.scope,
                     drag=True, save=self._magnifier.save_mode,
                     frames=found.frames, merged=found.merged)
        self._history.push(out)
        self._refresh_history_buttons()
        self._status_label.setText(tr(
            "Magnifier added {n} object(s) — Ctrl+Z to undo", n=len(added)))
        return added

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

    def _on_pick_folder(self):
        """Ask for a folder of images and open it."""
        d = QFileDialog.getExistingDirectory(self, "Pick images folder",
                                              self._folder or os.getcwd())
        if not d:
            return
        self._open_folder(d)

    def _open_folder(self, folder: str,
                     files: Optional[List[str]] = None) -> bool:
        """List the folder's images and load the first.

        :param folder: the folder to open.
        :param files: the file names to offer, in the order to offer them.
            ``None`` -- every caller but :meth:`open_queue` -- lists the
            folder itself, which is what a file dialog or a dropped folder
            means. A session built by ``spacr-make-masks`` passes its own
            list, because the queue has already dropped what is reviewed and
            sorted what is left.
        :returns: whether a folder was opened. ``False`` means there was
            nothing in it to edit, which the user has been told about.
        """
        files = list(files) if files is not None else engine.list_images(folder)
        if not files:
            self._warn("No images", f"Found no image files in: {folder}")
            return False
        self._queue = None
        self._folder = folder
        self._image_files = files
        self._current_index = 0
        self._src_label.setText(f"{folder}  —  {len(files)} images")
        self._load_current()
        self._sync_button_states()
        prefs.push_recent_source("make_masks", folder)
        self._body_stack.setCurrentWidget(self._body_splitter)
        return True

    def _load_current(self):
        """Show the current field and whatever mask it already has."""
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

        self._magnifier.close()
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
        self._recrop_children = []
        self._reset_flow_panes()
        self._history.clear()
        self._history.push(mask)
        self._refresh_history_buttons()
        self._btn_reset_zoom.setEnabled(False)
        self._log = self._open_ledger(filename)
        self._status_label.setText(
            f"{filename}  "
            f"({self._current_index + 1}/{len(self._image_files)})"
        )
        self.apply_object_filter(on_load=True)
        self._magnifier.refresh()

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
            LOG.warning("Unreadable curation ledger beside %s: %s",
                        artifact, exc)
            log = CurationLog()
        if not log.artifact:
            log.artifact = artifact
            log.source = engine.CURATION_SOURCE
        return log

    def _on_recrop_requested(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Handle a box dragged with the Recrop tool."""
        self.recrop(x0, y0, x1, y1)

    def recrop(self, x0: int, y0: int, x1: int, y1: int) -> Optional[str]:
        """Extract one selected region into a separate training field.

        Accepted crops are written immediately, inserted after the source
        field in the queue, and marked on the canvas. Rejected selections are
        reported in the status label without modifying the queue.

        :returns: Filename of the recropped field, or ``None`` if the
            selection was rejected or could not be written.
        """
        if not self._image_files or self._canvas.mask is None \
                or self._canvas.image is None:
            self._status_label.setText("Recrop: no field open to cut.")
            return None
        filename = self._image_files[self._current_index]
        try:
            box = engine.recrop_box(self._canvas.mask.shape, (x0, y0), (x1, y1),
                                     existing=self._canvas.recrop_boxes)
        except engine.RecropRefused as refusal:
            self._status_label.setText(f"Recrop: {refusal}")
            return None
        try:
            written = engine.write_recrop(
                self._folder, filename, self._canvas.image,
                self._canvas.mask, box)
        except Exception as exc:
            self._warn("Recrop failed", str(exc))
            return None

        name = os.path.splitext(written.name)[0]
        self._canvas.recrop_boxes.append((*box, name))
        self._canvas.update()
        self._image_files.insert(
            self._current_index + len(self._recrop_children) + 1, written.name)
        self._recrop_children.append(written.name)
        area = (box[2] - box[0]) * (box[3] - box[1])
        self._record(engine.RECROP_KIND, written.name, area,
                      box=[int(v) for v in box],
                      n_objects=int(written.n_objects))
        self._status_label.setText(
            f"Recrop {name}: {box[2] - box[0]}x{box[3] - box[1]} px, "
            f"{written.n_objects} whole object(s), queued next "
            f"({len(self._recrop_children)} so far). "
            f"{filename} is retired when you move on."
        )
        return written.name

    def finish_recrop(self) -> bool:
        """Archive a recropped source field and advance to its first child.

        The source image, mask, and ledger move to
        ``recropped_originals/`` and remain recoverable through the recrop
        manifest.

        :returns: ``True`` if a field was archived.
        """
        if not self._recrop_children or not self._image_files:
            return False
        filename = self._image_files[self._current_index]
        children = list(self._recrop_children)
        boxes = [tuple(int(v) for v in box[:4])
                 for box in self._canvas.recrop_boxes]
        if self._canvas.mask is not None:
            try:
                engine.save_mask(self._folder, filename, self._canvas.mask,
                                  log=self._log)
            except Exception as exc:
                LOG.warning("Could not save %s before retiring it: %s",
                            filename, exc)
        try:
            engine.retire_recropped_original(
                self._folder, filename, children=children, boxes=boxes)
        except Exception as exc:
            self._warn("Recrop failed", str(exc))
            return False
        self._image_files.pop(self._current_index)
        self._recrop_children = []
        self._canvas.recrop_boxes = []
        if not self._image_files:
            self._current_index = 0
            self._canvas.image = None
            self._canvas.mask = None
            self._canvas.clear()
            self._status_label.setText(
                f"{filename} retired to {engine.RECROP_ARCHIVE_DIRNAME}/ — "
                f"{len(children)} crop(s) written, queue empty.")
            self._sync_button_states()
            return True
        self._current_index = min(self._current_index,
                                   len(self._image_files) - 1)
        self._load_current()
        self._status_label.setText(
            f"{filename} retired to {engine.RECROP_ARCHIVE_DIRNAME}/ — "
            f"{len(children)} crop(s) next.")
        self._sync_button_states()
        return True

    def _on_prev(self):
        """Go to the previous field, retiring this one if it was cut up."""
        self.finish_recrop()
        if not self._image_files or self._current_index <= 0:
            return
        self._current_index -= 1
        self._load_current()

    def _on_next(self):
        """Go to the next field, retiring this one if it was cut up."""
        if self.finish_recrop():
            return
        if not self._image_files or self._current_index >= len(self._image_files) - 1:
            return
        self._current_index += 1
        self._load_current()

    def _on_save(self):
        """Write the mask for the field on screen."""
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
        objects = int(np.count_nonzero(np.unique(self._canvas.mask)))
        self._note_curated(self._image_files[self._current_index],
                           n_objects=objects)
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
        """Fill enclosed holes in every object."""
        self._apply_op(engine.fill_holes, "fill_holes")

    def _on_relabel(self):
        """Renumber the objects so the labels are consecutive."""
        self._apply_op(engine.relabel_objects, "relabel")

    def _on_invert(self):
        """Swap object and background."""
        self._apply_op(engine.invert_mask, "invert")

    def _on_remove_small(self):
        """Delete objects below the minimum area."""
        area = int(self._min_area.value())
        self._apply_op(lambda m: engine.remove_small_objects(m, area),
                        "remove_small", min_area=area)

    def _on_clear_mask(self):
        """Throw the whole mask away, after confirming."""
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
        """Snapshot the mask before a stroke mutates it in place.

        BRUSH STROKES EDIT IN PLACE, so undo has nothing to go back to unless
        the state is captured at the START of the stroke rather than after it.
        """
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

    def _sync_button_states(self):
        """Enable each control only when it has something to act on."""
        has_files = bool(self._image_files)
        editable = has_files and not self._loading
        for b in (self._btn_prev, self._btn_next, self._btn_save,
                   self._btn_filter, self._btn_otsu, self._btn_magnifier,
                   *self._mode_buttons.values()):
            b.setEnabled(editable)
