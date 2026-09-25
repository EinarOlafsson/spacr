"""Interactive editing and curation of segmentation masks.

Make Masks is filed under **Tools** on the home screen and is the manual
half of segmentation: it corrects masks a model got wrong, one field at a
time, and its masthead opens the Cellpose workflows that produced them.
:class:`MakeMasksScreen` loads images and labelled masks from
``<folder>/masks`` and saves edited labels as ``uint16`` TIFF files.

THE TEN TOOLS, in :data:`TOOL_MODES` order, because this vocabulary is what
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

**Ruler** is the tenth. Drag between image pixels to read a length without
editing labels. Zoom and pan preserve it; right-click with Ruler selected
clears it. The same ruler is available on the paired live preview canvases.

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
operations that are not gestures -- object filling and relabeling, swapping
object and background, size filtering and Otsu detection, with undo and redo
over all of them -- alongside the brush, wand and display controls, and can
be hidden to return its width to the canvas.

Image inversion and swapping object and background are separate operations.
The first changes the image seen by the detector; the second changes labels.

**Invert image**, in the Display category, is THE inversion. It draws the
field as its own negative and hands the detectors that same negative: both
detect buttons and the live magnifier segment
:meth:`MakeMasksScreen._detector_image`, so a threshold written for bright
objects takes dark ones, and a banner above the image says so for as long as
it is on. The corner readout's pixel intensity follows the inverted image.
Object mean intensities and the object filter still read the loaded pixels.
Saving writes label masks without changing the source image.

IT USED TO BE TWO SWITCHES AND THEY COULD DISAGREE. A display-only "Invert
image" drew a negative and left detection untouched; "Invert for detection"
inverted what the detectors read and drew nothing. A curator could hold
either without the other, and the first one's status line had to contradict
the second one's banner. They are one switch because the reason for the
switch is one intention: inverting a field of dark objects so that Otsu and
the magnifier can detect them. ``_cp_invert`` is now the same widget object as
``_invert_display`` under its old name, so the two cannot come apart.

**Swap object and background**, in Object operations, is the old "Invert
mask" under the name that describes it -- it flips the LABEL image, which on
an ordinary field leaves one object covering the frame. It is not
a picture invert and is deliberately not called one.

THE ARITHMETIC. :func:`~spacr.qt.mask_engine.invert_normalized` normalises
the field to 0..1 on its own range and takes ``1 - v``, and the
normalisation is not incidental: the Otsu correction is a
MULTIPLIER on an absolute level, so normalising first is what makes one
correction value mean the same thing on the next image.
:func:`~spacr.qt.mask_engine.invert_intensity` (dtype complement) and
:func:`~spacr.qt.mask_engine.invert_for_detection` (reflection about the
field's range) are what this screen used before and are kept for other
callers. :func:`~spacr.qt.mask_engine.invert_mask` flips labels.

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
import math
import os
import re
import threading
import time
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
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
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
    QComboBox,
    QDialog,
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
from .. import cpu_modes
from .. import detect_chain
from .. import iconset
from .. import mask_engine as engine
from .. import organelle_modes
from .. import prefs
from .. import wand_rescue
from ..hidpi import follow_device_ratio, logical_size, scaled_for
from ..theme import (SPACING, active_palette, block_surface,
                     ensure_widget_qss_applied, mark_surface,
                     register_widget_qss)
from ..widgets import Card, Divider, EmptyState
from ..widgets.fold_strip import FoldStrip
from ..widgets.section import Section
from ..widgets.toggle import Toggle
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.make_masks")

#: The registry key this screen answers to.
APP_KEY = "make_masks"

#: The masthead's name, matching the registry row so the page and the tile
#: that opens it say the same thing. The masthead carries no one-line
#: description beside it, so the name, the instruction under it and the
#: fold strip are the whole row.
HEADER_TITLE = "Make Masks"
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
MODE_RULER = "ruler"

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
    (MODE_RULER,        "Ruler",        "measure"),
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

#: Pixels between the settings, on the left, and the image: the splitter's
#: handle, so the gap is also where the settings are dragged wider.
SETTINGS_GAP = 12

#: Width of the shortcut list beside the views, in pixels. FIXED, so every
#: pixel a wider window gives the right-hand pane goes to the image; the list
#: is a dozen short lines and does not want the room. Wide enough that the
#: longest keys -- ``Magnifier, whole image: right`` -- wrap onto two lines
#: and no line of prose breaks after one word, which is what a narrower
#: column was doing when this was measured on a rendered screen.
SHORTCUTS_WIDTH = 230

RESTORATION_SETTLE_MS = 400

#: Where the settings panel's folded categories are remembered, as the titles
#: folded away -- :func:`spacr.qt.preferences.get_section_layout` keyed by
#: this name.
_SETTINGS_LAYOUT_KEY = "make_masks/settings"

#: Settings categories this panel has renamed, old title to new. The stored
#: layout is a list of TITLES, so a user who folded the old one away would
#: find it open again after a rename and would have to fold it a second time;
#: reading the stored list through this keeps their arrangement.
#: Cellpose-SAM became Object detection and Auto-filter objects became the
#: Filter category; item 473 then folded Otsu, Object detection and its own
#: Detection methods into ONE "Detection method" category, because they
#: were three categories answering one question -- what finds the objects
#: -- and only one of them was ever being read.
_RENAMED_CATEGORIES = {"Cellpose-SAM": "Detection method",
                       "Object detection": "Detection method",
                       "Otsu": "Detection method",
                       "Detection methods": "Detection method",
                       "Auto-filter objects": "Filter"}

#: The two inks this screen cannot take from the shipped stylesheet: the
#: Filter category's removal ledger, which is RED because it lists what was
#: deleted, and the Invert warning, which is a warning and not prose.
#: Registered rather than written inline, so both follow the user's theme --
#: a colour set on the widget at build time is the colour it keeps when the
#: theme changes under it.
MAKE_MASKS_QSS_NAME = "MakeMasksInk"

#: The Filter category's removal ledger, by object name.
FILTER_LOG_NAME = "MakeMasksFilterLog"

#: The banner that says the detectors are reading an inverted image.
INVERT_WARNING_NAME = "MakeMasksInvertWarning"

#: What that banner says. It reminds the curator that masks are generated
#: from the inverted image, and it is NOT INSIDE THE SETTINGS PANEL: the Settings toggle hides that
#: panel to give the image the width, and the magnifier goes on inverting
#: while it is hidden. It sits between the tool row and the image, where
#: nothing can fold it away.
INVERT_WARNING_TEXT = (
    "Invert image is on: the picture and detection use the INVERTED image. "
    "Hover pixel intensity follows the inversion; object mean intensity "
    "and Filter thresholds use the original loaded values. "
    "The loaded image data and existing mask are unchanged.")

#: How many removal rows the Filter category's ledger shows before it
#: scrolls. The ledger has one row per removed object, and a
#: filter tightened too far removes hundreds; a box that grew with them
#: would push every other category off the panel, so it is a fixed six rows
#: with the rest a scroll away.
FILTER_LOG_ROWS = 6

#: Gaussian sigma the Otsu mode smooths with before it cuts, and what the
#: Otsu category's Smoothing box starts at. It is
#: :data:`spacr.qt.mask_engine._CLASSICAL_SMOOTHING` -- the value the
#: magnifier's threshold mode has always used -- written here so the panel
#: can start on it without importing a private name.
OTSU_SMOOTHING = 1.0

#: Where the local-threshold window starts, in pixels. Comfortably
#: larger than a cell at the magnifications this screen is used at and far
#: smaller than the scale illumination falls off over, which is the band a
#: local threshold has to sit in to be worth switching on. It is odd because
#: the window is centred on the pixel it judges.
OTSU_LOCAL_WINDOW = 51

#: How many bars the Otsu histogram preview draws. 256 is what a reader can
#: tell apart at the width the dialog opens at, and enough that a 16-bit
#: field's two populations are two humps rather than one.
OTSU_HISTOGRAM_BINS = 256

#: The shortcut list beside the Mask / Cell probability / Flows
#: views, as ``(keys, what it does)``. ONE TERSE LINE EACH: the panel is read at a glance between strokes, and a paragraph
#: there would be read once and then never again. Every line is a gesture
#: this module actually implements -- :data:`PAN_MODIFIERS`,
#: :meth:`_MaskCanvas.wheelEvent`, :meth:`_MaskCanvas.mousePressEvent` and
#: :meth:`MakeMasksScreen._install_shortcuts` are where they live -- so the
#: panel is checked against the code by
#: ``tests/qt/test_make_masks_shortcuts_otsu_and_object_edits.py`` rather
#: than believed.
SHORTCUT_HINTS = (
    ("Shift or Alt + drag", "Pan, from any tool"),
    ("Wheel", "Zoom about the cursor"),
    ("Right button", "Sweep away the objects it passes"),
    ("Ctrl + left click", "Split the object at its waist"),
    ("Ctrl + right click", "Remove the object under the cursor"),
    ("Left / Right arrows", "Previous / next field"),
    ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
    ("Ctrl+S", "Save the mask"),
    ("Esc", "Reset the zoom"),
    ("B E W D V Z R", "Brush, erase, wand, draw, divide, zoom, recrop"),
    ("M", "Live magnifier"),
    ("Magnifier: wheel", "Box zoom"),
    ("Magnifier: Shift + wheel", "Box size"),
    ("Ctrl+L+right click", "Lock / unlock box"),
    ("Magnifier: drag", "Add the objects it passes over"),
    ("Magnifier, whole image: right", "Remove the object under it"),
)


def _make_masks_qss(palette, opacity) -> str:
    """This screen's two coloured inks, for one palette.

    Registered through :func:`spacr.qt.theme.register_widget_qss`, so the
    colours follow the user's theme without a line in ``theme.py`` and
    without either widget holding a colour of its own.

    The removal ledger is a text box and takes ``surface_alt`` under its red
    rows, the fill every other boxed control on this panel stands on, put
    through the page opacity the same way. The warning is a label over the
    page and stays transparent: a filled strip across the top of the image
    would be a second surface stacked on the one behind it.

    :param palette: the theme palette, surfaces already rendered through the
        page opacity.
    :param opacity: the user's page-opacity preference, passed through.
    """
    return f"""
QPlainTextEdit#{FILTER_LOG_NAME} {{
    color: {palette['error']};
    background: {block_surface('surface_alt', palette.get('theme'), opacity)};
    border: 1px solid {palette['border_soft']};
}}
QLabel#{INVERT_WARNING_NAME} {{
    color: {palette['warning']};
    background: transparent;
}}
"""


register_widget_qss(MAKE_MASKS_QSS_NAME, _make_masks_qss, replace=True)


class _MaskLoadWorker(QThread):
    """Decode one image/mask pair without blocking Qt's main thread."""

    def __init__(self, folder: str, filename: str, token: int, parent=None,
                 layout: Optional[dict] = None):
        """Load one image and its mask off the GUI thread.

        :param folder: the folder holding the pair.
        :param filename: the image to load; the mask is found beside it.
        :param token: identifies the request this worker answers. The screen
            compares it on completion, so a load the user has already
            navigated away from is discarded rather than drawn over the
            image now on screen.
        :param parent: parent object.
        :param layout: extra keywords for
            :func:`spacr.qt.mask_engine.load_image_and_mask` -- the sibling
            layout's ``masks_dir`` -- or ``None`` for the nested layout.

        The result and the original exception are both kept as attributes
        rather than raised, because a thread that raises loses the traceback
        the screen needs to say what failed.
        """
        super().__init__(parent)
        self.folder = folder
        self.filename = filename
        self.token = token
        self.layout = dict(layout or {})
        self.result = None
        self.error: Optional[Exception] = None

    def run(self) -> None:
        """Load the pair, retaining either the result or original exception."""
        try:
            self.result = engine.load_image_and_mask(
                self.folder, self.filename, **self.layout
            )
        except Exception as exc:
            self.error = exc
            LOG.exception(
                "Failed to load mask source %s from %s",
                self.filename,
                self.folder,
            )


class _StatusLabel(QLabel):
    """The corner readout: one short line here, every line in the console.

    Item 507. About a hundred places in this screen set the corner's text,
    and some of them set a whole failure -- a backend's message and the
    last forty lines it printed -- which piled up in the bottom right. The
    corner now shows the first line, cut short, with the whole text as its
    tooltip, and :attr:`said` hands every new text to the screen, which
    puts it in its console.

    :ivar said: ``(text,)``, each time the text changes to something new.
    """

    said = Signal(str)

    LIMIT = 160

    def __init__(self, text: str = "", parent=None):
        """Start with ``text``, which is not reported."""
        super().__init__(parent)
        self._full = ""
        self._quiet = False
        self._show(text)

    def text(self) -> str:
        """The whole text last set, not the shortened line shown."""
        return self._full

    def setText(self, text: str) -> None:
        """Show ``text``'s first line and report the whole of it once."""
        text = str(text or "")
        changed = text != self._full
        self._show(text)
        if changed and text.strip() and not self._quiet:
            self.said.emit(text)

    def set_quietly(self, text: str) -> None:
        """Show ``text`` without reporting it; its sender already did."""
        self._quiet = True
        try:
            self.setText(text)
        finally:
            self._quiet = False

    def _show(self, text: str) -> None:
        """Put the first line, shortened, on screen; the rest in the tooltip."""
        self._full = text
        lines = text.strip().splitlines()
        line = lines[0] if lines else ""
        if len(line) > self.LIMIT:
            line = line[:self.LIMIT - 1].rstrip() + "…"
        elif len(lines) > 1:
            line = line.rstrip() + " …"
        super().setText(line)
        self.setToolTip(text if line != text else "")


class _MasksConsole(QWidget):
    """Make Masks' console: every status, progress, warning and failure line.

    Item 507. It sits under the shortcut list, right of the image, in a
    section that folds (item 471's :class:`FoldSection`). It is a
    :class:`~spacr.qt.widgets.console_panel.ConsolePanel` without the chat
    and without the application-wide log, so it carries this screen's own
    lines, and under it ONE :class:`~spacr.qt.widgets.eliding.ProgressLine`
    for the task that is running now: a task that reports a hundred steps
    rewrites that line a hundred times rather than stacking a hundred lines.
    When the task ends, its last words go into the scrollback and the line
    hides.

    A line identical to the one before it is not written again, so a
    message repeated on every mouse move reads once.

    :ivar console: the scrollback.
    :ivar progress: the in-place progress line; hidden while nothing runs.
    """

    _relay = Signal(str, str)

    PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")

    def __init__(self, parent=None):
        """Build the scrollback and the hidden progress line."""
        super().__init__(parent)
        from ..widgets.console_panel import ConsolePanel
        from ..widgets.eliding import ProgressLine

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["xs"])
        self.console = ConsolePanel(active_app_label="",
                                    follow_log=False, chat=False)
        layout.addWidget(self.console, 1)
        self.progress = ProgressLine(self, detail=True, count_below=True)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self._last = None
        self._relay.connect(self.say)

    def say(self, text: str, kind: str = "info") -> None:
        """Write one line.

        :param text: what to say; blank is ignored.
        :param kind: ``progress`` rewrites the progress line; ``info``,
            ``warning`` and ``error`` go into the scrollback, in the
            console's colours for each, and end any progress shown.
        """
        if QThread.currentThread() is not self.thread():
            self._relay.emit(str(text or ""), str(kind or "info"))
            return
        text = str(text or "").rstrip()
        if not text.strip():
            return
        if kind == "progress":
            self.show_progress(text)
            return
        self.progress.setVisible(False)
        if (text, kind) == self._last:
            return
        self._last = (text, kind)
        if kind == "error":
            self.console.append_error(text)
        elif kind == "warning":
            self.console.append_warning(text)
        else:
            self.console.append_stdout(text + "\n")

    def show_progress(self, text: str) -> None:
        """Rewrite the one progress line with ``text``.

        A percentage in the text moves the bar; without one the bar is busy.
        """
        found = self.PERCENT.findall(text)
        if found:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(min(100.0, float(found[-1]))))
        else:
            self.progress.setRange(0, 0)
        self.progress.set_detail(text)
        self.progress.setToolTip(text)
        self.progress.setVisible(True)

    def progress_text(self) -> str:
        """What the progress line says, or ``''`` while it is hidden."""
        if not self.progress.isVisibleTo(self):
            return ""
        return self.progress.toolTip()

    def text(self) -> str:
        """The scrollback as plain text."""
        return self.console.as_text()


class _MethodGroup(QWidget):
    """One family of detection settings inside the Detection method category.

    The category holds four of these -- the threshold family's, Cellpose's,
    the organelle methods' and the propagation's -- and shows the one the
    chosen mode reads. A GROUP AND NOT A ROW RULE, because these families
    are not all forms: the threshold group has toggles, a histogram button
    and two nested form layouts, and hiding those one at a time would be a
    list of widget names that goes stale the day one is added.

    It carries ``body_layout`` so the builders that used to fill a
    :class:`~spacr.qt.widgets.section.Section` fill one of these instead,
    unchanged.

    :param parent: parent widget; ownership only.
    """

    def __init__(self, parent=None):
        """Build an empty group with a vertical body layout."""
        super().__init__(parent)
        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(SPACING["sm"])


class _EnhanceRequest(NamedTuple):
    """One whole-field enhanced picture to build, off the GUI thread.

    ``key`` is what :class:`_NewestRequestWorker` matches two requests by:
    the base array and the chain, which together are the whole of what the
    answer depends on.
    """

    key: tuple
    image: np.ndarray
    chain: Any
    cancelled: Any = None


class _CompareRequest(NamedTuple):
    """An immutable comparison snapshot with cooperative result cancellation."""

    key: tuple
    image: np.ndarray
    box: tuple
    chain: Any
    normalized: bool
    percentiles: tuple
    cancelled: Any


def _compare_picture_for(request):
    """Prepare the detector's input on a worker, stretching before cropping."""
    if request.cancelled.is_set():
        return None
    base = request.image
    if request.normalized:
        base = engine.normalize_for_detection(base, *request.percentiles)
    if request.cancelled.is_set():
        return None
    x0, y0, x1, y1 = request.box
    result = detect_chain.prepare(base[y0:y1, x0:x1], request.chain,
                                  cancel=request.cancelled)
    return None if request.cancelled.is_set() else result


class _EnhancedImage(NamedTuple):
    """Scientific intensities and their separately scaled display picture."""

    prepared: np.ndarray
    picture: np.ndarray


def _enhanced_picture_for(request: _EnhanceRequest) -> Optional[_EnhancedImage]:
    """The enhanced field as a drawable picture. ON THE WORKER THREAD.

    The chain in float (:func:`spacr.qt.detect_chain.prepare`), then back
    onto the loaded field's own unsigned range so
    :func:`_box_grey_table` can index it and :func:`refresh` can stretch
    it. A float field has no integer range to return to and is handed back
    as it is.

    :param request: the field and the chain.
    :returns: the picture, or None when there is nothing to draw.
    """
    base = request.image
    if base is None:
        return None
    out = detect_chain.prepare(base, request.chain, cancel=request.cancelled)
    dtype = np.dtype(base.dtype)
    if out is base or dtype.kind != "u":
        return _EnhancedImage(out, out)
    low = float(np.min(out)) if out.size else 0.0
    span = (float(np.max(out)) - low) if out.size else 1.0
    top = float(np.iinfo(dtype).max)
    scaled = (np.asarray(out, dtype=np.float64) - low) / (span or 1.0) * top
    return _EnhancedImage(out, np.clip(scaled, 0, top).astype(dtype))


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
    #: Something the user did needs a sentence on the status line. Emitted
    #: by the Ctrl+click edits, which are the canvas's
    #: only gestures that can decline to do anything for a reason worth
    #: telling: a click on background, and an object with no waist to cut.
    #: A gesture that did nothing and said nothing reads as a broken
    #: shortcut.
    status = Signal(str)
    #: A recrop box was dragged, in FULL-image pixels: (x0, y0, x1, y1).
    #: The canvas neither writes it nor judges it — the box may be too
    #: small, or a re-draw of one already cut — because what it becomes is
    #: two files and a queue position, and none of that is a canvas's job.
    recrop_requested = Signal(int, int, int, int)
    #: A whole-field enhanced picture finished on the worker thread, as
    #: ``(the base array it was made from, the chain, the picture)``.
    #: Connected to :meth:`_take_enhanced` in ``__init__``, which is what
    #: carries it from the worker thread to this one.
    enhanced_ready = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build an empty canvas: no image, no mask, no stroke in progress."""
        super().__init__(parent)
        from ..widgets.image_ruler import ImageRuler

        self.ruler = ImageRuler(self)
        self.ruler.changed.connect(self.update)
        #: What the corner readout says about the pixel under the mouse, or
        #: None while the mouse is off the image.
        self.readout: Optional[engine.PixelReadout] = None
        self._lookup: Optional[engine.ObjectLookup] = None
        self._lookup_mask: Optional[np.ndarray] = None
        self._lookup_image: Optional[np.ndarray] = None
        self._lookup_dirty = True
        self._readout_pos: Optional[QPointF] = None
        self._readout_queued = False
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.mode: str = MODE_NONE
        self.brush_radius: int = 10
        self.norm_lo: float = 1.0
        self.norm_hi: float = 99.9
        #: Draw the photographic complement of the image. A VIEW
        #: setting beside the two percentiles, not an edit: :attr:`image`
        #: keeps the pixels that were read off disk, so the readout, the
        #: filter, every detector and the save all see the original numbers
        #: whether this is on or off.
        self.invert_display: bool = False
        self._inverted: Optional[np.ndarray] = None
        self._inverted_of: Optional[np.ndarray] = None
        #: Give the detectors the field as it is DRAWN -- stretched between
        #: the two display percentiles -- rather than as it was loaded.
        #: Off by default, which keeps the stretch a view setting.
        self.detect_on_normalized: bool = False
        self._detection_cache: Optional[tuple] = None
        #: The pre-detection chain the detectors read the field through
        #: (:mod:`spacr.qt.detect_chain`), and whether the canvas DRAWS
        #: what that chain produced instead of the field as loaded. Both
        #: are view-and-detection settings in the same sense the two
        #: percentiles are: :attr:`image` keeps the numbers off disk.
        self.enhance_chain = detect_chain.NO_CHAIN
        self.enhance_display: bool = False
        self._enhanced_cache: Optional[tuple] = None
        self._enhanced_picture: Optional[tuple] = None
        self._enhance_failure = None
        #: The whole-field enhanced picture is built OFF THIS THREAD; see
        #: :meth:`enhanced_picture`. ``_enhance_asked`` is the
        #: ``(base, chain)`` a request is already out for, so a repaint
        #: while one is running does not ask again.
        self._enhance_worker = None
        self._enhance_cancel = threading.Event()
        self._enhance_pending = None
        self._enhance_started = None
        self._enhance_timer = QTimer(self)
        self._enhance_timer.setSingleShot(True)
        self._enhance_timer.setInterval(RESTORATION_SETTLE_MS)
        self._enhance_timer.timeout.connect(self._submit_pending_enhance)
        self._enhance_asked: Optional[tuple] = None
        self.enhanced_ready.connect(self._take_enhanced)
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
        #: The smallest object the screen is willing to keep, in pixels, as
        #: the Min area box holds it. Ctrl + left click reads it for its
        #: seed spacing (:func:`mask_engine.split_object_at`), so the size
        #: that decides what is debris also decides what is too small to be
        #: two things. Set by the screen; 0 leaves the engine's own floor.
        self.split_min_area: int = 0
        #: The BUTTON now down that was pressed with Ctrl, or None. It makes
        #: that press one of the Ctrl+click edits and NOT the start of a drag:
        #: without it the release reaches the magnifier, whose own release
        #: with no stroke open IS a click -- so a Ctrl+click that split one
        #: object would commit every object in the box on the way back up.
        #: It is the button and not a flag so that a SECOND button pressed
        #: while the edit is open can have its own release consumed without
        #: ending the edit early.
        self._ctrl_click = None
        #: Buttons whose PRESS this canvas declined to act on, and whose
        #: release must therefore be declined too. A release that falls
        #: through reaches the generic stroke end at the foot of
        #: :meth:`_MaskCanvas.mouseReleaseEvent`, which would close whatever
        #: OTHER gesture is open and label it a paint.
        self._swallowed: set = set()
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
        self.ruler.clear()
        self.ruler.set_spacing()
        self._gesture_points = []
        self.recrop_boxes = []
        self._inverted = self._inverted_of = None
        self._detection_cache = None
        self._enhanced_cache = self._enhanced_picture = None
        self._enhance_failure = None
        self._enhance_asked = None
        self._enhance_cancel.set()
        self._lookup = self._lookup_mask = self._lookup_image = None
        self.readout = None
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

    def displayed_source(self) -> Optional[np.ndarray]:
        """The intensities this canvas DRAWS, before the contrast stretch.

        :attr:`image` unless :attr:`invert_display` is on, in which case the
        photographic complement of it (:func:`spacr.qt.mask_engine.
        invert_intensity`). The complement is computed here and
        nowhere else, so everything that measures, detects, filters or saves
        goes on reading :attr:`image` and cannot silently be handed inverted
        pixels. A curator can leave Invert on all day and the numbers on
        disk are the numbers the microscope wrote.

        The complement is kept until the image itself changes, because a
        refresh runs on every stroke point and re-subtracting a megapixel
        field per point would be felt on the brush.

        :returns: the array to stretch and draw, or None with no image.
        """
        if self.image is None:
            return None
        if not self.invert_display:
            return self.image
        if self._inverted is None or self._inverted_of is not self.image:
            self._inverted = engine.invert_normalized(self.image)
            self._inverted_of = self.image
        return self._inverted

    def detection_base(self) -> Optional[np.ndarray]:
        """The field the enhancement chain starts from.

        :meth:`displayed_source` -- the loaded field, or its inversion with
        Invert on -- and, with :attr:`detect_on_normalized` on, that array
        stretched between :attr:`norm_lo` and :attr:`norm_hi` exactly as
        :meth:`refresh` stretches it for drawing, so Otsu, Cellpose and the
        magnifier segment the picture the curator is looking at.

        THE STRETCH IS THE CHAIN'S FIRST STAGE and it is here rather than
        in :mod:`spacr.qt.detect_chain` because its two levels are
        percentiles of the WHOLE FIELD: taken inside the magnifier's box
        they would be a different stretch in every box. Everything after it
        is applied to whatever region is being detected, which is the box
        for the magnifier and the field for the detect buttons.

        Cached against the array and the two percentiles: the magnifier asks
        on every mouse move, and a percentile pass over a megapixel field
        per move would be felt.

        :returns: the array, or None with no image.
        """
        base = self.displayed_source()
        if base is None or not self.detect_on_normalized:
            return base
        key = (base, float(self.norm_lo), float(self.norm_hi))
        cached = self._detection_cache
        if cached is not None and cached[0] is base and cached[1:3] == key[1:]:
            return cached[3]
        out = engine.normalize_for_detection(base, self.norm_lo, self.norm_hi)
        self._detection_cache = (base, key[1], key[2], out)
        return out

    def detection_source(self) -> Optional[np.ndarray]:
        """The WHOLE FIELD as a detector on it reads it: base plus the chain.

        What the detect buttons segment and what the active Apply button
        draws. The magnifier does NOT come through here: its box is
        enhanced on the worker thread, region by region, so a heavy step is
        never a frozen window and item 407's progress and Cancel have
        something to cover.

        Cached against the base array and the chain, because drawing the
        enhanced field asks on every repaint.

        :returns: the array, or None with no image.
        """
        base = self.detection_base()
        if base is None:
            return None
        chain = self.enhance_chain or detect_chain.NO_CHAIN
        cached = self._enhanced_cache
        if cached is not None and cached[0] is base and cached[1] == chain:
            return cached[2]
        if chain.psf_operation != 'none' or chain.restoration:
            from ..i18n import tr

            failure = self._enhance_failure
            if failure is not None and failure[0] is base and failure[1] == chain:
                raise ValueError(failure[2])
            self._ask_for_enhanced(base, chain)
            raise ValueError(tr('Image enhancement is updating. Wait for it to finish before detecting objects.'))
        out = detect_chain.prepare(base, chain)
        self._enhanced_cache = (base, chain, out)
        return out

    def enhanced_picture(self) -> Optional[np.ndarray]:
        """The enhanced field as a PICTURE, and never at the cost of a frame.

        The chain works in float and the box's picture is a look-up table
        indexed by an unsigned integer value (:func:`_box_grey_table`), so
        an enhanced field is put back on the loaded field's own integer
        range before anything draws it. The numbers a detector read are the
        float ones; this is the picture of them.

        IT IS BUILT ON A WORKER THREAD AND THIS METHOD NEVER WAITS. A chain
        over a whole field is filters over four megapixels -- a rolling ball
        at the default is about a second, and a curator is free to ask for
        the exact one, which is fifteen. This is called from
        :meth:`refresh`, which runs on every edit, every stroke point and
        every zoom, so a second spent here is a second the window does not
        answer in. The first call for a new field or a new chain starts the
        work and hands back the field as loaded; when the picture arrives
        (:attr:`enhanced_ready`) the canvas refreshes and shows it.

        :returns: the enhanced picture, the field as loaded while one is
            being built, or None with no image.
        """
        base = self.detection_base()
        if base is None:
            return None
        chain = self.enhance_chain or detect_chain.NO_CHAIN
        if not detect_chain.pre_active(chain):
            return base
        cached = self._enhanced_picture
        if cached is not None and cached[0] is base and cached[1] == chain:
            return cached[2]
        failure = self._enhance_failure
        if failure is not None and failure[0] is base and failure[1] == chain:
            return base
        self._ask_for_enhanced(base, chain)
        return base

    def _ask_for_enhanced(self, base: np.ndarray, chain) -> None:
        """Start the enhanced picture for ``(base, chain)`` on the worker.

        One request at a time and the newest wins
        (:class:`_NewestRequestWorker`): dragging the background radius
        makes a request per step, and every one but the last is about a
        picture nobody will see.

        A CHAIN WITH CELLPOSE 3 RESTORATION WAITS TO SETTLE (item 507). One
        restoration of a whole field is tens of seconds on a CPU, and a
        cancelled one still finishes in the worker before the next can
        start, so a request is sent only after the chain has stood still
        for :data:`RESTORATION_SETTLE_MS`; each change in between replaces
        the waiting request rather than sending it.
        """
        asked = self._enhance_asked
        if asked is not None and asked[0] is base and asked[1] == chain:
            return
        self._enhance_asked = (base, chain)
        self._enhance_failure = None
        self._enhance_cancel.set()
        self._enhance_cancel = threading.Event()
        if self._enhance_worker is None:
            self._enhance_worker = _NewestRequestWorker(
                _enhanced_picture_for, self._enhanced_done,
                name="spacr-enhance")
        request = _EnhanceRequest(
            key=(id(base), chain, id(self._enhance_cancel)),
            image=base, chain=chain, cancelled=self._enhance_cancel)
        if getattr(chain, "restoration", False):
            self._enhance_pending = request
            self._enhance_timer.start()
            return
        self._enhance_pending = None
        self._enhance_timer.stop()
        self._enhance_worker.submit(request)

    def _submit_pending_enhance(self) -> None:
        """Send the request that waited for the chain to settle, if still wanted."""
        request, self._enhance_pending = self._enhance_pending, None
        worker = self._enhance_worker
        if (request is None or worker is None
                or request.cancelled is None or request.cancelled.is_set()):
            return
        from ..i18n import tr

        self._enhance_started = (request.image, request.chain, time.monotonic())
        self.status.emit(tr("Cellpose 3 restoration is running on the whole field…"))
        worker.submit(request)

    def _enhanced_done(self, request, result, error) -> None:
        """Deliver the finished picture or exception to Qt from the worker."""
        if request.cancelled is not None and request.cancelled.is_set():
            return
        if error is not None:
            LOG.warning("the enhanced picture could not be built",
                        exc_info=error)
        if error is None and result is None:
            return
        try:
            self.enhanced_ready.emit((request.image, request.chain,
                                     error if error is not None else result))
        except RuntimeError:
            pass

    def _take_enhanced(self, payload) -> None:
        """Keep a finished enhanced picture and draw it, on the GUI thread."""
        from ..i18n import tr

        base, chain, picture = payload
        asked = self._enhance_asked
        if asked is not None and asked[0] is base and asked[1] == chain:
            self._enhance_asked = None
        if self.detection_base() is not base or self.enhance_chain != chain:
            return
        if isinstance(picture, Exception):
            self._enhance_failure = (base, chain, str(picture))
            self._enhance_started = None
            self.status.emit(tr('Image enhancement failed: {error}', error=str(picture)))
            return
        self._enhance_failure = None
        started = self._enhance_started
        if started is not None and started[0] is base and started[1] == chain:
            self._enhance_started = None
            self.status.emit(tr("Cellpose 3 restoration finished in {seconds} s.",
                                seconds=round(time.monotonic() - started[2], 1)))
        if isinstance(picture, _EnhancedImage):
            self._enhanced_cache = (base, chain, picture.prepared)
            picture = picture.picture
        self._enhanced_picture = (base, chain, picture)
        if self.enhance_display:
            self.refresh()

    def close_enhancer(self) -> bool:
        """Stop the enhanced-picture worker; True when none is left running."""
        self._enhance_cancel.set()
        self._enhance_timer.stop()
        self._enhance_pending = None
        worker = self._enhance_worker
        self._enhance_worker = None
        self._enhance_asked = None
        return True if worker is None else worker.close()

    def refresh(self) -> None:
        """Recompose image + mask overlay and repaint the canvas pixmap.

        Every edit ends in a refresh, so the corner readout is re-read after
        one: an object just erased must stop being reported under the mouse.
        """
        self._lookup_dirty = True
        self._schedule_readout()
        if self.image is None or self.mask is None:
            return
        source = (self.enhanced_picture() if self.enhance_display
                  else self.displayed_source())
        img = engine.normalize_uint16(source, self.norm_lo, self.norm_hi)
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
        if (self.magnifier is not None and self.magnifier.enabled
                and event.modifiers() & Qt.ShiftModifier):
            notches = notches or event.angleDelta().x()
            if notches:
                self.magnifier.wheel_size(notches > 0)
                self.update()
            event.accept()
            return
        if not notches:
            return super().wheelEvent(event)
        if self.magnifier is not None and self.magnifier.enabled:
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

    def effective_wand_tolerance(self, source=None) -> float:
        """The tolerance the wand will actually flood with, right now.

        Relative by default: a percentage of this image's own intensity
        range, so the same setting behaves the same on 8-bit and 16-bit
        data. Switching ``wand_relative`` off restores a plain absolute
        value for the case where somebody knows the exact grey-level
        distance they want.

        :param source: optional applied picture; defaults to loaded pixels.
        """
        source = self.image if source is None else source
        if self.wand_relative and source is not None:
            return engine.relative_tolerance(source, self.wand_tol_pct)
        return float(self.wand_tolerance)

    def wand_source(self):
        """Return original pixels with Apply off, or the ready applied picture.

        Classical filters use the display-scaled picture. PSF and deep
        restoration use the detector's float output, including normalized
        model units for restoration; absolute tolerances refer to that output.
        Relative tolerances
        follow its new intensity range. While enhancement is pending or
        failed, return None and explain why; no raw-pixel flood substitutes
        for an applied enhancement. Post-detection morphology and splitting
        remain detector operations, not manual Wand edits.
        """
        from ..i18n import tr

        if not self.enhance_display:
            return self.image
        base = self.detection_base()
        chain = self.enhance_chain or detect_chain.NO_CHAIN
        if base is None:
            return None
        if not detect_chain.pre_active(chain):
            return base
        cached = self._enhanced_picture
        if cached is not None and cached[0] is base and cached[1] == chain:
            if (chain.psf_operation != 'none' or chain.restoration) and self._enhanced_cache is not None:
                return self._enhanced_cache[2]
            return cached[2]
        failure = self._enhance_failure
        if failure is not None and failure[0] is base and failure[1] == chain:
            self.status.emit(tr('Image enhancement failed: {error}', error=failure[2]))
            return None
        self._ask_for_enhanced(base, chain)
        self.status.emit(tr('Image enhancement is updating. Try the Wand again when it finishes.'))
        return None

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
        self._paint_drag()
        self._paint_readout()
        if self.ruler.start is not None:
            painter = QPainter(self)
            self.ruler.paint(painter, lambda x, y: self._image_to_canvas(x + 0.5, y + 0.5))
            painter.end()

    def _paint_drag(self) -> None:
        """Draw the outline, cut or rectangle being dragged, if there is one."""
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
        painter.end()

    def _object_lookup(self) -> Optional[engine.ObjectLookup]:
        """The objects of the mask on screen, rebuilt only when it changed.

        A refresh marks the lookup stale, and a stale one is compared with a
        copy of the mask it was built from before it is rebuilt: a zoom or a
        pan refreshes without changing a label, and rebuilding for those
        would cost a 2048 px field tens of milliseconds per wheel notch.

        :returns: the lookup, or ``None`` with no field open.
        """
        if self.mask is None or self.image is None:
            return None
        lookup = self._lookup
        if lookup is not None and self._lookup_image is self.image and (
                not self._lookup_dirty
                or (self._lookup_mask.shape == self.mask.shape
                    and np.array_equal(self._lookup_mask, self.mask))):
            self._lookup_dirty = False
            return lookup
        try:
            lookup = engine.ObjectLookup(self.mask, self.image,
                                         preserve_ids=getattr(self, 'preserve_ids', False))
        except Exception:                                    # noqa: BLE001
            LOG.debug("the readout could not measure the mask", exc_info=True)
            return None
        self._lookup = lookup
        self._lookup_mask = np.array(self.mask, copy=True)
        self._lookup_image = self.image
        self._lookup_dirty = False
        return lookup

    def update_readout(self, pos=None, *, measure: bool = True):
        """Point the corner readout at the widget position ``pos``.

        :param pos: where the mouse is, in widget coordinates, or ``None``
            when it has left the canvas.
        THE READOUT READS :meth:`detection_base` AND NOT
        :meth:`detection_source`, so it never runs the enhancement chain.
        This method is called on EVERY MOUSE MOVE, and a chain is filters
        over a whole field: pointing the readout at the chain's output made
        choosing a background subtraction freeze the window, because the
        first move after choosing it ran the estimate on the GUI thread.
        What the readout is for is the value under the cursor -- as loaded,
        or stretched when "Detect on the normalized image" is on -- and the
        stretch is exactly what :meth:`detection_base` is.

        :param measure: also report the object under the pixel. False while
            a button is held, because a brush stroke changes the mask on every
            move and the object is re-read when the stroke ends.
        :returns: the readout now shown, or ``None``.
        """
        self._readout_pos = None if pos is None else QPointF(pos)
        readout = None
        spot = (None if pos is None or self.image is None
                else self._canvas_to_image(pos.x(), pos.y()))
        if spot is not None:
            source = self.detection_base()
            lookup = self._object_lookup() if measure else None
            if lookup is not None:
                readout = lookup.at(*spot)
                if readout is not None and (self.invert_display
                                            or self.detect_on_normalized):
                    readout = readout._replace(
                        intensity=self._value_at(source, spot))
            else:
                readout = engine.PixelReadout(
                    spot[0], spot[1], self._value_at(source, spot))
        if readout != self.readout:
            before = self.readout_rect()
            self.readout = readout
            for rect in (before, self.readout_rect()):
                if rect is not None:
                    self.update(rect.adjusted(-2, -2, 2, 2))
        return readout

    @staticmethod
    def _value_at(source, spot) -> float:
        """The intensity at ``spot`` of whichever array is handed in.

        Reading :attr:`image` always would leave the number under the mouse
        unchanged while the picture showed the negative with Invert on --
        and the readout is the instrument a curator checks the inversion
        WITH, so a readout that does not move makes Invert look broken.

        It now reads what is DRAWN, so the number agrees with the picture.
        What is saved and what the Filter measures still read the loaded
        pixels, which is why the caption says which of the two a number is.

        :param source: the array actually on screen.
        :param spot: ``(x, y)`` in image pixels.
        :returns: the value, averaged over channels for a colour field.
        """
        value = np.asarray(source[spot[1], spot[0]], dtype=np.float32)
        return float(value.mean())

    def _schedule_readout(self) -> None:
        """Re-read the readout once the current event has been handled.

        The canvas is the timer's context object, so a canvas deleted before
        the timer fires cancels it rather than being called after it has gone.
        """
        if self._readout_pos is None or self._readout_queued:
            return
        self._readout_queued = True
        QTimer.singleShot(0, self, self._reread_readout)

    def _reread_readout(self) -> None:
        """Measure the pixel the mouse is resting on again, after an edit."""
        self._readout_queued = False
        if self._readout_pos is None or QApplication.mouseButtons() \
                != Qt.NoButton:
            return
        self.update_readout(self._readout_pos)

    def readout_text(self) -> str:
        """The corner readout as it is painted, or ``""`` when there is none.

        The first line is the pixel -- its position and raw intensity -- and
        the second, over an object, is that object's id, area and mean
        intensity, in the units and to the decimals the filter's own boxes
        use, so a value read here can be typed there.
        """
        from ..i18n import tr

        readout = self.readout
        if readout is None:
            return ""
        if self.detect_on_normalized:
            lines = [tr("x {x}, y {y}   intensity {value} (normalized, as detected)",
                        x=readout.x, y=readout.y,
                        value=_readout_number(readout.intensity))]
        elif self.invert_display:
            lines = [tr("x {x}, y {y}   intensity {value} (inverted)",
                        x=readout.x, y=readout.y,
                        value=_readout_number(readout.intensity))]
        else:
            lines = [tr("x {x}, y {y}   intensity {value}", x=readout.x,
                        y=readout.y,
                        value=_readout_number(readout.intensity))]
        if readout.label:
            if self.invert_display or self.detect_on_normalized:
                lines.append(tr(
                    "Object {label}   area {area} px   mean intensity "
                    "{mean} (as loaded)",
                    label=readout.label, area=readout.area,
                    mean=_readout_number(readout.mean_intensity, decimals=2)))
            else:
                lines.append(tr(
                    "Object {label}   area {area} px   mean intensity {mean}",
                    label=readout.label, area=readout.area,
                    mean=_readout_number(readout.mean_intensity, decimals=2)))
        return "\n".join(lines)

    def readout_rect(self) -> Optional[QRect]:
        """Where the readout is painted: the top-left corner of the image.

        :returns: the box in widget coordinates, or ``None`` when there is
            nothing to show.
        """
        text = self.readout_text()
        rendered = self.pixmap()
        if not text or rendered is None or rendered.isNull():
            return None
        shown = logical_size(rendered)
        left = max(0, (self.width() - shown.width()) // 2)
        top = max(0, (self.height() - shown.height()) // 2)
        margin = SPACING["xs"]
        metrics = self.fontMetrics()
        bounds = metrics.boundingRect(QRect(0, 0, 10_000, 10_000),
                                      int(Qt.AlignLeft | Qt.AlignTop), text)
        return QRect(left + margin, top + margin,
                     bounds.width() + 2 * SPACING["sm"],
                     bounds.height() + 2 * SPACING["xs"])

    def _paint_readout(self) -> None:
        """Paint the readout in the image's top-left corner, over everything.

        On a translucent plate of the page colour, so it reads over a bright
        field and a dark one alike.
        """
        rect = self.readout_rect()
        if rect is None:
            return
        palette = active_palette()
        plate = QColor(palette["bg"])
        plate.setAlpha(215)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(plate)
            painter.drawRoundedRect(QRectF(rect), 4, 4)
            painter.setPen(QPen(QColor(palette["fg"])))
            painter.drawText(
                rect.adjusted(SPACING["sm"], SPACING["xs"], 0, 0),
                int(Qt.AlignLeft | Qt.AlignTop), self.readout_text())
        finally:
            painter.end()

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

    def _ctrl_edit_at(self, pt, *, split: bool) -> bool:
        """Split or remove the object under ``pt``.

        Ctrl + left click splits the hovered object; Ctrl + right click
        removes it.

        BOTH ARE ONE STROKE, so each is one undo step and one ledger entry,
        like every other edit on this canvas. The stroke is opened only once
        something is actually going to change: a Ctrl+click on background,
        or on an object with no waist to cut, leaves no empty undo step to
        step back through and says why on the status line instead.

        :param pt: the image pixel clicked, or None when the click was off
            the image.
        :param split: True for the left button's split, False for the right
            button's removal.
        :returns: whether the mask changed.
        """
        if pt is None or self.mask is None:
            return False
        x, y = pt
        height, width = self.mask.shape[:2]
        if not (0 <= y < height and 0 <= x < width):
            return False
        target = int(self.mask[y, x])
        if target <= 0:
            self.status.emit(
                "Ctrl+click acts on an object — there is none under the "
                "cursor.")
            return False
        if not split:
            self._emit_stroke_start()
            self.mask = engine.erase_object_at(self.mask, x, y)
            self.refresh()
            self._emit_stroke_end(kind="delete", target=target,
                                   gesture="ctrl_click")
            self.status.emit(
                f"Removed object {target} — Ctrl+Z to undo")
            return True
        out, new_ids = engine.split_object_at(
            self.mask, x, y, min_area=int(self.split_min_area))
        if not new_ids:
            self.status.emit(
                f"Object {target} has one centre, so there is no waist to "
                "split it at. The Divide tool cuts along a line you draw.")
            return False
        self._emit_stroke_start()
        self.mask = out
        self.refresh()
        self._emit_stroke_end(kind="split", target=target,
                               gesture="ctrl_click", into=list(new_ids),
                               n_objects=len(new_ids) + 1)
        made = ", ".join(str(new) for new in new_ids)
        self.status.emit(
            f"Split object {target} into {len(new_ids) + 1} — new id(s) "
            f"{made}. Ctrl+Z to undo")
        return True

    def mousePressEvent(self, event):
        """Dispatch a click to the current tool (brush/erase/wand/zoom/…).

        Three gestures are checked before the tool, because they work from
        *any* tool: Ctrl + left splits and Ctrl + right removes the object
        under the cursor, the right button sweep-deletes,
        and Shift/Alt + left pans. All of them are things you want mid-edit
        without putting the brush down and picking it up again. With the
        magnifier on in whole-image scope the right button removes the one
        object under it instead of sweeping.

        CTRL IS TESTED FIRST, before the magnifier and before the pan, or
        the two edits would exist only while the magnifier was off: its own
        press handler takes every left click, and its whole-image mode takes
        every right one.

        A CTRL EDIT OWNS THE MOUSE, and only starts when it can: the whole
        branch is behind ``event.buttons() == event.button()``, so it runs
        only when nothing else is already down. Pressing Ctrl+left in the
        middle of a right-button sweep would otherwise end the SWEEP's
        stroke and label it a split, and the sweep's own release would then
        be swallowed as this gesture's -- leaving the sweep open with no
        ledger entry.

        SUCH A PRESS IS SWALLOWED RATHER THAN HANDED ON. Falling through is
        not the safe default it looks like: the rest of this handler would
        read a Ctrl+left as an ordinary left press and open a PAINT stroke,
        so a chord meant to split an object would paint over one instead.
        Ctrl with another button already down does nothing at all, and so
        does a second button pressed while a Ctrl edit is open -- whose
        release is then consumed without ending that edit.

        ITS RELEASE IS SWALLOWED WITH IT, through :attr:`_swallowed`. A
        release that falls through reaches the generic stroke end at the
        foot of :meth:`mouseReleaseEvent`, which closes whatever stroke is
        open -- the SWEEP's -- and labels it a paint, which is the whole
        damage this guard exists to stop.

        :attr:`_ctrl_click` is still rewritten by every press that arrives
        alone, so a release that never came -- a grab lost to a dialog --
        cannot leave it standing and swallow the next drag.
        """
        if self.mask is None:
            return super().mousePressEvent(event)

        if self.ruler.handle(event, lambda p: self._canvas_to_image(p.x(), p.y())):
            return

        magnifier = self.magnifier
        if (magnifier is not None and magnifier.enabled
                and magnifier._lock_key_down
                and event.modifiers() & Qt.ControlModifier
                and event.button() == Qt.RightButton):
            if event.buttons() == event.button():
                self._swallowed.clear()
                self._ctrl_click = None
                if not magnifier.locked:
                    magnifier.hover(event.position())
                magnifier.set_locked(not magnifier.locked)
            self._swallowed.add(event.button())
            event.accept()
            return

        if event.buttons() == event.button():
            self._swallowed.clear()
            self._ctrl_click = (
                event.button()
                if (event.modifiers() & Qt.ControlModifier
                    and event.button() in (Qt.LeftButton, Qt.RightButton))
                else None)
            if self._ctrl_click is not None:
                self._ctrl_edit_at(
                    self._canvas_to_image(event.position().x(),
                                           event.position().y()),
                    split=event.button() == Qt.LeftButton)
                self.update()
                return
        elif (self._ctrl_click is not None
              or event.modifiers() & Qt.ControlModifier):
            self._swallowed.add(event.button())
            return

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

        wand_input = None
        if self.mode in (MODE_WAND_ADD, MODE_WAND_ERASE):
            wand_input = self.wand_source()
            if wand_input is None:
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
            tolerance = self.effective_wand_tolerance(wand_input)
            self.mask, report = wand_rescue.magic_wand(
                wand_input, self.mask, pt[0], pt[1],
                tolerance, self.wand_max_pixels, action=action,
                **self.wand_rescue_settings(),
            )
            self.refresh()
            self._emit_stroke_end(
                kind="wand", target=(255 if action == "add" else 0),
                action=action, tolerance=round(float(tolerance), 3),
                relative=bool(self.wand_relative), **report,
                input_kind='enhanced_picture' if self.enhance_display else 'as_loaded',
                invert=bool(self.enhance_display and self.invert_display),
                normalization_percentiles=([float(self.norm_lo), float(self.norm_hi)]
                    if self.enhance_display and self.detect_on_normalized else None),
                **detect_chain.provenance(
                    self.enhance_chain._replace(morphology='none', split=False)
                    if self.enhance_display else detect_chain.NO_CHAIN,
                    percentile_stretch=self.enhance_display and self.detect_on_normalized),
            )
            return

        radius = self._mask_radius_for_brush()
        value = 255 if self.mode == MODE_BRUSH else 0
        engine.paint_disk(self.mask, pt[0], pt[1], radius, value)
        self._last_pt = QPoint(*pt)
        self.refresh()

    def mouseMoveEvent(self, event):
        """Move the readout; extend a sweep, a pan, a stroke or a zoom drag.

        A move while a Ctrl+click is still down moves the readout and
        nothing else: that click was one edit and is finished, so dragging
        away from it must not turn into a magnifier stroke or a sweep.
        """
        if self.mask is None:
            return
        if self.ruler.handle(event, lambda p: self._canvas_to_image(p.x(), p.y())):
            return
        self.update_readout(event.position(),
                            measure=event.buttons() == Qt.NoButton)
        if self._ctrl_click is not None:
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
        """Close a sweep or pan, commit a zoom rect, or finalize a stroke.

        The readout is re-measured once the release has been handled, since a
        held button kept it to the pixel while the mask was changing.

        A Ctrl+click is over the moment it was pressed, so its release is
        consumed here rather than handed on — see :attr:`_ctrl_click`. Any
        release arriving while one is open is consumed, but only the button
        that OPENED it closes it: a second button pressed meanwhile started
        nothing, so its release must end nothing. A button whose press was
        declined outright (:attr:`_swallowed`) is dropped for the same
        reason, before the generic stroke end below can close somebody
        else's stroke with it.
        """
        if self.ruler.handle(event, lambda p: self._canvas_to_image(p.x(), p.y())):
            return
        self._schedule_readout()
        if self._ctrl_click is not None:
            if event.button() == self._ctrl_click:
                self._ctrl_click = None
            return
        if event.button() in self._swallowed:
            self._swallowed.discard(event.button())
            return
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
        """Put the magnifier's box and the readout away as the mouse leaves."""
        self.update_readout(None)
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
    "Run Object detection to see the cell-probability map\n"
    "and the flow field for this field."
)


def _readout_number(value, decimals: int = 0) -> str:
    """A number as the corner readout writes it.

    A whole number, which every pixel of an integer image is, is written
    without a decimal point; anything else to ``decimals`` places, which for
    a mean is the two the filter's intensity boxes take.

    :param value: the number, or ``None``.
    :param decimals: places for a number that is not whole.
    :returns: the text.
    """
    if value is None:
        return ""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.{max(1, int(decimals))}f}"


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

    :param array: any numeric array; it is read as float32, and an empty array
        gives an empty uint8 array.
    :param lower_pct: percentile mapped to 0.
    :param upper_pct: percentile mapped to 255.
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

    :param cellprob: Cellpose's cell-probability map, a 2-D float array; it is
        percentile-stretched with :func:`stretch_to_uint8` before colouring.
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

    :param flow: either Cellpose's RGB flow picture, an array of shape (H, W, 3
        or more), or the raw vector field of shape (2, H, W); ``None`` or any
        other shape gives ``None``.
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


#: The item-data role marking a Model box row that came from the model zoo.
_ZOO_ROLE = int(Qt.UserRole) + 17

#: Item-data role holding the zoo entry of a Model row that is listed but not
#: downloaded. Such a row stores no path, so it can never be what a detect
#: run loads; choosing it starts the download instead.
_ZOO_PENDING_ROLE = int(Qt.UserRole) + 18


def _zoo_cellpose_models() -> List[tuple]:
    """``(key, path or None, entry)`` for every Cellpose model in the zoo.

    ``path`` is where the model is on this machine -- the entry's own path,
    or its file in the folder the Model zoo picker downloads into -- and None
    for one not downloaded. ``entry`` is the zoo's own record, which is what
    a download starts from. Read without waiting on the network (the
    community rows come from the zoo's cache), and never raises: a zoo that
    cannot be read leaves the Model box with the Cellpose installed here.

    OFFERS WHAT THE ZOO OFFERS, AND NOTHING ELSE. The source headings the
    Model zoo picker carries are a preference, not a property of
    one dialog: a user who folded bioimage.io away in the picker has said
    they do not want those models, and a Mode box that listed them anyway
    would be the one place that ignored them. So the same persisted headings
    filter this list.
    """
    try:
        from ... import model_zoo
        from ..widgets.model_zoo_picker import (remembered_model_dir,
                                                remembered_sources)

        entries = model_zoo.catalogue(remote=True, block=False)
        folder = remembered_model_dir()
        sources = set(remembered_sources())
    except Exception:                                        # noqa: BLE001
        LOG.debug("the model zoo could not be read", exc_info=True)
        return []
    found = []
    for entry in entries:
        if getattr(entry, "kind", "") != "cellpose":
            continue
        if model_zoo.source_of(entry) not in sources:
            continue
        path = str(getattr(entry, "path", "") or "")
        if not (path and os.path.isfile(path)):
            candidate = os.path.join(folder, str(entry.name))
            path = candidate if os.path.isfile(candidate) else ""
        found.append((str(entry.key or entry.name), path or None, entry))
    return found


def load_cellpose_model(model_name: str):
    """Load a Cellpose model through spaCR's own resolver.

    :func:`spacr.utils._resolve_cellpose_pretrained` is what the pipeline
    itself calls: it maps every pre-SAM name onto ``cpsam``, keeps a
    fine-tuned checkpoint path as itself, and raises rather than quietly
    substituting stock weights for a checkpoint that is not there. Going
    around it with a second, simpler call would give this screen a
    different answer from the run it is meant to be correcting.

    ON A CPU THE WEIGHTS ARE LOADED IN FLOAT32. Cellpose-SAM loads them in
    bfloat16 by default, and a processor without native bfloat16
    arithmetic runs every tile through a slow emulation of it: measured on
    an AMD Ryzen 9 5950X, one 256 px tile took 162-197 s in bfloat16 and
    31-46 s in float32, alternating, under the same load. A whole field is a hundred such tiles, so this is
    the difference between most of an hour and most of a day. On a GPU
    nothing changes: :func:`spacr.accelerator.cellpose_kwargs` decides
    there, as it does for the pipeline.

    A ``cellpose3:<name or path>`` model -- a stock Cellpose 3 model or a
    bioimage.io Cellpose 3 checkpoint -- is not a Cellpose 4 model at all:
    Cellpose 4 would load such a checkpoint and segment nonsense with it. It
    is loaded by :func:`_backend_model`, in the Cellpose 3 backend's own
    environment, as Mask generation loads it. A ``cellpose_dino:<path>``
    model, a Cellpose-DINO checkpoint, is loaded the same way in the
    Cellpose-DINO backend (item 525).

    :param model_name: a Cellpose model name, the path of a fine-tuned
        checkpoint, resolved by
        :func:`spacr.utils._resolve_cellpose_pretrained`, or
        ``cellpose3:<name or path>``.
    """
    import inspect

    from ..._segmentation_backends import (_cellpose3_choice,
                                           _cellpose_dino_choice)

    if (_cellpose3_choice(model_name) is not None
            or _cellpose_dino_choice(model_name) is not None):
        return _backend_model(str(model_name).strip())

    import torch
    from cellpose import models as cp_models

    from ...utils import _resolve_cellpose_pretrained

    pretrained = _resolve_cellpose_pretrained(model_name)
    from ...accelerator import cellpose_kwargs

    kwargs = cellpose_kwargs()
    if not kwargs.get("gpu"):
        try:
            accepts = "use_bfloat16" in inspect.signature(
                cp_models.CellposeModel).parameters
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            kwargs["use_bfloat16"] = False
    return cp_models.CellposeModel(pretrained_model=pretrained, **kwargs)


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



#: The magnifier's starting settings and their ranges. Size is the side of the
#: square region the model segments, in IMAGE pixels; zoom is how many times
#: larger than the canvas draws that region the box draws it; a sensitivity
#: of 0 is each model's own default cut.
_MAGNIFIER_SIZE = 128
#: The size's range BEFORE A FIELD IS OPEN. Once one is, the top of the range
#: is that field's own longer side, so the magnifier can cover the whole
#: field -- see :func:`_magnifier_size_range`.
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

#: The Overlap rule a fresh panel is on, and what the magnifier assumes
#: until the screen's Overlap box says otherwise.
_MAGNIFIER_OVERLAP_DEFAULT = "clip"

#: Megabytes of whole-image objects the magnifier may keep for fields it has
#: left, so coming back to one does not segment it again -- minutes, on a CPU.
#: A label image is int32, so this is sixteen 2048 px fields or four 4096 px
#: ones. It is a CEILING and not an allowance: the user's own cache ceiling
#: (:func:`spacr.qt.preferences.get_cache_ceiling_mb`) wins when it is lower,
#: and the idle timeout beside it releases what nobody has come back to.
_MAGNIFIER_IMAGE_CACHE_MB = 256

#: Serialises every use of a Cellpose model between the magnifier's worker
#: thread and the screen's own detect button, which share the loaded models.
#: Re-entrant, because the detect button loads a model inside the same hold.
_CELLPOSE_LOCK = threading.RLock()


def _magnifier_size_range(shape=None) -> tuple:
    """``(smallest, largest)`` box side, in image pixels, for a field shaped so.

    The box is square, so its largest useful side is the field's LONGER side:
    a box that size, centred anywhere, reaches across the field the long way.
    A field narrower than the usual smallest box lowers the floor with it,
    so the range is never empty.

    :param shape: the open field's shape, ``(height, width, ...)``, or None
        when no field is open -- which gives :data:`_MAGNIFIER_SIZE_RANGE`.
    """
    if shape is None or len(shape) < 2:
        return _MAGNIFIER_SIZE_RANGE
    largest = max(1, int(shape[0]), int(shape[1]))
    return (min(_MAGNIFIER_SIZE_RANGE[0], largest), largest)


class _MagnifierRequest(NamedTuple):
    """One region to segment, with everything the model reads copied in.

    Copied rather than referenced: the worker reads it on another thread
    while the GUI thread goes on editing the mask and moving the box.
    ``key`` is what makes two requests the same request -- the field, the box
    (or ``"image"`` for the whole field), every setting a model reads and,
    for a region, whether cut objects are left out -- and it is what a click
    is matched to its result by.

    ``otsu_classes`` and ``otsu_foreground_class`` snapshot Multi-Otsu's
    class count and selected zero-based intensity band. The default count
    of 2 preserves legacy callers; Multi-Otsu enforces at least 3 classes.
    A None foreground class selects the brightest band. Both fields are
    appended to the request and settings key to preserve existing positions.
    ``detection_percentiles`` is None for raw detector input or the saved
    whole-field (low, high) display percentiles applied before cropping.
    It is separate from model normalization and is also part of the key.
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
    #: The Object detection settings' flow threshold, cell-probability
    #: threshold and normalization, which the models read.
    flow_threshold: float = FLOW_THRESHOLD
    cellprob_threshold: float = CELLPROB_THRESHOLD
    normalize: bool = True
    #: What Otsu's level is multiplied by in the Otsu mode.
    otsu_correction: float = 1.0
    #: Gaussian sigma the Otsu mode smooths the region with before it cuts.
    otsu_smoothing: float = OTSU_SMOOTHING
    #: Whether the Otsu mode closes the holes inside what it thresholded.
    otsu_fill_holes: bool = True
    #: Whether the Otsu mode cuts a blob with two centres into two objects.
    otsu_split: bool = True
    #: Whether :attr:`crop` was inverted before it was copied in. It is
    #: carried on the request, though nothing downstream
    #: reads it, because it is part of what makes two requests the same
    #: request: the same box under the same settings with Invert on and off
    #: are two different questions with two different answers.
    invert: bool = False
    #: The pre- and post-detection chain (:mod:`spacr.qt.detect_chain`).
    #: Applied by :func:`_segment_region` on the WORKER thread: a non-local
    #: means over a whole field is minutes, and item 407's progress and
    #: Cancel are only worth having if the slow part is behind them.
    chain: Any = detect_chain.NO_CHAIN
    #: Every organelle method's parameters
    #: (:class:`spacr.qt.organelle_modes.MethodParams`). All of them
    #: whichever mode is chosen, for the reason every other setting is
    #: here: a mode that cannot run hands the request to another, which
    #: must find its own settings in it.
    method_params: Any = organelle_modes.DEFAULT_PARAMS
    #: The CPU modes' parameters
    #: (:class:`spacr.qt.cpu_modes.CpuParams`) -- Sauvola's and Niblack's
    #: k, and everything Maxima + propagate reads.
    cpu_params: Any = cpu_modes.DEFAULT_PARAMS
    #: The local window, in pixels, that Sauvola and Niblack measure in.
    #: The Otsu category's own "Local window", read by them too.
    otsu_window: int = OTSU_LOCAL_WINDOW
    #: The Overlap rule the box is to draw its promise in, and the mask it
    #: is to be read against: the pixels the mask already owns inside
    #: ``box``, and a number that changes whenever the canvas is handed a
    #: different mask. NOT PART OF ``key``, because neither changes what the
    #: model is asked -- a rule and a mask edit change what the ANSWER
    #: means, which is why the worker is given both and
    #: :meth:`_LiveMagnifier.refresh` compares them beside the key.
    overlap: str = "replace"
    occupied: Optional[np.ndarray] = None
    mask_token: int = 0
    #: A whole-image run's :class:`_RunTicket`: how far the model has got
    #: and whether the run is still wanted. None for a region, which is
    #: small enough to finish. Not part of ``key``, for the same reason.
    ticket: Any = None
    otsu_classes: int = 2
    otsu_foreground_class: Optional[int] = None
    detection_percentiles: Optional[tuple] = None
    primary_token: tuple = ()
    primary_labels: Optional[np.ndarray] = None
    primary_provenance: Optional[dict] = None


class _RunCancelled(Exception):
    """A whole-image run was stopped, between two of the model's tiles."""


class _RunTicket:
    """How far one whole-image run has got, and whether it is still wanted.

    A MODEL CALL COULD NOT BE STOPPED, and on a CPU that was the difference
    between a tool and a trap. Cellpose-SAM on a CPU takes most of an hour
    for one 2,000 px field, and Cancel could only throw the answer
    away once it arrived: the model went on holding the processor, and the
    shared model lock with it, for the rest of those minutes. Cellpose runs
    a field as a series of 256 px tiles, one network call each, so the run
    CAN be asked between two of them -- this is what it is asked.

    Shared between the GUI thread, which reads the progress and cancels,
    and the worker, which counts and checks; every field is one assignment
    or one Event, so neither side takes a lock.

    :param total: how many tiles the run is expected to take, when known.
    """

    def __init__(self, total: int = 0):
        """A ticket for a run that has not started a tile yet."""
        self._stop = threading.Event()
        #: Tiles the model has STARTED, and how many it will run in all
        #: (0 while unknown).
        self.done = 0
        self.total = int(total)
        #: ``time.monotonic()`` when the first and the latest tile started.
        self.first_at: Optional[float] = None
        self.last_at: Optional[float] = None

    def cancel(self) -> None:
        """Ask the run to stop at its next tile."""
        self._stop.set()

    def cancelled(self) -> bool:
        """Whether the run has been asked to stop."""
        return self._stop.is_set()

    def check(self) -> None:
        """Raise :class:`_RunCancelled` if the run has been asked to stop."""
        if self._stop.is_set():
            raise _RunCancelled()

    def step(self, tiles: int = 1) -> None:
        """Count ``tiles`` starting now, unless the run is to stop instead.

        :raises _RunCancelled: when :meth:`cancel` was called.
        """
        self.check()
        now = time.monotonic()
        if self.first_at is None:
            self.first_at = now
        self.last_at = now
        self.done += max(1, int(tiles))

    def remaining_seconds(self) -> Optional[float]:
        """How long the tiles still to run will take, from this run's pace.

        MEASURED ON THIS RUN, so the first run of a session has an estimate
        too -- the one run that most needs one, on a CPU, and the one the
        per-megapixel pace (:meth:`_LiveMagnifier.remaining_seconds`) has to
        stay silent through. The pace is the time between the first tile
        and the latest one, so the model's load, which comes before the
        first, is not counted as tile time.

        :returns: seconds, or None before two tiles have started, when the
            total is unknown, or once the last tile is under way -- what
            follows it, turning the network's output into objects, is not
            made of tiles and is not guessed at.
        """
        done, total = self.done, self.total
        first, last = self.first_at, self.last_at
        if total <= 0 or done < 2 or first is None or last is None \
                or done >= total:
            return None
        pace = (last - first) / (done - 1)
        left = pace * (total - done + 1) - (time.monotonic() - last)
        return max(0.0, left)


def _cellpose_tile_count(shape, diameter: int = 0, bsize: int = 256,
                         tile_overlap: float = 0.1) -> int:
    """How many network calls Cellpose makes for one field of ``shape``.

    ``cellpose.core.run_net``'s own arithmetic, for one 2-D image: the
    field is rescaled by ``30 / diameter`` when a diameter is set, padded
    by ``cellpose.transforms.get_pad_yx``, and cut into overlapping tiles of
    ``bsize``. Cellpose-SAM's ``bsize`` is 256 and cannot be changed. Used
    only to say how far a run has got; a Cellpose that tiles differently
    makes the bar wrong and nothing else, and the bar never claims more than
    it has counted (:meth:`_RunTicket.remaining_seconds`).

    :param shape: the field's ``(height, width, ...)``.
    :param diameter: the Object detection diameter; 0 for none.
    :returns: the number of tiles, at least 1.
    """
    height, width = int(shape[0]), int(shape[1])
    if diameter and int(diameter) > 0:
        rescale = 30.0 / float(diameter)
        height, width = int(height * rescale), int(width * rescale)
    try:
        from cellpose.transforms import get_pad_yx

        ypad1, ypad2, xpad1, xpad2 = get_pad_yx(
            height, width, min_size=(bsize, bsize))
    except Exception:                                       # noqa: BLE001
        ypad1 = ypad2 = xpad1 = xpad2 = 0
    padded_y = height + ypad1 + ypad2
    padded_x = width + xpad1 + xpad2
    ny = 1 if padded_y <= bsize else int(
        math.ceil((1.0 + 2 * tile_overlap) * padded_y / bsize))
    nx = 1 if padded_x <= bsize else int(
        math.ceil((1.0 + 2 * tile_overlap) * padded_x / bsize))
    return max(1, ny * nx)


def _counting_tiles(model, ticket):
    """Count ``model``'s network calls on ``ticket``, and stop on a cancel.

    A forward PRE-hook on the model's network: it runs before each tile, so
    a cancelled run stops before spending another tile on an answer nobody
    wants, and the exception it raises leaves ``eval`` the way any error
    would -- the model lock is released by its own ``with``.

    :returns: the hook's handle, to ``remove()`` afterwards, or None when
        there is no ticket or the model has no torch network to hook -- a
        stand-in, or a backend that runs out of process.
    """
    net = getattr(model, "net", None)
    register = getattr(net, "register_forward_pre_hook", None)
    if ticket is None or not callable(register):
        return None

    def before_a_tile(_module, inputs):
        """Count this call's tiles, or stop the run here."""
        batch = inputs[0] if inputs else None
        ticket.step(int(getattr(batch, "shape", (1,))[0] or 1))

    return register(before_a_tile)


#: Everything a model reads, in the order :meth:`_LiveMagnifier._model_settings`
#: gives it and a request key carries it after ``(field, box)``. The mode is
#: first, which is what a key's ``[2]`` is read as. ADDING ONE GOES AT THE
#: END, beside the same name in :meth:`_LiveMagnifier._model_settings`: a
#: request key is this tuple positionally, and an insertion in the middle
#: would make every key already cached mean something else.
def _cellpose3_auto_diameter_note() -> str:
    """Why a Cellpose 3 run with Diameter 0 over a whole field is slow.

    Measured for item 507 on a real 1994 x 1994 Toxoplasma field: Cellpose 3
    estimated 16 px for vacuoles whose median is 44 px, rescaled the field
    by 30/16 and took 153 s on the CPU, matching 7 of 33 objects; Diameter
    44 took 12 s and matched 24.
    """
    from ..i18n import tr

    return tr(
        "Cellpose 3 with Diameter 0 first estimates the object size and "
        "rescales the whole field to it. On a 1994 x 1994 Toxoplasma field on "
        "the CPU that took 153 s and guessed 16 px for vacuoles of 44 px; with "
        "Diameter 44 it took 12 s. Set Diameter to the objects' size in pixels "
        "to skip the estimate.")


_MODEL_SETTING_FIELDS = ("mode", "sensitivity", "bright", "min_area",
                         "model_name", "diameter", "flow_threshold",
                         "cellprob_threshold", "normalize", "otsu_correction",
                         "otsu_smoothing", "otsu_fill_holes", "otsu_split",
                         "invert", "chain", "method_params", "cpu_params",
                         "otsu_window", "otsu_classes", "otsu_foreground_class",
                         "detection_percentiles", "primary_token")


class _MagnifierResult(NamedTuple):
    """The objects found for one request, in that request's CROP coordinates.

    ``labels`` is exactly what the box outlines and exactly what a click
    commits: unless the request keeps them, it has already lost the objects
    the box edge cut (:func:`spacr.qt.mask_engine._drop_cut_objects`). A
    whole-image request's crop is the whole field, so nothing in it is cut.
    ``mode`` is the mode that actually ran and ``note`` says why when that is
    not the one asked for; ``overlay`` is the RGBA picture of the outlines
    -- for a whole-image result, of the whole field, which the box slices
    rather than outlining its part of the field again on every move;
    ``count`` is how many objects ``labels`` holds.

    ``ghost`` is ``overlay`` again with the pixels the request's Overlap
    rule would NOT add faded to a quarter of their alpha -- what the box
    draws, when the rule takes something away. It is built beside the
    outlines on the worker thread and is None when there is nothing to fade:
    under Replace, over an empty mask, or when the rule keeps everything.
    See :func:`_ghosted_overlay`.
    """

    request: _MagnifierRequest
    labels: np.ndarray
    mode: str
    note: str
    overlay: Optional[np.ndarray]
    count: int = 0
    ghost: Optional[np.ndarray] = None
    #: A whole-image result's ``scipy.ndimage.find_objects`` of ``labels``:
    #: every object's bounding box, from one pass on the worker. What the
    #: Overlap rule's promise for one object is computed over, since the
    #: rule's answer for an object depends on all of it and not on the
    #: part the box shows. See :func:`_object_window`.
    extents: Optional[tuple] = None

    def nbytes(self) -> int:
        """What this result holds in memory: its labels and its pictures."""
        return int(sum(np.asarray(part).nbytes
                       for part in (self.labels, self.overlay, self.ghost)
                       if part is not None))


def _cellpose_segmenter(request: _MagnifierRequest, load_model=None):
    """Segment the region with the Object detection settings the screen has.

    ONE SOURCE OF TRUTH: the model, the flow threshold, the
    cell-probability threshold, the diameter and the normalization are the
    Object detection category's, exactly as the detect button passes them, so
    the box and the button cannot disagree about what Cellpose was asked. The
    magnifier's own sensitivity is the Otsu mode's and is not read here.

    A WHOLE-IMAGE RUN COUNTS ITS TILES AND CAN BE STOPPED between two of
    them, through the request's :class:`_RunTicket`: on a CPU that run is
    minutes long, and a Cancel that could only discard the answer left the
    processor and the model lock taken until it arrived.
    """
    loader = load_model or load_cellpose_model
    ticket = request.ticket
    if ticket is not None:
        ticket.check()
        ticket.total = _cellpose_tile_count(request.crop.shape,
                                            int(request.diameter))
    with _CELLPOSE_LOCK:
        if ticket is not None:
            ticket.check()
        model = loader(request.model_name)
        hook = _counting_tiles(model, ticket)
        try:
            labels, _cellprob, _flow = cellpose_detect(
                request.crop, model,
                diameter=int(request.diameter),
                normalize=bool(request.normalize),
                flow_threshold=float(request.flow_threshold),
                cellprob_threshold=float(request.cellprob_threshold),
                min_size=int(request.min_area),
            )
        finally:
            if hook is not None:
                hook.remove()
    if ticket is not None:
        ticket.check()
    return labels


#: The optional models the magnifier runs through
#: :mod:`spacr._segmentation_backends`: ``mode -> (backend, the label the
#: Mode box shows)``. A Cellpose 3 mode names its model after the colon.
#: Each is always listed, greyed until its backend is installed, and
#: choosing a greyed one offers the install -- into an environment of its
#: own rather than into spaCR's, so a backend's dependencies cannot break
#: spaCR's.
_MAGNIFIER_BACKENDS = {
    "cellpose3:cyto3": ("cellpose3", "Cellpose 3 · cyto3"),
    "cellpose3:cyto2": ("cellpose3", "Cellpose 3 · cyto2"),
    "cellpose3:cyto": ("cellpose3", "Cellpose 3 · cyto"),
    "cellpose3:nuclei": ("cellpose3", "Cellpose 3 · nuclei"),
    "dinocell": ("dinocell", "DINOCell"),
    "samcell": ("samcell", "SAMCell"),
}

#: Loaded DINOCell and SAMCell models, by backend name, for the life of the
#: process. Building one loads a ViT checkpoint, and the box asks on every
#: move; a backend that fails to build is not kept, so it is tried again.
_BACKEND_MODELS: dict = {}

#: ``mode -> the environment folder the cached model was built against``, for
#: the modes whose backend really had one. See :func:`_backend_model`.
_BACKEND_MODEL_ENVS: dict = {}


def _state_ready(backend: str) -> bool:
    """Whether ``backend``'s environment is here and can segment now.

    File checks only -- no import, no subprocess -- so the Mode box can ask
    while it is built and a cached model can be checked before it is used.
    """
    from ... import _segmentation_backends

    try:
        return _segmentation_backends._backend_state(backend).ready
    except (OSError, ValueError):
        return False


def _backend_model(name: str):
    """The backend model a magnifier mode names, built once.

    Built by :func:`spacr._segmentation_backends._load_backend`, the same
    loader the mask pipeline's ``segmentation_backend`` setting uses. A
    ``cellpose3:<model>`` mode loads that Cellpose 3 model.

    A CACHED MODEL IS CHECKED AGAINST THE FOLDER IT WAS BUILT FROM.
    Uninstalling the backend from the Model Zoo while Make Masks is open left
    this dictionary holding a model whose environment had been deleted;
    asking it to segment then tried to start a Python that is no longer
    there, and the magnifier reported a missing file instead of a missing
    backend. The folder, and not the backend's state, because a model built
    without one -- a stand-in, or a backend that lives in spaCR's own
    environment -- was never on disk to lose.

    :raises ImportError: naming the Model Zoo, when the backend is missing.
    """
    with _CELLPOSE_LOCK:
        from ... import _segmentation_backends

        backend, _colon, model_name = str(name).partition(":")
        model = _BACKEND_MODELS.get(name)
        env = _BACKEND_MODEL_ENVS.get(name)
        if model is not None and env and not os.path.isdir(env):
            _BACKEND_MODELS.pop(name, None)
            _BACKEND_MODEL_ENVS.pop(name, None)
            model = None
        if model is None:
            model = _segmentation_backends._load_backend(
                backend, model_name=model_name or None)
            _BACKEND_MODELS[name] = model
            _BACKEND_MODEL_ENVS[name] = _model_env(backend)
        return model


def _model_env(backend: str) -> str:
    """The folder a model of ``backend`` was just built from, or ``''``.

    Empty for a backend that is not installed in one of its own -- a
    stand-in, or one an older spaCR put in spaCR's own environment -- which
    is a model :func:`_backend_model` must not go on to drop.
    """
    from ... import _segmentation_backends

    try:
        state = _segmentation_backends._backend_state(backend)
    except (OSError, ValueError):
        return ""
    return state.env if state.ready and not state.in_process else ""


def _backend_ready(mode: str) -> bool:
    """Whether the backend a magnifier mode needs can segment now."""
    return _state_ready(_MAGNIFIER_BACKENDS[mode][0])


def _backend_segmenter(request: _MagnifierRequest, load_model=None):
    """Segment the region with Cellpose 3, DINOCell or SAMCell.

    A backend answers ``CellposeModel.eval``'s own call, so the region goes
    through :func:`cellpose_detect` with the Object detection settings as
    Cellpose's does: Cellpose 3 reads both thresholds and the diameter (0
    lets its size model estimate it), DINOCell reads the cell-probability
    threshold (through the logistic function, so 0 is its own 0.5) and
    SAMCell uses its own thresholds. Each runs in its own environment, out
    of process. ``load_model`` is the CELLPOSE loader and is not used: a
    backend name handed to it would load stock cpsam without a word.
    """
    with _CELLPOSE_LOCK:
        model = _backend_model(request.mode)
        labels, _cellprob, _flow = cellpose_detect(
            request.crop, model,
            diameter=int(request.diameter),
            normalize=bool(request.normalize),
            flow_threshold=float(request.flow_threshold),
            cellprob_threshold=float(request.cellprob_threshold),
            min_size=int(request.min_area),
        )
    return labels


def _cellpose_installed() -> bool:
    """Whether Cellpose can be imported here, WITHOUT importing it.

    :func:`importlib.util.find_spec`, so a package that pulls in torch is
    located and not loaded. A spec lookup that raises counts as absent.
    """
    try:
        return find_spec("cellpose") is not None
    except (ImportError, ValueError):
        return False


def _threshold_label(name: str) -> str:
    """The caption a threshold algorithm goes under in a box."""
    if name == "otsu":
        return "Otsu"
    return cpu_modes.THRESHOLD_LABELS.get(name, str(name).title())


def _threshold_segmenter(request: _MagnifierRequest, load_model=None):
    """Cut the region at the level the chosen ALGORITHM finds.

    Otsu, Li's minimum cross entropy, Yen, Triangle, IsoData, Mean,
    Minimum, Multi-Otsu, Sauvola and Niblack all arrive here, and the mode
    name maps to the algorithm name
    (:func:`spacr.qt.cpu_modes.engine_algorithm`). Every named threshold
    except plain Otsu uses :func:`spacr.qt.mask_engine._otsu_instances`,
    the same engine as the detect button. The request supplies smoothing,
    correction, polarity, hole filling, splitting and minimum area. Only
    Multi-Otsu reads its class count and band; Sauvola and Niblack read
    window and local k. None of these modes reads magnifier Sensitivity
    or applies the legacy crop stretch, opening or noise-floor fallback.
    Thresholds are estimated from the requested pixels, so different
    crops can still yield different results.

    Plain Otsu keeps :func:`spacr.qt.mask_engine._classical_region_labels`
    and its existing sensitivity/noise-floor behavior.
    """
    mode = canonical_magnifier_mode(request.mode)
    params = request.cpu_params
    if mode in cpu_modes.THRESHOLD_LABELS:
        multi = mode == cpu_modes.MULTIOTSU
        return engine._otsu_instances(
            request.crop, bright=request.bright, min_area=request.min_area,
            correction=request.otsu_correction,
            smoothing=request.otsu_smoothing,
            fill_holes=request.otsu_fill_holes,
            split_touching=request.otsu_split,
            algorithm=cpu_modes.engine_algorithm(mode),
            classes=max(3, int(request.otsu_classes)) if multi else 2,
            foreground_class=request.otsu_foreground_class if multi else None,
            window=int(request.otsu_window), local_k=float(params.local_k))
    return engine._classical_region_labels(
        request.crop, sensitivity=request.sensitivity,
        bright=request.bright, min_area=request.min_area,
        correction=request.otsu_correction,
        smoothing=request.otsu_smoothing,
        fill_holes=request.otsu_fill_holes,
        split_touching=request.otsu_split,
        algorithm=cpu_modes.engine_algorithm(mode),
        window=int(request.otsu_window),
        local_k=float(params.local_k))


def _propagate_segmenter(request: _MagnifierRequest, load_model=None):
    """Grow one object out of each bright centre of the region.

    Seeded watershed on inverted intensity; see
    :func:`spacr.qt.mask_engine.maxima_propagate_instances`. How many
    centres were found is carried back on the request's ticket-free path
    through :data:`_LAST_PROPAGATE_SEEDS`, because it is the number a
    curator tunes the settings against and the labels alone do not show
    it: an object that never grew and a centre that was never found look
    the same.
    """
    ticket = request.ticket
    if ticket is not None:
        ticket.check()
    found = cpu_modes.propagate(
        request.crop, request.cpu_params, min_area=int(request.min_area),
        fill_holes=bool(request.otsu_fill_holes))
    _LAST_PROPAGATE_SEEDS[request.scope] = (found.seeds, found.level)
    if ticket is not None:
        ticket.check()
    return found.labels


def _secondary_segmenter(request: _MagnifierRequest, load_model=None):
    """Grow the request's copied primary labels without discovering new seeds."""
    if request.primary_labels is None:
        raise ValueError("Choose and load a primary mask before growing secondary objects.")
    return cpu_modes.secondary(
        request.crop, request.primary_labels, request.cpu_params,
        min_area=request.min_area, fill_holes=request.otsu_fill_holes).labels


#: ``scope -> (centres found, the level they were grown to)`` for the last
#: propagation of each scope. Read by the screen's status line. A plain
#: dict and not a signal: it is written on the worker and read on the GUI
#: thread right after the result arrives, and a stale entry can only say a
#: number about a run that has just been superseded.
_LAST_PROPAGATE_SEEDS: dict = {}


def _organelle_segmenter(request: _MagnifierRequest, load_model=None):
    """Segment the region with one of organelle detection's own methods.

    Adaptive, LoG, DoG, ridge, hysteresis and U-Net reach this one
    function, and it reaches :mod:`spacr.qt.organelle_modes`, which reaches
    :func:`spacr.object._segment_single_image` -- the routine the organelle
    mask pipeline's own workers call. THERE IS NO SECOND COPY of any of
    these methods: a block size tuned on the box under the mouse is the
    block size a mask run will read, and a method that is fixed in the
    engine is fixed here on the same day.

    ``load_model`` is the CELLPOSE loader and is not used; a U-Net is
    loaded from the path the U-Net parameters name.

    A WHOLE-IMAGE RUN CAN BE CANCELLED BETWEEN STEPS AND NOT INSIDE ONE.
    :class:`_RunTicket`'s progress is Cellpose's tiles, counted by a
    forward hook on its network (:func:`_counting_tiles`); a scikit-image
    filter has no tiles, so the bar stays indeterminate and Cancel is
    honoured at the boundaries -- before the chain, before the detector,
    after it. That is the honest bar rather than a thin one: these runs are
    seconds on a field where Cellpose-SAM on a CPU is most of an hour. The
    exception a curator can feel is a heavy chain step over a whole field,
    which :func:`spacr.qt.detect_chain.heavy_steps` warns about before it
    is switched on.
    """
    ticket = request.ticket
    if ticket is not None:
        ticket.check()
    labels = organelle_modes.segment(
        request.crop, canonical_magnifier_mode(request.mode),
        request.method_params, min_area=int(request.min_area))
    if ticket is not None:
        ticket.check()
    return labels


#: ``mode -> segmenter``, in the order the Mode box offers them. A segmenter
#: takes ``(request, load_model)`` and returns labels shaped like
#: ``request.crop``. ADDING A MODEL IS ONE FUNCTION AND ONE LINE HERE, and
#: :func:`_segment_region` gives it the Otsu fallback for nothing.
_MAGNIFIER_SEGMENTERS = {
    "otsu": _threshold_segmenter,
    **{mode: _threshold_segmenter for mode in cpu_modes.threshold_modes()},
    cpu_modes.PROPAGATE: _propagate_segmenter,
    cpu_modes.SECONDARY: _secondary_segmenter,
    **{mode: _organelle_segmenter for mode in organelle_modes.modes()},
    "cellpose": _cellpose_segmenter,
    **{mode: _backend_segmenter for mode in _MAGNIFIER_BACKENDS},
}

#: Mode names this screen used to carry, and what they are called now. The
#: magnifier's threshold mode was once called ``classical`` and is now
#: ``otsu``; a mode name reaches this module from
#: a saved session, a script or a test, so the old one still arrives and is
#: translated rather than refused.
_MAGNIFIER_MODE_ALIASES = {"classical": "otsu"}


def canonical_magnifier_mode(mode) -> str:
    """The name a magnifier mode goes under today.

    :param mode: a mode name, possibly one this screen has renamed.
    :returns: the current name; an unknown mode is handed back unchanged, so
        :func:`_segment_region` can still say it does not know it.
    """
    name = str(mode or "otsu")
    return _MAGNIFIER_MODE_ALIASES.get(name, name)


#: The modes the Mode box builds itself, as ``mode -> its caption``: Otsu,
#: organelle detection's own methods (:mod:`spacr.qt.organelle_modes`) and
#: Cellpose. The rest are named by :data:`_MAGNIFIER_BACKENDS`.
_MAGNIFIER_MODE_LABELS = {"otsu": "Otsu",
                          **cpu_modes.MODE_LABELS,
                          **organelle_modes.MODE_LABELS,
                          "cellpose": "Cellpose"}


def _magnifier_mode_label(mode: str) -> str:
    """The caption the Mode box shows for ``mode``, translated.

    A status line that names a mode has to name it the way the box the user
    picked it from does. A mode under its old name is named under the new
    one, through :func:`canonical_magnifier_mode`; an unknown mode is handed
    back unchanged rather than hidden, so a key that has lost its caption
    still reads as itself.

    :param mode: a key of :data:`_MAGNIFIER_SEGMENTERS`, or one this screen
        has renamed.
    :returns: the caption, in the current language.
    """
    from ..i18n import tr

    key = canonical_magnifier_mode(mode)
    name = _MAGNIFIER_MODE_LABELS.get(key)
    if name is None:
        backend = _MAGNIFIER_BACKENDS.get(key)
        name = backend[1] if backend is not None else str(mode)
    return tr(name)


def _updating_caption(name: str) -> str:
    """The magnifier's "Updating <model>…" mark, in the current language.

    A language whose catalog has no row for the named form yet keeps its
    reviewed "Updating…" and names the model after it, rather than showing
    English words in a translated window.

    :param name: what is running, from :meth:`running_name`.
    :returns: the caption.
    """
    from ..i18n import current_language, tr

    template = "Updating {name}…"
    translated = tr("Updating {name}…", name=name)
    if (current_language() or "en").startswith("en") or \
            translated != template.format(name=name):
        return translated
    return f"{tr('Updating…')} {name}"


def _segment_region(request: _MagnifierRequest, load_model=None) -> tuple:
    """Run the request's mode, falling back to Otsu when it cannot run.

    The fallback is why the magnifier never simply does nothing: a model
    whose import or weights fail answers with the Otsu mode's objects
    and a note saying why, rather than with an empty box.

    :param request: the region and its settings.
    :param load_model: ``name -> model`` for modes that load one; called on
        the worker thread.
    :returns: ``(labels, mode_used, note)``; ``note`` is empty unless the mode
        asked for could not run.

    Classical CPU detection errors are reported to the caller without
    substituting Otsu. For example, Minimum may not find two histogram
    maxima and Multi-Otsu may have too few distinct intensities. Neither
    failure means the method is unavailable; a later crop can try again.
    """
    if request.ticket is not None:
        request.ticket.check()
    chain = request.chain or detect_chain.NO_CHAIN
    prepared = detect_chain.prepare(
        request.crop, chain,
        cancel=request.ticket.cancelled if request.ticket is not None else None)
    if prepared is not request.crop:
        request = request._replace(crop=prepared)
    if request.ticket is not None:
        request.ticket.check()
    mode = canonical_magnifier_mode(request.mode)
    segmenter = _MAGNIFIER_SEGMENTERS.get(mode)
    note = ""
    if mode == cpu_modes.SECONDARY:
        labels = segmenter(request, load_model)
        if request.ticket is not None:
            request.ticket.check()
        return labels, mode, ""
    if segmenter is None:
        note = f"no magnifier mode is called {request.mode!r}"
    elif (mode in cpu_modes.THRESHOLD_LABELS
          or mode in organelle_modes.MODE_LABELS and mode != "unet"):
        labels = segmenter(request, load_model)
        return _finished_labels(labels, chain, prepared), mode, ""
    elif mode != "otsu":
        try:
            labels = segmenter(request, load_model)
            return _finished_labels(labels, chain, prepared), mode, ""
        except _RunCancelled:
            raise
        except Exception as exc:                            # noqa: BLE001
            LOG.warning("magnifier mode %s could not run; using Otsu",
                        request.mode, exc_info=True)
            note = f"{type(exc).__name__}: {exc}"
    return (_threshold_segmenter(request._replace(mode="otsu"), load_model),
            "otsu", note)


def _finished_labels(labels, chain, image):
    """The chain's last two stages, for every mode but Otsu.

    THE OTSU MODE IS LEFT ALONE and keeps the "Fill holes inside an object"
    and "Split objects that touch" of its own category, which it has always
    had. The chain's morphology and split would arrive on top of them, and
    the first thing either does is read the detection back as ONE
    foreground -- so a pair Otsu had just cut apart would be joined again
    before being cut a second time. The chain's split is there to give the
    other threshold methods what Otsu already has, not to give Otsu it
    twice.

    :param labels: what the mode found.
    :param chain: the request's chain.
    :param image: the image the mode read, as the split's landscape.
    """
    return detect_chain.finish(labels, chain, intensity=image)


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


#: How many pixels of a box the contrast stretch may look at. The stretch
#: is two percentiles, and a percentile over four megapixels is most of what
#: a move at the largest box used to cost; over a grid of at most this many
#: of them the two levels are the same to a fraction of a step, and the box
#: goes on being stretched over the WHOLE region it covers rather than over
#: whichever corner of it is on screen.
_STRETCH_SAMPLES = 262144


def _stretch_for_box(image: np.ndarray, box, part,
                     lower_pct: float, upper_pct: float) -> np.ndarray:
    """``part`` of ``box``, contrast-stretched over all of ``box``'s levels.

    :func:`spacr.qt.mask_engine.normalize_uint16` reads the levels from the
    pixels it is handed and rescales the same pixels; the box needs the two
    halves of that separated, because what it must stretch over is the
    region it magnifies and what it must PAY FOR is the part of that region
    a user can see. At the default box the two are one thing and this
    returns exactly what that function would (``part`` is the whole box and
    the sample is every pixel of it); at the largest box allowed,
    with most of the lens off the canvas, it is the difference between a
    move a user feels and one nobody can.

    :param image: the field.
    :param box: ``(x0, y0, x1, y1)`` the box magnifies, in image pixels.
    :param part: ``(x0, y0, x1, y1)`` of it to return, inside ``box``.
    :param lower_pct: the percentile that becomes black.
    :param upper_pct: the percentile that becomes white.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    whole = image[y0:y1, x0:x1]
    if not whole.size:
        return np.ascontiguousarray(whole)
    low, high = _box_levels(image, box, lower_pct, upper_pct)
    vx0, vy0, vx1, vy1 = (int(v) for v in part)
    crop = np.ascontiguousarray(image[vy0:vy1, vx0:vx1])
    out = (np.clip(crop, low, high).astype(np.float64) - low) / (high - low)
    return (out * float(np.iinfo(crop.dtype).max)).astype(crop.dtype)


def _box_levels(image: np.ndarray, box, lower_pct: float,
                upper_pct: float) -> tuple:
    """The two levels the box is stretched between: ``(black, white)``.

    Read from the WHOLE region the box magnifies, over a grid of at most
    :data:`_STRETCH_SAMPLES` of its pixels -- every pixel, at a box the
    window holds. :func:`_stretch_for_box` and :func:`_box_picture` both
    read them here, so the two cannot stretch one box two ways.

    :param image: the field.
    :param box: ``(x0, y0, x1, y1)`` in image pixels; not empty.
    :param lower_pct: the percentile that becomes black.
    :param upper_pct: the percentile that becomes white.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    whole = image[y0:y1, x0:x1]
    step = max(1, int(math.sqrt(whole.size / _STRETCH_SAMPLES)))
    sample = whole[::step, ::step]
    low, high = np.percentile(sample, [lower_pct, upper_pct])
    if high <= low:
        high = low + 1
    return low, high


#: ``(dtype, black, white) -> the pixel each value is drawn as``, for the
#: last levels the box was drawn at. One entry: the levels change when the
#: box moves over a different region, and a table for levels nobody is
#: looking at any more is 256 KB held for nothing.
_BOX_GREY_TABLE: dict = {}


def _rgb32(rgb: np.ndarray) -> np.ndarray:
    """``(h, w, 3)`` uint8 as ``(h, w)`` ``0xFFRRGGBB`` -- Qt's own RGB32."""
    rgb = np.asarray(rgb, dtype=np.uint32)
    return (np.uint32(0xFF000000) | (rgb[..., 0] << 16)
            | (rgb[..., 1] << 8) | rgb[..., 2]).astype(np.uint32)


def _box_grey_table(dtype, low, high) -> Optional[np.ndarray]:
    """Every value of ``dtype`` as the grey pixel the box draws it as, or None.

    The box's picture is two passes that read ONE value each --
    :func:`_stretch_for_box` rescales a pixel between the levels,
    :func:`spacr.qt.mask_engine.overlay_mask` turns the result into 8 bits
    -- so the pair is a function of the pixel's value and the levels alone,
    and for a 16-bit field it is a table of 65,536 entries. Built by
    running those same two formulas over every value, which is what makes a
    look-up in it byte-identical to the two passes rather than close to
    them; `test_the_fast_box_picture_is_the_slow_one_to_the_byte` holds
    it to that.

    :param dtype: the field's dtype.
    :param low: the black level, from :func:`_box_levels`.
    :param high: the white level.
    :returns: a ``0xFFgggggg`` uint32 table indexed by value, or None for a
        dtype it cannot index -- a float field, a signed one, or one wider
        than 16 bits -- which then takes the two passes.
    """
    dtype = np.dtype(dtype)
    if dtype.kind != "u" or dtype.itemsize > 2:
        return None
    key = (dtype.str, float(low), float(high))
    table = _BOX_GREY_TABLE.get(key)
    if table is None:
        values = np.arange(np.iinfo(dtype).max + 1, dtype=dtype)
        out = (np.clip(values, low, high).astype(np.float64) - low) \
            / (high - low)
        stretched = (out * float(np.iinfo(dtype).max)).astype(dtype)
        grey = (stretched.astype(np.float32) / 256.0).clip(0, 255) \
            .astype(np.uint32)
        table = (np.uint32(0xFF000000) | (grey << 16) | (grey << 8)
                 | grey).astype(np.uint32)
        _BOX_GREY_TABLE.clear()
        _BOX_GREY_TABLE[key] = table
    return table


def _box_picture(image: np.ndarray, box, part, mask_part: np.ndarray,
                 lower_pct: float, upper_pct: float,
                 step: int = 1) -> np.ndarray:
    """What the box draws of ``part``: stretched, in grey, with the mask over it.

    EXACTLY ``overlay_mask(_stretch_for_box(...), mask, alpha=0.5)``, and
    made to cost a frame at the largest box the Size box allows. That box
    is as wide as the field and most of it is on the canvas: 1.4 Mpx per
    move on a 2,048 px field, and the two passes spent 41.5 ms of float
    arithmetic over 4.2 M channel values on it, per move, on the GUI thread
    -- three frames, measured by ``tools/perf_paint.py --only magnifier``.

    THE DESIGN DECISION: NOTHING IS CACHED BETWEEN MOVES. The picture could
    have been built once per field, mask and levels and blitted from, but
    the levels move with the box and the mask moves with every edit, so
    that cache misses exactly when the user is working -- and its miss is
    the whole field, a 150 ms frame of its own. Instead every move is made
    cheap, and there is no first move that pays for the rest:

    * the stretch and the 8-bit conversion become one look-up in a table
      built for the levels (:func:`_box_grey_table`), straight into Qt's
      native 32-bit pixel, so the picture is never converted again on its
      way to the screen, as a 24-bit one is on every draw;
    * the blend, which at ``alpha`` 0.5 is the floor of the mean of grey
      and colour, is the carry-free byte average ``(a & b) + ((a ^ b) >> 1)``
      over whole pixels, and only where the mask has an object.

    Nothing is approximated --
    `test_the_fast_box_picture_is_the_slow_one_to_the_byte`.

    ``step`` thins the picture when the lens shows more image pixels than
    the screen has device pixels for them (a zoom below one device pixel per
    image pixel): drawing unsmoothed, Qt would drop those pixels anyway, so
    they are not made. At every zoom that magnifies, it is 1 and nothing is
    dropped -- the box stays exactly as sharp as it was.

    :param image: the field the box magnifies.
    :param box: ``(x0, y0, x1, y1)`` the box covers; the levels come from it.
    :param part: ``(x0, y0, x1, y1)`` of it to draw, inside ``box``.
    :param mask_part: the mask's labels over ``part``.
    :param lower_pct: the percentile that becomes black.
    :param upper_pct: the percentile that becomes white.
    :param step: take every ``step``-th pixel of ``part`` each way.
    :returns: a C-contiguous ``(h, w)`` uint32 picture, ``0xFFRRGGBB``.
    """
    step = max(1, int(step))
    vx0, vy0, vx1, vy1 = (int(v) for v in part)
    crop = image[vy0:vy1:step, vx0:vx1:step]
    labels = np.asarray(mask_part)[::step, ::step]
    low, high = _box_levels(image, box, lower_pct, upper_pct)
    table = _box_grey_table(image.dtype, low, high)
    if table is None:
        stretched = _stretch_for_box(image, box, part, lower_pct, upper_pct)
        return np.ascontiguousarray(_rgb32(engine.overlay_mask(
            stretched[::step, ::step], labels, alpha=0.5)))
    out = table[crop]
    flat = np.ascontiguousarray(labels).reshape(-1)
    inside = np.flatnonzero(flat != 0)
    if inside.size:
        top = int(flat.max())
        rng = np.random.default_rng(0)
        colours = rng.integers(30, 255, size=(top + 1, 3), dtype=np.uint8)
        colours[0] = [0, 0, 0]
        paint = _rgb32(colours)
        pixels = out.reshape(-1)
        grey = pixels[inside]
        colour = paint[flat[inside]]
        pixels[inside] = (grey & colour) + (
            ((grey ^ colour) >> 1) & np.uint32(0x7F7F7F))
    return out


def _canonical_overlap_rule(rule) -> str:
    """The Overlap rule ``rule`` names, or ``replace`` when it names none.

    A rule reaches the worker inside a request, and a request outlives the
    box it was built for: a name the engine does not know must leave the
    box drawing the model's own outlines rather than raise on a thread
    whose exception the user would meet as an empty box.
    """
    name = str(rule or "")
    return name if name in engine._MAGNIFIER_OVERLAP_RULES else "replace"


def _ghosted_overlay(labels: np.ndarray, overlay: np.ndarray,
                     request: _MagnifierRequest) -> Optional[np.ndarray]:
    """``overlay`` with what the Overlap rule would NOT add ghosted.

    The box used to outline what the MODEL found, which is not what a click
    adds: the Overlap rule and Min area stand between the two, and a click
    that added half an object, or nothing, had said nothing first. The
    pixels the rule takes away keep a quarter of their alpha, so they read
    as "found, not yours" beside the solid objects a click would commit.

    RUN HERE, ON THE WORKER, AND NOT WHERE THE BOX IS DRAWN. The first
    version of this was a paintEvent's work, and the rule is counted over
    the box's pixels: on the largest box allowed -- the field's own
    longer side -- that measured 125.5 ms per delivered result on the GUI
    thread, nine frames of the interactive frame budget, for a picture the worker
    could have brought with it. What it needs from the GUI thread is the
    mask, and the mask is copied into the request beside the crop
    (:meth:`_LiveMagnifier.build_request`), which is how everything else a
    model reads gets here.

    :param labels: the objects found, in crop coordinates.
    :param overlay: their outlines as RGBA, from :func:`_candidate_overlay`.
    :param request: the request, for its rule, its mask crop and Min area.
    :returns: the picture, or None when the rule takes nothing away --
        under Replace, over an empty mask, or when everything survives.
    """
    occupied = request.occupied
    rule = _canonical_overlap_rule(request.overlap)
    if occupied is None or overlay is None or rule == "replace":
        return None
    exact_ids = request.mode == cpu_modes.SECONDARY
    occupied = ((np.asarray(occupied) > 0) & (np.asarray(occupied) != labels)
                if exact_ids else np.asarray(occupied) > 0)
    if occupied.shape != np.asarray(labels).shape or not occupied.any():
        return None
    kept = engine._surviving_region_objects(
        labels, occupied, overlap=rule, min_area=int(request.min_area),
        preserve_ids=exact_ids)
    lost = (np.asarray(labels) > 0) & (kept == 0)
    if not lost.any():
        return None
    rgba = np.array(overlay, copy=True)
    rgba[lost, 3] = rgba[lost, 3] // 4
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


def _object_extents(labels: np.ndarray) -> tuple:
    """Every object's bounding box in ``labels``, in one pass.

    ``scipy.ndimage.find_objects``: entry ``id - 1`` is ``(rows, cols)`` as
    slices, or None for an id with no pixels.
    """
    from scipy import ndimage

    lab = np.asarray(labels)
    if not lab.size or int(lab.max()) <= 0:
        return ()
    return tuple(ndimage.find_objects(lab.astype(np.int32, copy=False)))


def _object_window(result: _MagnifierResult, label: int) -> Optional[tuple]:
    """``(x0, y0, x1, y1)`` around object ``label`` of a whole-image result.

    Read from :attr:`_MagnifierResult.extents` when the worker counted
    them, which is every whole-image result it builds; otherwise found by a
    pass over the whole field, which is what this used to cost per click.

    :returns: the window, exclusive ends, or None when no pixel has the id.
    """
    label = int(label)
    extents = result.extents
    if extents is not None and 0 < label <= len(extents):
        found = extents[label - 1]
        if found is None:
            return None
        rows, cols = found
        return (int(cols.start), int(rows.start),
                int(cols.stop), int(rows.stop))
    body = np.asarray(result.labels) == label
    rows = np.flatnonzero(body.any(axis=1))
    cols = np.flatnonzero(body.any(axis=0))
    if not rows.size:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _magnifier_provenance(request: _MagnifierRequest, mode: str,
                          note: str = "") -> dict:
    """JSON-safe detector settings from a completed request, never the panel.

    ``mode`` is the algorithm that actually ran; ``request.mode`` and
    ``note`` preserve a model fallback. Common model controls retain their
    historical keys; ``method_parameters`` holds the applicable CPU or
    organelle settings. Display percentile normalization is separate from
    the model's ``normalize`` option. A whole-image object pick records the
    full detection box as well as its smaller paste box at the call site.
    Paste-time overlap and minimum area are added by the commit handler.
    """
    percentiles = request.detection_percentiles
    height, width = request.shape
    detail = {
        "mode": mode, "requested_mode": request.mode, "fallback_note": str(note),
        "scope": request.scope,
        "detection_box": ([0, 0, int(width), int(height)] if request.scope == "image"
                          else [int(v) for v in request.box]),
        "sensitivity": float(request.sensitivity), "bright": bool(request.bright),
        "min_area": int(request.min_area), "exclude_border": bool(request.exclude_border),
        "invert": bool(request.invert), "model": str(request.model_name),
        "flow_threshold": float(request.flow_threshold),
        "cellprob_threshold": float(request.cellprob_threshold),
        "diameter": int(request.diameter), "normalize": bool(request.normalize),
        "otsu_correction": float(request.otsu_correction),
        "detect_on_normalized": percentiles is not None,
        "normalization_percentiles": (None if percentiles is None else
                                       [float(v) for v in percentiles]),
        "method_parameters": (organelle_modes.provenance(mode, request.method_params)
                              or cpu_modes.provenance(mode, request.cpu_params)),
    }
    if mode == "otsu" or mode in cpu_modes.THRESHOLD_LABELS:
        multi = mode == cpu_modes.MULTIOTSU
        count = max(3, int(request.otsu_classes)) if multi else 2
        detail.update(
            otsu_smoothing=float(request.otsu_smoothing),
            otsu_fill_holes=bool(request.otsu_fill_holes),
            otsu_split=bool(request.otsu_split), otsu_classes=count,
            otsu_foreground_class=(count - 1 if request.otsu_foreground_class is None
                                   else int(request.otsu_foreground_class)) if multi else 1,
            otsu_local=False)
        if mode in engine.LOCAL_THRESHOLDS:
            detail["otsu_window"] = int(request.otsu_window)
    elif mode == cpu_modes.PROPAGATE:
        detail["otsu_fill_holes"] = bool(request.otsu_fill_holes)
    detail.update(detect_chain.provenance(
        request.chain or detect_chain.NO_CHAIN, percentile_stretch=percentiles is not None))
    if mode == cpu_modes.SECONDARY:
        detail['primary_source'] = request.primary_provenance
        detail['preserve_ids'] = True
        detail['otsu_fill_holes'] = bool(request.otsu_fill_holes)
    return detail


def _single_object(result: _MagnifierResult, label: int) -> _MagnifierResult:
    """One object of a whole-image result, as a result of its own.

    Cut to the object's bounding box, so a commit pastes that small window at
    its corner rather than the whole field. Every pixel of the object goes
    in, including any part outside the box on screen: the whole image was
    segmented, so nothing cut it.

    :param result: a whole-image result.
    :param label: an id present in ``result.labels``.
    """
    x0, y0, x1, y1 = _object_window(result, label)
    body = np.asarray(result.labels)[y0:y1, x0:x1] == int(label)
    labels = np.where(body, int(label), 0).astype(np.int32)
    request = result.request._replace(box=(x0, y0, x1, y1))
    return result._replace(request=request, labels=labels, overlay=None,
                           count=1, ghost=None, extents=None)


def _detect_cellpose_snapshot(request, models):
    """Prepare and segment a captured field without reading any Qt object."""
    image = request['image']
    if request['invert']:
        image = engine.invert_normalized(image)
    if request['percentiles'] is not None:
        image = engine.normalize_for_detection(image, *request['percentiles'])
    image = detect_chain.prepare(image, request['chain'])
    with _CELLPOSE_LOCK:
        name = request['model']
        if name not in models:
            models[name] = load_cellpose_model(name)
        labels, cellprob, flow = cellpose_detect(image, models[name], **request['parameters'])
    labels = detect_chain.finish(labels, request['chain'], intensity=image)
    return labels, cellprob, flow


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
        -- ``model_name``, ``diameter``, ``bright``, ``min_area``,
        ``flow_threshold``, ``cellprob_threshold``, ``normalize`` and
        ``otsu_correction`` -- read on the GUI thread whenever a request is
        built.
    """

    _delivered = Signal(object)
    #: The result a click was waiting on; the screen commits it.
    commit_ready = Signal(object)
    #: The wheel moved the zoom; carries the new value.
    zoom_changed = Signal(float)
    #: Shift + the wheel moved the size; carries the new value.
    size_changed = Signal(int)
    #: A field opened and the size's range is now ``(smallest, largest)``.
    size_range_changed = Signal(int, int)
    #: A sentence for the status line.
    status = Signal(str)
    #: The mask object under the mouse should go; carries image ``(x, y)``.
    remove_requested = Signal(int, int)
    #: A whole-image run started (True) or ended (False).
    busy_changed = Signal(bool)
    #: A press-and-drag's objects as ``(outcome, final)``: shown
    #: while the button is down, committed once when ``final``.
    drag_ready = Signal(object)
    locked_changed = Signal(bool)

    def __init__(self, canvas, parent=None, *, load_model=None, context=None):
        """Build a magnifier that is off and holds no thread."""
        super().__init__(parent)
        from ..bridge import emit_safely

        self._emit_safely = emit_safely
        self.canvas = canvas
        self.enabled = False
        self.locked = False
        self._lock_key_down = False
        self.mode = "otsu"
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
        self._said_error: Optional[str] = None
        self._said_diameter_note = False
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
        #: ``(result, (box, object under the mouse, rule, mask), picture,
        #: its pixels)`` last drawn; the pixels are what the picture reads.
        self._image_view: Optional[tuple] = None
        #: ``(result, (object, rule, mask), what the rule takes)`` -- see
        #: :meth:`_image_promise`.
        self._image_promised: Optional[tuple] = None
        #: What a click would add where the mask already has objects; the
        #: Overlap rule the screen's box is on. See :func:`_ghosted_overlay`.
        self.overlap = _MAGNIFIER_OVERLAP_DEFAULT
        #: The mask the canvas was last seen holding, and how many different
        #: ones it has held. See :meth:`mask_generation`.
        self._mask_seen = None
        self._mask_token = 0
        #: Whether a request for a ghost gone stale is already on its way,
        #: so a stream of repaints asks once rather than once each.
        self._asking = False
        #: :meth:`_stamp` of the request the box is waiting on, or None.
        self._requested_stamp: Optional[tuple] = None
        #: When the whole-image run on its way was handed to its worker, and
        #: how long it is expected to take -- None when nothing says.
        self._image_started: Optional[float] = None
        self._image_estimate: Optional[float] = None
        #: The :class:`_RunTicket` of the whole-image run on its way, which
        #: counts its tiles and is how that run is stopped.
        self._image_ticket: Optional[_RunTicket] = None
        #: ``(mode, model) -> seconds per megapixel`` from the last run
        #: that finished under it and was worth believing. See
        #: :meth:`_note_pace`, which says which runs those are, and
        #: ``_image_paced`` beside it, which is every pair measured at all.
        self._image_pace: dict = {}
        self._image_paced: set = set()
        #: Which field is on screen, as the screen names it -- the path of
        #: the image file. It identifies a field across a trip to another
        #: one and back, which the generation counter above cannot: that
        #: counts loads, so the same field twice is two numbers.
        self._field_name: str = ""
        #: ``(field name, settings) -> result`` for fields segmented whole
        #: already, and ``-> when it was last used`` beside it. See
        #: :meth:`_keep_image_result`.
        self._image_cache: dict = {}
        self._image_cache_used: dict = {}
        #: ``(the field's array, that field inverted)`` -- see
        #: :meth:`inverted_field`.
        self._inverted_field: Optional[tuple] = None
        self._busy = False
        self._worker = _NewestRequestWorker(self._run, self._hand_over)
        self._image_worker = _NewestRequestWorker(
            self._run, self._hand_over, name="spacr-magnifier-image")
        self._delivered.connect(self._on_delivered, Qt.QueuedConnection)
        self._init_stroke()

    def eventFilter(self, watched, event):
        """Track the held L in Ctrl+L+right-click without stealing edit keys."""
        from PySide6.QtCore import QEvent

        kind = event.type()
        if kind in (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate):
            self._lock_key_down = False
        elif kind == QEvent.KeyRelease and event.key() in (Qt.Key_L, Qt.Key_Control):
            if not event.isAutoRepeat():
                self._lock_key_down = False
        elif (kind == QEvent.KeyPress and event.key() == Qt.Key_L
              and event.modifiers() & Qt.ControlModifier
              and self.enabled and self.canvas.isVisible()
              and isinstance(watched, QWidget)
              and watched.window() == self.canvas.window()):
            self._lock_key_down = True
            return True
        return super().eventFilter(watched, event)

    def set_locked(self, locked: bool) -> None:
        """Pin or release the current image region, lens position, size and zoom.

        Locking needs an enabled lens over the image. Detector settings can
        still refresh this region. Disabling the lens or opening another
        field releases it. This transient viewing state is never saved.
        """
        from ..i18n import tr

        locked = bool(locked and self.enabled and self._cursor is not None)
        if locked == self.locked:
            return
        self.locked = locked
        self.locked_changed.emit(locked)
        self.status.emit(tr("Magnifier locked: Ctrl+L+right-click to unlock.")
                         if locked else tr("Magnifier unlocked."))
        self.canvas.update()


    def set_enabled(self, on: bool) -> None:
        """Turn the box on or off; objects already committed are untouched.

        Turning it on in whole-image scope starts a run for the field on
        screen, unless its objects are already found. Turning it off leaves
        a run in progress going, so its objects are ready when it comes back.
        """
        self.enabled = bool(on)
        if self.enabled:
            QApplication.instance().installEventFilter(self)
            self._image_halted = None
            self.refresh()
        else:
            QApplication.instance().removeEventFilter(self)
            self.set_locked(False)
            self._lock_key_down = False
            self._cursor = None
            self._anchor = None
        self.canvas.update()

    def set_mode(self, mode: str) -> None:
        """Choose the model, and forget which models failed to run.

        A mode this screen has renamed arrives under either name and is
        stored under the current one (:func:`canonical_magnifier_mode`), so
        a session or a script written under an old mode name still selects
        a mode that exists.
        """
        self.mode = canonical_magnifier_mode(mode)
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

    def set_overlap(self, rule: str) -> None:
        """Choose what a new object does where the mask already has one.

        The MODEL IS NOT ASKED AGAIN, and the whole-image objects are not
        discarded: the rule is applied to what was found, not by the thing
        that finds it. Only the box's promise of what a click would add is
        built afresh -- and it is built where the objects are, on the
        worker, which is why this asks for the region again rather than
        clearing a picture. The rule is not part of a request key, so the
        worker answers from the model's last answer for that box.

        :param rule: one of
            :data:`spacr.qt.mask_engine._MAGNIFIER_OVERLAP_RULES`; anything
            else leaves the rule where it is, because the box is about to
            draw a promise in its name.
        """
        name = str(rule or "")
        if name not in engine._MAGNIFIER_OVERLAP_RULES:
            return
        self.overlap = name
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

    def size_range(self) -> tuple:
        """``(smallest, largest)`` side for the field on screen; see
        :func:`_magnifier_size_range`."""
        image = self.canvas.image
        return _magnifier_size_range(None if image is None else image.shape)

    def set_size(self, size: int) -> None:
        """Set the region's side in image pixels, within its range."""
        if self.locked:
            return
        low, high = self.size_range()
        self.size = max(low, min(high, int(size)))
        self.refresh()
        self.canvas.update()

    def wheel_size(self, up: bool) -> int:
        """Step the size one Shift + wheel notch; return the new size.

        A notch moves the side by the canvas's own zoom per notch, and by at
        least one pixel, so a box of 32 px and one of 4,000 px both take
        about as many notches to double. The size stops at its range.
        """
        speed = max(1.001, float(getattr(self.canvas, "zoom_speed", 1.15)))
        step = max(1, int(round(self.size * (speed - 1.0))))
        self.set_size(self.size + step if up else self.size - step)
        self.size_changed.emit(self.size)
        return self.size

    def _sync_size_range(self) -> None:
        """Say the size's range for the field now, and keep the size inside it."""
        low, high = self.size_range()
        self.size_range_changed.emit(int(low), int(high))
        clamped = max(low, min(high, int(self.size)))
        if clamped != self.size:
            self.size = clamped
            self.size_changed.emit(self.size)

    def set_zoom(self, zoom: float) -> None:
        """Set the magnification, within its range. The model is not asked."""
        if self.locked:
            return
        low, high = _MAGNIFIER_ZOOM_RANGE
        self.zoom = max(low, min(high, float(zoom)))
        self.canvas.update()

    def set_sensitivity(self, value: float) -> None:
        """Set how readily an object is accepted, within its range."""
        low, high = _MAGNIFIER_SENSITIVITY_RANGE
        self.sensitivity = max(low, min(high, float(value)))
        self.refresh()
        self.canvas.update()

    def set_field(self, name: str) -> None:
        """Name the field about to be loaded, so its objects can be found again.

        Called by the screen BEFORE the pair reaches the canvas, because the
        canvas calls :meth:`forget` on the way in and the new name is what
        the next whole-image run is kept under.

        :param name: the image file's path; "" for a canvas handed an array
            with no file behind it, which is then never cached.
        """
        self._field_name = str(name or "")

    def forget(self) -> None:
        """Drop everything tied to the field on screen; another is loading.

        The whole-image objects are KEPT, in the cache rather than on the
        magnifier: coming back to this field offers them again instead of
        segmenting it a second time. See :meth:`_keep_image_result`.

        The canvas calls this with the new field already in place, which is
        what lets the size's range follow the field that has just opened.
        """
        self.set_locked(False)
        self._lock_key_down = False
        self._cursor = None
        self._anchor = None
        self._field += 1
        self._said_error = None
        self._said_diameter_note = False
        self._shown = None
        self._shown_image = None
        self._waiting.clear()
        self._requested_key = None
        self._requested_stamp = None
        self._stop_image()
        self._image_result = None
        self._image_view = None
        self._image_halted = None
        self._sync_size_range()

    def close(self) -> bool:
        """Stop both workers, waiting briefly for a model call in flight.

        The kept whole-image objects go with them: the screen is closing,
        and a label image per field is the largest thing this object holds.
        """
        QApplication.instance().removeEventFilter(self)
        if self._image_ticket is not None:
            self._image_ticket.cancel()
        region = self._worker.close()
        image = self._image_worker.close()
        self._image_cache.clear()
        self._image_cache_used.clear()
        return region and image


    def updating(self) -> bool:
        """Whether what the box draws is not yet what it was last asked for.

        The region asked for, and also the promise made about it: a rule
        change and an edit to the mask under the box both leave the ghost
        drawn against something that is no longer true, and the frame says
        so in the way it already says it -- dashed, with "Updating…" --
        until the worker answers again.

        In whole-image scope: whether the objects for the field and the
        settings now are not found yet.
        """
        if self.scope == "image":
            result = self._image_result
            return (result is None
                    or result.request.key != self._image_key_now())
        return (self._shown is None
                or self._stamp(self._shown.request) != self._requested_stamp
                or self._ghost_is_stale())

    def hover(self, pos) -> None:
        """Follow the mouse to widget point ``pos``; None puts the box away."""
        if self.locked:
            return
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
        self._requested_stamp = self._stamp(request)
        if (self._shown is not None
                and self._stamp(self._shown.request) == self._requested_stamp):
            return
        self._worker.submit(request)

    def _model_settings(self) -> tuple:
        """Everything a model reads, in :data:`_MODEL_SETTING_FIELDS` order.

        As it would read it now: the screen's own settings through
        ``context``, and Otsu in place of a mode that has already failed
        to load. Every setting is here whichever mode is chosen, because a
        model that cannot run hands the request to the Otsu mode, which
        must then find its own settings in it.
        """
        context = {"model_name": "cpsam", "diameter": 0, "bright": True,
                   "min_area": 0, "flow_threshold": FLOW_THRESHOLD,
                   "cellprob_threshold": CELLPROB_THRESHOLD,
                   "normalize": True, "otsu_correction": 1.0,
                   "otsu_smoothing": OTSU_SMOOTHING,
                   "otsu_fill_holes": True, "otsu_split": True,
                   "invert": False,
                   "chain": detect_chain.NO_CHAIN,
                   "method_params": organelle_modes.DEFAULT_PARAMS,
                   "cpu_params": cpu_modes.DEFAULT_PARAMS,
                   "otsu_window": OTSU_LOCAL_WINDOW,
                   "otsu_classes": 2, "otsu_foreground_class": None}
        if self._context is not None:
            context.update(self._context())
        model_name = str(context["model_name"])
        mode = canonical_magnifier_mode(self.mode)
        if (mode, model_name) in self._unavailable:
            mode = "otsu"
        return (mode, round(float(self.sensitivity), 4),
                bool(context["bright"]), int(context["min_area"]),
                model_name, int(context["diameter"]),
                round(float(context["flow_threshold"]), 4),
                round(float(context["cellprob_threshold"]), 4),
                bool(context["normalize"]),
                round(float(context["otsu_correction"]), 4),
                round(float(context["otsu_smoothing"]), 4),
                bool(context["otsu_fill_holes"]),
                bool(context["otsu_split"]),
                bool(context["invert"]),
                context["chain"],
                context["method_params"],
                context["cpu_params"],
                int(context["otsu_window"]),
                max(3, int(context["otsu_classes"])) if mode == cpu_modes.MULTIOTSU
                else int(context["otsu_classes"]),
                (None if context["otsu_foreground_class"] is None else
                 int(context["otsu_foreground_class"])),
                ((float(self.canvas.norm_lo), float(self.canvas.norm_hi))
                 if self.canvas.detect_on_normalized else None),
                context.get('primary_token', ()) if mode == cpu_modes.SECONDARY else ())

    def running_name(self) -> str:
        """What the box is running, as the Updating mark names it.

        The Cellpose model by its own name (``cpsam``,
        ``toxoplasma_plaque_v1``), since that is what the user chose and what
        the wait depends on; any other mode by the caption its Mode box shows
        (``Otsu``, ``DINOCell``). A mode that could not load is named as the
        Otsu it fell back to, because that is what is running.

        :returns: the name, in the current language where it is a caption.
        """
        settings = self._model_settings()
        if settings[0] == "cellpose":
            return str(settings[4])
        return _magnifier_mode_label(settings[0])

    @staticmethod
    def _accent() -> tuple:
        """The theme's accent colour as ``(red, green, blue)``."""
        accent = QColor(active_palette()["accent"])
        return (accent.red(), accent.green(), accent.blue())

    def mask_generation(self) -> int:
        """How many different masks the canvas has been handed, counted here.

        The canvas REBINDS its mask for every edit rather than writing into
        the one it has, so identity is what "the mask changed" means, and a
        counter over it is what a request can carry to the worker and a
        result can be compared against when it comes back. Reading it is
        the only place the change is noticed, which is why the box's own
        paint asks (:meth:`paint`).
        """
        mask = self.canvas.mask
        if mask is not self._mask_seen:
            self._mask_seen = mask
            self._mask_token += 1
        return self._mask_token

    def inverting(self) -> bool:
        """Whether Invert is on, as the screen's panel holds it now.

        Asked by :meth:`paint`, which runs outside the request path and so
        cannot read the answer off a request: the box shows the inverted
        image while Invert is on, and the box repaints on moves that
        never build a request at all.
        """
        if self._context is None:
            return False
        return bool(self._context().get("invert", False))

    def inverted_field(self) -> np.ndarray:
        """The WHOLE open field reflected about its own range, cached.

        The inversion is what the DETECTOR sees, so the
        box and the detect button have to invert the same way -- and a
        region reflected about ITS OWN extremes is reflected differently
        wherever the box is put. Inverting the whole field once and cutting
        from that is what keeps the box a preview of the button, and it is
        ONE array that both the painted picture and the request's crop come
        off, so the two cannot disagree about which way up the field was.

        Computed once per field and not once per mouse move, keyed on the
        array's identity: a full-field pass on every hover is work on the
        GUI thread for an answer that cannot have changed, and another field
        is another array and asks again. The canvas caches its
        display complement against its own image the same way.
        """
        image = self.canvas.image
        cached = self._inverted_field
        if cached is not None and cached[0] is image:
            return cached[1]
        out = engine.invert_normalized(image)
        self._inverted_field = (image, out)
        return out

    def detector_field(self) -> np.ndarray:
        """The field the box magnifies: enhanced, inverted, or the canvas's own.

        With "Show the enhanced image" on it is the canvas's own enhanced
        picture (:meth:`_MaskCanvas.enhanced_picture`), so the box and the
        canvas under it show one image and a curator comparing them is
        comparing the same chain. The box's DETECTION still enhances the
        box's own region, on the worker; the difference between the two is
        the difference between a CLAHE tile grid laid over a field and one
        laid over a box, and it is why the Compare window shows the box's
        own region.
        """
        if self.canvas.enhance_display:
            return self.canvas.enhanced_picture()
        if self.canvas.detect_on_normalized:
            return self.canvas.detection_base()
        if not self.inverting():
            return self.canvas.image
        return self.inverted_field()

    def refresh_view(self) -> None:
        """Repaint the box because the PICTURE under it changed.

        Not :meth:`refresh`, which asks the model again: switching the
        enhanced view on and off changes what is drawn and not what was
        detected, and the objects on screen are still the objects the
        settings ask for.
        """
        self.canvas.update()

    def compare_box(self) -> tuple:
        """The region a raw-versus-enhanced comparison is to show.

        THE BOX UNDER THE MOUSE when there is one, because that is the
        region the chain really ran on for the detection that is on screen;
        the whole field otherwise, which is the region the detect buttons
        run it on.

        :returns: ``(x0, y0, x1, y1)`` in image pixels.
        """
        image = self.canvas.image
        height, width = (int(v) for v in image.shape[:2])
        if self.enabled and self._cursor is not None:
            return engine._magnifier_box(image.shape, self._cursor[0],
                                         self._cursor[1], self.size)
        return (0, 0, width, height)

    def region_for(self, box, *, invert: bool) -> np.ndarray:
        """A copy of ``box`` of the open field, inverted if Invert is on.

        The one place the box's pixels are taken, so what the model is given
        and what the box paints cannot disagree about whether they were
        inverted.

        THE ENHANCEMENT CHAIN IS NOT APPLIED HERE. This runs on the GUI
        thread, once per mouse move, and a non-local means or a wide
        background radius over a region is not something to do between two
        frames; :func:`_segment_region` applies it on the worker, to this
        crop, which is also what makes every step live on the box. The
        percentile stretch is the exception and is already in
        :meth:`_MaskCanvas.detection_base`, because its levels are the
        whole field's.

        The inverted case is a slice of :meth:`inverted_field`, which uses
        :func:`mask_engine.invert_for_detection` and NOT
        :func:`mask_engine.invert_intensity`: the crop is about to be
        thresholded, and the second moves the field's span in a way the
        Otsu threshold correction cannot survive. The first function's
        docstring carries the measurement.

        :param box: ``(x0, y0, x1, y1)`` in image pixels.
        :param invert: whether Invert for detection is on.
        """
        x0, y0, x1, y1 = box
        if self.canvas.detect_on_normalized:
            source = self.canvas.detection_base()
        else:
            source = self.inverted_field() if invert else self.canvas.image
        return np.array(source[y0:y1, x0:x1], copy=True)

    def build_request(self, *, ghost: bool = True
                      ) -> Optional[_MagnifierRequest]:
        """The request for the region under the mouse now, or None.

        :param ghost: whether the worker is to draw what the Overlap rule
            would leave as well as what the model found. False for a drag's
            frames, which are never drawn as a box: the mask crop would be
            copied and the rule counted over it for a picture nobody sees.
        """
        canvas = self.canvas
        image = canvas.image
        if (not self.enabled or self._cursor is None or image is None
                or canvas.mask is None):
            return None
        settings = self._model_settings()
        values = dict(zip(_MODEL_SETTING_FIELDS, settings))
        box = engine._magnifier_box(image.shape, self._cursor[0],
                                    self._cursor[1], self.size)
        x0, y0, x1, y1 = box
        exclude = bool(self.exclude_border)
        rule = _canonical_overlap_rule(self.overlap)
        occupied = None
        token = 0
        if ghost and rule != "replace":
            token = self.mask_generation()
            occupied = (np.array(canvas.mask[y0:y1, x0:x1], copy=True)
                        if values['mode'] == cpu_modes.SECONDARY else
                        np.asarray(canvas.mask)[y0:y1, x0:x1] > 0)
        primary = self._primary_request_values(box, values)
        if primary is None:
            return None
        return _MagnifierRequest(
            key=(self._field, box) + settings + (exclude,),
            crop=self.region_for(box, invert=values["invert"]),
            box=box,
            shape=tuple(int(v) for v in image.shape[:2]),
            colour=self._accent(),
            exclude_border=exclude,
            overlap=rule if ghost else "replace",
            occupied=occupied,
            mask_token=token,
            **primary,
            **values,
        )

    def _primary_request_values(self, box, settings):
        """Snapshot this field's primary crop and provenance for secondary mode."""
        if settings['mode'] != cpu_modes.SECONDARY:
            return {}
        context = self._context() if self._context is not None else {}
        source = context.get('primary_source')
        if source is None or source.identity != settings['primary_token']:
            return None
        return {'primary_labels': source.crop(box),
                'primary_provenance': dict(source.provenance(),
                                           selection=context.get('primary_selection', source.path))}

    @staticmethod
    def _stamp(request: _MagnifierRequest) -> tuple:
        """What makes a result the one the box wants NOW.

        The request key says which region the model was asked about; the
        rule and the mask say what its answer means. A click while the mask
        under the box has changed must not be answered from a picture drawn
        against the mask before it.
        """
        return (request.key, request.overlap, request.mask_token)

    def click(self) -> bool:
        """Commit the objects for the region under the mouse.

        In whole-image scope, commit only the object under the mouse; see
        :meth:`_pick`.

        THE RESULT ON SCREEN IS MATCHED BY ITS KEY AND NOT BY :meth:`_stamp`.
        What a click commits is the model's objects; the Overlap rule is
        applied to them as they go in, by the screen, from the Overlap box as
        it reads at that moment. A box whose ghost is one mask edit behind is
        still drawn from the right objects, and making the click wait for a
        picture would delay the edit itself for the sake of a promise about
        it.

        :returns: False when there is no region to commit -- the magnifier is
            off, the mouse is off the image, or no field is open.
        """
        from ..i18n import tr

        if self.scope == "image":
            return self._pick()
        request = self.build_request()
        if request is None:
            return False
        self._requested_key = request.key
        self._requested_stamp = self._stamp(request)
        if self._shown is not None and self._shown.request.key == request.key:
            self.commit_ready.emit(self._shown)
            return True
        self._waiting.add(request.key)
        self._worker.submit(request, pin=True)
        self.status.emit(tr(
            "Magnifier: segmenting this region — its objects are added as "
            "soon as the box is up to date."))
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

    def _cache_key(self, key: Optional[tuple]) -> Optional[tuple]:
        """The cache's name for a run key: the FIELD and what a model reads.

        A run key opens with the generation counter, which counts loads
        rather than fields, so the same field opened twice carries two
        different keys and would never match itself.

        :param key: a key from :meth:`_image_key_now`.
        :returns: None when no field is named, so nothing is kept for a
            canvas that was handed an array without one.
        """
        if key is None or not self._field_name:
            return None
        return (self._field_name,) + tuple(key[2:])

    def _keep_image_result(self, result) -> None:
        """Keep a finished whole-image answer for when the field comes back.

        A field this session has already segmented whole is minutes of
        Cellpose on a CPU; leaving it and coming back paid that again. What
        is kept is the label image, which is what a click reads, and it is
        valid for exactly the settings it was found under: the key carries
        them, so a run under a new Sensitivity neither matches nor evicts
        the old one.
        """
        key = self._cache_key(result.request.key)
        if key is None:
            return
        self._image_cache[key] = result
        self._image_cache_used[key] = time.time()
        self._trim_image_cache()

    def _cached_image_result(self, key: tuple):
        """The kept answer for run ``key``, with its key moved to this load.

        :returns: a result whose request carries ``key``, so everything that
            compares the two goes on doing it, or None.
        """
        name = self._cache_key(key)
        if name is None:
            return None
        result = self._image_cache.get(name)
        if result is None:
            return None
        self._image_cache_used[name] = time.time()
        return result._replace(request=result.request._replace(key=key))

    def _trim_image_cache(self) -> None:
        """Drop what the memory budget says must go, least recently used first.

        The policy is :func:`spacr.qt.memory_budget.what_to_drop`, which is
        what every other cache in the application is trimmed by: the user's
        own idle timeout releases a field nobody has gone back to, and the
        lower of the user's cache ceiling and
        :data:`_MAGNIFIER_IMAGE_CACHE_MB` bounds the rest. The field on
        screen is held by :attr:`_image_result` as well, so a trim can never
        take the objects out from under the box.
        """
        from ..memory_budget import what_to_drop

        ceiling = _MAGNIFIER_IMAGE_CACHE_MB
        try:
            from ..preferences import get_cache_ceiling_mb

            ceiling = min(ceiling, int(get_cache_ceiling_mb()))
        except Exception:                                    # noqa: BLE001
            pass
        names = list(self._image_cache)
        entries = [(index,
                    self._image_cache[name].nbytes() / 1e6,
                    self._image_cache_used.get(name, 0.0))
                   for index, name in enumerate(names)]
        for dropped in what_to_drop(entries, time.time(), ceiling_mb=ceiling):
            name = names[int(dropped)]
            self._image_cache.pop(name, None)
            self._image_cache_used.pop(name, None)

    def _refresh_image(self) -> None:
        """Start a whole-image run, unless one for the settings now is found,
        on its way, or was cancelled.

        A field segmented whole earlier in the session, under the settings
        now, is taken from the cache instead of being segmented again.
        """
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
        kept = self._cached_image_result(key)
        if kept is not None:
            self._stop_image()
            self._image_result = kept
            self._image_view = None
            self.status.emit(tr(
                "Magnifier: {n} object(s) found in the whole image. Click one "
                "to add it; right-click an object in the mask to remove it.",
                n=kept.count))
            self.canvas.update()
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
        """Hand a copy of the whole field to the image worker under ``key``.

        Inverted first when Invert is on, like every other request: the
        whole field IS the region here, so it is the same call.
        """
        image = self.canvas.image
        height, width = (int(v) for v in image.shape[:2])
        values = dict(zip(_MODEL_SETTING_FIELDS, key[2:]))
        if (str(values.get("mode") or "").startswith("cellpose3")
                and not values.get("diameter")
                and not self._said_diameter_note):
            self._said_diameter_note = True
            self.status.emit(_cellpose3_auto_diameter_note())
        primary = self._primary_request_values((0, 0, width, height), values)
        if primary is None:
            from ..i18n import tr
            self.status.emit(tr('Load a primary mask before growing secondary objects.'))
            self._image_halted = key
            return
        if self._image_ticket is not None:
            self._image_ticket.cancel()
        self._image_ticket = _RunTicket()
        self._image_key = key
        self._image_halted = None
        self._image_started = time.monotonic()
        pace = self._image_pace.get(self._pace_key(key))
        self._image_estimate = (None if pace is None
                                else pace * height * width / 1e6)
        self._image_worker.submit(_MagnifierRequest(
            key=key,
            crop=self.region_for((0, 0, width, height),
                                  invert=values["invert"]),
            box=(0, 0, width, height), shape=(height, width),
            colour=self._accent(), exclude_border=False, scope="image",
            ticket=self._image_ticket, **primary, **values))
        self._set_busy(True)

    @staticmethod
    def _pace_key(key: tuple) -> tuple:
        """What a run's duration is worth remembering against: mode and model.

        Not the field and not its size: the whole point is to answer for a
        field nothing has been measured on, and seconds per megapixel is what
        carries across. Sensitivity and Min area are left out because they
        move the objects found rather than the work done.
        """
        return (key[2], key[6])

    def remaining_seconds(self) -> Optional[float]:
        """How long the whole-image run on its way still has, or None.

        A GUESS FROM A MEASUREMENT, and only ever from one: the seconds per
        megapixel the last run under this mode and model took, times this
        field's megapixels -- see :meth:`_note_pace` for which run that is
        allowed to be. Before there is such a measurement the answer is
        None and the bar stays indeterminate, which is the honest answer.

        None once the estimate is spent, too, so a bar that has run out goes
        back to saying only that something is happening.

        :returns: seconds, never below zero, or None.
        """
        if (not self._busy or self._image_started is None
                or self._image_estimate is None):
            return None
        left = self._image_estimate - (time.monotonic() - self._image_started)
        return left if left > 0 else None

    def image_progress(self) -> Optional[tuple]:
        """``(tiles started, tiles in all, seconds left or None)``, or None.

        What the whole-image run on its way has itself counted, which a
        model that tiles -- Cellpose -- does from its first tile
        (:class:`_RunTicket`). None for a run that counts nothing (Otsu, a
        backend out of process), for which :meth:`remaining_seconds` is the
        only estimate there is.
        """
        ticket = self._image_ticket
        if not self._busy or ticket is None or ticket.total <= 0 \
                or ticket.done <= 0:
            return None
        return (min(ticket.done, ticket.total), ticket.total,
                ticket.remaining_seconds())

    def estimated_seconds(self) -> Optional[float]:
        """The whole run's estimate, or None when nothing was estimated.

        What :meth:`remaining_seconds` is counting down FROM, which a bar
        needs as well as the remainder to show a fraction. Public for the
        same reason the remainder is: the screen draws the bar, and a
        screen reaching into this object for the other half of one answer
        was reaching past the answer it had just been given.
        """
        return self._image_estimate

    def _note_pace(self, key: tuple, pixels: int, seconds: float) -> None:
        """Remember what this mode and model cost per megapixel, last time.

        A RUN THAT LOADED ITS MODEL IS NOT WHAT THE NEXT RUN COSTS. The
        same field measures 10.3 s cold and 3.9 s warm on the
        same Cellpose model: most of a first run is the load, and a bar
        that counted down from it would promise two and a half times the
        time the run it is drawn over actually takes, then finish while it
        still said seven seconds left. A mode that loads a model therefore
        spends its first measurement of a session learning that the model
        is now in memory -- the bar stays indeterminate through the second
        run, and counts down from the third. The Otsu mode loads nothing,
        so its first run is its pace and its second run counts down.

        :param key: the whole-image request key the run answered.
        :param pixels: how many pixels were segmented.
        :param seconds: how long it took, from handing it over to its
            result arriving.
        """
        if pixels <= 0 or seconds <= 0:
            return
        pace_key = self._pace_key(key)
        first = pace_key not in self._image_paced
        self._image_paced.add(pace_key)
        if first and canonical_magnifier_mode(pace_key[0]) != "otsu":
            return
        self._image_pace[pace_key] = seconds / (pixels / 1e6)

    def _stop_image(self) -> None:
        """Forget the whole-image run on its way, and stop it.

        One not started yet is dropped; one running is asked to stop at the
        model's next tile (:class:`_RunTicket`), which frees the processor
        and the model lock within one tile rather than at the end of the
        run. A model that does not tile finishes, and its answer is thrown
        away when it arrives, as before.
        """
        if self._image_ticket is not None:
            self._image_ticket.cancel()
            self._image_ticket = None
        self._image_key = None
        self._image_started = None
        self._image_estimate = None
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


    def _init_stroke(self) -> None:
        """Hold no press, and start in the mode that adds every object in the box."""
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
        self._blocked_stroke_press = False
        self._stroke_timer = QTimer(self)
        self._stroke_timer.setSingleShot(True)
        self._stroke_timer.setInterval(_PREVIEW_MS)
        self._stroke_timer.timeout.connect(self._stroke_show)
        self._delivered.connect(self._stroke_delivered, Qt.QueuedConnection)

    def press(self) -> bool:
        """Start a stroke under the mouse: what a left press does while on.

        Nothing is added on the press. A release that has not moved is a
        click -- :meth:`click` -- except that with only
        objects touching the mouse it adds just the object under the cursor.
        A press that pulls is a drag; what a drag adds is
        :mod:`spacr.qt._magnifier_drag`'s to say. Under Whole image a drag
        reads the objects already found and asks no model; before they are
        found a press starts nothing, and its release is a click that says so.
        A previous stroke waiting for segmentation keeps ownership until it
        commits; a press during that wait and its matching release are ignored.

        :returns: False when no stroke started.
        """
        from .._magnifier_drag import _DragStroke, _frame_step
        from ..i18n import tr

        self._blocked_stroke_press = bool(
            self._stroke is not None and self._stroke_from[1] == self._field)
        if self._blocked_stroke_press:
            self.status.emit(tr(
                "Magnifier: segmenting the last regions — the objects are "
                "added as soon as they are done."))
            return False
        self._stroke = None
        canvas = self.canvas
        if (not self.enabled or self._cursor is None or canvas.image is None
                or canvas.mask is None):
            return False
        if self.mode == cpu_modes.SECONDARY:
            return True
        whole = self.scope == "image"
        found = self._image_result
        if whole and (found is None
                      or found.request.key != self._image_key_now()):
            return False
        stroke = _DragStroke(
            canvas.image.shape, self._cursor,
            step=0 if whole else _frame_step(self.size),
            keep_untouched=not whole and self.save_mode != "touching",
            provenance={"save": self.save_mode})
        self._stroke = stroke
        self._stroke_from = (QPointF(self._anchor), self._field)
        self._stroke_moved = False
        if whole:
            stroke.expect("image")
            stroke.deliver("image", found.labels, found.request.box,
                           provenance=_magnifier_provenance(found.request, found.mode, found.note))
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
        if self._blocked_stroke_press:
            return
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

        if self._blocked_stroke_press:
            self._blocked_stroke_press = False
            return False
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
        request = self.build_request(ghost=False)
        self._cursor = cursor
        self._stroke.expect(request.key)
        shown = self._shown
        if shown is not None and shown.request.key == request.key:
            self._stroke.deliver(request.key, shown.labels, request.box,
                                 provenance=_magnifier_provenance(shown.request, shown.mode, shown.note))
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
        elif stroke.deliver(request.key, result.labels, request.box,
                            provenance=_magnifier_provenance(result.request, result.mode, result.note)):
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


    def _run(self, request: _MagnifierRequest) -> _MagnifierResult:
        """Segment one request. Runs on a worker thread, never the GUI's.

        A region asked for again with only the border option changed reuses
        the model's last answer, which is kept here on the region worker.
        """
        if request.scope == "image":
            labels, used, note = self._segment_now(request)
            return _MagnifierResult(
                request, labels, used, note,
                _candidate_overlay(labels, request.colour),
                _object_count(labels), extents=_object_extents(labels))
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
        overlay = _candidate_overlay(labels, request.colour)
        return _MagnifierResult(request, labels, used, note, overlay,
                                _object_count(labels),
                                _ghosted_overlay(labels, overlay, request))

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
        """Show a finished result, and commit it if a click was waiting.

        The picture the box draws is the result's ghost when it has one,
        which is the whole picture and not a layer over the outlines: it IS
        the outlines, with what the Overlap rule would not add faded. One
        QImage is built per result either way.

        A FAILURE IS SAID ONCE (item 507). While a whole-field enhancement
        runs, every region fails with the same "enhancement is updating"
        reason, and the magnifier asks for a region on every mouse move; the
        reason used to be said, and logged, each time. It is said again only
        when it changes, after a region succeeds, or on a new field.
        """
        from ..i18n import tr

        request, result, error = payload
        if request.key[0] != self._field:
            return
        if request.scope == "image":
            self._on_image_delivered(request, result, error)
            return
        if error is not None:
            self._waiting.discard(request.key)
            if request.key == self._requested_key:
                self._shown = None
                self._shown_image = None
                self.canvas.update()
            if str(error) != self._said_error:
                self._said_error = str(error)
                LOG.warning("magnifier could not segment %s: %s",
                            request.box, error)
                self.status.emit(tr(
                    "Magnifier could not segment this region: {error}",
                    error=error))
            return
        self._said_error = None
        self._note_fallback(request, result)
        self._shown = result
        self._shown_image = _rgba_qimage(
            result.overlay if result.ghost is None else result.ghost)
        if request.key in self._waiting:
            self._waiting.discard(request.key)
            self.commit_ready.emit(result)
        self.canvas.update()

    def _note_fallback(self, request, result) -> bool:
        """Say, once per model, that it could not run and Otsu stood in.

        The mode is named the way the Mode box names it, not by its internal
        key: a sentence about ``cellpose3:cyto3`` sends the reader looking for
        a row that says "Cellpose 3 · cyto3".

        :returns: True when this call said it.
        """
        from ..i18n import tr

        if result.mode == request.mode:
            return False
        marker = (request.mode, request.model_name)
        if marker in self._unavailable:
            return False
        self._unavailable[marker] = result.note
        self.status.emit(tr(
            "Magnifier: {mode} could not run ({reason}); the Otsu mode is "
            "segmenting instead.",
            mode=_magnifier_mode_label(request.mode), reason=result.note))
        return True

    def _on_image_delivered(self, request, result, error) -> None:
        """Keep a finished whole-image run, unless it is no longer wanted."""
        from ..i18n import tr

        if request.key != self._image_key:
            return
        started, self._image_started = self._image_started, None
        self._image_key = None
        self._image_estimate = None
        if request.ticket is self._image_ticket:
            self._image_ticket = None
        self._set_busy(False)
        if isinstance(error, _RunCancelled):
            return
        if error is None and started is not None:
            height, width = (int(v) for v in request.shape[:2])
            self._note_pace(request.key, height * width,
                            time.monotonic() - started)
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
            stored = stored._replace(key=key)
        self._image_result = result._replace(request=stored)
        self._image_view = None
        self._keep_image_result(self._image_result)
        if not noted:
            self.status.emit(tr(
                "Magnifier: {n} object(s) found in the whole image. Click one "
                "to add it; right-click an object in the mask to remove it.",
                n=result.count))
        self.canvas.update()


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
        turns dashed with an "Updating <model>…" mark until the result for this
        region arrives -- the box never blanks while it waits.

        In whole-image scope the box draws its slice of the whole-image
        objects instead, with the object a click would add filled more
        strongly and what the Overlap rule would take from it ghosted, and
        the frame is dashed while a run is on its way.

        NOTHING IS COMPUTED FROM PIXELS HERE beyond the magnified region
        itself: the picture the box draws over it was built on the worker,
        ghosted and all (:func:`_ghosted_overlay`). What this does notice
        is a mask it has not been drawn against -- the canvas rebinds its
        mask for every edit -- and it asks for the region again rather than
        redrawing a promise made about a mask that is gone.

        WITH INVERT ON THE BOX SHOWS THE INVERTED REGION,
        off the same :meth:`inverted_field` the
        request's crop is cut from, so what the user is looking at inside
        the box is what the model was given. The canvas under it is
        untouched.
        """
        geometry = self.lens_geometry()
        if geometry is None:
            return
        box, lens, scale = geometry
        x0, y0, x1, y1 = box
        canvas = self.canvas
        part, area = self._visible_part(box, lens, scale)
        if part is None:
            return
        vx0, vy0, vx1, vy1 = part
        step = self._picture_step(scale)
        rgb = _box_picture(self.detector_field(), box, part,
                           canvas.mask[vy0:vy1, vx0:vx1],
                           canvas.norm_lo, canvas.norm_hi, step)
        height, width = rgb.shape[:2]
        picture = QImage(rgb.data, width, height, 4 * width,
                         QImage.Format_RGB32)
        if step > 1:
            area = QRectF(area.left(), area.top(), width * step * scale,
                          height * step * scale)
        palette = active_palette()
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.setClipRect(lens)
        painter.drawImage(area, picture)
        if self.scope == "image":
            self.mask_generation()
            view = self._image_slice(part)
            if view is not None:
                painter.drawImage(area, view)
            updating = self._image_key is not None
        else:
            self.mask_generation()
            if self._ghost_is_stale() and not self._asking:
                self._asking = True
                QTimer.singleShot(0, self, self._ask_again)
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
        if updating or self.locked:
            from ..i18n import tr

            caption = _updating_caption(self.running_name()) if updating else ""
            if self.locked:
                caption = tr("Locked") + (" · " + caption if caption else "")
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

    def _picture_step(self, scale: float) -> int:
        """How many image pixels one pixel of the box's picture stands for.

        One whenever the lens magnifies -- a device pixel or more per image
        pixel -- which is every zoom at which the box is used to look
        closely. Below that, Qt drawing the picture unsmoothed would keep
        one image pixel in every ``1 / (scale x device ratio)`` anyway, so
        :func:`_box_picture` does not make the others.

        :param scale: widget pixels per image pixel inside the lens.
        """
        ratio_of = getattr(self.canvas, "devicePixelRatioF", None)
        ratio = float(ratio_of()) if callable(ratio_of) else 1.0
        device = float(scale) * (ratio or 1.0)
        if device <= 0 or device >= 1.0:
            return 1
        return max(1, int(1.0 / device))

    def _visible_part(self, box, lens, scale):
        """The part of ``box`` that is on the canvas, and where it is drawn.

        THE BOX IS NOT BOUNDED BY THE WINDOW. Its side is the Size box's
        value in image pixels and may be as large as the field's own
        longer side, and the lens then draws it ``zoom`` times larger
        again: on a 2,048 px field most of the lens is off the widget, and
        every pixel of it was being stretched, coloured and handed to Qt on
        the GUI thread for every move. Measured by the responsiveness
        harness at 2,048 px: a median move of 183.9 ms, 23 frames dropped.

        :param box: ``(x0, y0, x1, y1)`` in image pixels.
        :param lens: where the whole box would be drawn.
        :param scale: widget pixels per image pixel inside the lens.
        :returns: ``(part, rect)``, or ``(None, None)`` when none of the
            box is on the canvas.
        """
        x0, y0, x1, y1 = (int(v) for v in box)
        area = lens.intersected(QRectF(self.canvas.rect()))
        if area.isEmpty() or scale <= 0:
            return None, None
        left = max(x0, x0 + int(math.floor((area.left() - lens.left()) / scale)))
        top = max(y0, y0 + int(math.floor((area.top() - lens.top()) / scale)))
        right = min(x1, x0 + int(math.ceil((area.right() - lens.left()) / scale)))
        bottom = min(y1, y0 + int(math.ceil((area.bottom() - lens.top()) / scale)))
        if right <= left or bottom <= top:
            return None, None
        rect = QRectF(lens.left() + (left - x0) * scale,
                      lens.top() + (top - y0) * scale,
                      (right - left) * scale, (bottom - top) * scale)
        return (left, top, right, bottom), rect

    def _ghost_is_stale(self) -> bool:
        """Whether the box promises against a mask the canvas no longer has.

        Asks nothing of the canvas: it compares what the shown result was
        built against with the number :meth:`mask_generation` has reached,
        and it is :meth:`paint` that brings that number up to date -- which
        is where a mask edit is first seen, every edit rebinding the mask
        and repainting, and no single place on the screen owning all of
        them. Under Replace nothing is ghosted, so nothing goes stale.
        """
        shown = self._shown
        return (shown is not None and shown.request.overlap != "replace"
                and shown.request.mask_token != self._mask_token)

    def _ask_again(self) -> None:
        """Ask for the region under the mouse and redraw when it lands."""
        self._asking = False
        if self.enabled:
            self.refresh()
            self.canvas.update()

    def _image_slice(self, box) -> Optional[QImage]:
        """The whole-image objects inside ``box``, outlined, as a picture.

        ``box`` is the part of the lens that is on the canvas
        (:meth:`_visible_part`) and not necessarily the whole region the box
        magnifies, for the reason given there: what is off the widget costs
        the same to build and shows nobody anything.

        The outlines are CUT FROM THE WORKER'S PICTURE of the whole field
        (:attr:`_MagnifierResult.overlay`), not drawn again here: at the
        largest box, outlining the part of the field on the canvas was a
        boundary pass over a megapixel and more per move.

        The object under the mouse -- the one a click would add -- is filled
        more strongly than the rest, and THE OVERLAP RULE'S ANSWER FOR IT IS
        DRAWN THE WAY THE REGION MODE DRAWS ITS OWN: what a click would add
        is solid, what the rule would take away keeps a quarter of its
        alpha. See :meth:`_image_promise`. The picture is kept until the
        box, that object, the rule, the mask or the objects themselves
        change, so repainting without moving recomputes nothing -- and
        nothing here calls a model.
        """
        result = self._image_result
        if result is None or self._cursor is None:
            return None
        x0, y0, x1, y1 = (int(v) for v in box)
        cx, cy = self._cursor
        labels = result.labels
        under = int(labels[cy, cx])
        rule = _canonical_overlap_rule(self.overlap)
        where = ((x0, y0, x1, y1), under, rule,
                 self._mask_token if rule != "replace" else 0)
        cached = self._image_view
        if cached is not None and cached[0] is result and cached[1] == where:
            return cached[2]
        if result.overlay is not None:
            rgba = np.array(result.overlay[y0:y1, x0:x1], copy=True)
        else:
            rgba = _candidate_overlay(labels[y0:y1, x0:x1],
                                      result.request.colour)
        window = _object_window(result, under) if under > 0 else None
        if window is not None:
            wx0, wy0, wx1, wy1 = window
            ix0, iy0 = max(x0, wx0), max(y0, wy0)
            ix1, iy1 = min(x1, wx1), min(y1, wy1)
            if ix1 > ix0 and iy1 > iy0:
                body = labels[iy0:iy1, ix0:ix1] == under
                alpha = rgba[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0, 3]
                alpha[body & (alpha < 140)] = 140
                lost = self._image_promise(result, under)
                if lost is not None:
                    gone = lost[iy0 - wy0:iy1 - wy0, ix0 - wx0:ix1 - wx0]
                    alpha[gone] = alpha[gone] // 4
        rgba = np.ascontiguousarray(rgba)
        height, width = rgba.shape[:2]
        picture = QImage(rgba.data, width, height, 4 * width,
                         QImage.Format_RGBA8888)
        self._image_view = (result, where, picture, rgba)
        return picture

    def _image_promise(self, result: _MagnifierResult,
                       label: int) -> Optional[np.ndarray]:
        """What the Overlap rule would take from whole-image object ``label``.

        A whole-image click adds ONE WHOLE OBJECT, and most of it can lie
        outside the box, so the rule's answer cannot be read off the box's
        slice: Skip asks whether the object touches the mask ANYWHERE, and
        Clip keeps its largest surviving piece, which may be off the box.
        The answer is therefore computed over the object's own bounding box
        (:func:`_object_window`, from the worker's one ``find_objects``
        pass), against the mask there, by
        :func:`spacr.qt.mask_engine._surviving_region_objects` -- the same
        function, over the same window, that the click's paste runs
        (:func:`_single_object`, then
        :func:`spacr.qt.mask_engine._paste_region_objects`), so the promise
        and the edit cannot disagree.

        Its cost is the object's bounding box, once per object, rule and
        mask: kept in :attr:`_image_promised`.

        :param result: the whole-image result on screen.
        :param label: the object under the mouse.
        :returns: a bool array over the object's window, True where a click
            would NOT add the pixel, or None when the rule takes nothing --
            under Replace, over an empty window, or when all of it survives.
        """
        rule = _canonical_overlap_rule(self.overlap)
        mask = self.canvas.mask
        if rule == "replace" or mask is None \
                or tuple(mask.shape[:2]) != tuple(result.labels.shape[:2]):
            return None
        key = (label, rule, self._mask_token)
        cached = self._image_promised
        if cached is not None and cached[0] is result and cached[1] == key:
            return cached[2]
        lost = None
        window = _object_window(result, label)
        if window is not None:
            x0, y0, x1, y1 = window
            occupied = np.asarray(mask)[y0:y1, x0:x1] > 0
            if occupied.any():
                single = _single_object(result, label).labels
                exact_ids = result.mode == cpu_modes.SECONDARY
                if exact_ids:
                    occupied &= np.asarray(mask)[y0:y1, x0:x1] != single
                kept = engine._surviving_region_objects(
                    single, occupied, overlap=rule,
                    min_area=int(result.request.min_area), preserve_ids=exact_ids)
                gone = (single > 0) & (kept == 0)
                lost = gone if gone.any() else None
        self._image_promised = (result, key, lost)
        return lost


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


class _ThresholdHistogramRequest(NamedTuple):
    """A whole-field histogram snapshot, independent of later panel edits.

    ``image`` is a private copy of the loaded pixels; ``invert`` and optional
    ``normalization`` (low/high percentiles) precede ``chain``. ``settings``
    copies the effective Otsu-category settings, including smoothing and
    correction. ``mode`` names the selected threshold and ``bright`` its
    polarity. ``ticket`` cancels between preparation and threshold stages;
    an active NumPy/scikit-image call finishes before cancellation is read.
    """

    image: np.ndarray
    mode: str
    settings: dict
    bright: bool
    invert: bool
    normalization: Optional[tuple]
    chain: Any
    ticket: Any


def _threshold_histogram(request: _ThresholdHistogramRequest) -> tuple:
    """Compute detector-input counts and global levels on a worker thread.

    Uses the canvas's inversion, normalization and enhancement order, then
    the threshold engine's smoothing and level calculation. Local methods
    return no global marker: their per-pixel thresholds cannot be represented
    by one vertical line. Their histogram is the smoothed input before local
    thresholding (and before Local Otsu's internal 8-bit rank-filter scaling).

    :returns: ``(counts, edges, levels, local)``. Algorithm errors propagate;
        the preview must never silently replace the selected method by Otsu.
    """
    request.ticket.check()
    image = request.image
    if request.invert:
        image = engine.invert_normalized(image)
    if request.normalization is not None:
        image = engine.normalize_for_detection(image, *request.normalization)
    image = detect_chain.prepare(image, request.chain, cancel=request.ticket.cancelled)
    request.ticket.check()
    settings = request.settings
    values = engine._otsu_values(image, settings["smoothing"])
    counts, edges = engine._otsu_histogram(values, bins=OTSU_HISTOGRAM_BINS)
    local = settings["local"] or request.mode in engine.LOCAL_THRESHOLDS
    levels = [] if local else engine._otsu_levels(
        values, bright=request.bright, correction=settings["correction"],
        classes=settings["classes"],
        algorithm=cpu_modes.engine_algorithm(request.mode))
    request.ticket.check()
    return counts, edges, levels, local


class _OtsuHistogramPlot(QWidget):
    """The field's intensity histogram with the chosen level drawn on it.

    A preview of the histogram with the chosen level marked, so a curator
    can see where the cut falls. It is painted rather than plotted: the whole figure is a few
    hundred bars and two or three vertical lines, and a chart library on this
    screen would mean importing one on the path that opens Make Masks.

    THE MARKER IS NOT COMPUTED HERE. It is handed in, already read from
    :func:`spacr.qt.mask_engine._otsu_levels` -- the same function the detect
    button's cut comes from -- because a preview that found its own level
    would be a second opinion and could be right while the button was wrong.

    :param counts: bar heights, as :func:`numpy.histogram` returns them.
    :param edges: bin edges, one longer than ``counts``.
    :param levels: the intensities the field is cut at.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, counts, edges, levels, parent=None):
        """Keep the figure and set a size a reader can tell two humps apart in."""
        super().__init__(parent)
        self.counts = np.asarray(counts, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.levels = [float(level) for level in levels]
        self.setMinimumSize(420, 220)

    def level_x(self, level: float) -> float:
        """Where ``level`` falls across the plot, in widget pixels.

        The same mapping the bars are drawn with, so a test can ask the
        picture where it put the marker instead of trusting that it did.
        Its range includes corrected levels outside the histogram's bin
        edges, keeping those markers distinct from the maximum intensity.

        :param level: an intensity.
        :returns: the x coordinate, clamped to the plot's own width.
        """
        low = min(float(self.edges[0]), *self.levels) if self.levels else float(self.edges[0])
        high = max(float(self.edges[-1]), *self.levels) if self.levels else float(self.edges[-1])
        width = max(1, self.width() - 1)
        if high <= low:
            return 0.0
        fraction = (float(level) - low) / (high - low)
        return max(0.0, min(1.0, fraction)) * width

    def paintEvent(self, event):
        """Draw bars and threshold lines on the same intensity axis."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        palette = active_palette()
        painter.fillRect(self.rect(), QColor(palette["bg"]))
        height = max(1, self.height())
        tallest = float(self.counts.max()) if self.counts.size else 0.0
        if tallest > 0.0:
            bar_colour = QColor(palette.get("fg", "#c8c8c8"))
            bar_colour.setAlpha(160)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(bar_colour))
            for index, value in enumerate(self.counts):
                tall = int(round(height * float(value) / tallest))
                if tall <= 0:
                    continue
                left = int(self.level_x(self.edges[index]))
                right = int(self.level_x(self.edges[index + 1]))
                painter.drawRect(QRect(left, height - tall,
                                       max(1, right - left), tall))
        pen = QPen(QColor(palette["accent"]))
        pen.setWidth(2)
        painter.setPen(pen)
        for level in self.levels:
            x = int(round(self.level_x(level)))
            painter.drawLine(x, 0, x, height)
        painter.end()


class _LevelsPlot(_OtsuHistogramPlot):
    """Drag the nearest black/white marker along the intensity histogram."""

    cutoff_changed = Signal(int, float)

    def mousePressEvent(self, event):
        """Choose the nearest cutoff; right and middle clicks do nothing."""
        if event.button() != Qt.LeftButton or len(self.levels) != 2:
            return
        self._drag_cutoff = min(range(2), key=lambda i:
                                abs(self.level_x(self.levels[i]) - event.position().x()))
        self._move_cutoff(event.position().x())

    def mouseMoveEvent(self, event):
        """Move a held cutoff without recomputing the histogram."""
        if event.buttons() & Qt.LeftButton and hasattr(self, '_drag_cutoff'):
            self._move_cutoff(event.position().x())

    def mouseReleaseEvent(self, event):
        """Finish a cutoff drag at the released position."""
        if event.button() == Qt.LeftButton and hasattr(self, '_drag_cutoff'):
            self._move_cutoff(event.position().x())
            del self._drag_cutoff

    def _move_cutoff(self, x):
        """Convert widget x into an intensity on the histogram's axis."""
        fraction = max(0.0, min(1.0, x / max(1, self.width() - 1)))
        value = float(self.edges[0] + fraction * (self.edges[-1] - self.edges[0]))
        self.cutoff_changed.emit(self._drag_cutoff, value)


def _levels_histogram(image):
    """Sort finite field intensities and count bins off the GUI thread.

    The sorted values provide exact percentile positions for dragged levels;
    no downsampling or histogram-bin approximation changes the chosen cut.
    """
    values = np.sort(image[np.isfinite(image)], axis=None)
    if not values.size:
        from ..i18n import tr

        raise ValueError(tr('The image has no finite intensities.'))
    counts, edges = np.histogram(values, bins=256)
    return counts, edges, values


class _LevelsDialog(QDialog):
    """Edit black/white percentile cutoffs by histogram or intensity value.

    The histogram describes the full displayed source before enhancement,
    including inversion when enabled. It is computed on a worker. Changes
    emit percentiles used by the existing display/detection normalization;
    source pixels and masks are never edited. Closing releases the sorted
    field; a late worker result cannot reopen the dialog.
    """

    levels_changed = Signal(float, float)
    _delivered = Signal(object)

    def __init__(self, image, percentiles, parent=None):
        """Build linked percentile controls and submit the source histogram for background computation."""
        super().__init__(parent)
        from ..i18n import tr

        self.setWindowTitle(tr('Levels'))
        self.closed = False
        self.ready = False
        self.values = None
        self.percentiles = tuple(percentiles)
        layout = QVBoxLayout(self)
        self.caption = QLabel(tr('Calculating image histogram…'))
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.plot = _LevelsPlot(np.zeros(2), np.arange(3), [], self)
        self.plot.setMinimumHeight(100)
        self.plot.setEnabled(False)
        layout.addWidget(self.plot, 1)
        form = QFormLayout()
        self.black, self.white = QDoubleSpinBox(), QDoubleSpinBox()
        for index, control in enumerate((self.black, self.white)):
            control.setDecimals(6)
            control.setEnabled(False)
            control.valueChanged.connect(lambda value, i=index: self._choose(i, value))
        form.addRow(tr('Black cutoff'), self.black)
        form.addRow(tr('White cutoff'), self.white)
        layout.addLayout(form)
        self.plot.cutoff_changed.connect(self._choose)
        self.detect = Toggle(tr('Detect on the normalized image'))
        layout.addWidget(self.detect)
        note = QLabel(tr('Drag a marker or enter an intensity. Values below the black '
                         'cutoff become black; values above the white cutoff become white. '
                         'The range between them is stretched. Enable detection here to '
                         'use these levels before any applied enhancement. Original image '
                         'values and existing masks are preserved.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.reset = buttons.addButton(tr('Reset levels'), QDialogButtonBox.ResetRole)
        self.reset.setEnabled(False)
        self.reset.clicked.connect(lambda: self._publish(0.0, 100.0))
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self._delivered.connect(self._take, Qt.QueuedConnection)
        self._worker = _NewestRequestWorker(_levels_histogram, self._deliver,
                                             name='spacr-levels-histogram')
        self._worker.submit(image)
        self.resize(560, 520)

    def _deliver(self, request, result, error):
        """Carry computation back to Qt, tolerating destruction while busy."""
        try:
            self._delivered.emit((result, error))
        except RuntimeError:
            pass

    def _take(self, payload):
        """Install a completed histogram only while this editor is open."""
        from ..i18n import tr

        if self.closed:
            return
        result, error = payload
        if error is not None:
            self.caption.setText(tr('Could not calculate levels: {error}', error=str(error)))
            return
        counts, edges, self.values = result
        self.ready = True
        self.plot.counts, self.plot.edges = counts, edges
        varying = self.values[0] < self.values[-1]
        for control in (self.black, self.white):
            blocked = control.blockSignals(True)
            control.setRange(float(self.values[0]), float(self.values[-1]))
            control.blockSignals(blocked)
            control.setEnabled(bool(varying))
        self.plot.setEnabled(bool(varying))
        self.reset.setEnabled(True)
        self.set_percentiles(*self.percentiles)
        if not varying:
            self.caption.setText(tr('This image has one intensity; there is no range to stretch.'))

    def set_percentiles(self, low, high):
        """Follow changes from the screen without emitting another edit."""
        from ..i18n import tr

        self.percentiles = (float(low), float(high))
        if self.values is None:
            return
        positions = np.asarray(self.percentiles) * (len(self.values) - 1) / 100.0
        left = np.floor(positions).astype(int)
        right = np.ceil(positions).astype(int)
        levels = (self.values[left].astype(float) * (1 - positions + left)
                  + self.values[right].astype(float) * (positions - left))
        self.plot.levels = list(levels)
        self.plot.update()
        for control, value in zip((self.black, self.white), levels):
            blocked = control.blockSignals(True)
            control.setValue(float(value))
            control.blockSignals(blocked)
        self.caption.setText(tr('Full-field intensity histogram · black {low:.4g}, white {high:.4g}',
                                low=float(levels[0]), high=float(levels[1])))

    def _choose(self, index, value):
        """Map an absolute intensity to its interpolated percentile rank."""
        if self.values is None or self.values[-1] <= self.values[0]:
            return
        values = self.values
        upper = int(np.searchsorted(values, value, side='left'))
        if upper == 0:
            percentile = 0.0
        elif upper >= values.size:
            percentile = 100.0
        else:
            a, b = float(values[upper - 1]), float(values[upper])
            fraction = (float(value) - a) / (b - a) if b > a else 0.0
            percentile = (upper - 1 + fraction) * 100.0 / (values.size - 1)
        low, high = self.percentiles
        if index == 0:
            low = min(percentile, high - 0.000001)
        else:
            high = max(percentile, low + 0.000001)
        self._publish(max(0.0, low), min(100.0, high))

    def _publish(self, low, high):
        """Apply an ordered pair of percentiles to the screen and markers."""
        self.set_percentiles(low, high)
        self.levels_changed.emit(float(low), float(high))

    def closeEvent(self, event):
        """Discard pending work and release the field held for this histogram."""
        self.closed = True
        self.values = None
        self._worker.close(timeout=0)
        super().closeEvent(event)


class _OtsuHistogramDialog(QDialog):
    """A window holding :class:`_OtsuHistogramPlot` and what it is showing.

    Modeless on purpose: the point of the preview is to change a setting and
    look again, and a modal window would make that a close, a change and a
    reopen each time. It is also what keeps it testable -- a static modal
    runs its event loop in C++ and hangs a headless run.

    :param counts: histogram bar heights.
    :param edges: histogram bin edges.
    :param levels: the intensities the field is cut at.
    :param description: how the cut was taken, for the caption.
    :param local: local thresholds have no single marker to draw.
    :param parent: parent widget.
    :param method: selected threshold key, named in the window title.
    :param pending: show indeterminate progress until the snapshot arrives.
    """

    def __init__(self, counts, edges, levels, description: str,
                 local: bool = False, parent=None, *, method="otsu",
                 pending: bool = False):
        """Build the plot, the caption above it and the Close button."""
        super().__init__(parent)
        from ..i18n import tr

        self.setWindowTitle(tr("{method} histogram", method=_magnifier_mode_label(method)))
        self.description = description
        self.request = None
        self.ready = not pending
        self.closed = False
        self.error = None
        layout = QVBoxLayout(self)
        layout.setSpacing(SPACING["sm"])
        self.caption = QLabel()
        self.caption.setTextFormat(Qt.PlainText)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(pending)
        layout.addWidget(self.progress)
        self.plot = _OtsuHistogramPlot(counts, edges, levels, self)
        layout.addWidget(self.plot, 1)
        self.plot.setVisible(not pending)
        if pending:
            self.caption.setText(tr("Calculating threshold histogram…"))
        else:
            self.show_result((counts, edges, levels, local))
        note = QLabel(tr("Snapshot of the full field and settings when opened. "
                         "Open the histogram again after changing the image or settings."))
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.resize(520, 340)

    def show_result(self, result, error=None) -> None:
        """Display a completed snapshot or its error, without a fallback marker."""
        from ..i18n import tr

        self.ready = True
        self.error = error
        self.progress.hide()
        if error is not None:
            self.plot.hide()
            self.caption.setText(tr("Threshold histogram failed: {error}", error=str(error)))
            self._fit_height()
            return
        counts, edges, levels, local = result
        self.plot.counts = np.asarray(counts, dtype=np.float64)
        self.plot.edges = np.asarray(edges, dtype=np.float64)
        self.plot.levels = [float(level) for level in levels]
        self.plot.show()
        self.plot.update()
        if local:
            text = tr("Local threshold varies by pixel ({description}); no single "
                      "level is drawn. The histogram shows the smoothed detector "
                      "input before local thresholding.", description=self.description)
        else:
            text = tr("Level: {levels} ({description}). These are the thresholds "
                      "the detect button uses on the smoothed detector input.",
                      levels=", ".join(f"{level:.4g}" for level in levels),
                      description=self.description)
        self.caption.setText(text)
        self._fit_height()

    def _fit_height(self) -> None:
        """Reserve the wrapped captions' height so they cannot overlap the plot."""
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            needed = layout.totalHeightForWidth(self.width())
            if needed > 0 and self.minimumHeight() != needed:
                self.setMinimumHeight(needed)

    def resizeEvent(self, event):
        """Recompute the text's height as the histogram window is widened."""
        super().resizeEvent(event)
        self._fit_height()

    def closeEvent(self, event):
        """Retire this snapshot; a running library call may finish in the background."""
        self.closed = True
        if self.request is not None:
            self.request.ticket.cancel()
        super().closeEvent(event)


def _parsed_sigmas(text: str) -> tuple:
    """The filament widths a text box holds, as numbers the engine can read.

    A LIST OF SCALES IS A TEXT BOX and not a row of spin boxes, because how
    many scales the ridge filter is given is part of the answer: one width
    for a uniform bundle, four to cover fine tubules and thick ones at
    once. So the box is parsed rather than validated -- anything that is
    not a positive number is dropped, and a box that holds nothing usable
    falls back to the engine's own default rather than raising on the
    worker thread, where the only way to show the error would be an empty
    magnifier box.

    :param text: what the user typed, numbers separated by commas or spaces.
    :returns: the scales, in the order typed, never empty.
    """
    out = []
    for piece in str(text or "").replace(",", " ").split():
        try:
            value = float(piece)
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return tuple(out) or organelle_modes.DEFAULT_PARAMS.ridge_sigmas


def _grey_pixmap(image: np.ndarray, lower_pct: float,
                 upper_pct: float) -> QPixmap:
    """``image`` stretched between two percentiles, as a grey pixmap.

    One conversion for both halves of the Compare window, so the raw
    picture and the enhanced one are drawn by the same arithmetic and a
    difference between them is a difference the chain made.

    :param image: any 2-D array; float and integer fields alike.
    :param lower_pct: the percentile drawn black.
    :param upper_pct: the percentile drawn white.
    """
    data = np.asarray(image, dtype=np.float64)
    if not data.size:
        return QPixmap()
    low = float(np.percentile(data, lower_pct))
    high = float(np.percentile(data, upper_pct))
    if high <= low:
        high = low + 1.0
    grey = np.ascontiguousarray(
        (np.clip(data, low, high) - low) / (high - low) * 255.0
    ).astype(np.uint8)
    height, width = grey.shape[:2]
    picture = QImage(grey.data, width, height, width,
                     QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(picture)


#: Saved-layout key and initial window size for the side-by-side comparison.
#: The initial size leaves enough room to inspect both images.
COMPARE_LAYOUT_KEY = "make_masks::compare"
COMPARE_DEFAULT_SIZE = (1100, 700)


class _ComparePreview(QDialog):
    """The image as loaded beside the image the detector reads.

    ONE CLICK IS THE WHOLE FEATURE. A chain of eight optional steps is a
    chain a curator cannot judge from the objects alone -- a threshold that
    found nothing may have been given an image with nothing left in it --
    and the cheapest way to say which it was is to put the two pictures
    next to each other under the list of what ran.

    IT IS A WINDOW TO LOOK IN, so it is resizable, it remembers the size it
    was left at (:data:`COMPARE_LAYOUT_KEY`), the two pictures grow with it
    rather than sitting at a fixed size, and both can be zoomed into. THE
    TWO ZOOMS ARE ONE ZOOM (:meth:`spacr.qt.widgets.zoom_view.
    ZoomableImageView.link_to`): a difference between the pictures is what
    the window is for, and a difference in where they are pointing is the
    one difference that is not information.

    Modeless, for :class:`_OtsuHistogramDialog`'s reason: the point is to
    change a step and look again.

    :param raw: the region as loaded (or inverted, as it is drawn).
    :param enhanced: the same region after the chain, or None while its
        worker runs. Cancel closes a pending comparison without blocking.
    :param steps: the steps that ran, in words.
    :param parent: parent widget.
    """

    def __init__(self, raw: np.ndarray, enhanced: Optional[np.ndarray],
                 steps: str, parent=None):
        """Build the two linked views, the caption over them and the tools."""
        from ..i18n import tr
        from ..widgets.zoom_view import ZoomableImageView

        super().__init__(parent)
        self.setWindowTitle(tr("Raw and enhanced"))
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        layout.setSpacing(SPACING["sm"])
        self.caption = QLabel(steps)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        row = QHBoxLayout()
        self.views: List[ZoomableImageView] = []
        for title, array in ((tr("As loaded"), raw),
                             (tr("As the detector reads it"), enhanced)):
            column = QVBoxLayout()
            heading = QLabel(title)
            heading.setObjectName("Muted")
            column.addWidget(heading)
            view = ZoomableImageView(self)
            view.setMinimumSize(200, 200)
            if array is not None:
                view.set_pixmap(_grey_pixmap(array, 1.0, 99.9))
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.views.append(view)
            column.addWidget(view, 1)
            row.addLayout(column, 1)
        layout.addLayout(row, 1)
        self.views[0].link_to(self.views[1])

        tools = QHBoxLayout()
        self.hint = QLabel(tr(
            "Scroll to zoom, drag to pan. Both pictures move together."))
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        tools.addWidget(self.hint, 1)
        for caption, action in ((tr("Zoom out"), lambda: self.zoom(1 / 1.4)),
                                (tr("Zoom in"), lambda: self.zoom(1.4)),
                                (tr("Fit"), self.fit)):
            button = QPushButton(caption)
            button.clicked.connect(action)
            tools.addWidget(button)
        layout.addLayout(tools)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Cancel if enhanced is None else QDialogButtonBox.Close)
        self._buttons.rejected.connect(self.close)
        layout.addWidget(self._buttons)
        self.resize(*_remembered_compare_size())

    def _show_result(self, enhanced, caption):
        """Adopt a finished comparison, or explain why it was discarded."""
        if enhanced is not None:
            self.views[1].set_pixmap(_grey_pixmap(enhanced, 1.0, 99.9))
            self.fit()
        self.caption.setText(caption)
        self._buttons.setStandardButtons(QDialogButtonBox.Close)

    def zoom(self, factor: float) -> None:
        """Zoom both pictures by ``factor``; the link does the second one."""
        if self.views:
            self.views[0].zoom_by(factor)

    def fit(self) -> None:
        """Put both pictures back to fitting their pane."""
        if self.views:
            self.views[0].fit()

    def closeEvent(self, event):                            # noqa: N802
        """Remember the size the window was left at, then close.

        On the way out rather than on every resize: a drag is a hundred
        resize events and this writes to the preference store.
        """
        _remember_compare_size(self.width(), self.height())
        super().closeEvent(event)


def _remembered_compare_size() -> tuple:
    """``(width, height)`` the Compare window was last left at."""
    try:
        from ..preferences import get_section_layout

        sizes = get_section_layout(COMPARE_LAYOUT_KEY).get("sizes") or ()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the Compare window size", exc_info=True)
        sizes = ()
    if len(sizes) == 2 and all(int(value) > 200 for value in sizes):
        return (int(sizes[0]), int(sizes[1]))
    return COMPARE_DEFAULT_SIZE


def _remember_compare_size(width: int, height: int) -> None:
    """Keep the Compare window's size for the next time it is opened."""
    try:
        from ..preferences import set_section_layout

        set_section_layout(COMPARE_LAYOUT_KEY,
                           sizes=(int(width), int(height)))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not keep the Compare window size", exc_info=True)


def fold_description(key: str) -> tuple:
    """``(name, description, stage)`` for a folded module.

    The app registry answers while it still holds the module's row; once the
    row has been dropped — which is what folding a module ends in — the
    answer comes from :data:`FOLD_FALLBACK`, so the button goes on carrying
    the name, the sentence and the maturity colour its tile had.

    :param key: the app registry key of the folded module; an unknown key gives
        empty strings.
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
        """Display a status message and return the complete displayed text.

        :param text: the message; converted to ``str``.
        :param append: add the message below the current text, after a blank
            line, instead of replacing it.
        """
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



class ObjectFilterList(QWidget):
    """Make Masks' object filters: one row per regionprop the user added.

    Item 511. The Filter category used to hold four fixed boxes -- minimum
    and maximum area, minimum and maximum mean intensity. It now starts
    EMPTY, and "Add a filter" offers every scalar property
    :func:`skimage.measure.regionprops` computes (see
    :func:`mask_engine.filter_properties`); each added row carries the
    property, a minimum, a maximum and a Remove button. A blank bound is
    off. The old four are two of the rows a user can add: ``area`` and
    ``intensity_mean``.

    The intensity statistics are offered only while an intensity image is
    open (:meth:`set_intensity_available`), so a property that cannot be
    measured is never offered, rather than offered and refused later.

    :meth:`filters` is the serialised list -- the same one Mask generation
    reads from ``object_filters`` -- and :attr:`changed` fires whenever it
    may have changed, which is what applies the list live.
    """

    changed = Signal()
    row_added = Signal(object)

    def __init__(self, parent=None):
        """Build the Add control and the empty row list."""
        from ..i18n import tr

        super().__init__(parent)
        self._rows: List[dict] = []
        self._intensity = False
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING["xs"])
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(SPACING["xs"])
        column.addLayout(self._rows_layout)
        adder = QHBoxLayout()
        self.property_box = QComboBox()
        self.property_box.setToolTip(tr(
            "The scikit-image regionprop the next filter row judges objects "
            "by. Intensity statistics are listed only while an image is "
            "open, since they read the raw pixel values."))
        self.add_button = QPushButton(tr("Add a filter"))
        self.add_button.setCursor(Qt.PointingHandCursor)
        self.add_button.setToolTip(tr(
            "Add a row for the property on the left, with a minimum and a "
            "maximum. A blank bound is off; an object outside a bound is "
            "hidden, and removing the row brings it back."))
        self.add_button.clicked.connect(self._on_add)
        adder.addWidget(self.property_box, 1)
        adder.addWidget(self.add_button)
        column.addLayout(adder)
        self._offer()

    def set_intensity_available(self, available: bool) -> None:
        """Offer the intensity properties only when an image is open.

        :param available: whether an intensity image is open.
        """
        self._intensity = bool(available)
        self._offer()

    def _offer(self) -> None:
        """Fill the property box with what can be measured right now."""
        chosen = self.property_box.currentText()
        names = engine.filter_properties(intensity=self._intensity)
        self.property_box.blockSignals(True)
        self.property_box.clear()
        self.property_box.addItems(list(names))
        if chosen in names:
            self.property_box.setCurrentText(chosen)
        self.property_box.blockSignals(False)
        self.add_button.setEnabled(bool(names))

    def offered(self) -> List[str]:
        """The properties the Add control offers now."""
        return [self.property_box.itemText(i)
                for i in range(self.property_box.count())]

    def _on_add(self) -> None:
        """Add a row for the property the box shows."""
        name = self.property_box.currentText()
        if name:
            self.add_filter(name)

    def _bound_edit(self, value, placeholder: str) -> QLineEdit:
        """One bound's box: a number or blank, with blank meaning off."""
        from PySide6.QtCore import QLocale
        from PySide6.QtGui import QDoubleValidator

        edit = QLineEdit()
        validator = QDoubleValidator(edit)
        validator.setLocale(QLocale.c())
        validator.setNotation(QDoubleValidator.StandardNotation)
        edit.setValidator(validator)
        edit.setPlaceholderText(placeholder)
        edit.setText("" if value is None else format(float(value), ".12g"))
        edit.editingFinished.connect(self.changed.emit)
        return edit

    def add_filter(self, name, minimum=None, maximum=None, *,
                   notify: bool = True) -> dict:
        """Add one row; return it as ``{widget, property, min, max, remove}``.

        :param name: the regionprop the row filters on.
        :param minimum: the lower bound, or ``None`` for none.
        :param maximum: the upper bound, or ``None`` for none.
        :param notify: emit ``changed`` once the row is added.
        :raises ValueError: when ``name`` is not a scalar regionprop, or is an
            intensity property while no intensity image is open.
        """
        from ..i18n import tr

        name = engine.canonical_property(name)
        if name not in self.offered():
            raise ValueError(tr(
                "{name} measures pixel values, and no intensity image is "
                "open, so it cannot filter this mask.", name=name))
        widget = QWidget()
        line = QHBoxLayout(widget)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(SPACING["xs"])
        label = QLabel(name)
        label.setToolTip(tr(
            "The regionprop this row judges each object by, measured once "
            "per mask together with every other row."))
        low = self._bound_edit(minimum, tr("no minimum"))
        low.setToolTip(tr(
            "Hide objects whose value is below this. Blank is no minimum; "
            "an object equal to the bound is kept."))
        high = self._bound_edit(maximum, tr("no maximum"))
        high.setToolTip(tr(
            "Hide objects whose value is above this. Blank is no maximum; "
            "an object equal to the bound is kept."))
        remove = QPushButton(tr("Remove"))
        remove.setCursor(Qt.PointingHandCursor)
        remove.setToolTip(tr(
            "Remove this filter. The objects only it was hiding come back."))
        line.addWidget(label, 1)
        line.addWidget(low)
        line.addWidget(high)
        line.addWidget(remove)
        row = {"widget": widget, "property": name, "min": low, "max": high,
               "remove": remove}
        remove.clicked.connect(lambda: self.remove_filter(self._rows.index(row)))
        self._rows.append(row)
        self._rows_layout.addWidget(widget)
        self.row_added.emit(widget)
        if notify:
            self.changed.emit()
        return row

    def set_bounds(self, index: int, minimum=None, maximum=None) -> None:
        """Set row ``index``'s bounds as if typed, and apply them.

        :param index: the row, in the order rows were added.
        :param minimum: the lower bound, or ``None`` to clear it.
        :param maximum: the upper bound, or ``None`` to clear it.
        """
        row = self._rows[index]
        row["min"].setText("" if minimum is None else format(float(minimum), ".12g"))
        row["max"].setText("" if maximum is None else format(float(maximum), ".12g"))
        self.changed.emit()

    def set_filter(self, name, minimum=None, maximum=None) -> None:
        """Set the bounds of the first row for ``name``, adding it if absent.

        :param name: the regionprop, current or legacy spelling.
        :param minimum: the lower bound, or ``None`` to clear it.
        :param maximum: the upper bound, or ``None`` to clear it.
        """
        name = engine.canonical_property(name)
        index = next((i for i, row in enumerate(self._rows)
                      if row["property"] == name), None)
        if index is None:
            self.add_filter(name, notify=False)
            index = len(self._rows) - 1
        self.set_bounds(index, minimum, maximum)

    def remove_filter(self, index: int) -> None:
        """Remove row ``index``; the objects only it hid come back.

        :param index: the row, in the order rows were added.
        """
        row = self._rows.pop(index)
        row["widget"].hide()
        row["widget"].setParent(None)
        row["widget"].deleteLater()
        self.changed.emit()

    def rows(self) -> List[dict]:
        """The rows, in the order they were added."""
        return list(self._rows)

    def filters(self) -> List[dict]:
        """The rows as the serialised filter list the engine and a run read.

        :raises ValueError: when a row's minimum is above its maximum.
        """
        return engine.normalise_filters([
            {"property": row["property"], "min": row["min"].text(),
             "max": row["max"].text()} for row in self._rows])

    def set_filters(self, filters) -> None:
        """Replace every row with ``filters``, a list or a legacy bounds dict.

        A dict of the old four bounds (``min_area`` and the rest) is migrated
        by :func:`mask_engine.legacy_filters`, so a saved state from before
        item 511 opens as the rows it meant.

        :param filters: a filter list in any form
            :func:`mask_engine.normalise_filters` accepts, or the legacy dict.
        """
        if isinstance(filters, dict) and set(filters) <= set(engine.FILTER_BOUNDS):
            filters = engine.legacy_filters(**filters)
        entries = engine.normalise_filters(filters)
        while self._rows:
            row = self._rows.pop()
            row["widget"].setParent(None)
            row["widget"].deleteLater()
        for entry in entries:
            self.add_filter(entry["property"], entry["min"], entry["max"],
                            notify=False)
        self.changed.emit()


class MakeMasksScreen(QWidget):
    """Qt widget for the Make Masks app — the successor to Tk ModifyMaskApp.

    Owns the canvas, the tools panel, and the file-navigation state; see
    the module docstring for the full feature list.

    :param parent: parent widget.
    """

    _histogram_delivered = Signal(object)
    _detection_delivered = Signal(object)
    _comparison_delivered = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the editor, its canvas and its tool panel.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self._folder: str = ""
        self._image_files: List[str] = []
        self._field_folders: Optional[List[str]] = None
        #: The terminal-built session this screen is working through, or
        #: ``None`` when the folder was opened from the file dialog. Set by
        #: :meth:`open_queue`; what makes a save reach
        #: ``curate_status.csv``.
        self._queue = None
        #: The masks folder of a sibling-layout session, which is beside the
        #: images rather than beneath them; ``None`` means ``<folder>/masks``.
        #: Set with the folder by :meth:`_open_folder`, so no field of one
        #: set is ever read or saved against another set's masks.
        self._masks_dir: Optional[str] = None
        #: What a terminal-built session had to say when it opened, still
        #: waiting for its first field to land. A field large enough to load
        #: off the GUI thread arrives after :meth:`open_queue` has returned,
        #: and its status line would otherwise replace the notice before
        #: anybody could read it. Emptied once shown, and by any other
        #: folder being opened.
        self._session_notice: str = ""
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
        self._detection_worker = None
        self._detection_request = None
        self._detection_delivered.connect(self._take_detection)
        self._comparison_worker = None
        self._comparison_request = None
        self._comparison_serial = 0
        self._comparison_delivered.connect(self._take_comparison)
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
        ensure_widget_qss_applied(MAKE_MASKS_QSS_NAME, root=self)
        self._build_ui()
        self._install_shortcuts()
        self._sync_button_states()

        try:
            from ..dnd import install_dropzone
            from ..dnd_handlers import MakeMasksDropHandler
            install_dropzone(self, MakeMasksDropHandler(), self)
        except Exception:
            pass
        from ..widgets.make_masks_help import install_make_masks_help
        install_make_masks_help(self)
        self._take_any_terminal_queue()

    def _take_any_terminal_queue(self) -> bool:
        """Open the queue ``spacr-make-masks`` handed over, if there is one.

        The terminal entry point hands over here. ``spacr-make-masks``
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
        them. All three layouts are edited where they lie:

        * ``nested`` opens the queue folder, masks in ``<folder>/masks``;
        * ``sibling`` opens ``<folder>/images`` and reads and saves the
          masks in ``<folder>/masks`` beside it, never ``images/masks``;
        * ``seg`` opens the queue folder with the ``_seg.npy`` bundles as
          its fields, each saved back into itself.

        What the session had to say -- a ``prob`` or ``easy`` order that
        fell back for want of scores, fields the scores do not name, a
        resume record beside the folder rather than in it -- is put on
        screen with it, not only in the terminal that started it. It stays
        on the status line after the first field loads, including a field
        large enough to load in the background, which lands after this
        returns.

        :param queue: a :class:`spacr.curation_queue.CurationQueue`.
        :returns: whether the editor is now on that session.
        """
        from ...curation_queue import LAYOUT_NESTED, LAYOUT_SEG, LAYOUT_SIBLING

        layout = queue.layout
        kind = getattr(layout, "kind", None)
        masks_dir: Optional[str] = None
        if kind == LAYOUT_SEG:
            folder = str(layout.folder)
            files = [item.bundle.name for item in queue.items
                     if item.bundle is not None]
        elif kind in (LAYOUT_NESTED, LAYOUT_SIBLING):
            folder = str(layout.images_dir)
            files = [item.image.name for item in queue.items
                     if item.image is not None]
            if kind == LAYOUT_SIBLING:
                masks_dir = str(layout.masks_dir)
        else:
            LOG.warning("Make Masks cannot edit the %s layout of %s",
                        kind, queue.folder)
            return False
        if not files:
            LOG.info("%s has nothing left to curate", queue.folder)
            return False
        if not self._open_folder(folder, files=files, masks_dir=masks_dir):
            return False
        self._queue = queue
        notices = tuple(getattr(queue, "notices", ()) or ())
        self._src_label.setText(
            f"{queue.folder}  --  {len(files)} to curate this session, "
            f"{queue.order_phrase}")
        self._session_notice = "  ".join((queue.describe(),) + notices)
        self._show_session_notice(keep=self._loading)
        return True

    def _show_session_notice(self, *, keep: bool = False) -> None:
        """Put the session's opening notice after what the status line says.

        :param keep: hold the notice for the field still loading in the
            background, so it is shown again when that field lands.
        """
        notice = self._session_notice
        if not notice:
            return
        if not keep:
            self._session_notice = ""
        current = self._status_label.text()
        self._status_label.setText(f"{current}  {notice}" if current
                                   else notice)

    def _layout_kwargs(self) -> dict:
        """What every mask read and write passes for this folder's layout.

        Empty for the nested layout, so those calls are exactly what they
        were before a sibling set could be opened.

        :returns: ``{"masks_dir": ...}`` for a sibling session, else ``{}``.
        """
        return {"masks_dir": self._masks_dir} if self._masks_dir else {}

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

        stem = engine.field_stem(filename)
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

        from ..i18n import tr
        self._invert_warning = QLabel(tr(INVERT_WARNING_TEXT))
        self._invert_warning.setObjectName(INVERT_WARNING_NAME)
        self._invert_warning.setWordWrap(True)
        self._invert_warning.hide()
        outer.addWidget(self._invert_warning)

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

        from ..widgets.collapsible_splitter import CollapsibleSplitter, EDGE

        self._body_splitter = CollapsibleSplitter(
            Qt.Horizontal, persist_key="make_masks::body")
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
        #: The Otsu histogram preview while it is open, kept so pressing
        #: the button twice reuses one window rather than stacking them and
        #: so the screen can take it down with itself.
        self._otsu_histogram_dialog: Optional[QDialog] = None
        self._levels_dialog = None
        self._histogram_worker = None
        self._histogram_delivered.connect(self._take_histogram)
        self._masks_console = _MasksConsole()
        self._magnifier.status.connect(
            lambda text: self._status_label.setText(text))
        self._canvas.status.connect(
            lambda text: self._status_label.setText(text))
        self._view_tabs = self._build_view_tabs()

        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QScrollArea.NoFrame)
        self._settings_scroll.setWidget(self._build_tools_panel())
        for changed in (self._cp_model.currentIndexChanged,
                        self._cp_diameter.valueChanged,
                        self._cp_flow.valueChanged,
                        self._cp_cellprob.valueChanged,
                        self._cp_normalize.toggled,
                        self._otsu_correction.valueChanged,
                        self._otsu_smoothing.valueChanged,
                        self._otsu_fill_holes.toggled,
                        self._otsu_split.toggled,
                        self._otsu_bright.toggled,
                        self._cp_invert.toggled,
                        self._min_area.valueChanged):
            changed.connect(self._on_magnifier_context_changed)
        self._min_area.valueChanged.connect(self._on_min_area_changed)
        self._on_min_area_changed(self._min_area.value())
        self._body_splitter.add_pane(
            self._settings_scroll, "Settings", mode=EDGE, stretch=1,
            extent=SETTINGS_WIDTH, fold_key="make_masks/Settings",
            hint="or drag to make the settings wider or narrower")
        self._body_splitter.add_pane(self._build_view_pane(), "Masks",
                                     stretch=3, extent=900)
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
        from ..make_masks_datasets import install_dataset_button
        nav_row.addWidget(install_dataset_button(self))

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

        self._btn_discard = QPushButton("Discard")
        self._btn_discard.setIcon(iconset.icon("trash"))
        self._btn_discard.setCheckable(True)
        self._btn_discard.setCursor(Qt.PointingHandCursor)
        self._btn_discard.setToolTip(
            "Mark this field as one to discard and move to the next. "
            "Nothing is deleted: the verdict goes to csv/keep_discard.csv "
            "beside the images, and the field, its mask and its objects "
            "stay as they are.")
        self._btn_discard.clicked.connect(lambda: self._on_curate(False))
        nav_row.addWidget(self._btn_discard)

        self._btn_keep = QPushButton("Keep")
        self._btn_keep.setIcon(iconset.icon("check"))
        self._btn_keep.setCheckable(True)
        self._btn_keep.setCursor(Qt.PointingHandCursor)
        self._btn_keep.setToolTip(
            "Mark this field as one to keep and move to the next. The "
            "verdict is written to csv/keep_discard.csv beside the images, "
            "with the image, its mask and the number of objects the mask "
            "holds right now.")
        self._btn_keep.clicked.connect(lambda: self._on_curate(True))
        nav_row.addWidget(self._btn_keep)

        self._btn_save = QPushButton("Save mask")
        self._btn_save.setObjectName("PrimaryButton")
        self._btn_save.setIcon(iconset.contrast_icon("save"))
        self._btn_save.setCursor(Qt.PointingHandCursor)
        self._btn_save.clicked.connect(self._on_save)
        nav_row.addWidget(self._btn_save)

        nav_row.addStretch(1)
        self._status_label = _StatusLabel("Ready.")
        self._status_label.setObjectName("SubtitleSmall")
        self._status_label.said.connect(self._report_status)
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
        if engine.is_seg_bundle(filename):
            self._status_label.setText(
                f"{filename} holds its image and mask inside one Cellpose "
                f"bundle; this module opens image and mask files, so it was "
                f"not pointed at it.")
            return {}
        mask_path = engine.mask_save_path(self._folder, filename,
                                          **self._layout_kwargs())
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

        The Apply half writes one mask per image into ``<src>/masks``, which
        is the nested layout's masks folder and nobody else's: in a sibling
        session it would be ``images/masks``, beside the set's real masks
        rather than in them, and a folder of ``_seg.npy`` bundles has no
        images for it to read. Both are refused, in the status line.

        :returns: whether a run was started. A folder that is not open, a
            session whose masks are not in ``<folder>/masks``, and a
            confirmation that is declined, all answer ``False``.
        """
        if not self._folder or not self._image_files:
            self._status_label.setText(
                "Open a folder of images before masking it.")
            return False
        if self._field_folders:
            from ..i18n import tr

            self._status_label.setText(tr(
                "Mask the whole folder works on one folder, and this queue "
                "was dropped from several, so it was not started."))
            return False
        if self._masks_dir or any(engine.is_seg_bundle(name)
                                  for name in self._image_files):
            self._status_label.setText(
                f"Mask the whole folder writes its masks into "
                f"{os.path.join(self._folder, 'masks')}, and this set keeps "
                f"its masks somewhere else, so it was not started.")
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
        if self._otsu_histogram_dialog is not None:
            self._otsu_histogram_dialog.close()
            self._otsu_histogram_dialog = None
        self._close_levels()
        if self._histogram_worker is not None:
            self._histogram_worker.close(timeout=0)
            self._histogram_worker = None
        if self._detection_worker is not None:
            self._detection_worker.close(timeout=0)
            self._detection_worker = None
            self._detection_request = None
            self._btn_cellpose.setEnabled(True)

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

        The row ends with the Magnifier and, directly right of it, the
        settings toggle, which is checkable because it reports a state
        rather than firing an action: it stays lit for as long as the
        settings are on screen. A stretch after the toggle keeps the row
        against the left edge, above the settings it hides.
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
            from ..i18n import tr

            btn = QPushButton(tr(label))
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
        self._mode_buttons[MODE_RULER].setToolTip(tr(
            "Drag a line to measure its length in image pixels. "
            "Right-click with Ruler selected to clear it. Zoom and pan preserve the measurement."))

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

        self._btn_features = QPushButton("Features")
        self._btn_features.setIcon(iconset.icon("run"))
        self._btn_features.setMinimumHeight(32)
        self._btn_features.setCursor(Qt.PointingHandCursor)
        self._btn_features.setToolTip(
            "Measure the masks you drew. Opens a table where each row is a "
            "field and each column is a channel or a mask type; the run "
            "goes through the Measure module itself, so the folders and the "
            "measurements database are the ones a Measure run produces.")
        self._btn_features.clicked.connect(self._on_open_features)
        row.addWidget(self._btn_features)

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setIcon(iconset.icon("settings"))
        self._btn_settings.setCheckable(True)
        self._btn_settings.setMinimumHeight(32)
        self._btn_settings.setCursor(Qt.PointingHandCursor)
        self._btn_settings.setToolTip(
            "Show or hide the settings — brush, wand, display, auto-filter, "
            "object operations, Otsu, object detection and the live "
            "magnifier, as one group. The canvas takes the width they give "
            "up.")
        self._btn_settings.setChecked(True)
        self._btn_settings.toggled.connect(self._on_toggle_settings)
        row.addWidget(self._btn_settings)
        row.addStretch(1)

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

    def _on_open_features(self, _checked: bool = False):
        """Open the measurement-input window on the folder being drawn in.

        The whole of this screen's part in measuring: the window, the
        table and the run live in
        :mod:`spacr.qt.screens.measure_inputs`, so Make Masks holds a button
        and a folder and nothing else about measuring.

        :param _checked: Qt's toggled flag, unused.
        :returns: the window, so a test can drive it.
        """
        from .measure_inputs import open_measure_inputs

        return open_measure_inputs(self, folder=self._folder or None)

    def _curation_paths(self):
        """The field being judged and the mask it is judged with.

        :returns: ``(image path, mask path)``, or ``(None, None)`` when no
            field is open.
        """
        if not self._folder or not self._image_files:
            return None, None
        try:
            filename = self._image_files[self._current_index]
        except (IndexError, TypeError):
            return None, None
        image_path = os.path.join(self._folder, filename)
        mask_path = engine.mask_save_path(self._folder, filename,
                                          **self._layout_kwargs())
        return image_path, mask_path

    def _on_curate(self, keep: bool):
        """Record a keep or discard verdict for the field on screen.

        THE COUNT IS READ NOW, not when the field was loaded, so a field
        curated after hand editing records the objects the user was looking
        at when they pressed the button.

        :param keep: True for Keep, False for Discard.
        :returns: the CSV's path, or None when there was nothing to record.
        """
        image_path, mask_path = self._curation_paths()
        if image_path is None:
            self._show_curation_verdict(None)
            return None
        try:
            written = engine.record_curation(
                self._folder, image_path, mask_path, self._objects_now(),
                keep)
        except OSError as exc:
            LOG.warning("Could not record the curation verdict: %s", exc)
            self._warn("Verdict not recorded",
                       f"{os.path.basename(image_path)} could not be marked: "
                       f"{exc}")
            self._show_curation_verdict(
                engine.curation_verdict(self._folder, image_path))
            return None
        self._show_curation_verdict(keep)
        self._advance_after_verdict(keep, os.path.basename(image_path))
        return written

    def _advance_after_verdict(self, keep: bool, judged: str) -> bool:
        """Move to the next field, and say what the verdict was on the way.

        Pressing Keep or Discard moves to the next image. Curation is a walk,
        and a verdict is the thing that ends a field -- a curator who has to
        press Keep and then Next presses twice per field for a thousand
        fields.

        :meth:`_on_next` is called rather than the index being moved here,
        so there is ONE definition of what next means -- it retires a recrop
        first, and it stops at the end of the folder instead of wrapping.

        THE MESSAGE IS BUILT HERE AND NOT IN A HELPER THAT APPENDS TO THE
        STATUS LINE. Loading the next field rewrites that line, so a verdict
        appended before the move is gone a moment later, and one appended
        after the move reads as a verdict on the field that just opened --
        which is the opposite of what happened. The name of the field that
        was judged is therefore carried in and spelled out.

        :param keep: True for Keep, False for Discard.
        :param judged: the basename of the field the verdict was about.
        :returns: whether the screen moved to another field.
        """
        said = "kept" if keep else "discarded — nothing was deleted"
        was = self._current_index
        self._on_next()
        moved = self._current_index != was
        if moved:
            now = os.path.basename(self._image_files[self._current_index])
            self._status_label.setText(f"{judged} {said}  —  now on {now}")
        elif (self._image_files
                and self._current_index >= len(self._image_files) - 1):
            self._status_label.setText(
                f"{judged} {said}  —  that was the last field in this folder")
        else:
            self._status_label.setText(f"{judged} {said}")
        return moved

    def _show_curation_verdict(self, verdict):
        """Put the field's current verdict on the two buttons.

        A field already marked shows it rather than looking unpressed,
        which is the difference between a record and a button that does
        something invisible.

        :param verdict: True, False, or None for no verdict.
        """
        for button, state in ((getattr(self, "_btn_keep", None),
                               verdict is True),
                              (getattr(self, "_btn_discard", None),
                               verdict is False)):
            if button is None:
                continue
            blocked = button.blockSignals(True)
            button.setChecked(bool(state))
            button.blockSignals(blocked)

    def _refresh_curation_buttons(self):
        """Show the verdict of whichever field is open now."""
        image_path, _mask_path = self._curation_paths()
        if image_path is None:
            self._show_curation_verdict(None)
            return
        try:
            self._show_curation_verdict(
                engine.curation_verdict(self._folder, image_path))
        except OSError:
            self._show_curation_verdict(None)

    def add_toolbar_action(self, button: QPushButton) -> QPushButton:
        """Insert a non-mode action into the editor toolbar.

        The button is placed with the other actions, before the Magnifier and
        the settings toggle, so that pair stays together at the end of the
        row whatever is added after them.

        :param button: Action button to insert.
        :returns: The same button.
        """
        row = self._tool_row_layout
        anchor = getattr(self, "_btn_magnifier", None)
        if anchor is None or row.indexOf(anchor) < 0:
            anchor = self._btn_settings
        row.insertWidget(row.indexOf(anchor), button)
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
        The settings are the splitter's first pane, left of the image.
        """
        splitter = self._body_splitter
        if not shown:
            sizes = splitter.sizes()
            if len(sizes) > 1 and sizes[0] > 0:
                self._settings_width = sizes[0]
        self._settings_scroll.setVisible(shown)
        if shown:
            sizes = splitter.sizes()
            total = sum(sizes) or (900 + SETTINGS_WIDTH)
            side = max(min(self._settings_width, total - 1), 1)
            splitter.setSizes([side, total - side])

    def _build_tools_panel(self) -> QWidget:
        """Build the tool column: mode, brush, wand and mask operations.

        :returns: the assembled panel.
        """
        from ..i18n import tr
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])
        self._settings_categories: List[tuple] = []
        try:
            from ..preferences import get_section_layout

            folded = get_section_layout(_SETTINGS_LAYOUT_KEY).get("folded")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the folded categories", exc_info=True)
            folded = None
        self._folded_categories = {
            _RENAMED_CATEGORIES.get(str(t), str(t)) for t in (folded or ())}

        brush_card = self._settings_category("Brush")
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

        wand_card = self._settings_category("Magic wand")
        wand_form = QFormLayout()
        self._wand_relative = Toggle("Tolerance is % of image range")
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
        self._wand_salvage = Toggle("Keep the nearest pixels at the cap")
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
        self._wand_intensity_border = Toggle("Re-flood below the escape")
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
        self._wand_gradient_taper = Toggle("Taper onto the gradient")
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

        norm_card = self._settings_category("Display")
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
        self._btn_levels = QPushButton(tr("Levels…"))
        self._btn_levels.setToolTip(tr(
            "Set black and white cutoffs by dragging on the image histogram."))
        self._btn_levels.clicked.connect(self._on_levels)
        norm_form.addRow(self._btn_levels)
        self._detect_normalized = Toggle("Detect on the normalized image")
        self._detect_normalized.setChecked(False)
        self._detect_normalized.setToolTip(
            "Off: Otsu, Cellpose and the magnifier read the image as it was "
            "loaded (inverted if Invert is on), and Lower % / Upper % only "
            "change how it is drawn. On: they read it stretched between Lower "
            "% and Upper %, exactly as drawn, and the intensity in the "
            "top-left corner shows that stretched value. What is saved is "
            "never changed.")
        self._detect_normalized.toggled.connect(self._on_detect_normalized)
        norm_form.addRow(self._detect_normalized)
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

        self._invert_display = Toggle("Invert image")
        self._invert_display.setToolTip(tr(
            "Invert the picture and the pixels used for detection, so dark "
            "objects become bright. The field is normalized to 0..1, then "
            "each pixel becomes 1 minus itself. Hover pixel intensity follows "
            "the inversion; object mean intensity and Filter thresholds use "
            "the original loaded values. The loaded image data and existing "
            "mask are unchanged. To swap foreground and background in a "
            "finished mask, use 'Swap object and background' in Object operations."
        ))
        self._invert_display.toggled.connect(self._on_invert_display)
        self._invert_display.toggled.connect(self._on_invert_toggled)
        norm_card.body_layout.addWidget(self._invert_display)
        col.addWidget(norm_card)

        filter_card = self._settings_category(
            "Filter",
            tr("Add a filter for any regionprop scikit-image measures. "
               "The list applies as you edit it and when a field opens; "
               "each object it hides is listed below, and removing a row "
               "brings back what it hid."),
        )
        self._filter_list = ObjectFilterList()
        self._filter_list.changed.connect(self._on_filters_changed)
        self._filter_list.row_added.connect(self._on_filter_row_added)
        self._filter_add = self._filter_list.add_button
        self._filter_property = self._filter_list.property_box
        filter_card.body_layout.addWidget(self._filter_list)
        self._btn_filter = QPushButton("Filter")
        self._btn_filter.setCursor(Qt.PointingHandCursor)
        self._btn_filter.setToolTip(tr(
            "Apply the filter list to the mask on screen again. Every "
            "object it hides is listed below it, with the value that hid "
            "it. One undo step."))
        self._btn_filter.clicked.connect(self._on_apply_filter)
        filter_card.body_layout.addWidget(self._btn_filter)
        self._filter_log = QPlainTextEdit()
        self._filter_log.setObjectName(FILTER_LOG_NAME)
        self._filter_log.setReadOnly(True)
        self._filter_log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._filter_log.setFixedHeight(
            self._filter_log.fontMetrics().lineSpacing() * FILTER_LOG_ROWS + 12)
        self._filter_log.setToolTip(tr(
            "What the filter list hides, one row per object: its id, its "
            "area and mean intensity, and each bound that hid it with the "
            "value it was judged on. The ids are the ones the hover readout "
            "shows."))
        self._set_filter_log([])
        filter_card.body_layout.addWidget(self._filter_log)
        col.addWidget(filter_card)

        obj_card = self._settings_category("Object operations")
        ops_col = QVBoxLayout()
        ops_col.setSpacing(SPACING["xs"])
        for label, cb, hint in (
            ("Fill holes", self._on_fill_holes,
             "Close every enclosed hole inside an object, so a nucleus "
             "outlined as a ring becomes a filled disc."),
            ("Relabel", self._on_relabel,
             "Renumber the objects 1, 2, 3… with no gaps. The picture does "
             "not change; the ids underneath it do."),
            ("Swap object and background", self._on_invert,
             "Turn every labelled pixel into background and every "
             "background pixel into an object. On an ordinary field that "
             "gives ONE object covering the whole frame with holes where "
             "your objects were — useful only when you have outlined the "
             "space between the cells and wanted the cells. This is not the "
             "picture invert: that is 'Invert image', in the Display "
             "category, and it leaves the mask alone."),
        ):
            btn = QPushButton(label)
            btn.setToolTip(hint)
            btn.clicked.connect(cb)
            ops_col.addWidget(btn)
        remove_row = QHBoxLayout()
        remove_row.setSpacing(SPACING["sm"])
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 1_000_000)
        self._min_area.setValue(100)
        self._min_area.setToolTip(
            "Smallest object worth keeping, in pixels. Removing small "
            "objects drops everything under it, and neither Otsu detect nor "
            "Object detection will produce an object below it, so one "
            "judgement about debris is made in one box."
        )
        remove_row.addWidget(QLabel("Min area:"))
        remove_row.addWidget(self._min_area, 1)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_small)
        remove_row.addWidget(remove_btn)
        remove_wrap = QWidget(); remove_wrap.setLayout(remove_row)
        ops_col.addWidget(remove_wrap)
        size_row = QHBoxLayout()
        size_row.setSpacing(SPACING["sm"])
        self._grow_step = QSpinBox()
        self._grow_step.setRange(1, 50)
        self._grow_step.setValue(1)
        self._grow_step.setSuffix(" px")
        self._grow_step.setToolTip(
            "How many pixels one press of Dilate or Shrink moves every "
            "object's boundary. The distance is the straight-line one, so 1 "
            "moves the edge onto its four neighbours and not its corners, "
            "and a Shrink undoes a Dilate of the same size on an object that "
            "had room to grow.")
        size_row.addWidget(QLabel("Grow / shrink:"))
        size_row.addWidget(self._grow_step, 1)
        self._btn_dilate = QPushButton("Dilate")
        self._btn_dilate.setCursor(Qt.PointingHandCursor)
        self._btn_dilate.setToolTip(
            "Grow every object by that many pixels, into background only: an "
            "object never takes a pixel from its neighbour and two objects "
            "never become one, so the ids and the object count are the ones "
            "you had. One undo step.")
        self._btn_dilate.clicked.connect(self._on_dilate)
        size_row.addWidget(self._btn_dilate)
        self._btn_shrink = QPushButton("Shrink")
        self._btn_shrink.setCursor(Qt.PointingHandCursor)
        self._btn_shrink.setToolTip(
            "Pull every object's boundary in by that many pixels. Each "
            "object is pulled back from its neighbours as well as from the "
            "background, so a pair that was touching comes apart. AN OBJECT "
            "THINNER THAN TWICE THE DISTANCE DISAPPEARS, and the status line "
            "says how many did. One undo step.")
        self._btn_shrink.clicked.connect(self._on_shrink)
        size_row.addWidget(self._btn_shrink)
        size_wrap = QWidget(); size_wrap.setLayout(size_row)
        ops_col.addWidget(size_wrap)
        detect_row = QHBoxLayout()
        detect_row.setSpacing(SPACING["sm"])
        self._btn_otsu = QPushButton("Otsu detect")
        self._btn_otsu.setCursor(Qt.PointingHandCursor)
        self._btn_otsu.setToolTip(
            "Run the CPU method chosen under Detection method on the whole "
            "image and label what it finds, honouring the minimum area "
            "above. Everything the method reads is that category: the "
            "level's algorithm, the correction, the smoothing, which side "
            "is the object, filling holes, splitting a pair that touches "
            "and dropping what the frame cut. With Cellpose or a backend "
            "chosen this button falls back to Otsu, because those have "
            "the Object detection button of their own.")
        self._btn_otsu.clicked.connect(self._on_detect_otsu)
        detect_row.addWidget(self._btn_otsu)
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
        from ..i18n import tr

        self._btn_clear = QPushButton(tr("Clear all objects"))
        self._btn_clear.setObjectName("DangerButton")
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.setToolTip(
            "Remove every object from this field. It asks first, and it is "
            "one undo step, so a press by accident costs one Ctrl+Z.")
        self._btn_clear.clicked.connect(self._on_clear_mask)
        ops_col.addWidget(self._btn_clear)
        obj_ops_wrap = QWidget(); obj_ops_wrap.setLayout(ops_col)
        obj_card.body_layout.addWidget(obj_ops_wrap)
        col.addWidget(obj_card)

        self._methods_card = self._build_detection_card()
        col.addWidget(self._methods_card)
        self._sync_method_controls()
        col.addWidget(self._build_enhance_card())
        col.addWidget(self._build_magnifier_card())

        col.addStretch(1)
        return wrap

    def _settings_category(self, title: str, subtitle: str = "") -> Section:
        """One settings category, folding the way the core applications' do.

        The same :class:`~spacr.qt.widgets.section.Section` a core module's
        settings panel is built from -- the chevron heading, the uppercase
        title, the card it draws -- so a category here looks and folds like
        one there. Its body is a plain vertical layout, which is what the
        panel's controls were laid out in.

        Categories START OPEN, unlike a core module's. This panel is the
        editor's tool column, used between strokes, and a first visit that
        showed seven closed headings would hide the brush radius behind a
        click. What the user folds is remembered, per category, through
        :func:`spacr.qt.preferences.set_section_layout`, and comes back
        folded on the next visit.

        :param title: the category's name, as written; the heading translates
            and uppercases it.
        :param subtitle: an optional sentence under the heading.
        :returns: the category, with its layout as ``body_layout``.
        """
        section = Section(title, expanded=True)
        body = QWidget(section)
        body.setObjectName("MakeMasksCategoryBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])
        if subtitle:
            note = QLabel(subtitle, body)
            note.setObjectName("CardSubtitle")
            note.setWordWrap(True)
            layout.addWidget(note)
        section.add_prose(body)
        section.body_layout = layout
        if title in self._folded_categories:
            section.set_expanded(False)
        section.toggled.connect(self._remember_folded_categories)
        self._settings_categories.append((title, section))
        return section

    def _remember_folded_categories(self, *_args) -> None:
        """Store which settings categories are folded away, by title."""
        folded = [title for title, section in self._settings_categories
                  if not section.is_expanded()]
        self._folded_categories = set(folded)
        try:
            from ..preferences import set_section_layout

            set_section_layout(_SETTINGS_LAYOUT_KEY, folded=folded)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not store the folded categories", exc_info=True)

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
        QShortcut(QKeySequence("M"), self, self._toggle_magnifier_key)
        QShortcut(QKeySequence("Escape"), self, self._on_reset_zoom)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    def _toggle_magnifier_key(self) -> bool:
        """Turn the live magnifier on or off from the keyboard: the M key.

        THE SAME TOGGLE AS THE TOOL-ROW BUTTON, pressed rather than
        imitated: ``toggle()`` on :attr:`_btn_magnifier` sends the button's
        own ``toggled`` signal, so the status line, the box and the button's
        checked look go through :meth:`_on_toggle_magnifier` exactly as a
        click on it does, and the two can never disagree about whether the
        magnifier is on. A button that is disabled or hidden is not pressed
        by its key either.

        WHY M. Every other letter this screen binds names its tool (B brush,
        E erase, W wand, D draw, V divide, Z zoom, R recrop), and M is the
        first free letter of "magnifier". It is bare, like theirs, so it
        carries no Ctrl and therefore no Command on macOS, where Cmd+M
        minimises the window. :data:`spacr.qt.shortcuts.SCREEN_SHORTCUTS`
        lists it with this screen as its scope, under the card's own caption
        "Live magnifier", which the generated catalog already translates.

        :returns: whether the magnifier changed state.
        """
        button = getattr(self, "_btn_magnifier", None)
        if button is None or not button.isEnabled() \
                or not button.isVisibleTo(self):
            return False
        button.toggle()
        return True

    def _set_mode(self, mode: str):
        """Select an editing, navigation or measurement tool.

        :param mode: the mode's name.
        """
        from ..i18n import tr

        self._canvas.mode = mode
        self._canvas.ruler.set_active(mode == MODE_RULER)
        if mode == MODE_RULER:
            self._btn_magnifier.setChecked(False)
        row = getattr(self, '_shortcut_rows', {}).get('Right button')
        if row is not None:
            row[1].setText(tr('Clear the ruler line') if mode == MODE_RULER
                           else tr('Sweep away the objects it passes'))
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

    def _on_brush_size_changed(self, v: int):
        """Resize the brush.

        :param v: the new radius in pixels.
        """
        self._canvas.brush_radius = int(v)
        self._brush_size_label.setText(f"{v} px")

    def _close_levels(self):
        """Close the levels editor before its field or inversion changes."""
        dialog = self._levels_dialog
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
            self._levels_dialog = None

    def _on_levels(self):
        """Open an interactive histogram for the current normalization levels."""
        image = self._canvas.displayed_source()
        if image is None:
            return
        if self._levels_dialog is not None and not self._levels_dialog.closed:
            self._levels_dialog.raise_()
            self._levels_dialog.activateWindow()
            return
        self._close_levels()
        dialog = _LevelsDialog(image, (self._norm_lo.value(), self._norm_hi.value()), self)
        self._levels_dialog = dialog
        dialog.detect.setChecked(self._detect_normalized.isChecked())
        dialog.detect.toggled.connect(self._detect_normalized.setChecked)
        dialog.levels_changed.connect(self._set_levels)
        dialog.show()

    def _set_levels(self, low, high):
        """Apply both histogram percentiles together, refreshing only once."""
        for control, value in ((self._norm_lo, low), (self._norm_hi, high)):
            blocked = control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(blocked)
        self._on_normalize_changed(0)

    def _on_normalize_changed(self, _v: float):
        """Re-stretch the displayed intensity range.

        The loaded pixels and existing masks are untouched. Detection also
        uses the stretch when Detect on the normalized image is enabled.

        :param _v: the changed value; both ends are re-read from the widgets.
        """
        self._canvas.norm_lo = float(self._norm_lo.value())
        self._canvas.norm_hi = float(self._norm_hi.value())
        if self._levels_dialog is not None and not self._levels_dialog.closed:
            self._levels_dialog.set_percentiles(self._canvas.norm_lo, self._canvas.norm_hi)
        self._canvas.refresh()
        if self._canvas.detect_on_normalized:
            self._on_magnifier_context_changed()

    def _on_detect_normalized(self, on: bool) -> None:
        """Hand the detectors the stretched field, or the loaded one.

        :param on: whether detection reads the normalized image.
        """
        self._canvas.detect_on_normalized = bool(on)
        if self._levels_dialog is not None and not self._levels_dialog.closed:
            blocked = self._levels_dialog.detect.blockSignals(True)
            self._levels_dialog.detect.setChecked(bool(on))
            self._levels_dialog.detect.blockSignals(blocked)
        self._canvas.refresh()
        self._on_magnifier_context_changed()
        self._status_label.setText(
            "Detection reads the image as drawn (normalized)." if on else
            "Detection reads the image as loaded.")

    def _on_invert_display(self, on: bool) -> None:
        """Draw the image as its own negative, or stop.

        There is ONE of these. The screen used to carry two inversions that
        a user could hold in disagreeing states: this one, which drew a
        negative and left detection unaffected, and "Invert for detection",
        which inverted what the detectors read and drew nothing. The reason
        to invert is the reason they are merged -- inverting a field of dark
        objects so that Otsu and the magnifier can find them is one
        intention, and a curator who inverts the picture to see dark objects
        means the detector to see them too.

        So this switch now drives both, through
        :func:`spacr.qt.mask_engine.invert_normalized`, and
        ``_cp_invert`` is this same widget under its old name rather than a
        second one that can disagree with it.

        WHAT STILL READS THE LOADED PIXELS, and the status line says it:
        the Filter category and the mask that is saved. THE HOVER READOUT
        NO LONGER DOES: it once reported the loaded value
        while the picture showed the negative, and since the readout is the
        instrument a curator checks an inversion WITH, that read as the
        switch doing nothing. The pixel intensity now follows the picture
        and says "(inverted)"; the object mean still follows the Filter,
        and says "(as loaded)" so the two cannot be confused.
        Detection no longer does, which is the point of the switch, and the
        warning banner above the image says that for as long as it is on.

        :param on: the switch's new state.
        """
        self._close_levels()
        self._canvas.invert_display = bool(on)
        self._canvas.refresh()
        if self._canvas.image is None:
            return
        if not on:
            self._status_label.setText("Showing the image as it was loaded.")
        else:
            self._status_label.setText(
                "Showing the image inverted — dark is bright — and "
                "detecting on it. The hover intensity follows the picture "
                "and says so; filtering, the object mean and saving still "
                "use the original pixels."
            )

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

        IT IS ALSO WHERE THE FILTER'S REMOVAL ROWS ARE DROPPED, because it
        is the one place every edit passes through. Those rows promise to
        name objects in the mask ON SCREEN (:meth:`_set_filter_log`), and
        the promise is broken by the next edit whatever it was: Ctrl+Z puts
        a removed object back under the id the row still lists, and a detect
        in replace mode rebuilds the mask around it. Any recorded edit that
        is not the filter's own therefore empties the box, and the user
        presses Filter again to ask the question again.

        :param kind: what happened, as the curation ledger names it.
        """
        self._refresh_secondary_report()
        if int(n_changed) <= 0:
            return None
        if kind != "filter":
            self._set_filter_log([])
        if kind in ("undo", "redo"):
            self._filter_baseline = None
            self._filter_shown = None
        if self._log is None:
            return None
        if getattr(self._canvas, 'preserve_ids', False):
            detail.update(self._secondary_detail())
        return self._log.append(kind, target, n_changed=int(n_changed),
                                 **detail)

    def _require_primary_source(self):
        """Require a validated primary snapshot belonging to this queue field."""
        from ..i18n import tr

        source = self._primary_selector.snapshot
        if source is None:
            raise ValueError(tr('Load a valid primary mask before growing secondary objects.'))
        filename = self._image_files[self._current_index]
        image_path = os.path.realpath(os.path.join(self._folder, filename))
        if source.image_path != image_path or source.labels.shape != self._canvas.mask.shape:
            raise ValueError(tr('The primary mask belongs to a different field. Reload it for this image.'))
        source.validate_destination(engine.mask_save_path(
            self._folder, filename, **self._layout_kwargs()))
        return source

    def _on_primary_source_changed(self):
        """Invalidate asynchronous detections as soon as their primary changes."""
        self._refresh_secondary_report()
        if hasattr(self, '_magnifier'):
            self._magnifier.refresh()

    def _require_secondary_merge(self, source):
        """Refuse numeric ID collisions with an unrelated nonempty target mask.

        :param source: the validated primary-mask snapshot for this detection.
        :raises ValueError: when the existing output belongs to another source
            or was not created with primary IDs preserved.
        """
        from ..i18n import tr

        if self._canvas.mask is None or not self._canvas.mask.any():
            return
        record = getattr(self, '_paired_source', None) or {}
        expected = source.provenance()
        if not getattr(self._canvas, 'preserve_ids', False) or expected != {
                key: record.get(key) for key in expected}:
            raise ValueError(tr('The existing mask is not paired with this primary source. Use whole-image Replace or Clear all objects before accepting secondary objects.'))

    def _refresh_secondary_report(self):
        """Show explicit missing, orphaned and incompletely enclosing IDs."""
        from ..i18n import tr

        label = getattr(self, '_secondary_relations', None)
        if label is None:
            return
        source = self._primary_selector.snapshot
        mask = self._canvas.mask
        if source is None or mask is None or source.labels.shape != mask.shape:
            label.setText(tr('Load a primary mask to inspect object relationships.'))
            return
        report = engine.primary_secondary_report(source.labels, mask)
        names = (('matched_ids', tr('Matched')), ('missing_secondary_ids', tr('Missing secondary')),
                 ('orphan_secondary_ids', tr('No primary')), ('incomplete_primary_ids', tr('Primary not enclosed')),
                 ('unexpanded_primary_ids', tr('Not expanded')))
        label.setText('\n'.join(tr('{name}: {count} ({ids})', name=name,
                                    count=len(getattr(report, key)),
                                    ids=', '.join(map(str, getattr(report, key)[:12])) +
                                    ('…' if len(getattr(report, key)) > 12 else ''))
                                for key, name in names))

    def _secondary_detail(self):
        """Source association and exact identity diagnostics for the current edit."""
        record = getattr(self, '_paired_source', None)
        if not getattr(self._canvas, 'preserve_ids', False) or record is None:
            return {}
        detail = {'preserve_ids': True, 'primary_source': record}
        source = self._primary_selector.snapshot
        if source is not None and source.provenance() == {key: record.get(key) for key in source.provenance()}:
            report = engine.primary_secondary_report(source.labels, self._canvas.mask)
            detail['primary_secondary'] = {key: list(value) for key, value in report._asdict().items()}
        return detail

    def _retain_secondary_ids(self, record):
        """Associate an accepted detection with its immutable primary identity."""
        self._paired_source = dict(record)
        self._canvas.preserve_ids = True
        self._canvas._lookup = None
        self._canvas._lookup_dirty = True
        self._refresh_secondary_report()
        if self._log is not None:
            previous = next((edit.detail.get('primary_source') for edit in reversed(self._log.edits)
                             if edit.detail.get('preserve_ids')), None)
            if previous != self._paired_source:
                self._log.append('secondary_source', None, **self._secondary_detail())

    def _validate_secondary_save(self):
        """Keep every recorded primary file separate from the editable output."""
        from ..secondary_masks import _same_file
        from ..i18n import tr

        destination = engine.mask_save_path(self._folder, self._image_files[self._current_index],
                                            **self._layout_kwargs())
        records = [getattr(self, '_paired_source', None)]
        if self._log is not None:
            records.extend(edit.detail.get('primary_source') for edit in self._log.edits)
        source = self._primary_selector.snapshot
        if source is not None:
            source.validate_destination(destination)
        for record in records:
            if record and record.get('path') and _same_file(record['path'], destination):
                raise ValueError(tr('Primary and secondary masks must be saved to different files.'))

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

    def _filter_rules(self) -> list:
        """The filter list as the rows say, serialised for the engine."""
        return self._filter_list.filters()

    @staticmethod
    def _filter_number(name: str, value: float) -> str:
        """A bound or a measurement, as the ledger prints it."""
        value = float(value)
        if name.startswith("intensity_"):
            return f"{value:.2f}"
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.4g}"

    def _filter_removal_line(self, removal) -> str:
        """One hidden object as the Filter category's ledger prints it.

        The line reads like "Object 22 with area 31 px and intensity 12.50
        was removed by minimum area 40", with each bound the object fell
        outside and the bound's own number: a row saying only which filter
        removed an object leaves the reader looking for the row it came
        from. A property other than area and mean intensity also gives the
        object's own value, since that is what the bound judged.

        The id is :func:`mask_engine.canonical_labels`' id, which is the id
        the hover readout showed for the same object.

        :param removal: a :class:`mask_engine.ObjectRemoval`.
        """
        from ..i18n import tr

        names = {"area": tr("area"), "intensity_mean": tr("intensity")}
        reasons = []
        for failed in removal.failed:
            side = tr("minimum") if failed.side == "min" else tr("maximum")
            label = names.get(failed.property, failed.property)
            reason = f"{side} {label} {self._filter_number(failed.property, failed.bound)}"
            if failed.property not in names:
                reason += " " + tr("(was {value})", value=self._filter_number(
                    failed.property, failed.value))
            reasons.append(reason)
        area = int(round(removal.values.get("area", 0)))
        mean = removal.values.get("intensity_mean")
        head = tr("Object {label} with area {area} px", label=removal.label,
                  area=area)
        if mean is not None:
            head += " " + tr("and intensity {mean}", mean=f"{mean:.2f}")
        return (head + " " + tr("was removed by") + " "
                + (" " + tr("and") + " ").join(reasons))

    def _set_filter_log(self, lines) -> None:
        """Show ``lines`` in the Filter category's removal ledger.

        Empty puts the box back to its placeholder rather than to a blank
        red field: the rows name objects in the mask ON SCREEN, so carrying
        the last field's rows into the next field would name objects that
        are not there.
        """
        from ..i18n import tr

        self._filter_log.setPlainText("\n".join(str(line) for line in lines))
        self._filter_log.setPlaceholderText(tr(
            "Nothing is hidden. Add a filter above and set a bound."))

    def _refresh_filter_properties(self) -> None:
        """Offer the intensity properties only while an image is open."""
        self._filter_list.set_intensity_available(
            self._canvas.image is not None)

    def _on_filter_row_added(self, _widget) -> None:
        """Give a new filter row the linked help every other control has."""
        if getattr(self, "_api_tooltip_filter", None) is None:
            return
        from ..widgets.make_masks_help import install_make_masks_help
        install_make_masks_help(self)

    def _on_filters_changed(self) -> None:
        """Apply the edited filter list live to the field on screen."""
        if self._canvas.mask is None or self._canvas.image is None:
            return
        self.apply_object_filter(on_load=False)

    def _filter_base(self):
        """The mask the filter list is applied to, with later edits folded in.

        The list HIDES objects rather than deleting them for good: it is
        applied to the mask as it stood before any filter ran, so removing a
        row, or loosening a bound, brings back what that row hid. Edits made
        since the last run (a stroke, a detect, a merge) are carried into
        that baseline pixel for pixel, so re-applying the list does not
        undo them. Undo, redo and opening a field start a new baseline from
        the mask on screen.
        """
        mask = self._canvas.mask
        base = getattr(self, "_filter_baseline", None)
        shown = getattr(self, "_filter_shown", None)
        if (base is None or shown is None or base.shape != mask.shape
                or shown.shape != mask.shape):
            base = np.array(mask, copy=True)
        else:
            edited = shown != mask
            if edited.any():
                base = base.copy()
                base[edited] = mask[edited]
        self._filter_baseline = base
        return base

    def apply_object_filter(self, *, on_load: bool = False) -> int:
        """Apply the filter list to the field on screen; return how many it hides.

        Runs itself when a field loads -- a draft segmentation usually
        arrives with the same class of junk in every field -- whenever the
        list is edited, and when the user presses Filter.

        One engine: :func:`mask_engine.apply_filters`, the same one Mask
        generation's ``object_filters`` setting runs, with one
        ``regionprops_table`` pass for every property the list names. The
        result is one undo step and one ledger entry that records the whole
        list, so the mask's provenance says which filters shaped it.

        Every hidden object is written into the Filter category's ledger,
        one red row each, cleared at the start of every run.

        :param on_load: True when this is the automatic run on opening a
            field. It starts a fresh baseline and keeps a run that hid
            nothing quiet.
        """
        from ..i18n import tr

        self._set_filter_log([])
        self._refresh_filter_properties()
        if on_load:
            self._filter_baseline = None
            self._filter_shown = None
        if self._canvas.mask is None or self._canvas.image is None:
            return 0
        try:
            rules = self._filter_rules()
        except ValueError as error:
            self._status_label.setText(str(error))
            return 0
        if not rules and getattr(self, "_filter_baseline", None) is None:
            if not on_load:
                self._status_label.setText(
                    tr("Object filters: nothing outside the bounds."))
            return 0
        base = self._filter_base()
        try:
            out, removals = engine.apply_filters(
                base, self._canvas.image, rules,
                preserve_ids=getattr(self._canvas, 'preserve_ids', False))
        except ValueError as error:
            self._status_label.setText(str(error))
            return 0
        if out is base:
            out = np.array(base, copy=True)
        changed = int(np.count_nonzero(self._canvas.mask != out))
        self._set_filter_log(
            self._filter_removal_line(removal) for removal in removals)
        if changed:
            dropped = [removal.label for removal in removals]
            self._canvas.mask = out
            self._canvas.refresh()
            self._record("filter", dropped, changed, n_objects=len(dropped),
                         automatic=bool(on_load), filters=rules)
            self._history.push(out)
            self._refresh_history_buttons()
        self._filter_shown = np.array(self._canvas.mask, copy=True)
        if not removals:
            if not on_load:
                self._status_label.setText(
                    tr("Object filters: nothing outside the bounds."))
            return 0
        self._status_label.setText(tr(
            "Object filter removed {count} object(s) — Ctrl+Z to undo",
            count=len(removals)))
        return len(removals)

    def _on_apply_filter(self):
        """Apply the object filter list to the mask on screen."""
        self.apply_object_filter(on_load=False)

    def _cpu_detect(self, image, method: str, otsu: dict) -> tuple:
        """Run the CPU method ``method`` on ``image``.

        The one place the detect button's method is dispatched, so the
        button and the magnifier's box cannot end up running different
        things under one name. A model mode (Cellpose, a backend) falls
        back to Otsu, because those have the Object detection button.

        Otsu's saved local toggle and class/band choices cannot override
        another named threshold. Multi-Otsu alone reads the class and band;
        every other named threshold forces a two-class, non-Local-Otsu run.

        :returns: ``(labels, centres)``; ``centres`` is the number of
            maxima for the propagation and None for everything else.
        """
        if method in organelle_modes.MODE_LABELS:
            return (organelle_modes.segment(
                image, method, self._method_params(),
                min_area=self._detect_min_area()), None)
        if method == cpu_modes.PROPAGATE:
            found = cpu_modes.propagate(
                image, self._cpu_params(), min_area=self._detect_min_area(),
                fill_holes=bool(otsu["fill_holes"]))
            return (found.labels, found.seeds)
        if method == cpu_modes.SECONDARY:
            found = cpu_modes.secondary(image, self._require_primary_source().labels,
                                         self._cpu_params(), min_area=self._detect_min_area(),
                                         fill_holes=bool(otsu['fill_holes']))
            return found.labels, None
        algorithm = cpu_modes.engine_algorithm(
            method if method in cpu_modes.THRESHOLD_LABELS else "otsu")
        settings = dict(otsu)
        if method == cpu_modes.MULTIOTSU:
            settings["classes"] = max(3, int(settings["classes"]))
            settings["local"] = False
        elif method in cpu_modes.THRESHOLD_LABELS:
            settings.update(classes=2, foreground_class=1, local=False)
        return (engine._otsu_instances(
            image, bright=self._otsu_bright.isChecked(),
            min_area=int(self._min_area.value()), algorithm=algorithm,
            local_k=float(self._otsu_local_k.value()), **settings), None)

    def _on_detect_otsu(self):
        """Detect on the whole image and fold the result in per replace/merge.

        IT RUNS WHATEVER THE DETECTION METHOD CATEGORY IS SET TO, not Otsu
        alone: after item 473 that category holds ten threshold
        algorithms, the propagation and the organelle methods, and a
        button that went on running Otsu while the box under the mouse ran
        Li would be two answers to one question. Only a model mode falls
        back, and only because it has a button of its own.

        With Invert on it detects on the INVERTED field
        (:meth:`_detector_image`), which is what lets it take dark objects:
        Bright and Invert together are the dark side of the dark side, and
        the ledger records which way up the image was.
        """
        from ..i18n import tr

        if self._canvas.image is None or self._canvas.mask is None:
            return
        mode = self._combine_mode.currentData()
        otsu = self._otsu_settings()
        correction = otsu["correction"]
        method = canonical_magnifier_mode(
            getattr(self._magnifier, "mode", None))
        if method == cpu_modes.SECONDARY:
            otsu['fill_holes'] = self._secondary_fill_holes.isChecked()
        try:
            detected, seeds = self._cpu_detect(self._detector_image(),
                                               method, otsu)
        except Exception as exc:
            self._warn("Detect failed", str(exc))
            return
        if method not in (cpu_modes.PROPAGATE, cpu_modes.SECONDARY):
            detected = detect_chain.finish(detected, self._detect_chain(),
                                           intensity=self._detector_image())
        found = _object_count(detected)
        centres = ("" if seeds is None
                   else tr(" from {n} centre(s)", n=seeds))
        if not found and method != cpu_modes.SECONDARY:
            self._status_label.setText(tr(
                "{method}{centres} found no objects — the mask is "
                "unchanged. Lower the minimum area, or try the other side.",
                method=_magnifier_mode_label(method), centres=centres))
            return
        try:
            if method == cpu_modes.SECONDARY:
                source = self._require_primary_source()
                if mode != 'replace':
                    self._require_secondary_merge(source)
                out = (engine.canonical_labels(detected, preserve_ids=True) if mode == 'replace'
                       else engine._paste_region_objects(self._canvas.mask, detected, (0, 0),
                                                         overlap='clip', preserve_ids=True)[0])
            else:
                out = engine.combine_masks(self._canvas.mask, detected, mode)
        except Exception as exc:
            self._warn("Detect failed", str(exc))
            return
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        if method == cpu_modes.SECONDARY:
            self._retain_secondary_ids(dict(source.provenance(), selection=self._primary_selector.path.text()))
        self._record("detect", mode, changed, method=method,
                      n_objects=found,
                      invert=bool(self._cp_invert.isChecked()),
                      bright=bool(self._otsu_bright.isChecked()),
                      min_area=int(self._min_area.value()),
                      otsu_correction=correction,
                      otsu_smoothing=otsu["smoothing"],
                      otsu_fill_holes=otsu["fill_holes"],
                      otsu_split=otsu["split_touching"],
                      otsu_exclude_border=otsu["exclude_border"],
                      otsu_classes=otsu["classes"],
                      otsu_foreground_class=otsu["foreground_class"],
                      otsu_local=otsu["local"],
                      otsu_window=otsu["window"],
                      method_parameters=(
                          organelle_modes.provenance(
                              method, self._method_params())
                          or cpu_modes.provenance(method, self._cpu_params())),
                      **self._chain_provenance())
        self._history.push(out)
        self._refresh_history_buttons()
        inverted = (tr(" of the INVERTED image")
                    if self._cp_invert.isChecked() else "")
        described = (self._otsu_description()
                     if method in ("otsu",) + cpu_modes.threshold_modes()
                     else _magnifier_mode_label(method))
        self._status_label.setText(tr(
            "{method} ({how}){inverted}{centres} found {n} object(s) — "
            "{combine}d into the mask",
            method=_magnifier_mode_label(method), how=described,
            inverted=inverted, centres=centres, n=found, combine=mode))

    def _otsu_description(self) -> str:
        """How the threshold was taken, for a status line and the preview.

        Local and multi-class Otsu are two ways of cutting that are not
        "bright" or "dark",
        and a line that went on saying one of those two would be describing
        a run that had not happened.
        """
        settings = self._otsu_settings()
        mode = canonical_magnifier_mode(self._magnifier.mode)
        if settings["local"] or mode in ("sauvola", "niblack"):
            side = "bright" if self._otsu_bright.isChecked() else "dark"
            return f"local {settings['window']} px, {side}"
        classes = settings["classes"]
        if classes > 2:
            return (f"{classes} classes, class "
                    f"{settings['foreground_class']}")
        return "bright" if self._otsu_bright.isChecked() else "dark"

    def _on_show_otsu_histogram(self) -> None:
        """Snapshot a threshold preview and compute it off the GUI thread.

        The historical method name is retained for callers. All named CPU
        thresholds use their own levels on the full detector input, including
        inversion, normalization, enhancement and smoothing. Local methods
        show counts without a misleading global marker. Reopening replaces
        the previous snapshot; only the newest result may update its dialog.
        """
        from ..i18n import tr

        mode = canonical_magnifier_mode(self._magnifier.mode)
        if mode != "otsu" and mode not in cpu_modes.THRESHOLD_LABELS:
            return
        image = self._canvas.image
        if image is None:
            self._status_label.setText(
                tr("Open a field first — a histogram needs an image."))
            return
        previous = self._otsu_histogram_dialog
        if previous is not None:
            previous.close()
            previous.deleteLater()
        dialog = _OtsuHistogramDialog(
            np.zeros(2), np.arange(3), [], self._otsu_description(),
            parent=self, method=mode, pending=True)
        request = _ThresholdHistogramRequest(
            np.array(image, copy=True), mode, self._otsu_settings(),
            bool(self._otsu_bright.isChecked()), bool(self._canvas.invert_display),
            (float(self._canvas.norm_lo), float(self._canvas.norm_hi))
            if self._canvas.detect_on_normalized else None,
            self._detect_chain(), _RunTicket())
        dialog.request = request
        self._otsu_histogram_dialog = dialog
        dialog.show()
        if self._histogram_worker is None:
            self._histogram_worker = _NewestRequestWorker(
                _threshold_histogram, self._histogram_done, name="spacr-threshold-histogram")
        self._histogram_worker.submit(request)

    def _histogram_done(self, request, result, error) -> None:
        """Deliver a worker result to Qt; the screen may already be destroyed."""
        try:
            self._histogram_delivered.emit((request, result, error))
        except RuntimeError:
            pass

    def _take_histogram(self, payload) -> None:
        """Accept only the currently open histogram's snapshot on the GUI thread."""
        request, result, error = payload
        dialog = self._otsu_histogram_dialog
        if dialog is None or dialog.closed or dialog.request is not request:
            return
        dialog.show_result(result, error)


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

    def _build_view_pane(self) -> QWidget:
        """The views, with the shortcut list down their right side.

        The shortcuts sit to the right of the Mask, Cell probability and
        Flows views -- the three view tabs -- so the pane the
        splitter holds is the tabs and the list side by side, and the list
        travels with the views when the settings are hidden and the image
        takes their width.

        THE LIST IS NOT SETTINGS, so the Settings toggle does not take it
        away: a shortcut list that disappears the moment the screen is
        cleared for work is a list you can only read when you do not need
        it. It hides independently as an EDGE pane of its own splitter. Its handle
        folds it to the right edge and drags it wider, and the image takes
        the room it leaves.

        THE CONSOLE IS UNDER THE LIST (item 507), in the same right-hand
        column, where it was asked for: "to the right of the image and
        below the hot key map". The column is a vertical splitter of the
        list and the console's :class:`FoldSection`, so the console folds to
        its heading at the bottom of the column and drags taller, and the
        whole column still folds away as the one "Shortcuts" pane.
        """
        from ..widgets.collapsible_splitter import CollapsibleSplitter, EDGE

        pane = CollapsibleSplitter(Qt.Horizontal,
                                   persist_key="make_masks::views")
        pane.setObjectName("MakeMasksViewPane")
        self._shortcut_card = self._build_shortcut_panel()
        column = CollapsibleSplitter(Qt.Vertical,
                                     persist_key="make_masks::side")
        column.setObjectName("MakeMasksSideColumn")
        column.add_pane(self._shortcut_card, "Shortcut list", stretch=1)
        self._console_section = column.add_section(
            self._masks_console, "Console", persist_key="make_masks/Console",
            stretch=1, extent=240, minimum=120)
        self._shortcut_panel = column
        pane.add_pane(self._view_tabs, "Views", stretch=1, extent=900)
        pane.add_pane(column, "Shortcuts", mode=EDGE, stretch=0,
                      extent=SHORTCUTS_WIDTH, minimum=SHORTCUTS_WIDTH,
                      fold_key="make_masks/Shortcuts",
                      hint="or drag to make the shortcut list wider")
        self._view_pane = pane
        return pane

    def _report(self, text: str, kind: str = "info") -> None:
        """Say ``text`` in the console and show it in the corner.

        :param text: the line.
        :param kind: ``progress``, ``info``, ``warning`` or ``error``; see
            :meth:`_MasksConsole.say`.
        """
        self._masks_console.say(text, kind)
        self._status_label.set_quietly(text)

    def _report_status(self, text: str) -> None:
        """Copy a new corner text into the console.

        A text that ends in an ellipsis says that something is under way,
        and rewrites the console's one progress line; anything else is a
        line of the scrollback and ends that progress.
        """
        stripped = str(text or "").rstrip()
        running = stripped.endswith(("…", "..."))
        self._masks_console.say(stripped, "progress" if running else "info")

    def _build_shortcut_panel(self) -> QWidget:
        """The gestures, one terse line each.

        Every row comes from :data:`SHORTCUT_HINTS`, so the list and the
        gestures cannot drift apart without the table being edited, and a
        test reads the same table off the built widget.

        Read between strokes rather than studied, which is why nothing here
        is a sentence. The keys sit on their own line ABOVE what they do,
        rather than in a column beside it: measured on a rendered screen, a
        two-column list 230 px wide broke every line of prose after one word.

        IT IS A CARD, with the card's own title and subtitle labels. A bare
        QLabel with an object name the theme does not style takes its colour
        from the palette instead of the stylesheet, and this panel was
        drawing its keys in near-black on the dark canvas -- invisible, and
        visible as such only in a rendered grab.
        """
        from ..i18n import tr

        panel = Card("Shortcuts")
        panel.setObjectName("Card")
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(SPACING["xs"])
        self._shortcut_scroll = QScrollArea(panel)
        self._shortcut_scroll.setWidgetResizable(True)
        self._shortcut_scroll.setFrameShape(QScrollArea.NoFrame)
        self._shortcut_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._shortcut_scroll.setWidget(content)
        panel.body_layout.addWidget(self._shortcut_scroll, 1)
        #: ``keys -> (key label, what it does label)``, so a test can ask the
        #: built panel what it is telling the user rather than re-reading the
        #: table it was built from.
        self._shortcut_rows = {}
        for index, (keys, does) in enumerate(SHORTCUT_HINTS):
            if index:
                body.addSpacing(SPACING["xs"])
            key_label = QLabel(tr(keys), panel)
            key_label.setObjectName("CardSubtitle")
            key_label.setWordWrap(True)
            font = key_label.font()
            font.setBold(True)
            key_label.setFont(font)
            does_label = QLabel(tr(does), panel)
            does_label.setObjectName("Muted")
            does_label.setWordWrap(True)
            key_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            does_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            body.addWidget(key_label)
            body.addWidget(does_label)
            self._shortcut_rows[keys] = (key_label, does_label)
        body.addStretch(1)
        return panel

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

    def _build_cellpose_card(self) -> _MethodGroup:
        """The Object detection settings, and the detect button they drive.

        Called Object detection rather than "Cellpose-SAM", because what
        belongs in it is every model that finds objects and not one of them.
        The Otsu settings have a category of their own.

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

        card = _MethodGroup()
        form = QFormLayout()

        self._cp_model = QComboBox()
        for name in cellpose_model_choices():
            self._cp_model.addItem(name, name)
        #: The zoo download running now, if one is.
        self._cp_download = None
        #: ``zoo key -> path`` of the models downloaded from this box this
        #: session. The zoo files a download under the first free versioned
        #: name, which is not always the entry's own name -- a name ending in
        #: ``_v1`` lands without it -- so the path the download reported is
        #: what says the model is here.
        self._cp_fetched: dict = {}
        self._cp_last_model = 0
        self._fill_zoo_models()
        self._cp_last_model = self._cp_model.currentIndex()
        self._cp_model.currentIndexChanged.connect(self._keep_model_loadable)
        self._cp_model.activated.connect(self._on_model_activated)
        self._cp_model.setToolTip(
            "Which weights segment this field, and the Live magnifier's box "
            "in Cellpose mode. The list is the Cellpose installed on this "
            "machine and every Cellpose model in the model zoo; a zoo model "
            "not downloaded yet is greyed out, and choosing it downloads it "
            "and selects it.")
        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(SPACING["xs"])
        model_row_layout.addWidget(self._cp_model, 1)
        self._cp_model_zoo_btn = QPushButton("Model zoo…", model_row)
        self._cp_model_zoo_btn.setToolTip(
            "Browse the model zoo, download a Cellpose model and segment with "
            "it. The model chosen there is selected in the list beside this "
            "button.")
        self._cp_model_zoo_btn.clicked.connect(
            lambda _checked=False: self._choose_cellpose_model_from_zoo())
        model_row_layout.addWidget(self._cp_model_zoo_btn)
        form.addRow("Model", model_row)
        from ..widgets.eliding import ProgressLine

        self._cp_download_bar = ProgressLine(count_below=True)
        self._cp_download_bar.hide()
        form.addRow(self._cp_download_bar)

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

        self._cp_normalize = Toggle("Normalize each field")
        self._cp_normalize.setChecked(True)
        self._cp_normalize.setToolTip(
            "Percentile-normalize the field before segmenting it, which is "
            "what Cellpose expects. Turn it off only for data already "
            "normalized upstream, where doing it twice changes the result.")
        card.body_layout.addWidget(self._cp_normalize)

        self._cp_invert = self._invert_display

        inverts = QLabel(
            "Detecting dark objects? Turn on Invert image, in the Display "
            "category. It inverts the picture you see AND the pixels these "
            "buttons and the Live magnifier read, because those were two "
            "switches until 2026-09-20 and a curator could have either one "
            "without the other.")
        inverts.setObjectName("CardSubtitle")
        inverts.setWordWrap(True)
        card.body_layout.addWidget(inverts)

        drives = QLabel(
            "The Live magnifier reads these settings too: Cellpose mode uses "
            "the model, both thresholds, the diameter and the normalization, "
            "and DINOCell the cell probability. Choosing another method "
            "above shows that method's settings here instead.")
        drives.setObjectName("CardSubtitle")
        drives.setWordWrap(True)
        card.body_layout.addWidget(drives)

        self._btn_cellpose = QPushButton("Object detection")
        self._btn_cellpose.setIcon(iconset.icon("run"))
        self._btn_cellpose.setCursor(Qt.PointingHandCursor)
        self._btn_cellpose.setToolTip(
            "Segment the open field with the Object detection model and "
            "fold the result "
            "in as the replace/merge setting says. Fills the Cell "
            "probability and Flows tabs with what the run was thinking.")
        self._btn_cellpose.clicked.connect(self._on_detect_cellpose)
        self.add_toolbar_action(self._btn_cellpose)
        return card

    def _build_otsu_card(self) -> _MethodGroup:
        """The Otsu settings, driving both the button and the magnifier.

        The threshold correction once sat at the bottom of the Cellpose-SAM
        category, where a user looking for the Otsu settings had no reason
        to open it. It lives here with the Bright switch, which was a bare
        tick-box beside the detect button, and four more settings.

        ALL SIX DRIVE BOTH PLACES OTSU RUNS -- the Otsu detect button on the
        whole field, and the Live magnifier's Otsu mode on the box under the
        mouse -- so the box under the mouse is a close preview of what the
        button will do rather than a second opinion. The one exception is
        Exclude at the image border, which the magnifier answers with its
        own "Exclude objects touching the box border"; applying both to a
        box would drop everything the box cut twice over.

        THE DEFAULTS ARE THE MAGNIFIER'S, AND THAT MOVES THE BUTTON. Before
        this category the two disagreed in three ways and nothing on the
        screen said so: the magnifier smoothed the region, filled the holes
        and cut a blob with two centres in two, and Otsu detect thresholded
        and labelled and did none of it. A user who set the correction by
        watching the box and then pressed the button got a different mask.
        The three boxes start where the preview has always been; three
        clicks put the plain threshold back, which is what
        :func:`spacr.qt.mask_engine._otsu_instances` still does when it is
        asked for nothing.

        "PREVIEW", NOT "THE SAME FUNCTION", AND THE DIFFERENCE IS MEASURED.
        Closing those three gaps does not make the two one routine.
        :func:`spacr.qt.mask_engine._classical_region_labels` still opens the
        binary image, still offsets Otsu's level by the magnifier's own
        Sensitivity, and still falls back to a noise-floor cut where a region
        holds no two clear populations;
        :func:`spacr.qt.mask_engine._otsu_instances` does none of the three.
        Driven from these defaults over twelve synthetic 96x96 fields of
        three to six bright discs on noise, the two agreed on
        the object COUNT in twelve of twelve and were pixel-identical in two
        of twelve, the other ten differing by 3 to 16 boundary pixels out of
        9,216. So the box tells a curator what the button is about to do; it
        does not promise the same array.
        """
        card = _MethodGroup()
        form = QFormLayout()

        self._otsu_correction = QDoubleSpinBox()
        self._otsu_correction.setDecimals(2)
        self._otsu_correction.setRange(0.1, 5.0)
        self._otsu_correction.setSingleStep(0.05)
        self._otsu_correction.setValue(1.0)
        self._otsu_correction.setToolTip(
            "Threshold factor (default 1; range 0.1 to 5), applied before "
            "foreground selection. For positive global thresholds and local "
            "Otsu, increasing it keeps fewer bright or dark pixels. Sauvola "
            "and Niblack multiply their direct local levels: at positive "
            "levels, increasing it keeps fewer bright but more dark pixels; "
            "negative levels reverse that direction. Multi-Otsu shifts class "
            "boundaries, so a selected middle band can gain and lose pixels.")
        form.addRow("Threshold correction", self._otsu_correction)

        self._otsu_smoothing = QDoubleSpinBox()
        self._otsu_smoothing.setDecimals(1)
        self._otsu_smoothing.setRange(0.0, 10.0)
        self._otsu_smoothing.setSingleStep(0.5)
        self._otsu_smoothing.setValue(OTSU_SMOOTHING)
        self._otsu_smoothing.setToolTip(
            "Gaussian blur, in pixels, before the level is found and before "
            "the image is cut at it. It is what stops a noisy field coming "
            "back as a thousand single-pixel objects. 0 thresholds the raw "
            "data.")
        form.addRow("Smoothing (sigma)", self._otsu_smoothing)
        card.body_layout.addLayout(form)

        self._otsu_bright = Toggle("Objects are brighter than background")
        self._otsu_bright.setChecked(True)
        self._otsu_bright.setToolTip(
            "On: objects are brighter than background, as in fluorescence. "
            "Off: take the dark side instead, for brightfield or stain.")
        card.body_layout.addWidget(self._otsu_bright)

        self._otsu_fill_holes = Toggle("Fill holes inside an object")
        self._otsu_fill_holes.setChecked(True)
        self._otsu_fill_holes.setToolTip(
            "Close the holes inside what was thresholded, before it is "
            "labelled. A nucleus dimmer in the middle than at its rim comes "
            "back as a ring without this. The magnifier's Otsu mode has "
            "always done it and Otsu detect did not, which is why the two "
            "could disagree about the same field; now they both read this "
            "box.")
        card.body_layout.addWidget(self._otsu_fill_holes)

        self._otsu_split = Toggle("Split objects that touch")
        self._otsu_split.setChecked(True)
        self._otsu_split.setToolTip(
            "Cut a blob with two centres in two, at the ridge between them "
            "(a watershed on the distance to the background). A blob with "
            "one centre is left whole, so this is not a splitter that cuts "
            "everything. Off, a pair of touching cells arrives as one "
            "object.")
        card.body_layout.addWidget(self._otsu_split)

        self._otsu_exclude_border = Toggle(
            "Drop objects the image border cuts")
        self._otsu_exclude_border.setChecked(False)
        self._otsu_exclude_border.setToolTip(
            "Leave out the objects the edge of the field runs through. Their "
            "area and their mean intensity are properties of where the frame "
            "fell rather than of the object, so a detection meant to be "
            "measured is better without them. Otsu detect only: the Live "
            "magnifier has its own box-border switch.")
        card.body_layout.addWidget(self._otsu_exclude_border)

        more = QFormLayout()
        self._otsu_classes = QSpinBox()
        self._otsu_classes.setRange(2, 6)
        self._otsu_classes.setValue(2)
        self._otsu_classes.setToolTip(
            "How many brightness bands to cut the field into. 2 is Otsu's "
            "own split, one level between background and objects. Raise it "
            "where a field holds more than two populations — background, a "
            "dim halo and bright nuclei — and the cut moves off the one "
            "compromise level between all three onto the boundary you "
            "actually want, chosen below. Multi-Otsu uses this count in both "
            "magnifier scopes and whole-image detection. Each scope estimates "
            "thresholds from its own pixels; a small region may contain too "
            "few distinct intensities for the requested class count. In plain "
            "Otsu mode, this count still applies only to whole-image detect.")
        more.addRow("Classes", self._otsu_classes)

        self._otsu_foreground = QSpinBox()
        self._otsu_foreground.setRange(0, 5)
        self._otsu_foreground.setValue(1)
        self._otsu_foreground.setToolTip(
            "Which band becomes the objects, counting 0 for the dimmest. "
            "With 3 classes, 2 takes the bright nuclei and 1 takes the dim "
            "halo AROUND them without the nuclei inside it — exactly that "
            "band, which is what more than two classes is for. With more "
            "than two classes this replaces 'Objects are brighter than "
            "background': the number says which side you mean.")
        more.addRow("Foreground class", self._otsu_foreground)

        self._otsu_window = QSpinBox()
        self._otsu_window.setRange(3, 999)
        self._otsu_window.setSingleStep(2)
        self._otsu_window.setValue(OTSU_LOCAL_WINDOW)
        self._otsu_window.setSuffix(" px")
        self._otsu_window.setToolTip(
            "The square the local level is measured in, centred on each "
            "pixel. Make it comfortably bigger than one object and smaller "
            "than the illumination's own scale: too small and the inside of "
            "a large object becomes its own background, so it comes back "
            "hollow; too large and it is the whole-field threshold again. "
            "Even numbers are rounded up, so the square has a centre.")
        more.addRow("Local window", self._otsu_window)

        self._otsu_local_k = QDoubleSpinBox()
        self._otsu_local_k.setDecimals(3)
        self._otsu_local_k.setRange(-2.0, 2.0)
        self._otsu_local_k.setSingleStep(0.05)
        self._otsu_local_k.setValue(0.2)
        self._otsu_local_k.setToolTip(
            "Dimensionless local contrast weight (default 0.2; range -2 to 2). "
            "Niblack uses T=m-k*s: increasing k lowers the threshold and keeps "
            "more bright pixels, fewer dark pixels. Sauvola uses "
            "T=m*(1+k*(s/R-1)), where m is the local mean and s its standard "
            "deviation. Here R=1 because the detector receives floats without "
            "range rescaling; its response to k depends on m and s/R. "
            "Check the preview after changing intensity scale.")
        self._otsu_local_k_label = QLabel("Local k")
        more.addRow(self._otsu_local_k_label, self._otsu_local_k)
        card.body_layout.addLayout(more)

        self._otsu_local = Toggle("Local threshold (uneven illumination)")
        self._otsu_local.setChecked(False)
        self._otsu_local.setToolTip(
            "Find Otsu's level separately in a window around every pixel "
            "instead of once for the whole field. A corner that the lamp "
            "falls away from is then judged against its own corner, so the "
            "objects in it stop being lost while the bright middle stays "
            "clean. It costs more objects on a noisy background, which the "
            "minimum area is there to take back. Two classes only, and Otsu "
            "detect only.")
        self._otsu_local.toggled.connect(self._sync_otsu_controls)
        self._otsu_classes.valueChanged.connect(self._sync_otsu_controls)
        self._otsu_classes.valueChanged.connect(self._on_magnifier_context_changed)
        self._otsu_foreground.valueChanged.connect(self._on_magnifier_context_changed)
        self._otsu_local_k.valueChanged.connect(
            self._on_magnifier_context_changed)
        self._otsu_window.valueChanged.connect(
            self._on_magnifier_context_changed)
        card.body_layout.addWidget(self._otsu_local)

        min_area_note = QLabel(
            "Objects under the Min area in Object operations are dropped "
            "after the threshold."
        )
        min_area_note.setWordWrap(True)
        card.body_layout.addWidget(min_area_note)

        self._btn_otsu_hist = QPushButton("Show histogram and level")
        self._btn_otsu_hist.setCursor(Qt.PointingHandCursor)
        self._btn_otsu_hist.setToolTip(
            "Preview the full-field detector input after inversion, normalization, "
            "enhancement and smoothing, with the selected method's thresholds. "
            "Local methods have no single threshold marker. Computation runs in "
            "the background; close the window to discard it. Open again to "
            "refresh after changing settings.")
        self._btn_otsu_hist.clicked.connect(self._on_show_otsu_histogram)
        card.body_layout.addWidget(self._btn_otsu_hist)
        self._sync_otsu_controls()
        return card

    def _build_methods_card(self) -> _MethodGroup:
        """The parameters of organelle detection's methods, one mode at a time.

        Six of the Mode box's rows are organelle detection's own methods
        (:mod:`spacr.qt.organelle_modes`), and each reads different numbers:
        a block size and an offset, a pair of sigmas, a list of filament
        widths, two hysteresis levels, a checkpoint. Putting all nineteen on
        the panel at once would put eighteen controls that are being ignored
        in front of a curator who cannot tell which is which -- the argument
        :meth:`_sync_otsu_controls` already makes about three of them.

        SO THE CARD SHOWS ONE MODE'S PARAMETERS AND HIDES THE REST, from
        :data:`spacr.qt.organelle_modes.PARAMETERS_FOR`, which is also what
        the ledger records (:func:`spacr.qt.organelle_modes.provenance`). One
        list, read by the form, by the recorder and by the engine's own
        settings dict, so the three cannot disagree about what a method
        read.

        The sentence at the top is
        :func:`spacr.organelle_types.method_guidance`, built from
        :data:`spacr.organelle_types.LEGAL_METHODS` -- the shapes spaCR
        already records the method as a legal detector for.
        """
        card = _MethodGroup()
        form = self._method_form = QFormLayout()
        #: ``field of MethodParams -> its control``. One control per
        #: parameter and not one per mode-and-parameter, so the block size
        #: the Adaptive mode was tuned at is the block size the Ridge
        #: mode's adaptive threshold then uses.
        self._method_widgets: dict = {}

        def row(field: str, caption: str, widget, tip: str) -> None:
            """Add one parameter row and remember it under its field name."""
            widget.setToolTip(tip)
            self._method_widgets[field] = widget
            if caption:
                form.addRow(caption, widget)
            else:
                form.addRow(widget)

        block = QSpinBox()
        block.setRange(3, 999)
        block.setSingleStep(2)
        block.setValue(51)
        block.setSuffix(" px")
        row("adaptive_block", "Block size", block,
            "The square the local threshold is measured in, centred on each "
            "pixel; the engine forces it odd. A few times the object "
            "diameter is the starting point: too small and the middle of a "
            "large object becomes its own background, too large and it is a "
            "whole-field threshold again.")

        offset = QDoubleSpinBox()
        offset.setDecimals(2)
        offset.setRange(-1000.0, 1000.0)
        offset.setSingleStep(1.0)
        offset.setValue(5.0)
        row("adaptive_offset", "Offset", offset,
            "Subtracted from the Gaussian-weighted local mean (default 5). "
            "Increasing the offset lowers the threshold and keeps more "
            "foreground pixels before cleanup; a negative offset is stricter. "
            "Units follow the processed image: smoothed intensity for Adaptive "
            "threshold, ridge response for Ridge filter with an adaptive "
            "threshold. A raw-intensity offset can overwhelm a 0-to-1 response.")

        morph = QSpinBox()
        morph.setRange(0, 50)
        morph.setValue(3)
        morph.setSuffix(" px")
        row("morph_radius", "Cleanup radius", morph,
            "The disk the detection is closed and opened with after it is "
            "thresholded. Raise it to smooth ragged outlines, lower it to "
            "keep fine detail. The Adaptive mode also pre-smooths with half "
            "of it; the network methods close with half of it.")

        holes = QSpinBox()
        holes.setRange(0, 1_000_000)
        holes.setValue(64)
        holes.setSuffix(" px²")
        row("fill_holes", "Fill holes up to", holes,
            "Holes inside an object smaller than this are filled, so an "
            "object dimmer in the middle than at its rim does not come back "
            "as a ring. 0 leaves every hole where it is.")

        watershed = Toggle("Split touching spots")
        watershed.setChecked(True)
        row("watershed_spots", "", watershed,
            "Grow a watershed from each blob centre instead of stamping a "
            "disk whose radius comes from that blob's own scale. Turn it "
            "off when single spots are being fragmented.")

        log_min = QDoubleSpinBox()
        log_min.setDecimals(2)
        log_min.setRange(0.1, 100.0)
        log_min.setValue(1.0)
        row("log_min_sigma", "Min sigma", log_min,
            "The smallest scale searched, in pixels; a blob's radius is "
            "about sigma times the square root of two, so 1 finds roughly "
            "1.4 px puncta. Raise it to ignore single-pixel noise.")

        log_max = QDoubleSpinBox()
        log_max.setDecimals(2)
        log_max.setRange(0.1, 200.0)
        log_max.setValue(10.0)
        row("log_max_sigma", "Max sigma", log_max,
            "The largest scale searched, in pixels. Raise it to catch large "
            "puncta; the filter runs once per scale, so it costs time.")

        log_num = QSpinBox()
        log_num.setRange(1, 50)
        log_num.setValue(10)
        row("log_num_sigma", "Scales", log_num,
            "How many scales are evaluated between the two sigmas. More "
            "resolves a wider spread of spot sizes and costs one filter "
            "pass each; 3 to 5 is enough when the spots are all one size.")

        log_thresh = QDoubleSpinBox()
        log_thresh.setDecimals(4)
        log_thresh.setRange(0.0001, 1.0)
        log_thresh.setSingleStep(0.005)
        log_thresh.setValue(0.01)
        row("log_threshold", "Blob threshold", log_thresh,
            "The blob response a spot has to reach to be kept. Lower it to "
            "find fainter spots and more noise. DoG reads this one too: it "
            "has no threshold of its own.")

        dog_low = QDoubleSpinBox()
        dog_low.setDecimals(2)
        dog_low.setRange(0.1, 100.0)
        dog_low.setValue(1.0)
        row("dog_sigma_low", "Low sigma", dog_low,
            "The smaller of the two Gaussians, in pixels: the finest "
            "detail kept. Raise it to suppress noise.")

        dog_high = QDoubleSpinBox()
        dog_high.setDecimals(2)
        dog_high.setRange(0.1, 200.0)
        dog_high.setValue(3.0)
        row("dog_sigma_high", "High sigma", dog_high,
            "The larger of the two Gaussians, in pixels. Scales step up "
            "from the low sigma by a factor of 1.6 until this bound, so a "
            "wider gap covers more spot sizes and costs more passes.")

        ridge_filter = QComboBox()
        for name in ("frangi", "sato", "meijering"):
            ridge_filter.addItem(name, name)
        row("ridge_filter", "Ridge filter", ridge_filter,
            "Which vesselness filter measures how tube-like each pixel's "
            "neighbourhood is. Frangi is the usual choice; Sato responds "
            "more to bright tubes, Meijering to thin neurite-like ones.")

        ridge_sigmas = QLineEdit("1, 2, 3")
        row("ridge_sigmas", "Filament widths", ridge_sigmas,
            "The scales the filter looks for, in pixels, separated by "
            "commas; each should be about the half-width of a filament. Add "
            "a larger value for thick bundles and keep the small ones for "
            "fine tubules. Runtime grows with the list.")

        ridge_threshold = QComboBox()
        ridge_threshold.addItem("Otsu", "otsu")
        ridge_threshold.addItem("Adaptive", "adaptive")
        row("ridge_threshold", "Cut the response at", ridge_threshold,
            "How the filter's response becomes a foreground. Otsu takes one "
            "level for the whole region; Adaptive uses the block size and "
            "offset above and keeps faint filaments in dim corners, at the "
            "cost of background elsewhere.")

        skeleton = Toggle("Reduce to a one-pixel skeleton")
        row("skeletonize", "", skeleton,
            "Label the centre line of the network instead of the filled "
            "filaments, so a measured area tracks network LENGTH rather "
            "than filament thickness. Leave it off to measure filament "
            "mass.")

        hyst_low = QDoubleSpinBox()
        hyst_low.setDecimals(3)
        hyst_low.setRange(0.0, 1_000_000.0)
        hyst_low.setValue(0.2)
        row("hysteresis_low", "Weak level", hyst_low,
            "Pixels above this are kept only where they connect to a seed "
            "above the strong level. Under 1.0 it is read as a fraction and "
            "becomes that percentile of the region (0.2 is the 20th); 1.0 "
            "and above is an absolute intensity.")

        hyst_high = QDoubleSpinBox()
        hyst_high.setDecimals(3)
        hyst_high.setRange(0.0, 1_000_000.0)
        hyst_high.setValue(0.6)
        row("hysteresis_high", "Strong level", hyst_high,
            "Only pieces holding a pixel above this survive at all, and "
            "they then grow outward down to the weak level. Read as a "
            "percentile below 1.0, as an absolute intensity above it.")

        unet_path = QWidget()
        unet_row = QHBoxLayout(unet_path)
        unet_row.setContentsMargins(0, 0, 0, 0)
        self._unet_path_edit = QLineEdit()
        self._unet_path_edit.setPlaceholderText("model.pt")
        unet_browse = QPushButton("Browse…")
        unet_browse.setCursor(Qt.PointingHandCursor)
        unet_browse.clicked.connect(self._on_pick_unet_model)
        unet_row.addWidget(self._unet_path_edit, 1)
        unet_row.addWidget(unet_browse)
        row("unet_model_path", "U-Net checkpoint", unet_path,
            "A .pt or .pth file holding a model that takes one channel and "
            "returns one channel of logits. HEAVY: the file is loaded and "
            "the network is run, which on a CPU is seconds per box and "
            "minutes per field.")

        unet_threshold = QDoubleSpinBox()
        unet_threshold.setDecimals(3)
        unet_threshold.setRange(0.0, 1.0)
        unet_threshold.setSingleStep(0.05)
        unet_threshold.setValue(0.5)
        row("unet_threshold", "Probability cut-off", unet_threshold,
            "Where the network's output is cut. Lower it to recover faint "
            "branches along with false positives; raise it to keep only "
            "confident pixels, which tends to break weak connections.")

        card.body_layout.addLayout(form)
        for widget in self._method_widgets.values():
            for signal in ("valueChanged", "currentIndexChanged",
                           "textChanged", "toggled"):
                changed = getattr(widget, signal, None)
                if changed is not None:
                    changed.connect(self._on_magnifier_context_changed)
        self._sync_method_controls()
        return card

    def _build_detection_card(self) -> Section:
        """Build the shared detection-mode selector and its method controls.

        The mode drives the detect buttons, whole-image runs and Live
        magnifier. The magnifier's size, zoom, scope, overlap rule and
        sensitivity remain in its own category.

        Inside, four :class:`_MethodGroup` s, of which one is shown:
        the threshold family's settings (Otsu's own, and every algorithm in
        :mod:`spacr.qt.cpu_modes` reads them), Cellpose's, the organelle
        methods' and the propagation's. :meth:`_sync_method_controls` is
        what shows one and hides three.
        """
        card = self._settings_category(
            "Detection method",
            "What finds the objects, and the settings that method reads. "
            "Drives the detect buttons and the Live magnifier alike.",
        )
        form = QFormLayout()
        self._mag_mode = QComboBox()
        self._mag_mode.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._mag_mode.setMinimumContentsLength(14)
        self._mag_mode.addItem("Otsu", "otsu")
        from ..i18n import tr

        for source in (cpu_modes.MODE_LABELS, organelle_modes.MODE_LABELS):
            for mode, label in source.items():
                self._mag_mode.addItem(tr(label), mode)
                self._mag_mode.setItemData(
                    self._mag_mode.count() - 1, self._mode_guidance(mode),
                    Qt.ToolTipRole)
        if _cellpose_installed():
            self._mag_mode.addItem("Cellpose", "cellpose")
        self._mag_uninstalled = set()
        for mode, (_backend, label) in _MAGNIFIER_BACKENDS.items():
            self._mag_mode.addItem(label, mode)
        self._resync_magnifier_modes()
        self._mag_mode.setToolTip(
            "Which algorithm finds the objects, for the detect buttons and "
            "for the Live magnifier alike. Otsu and the threshold "
            "algorithms under it cut the field at one level and need "
            "nothing installed; Maxima + propagate grows an object out of "
            "each bright centre, which is how two touching objects come "
            "apart; the organelle methods are the ones organelle detection "
            "runs, through the same code; Cellpose and the backends are "
            "models. Choosing a row changes which settings this category "
            "shows, and the row's own tooltip says what it suits.")
        self._mag_mode.currentIndexChanged.connect(self._on_mode_row_changed)
        self._mag_mode.activated.connect(self._on_magnifier_mode_activated)
        form.addRow("Method", self._mag_mode)
        card.body_layout.addLayout(form)

        self._method_note = QLabel()
        self._method_note.setWordWrap(True)
        card.body_layout.addWidget(self._method_note)

        self._method_groups = {
            "threshold": self._build_otsu_card(),
            "organelle": self._build_methods_card(),
            "propagate": self._build_propagate_card(),
            "secondary": self._build_propagate_card(secondary=True),
            "cellpose": self._build_cellpose_card(),
        }
        for group in self._method_groups.values():
            card.body_layout.addWidget(group)
        return card

    def _mode_guidance(self, mode: str) -> str:
        """What the mode ``mode`` suits, from whichever module owns it.

        A backend is named after Cellpose's sentence: they are all models
        that know what a cell looks like, and what distinguishes them from
        each other is the training set rather than the kind of object.
        """
        if mode in organelle_modes.MODE_LABELS:
            return organelle_modes.guidance(mode)
        if mode in _MAGNIFIER_BACKENDS:
            return cpu_modes.guidance("cellpose")
        return cpu_modes.guidance(mode)

    @staticmethod
    def _mode_family(mode: str) -> str:
        """Which :attr:`_method_groups` family ``mode`` belongs to."""
        if mode in organelle_modes.MODE_LABELS:
            return "organelle"
        if mode == cpu_modes.PROPAGATE:
            return "propagate"
        if mode == cpu_modes.SECONDARY:
            return "secondary"
        if mode == "cellpose" or mode in _MAGNIFIER_BACKENDS:
            return "cellpose"
        return "threshold"

    def _build_propagate_card(self, *, secondary=False) -> _MethodGroup:
        """The settings of Maxima + propagate, an intensity watershed.

        Four steps with a setting each, in the order they run: blur, find
        the maxima, grow, stop. The engine is
        :func:`spacr.qt.mask_engine.maxima_propagate_instances`, whose
        docstring describes the four stop rules, parameter units and
        defaults, and the difference between seed and surviving-label counts.
        ``secondary=True`` selects existing primary masks instead of finding
        centres, with independent settings and a default global threshold.
        """
        from ..i18n import tr

        card = _MethodGroup()
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        widgets = {}
        if secondary:
            from ..widgets.primary_mask_selector import PrimaryMaskSelector
            self._secondary_widgets = widgets
            self._primary_selector = PrimaryMaskSelector(card)
            self._primary_selector.changed.connect(self._on_primary_source_changed)
            card.body_layout.addWidget(self._primary_selector)
            self._secondary_relations = QLabel(tr('Primary/secondary relationships will appear after detection.'))
            self._secondary_relations.setWordWrap(True)
            card.body_layout.addWidget(self._secondary_relations)
        else:
            self._propagate_form = form
            self._propagate_widgets = widgets

        def row(field: str, caption: str, widget, tip: str) -> None:
            """Add one parameter row and remember it under its field name."""
            widget.setToolTip(tip)
            widgets[field] = widget
            if caption:
                form.addRow(caption, widget)
            else:
                form.addRow(widget)

        if secondary:
            growth = QComboBox()
            growth.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            growth.setMinimumContentsLength(14)
            growth.addItem(tr('Intensity watershed'), 'intensity')
            growth.addItem(tr('Distance watershed'), 'distance')
            row('secondary_growth', tr('Growth'), growth,
                tr('Intensity follows bright structures. Distance spreads from primary pixels. Common thresholds constrain the paths; fraction-of-peak trims after growth. Neither is CellProfiler Propagation.'))

        sigma = QDoubleSpinBox()
        sigma.setDecimals(2)
        sigma.setRange(0.0, 50.0)
        sigma.setSingleStep(0.5)
        sigma.setValue(2.0)
        sigma.setSuffix(" px")
        row("propagate_sigma", "Blur first", sigma,
            "Gaussian blur applied before the centres are found, in "
            "pixels. It is what makes ONE object have ONE centre: a raw "
            "object has a dozen maxima in its own noise. About a third of "
            "the object radius is a starting point. This is NOT the Image "
            "enhancement card's denoise, which has already run by now; "
            "leave that one off unless the field is genuinely noisy, or "
            "the two blurs compound.")

        distance = QSpinBox()
        distance.setRange(1, 500)
        distance.setValue(10)
        distance.setSuffix(" px")
        row("propagate_min_distance", "Min centre spacing", distance,
            "No two centres closer together than this, so one object "
            "cannot become two. About one object radius. Raise it when "
            "objects are being split, lower it when two touching objects "
            "come back as one.")

        level = QDoubleSpinBox()
        level.setDecimals(2)
        level.setRange(0.0, 1_000_000.0)
        level.setValue(90.0)
        row("propagate_seed_level", "Centre level", level,
            "How bright a maximum has to be to count as a centre. Read as "
            "a percentile of the blurred image by default, so one setting "
            "suits any exposure: 90 means the top tenth of the pixels. "
            "This and the distance together decide HOW MANY objects there "
            "will be, and the status line says how many centres were "
            "found.")

        percentile = Toggle("Centre level is a percentile")
        percentile.setChecked(True)
        row("propagate_seed_percentile", "", percentile,
            "On: the number above is a percentile of this image. Off: it "
            "is an absolute intensity, which is what to use when the "
            "same setting must mean the same thing across fields of "
            "different exposure.")

        border = Toggle("Drop centres near the edge")
        row("propagate_exclude_border", "", border,
            "Leave out maxima within one minimum distance of the frame. "
            "An object the edge cuts has its centre in the wrong place, "
            "so what grows from it is the wrong shape.")

        stop = QComboBox()
        stop.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        stop.setMinimumContentsLength(18)
        stop.addItem("Fraction of this centre's own peak", "seed_fraction")
        stop.addItem("Absolute intensity", "absolute")
        stop.addItem("Percentile of the image", "percentile")
        stop.addItem("A threshold algorithm's level", "threshold")
        row("propagate_stop", "Grow until", stop,
            tr("Fraction of peak: trim each watershed basin at the chosen "
               "fraction of its seed intensity. This is an intensity ratio, "
               "not a percentile; background offsets affect the result. "
               "The other three rules restrict the watershed with one "
               "threshold for the processed field or region. Hole filling "
               "and minimum-area filtering run afterward."))
        stop.currentIndexChanged.connect(self._sync_propagate_controls)

        stop_value = QDoubleSpinBox()
        stop_value.setDecimals(3)
        stop_value.setRange(0.0, 1_000_000.0)
        stop_value.setValue(0.4)
        row("propagate_stop_value", "Stop at", stop_value,
            "The number the rule above reads: a fraction from 0 to 1 for "
            "the per-centre rule, an intensity for the absolute one, a "
            "percentile from 0 to 100 for the third. Not read when a "
            "threshold algorithm provides the level.")

        algorithm = QComboBox()
        for name in engine.GLOBAL_THRESHOLDS:
            algorithm.addItem(_threshold_label(name), name)
        row("propagate_stop_algorithm", "Stop threshold", algorithm,
            "Which global threshold provides the floor, when the rule "
            "above is a threshold algorithm's level. The same algorithms "
            "the Method box offers on their own.")

        card.body_layout.addLayout(form)
        if secondary:
            self._secondary_fill_holes = Toggle(
                tr('Fill holes inside secondary objects'), word_wrap=True)
            self._secondary_fill_holes.setToolTip(tr(
                'Fill enclosed background holes within each secondary object after growth '
                'and before minimum-area filtering. Pixels belonging to another object are preserved.'))
            self._secondary_fill_holes.setChecked(True)
            self._secondary_fill_holes.toggled.connect(self._on_magnifier_context_changed)
            card.body_layout.addWidget(self._secondary_fill_holes)
            for field, widget in widgets.items():
                form.setRowVisible(widget, field in cpu_modes.PARAMETERS_FOR[cpu_modes.SECONDARY])
            stop.setCurrentIndex(stop.findData('threshold'))
            stop.setItemText(stop.findData('seed_fraction'), tr("Fraction of the primary object's peak"))
            sigma.setToolTip(tr('Gaussian smoothing before growth, in pixels. Primary labels are kept unchanged; no centres are detected.'))
        for widget in widgets.values():
            for signal in ("valueChanged", "currentIndexChanged", "toggled"):
                changed = getattr(widget, signal, None)
                if changed is not None:
                    changed.connect(self._on_magnifier_context_changed)
        self._sync_propagate_controls()
        return card

    def _sync_propagate_controls(self, *_args) -> None:
        """Leave enabled only the propagation boxes that answer anything."""
        for attribute in ('_propagate_widgets', '_secondary_widgets'):
            widgets = getattr(self, attribute, None)
            if widgets:
                rule = str(widgets["propagate_stop"].currentData())
                widgets["propagate_stop_value"].setEnabled(rule != "threshold")
                widgets["propagate_stop_algorithm"].setEnabled(rule == "threshold")

    def _cpu_params(self) -> "cpu_modes.CpuParams":
        """The CPU modes' settings, as the engine's parameters."""
        secondary = getattr(self, '_mag_mode', None) is not None and self._mag_mode.currentData() == cpu_modes.SECONDARY
        widgets = getattr(self, "_secondary_widgets" if secondary else "_propagate_widgets", None)
        if not widgets:
            return cpu_modes.DEFAULT_PARAMS
        return cpu_modes.CpuParams(
            local_k=float(self._otsu_local_k.value()),
            propagate_sigma=float(widgets["propagate_sigma"].value()),
            propagate_min_distance=int(
                widgets["propagate_min_distance"].value()),
            propagate_seed_level=float(widgets["propagate_seed_level"].value()),
            propagate_seed_percentile=bool(
                widgets["propagate_seed_percentile"].isChecked()),
            propagate_exclude_border=bool(
                widgets["propagate_exclude_border"].isChecked()),
            propagate_stop=str(widgets["propagate_stop"].currentData()),
            propagate_stop_value=float(
                widgets["propagate_stop_value"].value()),
            propagate_stop_algorithm=str(
                widgets["propagate_stop_algorithm"].currentData()),
            secondary_growth=(str(widgets['secondary_growth'].currentData())
                              if 'secondary_growth' in widgets else 'intensity'),
        )

    def _on_pick_unet_model(self) -> None:
        """Choose the U-Net checkpoint the U-Net mode is to load."""
        from ..i18n import tr

        path, _filter = QFileDialog.getOpenFileName(
            self, tr("Choose a U-Net checkpoint"), "",
            tr("Torch checkpoints (*.pt *.pth)"))
        if path:
            self._unet_path_edit.setText(path)

    def _sync_method_controls(self, *_args) -> None:
        """Show the chosen mode's parameters and hide every other method's.

        A control that is being read and a control that is being ignored
        look identical; the modes each read four to seven of nineteen, so
        the card would otherwise be mostly controls that do nothing.

        THE CARD ITSELF STAYS, empty but for a sentence, under Otsu,
        Cellpose and the backends, which read none of these parameters. A
        category that comes and goes is a category whose folded state, and
        whose place on the panel, a user cannot learn -- and the note is
        where they are told which category their mode reads instead.
        """
        from ..i18n import tr

        mode = canonical_magnifier_mode(getattr(self._magnifier, "mode", None))
        family = self._mode_family(mode)
        for name, group in getattr(self, "_method_groups", {}).items():
            group.setVisible(name == family)
        shown = organelle_modes.PARAMETERS_FOR.get(mode, ())
        for field, widget in self._method_widgets.items():
            self._method_form.setRowVisible(widget, field in shown)
        self._sync_otsu_controls()
        note = tr(self._mode_guidance(mode)) if self._mode_guidance(mode) \
            else ""
        heavy = organelle_modes.HEAVY_MODES.get(mode)
        if heavy:
            note = f"{note} {tr('Heavy: this mode {what}.', what=tr(heavy))}"
        self._method_note.setText(note)
        for name in ('_enh_morphology', '_enh_morphology_radius', '_enh_split'):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(mode != cpu_modes.SECONDARY)
        self._sync_detect_button(mode)

    def _sync_detect_button(self, mode: str) -> None:
        """Name the whole-image CPU detect button after the chosen method.

        The button runs the method, so it says the method. A model mode
        leaves it reading "Otsu detect", because that is what it falls back
        to and a button that claimed to run Cellpose while running Otsu
        would be the disagreement this fold was meant to end.
        """
        from ..i18n import tr

        button = getattr(self, "_btn_otsu", None)
        if button is None:
            return
        named = ("otsu" if self._mode_family(mode) == "cellpose" else mode)
        button.setText(tr("{method} detect",
                          method=_magnifier_mode_label(named)))

    def _method_params(self) -> "organelle_modes.MethodParams":
        """The Detection methods card as the engine's parameters.

        Read on the GUI thread whenever a request is built, like every
        other setting a model reads, so the box under the mouse and the
        whole-image run cannot be answering under different numbers.
        """
        widgets = getattr(self, "_method_widgets", None)
        if not widgets:
            return organelle_modes.DEFAULT_PARAMS
        return organelle_modes.MethodParams(
            adaptive_block=int(widgets["adaptive_block"].value()),
            adaptive_offset=float(widgets["adaptive_offset"].value()),
            morph_radius=int(widgets["morph_radius"].value()),
            fill_holes=int(widgets["fill_holes"].value()),
            watershed_spots=bool(widgets["watershed_spots"].isChecked()),
            log_min_sigma=float(widgets["log_min_sigma"].value()),
            log_max_sigma=float(widgets["log_max_sigma"].value()),
            log_num_sigma=int(widgets["log_num_sigma"].value()),
            log_threshold=float(widgets["log_threshold"].value()),
            dog_sigma_low=float(widgets["dog_sigma_low"].value()),
            dog_sigma_high=float(widgets["dog_sigma_high"].value()),
            ridge_filter=str(widgets["ridge_filter"].currentData()),
            ridge_sigmas=_parsed_sigmas(widgets["ridge_sigmas"].text()),
            ridge_threshold=str(widgets["ridge_threshold"].currentData()),
            skeletonize=bool(widgets["skeletonize"].isChecked()),
            hysteresis_low=float(widgets["hysteresis_low"].value()),
            hysteresis_high=float(widgets["hysteresis_high"].value()),
            unet_model_path=str(self._unet_path_edit.text()).strip(),
            unet_threshold=float(widgets["unet_threshold"].value()),
        )

    def _build_enhance_card(self) -> Section:
        """The pre- and post-detection chain, for every mode alike.

        :mod:`spacr.qt.detect_chain` is what each row means and
        :data:`spacr.qt.detect_chain.CHAIN_ORDER` is the order they run in,
        which is fixed and is printed at the top of the card rather than
        left to be inferred from the order the rows happen to sit in.

        THE FIRST STAGE IS NOT HERE. The percentile stretch is the Display
        category's "Detect on the normalized image", because its two levels
        are percentiles of the WHOLE FIELD and the magnifier's box is not
        the field. The note says where it is rather than putting a second
        switch for it on this card.

        Apply enables the configured chain for image display and detection;
        Compare previews it independently. "Use in Mask generation" writes
        the configured chain and PSF into the Mask module's settings as the
        ``enhance_*`` and ``psf_*`` keys (:meth:`mask_settings`), so a plate
        run applies the steps tuned here; see
        :func:`spacr.psf_pipeline.prepare_chain`.
        """
        from ..i18n import tr
        from ..widgets.psf_controls import _PSFControls
        from ..widgets.restoration_controls import _RestorationControls

        card = self._settings_category(
            "Image enhancement",
            "Optional steps applied to what the detector reads, in one "
            "fixed order. The image on disk is never changed.",
        )
        order = QLabel(tr(
            "Order: percentile stretch (Display) → background → PSF → restoration → denoise → "
            "contrast → sharpen → detect → morphology → split."))
        order.setWordWrap(True)
        order.setObjectName("Muted")
        card.body_layout.addWidget(order)

        form = QFormLayout()
        self._enh_background = QComboBox()
        self._enh_background.addItem("None", "none")
        self._enh_background.addItem("Rolling ball", "rolling_ball")
        self._enh_background.addItem("Top-hat", "tophat")
        self._enh_background.setToolTip(
            "Subtract the slowly varying background before anything else. "
            "Rolling ball fits a surface of the radius below and takes it "
            "away, which is what flattens uneven illumination; Top-hat "
            "keeps only what is brighter than its surroundings within that "
            "radius and is much faster. Set the radius comfortably LARGER "
            "than the largest object: a radius under the object size eats "
            "the objects with the background. The background is ESTIMATED "
            "on a smaller copy, at the scale below, because a background is "
            "by definition what varies slowly across the field and so is "
            "the one thing that survives being looked at smaller.")
        form.addRow("Background", self._enh_background)

        self._enh_background_radius = QSpinBox()
        self._enh_background_radius.setRange(1, 2000)
        self._enh_background_radius.setValue(50)
        self._enh_background_radius.setSuffix(" px")
        self._enh_background_radius.setToolTip(
            "The ball's or the top-hat disk's radius, in pixels. Larger "
            "than the largest object and smaller than the scale the "
            "illumination itself varies on.")
        form.addRow("Background radius", self._enh_background_radius)

        self._enh_background_scale = QDoubleSpinBox()
        self._enh_background_scale.setDecimals(2)
        self._enh_background_scale.setRange(0.10, 1.00)
        self._enh_background_scale.setSingleStep(0.05)
        self._enh_background_scale.setValue(0.50)
        self._enh_background_scale.setToolTip(
            "What fraction of full size the background is measured at. The "
            "surface is then scaled back up and subtracted from the "
            "full-size image, so only the ESTIMATE is smaller. On one "
            "1,994 px field a rolling ball at radius 50 takes 14.5 s at "
            "1.00 and about 1 s at 0.50. 1.00 is scikit-image's own answer "
            "exactly, for when you want it and will wait; lower it further "
            "on a very large field, and raise it if the surface is missing "
            "illumination that changes over a short distance.")
        form.addRow("Background scale", self._enh_background_scale)

        self._psf_controls = _PSFControls(image_paths=self._current_image_paths)
        form.addRow(self._psf_controls)
        self._restoration_controls = _RestorationControls()
        self._restoration_controls.said.connect(self._report)
        form.addRow(self._restoration_controls)

        self._enh_denoise = QComboBox()
        self._enh_denoise.addItem("None", "none")
        self._enh_denoise.addItem("Gaussian", "gaussian")
        self._enh_denoise.addItem("Median", "median")
        self._enh_denoise.addItem("Bilateral", "bilateral")
        self._enh_denoise.addItem("Non-local means", "nlm")
        self._enh_denoise.addItem(tr("Total variation"), "tv")
        self._enh_denoise.setToolTip(
            "Smooth the noise before the contrast step amplifies it. "
            "Gaussian is a blur and softens edges with the noise; Median "
            "removes speckle and keeps edges; Bilateral and Non-local means "
            "keep edges better still and are much slower. HEAVY: non-local "
            "means is minutes on a whole 2,000 px field, and seconds on a "
            "magnifier box.")
        form.addRow("Denoise", self._enh_denoise)

        self._enh_denoise_strength = QDoubleSpinBox()
        self._enh_denoise_strength.setDecimals(2)
        self._enh_denoise_strength.setRange(0.1, 50.0)
        self._enh_denoise_strength.setValue(1.0)
        self._enh_denoise_strength.setToolTip(
            "How much smoothing: the Gaussian's sigma in pixels, the "
            "median's and the bilateral's disk radius, or the non-local "
            "means' cut-off in multiples of the noise it measures.")
        form.addRow("Denoise strength", self._enh_denoise_strength)

        self._enh_gamma = QDoubleSpinBox()
        self._enh_gamma.setDecimals(2)
        self._enh_gamma.setRange(0.05, 5.0)
        self._enh_gamma.setSingleStep(0.05)
        self._enh_gamma.setValue(1.0)
        self._enh_gamma.setToolTip(
            "The exponent the intensities are raised to on 0..1. Below 1 "
            "lifts the dim end, so faint objects rise out of the "
            "background; above 1 pushes it down and leaves only the bright "
            "ones. 1.00 is off.")
        form.addRow("Gamma", self._enh_gamma)
        card.body_layout.addLayout(form)

        self._enh_percentile_clip = Toggle(tr("Percentile clip"))
        self._enh_percentile_clip.setToolTip(tr(
            "Clip the image to two percentiles of its own intensities before "
            "the contrast curves, so a hot or dead pixel cannot set the range "
            "they are drawn on. Nothing is stretched; intensities keep their "
            "units."))
        card.body_layout.addWidget(self._enh_percentile_clip)
        curve_body = QWidget()
        curve_form = QFormLayout(curve_body)
        curve_form.setContentsMargins(0, 0, 0, 0)
        self._enh_percentile_low = QDoubleSpinBox()
        self._enh_percentile_low.setRange(0.0, 99.9)
        self._enh_percentile_low.setValue(1.0)
        self._enh_percentile_low.setToolTip(tr(
            "The lower percentile of the clip, 0 to 100, below the upper."))
        curve_form.addRow(tr("Clip low percentile"), self._enh_percentile_low)
        self._enh_percentile_high = QDoubleSpinBox()
        self._enh_percentile_high.setRange(0.1, 100.0)
        self._enh_percentile_high.setValue(99.0)
        self._enh_percentile_high.setToolTip(tr(
            "The upper percentile of the clip, 0 to 100, above the lower."))
        curve_form.addRow(tr("Clip high percentile"), self._enh_percentile_high)
        self._enh_percentile_details = self._folded_rows(
            curve_body, "Percentile clip settings", "make_masks/percentile_details")
        card.body_layout.addWidget(self._enh_percentile_details)

        self._enh_log = Toggle(tr("Logarithm"))
        self._enh_log.setToolTip(tr(
            "A logarithmic curve on 0..1, log(1 + gain x) / log(1 + gain): "
            "it compresses the bright end and lifts the dim one, more "
            "strongly near zero than a gamma below 1."))
        card.body_layout.addWidget(self._enh_log)
        log_body = QWidget()
        log_form = QFormLayout(log_body)
        log_form.setContentsMargins(0, 0, 0, 0)
        self._enh_log_gain = QDoubleSpinBox()
        self._enh_log_gain.setRange(0.01, 1000.0)
        self._enh_log_gain.setValue(10.0)
        self._enh_log_gain.setToolTip(tr(
            "What the intensities are multiplied by before the logarithm. "
            "Larger compresses the bright end harder."))
        log_form.addRow(tr("Logarithm gain"), self._enh_log_gain)
        self._enh_log_details = self._folded_rows(
            log_body, "Logarithm settings", "make_masks/log_details")
        card.body_layout.addWidget(self._enh_log_details)

        self._enh_sqrt = Toggle(tr("Square root"))
        self._enh_sqrt.setToolTip(tr(
            "A square-root curve on 0..1, the curve a gamma of 0.5 draws: "
            "it lifts the dim end."))
        card.body_layout.addWidget(self._enh_sqrt)

        self._enh_clahe = Toggle("CLAHE (local histogram equalisation)")
        self._enh_clahe.setToolTip(
            "Equalise the histogram inside each tile rather than over the "
            "whole field, with a limit on how much any one level may be "
            "stretched. It is what brings out objects in a dim corner "
            "without blowing out the bright middle. It also amplifies "
            "noise in empty tiles, which is what the clip limit is for.")
        card.body_layout.addWidget(self._enh_clahe)

        clahe_body = QWidget()
        clahe_form = QFormLayout(clahe_body)
        clahe_form.setContentsMargins(0, 0, 0, 0)
        self._enh_clahe_tile = QSpinBox()
        self._enh_clahe_tile.setRange(8, 1024)
        self._enh_clahe_tile.setValue(64)
        self._enh_clahe_tile.setSuffix(" px")
        self._enh_clahe_tile.setToolTip(
            "The side of one tile, in pixels. Comfortably larger than one "
            "object and smaller than the scale the illumination varies on; "
            "a tile the size of one object equalises the object against "
            "itself.")
        clahe_form.addRow("CLAHE tile", self._enh_clahe_tile)

        self._enh_clahe_clip = QDoubleSpinBox()
        self._enh_clahe_clip.setDecimals(3)
        self._enh_clahe_clip.setRange(0.001, 1.0)
        self._enh_clahe_clip.setSingleStep(0.005)
        self._enh_clahe_clip.setValue(0.01)
        self._enh_clahe_clip.setToolTip(
            "How much contrast a tile may be given, 0 to 1. Higher is more "
            "contrast and more amplified noise in tiles that hold only "
            "background.")
        clahe_form.addRow("CLAHE clip limit", self._enh_clahe_clip)
        self._enh_clahe_details = self._folded_rows(
            clahe_body, "CLAHE settings", "make_masks/clahe_details")
        card.body_layout.addWidget(self._enh_clahe_details)

        self._enh_equalize = Toggle("Histogram equalisation (whole image)")
        self._enh_equalize.setToolTip(
            "Flatten the histogram of the whole region at once, so every "
            "brightness band ends up with the same number of pixels. It is "
            "the strongest of the contrast steps and the least respectful "
            "of the data: a field that is mostly background has its "
            "background stretched across half the range.")
        card.body_layout.addWidget(self._enh_equalize)

        self._enh_sharpen = Toggle("Unsharp mask")
        self._enh_sharpen.setToolTip(
            "Add back a high-pass copy of the image, which makes edges "
            "steeper and helps a threshold land on the boundary rather "
            "than in the halo. Too much amount puts a bright rim around "
            "every object and a dark moat outside it.")
        card.body_layout.addWidget(self._enh_sharpen)

        sharpen_body = QWidget()
        sharpen_form = QFormLayout(sharpen_body)
        sharpen_form.setContentsMargins(0, 0, 0, 0)
        self._enh_sharpen_radius = QDoubleSpinBox()
        self._enh_sharpen_radius.setDecimals(2)
        self._enh_sharpen_radius.setRange(0.1, 50.0)
        self._enh_sharpen_radius.setValue(1.0)
        self._enh_sharpen_radius.setSuffix(" px")
        self._enh_sharpen_radius.setToolTip(
            "The blur the mask is built from, in pixels: about the scale "
            "of the edges to sharpen.")
        sharpen_form.addRow("Sharpen radius", self._enh_sharpen_radius)

        self._enh_sharpen_amount = QDoubleSpinBox()
        self._enh_sharpen_amount.setDecimals(2)
        self._enh_sharpen_amount.setRange(0.0, 10.0)
        self._enh_sharpen_amount.setValue(1.0)
        self._enh_sharpen_amount.setToolTip(
            "How much of the mask is added back. 1 is a normal sharpen; "
            "above 2 the halos start to become objects of their own.")
        sharpen_form.addRow("Sharpen amount", self._enh_sharpen_amount)
        self._enh_sharpen_details = self._folded_rows(
            sharpen_body, "Unsharp mask settings", "make_masks/sharpen_details")
        card.body_layout.addWidget(self._enh_sharpen_details)

        after_form = QFormLayout()
        self._enh_morphology = QComboBox()
        self._enh_morphology.addItem("None", "none")
        self._enh_morphology.addItem("Opening (separate)", "open")
        self._enh_morphology.addItem("Closing (join)", "close")
        self._enh_morphology.addItem("Opening then closing", "open_close")
        self._enh_morphology.setToolTip(
            "Applied to what the detector found, not to the image. Opening "
            "erases what is thinner than the radius, which breaks two "
            "objects joined by a bridge; Closing fills what is thinner, "
            "which joins one object broken into pieces.")
        after_form.addRow("Morphology", self._enh_morphology)

        self._enh_morphology_radius = QSpinBox()
        self._enh_morphology_radius.setRange(1, 50)
        self._enh_morphology_radius.setValue(1)
        self._enh_morphology_radius.setSuffix(" px")
        self._enh_morphology_radius.setToolTip(
            "The disk the opening or closing uses, in pixels. It is a "
            "length: a bridge narrower than twice this is broken, a gap "
            "narrower than twice this is filled.")
        after_form.addRow("Morphology radius", self._enh_morphology_radius)
        card.body_layout.addLayout(after_form)

        self._enh_split = Toggle("Split objects that touch")
        self._enh_split.setToolTip(
            "Cut an object with two centres in two, at the ridge between "
            "them, on the distance to the background. It is the Otsu "
            "category's own split offered to every other method. The Otsu "
            "mode is not affected by this box and keeps using its own: "
            "applying both would join what Otsu had just separated and cut "
            "it again.")
        card.body_layout.addWidget(self._enh_split)

        from ..i18n import tr

        self._enh_heavy = QLabel()
        self._enh_heavy.setWordWrap(True)
        card.body_layout.addWidget(self._enh_heavy)
        actions = QHBoxLayout()

        self._btn_compare = QPushButton("Compare raw and enhanced")
        self._btn_compare.setCursor(Qt.PointingHandCursor)
        self._btn_compare.setToolTip(tr(
            "Preview the configured enhancements beside the unenhanced image, "
            "for the magnifier's box when it has one and for the whole "
            "field otherwise, so every step of the chain can be judged by "
            "looking at what it did. Processing runs in the background. "
            "Cancel closes the comparison; a running filter finishes without "
            "displaying its result. Whole-field normalization is applied "
            "before cropping, as it is for detection."))
        self._btn_compare.clicked.connect(self._on_compare_enhanced)
        actions.addWidget(self._btn_compare)
        self._btn_apply = QPushButton(tr("Apply"))
        self._btn_apply.setCheckable(True)
        self._btn_apply.setCursor(Qt.PointingHandCursor)
        self._btn_apply.setToolTip(tr(
            "Apply these enhancements to the displayed image and subsequent "
            "detections. Blue means active. Click again to use unenhanced "
            "input while keeping these settings and existing masks. "
            "Measurements retain the original image values."))
        self._btn_apply.setStyleSheet(
            "QPushButton:checked { background-color: #2563eb; color: #ffffff; "
            "border: 1px solid #60a5fa; }")
        self._enh_show = self._btn_apply
        self._btn_apply.toggled.connect(self._on_show_enhanced)
        actions.addWidget(self._btn_apply)
        self._btn_to_mask = QPushButton(tr("Use in Mask generation"))
        self._btn_to_mask.setCursor(Qt.PointingHandCursor)
        self._btn_to_mask.setToolTip(tr(
            "Write the configured chain and PSF into the Mask module's "
            "Image Enhancement and Point Spread Function settings, so a "
            "plate run applies these steps to every selected channel after "
            "illumination correction and before normalization. Morphology "
            "and split reshape a detector's labels and stay here."))
        self._btn_to_mask.clicked.connect(self._send_chain_to_mask)
        actions.addWidget(self._btn_to_mask)
        card.body_layout.addLayout(actions)

        for widget in (self._enh_background, self._enh_denoise,
                       self._enh_morphology):
            widget.currentIndexChanged.connect(self._on_chain_changed)
        for widget in (self._enh_background_radius,
                       self._enh_background_scale, self._enh_gamma,
                       self._enh_denoise_strength, self._enh_clahe_tile,
                       self._enh_clahe_clip, self._enh_sharpen_radius,
                       self._enh_sharpen_amount,
                       self._enh_morphology_radius,
                       self._enh_percentile_low, self._enh_percentile_high,
                       self._enh_log_gain):
            widget.valueChanged.connect(self._on_chain_changed)
        for widget in (self._enh_clahe, self._enh_equalize,
                       self._enh_sharpen, self._enh_split,
                       self._enh_percentile_clip, self._enh_log,
                       self._enh_sqrt):
            widget.toggled.connect(self._on_chain_changed)
        self._psf_controls.changed.connect(self._on_chain_changed)
        self._restoration_controls.changed.connect(self._on_chain_changed)
        self._on_chain_changed()
        return card

    @staticmethod
    def _folded_rows(body: QWidget, name: str, key: str) -> QWidget:
        """Put a step's parameter rows under a fold that starts shut (item 509).

        The step's own switch stays visible above it; the fold remembers
        being opened.
        """
        from ..widgets.collapsible_splitter import FoldSection

        section = FoldSection(body, name, persist_key=key, follow_body=False,
                              stretch=0, folded=True)
        section.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        return section

    def _current_image_paths(self) -> list:
        """The open field's file, which "Infer from images…" reads first."""
        files = getattr(self, "_image_files", None) or []
        index = getattr(self, "_current_index", 0)
        if not files or not 0 <= index < len(files):
            return []
        return [os.path.join(self._folder, files[index])]

    def _detect_chain(self) -> "detect_chain.Chain":
        """The applied enhancement chain, or no changes while Apply is off."""
        button = getattr(self, "_btn_apply", None)
        if button is None or not button.isChecked():
            return detect_chain.NO_CHAIN
        chain = self._enhancement_chain()
        if getattr(self, '_mag_mode', None) is not None and self._mag_mode.currentData() == cpu_modes.SECONDARY:
            chain = chain._replace(morphology='none', split=False)
        return chain

    def _enhancement_chain(self) -> "detect_chain.Chain":
        """Configured enhancement settings, available to Compare before applying."""
        if not hasattr(self, "_enh_background"):
            return detect_chain.NO_CHAIN
        return detect_chain.Chain(
            background=str(self._enh_background.currentData()),
            background_radius=int(self._enh_background_radius.value()),
            background_scale=float(self._enh_background_scale.value()),
            denoise=str(self._enh_denoise.currentData()),
            denoise_strength=float(self._enh_denoise_strength.value()),
            gamma=float(self._enh_gamma.value()),
            percentile_clip=bool(self._enh_percentile_clip.isChecked()),
            percentile_low=float(self._enh_percentile_low.value()),
            percentile_high=float(self._enh_percentile_high.value()),
            log=bool(self._enh_log.isChecked()),
            log_gain=float(self._enh_log_gain.value()),
            sqrt=bool(self._enh_sqrt.isChecked()),
            clahe=bool(self._enh_clahe.isChecked()),
            clahe_tile=int(self._enh_clahe_tile.value()),
            clahe_clip=float(self._enh_clahe_clip.value()),
            equalize=bool(self._enh_equalize.isChecked()),
            sharpen=bool(self._enh_sharpen.isChecked()),
            sharpen_radius=float(self._enh_sharpen_radius.value()),
            sharpen_amount=float(self._enh_sharpen_amount.value()),
            morphology=str(self._enh_morphology.currentData()),
            morphology_radius=int(self._enh_morphology_radius.value()),
            split=bool(self._enh_split.isChecked()),
            **self._psf_controls._chain_fields(),
            **self._restoration_controls._chain_fields(),
        )

    def mask_settings(self) -> dict:
        """The configured chain and PSF as the Mask module's settings.

        :func:`spacr.qt.detect_chain.chain_settings` writes the image steps
        as ``enhance_*`` keys and the PSF controls write the ``psf_*`` keys,
        which is exactly what :func:`spacr.psf_pipeline.prepare_chain` reads
        back, so the chain a plate run applies is the chain configured here
        -- whether or not Apply is on, since Apply is this screen's switch.
        """
        settings = detect_chain.chain_settings(self._enhancement_chain())
        controls = getattr(self, "_psf_controls", None)
        if controls is not None:
            settings.update(controls.mask_settings())
        return settings

    def _send_chain_to_mask(self) -> None:
        """Write :meth:`mask_settings` into the Mask module and show it.

        The Mask screen gets the values if it is built, and is built for
        them otherwise; a Timelapse screen already built gets them too, since
        it runs the same preprocessing. Standalone, with no application
        window around this screen, there is nowhere to write and the status
        line says so.
        """
        from ..i18n import tr

        settings = self.mask_settings()
        window = self.window()
        screens = getattr(window, "_screens", None)
        written = []
        for key in ("mask", "timelapse"):
            screen = screens.get(key) if isinstance(screens, dict) else None
            apply = getattr(screen, "apply_settings_dict", None)
            if callable(apply):
                apply(settings)
                written.append(key)
        rebuild = getattr(window, "rebuild_app_screen", None)
        if "mask" not in written and callable(rebuild):
            rebuild("mask", settings)
            written.append("mask")
        if not written:
            self._status_label.setText(tr(
                "Open spaCR's Mask module to receive the enhancement chain."))
            return
        self._status_label.setText(tr(
            "Enhancement chain written to the Mask settings: {steps}.",
            steps=detect_chain.describe(self._enhancement_chain())
            or tr("every step off")))
        navigate = getattr(window, "_on_nav_selected", None)
        if callable(navigate):
            navigate("mask")

    def _chain_provenance(self) -> dict:
        """The chain as a mask's ledger entry records it.

        The percentile stretch is read from the Display category, because
        it is the chain's first stage and lives there; see
        :meth:`_build_enhance_card`.
        """
        return detect_chain.provenance(
            self._detect_chain(),
            percentile_stretch=bool(self._canvas.detect_on_normalized))

    def _on_chain_changed(self, *_args) -> None:
        """A chain step changed: warn about the slow ones and re-detect."""
        from ..i18n import tr

        chain = self._detect_chain()
        heavy = detect_chain.heavy_steps(self._enhancement_chain())
        self._enh_heavy.setText(
            tr("Heavy: {steps}. A whole-image run shows progress and can be "
               "cancelled.", steps=", ".join(tr(step) for step in heavy))
            if heavy else "")
        self._canvas.enhance_chain = chain
        self._canvas._enhance_cancel.set()
        self._canvas._enhance_asked = None
        self._canvas.refresh()
        self._on_magnifier_context_changed()

    def _on_show_enhanced(self, on: bool) -> None:
        """Apply or bypass enhancements for display and subsequent detections."""
        from ..i18n import tr

        self._canvas.enhance_display = bool(on)
        if on:
            self._canvas._enhance_failure = None
        self._on_chain_changed()
        self._magnifier.refresh_view()
        self._status_label.setText(
            tr("Showing the enhanced image the detector reads.") if on else
            tr("Showing the image as loaded."))

    def _on_compare_enhanced(self) -> None:
        """Open the raw image beside the enhanced one, and say what ran.

        The magnifier's box when there is one, because that is the region a
        curator is judging and the region every step ran on live; the whole
        field otherwise. A worker prepares a captured snapshot; closing the
        dialog discards it, and changed fields/settings reject late results.
        """
        from ..i18n import tr

        image = self._canvas.image
        if image is None:
            return
        chain = self._enhancement_chain()
        box = self._magnifier.compare_box()
        self._cancel_comparison()
        old_dialog = getattr(self, '_compare_dialog', None)
        if old_dialog is not None:
            old_dialog.close()
        self._comparison_serial += 1
        request = _CompareRequest(
            key=(self._comparison_serial, self._comparison_context()),
            image=np.array(self._canvas.displayed_source(), copy=True),
            box=tuple(box), chain=chain,
            normalized=bool(self._canvas.detect_on_normalized),
            percentiles=(float(self._canvas.norm_lo), float(self._canvas.norm_hi)),
            cancelled=threading.Event())
        self._comparison_request = request
        self._compare_dialog = _ComparePreview(
            request.image[box[1]:box[3], box[0]:box[2]], None,
            tr("Preparing enhanced image… Cancel closes this comparison; a running filter finishes in the background."), self)
        self._compare_dialog.finished.connect(self._cancel_comparison)
        self._compare_dialog.show()
        if self._comparison_worker is None:
            self._comparison_worker = _NewestRequestWorker(
                _compare_picture_for, self._comparison_done, name='spacr-compare')
        self._comparison_worker.submit(request)

    def _comparison_context(self):
        """Settings and field identity whose comparison remains meaningful."""
        canvas = self._canvas
        return (self._load_token, id(canvas.image), bool(canvas.invert_display),
                bool(canvas.detect_on_normalized), float(canvas.norm_lo),
                float(canvas.norm_hi), self._enhancement_chain())

    def _cancel_comparison(self, *_args):
        """Discard a comparison without waiting for an active native filter."""
        request, self._comparison_request = self._comparison_request, None
        if request is not None:
            request.cancelled.set()
        if self._comparison_worker is not None:
            self._comparison_worker.drop_waiting()

    def _comparison_done(self, request, result, error):
        """Marshal a worker completion back to the owning Qt screen."""
        try:
            self._comparison_delivered.emit((request, result, error))
        except RuntimeError:
            pass

    def _take_comparison(self, payload):
        """Only show results from the still-open, unchanged comparison."""
        from ..i18n import tr

        request, result, error = payload
        if request is not self._comparison_request or request.cancelled.is_set():
            return
        self._comparison_request = None
        if request.key[1] != self._comparison_context():
            self._compare_dialog._show_result(None, tr(
                "The image or enhancement settings changed. Choose Compare again."))
            return
        if error is not None:
            self._compare_dialog._show_result(None, tr(
                "Image enhancement failed: {error}", error=str(error)))
            return
        steps = " → ".join(tr(name) for name in detect_chain.step_names(
            request.chain, percentile_stretch=request.normalized))
        self._compare_dialog._show_result(
            result, steps or tr("No enhancement step is switched on."))

    def _sync_otsu_controls(self, *_args) -> None:
        """Leave enabled only the threshold boxes that answer anything.

        A control that is being read and a control that is being ignored
        look identical, and a curator cannot tell which is which. So:
        the foreground class is a choice only once there is more than one
        band to choose from, the local window only matters while the local
        threshold is on, and the two cannot both be on -- a level per window
        and a split into several bands have no joint meaning, and the engine
        refuses the pair rather than quietly dropping one.
        """
        mode = canonical_magnifier_mode(getattr(self._magnifier, "mode", None))
        multi = mode == cpu_modes.MULTIOTSU
        plain = mode == "otsu"
        window_family = mode in ("sauvola", "niblack")
        if multi and int(self._otsu_classes.value()) < 3:
            self._otsu_classes.setValue(3)
        self._otsu_local_k.setEnabled(window_family)
        self._otsu_local.setEnabled(plain)
        self._otsu_local_k_label.setVisible(window_family)
        self._otsu_local_k.setVisible(window_family)
        classes = int(self._otsu_classes.value())
        local = bool(self._otsu_local.isChecked()) and plain
        bands = (plain or multi) and not local
        self._otsu_classes.setEnabled(bands)
        self._otsu_foreground.setEnabled(bands and classes > 2)
        self._otsu_foreground.setRange(0, max(1, classes - 1))
        if classes > 2 and self._otsu_foreground.value() > classes - 1:
            self._otsu_foreground.setValue(classes - 1)
        self._otsu_window.setEnabled(local or window_family)
        self._otsu_bright.setEnabled(not bands or classes == 2)

    def _otsu_settings(self) -> dict:
        """What the Otsu category says, as :func:`_otsu_instances` keywords.

        One reader for the button and the magnifier, so a setting added to
        the category reaches both by being read here once.

        Multi-Otsu reads the class count and foreground band in both
        magnifier scopes through :meth:`_magnifier_context`. Its local
        Otsu toggle is disabled and ignored because one threshold per
        window cannot be combined with multiple intensity bands. Plain
        Otsu's magnifier keeps its existing region-specific algorithm;
        that mode's class count and Local Otsu toggle remain button-only.
        Every other threshold ignores these saved Otsu settings and uses
        two classes with its named algorithm. Disabled controls retain
        their values for a later return to Otsu or Multi-Otsu.
        """
        mode = canonical_magnifier_mode(self._magnifier.mode)
        local = bool(self._otsu_local.isChecked()) and mode == "otsu"
        bands = mode in ("otsu", cpu_modes.MULTIOTSU) and not local
        return {
            "correction": float(self._otsu_correction.value()),
            "smoothing": float(self._otsu_smoothing.value()),
            "fill_holes": bool(self._otsu_fill_holes.isChecked()),
            "split_touching": bool(self._otsu_split.isChecked()),
            "exclude_border": bool(self._otsu_exclude_border.isChecked()),
            "classes": int(self._otsu_classes.value()) if bands else 2,
            "foreground_class": int(self._otsu_foreground.value()) if bands else 1,
            "local": local,
            "window": int(self._otsu_window.value()),
        }

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

    def _fill_zoo_models(self) -> None:
        """List every Cellpose model in the model zoo in the Model box.

        A model on this machine is listed by its zoo key and carries its path,
        which is what :func:`load_cellpose_model` loads. One that is not
        downloaded is listed greyed, carrying no path and its zoo entry under
        :data:`_ZOO_PENDING_ROLE`: choosing it downloads it
        (:meth:`_on_model_activated`) rather than selecting it, so the box
        never rests on a model there is nothing to load for. Called again
        after the picker closes or a download ends, the zoo rows are rebuilt
        -- so a model just downloaded becomes selectable -- and the model
        chosen stays chosen, without a change signal when it did not change.
        """
        from ..i18n import tr
        from ..model_install import UNINSTALLED_GREY

        combo = self._cp_model
        chosen = combo.currentData()
        combo.blockSignals(True)
        try:
            for index in reversed(range(combo.count())):
                if combo.itemData(index, _ZOO_ROLE):
                    combo.removeItem(index)
            for key, path, entry in _zoo_cellpose_models():
                fetched = self._cp_fetched.get(key)
                if not path and fetched and os.path.isfile(fetched):
                    path = fetched
                if path and combo.findData(path) >= 0:
                    continue
                if path:
                    combo.addItem(key, path)
                    combo.setItemData(combo.count() - 1, path, Qt.ToolTipRole)
                else:
                    combo.addItem(tr("{name} (not downloaded)", name=key))
                    row = combo.count() - 1
                    combo.setItemData(row, entry, _ZOO_PENDING_ROLE)
                    combo.setItemData(row, QBrush(UNINSTALLED_GREY),
                                      Qt.ForegroundRole)
                    combo.setItemData(row, tr(
                        "{name} is not downloaded. Choosing it downloads it "
                        "from the model zoo and selects it.", name=key),
                        Qt.ToolTipRole)
                combo.setItemData(combo.count() - 1, True, _ZOO_ROLE)
            index = combo.findData(chosen) if chosen is not None else -1
            combo.setCurrentIndex(max(index, 0))
        finally:
            combo.blockSignals(False)
        self._cp_last_model = combo.currentIndex()
        if combo.currentData() != chosen:
            combo.currentIndexChanged.emit(combo.currentIndex())

    def _keep_model_loadable(self, index: int) -> None:
        """Never let the Model box rest on a model that is not downloaded.

        A click is handled by :meth:`_on_model_activated`; this catches the
        other ways a row becomes current -- the keyboard, the wheel -- and
        puts the box back on the last model that can be loaded.
        """
        combo = self._cp_model
        if index >= 0 and combo.itemData(index, _ZOO_PENDING_ROLE) is not None:
            back = self._cp_last_model
            if back == index or not 0 <= back < combo.count():
                back = 0
            combo.setCurrentIndex(back)
            return
        self._cp_last_model = index

    def _on_model_activated(self, index: int) -> None:
        """A person chose a Model row: download it if it is not here yet."""
        entry = self._cp_model.itemData(index, _ZOO_PENDING_ROLE)
        if entry is not None:
            self.download_zoo_model(entry)

    def download_zoo_model(self, entry) -> bool:
        """Ask, then download a zoo Cellpose model and select it when it lands.

        The download goes through :func:`spacr.model_zoo.install` on a worker
        thread (:class:`spacr.qt.model_install.CheckpointDownload`), into the
        folder the Model zoo picker uses, so either route finds the file the
        other fetched. The bar under the Model row shows the bytes as they
        arrive, and the screen stays usable. A model that publishes no
        checksum is downloaded only after the user has been told spaCR
        cannot then check it.

        :param entry: the zoo's record of the model.
        :returns: True when a download was started.
        """
        from ..i18n import tr
        from ..model_install import CheckpointDownload, human_bytes
        from ..widgets.model_zoo_picker import remembered_model_dir

        name = str(getattr(entry, "key", "") or getattr(entry, "name", ""))
        running = self._cp_download
        if running is not None and running.is_running():
            self._status_label.setText(tr(
                "A model is already downloading; wait for it to finish."))
            return False
        folder = remembered_model_dir()
        size = human_bytes(getattr(entry, "size_bytes", 0))
        unverified = not str(getattr(entry, "sha256", "") or "")
        text = tr("Download {name}{size} into {folder}?", name=name,
                  size=f" ({size})" if size else "", folder=folder)
        if unverified:
            text += "\n\n" + tr(
                "This model publishes no checksum, so spaCR cannot tell a "
                "truncated or substituted file from the real one.")
        if not self._confirm(tr("Download {name}?", name=name), text):
            return False
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            self._warn(tr("Download failed"), str(exc))
            return False
        job = CheckpointDownload(entry, folder, unverified=unverified)
        self._cp_download = job
        job.progressed.connect(self._on_model_download_progress)
        job.finished.connect(self._on_model_downloaded)
        self._cp_download_bar.setRange(0, 0)
        self._cp_download_bar.setFormat("")
        self._cp_download_bar.set_detail(tr("Downloading {name}…", name=name))
        self._cp_download_bar.show()
        self._status_label.setText(tr(
            "Downloading {name} in the background.", name=name))
        return job.start()

    def _on_model_download_progress(self, done: int, total: int) -> None:
        """Move the download bar; a server that sent no size keeps it busy."""
        bar = self._cp_download_bar
        if total > 0:
            from ... import model_zoo as zoo

            bar.setRange(0, 1000)
            bar.setValue(int(1000 * min(done, total) / total))
            bar.setFormat(f"{zoo._human_bytes(min(done, total))} / "
                          f"{zoo._human_bytes(total)} (%p%)")

    def _on_model_downloaded(self, worked: bool, message: str) -> None:
        """Select the model just downloaded, or say why it did not arrive."""
        from ..i18n import tr

        job, self._cp_download = self._cp_download, None
        self._cp_download_bar.hide()
        name = str(getattr(getattr(job, "entry", None), "key", "") or "")
        if not worked:
            if message != "cancelled":
                self._warn(tr("Download failed"), message)
            return
        if name:
            self._cp_fetched[name] = message
        self._fill_zoo_models()
        combo = self._cp_model
        target = os.path.realpath(message)
        index = next((row for row in range(combo.count())
                      if isinstance(combo.itemData(row), str)
                      and os.path.realpath(combo.itemData(row)) == target),
                     -1)
        if index < 0:
            combo.addItem(name or os.path.basename(message) or message,
                          message)
            index = combo.count() - 1
            combo.setItemData(index, message, Qt.ToolTipRole)
        combo.setCurrentIndex(index)
        self._status_label.setText(tr(
            "{name} is downloaded and selected.", name=name or message))

    def _choose_cellpose_model_from_zoo(self) -> Optional[str]:
        """Open the model zoo on its Cellpose models and select what is picked.

        Cellpose-SAM and Cellpose 3 models, not the zoo's YOLO well detector,
        which no Cellpose can load. A Cellpose 3 model comes back as
        ``cellpose3:<name or path>`` -- a bioimage.io Cellpose 3 checkpoint
        among them -- and :func:`load_cellpose_model` runs it through the
        Cellpose 3 backend, as Mask generation does. A picked model the list
        does not hold is added to it, under its file name.

        :returns: the model setting chosen, or None when the picker was
            cancelled.
        """
        from ..i18n import tr
        from ..widgets import model_zoo_picker
        from ..._segmentation_backends import (_cellpose3_choice,
                                               _cellpose_dino_choice)

        path = model_zoo_picker.choose_model(
            self, kinds=("cellpose", "cellpose3", "cellpose_dino"))
        if not path:
            return None
        path = str(path)
        self._fill_zoo_models()
        index = self._cp_model.findData(path)
        if index < 0:
            chosen = _cellpose3_choice(path)
            dino = _cellpose_dino_choice(path)
            if chosen is not None:
                label = tr("Cellpose 3 · {model}",
                           model=os.path.basename(chosen) or chosen)
            elif dino is not None:
                label = tr("Cellpose-DINO · {model}",
                           model=os.path.basename(dino) or dino)
            else:
                label = os.path.basename(path) or path
            self._cp_model.addItem(label, path)
            self._cp_model.setItemData(self._cp_model.count() - 1, path,
                                       Qt.ToolTipRole)
            index = self._cp_model.count() - 1
        self._cp_model.setCurrentIndex(index)
        return path

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
        """Segment the open field with the chosen model; objects found.

        The two panes are filled BEFORE the mask is touched, and they are
        filled even when the run found nothing at all. A run that returns
        an empty mask is exactly the run whose probability map you need to
        see: it says whether the network found nothing, or found plenty
        and the threshold threw it away.

        This programmatic method is synchronous and returns the object
        count. The toolbar uses a background worker instead, taking a
        snapshot and discarding results after field changes or mask edits.
        If toolbar detection is already running, this method returns zero
        without starting a second run.

        With Invert on the model is given the INVERTED field
        (:meth:`_detector_image`), and the status
        line and the ledger entry both say so.
        """
        if self._detection_request is not None:
            return 0
        if self._canvas.image is None or self._canvas.mask is None:
            self._status_label.setText(
                "Open a folder before running Object detection.")
            return 0

        model_name = self._cp_model.currentData() or "cpsam"
        app = QApplication.instance()
        self._btn_cellpose.setEnabled(False)
        self._status_label.setText(
            f"Object detection ({model_name}) running…")
        if app is not None:
            app.setOverrideCursor(Qt.WaitCursor)
            app.processEvents()
        try:
            with _CELLPOSE_LOCK:
                labels, cellprob, flow = cellpose_detect(
                    self._detector_image(),
                    self._cellpose_model(model_name),
                    diameter=int(self._cp_diameter.value()),
                    normalize=bool(self._cp_normalize.isChecked()),
                    flow_threshold=float(self._cp_flow.value()),
                    cellprob_threshold=float(self._cp_cellprob.value()),
                    min_size=self._detect_min_area(),
                )
        except Exception as exc:
            LOG.exception("Object detection failed")
            self._warn("Object detection failed", str(exc))
            return 0
        finally:
            if app is not None:
                app.restoreOverrideCursor()
            self._btn_cellpose.setEnabled(True)

        labels = detect_chain.finish(labels, self._detect_chain(),
                                     intensity=self._detector_image())
        details = dict(model=model_name, invert=bool(self._cp_invert.isChecked()),
                       cellprob_threshold=float(self._cp_cellprob.value()),
                       flow_threshold=float(self._cp_flow.value()),
                       diameter=int(self._cp_diameter.value()),
                       min_size=self._detect_min_area(), **self._chain_provenance())
        return self._apply_detection((labels, cellprob, flow),
                                     self._combine_mode.currentData(), details)

    def _apply_detection(self, result, mode, details) -> int:
        """Commit one accepted result and its captured provenance on Qt's thread."""
        labels, cellprob, flow = result
        self._show_intermediates(cellprob, flow)
        self._sync_model_choices()
        found = int(labels.max()) if labels.size else 0
        if not found:
            self._status_label.setText(
                "Object detection found no objects — the mask is "
                "unchanged. The "
                "Cell probability tab shows what it had to work with.")
            return 0

        try:
            out = engine.combine_masks(self._canvas.mask, labels, mode)
        except Exception as exc:
            self._warn("Object detection failed", str(exc))
            return 0
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        self._record("detect", mode, changed, method="cellpose",
                      n_objects=found, **details)
        self._history.push(out)
        self._refresh_history_buttons()
        inverted = (" from the INVERTED image"
                    if details['invert'] else "")
        model_name = details['model']
        self._status_label.setText(
            f"Object detection ({model_name}){inverted} found {found} "
            f"object(s) — {mode}d into the mask. See the Cell probability "
            "and Flows tabs."
        )
        return found

    def _on_detect_cellpose(self):
        """Capture the field/settings and start detection without blocking Qt."""
        from ..i18n import tr

        if self._detection_request is not None:
            return
        if self._canvas.image is None or self._canvas.mask is None:
            self._status_label.setText(tr("Open a folder before running Object detection."))
            return
        parameters = dict(diameter=int(self._cp_diameter.value()),
                          normalize=bool(self._cp_normalize.isChecked()),
                          flow_threshold=float(self._cp_flow.value()),
                          cellprob_threshold=float(self._cp_cellprob.value()),
                          min_size=self._detect_min_area())
        model = self._cp_model.currentData() or 'cpsam'
        request = dict(image=np.array(self._canvas.image, copy=True),
                       image_reference=self._canvas.image, token=self._load_token,
                       mask=np.array(self._canvas.mask, copy=True), model=model,
                       parameters=parameters, chain=self._detect_chain(),
                       invert=bool(self._cp_invert.isChecked()),
                       percentiles=(float(self._canvas.norm_lo), float(self._canvas.norm_hi))
                       if self._canvas.detect_on_normalized else None,
                       mode=self._combine_mode.currentData(),
                       details=dict(model=model, invert=bool(self._cp_invert.isChecked()),
                                    **{key: value for key, value in parameters.items() if key != 'normalize'},
                                    **self._chain_provenance()))
        self._detection_request = request
        self._btn_cellpose.setEnabled(False)
        if str(model).startswith('cellpose3') and not parameters.get('diameter'):
            self._report(_cellpose3_auto_diameter_note(), "warning")
        self._status_label.setText(tr("Object detection ({model}) running…", model=model))
        if self._detection_worker is None:
            self._detection_worker = _NewestRequestWorker(
                partial(_detect_cellpose_snapshot, models=self._cp_loaded),
                self._detection_done, name='spacr-object-detection')
        self._detection_worker.submit(request)

    def _detection_done(self, request, result, error) -> None:
        """Send completion to Qt while tolerating a screen already destroyed."""
        try:
            self._detection_delivered.emit((request, result, error))
        except RuntimeError:
            pass

    def _take_detection(self, payload) -> None:
        """Reject stale field/mask results before changing any editor state."""
        from ..i18n import tr

        request, result, error = payload
        if request is not self._detection_request:
            return
        self._detection_request = None
        self._btn_cellpose.setEnabled(True)
        if (request['token'] != self._load_token
                or request['image_reference'] is not self._canvas.image
                or not np.array_equal(request['image'], self._canvas.image, equal_nan=True)
                or not np.array_equal(request['mask'], self._canvas.mask)):
            self._status_label.setText(tr(
                "Detection result discarded because the field or mask changed. Run detection again to use the current field."))
            return
        if error is not None:
            self._status_label.setText(tr("Object detection failed"))
            self._warn(tr("Object detection failed"), str(error))
            return
        try:
            self._apply_detection(result, request['mode'], request['details'])
        except Exception as exc:
            LOG.exception("Object detection result could not be applied")
            self._warn(tr("Object detection failed"), str(exc))

    def _build_magnifier_card(self) -> Section:
        """The live magnifier's settings, and the toggle that turns it on.

        The toggle goes in the tool row beside Object detection, because it
        has to stay reachable with the settings hidden; the four settings the
        request named -- mode, size, zoom, sensitivity -- and the overlap rule
        go here, with what is segmented (the region under the mouse or the
        whole image once), whether objects cut by the box are offered, and
        the progress and Cancel of a whole-image run.

        THE METHOD IS NOT HERE ANY MORE. It moved to the Detection method
        category with item 473, because it is not the box's: the same
        choice drives the detect buttons and the whole-image run, and the
        category whose settings it changes is the one that should hold it.
        Min area is still Object operations', for the same reason it
        always was. No value here persists between sessions, like every
        other setting on this panel; only which categories are folded does.
        """
        magnifier = self._magnifier
        card = self._settings_category(
            "Live magnifier",
            "Segments the region under the mouse, or the whole image "
            "once, and shows its objects magnified. A click adds "
            "objects to the mask.",
        )
        form = QFormLayout()

        #: Kept empty: the install sentence used to live on the panel, and
        #: now the greyed Mode row offers the install itself.
        self._mag_install_notes = {}

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
        self._mag_size.setRange(*magnifier.size_range())
        self._mag_size.setSingleStep(16)
        self._mag_size.setValue(_MAGNIFIER_SIZE)
        self._mag_size.setToolTip(
            "Side of the square box, in image pixels, up to the open image's "
            "own height or width. Under Region under the mouse it is also the "
            "region the model segments, so make it wider than the largest "
            "object you want to add. Shift + mouse wheel changes it while the "
            "magnifier is on.")
        self._mag_size.valueChanged.connect(magnifier.set_size)
        magnifier.size_changed.connect(self._mag_size.setValue)
        magnifier.size_range_changed.connect(self._mag_size.setRange)
        form.addRow("Size (px)", self._mag_size)

        self._mag_exclude_border = Toggle(
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

        from ..i18n import tr

        self._mag_lock_hint = QLabel(tr(
            "Hold Ctrl+L and right-click to lock the region and zoom. "
            "Repeat to unlock."))
        self._mag_lock_hint.setWordWrap(True)
        form.addRow(self._mag_lock_hint)
        magnifier.locked_changed.connect(
            lambda locked: self._mag_size.setEnabled(not locked))
        magnifier.locked_changed.connect(
            lambda locked: self._mag_zoom.setEnabled(not locked))

        self._mag_sensitivity = QDoubleSpinBox()
        self._mag_sensitivity.setDecimals(2)
        self._mag_sensitivity.setRange(*_MAGNIFIER_SENSITIVITY_RANGE)
        self._mag_sensitivity.setSingleStep(0.25)
        self._mag_sensitivity.setValue(_MAGNIFIER_SENSITIVITY)
        self._mag_sensitivity.setToolTip(
            "How readily the Otsu mode accepts an object. Raise it to "
            "take in dimmer or less certain objects, lower it to keep only "
            "clear ones; 0 is the default cut. The models read their "
            "thresholds from the Object detection settings instead, so this "
            "is "
            "greyed out while another mode is chosen.")
        self._mag_sensitivity.valueChanged.connect(magnifier.set_sensitivity)
        form.addRow("Sensitivity", self._mag_sensitivity)
        self._mag_sensitivity.setEnabled(
            self._mag_mode.currentData() == "otsu")

        self._mag_overlap = QComboBox()
        self._mag_overlap.addItem("Clip", "clip")
        self._mag_overlap.addItem("Skip", "skip")
        self._mag_overlap.addItem("Replace", "replace")
        self._mag_overlap.setToolTip(
            "What a new object does where the mask already has an object. "
            "Clip keeps only its unlabelled pixels, so no existing object "
            "loses a pixel. Skip leaves out any object that touches an "
            "existing one. Replace lets the new object take every pixel it "
            "covers. Under Region under the mouse the box shows what the "
            "rule leaves: what a click would add is drawn solid, and what it "
            "would take away is ghosted.")
        self._mag_overlap.currentIndexChanged.connect(
            lambda _index: magnifier.set_overlap(
                self._mag_overlap.currentData()))
        form.addRow("Overlap", self._mag_overlap)
        card.body_layout.addLayout(form)

        progress = QHBoxLayout()
        from ..widgets.eliding import ProgressLine

        self._mag_progress = ProgressLine(detail=False, count_below=True)
        self._mag_progress.setRange(0, 0)
        self._mag_progress.setTextVisible(False)
        self._mag_progress.hide()
        #: Moves the bar's estimate while a whole-image run is on its way.
        #: Half a second, because the number it writes is whole seconds.
        self._mag_eta_timer = QTimer(self)
        self._mag_eta_timer.setInterval(500)
        self._mag_eta_timer.timeout.connect(self._tick_magnifier_eta)
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
        row = self._tool_row_layout
        row.insertWidget(row.indexOf(self._btn_settings), self._btn_magnifier)
        return card

    def _build_magnifier_save_mode(self, form: QFormLayout) -> None:
        """Which objects a click or a drag adds.

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
        """The settings the magnifier's models read from elsewhere on the panel.

        The Object detection and Otsu categories' own controls, read the
        moment a request is built: the detect buttons read the same boxes, so
        there is one set of settings on the panel and not one per tool.
        """
        selector = getattr(self, '_primary_selector', None)
        source = selector.snapshot if selector is not None else None
        return {
            'primary_source': source,
            'primary_token': source.identity if source is not None else (),
            'primary_selection': selector.path.text() if selector is not None else '',
            "model_name": self._cp_model.currentData() or "cpsam",
            "diameter": int(self._cp_diameter.value()),
            "flow_threshold": float(self._cp_flow.value()),
            "cellprob_threshold": float(self._cp_cellprob.value()),
            "normalize": bool(self._cp_normalize.isChecked()),
            "otsu_correction": float(self._otsu_correction.value()),
            "otsu_smoothing": float(self._otsu_smoothing.value()),
            "otsu_fill_holes": bool(self._secondary_fill_holes.isChecked()
                                     if self._mag_mode.currentData() == cpu_modes.SECONDARY
                                     else self._otsu_fill_holes.isChecked()),
            "otsu_split": bool(self._otsu_split.isChecked()),
            "bright": bool(self._otsu_bright.isChecked()),
            "min_area": self._detect_min_area(),
            "invert": bool(self._cp_invert.isChecked()),
            "chain": self._detect_chain(),
            "method_params": self._method_params(),
            "cpu_params": self._cpu_params(),
            "otsu_window": int(self._otsu_window.value()),
            "otsu_classes": int(self._otsu_classes.value()),
            "otsu_foreground_class": int(self._otsu_foreground.value()),
        }

    def _on_magnifier_mode(self, mode) -> None:
        """Choose the magnifier's model; Sensitivity is the Otsu mode's.

        The Detection methods card follows the mode, so the parameters on
        screen are the ones the mode just chosen reads and no others.
        """
        name = canonical_magnifier_mode(mode)
        self._mag_sensitivity.setEnabled(name == "otsu")
        self._magnifier.set_mode(name)
        self._sync_method_controls()

    def _on_mode_row_changed(self, _index: int) -> None:
        """The Mode box's row changed: hand the mode on, if it can run.

        A row whose package is missing never reaches the magnifier: the box
        is put back on the mode the magnifier is running, and
        :meth:`_on_magnifier_mode_activated`, which fires next for a click,
        offers the install. Handing the mode over for the moment between the
        two would start a model load that can only fail.
        """
        mode = self._mag_mode.currentData()
        if mode in getattr(self, "_mag_uninstalled", ()):
            running = self._mag_mode.findData(
                canonical_magnifier_mode(
                    getattr(self._magnifier, "mode", None)))
            self._mag_mode.setCurrentIndex(
                running if running >= 0
                else self._mag_mode.findData("otsu"))
            return
        self._on_magnifier_mode(mode)

    def _grey_uninstalled_modes(self) -> None:
        """Grey the Mode rows whose package is missing, and only those.

        A model absent from the box teaches nobody it exists; a greyed row
        that offers to install itself does. Greying is a colour and a
        tooltip, not a disabled row, because a disabled row cannot be
        chosen and choosing it is how the install is asked for.
        """
        from ..i18n import tr
        from ..model_install import UNINSTALLED_GREY

        box = self._mag_mode
        for index in range(box.count()):
            mode = box.itemData(index)
            if mode in self._mag_uninstalled:
                box.setItemData(index, QBrush(UNINSTALLED_GREY),
                                Qt.ForegroundRole)
                box.setItemData(index, tr(
                    "{name} is not installed. Choosing it offers to install "
                    "it.", name=box.itemText(index)), Qt.ToolTipRole)
            elif mode in _MAGNIFIER_BACKENDS:
                box.setItemData(index, None, Qt.ForegroundRole)
                box.setItemData(index, None, Qt.ToolTipRole)

    def _on_magnifier_mode_activated(self, index: int) -> None:
        """A person chose this mode: offer the install if it is missing.

        The offer hangs off ``activated``, which fires only when a person
        picks a row -- not when code calls ``setCurrentIndex`` -- so no
        question opens while settings are being restored. The box goes back
        to the mode it was on at once, because a model that is not installed
        cannot segment anything; a finished install selects it.
        """
        mode = self._mag_mode.itemData(index)
        self._resync_magnifier_modes()
        if mode not in getattr(self, "_mag_uninstalled", ()):
            return
        previous = self._mag_mode.findData(
            canonical_magnifier_mode(getattr(self._magnifier, "mode", None)))
        if previous < 0 or previous == index:
            previous = self._mag_mode.findData("otsu")
        self._mag_mode.setCurrentIndex(max(previous, 0))
        self._offer_backend_install(mode)

    def _resync_magnifier_modes(self) -> None:
        """Re-read where each backend stands and redraw the Mode box.

        THE BOX USED TO LEARN THIS ONCE, WHEN THE SCREEN WAS BUILT. Uninstall
        Cellpose 3 from the Model Zoo with Make Masks still open and its four
        modes stayed un-greyed and out of ``_mag_uninstalled``, so choosing
        one offered no install and went straight to a backend that was no
        longer there. Installing one from the Model Zoo screen left the
        reverse: four greyed modes and an install dialog that returned at
        once.

        File checks only (:func:`_state_ready`), so it is cheap enough to run
        on every choice, which is the moment it has to be right. The drawing
        is :meth:`_grey_uninstalled_modes`, so there is one
        description of what a greyed row looks like and it stays translated.
        """
        for mode in _MAGNIFIER_BACKENDS:
            if _backend_ready(mode):
                self._mag_uninstalled.discard(mode)
            else:
                self._mag_uninstalled.add(mode)
        self._grey_uninstalled_modes()

    def _offer_backend_install(self, mode) -> bool:
        """Install the backend ``mode`` needs, into an environment of its own.

        A greyed row installs its backend when it is chosen, and each
        backend gets an isolated environment of its own, so this goes
        through the Model Zoo's own install dialog: off the GUI thread, with
        progress and Cancel, into ~/.spacr/backends/<name>, and spaCR's own
        environment is never touched. It is used instead of
        :class:`spacr.qt.model_install.PackageInstall`, which ran
        `pip install "spacr[<backend>]"` against the environment spaCR is
        running in; that module is unchanged and still serves the Mask
        settings' backend dropdown.

        Every mode of that backend -- all four Cellpose 3 models at once --
        stops being greyed when it lands, and the row that was chosen is
        selected, as the earlier in-process install also did at its end.

        :param mode: a key of :data:`_MAGNIFIER_BACKENDS`.
        :returns: True when the backend can segment afterwards.
        """
        from ..i18n import tr
        from ..widgets import model_zoo_picker

        backend, label = _MAGNIFIER_BACKENDS[mode]
        if not model_zoo_picker.install_backend(self, backend):
            return False
        for other, (needs, _label) in _MAGNIFIER_BACKENDS.items():
            if needs == backend:
                self._mag_uninstalled.discard(other)
        self._grey_uninstalled_modes()
        self._mag_mode.setCurrentIndex(self._mag_mode.findData(mode))
        self._status_label.setText(tr(
            "{name} is installed and selected.", name=label))
        return True

    def _on_magnifier_scope(self, scope) -> None:
        """Segment the region under the mouse or the whole image.

        The border option is greyed out for the whole image, whose objects
        the box never cuts.
        """
        self._mag_exclude_border.setEnabled(scope != "image")
        self._magnifier.set_scope(scope)

    def _on_magnifier_busy(self, busy: bool) -> None:
        """Show the whole-image run's progress and Cancel while it runs."""
        busy = bool(busy)
        self._mag_progress.setVisible(busy)
        self._mag_cancel.setVisible(busy)
        if busy:
            self._mag_eta_timer.start()
            self._tick_magnifier_eta()
        else:
            self._mag_eta_timer.stop()
            self._show_indeterminate_magnifier_bar()

    def _show_indeterminate_magnifier_bar(self) -> None:
        """Say only that something is happening, which is what a bar with no
        measurement behind it can honestly say."""
        self._mag_progress.setTextVisible(False)
        self._mag_progress.setFormat("")
        self._mag_progress.setRange(0, 0)

    def _tick_magnifier_eta(self) -> None:
        """Put the time the whole-image run still has on the busy bar.

        Only when there is a MEASUREMENT behind it -- the last run under this
        mode and model, per megapixel. Before there is one, and once an
        estimate has run out, the bar goes back to indeterminate rather than
        counting down past zero or sitting at 99%.

        A RUN THAT COUNTS ITS OWN TILES SAYS HOW FAR IT HAS GOT instead
        (:meth:`_LiveMagnifier.image_progress`): Cellpose on a whole field,
        which is the run that takes minutes on a CPU. The bar then fills by
        tiles, from the first run of the session on, and the time left is
        this run's own pace; after the last tile, while the network's
        output is turned into objects, it holds its place and says no time,
        because that step is not made of tiles and is not guessed at.
        """
        from ..i18n import tr

        counted = self._magnifier.image_progress()
        if counted is not None:
            done, total, tiles_left = counted
            self._mag_progress.setRange(0, int(total))
            self._mag_progress.setValue(max(0, int(done) - 1))
            if tiles_left is None:
                self._mag_progress.setFormat("%p%")
            else:
                self._mag_progress.setFormat(tr(
                    "about {seconds} s left", seconds=int(tiles_left) + 1))
            self._mag_progress.setTextVisible(True)
            return
        left = self._magnifier.remaining_seconds()
        estimate = self._magnifier.estimated_seconds()
        if left is None or not estimate:
            self._show_indeterminate_magnifier_bar()
            return
        done = max(0.0, min(0.99, 1.0 - left / float(estimate)))
        self._mag_progress.setRange(0, 1000)
        self._mag_progress.setValue(int(done * 1000))
        self._mag_progress.setFormat(
            tr("about {seconds} s left", seconds=int(left) + 1))
        self._mag_progress.setTextVisible(True)

    def _on_invert_toggled(self, on: bool) -> None:
        """Show or hide the Invert warning.

        The magnifier is told separately, by the same signal reaching
        :meth:`_on_magnifier_context_changed`, which throws away objects
        found from the image the other way up.
        """
        self._invert_warning.setVisible(bool(on))

    def _detector_image(self) -> Optional[np.ndarray]:
        """The field as the detectors must read it: inverted when Invert is on.

        The inversion is not a display trick, so the one
        thing the detect buttons segment comes from here rather than from
        the canvas directly, and Otsu detect and Object detection cannot end
        up disagreeing about which way up the image was.

        The canvas's own array is never changed: the hover readout and the
        Filter category read that one and go on reporting the field's real
        values -- a user filtering by
        intensity would otherwise be judging inverted numbers.

        :func:`mask_engine.invert_for_detection`, NOT
        :func:`mask_engine.invert_intensity` -- the dtype complement, which
        would leave the Otsu threshold correction
        pointing at intensities the field does not contain. The measurement
        is in the first function's docstring.

        IT IS ONE CALL because the canvas already assembles exactly this
        array for the Image enhancement card: the inversion is
        :meth:`_MaskCanvas.displayed_source`, the percentile stretch is
        :meth:`_MaskCanvas.detection_base`, and the chain is
        :meth:`_MaskCanvas.detection_source`. This method used to repeat
        the first two, which is how a detect button and the magnifier could
        have ended up reading different arrays the day one of them changed.
        """
        image = self._canvas.image
        if image is None:
            return None
        return self._canvas.detection_source()

    def _on_min_area_changed(self, value) -> None:
        """Hand Min area to the canvas, for Ctrl + left click's seed spacing.

        The same judgement about debris as the detectors read
        (:meth:`_detect_min_area`), so an object the screen would not keep
        is not one the split gesture cuts in two either.
        """
        self._canvas.split_min_area = int(value)

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

        if on and self._canvas.ruler.active:
            self._set_mode(MODE_NONE)
        if on and self._magnifier.scope == "image":
            self._status_label.setText(tr(
                "Magnifier on: a click adds the object under it and a "
                "right-click removes the mask object under it; the mouse "
                "wheel changes its zoom."))
        elif on:
            self._status_label.setText(tr(
                "Magnifier on: a click adds the objects outlined in the box; "
                "the mouse wheel changes its zoom."))
        else:
            self._status_label.setText(tr(
                "Magnifier off. The objects it added stay in the mask."))
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

        Detector settings, CPU/organelle parameters and enhancement come
        from the completed request, even if the panel has since changed.
        ``min_area`` records detection's filter; ``paste_min_area`` records
        the current filter applied while pasting through the overlap rule.

        :returns: the ids added; empty when nothing was.
        """
        from ..i18n import tr

        mask = self._canvas.mask
        request = result.request
        if mask is None or tuple(mask.shape[:2]) != tuple(request.shape):
            return []
        exact_ids = result.mode == cpu_modes.SECONDARY
        if exact_ids:
            try:
                source = self._require_primary_source()
                if source.identity != request.primary_token:
                    raise ValueError(tr('The primary mask changed. Wait for a new preview before accepting objects.'))
                self._require_secondary_merge(source)
            except ValueError as exc:
                self._status_label.setText(str(exc))
                return []
        overlap = self._mag_overlap.currentData() or "clip"
        try:
            out, added = engine._paste_region_objects(
                mask, result.labels, request.box[:2], overlap=overlap,
                min_area=self._detect_min_area(), preserve_ids=exact_ids)
        except ValueError as exc:
            self._status_label.setText(tr(
                "Magnifier could not add objects: {error}", error=exc))
            return []
        if not added and request.scope == "image":
            self._status_label.setText(tr(
                "Magnifier: nothing was added — the Overlap rule or Min area "
                "leaves nothing of the object under the click."))
            return []
        if not added:
            self._status_label.setText(tr(
                "Magnifier: nothing to add — the box outlines no object, or "
                "every object it outlines overlaps one already in the mask."))
            return []
        changed = self._pixels_changed(out)
        self._canvas.mask = out
        self._canvas.refresh()
        if exact_ids:
            self._retain_secondary_ids(request.primary_provenance)
        self._record("magnifier", list(added), changed,
                      overlap=overlap, paste_min_area=self._detect_min_area(),
                      box=[int(v) for v in request.box],
                      n_objects=len(added),
                      **({"source_labels": [int(v) for v in np.unique(result.labels) if v > 0]}
                         if request.scope == "image" else {}),
                      **_magnifier_provenance(request, result.mode, result.note))
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
            Outcome provenance supplies the cursor path, saved selection
            rule and each accepted frame's detector request in delivery order.
            A mixed-method stroke is marked ``mode="mixed"``; its per-frame
            records retain the actual methods, settings and fallback notes.
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
        provenance = found.provenance or {}
        frames = provenance.get("frame_requests", [])
        def common(field, default):
            """Return a shared frame value, or the explicit mixed/unknown value."""
            values = [frame.get(field, default) for frame in frames]
            return values[0] if values and all(value == values[0] for value in values) else default
        self._record("magnifier", list(added), self._pixels_changed(out),
                     mode=common("mode", "mixed" if frames else "unknown"), overlap=overlap,
                     paste_min_area=self._detect_min_area(),
                     box=[x0, y0, x0 + width, y0 + height],
                     sensitivity=common("sensitivity", None),
                     n_objects=len(added), scope=common("scope", "mixed" if frames else "unknown"),
                     drag=True, frames=found.frames, merged=found.merged,
                     **provenance)
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

    def open_paths(self, paths) -> bool:
        """Open dropped image files and folders as one queue, in drop order.

        One folder alone is what it always was, :meth:`_open_folder` on it.
        Anything else -- one file, several, or files and folders together --
        becomes a queue of the fields named, a folder standing for its
        images, each field edited where it lies with its mask in its own
        ``<folder>/masks``. Nothing is copied.

        :param paths: the dropped files and folders, in the order dropped.
        :returns: whether a queue was opened.
        """
        from ..i18n import tr

        paths = [os.path.abspath(str(p)) for p in paths]
        if len(paths) == 1 and os.path.isdir(paths[0]):
            return self._open_folder(paths[0])
        fields: list = []
        for path in paths:
            if os.path.isdir(path):
                found = [(path, name) for name in engine.list_images(path)]
            elif (os.path.isfile(path)
                  and path.lower().endswith(engine.IMAGE_EXTS)):
                found = [(os.path.dirname(path), os.path.basename(path))]
            else:
                found = []
            for field in found:
                if field not in fields:
                    fields.append(field)
        if not fields:
            self._warn(tr("No images"),
                       tr("None of the dropped items is an image Make Masks "
                          "can open."))
            return False
        folders = [folder for folder, _name in fields]
        spans = list(dict.fromkeys(folders))
        if not self._open_folder(
                folders[0], files=[name for _folder, name in fields],
                field_folders=folders if len(spans) > 1 else None):
            return False
        if len(spans) > 1:
            self._src_label.setText(tr(
                "{n} images from {k} folders, in the order dropped",
                n=len(fields), k=len(spans)))
        return True

    def _open_folder(self, folder: str,
                     files: Optional[List[str]] = None,
                     masks_dir: Optional[str] = None,
                     field_folders: Optional[List[str]] = None) -> bool:
        """List the folder's images and load the first.

        :param folder: the folder to open.
        :param files: the file names to offer, in the order to offer them.
            ``None`` -- every caller but :meth:`open_queue` -- lists the
            folder itself, which is what a file dialog or a dropped folder
            means. A session built by ``spacr-make-masks`` passes its own
            list, because the queue has already dropped what is reviewed and
            sorted what is left.
        :param masks_dir: the masks folder when it is not ``<folder>/masks``
            -- a sibling session's. Set BEFORE the first field loads, so the
            first draft shown is the set's own.
        :param field_folders: the folder of each of ``files``, when a drop
            queued fields from more than one folder; see :meth:`open_paths`.
        :returns: whether a folder was opened. ``False`` means there was
            nothing in it to edit, which the user has been told about.
        """
        files = list(files) if files is not None else engine.list_images(folder)
        if not files:
            self._warn("No images", f"Found no image files in: {folder}")
            return False
        self._queue = None
        self._session_notice = ""
        self._masks_dir = masks_dir
        self._folder = folder
        self._image_files = files
        self._field_folders = (list(field_folders)
                               if field_folders is not None else None)
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
        if self._field_folders:
            self._folder = self._field_folders[self._current_index]
        self._primary_selector.clear_field()
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
        worker = _MaskLoadWorker(folder, filename, token, self,
                                 layout=self._layout_kwargs())
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
        be told to stop rather than be collected out from under Qt. A model
        download is cancelled; a backend install is left to finish, because
        ``pip`` stopped half way can leave the environment broken.

        :param event: the close event; it is passed on to the base class once
            the workers and folded modules have been stopped.
        """
        from ..bridge import drain_thread

        download, self._cp_download = self._cp_download, None
        if download is not None:
            download.cancel()
        self._magnifier.close()
        self._primary_selector.shutdown()
        self._psf_controls._shutdown()
        self._restoration_controls._shutdown()
        self._canvas.close_enhancer()
        self._cancel_comparison()
        if self._comparison_worker is not None:
            self._comparison_worker.close(timeout=0)
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
            image, mask = engine.load_image_and_mask(
                folder, filename, **self._layout_kwargs())
        except Exception as exc:
            self._handle_load_failure(exc)
            return
        self._apply_loaded_pair(filename, token, image, mask)

    def _handle_load_failure(self, error: Exception) -> None:
        """Clear stale canvas state and visibly report an image-load error."""
        self._primary_selector.clear_field()
        self._canvas.preserve_ids = False
        self._paired_source = None
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
        """Install a decoded pair if it still represents the selected field.

        The magnifier is told which field this is BEFORE the pair reaches the
        canvas: the canvas makes it forget the last one on the way in, and
        the name is what its whole-image objects are kept under, so coming
        back to this field offers them rather than segmenting it again.
        """
        if token != self._load_token:
            return
        self._canvas.preserve_ids = False
        self._paired_source = None
        self._magnifier.set_field(os.path.join(self._folder or "", filename))
        self._close_levels()
        self._canvas.set_image_and_mask(image, mask)
        self._recrop_children = []
        self._reset_flow_panes()
        self._history.clear()
        self._history.push(mask)
        self._refresh_history_buttons()
        self._btn_reset_zoom.setEnabled(False)
        self._log = self._open_ledger(filename)
        record = next((edit.detail.get('primary_source') for edit in reversed(self._log.edits)
                       if edit.detail.get('preserve_ids')), None)
        if record:
            self._canvas.preserve_ids = True
            self._canvas._lookup = None
            self._canvas._lookup_dirty = True
            self._paired_source = record
        self._primary_selector.bind_field(os.path.join(self._folder, filename), mask.shape,
                                          engine.mask_save_path(self._folder, filename, **self._layout_kwargs()))
        if record:
            self._primary_selector.restore_source(record)
        self._status_label.setText(
            f"{filename}  "
            f"({self._current_index + 1}/{len(self._image_files)})"
        )
        self.apply_object_filter(on_load=True)
        self._refresh_curation_buttons()
        self._show_session_notice()
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
        artifact = engine.mask_save_path(self._folder, filename,
                                         **self._layout_kwargs())
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

        :param x0: x of the first selection corner, in image pixels.
        :param y0: y of the first selection corner, in image pixels.
        :param x1: x of the opposite corner, in image pixels. The corners may
            come in either order and are clipped to the image.
        :param y1: y of the opposite corner, in image pixels.
        :returns: Filename of the recropped field, or ``None`` if the
            selection was rejected or could not be written.
        """
        if getattr(self._canvas, 'preserve_ids', False):
            from ..i18n import tr

            self._status_label.setText(tr('Recrop requires a matching crop of both primary and secondary masks. Save this paired field before creating a separate crop.'))
            return None
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
                self._canvas.mask, box, **self._layout_kwargs())
        except Exception as exc:
            self._warn("Recrop failed", str(exc))
            return None

        name = engine.field_stem(written.name)
        self._canvas.recrop_boxes.append((*box, name))
        self._canvas.update()
        self._image_files.insert(
            self._current_index + len(self._recrop_children) + 1, written.name)
        if self._field_folders:
            self._field_folders.insert(
                self._current_index + len(self._recrop_children) + 1,
                self._folder)
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
                                  log=self._log, preserve_ids=getattr(self._canvas, 'preserve_ids', False),
                                  **self._layout_kwargs())
            except Exception as exc:
                LOG.warning("Could not save %s before retiring it: %s",
                            filename, exc)
        try:
            engine.retire_recropped_original(
                self._folder, filename, children=children, boxes=boxes,
                **self._layout_kwargs())
        except Exception as exc:
            self._warn("Recrop failed", str(exc))
            return False
        self._image_files.pop(self._current_index)
        if self._field_folders:
            self._field_folders.pop(self._current_index)
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
            self._validate_secondary_save()
            path = engine.save_mask(
                self._folder,
                self._image_files[self._current_index],
                self._canvas.mask,
                log=self._log,
                preserve_ids=getattr(self._canvas, 'preserve_ids', False),
                **self._layout_kwargs(),
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
        self._apply_op(lambda mask: engine.fill_holes(
            mask, preserve_ids=getattr(self._canvas, 'preserve_ids', False)), 'fill_holes')

    def _on_relabel(self):
        """Renumber the objects so the labels are consecutive."""
        if getattr(self._canvas, 'preserve_ids', False):
            from ..i18n import tr

            self._status_label.setText(tr('Paired secondary objects retain primary IDs; consecutive relabeling would break that association.'))
            return
        self._apply_op(engine.relabel_objects, "relabel")

    def _on_invert(self):
        """Swap object and background in the MASK, and say what that gave.

        Not the picture invert -- that is :meth:`_on_invert_display`, which
        is what a curator expecting Invert to change the image wants.
        This one is kept under the name that describes it, and it now
        reports its own result, because the thing that made it look broken
        is that its result is invisible: a field whose background is one
        connected region comes back as a single object covering the frame,
        which the overlay draws as one flat wash.
        """
        if self._canvas.mask is None:
            return
        self._apply_op(engine.invert_mask, "invert")
        found = self._objects_now()
        self._status_label.setText(
            f"Swapped object and background — {found} object(s) now, and "
            f"what was an object is background. Ctrl+Z to undo."
        )

    def _on_remove_small(self):
        """Delete objects below the minimum area."""
        area = int(self._min_area.value())
        if getattr(self._canvas, 'preserve_ids', False):
            self._apply_op(lambda mask: engine.filter_report(
                mask, self._canvas.image, min_area=area, preserve_ids=True)[0],
                'remove_small', min_area=area)
            return
        self._apply_op(lambda m: engine.remove_small_objects(m, area),
                        "remove_small", min_area=area)

    def _objects_now(self) -> int:
        """How many objects the open mask holds, by distinct id.

        Not ``mask.max()``: a mask that has had objects deleted out of the
        middle of it has ids with gaps, and the largest id is then a count
        of what has ever been there rather than of what is there.
        :func:`_object_count` is the same count, and is what the magnifier
        reports its own results with.
        """
        field = self._canvas.mask
        if field is None or not np.asarray(field).size:
            return 0
        return _object_count(field)

    def _on_dilate(self):
        """Grow every object by the step, into background only."""
        if self._canvas.mask is None:
            return
        step = int(self._grow_step.value())
        before = self._objects_now()
        self._apply_op(lambda m: engine.dilate_objects(m, step),
                       "dilate", step=step)
        self._status_label.setText(
            f"Dilated {before} object(s) by {step} px — no object took a "
            f"pixel from another.")

    def _on_shrink(self):
        """Pull every object in by the step, and say what that cost.

        THE COUNT IS THE POINT OF THE MESSAGE. Erosion deletes anything
        thinner than twice the step, and a curator who has just lost eleven
        objects to a step of 3 needs to be told so while the undo is still
        the obvious thing to do.

        BOTH NUMBERS ARE COUNTED BEFORE THE EDIT, and the first of them is
        deliberately the count the button was PRESSED on rather than the
        count that survived. Reporting the survivors read as an arithmetic
        puzzle -- ten objects, seven erased, "Shrank 3 object(s) ... 7
        object(s) are gone" -- and erasing every object reported "Shrank 0
        object(s)" over a field that had just been emptied. Every object
        was eroded; some of them did not survive it, which is the second
        clause.
        """
        if self._canvas.mask is None:
            return
        step = int(self._grow_step.value())
        before = self._objects_now()
        self._apply_op(lambda m: engine.shrink_objects(m, step),
                       "shrink", step=step)
        gone = max(0, before - self._objects_now())
        lost = (f" — {gone} of them were thinner than {2 * step} px and "
                f"are gone; Undo brings them back" if gone else "")
        self._status_label.setText(
            f"Shrank {before} object(s) by {step} px{lost}.")

    def _on_clear_mask(self):
        """Throw the whole mask away, after confirming.

        The confirmation says how many objects are about to go: "Zero out
        the current mask?" alone did not, and the count is the one fact that
        decides the question.
        """
        if self._canvas.mask is None:
            return
        count = self._objects_now()
        if not self._confirm(
                "Clear mask",
                f"Remove all {count} object(s) from this field? The field "
                f"itself is untouched, and Undo brings the objects back."):
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
                   self._btn_discard, self._btn_keep,
                   self._btn_filter, self._btn_otsu, self._btn_magnifier,
                   self._btn_dilate, self._btn_shrink, self._btn_clear,
                   self._btn_levels,
                   *self._mode_buttons.values()):
            b.setEnabled(editable)
