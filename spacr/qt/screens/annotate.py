"""
AnnotateScreen — Qt widget replacing the Tk AnnotateApp.

Displays a paginated grid of clickable image thumbnails backed by
`png_list` in a `measurements/measurements.db`. Left-click = value 1,
right-click = value 2, re-click the same value = clear. Annotations
are persisted through a background SaveWorker (see
`spacr.qt.annotate_engine.SaveWorker`).

A keyboard-only rapid-annotation layer sits on top of the same write
path (see :meth:`AnnotateScreen.handle_key`): ``1``–``9`` assign a class
and auto-advance to the next unlabelled crop, ``0`` clears, arrows /
``hjkl`` move focus, ``Space``/``Backspace`` step without labelling,
``u`` undoes and ``Enter`` commits the page.

Every crop is drawn as a rounded square (see :class:`_Thumbnail`) with
two independent bands of colour, so its three states stay readable
together rather than overwriting each other:

* **resting** — a thin gray ring hugging the image
* **classified** — that same ring in the class colour
  (:func:`~spacr.qt.annotate_engine.label_to_hex`, the app's one
  class→colour map)
* **current** — an *extra* white ring outside it on the single tile the
  next click or keystroke will hit

The cursor and the keyboard move the same current tile: entering a tile
makes it current, and an arrow key moves it away. There is no second
"hovered" highlight that could point somewhere else.

Advanced features that are *not* yet ported (marked as TODOs in the UI):
UMAP window, Deep Spacr training launcher, measurement-threshold
filtering (the threshold filter can be entered in settings but only
plain per-page fetch is used at query time in this MVP).
"""
from __future__ import annotations

import os
from copy import deepcopy
from collections import deque
from functools import partial
from typing import Deque, Dict, List, Optional, Tuple

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    Qt,
    QEvent,
    QRectF,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (QColor, QImage, QKeySequence, QPainter,
                           QPainterPath, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from ..widgets.toggle import Toggle

from ..annotate_engine import (
    AnnotateSettings,
    SaveWorker,
    class_counts,
    clear_column,
    count_rows,
    ensure_annotation_column,
    fetch_filtered_paths,
    fetch_page,
    filter_channels_pil,
    find_last_annotated_offset,
    label_to_hex,
    load_crop_image,
    normalize_pil,
    outline_image,
)
from .. import iconset, prefs
from ..theme import SPACING, palette_for
from ..widgets.column_picker import attach_column_picker
from ..widgets import Divider, EmptyState


# ---------------------------------------------------------------------------
# Tile chrome
#
# Every crop is a rounded square drawn by `_Thumbnail.paintEvent` as three
# concentric pieces:
#
#     ┌── current ring  (white) — ONLY on the tile the next action hits
#     │ ┌── state ring          — resting gray, or the crop's class colour
#     │ │ ┌── the crop itself, CLIPPED to a rounded rect (a real round
#     │ │ │   corner, not a rounded frame laid over a square image)
#
# The two rings sit at fixed insets, so nothing moves or resizes when the
# cursor arrives: hover ADDS the outer ring, it never recolours the inner
# one, and a class colour never hides the fact that a tile is the current
# one. That is the whole composition rule — the two states are drawn in
# two different bands and cannot overwrite each other.
# ---------------------------------------------------------------------------

BORDER_WIDTH = 2          # state ring — the thin line around every crop
HOVER_RING_WIDTH = 3      # current-tile ring, drawn outside the state ring
TILE_INSET = HOVER_RING_WIDTH + BORDER_WIDTH   # chrome per side, in px
TILE_RADIUS = 10          # outer corner radius of the rounded square
IMAGE_RADIUS = max(1, TILE_RADIUS - TILE_INSET)

# How many keyboard assignments can be walked back with `u`. Bounded so a
# long session can't grow the stack without limit.
UNDO_LIMIT = 128


def tile_palette() -> Dict[str, str]:
    """Palette for the theme the app is actually showing right now.

    The Annotate grid paints raw colours (it is not QSS-styled), so it has
    to resolve dark/light itself instead of importing the dark ``PALETTE``
    at module scope — a hard-coded gray is invisible on one of the two
    themes.
    """
    try:
        from ..preferences import resolve_effective_theme
        return palette_for(resolve_effective_theme())
    except Exception:
        return palette_for("dark")


def resting_border_color() -> str:
    """The thin gray line every unlabelled crop carries."""
    return tile_palette()["border"]


def current_ring_color() -> str:
    """Colour of the "this is the tile you are on" ring.

    ``fg`` is pure white on the (default) dark theme — exactly the white
    border the feature asks for — and flips to the near-black foreground
    on the light theme, where white would vanish into the background.
    Either way it is a colour :func:`label_to_hex` can never produce
    (class 1 is blue, 2 red, 3+ are HSV rotations at saturation 0.65), so
    the current ring is never mistaken for a class.
    """
    return tile_palette()["fg"]


# ---------------------------------------------------------------------------
# Keyboard tokens
#
# `handle_key` is the single entry point for every keystroke so tests can
# drive the whole feature without synthesising Qt key events. It accepts a
# Qt key code, a Qt key *name* ("Left"), or a literal character ("1", "h"),
# and normalises all of them onto the small token vocabulary below.
# ---------------------------------------------------------------------------

# canonical tokens: "0".."9", "left", "right", "up", "down", "space",
#                   "backspace", "undo", "enter", "help", "escape"

_TEXT_TOKENS = {
    "left": "left", "right": "right", "up": "up", "down": "down",
    # vi-style motion
    "h": "left", "j": "down", "k": "up", "l": "right",
    "space": "space",
    "backspace": "backspace", "back": "backspace",
    "u": "undo", "undo": "undo",
    "enter": "enter", "return": "enter",
    "?": "help", "help": "help",
    "escape": "escape", "esc": "escape",
}

_QT_NAME_TOKENS = (
    ("Key_Left", "left"), ("Key_Right", "right"), ("Key_Up", "up"),
    ("Key_Down", "down"), ("Key_Space", "space"),
    ("Key_Backspace", "backspace"), ("Key_Return", "enter"),
    ("Key_Enter", "enter"), ("Key_Question", "help"),
    ("Key_Escape", "escape"),
)


def _qt_code_tokens() -> Dict[int, str]:
    """Build {Qt key code -> token} once, tolerating enum-shape differences."""
    out: Dict[int, str] = {}
    for name, token in _QT_NAME_TOKENS:
        code = getattr(Qt, name, None)
        if code is None:
            code = getattr(getattr(Qt, "Key", None), name, None)
        if code is None:
            continue
        try:
            out[int(code)] = token
        except (TypeError, ValueError):   # pragma: no cover - defensive
            continue
    return out


_QT_CODE_TOKENS: Dict[int, str] = _qt_code_tokens()


def _token_from_text(text: str) -> Optional[str]:
    """Map a literal character or key name onto a canonical token."""
    if text == " ":
        return "space"
    low = text.strip().lower()
    if not low:
        return None
    if low in _TEXT_TOKENS:
        return _TEXT_TOKENS[low]
    if len(low) == 1 and low.isdigit():
        return low
    return None


def key_token(key, text: str = "") -> Optional[str]:
    """Normalise ``key`` (Qt code, key name or character) to an action token.

    Returns ``None`` for anything the annotate screen does not bind, so
    callers can fall through to the default Qt handling.
    """
    if isinstance(key, str):
        token = _token_from_text(key)
        return token if token is not None else (_token_from_text(text)
                                                 if text else None)
    code: Optional[int]
    try:
        code = int(key)
    except (TypeError, ValueError):
        code = None
    if code is not None:
        token = _QT_CODE_TOKENS.get(code)
        if token:
            return token
        if 0x30 <= code <= 0x39:          # Qt.Key_0 .. Qt.Key_9
            return chr(code)
        if 0x41 <= code <= 0x5A:          # Qt.Key_A .. Qt.Key_Z
            token = _token_from_text(chr(code))
            if token:
                return token
    return _token_from_text(text) if text else None


# ---------------------------------------------------------------------------
# Click-aware thumbnail label
# ---------------------------------------------------------------------------

class _PageLoadWorker(QThread):
    """Loads + processes a page of thumbnail images OFF the GUI thread.

    ``_load_thumb_image`` (normalise + optional Otsu/Cellpose outline) is
    expensive; running it inline froze the UI when settings changed. This runs
    the whole page in a worker and emits the finished (PIL image, annotation)
    list back to the main thread, which does only the cheap pixmap conversion.
    ``gen`` lets the screen ignore results from a superseded load.
    """

    done = Signal(int, object)   # (gen, list[(PIL.Image, annotation)])

    def __init__(self, gen: int, paths: list, load_fn, parent=None):
        super().__init__(parent)
        self._gen = gen
        self._paths = paths
        self._load_fn = load_fn

    def run(self):
        try:
            loaded = []
            for row in self._paths:
                if self.isInterruptionRequested():
                    return
                loaded.append(self._load_fn(row))
        except Exception:
            loaded = []
        if not self.isInterruptionRequested():
            self.done.emit(self._gen, loaded)


class _Thumbnail(QLabel):
    """One crop in the grid: a rounded square wearing up to two rings.

    Everything is drawn in :meth:`paintEvent`, which is what makes the
    borders cheap: changing a border is one ``update()`` on ONE widget, not
    a rebuilt pixmap. The pixmap handed to ``setPixmap`` is the bare crop —
    the rounded corners come from clipping it here, so the corner is
    actually round instead of a rounded frame sitting on a square image.

    Three visual states, drawn in two separate bands so they compose
    instead of overwriting each other:

    * resting — thin ``resting_border_color()`` gray ring
    * classified — the same ring, recoloured to the class colour
    * current (cursor is on it, or the keyboard is) — an ADDITIONAL white
      ring outside the first one, leaving the class colour untouched
    """

    left_clicked = Signal(int)
    right_clicked = Signal(int)
    # (slot, entered). Emitted on Enter/Leave only — never per mouse-move —
    # so tracking the cursor across the grid costs two repaints per tile
    # boundary crossed and nothing at all in between.
    hover_changed = Signal(int, bool)

    def __init__(self, slot: int, parent: Optional[QWidget] = None,
                 border_color: Optional[str] = None,
                 ring_color: Optional[str] = None):
        super().__init__(parent)
        self.slot = slot
        # Colours are resolved by the screen once per grid rebuild and
        # handed down, so the hover path never has to look up a palette.
        self._border_color = border_color or resting_border_color()
        self._ring_color = ring_color or current_ring_color()
        self._current = False
        self._occupied = False
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # Transparent so the rounded tile sits cleanly on the grid canvas
        # (no grey square peeking out at the corners).
        self.setStyleSheet("background: transparent;")
        self.setProperty("kbdFocused", False)

    # -- state ---------------------------------------------------------
    def border_color(self) -> str:
        """Colour of the ring hugging the image: resting gray or class colour."""
        return self._border_color

    def ring_color(self) -> Optional[str]:
        """The current-tile ring colour, or ``None`` when this isn't it."""
        return self._ring_color if self._current else None

    def outline_color(self) -> str:
        """Outermost colour drawn — what a user would call "the border"."""
        return self._ring_color if self._current else self._border_color

    def is_current(self) -> bool:
        """True when this is the one tile the next action applies to."""
        return self._current

    def is_occupied(self) -> bool:
        """True when this cell holds a crop (empty cells draw nothing)."""
        return self._occupied

    def set_border_color(self, color: Optional[str]) -> bool:
        """Recolour the state ring; returns True when a repaint was needed."""
        color = str(color or resting_border_color())
        if color.lower() == self._border_color.lower():
            return False
        self._border_color = color
        self.update()
        return True

    def set_current(self, on: bool) -> bool:
        """Add/remove the current-tile ring; returns True when it changed."""
        on = bool(on)
        # Mirrored onto a Qt property so QSS and tests can both see it, and
        # so there is exactly one notion of "the current tile".
        self.setProperty("kbdFocused", on)
        if on == self._current:
            return False
        self._current = on
        self.update()
        return True

    def set_occupied(self, on: bool) -> bool:
        """Mark whether this cell holds a crop; empty cells paint nothing."""
        on = bool(on)
        if on == self._occupied:
            return False
        self._occupied = on
        self.update()
        return True

    # -- painting ------------------------------------------------------
    def paintEvent(self, event):        # noqa: N802  (Qt naming)
        """Draw the clipped crop, the state ring and (if current) the ring."""
        if not self._occupied:
            return
        w = float(self.width())
        h = float(self.height())
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(Qt.NoBrush)
            inner = QRectF(TILE_INSET, TILE_INSET,
                           max(0.0, w - 2 * TILE_INSET),
                           max(0.0, h - 2 * TILE_INSET))
            pm = self.pixmap()
            if pm is not None and not pm.isNull() \
                    and inner.width() > 0 and inner.height() > 0:
                clip = QPainterPath()
                clip.addRoundedRect(inner, IMAGE_RADIUS, IMAGE_RADIUS)
                painter.save()
                painter.setClipPath(clip)
                painter.drawPixmap(_cover_rect(pm, inner), pm,
                                   QRectF(pm.rect()))
                painter.restore()
            self._stroke(painter, self._border_color, BORDER_WIDTH,
                         HOVER_RING_WIDTH + BORDER_WIDTH / 2.0, w, h)
            if self._current:
                self._stroke(painter, self._ring_color, HOVER_RING_WIDTH,
                             HOVER_RING_WIDTH / 2.0, w, h)
        finally:
            painter.end()

    @staticmethod
    def _stroke(painter: QPainter, color: str, width: int, inset: float,
                w: float, h: float) -> None:
        """Stroke one rounded rect inset by ``inset`` from the widget edge."""
        if w - 2 * inset <= 0 or h - 2 * inset <= 0:
            return
        pen = QPen(QColor(color))
        pen.setWidth(width)
        painter.setPen(pen)
        radius = max(1.0, TILE_RADIUS - inset)
        painter.drawRoundedRect(
            QRectF(inset, inset, w - 2 * inset, h - 2 * inset),
            radius, radius)

    # -- mouse ---------------------------------------------------------
    def mousePressEvent(self, event):
        """Route left/right mouse buttons to typed signals; ignore others."""
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(self.slot)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(self.slot)
        else:
            super().mousePressEvent(event)

    def enterEvent(self, event):        # noqa: N802  (Qt naming)
        """Cursor arrived — tell the screen this tile is now the current one."""
        self.hover_changed.emit(self.slot, True)
        super().enterEvent(event)

    def leaveEvent(self, event):        # noqa: N802  (Qt naming)
        """Cursor left — the screen drops the hover if it still points here."""
        self.hover_changed.emit(self.slot, False)
        super().leaveEvent(event)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

def _csv_to_list(text: str) -> Optional[List[str]]:
    """Parse a comma-separated string into a stripped list, or ``None`` when empty."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts or None


def _list_to_csv(vals: Optional[List[str]]) -> str:
    """Format a list as a comma-separated string; empty/None becomes ``""``."""
    return ", ".join(str(v) for v in vals) if vals else ""


def _cover_rect(pm: QPixmap, box: QRectF) -> QRectF:
    """Rect to draw ``pm`` into so it fills ``box`` at its own aspect ratio.

    Crops rather than letterboxes (the clip path trims the overflow), so a
    tile is always a complete rounded square with no canvas showing through
    at the edges. Crops normally arrive already resized to the box, in
    which case this is the identity.
    """
    pw = float(pm.width())
    ph = float(pm.height())
    if pw <= 0 or ph <= 0:
        return box
    scale = max(box.width() / pw, box.height() / ph)
    w = pw * scale
    h = ph * scale
    return QRectF(box.x() + (box.width() - w) / 2.0,
                  box.y() + (box.height() - h) / 2.0, w, h)


def _reanchor_png_path(path: str, db_path: str) -> str:
    """Re-anchor a stored ``png_path`` against the opened database's location.

    The measurements DB records absolute png paths built at measure time. If the
    dataset was moved (or measure ran with a relative ``src``), those paths no
    longer resolve and the Annotate grid shows grey placeholders instead of
    images. The DB always sits at ``<root>/measurements/measurements.db`` beside
    the ``<root>/data/...`` crops, so when the stored path fails we rebuild it
    from the ``/data/`` segment onward under this DB's own root.
    """
    if not path or os.path.isfile(path):
        return path
    if not db_path:
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(db_path)))
    norm = str(path).replace("\\", "/")
    i = norm.rfind("/data/")
    if i != -1:
        cand = os.path.join(root, norm[i + 1:])   # data/.../x.png
        if os.path.isfile(cand):
            return cand
    if norm.startswith("data/"):                   # relative-path case
        cand = os.path.join(root, norm)
        if os.path.isfile(cand):
            return cand
    return path


def _load_thumb_image_worker(row, src, settings):
    """Load one thumbnail from an immutable page-request snapshot.

    This function deliberately receives no :class:`AnnotateScreen`.  Calling a
    bound QWidget method from a worker kept the screen wrapper alive after its
    C++ object had been destroyed and let background threads read settings
    while the GUI thread replaced them.
    """
    if isinstance(row, dict):
        annotation = row.get("annotation")
    else:
        path, annotation = row
        row = {"png_path": path}

    s = settings
    if src is not None and getattr(src, "kind", "png") == "merged":
        try:
            img = Image.fromarray(src.get(row)).convert("RGB")
        except Exception:
            return Image.new("RGB", s.image_size, (30, 30, 30)), annotation
    else:
        path = _reanchor_png_path(row.get("png_path"), s.db_path)
        if not path or not os.path.isfile(path):
            return Image.new("RGB", s.image_size, color=(20, 20, 20)), annotation
        try:
            img = load_crop_image(
                path, db_path=s.db_path,
                stored_channel_order=getattr(
                    s, "stored_channel_order", "rgb"))
        except Exception:
            return Image.new("RGB", s.image_size, (30, 30, 30)), annotation

    img = normalize_pil(img, s.percentiles, s.normalize_channels)
    # Keep the full image for outline detection even when the display filter
    # hides one of its channels.
    full_img = img
    img = filter_channels_pil(img, s.channels)
    if s.outline:
        try:
            img = outline_image(
                base_img=img,
                full_img=full_img,
                outline_channels=s.outline,
                edge_sigma=s.outline_sigma,
                edge_thickness=s.edge_thickness,
                edge_transparency=s.edge_transparency,
                edge_image=s.edge_image,
                outline_threshold_factor=s.outline_threshold_factor,
                object_size=s.object_size,
                outline_method=getattr(s, "outline_method", "otsu"),
            )
        except Exception:
            pass
    return img.resize(s.image_size), annotation


class _SettingsDialog(QDialog):
    """Modal dialog that edits an :class:`AnnotateSettings` in place."""

    def __init__(self, settings: AnnotateSettings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Annotate — Settings")
        self.setMinimumWidth(480)
        self._settings = settings

        form = QFormLayout()

        self._src_edit = QLineEdit(settings.src)
        src_row = QHBoxLayout()
        src_row.setContentsMargins(0, 0, 0, 0)
        src_row.addWidget(self._src_edit, 1)
        src_btn = QPushButton("Browse…")
        src_btn.clicked.connect(self._pick_src)
        src_row.addWidget(src_btn)
        src_wrap = QWidget(); src_wrap.setLayout(src_row)
        form.addRow("Source folder", src_wrap)

        self._ann_col = QLineEdit(settings.annotation_column)
        form.addRow("Annotation column", self._ann_col)
        # "SQL" — show what png_list already holds, so a mistyped name cannot
        # quietly start a second annotation pass that then looks like a second
        # annotator who agrees with nobody. Opens read-only.
        attach_column_picker(self._ann_col, self._picker_db_path, "png_list",
                             layout=form)

        self._img_size = QSpinBox()
        self._img_size.setRange(48, 800)
        self._img_size.setValue(settings.image_size[0])
        form.addRow("Image size (px)", self._img_size)

        self._image_type = QLineEdit(settings.image_type or "")
        self._image_type.setPlaceholderText("e.g. cell (blank = all types)")
        form.addRow("Image type filter", self._image_type)

        self._channels = QLineEdit(_list_to_csv(settings.channels))
        self._channels.setPlaceholderText("r, g, b (blank = all)")
        form.addRow("Show channels", self._channels)

        self._stored_channel_order = QComboBox()
        self._stored_channel_order.addItem("RGB (standard)", "rgb")
        self._stored_channel_order.addItem(
            "Auto (use spaCR format marker)", "auto")
        self._stored_channel_order.addItem(
            "Legacy BGR (old unmarked crops)", "legacy_bgr")
        current_order = str(
            getattr(settings, "stored_channel_order", "rgb")).lower()
        order_index = self._stored_channel_order.findData(current_order)
        self._stored_channel_order.setCurrentIndex(max(0, order_index))
        self._stored_channel_order.setToolTip(
            "Order stored in the PNG file. RGB keeps standard PNG channels "
            "unchanged. Auto uses spaCR's sidecar/database format marker. "
            "Legacy BGR repairs crops written by older cv2-based releases. "
            "After decoding, Annotate always uses RGB arrays.")
        form.addRow("Stored PNG order", self._stored_channel_order)

        self._norm_channels = QLineEdit(_list_to_csv(settings.normalize_channels))
        self._norm_channels.setPlaceholderText("r, g, b (blank = off)")
        form.addRow("Normalize channels", self._norm_channels)

        self._pct_lo = QDoubleSpinBox()
        self._pct_lo.setRange(0.0, 100.0)
        self._pct_lo.setValue(float(settings.percentiles[0]))
        self._pct_hi = QDoubleSpinBox()
        self._pct_hi.setRange(0.0, 100.0)
        self._pct_hi.setValue(float(settings.percentiles[1]))
        pct_row = QHBoxLayout(); pct_row.setContentsMargins(0, 0, 0, 0)
        pct_row.addWidget(self._pct_lo); pct_row.addWidget(QLabel("–"))
        pct_row.addWidget(self._pct_hi)
        pct_wrap = QWidget(); pct_wrap.setLayout(pct_row)
        form.addRow("Percentiles", pct_wrap)

        self._outline = QLineEdit(_list_to_csv(settings.outline))
        self._outline.setPlaceholderText("channels to outline, e.g. g")
        form.addRow("Outline channels", self._outline)

        self._outline_method = QComboBox()
        self._outline_method.addItems(["otsu", "cellpose"])
        self._outline_method.setCurrentText(
            getattr(settings, "outline_method", "otsu"))
        self._outline_method.setToolTip(
            "How object outlines are found: 'otsu' (fast threshold) or "
            "'cellpose' (a small Cellpose model — cleaner, slower).")
        form.addRow("Outline method", self._outline_method)

        self._out_factor = QDoubleSpinBox()
        self._out_factor.setRange(0.0, 100.0)
        self._out_factor.setValue(float(settings.outline_threshold_factor))
        form.addRow("Outline threshold factor", self._out_factor)

        self._out_sigma = QDoubleSpinBox()
        self._out_sigma.setRange(0.0, 100.0)
        self._out_sigma.setValue(float(settings.outline_sigma))
        form.addRow("Outline sigma", self._out_sigma)

        self._edge_thick = QDoubleSpinBox()
        self._edge_thick.setRange(0.0, 20.0)
        self._edge_thick.setDecimals(2)
        self._edge_thick.setValue(float(settings.edge_thickness))
        form.addRow("Edge thickness", self._edge_thick)

        self._edge_transp = QDoubleSpinBox()
        self._edge_transp.setRange(0.0, 100.0)
        self._edge_transp.setValue(float(settings.edge_transparency))
        form.addRow("Edge transparency", self._edge_transp)

        self._edge_image = Toggle("Show original image under outline")
        self._edge_image.setChecked(bool(settings.edge_image))
        form.addRow("", self._edge_image)

        self._obj_min = QSpinBox(); self._obj_min.setRange(0, 10_000_000)
        self._obj_min.setValue(int(settings.object_size[0]))
        self._obj_max = QSpinBox(); self._obj_max.setRange(0, 10_000_000)
        self._obj_max.setValue(int(settings.object_size[1]))
        obj_row = QHBoxLayout(); obj_row.setContentsMargins(0, 0, 0, 0)
        obj_row.addWidget(self._obj_min); obj_row.addWidget(QLabel("–"))
        obj_row.addWidget(self._obj_max)
        obj_wrap = QWidget(); obj_wrap.setLayout(obj_row)
        form.addRow("Object size (px area)", obj_wrap)

        # ── Threshold filter (measurement > / < threshold on merged tables)
        self._measurement = QLineEdit(
            ", ".join(settings.measurement) if isinstance(settings.measurement, (list, tuple))
            else (str(settings.measurement) if settings.measurement else "")
        )
        self._measurement.setPlaceholderText("e.g. cell_area (blank = off)")
        form.addRow("Measurement column(s)", self._measurement)
        attach_column_picker(self._measurement, self._picker_db_path,
                             layout=form, multi=True)

        self._threshold = QLineEdit(
            ", ".join(str(x) for x in settings.threshold) if isinstance(settings.threshold, (list, tuple))
            else (str(settings.threshold) if settings.threshold is not None else "")
        )
        self._threshold.setPlaceholderText("e.g. 500 (comma-separated to match)")
        form.addRow("Threshold(s)", self._threshold)

        self._threshold_dir = QComboBox()
        for d in ("higher", "lower"):
            self._threshold_dir.addItem(d)
        idx = 0
        if settings.threshold_direction == "lower":
            idx = 1
        elif isinstance(settings.threshold_direction, (list, tuple)) \
                and settings.threshold_direction \
                and str(settings.threshold_direction[0]).lower() == "lower":
            idx = 1
        self._threshold_dir.setCurrentIndex(idx)
        form.addRow("Direction", self._threshold_dir)

        # -- active-learning queue (spacr.active_learning) -------------------
        self._queue_on = Toggle("Order by model uncertainty")
        self._queue_on.setChecked(bool(getattr(settings, "queue_by_uncertainty", False)))
        self._queue_on.setToolTip(
            "Show the unlabelled crops the classifier is least sure about "
            "first. Needs model scores in png_list, so run Classify (CV) "
            "before turning this on.")
        form.addRow("Queue", self._queue_on)

        self._queue_measure = QComboBox()
        for m in ("entropy", "least_confidence", "margin"):
            self._queue_measure.addItem(m)
        self._queue_measure.setCurrentText(
            str(getattr(settings, "queue_measure", "entropy")))
        self._queue_measure.setToolTip(
            "How uncertainty is scored. With two classes, margin and "
            "least_confidence give the identical ranking; they only diverge "
            "at three classes or more.")
        form.addRow("Uncertainty measure", self._queue_measure)

        self._queue_diversity = QComboBox()
        for d in ("well", "field", "plate", "none"):
            self._queue_diversity.addItem(d)
        self._queue_diversity.setCurrentText(
            str(getattr(settings, "queue_diversity", "well")))
        self._queue_diversity.setToolTip(
            "Spread the queue across wells rather than serving the most "
            "uncertain crops in ranked order. Pure uncertainty collapses onto "
            "one or two wells, so you end up labelling the same ambiguity a "
            "hundred times. 'none' turns that protection off.")
        form.addRow("Queue diversity", self._queue_diversity)

        self._queue_limit = QSpinBox()
        self._queue_limit.setRange(0, 1_000_000)
        self._queue_limit.setValue(int(getattr(settings, "queue_limit", 0) or 0))
        self._queue_limit.setSpecialValueText("all unlabelled")
        form.addRow("Queue length", self._queue_limit)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

        from .settings_model import install_api_tooltips
        install_api_tooltips(self, "annotate", {
            self._src_edit: "src",
            self._ann_col: "annotation_column",
            self._img_size: "image_size",
            self._image_type: "image_type",
            self._channels: "channels",
            self._stored_channel_order: "stored_channel_order",
            self._norm_channels: "normalize_channels",
            self._pct_lo: "lower_percentile",
            self._pct_hi: "upper_percentile",
            self._outline: "outline",
            self._outline_method: "outline_method",
            self._out_factor: "outline_threshold_factor",
            self._out_sigma: "outline_sigma",
            self._edge_thick: "edge_thickness",
            self._edge_transp: "edge_transparency",
            self._edge_image: "edge_image",
            self._obj_min: "object_min_size",
            self._obj_max: "object_max_size",
            self._measurement: "measurement",
            self._threshold: "threshold",
            self._threshold_dir: "threshold_direction",
            self._queue_on: "queue_by_uncertainty",
            self._queue_measure: "queue_measure",
            self._queue_diversity: "queue_diversity",
            self._queue_limit: "queue_limit",
        })

    def _picker_db_path(self) -> str:
        """Where the SQL picker looks — the src folder as it reads right now.

        A callable rather than a captured string: users routinely set the
        source folder and the annotation column in the same visit to this
        dialog, and a value captured at construction would point at the
        previous folder.
        """
        return self._src_edit.text().strip()

    def _pick_src(self):
        d = QFileDialog.getExistingDirectory(self, "Pick experiment source",
                                              self._src_edit.text() or os.getcwd())
        if d:
            self._src_edit.setText(d)

    def collect(self) -> AnnotateSettings:
        """Read every editor and return the updated settings object."""
        s = self._settings
        s.src = self._src_edit.text().strip()
        s.db_path = os.path.join(s.src, "measurements", "measurements.db")
        s.annotation_column = self._ann_col.text().strip() or "annotate"
        size = int(self._img_size.value())
        s.image_size = (size, size)
        s.image_type = self._image_type.text().strip() or None
        s.channels = _csv_to_list(self._channels.text())
        s.stored_channel_order = str(
            self._stored_channel_order.currentData() or "rgb")
        s.normalize_channels = _csv_to_list(self._norm_channels.text())
        s.percentiles = (float(self._pct_lo.value()), float(self._pct_hi.value()))
        s.outline = _csv_to_list(self._outline.text())
        s.outline_method = self._outline_method.currentText()
        s.outline_threshold_factor = float(self._out_factor.value())
        s.outline_sigma = float(self._out_sigma.value())
        s.edge_thickness = float(self._edge_thick.value())
        s.edge_transparency = float(self._edge_transp.value())
        s.edge_image = bool(self._edge_image.isChecked())
        s.object_size = (int(self._obj_min.value()), int(self._obj_max.value()))
        # Threshold filter
        meas_txt = self._measurement.text().strip()
        s.measurement = _csv_to_list(meas_txt)
        thr_txt = self._threshold.text().strip()
        if thr_txt:
            parts = [p.strip() for p in thr_txt.split(",") if p.strip()]
            parsed: List[float] = []
            for p in parts:
                try:
                    parsed.append(float(p))
                except ValueError:
                    pass
            s.threshold = parsed or None
        else:
            s.threshold = None
        s.threshold_direction = self._threshold_dir.currentText() \
            if (s.measurement and s.threshold) else None
        s.queue_by_uncertainty = bool(self._queue_on.isChecked())
        s.queue_measure = self._queue_measure.currentText()
        s.queue_diversity = self._queue_diversity.currentText()
        s.queue_limit = int(self._queue_limit.value())
        return s


# ---------------------------------------------------------------------------
# AnnotateScreen
# ---------------------------------------------------------------------------

class AnnotateScreen(QWidget):
    """Main Qt widget for the annotate app."""

    # Emitted with (target_app_key, seed_settings_dict); MainWindow
    # picks this up to switch to that screen and preseed values.
    train_requested = Signal(str, dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._settings = AnnotateSettings()
        self._offset = 0
        self._total = 0
        self._page_paths: List[Tuple[str, Optional[int]]] = []
        self._filtered_rows: Optional[List[Tuple[str, Optional[int]]]] = None
        #: rendered spread/class-balance summary when the uncertainty queue is on
        self._queue_summary: str = ""
        self._pending_updates: Dict[str, Optional[int]] = {}
        self._worker: Optional[SaveWorker] = None
        self._thumbs: List[_Thumbnail] = []
        self._thumb_pixmaps: List[Optional[QPixmap]] = []
        self._raw_thumb_images: List[Optional[Image.Image]] = []
        self._page_worker: Optional[_PageLoadWorker] = None
        self._pending_page_load = None
        self._page_gen = 0
        self._closing = False
        # A drag-resize used to launch one QThread (and one inner thread pool)
        # per geometry event.  Debounce it and keep only the newest page
        # request so native image/model code never overlaps with itself.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._reload_after_resize)
        self._suggested_source = prefs.get_last_source("annotate")

        # ── The current tile ───────────────────────────────────────────────
        # ONE notion, shared by mouse and keyboard: `_focus_slot` is the crop
        # the next action hits and the only tile that wears the white ring.
        # The cursor entering a tile moves it; an arrow key moves it. There
        # is deliberately no second "hovered tile" that could disagree.
        self._focus_slot = 0
        # Bookkeeping only: which tile the cursor is inside right now, or
        # None when it is between tiles / outside the grid. Whenever it is
        # set it equals `_focus_slot` (see `_set_hover_slot`), so the white
        # ring never has two candidates.
        self._hover_slot: Optional[int] = None
        # (slot, png_path, previous_value) for `u`. Bounded — a long session
        # must not grow this without limit. Cleared on every page load since
        # slot indices change meaning.
        self._undo_stack: Deque[Tuple[int, str, Optional[int]]] = deque(
            maxlen=UNDO_LIMIT)
        self._legend_expanded = False

        self._build_ui()
        self._install_shortcuts()
        # The screen itself owns keystrokes; the thumbnails are NoFocus
        # QLabels so nothing inside the grid competes for them.
        self.setFocusPolicy(Qt.StrongFocus)

        # Drag & drop — accepts a plate folder with
        # measurements/measurements.db (or the .db file directly).
        try:
            from ..dnd import install_dropzone
            from ..dnd_handlers import AnnotateDropHandler
            install_dropzone(self, AnnotateDropHandler(), self)
        except Exception:
            pass

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status_label)
        self._status_timer.start()

        if self._suggested_source and os.path.isdir(self._suggested_source):
            self._src_label.setText(
                f"Suggested (last used): {self._suggested_source}"
            )

    # ------------------------------------------------------------------
    def _build_ui(self):
        # Resolved once here rather than imported at module scope, so the
        # grid canvas and the tile chrome agree with the theme the user is
        # actually running (see `tile_palette`).
        PALETTE = tile_palette()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # Header
        header = QWidget()
        hbox = QVBoxLayout(header)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        title = QLabel("Annotate")
        title.setObjectName("TitleHeading")
        hbox.addWidget(title)
        self._src_label = QLabel("No source selected — click Open source…")
        self._src_label.setObjectName("SubtitleSmall")
        hbox.addWidget(self._src_label)
        outer.addWidget(header)
        outer.addWidget(Divider())

        # Toolbar
        toolbar = QWidget()
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        self._btn_open = QPushButton("Open source…")
        self._btn_open.setObjectName("PrimaryButton")
        self._btn_open.setIcon(iconset.contrast_icon("open"))
        self._btn_open.setCursor(Qt.PointingHandCursor)
        self._btn_open.clicked.connect(self._on_pick_source)
        row.addWidget(self._btn_open)

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setIcon(iconset.icon("settings"))
        self._btn_settings.setCursor(Qt.PointingHandCursor)
        self._btn_settings.clicked.connect(self._on_open_settings)
        row.addWidget(self._btn_settings)

        self._btn_prev = QPushButton("Back")
        self._btn_prev.setIcon(iconset.icon("prev"))
        self._btn_prev.setCursor(Qt.PointingHandCursor)
        self._btn_prev.clicked.connect(self._on_prev)
        row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next")
        self._btn_next.setIcon(iconset.icon("next"))
        self._btn_next.setLayoutDirection(Qt.RightToLeft)   # icon on the right
        self._btn_next.setCursor(Qt.PointingHandCursor)
        self._btn_next.clicked.connect(self._on_next)
        row.addWidget(self._btn_next)

        self._btn_skip = QPushButton("Skip to last annotated")
        self._btn_skip.setIcon(iconset.icon("skip"))
        self._btn_skip.setCursor(Qt.PointingHandCursor)
        self._btn_skip.clicked.connect(self._on_skip)
        row.addWidget(self._btn_skip)

        self._btn_count = QPushButton("Class counts")
        self._btn_count.setIcon(iconset.icon("chart"))
        self._btn_count.setCursor(Qt.PointingHandCursor)
        self._btn_count.clicked.connect(self._on_class_counts)
        row.addWidget(self._btn_count)

        self._btn_train_cv = QPushButton("Train CV")
        self._btn_train_cv.setIcon(iconset.icon("classify"))
        self._btn_train_cv.setCursor(Qt.PointingHandCursor)
        self._btn_train_cv.setToolTip(
            "Generate a training dataset from the current annotations "
            "and train a Torch CNN / Transformer classifier, then apply "
            "it to the full dataset. Opens the Classify screen with "
            "this source pre-selected."
        )
        self._btn_train_cv.clicked.connect(self._on_train_cv)
        row.addWidget(self._btn_train_cv)

        self._btn_train_xg = QPushButton("Train XG")
        self._btn_train_xg.setIcon(iconset.icon("chart"))
        self._btn_train_xg.setCursor(Qt.PointingHandCursor)
        self._btn_train_xg.setToolTip(
            "Train an XGBoost model on the measurement features "
            "using the current annotations as class labels, then apply "
            "it to score the full dataset. Opens the ML Analyze screen "
            "with this source pre-selected."
        )
        self._btn_train_xg.clicked.connect(self._on_train_xg)
        row.addWidget(self._btn_train_xg)

        self._btn_clear = QPushButton("Clear column")
        self._btn_clear.setObjectName("DangerButton")
        self._btn_clear.setIcon(iconset.icon("clear", color=PALETTE["error"]))
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(self._on_clear_column)
        row.addWidget(self._btn_clear)

        row.addStretch(1)
        self._page_label = QLabel("")
        self._page_label.setObjectName("SubtitleSmall")
        row.addWidget(self._page_label)
        outer.addWidget(toolbar)

        outer.addWidget(self._build_key_legend())

        # Content stack: empty-state until a source is opened, then grid
        self._content_stack = QStackedWidget()

        self._empty_state = EmptyState(
            title="Open an experiment to start annotating",
            subtitle=(
                "Pick a folder that contains "
                "`measurements/measurements.db`. Left-click an image to "
                "assign class 1, right-click for class 2, click again to "
                "clear — or go keyboard-only: 1–9 label and jump to the "
                "next crop. Annotations save in the background."
            ),
            icon=iconset.accent_icon("tag"),
            cta_label="Open source…",
            on_action=self._on_pick_source,
        )
        self._content_stack.addWidget(self._empty_state)

        # Grid inside a scroll area
        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setFrameShape(QScrollArea.NoFrame)
        # Dark-gray canvas (not black) behind the rounded thumbnails.
        self._grid_scroll.viewport().setStyleSheet(
            f"background: {PALETTE['surface_alt']};")
        self._grid_holder = QWidget()
        self._grid_holder.setObjectName("AnnotateGrid")
        self._grid_holder.setStyleSheet(
            f"QWidget#AnnotateGrid {{ background: {PALETTE['surface_alt']}; }}")
        self._grid_layout = QGridLayout(self._grid_holder)
        self._grid_layout.setSpacing(SPACING["sm"])
        self._grid_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                              SPACING["sm"], SPACING["sm"])
        self._grid_scroll.setWidget(self._grid_holder)
        # Without these the scroll area swallows the arrow keys and scrolls
        # instead of moving grid focus.
        self._grid_scroll.installEventFilter(self)
        self._grid_scroll.viewport().installEventFilter(self)
        self._grid_holder.installEventFilter(self)
        self._content_stack.addWidget(self._grid_scroll)
        self._content_stack.setCurrentWidget(self._empty_state)

        # The grid and the optional Console + AI pane share a vertical
        # splitter.  Annotate starts grid-first; the bottom controls reveal
        # the console on demand without opening a separate window.
        self._runtime_splitter = QSplitter(Qt.Vertical, self)
        self._runtime_splitter.setChildrenCollapsible(False)
        self._runtime_splitter.addWidget(self._content_stack)

        self._console_wrap = QWidget(self)
        console_layout = QVBoxLayout(self._console_wrap)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(SPACING["xs"])
        console_title = QLabel("Console + AI", self._console_wrap)
        console_title.setObjectName("CardTitle")
        console_layout.addWidget(console_title)
        from ..widgets import ConsolePanel
        self._console = ConsolePanel(
            active_app_label="Annotate", parent=self._console_wrap)
        self._console.setMinimumHeight(180)
        console_layout.addWidget(self._console, 1)
        self._runtime_splitter.addWidget(self._console_wrap)
        self._runtime_splitter.setStretchFactor(0, 4)
        self._runtime_splitter.setStretchFactor(1, 2)
        self._console_wrap.hide()
        outer.addWidget(self._runtime_splitter, 1)

        # Status and Console/AI controls stay at the bottom, matching the
        # generic module screens.
        bottom = QWidget(self)
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(SPACING["sm"])
        self._status_label = QLabel("Ready.")
        self._status_label.setObjectName("SubtitleSmall")
        bottom_row.addWidget(self._status_label, 1)

        self._console_switch = QToolButton(self)
        self._console_switch.setText("Console ▾")
        self._console_switch.setCheckable(True)
        self._console_switch.setCursor(Qt.PointingHandCursor)
        self._console_switch.setFocusPolicy(Qt.NoFocus)
        self._console_switch.setToolTip("Show or hide the Console + AI pane.")
        self._console_switch.toggled.connect(self._on_console_switch)
        bottom_row.addWidget(self._console_switch)

        from ..widgets import AiToggleLabel
        self._ai_switch = AiToggleLabel()
        self._ai_switch.toggled.connect(self._on_ai_switch)
        bottom_row.addWidget(self._ai_switch)

        self._ai_menu_btn = QToolButton(self)
        self._ai_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._ai_menu_btn.setCursor(Qt.PointingHandCursor)
        self._ai_menu_btn.setFocusPolicy(Qt.NoFocus)
        self._ai_menu_btn.setToolTip("Pick provider · Providers…")
        self._ai_menu_btn.setText("▾")
        self._ai_menu = QMenu(self._ai_menu_btn)
        self._ai_menu_btn.setMenu(self._ai_menu)
        bottom_row.addWidget(self._ai_menu_btn)
        self._refresh_ai_menu()
        outer.addWidget(bottom)

        self._rebuild_grid()

    def _on_console_switch(self, on: bool) -> None:
        """Expand or collapse Annotate's merged Console + AI pane."""
        self._console_wrap.setVisible(on)
        self._console_switch.setText("Console ▴" if on else "Console ▾")
        if on:
            height = max(480, self._runtime_splitter.height())
            self._runtime_splitter.setSizes(
                [max(240, int(height * 0.62)), max(180, int(height * 0.38))])

    def _on_ai_switch(self, on: bool) -> None:
        """Enable chat routing and reveal the console when AI is selected."""
        self._console.set_ai_active(on)
        if not on:
            return
        if not self._console_switch.isChecked():
            self._console_switch.setChecked(True)
        from .. import ai as ai_module
        if not self._console._current_provider_name:
            configured = ai_module.configured_providers()
            if configured:
                self._console.set_ai_provider(configured[0].name)
                self._refresh_ai_menu()
            else:
                self._console.append_stdout(
                    "[AI] No vendor CLI installed. Click ▾ next to AI → "
                    "Providers…\n")
                self._ai_switch.setChecked(False)

    def _refresh_ai_menu(self) -> None:
        """Rebuild the provider dropdown beside Annotate's AI button."""
        from .. import ai as ai_module
        self._ai_menu.clear()
        configured = ai_module.configured_providers()
        current = self._console._current_provider_name
        if configured:
            for provider in configured:
                action = self._ai_menu.addAction(provider.label)
                action.setCheckable(True)
                action.setChecked(provider.name == current)
                action.triggered.connect(
                    lambda _checked=False, name=provider.name:
                    self._on_pick_provider(name))
            self._ai_menu.addSeparator()
        else:
            self._ai_menu.addAction(
                "(no vendor CLI installed)").setEnabled(False)
            self._ai_menu.addSeparator()
        action = self._ai_menu.addAction("Providers…")
        action.triggered.connect(self._on_open_providers_dialog)

    def _on_pick_provider(self, name: str) -> None:
        self._console.set_ai_provider(name)
        self._refresh_ai_menu()

    def _on_open_providers_dialog(self) -> None:
        from ..widgets.ai_chat_panel import _ProvidersDialog
        dialog = _ProvidersDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self._refresh_ai_menu()

    # ------------------------------------------------------------------
    # Key legend
    # ------------------------------------------------------------------
    LEGEND_COMPACT = (
        "<b>1</b>–<b>9</b> label + advance &nbsp;·&nbsp; <b>0</b> clear "
        "&nbsp;·&nbsp; <b>← ↑ ↓ →</b> / <b>hjkl</b> move &nbsp;·&nbsp; "
        "<b>Space</b> skip &nbsp;·&nbsp; <b>Backspace</b> back "
        "&nbsp;·&nbsp; <b>u</b> undo &nbsp;·&nbsp; <b>Enter</b> next batch"
    )
    LEGEND_FULL = (
        "<b>1</b>–<b>9</b> assign that class to the focused crop and jump to "
        "the next unlabelled one &nbsp;·&nbsp; <b>0</b> clear the focused "
        "crop &nbsp;·&nbsp; <b>← ↑ ↓ →</b> or <b>h j k l</b> move focus "
        "without labelling &nbsp;·&nbsp; <b>Space</b> skip forward one "
        "&nbsp;·&nbsp; <b>Backspace</b> step back one &nbsp;·&nbsp; "
        "<b>u</b> undo the last label &nbsp;·&nbsp; <b>Enter</b> save this "
        "page and load the next batch &nbsp;·&nbsp; mouse still works: "
        "left-click = class 1, right-click = class 2."
    )

    def _build_key_legend(self) -> QWidget:
        """Build the always-visible keyboard cheat strip.

        Nothing in it accepts focus — a legend that stole focus would break
        the very keyboard flow it documents.
        """
        PALETTE = tile_palette()
        legend = QWidget()
        legend.setObjectName("AnnotateKeyLegend")
        legend.setFocusPolicy(Qt.NoFocus)
        legend.setStyleSheet(
            f"QWidget#AnnotateKeyLegend {{ background: {PALETTE['surface']};"
            f" border: 1px solid {PALETTE['border_soft']};"
            f" border-radius: 6px; }}"
        )
        lay = QHBoxLayout(legend)
        lay.setContentsMargins(SPACING["sm"], SPACING["xs"],
                                SPACING["sm"], SPACING["xs"])
        lay.setSpacing(SPACING["sm"])

        self._legend_label = QLabel(self.LEGEND_COMPACT)
        self._legend_label.setObjectName("SubtitleSmall")
        self._legend_label.setTextFormat(Qt.RichText)
        self._legend_label.setWordWrap(True)
        self._legend_label.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self._legend_label, 1)

        # Transient keyboard feedback ("end of page", "nothing to undo").
        # Deliberately NOT the shared status label: that one is rewritten
        # every 500 ms by the save-state timer, which would eat the message.
        self._kbd_hint = QLabel("")
        self._kbd_hint.setObjectName("SubtitleSmall")
        self._kbd_hint.setFocusPolicy(Qt.NoFocus)
        self._kbd_hint.setStyleSheet(f"color: {PALETTE['warning']};")
        lay.addWidget(self._kbd_hint, 0)

        self._legend_toggle = QPushButton("?")
        self._legend_toggle.setFocusPolicy(Qt.NoFocus)
        self._legend_toggle.setCursor(Qt.PointingHandCursor)
        self._legend_toggle.setFixedWidth(28)
        self._legend_toggle.setToolTip("Show the full keyboard reference")
        self._legend_toggle.clicked.connect(self._toggle_legend)
        lay.addWidget(self._legend_toggle, 0)

        self._legend = legend
        return legend

    def _toggle_legend(self) -> bool:
        """Flip the legend between the compact strip and the full reference."""
        self._legend_expanded = not self._legend_expanded
        self._legend_label.setText(
            self.LEGEND_FULL if self._legend_expanded else self.LEGEND_COMPACT)
        return True

    def _set_kbd_hint(self, text: str = "") -> None:
        """Show (or clear) the transient keyboard-mode message."""
        if getattr(self, "_kbd_hint", None) is not None:
            self._kbd_hint.setText(text)

    def _install_shortcuts(self):
        # Bare arrow keys now drive grid focus (see `handle_key`), so page
        # navigation moved to PageUp/PageDown with Alt+Arrow kept as an
        # alias for anyone with the old muscle memory.
        QShortcut(QKeySequence(Qt.Key_PageUp), self, self._on_prev)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, self._on_next)
        QShortcut(QKeySequence("Alt+Left"), self, self._on_prev)
        QShortcut(QKeySequence("Alt+Right"), self, self._on_next)

    # ------------------------------------------------------------------
    def _compute_grid_dims(self):
        """Fit as many `image_size`-thumbnails as possible into the
        scroll viewport, then update settings.grid_rows/grid_cols."""
        w, h = self._settings.image_size
        gap = SPACING["xs"]
        pad = TILE_INSET * 2
        cell_w = w + pad + gap
        cell_h = h + pad + gap
        vp = self._grid_scroll.viewport() if self._grid_scroll else None
        if vp is not None and vp.width() > cell_w and vp.height() > cell_h:
            cols = max(1, vp.width() // cell_w)
            rows = max(1, vp.height() // cell_h)
        else:
            # No viewport yet — fall back to previous values (or a
            # sensible default of a 5x5 grid).
            cols = max(1, self._settings.grid_cols or 5)
            rows = max(1, self._settings.grid_rows or 5)
        self._settings.grid_cols = cols
        self._settings.grid_rows = rows

    def _rebuild_grid(self):
        """Regenerate empty thumbnail widgets sized for current settings."""
        # Recompute page-fit before we create widgets
        self._compute_grid_dims()
        for w in self._thumbs:
            w.setParent(None)
            w.deleteLater()
        self._thumbs.clear()
        # Every widget the cursor could have been inside is gone; keeping the
        # index would leave a hover pointing at a tile that no longer exists.
        self._hover_slot = None
        self._thumb_pixmaps = [None] * (self._settings.grid_rows *
                                         self._settings.grid_cols)
        self._raw_thumb_images = [None] * len(self._thumb_pixmaps)

        cols = self._settings.grid_cols
        rows = self._settings.grid_rows
        w, h = self._settings.image_size
        pad = TILE_INSET * 2
        # One palette lookup for the whole grid — the hover path must not
        # pay for a theme resolution on every mouse move.
        resting = resting_border_color()
        ring = current_ring_color()
        for i in range(rows * cols):
            thumb = _Thumbnail(i, border_color=resting, ring_color=ring)
            thumb.setFixedSize(w + pad, h + pad)
            thumb.left_clicked.connect(self._on_thumb_left)
            thumb.right_clicked.connect(self._on_thumb_right)
            thumb.hover_changed.connect(self._on_thumb_hover)
            self._grid_layout.addWidget(thumb, i // cols, i % cols)
            self._thumbs.append(thumb)

        # Widgets were just recreated — re-establish the focus marker.
        self._focus_slot = max(0, min(self._focus_slot, len(self._thumbs) - 1))
        self._refresh_focus_marks()

    def resizeEvent(self, event):
        """Re-fit the thumbnail grid after resize activity settles."""
        super().resizeEvent(event)
        if not getattr(self, "_grid_scroll", None):
            return
        prev = (self._settings.grid_rows, self._settings.grid_cols)
        self._compute_grid_dims()
        new = (self._settings.grid_rows, self._settings.grid_cols)
        if new != prev and self._worker is not None:
            self._resize_timer.start()

    @Slot()
    def _reload_after_resize(self):
        """Apply the final geometry after a burst of resize events."""
        if self._closing or self._worker is None:
            return
        self._flush_pending()
        self._rebuild_grid()
        self._refresh_total()
        self._load_page()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_pick_source(self):
        starting = (self._settings.src
                    or self._suggested_source
                    or os.getcwd())
        d = QFileDialog.getExistingDirectory(self, "Pick experiment source",
                                              starting)
        if not d:
            return
        self._open_source(d)

    def _open_source(self, src: str):
        db_path = os.path.join(src, "measurements", "measurements.db")
        if not os.path.isfile(db_path):
            answer = QMessageBox.question(
                self, "Database not found",
                f"No file at:\n{db_path}\n\nUse it anyway?",
            )
            if answer != QMessageBox.Yes:
                return
        # Tear down previous worker
        self._flush_pending()
        if self._worker:
            self._worker.stop(wait=True)
            self._worker = None
        self._settings.src = src
        self._settings.db_path = db_path
        ensure_annotation_column(db_path, self._settings.annotation_column)
        self._worker = SaveWorker(db_path, self._settings.annotation_column)
        self._worker.start()
        self._offset = 0
        self._src_label.setText(f"{src}  →  {db_path}")
        self._console.append_stdout(
            f"[Annotate] Opened {db_path}\n")
        prefs.push_recent_source("annotate", src)
        # Show the grid page FIRST so its viewport is realized, then defer the
        # grid build + first load to the next event-loop tick. Otherwise
        # _compute_grid_dims measures a zero-size viewport and builds a 5x5
        # fallback, so the first open showed only a few images that don't fill
        # the view (until the user opened Settings, which rebuilt the grid).
        self._content_stack.setCurrentWidget(self._grid_scroll)
        # Take keyboard focus so the user can start keying classes straight
        # away without first clicking into the grid.
        self.setFocus(Qt.OtherFocusReason)
        self._refresh_total()
        QTimer.singleShot(0, self._rebuild_and_load)

    def _rebuild_and_load(self):
        """Rebuild the grid against the (now realized) viewport, then load."""
        self._rebuild_grid()
        self._load_page()

    def _on_open_settings(self):
        dlg = _SettingsDialog(self._settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        old_src = self._settings.src
        old_col = self._settings.annotation_column
        self._settings = dlg.collect()
        self._rebuild_grid()
        # Restart worker if src/col changed
        if self._settings.src != old_src or self._settings.annotation_column != old_col:
            self._open_source(self._settings.src)
        else:
            self._refresh_total()
            self._load_page()

    def _on_next(self):
        self._flush_pending()
        page = self._settings.page_size
        if self._offset + page < max(self._total, 1):
            self._offset += page
            self._load_page()

    def _on_prev(self):
        self._flush_pending()
        page = self._settings.page_size
        self._offset = max(0, self._offset - page)
        self._load_page()

    def _on_skip(self):
        self._flush_pending()
        offset = find_last_annotated_offset(
            self._settings.db_path,
            self._settings.annotation_column,
            self._settings.page_size,
            self._settings.image_type,
        )
        if offset is None:
            self._status_label.setText("No annotated images found.")
            return
        self._offset = offset
        self._load_page()

    def _on_class_counts(self):
        rows = class_counts(self._settings.db_path, self._settings.annotation_column)
        if not rows:
            QMessageBox.information(self, "Class counts", "No annotated rows yet.")
            return
        lines = ["Class    Count    Color"]
        for cls, cnt in rows:
            lines.append(f"{cls:>5}  {cnt:>7}    {label_to_hex(cls) or ''}")
        QMessageBox.information(self, "Class counts", "\n".join(lines))

    def _on_train_cv(self):
        """Save any pending annotations, then hand off to Classify."""
        if not self._settings.src:
            QMessageBox.information(
                self, "Open a source first",
                "Open an experiment source before training a classifier.",
            )
            return
        self._flush_pending()
        seed = {
            "src": self._settings.src,
            "annotation_column": self._settings.annotation_column,
            # nudge the train pipeline into the "annotation → train → apply"
            # mode. dataset_mode is what actually selects the classes:
            # generate_training_dataset alone left it at the Classify panel's
            # default, 'metadata', so "Train CV" from the Annotate app built
            # its classes from well metadata and ignored the annotations that
            # had just been made. It used to die on the way there
            # (KeyError: 'condition'); now that metadata mode works, leaving
            # this unset would silently train on the wrong labels.
            "dataset_mode": "annotation",
            "generate_training_dataset": True,
            "train": True,
            "apply_model_to_dataset": True,
        }
        self.train_requested.emit("classify", seed)

    def _on_train_xg(self):
        """Save any pending annotations, then hand off to ML Analyze."""
        if not self._settings.src:
            QMessageBox.information(
                self, "Open a source first",
                "Open an experiment source before training an XGBoost model.",
            )
            return
        self._flush_pending()
        seed = {
            "src": self._settings.src,
            "annotation_column": self._settings.annotation_column,
            "model_type": "xgboost",
        }
        self.train_requested.emit("ml_analyze", seed)

    def _on_clear_column(self):
        col = self._settings.annotation_column
        answer = QMessageBox.question(
            self, "Confirm clear",
            f'Clear ALL annotations in column "{col}"?\nThis cannot be undone.',
        )
        if answer != QMessageBox.Yes:
            return
        self._pending_updates.clear()
        clear_column(self._settings.db_path, col)
        self._refresh_total()
        self._load_page()

    def _on_thumb_left(self, slot: int):
        self._toggle_annotation(slot, 1)

    def _on_thumb_right(self, slot: int):
        self._toggle_annotation(slot, 2)

    # ------------------------------------------------------------------
    # Page loading + rendering
    # ------------------------------------------------------------------
    def _filter_active(self) -> bool:
        s = self._settings
        return bool(s.measurement and s.threshold and s.threshold_direction)

    def _refresh_total(self):
        s = self._settings
        if s.queue_by_uncertainty:
            # Order the unlabelled crops by how unsure the model is about them,
            # so the annotator spends their time on the decision boundary. The
            # queue is a snapshot, rebuilt on every settings apply, so crops
            # labelled since the last rebuild drop out then rather than now.
            from ... import active_learning as al
            try:
                queue = al.build_queue(
                    s.db_path, s.annotation_column,
                    measure=s.queue_measure,
                    diversity=(s.queue_diversity or "none"),
                    limit=(s.queue_limit or None),
                    image_type=s.image_type, seed=0)
            except (FileNotFoundError, ValueError) as exc:
                # No model scores yet is the ordinary case before a classifier
                # has run, so fall back to page order and say why rather than
                # showing an empty grid.
                self._filtered_rows = None
                self._total = count_rows(s.db_path, s.image_type)
                self._queue_summary = ""
                self._page_label.setText(f"Uncertainty queue unavailable: {exc}")
                return
            self._queue_summary = al.format_queue_summary(queue)
            self._filtered_rows = al.queue_rows(queue)
            self._total = len(self._filtered_rows)
            return
        self._queue_summary = ""
        if self._filter_active():
            # Cache the filtered set once so pagination + total agree
            self._filtered_rows = fetch_filtered_paths(
                self._settings.db_path,
                self._settings.annotation_column,
                self._settings.measurement if isinstance(self._settings.measurement, list)
                else [self._settings.measurement],
                self._settings.threshold if isinstance(self._settings.threshold, list)
                else [self._settings.threshold],
                self._settings.threshold_direction if isinstance(
                    self._settings.threshold_direction, list
                ) else [self._settings.threshold_direction],
                self._settings.image_type,
            )
            self._total = len(self._filtered_rows)
        else:
            self._filtered_rows = None
            self._total = count_rows(self._settings.db_path, self._settings.image_type)

    def _load_page(self):
        if self._closing:
            return
        page = self._settings.page_size
        if self._filtered_rows is not None:
            self._page_paths = list(self._filtered_rows[self._offset:self._offset + page])
        else:
            self._page_paths = fetch_page(
                self._settings.db_path,
                self._settings.annotation_column,
                self._offset,
                page,
                self._settings.image_type,
            )
        # Clear all thumbs
        for i in range(len(self._thumbs)):
            self._set_slot_image(i, None)

        # Slot indices now mean different crops — an undo entry from the old
        # page would write a label onto the wrong image.
        self._undo_stack.clear()
        self._set_kbd_hint("")
        # The crops under the grid just changed. A hover recorded against
        # the previous page is only still true if the cursor is genuinely
        # inside that same widget.
        self._revalidate_hover()
        # Park the keyboard on the first crop that still needs a label.
        first = self._next_unannotated(0)
        self._set_focus_slot(first if first is not None else 0)
        # Repaint every cell so occupancy + resting borders match the new
        # page (cells past the end of a short last page draw nothing).
        for i in range(len(self._thumbs)):
            self._repaint_slot(i)

        # Process the page (normalise + outline) on a worker thread so the UI
        # stays responsive even when the recompute is slow. A generation token
        # discards results from a page/settings change the user has since
        # superseded.
        self._page_gen += 1
        self._page_label.setText(f"Loading {len(self._page_paths)} images…")
        crop_src = self._crop_source()
        # Lists inside AnnotateSettings are mutable. Freeze the complete view
        # configuration now so a Settings change cannot race the decoder.
        settings = deepcopy(self._settings)
        request = (self._page_gen, list(self._page_paths), crop_src, settings)
        self._queue_page_load(request)

    def _queue_page_load(self, request):
        """Run at most one page QThread, retaining only the newest request."""
        if self._closing:
            return
        worker = self._page_worker
        # A finished QThread is still owned here until its queued ``finished``
        # slot runs on the GUI thread. Replacing that reference in the small
        # gap would let the old slot retire the new live worker.
        if worker is not None:
            self._pending_page_load = request
            return
        self._start_page_worker(request)

    def _start_page_worker(self, request):
        gen, paths, crop_src, settings = request
        load_fn = partial(
            _load_thumb_image_worker,
            src=crop_src,
            settings=settings,
        )
        worker = _PageLoadWorker(gen, paths, load_fn)
        worker.setObjectName(f"annotate-page-{gen}")
        worker.done.connect(self._on_page_loaded, Qt.QueuedConnection)
        worker.finished.connect(
            self._on_page_worker_finished,
            Qt.QueuedConnection,
        )
        self._page_worker = worker
        worker.start()

    @Slot()
    def _on_page_worker_finished(self):
        """Retire the QThread on the GUI thread and launch the newest request."""
        worker = self._page_worker
        self._page_worker = None
        if worker is not None:
            worker.deleteLater()
        if self._pending_page_load is not None and not self._closing:
            request = self._pending_page_load
            self._pending_page_load = None
            self._start_page_worker(request)

    @Slot(int, object)
    def _on_page_loaded(self, gen: int, loaded):
        if gen != self._page_gen:
            return   # superseded by a newer load
        page = self._settings.page_size
        for i, (img, _annotation) in enumerate(loaded):
            if i >= len(self._thumbs):
                break
            self._set_slot_image(i, img)
            # Paint from `_page_paths`, not the annotation the worker
            # snapshotted: the user may have keyed labels in while the page
            # was still decoding, and those are the fresher truth.
            self._repaint_slot(i)
        self._page_label.setText(
            f"Page rows {self._offset}–{min(self._offset + page, self._total)} / {self._total}"
        )

    def _crop_source(self):
        """Resolve the crop source once per settings change, then cache it.

        'auto' keeps today's behaviour: the PNG folder is used whenever one
        exists, and only a project without one falls back to cutting crops out
        of merged/*.npy. See spacr.crops.resolve_crop_source.
        """
        s = self._settings
        key = (s.db_path, getattr(s, "crop_source", "auto"), s.image_type)
        if getattr(self, "_cropsrc_key", None) != key:
            from ...crops import resolve_crop_source
            root = os.path.dirname(os.path.dirname(s.db_path or ""))
            obj = (s.image_type or "cell_png").replace("_png", "") or "cell"
            try:
                self._cropsrc = resolve_crop_source(
                    {"src": root, "crop_source": getattr(s, "crop_source", "auto")},
                    object_type=obj)
            except Exception:
                self._cropsrc = None      # PNG path below still works
            self._cropsrc_key = key
        return self._cropsrc

    def _load_thumb_image(self, row, src=None, settings=None):
        """Compatibility wrapper for direct callers and older tests.

        Page workers call :func:`_load_thumb_image_worker` with snapshots and
        never retain this bound QWidget method.
        """
        if src is None:
            src = self._crop_source()
        if settings is None:
            settings = deepcopy(self._settings)
        return _load_thumb_image_worker(row, src, settings)

    def _image_to_pixmap(self, img: Image.Image) -> QPixmap:
        """Convert one decoded crop to a bare pixmap.

        No border and no corner rounding are baked in — both are painted by
        :class:`_Thumbnail`, so changing either never costs a conversion.
        """
        qimg = ImageQt(img.convert("RGB"))
        return QPixmap.fromImage(QImage(qimg))

    # ------------------------------------------------------------------
    # Annotation write path
    #
    # `_set_annotation` is the ONE place a label is recorded. Mouse clicks
    # (`_toggle_annotation`), keyboard assignment, clearing and undo all
    # funnel through it, so they can never drift apart.
    # ------------------------------------------------------------------
    def _slot_is_valid(self, slot: int) -> bool:
        """True when ``slot`` addresses a crop present on the current page."""
        return 0 <= slot < self._slot_count()

    def _slot_count(self) -> int:
        """Number of keyboard-navigable crops on this page."""
        return min(len(self._page_paths), len(self._thumbs))

    def _current_value(self, slot: int) -> Optional[int]:
        """The label ``slot`` carries right now, pending writes included."""
        if not (0 <= slot < len(self._page_paths)):
            return None
        path, current = self._page_paths[slot]
        if path in self._pending_updates:
            return self._pending_updates[path]
        return current

    def _is_annotated(self, slot: int) -> bool:
        """True when ``slot`` already carries a non-zero class label."""
        value = self._current_value(slot)
        return value is not None and value != 0

    def _set_annotation(self, slot: int, value: Optional[int]) -> bool:
        """Record ``value`` (or ``None`` to clear) as ``slot``'s label."""
        if not (0 <= slot < len(self._page_paths)):
            return False
        path, _ = self._page_paths[slot]
        self._pending_updates[path] = value
        self._page_paths[slot] = (path, value)
        self._repaint_slot(slot)
        return True

    def _set_slot_image(self, slot: int, img: Optional[Image.Image]) -> None:
        """Install (or clear) the decoded crop for ``slot``.

        This is the ONLY place a pixmap is built. Border and ring changes go
        through :meth:`_repaint_slot`, which never touches pixels.
        """
        if not (0 <= slot < len(self._thumbs)):
            return
        self._raw_thumb_images[slot] = img
        if img is None:
            self._thumb_pixmaps[slot] = None
            self._thumbs[slot].setPixmap(QPixmap())
            return
        pm = self._image_to_pixmap(img)
        self._thumb_pixmaps[slot] = pm
        self._thumbs[slot].setPixmap(pm)

    def _border_color_for(self, slot: int) -> str:
        """The state-ring colour for ``slot``: its class colour, else gray.

        ``label_to_hex`` is the app's one class→colour map (the Class counts
        dialog reads the same function), so the border can never disagree
        with the colour shown anywhere else.
        """
        return label_to_hex(self._current_value(slot)) or resting_border_color()

    def _repaint_slot(self, slot: int) -> None:
        """Sync one tile's chrome with the model. Cheap: no pixmap work.

        Sets the state ring (class colour or resting gray) and whether this
        is the current tile. Both setters no-op when nothing changed, so a
        redundant call costs nothing and a real change costs one
        ``update()`` on one widget.
        """
        if not (0 <= slot < len(self._thumbs)):
            return
        thumb = self._thumbs[slot]
        thumb.set_occupied(slot < len(self._page_paths))
        thumb.set_border_color(self._border_color_for(slot))
        thumb.set_current(slot == self._focus_slot)

    def _toggle_annotation(self, slot: int, new_value: int):
        """Mouse semantics: same class again clears, otherwise assign."""
        if not self._slot_is_valid(slot):
            return
        existing = self._current_value(slot)
        resolved = None if existing == new_value else new_value
        if self._set_annotation(slot, resolved):
            path = str(self._page_paths[slot][0])
            # A filesystem path can legally contain line breaks. Escape them
            # so every click remains one searchable console record.
            path = path.replace("\r", r"\r").replace("\n", r"\n")
            self._console.append_stdout(
                f"path={path} | annotation={resolved}\n"
            )

    # ------------------------------------------------------------------
    # Keyboard-only rapid annotation
    # ------------------------------------------------------------------
    def _refresh_focus_marks(self) -> None:
        """Re-apply the current-tile marker across every thumbnail."""
        for i in range(len(self._thumbs)):
            self._repaint_slot(i)

    def _set_focus_slot(self, slot: int, ensure_visible: bool = True) -> None:
        """Move the current tile to ``slot``, repainting the old and new cells.

        ``ensure_visible`` is False on the hover path: the tile the cursor is
        inside is visible by definition, and scrolling under the cursor could
        drag a fresh tile under it and set off a feedback loop.
        """
        if not self._thumbs:
            self._focus_slot = max(0, slot)
            return
        slot = max(0, min(int(slot), len(self._thumbs) - 1))
        previous = self._focus_slot
        self._focus_slot = slot
        # A keyboard move away from the hovered tile makes the recorded
        # hover stale — the cursor has not moved, but it is no longer on the
        # tile the next action hits, and only that tile may wear the ring.
        if self._hover_slot is not None and self._hover_slot != slot:
            self._hover_slot = None
        if previous != slot and 0 <= previous < len(self._thumbs):
            self._repaint_slot(previous)
        self._repaint_slot(slot)
        if ensure_visible:
            try:
                self._grid_scroll.ensureWidgetVisible(self._thumbs[slot])
            except Exception:
                pass           # no viewport yet — nothing to scroll into

    # -- hover ----------------------------------------------------------
    def _on_thumb_hover(self, slot: int, entered: bool) -> None:
        """Handle a tile's Enter/Leave. Runs once per boundary crossed."""
        if entered:
            self._set_hover_slot(slot)
        elif self._hover_slot == slot:
            self._set_hover_slot(None)

    def _set_hover_slot(self, slot: Optional[int]) -> None:
        """Record which tile the cursor is inside and follow it.

        Entering a tile makes it the current tile, so the white ring is
        always on the crop the next click or keystroke will hit. Leaving
        only forgets the cursor position: the ring stays where it is,
        because the keyboard still targets that crop.
        """
        if slot is not None:
            slot = int(slot)
            # Empty cells past the end of a short page hold no crop, so
            # there is nothing there to be "on".
            if not (0 <= slot < self._slot_count()):
                slot = None
        if slot == self._hover_slot:
            return
        self._hover_slot = slot
        if slot is not None:
            self._set_focus_slot(slot, ensure_visible=False)

    def _revalidate_hover(self) -> None:
        """Drop a hover the cursor is no longer actually inside.

        Called after a page load: the widgets stay put but the crops under
        them change, and Qt only re-sends Enter/Leave when the cursor
        crosses a boundary. ``underMouse`` is the authority on where the
        cursor really is.
        """
        slot = self._hover_slot
        if slot is None:
            return
        if not (0 <= slot < self._slot_count()) \
                or not self._thumbs[slot].underMouse():
            self._hover_slot = None

    @property
    def focus_slot(self) -> int:
        """Index of the crop the keyboard currently acts on."""
        return self._focus_slot

    @property
    def hover_slot(self) -> Optional[int]:
        """Index of the tile the cursor is inside, or ``None``."""
        return self._hover_slot

    @property
    def current_slot(self) -> int:
        """The one tile wearing the white ring — mouse and keyboard agree."""
        return self._focus_slot

    def _next_unannotated(self, start: int) -> Optional[int]:
        """First unlabelled slot at or after ``start``; ``None`` if there is none."""
        for i in range(max(0, start), self._slot_count()):
            if not self._is_annotated(i):
                return i
        return None

    def _push_undo(self, slot: int, path: str, previous: Optional[int]) -> None:
        self._undo_stack.append((slot, path, previous))

    def handle_key(self, key, text: str = "") -> bool:
        """Run the annotate keybinding for ``key``.

        ``key`` may be a Qt key code, a Qt key name (``"Left"``) or a literal
        character (``"1"``, ``"h"``). Returns True when the key is bound —
        unbound keys return False and are left for Qt's default handling.
        This is the single entry point for the whole keyboard feature so it
        can be driven directly, without synthesising key events.
        """
        token = key_token(key, text)
        if token is None:
            return False
        if token.isdigit():
            value = int(token)
            return self._kbd_clear() if value == 0 else self._kbd_assign(value)
        if token in ("left", "right", "up", "down"):
            return self._kbd_move(token)
        if token == "space":
            return self._kbd_step(+1)
        if token == "backspace":
            return self._kbd_step(-1)
        if token == "undo":
            return self._kbd_undo()
        if token == "enter":
            return self._kbd_commit_page()
        if token == "help":
            return self._toggle_legend()
        if token == "escape":
            # Only meaningful while the full reference is showing; otherwise
            # leave Escape to whatever dialog/window wants it.
            if self._legend_expanded:
                return self._toggle_legend()
            return False
        return False      # pragma: no cover - every token above is handled

    # -- individual actions --------------------------------------------
    def _kbd_assign(self, value: int) -> bool:
        """Label the focused crop with ``value`` and advance."""
        slot = self._focus_slot
        if not self._slot_is_valid(slot):
            self._set_kbd_hint("Nothing to annotate — open a source first.")
            return True
        path = self._page_paths[slot][0]
        previous = self._current_value(slot)
        self._set_annotation(slot, value)
        self._push_undo(slot, path, previous)
        self._advance_after_assign()
        return True

    def _kbd_clear(self) -> bool:
        """Clear the focused crop's label, staying put so it can be re-keyed."""
        slot = self._focus_slot
        if not self._slot_is_valid(slot):
            self._set_kbd_hint("Nothing to annotate — open a source first.")
            return True
        path = self._page_paths[slot][0]
        previous = self._current_value(slot)
        self._set_annotation(slot, None)
        self._push_undo(slot, path, previous)
        self._set_kbd_hint("Cleared.")
        return True

    def _advance_after_assign(self) -> bool:
        """Jump to the next unlabelled crop; never wrap without saying so."""
        nxt = self._next_unannotated(self._focus_slot + 1)
        if nxt is not None:
            self._set_focus_slot(nxt)
            self._set_kbd_hint("")
            return True
        # No unlabelled crop AFTER the focus. Stay put rather than silently
        # wrapping to the top, and say which of the two situations this is.
        behind = sum(1 for i in range(self._focus_slot)
                     if not self._is_annotated(i))
        if behind:
            self._set_kbd_hint(
                f"End of page — {behind} unlabelled crop(s) above. "
                "Press Enter for the next batch."
            )
        else:
            self._set_kbd_hint(
                "End of page — all crops labelled. "
                "Press Enter to load the next batch."
            )
        return False

    def _kbd_move(self, token: str) -> bool:
        """Move focus one cell in ``token``'s direction, clamped to the grid."""
        count = self._slot_count()
        if count <= 0:
            self._set_kbd_hint("Nothing to annotate — open a source first.")
            return True
        cols = max(1, int(self._settings.grid_cols))
        slot = self._focus_slot
        target = slot
        if token == "left":
            if slot % cols > 0:
                target = slot - 1
        elif token == "right":
            if slot % cols < cols - 1 and slot + 1 < count:
                target = slot + 1
        elif token == "up":
            if slot - cols >= 0:
                target = slot - cols
        elif token == "down":
            if slot + cols < count:
                target = slot + cols
        if target != slot:
            self._set_focus_slot(target)
            self._set_kbd_hint("")
        return True

    def _kbd_step(self, delta: int) -> bool:
        """Step focus by ``delta`` in reading order without touching labels."""
        count = self._slot_count()
        if count <= 0:
            self._set_kbd_hint("Nothing to annotate — open a source first.")
            return True
        target = self._focus_slot + delta
        if target < 0:
            self._set_kbd_hint("Start of page.")
            return True
        if target >= count:
            self._set_kbd_hint(
                "End of page — press Enter to load the next batch.")
            return True
        self._set_focus_slot(target)
        self._set_kbd_hint("")
        return True

    def _kbd_undo(self) -> bool:
        """Walk back the most recent keyboard label assignment."""
        while self._undo_stack:
            slot, path, previous = self._undo_stack.pop()
            # Skip entries whose slot no longer holds the same crop; writing
            # them back would label the wrong image.
            if slot < len(self._page_paths) and self._page_paths[slot][0] == path:
                self._set_annotation(slot, previous)
                self._set_focus_slot(slot)
                self._set_kbd_hint("Undone.")
                return True
        self._set_kbd_hint("Nothing to undo.")
        return True

    def _kbd_commit_page(self) -> bool:
        """Save this page and load the next batch — same as the Next button."""
        before = self._offset
        self._on_next()          # flushes pending writes, then paginates
        # `_load_page` clears the hint on a successful page turn, so only the
        # "nothing more to load" case needs to say anything.
        if self._offset == before:
            self._set_kbd_hint("Saved — this is the last page.")
        return True

    # -- event plumbing -------------------------------------------------
    def keyPressEvent(self, event):
        """Route keystrokes through :meth:`handle_key` before Qt's default."""
        if self.handle_key(event.key(), event.text()):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):      # noqa: N802  (Qt naming)
        """Catch keys landing on the scroll area so arrows don't just scroll.

        Also catches the cursor leaving the grid as a whole. A tile's own
        Leave normally clears the hover, but the cursor can quit the grid
        without one (window hidden, cursor warped), and a hover nobody is
        pointing at any more must not survive.
        """
        try:
            etype = event.type()
        except Exception:
            return False       # not something we can reason about
        if etype == QEvent.KeyPress and self.handle_key(event.key(),
                                                         event.text()):
            return True
        if etype == QEvent.Leave:
            self._set_hover_slot(None)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    def _flush_pending(self):
        if not self._pending_updates or self._worker is None:
            return
        self._worker.submit(self._pending_updates)
        self._pending_updates.clear()

    def _refresh_status_label(self):
        w = self._worker
        if w is None:
            self._status_label.setText("Ready.")
            return
        parts = []
        if self._pending_updates:
            parts.append(f"{len(self._pending_updates)} unsaved change(s)")
        if w.busy:
            parts.append("saving…")
        elif w.pending_batches > 0:
            parts.append(f"{w.pending_batches} batch queued")
        if w.last_save_ts is not None and not parts:
            parts.append("saved")
        self._status_label.setText(" · ".join(parts) if parts else "Ready.")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """Drain every native/Python worker before Qt destroys this screen."""
        self._closing = True
        self._resize_timer.stop()
        self._pending_page_load = None
        self._flush_pending()
        if self._worker:
            self._worker.stop(wait=True)
            self._worker = None
        self._page_gen += 1   # invalidate any in-flight results
        worker = self._page_worker
        if worker is not None:
            # Do not use a timeout here. Cellpose/PyTorch can remain in native
            # inference longer than four seconds; letting QWidget destruction
            # continue in that window is the intermittent SIGSEGV/abort.
            worker.requestInterruption()
            worker.wait()
            self._page_worker = None
            try:
                worker.done.disconnect(self._on_page_loaded)
                worker.finished.disconnect(self._on_page_worker_finished)
            except (RuntimeError, TypeError):
                pass
            worker.deleteLater()
        try:
            self._console.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
