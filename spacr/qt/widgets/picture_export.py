"""Save a rendered picture as a PNG or a one-page PDF.

Exports use the supplied ``QImage`` or ``QPixmap`` rather than capturing
the surrounding widget. PNG preserves the image's pixel dimensions and
records the selected resolution as dots-per-metre metadata. PDF embeds
the raster image on a page; it does not convert pixels into vector artwork.

PDF page dimensions are computed from the image's pixel dimensions and
export DPI, with a minimum of one millimetre for each dimension. The image
fills the page with zero margins. This sets the printed size without using
microscope pixel calibration.

Default resolution and filename suffix come from
:func:`spacr.qt.preferences.get_figure_png_dpi` and
:func:`spacr.qt.preferences.get_figure_format`. The right-click menu's
explicit PNG or PDF choice determines the final export format.
"""
from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Callable, Optional

from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPageLayout, QPageSize, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QGraphicsItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QPlainTextEdit,
                               QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..i18n import tr
from .toggle import Toggle

LOG = logging.getLogger(__name__)

#: Millimetres per inch. Named because the conversion appears three times
#: and a wrong one is a page that is silently the wrong size.
MM_PER_INCH = 25.4

#: What the save dialog offers, in the order it offers it. PNG first: it is
#: what a picture of pixels usually wants to be, and the PDF is there for
#: the person putting it in a figure.
FILTERS = "PNG image (*.png);;PDF document (*.pdf)"

#: Fallback resolution when the preference cannot be read. The same 300 the
#: Figures preference defaults to.
FALLBACK_DPI = 300


def picture_dpi() -> int:
    """The resolution a saved picture is written at.

    :returns: the Figures preference, or :data:`FALLBACK_DPI`.
    """
    try:
        from ..preferences import get_figure_png_dpi
        dpi = int(get_figure_png_dpi())
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not read the figure DPI", exc_info=True)
        return FALLBACK_DPI
    return dpi if dpi > 0 else FALLBACK_DPI


def preferred_suffix() -> str:
    """``".pdf"`` or ``".png"``, following the Figures format preference."""
    try:
        from ..preferences import get_figure_format
        chosen = str(get_figure_format() or "").strip().lower()
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not read the figure format", exc_info=True)
        return ".png"
    return ".pdf" if chosen == "pdf" else ".png"


def as_image(picture) -> Optional[QImage]:
    """Whatever was handed over, as a :class:`QImage`, or ``None``.

    A caller may hold a ``QPixmap`` (what a view shows), a ``QImage`` (what
    a renderer produced) or nothing at all yet.

    :param picture: a ``QPixmap``, a ``QImage`` or ``None``; a null image or
        any other type also gives ``None``.
    """
    if picture is None:
        return None
    if isinstance(picture, QImage):
        return picture if not picture.isNull() else None
    if isinstance(picture, QPixmap):
        if picture.isNull():
            return None
        return picture.toImage()
    return None


def save_picture(picture, path, dpi: Optional[int] = None) -> bool:
    """Write ``picture`` to ``path``, as a PNG or a PDF by its suffix.

    :param picture: a ``QPixmap`` or ``QImage``.
    :param path: where to write. ``.pdf`` writes a page; anything else
        writes a PNG.
    :param dpi: resolution; :func:`picture_dpi` when omitted.
    :returns: whether a file was written.
    """
    image = as_image(picture)
    if image is None:
        return False
    resolution = int(dpi or picture_dpi())
    if resolution <= 0:
        resolution = FALLBACK_DPI
    path = str(path)
    fmt = "pdf" if path.lower().endswith(".pdf") else "png"
    report = None
    try:
        flag = os.environ.get("SPACR_FIGURE_INTEGRITY", "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            enabled = True
        elif flag in ("0", "false", "no", "off"):
            enabled = False
        else:
            from ..preferences import _get_figure_integrity
            enabled = bool(_get_figure_integrity())
        if enabled:
            import numpy as np
            from matplotlib.figure import Figure

            from ...plot import _PNG_PROVENANCE_KEY, _integrity_report

            rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(bytes(rgba.constBits()), dtype=np.uint8)
            pixels = pixels.reshape(rgba.height(), rgba.bytesPerLine())
            pixels = pixels[:, :rgba.width() * 4].reshape(
                rgba.height(), rgba.width(), 4).copy()
            figure = Figure(figsize=(rgba.width() / resolution,
                                     rgba.height() / resolution),
                            dpi=resolution)
            axes = figure.add_axes([0, 0, 1, 1])
            axes.imshow(pixels, interpolation="nearest")
            axes.set_axis_off()
            report = _integrity_report(figure, fmt=fmt, dpi=resolution,
                                       destination=path)
            if report is not None:
                report["panels"][0]["capture"] = "rendered QImage"
                report["panels"][0]["reproducible"] = False
                if fmt == "png":
                    import json
                    image = QImage(image)
                    image.setText(_PNG_PROVENANCE_KEY, json.dumps(report))
    except Exception:
        LOG.debug("could not check the rendered picture", exc_info=True)
        report = None
    written = (_save_pdf(image, path, resolution) if fmt == "pdf"
               else _save_png(image, path, resolution))
    if written and report is not None:
        try:
            from ...plot import _finish_integrity
            _finish_integrity(report, path)
        except Exception:
            LOG.debug("could not save picture provenance", exc_info=True)
    return written


def _save_png(image: QImage, path: str, dpi: int) -> bool:
    """Write PNG pixels with horizontal and vertical resolution metadata.

    A copy receives the requested DPI converted to integer dots per metre.
    If setting the metadata fails, writing the pixel image is still attempted.
    """
    stamped = QImage(image)
    try:
        per_metre = int(round(dpi / MM_PER_INCH * 1000.0))
        stamped.setDotsPerMeterX(per_metre)
        stamped.setDotsPerMeterY(per_metre)
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not stamp the resolution", exc_info=True)
    try:
        return bool(stamped.save(path, "PNG"))
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not write %s", path, exc_info=True)
        return False


def _save_pdf(image: QImage, path: str, dpi: int) -> bool:
    """Draw a raster image across a one-page PDF at the selected DPI.

    Width and height are computed from pixels divided by DPI, converted to
    millimetres, and independently clamped to at least one millimetre. The
    image is drawn over the resulting page with zero margins.
    """
    from PySide6.QtGui import QPdfWriter

    width_mm = max(1.0, image.width() / float(dpi) * MM_PER_INCH)
    height_mm = max(1.0, image.height() / float(dpi) * MM_PER_INCH)
    painter = None
    try:
        writer = QPdfWriter(path)
        writer.setResolution(int(dpi))
        writer.setPageSize(QPageSize(QSizeF(width_mm, height_mm),
                                     QPageSize.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
        painter = QPainter(writer)
        target = QRectF(0, 0, writer.width(), writer.height())
        painter.drawImage(target, image, QRectF(image.rect()))
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not write %s", path, exc_info=True)
        return False
    finally:
        if painter is not None:
            try:
                painter.end()
            except Exception:                                # noqa: BLE001
                LOG.debug("the PDF painter would not close", exc_info=True)
    return os.path.isfile(path)


def suggested_name(stem: str) -> str:
    """A file name for ``stem``, with the preferred suffix on it.

    :param stem: the base name; whitespace runs become underscores, and an
        empty result (or ``None``) becomes ``picture``.
    """
    cleaned = "_".join(str(stem or "picture").split()).strip("_")
    return f"{cleaned or 'picture'}{preferred_suffix()}"


def ask_where_to_save(parent, stem: str) -> str:
    """Ask for an output path; return ``""`` when the dialog is cancelled.

    If the chosen name has no suffix, append ``.pdf`` or ``.png`` according
    to the selected file filter. An existing suffix is retained here;
    :func:`save_as` subsequently enforces the format chosen in the menu.

    :param parent: the widget the save dialog is parented to.
    :param stem: the file name offered, without a suffix; cleaned by
        :func:`suggested_name`.
    """
    from PySide6.QtWidgets import QFileDialog

    start = suggested_name(stem)
    chosen, selected = QFileDialog.getSaveFileName(
        parent, "Save picture", start, FILTERS)
    if not chosen:
        return ""
    if not os.path.splitext(chosen)[1]:
        chosen += ".pdf" if "pdf" in str(selected).lower() else ".png"
    return chosen


def build_menu(parent, enabled: bool):
    """The right-click menu: save as PNG, save as PDF.

    Each action carries its suffix as its ``data``, so the caller reads the
    choice off the action rather than comparing it against two references
    it had to keep.

    :param parent: the widget that owns the menu.
    :param enabled: ``False`` when there is no picture yet. The actions are
        SHOWN AND GREYED rather than left out, and a line underneath says
        why: a menu that is empty on one field and full on the next reads
        as a bug, not as a state.
    """
    from PySide6.QtWidgets import QMenu

    menu = QMenu(parent)
    png = menu.addAction("Save as PNG…")
    png.setData(".png")
    pdf = menu.addAction("Save as PDF…")
    pdf.setData(".pdf")
    for action in (png, pdf):
        action.setEnabled(bool(enabled))
    if not enabled:
        menu.addSeparator()
        menu.addAction("There is no picture here yet").setEnabled(False)
    return menu


def choose_format(view, point, enabled: bool) -> str:
    """Pop the menu at ``point`` and return the chosen suffix, or ``""``.

    THE MODAL CALL IS ALONE IN HERE so everything around it can be driven
    by a test. ``QMenu.exec`` spins an event loop of its own, and a test
    that reached it would hang rather than fail; substituting this one
    function is how the rest of the gesture is checked at all.

    :param view: the widget the menu belongs to; ``point`` is mapped to
        global coordinates through it.
    :param point: where the right-click happened, in ``view`` coordinates.
    :param enabled: ``False`` greys out the save actions, as in
        :func:`build_menu`.
    """
    menu = build_menu(view, enabled)
    chosen = menu.exec(view.mapToGlobal(point))
    if chosen is None:
        return ""
    return str(chosen.data() or "")


def save_as(view, picture, stem: str, want: str) -> str:
    """Ask for a path and write ``picture`` there. ``""`` if nothing was.

    :param view: the widget the save dialog and any warning are parented to.
    :param picture: the ``QPixmap`` or ``QImage`` to write.
    :param stem: the file name offered, without a suffix.
    :param want: the suffix the user picked in the menu. It WINS over what
        the file dialog came back with, so "save as PDF" followed by a name
        ending in ``.png`` still writes a PDF rather than quietly changing
        format because of a typed extension.
    :returns: the path written, or ``""``.
    """
    from PySide6.QtWidgets import QMessageBox

    path = ask_where_to_save(view, stem)
    if not path:
        return ""
    if os.path.splitext(path)[1].lower() != want:
        path = os.path.splitext(path)[0] + want
    if save_picture(picture, path):
        return path
    QMessageBox.warning(view, "Not saved",
                        f"{os.path.basename(path)} could not be written.")
    return ""


def install_picture_save(view, picture: Callable[[], object],
                         stem: str = "picture",
                         unless: Optional[Callable[[], bool]] = None) -> bool:
    """Give ``view`` a right-click menu that saves what it is showing.

    :param view: the widget the user right-clicks.
    :param picture: called before the context menu opens; returns the
        ``QPixmap`` or ``QImage`` to write. The returned image is used after
        the format and path dialogs; the callback is not called again
        immediately before writing.
    :param stem: the file name offered, without a suffix. A callable is
        called after the format is selected and before the path dialog opens.
    :param unless: asked first; ``True`` means the right button belongs to
        something else at this moment and no menu is shown. The live
        preview's ruler is cleared with a right-click while it is active,
        and a menu appearing over that would take a tool away to add a file
        dialog nobody asked for.
    :returns: whether the menu was installed.
    """
    def _menu(point) -> str:
        """Offer the save menu at ``point``; the path written, or ``""``.

        :param point: where the right-click landed, in ``view`` coordinates.
        """
        if unless is not None:
            try:
                if unless():
                    return ""
            except Exception:                                # noqa: BLE001
                LOG.debug("a picture menu guard raised", exc_info=True)
        image = as_image(picture())
        want = choose_format(view, point, image is not None)
        if not want or image is None:
            return ""
        name = stem() if callable(stem) else stem
        return save_as(view, image, name, want)

    try:
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(_menu)
        view._spacr_picture_menu = _menu
        return True
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not install the picture menu", exc_info=True)
        return False


_CANVAS_PAGE_SIZES = {"A4": (595.2756, 841.8898), "Letter": (612.0, 792.0)}
_CANVAS_LAYOUT_SCHEMA = "spacr.canvas_layout/1"
_CANVAS_MIN_SIDE = 6.0
_CANVAS_UNDO_DEPTH = 100
_CANVAS_TEXT_PAD = 2.0
_CANVAS_LINE_SPACING = 1.2
_CANVAS_ALIGNMENTS = ("left", "center", "right")
_CANVAS_SCALE_STEPS = (1.0, 2.0, 5.0)


class _CanvasLayout:
    """Pages of text and image boxes on a snap grid, in PDF points from the top left.

    Boxes are plain dictionaries so a layout saves to JSON as it is; a
    page's list order is its z-order, the last box drawn on top. Every
    change made through a method can be undone; a change made by writing to
    a box directly cannot.

    :param page_size: ``"A4"`` or ``"Letter"``.
    :param orientation: ``"portrait"`` or ``"landscape"``.
    :param grid: grid spacing in points.
    :param show_grid: whether the grid is drawn.
    :param snap: whether positions and sizes snap to the grid.
    """

    def __init__(self, page_size="A4", orientation="portrait", grid=18.0, show_grid=True, snap=True):
        """Start with one empty page."""
        self.page_size = page_size if page_size in _CANVAS_PAGE_SIZES else "A4"
        self.orientation = "landscape" if orientation == "landscape" else "portrait"
        self.grid = max(1.0, float(grid))
        self.show_grid = bool(show_grid)
        self.snap = bool(snap)
        self.pages = [[]]
        self._undo = []
        self._redo = []
        self._next_id = 1

    @property
    def page_points(self):
        """The page's ``(width, height)`` in points, in its orientation."""
        width, height = _CANVAS_PAGE_SIZES[self.page_size]
        return (height, width) if self.orientation == "landscape" else (width, height)

    def snapped(self, value):
        """``value`` on the nearest grid line when snapping is on."""
        if not self.snap:
            return float(value)
        return round(float(value) / self.grid) * self.grid

    def checkpoint(self):
        """Remember the layout as it is, for :meth:`undo`; clears the redo history."""
        self._undo.append(self._state())
        del self._undo[:-_CANVAS_UNDO_DEPTH]
        self._redo.clear()

    def can_undo(self):
        """Whether there is a change to undo."""
        return bool(self._undo)

    def can_redo(self):
        """Whether there is an undone change to redo."""
        return bool(self._redo)

    def undo(self):
        """Go back one change; returns whether there was one."""
        if not self._undo:
            return False
        self._redo.append(self._state())
        self._restore(self._undo.pop())
        return True

    def redo(self):
        """Repeat the last undone change; returns whether there was one."""
        if not self._redo:
            return False
        self._undo.append(self._state())
        self._restore(self._redo.pop())
        return True

    def _state(self):
        """The undoable state: settings and pages, deep-copied."""
        import copy

        return copy.deepcopy(self.to_dict())

    def _restore(self, state):
        """Put back a state from :meth:`_state`, keeping the undo history."""
        undo, redo = self._undo, self._redo
        self._apply(state)
        self._undo, self._redo = undo, redo

    def _apply(self, data):
        """Load settings and pages from a :meth:`to_dict` mapping."""
        self.page_size = data.get("page_size") if data.get("page_size") in _CANVAS_PAGE_SIZES else "A4"
        self.orientation = "landscape" if data.get("orientation") == "landscape" else "portrait"
        self.grid = max(1.0, float(data.get("grid", 18.0)))
        self.show_grid = bool(data.get("show_grid", True))
        self.snap = bool(data.get("snap", True))
        self.pages = [[_canvas_box(**box) for box in page] for page in (data.get("pages") or [[]])] or [[]]
        self._next_id = 1 + max((int(box["id"]) for page in self.pages for box in page), default=0)

    def set_page(self, page_size=None, orientation=None):
        """Change the page size or orientation; boxes keep their place, kept on the page."""
        self.checkpoint()
        if page_size in _CANVAS_PAGE_SIZES:
            self.page_size = page_size
        if orientation in ("portrait", "landscape"):
            self.orientation = orientation
        for page in self.pages:
            for box in page:
                self._keep_on_page(box)

    def set_grid(self, grid=None, show_grid=None, snap=None):
        """Change the grid spacing, whether it shows and whether boxes snap to it."""
        if grid is not None:
            self.grid = max(1.0, float(grid))
        if show_grid is not None:
            self.show_grid = bool(show_grid)
        if snap is not None:
            self.snap = bool(snap)

    def add_page(self, at=None):
        """Insert an empty page at ``at`` (the end by default); returns its index."""
        self.checkpoint()
        index = len(self.pages) if at is None else max(0, min(int(at), len(self.pages)))
        self.pages.insert(index, [])
        return index

    def remove_page(self, index):
        """Remove page ``index``; the last page is emptied instead of removed."""
        self.checkpoint()
        if len(self.pages) == 1:
            self.pages[0] = []
            return
        del self.pages[index]

    def move_page(self, index, target):
        """Move page ``index`` to position ``target``."""
        target = max(0, min(int(target), len(self.pages) - 1))
        if target == index:
            return
        self.checkpoint()
        self.pages.insert(target, self.pages.pop(index))

    def add_box(self, page, kind, x, y, w, h, **properties):
        """Add a text or image box to page ``page`` on top of the others; returns it.

        :param kind: ``"text"`` or ``"image"``.
        :param x: left edge in points; snapped.
        :param y: top edge in points; snapped.
        :param w: width in points; snapped.
        :param h: height in points; snapped, or derived from the width for
            an image that keeps its aspect ratio.
        :param properties: any other box keys (see :func:`_canvas_box`).
        """
        self.checkpoint()
        box = _canvas_box(id=self._next_id, kind=kind, x=x, y=y, w=w, h=h, **properties)
        self._next_id += 1
        self._place(box, self.snapped(x), self.snapped(y), w, h)
        self.pages[page].append(box)
        return box

    def find(self, box_id):
        """``(page index, box)`` for ``box_id``, or ``(None, None)``."""
        for index, page in enumerate(self.pages):
            for box in page:
                if box["id"] == box_id:
                    return index, box
        return None, None

    def move_boxes(self, ids, dx, dy):
        """Move boxes by ``(dx, dy)`` points, snapping each top-left corner and keeping it on the page."""
        self.checkpoint()
        for box_id in ids:
            box = self.find(box_id)[1]
            if box is not None:
                box["x"], box["y"] = self.snapped(box["x"] + dx), self.snapped(box["y"] + dy)
                self._keep_on_page(box)

    def set_rect(self, box_id, x, y, w, h):
        """Place a box at ``(x, y)`` with size ``(w, h)``, snapped, keeping a locked image's aspect."""
        box = self.find(box_id)[1]
        if box is None:
            return
        self.checkpoint()
        self._place(box, self.snapped(x), self.snapped(y), w, h)

    def resize_box(self, box_id, w, h):
        """Resize a box from its top-left corner, snapping its right and bottom edges."""
        box = self.find(box_id)[1]
        if box is not None:
            self.set_rect(box_id, box["x"], box["y"], w, h)

    def _place(self, box, x, y, w, h):
        """Set a box's rectangle: edges snapped, a locked image's height from its width."""
        right = self.snapped(x + float(w))
        width = max(_CANVAS_MIN_SIDE, right - x)
        if box["kind"] == "image" and box.get("lock_aspect", True) and box.get("aspect"):
            height = width / float(box["aspect"])
        else:
            height = max(_CANVAS_MIN_SIDE, self.snapped(y + float(h)) - y)
        box.update(x=float(x), y=float(y), w=float(width), h=float(height))
        self._keep_on_page(box)

    def _keep_on_page(self, box):
        """Shrink a box to fit the page and move it back onto it."""
        width, height = self.page_points
        if box["w"] > width or box["h"] > height:
            scale = min(width / box["w"], height / box["h"])
            if box["kind"] == "image" and box.get("lock_aspect", True) and box.get("aspect"):
                box["w"], box["h"] = box["w"] * scale, box["h"] * scale
            else:
                box["w"], box["h"] = min(box["w"], width), min(box["h"], height)
        box["x"] = min(max(0.0, box["x"]), width - box["w"])
        box["y"] = min(max(0.0, box["y"]), height - box["h"])

    def update_box(self, box_id, **properties):
        """Change a box's properties (text, font, alignment, aspect lock, scale bar...)."""
        box = self.find(box_id)[1]
        if box is None:
            return
        self.checkpoint()
        box.update(properties)
        if properties.get("lock_aspect") and box["kind"] == "image" and box.get("aspect"):
            box["h"] = box["w"] / float(box["aspect"])
            self._keep_on_page(box)

    def delete_boxes(self, ids):
        """Remove boxes from their pages."""
        self.checkpoint()
        wanted = set(ids)
        self.pages = [[box for box in page if box["id"] not in wanted] for page in self.pages]

    def restack(self, ids, where):
        """Change the z-order of boxes on their page.

        :param where: ``"front"``, ``"back"``, ``"forward"`` (one step up) or
            ``"backward"`` (one step down).
        """
        self.checkpoint()
        wanted = set(ids)
        for index, page in enumerate(self.pages):
            chosen = [box for box in page if box["id"] in wanted]
            if not chosen:
                continue
            others = [box for box in page if box["id"] not in wanted]
            if where == "front":
                self.pages[index] = others + chosen
            elif where == "back":
                self.pages[index] = chosen + others
            else:
                order = list(page)
                step = 1 if where == "forward" else -1
                for box in (reversed(chosen) if step > 0 else chosen):
                    position = order.index(box)
                    target = position + step
                    if 0 <= target < len(order) and order[target]["id"] not in wanted:
                        order[position], order[target] = order[target], order[position]
                self.pages[index] = order

    def to_dict(self):
        """The layout as a JSON-ready mapping."""
        return {"schema": _CANVAS_LAYOUT_SCHEMA, "page_size": self.page_size,
                "orientation": self.orientation, "grid": self.grid,
                "show_grid": self.show_grid, "snap": self.snap,
                "pages": [[dict(box) for box in page] for page in self.pages]}

    @classmethod
    def from_dict(cls, data):
        """A layout from :meth:`to_dict`'s mapping.

        :raises ValueError: when the mapping is not a canvas layout.
        """
        if not isinstance(data, dict) or data.get("schema") != _CANVAS_LAYOUT_SCHEMA:
            raise ValueError(tr("This file is not a spaCR canvas layout."))
        layout = cls()
        layout._apply(data)
        return layout

    def save(self, path):
        """Write the layout as JSON to ``path``, replacing it in one step; returns the path."""
        import json

        path = str(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        pending = path + ".partial"
        with open(pending, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1)
        os.replace(pending, path)
        return path

    @classmethod
    def load(cls, path):
        """Read a layout saved by :meth:`save`."""
        import json

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _canvas_box(id, kind, x, y, w, h, **properties):
    """One box's dictionary with every key it can carry.

    :param id: the box's number, unique within its layout.
    :param kind: ``"text"`` or ``"image"``.
    :param properties: ``text``, ``font_size`` (points), ``bold``,
        ``italic``, ``align`` (``left``/``center``/``right``) for text;
        ``source`` (``{"type": "file", "path": ...}`` or an Overview field),
        ``label``, ``lock_aspect``, ``aspect`` (width over height),
        ``native`` (pixel width and height), ``scale_bar``,
        ``pixel_size_um`` and ``fit`` (drawn whole inside the box, centred,
        without stretching) for images.
    """
    box = {"id": int(id), "kind": "image" if kind == "image" else "text",
           "x": float(x), "y": float(y), "w": float(w), "h": float(h),
           "text": "", "font_size": 10.0, "bold": False, "italic": False, "align": "left",
           "source": None, "label": "", "lock_aspect": True, "aspect": None, "native": None,
           "scale_bar": False, "pixel_size_um": None, "fit": False}
    box.update({key: value for key, value in properties.items() if key in box or key == "role"})
    if box["align"] not in _CANVAS_ALIGNMENTS:
        box["align"] = "left"
    return box


def _canvas_file_pixels(path, edge=None):
    """An image file's pixels as 8-bit RGB, at full size or fitted within ``edge``.

    Integer images wider than 8 bits and float images are stretched between
    their minimum and maximum; a single-channel image becomes grey.

    :param path: a PNG, JPEG, BMP or TIFF file.
    :param edge: longest side wanted, or ``None`` for every pixel.
    :raises ValueError: when the file holds no image.
    """
    import numpy as np

    path = str(path)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        pixels = np.asarray(tifffile.imread(path))
    else:
        from PIL import Image

        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGBA" if "A" in image.getbands() else "RGB"))
    pixels = np.squeeze(pixels)
    if pixels.ndim == 3 and pixels.shape[0] in (3, 4) and pixels.shape[-1] not in (3, 4):
        pixels = np.moveaxis(pixels, 0, -1)
    if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
        pixels = pixels[..., 0]
    if pixels.ndim not in (2, 3) or pixels.size == 0:
        raise ValueError(tr("{name} holds no image.", name=os.path.basename(path)))
    if pixels.dtype != np.uint8:
        values = pixels.astype(np.float64)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        pixels = np.clip((values - low) / ((high - low) or 1.0) * 255.0, 0, 255).astype(np.uint8)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if edge:
        longest = max(pixels.shape[:2])
        step = max(1, int(np.ceil(longest / float(edge))))
        pixels = pixels[::step, ::step]
    return np.ascontiguousarray(pixels)


def _canvas_scale_length(width_um):
    """A round scale-bar length near a quarter of an image ``width_um`` micrometres wide."""
    import math

    target = max(width_um, 1e-9) / 4.0
    power = 10.0 ** math.floor(math.log10(target))
    return max((step * power for step in _CANVAS_SCALE_STEPS if step * power <= target), default=power)


def _canvas_wrap(text, width, measure):
    """``text`` broken into lines no wider than ``width``, keeping its own line breaks.

    :param measure: returns a line's width in the same units as ``width``.
    :returns: the lines.
    """
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = word if not line else line + " " + word
            if line and measure(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def _canvas_text_width(figure, renderer, size, weight, style, text):
    """How wide ``text`` prints in points at ``size`` with this weight and style."""
    from matplotlib.text import Text

    if not text:
        return 0.0
    probe = Text(0, 0, text, fontsize=size, fontweight=weight, fontstyle=style, figure=figure)
    return probe.get_window_extent(renderer).width * 72.0 / figure.dpi


def _canvas_draw_text(figure, renderer, box, page_width, page_height):
    """Draw a text box as vector text, wrapped to the box's width."""
    weight = "bold" if box.get("bold") else "normal"
    style = "italic" if box.get("italic") else "normal"
    size = float(box.get("font_size") or 10.0)
    usable = max(1.0, box["w"] - 2 * _CANVAS_TEXT_PAD)
    lines = _canvas_wrap(box.get("text") or "", usable, functools.partial(
        _canvas_text_width, figure, renderer, size, weight, style))
    align = box.get("align") or "left"
    left = box["x"] + _CANVAS_TEXT_PAD
    x = {"left": left, "center": box["x"] + box["w"] / 2.0,
         "right": box["x"] + box["w"] - _CANVAS_TEXT_PAD}[align]
    figure.text(x / page_width, 1.0 - (box["y"] + _CANVAS_TEXT_PAD) / page_height, "\n".join(lines),
                fontsize=size, fontweight=weight, fontstyle=style, ha=align, va="top",
                multialignment=align, linespacing=_CANVAS_LINE_SPACING, color="black")


def _canvas_draw_scale_bar(axis, pixels, pixel_size_um):
    """Draw a vector scale bar and its length in the lower right of an image axis."""
    import numpy as np
    from matplotlib.patches import Rectangle

    rows, columns = pixels.shape[:2]
    length_um = _canvas_scale_length(columns * float(pixel_size_um))
    length = length_um / float(pixel_size_um)
    margin = .04 * columns
    thickness = max(1.0, .012 * rows)
    corner = pixels[int(rows * .8):, int(columns * .6):]
    bright = corner.size and float(np.mean(corner[..., :3])) > 127
    colour = "black" if bright else "white"
    left, top = columns - margin - length, rows - margin - thickness
    axis.add_patch(Rectangle((left, top), length, thickness, color=colour, linewidth=0))
    label = f"{length_um:g} µm"
    axis.text(left + length / 2.0, top - thickness, label, color=colour, fontsize=8,
              ha="center", va="bottom")
    return length_um


def _canvas_page_figure(layout, page_index, pixels_for, cancelled=None):
    """Draw one page of ``layout`` on a Matplotlib figure the page's size.

    :param pixels_for: called with an image box; returns its pixels at full
        resolution as an 8-bit array of rows by columns (by channels).
    :returns: ``(figure, images, records, errors)``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    width, height = layout.page_points
    figure = Figure(figsize=(width / 72.0, height / 72.0), dpi=72)
    figure.patch.set_facecolor("white")
    renderer = FigureCanvasAgg(figure).get_renderer()
    images, records, errors = [], [], []
    for box in layout.pages[page_index]:
        if cancelled is not None and cancelled.is_set():
            break
        if box["kind"] == "text":
            _canvas_draw_text(figure, renderer, box, width, height)
            records.append({"id": box["id"], "kind": "text", "text": box.get("text", "")})
            continue
        axis = figure.add_axes([box["x"] / width, 1.0 - (box["y"] + box["h"]) / height,
                                box["w"] / width, box["h"] / height])
        axis.set_axis_off()
        try:
            pixels = pixels_for(box)
        except Exception as exc:
            axis.text(.5, .5, str(exc), ha="center", va="center", wrap=True, fontsize=8)
            errors.append((box.get("label") or "", str(exc)))
            continue
        aspect = "equal" if box.get("fit") else "auto"
        if pixels.ndim == 2:
            axis.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="none", aspect=aspect)
        else:
            axis.imshow(pixels, interpolation="none", aspect=aspect)
        bar = None
        if box.get("scale_bar") and box.get("pixel_size_um"):
            bar = _canvas_draw_scale_bar(axis, pixels, box["pixel_size_um"])
        images.append(pixels)
        records.append({"id": box["id"], "kind": "image", "label": box.get("label", ""),
                        "source": box.get("source"), "pixels": list(pixels.shape[:2]),
                        "rect_points": [box["x"], box["y"], box["w"], box["h"]],
                        "scale_bar_um": bar})
    from ...figures.bundle import _register_figure_data

    _register_figure_data(figure, images, kind="montage", boxes=records, page=page_index + 1,
                          page_size=layout.page_size, orientation=layout.orientation)
    return figure, images, records, errors


def _canvas_pdf(layout, path, pixels_for, cancelled=None):
    """Write every page of ``layout`` to one PDF through spaCR's figure writer.

    Text is vector; each image is embedded with all of the pixels
    ``pixels_for`` returns, never resampled to the screen. Each page goes
    through :func:`spacr.plot.save_figure` and the pages are joined with
    pypdf, so the house style, the print colours and the figure-integrity
    check reach every page.

    :param pixels_for: called with an image box; returns its full pixels.
    :param cancelled: an optional ``threading.Event``; once set no file is written.
    :returns: ``(path written or None when cancelled, errors)``.
    """
    import shutil
    import tempfile

    from pypdf import PdfWriter

    from ...plot import save_figure

    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    errors = []
    staging = tempfile.mkdtemp(prefix=".canvas_", dir=directory)
    try:
        pages = []
        for index in range(len(layout.pages)):
            if cancelled is not None and cancelled.is_set():
                return None, errors
            figure, _images, _records, page_errors = _canvas_page_figure(layout, index, pixels_for, cancelled)
            pages.append(save_figure(figure, os.path.join(staging, f"page_{index + 1:04d}.pdf"),
                                     fmt="pdf", close=True))
            errors.extend(page_errors)
        if cancelled is not None and cancelled.is_set():
            return None, errors
        writer = PdfWriter()
        for page in pages:
            writer.append(str(page))
        pending = os.path.join(staging, "joined.pdf")
        with open(pending, "wb") as handle:
            writer.write(handle)
        os.replace(pending, path)
        for index, page in enumerate(pages):
            sidecar = str(page) + ".provenance.json"
            if os.path.isfile(sidecar):
                shutil.move(sidecar, f"{path}.page{index + 1}.provenance.json")
        return path, errors
    finally:
        shutil.rmtree(staging, ignore_errors=True)


_CANVAS_MIME = "application/x-spacr-canvas-images"
_CANVAS_PREVIEW_EDGE = 640
_CANVAS_HANDLE = 8.0
_CANVAS_GRID_INK = "#c9d5e3"
_CANVAS_PAPER = "#ffffff"
_CANVAS_FILE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _canvas_mime(images):
    """Drag data carrying images for the canvas composer.

    :param images: dictionaries with ``source`` and optionally ``label``,
        ``aspect``, ``native`` and ``pixel_size_um``.
    :returns: a ``QMimeData``.
    """
    import json

    from PySide6.QtCore import QMimeData

    data = QMimeData()
    data.setData(_CANVAS_MIME, json.dumps(list(images)).encode("utf-8"))
    data.setText("\n".join(str(image.get("label") or "") for image in images))
    return data


def _canvas_dropped_images(mime):
    """The images a drop carries: canvas drag data, or image files from disk.

    :returns: dictionaries ready for :meth:`_CanvasLayout.add_box`.
    """
    import json

    if mime.hasFormat(_CANVAS_MIME):
        try:
            images = json.loads(bytes(mime.data(_CANVAS_MIME)).decode("utf-8"))
        except ValueError:
            return []
        return [dict(image) for image in images if isinstance(image, dict) and image.get("source")]
    images = []
    for url in mime.urls() if mime.hasUrls() else ():
        path = url.toLocalFile()
        if path and path.lower().endswith(_CANVAS_FILE_SUFFIXES) and os.path.isfile(path):
            images.append({"source": {"type": "file", "path": path}, "label": os.path.basename(path)})
    return images


def _canvas_box_pixels(box, edge, field_pixels):
    """An image box's pixels: a file read here, anything else from ``field_pixels``.

    :param edge: the longest side wanted, or ``None`` for every pixel.
    :param field_pixels: called with ``(box, edge)`` for sources that are
        not files, or ``None`` when there are none.
    :raises ValueError: when nothing can supply the pixels.
    """
    source = box.get("source") or {}
    if source.get("type") == "file":
        return _canvas_file_pixels(source.get("path", ""), edge)
    if field_pixels is None:
        raise ValueError(tr("The image source is not loaded: {label}", label=box.get("label") or "?"))
    return field_pixels(box, edge)


def _canvas_qimage(pixels):
    """An 8-bit RGB or RGBA array as a ``QImage`` that owns its pixels."""
    import numpy as np

    pixels = np.ascontiguousarray(pixels)
    rows, columns = pixels.shape[:2]
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        image = QImage(pixels.data, columns, rows, columns * 4, QImage.Format.Format_RGBA8888)
    else:
        if pixels.ndim == 2:
            pixels = np.ascontiguousarray(np.repeat(pixels[..., None], 3, axis=2))
        image = QImage(pixels.data, columns, rows, columns * 3, QImage.Format.Format_RGB888)
    return image.copy()


class _CanvasBoxItem(QGraphicsRectItem):
    """One box drawn on the composer's page: text, or an image preview."""

    def __init__(self, composer, box):
        """Place the item at the box's rectangle, in points."""
        super().__init__(0.0, 0.0, box["w"], box["h"])
        self._composer = composer
        self.box_id = box["id"]
        self._resizing = False
        self.setPos(box["x"], box["y"])
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.setToolTip(tr("Drag to move; drag the lower-right corner to resize."))

    def box(self):
        """The model box this item shows."""
        return self._composer.canvas.find(self.box_id)[1]

    def _in_handle(self, point):
        """Whether ``point`` (item coordinates) is on the resize corner."""
        rect = self.rect()
        return (rect.right() - _CANVAS_HANDLE * 1.5 <= point.x() <= rect.right() + 1
                and rect.bottom() - _CANVAS_HANDLE * 1.5 <= point.y() <= rect.bottom() + 1)

    def hoverMoveEvent(self, event):
        """Show the resize cursor over the corner."""
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if self._in_handle(event.pos())
                       else Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        """Start a resize on the corner, otherwise a move."""
        if event.button() == Qt.MouseButton.LeftButton and self._in_handle(event.pos()):
            self._resizing = True
            if not self.isSelected():
                self.scene().clearSelection()
                self.setSelected(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Follow the pointer while resizing."""
        if self._resizing:
            point = event.pos()
            self.prepareGeometryChange()
            self.setRect(0.0, 0.0, max(_CANVAS_MIN_SIDE, point.x()), max(_CANVAS_MIN_SIDE, point.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Hand the finished move or resize to the composer, which snaps it."""
        if self._resizing:
            self._resizing = False
            rect = self.rect()
            self._composer._resized(self.box_id, rect.width(), rect.height())
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._composer._moved(self.box_id)

    def paint(self, painter, option, widget=None):
        """Draw the text or image, the frame and, when selected, the resize corner."""
        from PySide6.QtGui import QBrush, QColor, QFont, QPen

        box = self.box()
        if box is None:
            return
        rect = self.rect()
        if box["kind"] == "image":
            image = self._composer._preview_of(box)
            if image is not None:
                target = rect
                if box.get("fit") and image.width() and image.height():
                    scale = min(rect.width() / image.width(), rect.height() / image.height())
                    size = QSizeF(image.width() * scale, image.height() * scale)
                    target = QRectF(rect.center().x() - size.width() / 2, rect.center().y() - size.height() / 2,
                                    size.width(), size.height())
                painter.drawImage(target, image)
                if box.get("scale_bar") and box.get("pixel_size_um") and box.get("native"):
                    self._paint_scale_bar(painter, box, target)
            else:
                painter.fillRect(rect, QColor("#e6ebf2"))
                painter.setPen(QColor("#4b5460"))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                                 box.get("label") or tr("Image"))
        else:
            font = QFont(painter.font())
            font.setPixelSize(max(1, int(round(float(box.get("font_size") or 10.0)))))
            font.setBold(bool(box.get("bold")))
            font.setItalic(bool(box.get("italic")))
            painter.setFont(font)
            painter.setPen(QColor("#000000"))
            align = {"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
                     "right": Qt.AlignmentFlag.AlignRight}[box.get("align") or "left"]
            inner = rect.adjusted(_CANVAS_TEXT_PAD, _CANVAS_TEXT_PAD, -_CANVAS_TEXT_PAD, -_CANVAS_TEXT_PAD)
            text = box.get("text") or ""
            if not text:
                painter.setPen(QColor("#8a94a3"))
                text = tr("Text box")
            painter.drawText(inner, align | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, text)
        accent = QColor(self._composer._accent())
        if self.isSelected():
            pen = QPen(accent, 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            painter.setBrush(QBrush(accent))
            painter.drawRect(rect.right() - _CANVAS_HANDLE, rect.bottom() - _CANVAS_HANDLE,
                             _CANVAS_HANDLE, _CANVAS_HANDLE)
        elif box["kind"] == "text":
            pen = QPen(QColor("#a9b6c8"), .6, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

    def _paint_scale_bar(self, painter, box, rect):
        """Preview the scale bar where the PDF draws it."""
        from PySide6.QtGui import QColor

        columns, rows = box["native"]
        length_um = _canvas_scale_length(columns * float(box["pixel_size_um"]))
        scale = rect.width() / float(columns)
        length = length_um / float(box["pixel_size_um"]) * scale
        margin = .04 * columns * scale
        thickness = max(1.0, .012 * rows * (rect.height() / float(rows)))
        left = rect.right() - margin - length
        top = rect.bottom() - margin - thickness
        painter.fillRect(left, top, length, thickness, QColor("#ffffff"))


class _CanvasScene(QGraphicsScene):
    """The page, its grid and the boxes, in points."""

    def __init__(self, composer):
        """An empty scene owned by ``composer``."""
        super().__init__(composer)
        self._composer = composer

    def drawBackground(self, painter, rect):
        """Paint the desk, the paper and, when shown, the snap grid."""
        from PySide6.QtCore import QLineF
        from PySide6.QtGui import QColor, QPen

        super().drawBackground(painter, rect)
        layout = self._composer.canvas
        width, height = layout.page_points
        painter.fillRect(QRectF(3, 3, width, height), QColor(0, 0, 0, 40))
        painter.fillRect(QRectF(0, 0, width, height), QColor(_CANVAS_PAPER))
        if not layout.show_grid:
            return
        painter.setPen(QPen(QColor(_CANVAS_GRID_INK), 0))
        lines = []
        step = layout.grid
        x = step
        while x < width:
            lines.append(QLineF(x, 0, x, height))
            x += step
        y = step
        while y < height:
            lines.append(QLineF(0, y, width, y))
            y += step
        painter.drawLines(lines)

    def mousePressEvent(self, event):
        """Start drawing a text box when the text tool is chosen."""
        if (event.button() == Qt.MouseButton.LeftButton and self._composer._tool == "text"
                and self.itemAt(event.scenePos(), self.views()[0].transform()) is None):
            self._composer._begin_text(event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Stretch the text box being drawn."""
        if self._composer._drawing is not None:
            self._composer._stretch_text(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish the text box being drawn."""
        if self._composer._drawing is not None:
            self._composer._finish_text(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event):
        """Accept Overview images and image files."""
        if _canvas_dropped_images(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):
        """Keep accepting while the drag is over the page."""
        if _canvas_dropped_images(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        """Add the dropped images where they were dropped."""
        images = _canvas_dropped_images(event.mimeData())
        if images:
            self._composer.add_images(images, event.scenePos())
            event.acceptProposedAction()
            return
        event.ignore()


class _CanvasView(QGraphicsView):
    """The composer's zoomable view of the page."""

    def __init__(self, scene, parent=None):
        """Show ``scene`` fitted to the view, with rubber-band selection."""
        super().__init__(scene, parent)
        from PySide6.QtGui import QPainter

        self.setObjectName("CanvasComposerView")
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAcceptDrops(True)
        self.setToolTip(tr("Drag boxes to move them, drag around boxes to select several, and drop "
                           "Overview images or image files here."))

    def fit(self):
        """Fit the whole page in the view."""
        width, height = self.scene()._composer.canvas.page_points
        self.fitInView(QRectF(-8, -8, width + 16, height + 16), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        """Keep the page fitted as the view resizes."""
        super().resizeEvent(event)
        self.fit()


class _CanvasComposer(QDialog):
    """Compose PDF pages from text boxes and images on a snap grid.

    :param layout: a :class:`_CanvasLayout` to edit; a new one by default.
    :param field_pixels: called with ``(box, edge)`` for image sources that
        are not files (Overview fields); ``edge`` ``None`` means every pixel.
    :param folder: the results folder layouts and PDFs are offered in.
    :param parent: the owning widget.
    :param actions: extra toolbar buttons, ``(caption, tooltip, callback)``;
        each callback is called with the composer.
    """

    changed = Signal()

    def __init__(self, layout=None, field_pixels=None, folder="", parent=None, actions=()):
        """Build the toolbar, the pages, the page view and the box settings."""
        super().__init__(parent)
        from ..job_runner import JobRunner

        self.setObjectName("CanvasComposer")
        self.setWindowTitle(tr("Custom export"))
        self.canvas = layout if layout is not None else _CanvasLayout()
        self._field_pixels = field_pixels
        self._folder = str(folder or "")
        self._page = 0
        self._tool = "select"
        self._drawing = None
        self._rubber = None
        self._previews = {}
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self._rebuild)
        self._requested = set()
        self._editing_text = None
        self._syncing = False
        self._items = {}
        self._preview_jobs = JobRunner(self, app_key="canvas_composer", user_visible=False)
        self._export_jobs = JobRunner(self, app_key="canvas_composer")
        self._export_jobs.job_failed.connect(self._export_failed)
        self._cancelled = threading.Event()
        self.last_export = None
        self._actions = tuple(actions)
        self._build()
        self._rebuild()

    def _accent(self):
        """The spaCR accent of the current theme, for selection."""
        from ..iconset import active_theme
        from ..theme import palette_for

        return palette_for(active_theme())["accent"]

    def _build(self):
        """Lay out the toolbar, the page list, the view and the box settings."""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addLayout(self._build_toolbar())
        body = QHBoxLayout()
        body.setSpacing(8)
        body.addLayout(self._build_pages())
        self.scene = _CanvasScene(self)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.view = _CanvasView(self.scene, self)
        body.addWidget(self.view, 1)
        body.addWidget(self._build_settings())
        root.addLayout(body, 1)
        self.status = QLabel("")
        self.status.setObjectName("CanvasComposerStatus")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.resize(1280, 860)
        for keys, slot in ((QKeySequence.StandardKey.Undo, self.undo),
                           (QKeySequence.StandardKey.Redo, self.redo),
                           (QKeySequence.StandardKey.Delete, self.delete_selected),
                           (QKeySequence(Qt.Key.Key_Backspace), self.delete_selected)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(functools.partial(self._shortcut, slot))
        for key, dx, dy in ((Qt.Key.Key_Left, -1, 0), (Qt.Key.Key_Right, 1, 0),
                            (Qt.Key.Key_Up, 0, -1), (Qt.Key.Key_Down, 0, 1)):
            shortcut = QShortcut(QKeySequence(key), self.view)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(functools.partial(self.nudge, dx, dy))

    def _run_action(self, callback, *_):
        """Run an extra toolbar action with this composer."""
        callback(self)

    def _shortcut(self, slot):
        """Run an edit shortcut unless a text field has the keyboard."""
        from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit

        if isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit, QSpinBox)):
            return
        slot()

    def _build_toolbar(self):
        """Page, grid, tools, history and file controls."""
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.page_size = QComboBox()
        for name in _CANVAS_PAGE_SIZES:
            self.page_size.addItem(tr(name), name)
        self.orientation = QComboBox()
        self.orientation.addItem(tr("Portrait"), "portrait")
        self.orientation.addItem(tr("Landscape"), "landscape")
        self.page_size.currentIndexChanged.connect(self._page_changed)
        self.orientation.currentIndexChanged.connect(self._page_changed)
        self.grid_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_slider.setRange(4, 72)
        self.grid_slider.setFixedWidth(110)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(4, 72)
        self.grid_spin.setSuffix(" pt")
        for control in (self.grid_slider, self.grid_spin):
            control.setToolTip(tr("Grid spacing in points (72 points per inch)."))
        self.grid_slider.valueChanged.connect(self.grid_spin.setValue)
        self.grid_spin.valueChanged.connect(self.grid_slider.setValue)
        self.grid_spin.valueChanged.connect(self._grid_changed)
        self.show_grid = Toggle(tr("Grid"))
        self.show_grid.setToolTip(tr("Show the grid on the page; it is never printed."))
        self.snap = Toggle(tr("Snap"))
        self.snap.setToolTip(tr("Snap box edges to the grid."))
        self.show_grid.toggled.connect(self._grid_changed)
        self.snap.toggled.connect(self._grid_changed)
        self.select_tool = QPushButton(tr("Select"))
        self.select_tool.setToolTip(tr("Select, move and resize boxes."))
        self.text_tool = QPushButton(tr("Draw text box"))
        self.text_tool.setToolTip(tr("Drag on the page to draw a text box."))
        for button in (self.select_tool, self.text_tool):
            button.setCheckable(True)
        self.select_tool.setChecked(True)
        self.select_tool.clicked.connect(functools.partial(self.set_tool, "select"))
        self.text_tool.clicked.connect(functools.partial(self.set_tool, "text"))
        add_image = QPushButton(tr("Add image…"))
        add_image.setToolTip(tr("Place an image file from disk on the page."))
        add_image.clicked.connect(self._choose_images)
        self.undo_button = QPushButton(tr("Undo"))
        self.undo_button.setToolTip(tr("Undo the last change (Ctrl+Z)."))
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton(tr("Redo"))
        self.redo_button.setToolTip(tr("Redo the last undone change (Ctrl+Shift+Z)."))
        self.redo_button.clicked.connect(self.redo)
        open_button = QPushButton(tr("Open layout…"))
        open_button.setToolTip(tr("Open a layout saved as JSON."))
        open_button.clicked.connect(self._ask_open)
        save_button = QPushButton(tr("Save layout…"))
        save_button.setToolTip(tr("Save this layout as JSON in the results folder, to reopen and edit later."))
        save_button.clicked.connect(self._ask_save)
        self.export_button = QPushButton(tr("Export PDF…"))
        self.export_button.setObjectName("CanvasComposerExport")
        self.export_button.setToolTip(tr("Write every page to one PDF: vector text, images at full resolution "
                                         "with the current display settings."))
        self.export_button.clicked.connect(self._ask_export)
        page_label = QLabel(tr("Page"))
        page_label.setToolTip(tr("Paper size of every page.") + " " + tr("Page orientation of every page."))
        for widget in (page_label, self.page_size, self.orientation, QLabel(tr("Grid")),
                       self.grid_slider, self.grid_spin, self.show_grid, self.snap):
            bar.addWidget(widget)
        bar.addSpacing(10)
        for widget in (self.select_tool, self.text_tool, add_image):
            bar.addWidget(widget)
        for caption, tooltip, callback in self._actions:
            button = QPushButton(tr(caption))
            button.setToolTip(tr(tooltip))
            button.clicked.connect(functools.partial(self._run_action, callback))
            bar.addWidget(button)
        for widget in (self.undo_button, self.redo_button):
            bar.addWidget(widget)
        bar.addStretch(1)
        for widget in (open_button, save_button, self.export_button):
            bar.addWidget(widget)
        return bar

    def _build_pages(self):
        """The page list and its add, remove and reorder buttons."""
        column = QVBoxLayout()
        column.setSpacing(4)
        title = QLabel(tr("Pages"))
        title.setObjectName("CanvasComposerHeading")
        self.pages = QListWidget()
        self.pages.setObjectName("CanvasComposerPages")
        self.pages.setFixedWidth(150)
        self.pages.setToolTip(tr("Pages of the PDF, in order; click one to edit it."))
        self.pages.currentRowChanged.connect(self._page_chosen)
        column.addWidget(title)
        column.addWidget(self.pages, 1)
        for caption, tooltip, slot in (
                ("Add page", "Add an empty page after this one.", self.add_page),
                ("Remove page", "Remove this page and its boxes.", self.remove_page),
                ("Move up", "Move this page one place earlier.", functools.partial(self.move_page, -1)),
                ("Move down", "Move this page one place later.", functools.partial(self.move_page, 1))):
            button = QPushButton(tr(caption))
            button.setToolTip(tr(tooltip))
            button.clicked.connect(slot)
            column.addWidget(button)
        return column

    def _build_settings(self):
        """Settings of the selected boxes: text, image and arrangement."""
        panel = QWidget()
        panel.setObjectName("CanvasComposerSettings")
        panel.setFixedWidth(270)
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        self.text_group = QGroupBox(tr("Text"))
        form = QFormLayout(self.text_group)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("CanvasComposerText")
        self.text_edit.setToolTip(tr("Text of the selected text box."))
        self.text_edit.setFixedHeight(110)
        self.text_edit.textChanged.connect(self._text_edited)
        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(4, 96)
        self.font_slider.setValue(10)
        self.font_size = QSpinBox()
        self.font_size.setRange(4, 96)
        self.font_size.setSuffix(" pt")
        self.font_size.setValue(10)
        for control in (self.font_slider, self.font_size):
            control.setToolTip(tr("Font size in points."))
        self.font_slider.valueChanged.connect(self.font_size.setValue)
        self.font_size.valueChanged.connect(self.font_slider.setValue)
        self.font_size.valueChanged.connect(self._font_changed)
        size_row = QHBoxLayout()
        size_row.addWidget(self.font_slider, 1)
        size_row.addWidget(self.font_size)
        self.bold = Toggle(tr("Bold"))
        self.italic = Toggle(tr("Italic"))
        self.bold.setToolTip(tr("Bold text."))
        self.italic.setToolTip(tr("Italic text."))
        self.bold.toggled.connect(self._font_changed)
        self.italic.toggled.connect(self._font_changed)
        style_row = QHBoxLayout()
        style_row.addWidget(self.bold)
        style_row.addWidget(self.italic)
        self.align = QComboBox()
        for caption, value in (("Left", "left"), ("Centre", "center"), ("Right", "right")):
            self.align.addItem(tr(caption), value)
        self.align.currentIndexChanged.connect(self._font_changed)
        form.addRow(self.text_edit)
        form.addRow(tr("Size"), size_row)
        form.addRow(style_row)
        align_label = QLabel(tr("Align"))
        align_label.setToolTip(tr("Horizontal alignment of the text in its box."))
        form.addRow(align_label, self.align)
        self.image_group = QGroupBox(tr("Image"))
        image_form = QVBoxLayout(self.image_group)
        self.image_label = QLabel("")
        self.image_label.setWordWrap(True)
        self.lock_aspect = Toggle(tr("Keep aspect ratio"))
        self.lock_aspect.setToolTip(tr("Resize images without stretching them; turn off to stretch."))
        self.lock_aspect.toggled.connect(self._image_changed)
        self.scale_bar = Toggle(tr("Scale bar"))
        self.scale_bar.setToolTip(tr("Draw a scale bar on the image; needs the pixel size."))
        self.scale_bar.toggled.connect(self._image_changed)
        for widget in (self.image_label, self.lock_aspect, self.scale_bar):
            image_form.addWidget(widget)
        self.arrange_group = QGroupBox(tr("Arrange"))
        arrange = QGridLayout(self.arrange_group)
        for index, (caption, tooltip, where) in enumerate((
                ("Bring to front", "Draw the selected boxes above every other box.", "front"),
                ("Bring forward", "Draw the selected boxes one step higher.", "forward"),
                ("Send backward", "Draw the selected boxes one step lower.", "backward"),
                ("Send to back", "Draw the selected boxes below every other box.", "back"))):
            button = QPushButton(tr(caption))
            button.setToolTip(tr(tooltip))
            button.clicked.connect(functools.partial(self.restack, where))
            arrange.addWidget(button, index // 2, index % 2)
        delete = QPushButton(tr("Delete"))
        delete.setToolTip(tr("Delete the selected boxes (Delete key)."))
        delete.clicked.connect(self.delete_selected)
        arrange.addWidget(delete, 2, 0, 1, 2)
        for group in (self.text_group, self.image_group, self.arrange_group):
            column.addWidget(group)
        column.addStretch(1)
        return panel

    def _sync_controls(self):
        """Show the page and grid settings of the layout in the toolbar."""
        self._syncing = True
        try:
            self.page_size.setCurrentIndex(max(0, self.page_size.findData(self.canvas.page_size)))
            self.orientation.setCurrentIndex(max(0, self.orientation.findData(self.canvas.orientation)))
            self.grid_spin.setValue(int(round(self.canvas.grid)))
            self.show_grid.setChecked(self.canvas.show_grid)
            self.snap.setChecked(self.canvas.snap)
            self.undo_button.setEnabled(self.canvas.can_undo())
            self.redo_button.setEnabled(self.canvas.can_redo())
        finally:
            self._syncing = False

    def _rebuild(self):
        """Redraw the page list and the current page from the layout."""
        selected = set(self.selected_ids())
        self._page = max(0, min(self._page, len(self.canvas.pages) - 1))
        self._syncing = True
        try:
            self.pages.clear()
            for index, page in enumerate(self.canvas.pages):
                self.pages.addItem(tr("Page {number} ({count} boxes)", number=index + 1, count=len(page)))
            self.pages.setCurrentRow(self._page)
            self.scene.clear()
            self._items = {}
            width, height = self.canvas.page_points
            self.scene.setSceneRect(QRectF(-40, -40, width + 80, height + 80))
            for z, box in enumerate(self.canvas.pages[self._page]):
                item = _CanvasBoxItem(self, box)
                item.setZValue(z)
                self.scene.addItem(item)
                item.setSelected(box["id"] in selected)
                self._items[box["id"]] = item
        finally:
            self._syncing = False
        self._sync_controls()
        self._selection_changed()
        self.view.fit()
        self.changed.emit()

    def selected_ids(self):
        """Ids of the selected boxes on the current page, in z-order."""
        if not hasattr(self, "scene"):
            return []
        chosen = {item.box_id for item in self.scene.selectedItems() if isinstance(item, _CanvasBoxItem)}
        return [box["id"] for box in self.canvas.pages[self._page] if box["id"] in chosen] if self._items else []

    def select(self, ids):
        """Select exactly the boxes ``ids`` on the current page."""
        self.scene.clearSelection()
        for box_id in ids:
            item = self._items.get(box_id)
            if item is not None:
                item.setSelected(True)

    def _selection_changed(self):
        """Fill the box settings from the selection."""
        if self._syncing:
            return
        boxes = [self.canvas.find(box_id)[1] for box_id in self.selected_ids()]
        texts = [box for box in boxes if box["kind"] == "text"]
        images = [box for box in boxes if box["kind"] == "image"]
        self._editing_text = None
        self._syncing = True
        try:
            self.text_group.setEnabled(bool(texts))
            self.image_group.setEnabled(bool(images))
            self.arrange_group.setEnabled(bool(boxes))
            if texts:
                first = texts[0]
                self.text_edit.setPlainText((first.get("text") or "") if len(texts) == 1 else "")
                self.text_edit.setEnabled(len(texts) == 1)
                self.font_size.setValue(int(round(first.get("font_size") or 10)))
                self.bold.setChecked(bool(first.get("bold")))
                self.italic.setChecked(bool(first.get("italic")))
                self.align.setCurrentIndex(max(0, self.align.findData(first.get("align") or "left")))
            if images:
                first = images[0]
                known = all(box.get("pixel_size_um") for box in images)
                self.image_label.setText("\n".join(box.get("label") or "" for box in images[:4]))
                self.lock_aspect.setChecked(bool(first.get("lock_aspect", True)))
                self.scale_bar.setChecked(bool(first.get("scale_bar")))
                self.scale_bar.setEnabled(known)
                self.scale_bar.setToolTip(tr("Draw a scale bar on the image; needs the pixel size.") if known
                                          else tr("The pixel size of this image is not known, so no scale bar can be drawn."))
        finally:
            self._syncing = False

    def _commit(self):
        """Redraw after a change to the layout."""
        self._rebuild()

    def _moved(self, box_id):
        """Turn a finished drag into a snapped move of every selected box."""
        if self._syncing:
            return
        item = self._items.get(box_id)
        box = self.canvas.find(box_id)[1]
        if item is None or box is None:
            return
        dx, dy = item.pos().x() - box["x"], item.pos().y() - box["y"]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        ids = self.selected_ids() or [box_id]
        self.canvas.move_boxes(ids, dx, dy)
        self._rebuild_timer.start()

    def _resized(self, box_id, width, height):
        """Turn a finished corner drag into a snapped resize."""
        self.canvas.resize_box(box_id, width, height)
        self._rebuild_timer.start()

    def set_tool(self, tool):
        """Choose ``"select"`` or ``"text"`` (draw text boxes)."""
        self._tool = "text" if tool == "text" else "select"
        self.select_tool.setChecked(self._tool == "select")
        self.text_tool.setChecked(self._tool == "text")
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag if self._tool == "text"
                              else QGraphicsView.DragMode.RubberBandDrag)
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor if self._tool == "text"
                                       else Qt.CursorShape.ArrowCursor)

    def _begin_text(self, point):
        """Start drawing a text box at ``point``."""
        from PySide6.QtGui import QColor, QPen

        self._drawing = point
        self._rubber = self.scene.addRect(QRectF(point, point), QPen(QColor(self._accent()), 1, Qt.PenStyle.DashLine))

    def _stretch_text(self, point):
        """Follow the pointer with the text box being drawn."""
        self._rubber.setRect(QRectF(self._drawing, point).normalized())

    def _finish_text(self, point):
        """Add the drawn text box and select it for typing."""
        rect = QRectF(self._drawing, point).normalized()
        self._drawing = None
        self.scene.removeItem(self._rubber)
        self._rubber = None
        if rect.width() < 2 and rect.height() < 2:
            rect = QRectF(rect.topLeft(), QSizeF(self.canvas.grid * 8, self.canvas.grid * 2))
        self.add_text(rect.x(), rect.y(), rect.width(), rect.height())
        self.set_tool("select")
        self.text_edit.setFocus()

    def add_text(self, x, y, w, h, text="", **properties):
        """Add a text box to the current page and select it; returns the box."""
        box = self.canvas.add_box(self._page, "text", x, y, w, h, text=text, **properties)
        self._commit()
        self.select([box["id"]])
        return box

    def add_images(self, images, at=None):
        """Add images to the current page, side by side from ``at`` (a scene point).

        :param images: dictionaries from :func:`_canvas_dropped_images`.
        :returns: the boxes added.
        """
        width, _height = self.canvas.page_points
        x, y = (at.x(), at.y()) if at is not None else (self.canvas.grid * 2, self.canvas.grid * 2)
        side = max(self.canvas.grid * 4, self.canvas.snapped(width / 3.0))
        boxes = []
        for image in images:
            aspect = image.get("aspect") or (image["native"][0] / float(image["native"][1])
                                             if image.get("native") else 1.0)
            properties = {key: image[key] for key in ("source", "label", "native", "pixel_size_um")
                          if key in image}
            box = self.canvas.add_box(self._page, "image", x, y, side, side / aspect,
                                      aspect=aspect if image.get("aspect") or image.get("native") else None,
                                      **properties)
            boxes.append(box)
            x = box["x"] + box["w"] + self.canvas.grid
            if x + side > width:
                x, y = self.canvas.grid * 2, box["y"] + box["h"] + self.canvas.grid
        self._commit()
        self.select([box["id"] for box in boxes])
        return boxes

    def _choose_images(self):
        """Ask for image files and place them on the page."""
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Add image…"), self._folder,
                                                tr("Images") + " (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if paths:
            self.add_images([{"source": {"type": "file", "path": path}, "label": os.path.basename(path)}
                             for path in paths])

    def _preview_of(self, box):
        """The box's preview image, asking for it off the GUI thread the first time."""
        import json

        key = json.dumps(box.get("source"), sort_keys=True)
        if key in self._previews:
            return self._previews[key]
        if key not in self._requested:
            self._requested.add(key)
            self._preview_jobs.submit(
                functools.partial(_canvas_box_pixels, dict(box), _CANVAS_PREVIEW_EDGE, self._reader()),
                functools.partial(self._preview_ready, key))
        return None

    def _preview_ready(self, key, pixels):
        """Keep a finished preview, give images their aspect ratio and redraw."""
        self._previews[key] = _canvas_qimage(pixels)
        rows, columns = pixels.shape[:2]
        for page in self.canvas.pages:
            for box in page:
                if (box["kind"] == "image" and not box.get("aspect") and not box.get("fit")
                        and key == self._source_key(box)):
                    box["aspect"] = columns / float(rows)
                    if box.get("lock_aspect", True):
                        box["h"] = box["w"] / box["aspect"]
                        self.canvas._keep_on_page(box)
        self._rebuild()

    @staticmethod
    def _source_key(box):
        """The cache key of a box's image source."""
        import json

        return json.dumps(box.get("source"), sort_keys=True)

    def _text_edited(self):
        """Write the typed text into the selected text box; one undo step per editing run."""
        if self._syncing:
            return
        ids = [box_id for box_id in self.selected_ids() if self.canvas.find(box_id)[1]["kind"] == "text"]
        if len(ids) != 1:
            return
        if self._editing_text != ids[0]:
            self.canvas.checkpoint()
            self._editing_text = ids[0]
        self.canvas.find(ids[0])[1]["text"] = self.text_edit.toPlainText()
        self._items[ids[0]].update()
        self._sync_controls()

    def _font_changed(self, *_):
        """Apply size, weight, slant and alignment to the selected text boxes."""
        if self._syncing:
            return
        ids = [box_id for box_id in self.selected_ids() if self.canvas.find(box_id)[1]["kind"] == "text"]
        if not ids:
            return
        self.canvas.checkpoint()
        for box_id in ids:
            self.canvas.find(box_id)[1].update(font_size=float(self.font_size.value()), bold=self.bold.isChecked(),
                                               italic=self.italic.isChecked(), align=self.align.currentData())
        self._editing_text = None
        self._commit()

    def _image_changed(self, *_):
        """Apply the aspect lock and the scale bar to the selected images."""
        if self._syncing:
            return
        ids = [box_id for box_id in self.selected_ids() if self.canvas.find(box_id)[1]["kind"] == "image"]
        if not ids:
            return
        self.canvas.checkpoint()
        for box_id in ids:
            box = self.canvas.find(box_id)[1]
            box.update(lock_aspect=self.lock_aspect.isChecked(),
                       scale_bar=self.scale_bar.isChecked() and bool(box.get("pixel_size_um")))
            if box["lock_aspect"] and box.get("aspect"):
                box["h"] = box["w"] / float(box["aspect"])
                self.canvas._keep_on_page(box)
        self._commit()

    def _page_changed(self, *_):
        """Apply the paper size and orientation."""
        if self._syncing:
            return
        self.canvas.set_page(self.page_size.currentData(), self.orientation.currentData())
        self._commit()

    def _grid_changed(self, *_):
        """Apply the grid spacing, visibility and snapping."""
        if self._syncing:
            return
        self.canvas.set_grid(self.grid_spin.value(), self.show_grid.isChecked(), self.snap.isChecked())
        self.scene.invalidate(self.scene.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)
        self.changed.emit()

    def _page_chosen(self, row):
        """Edit page ``row``."""
        if self._syncing or row < 0:
            return
        self._page = row
        self._rebuild()

    def show_page(self, index):
        """Edit page ``index``."""
        self._page = max(0, min(int(index), len(self.canvas.pages) - 1))
        self._rebuild()

    @property
    def page_index(self):
        """The page being edited."""
        return self._page

    def add_page(self):
        """Add an empty page after the current one and edit it."""
        self._page = self.canvas.add_page(self._page + 1)
        self._rebuild()

    def remove_page(self):
        """Remove the current page."""
        self.canvas.remove_page(self._page)
        self._page = max(0, self._page - 1)
        self._rebuild()

    def move_page(self, step):
        """Move the current page ``step`` places."""
        target = max(0, min(self._page + int(step), len(self.canvas.pages) - 1))
        self.canvas.move_page(self._page, target)
        self._page = target
        self._rebuild()

    def nudge(self, dx, dy):
        """Move the selected boxes by one grid step."""
        ids = self.selected_ids()
        if ids:
            self.canvas.move_boxes(ids, dx * self.canvas.grid, dy * self.canvas.grid)
            self._commit()

    def restack(self, where):
        """Change the z-order of the selected boxes (see :meth:`_CanvasLayout.restack`)."""
        ids = self.selected_ids()
        if ids:
            self.canvas.restack(ids, where)
            self._commit()

    def delete_selected(self):
        """Delete the selected boxes."""
        ids = self.selected_ids()
        if ids:
            self.canvas.delete_boxes(ids)
            self._commit()

    def undo(self):
        """Undo the last change."""
        if self.canvas.undo():
            self._editing_text = None
            self._rebuild()

    def redo(self):
        """Redo the last undone change."""
        if self.canvas.redo():
            self._editing_text = None
            self._rebuild()

    def set_layout(self, layout):
        """Edit ``layout`` from its first page."""
        self.canvas = layout
        self._page = 0
        self._rebuild()

    def _default_path(self, stem, suffix):
        """A new file name in the results folder."""
        from datetime import datetime

        folder = os.path.join(self._folder, "results") if self._folder else ""
        return os.path.join(folder, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}")

    def save_layout(self, path):
        """Save the layout as JSON to ``path``; returns the path."""
        written = self.canvas.save(path)
        self.status.setText(tr("Saved the layout to {path}", path=written))
        return written

    def open_layout(self, path):
        """Open a layout saved as JSON; returns whether it could be read."""
        try:
            layout = _CanvasLayout.load(path)
        except (OSError, ValueError) as exc:
            self.status.setText(tr("Could not open the layout: {error}", error=str(exc)))
            return False
        self.set_layout(layout)
        self.status.setText(tr("Opened {path}", path=path))
        return True

    def _ask_save(self):
        """Ask where to save the layout."""
        path, _ = QFileDialog.getSaveFileName(self, tr("Save layout…"), self._default_path("canvas_layout", ".json"),
                                              tr("Layout") + " (*.json)")
        if path:
            self.save_layout(path)

    def _ask_open(self):
        """Ask for a layout to open."""
        folder = os.path.join(self._folder, "results") if self._folder else ""
        path, _ = QFileDialog.getOpenFileName(self, tr("Open layout…"), folder, tr("Layout") + " (*.json)")
        if path:
            self.open_layout(path)

    def _ask_export(self):
        """Ask where to write the PDF, then write it."""
        path, _ = QFileDialog.getSaveFileName(self, tr("Export PDF…"), self._default_path("canvas", ".pdf"),
                                              "PDF (*.pdf)")
        if path:
            self.export_pdf(path if path.lower().endswith(".pdf") else path + ".pdf")

    def export_pdf(self, path):
        """Write the layout to ``path`` off the GUI thread; returns whether it started."""
        if self._export_jobs.is_busy():
            self.status.setText(tr("Wait for the current export to finish or cancel it."))
            return False
        self._cancelled = threading.Event()
        layout = _CanvasLayout.from_dict(self.canvas.to_dict())
        reader = functools.partial(_canvas_full_pixels, self._reader())
        self.export_button.setEnabled(False)
        self.status.setText(tr("Writing {path}…", path=path))
        return self._export_jobs.submit(
            functools.partial(_canvas_pdf, layout, path, reader, self._cancelled), self._exported)

    def _reader(self):
        """The image reader for work off the GUI thread.

        A reader with a ``fresh`` method is asked for a snapshot here, on the
        GUI thread, so the work reads the display settings as they are now.
        """
        fresh = getattr(self._field_pixels, "fresh", None)
        return fresh() if fresh is not None else self._field_pixels

    def _export_failed(self, message):
        """Report an export that raised and allow another."""
        self.export_button.setEnabled(True)
        self.status.setText(tr("The PDF could not be written: {error}", error=message))

    def _exported(self, result):
        """Report the written PDF and any image that could not be read."""
        path, errors = result
        self.export_button.setEnabled(True)
        self.last_export = path
        if path is None:
            self.status.setText(tr("Export cancelled; no file was written."))
            return
        message = tr("Wrote {path}", path=path)
        if errors:
            message += " " + tr("{count} images could not be read: {names}", count=len(errors),
                                names=", ".join(name for name, _error in errors[:5]))
        self.status.setText(message)

    def closeEvent(self, event):
        """Stop previews and exports with the window."""
        self._cancelled.set()
        for runner in (self._preview_jobs, self._export_jobs):
            runner.cancel()
        super().closeEvent(event)


def _canvas_full_pixels(field_pixels, box):
    """Every pixel of an image box, for the PDF."""
    return _canvas_box_pixels(box, None, field_pixels)


_CANVAS_UI_SOURCES = frozenset({
    "Letter",
    "Add page",
    "Add an empty page after this one.",
    "Remove page",
    "Remove this page and its boxes.",
    "Move up",
    "Move this page one place earlier.",
    "Move down",
    "Move this page one place later.",
    "Bring to front",
    "Draw the selected boxes above every other box.",
    "Bring forward",
    "Draw the selected boxes one step higher.",
    "Send backward",
    "Draw the selected boxes one step lower.",
    "Send to back",
    "Draw the selected boxes below every other box.",
    "Left",
    "Centre",
    "Right",
})
"""Captions the Custom export canvas passes to ``tr`` through loop variables.

The runtime catalog generator cannot see them at a literal call site and
imports this set instead, as it does ``_DOSE_RESPONSE_UI_SOURCES``.
"""
