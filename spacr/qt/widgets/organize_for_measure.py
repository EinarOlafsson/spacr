"""Make Masks' "Organize for Measure…" popup: images and masks into merged/.

Item 600, replacing item 593's string of prompts and its separate "Sort into
channels" dialog as the entry point. ONE popup puts images and their masks
into the layout Measure reads -- Yokogawa-named channel folders, the masks,
and ``merged/*.npy`` -- through :mod:`spacr.channel_sorting` and
:mod:`spacr.folder_consolidation`, which hold every decision:

* a SOURCE folder (typed, browsed or dropped), optionally consolidated
  first (its subfolders' images copied into one folder, named after their
  folders);
* the REGEX box: Mask generation's own ``metadata_type`` field -- the same
  grouped convention menu, example and "Test on my folder" -- with its
  ``custom_regex``, plus "Sort by regex", "Auto regex" and "Detect sets";
* the DIMENSIONS: which columns are image channels and which are masks,
  each mask with its role (cell, nucleus, pathogen) and the channel whose
  images it outlines;
* a TABLE with one column per channel and per mask and one row per field.
  Every column takes files and folders dropped from the file manager (a
  folder brings its images, recursively) and cells dragged from another
  column. Dropped files sort themselves by name and are matched across the
  columns the way Detect sets pairs images -- file-name tokens, then order,
  never two images of different size (:func:`spacr.channel_sorting.
  _match_columns`). A cell nothing matched stays empty and flagged.

Mode 1: fill the source, choose or test the regex, press "Sort by regex",
Apply. Mode 2: fill the columns by dropping, Apply. Mode 3, "Teach me…":
answer "Which channel is this?" for one image at a time while spaCR learns
a regex from the answers (:func:`spacr.channel_sorting._teach_step`), until
every image is placed. Each works without the others. A folder made by
folder consolidation is read under its files' ORIGINAL names, from its
``rename_manifest.csv``, because the copies' names may have lost what told
the channels apart. Apply builds a :class:`spacr.channel_sorting.SortPlan`; incomplete
rows block it (the regex box is where to fix them), RGB images and z-stacks
are offered for conversion, and detected sets are shown three at a time
before they are used. Nothing moves before Apply; the screen applies the
plan off the GUI thread and every move is written to the manifest.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import (QByteArray, QEvent, QMimeData, QRect, QSize, Qt,
                            Signal, QTimer)
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QStyle, QStyledItemDelegate, QTableWidget,
    QToolTip, QVBoxLayout,
)

from ... import channel_sorting as cs
from ..hidpi import scaled_for
from ..i18n import tr
from .colour_picker import pick_colour
from .sortable_table import SortableTableItem

#: The drag format of cells dragged between the table's columns: JSON
#: ``{"column": int, "paths": [str]}``.
_CELLS_MIME = "application/x-spacr-organize-cells"

#: Subfolders consolidation leaves out: masks, raw originals, earlier sorts.
_NOT_CONSOLIDATED = ("masks", "orig", cs.DEFAULT_DEST_NAME)

#: The convention the regex box starts on: spaCR proposes a regex.
_DEFAULT_METADATA_TYPE = "auto"


def _headless() -> bool:
    """Whether no one can answer a modal box (offscreen/minimal platform)."""
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.platformName() in ("offscreen", "minimal")


def _mime_paths(mime: Optional[QMimeData]) -> List[str]:
    """The local files and folders a drag from the file manager carries.

    :param mime: the drag's data.
    :returns: absolute paths, in drag order.
    """
    if mime is None or not mime.hasUrls():
        return []
    return [os.path.abspath(url.toLocalFile()) for url in mime.urls()
            if url.isLocalFile() and url.toLocalFile()]


def _mime_cells(mime: Optional[QMimeData]):
    """The column and files a drag out of the table carries.

    :param mime: the drag's data.
    :returns: ``(column, [paths])``, or ``(None, [])`` for any other drag.
    """
    if mime is None or not mime.hasFormat(_CELLS_MIME):
        return None, []
    try:
        data = json.loads(bytes(mime.data(_CELLS_MIME)).decode("utf-8"))
        return int(data["column"]), [str(p) for p in data["paths"]]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        return None, []


def _mime_slots(mime: Optional[QMimeData]) -> List[List[int]]:
    """The table slots a drag out of the table carries.

    :param mime: the drag's data.
    :returns: ``[[row, column], ...]``, empty for any other drag.
    """
    if mime is None or not mime.hasFormat(_CELLS_MIME):
        return []
    try:
        data = json.loads(bytes(mime.data(_CELLS_MIME)).decode("utf-8"))
        return [[int(r), int(c)] for r, c in data.get("cells", [])]
    except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
        return []


def _mime_anchor(mime: Optional[QMimeData]) -> Optional[List[int]]:
    """The slot a drag out of the table started on.

    :param mime: the drag's data.
    :returns: ``[row, column]``, or None for any other drag.
    """
    if mime is None or not mime.hasFormat(_CELLS_MIME):
        return None
    try:
        data = json.loads(bytes(mime.data(_CELLS_MIME)).decode("utf-8"))
        anchor = data.get("anchor")
        return [int(anchor[0]), int(anchor[1])] if anchor else None
    except (ValueError, TypeError, AttributeError, IndexError, KeyError,
            UnicodeDecodeError):
        return None


def _images_in(paths: Iterable[str]):
    """Expand dropped files and folders into image files.

    :param paths: files and folders.
    :returns: ``([image paths], [names left out])``.
    """
    from ... import drop_classification as dc

    found: List[str] = []
    skipped: List[str] = []
    for path in paths:
        path = os.path.abspath(os.fspath(path))
        if os.path.isdir(path):
            found.extend(dc._folder_images(path))
        elif os.path.isfile(path) and path.lower().endswith(cs.IMAGE_EXTS):
            found.append(path)
        else:
            skipped.append(os.path.basename(path) or path)
    return list(dict.fromkeys(found)), skipped


@dataclass
class _Column:
    """One table column: an image channel or a mask.

    :ivar kind: ``"channel"`` or ``"mask"``.
    :ivar role: a mask's role (cell, nucleus, pathogen); unused for a channel.
    :ivar of_channel: a mask's channel -- the 1-based channel whose images
        it outlines; unused for a channel.
    """

    kind: str
    role: str = ""
    of_channel: int = 1


class _PathEdit(QLineEdit):
    """The source field, which also takes a folder dropped on it."""

    def __init__(self, parent=None):
        """A line edit that accepts drops.

        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        """Accept a drag holding a folder.

        :param event: the drag event.
        """
        if any(os.path.isdir(p) for p in _mime_paths(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        """Keep accepting while the drag moves over the field.

        :param event: the drag event.
        """
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        """Put the first dropped folder in the field.

        :param event: the drop event.
        """
        folders = [p for p in _mime_paths(event.mimeData()) if os.path.isdir(p)]
        if not folders:
            event.ignore()
            return
        event.acceptProposedAction()
        self.setText(folders[0])


#: How the table shows a cell: its name, its thumbnail, or both.
_VIEWS = ("text", "image", "both")

#: The thumbnail side in the table's image views, in pixels.
_CELL_THUMB = 96

#: The thumbnail-size slider's range, in pixels.
_THUMB_RANGE = (32, 320)

#: The side of the × that clears a cell, in pixels.
_CLOSE_SIZE = 14


def _swap_cells(rows: List[List[Optional[str]]], mask_of: Dict[int, int],
                source, target) -> List[List[Optional[str]]]:
    """Swap two table slots; the images' masks move with them.

    Dropping on an empty slot is the same swap with nothing. A target row
    one past the last makes a new row. When both slots are in channel
    columns that have a mask column, the masks in those rows swap too, so
    an image never leaves its mask behind.

    :param rows: the table, one list per row, changed in place.
    :param mask_of: ``{channel column: its mask column}``.
    :param source: ``(row, column)`` dragged.
    :param target: ``(row, column)`` dropped on.
    :returns: ``rows``.
    """
    (r1, c1), (r2, c2) = source, target
    width = len(rows[0]) if rows else max(c1, c2) + 1
    while r2 >= len(rows):
        rows.append([None] * width)
    rows[r1][c1], rows[r2][c2] = rows[r2][c2], rows[r1][c1]
    if c1 in mask_of and c2 in mask_of and (r1, c1) != (r2, c2):
        m1, m2 = mask_of[c1], mask_of[c2]
        rows[r1][m1], rows[r2][m2] = rows[r2][m2], rows[r1][m1]
    return rows


def _clear_cell(rows: List[List[Optional[str]]], mask_of: Dict[int, int],
                cell) -> Optional[str]:
    """Empty one slot; an image's mask leaves the table with it.

    :param rows: the table, changed in place.
    :param mask_of: ``{channel column: its mask column}``.
    :param cell: ``(row, column)``.
    :returns: the file that was there, or None.
    """
    row, column = cell
    if not 0 <= row < len(rows):
        return None
    removed, rows[row][column] = rows[row][column], None
    if removed and column in mask_of:
        rows[row][mask_of[column]] = None
    return removed


def _block_moves(cells, anchor, target, width: int,
                 mask_of: Dict[int, int]) -> Dict[tuple, tuple]:
    """Where each cell of a dragged block lands.

    The block keeps its shape: every cell moves by the offset from the cell
    the drag started on (``anchor``) to the slot it was dropped on. A cell
    in a channel column with a mask column takes its mask along, into the
    target channel's mask column, unless that mask slot is already spoken
    for. A block that would leave the table sideways or above row 0 does
    not move at all.

    :param cells: ``[(row, column), ...]`` the dragged cells.
    :param anchor: ``(row, column)`` the cell the drag started on.
    :param target: ``(row, column)`` the slot it was dropped on.
    :param width: the number of columns.
    :param mask_of: ``{channel column: its mask column}``.
    :returns: ``{source slot: target slot}``, empty when the block cannot
        move.
    """
    dr, dc = target[0] - anchor[0], target[1] - anchor[1]
    moves: Dict[tuple, tuple] = {}
    for row, column in cells:
        to = (row + dr, column + dc)
        if to[0] < 0 or not 0 <= to[1] < width:
            return {}
        moves[(row, column)] = to
    taken = set(moves.values())
    for (row, column), (to_row, to_column) in list(moves.items()):
        if column in mask_of and to_column in mask_of:
            source = (row, mask_of[column])
            to = (to_row, mask_of[to_column])
            if source not in moves and to not in taken and source != to:
                moves[source] = to
                taken.add(to)
    return {s: t for s, t in moves.items() if s != t}


def _move_block(rows: List[List[Optional[str]]], moves: Dict[tuple, tuple]
                ) -> List[List[Optional[str]]]:
    """Move several slots at once; what they land on swaps back.

    Every source's file goes to its target. A file already on a target that
    is not itself moving goes to the slot freed at the start of that chain
    of moves -- for a block dropped clear of itself that is exactly a
    pairwise swap, and for a block shifted onto part of itself nothing is
    lost. Rows past the end are added.

    :param rows: the table, changed in place.
    :param moves: ``{(row, column): (row, column)}`` from :func:`_block_moves`.
    :returns: ``rows``.
    """
    if not moves:
        return rows
    width = len(rows[0]) if rows else 1 + max(
        c for pair in moves.items() for _r, c in pair)
    last = max(r for pair in moves.items() for r, _c in pair)
    while last >= len(rows):
        rows.append([None] * width)
    before = {slot: rows[slot[0]][slot[1]]
              for pair in moves.items() for slot in pair}
    back = {t: s for s, t in moves.items()}
    for source in moves:
        rows[source[0]][source[1]] = None
    for source, to in moves.items():
        rows[to[0]][to[1]] = before[source]
    for to in moves.values():
        if to in moves or before[to] is None:
            continue
        free = back[to]
        while free in back:
            free = back[free]
        rows[free[0]][free[1]] = before[to]
    return rows


def _selection_cells(indexes) -> List[List[int]]:
    """The filled cells among selected indexes, in reading order.

    :param indexes: model indexes (or anything with ``row``, ``column`` and
        ``data``).
    :returns: ``[[row, column], ...]`` of the cells holding a file.
    """
    return sorted([index.row(), index.column()] for index in indexes
                  if index.data(Qt.UserRole))


def _close_rect(cell_rect):
    """Where a cell's × sits: its top-right corner.

    :param cell_rect: the cell's QRect.
    :returns: the × QRect.
    """
    from PySide6.QtCore import QRect

    return QRect(cell_rect.right() - _CLOSE_SIZE - 3, cell_rect.top() + 3,
                 _CLOSE_SIZE, _CLOSE_SIZE)


class _CellDelegate(QStyledItemDelegate):
    """Draws a table cell as text, thumbnail or both, with its ×.

    The × is painted, not a widget per cell: white with a dark outline, red
    while the mouse is over it (the table tracks the mouse and the click).
    """

    def __init__(self, table, parent=None):
        """Remember the table whose view, colour and thumbnails it reads.

        :param table: the :class:`_OrganizeTable`.
        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self._table = table
        self.view = "text"
        self.text_color = QColor("white")
        self.pixmaps: Dict[str, QPixmap] = OrderedDict()
        #: The (row, column) whose × the mouse is over, or None.
        self.hover = None

    def paint(self, painter, option, index) -> None:
        """Draw the cell in the chosen view, then its ×.

        :param painter: the painter.
        :param option: the style option.
        :param index: the cell.
        """
        path = index.data(Qt.UserRole)
        pixmap = self.pixmaps.get(path) if path else None
        if self.view == "text" or pixmap is None:
            super().paint(painter, option, index)
        else:
            painter.save()
            if option.state & QStyle.State_Selected:
                painter.fillRect(option.rect, option.palette.highlight())
            scaled = scaled_for(pixmap, self._table,
                                option.rect.size() - QSize(4, 4))
            shown = scaled.deviceIndependentSize()
            x = option.rect.x() + int(option.rect.width() - shown.width()) // 2
            y = option.rect.y() + int(option.rect.height() - shown.height()) // 2
            painter.drawPixmap(x, y, scaled)
            if self.view == "both":
                text = os.path.basename(path)
                box = QRect(x, y, int(shown.width()),
                            int(shown.height())).adjusted(3, 3, -3, -3)
                painter.setPen(QColor(0, 0, 0, 200))
                painter.drawText(box.translated(1, 1),
                                 Qt.AlignBottom | Qt.AlignLeft | Qt.TextWordWrap,
                                 text)
                painter.setPen(self.text_color)
                painter.drawText(box, Qt.AlignBottom | Qt.AlignLeft
                                 | Qt.TextWordWrap, text)
            painter.restore()
        if path:
            self._paint_close(painter, _close_rect(option.rect),
                              self.hover == (index.row(), index.column()))

    @staticmethod
    def _paint_close(painter, rect, hovered: bool) -> None:
        """Draw the × in ``rect``: white, red when hovered.

        :param painter: the painter.
        :param rect: where.
        :param hovered: whether the mouse is over it.
        """
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        inset = rect.adjusted(3, 3, -3, -3)
        outline = QPen(QColor(0, 0, 0, 170), 4, Qt.SolidLine, Qt.RoundCap)
        stroke = QPen(QColor(220, 40, 40) if hovered else QColor("white"),
                      2, Qt.SolidLine, Qt.RoundCap)
        for pen in (outline, stroke):
            painter.setPen(pen)
            painter.drawLine(inset.topLeft(), inset.bottomRight())
            painter.drawLine(inset.topRight(), inset.bottomLeft())
        painter.restore()

    def helpEvent(self, event, view, option, index) -> bool:
        """The ×'s tooltip.

        :param event: the help event.
        :param view: the view.
        :param option: the style option.
        :param index: the cell.
        :returns: True when a tooltip was shown.
        """
        if index.data(Qt.UserRole) and _close_rect(option.rect).contains(
                event.pos()):
            QToolTip.showText(event.globalPos(), tr(
                "Remove from the table (Delete); the file is not touched."),
                view)
            return True
        return super().helpEvent(event, view, option, index)


#: Where the table's view and overlay colour are remembered, per viewer.
_PREFS_VIEW = "organize_for_measure/view"
_PREFS_COLOR = "organize_for_measure/text_color"


def _load_view_prefs():
    """The remembered view and overlay colour.

    :returns: ``(view, colour name)``; ``("text", "#ffffff")`` when nothing
        is stored or the preferences cannot be read.
    """
    try:
        from ..prefs import _s

        store = _s()
        view = str(store.value(_PREFS_VIEW, "text") or "text")
        color = str(store.value(_PREFS_COLOR, "#ffffff") or "#ffffff")
    except Exception:
        return "text", "#ffffff"
    return (view if view in _VIEWS else "text"), color


def _save_view_prefs(view: str, color: str) -> None:
    """Remember the view and overlay colour for the next popup.

    :param view: one of :data:`_VIEWS`.
    :param color: a colour name.
    """
    try:
        from ..prefs import _s

        store = _s()
        store.setValue(_PREFS_VIEW, view)
        store.setValue(_PREFS_COLOR, color)
    except Exception:
        pass


def _legible_backdrop(color: QColor, window: QColor) -> str:
    """A backdrop that keeps a caption in ``color`` readable on ``window``.

    The "Text colour" button writes its caption in the chosen colour; white
    or yellow on a light window, or black on a dark one, would vanish. Below
    a 3:1 contrast ratio (WCAG relative luminance) a translucent chip of the
    opposite shade goes behind it.

    :param color: the caption's colour.
    :param window: the colour behind the button.
    :returns: a style-sheet colour, ``transparent`` when none is needed.
    """
    def luminance(c: QColor) -> float:
        """The colour's relative luminance (WCAG 2), 0 for black to 1 for white."""
        def linear(v: float) -> float:
            """One sRGB channel, 0..1, converted to linear light."""
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        c = QColor(c)
        return (0.2126 * linear(c.redF()) + 0.7152 * linear(c.greenF())
                + 0.0722 * linear(c.blueF()))

    fore, back = luminance(color), luminance(window)
    ratio = (max(fore, back) + 0.05) / (min(fore, back) + 0.05)
    if ratio >= 3.0:
        return "transparent"
    if fore >= 0.18:
        return "rgba(0, 0, 0, 160)"
    return "rgba(255, 255, 255, 200)"


#: Where the thumbnail size is remembered.
_PREFS_THUMB = "organize_for_measure/thumb_size"


def _clamp_thumb(value) -> int:
    """A thumbnail size inside :data:`_THUMB_RANGE`.

    :param value: anything ``int()`` takes; junk gives :data:`_CELL_THUMB`.
    :returns: the size in pixels.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return _CELL_THUMB
    return max(_THUMB_RANGE[0], min(_THUMB_RANGE[1], value))


def _load_thumb_pref() -> int:
    """The remembered thumbnail size, or :data:`_CELL_THUMB`."""
    try:
        from ..prefs import _s

        return _clamp_thumb(_s().value(_PREFS_THUMB, _CELL_THUMB))
    except Exception:
        return _CELL_THUMB


def _save_thumb_pref(size: int) -> None:
    """Remember the thumbnail size for the next popup.

    :param size: pixels.
    """
    try:
        from ..prefs import _s

        _s().setValue(_PREFS_THUMB, _clamp_thumb(size))
    except Exception:
        pass


class _OrganizeTable(QTableWidget):
    """The channel/mask table; every column is a drop target.

    :ivar dropped: ``(column, paths, from_column)`` -- files dropped on a
        column; ``from_column`` is -1 for a drop from the file manager.
    :ivar cell_moved: ``(from row, from column, to row, to column)`` -- one
        cell dragged onto a slot; the dialog swaps the two.
    :ivar clear_requested: ``([[row, column], ...])`` -- cells whose × was
        clicked, or that were selected when Delete was pressed.
    :ivar block_moved: ``([[row, column], ...], [anchor row, column],
        [to row, column])`` -- several cells dragged together.
    :ivar placeholder: the hint painted over the table while it holds no
        file.
    """

    dropped = Signal(int, list, int)
    cell_moved = Signal(int, int, int, int)
    clear_requested = Signal(list)
    block_moved = Signal(list, list, list)

    def __init__(self, parent=None):
        """A table that drags cells out and takes drops on any column.

        :param parent: the Qt parent.
        """
        super().__init__(0, 0, parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setMouseTracking(True)
        self._pressed_close = None
        self.placeholder = tr("Drag images or folders here")
        #: Where a rubber-band selection started, and its band.
        self._band_origin = None
        self._band = None
        self._band_extend = False
        self._drag_anchor = None

    def _is_empty(self) -> bool:
        """Whether no cell holds a file (the placeholder is shown)."""
        for row in range(self.rowCount()):
            for column in range(self.columnCount()):
                item = self.item(row, column)
                if item is not None and item.data(Qt.UserRole):
                    return False
        return True

    def paintEvent(self, event) -> None:
        """Paint the table, and the drop hint over it while it is empty.

        :param event: the paint event.
        """
        super().paintEvent(event)
        if not self._is_empty():
            return
        painter = QPainter(self.viewport())
        color = QColor(self.palette().color(self.foregroundRole()))
        color.setAlpha(150)
        painter.setPen(color)
        font = painter.font()
        font.setPointSizeF(font.pointSizeF() * 1.4)
        painter.setFont(font)
        painter.drawText(self.viewport().rect(), Qt.AlignCenter
                         | Qt.TextWordWrap, self.placeholder)
        painter.end()

    def _select_band(self, pos) -> None:
        """Select every cell the band from its origin to ``pos`` touches.

        :param pos: a point in the viewport.
        """
        from PySide6.QtCore import QItemSelectionModel, QRect
        from PySide6.QtWidgets import QRubberBand

        rect = QRect(self._band_origin, pos).normalized()
        if self._band is None:
            self._band = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self._band.setGeometry(rect)
        self._band.show()
        flag = (QItemSelectionModel.Select if self._band_extend
                else QItemSelectionModel.ClearAndSelect)
        self.setSelection(rect, flag)

    def _end_band(self) -> bool:
        """Hide the rubber band and forget its origin.

        :returns: whether a band was showing.
        """
        showing = self._band is not None and self._band.isVisible()
        self._band_origin = None
        if self._band is not None:
            self._band.hide()
        return showing

    def _close_hit(self, pos):
        """The cell whose × is under ``pos``, or None.

        :param pos: a point in the viewport.
        :returns: ``(row, column)`` or None.
        """
        index = self.indexAt(pos)
        if index.isValid() and index.data(Qt.UserRole) and _close_rect(
                self.visualRect(index)).contains(pos):
            return index.row(), index.column()
        return None

    def _set_hover(self, hit) -> None:
        """Turn the × under the mouse red, and the last one white again.

        :param hit: ``(row, column)`` or None.
        """
        delegate = self.itemDelegate()
        if getattr(delegate, "hover", None) != hit:
            delegate.hover = hit
            self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        """Follow the mouse over the ×s.

        :param event: the mouse event.
        """
        pos = event.position().toPoint()
        self._set_hover(self._close_hit(pos))
        if self._band_origin is not None and event.buttons() & Qt.LeftButton:
            if (self._band is not None and self._band.isVisible()) or (
                    pos - self._band_origin).manhattanLength() >= \
                    QApplication.startDragDistance():
                self._select_band(pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        """No × is hovered once the mouse leaves.

        :param event: the leave event.
        """
        self._set_hover(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """A press on a × is the ×'s, not the start of a drag.

        :param event: the mouse event.
        """
        pos = event.position().toPoint()
        hit = self._close_hit(pos)
        self._pressed_close = hit
        if hit is not None and event.button() == Qt.LeftButton:
            event.accept()
            return
        index = self.indexAt(pos)
        self._drag_anchor = ([index.row(), index.column()]
                             if index.isValid() else None)
        extend = bool(event.modifiers()
                      & (Qt.ControlModifier | Qt.ShiftModifier))
        if event.button() == Qt.LeftButton and (
                extend or not index.isValid()
                or not index.data(Qt.UserRole)):
            # A band starts from an empty slot, from outside the cells, or
            # with Ctrl/Shift held; a plain press on a file drags it.
            self._band_origin = pos
            self._band_extend = extend
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Releasing on the × that was pressed clears its cell.

        :param event: the mouse event.
        """
        if self._end_band():
            self._pressed_close = None
            event.accept()
            return
        hit = self._close_hit(event.position().toPoint())
        pressed, self._pressed_close = self._pressed_close, None
        if hit is not None and hit == pressed:
            event.accept()
            chosen = _selection_cells(self.selectedIndexes())
            self.clear_requested.emit(chosen if list(hit) in chosen
                                      else [list(hit)])
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        """Delete or Backspace clears the selected cells.

        :param event: the key event.
        """
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            cells = [[item.row(), item.column()]
                     for item in self.selectedItems()
                     if item.data(Qt.UserRole)]
            if cells:
                self.clear_requested.emit(cells)
                event.accept()
                return
        super().keyPressEvent(event)

    def mimeTypes(self) -> List[str]:
        """The formats a drag out of the table offers.

        :returns: the cells format only.
        """
        return [_CELLS_MIME]

    def mimeData(self, items) -> QMimeData:
        """Put the dragged cells' files and slots into the drag.

        ``column`` is the column of the cell the drag started on (or of the
        first cell) and ``paths`` that column's files, which a drop on the
        "new channel" strip moves; ``cells`` are every dragged slot and
        ``anchor`` the one the drag started on, which a drop on the table
        moves as a block.

        :param items: the dragged items.
        :returns: the drag's data.
        """
        mime = QMimeData()
        items = sorted((item for item in items if item.data(Qt.UserRole)),
                       key=lambda item: (item.row(), item.column()))
        cells = [[item.row(), item.column()] for item in items]
        anchor = self._drag_anchor if self._drag_anchor in cells else (
            cells[0] if cells else None)
        column = anchor[1] if anchor else -1
        chosen = [item for item in items if item.column() == column]
        mime.setData(_CELLS_MIME, QByteArray(json.dumps(
            {"column": column,
             "paths": [item.data(Qt.UserRole) for item in chosen],
             "cells": cells, "anchor": anchor}
        ).encode("utf-8")))
        return mime

    def _accepts(self, event) -> bool:
        """Whether a drag carries files or cells, and there is a column.

        :param event: the drag event.
        """
        mime = event.mimeData()
        if _mime_cells(mime)[1]:
            return bool(self.columnCount())
        return bool(_mime_paths(mime))

    def dragEnterEvent(self, event) -> None:
        """Accept files from outside and cells from another column.

        :param event: the drag event.
        """
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        """Keep accepting while the drag moves over a column.

        :param event: the drag event.
        """
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        """Hand the drop, and the column it landed in, to the dialog.

        :param event: the drop event.
        """
        if not self._accepts(event):
            event.ignore()
            return
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if not self.columnCount():
            # The empty table's placeholder: the drop makes the first
            # channel column.
            self.dropped.emit(-1, _mime_paths(event.mimeData()), -1)
            return
        column = self.columnAt(int(event.position().x()))
        if column < 0:
            column = self.columnCount() - 1
        source, cells = _mime_cells(event.mimeData())
        slots = _mime_slots(event.mimeData())
        if slots:
            row = self.rowAt(int(event.position().y()))
            if row < 0:
                row = self.rowCount()
            if len(slots) == 1:
                self.cell_moved.emit(slots[0][0], slots[0][1], row, column)
            else:
                anchor = _mime_anchor(event.mimeData()) or slots[0]
                self.block_moved.emit(slots, anchor, [row, column])
            return
        if cells:
            self.dropped.emit(column, cells, -1 if source is None else source)
        else:
            self.dropped.emit(column, _mime_paths(event.mimeData()), -1)


class _NewColumnZone(QFrame):
    """"Drop here for a new channel": makes a channel column of the drop.

    :ivar dropped: ``(paths, from_column)``.
    """

    dropped = Signal(list, int)

    def __init__(self, parent=None):
        """Build the strip.

        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self.setObjectName("OrganizeNewColumnZone")
        self.setFrameShape(QFrame.StyledPanel)
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        label = QLabel(tr("Drop images or folders here for a new channel"))
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)

    def dragEnterEvent(self, event) -> None:
        """Accept files from outside and cells from the table.

        :param event: the drag event.
        """
        mime = event.mimeData()
        if _mime_paths(mime) or _mime_cells(mime)[1]:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        """Keep accepting while the drag moves over the strip.

        :param event: the drag event.
        """
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        """Hand the drop to the dialog.

        :param event: the drop event.
        """
        mime = event.mimeData()
        source, cells = _mime_cells(mime)
        paths = cells or _mime_paths(mime)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.dropped.emit(paths, -1 if source is None else source)


class _TeachQuestion(QDialog):
    """"Which channel is this?" -- one image, one answer, for "Teach me".

    :ivar answer: ``("channel", n)``, ``("mask", n, role)``, ``"skip"`` or
        ``"stop"``.
    """

    def __init__(self, path: str, known: List[object], parent=None):
        """Show the image and one button per possible answer.

        :param path: the image.
        :param known: labels already used, channels first.
        :param parent: the Qt parent.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Which channel is this?"))
        self.answer: object = "stop"
        layout = QVBoxLayout(self)
        picture = QLabel()
        array = cs.thumbnail(path, 320)
        if array is not None:
            from PySide6.QtGui import QImage, QPixmap

            array = np.ascontiguousarray(array)
            image = QImage(array.data, array.shape[1], array.shape[0],
                           array.shape[1], QImage.Format_Grayscale8)
            picture.setPixmap(QPixmap.fromImage(image.copy()))
        layout.addWidget(picture)
        caption = QLabel(os.path.basename(path))
        caption.setWordWrap(True)
        layout.addWidget(caption)
        channels = sorted({lab[1] for lab in known if lab[0] == "channel"})
        row = QHBoxLayout()
        for n in channels + [max(channels, default=0) + 1]:
            text = (tr("Channel {n}", n=n) if n in channels
                    else tr("New channel ({n})", n=n))
            button = QPushButton(text)
            button.clicked.connect(
                lambda _c=False, n=n: self._choose(("channel", n)))
            row.addWidget(button)
        layout.addLayout(row)
        mask_row = QHBoxLayout()
        self.mask_channel = QComboBox()
        for n in channels or [1]:
            self.mask_channel.addItem(tr("of channel {n}", n=n), n)
        self.mask_role = QComboBox()
        for role in cs.MASK_ROLES:
            self.mask_role.addItem(role, role)
        mask_button = QPushButton(tr("It is a mask"))
        mask_button.clicked.connect(lambda: self._choose((
            "mask", int(self.mask_channel.currentData()),
            self.mask_role.currentData())))
        for widget in (mask_button, self.mask_role, self.mask_channel):
            mask_row.addWidget(widget)
        layout.addLayout(mask_row)
        end_row = QHBoxLayout()
        skip = QPushButton(tr("Skip this image"))
        skip.clicked.connect(lambda: self._choose("skip"))
        stop = QPushButton(tr("Stop: the rest are not needed"))
        stop.clicked.connect(lambda: self._choose("stop"))
        end_row.addWidget(skip)
        end_row.addWidget(stop)
        layout.addLayout(end_row)

    def _choose(self, answer) -> None:
        """Keep the answer and close.

        :param answer: the label chosen.
        """
        self.answer = answer
        self.accept()


class OrganizeForMeasureDialog(QDialog):
    """Put images and masks into channel folders and ``merged/`` for Measure.

    :ivar plan: the confirmed :class:`spacr.channel_sorting.SortPlan` once the
        dialog is accepted, else None.
    :ivar columns: the table's columns, in order.
    :ivar rows: one list per field, one path or None per column.
    """

    def __init__(self, source: str = "", parent=None,
                 channel_folders: Optional[Sequence[str]] = None,
                 masks_dir: Optional[str] = None):
        """Build the popup, prefilled from a drop when there was one.

        :param source: the source folder, if known.
        :param parent: the Qt parent.
        :param channel_folders: folders to put in the table, one channel
            column each (their masks from each folder's ``masks/``) -- the
            Make Masks drop of several folders or of channel-like
            subfolders.
        :param masks_dir: the source's masks folder, when not its ``masks/``.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Organize for Measure"))
        self.resize(1000, 760)
        self.masks_dir = masks_dir
        self.columns: List[_Column] = []
        self.rows: List[List[Optional[str]]] = []
        self.row_keys: List[Optional[cs.SetKey]] = []
        self.plan: Optional[cs.SortPlan] = None
        self._output_auto = True
        self._note = ""
        self._alias_cache: Dict[str, Dict[str, str]] = {}
        #: The regex "Teach me" learned and its ``{marker: label}``.
        self._taught: tuple = ("", None)
        self._skipped: set = set()
        #: Asks "Which channel is this?" for one image; replaced in tests.
        self.ask_label = self._ask_label
        #: Asks what a part of the names means; replaced in tests.
        self.ask_marker = self._ask_marker
        #: Answers yes/no questions; replaced in tests.
        self.ask = self._ask
        #: Shows three detected sets and asks; replaced in tests.
        self.confirm_examples = self._ask_examples

        layout = QVBoxLayout(self)
        intro = QLabel(tr(
            "Organise images and their masks into the layout Measure reads. "
            "Either give a source folder and sort it with a regex, or drop "
            "files and folders into the columns below: each column is a "
            "channel or a mask, each row one field, and dropped files are "
            "matched across the columns by name. On Apply the images and "
            "masks are MOVED into Yokogawa-named channel folders and merged "
            "into merged/; every move is written to a manifest."))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # The fields on aligned rows above, every action button on
        # one line below them, then the view controls over the table.
        from PySide6.QtWidgets import QGridLayout, QSlider

        fields = QGridLayout()
        fields.setColumnStretch(1, 1)
        fields.addWidget(QLabel(tr("Source folder")), 0, 0)
        self.source_edit = _PathEdit()
        self.source_edit.setPlaceholderText(tr(
            "A folder of images; drop one here"))
        self.source_edit.setText(source or "")
        self.source_edit.textChanged.connect(lambda _t: self._suggest_output())
        fields.addWidget(self.source_edit, 0, 1)
        self.browse_button = QPushButton(tr("Browse…"))
        self.browse_button.clicked.connect(self._browse)
        fields.addWidget(self.browse_button, 0, 2)
        self.consolidate_check = QCheckBox(tr(
            "Consolidate subfolders into filenames first (copies the images "
            "into a new folder, each named after its folders; the originals "
            "are not touched)"))
        fields.addWidget(self.consolidate_check, 1, 1, 1, 2)

        from ..screens.settings_model import _MetadataTypeField

        regex_label = QLabel(tr("Regex"))
        regex_label.setToolTip(tr(
            "Regex: the same filename conventions as Mask generation. "
            "'auto' lets spaCR propose one; 'custom' uses the regex below."))
        fields.addWidget(regex_label, 2, 0)
        self.metadata_field = _MetadataTypeField(
            _DEFAULT_METADATA_TYPE,
            source_folder=lambda: self.source_edit.text(),
            custom_regex=lambda: self.custom_edit.text(), parent=self)
        fields.addWidget(self.metadata_field, 2, 1, 1, 2)
        fields.addWidget(QLabel(tr("custom_regex")), 3, 0)
        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText(tr(
            "Regex with named groups: chanID, and wellID, fieldID..."))
        fields.addWidget(self.custom_edit, 3, 1, 1, 2)
        fields.addWidget(QLabel(tr("Output folder")), 4, 0)
        self.output_edit = QLineEdit()
        self.output_edit.textEdited.connect(self._output_edited)
        fields.addWidget(self.output_edit, 4, 1, 1, 2)
        layout.addLayout(fields)

        self.sort_button = QPushButton(tr("Sort by regex"))
        self.sort_button.setToolTip(tr(
            "Read the source folder (or the files in the table) with the "
            "regex: chanID gives the column, the other groups the row."))
        self.sort_button.clicked.connect(lambda: self.sort_by_regex())
        self.auto_button = QPushButton(tr("Auto regex"))
        self.auto_button.setToolTip(tr(
            "Find a regex that puts every image in exactly one complete set, "
            "put it in custom_regex and sort by it."))
        self.auto_button.clicked.connect(lambda: self.auto_regex())
        self.detect_button = QPushButton(tr("Detect sets"))
        self.detect_button.setToolTip(tr(
            "Ignore the names' structure: match every column's files by "
            "similarity, order and size, then show three sets to check."))
        self.detect_button.clicked.connect(lambda: self.detect_sets())
        self.teach_button = QPushButton(tr("Teach me…"))
        self.teach_button.setToolTip(tr(
            "Show one image at a time and ask which channel (or mask) it is, "
            "learning a regex from the answers until every image is placed."))
        self.teach_button.clicked.connect(lambda: self.teach())
        self.add_channel_button = QPushButton(tr("Add channel"))
        self.add_channel_button.clicked.connect(
            lambda: self.add_column("channel"))
        self.add_mask_button = QPushButton(tr("Add mask"))
        self.add_mask_button.clicked.connect(lambda: self.add_column("mask"))
        self.remove_button = QPushButton(tr("Remove selected files"))
        self.remove_button.setToolTip(tr(
            "Take the selected cells' files out of the table; the files "
            "themselves are not touched."))
        self.remove_button.clicked.connect(self._remove_selected)
        #: The one line of action buttons, in order.
        self.action_row = QHBoxLayout()
        for button in (self.sort_button, self.auto_button, self.detect_button,
                       self.teach_button):
            self.action_row.addWidget(button)
        self.action_row.addSpacing(12)
        for button in (self.add_channel_button, self.add_mask_button,
                       self.remove_button):
            self.action_row.addWidget(button)
        self.action_row.addStretch(1)
        layout.addLayout(self.action_row)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel(tr("Channels and masks")))
        view_row.addStretch(1)
        self.color_button = QPushButton(tr("Text colour"))
        self.color_button.setFlat(True)
        self.color_button.setCursor(Qt.PointingHandCursor)
        self.color_button.setToolTip(tr(
            "The colour of the names written over the images."))
        self.color_button.clicked.connect(lambda: self._pick_text_color())
        view_row.addWidget(self.color_button)
        show_label = QLabel(tr("Show"))
        show_label.setToolTip(tr(
            "Show each cell as its file name, its image, or its image with "
            "the name written over it."))
        view_row.addWidget(show_label)
        self.view_box = QComboBox()
        for value, caption in (("text", tr("Text")), ("image", tr("Image")),
                               ("both", tr("Image + text"))):
            self.view_box.addItem(caption, value)
        view_row.addWidget(self.view_box)
        size_label = QLabel(tr("Size"))
        size_label.setToolTip(tr("The size of the images in the table."))
        view_row.addWidget(size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(*_THUMB_RANGE)
        self.size_slider.setFixedWidth(140)
        self.size_slider.setToolTip(tr("The size of the images in the table."))
        self.thumb_size = _load_thumb_pref()
        self.size_slider.setValue(self.thumb_size)
        self.size_slider.valueChanged.connect(self._set_thumb_size)
        view_row.addWidget(self.size_slider)
        from ..mask_thumbnail_quality import quality_combo, changes

        view_row.addWidget(QLabel(tr("Thumbnail quality")))
        self.quality_box = quality_combo(self)
        view_row.addWidget(self.quality_box)
        changes.changed.connect(self._quality_changed)
        layout.addLayout(view_row)
        self._editors_row = QHBoxLayout()
        layout.addLayout(self._editors_row)

        table_row = QHBoxLayout()
        self.table = _OrganizeTable()
        self.table.dropped.connect(self._on_table_drop)
        self.table.cell_moved.connect(self._swap_slots)
        self.table.block_moved.connect(self._move_slots)
        self.table.clear_requested.connect(self._clear_slots)
        self.delegate = _CellDelegate(self.table, self.table)
        self.table.setItemDelegate(self.delegate)
        self._thumb_workers: list = []
        self._thumb_pending: set = set()
        self._thumb_generation = 0
        self._thumb_signatures = {}
        self._thumb_requested = {}
        self._thumb_closed = False
        self._thumb_render_ratio = None
        self._thumb_poll = QTimer(self)
        self._thumb_poll.setInterval(750)
        self._thumb_poll.timeout.connect(self._load_thumbnails)
        self._thumb_poll.start()
        self.table.verticalScrollBar().valueChanged.connect(self._load_thumbnails)
        self.table.horizontalScrollBar().valueChanged.connect(self._load_thumbnails)
        self.finished.connect(lambda _code: self._stop_thumbs())
        table_row.addWidget(self.table, 1)
        self.new_zone = _NewColumnZone()
        self.new_zone.setMaximumWidth(150)
        self.new_zone.dropped.connect(self._on_new_zone_drop)
        table_row.addWidget(self.new_zone)
        layout.addLayout(table_row, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton(tr("Cancel"))
        self.apply_button = QPushButton(tr("Apply"))
        self.apply_button.setObjectName("PrimaryButton")
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self._on_apply)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        for folder in channel_folders or ():
            self.add_files(self.add_column("channel"),
                           [os.path.join(folder, n)
                            for n in cs.list_folder_images(folder)],
                           rematch=False)
        if channel_folders:
            self._rematch()
        self._suggest_output()
        stored = _load_view_prefs()
        self._set_text_color(stored[1])
        self.view_box.setCurrentIndex(max(0, self.view_box.findData(stored[0])))
        self.view_box.currentIndexChanged.connect(
            lambda _i: self._set_view(self.view_box.currentData()))
        self._set_view(stored[0], remember=False)
        self._refresh()
        from ..screens.settings_model import retarget_field_tooltips

        retarget_field_tooltips(self)

    # -- columns -----------------------------------------------------------

    def _channel_columns(self) -> List[int]:
        """The indices of the channel columns, in order."""
        return [i for i, c in enumerate(self.columns) if c.kind == "channel"]

    def _channel_of(self, column: int) -> int:
        """The 1-based channel number of channel column ``column``.

        :param column: a channel column's index.
        """
        return self._channel_columns().index(column) + 1

    def _column_for_channel(self, channel: int) -> Optional[int]:
        """The index of channel ``channel``'s column, or None.

        :param channel: a 1-based channel.
        """
        channels = self._channel_columns()
        return channels[channel - 1] if 0 < channel <= len(channels) else None

    def _mask_column(self, channel: int) -> Optional[int]:
        """The first mask column of ``channel``, or None.

        :param channel: a 1-based channel.
        """
        for index, column in enumerate(self.columns):
            if column.kind == "mask" and column.of_channel == channel:
                return index
        return None

    def add_column(self, kind: str, role: str = "",
                   of_channel: Optional[int] = None) -> int:
        """Add a channel or mask column at the end.

        :param kind: ``"channel"`` or ``"mask"``.
        :param role: a mask's role; default the first role no mask has.
        :param of_channel: a mask's channel; default the first channel with
            no mask column, else 1.
        :returns: the new column's index.
        """
        column = _Column(kind)
        if kind == "mask":
            used = {c.role for c in self.columns if c.kind == "mask"}
            column.role = role or next(
                (r for r in cs.MASK_ROLES if r not in used), cs.MASK_ROLES[0])
            if of_channel is None:
                masked = {c.of_channel for c in self.columns if c.kind == "mask"}
                of_channel = next(
                    (n for n in range(1, len(self._channel_columns()) + 1)
                     if n not in masked), 1)
            column.of_channel = int(of_channel)
        self.columns.append(column)
        for row in self.rows:
            row.append(None)
        self._refresh()
        return len(self.columns) - 1

    def remove_column(self, index: int) -> None:
        """Remove a column and its files from the table.

        :param index: the column.
        """
        if not 0 <= index < len(self.columns):
            return
        removed = self.columns.pop(index)
        if removed.kind == "channel":
            channel = len([i for i in range(index)
                           if self.columns[i].kind == "channel"]) + 1
            for column in self.columns:
                if column.kind == "mask" and column.of_channel > channel:
                    column.of_channel -= 1
        for row in self.rows:
            row.pop(index)
        self._drop_empty_rows()
        self._refresh()

    def _column_files(self, index: int) -> List[str]:
        """The files in column ``index``, in row order.

        :param index: a column.
        """
        return [row[index] for row in self.rows if row[index]]

    # -- files -------------------------------------------------------------

    def add_files(self, column: int, paths: Iterable[str],
                  rematch: bool = True) -> List[str]:
        """Put dropped files and folders into a column, and match the rows.

        A folder brings its images, recursively, not its ``masks/``. Images
        dropped into a channel column bring the masks Make Masks saved for
        them (``<their folder>/masks/<stem>.tif``) into that channel's mask
        column, which is made when there is none. A file already in the
        table moves to this column. Nothing on disk changes.

        :param column: the column index.
        :param paths: files and folders.
        :param rematch: match the rows afterwards (off while prefilling).
        :returns: the image files added.
        """
        images, skipped = _images_in(paths)
        self._note = (tr("Not images, left out: {names}",
                         names=", ".join(skipped)) if skipped else "")
        if not images or not 0 <= column < len(self.columns):
            self._refresh()
            return []
        self._forget(images)
        for path in images:
            row = [None] * len(self.columns)
            row[column] = path
            self.rows.append(row)
            self.row_keys.append(None)
        self._bring_masks(images, column)
        if not self.source_edit.text().strip():
            self._suggest_output()
        if rematch:
            self._rematch()
        else:
            self._refresh()
        return images

    def _bring_masks(self, images: Sequence[str], column: int) -> None:
        """Put the saved masks of images in a channel column beside them.

        Each image's mask where Make Masks saves it
        (``<its folder>/masks/<stem>.tif``) goes into that channel's mask
        column, which is made -- its role guessed from the names -- when
        there is none. A mask already elsewhere in the table moves.

        :param images: images just put in ``column``.
        :param column: their column.
        """
        if self.columns[column].kind != "channel":
            return
        channel = self._channel_of(column)
        masks = [m for m in (cs.mask_for(os.path.dirname(p), p)
                             for p in images) if m]
        if not masks:
            return
        mask_column = self._mask_column(channel)
        if mask_column is None:
            roles = cs.default_mask_roles({channel: list(images)}, [channel])
            used = {c.role for c in self.columns if c.kind == "mask"}
            role = roles.get(channel, "")
            mask_column = self.add_column(
                "mask", role if role not in used else "", channel)
        self._forget(masks)
        for mask in masks:
            row = [None] * len(self.columns)
            row[mask_column] = mask
            self.rows.append(row)
            self.row_keys.append(None)

    def _forget(self, paths: Iterable[str]) -> None:
        """Take files out of whatever cells hold them.

        :param paths: files.
        """
        paths = set(paths)
        for row in self.rows:
            for index, value in enumerate(row):
                if value in paths:
                    row[index] = None
        self._drop_empty_rows()

    def _drop_empty_rows(self) -> None:
        """Remove rows that hold no file at all."""
        kept = [(row, key) for row, key in zip(self.rows, self.row_keys)
                if any(row)]
        self.rows = [row for row, _key in kept]
        self.row_keys = [key for _row, key in kept]

    def _move_files(self, paths: Iterable[str], column: int) -> None:
        """Move files already in the table into another column, and rematch.

        :param paths: files in the table.
        :param column: the column they go to.
        """
        paths = [p for p in paths if p]
        if not paths or not 0 <= column < len(self.columns):
            return
        self._forget(paths)
        for path in paths:
            row = [None] * len(self.columns)
            row[column] = path
            self.rows.append(row)
            self.row_keys.append(None)
        self._bring_masks(paths, column)
        self._rematch()

    def _remove_selected(self) -> int:
        """Take the selected cells' files out of the table.

        :returns: how many files were taken out.
        """
        paths = [item.data(Qt.UserRole) for item in self.table.selectedItems()
                 if item.data(Qt.UserRole)]
        self._forget(paths)
        self._refresh()
        return len(paths)

    def _on_table_drop(self, column: int, paths: list, source: int) -> None:
        """Files dropped on a column: from outside, or from another column.

        :param column: the column dropped on.
        :param paths: the files or folders.
        :param source: the column they were dragged from, or -1.
        """
        if column < 0:
            self._on_new_zone_drop(paths, source)
            return
        if source >= 0:
            if source != column:
                self._move_files(paths, column)
            return
        self.add_files(column, paths)

    def _on_new_zone_drop(self, paths: list, source: int) -> None:
        """Files dropped on "new channel": a channel column of their own.

        :param paths: the files or folders.
        :param source: the column they were dragged from, or -1.
        """
        column = self.add_column("channel")
        if source >= 0:
            self._move_files(paths, column)
        else:
            self.add_files(column, paths)

    def _partners(self) -> Dict[int, int]:
        """``{mask column: its channel's column}`` for matching.

        :returns: the pairs whose channel column exists.
        """
        partners = {}
        for index, column in enumerate(self.columns):
            if column.kind == "mask":
                target = self._column_for_channel(column.of_channel)
                if target is not None:
                    partners[index] = target
        return partners

    def _rematch(self) -> None:
        """Line every column's files up in rows by name, order and size."""
        files = [self._column_files(i) for i in range(len(self.columns))]
        self.rows = cs._match_columns(files, self._partners())
        self.row_keys = [None] * len(self.rows)
        self._refresh()

    # -- the regex ---------------------------------------------------------

    def _source(self) -> str:
        """The source folder as typed, or ``""``."""
        return self.source_edit.text().strip()

    def _regex_for(self, names: Sequence[str]) -> str:
        """The regex the regex box stands for, for these file names.

        ``auto`` is the custom regex when one is typed, else spaCR's proposal
        (:func:`spacr.channel_sorting.infer_regex`); every other convention
        is Mask generation's own pattern for the names' extensions.

        :param names: file names or paths.
        :returns: the regex, or ``""`` when there is none.
        """
        from ... import regex_infer

        key = self.metadata_field.get_value() or _DEFAULT_METADATA_TYPE
        custom = self.custom_edit.text().strip()
        if key == "auto":
            return custom or cs.infer_regex(list(names)) or ""
        if key == "custom":
            return custom
        extensions = sorted({os.path.splitext(n)[1].lstrip(".") or "tif"
                             for n in names}) or ["tif"]
        try:
            if len(extensions) == 1:
                return regex_infer._metadata_pattern(key, extensions[0],
                                                     custom or None)
            return regex_infer._metadata_pattern_any_extension(key, extensions)
        except KeyError:
            return ""

    def _consolidate(self, source: str) -> str:
        """Copy the source's subfolders' images into one folder, if asked.

        :param source: the source folder.
        :returns: the folder to read: the consolidated copy, or ``source``.
        """
        from ... import folder_consolidation as fc

        if not self.consolidate_check.isChecked():
            return source
        files, _folders = fc.nested_file_count(source, cs.IMAGE_EXTS,
                                               _NOT_CONSOLIDATED)
        if not files:
            return source
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = fc.consolidate_folder(
                source, extensions=cs.IMAGE_EXTS, skip_dirs=_NOT_CONSOLIDATED,
                log=lambda _text: None)
        finally:
            QApplication.restoreOverrideCursor()
        self.consolidate_check.setChecked(False)
        self.source_edit.setText(str(result.output))
        return str(result.output)

    def _alias(self, path: str) -> str:
        """The name the regex reads for ``path``.

        A file copied by folder consolidation is read under its ORIGINAL
        name, from the folder's ``rename_manifest.csv``
        (:func:`spacr.channel_sorting._original_names`): the copy's name may
        have lost what told the channels apart.

        :param path: an image path.
        """
        folder = os.path.dirname(path)
        if folder not in self._alias_cache:
            self._alias_cache[folder] = cs._original_names(folder)
        # Kept in its folder, so two folders' field1.tif stay two names; the
        # regex reads the file name alone.
        return os.path.join(folder, self._alias_cache[folder].get(
            os.path.basename(path), os.path.basename(path)))

    def _paths_to_sort(self):
        """The images a regex or "Teach me" reads, their channels and masks.

        :returns: ``(paths, {path: channel}, {image: mask})`` -- the source
            folder's images and saved masks, or, without a source, the
            channel columns' files and the masks matched to them.
        """
        source = self._source()
        channels: Dict[str, int] = {}
        masks: Dict[str, str] = {}
        if source and os.path.isdir(source):
            source = self._consolidate(source)
            paths = [os.path.join(source, n)
                     for n in cs.list_folder_images(source)]
            for path in paths:
                mask = cs.mask_for(source, os.path.basename(path),
                                   self.masks_dir)
                if mask:
                    masks[path] = mask
        else:
            paths = []
            for column in self._channel_columns():
                for path in self._column_files(column):
                    paths.append(path)
                    channels[path] = self._channel_of(column)
            partners = self._partners()
            for index, column in enumerate(self.columns):
                if column.kind == "mask" and index in partners:
                    pairs = cs._match_columns([
                        self._column_files(partners[index]),
                        self._column_files(index)])
                    masks.update({row[0]: row[1] for row in pairs
                                  if row[0] and row[1]})
        paths = [p for p in paths if p not in self._skipped]
        return paths, channels, masks

    def sort_by_regex(self, pattern: Optional[str] = None,
                      chan_map: Optional[Dict[str, object]] = None
                      ) -> Optional[cs.SetReport]:
        """Fill the table from the regex: chanID is the column, the rest the row.

        With a source folder its images are read (consolidated first when
        asked), and their masks found where Make Masks saves them; with none,
        the files already in the channel columns are re-sorted, keeping the
        column each is in. Masks go into a mask column per channel. A mask
        two images share (two files with one stem) is left out and named.

        :param pattern: the regex; default what the regex box stands for.
        :param chan_map: ``{chanID value: label}`` learned by "Teach me",
            a label being ``("channel", n)`` or ``("mask", n, role)``; by
            default the regex's chanID values are channels in natural order.
        :returns: the set report, or None when there was nothing to read.
        """
        paths, channels, masks = self._paths_to_sort()
        if not paths:
            self.status.setText(tr(
                "Nothing to sort: give a source folder of images, or drop "
                "images into the channel columns."))
            return None
        if chan_map is None and pattern is None and \
                self.custom_edit.text().strip() == self._taught[0]:
            chan_map = self._taught[1]
        aliases = [self._alias(p) for p in paths]
        back = dict(zip(aliases, paths))
        if pattern is None:
            pattern = self._regex_for(aliases)
        parsed = cs.parse_names(
            aliases, pattern,
            {self._alias(p): c for p, c in channels.items()} or None)
        name_masks: Dict[str, List[str]] = {}
        if chan_map is not None:
            compiled = cs.compile_regex(pattern)[0]
            for item in parsed:
                item.channel = None
                if not item.matched or compiled is None:
                    continue
                label = chan_map.get(item.groups.get(cs.CHANNEL_GROUP, ""))
                if isinstance(label, tuple) and label[0] == "channel":
                    item.channel = int(label[1])
                elif isinstance(label, tuple) and label[0] == "mask":
                    name_masks.setdefault(item.set_key, []).append(
                        (int(label[1]), label[2] if len(label) > 2 else "",
                         back[item.name]))
                    item.matched = False
        taken_as_masks = {m for entries in name_masks.values()
                          for _c, _r, m in entries}
        report = cs.check_sets([p for p in parsed
                                if back[p.name] not in taken_as_masks])
        chosen: Dict[cs.SetKey, Dict[int, str]] = {}
        for item in parsed:
            if item.matched and item.channel is not None:
                chosen.setdefault(item.set_key, {}).setdefault(
                    item.channel, back[item.name])
        claimed: Dict[str, List[str]] = {}
        for image, mask in masks.items():
            claimed.setdefault(mask, []).append(image)
        shared = sorted(m for m, images in claimed.items() if len(images) > 1)
        masks = {i: m for i, m in masks.items() if m not in shared}
        present = sorted({c for m in chosen.values() for c in m})
        position = {c: i + 1 for i, c in enumerate(present)}
        explicit = {}
        for key, entries in name_masks.items():
            for channel, role, mask in entries:
                image = chosen.get(key, {}).get(channel)
                if image is not None:
                    masks[image] = mask
                    if role:
                        explicit[channel] = role
        roles = cs.default_mask_roles(
            {c: [m[c] for m in chosen.values() if c in m] for c in present},
            sorted({c for m in chosen.values() for c, p in m.items()
                    if p in masks}))
        roles.update({c: r for c, r in explicit.items() if c in roles})
        self.columns = [_Column("channel") for _c in present]
        for channel in sorted(roles):
            self.columns.append(_Column("mask", roles[channel],
                                        position[channel]))
        self.rows, self.row_keys = [], []
        for key in sorted(chosen, key=lambda k: cs.natural_key(cs.set_label(k))):
            row: List[Optional[str]] = [None] * len(self.columns)
            for index, channel in enumerate(present):
                row[index] = chosen[key].get(channel)
            for index, column in enumerate(self.columns):
                if column.kind == "mask":
                    image = chosen[key].get(present[column.of_channel - 1])
                    row[index] = masks.get(image) if image else None
            self.rows.append(row)
            self.row_keys.append(key)
        self._note = report.summary()
        if shared:
            self._note += "\n" + tr(
                "{n} mask(s) are shared by two images with one name stem and "
                "were left out: {names}", n=len(shared),
                names=", ".join(os.path.basename(m) for m in shared[:4]))
        self._suggest_output()
        self._refresh()
        return report

    def auto_regex(self) -> Optional[str]:
        """Propose a regex, put it in custom_regex, and sort by it.

        :returns: the regex, or None when none was found.
        """
        paths, by_path, _masks = self._paths_to_sort()
        names = [self._alias(p) for p in paths]
        channels = ({self._alias(p): c for p, c in by_path.items()}
                    if by_path else None)
        pattern = cs.infer_regex(names, channels) if names else None
        if not pattern:
            self.status.setText(tr(
                "spaCR found no regex that puts every image in exactly one "
                "complete set. Drop the images into channel columns, or use "
                "Detect sets."))
            return None
        self.metadata_field.set_value("custom")
        self.custom_edit.setText(pattern)
        self.sort_by_regex(pattern)
        return pattern

    def teach(self) -> Optional[str]:
        """"Teach me": ask what images are until a regex places all of them.

        One image is shown at a time with "Which channel is this?"
        (:attr:`ask_label`). From the answers
        :func:`spacr.channel_sorting._teach_step` learns what in the names
        marks each channel or mask and writes a regex; each new marker is
        confirmed with :attr:`ask_marker` (a channel, a mask, or only part
        of the field's name). The table fills as it goes, and the regex is
        put in custom_regex, where it can be edited. It ends when every
        image is placed, or when the user says stop.

        :returns: the learned regex, or None when none was learned.
        """
        paths, _channels, _masks = self._paths_to_sort()
        if not paths:
            self.status.setText(tr(
                "Nothing to sort: give a source folder of images, or drop "
                "images into the channel columns."))
            return None
        back = {self._alias(p): p for p in paths}
        names = list(back)
        answers: Dict[str, object] = {}
        confirmed: Dict[str, object] = {}
        pattern = None
        by_marker: Dict[str, object] = {}
        for _round in range(len(names) + 8):
            pattern, by_marker, nxt = cs._teach_step(names, answers)
            relabelled = False
            for marker, label in sorted(by_marker.items()):
                if marker in confirmed:
                    continue
                verdict = self.ask_marker(marker, label)
                if verdict == "field":
                    others = [lab for lab in by_marker.values() if lab != label]
                    for name, answer in list(answers.items()):
                        if answer == label and others:
                            answers[name] = others[0]
                    relabelled = True
                    break
                confirmed[marker] = verdict or label
                if verdict and verdict != label:
                    for name, answer in list(answers.items()):
                        if answer == label:
                            answers[name] = verdict
                    relabelled = True
                    break
            if relabelled:
                continue
            if pattern:
                self.metadata_field.set_value("custom")
                self.custom_edit.setText(pattern)
                self._taught = (pattern, dict(by_marker))
                self.sort_by_regex(pattern, dict(by_marker))
            if nxt is None:
                break
            answer = self.ask_label(back[nxt], self._known_labels(answers))
            if answer == "stop" or answer is None:
                break
            if answer == "skip":
                self._skipped.add(back[nxt])
            answers[nxt] = answer
        if pattern:
            self.status.setText(self.status.text() + "\n" + tr(
                "Learned regex: {regex}", regex=pattern))
        return pattern

    @staticmethod
    def _known_labels(answers: Dict[str, object]) -> List[object]:
        """The labels answered so far, channels first.

        :param answers: ``{name: label}``.
        """
        labels = {a for a in answers.values() if isinstance(a, tuple)}
        return sorted(labels, key=lambda lab: (lab[0] != "channel", lab[1:]))

    def _ask_label(self, path: str, known: List[object]):
        """Show one image and ask which channel it is (a modal box).

        :param path: the image.
        :param known: labels already used.
        :returns: ``("channel", n)``, ``("mask", n, role)``, ``"skip"`` or
            ``"stop"`` (always ``"stop"`` when no one can answer).
        """
        if _headless():
            return "stop"
        dialog = _TeachQuestion(path, known, self)
        dialog.exec()
        return dialog.answer

    def _ask_marker(self, marker: str, label):
        """Ask what the text that tells images apart means.

        :param marker: the text (``""`` for nothing).
        :param label: what the answers say it marks.
        :returns: the label, another label, or ``"field"`` when the text is
            only part of the field's name; headless, the label unchanged.
        """
        if _headless():
            return label
        shown = marker or tr("(nothing)")
        kind = (tr("channel {n}", n=label[1]) if label[0] == "channel"
                else tr("the {role} mask of channel {n}", role=label[2],
                        n=label[1]))
        box = QMessageBox(self)
        box.setWindowTitle(tr("What does this part of the name mean?"))
        box.setText(tr(
            "Images whose names have “{marker}” here look like {kind}. "
            "Does this part of the name mark a channel or a mask, or is it "
            "only part of the field's name?", marker=shown, kind=kind))
        keep = box.addButton(tr("It marks {kind}", kind=kind),
                             QMessageBox.AcceptRole)
        box.addButton(tr("It is part of the field's name"),
                      QMessageBox.RejectRole)
        box.exec()
        return label if box.clickedButton() is keep else "field"

    def detect_sets(self) -> bool:
        """Match the columns' files by name, order and size, and check three.

        With an empty table and a source folder, the source is sorted by the
        regex first, for its channels. The sets are shown three at a time
        (:class:`spacr.qt.widgets.channel_sort_dialog.ExampleSetsDialog`)
        and kept only when the user says they are right.

        :returns: whether the detected sets were kept.
        """
        if not self.rows and self._source():
            self.sort_by_regex()
        if len([c for c in self._channel_columns() if self._column_files(c)]) < 2:
            self.status.setText(tr(
                "Detect sets needs at least two channel columns with images: "
                "drop them into the columns, or sort by a regex with chanID."))
            return False
        before = ([list(r) for r in self.rows], list(self.row_keys))
        self._rematch()
        sets = self._complete_sets(detected=True)
        if not sets or not self.confirm_examples(sets):
            self.rows, self.row_keys = before
            self._refresh()
            self.status.setText(tr("The detected sets were not used."))
            return False
        self.row_keys = [((cs.DETECTED_GROUP, f"{n + 1:06d}"),)
                         for n in range(len(self.rows))]
        self._refresh()
        return True

    def _ask_examples(self, sets) -> bool:
        """Show three detected sets and ask; False when no one can answer.

        :param sets: ``{set key: {channel: path}}``.
        """
        if _headless():
            return False
        from .channel_sort_dialog import ExampleSetsDialog

        dialog = ExampleSetsDialog(self._source() or "", sets, None, self)
        return dialog.exec() == QDialog.Accepted

    # -- the plan ----------------------------------------------------------

    def _complete_sets(self, detected: bool = False):
        """``{set key: {channel: image}}`` for the rows with every channel.

        :param detected: key every set by its number, as Detect sets does,
            even when the rows came from a regex.
        """
        channels = self._channel_columns()
        rows = [(row, key) for row, key in zip(self.rows, self.row_keys)
                if channels and all(row[c] for c in channels)]
        use_keys = not detected and rows and all(key is not None
                                                 for _r, key in rows)
        sets = {}
        for number, (row, key) in enumerate(rows):
            if not use_keys:
                key = ((cs.DETECTED_GROUP, f"{number + 1:06d}"),)
            sets[key] = {self._channel_of(c): row[c] for c in channels}
        return sets

    def _incomplete_rows(self) -> List[int]:
        """The rows missing an image in some channel column."""
        channels = self._channel_columns()
        return [i for i, row in enumerate(self.rows)
                if any(row[c] is None for c in channels)]

    def _base_folder(self) -> str:
        """The folder the output goes beside: the source, else the files'.

        :returns: a folder, or ``""`` when there is nothing yet.
        """
        source = self._source()
        if source and os.path.isdir(source):
            return source
        folders = sorted({os.path.dirname(p) for row in self.rows
                          for p in row if p})
        if not folders:
            return ""
        common = os.path.commonpath(folders)
        if os.path.dirname(common) == common:
            common = folders[0]
        return common

    def _suggest_output(self) -> None:
        """Propose ``<base>/sorted_channels`` until the user types their own."""
        if not self._output_auto:
            return
        base = self._base_folder()
        self.output_edit.setText(
            cs.unused_folder(base, cs.DEFAULT_DEST_NAME) if base else "")

    def _output_edited(self, _text: str) -> None:
        """The user typed an output folder: stop proposing one.

        :param _text: the new text.
        """
        self._output_auto = False

    def prepare_plan(self, convert: bool = False) -> Optional[cs.SortPlan]:
        """Build the plan from the table, or None while rows are incomplete.

        :param convert: convert RGB images and z-stacks to one plane.
        :returns: the plan, which may still carry problems.
        """
        if not self._channel_columns() or not self.rows or self._incomplete_rows():
            return None
        sets = self._complete_sets()
        masks: Dict[str, Optional[str]] = {}
        roles: Dict[int, str] = {}
        problems = []
        for index, column in enumerate(self.columns):
            if column.kind != "mask":
                continue
            target = self._column_for_channel(column.of_channel)
            if target is None:
                problems.append(tr(
                    "A {role} mask column belongs to channel {n}, which "
                    "there is not.", role=column.role, n=column.of_channel))
                continue
            if column.of_channel in roles:
                problems.append(tr(
                    "Channel {n} has two mask columns; a channel's images "
                    "have one mask each.", n=column.of_channel))
                continue
            roles[column.of_channel] = column.role
            for row in self.rows:
                if row[target] and row[index]:
                    masks[row[target]] = row[index]
        base = self._base_folder()
        plan = cs.build_plan(base, sets, masks=masks, mask_roles=roles,
                             dest=self.output_edit.text().strip() or None,
                             convert=convert)
        plan.problems.extend(problems)
        return plan

    def _ask(self, title: str, text: str) -> bool:
        """Ask a yes/no question; no when no one can answer.

        :param title: the box's title.
        :param text: the question.
        """
        if _headless():
            return False
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _on_apply(self) -> None:
        """Plan, confirm and accept -- or say what is missing."""
        plan = self.prepare_plan()
        if plan is None:
            missing = self._incomplete_rows()
            self.status.setText(tr(
                "{n} row(s) lack an image in some channel column, so they "
                "cannot become Yokogawa sets. Change the regex (Sort by "
                "regex, Auto regex) or use Detect sets, drop the missing "
                "images into their columns, or remove those files.",
                n=len(missing)) if missing else tr(
                "Add images to at least one channel column first."))
            self.custom_edit.setFocus()
            return
        if plan.convertible:
            shown = "\n".join(os.path.basename(p) for p in plan.convertible[:8])
            if self.ask(tr("Convert RGB images and z-stacks?"), tr(
                    "{n} image(s) are not a single grey plane:\n{names}\n\n"
                    "Convert them? RGB images become grey (the mean of their "
                    "colours) and z-stacks become their maximum projection. "
                    "The originals are kept in originals/ of the sorted "
                    "folder.", n=len(plan.convertible), names=shown)):
                plan = self.prepare_plan(convert=True)
        if not plan.ok:
            self.status.setText(plan.summary())
            return
        if not self.ask(tr("Move and merge?"), plan.summary()):
            return
        self.plan = plan
        self.accept()

    # -- drawing -----------------------------------------------------------

    def _caption(self, index: int) -> str:
        """A column's header.

        :param index: the column.
        """
        column = self.columns[index]
        if column.kind == "channel":
            return tr("Channel {n}", n=self._channel_of(index))
        return tr("{role} mask (channel {n})", role=column.role,
                  n=column.of_channel)

    def _rebuild_editors(self) -> None:
        """One small editor per column: its role and channel, and Remove."""
        while self._editors_row.count():
            widget = self._editors_row.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        channels = len(self._channel_columns())
        for index, column in enumerate(self.columns):
            box = QFrame()
            row = QHBoxLayout(box)
            row.setContentsMargins(2, 0, 2, 0)
            row.addWidget(QLabel(self._caption(index)))
            if column.kind == "mask":
                role = QComboBox()
                captions = {"cell": tr("cell"), "nucleus": tr("nucleus"),
                            "pathogen": tr("pathogen"),
                            "organelle": tr("organelle")}
                for value in cs.MASK_ROLES:
                    role.addItem(captions[value], value)
                role.setCurrentIndex(max(0, role.findData(column.role)))
                role.currentIndexChanged.connect(
                    lambda _i, c=column, b=role: self._set_role(c, b))
                row.addWidget(role)
                owner = QComboBox()
                for n in range(1, max(channels, 1) + 1):
                    owner.addItem(tr("of channel {n}", n=n), n)
                owner.setCurrentIndex(max(0, owner.findData(column.of_channel)))
                owner.currentIndexChanged.connect(
                    lambda _i, c=column, b=owner: self._set_owner(c, b))
                row.addWidget(owner)
            remove = QPushButton(tr("Remove"))
            remove.setToolTip(tr("Remove this column and its files from the "
                                 "table."))
            remove.clicked.connect(
                lambda _checked=False, c=column: self.remove_column(
                    self.columns.index(c)))
            row.addWidget(remove)
            self._editors_row.addWidget(box)
        self._editors_row.addStretch(1)

    def _set_role(self, column: _Column, box: QComboBox) -> None:
        """A mask column's role was chosen.

        :param column: the column.
        :param box: its role box.
        """
        column.role = box.currentData()
        self._refresh_table()

    def _set_owner(self, column: _Column, box: QComboBox) -> None:
        """A mask column's channel was chosen: rematch its masks.

        :param column: the column.
        :param box: its channel box.
        """
        column.of_channel = int(box.currentData())
        self._rematch()

    def _refresh(self) -> None:
        """Redraw the column editors, the table and the status line."""
        self._rebuild_editors()
        self._refresh_table()

    # -- the table's view, slots and × ---------------------------------------

    def _mask_of(self) -> Dict[int, int]:
        """``{channel column: its mask column}``, for moving masks along."""
        return {channel: mask for mask, channel in self._partners().items()}

    def _swap_slots(self, from_row: int, from_column: int, to_row: int,
                   to_column: int) -> None:
        """Move one cell to another slot; an occupied slot swaps with it.

        The images' masks move with them. Rows left with nothing are dropped.

        :param from_row: the dragged cell's row.
        :param from_column: its column.
        :param to_row: the slot's row; one past the last makes a new row.
        :param to_column: the slot's column.
        """
        if not (0 <= from_row < len(self.rows)
                and 0 <= from_column < len(self.columns)
                and 0 <= to_column < len(self.columns) and to_row >= 0):
            return
        before = len(self.rows)
        _swap_cells(self.rows, self._mask_of(), (from_row, from_column),
                    (to_row, to_column))
        self.row_keys.extend([None] * (len(self.rows) - before))
        self._drop_empty_rows()
        self._refresh_table()

    def _move_slots(self, cells, anchor, target) -> int:
        """Move several cells together as a block.

        Each cell moves by the offset from ``anchor`` to ``target``; what
        they land on swaps back into the slots they left, and images keep
        their masks. Rows left with nothing are dropped.

        :param cells: ``[[row, column], ...]``.
        :param anchor: ``[row, column]`` the drag started on.
        :param target: ``[row, column]`` dropped on.
        :returns: how many slots moved.
        """
        cells = [(int(r), int(c)) for r, c in cells
                 if 0 <= int(r) < len(self.rows)
                 and 0 <= int(c) < len(self.columns)]
        moves = _block_moves(cells, tuple(anchor), tuple(target),
                             len(self.columns), self._mask_of())
        before = len(self.rows)
        _move_block(self.rows, moves)
        self.row_keys.extend([None] * (len(self.rows) - before))
        self._drop_empty_rows()
        self._refresh_table()
        return len(moves)

    def _set_thumb_size(self, size: int, remember: bool = True) -> None:
        """Resize the table's images live (the Size slider).

        :param size: pixels; clamped to :data:`_THUMB_RANGE`.
        :param remember: store it in the preferences.
        """
        self.thumb_size = _clamp_thumb(size)
        if remember:
            _save_thumb_pref(self.thumb_size)
        self._size_cells()
        self._quality_changed("")

    def _size_cells(self) -> None:
        """Size the rows and columns for the view and the thumbnail size.

        In the image views a column is never narrower than its heading.
        """
        if self.delegate.view == "text":
            self.table.resizeColumnsToContents()
            self.table.resizeRowsToContents()
            return
        header = self.table.horizontalHeader()
        for column in range(self.table.columnCount()):
            self.table.setColumnWidth(column, max(
                self.thumb_size + 24, header.sectionSizeHint(column)))
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, self.thumb_size + 8)

    def _clear_slots(self, cells) -> int:
        """Empty cells (the × or Delete); their images' masks go too.

        :param cells: ``[[row, column], ...]``.
        :returns: how many files left the table.
        """
        mask_of = self._mask_of()
        removed = sum(1 for row, column in cells
                      if _clear_cell(self.rows, mask_of, (row, column)))
        self._drop_empty_rows()
        self._refresh_table()
        return removed

    def _set_view(self, view: str, remember: bool = True) -> None:
        """Show cells as ``text``, ``image`` or ``both``.

        :param view: one of :data:`_VIEWS`.
        :param remember: store it in the preferences.
        """
        view = view if view in _VIEWS else "text"
        self.delegate.view = view
        self.color_button.setEnabled(view == "both")
        self.size_slider.setEnabled(view != "text")
        if remember:
            _save_view_prefs(view, self.delegate.text_color.name())
        self._refresh_table()

    def _set_text_color(self, color, remember: bool = False) -> None:
        """The colour of the names written over the images.

        :param color: a QColor or a colour name such as ``#ffff00``.
        :param remember: store it in the preferences.
        """
        color = QColor(color)
        if not color.isValid():
            color = QColor("white")
        self.delegate.text_color = color
        backdrop = _legible_backdrop(
            color, self.palette().color(QPalette.Window))
        self.color_button.setStyleSheet(
            f"QPushButton {{ color: {color.name()}; border: none; "
            f"background: {backdrop}; border-radius: 3px; "
            f"padding: 2px 4px; }}"
            f"QPushButton:disabled {{ color: {color.name()}80; }}")
        if remember:
            _save_view_prefs(self.delegate.view, color.name())
        self.table.viewport().update()

    def _pick_text_color(self) -> None:
        """Choose the overlay colour in a colour dialog (not headless)."""
        if _headless():
            return
        color = pick_colour(self, self.delegate.text_color, tr("Text colour"))
        if color.isValid():
            self._set_text_color(color, remember=True)

    def _quality_changed(self, _quality: str) -> None:
        """Discard display samples and asynchronously reload at the new quality.

        :param _quality: the newly persisted quality key.
        """
        self._thumb_generation += 1
        for worker in self._thumb_workers:
            worker.stop()
        self._thumb_pending.clear()
        self._thumb_signatures.clear()
        self._thumb_requested.clear()
        self.delegate.pixmaps.clear()
        self.table.viewport().update()
        self._load_thumbnails()

    @staticmethod
    def _thumbnail_signature(path):
        """Identify the current file contents for cache invalidation.

        :param path: image or mask file.
        :returns: modification time and size, or None for an unavailable file.
        """
        try:
            stat = os.stat(path)
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def _load_thumbnails(self, *_args) -> None:
        """Read only visible rows off-thread, with one bounded batch at a time."""
        if (self._thumb_closed or self.delegate.view == "text"
                or not self.rows or self._thumb_workers):
            return
        from ..hidpi import device_ratio

        ratio = device_ratio(self.table)
        previous_ratio, self._thumb_render_ratio = self._thumb_render_ratio, ratio
        if previous_ratio is not None and previous_ratio != ratio:
            self._quality_changed("")
            return
        viewport = self.table.viewport()
        first = max(0, self.table.rowAt(0))
        last = self.table.rowAt(max(0, viewport.height() - 1))
        if last < 0:
            last = len(self.rows) - 1
        first_col = max(0, self.table.columnAt(0))
        last_col = self.table.columnAt(max(0, viewport.width() - 1))
        if last_col < 0:
            last_col = len(self.columns) - 1
        paths = list(dict.fromkeys(
            p for row in self.rows[first:last + 1]
            for p in row[first_col:last_col + 1] if p))
        wanted = []
        for path in paths:
            signature = self._thumbnail_signature(path)
            if (path in self.delegate.pixmaps
                    and self._thumb_signatures.get(path) == signature):
                self.delegate.pixmaps.move_to_end(path)
                continue
            self.delegate.pixmaps.pop(path, None)
            self._thumb_requested[path] = signature
            wanted.append(path)
        if not wanted:
            return
        from .channel_sort_dialog import _ThumbWorker

        self._thumb_pending.update(wanted)
        worker = _ThumbWorker("", wanted, None, self)
        worker.generation = self._thumb_generation
        worker.ready.connect(self._take_thumbnail)
        worker.finished.connect(self._thumbnail_worker_finished)
        self._thumb_workers.append(worker)
        worker.start()

    def _thumbnail_worker_finished(self) -> None:
        """Release completed workers and service the latest quality/viewport."""
        worker = self.sender()
        if worker in self._thumb_workers:
            self._thumb_workers.remove(worker)
        if worker is not None:
            worker.deleteLater()
        if (not self._thumb_closed and worker is not None
                and worker.generation != self._thumb_generation):
            QTimer.singleShot(0, self._load_thumbnails)

    def _take_thumbnail(self, path: str, image, _mask) -> None:
        """Cache a current thumbnail, with a 64 MiB/512-entry display budget.

        :param path: the image or mask file.
        :param image: its 2-D uint8 source-sampled thumbnail, or None.
        :param _mask: paired overlay, unused in this separate-column table.
        """
        worker = self.sender()
        if (self._thumb_closed or (worker is not None
                and worker.generation != self._thumb_generation)):
            return
        self._thumb_pending.discard(path)
        if image is None:
            return
        signature = self._thumb_requested.get(path)
        if signature != self._thumbnail_signature(path):
            return
        array = np.ascontiguousarray(image)
        qimage = QImage(array.data, array.shape[1], array.shape[0],
                        array.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage.copy())
        # Keep only useful display pixels; source reads still use the selected
        # quality. This bounds a large grid without retaining full-size fields.
        from ..hidpi import device_ratio

        display_cap = max(1, round((self.thumb_size + 24) * device_ratio(self.table)))
        if max(pixmap.width(), pixmap.height()) > display_cap:
            pixmap = pixmap.scaled(display_cap, display_cap, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
        self.delegate.pixmaps[path] = pixmap
        self._thumb_signatures[path] = signature
        cache = self.delegate.pixmaps
        budget = 64 * 1024 * 1024
        total = sum(p.width() * p.height() * 4 for p in cache.values())
        while len(cache) > 512 or (total > budget and len(cache) > 1):
            old, pixmap = cache.popitem(last=False)
            total -= pixmap.width() * pixmap.height() * 4
            self._thumb_signatures.pop(old, None)
            self._thumb_requested.pop(old, None)
        self.table.viewport().update()

    def _stop_thumbs(self) -> None:
        """Stop polling and end thumbnail threads before destroying the popup."""
        self._thumb_closed = True
        self._thumb_poll.stop()
        for worker in self._thumb_workers:
            if worker.isRunning():
                worker.stop()
                worker.wait(5000)

    def _refresh_table(self) -> None:
        """Fill the table from the rows, empty cells flagged."""
        from PySide6.QtGui import QBrush, QColor

        self.table.clear()
        self.table.setColumnCount(len(self.columns))
        self.table.setRowCount(len(self.rows))
        self.table.setHorizontalHeaderLabels(
            [self._caption(i) for i in range(len(self.columns))])
        channels = set(self._channel_columns())
        flag = QBrush(QColor(200, 60, 60, 60))
        for r, row in enumerate(self.rows):
            for c, path in enumerate(row):
                item = SortableTableItem(
                    os.path.basename(path) if path else "—")
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                              | Qt.ItemIsDropEnabled
                              | (Qt.ItemIsDragEnabled if path else Qt.NoItemFlags))
                if path:
                    item.setData(Qt.UserRole, path)
                    item.setToolTip(path + "\n" + tr(
                        "Drag it to another slot to move it (an occupied "
                        "slot swaps); the \u00d7 or Delete removes it."))
                else:
                    item.setToolTip(tr("Nothing matched here"))
                    if c in channels:
                        item.setBackground(flag)
                self.table.setItem(r, c, item)
        self._size_cells()
        if self.delegate.view != "text":
            self._load_thumbnails()
        # The drop hint lives in the table while it is empty; the "new
        # channel" strip only once there is something beside it.
        self.new_zone.setVisible(not self.table._is_empty())
        self.table.viewport().update()
        missing = len(self._incomplete_rows())
        lines = [tr("{rows} row(s) in {channels} channel(s) and {masks} "
                    "mask column(s); {missing} row(s) incomplete.",
                    rows=len(self.rows), channels=len(channels),
                    masks=len(self.columns) - len(channels), missing=missing)]
        if self._note:
            lines.append(self._note)
        self.status.setText("\n".join(lines))

    def _browse(self) -> None:
        """Pick the source folder."""
        folder = QFileDialog.getExistingDirectory(
            self, tr("Pick the source folder"), self._source() or os.getcwd())
        if folder:
            self.source_edit.setText(folder)
