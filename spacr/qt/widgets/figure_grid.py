"""A grid of search figures that fills as the search runs.

:class:`FigureQueue` is a one-at-a-time gallery: it shows the figure you
navigated to. That is the right shape for a pipeline that produces a dozen
plots over an hour, and the wrong one for a hyperparameter search, where
the whole point is to compare fifty embeddings against each other and stop
early when the range is obviously wrong.

This module is the comparison view. Figures land as they are produced, the
grid reflows to whatever space the container has, and where a cell SITS
says which parameter values produced it.

Two halves, deliberately separated:

* :func:`reflow_shape` and :func:`axis_layout` are pure. They decide how
  many columns fit and which cell a trial belongs in, and they are tested
  without a display. Nearly all of the behaviour worth defending is here.
* :class:`SearchFigureGrid` is the Qt shell around them.

Rendering never happens on the GUI thread. The figure arrives as a
pre-rendered PNG from the worker that made it, exactly as
:class:`FigureQueue` receives one; a grid of fifty PDFs rasterised in the
paint path would freeze the application, which is the one thing it must
never do.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

from ..preferences import get_figure_format, get_figure_png_dpi
from ..theme import active_palette, css_color, make_transparent

#: A cell narrower than this is a coloured smudge, not a figure. Past it the
#: grid stops adding columns and scrolls instead. Measured against the UMAP
#: panels, whose axis labels stop being legible just under this width.
MIN_CELL_PX = 148

#: Cells are drawn at the aspect ratio matplotlib figures default to.
DEFAULT_CELL_ASPECT = 4.0 / 3.0

#: Resize is a stream of events, and re-laying out on every one of them is
#: how a smooth drag turns into a stutter. Same reason and same value as
#: FigureQueue's re-render debounce.
REFLOW_DEBOUNCE_MS = 220


def reflow_shape(count: int,
                 width: int,
                 height: int,
                 *,
                 min_cell: int = MIN_CELL_PX,
                 aspect: float = DEFAULT_CELL_ASPECT,
                 spacing: int = 6,
                 ) -> Tuple[int, int, int]:
    """How many columns and rows fit ``count`` figures in ``width`` x ``height``.

    The grid prefers to fill the container in BOTH directions rather than
    to make one long row: with nine figures in a square panel the useful
    answer is 3x3, not 9x1, because comparing nine embeddings side by side
    is the entire reason the view exists.

    Columns are capped by ``min_cell`` and by ``count``. When the result
    does not fit vertically the grid scrolls -- it does not shrink cells
    below the size at which a figure stops being readable, because an
    unreadable grid is not a cheaper version of a readable one.

    :param count: how many figures are in the grid.
    :param width: usable container width in pixels.
    :param height: usable container height in pixels.
    :param min_cell: narrowest cell worth drawing.
    :param aspect: cell width divided by cell height.
    :param spacing: gap between cells in pixels.
    :returns: ``(columns, rows, cell_width)``. ``(0, 0, 0)`` for no figures.
    """
    if count <= 0:
        return 0, 0, 0
    width = max(0, int(width))
    height = max(0, int(height))
    min_cell = max(1, int(min_cell))
    aspect = float(aspect) if aspect and aspect > 0 else DEFAULT_CELL_ASPECT
    spacing = max(0, int(spacing))

    # The most columns the width allows at all, ignoring how many figures
    # there are. One column is always offered: a container too narrow for a
    # readable cell still has to show something.
    fitting = max(1, (width + spacing) // (min_cell + spacing))
    columns_cap = int(min(count, fitting))

    best: Optional[Tuple[int, int, int]] = None
    best_score: Optional[Tuple[float, int]] = None
    for columns in range(1, columns_cap + 1):
        rows = math.ceil(count / columns)
        cell_w = (width - spacing * (columns - 1)) / columns
        if cell_w < 1:
            continue
        cell_h = cell_w / aspect
        used_h = cell_h * rows + spacing * (rows - 1)
        # Prefer the shape that comes closest to filling the height without
        # overflowing it; among those, the larger cell.
        overflow = used_h > height and height > 0
        waste = (abs(height - used_h) / max(1.0, float(height))
                 if height > 0 else 0.0)
        score = (waste + (10.0 if overflow else 0.0), -int(cell_w))
        if best_score is None or score < best_score:
            best_score = score
            best = (columns, rows, int(cell_w))
    if best is None:
        return 1, count, max(1, width)
    return best


def axis_layout(coordinates: Sequence[Mapping[str, Any]],
                parameters: Sequence[str],
                ) -> Tuple[List[str], List[Any], List[Tuple[int, int]]]:
    """Place each trial on a grid whose axes are the searched parameters.

    With ONE searched parameter the grid is a single row ordered by it.
    With two it is the familiar table: one parameter across, the other
    down. With more than two there is no honest two-dimensional picture, so
    the widest parameter goes across and every distinct combination of the
    rest gets its own row -- a small multiple of small multiples. That is
    the arrangement that keeps the promise the position makes: two cells in
    the same row differ in exactly one parameter.

    Trials whose coordinates repeat (a resumed search, a duplicate) share a
    cell; the caller decides which figure wins.

    :param coordinates: one mapping of parameter values per figure, in
        arrival order.
    :param parameters: the searched parameter names. Order decides which
        axis is which; the caller passes them in the order the user chose.
    :returns: ``(row_parameters, column_values, cells)`` where ``cells[i]``
        is ``(row, column)`` for ``coordinates[i]``. ``row_parameters`` is
        empty when there is only one axis.
    """
    coordinates = list(coordinates)
    names = [p for p in parameters if any(p in c for c in coordinates)]
    if not coordinates:
        return [], [], []
    if not names:
        # No axes to speak of: arrival order, one long row. The widget
        # reflows it; there is nothing meaningful to say about position.
        return [], [], [(0, i) for i in range(len(coordinates))]

    def distinct(name: str) -> List[Any]:
        seen: List[Any] = []
        for coord in coordinates:
            value = coord.get(name)
            if value not in seen:
                seen.append(value)
        return sorted(seen, key=_sort_key)

    widths = {name: len(distinct(name)) for name in names}
    column_name = max(names, key=lambda n: (widths[n], -names.index(n)))
    row_names = [n for n in names if n != column_name]

    column_values = distinct(column_name)
    column_index = {value: i for i, value in enumerate(column_values)}

    row_keys: List[Tuple[Any, ...]] = []
    for coord in coordinates:
        key = tuple(coord.get(n) for n in row_names)
        if key not in row_keys:
            row_keys.append(key)
    row_keys.sort(key=lambda key: tuple(_sort_key(v) for v in key))
    row_index = {key: i for i, key in enumerate(row_keys)}

    cells = [
        (row_index[tuple(coord.get(n) for n in row_names)],
         column_index.get(coord.get(column_name), 0))
        for coord in coordinates
    ]
    return row_names, column_values, cells


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Order numbers numerically, everything else as text.

    A grid axis mixing 5 and "spectral" has to be ordered somehow, and
    comparing an int with a str raises. Numbers first, then text.
    """
    if isinstance(value, bool):
        return (1, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def cell_caption(coordinate: Mapping[str, Any],
                 parameters: Sequence[str]) -> str:
    """A one-line label naming the parameter values behind one figure."""
    parts = []
    for name in parameters:
        if name not in coordinate:
            continue
        value = coordinate[name]
        if isinstance(value, float):
            parts.append(f"{name}={value:g}")
        else:
            parts.append(f"{name}={value}")
    return "  ".join(parts)


@dataclass
class GridCell:
    """One figure in the grid, with where it came from."""

    pixmap: Optional[QPixmap] = None
    coordinate: Dict[str, Any] = field(default_factory=dict)
    caption: str = ""
    source_path: str = ""


class SearchFigureGrid(QWidget):
    """Figures on a reflowing grid, filling as a search produces them.

    :param parameters: the searched parameter names, in the order that
        decides the axes. May be set later with :meth:`set_parameters`.
    :param parent: optional Qt parent.
    :ivar cell_clicked: emitted with the index of a clicked figure.
    """

    cell_clicked = Signal(int)

    def __init__(self,
                 parameters: Optional[Sequence[str]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SearchFigureGrid")
        self._parameters: List[str] = list(parameters or [])
        self._cells: List[GridCell] = []
        self._labels: List[QLabel] = []
        self._columns = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("SearchFigureGridScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._page = QWidget(self._scroll)
        self._page.setObjectName("SearchFigureGridPage")
        # An anonymous QWidget inherits the blanket `QWidget { background:
        # bg }` rule and paints the window colour as a solid rectangle over
        # whatever is behind it. See INVARIANTS 1 and 3.
        make_transparent(self._page)
        self._grid = QGridLayout(self._page)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._grid.setSpacing(6)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._page)
        make_transparent(self._scroll.viewport())
        outer.addWidget(self._scroll, 1)

        self._empty = QLabel("No figures yet.", self)
        self._empty.setObjectName("SearchFigureGridEmpty")
        self._empty.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._empty, 1)

        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.setInterval(REFLOW_DEBOUNCE_MS)
        self._reflow_timer.timeout.connect(self.relayout)
        self._scroll.viewport().installEventFilter(self)
        self._update_empty()

    # -- content -----------------------------------------------------------

    def set_parameters(self, parameters: Sequence[str]) -> None:
        """Set the parameters whose values place a figure, then re-lay out."""
        self._parameters = list(parameters or [])
        self.relayout()

    def parameters(self) -> List[str]:
        """The parameters currently deciding the axes."""
        return list(self._parameters)

    def add_figure(self,
                   png_path: str,
                   coordinate: Optional[Mapping[str, Any]] = None) -> int:
        """Add one already-rendered figure.

        The PNG is produced by whoever ran the trial, on the worker thread
        that ran it. This method only loads and places it -- a search that
        rendered its figures here would render them on the GUI thread.

        :param png_path: path to the rendered image.
        :param coordinate: the parameter values behind this figure.
        :returns: the index of the new cell.
        """
        coordinate = dict(coordinate or {})
        pixmap = QPixmap(str(png_path))
        cell = GridCell(
            pixmap=(None if pixmap.isNull() else pixmap),
            coordinate=coordinate,
            caption=cell_caption(coordinate, self._parameters or
                                 sorted(coordinate)),
            source_path=str(png_path),
        )
        self._cells.append(cell)
        self.relayout()
        return len(self._cells) - 1

    def count(self) -> int:
        """How many figures are in the grid."""
        return len(self._cells)

    def columns(self) -> int:
        """The current column count. 0 while empty."""
        return self._columns

    def figure_path(self, index: int) -> str:
        """The best file for cell ``index`` -- the PDF when one exists.

        `render_figure_to_png` writes a sibling `.pdf` when the figure
        format preference is PDF, and the grid necessarily DISPLAYS the
        PNG, because a PDF cannot be painted into a label. So the file a
        user should be handed is not always the one on screen.
        """
        if not 0 <= index < len(self._cells):
            return ""
        png = Path(self._cells[index].source_path)
        pdf = png.with_suffix(".pdf")
        return str(pdf if pdf.is_file() else png)

    def coordinates(self) -> List[Dict[str, Any]]:
        """The parameter values behind each figure, in arrival order."""
        return [dict(cell.coordinate) for cell in self._cells]

    def clear(self) -> None:
        """Drop every figure."""
        self._cells.clear()
        self.relayout()

    # -- layout ------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Debounce reflow while the container is being resized."""
        if obj is self._scroll.viewport() and event.type() == QEvent.Resize:
            self._reflow_timer.start()
        return super().eventFilter(obj, event)

    def relayout(self) -> None:
        """Rebuild the grid for the current size and figure count."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._labels = []
        self._update_empty()
        if not self._cells:
            self._columns = 0
            return

        viewport = self._scroll.viewport()
        width = max(1, viewport.width() - 12)
        height = max(1, viewport.height() - 12)
        placed = self._placement(len(self._cells))
        columns = max(1, max((column for _row, column in placed), default=0) + 1)
        _cols, _rows, cell_w = reflow_shape(
            len(self._cells), width, height,
            spacing=self._grid.spacing())
        if not self._parameters:
            columns = max(1, _cols)
            placed = [(index // columns, index % columns)
                      for index in range(len(self._cells))]
        else:
            # The axes decide the columns; the container decides how wide
            # each one is. A search space is not free to be reshaped to fit
            # a window -- moving a cell would change what it claims.
            cell_w = max(
                1,
                (width - self._grid.spacing() * (columns - 1)) // columns)
        self._columns = columns

        for index, (row, column) in enumerate(placed):
            label = self._make_label(index, cell_w)
            self._grid.addWidget(label, row, column)
            self._labels.append(label)

    def _placement(self, count: int) -> List[Tuple[int, int]]:
        """Grid coordinates for each figure."""
        if not self._parameters:
            return [(0, index) for index in range(count)]
        _rows, _values, cells = axis_layout(
            [cell.coordinate for cell in self._cells], self._parameters)
        return cells or [(0, index) for index in range(count)]

    def _make_label(self, index: int, cell_w: int) -> QLabel:
        """One cell: the figure, scaled, with its coordinates under it."""
        cell = self._cells[index]
        label = QLabel(self._page)
        label.setObjectName(f"SearchFigureCell_{index}")
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setToolTip(cell.caption or cell.source_path)
        if cell.pixmap is not None and not cell.pixmap.isNull():
            label.setPixmap(cell.pixmap.scaled(
                QSize(cell_w, int(cell_w / DEFAULT_CELL_ASPECT)),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # A figure that failed to render is a missing result, not a
            # missing widget: the cell stays so the grid keeps its shape and
            # says which configuration produced nothing.
            label.setText(cell.caption or "no figure")
            label.setWordWrap(True)
        label.mouseReleaseEvent = (
            lambda _event, i=index: self.cell_clicked.emit(i))
        return label

    def _update_empty(self) -> None:
        """Show the placeholder only while there is nothing to compare."""
        empty = not self._cells
        self._empty.setVisible(empty)
        self._scroll.setVisible(not empty)

    # -- preferences -------------------------------------------------------

    @staticmethod
    def figure_format() -> str:
        """The figure format from Preferences, not from a second setting.

        Instruction 35 is explicit that this must not grow its own control:
        a user who sets PDF once should get PDF everywhere.
        """
        return get_figure_format()

    @staticmethod
    def figure_dpi() -> int:
        """The PNG resolution from Preferences."""
        return get_figure_png_dpi()
