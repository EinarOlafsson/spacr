"""Tabulate — drag columns onto rows and columns, get a table with its n.

The chrome over :mod:`spacr.qt.widgets.pivot_spec`. Everything about *what the
table contains* is in there; this module is three drop wells, a stack of
aggregation tick boxes and a grid.

Reused rather than rebuilt
--------------------------
The column well is
:class:`spacr.qt.widgets.graph_builder.ColumnWell` and the drag payload is
:data:`~spacr.qt.widgets.graph_builder.COLUMN_MIME` — the same list, the same
classification of what is worth offering, and the same payload type, so a
column dragged in the Graph Builder and a column dragged here are the same
gesture. The drop targets are different, and that is the only reason there is a
widget in here at all: a Graph Builder zone holds exactly one column, and a
pivot axis holds a *nest* of them, outermost first.

What the grid shows, and what it refuses to
-------------------------------------------
**Every cell prints its n**, under the statistics, whether or not ``n`` was
ticked. An aggregate over 4 objects and one over 4 000 are three digits either
way, and the only thing that tells them apart is the number this panel refuses
to hide. Cells at or below :data:`~spacr.qt.widgets.pivot_spec.LOW_N` are drawn
muted, so a table can be scanned for "which of these am I allowed to believe".

**An empty cell is blank, not zero.** The rule and its reasoning are in
:mod:`spacr.qt.widgets.pivot_spec`; the panel's part is not to paper over it
with a ``0`` because a ``QTableWidgetItem`` would rather have a number.

Plotting the result
-------------------
There is no chart in here. :meth:`PivotPanel.long_frame` hands the Graph
Builder a tidy frame — one row per non-empty cell, one column per statistic —
and the Graph Builder does what it already does. A second plotting
implementation would be a second set of scale, facet and colour rules to keep
in step with the first.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from ..theme import (RADIUS, SPACING, active_palette, font_px, mark_surface,
                     register_widget_qss)
from .graph_builder import COLUMN_MIME, ColumnWell
from .pivot_spec import (
    AGGREGATION_LABELS, AGGREGATIONS, COUNT_ONLY, LOW_N, MEAN, N, SD,
    WELL_HIERARCHY, PivotError, PivotResult, PivotSpec, format_value, pivot,
)

LOG = logging.getLogger("spacr.qt.pivot")

__all__ = ["AXIS_ROWS", "AXIS_COLS", "AXIS_VALUES", "AXIS_LABELS",
           "DropWell", "PivotTable", "PivotPanel"]

AXIS_ROWS = "rows"
AXIS_COLS = "cols"
AXIS_VALUES = "values"

AXIS_LABELS = {
    AXIS_ROWS: "Rows",
    AXIS_COLS: "Columns",
    AXIS_VALUES: "Values",
}

AXIS_HINTS = {
    AXIS_ROWS: "Keys nesting down the rows, outermost first — plateID, then "
               "rowID, then columnID is the usual one. Drop order is nesting "
               "order.",
    AXIS_COLS: "Keys nesting across the columns. Keep this axis short; a "
               "table is read down the page.",
    AXIS_VALUES: "The measurements to aggregate. Leave it empty for a "
                 "contingency table of counts.",
}

#: Recomputes are coalesced this long, so ticking three aggregations in a row
#: costs one pass over the frame rather than three.
DEBOUNCE_MS = 150

#: Beyond this many cells the grid is built but not populated cell by cell:
#: a QTableWidget with a million items is minutes of construction for a table
#: nobody will scroll. Export is the answer, and the notice says so.
MAX_RENDERED_CELLS = 20_000


class DropWell(QWidget):
    """One pivot axis: an ordered list of columns, filled by dropping.

    A list rather than a single-slot zone, because a pivot axis is a *nest*:
    ``plateID`` then ``rowID`` then ``columnID`` is three keys on one axis and
    the order is the hierarchy. Drop order is that order, which is the only
    rule that needs no explanation at the moment a user drags something.

    Removing is Delete, Backspace or a double-click. There is no drag-out: a
    column dragged from here to another well would have to decide whether it
    was a move or a copy, and getting that wrong silently loses an axis.
    """

    changed = Signal()

    def __init__(self, axis: str, parent=None):
        super().__init__(parent)
        if axis not in AXIS_LABELS:
            raise ValueError(f"unknown pivot axis {axis!r}")
        self.axis = axis
        self.setObjectName("PivotDropWell")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["xs"])
        title = QLabel(AXIS_LABELS[axis], self)
        title.setObjectName("PivotWellName")
        head.addWidget(title, 1)
        clear = QPushButton("×", self)
        clear.setObjectName("PivotWellClear")
        clear.setFixedWidth(20)
        clear.setToolTip(f"Empty the {AXIS_LABELS[axis]} well")
        clear.clicked.connect(self.clear)
        head.addWidget(clear)
        outer.addLayout(head)

        self._list = _AxisList(self)
        self._list.setObjectName("PivotWellList")
        self._list.setToolTip(AXIS_HINTS[axis])
        self._list.dropped.connect(self._on_dropped)
        self._list.remove_requested.connect(self._on_remove)
        outer.addWidget(self._list, 1)

    # -- state ------------------------------------------------------------
    def columns(self) -> Tuple[str, ...]:
        return tuple(self._list.item(i).data(Qt.UserRole)
                     for i in range(self._list.count()))

    def set_columns(self, columns) -> None:
        """Replace the contents. Silent when nothing changes — the panel
        recomputes on every emission and a redundant pass over a million rows
        is a visible pause for no visible difference."""
        wanted = tuple(str(c) for c in columns if c)
        if wanted == self.columns():
            return
        self._list.clear()
        for name in wanted:
            self._add(name)
        self.changed.emit()

    def clear(self) -> None:
        self.set_columns(())

    def _add(self, name: str) -> None:
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        item.setToolTip(f"{name}\nDelete or double-click to remove it.")
        self._list.addItem(item)

    def _on_dropped(self, name: str) -> None:
        if name and name not in self.columns():
            self._add(name)
            self.changed.emit()

    def _on_remove(self, row: int) -> None:
        if 0 <= row < self._list.count():
            self._list.takeItem(row)
            self.changed.emit()


class _AxisList(QListWidget):
    """The list inside a :class:`DropWell`: takes :data:`COLUMN_MIME`."""

    dropped = Signal(str)
    remove_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setMaximumHeight(84)
        self.itemDoubleClicked.connect(
            lambda item: self.remove_requested.emit(self.row(item)))

    def _accepts(self, event) -> bool:
        return event.mimeData() is not None and \
            event.mimeData().hasFormat(COLUMN_MIME)

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt name
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt name
        if not self._accepts(event):
            event.ignore()
            return
        raw = bytes(event.mimeData().data(COLUMN_MIME)).decode("utf-8")
        if raw:
            self.dropped.emit(raw)
        event.acceptProposedAction()

    def keyPressEvent(self, event):  # noqa: N802 - Qt name
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            row = self.currentRow()
            if row >= 0:
                self.remove_requested.emit(row)
                return
        super().keyPressEvent(event)


class PivotTable(QTableWidget):
    """The grid. Renders a :class:`~spacr.qt.widgets.pivot_spec.PivotResult`.

    One column per row key so the table can be copied out whole, then one per
    column-level combination. Every populated cell ends with its ``n``; every
    empty one is blank.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PivotTable")
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.setWordWrap(True)
        self._result: Optional[PivotResult] = None
        self._truncated = 0

    @property
    def result(self) -> Optional[PivotResult]:
        return self._result

    @property
    def truncated_cells(self) -> int:
        """Cells the grid refused to build. Non-zero means export, not scroll."""
        return self._truncated

    def cell_text(self, row: int, col: int) -> str:
        """What is actually painted in a body cell — for a test, and for a
        caller that wants the string rather than the float."""
        item = self.item(row, col + len(self._header_offset_keys()))
        return item.text() if item is not None else ""

    def _header_offset_keys(self) -> Tuple[str, ...]:
        result = self._result
        if result is None:
            return ()
        return result.row_keys or ("rows",)

    def set_result(self, result: Optional[PivotResult]) -> None:
        self._result = result
        self.clear()
        self._truncated = 0
        if result is None:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        palette = active_palette()
        key_columns = self._header_offset_keys()
        n_rows, n_cols = result.shape
        self.setRowCount(n_rows)
        self.setColumnCount(len(key_columns) + n_cols)
        headers = list(key_columns) + [
            (result.col_label(c) or "all") for c in range(n_cols)]
        self.setHorizontalHeaderLabels(headers)

        if result.n_cells > MAX_RENDERED_CELLS:
            # Build the shape but not the contents: a QTableWidget with a
            # million items takes minutes to construct, for a table nobody is
            # going to scroll to the end of.
            self._truncated = result.n_cells
            note = QTableWidgetItem(
                f"{result.n_cells:,} cells is past the {MAX_RENDERED_CELLS:,} "
                f"this grid draws — narrow the axes, or export the CSV.")
            self.setRowCount(1)
            self.setColumnCount(1)
            self.setHorizontalHeaderLabels(["too large to draw"])
            self.setItem(0, 0, note)
            return

        low_font = QFont()
        low_font.setItalic(True)
        for r in range(n_rows):
            for i, key in enumerate(key_columns):
                text = (result.row_levels[r][i] if result.row_keys else "all")
                item = QTableWidgetItem(str(text))
                item.setToolTip(f"{key} = {text}")
                self.setItem(r, i, item)
            for c in range(n_cols):
                item = QTableWidgetItem(self._body_text(result, r, c))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setToolTip(self._tooltip(result, r, c))
                smallest = self._smallest_n(result, r, c)
                if smallest is not None and smallest <= LOW_N:
                    item.setFont(low_font)
                    item.setForeground(
                        QBrush(QColor(str(palette["warning"]))))
                self.setItem(r, len(key_columns) + c, item)

    # -- one cell ---------------------------------------------------------
    @staticmethod
    def _smallest_n(result: PivotResult, row: int, col: int) -> Optional[int]:
        counts = [result.n_at(value, row, col)
                  for value in (result.spec.values or (COUNT_ONLY,))]
        found = [c for c in counts if c is not None]
        return min(found) if found else None

    @staticmethod
    def _body_text(result: PivotResult, row: int, col: int) -> str:
        """The statistics, then the n. Blank when the combination has no rows.

        ``n`` is printed whether or not it was ticked and always last, because
        it is the qualifier on everything above it.
        """
        if result.is_empty(row, col):
            return ""
        multiple = len(result.spec.values) > 1
        lines: List[str] = []
        for value in (result.spec.values or (COUNT_ONLY,)):
            prefix = f"{value} " if multiple else ""
            for agg in (result.spec.aggs if result.spec.values else (N,)):
                if agg == N:
                    continue
                lines.append(
                    f"{prefix}{agg} {format_value(result.value_at(value, agg, row, col))}")
            count = result.n_at(value, row, col)
            lines.append(f"{prefix}n {0 if count is None else count:,}")
        return "\n".join(lines)

    @staticmethod
    def _tooltip(result: PivotResult, row: int, col: int) -> str:
        where = " · ".join(p for p in (result.row_label(row),
                                       result.col_label(col)) if p) or "all"
        if result.is_empty(row, col):
            return (f"{where}\nNo objects here at all — this combination was "
                    f"not measured, or nothing survived the filter. It is "
                    f"blank rather than zero on purpose.")
        lines = [where, f"{int(result.sizes[row, col]):,} source row(s)"]
        for value, agg in result.layer_keys:
            name = f"{agg}({value})" if value else "n"
            lines.append(
                f"{name} = "
                f"{format_value(result.value_at(value, agg, row, col)) or '—'}")
        smallest = PivotTable._smallest_n(result, row, col)
        if smallest is not None and smallest <= LOW_N:
            lines.append(f"n ≤ {LOW_N}: read this cell as an anecdote.")
        return "\n".join(lines)


class PivotPanel(QWidget):
    """The well, the three axes, the aggregations and the grid."""

    #: Emitted after every successful pivot.
    computed = Signal(object)
    #: Emitted with :meth:`long_frame` when the user asks for a chart.
    plot_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PivotPanel")
        self._frame: Optional[pd.DataFrame] = None
        self._result: Optional[PivotResult] = None
        self._building = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        shelf = QWidget(self)
        shelf.setObjectName("PivotShelf")
        shelf_layout = QVBoxLayout(shelf)
        shelf_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                        SPACING["sm"], SPACING["sm"])
        shelf_layout.setSpacing(SPACING["sm"])

        self.well = ColumnWell(shelf)
        shelf_layout.addWidget(self.well, 1)

        self.wells: Dict[str, DropWell] = {}
        for axis in (AXIS_ROWS, AXIS_COLS, AXIS_VALUES):
            well = DropWell(axis, shelf)
            well.changed.connect(self._on_axis_changed)
            self.wells[axis] = well
            shelf_layout.addWidget(well)

        preset = QPushButton("Plate / row / column", shelf)
        preset.setToolTip(
            "Put the well hierarchy on the rows — what a plate summary is")
        preset.clicked.connect(self.use_well_hierarchy)
        shelf_layout.addWidget(preset)

        aggs = QGridLayout()
        aggs.setContentsMargins(0, 0, 0, 0)
        aggs.setSpacing(SPACING["xs"])
        self._agg_boxes: Dict[str, QCheckBox] = {}
        for i, agg in enumerate(AGGREGATIONS):
            box = QCheckBox(agg, shelf)
            box.setToolTip(AGGREGATION_LABELS[agg])
            box.setChecked(agg in (N, MEAN, SD))
            if agg == N:
                # n is not a choice. Every cell carries it, and a table where
                # the user could turn it off is a table where a mean over four
                # objects looks like a mean over four thousand.
                box.setEnabled(False)
            box.toggled.connect(self._on_axis_changed)
            self._agg_boxes[agg] = box
            aggs.addWidget(box, i // 2, i % 2)
        shelf_layout.addLayout(aggs)

        self._quantile = QDoubleSpinBox(shelf)
        self._quantile.setRange(0.0, 1.0)
        self._quantile.setSingleStep(0.05)
        self._quantile.setDecimals(3)
        self._quantile.setValue(0.75)
        self._quantile.setToolTip("Which quantile, when quantile is ticked")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Quantile at", shelf))
        row.addWidget(self._quantile, 1)
        self._quantile.valueChanged.connect(self._on_axis_changed)
        shelf_layout.addLayout(row)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(SPACING["xs"])
        self._export = QPushButton("Export CSV…", shelf)
        self._export.setToolTip("Write the table as it is shown")
        self._export.clicked.connect(self.export_csv)
        buttons.addWidget(self._export)
        self._plot = QPushButton("Plot this table", shelf)
        self._plot.setToolTip(
            "Hand the summary to the Graph Builder — one row per cell, one "
            "column per statistic")
        self._plot.clicked.connect(self._on_plot)
        buttons.addWidget(self._plot)
        shelf_layout.addLayout(buttons)
        splitter.addWidget(shelf)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])
        self.table = PivotTable(right)
        # The shelf half of the splitter has `PivotShelf` for a surface;
        # the grid is the other half and sits straight on the page.
        mark_surface(self.table)
        right_layout.addWidget(self.table, 1)
        self.notice = QLabel("", right)
        self.notice.setObjectName("PivotNotice")
        self.notice.setWordWrap(True)
        right_layout.addWidget(self.notice)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.recompute)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Point the panel at a table.

        Axis columns the new table does not have are dropped rather than
        carried over — a pivot half-resolved against the wrong frame would
        group by fewer keys than the wells claim.
        """
        self._frame = frame
        self.well.set_frame(frame)
        if frame is not None:
            for well in self.wells.values():
                well.set_columns([c for c in well.columns()
                                  if c in frame.columns])
        self.recompute()

    @property
    def result(self) -> Optional[PivotResult]:
        return self._result

    def spec(self) -> PivotSpec:
        """The spec the wells and tick boxes currently describe."""
        aggs = [agg for agg in AGGREGATIONS
                if self._agg_boxes[agg].isChecked()]
        return PivotSpec(
            rows=self.wells[AXIS_ROWS].columns(),
            cols=self.wells[AXIS_COLS].columns(),
            values=self.wells[AXIS_VALUES].columns(),
            aggs=tuple(aggs), quantile=self._quantile.value())

    def set_spec(self, spec: PivotSpec) -> None:
        """Push a whole spec in — restoring a saved table, or a preset."""
        self._building = True
        try:
            self.wells[AXIS_ROWS].set_columns(spec.rows)
            self.wells[AXIS_COLS].set_columns(spec.cols)
            self.wells[AXIS_VALUES].set_columns(spec.values)
            for agg, box in self._agg_boxes.items():
                box.setChecked(agg in spec.aggs)
            self._quantile.setValue(spec.quantile)
        finally:
            self._building = False
        self.recompute()

    def use_well_hierarchy(self) -> None:
        """The preset: plate / row / column down the rows."""
        frame = self._frame
        if frame is None:
            return
        self.wells[AXIS_ROWS].set_columns(
            [c for c in WELL_HIERARCHY if c in frame.columns])

    def long_frame(self) -> pd.DataFrame:
        """The tidy summary — what :attr:`plot_requested` carries."""
        if self._result is None:
            return pd.DataFrame()
        return self._result.to_long()

    # -- computing ---------------------------------------------------------
    def recompute(self) -> Optional[PivotResult]:
        """Rebuild the table. Refusals become a message, never a traceback."""
        self._debounce.stop()
        if self._frame is None or self._frame.empty:
            self._result = None
            self.table.set_result(None)
            self.notice.setText("Load a table, then drop a column onto Rows.")
            return None
        spec = self.spec()
        if spec.is_empty:
            self._result = None
            self.table.set_result(None)
            self.notice.setText(
                "Drop a column onto Rows or Columns to group by it, and a "
                "measurement onto Values to summarise it. With no Values you "
                "get a table of counts.")
            return None
        try:
            result = pivot(self._frame, spec)
        except PivotError as exc:
            self._result = None
            self.table.set_result(None)
            self.notice.setText(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            LOG.info("the pivot failed", exc_info=True)
            self._result = None
            self.table.set_result(None)
            self.notice.setText(f"could not build that table: {exc}")
            return None
        self._result = result
        self.table.set_result(result)
        parts = [result.summary()]
        if self.table.truncated_cells:
            parts.append("shown as a message rather than a grid — export it")
        self.notice.setText(" · ".join(parts))
        self.computed.emit(result)
        return result

    def _on_axis_changed(self, *_args) -> None:
        if self._building:
            return
        self._debounce.start()

    def _on_plot(self) -> None:
        frame = self.long_frame()
        if frame.empty:
            self.notice.setText(
                "Nothing to plot yet — build a table with at least one "
                "non-empty cell.")
            return
        self.plot_requested.emit(frame)

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        """Write the table as shown. Returns the path written, or ``None``."""
        if self._result is None:
            self.notice.setText("Nothing to export — build a table first.")
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export the table", "tabulate.csv", "CSV (*.csv)")
        if not path:
            return None
        try:
            self._result.to_csv(path)
        except OSError as exc:
            LOG.info("could not export the pivot", exc_info=True)
            self.notice.setText(f"could not write that file: {exc}")
            return None
        self.notice.setText(f"wrote {path}")
        return path

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self._debounce.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Styling, through the seam
# ---------------------------------------------------------------------------

def _pivot_qss(palette, opacity) -> str:
    from ..theme import block_surface
    surface_alt = block_surface("surface_alt", palette["theme"], opacity)
    return f"""
QWidget#PivotShelf {{
    background: {surface_alt};
    border-radius: {RADIUS["md"]}px;
}}
QLabel#PivotWellName {{
    color: {palette["fg_muted"]};
    font-weight: 600;
}}
QPushButton#PivotWellClear {{
    border: none;
    background: transparent;
    color: {palette["fg_muted"]};
}}
QListWidget#PivotWellList {{
    background: transparent;
    border: 1px dashed {palette["border"]};
    border-radius: {RADIUS["sm"]}px;
}}
QLabel#PivotNotice {{
    color: {palette["fg_muted"]};
    font-size: {font_px(11)}px;
}}
QTableWidget#PivotTable {{
    border: 1px solid {palette["border_soft"]};
    border-radius: {RADIUS["sm"]}px;
    gridline-color: {palette["border_soft"]};
}}
"""


# Registered at import of this module, which happens when the screen module
# is imported — and the row that does that lives in ``app.py``'s
# ``_SELF_REGISTERING_APPS``, whose loop runs while ``app.py`` itself is being
# imported. That is before ``launch()`` calls ``stylesheet()``, which is the
# deadline: a block registered after the stylesheet is built is missing from
# the one the application was actually given. `spacr.qt.widgets.__init__`
# imports `graph_builder` eagerly for exactly this reason; this module needs no
# such entry only because its screen is imported earlier still.
register_widget_qss("Pivot", _pivot_qss, replace=True)
