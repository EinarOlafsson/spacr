"""
Plate Viewer — any measurement as a plate heatmap, with the edge-effect
test sitting right next to it.

The heatmap is the delivery mechanism; the statistic beside it is the
point. A screen hit in row A or column 24 is far more likely to be an
evaporation artefact than biology, and until now spaCR gave nobody a way
to notice that before the follow-up experiment failed.

Layout::

    ┌───────────────────────────────────────────────────────────────────┐
    │ /data/plate1/measurements/measurements.db   [DB…] [Run folder…]   │
    │ Table [cell ▾] Value [cell_area ▾] Plate [plate1 ▾] [mean ▾]      │
    │ Colour [2–98 % ▾]  Min objects/well [20]            [Render]      │
    ├───────────────────────────────────┬───────────────────────────────┤
    │      1  2  3  4 …             24  │ Plate plate1 — mean cell_area │
    │  A  ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓  │ 384-well, 384 wells tested    │
    │  B  ▓▓ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ▓▓  │                               │
    │  C  ▓▓ ░░ ▒▒ ▒▒ ▒▒ ▒▒ ▒▒ ░░ ░░ ▓▓  │ Edge effect: the outer ring   │
    │  …                                │ reads +31.2 % vs the interior │
    │  P  ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓  │ (δ = 0.78, p < 1e-4).         │
    ├───────────────────────────────────┴───────────────────────────────┤
    │ C07 · 142 objects · mean cell_area = 1204.53 · ring 2 (interior)  │
    │ scale 812.4 → 1655.9 (viridis, 2–98 %)     [Export well grid CSV…]│
    └───────────────────────────────────────────────────────────────────┘

Design notes:

* **Read-only, structurally.** Every query goes through
  :mod:`spacr.plate_qc`, which opens the file with ``file:…?mode=ro`` and
  ``PRAGMA query_only = ON`` — the same approach as the Database Browser.
* **The statistics live outside the GUI.** All of the analysis is in
  :mod:`spacr.plate_qc`, which imports neither torch nor cellpose, so it
  is testable headless and the screen stays a view.
* **An empty well is drawn as empty.** Wells with no objects, or fewer
  than ``min_count``, are hatched — never coloured as if they measured
  zero. The count of dropped wells is on screen, because a heatmap
  quietly missing a third of its wells looks exactly like data.
* **Only the columns needed are read.** A spaCR feature table is 500
  columns wide; the query pulls the well identifiers plus the one
  measurement being plotted.
* **No modal dialogs on any error path.** "That column isn't in this
  table", "that file isn't a database", "this plate has no interior" —
  all of it lands in the inline status label. A QMessageBox would hang a
  headless run.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ... import plate_qc as pqc
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider
from .db_browser import resolve_db_path

__all__ = [
    "COLOUR_SCALES",
    "DEFAULT_CMAP",
    "PlateGridWidget",
    "PlateViewScreen",
    "PREFERRED_TABLES",
]


#: Colour-scale choices, mapping the label to the ``min_max`` spec
#: :func:`spacr.plate_qc.colour_limits` understands. The quantile scale is
#: the default: one dead well at 40x the plate median would otherwise
#: flatten every other well to the same colour.
COLOUR_SCALES: Tuple[Tuple[str, Any], ...] = (
    ("2–98 % (robust)", "allq"),
    ("Full range", "all"),
)

#: Colormap used for the wells. Perceptually uniform and colour-blind
#: safe; :func:`spacr.qt.preferences.color_blind_continuous_cmap` swaps it
#: for cividis when a colour-vision mode is on.
DEFAULT_CMAP = "viridis"

#: Tables tried first when a database opens, in order of usefulness.
PREFERRED_TABLES: Tuple[str, ...] = ("cell", "object", "nucleus", "pathogen",
                                     "cytoplasm", "png_list")

#: Pixels reserved for the row letters and the column numbers.
_ROW_LABEL_W = 30
_COL_LABEL_H = 20
_GRID_PAD = 6


def _cmap_lut(name: str, size: int = 256) -> List[QColor]:
    """Return ``size`` QColors sampled across the named matplotlib colormap.

    Sampling once into a lookup table keeps the paint loop free of any
    matplotlib call — a 1536-well plate would otherwise make 1536 of them
    on every repaint. Imported lazily so simply importing this module
    costs nothing.
    """
    try:
        from matplotlib import colormaps
        cmap = colormaps[name]
    except Exception:
        try:
            from matplotlib import cm
            cmap = cm.get_cmap(name)
        except Exception:
            cmap = None
    out: List[QColor] = []
    for i in range(size):
        t = i / (size - 1)
        if cmap is None:
            # Greyscale fallback — a plate is still readable without
            # matplotlib, which is better than a screen that won't paint.
            level = int(round(255 * t))
            out.append(QColor(level, level, level))
        else:
            r, g, b = cmap(t)[:3]
            out.append(QColor(int(round(r * 255)), int(round(g * 255)),
                              int(round(b * 255))))
    return out


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

class PlateGridWidget(QWidget):
    """A plate drawn as coloured wells, with row letters and column numbers.

    Painted directly rather than embedded as a matplotlib canvas for one
    reason that matters: a click has to map back to a *well*, exactly, and
    a hand-rolled grid gives that for free instead of via a coordinate
    round-trip through a figure's data transform.

    :param parent: parent widget.
    :ivar well_clicked: emitted with ``(row_index, column_index)``
        (1-based) whenever the user clicks inside the grid.
    """

    #: emitted with the 1-based ``(row_index, column_index)`` of a click
    well_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values: Dict[Tuple[int, int], float] = {}
        self._counts: Dict[Tuple[int, int], int] = {}
        self._n_rows = 0
        self._n_cols = 0
        self._vmin = 0.0
        self._vmax = 1.0
        self._lut = _cmap_lut(DEFAULT_CMAP)
        self._cmap_name = DEFAULT_CMAP
        self._selected: Optional[Tuple[int, int]] = None
        self._placeholder = ("Choose a database, a table and a measurement, "
                             "then press Render.")
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # -- data --------------------------------------------------------------

    def set_plate(self, layout: Optional[pd.DataFrame],
                  vmin: float = 0.0, vmax: float = 1.0,
                  cmap: str = DEFAULT_CMAP,
                  n_rows: Optional[int] = None,
                  n_cols: Optional[int] = None) -> None:
        """Show ``layout`` (a :func:`spacr.plate_qc.plate_layout` frame).

        :param layout: tidy well frame, or ``None`` to clear.
        :param vmin: value mapped to the bottom of the colormap.
        :param vmax: value mapped to the top.
        :param cmap: matplotlib colormap name.
        :param n_rows: nominal plate rows; taken from ``layout.attrs``
            when omitted.
        :param n_cols: nominal plate columns; likewise.
        """
        self._values.clear()
        self._counts.clear()
        self._selected = None
        if cmap != self._cmap_name:
            self._lut = _cmap_lut(cmap)
            self._cmap_name = cmap
        meta = dict(getattr(layout, "attrs", {}) or {}) if layout is not None else {}
        self._n_rows = int(n_rows or meta.get("n_rows") or 0)
        self._n_cols = int(n_cols or meta.get("n_cols") or 0)
        if layout is not None and len(layout):
            for r, c, n, v in zip(layout["row_index"], layout["column_index"],
                                  layout["n"], layout["value"]):
                key = (int(r), int(c))
                self._counts[key] = int(n)
                if pd.notna(v):
                    self._values[key] = float(v)
            self._n_rows = max(self._n_rows, int(layout["row_index"].max()))
            self._n_cols = max(self._n_cols, int(layout["column_index"].max()))
        self._vmin = float(vmin)
        self._vmax = float(vmax) if float(vmax) != float(vmin) else float(vmin) + 1e-6
        self.update()

    def clear(self) -> None:
        """Drop the current plate and repaint empty."""
        self.set_plate(None)

    def set_placeholder(self, text: str) -> None:
        """Text shown when there is no plate to draw."""
        self._placeholder = text
        self.update()

    def grid_size(self) -> Tuple[int, int]:
        """Return ``(n_rows, n_cols)`` of the grid currently drawn."""
        return self._n_rows, self._n_cols

    def has_plate(self) -> bool:
        """True when there is something to paint."""
        return self._n_rows > 0 and self._n_cols > 0

    def well_value(self, row_index: int, column_index: int) -> Optional[float]:
        """Value behind a well, or ``None`` when the well is blank."""
        return self._values.get((int(row_index), int(column_index)))

    def well_count(self, row_index: int, column_index: int) -> int:
        """Objects behind a well; ``0`` when nothing survived filtering."""
        return self._counts.get((int(row_index), int(column_index)), 0)

    def selected_well(self) -> Optional[Tuple[int, int]]:
        """The highlighted ``(row_index, column_index)``, if any."""
        return self._selected

    def select(self, row_index: Optional[int],
               column_index: Optional[int] = None) -> None:
        """Highlight a well (or clear the highlight with ``None``)."""
        if row_index is None or column_index is None:
            self._selected = None
        else:
            self._selected = (int(row_index), int(column_index))
        self.update()

    # -- geometry ----------------------------------------------------------

    def _cell_size(self) -> float:
        """Edge length of a square well, in pixels."""
        if not self.has_plate():
            return 0.0
        avail_w = max(self.width() - _ROW_LABEL_W - 2 * _GRID_PAD, 1)
        avail_h = max(self.height() - _COL_LABEL_H - 2 * _GRID_PAD, 1)
        return max(min(avail_w / self._n_cols, avail_h / self._n_rows), 1.0)

    def cell_rect(self, row_index: int, column_index: int) -> QRectF:
        """Return the rectangle a well occupies, in widget coordinates.

        Public because it is what makes a click testable: a test can aim
        at the centre of ``cell_rect(3, 7)`` instead of guessing at pixel
        arithmetic that would have to be kept in sync by hand.
        """
        size = self._cell_size()
        x = _GRID_PAD + _ROW_LABEL_W + (int(column_index) - 1) * size
        y = _GRID_PAD + _COL_LABEL_H + (int(row_index) - 1) * size
        return QRectF(x, y, size, size)

    def well_at(self, point) -> Optional[Tuple[int, int]]:
        """Map a widget-coordinate point to a 1-based ``(row, column)``.

        :param point: ``QPoint``/``QPointF`` in widget coordinates.
        :returns: the well under the point, or ``None`` when the point is
            in the margins or off the grid.
        """
        if not self.has_plate():
            return None
        size = self._cell_size()
        pt = QPointF(point)
        col = int((pt.x() - _GRID_PAD - _ROW_LABEL_W) // size) + 1
        row = int((pt.y() - _GRID_PAD - _COL_LABEL_H) // size) + 1
        if pt.x() < _GRID_PAD + _ROW_LABEL_W or pt.y() < _GRID_PAD + _COL_LABEL_H:
            return None
        if not (1 <= row <= self._n_rows and 1 <= col <= self._n_cols):
            return None
        return row, col

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        well = self.well_at(event.position() if hasattr(event, "position")
                            else event.pos())
        if well is not None:
            self.select(*well)
            self.well_clicked.emit(well[0], well[1])
        super().mousePressEvent(event)

    # -- painting ----------------------------------------------------------

    def _colour(self, value: float) -> QColor:
        span = self._vmax - self._vmin
        t = 0.0 if span == 0 else (value - self._vmin) / span
        t = min(max(t, 0.0), 1.0)
        return self._lut[int(round(t * (len(self._lut) - 1)))]

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        palette = active_palette()
        painter.fillRect(self.rect(), QColor(palette["surface"]))

        if not self.has_plate():
            painter.setPen(QColor(palette["fg_muted"]))
            painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                             Qt.AlignCenter | Qt.TextWordWrap,
                             self._placeholder)
            painter.end()
            return

        size = self._cell_size()
        font = QFont(self.font())
        font.setPointSizeF(max(min(size * 0.42, 11.0), 5.0))
        painter.setFont(font)

        # Column numbers along the top.
        palette = active_palette()
        painter.setPen(QColor(palette["fg_muted"]))
        for c in range(1, self._n_cols + 1):
            rect = QRectF(_GRID_PAD + _ROW_LABEL_W + (c - 1) * size, _GRID_PAD,
                          size, _COL_LABEL_H)
            painter.drawText(rect, Qt.AlignCenter, str(c))
        # Row letters down the side.
        for r in range(1, self._n_rows + 1):
            rect = QRectF(_GRID_PAD, _GRID_PAD + _COL_LABEL_H + (r - 1) * size,
                          _ROW_LABEL_W, size)
            painter.drawText(rect, Qt.AlignCenter, pqc.row_label(r))

        empty_pen = QPen(QColor(palette["border"]))
        empty_pen.setWidth(1)
        for r in range(1, self._n_rows + 1):
            for c in range(1, self._n_cols + 1):
                rect = self.cell_rect(r, c).adjusted(0.5, 0.5, -0.5, -0.5)
                value = self._values.get((r, c))
                if value is None:
                    # Blank, and visibly so: an absent well is not a zero.
                    painter.setPen(empty_pen)
                    painter.setBrush(QBrush(QColor(palette["surface_alt"])))
                    painter.drawRect(rect)
                    painter.drawLine(rect.topLeft(), rect.bottomRight())
                else:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(self._colour(value)))
                    painter.drawRect(rect)

        if self._selected is not None:
            r, c = self._selected
            if 1 <= r <= self._n_rows and 1 <= c <= self._n_cols:
                pen = QPen(QColor(palette["accent"]))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(self.cell_rect(r, c).adjusted(1, 1, -1, -1))
        painter.end()


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class PlateViewScreen(QWidget):
    """Plate heatmap + edge-effect QC for a spaCR measurements database.

    :param parent: parent widget.
    :param threaded: run database work on a worker thread (the default).
        Tests pass ``False`` for deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation succeeded. Errors are *only* ever reported here
        and in the inline status label — never in a modal dialog.
    """

    #: emitted with the resolved path whenever a database opens
    database_opened = Signal(str)
    #: emitted after a plate has been drawn
    plate_rendered = Signal(str)
    #: emitted after every job settles (ok or not)
    job_finished = Signal(bool)
    #: private. Re-emitted from ``PipelineWorker.finished`` purely to hop
    #: back onto the GUI thread — see :meth:`_run_job`.
    _job_settled = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._db_path: str = ""
        self._frame: Optional[pd.DataFrame] = None
        self._frame_key: Tuple[str, str, str] = ("", "", "")
        self._layout_df: Optional[pd.DataFrame] = None
        self._report: Optional[pqc.EdgeEffectReport] = None
        self._busy = False
        self._jobs: List[tuple] = []
        self._pending: List[Tuple[Dict[str, Any], Callable[[Any], None]]] = []
        self._thread = None
        self._worker = None
        self._loading = False
        self.last_error: str = ""

        self._job_settled.connect(self._on_job_settled)
        self._build_ui()
        from ..dnd import install_dropzone
        from ..dnd_handlers import get_handler
        install_dropzone(self, get_handler("plate_view"), self)
        self._set_status(
            "Choose a measurements.db, or a run folder containing "
            "measurements/measurements.db.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Plate Viewer")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Any measurement as a plate heatmap, with the edge-effect test "
            "beside it. The outer ring of a plate evaporates faster than the "
            "interior — a hit sitting in it is more likely to be an artefact "
            "than biology. Read-only: the database is opened with mode=ro.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText(
            "…/measurements/measurements.db  — or a run folder")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._on_open_typed_path)
        self._btn_pick_db = QPushButton("Choose database…", self)
        self._btn_pick_db.clicked.connect(self._pick_database)
        self._btn_pick_src = QPushButton("Choose run folder…", self)
        self._btn_pick_src.clicked.connect(self._pick_run_folder)
        self._btn_open = QPushButton("Open", self)
        self._btn_open.clicked.connect(self._on_open_typed_path)
        src_row.addWidget(self._path_edit, 1)
        src_row.addWidget(self._btn_pick_db)
        src_row.addWidget(self._btn_pick_src)
        src_row.addWidget(self._btn_open)
        outer.addLayout(src_row)

        # ── Selection row ─────────────────────────────────────────────
        pick_row = QHBoxLayout()
        pick_row.setSpacing(SPACING["sm"])
        pick_row.addWidget(QLabel("Table", self))
        self._table_combo = QComboBox(self)
        self._table_combo.setMinimumWidth(140)
        self._table_combo.setToolTip("(str) Measurement table to read.")
        self._table_combo.currentIndexChanged.connect(self._on_table_changed)
        pick_row.addWidget(self._table_combo)

        pick_row.addWidget(QLabel("Measurement", self))
        self._value_combo = QComboBox(self)
        self._value_combo.setMinimumWidth(240)
        self._value_combo.setToolTip(
            "(str) Numeric column aggregated per well and drawn as colour.")
        pick_row.addWidget(self._value_combo, 1)

        pick_row.addWidget(QLabel("Plate", self))
        self._plate_combo = QComboBox(self)
        self._plate_combo.setMinimumWidth(120)
        self._plate_combo.setToolTip("(str) Which plate to draw.")
        self._plate_combo.currentIndexChanged.connect(self._on_view_changed)
        pick_row.addWidget(self._plate_combo)
        outer.addLayout(pick_row)

        # ── Options row ───────────────────────────────────────────────
        opt_row = QHBoxLayout()
        opt_row.setSpacing(SPACING["sm"])
        opt_row.addWidget(QLabel("Per well", self))
        self._grouping_combo = QComboBox(self)
        self._grouping_combo.addItems(list(pqc.GROUPINGS))
        self._grouping_combo.setToolTip(
            "(str) How the objects in a well are collapsed to one number. "
            "'median' is the robust choice; 'count' plots objects per well "
            "and ignores the measurement.")
        self._grouping_combo.currentIndexChanged.connect(self._on_view_changed)
        opt_row.addWidget(self._grouping_combo)

        opt_row.addWidget(QLabel("Colour scale", self))
        self._scale_combo = QComboBox(self)
        for label, _spec in COLOUR_SCALES:
            self._scale_combo.addItem(label)
        self._scale_combo.setToolTip(
            "(str) Robust clips the colours to the 2nd–98th percentile so a "
            "single dead well cannot flatten the plate.")
        self._scale_combo.currentIndexChanged.connect(self._on_view_changed)
        opt_row.addWidget(self._scale_combo)

        opt_row.addWidget(QLabel("Min objects / well", self))
        self._min_count_box = QSpinBox(self)
        self._min_count_box.setRange(0, 100000)
        self._min_count_box.setValue(0)
        self._min_count_box.setToolTip(
            "(int) Wells with fewer objects than this are dropped. They are "
            "drawn blank, never as zero, and the number dropped is reported.")
        self._min_count_box.valueChanged.connect(self._on_view_changed)
        opt_row.addWidget(self._min_count_box)

        opt_row.addStretch(1)
        self._btn_render = QPushButton("Render", self)
        self._btn_render.setObjectName("Primary")
        self._btn_render.clicked.connect(self.render_plate)
        opt_row.addWidget(self._btn_render)
        outer.addLayout(opt_row)

        # ── Heatmap | report ──────────────────────────────────────────
        split = QSplitter(Qt.Horizontal, self)
        self._grid = PlateGridWidget(split)
        self._grid.well_clicked.connect(self._on_well_clicked)
        split.addWidget(self._grid)

        right = QWidget(split)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])
        right_layout.addWidget(QLabel("Edge-effect report", right))
        self._report_view = QPlainTextEdit(right)
        self._report_view.setReadOnly(True)
        self._report_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.Monospace)
        self._report_view.setFont(mono)
        self._report_view.setPlaceholderText(
            "The outer-ring test, the ring-by-ring profile and the "
            "row/column gradients appear here once a plate is rendered.")
        right_layout.addWidget(self._report_view, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([620, 520])
        outer.addWidget(split, 1)

        # ── Well readout ──────────────────────────────────────────────
        self._well_label = QLabel("Click a well to see what is behind it.", self)
        self._well_label.setWordWrap(True)
        self._well_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._well_label)

        # ── Footer ────────────────────────────────────────────────────
        foot_row = QHBoxLayout()
        foot_row.setSpacing(SPACING["sm"])
        self._scale_label = QLabel("", self)
        self._scale_label.setObjectName("Caption")
        foot_row.addWidget(self._scale_label, 1)
        self._btn_export = QPushButton("Export well grid CSV…", self)
        self._btn_export.clicked.connect(self._pick_export_path)
        foot_row.addWidget(self._btn_export)
        outer.addLayout(foot_row)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

    # -- status ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        palette = active_palette()
        colour = palette["error"] if error else palette["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    def report_text(self) -> str:
        """The rendered edge-effect report (test/introspection helper)."""
        return self._report_view.toPlainText()

    def well_info_text(self) -> str:
        """The per-well readout line (test/introspection helper)."""
        return self._well_label.text()

    def _update_controls(self) -> None:
        has_db = bool(self._db_path)
        ready = has_db and not self._busy
        self._btn_render.setEnabled(ready and self._value_combo.count() > 0)
        self._table_combo.setEnabled(ready)
        self._value_combo.setEnabled(ready)
        self._plate_combo.setEnabled(ready and self._plate_combo.count() > 0)
        self._grouping_combo.setEnabled(ready)
        self._scale_combo.setEnabled(ready)
        self._min_count_box.setEnabled(ready)
        self._btn_export.setEnabled(self._layout_df is not None
                                    and len(self._layout_df) > 0)

    # -- database ----------------------------------------------------------

    def _pick_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a measurements database", self._path_edit.text() or
            os.path.expanduser("~"), "SQLite databases (*.db);;All files (*)")
        if path:
            self._path_edit.setText(path)
            self.open_database(path)

    def _pick_run_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a run folder", self._path_edit.text() or
            os.path.expanduser("~"))
        if path:
            self._path_edit.setText(path)
            self.open_database(path)

    def _on_open_typed_path(self) -> None:
        self.open_database(self._path_edit.text())

    def open_database(self, path: str) -> bool:
        """Open ``path`` read-only and list the tables it holds.

        :param path: a ``measurements.db`` or a run folder containing one.
        :returns: True when the database opened.
        """
        try:
            resolved = resolve_db_path(path)
            names = pqc.tables(resolved)
        except Exception as e:
            self._db_path = ""
            self._frame = None
            self._layout_df = None
            self._grid.clear()
            self._report_view.setPlainText("")
            self._set_status(str(e) or e.__class__.__name__, error=True)
            self._update_controls()
            return False

        self._db_path = resolved
        self._path_edit.setText(resolved)
        self._frame = None
        self._frame_key = ("", "", "")
        self._layout_df = None
        self._report = None
        self._grid.clear()
        self._report_view.setPlainText("")

        self._loading = True
        try:
            self._table_combo.clear()
            self._table_combo.addItems(names)
            default = next((t for t in PREFERRED_TABLES if t in names),
                           names[0] if names else "")
            if default:
                self._table_combo.setCurrentText(default)
        finally:
            self._loading = False

        self.database_opened.emit(resolved)
        if not names:
            self._set_status(
                f"{os.path.basename(resolved)} has no tables.", error=True)
            self._update_controls()
            return True
        self._on_table_changed()
        return True

    def current_table(self) -> str:
        """The table currently selected."""
        return self._table_combo.currentText()

    def current_value_column(self) -> str:
        """The measurement column currently selected."""
        return self._value_combo.currentText()

    def set_table(self, table: str) -> None:
        """Select ``table`` and reload its numeric columns."""
        self._table_combo.setCurrentText(str(table))
        self._on_table_changed()

    def set_value_column(self, column: str) -> None:
        """Select ``column`` as the measurement to draw.

        A name that is not in the current table is *accepted* and selected
        anyway — it is exactly what happens when a user picks a column and
        then switches to a table that does not have it. Rendering then
        fails with an inline explanation rather than the combo silently
        snapping back to something the user did not choose.
        """
        name = str(column)
        if self._value_combo.findText(name) < 0:
            self._value_combo.addItem(name)
        self._value_combo.setCurrentText(name)

    def _on_table_changed(self, *_args) -> None:
        if self._loading or not self._db_path:
            return
        table = self._table_combo.currentText()
        if not table:
            return
        self._frame = None
        self._frame_key = ("", "", "")

        def _job():
            return pqc.numeric_columns(self._db_path, table)

        def _done(columns: List[str]) -> None:
            self._loading = True
            try:
                self._value_combo.clear()
                self._value_combo.addItems(columns)
            finally:
                self._loading = False
            if columns:
                self._set_status(
                    f"{table}: {len(columns)} numeric column(s). Pick a "
                    f"measurement and press Render.")
            else:
                self._set_status(
                    f"{table} has no numeric columns to plot — pick another "
                    f"table.", error=True)

        return self._run_job(_job, _done)

    # -- rendering ---------------------------------------------------------

    def _on_view_changed(self, *_args) -> None:
        """Plate / grouping / scale / min_count changed — recompute only.

        No database round-trip: the long frame is already in memory, and a
        user dragging the min-objects spin box should not re-query a
        500 000-row table on every tick.
        """
        if self._loading or self._frame is None:
            return
        self.recompute()

    def render_plate(self) -> bool:
        """Read the chosen measurement and draw the plate.

        The database is only touched when the table or measurement
        changed; otherwise this is the same recompute the option controls
        trigger.

        :returns: True when the render was started/completed.
        """
        if not self._db_path:
            self._set_status("Open a measurements database first.", error=True)
            return False
        table = self._table_combo.currentText()
        value_col = self._value_combo.currentText()
        if not table:
            self._set_status("Pick a table first.", error=True)
            return False
        if not value_col:
            self._set_status(
                "Pick a measurement column first — this table exposed no "
                "numeric columns.", error=True)
            return False

        key = (self._db_path, table, value_col)
        if self._frame is not None and self._frame_key == key:
            return self.recompute()

        def _job():
            return pqc.load_plate_frame(self._db_path, table, value_col)

        def _done(frame: pd.DataFrame) -> None:
            self._frame = frame
            self._frame_key = key
            self._refresh_plate_combo(frame)
            self.recompute()

        return self._run_job(_job, _done)

    def _refresh_plate_combo(self, frame: pd.DataFrame) -> None:
        """Repopulate the plate list, keeping the current choice if valid."""
        plates = pqc.plates_in(frame)
        previous = self._plate_combo.currentText()
        self._loading = True
        try:
            self._plate_combo.clear()
            self._plate_combo.addItems(plates)
            if previous and previous in plates:
                self._plate_combo.setCurrentText(previous)
        finally:
            self._loading = False

    def recompute(self) -> bool:
        """Rebuild the well grid + report from the frame already in memory.

        :returns: True when a plate was drawn.
        """
        if self._frame is None:
            self._set_status("Nothing loaded yet — press Render.", error=True)
            return False
        grouping = self._grouping_combo.currentText()
        plate = self._plate_combo.currentText() or None
        min_count = int(self._min_count_box.value())
        value_col = self._frame_key[2] or None
        try:
            layout = pqc.plate_layout(
                self._frame, value_col=value_col, plate=plate,
                grouping=grouping, min_count=min_count)
            report = pqc.detect_edge_effect(
                layout, value_col=value_col, grouping=grouping)
        except Exception as e:
            self._layout_df = None
            self._report = None
            self._grid.clear()
            self._report_view.setPlainText("")
            self._set_status(str(e) or e.__class__.__name__, error=True)
            self._update_controls()
            return False

        self._layout_df = layout
        self._report = report
        spec = COLOUR_SCALES[max(self._scale_combo.currentIndex(), 0)][1]
        vmin, vmax = pqc.colour_limits(layout, spec)
        cmap = self._colormap_name()
        self._grid.set_plate(layout, vmin, vmax, cmap)
        if not len(layout):
            self._grid.set_placeholder(
                "No wells survived filtering — lower 'Min objects / well', or "
                "pick another plate.")
        self._report_view.setPlainText(pqc.format_edge_report(report))
        self._scale_label.setText(
            f"colour scale {vmin:.4g} → {vmax:.4g} "
            f"({cmap}, {self._scale_combo.currentText()}) · "
            f"crossed wells are blank, not zero")
        self._well_label.setText("Click a well to see what is behind it.")

        n_blank = self._blank_well_count(layout)
        message = (f"{report.n_wells} well(s) drawn"
                   + (f", {report.n_dropped_min_count} dropped for holding "
                      f"fewer than {min_count} objects"
                      if report.n_dropped_min_count else "")
                   + (f", {n_blank} of the grid left blank" if n_blank else "")
                   + ". " + report.summary)
        self._set_status(message, error=False)
        self._update_controls()
        self.plate_rendered.emit(str(report.plate or ""))
        return True

    def _blank_well_count(self, layout: pd.DataFrame) -> int:
        """Wells on the nominal grid with nothing behind them."""
        meta = dict(getattr(layout, "attrs", {}) or {})
        total = int(meta.get("n_rows") or 0) * int(meta.get("n_cols") or 0)
        return max(total - int(len(layout)), 0)

    def _colormap_name(self) -> str:
        """Colormap honouring the user's colour-vision preference."""
        try:
            from ..preferences import color_blind_continuous_cmap
            return color_blind_continuous_cmap()
        except Exception:
            return DEFAULT_CMAP

    # -- well readout ------------------------------------------------------

    def _on_well_clicked(self, row_index: int, column_index: int) -> None:
        self.select_well(row_index, column_index)

    def select_well(self, row_index: int, column_index: int) -> str:
        """Report what is behind a well, and return the text shown.

        :param row_index: 1-based plate row.
        :param column_index: 1-based plate column.
        :returns: the readout line.
        """
        self._grid.select(row_index, column_index)
        well = pqc.well_id(row_index, column_index)
        layout = self._layout_df
        row = None
        if layout is not None and len(layout):
            match = layout[(layout["row_index"] == int(row_index))
                           & (layout["column_index"] == int(column_index))]
            if len(match):
                row = match.iloc[0]
        if row is None:
            min_count = int(self._min_count_box.value())
            reason = (f" — no objects, or fewer than the {min_count} required"
                      if min_count else " — no objects measured here")
            text = f"{well} · blank{reason}."
        else:
            grouping = self._grouping_combo.currentText()
            value_col = self._frame_key[2]
            what = ("objects per well" if grouping == "count"
                    else f"{grouping} {value_col}")
            ring = int(row["ring"])
            where = "outer ring (edge)" if ring == 0 else f"ring {ring} (interior)"
            value = row["value"]
            shown = "no value" if pd.isna(value) else f"{float(value):.6g}"
            text = (f"{well} · {int(row['n'])} object(s) · {what} = {shown} "
                    f"· {where}")
        self._well_label.setText(text)
        return text

    # -- export ------------------------------------------------------------

    def _pick_export_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export well grid", os.path.join(
                os.path.expanduser("~"), "plate_wells.csv"),
            "CSV files (*.csv);;All files (*)")
        if path:
            self.export_csv(path)

    def export_csv(self, out_path: str) -> bool:
        """Write the well grid to ``out_path`` as CSV.

        The tidy grid is exported — one row per well with its object
        count, value, ring index and edge flag — because that is what
        anybody re-analysing the plate outside spaCR needs.

        :param out_path: destination file.
        :returns: True on success; on failure the reason lands in the
            inline status label and ``last_error``.
        """
        if self._layout_df is None or not len(self._layout_df):
            self._set_status("Nothing to export — render a plate first.",
                             error=True)
            return False
        try:
            written = pqc.write_layout_csv(self._layout_df, out_path)
        except Exception as e:
            self._set_status(f"Export failed: {e}", error=True)
            return False
        self._set_status(f"Exported {len(self._layout_df)} well(s) → {written}")
        return True

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Mirrors ``DbBrowserScreen._run_job`` — one threading idiom for the
        whole Qt layer — with one difference that matters here.
        ``PipelineWorker.finished`` is emitted *in the worker thread*, and
        PySide6 invokes a plain closure connected to it directly, on that
        same thread. The Database Browser gets away with it because its
        completion handler only pokes a table model; this screen's fills a
        QPlainTextEdit, and building a QTextDocument's children off the
        GUI thread is undefined behaviour (Qt says so out loud:
        "Cannot create children for a parent that is in a different
        thread"). So ``finished`` is chained through :attr:`_job_settled`
        into a *bound method* of this widget, which has GUI-thread
        affinity — Qt then queues the call and the completion handler runs
        where every other widget call runs.

        With ``threaded=False`` the call runs inline and the same signals
        fire, so both paths behave identically from outside.
        """
        if not self._threaded:
            ok = True
            try:
                on_done(fn())
            except Exception as e:
                self._on_job_error(e)
                ok = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        box: Dict[str, Any] = {}

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a QThread garbage-collected
        # while still running takes the process down with it.
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append((box, on_done))
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def _on_job_settled(self, ok: bool) -> None:
        """Finish the oldest in-flight job. Always on the GUI thread."""
        self._busy = False
        box, on_done = self._pending.pop(0) if self._pending else ({}, None)
        ok = bool(ok)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as e:
                self._on_job_error(e)
                ok = False
        self._update_controls()
        self.job_finished.emit(ok)

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        """True while a database job is in flight."""
        return self._busy

    def _on_job_error(self, exc: Exception) -> None:
        self._busy = False
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._set_status(f"Plate view failed: {line}", error=True)
