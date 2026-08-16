"""Interactive regression plots drawn by Qt, not by matplotlib.

WHY THIS EXISTS

matplotlib redraws every artist, in Python, on every frame. On the screen's
volcano -- 1,215 points scattered once per LOPIT compartment, with a 27-entry
legend -- that is ~115 ms per redraw, and it is paid again for every pan, every
zoom, and every style change. No amount of debouncing, threading or resolution
capping removes it, because the cost is text layout and marker geometry rather
than pixels.

pyqtgraph draws into a QGraphicsScene. Pan, zoom and hover cost NOTHING, because
the scene is composited by Qt rather than re-rendered by Python; a log-axis
toggle is 4.7 ms against matplotlib's 115 ms; recolouring every point is 45 ms.
The same reason the 3D UMAP viewer can spin 10,000 points with edges while a
flat scatter plot stutters.

THE SPLIT

    on screen  -> pyqtgraph, because the user interacts with it
    on disk    -> matplotlib, because it still makes the better vector page

Publication figures are unchanged. This is only what the application shows.

Every plot here takes a DataFrame and returns a widget. They are deliberately
free of spaCR imports so they can be tested, and reused, on their own.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised by the import guard test
    import pyqtgraph as pg
    from pyqtgraph import ScatterPlotItem
    HAVE_PYQTGRAPH = True
except Exception:  # pragma: no cover - pyqtgraph is optional
    pg = None
    ScatterPlotItem = object
    HAVE_PYQTGRAPH = False

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

#: Colour-blind-safe qualitative order. A screen's categories are nominal, so a
#: sequential map would imply a ranking that is not there.
PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#4878CF", "#EE854A",
    "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979",
    "#D5BB67", "#82C6E2",
)

#: Points beyond this many stop getting individual hover hit-boxes, which is
#: what makes a large scatter slow to move over rather than slow to draw.
HOVER_LIMIT = 20000


def _require_pyqtgraph() -> None:
    if not HAVE_PYQTGRAPH:
        raise RuntimeError(
            "pyqtgraph is needed for the interactive plots. Install it with "
            "`pip install pyqtgraph`, or use the matplotlib figures.")


def colour_for(index: int, alpha: int = 255) -> QColor:
    """Stable colour for category ``index``."""
    colour = QColor(PALETTE[index % len(PALETTE)])
    colour.setAlpha(alpha)
    return colour


def _finite(values) -> np.ndarray:
    """Coerce to float and replace anything unplottable with NaN.

    A p-value column arrives with blanks, strings and the occasional inf from
    a log of zero. Left alone, one of those silently rescales the whole axis
    and the plot looks empty.
    """
    array = np.asarray(values, dtype="float64")
    return np.where(np.isfinite(array), array, np.nan)


class FastPlot(QWidget):
    """A pyqtgraph plot with the controls every plot here wants.

    :ivar point_clicked: emitted with the row index of a clicked point.
    """

    point_clicked = Signal(int)

    def __init__(self, title: str = "", x_label: str = "", y_label: str = "",
                 parent=None):
        super().__init__(parent)
        _require_pyqtgraph()
        pg.setConfigOptions(antialias=True, background=None, foreground="k")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot = pg.PlotWidget(title=title or None)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.plot, 1)

        controls = QHBoxLayout()
        self._log_x = QCheckBox("log x")
        self._log_y = QCheckBox("log y")
        for box in (self._log_x, self._log_y):
            box.toggled.connect(self._apply_log)
            controls.addWidget(box)
        self._grid = QCheckBox("grid")
        self._grid.setChecked(True)
        self._grid.toggled.connect(
            lambda on: self.plot.showGrid(x=on, y=on, alpha=0.25))
        controls.addWidget(self._grid)
        controls.addStretch(1)

        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self.plot.autoRange())
        controls.addWidget(reset)
        export = QPushButton("Export…")
        export.clicked.connect(self.export)
        controls.addWidget(export)
        layout.addLayout(controls)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._labels: Sequence[str] = ()

    # ----------------------------------------------------------------- state

    def _apply_log(self) -> None:
        self.plot.setLogMode(self._log_x.isChecked(), self._log_y.isChecked())

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def add_scatter(self, x, y, *, colours=None, size: float = 8.0,
                    labels: Sequence[str] = (), symbol: str = "o",
                    name: str = "") -> ScatterPlotItem:
        """Add points and wire up clicking them.

        :param colours: one QColor per point, or None for a single colour.
        :param labels: per-point text, shown on hover and on click.
        """
        x = _finite(x)
        y = _finite(y)
        keep = ~(np.isnan(x) | np.isnan(y))
        # Indices into the ORIGINAL frame, so a click still identifies the
        # right row after unplottable points have been dropped.
        original = np.nonzero(keep)[0]

        brushes = None
        if colours is not None:
            colours = list(colours)
            brushes = [pg.mkBrush(colours[i]) for i in original]

        # `data` must go in with the points: calling setData afterwards ADDS
        # points rather than annotating the ones already there.
        item = pg.ScatterPlotItem(
            x=x[keep], y=y[keep], size=size, symbol=symbol,
            pen=pg.mkPen(None),
            brush=brushes if brushes is not None else pg.mkBrush(colour_for(0)),
            hoverable=len(original) <= HOVER_LIMIT,
            data=list(original), name=name or None,
        )
        item.sigClicked.connect(self._on_points_clicked)
        self.plot.addItem(item)
        if labels is not None and len(labels):
            self._labels = labels
        return item

    def add_line(self, *, x=None, y=None, colour: str = "#C44E52",
                 style=Qt.DashLine, width: float = 1.5, label: str = ""):
        """A threshold line. ``x`` for vertical, ``y`` for horizontal."""
        pen = pg.mkPen(QColor(colour), width=width, style=style)
        line = pg.InfiniteLine(
            pos=(x if x is not None else y),
            angle=90 if x is not None else 0,
            pen=pen, label=label or None,
            labelOpts={"position": 0.92, "color": colour, "movable": False},
        )
        self.plot.addItem(line)
        return line

    def _on_points_clicked(self, _item, points) -> None:
        if not len(points):
            return
        index = points[0].data()
        if index is None:
            return
        index = int(index)
        if self._labels is not None and index < len(self._labels):
            self.set_status(str(self._labels[index]))
        self.point_clicked.emit(index)

    # ---------------------------------------------------------------- export

    def export(self, path: Optional[str] = None) -> Optional[str]:
        """Write the plot out. SVG when the name says so, else PNG.

        pyqtgraph's SVG export is real vector output, so what is exported from
        the screen is publishable rather than a screenshot of it.
        """
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self, "Export plot", "plot.png",
                "Vector (*.svg);;Image (*.png)")
            if not path:
                return None
        from pyqtgraph import exporters

        item = self.plot.plotItem
        if str(path).lower().endswith(".svg"):
            exporters.SVGExporter(item).export(path)
        else:
            exporters.ImageExporter(item).export(path)
        return path


class VolcanoPlot(FastPlot):
    """Effect against -log10(p), coloured by a category, every dot clickable."""

    def __init__(self, parent=None):
        super().__init__(title="Volcano", x_label="coefficient",
                         y_label="-log10(p)", parent=parent)

    def set_results(self, frame, *, effect: str = "coefficient",
                    p_column: str = "p_value", label_column: str = "feature",
                    category_column: Optional[str] = None,
                    alpha: float = 0.05,
                    effect_threshold: Optional[float] = None):
        """Draw ``frame``. Returns the number of points actually plotted."""
        self.plot.clear()
        if frame is None or not len(frame):
            self.set_status("No coefficients to plot.")
            return 0

        effects = _finite(frame[effect]) if effect in frame else np.zeros(len(frame))
        p_values = _finite(frame[p_column]) if p_column in frame \
            else np.full(len(frame), np.nan)
        # A p of exactly zero is a real result underflowing, not a mistake;
        # clamping keeps it on the plot instead of sending it to infinity.
        smallest = np.nanmin(p_values[p_values > 0]) if np.any(p_values > 0) \
            else 1e-300
        neglog = -np.log10(np.clip(p_values, smallest * 1e-3, 1.0))

        colours, legend = None, {}
        if category_column and category_column in frame:
            names = frame[category_column].astype(str).fillna("?")
            unique = sorted(names.unique())
            legend = {name: colour_for(i) for i, name in enumerate(unique)}
            colours = [legend[name] for name in names]

        labels = []
        for i in range(len(frame)):
            parts = [str(frame[label_column].iloc[i])] if label_column in frame else []
            parts.append(f"{effect}={effects[i]:.4g}")
            parts.append(f"{p_column}={p_values[i]:.3g}")
            labels.append("   ".join(parts))

        self.add_scatter(effects, neglog, colours=colours, labels=labels)
        self.add_line(y=-np.log10(alpha), label=f"p={alpha:g}")
        if effect_threshold:
            for sign in (-1, 1):
                self.add_line(x=sign * abs(effect_threshold), colour="#8C8C8C")

        if legend:
            self.plot.addLegend(offset=(-10, 10), labelTextSize="8pt")
            for name, colour in legend.items():
                marker = pg.ScatterPlotItem(
                    [], [], brush=pg.mkBrush(colour), pen=None, size=8)
                self.plot.plotItem.legend.addItem(marker, name)

        plotted = int(np.sum(~(np.isnan(effects) | np.isnan(neglog))))
        self.set_status(f"{plotted} coefficients. Click a point for detail.")
        return plotted


class PValueHistogram(FastPlot):
    """The single most informative check that a correction means anything.

    Under the null, p-values are uniform. A histogram that is flat with a spike
    at zero is a screen with real hits in it; one that slopes, or piles up near
    one, says the model is misspecified and every q-value downstream of it is
    decoration.
    """

    def __init__(self, parent=None):
        super().__init__(title="p-value distribution", x_label="p",
                         y_label="count", parent=parent)

    def set_p_values(self, values, bins: int = 50):
        self.plot.clear()
        p = _finite(values)
        p = p[~np.isnan(p)]
        if not len(p):
            self.set_status("No p-values.")
            return 0
        counts, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
        bars = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts,
                               brush=pg.mkBrush(colour_for(0, 190)),
                               pen=pg.mkPen(None))
        self.plot.addItem(bars)
        expected = len(p) / bins
        self.add_line(y=expected, colour="#C44E52", label="uniform")
        excess = max(int(counts[0] - expected), 0)
        self.set_status(
            f"{len(p)} p-values. The flat line is what a screen with no signal "
            f"would give; the first bin holds {excess} more than that.")
        return len(p)


class QQPlot(FastPlot):
    """Observed against expected quantiles -- is the null calibrated?

    Points on the diagonal mean the test is behaving. A curve that lifts off it
    early means inflation: the design is confounded, and the hits at the top of
    the volcano are partly an artefact of that rather than biology.
    """

    def __init__(self, parent=None):
        super().__init__(title="p-value Q-Q", x_label="expected -log10(p)",
                         y_label="observed -log10(p)", parent=parent)

    def set_p_values(self, values):
        self.plot.clear()
        p = _finite(values)
        p = np.sort(p[~np.isnan(p) & (p > 0)])
        if not len(p):
            self.set_status("No usable p-values.")
            return 0
        n = len(p)
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        observed = -np.log10(p)
        self.add_scatter(expected, observed, size=6)
        top = float(max(expected.max(), observed.max()))
        self.plot.plot([0, top], [0, top],
                       pen=pg.mkPen("#C44E52", width=1.5, style=Qt.DashLine))
        # Genomic inflation: the ratio at the median. 1.0 is calibrated.
        chi = np.median(observed) / np.median(expected) if np.median(expected) else float("nan")
        self.set_status(
            f"{n} tests. Inflation at the median is {chi:.2f} "
            f"(1.00 is calibrated; well above it means the null is not flat).")
        return n


class ResidualPlot(FastPlot):
    """Residual against fitted -- the check for a mis-specified mean.

    A horizontal band is what a well-specified model gives. A funnel means the
    variance grows with the fit and the standard errors are wrong, which is a
    p-value problem rather than a cosmetic one.
    """

    def __init__(self, parent=None):
        super().__init__(title="Residuals vs fitted", x_label="fitted",
                         y_label="residual", parent=parent)

    def set_residuals(self, fitted, residuals, labels: Sequence[str] = ()):
        self.plot.clear()
        f, r = _finite(fitted), _finite(residuals)
        if not len(f):
            self.set_status("No residuals.")
            return 0
        self.add_scatter(f, r, size=6, labels=labels)
        self.add_line(y=0.0, colour="#C44E52")
        good = ~(np.isnan(f) | np.isnan(r))
        if good.sum() > 2:
            # A crude trend line: if this is not flat, the mean is wrong.
            slope, intercept = np.polyfit(f[good], r[good], 1)
            xs = np.array([np.nanmin(f), np.nanmax(f)])
            self.plot.plot(xs, slope * xs + intercept,
                           pen=pg.mkPen("#DD8452", width=1.5))
            self.set_status(
                f"{int(good.sum())} residuals. Trend slope {slope:+.3g} -- "
                f"far from zero means the mean model is missing something.")
        return int(good.sum())


class ControlSeparation(FastPlot):
    """How far apart the positive and negative controls sit.

    This is the assay window. If the controls do not separate, nothing further
    down the pipeline can be trusted, and it is worth seeing before the volcano
    rather than after arguing about a hit list.
    """

    def __init__(self, parent=None):
        super().__init__(title="Control separation", x_label="",
                         y_label="effect", parent=parent)

    def set_groups(self, groups: dict):
        """:param groups: ``{'negative': array, 'positive': array, ...}``"""
        self.plot.clear()
        if not groups:
            self.set_status("No controls identified.")
            return 0
        summary, total = [], 0
        rng = np.random.default_rng(0)
        for position, (name, values) in enumerate(groups.items()):
            v = _finite(values)
            v = v[~np.isnan(v)]
            if not len(v):
                continue
            total += len(v)
            # Jitter, so overlapping points stay countable by eye.
            x = position + (rng.random(len(v)) - 0.5) * 0.35
            self.add_scatter(x, v, size=7,
                             colours=[colour_for(position, 200)] * len(v))
            median = float(np.median(v))
            self.plot.plot([position - 0.25, position + 0.25], [median, median],
                           pen=pg.mkPen("#000000", width=2))
            summary.append(f"{name} n={len(v)} median={median:.3g}")
        axis = self.plot.getAxis("bottom")
        axis.setTicks([[(i, name) for i, name in enumerate(groups)]])
        self.set_status("   ".join(summary) if summary else "No control values.")
        return total


class ResultsTable(QWidget):
    """The coefficient table, sortable and searchable, wired to a plot.

    A volcano answers "which points are extreme"; it cannot answer "what is
    the q-value of TGGT1_233460_4". Reading numbers off a scatter is the wrong
    tool, and until now the only way to see them was to open the CSV.

    :ivar row_selected: emitted with the frame row index of the selected row.
    """

    row_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import (QAbstractItemView, QLineEdit,
                                       QTableWidget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "Filter rows — type a gene, a guide, anything in the table")
        self._filter.textChanged.connect(self._apply_filter)
        top.addWidget(self._filter, 1)
        self._only_hits = QCheckBox("significant only")
        self._only_hits.toggled.connect(self._apply_filter)
        top.addWidget(self._only_hits)
        self._copy = QPushButton("Copy")
        self._copy.setToolTip("Copy the visible rows as TSV.")
        self._copy.clicked.connect(self.copy_visible)
        top.addWidget(self._copy)
        layout.addLayout(top)

        self.table = QTableWidget(0, 0)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self._count = QLabel("")
        layout.addWidget(self._count)

        self._frame = None
        self._alpha = 0.05

    def set_frame(self, frame, *, alpha: float = 0.05,
                  significance_column: Optional[str] = None) -> int:
        """Fill the table. Returns the row count."""
        from PySide6.QtWidgets import QTableWidgetItem

        self._frame = frame
        self._alpha = alpha
        self._significance = significance_column or self._guess_significance(frame)
        if frame is None or not len(frame):
            self.table.setRowCount(0)
            self._count.setText("Nothing to show.")
            return 0

        columns = list(frame.columns)
        # Sorting must be off while filling: with it on, Qt re-sorts after
        # every insert and the rows end up interleaved.
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column, name in enumerate(columns):
                value = frame.iloc[row][name]
                item = _NumericItem(value)
                # The frame row, so a click still maps home after sorting.
                item.setData(Qt.UserRole, row)
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self._apply_filter()
        return len(frame)

    @staticmethod
    def _guess_significance(frame) -> Optional[str]:
        """Prefer a corrected column: filtering on raw p would mislead."""
        if frame is None:
            return None
        for name in ("q_value", "adjusted_p_value", "p_value"):
            if name in frame.columns:
                return name
        return None

    def _apply_filter(self) -> None:
        text = self._filter.text().strip().lower()
        hits_only = self._only_hits.isChecked()
        shown = 0
        for row in range(self.table.rowCount()):
            visible = True
            if text:
                visible = any(
                    text in (self.table.item(row, c).text() or "").lower()
                    for c in range(self.table.columnCount())
                    if self.table.item(row, c) is not None)
            if visible and hits_only and self._significance:
                column = list(self._frame.columns).index(self._significance)
                item = self.table.item(row, column)
                try:
                    visible = float(item.text()) <= self._alpha
                except (TypeError, ValueError):
                    visible = False
            self.table.setRowHidden(row, not visible)
            shown += int(visible)
        total = self.table.rowCount()
        note = f"{shown} of {total} rows"
        if hits_only and self._significance:
            note += f" ({self._significance} <= {self._alpha:g})"
        self._count.setText(note)

    def _on_selection(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        index = items[0].data(Qt.UserRole)
        if index is not None:
            self.row_selected.emit(int(index))

    def select_frame_row(self, index: int) -> bool:
        """Scroll to and select the row for frame position ``index``.

        This is the other half of clicking a point on the volcano: the dot and
        the numbers behind it should be two views of one thing.
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == index:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return True
        return False

    def copy_visible(self) -> str:
        """Put the visible rows on the clipboard as TSV, and return them."""
        from PySide6.QtWidgets import QApplication

        lines = ["\t".join(
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount()))]
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            lines.append("\t".join(
                (self.table.item(row, c).text() if self.table.item(row, c)
                 else "")
                for c in range(self.table.columnCount())))
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        return text


try:  # pragma: no cover - trivial subclass
    from PySide6.QtWidgets import QTableWidgetItem

    class _NumericItem(QTableWidgetItem):
        """Sorts numerically when it holds a number, textually otherwise.

        A plain QTableWidgetItem sorts "10" before "9", which on a q-value
        column puts the answer in the wrong place.
        """

        def __init__(self, value):
            super().__init__("" if value is None else str(value))
            try:
                self._number = float(value)
            except (TypeError, ValueError):
                self._number = None

        def __lt__(self, other):
            mine = getattr(self, "_number", None)
            theirs = getattr(other, "_number", None)
            if mine is not None and theirs is not None:
                return mine < theirs
            return self.text() < other.text()
except Exception:  # pragma: no cover
    _NumericItem = None
