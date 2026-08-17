"""The volcano you can click, restyle and export.

The pipeline writes a volcano PDF, and until now that was the end of it: to
change an axis label, recolour by a covariate or find out which guide a dot
was, you re-ran the analysis or opened the CSV. This widget makes the plot the
thing you interrogate.

Three ideas hold it together:

* **One renderer.** Every pixel comes from
  :func:`spacr.volcano_style.render_volcano`, the same function the headless
  pipeline calls. Exporting re-renders from the same :class:`VolcanoStyle` at
  print size rather than screenshotting the widget, so what is saved is what
  was on screen, at full vector quality.
* **The style is a value.** Controls write into a
  :class:`~spacr.volcano_style.VolcanoStyle` and ask for a redraw. That makes
  the whole appearance saveable to JSON, reloadable, and reproducible.
* **Clicking is a lookup, not a hit test on pixels.** The nearest point is
  found in data space normalised by the axis ranges, so a click is as accurate
  on a squashed axis as on a square one.

Annotation files can be dropped in to colour or shape the points by anything:
the file is merged onto the results by a join column, and every column it
brings becomes available in the colour-by and shape-by menus.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...volcano_style import (
    COLORMAPS,
    FONT_FAMILIES,
    LINE_STYLES,
    MARKER_SHAPES,
    SCALES,
    VolcanoStyle,
    point_details,
    render_volcano,
)

#: Columns never offered as a colour/shape source: they are the plot's own
#: axes or bookkeeping, and mapping colour to the y axis says nothing.
_NON_MAPPING_COLUMNS = frozenset({
    "standardized_marginal_effect", "adjusted_p_value", "permutation_p_value",
    "p_value", "q_value", "coefficient", "significant", "alpha",
})


class VolcanoExplorer(QWidget):
    """An interactive volcano: click a point, restyle it, export it."""

    point_selected = Signal(dict)
    style_changed = Signal()

    def __init__(self, results: pd.DataFrame | None = None,
                 style: VolcanoStyle | None = None, parent=None):
        super().__init__(parent)
        self._results = pd.DataFrame() if results is None else results.reset_index(drop=True)
        self._style = style or VolcanoStyle()
        self._controls: dict[str, QWidget] = {}
        self._updating = False
        self._selected_index: int | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        outer.addWidget(splitter)

        # --- canvas -------------------------------------------------------
        from .graph_builder import _canvas_class
        from matplotlib.figure import Figure

        self._figure = Figure(figsize=(self._style.figure_width,
                                       self._style.figure_height))
        self._canvas = _canvas_class()(self._figure)
        self._canvas.setMinimumSize(420, 320)
        self._canvas.mpl_connect("button_press_event", self._on_click)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._canvas, 1)
        left_layout.addWidget(self._build_detail_panel())
        splitter.addWidget(left)

        # --- controls -----------------------------------------------------
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setWidget(self._build_controls())
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setAcceptDrops(True)
        if not self._results.empty:
            self.refresh()
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # ------------------------------------------------------------------ data

    def set_results(self, results: pd.DataFrame) -> None:
        self._results = results.reset_index(drop=True)
        self._selected_index = None
        self._repopulate_column_menus()
        self.refresh()

    def results(self) -> pd.DataFrame:
        return self._results.copy()

    def style(self) -> VolcanoStyle:
        return self._style

    def set_style(self, style: VolcanoStyle) -> None:
        self._style = style
        self._push_style_to_controls()
        self.refresh()

    def merge_annotation_file(self, path, *, on: str | None = None) -> int:
        """Merge a CSV/Excel of annotations onto the results.

        Every column it brings becomes selectable for colour and shape. The
        join column is inferred from whichever shared column matches most
        rows, so a file keyed on ``gene`` and one keyed on ``guide`` both work
        without the user being asked which is which.

        :returns: the number of columns added.
        """
        path = os.fspath(path)
        frame = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
                 else pd.read_csv(path))
        if frame.empty:
            return 0
        shared = [c for c in frame.columns if c in self._results.columns]
        if on is not None:
            key = on
        elif shared:
            # The column that actually matches the most rows, not the first
            # one that happens to share a name.
            key = max(shared, key=lambda c: self._results[c].astype(str).isin(
                frame[c].astype(str)).sum())
        else:
            raise ValueError(
                f"{os.path.basename(path)} shares no column with the results "
                f"({sorted(self._results.columns)[:10]}), so there is nothing "
                f"to join on.")
        incoming = frame.drop_duplicates(subset=[key])
        new_columns = [c for c in incoming.columns if c not in self._results.columns]
        if not new_columns:
            return 0
        # Join on the string form of the key, so 'TGGT1_225160' matches
        # whatever dtype each side happened to be read as -- a numeric-looking
        # guide id read as int on one side and str on the other would
        # otherwise match nothing and silently annotate every row with NaN.
        merged = self._results.copy()
        lookup = incoming.set_index(incoming[key].astype(str))
        keys = self._results[key].astype(str)
        for column in new_columns:
            merged[column] = keys.map(lookup[column])
        self._results = merged
        self._repopulate_column_menus()
        self.refresh()
        return len(new_columns)

    # -------------------------------------------------------------- controls

    def _build_controls(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._group("Data", [
            ("x_column", self._combo([], "Column plotted on the x axis")),
            ("y_column", self._combo([], "Column plotted on the y axis")),
            ("y_neg_log10", self._check("−log₁₀ the y column")),
            ("label_column", self._combo([], "Column holding point names")),
        ]))

        layout.addWidget(self._group("Axes", [
            ("title", self._line("Plot title")),
            ("x_label", self._line("X axis title")),
            ("y_label", self._line("Y axis title (blank = automatic)")),
            ("x_scale", self._combo(SCALES, "X axis scale")),
            ("y_scale", self._combo(SCALES, "Y axis scale")),
            ("invert_x", self._check("Invert x axis")),
            ("invert_y", self._check("Invert y axis")),
            ("split_axis", self._check("Split the y axis (broken axis)")),
            ("split_height_ratio", self._spin(0.1, 0.9, 0.05, 2,
                                              "Height of the upper panel")),
        ]))

        layout.addWidget(self._group("Thresholds", [
            ("alpha", self._spin(1e-6, 0.5, 0.01, 6, "Significance level")),
            ("threshold_method", self._combo(
                ["value", "std", "mad", "quantile"],
                "How the effect-size cut is derived")),
            ("threshold_multiplier", self._spin(
                0.0, 100.0, 0.5, 4,
                "Multiplier applied to the rule above (the quantile itself "
                "when the method is 'quantile')")),
            ("effect_threshold", self._spin(
                -1e6, 1e6, 0.05, 4,
                "Effect-size cut used when the method is 'value'")),
            ("show_alpha_line", self._check("Draw the significance line")),
            ("show_effect_lines", self._check("Draw the effect-size lines")),
            ("show_zero_line", self._check("Draw the zero line")),
        ]))

        layout.addWidget(self._group("Points", [
            ("marker", self._combo(
                [code for code, _ in MARKER_SHAPES], "Marker shape",
                labels=[label for _, label in MARKER_SHAPES])),
            ("marker_size", self._spin(1, 400, 2, 1, "Marker size")),
            ("significant_marker_size", self._spin(
                1, 400, 2, 1, "Marker size for significant points")),
            ("marker_alpha", self._spin(0.05, 1.0, 0.05, 2, "Opacity")),
            ("marker_edge_width", self._spin(0, 5, 0.05, 2, "Edge width")),
            ("marker_edge_color", self._line("Edge colour")),
            ("base_color", self._line("Colour of non-significant points")),
            ("significant_color", self._line("Colour of significant points")),
        ]))

        layout.addWidget(self._group("Colour & shape mapping", [
            ("color_by", self._combo([], "Column that chooses each colour")),
            ("colormap", self._combo(
                [name for group in COLORMAPS.values() for name in group],
                "Colormap")),
            ("shape_by", self._combo([], "Column that chooses each shape")),
            ("show_colorbar", self._check("Show the colour bar")),
        ]))

        layout.addWidget(self._group("Lines", [
            ("line_width", self._spin(0, 10, 0.1, 2, "Threshold line width")),
            ("line_color", self._line("Threshold line colour")),
            ("line_style", self._combo(
                [code for code, _ in LINE_STYLES], "Threshold line style",
                labels=[label for _, label in LINE_STYLES])),
            ("zero_line_width", self._spin(0, 10, 0.1, 2, "Zero line width")),
            ("zero_line_color", self._line("Zero line colour")),
        ]))

        layout.addWidget(self._group("Text", [
            ("font_family", self._combo(FONT_FAMILIES, "Font family")),
            ("font_size", self._spin(4, 48, 0.5, 1, "Base font size")),
            ("title_font_size", self._spin(4, 48, 0.5, 1, "Title size")),
            ("label_font_size", self._spin(4, 48, 0.5, 1, "Annotation size")),
            ("tick_font_size", self._spin(4, 48, 0.5, 1, "Tick label size")),
            ("font_weight", self._combo(
                ["normal", "bold", "light", "medium", "semibold", "heavy"],
                "Font weight")),
            ("annotate_significant", self._check("Label every hit")),
        ]))

        layout.addWidget(self._group("Frame", [
            ("figure_width", self._spin(1, 40, 0.2, 2, "Figure width (in)")),
            ("figure_height", self._spin(1, 40, 0.2, 2, "Figure height (in)")),
            ("dpi", self._int_spin(50, 1200, "Raster export dpi")),
            ("grid", self._check("Show grid")),
            ("grid_axis", self._combo(["x", "y", "both", "none"], "Grid axis")),
            ("grid_color", self._line("Grid colour")),
            ("grid_width", self._spin(0, 5, 0.1, 2, "Grid line width")),
            ("hide_top_right_spines", self._check("Hide top/right spines")),
            ("legend", self._check("Show legend")),
            ("legend_location", self._combo(
                ["best", "upper right", "upper left", "lower left",
                 "lower right", "right", "center left", "center right",
                 "lower center", "upper center", "center"],
                "Legend position")),
            ("transparent", self._check("Transparent background on export")),
        ]))

        buttons = QWidget(self)
        row = QVBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        for text, slot, tip in (
            ("Export PDF…", lambda: self.export("pdf"),
             "Write a vector PDF of exactly this plot"),
            ("Export PNG…", lambda: self.export("png"),
             "Write a raster PNG at the dpi set under Frame"),
        ):
            button = QPushButton(text, self)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            top.addWidget(button)
        row.addLayout(top)
        bottom = QHBoxLayout()
        for text, slot, tip in (
            ("Load annotations…", self._pick_annotation_file,
             "Merge a CSV/Excel of annotations, then colour or shape by any "
             "of its columns"),
            ("Save style", self._save_style, "Save this appearance as JSON"),
            ("Load style", self._load_style, "Restore a saved appearance"),
        ):
            button = QPushButton(text, self)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bottom.addWidget(button)
        row.addLayout(bottom)
        layout.addWidget(buttons)
        layout.addStretch(1)
        return container

    def _group(self, title: str, rows) -> QGroupBox:
        box = QGroupBox(title, self)
        form = QFormLayout(box)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)
        for key, widget in rows:
            self._controls[key] = widget
            form.addRow(widget.property("caption") or key, widget)
        return box

    # Each factory stores the caption on the widget so _group can label it
    # without a parallel table that can fall out of step.
    def _combo(self, options, caption: str, *, labels=None) -> QComboBox:
        widget = QComboBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        for index, option in enumerate(options):
            widget.addItem(labels[index] if labels else str(option), option)
        widget.currentIndexChanged.connect(self._on_control_changed)
        return widget

    def _check(self, caption: str) -> QCheckBox:
        widget = QCheckBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.toggled.connect(self._on_control_changed)
        return widget

    def _spin(self, low, high, step, decimals, caption) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.valueChanged.connect(self._on_control_changed)
        return widget

    def _int_spin(self, low, high, caption) -> QSpinBox:
        widget = QSpinBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.setRange(low, high)
        widget.valueChanged.connect(self._on_control_changed)
        return widget

    def _line(self, caption: str) -> QLineEdit:
        widget = QLineEdit(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.editingFinished.connect(self._on_control_changed)
        return widget

    def _build_detail_panel(self) -> QWidget:
        box = QGroupBox("Selected point", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self._detail_hint = QLabel("Click any point to see everything known "
                                   "about it.", self)
        self._detail_hint.setWordWrap(True)
        layout.addWidget(self._detail_hint)
        self._detail_table = QTableWidget(0, 2, self)
        self._detail_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._detail_table.verticalHeader().setVisible(False)
        self._detail_table.horizontalHeader().setStretchLastSection(True)
        self._detail_table.setMaximumHeight(190)
        layout.addWidget(self._detail_table)
        box.setMaximumHeight(260)
        return box

    # --------------------------------------------------------------- syncing

    def _repopulate_column_menus(self) -> None:
        """Refill the column dropdowns from whatever columns now exist."""
        columns = list(self._results.columns)
        numeric = [c for c in columns
                   if pd.api.types.is_numeric_dtype(self._results[c])]
        mappable = [c for c in columns if c not in _NON_MAPPING_COLUMNS]
        self._updating = True
        try:
            for key, options, allow_none in (
                ("x_column", numeric, False),
                ("y_column", numeric, False),
                ("label_column", columns, False),
                ("color_by", mappable, True),
                ("shape_by", mappable, True),
            ):
                widget = self._controls.get(key)
                if widget is None:
                    continue
                current = widget.currentData()
                widget.clear()
                if allow_none:
                    widget.addItem("— none —", None)
                for option in options:
                    widget.addItem(str(option), option)
                index = widget.findData(current)
                if index < 0:
                    index = widget.findData(getattr(self._style, key, None))
                widget.setCurrentIndex(max(index, 0))
        finally:
            self._updating = False

    def _push_style_to_controls(self) -> None:
        """Write the style into every control without triggering a redraw."""
        self._updating = True
        try:
            for key, widget in self._controls.items():
                value = getattr(self._style, key, None)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index < 0:
                        index = widget.findText(str(value))
                    widget.setCurrentIndex(max(index, 0))
                elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    if value is not None:
                        widget.setValue(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText("" if value is None else str(value))
        finally:
            self._updating = False

    def _pull_style_from_controls(self) -> None:
        for key, widget in self._controls.items():
            if isinstance(widget, QCheckBox):
                setattr(self._style, key, bool(widget.isChecked()))
            elif isinstance(widget, QComboBox):
                index = widget.currentIndex()
                setattr(self._style, key,
                        widget.itemData(index) if index >= 0 else None)
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                setattr(self._style, key, widget.value())
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                setattr(self._style, key, text)

    def _on_control_changed(self, *_args) -> None:
        if self._updating:
            return
        self._pull_style_from_controls()
        self.refresh()
        self.style_changed.emit()

    # -------------------------------------------------------------- painting

    def refresh(self) -> None:
        """Redraw the canvas from the current results and style."""
        if self._results.empty:
            return
        if not self._controls.get("x_column", QComboBox()).count():
            self._repopulate_column_menus()
            self._push_style_to_controls()
        try:
            # A split axis needs limits; derive sensible ones the first time
            # rather than refusing to draw.
            if self._style.split_axis and not self._style.split_y_lims:
                self._style.split_y_lims = self._suggest_split()
            _figure, self._panels = render_volcano(
                self._results, self._style, figure=self._figure)
        except Exception as error:  # noqa: BLE001 - a bad style must not crash
            self._figure.clear()
            axis = self._figure.add_subplot(111)
            axis.text(0.5, 0.5, f"Cannot draw this plot:\n{error}",
                      ha="center", va="center", wrap=True, fontsize=9,
                      color="#C44E52")
            axis.set_axis_off()
            self._panels = [axis]
        self._canvas.draw_idle()

    def _suggest_split(self):
        """Split just above the null cloud, at the 99th percentile of y."""
        y = self._plotted_y()
        if y.size == 0:
            return None
        cut = float(np.nanquantile(y, 0.99))
        top = float(np.nanmax(y))
        if not np.isfinite(cut) or not np.isfinite(top) or top <= cut:
            return None
        pad = (top - cut) * 0.1 or 0.1
        return ((0.0, cut), (cut + pad * 0.5, top + pad))

    def _plotted_y(self) -> np.ndarray:
        raw = pd.to_numeric(self._results[self._style.y_column],
                            errors="coerce").to_numpy(float)
        if self._style.y_neg_log10:
            return -np.log10(np.clip(raw, np.finfo(float).tiny, None))
        return raw

    # -------------------------------------------------------------- clicking

    def _on_click(self, event) -> None:
        if event.inaxes is None or self._results.empty:
            return
        index = self.nearest_point(event.xdata, event.ydata, event.inaxes)
        if index is None:
            return
        self.select_point(index)

    def nearest_point(self, x: float, y: float, axes=None) -> int | None:
        """Index of the point closest to ``(x, y)`` in axis-normalised space.

        Normalising by each axis's visible range is what makes a click land on
        the point that LOOKS closest. Raw Euclidean distance in data units
        picks the wrong point whenever the axes have different scales, which
        on a volcano they always do.
        """
        if self._results.empty:
            return None
        xs = pd.to_numeric(self._results[self._style.x_column],
                           errors="coerce").to_numpy(float)
        ys = self._plotted_y()
        x_span = y_span = 1.0
        if axes is not None:
            low, high = axes.get_xlim()
            x_span = abs(high - low) or 1.0
            low, high = axes.get_ylim()
            y_span = abs(high - low) or 1.0
        distance = np.hypot((xs - x) / x_span, (ys - y) / y_span)
        if not np.isfinite(distance).any():
            return None
        index = int(np.nanargmin(distance))
        # Ignore clicks on empty space: 5% of the diagonal is about the radius
        # of a marker at default size.
        return index if distance[index] <= 0.05 else None

    def select_point(self, index: int) -> dict:
        """Select a point by row index and show everything known about it."""
        detail = point_details(self._results, index, self._style)
        self._selected_index = int(index)
        rows = [(k, v) for k, v in detail.items() if not k.startswith("_")]
        rows += [(k.lstrip("_") + " (plotted)", v)
                 for k, v in detail.items() if k.startswith("_")]
        self._detail_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            self._detail_table.setItem(row, 0, QTableWidgetItem(str(key)))
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            self._detail_table.setItem(row, 1, item)
        name = detail.get(self._style.label_column, index)
        self._detail_hint.setText(f"Selected {name}")
        self.point_selected.emit(detail)
        return detail

    def selected_index(self) -> int | None:
        return self._selected_index

    # ---------------------------------------------------------------- export

    def export(self, fmt: str = "pdf", path: str | None = None) -> str | None:
        """Re-render at print size and write the file.

        Not a screenshot: the figure is drawn again from the same style, so a
        PDF stays vector and a PNG honours the dpi under Frame regardless of
        how large the widget happens to be on screen.
        """
        if self._results.empty:
            return None
        if path is None:
            filters = {"pdf": "PDF (*.pdf)", "png": "PNG (*.png)",
                       "svg": "SVG (*.svg)"}
            path, _selected = QFileDialog.getSaveFileName(
                self, f"Export volcano as {fmt.upper()}",
                f"volcano.{fmt}", filters.get(fmt, "All files (*)"))
            if not path:
                return None
        if not path.lower().endswith(f".{fmt}"):
            path = f"{path}.{fmt}"
        from matplotlib.figure import Figure

        figure = Figure(figsize=(self._style.figure_width,
                                 self._style.figure_height),
                        dpi=self._style.dpi)
        try:
            render_volcano(self._results, self._style, figure=figure,
                           save_path=path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(error))
            return None
        return path

    def _save_style(self) -> str | None:
        path, _selected = QFileDialog.getSaveFileName(
            self, "Save plot style", "volcano_style.json", "JSON (*.json)")
        if not path:
            return None
        return self._style.save(path)

    def _load_style(self) -> str | None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Load plot style", "", "JSON (*.json)")
        if not path:
            return None
        self.set_style(VolcanoStyle.load(path))
        return path

    def _pick_annotation_file(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Load an annotation table", "",
            "Tables (*.csv *.tsv *.txt *.xlsx);;All files (*)")
        if not path:
            return
        try:
            added = self.merge_annotation_file(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Could not merge annotations", str(error))
            return
        QMessageBox.information(
            self, "Annotations merged",
            f"Added {added} column{'s' if added != 1 else ''}. They are now "
            f"available under Colour & shape mapping.")

    # ------------------------------------------------------------ drag/drop

    @staticmethod
    def _dropped_paths(event) -> list[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if self._dropped_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt name
        paths = self._dropped_paths(event)
        if not paths:
            event.ignore()
            return
        for path in paths:
            if path.lower().endswith(".json"):
                self.set_style(VolcanoStyle.load(path))
            else:
                try:
                    self.merge_annotation_file(path)
                except Exception:  # noqa: BLE001 - drop is best-effort
                    continue
        event.acceptProposedAction()


__all__ = ["VolcanoExplorer"]
