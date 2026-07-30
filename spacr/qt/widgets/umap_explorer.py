"""Interactive Image UMAP viewer with click, lasso, and DB annotation."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ...umap_annotations import write_umap_annotations

LOG = logging.getLogger("spacr.qt.umap_explorer")


class _AnnotationWorker(QThread):
    """Commit a selection without blocking the Qt event loop on SQLite."""

    finished_result = Signal(int, int, str)

    def __init__(self, records, values, column, parent=None):
        super().__init__(parent)
        self._records = list(records)
        self._values = list(values)
        self._column = column

    def run(self):
        try:
            updated, skipped = write_umap_annotations(
                self._records, self._values, self._column)
            self.finished_result.emit(updated, skipped, "")
        except Exception as exc:
            LOG.info("UMAP annotation write failed", exc_info=True)
            self.finished_result.emit(0, len(self._records), str(exc))


class ImageUmapExplorer(QWidget):
    """Zoomable embedding: click a point, lasso a group, write labels."""

    annotation_finished = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._embedding = np.empty((0, 2), dtype=float)
        self._labels = np.empty(0, dtype=int)
        self._records: List[Dict] = []
        self._selected = np.empty(0, dtype=int)
        self._picked: Optional[int] = None
        self._worker: Optional[_AnnotationWorker] = None
        self._display = {
            "point_size": 26,
            "point_color": "cluster",
            "point_alpha": 0.65,
            "outline_width": 1.0,
            "canvas_width": 900,
            "sidebar_width": 280,
        }
        self._build_ui()

    def _build_ui(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT)

        class _OwnedTimerFigureCanvas(FigureCanvasQTAgg):
            """Figure canvas whose deferred draw cannot outlive the widget.

            Matplotlib's Qt canvas uses static ``QTimer.singleShot`` calls.
            Those callbacks are not owned by the canvas and can consequently
            run after Qt has deleted it. An owned timer is destroyed together
            with the canvas, so lasso/display updates cannot draw a dangling
            C++ object.
            """

            def __init__(self, figure):
                super().__init__(figure)
                self._spacr_draw_timer = QTimer(self)
                self._spacr_draw_timer.setSingleShot(True)
                self._spacr_draw_timer.timeout.connect(self._spacr_draw)

            def draw_idle(self):
                self._draw_pending = True
                if not self._spacr_draw_timer.isActive():
                    self._spacr_draw_timer.start(0)

            def _spacr_draw(self):
                if not self._draw_pending:
                    return
                self._draw_pending = False
                try:
                    self.draw()
                except RuntimeError:
                    # Qt may be closing the parent hierarchy in this same
                    # event-loop turn. There is nothing left to repaint.
                    return

            def cancel_pending_draw(self):
                self._spacr_draw_timer.stop()
                self._draw_pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._body_splitter = QSplitter(Qt.Horizontal, self)
        self._body_splitter.setChildrenCollapsible(False)
        from ..theme import active_palette
        surface = active_palette()["surface"]
        self._figure = Figure(figsize=(8, 6), facecolor=surface)
        self._canvas = _OwnedTimerFigureCanvas(self._figure)
        self._canvas.setStyleSheet(f"background: {surface};")
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        chart = QVBoxLayout()
        chart.addWidget(self._toolbar)
        chart.addWidget(self._canvas, 1)
        chart_wrap = QWidget(self)
        chart_wrap.setLayout(chart)
        chart_wrap.setStyleSheet(f"background: {surface};")
        self._body_splitter.addWidget(chart_wrap)

        side = QVBoxLayout()
        self._preview = QLabel("Click a point to preview its image.", self)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(220, 220)
        self._preview.setStyleSheet("border: 1px solid palette(mid);")
        side.addWidget(self._preview)
        self._point_label = QLabel("", self)
        self._point_label.setWordWrap(True)
        side.addWidget(self._point_label)

        form = QFormLayout()
        self._cluster_box = QComboBox(self)
        self._cluster_box.currentIndexChanged.connect(self._select_cluster)
        form.addRow("Select cluster", self._cluster_box)
        self._column = QLineEdit("umap_annotation", self)
        self._column.setToolTip(
            "Column created/updated on png_list when labels are applied.")
        form.addRow("DB column", self._column)
        self._value = QSpinBox(self)
        self._value.setRange(-1_000_000, 1_000_000)
        self._value.setValue(1)
        form.addRow("Manual label", self._value)
        side.addLayout(form)

        self._apply_selected = QPushButton("Label lasso selection", self)
        self._apply_selected.setObjectName("PrimaryButton")
        self._apply_selected.clicked.connect(self._write_selected)
        side.addWidget(self._apply_selected)
        self._apply_clusters = QPushButton(
            "Propagate automatic clusters", self)
        self._apply_clusters.setToolTip(
            "Write the current DBSCAN/KMeans cluster number for every point.")
        self._apply_clusters.clicked.connect(self._write_clusters)
        side.addWidget(self._apply_clusters)
        self._status = QLabel("Waiting for an embedding.", self)
        self._status.setWordWrap(True)
        side.addWidget(self._status)
        side.addStretch(1)
        side_wrap = QWidget(self)
        side_wrap.setLayout(side)
        side_wrap.setStyleSheet(f"background: {surface};")
        self._body_splitter.addWidget(side_wrap)
        root.addWidget(self._body_splitter, 1)

        self._axes = self._figure.add_subplot(111)
        self._axes.set_facecolor(surface)
        self._scatter = None
        self._selection_artist = None
        self._picked_artist = None
        self._lasso = None
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._body_splitter.setSizes([
            int(self._display["canvas_width"]),
            int(self._display["sidebar_width"]),
        ])

    def set_payload(self, payload: Dict) -> None:
        """Load the arrays/records attached by ``generate_image_umap``."""
        embedding = np.asarray(payload.get("embedding", []), dtype=float)
        if embedding.ndim != 2 or embedding.shape[1:] != (2,):
            raise ValueError("UMAP payload embedding must have shape (N, 2)")
        labels = np.asarray(payload.get("labels", []))
        records = list(payload.get("records", []))
        if len(labels) != len(embedding) or len(records) != len(embedding):
            raise ValueError("UMAP payload arrays must have equal lengths")
        self._embedding = embedding
        self._labels = labels
        self._records = records
        display = payload.get("display")
        if isinstance(display, dict):
            for key in self._display:
                if key in display and display[key] is not None:
                    self._display[key] = display[key]
            self._body_splitter.setSizes([
                int(self._display["canvas_width"]),
                int(self._display["sidebar_width"]),
            ])
        self._selected = np.empty(0, dtype=int)
        self._picked = None
        self._draw_embedding()

    def _draw_embedding(self) -> None:
        from matplotlib.widgets import LassoSelector
        from ..theme import active_palette

        palette = active_palette()
        background = palette["surface_alt"]
        foreground = palette["fg"]
        self._axes.clear()
        self._figure.patch.set_facecolor(background)
        self._axes.set_facecolor(background)
        requested_color = str(self._display["point_color"]).strip()
        color_key = requested_color.lower()
        scatter_kwargs = {}
        if color_key in {"", "cluster", "viridis"}:
            scatter_kwargs.update(c=self._labels, cmap="viridis")
        else:
            from matplotlib.colors import is_color_like
            if is_color_like(requested_color):
                scatter_kwargs["color"] = requested_color
            else:
                scatter_kwargs.update(c=self._labels, cmap="viridis")
        self._scatter = self._axes.scatter(
            self._embedding[:, 0], self._embedding[:, 1],
            s=float(self._display["point_size"]),
            alpha=float(self._display["point_alpha"]),
            **scatter_kwargs,
        )
        self._axes.set_xlabel("UMAP Dimension 1")
        self._axes.set_ylabel("UMAP Dimension 2")
        self._axes.set_title("Click a point to preview · drag a lasso to select")
        self._axes.tick_params(axis="both", colors=foreground)
        self._axes.xaxis.label.set_color(foreground)
        self._axes.yaxis.label.set_color(foreground)
        self._axes.title.set_color(foreground)
        for spine in self._axes.spines.values():
            spine.set_color(foreground)
        self._selection_artist = self._axes.scatter(
            [], [], s=70, facecolors="none", edgecolors=foreground,
            linewidths=float(self._display["outline_width"]))
        self._picked_artist = self._axes.scatter(
            [], [], s=110, facecolors="none", edgecolors="#ffcc33",
            linewidths=float(self._display["outline_width"]))
        if self._lasso is not None:
            self._lasso.disconnect_events()
        self._lasso = LassoSelector(
            self._axes, onselect=self._on_lasso,
            props={
                "color": foreground,
                "linewidth": float(self._display["outline_width"]),
            },
        )
        self._cluster_box.blockSignals(True)
        self._cluster_box.clear()
        self._cluster_box.addItem("—", None)
        for label in sorted(np.unique(self._labels), key=lambda value: str(value)):
            self._cluster_box.addItem(str(label), label)
        self._cluster_box.blockSignals(False)
        writable = sum(
            bool(row.get("db_path") and row.get("db_png_path"))
            for row in self._records)
        self._status.setText(
            f"{len(self._records)} points · {writable} database-backed · "
            "drag around points to select them.")
        if len(self._selected):
            self._selection_artist.set_offsets(
                self._embedding[self._selected])
        if self._picked is not None:
            self._picked_artist.set_offsets(
                self._embedding[self._picked].reshape(1, 2))
        self._canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        """Zoom around the pointer with the mouse wheel."""
        if (event.inaxes is not self._axes or event.xdata is None
                or event.ydata is None):
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = self._axes.get_xlim()
        y0, y1 = self._axes.get_ylim()
        self._axes.set_xlim(
            event.xdata - (event.xdata - x0) * factor,
            event.xdata + (x1 - event.xdata) * factor,
        )
        self._axes.set_ylim(
            event.ydata - (event.ydata - y0) * factor,
            event.ydata + (y1 - event.ydata) * factor,
        )
        self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if (event.inaxes is not self._axes or event.xdata is None
                or not len(self._embedding)):
            return
        click = np.array([event.xdata, event.ydata], dtype=float)
        spans = np.ptp(self._embedding, axis=0)
        spans[spans == 0] = 1.0
        distance = np.linalg.norm((self._embedding - click) / spans, axis=1)
        self.show_point(int(np.argmin(distance)))

    def show_point(self, index: int) -> None:
        """Preview one point's image and database identity."""
        if not (0 <= int(index) < len(self._records)):
            return
        index = int(index)
        self._picked = index
        point = self._embedding[index]
        self._picked_artist.set_offsets(point.reshape(1, 2))
        record = self._records[index]
        source = record.get("image")
        try:
            if source is None:
                raise ValueError("No image source for this point")
            if hasattr(source, "array"):
                image = Image.fromarray(np.asarray(source.array())).convert("RGB")
            else:
                with Image.open(source) as opened:
                    image = opened.convert("RGB")
            image.thumbnail((360, 360), Image.Resampling.LANCZOS)
            qimage = QImage(ImageQt(image)).copy()
            self._preview.setPixmap(QPixmap.fromImage(qimage))
        except Exception as exc:
            self._preview.setPixmap(QPixmap())
            self._preview.setText(f"Preview unavailable\n{exc}")
        self._point_label.setText(
            f"Point {index + 1}/{len(self._records)} · "
            f"cluster {self._labels[index]}\n"
            f"{record.get('db_png_path') or record.get('display_name') or ''}")
        self._canvas.draw_idle()

    def _on_lasso(self, vertices: Sequence) -> None:
        from matplotlib.path import Path

        inside = Path(vertices).contains_points(self._embedding)
        self._selected = np.flatnonzero(inside)
        self._refresh_selection()

    def _select_cluster(self, _index: int) -> None:
        label = self._cluster_box.currentData()
        if label is None:
            return
        self._selected = np.flatnonzero(self._labels == label)
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        points = (self._embedding[self._selected]
                  if len(self._selected) else np.empty((0, 2)))
        self._selection_artist.set_offsets(points)
        self._status.setText(f"{len(self._selected)} point(s) selected.")
        if len(self._selected):
            self.show_point(int(self._selected[0]))
        self._canvas.draw_idle()

    def _write_selected(self) -> None:
        if not len(self._selected):
            self._status.setText("Draw a lasso or select a cluster first.")
            return
        records = [self._records[i] for i in self._selected]
        values = [self._value.value()] * len(records)
        self._start_write(records, values, "manual selection")

    def _write_clusters(self) -> None:
        self._start_write(
            self._records, self._labels.tolist(), "automatic clusters")

    def _start_write(self, records, values, label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("An annotation write is already running.")
            return
        column = self._column.text().strip()
        self._set_write_enabled(False)
        self._status.setText(f"Writing {label} to {column or '(no column)'}…")
        worker = _AnnotationWorker(records, values, column, self)
        worker.finished_result.connect(self._on_write_done, Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _set_write_enabled(self, enabled: bool) -> None:
        self._apply_selected.setEnabled(enabled)
        self._apply_clusters.setEnabled(enabled)

    @Slot(int, int, str)
    def _on_write_done(self, updated: int, skipped: int, error: str) -> None:
        self._worker = None
        self._set_write_enabled(True)
        if error:
            self._status.setText(f"Database write failed: {error}")
        else:
            self._status.setText(
                f"Updated {updated} png_list row(s); skipped {skipped}.")
        self.annotation_finished.emit(updated, skipped)

    def closeEvent(self, event):
        worker = self._worker
        if worker is not None:
            worker.requestInterruption()
            worker.wait()
            self._worker = None
        # FigureCanvasQTAgg implements draw_idle with a zero-delay Qt timer.
        # Cancel that pending draw before Qt deletes the C++ canvas.
        if self._lasso is not None:
            self._lasso.disconnect_events()
            self._lasso = None
        if getattr(self, "_canvas", None) is not None:
            self._canvas.cancel_pending_draw()
        super().closeEvent(event)
