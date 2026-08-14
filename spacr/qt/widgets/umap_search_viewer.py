"""Native interactive viewer and gallery for Image UMAP search results.

Unlike the general spaCR figure queue, these widgets hold the coordinate
arrays themselves.  A 3-D map can therefore be spun and recoloured after it
has been computed, and clicking a gallery tile opens the exact embedding that
was scored rather than a PNG or a stochastic refit.

The renderer uses Qt's painter only.  It has no OpenGL or pyqtgraph dependency,
works in the offscreen test platform, and keeps the RAPIDS extra about compute
rather than pulling in a second GUI stack.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QIcon, QImage, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem,
    QMenu, QPushButton, QVBoxLayout, QWidget,
)

BACKGROUND = QColor(8, 10, 14)
FOREGROUND = QColor(226, 232, 240)
MUTED = QColor(148, 163, 184)
POINT = QColor(94, 194, 255, 205)
NOISE = QColor(115, 123, 135, 125)
CLUSTER_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 190, 64), (83, 190, 255), (102, 224, 139), (255, 102, 132),
    (183, 132, 255), (67, 222, 211), (255, 139, 72), (162, 213, 87),
    (240, 102, 218), (78, 151, 255), (239, 220, 91), (126, 231, 190),
)


def _coordinates(value: Any) -> np.ndarray:
    coords = np.asarray(value, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("UMAP coordinates must have shape (rows, 2) or (rows, 3).")
    if not len(coords):
        raise ValueError("UMAP coordinates are empty.")
    if not np.isfinite(coords).all():
        raise ValueError("UMAP coordinates contain NaN or infinite values.")
    if coords.shape[1] == 2:
        coords = np.column_stack((coords, np.zeros(len(coords))))
    return coords


@lru_cache(maxsize=128)
def _colormap_rgb(name: str, count: int) -> Tuple[Tuple[int, int, int], ...]:
    """Sample a Matplotlib colour map lazily; retain a Qt-only fallback."""
    count = max(1, int(count))
    if name == "spaCR":
        return tuple(CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
                     for index in range(count))
    try:
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(str(name))
        values = cmap(np.linspace(0.0, 1.0, count))
        return tuple(tuple(int(round(float(channel) * 255.0))
                           for channel in rgba[:3]) for rgba in values)
    except Exception:
        return tuple(CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
                     for index in range(count))


def available_colormaps() -> List[str]:
    """Every installed Matplotlib colour map, plus spaCR's native palette."""
    try:
        from matplotlib import colormaps
        return ["spaCR", *sorted(str(name) for name in colormaps)]
    except Exception:
        return ["spaCR", "viridis", "plasma", "inferno", "magma"]


def colors_for_labels(labels: Optional[Sequence[int]], count: int, *,
                      cmap: str = "spaCR", alpha: float = 0.86) -> List[QColor]:
    """One readable colour per point, with HDBSCAN noise in grey."""
    opacity = int(round(255.0 * float(np.clip(alpha, 0.05, 1.0))))
    if labels is None:
        if cmap == "spaCR":
            colour = QColor(POINT)
            colour.setAlpha(opacity)
            return [QColor(colour) for _ in range(count)]
        palette = _colormap_rgb(cmap, count)
        return [QColor(*rgb, opacity) for rgb in palette]
    values = np.asarray(labels)
    if values.shape != (count,):
        return [QColor(POINT) for _ in range(count)]
    ids = {value: index for index, value in enumerate(
        sorted({int(value) for value in values if int(value) >= 0}))}
    out: List[QColor] = []
    for raw in values:
        value = int(raw)
        if value < 0:
            out.append(QColor(NOISE))
            continue
        palette = _colormap_rgb(cmap, max(1, len(ids)))
        red, green, blue = palette[ids[value] % len(palette)]
        out.append(QColor(red, green, blue, opacity))
    return out


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    around_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0),
                         (-sy, 0.0, cy)))
    around_x = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp),
                         (0.0, sp, cp)))
    return around_x @ around_y


def project_points(coords: Any, width: int, height: int, *,
                   yaw: float = 0.0, pitch: float = 0.0,
                   zoom: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Orthographically project a 2-D/3-D map without distorting its axes."""
    xyz = _coordinates(coords)
    centred = xyz - np.mean(xyz, axis=0, keepdims=True)
    rotated = centred @ _rotation(float(yaw), float(pitch)).T
    span = float(np.max(np.ptp(rotated[:, :2], axis=0)))
    usable = max(1.0, min(float(width), float(height)) * 0.88)
    scale = usable * max(0.05, float(zoom)) / max(span, 1e-9)
    points = np.empty((len(rotated), 2), dtype=float)
    points[:, 0] = rotated[:, 0] * scale + float(width) / 2.0
    points[:, 1] = -rotated[:, 1] * scale + float(height) / 2.0
    return points, rotated[:, 2]


def axis_frame(coords: Any, width: int, height: int, *, yaw: float = 0.0,
               pitch: float = 0.0, zoom: float = 1.0) -> dict:
    """Return projected grid lines and exactly two or three primary axes."""
    raw = np.asarray(coords, dtype=float)
    dimensions = raw.shape[1] if raw.ndim == 2 else 0
    xyz = _coordinates(raw)
    centre = np.mean(xyz, axis=0, keepdims=True)
    rotation = _rotation(float(yaw), float(pitch))
    rotated = (xyz - centre) @ rotation.T
    span = float(np.max(np.ptp(rotated[:, :2], axis=0)))
    usable = max(1.0, min(float(width), float(height)) * 0.88)
    scale = usable * max(0.05, float(zoom)) / max(span, 1e-9)

    def projected(points: Sequence[Sequence[float]]) -> np.ndarray:
        values = (np.asarray(points, dtype=float) - centre) @ rotation.T
        result = np.empty((len(values), 2), dtype=float)
        result[:, 0] = values[:, 0] * scale + float(width) / 2.0
        result[:, 1] = -values[:, 1] * scale + float(height) / 2.0
        return result

    low = np.min(xyz, axis=0)
    high = np.max(xyz, axis=0)
    # Degenerate dimensions still receive a visible axis of finite length.
    high = np.where(np.isclose(high, low), low + 1.0, high)
    origin = low.copy()
    axes = []
    for index in range(dimensions):
        end = origin.copy()
        end[index] = high[index]
        line = projected((origin, end))
        axes.append((line[0], line[1], f"Dimension {index + 1}"))

    grid = []
    fractions = (0.2, 0.4, 0.6, 0.8)
    # A readable base-plane grid: X/Y in both 2D and 3D. The third axis rises
    # from the same origin in 3D and rotates with the map.
    for fraction in fractions:
        x = low[0] + (high[0] - low[0]) * fraction
        grid.append(tuple(projected(((x, low[1], low[2]),
                                     (x, high[1], low[2])))))
        y = low[1] + (high[1] - low[1]) * fraction
        grid.append(tuple(projected(((low[0], y, low[2]),
                                     (high[0], y, low[2])))))
    return {"dimensions": dimensions, "axes": axes, "grid": grid}


class UmapAppearanceDialog(QDialog):
    """Non-modal point renderer controls for one embedding view."""

    applied = Signal(dict)

    def __init__(self, appearance: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("UmapAppearanceDialog")
        self.setWindowTitle("Embedding appearance")
        form = QFormLayout(self)
        self.marker = QComboBox(self)
        self.marker.addItems(["circle", "square", "diamond", "cross"])
        self.marker.setCurrentText(str(appearance.get("marker", "circle")))
        form.addRow("Point rendering", self.marker)
        self.size = QDoubleSpinBox(self)
        self.size.setRange(1.0, 24.0)
        self.size.setSingleStep(0.5)
        self.size.setValue(float(appearance.get("size", 3.2)))
        form.addRow("Point size", self.size)
        self.alpha = QDoubleSpinBox(self)
        self.alpha.setRange(0.05, 1.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setDecimals(2)
        self.alpha.setValue(float(appearance.get("alpha", 0.86)))
        form.addRow("Opacity", self.alpha)
        self.cmap = QComboBox(self)
        self.cmap.addItems(available_colormaps())
        self.cmap.setCurrentText(str(appearance.get("cmap", "spaCR")))
        form.addRow("Colour map", self.cmap)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Close, parent=self)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.close)
        form.addRow(buttons)

    def values(self) -> dict:
        return {
            "marker": self.marker.currentText(),
            "size": self.size.value(),
            "alpha": self.alpha.value(),
            "cmap": self.cmap.currentText(),
        }

    def _apply(self) -> None:
        self.applied.emit(self.values())


def thumbnail_image(coords: Any, labels: Optional[Sequence[int]] = None,
                    *, size: int = 170) -> QImage:
    """Deterministic black-background thumbnail used by the all-map grid."""
    size = max(48, int(size))
    xyz = _coordinates(coords)
    points, depth = project_points(xyz, size, size, yaw=0.22, pitch=-0.16)
    colours = colors_for_labels(labels, len(xyz))
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(BACKGROUND)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen()
    pen.setWidthF(2.2)
    pen.setCapStyle(Qt.RoundCap)
    for index in np.argsort(depth):
        pen.setColor(colours[int(index)])
        painter.setPen(pen)
        x, y = points[int(index)]
        painter.drawPoint(QPointF(float(x), float(y)))
    painter.end()
    return image


class UmapEmbeddingView(QWidget):
    """A black-background 2-D/3-D point view; drag a 3-D map to spin it."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("UmapEmbeddingView")
        self.setMinimumSize(280, 260)
        self.setMouseTracking(True)
        self._coords: Optional[np.ndarray] = None
        self._dimensions = 0
        self._labels: Optional[np.ndarray] = None
        self._caption = "No search has been run yet."
        self._backend = ""
        self._yaw = 0.22
        self._pitch = -0.16
        self._zoom = 1.0
        self._drag_at = None
        self._point_size = 3.2
        self._marker = "circle"
        self._point_alpha = 0.86
        self._cmap = "spaCR"
        self._appearance_dialog: Optional[UmapAppearanceDialog] = None

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        return None if self._coords is None else self._coords.copy()

    @property
    def labels(self) -> Optional[np.ndarray]:
        return None if self._labels is None else self._labels.copy()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def appearance(self) -> dict:
        return {
            "marker": self._marker, "size": self._point_size,
            "alpha": self._point_alpha, "cmap": self._cmap,
        }

    def set_appearance(self, values: dict) -> None:
        """Apply rendering-only changes without changing the embedding."""
        marker = str(values.get("marker", self._marker))
        if marker not in {"circle", "square", "diamond", "cross"}:
            raise ValueError(f"Unknown point rendering: {marker!r}.")
        self._marker = marker
        self._point_size = float(np.clip(
            values.get("size", self._point_size), 1.0, 24.0))
        self._point_alpha = float(np.clip(
            values.get("alpha", self._point_alpha), 0.05, 1.0))
        cmap = str(values.get("cmap", self._cmap))
        self._cmap = cmap if cmap in available_colormaps() else "spaCR"
        self.update()

    def open_appearance_editor(self) -> UmapAppearanceDialog:
        dialog = UmapAppearanceDialog(self.appearance, self)
        dialog.applied.connect(self.set_appearance)
        dialog.finished.connect(lambda _result: setattr(
            self, "_appearance_dialog", None))
        self._appearance_dialog = dialog
        dialog.show()
        dialog.raise_()
        return dialog

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        appearance = menu.addAction("Appearance…")
        reset = menu.addAction("Reset view")
        chosen = menu.exec(event.globalPos())
        if chosen is appearance:
            self.open_appearance_editor()
        elif chosen is reset:
            self.reset_view()
        event.accept()

    def clear(self, message: str = "No search has been run yet.") -> None:
        self._coords = None
        self._dimensions = 0
        self._labels = None
        self._caption = str(message)
        self._backend = ""
        self.update()

    def set_embedding(self, coords: Any, *, labels: Optional[Sequence[int]] = None,
                      caption: str = "", backend: str = "") -> None:
        values = np.asarray(coords, dtype=float)
        dimensions = values.shape[1] if values.ndim == 2 else 0
        self._coords = _coordinates(values)
        self._dimensions = dimensions
        self._labels = None if labels is None else np.asarray(labels, dtype=int)
        if self._labels is not None and self._labels.shape != (len(self._coords),):
            raise ValueError("Cluster labels must contain one value per UMAP point.")
        self._caption = str(caption or f"{dimensions}D UMAP")
        self._backend = str(backend or "")
        self.reset_view()

    def set_labels(self, labels: Optional[Sequence[int]]) -> None:
        if self._coords is None:
            return
        values = None if labels is None else np.asarray(labels, dtype=int)
        if values is not None and values.shape != (len(self._coords),):
            raise ValueError("Cluster labels must contain one value per UMAP point.")
        self._labels = values
        self.update()

    def reset_view(self) -> None:
        self._yaw = 0.22
        self._pitch = -0.16
        self._zoom = 1.0
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._coords is not None:
            self._drag_at = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_at is not None and self._coords is not None:
            delta = event.position() - self._drag_at
            self._drag_at = event.position()
            if self.dimensions == 3:
                self._yaw += float(delta.x()) * 0.009
                self._pitch = float(np.clip(
                    self._pitch + float(delta.y()) * 0.009, -1.45, 1.45))
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_at = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._coords is None:
            return super().wheelEvent(event)
        steps = float(event.angleDelta().y()) / 120.0
        self._zoom = float(np.clip(self._zoom * (1.12 ** steps), 0.2, 8.0))
        self.update()
        event.accept()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._coords is None:
            painter.setPen(MUTED)
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                             self._caption)
            painter.end()
            return
        points, depth = project_points(
            self._coords, self.width(), self.height(), yaw=self._yaw,
            pitch=self._pitch if self.dimensions == 3 else 0.0,
            zoom=self._zoom)
        frame = axis_frame(
            self._coords[:, :self.dimensions], self.width(), self.height(),
            yaw=self._yaw, pitch=self._pitch if self.dimensions == 3 else 0.0,
            zoom=self._zoom)
        grid_pen = QPen(QColor(71, 85, 105, 105))
        grid_pen.setWidthF(0.8)
        painter.setPen(grid_pen)
        for start, end in frame["grid"]:
            painter.drawLine(QPointF(*start), QPointF(*end))
        axis_pen = QPen(QColor(148, 163, 184, 205))
        axis_pen.setWidthF(1.35)
        painter.setPen(axis_pen)
        for start, end, label in frame["axes"]:
            painter.drawLine(QPointF(*start), QPointF(*end))
            painter.drawText(QPointF(float(end[0]) + 4.0,
                                     float(end[1]) - 4.0), label)

        colours = colors_for_labels(
            self._labels, len(self._coords), cmap=self._cmap,
            alpha=self._point_alpha)
        radius = self._point_size / 2.0
        for index in np.argsort(depth):
            colour = colours[int(index)]
            x, y = points[int(index)]
            point = QPointF(float(x), float(y))
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            if self._marker == "circle":
                painter.drawEllipse(point, radius, radius)
            elif self._marker == "square":
                painter.drawRect(QRectF(
                    float(x) - radius, float(y) - radius,
                    self._point_size, self._point_size))
            elif self._marker == "diamond":
                painter.drawPolygon(QPolygonF((
                    QPointF(float(x), float(y) - radius),
                    QPointF(float(x) + radius, float(y)),
                    QPointF(float(x), float(y) + radius),
                    QPointF(float(x) - radius, float(y)),
                )))
            else:
                marker_pen = QPen(colour)
                marker_pen.setWidthF(max(1.0, self._point_size / 2.0))
                painter.setPen(marker_pen)
                painter.drawLine(QPointF(float(x) - radius, float(y)),
                                 QPointF(float(x) + radius, float(y)))
                painter.drawLine(QPointF(float(x), float(y) - radius),
                                 QPointF(float(x), float(y) + radius))
        painter.setPen(FOREGROUND)
        title = self._caption
        if self._backend:
            title += f"  ·  {self._backend}"
        painter.drawText(12, 22, title)
        painter.setPen(MUTED)
        hint = ("drag to spin · wheel to zoom · right-click appearance"
                if self.dimensions == 3
                else "wheel to zoom · right-click appearance")
        painter.drawText(12, self.height() - 12, hint)
        painter.end()


class UmapExplorer(QWidget):
    """Small shell around :class:`UmapEmbeddingView` with a reset control."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.title = QLabel("2D / 3D UMAP")
        self.reset = QPushButton("Reset view")
        self.reset.clicked.connect(self._reset)
        bar.addWidget(self.title)
        bar.addStretch(1)
        bar.addWidget(self.reset)
        layout.addLayout(bar)
        self.view = UmapEmbeddingView(self)
        layout.addWidget(self.view, 1)

    def _reset(self) -> None:
        self.view.reset_view()


class UmapGalleryDialog(QDialog):
    """All table embeddings on black, with a click returning the real trial."""

    trial_chosen = Signal(object)

    def __init__(self, trials: Iterable[Any] = (), parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("UmapGalleryDialog")
        self.setWindowTitle("All Image UMAPs")
        self.resize(900, 650)
        self._trials: List[Any] = []
        layout = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.list = QListWidget(self)
        self.list.setViewMode(QListView.IconMode)
        self.list.setIconSize(QPixmap(170, 170).size())
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setSpacing(8)
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setStyleSheet(
            "QListWidget { background: #080a0e; color: #e2e8f0; }"
            "QListWidget::item:selected { background: #17324d; }")
        self.list.itemClicked.connect(self._choose)
        layout.addWidget(self.list, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close)
        layout.addLayout(footer)
        self.set_trials(trials)

    def set_trials(self, trials: Iterable[Any]) -> None:
        self._trials = [trial for trial in trials
                        if getattr(trial, "extra_metrics", {}).get("embedding") is not None]
        self.list.clear()
        for index, trial in enumerate(self._trials):
            extra = trial.extra_metrics
            image = thumbnail_image(
                extra["embedding"], extra.get("cluster_labels"), size=170)
            params = ", ".join(
                f"{key}={trial.params[key]}" for key in sorted(trial.params))
            backend = str(extra.get("backend", "cpu"))
            score = "-" if trial.score is None else f"{float(trial.score):.4f}"
            clusters = extra.get("n_clusters")
            cluster_text = "" if clusters is None else f" · {int(clusters)} clusters"
            item = QListWidgetItem(
                QIcon(QPixmap.fromImage(image)),
                f"{params}\n{backend} · score {score}{cluster_text}")
            item.setData(Qt.UserRole, index)
            item.setToolTip("Click to load this exact embedding into the 2D / 3D viewer.")
            self.list.addItem(item)
        self.summary.setText(
            f"{len(self._trials)} maps from the search table. Click one to load "
            "its stored coordinates into the UMAP viewer.")

    def _choose(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.UserRole)
        if isinstance(index, int) and 0 <= index < len(self._trials):
            self.trial_chosen.emit(self._trials[index])
