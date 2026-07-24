"""Measure live preview — a grid of object crops from a merged image+mask
array, tuned with the same crop settings the Measure run will use.

Drop (or pick) a ``merged/*.npy`` array; the panel crops every object out of a
chosen mask slice with the current channel/size/area settings and shows them on
a dark-gray grid with rounded corners (like the annotate view). Click a crop to
inspect its label + area. A "Propagate settings" toggle pushes the crop
settings (png dims, png size, bounding box, normalise) into the main Measure
settings panel so the preview drives the real run — mirroring the Mask app's
live preview but tailored to measurement + image cropping, on the new
``crop_objects_from_array`` backend.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger("spacr.qt.measure_preview")

# Default mask-slice index per object class (4 image channels then masks).
_MASK_DIMS = {"cell": 4, "nucleus": 5, "pathogen": 6, "organelle": 7}
_SUPPORTED = (".npy",)


def _rounded_pixmap(pm: QPixmap, radius: int = 8) -> QPixmap:
    """Return ``pm`` with anti-aliased rounded corners."""
    if pm.isNull():
        return pm
    out = QPixmap(pm.size())
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, pm.width(), pm.height()), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pm)
    p.end()
    return out


def _parse_channels(text: str) -> List[int]:
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or [0, 1, 2]


class _CropThumb(QLabel):
    """Clickable crop thumbnail."""

    clicked = Signal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self._index = index
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: transparent;")
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class MeasurePreviewPanel(QWidget):
    """Interactive crop preview for the Measure app."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None
        self._data_path: Optional[str] = None
        self._crops: List[Dict[str, Any]] = []
        self._selected: set = set()
        self._propagate_cb = None
        self._thumb_px = 132
        self._build_ui()
        self.setAcceptDrops(True)

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # File picker row
        pick_row = QHBoxLayout()
        self._path_label = QLabel(
            "No array loaded — drop a merged .npy here, or choose one")
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        pick_btn = QPushButton("Choose merged array…")
        pick_btn.clicked.connect(self._pick_file)
        pick_row.addWidget(self._path_label, 1)
        pick_row.addWidget(pick_btn)
        root.addLayout(pick_row)

        # Settings row 1 — object + mask dim + channels
        r1 = QHBoxLayout()
        self._object_box = QComboBox()
        self._object_box.addItems(list(_MASK_DIMS.keys()))
        self._object_box.setToolTip("(str) Object class to crop by.")
        self._object_box.currentTextChanged.connect(self._on_object_changed)
        self._mask_dim = QSpinBox(); self._mask_dim.setRange(0, 32)
        self._mask_dim.setValue(_MASK_DIMS["cell"])
        self._mask_dim.setToolTip("(int) Mask slice index in the merged array.")
        self._channels = QLineEdit("0,1,2")
        self._channels.setToolTip("(list) Image channel indices to assemble (RGB order).")
        r1.addWidget(QLabel("Object")); r1.addWidget(self._object_box)
        r1.addWidget(QLabel("Mask dim")); r1.addWidget(self._mask_dim)
        r1.addWidget(QLabel("Channels")); r1.addWidget(self._channels, 1)
        root.addLayout(r1)

        # Settings row 2 — size, area, buffer, normalise, max
        r2 = QHBoxLayout()
        self._crop_size = QSpinBox(); self._crop_size.setRange(16, 1024)
        self._crop_size.setValue(128)
        self._crop_size.setToolTip("(int, px) Output crop size (png_size).")
        self._min_area = QSpinBox(); self._min_area.setRange(0, 10_000_000)
        self._min_area.setToolTip("(int) Skip objects smaller than this area.")
        self._buffer = QSpinBox(); self._buffer.setRange(0, 200)
        self._buffer.setValue(10)
        self._buffer.setToolTip("(int, px) Padding around each object's bounding box.")
        self._normalise = QCheckBox("Normalise"); self._normalise.setChecked(True)
        self._normalise.setToolTip("(bool) Percentile-normalise each crop.")
        self._max_crops = QSpinBox(); self._max_crops.setRange(1, 500)
        self._max_crops.setValue(60)
        self._max_crops.setToolTip("(int) Max crops shown in the preview.")
        r2.addWidget(QLabel("Crop size")); r2.addWidget(self._crop_size)
        r2.addWidget(QLabel("Min area")); r2.addWidget(self._min_area)
        r2.addWidget(QLabel("Buffer")); r2.addWidget(self._buffer)
        r2.addWidget(self._normalise)
        r2.addWidget(QLabel("Max")); r2.addWidget(self._max_crops)
        root.addLayout(r2)

        # Re-crop live on setting changes.
        for w in (self._mask_dim, self._crop_size, self._min_area, self._buffer,
                  self._max_crops):
            w.valueChanged.connect(lambda *_: self.refresh())
        self._channels.editingFinished.connect(self.refresh)
        self._normalise.toggled.connect(lambda *_: self.refresh())

        # Action row
        act = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh crops")
        self._refresh_btn.clicked.connect(self.refresh)
        self._propagate_btn = QPushButton("Propagate settings")
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, the crop settings here are copied into the main Measure "
            "settings so the run uses them.")
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        self._status = QLabel("")
        act.addWidget(self._refresh_btn)
        act.addWidget(self._propagate_btn)
        act.addWidget(self._status, 1)
        root.addLayout(act)

        # Crop grid on a dark-gray canvas.
        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setFrameShape(QScrollArea.NoFrame)
        try:
            from ..theme import PALETTE
            bg = PALETTE["surface_alt"]
        except Exception:
            bg = "#161719"
        self._grid_scroll.viewport().setStyleSheet(f"background: {bg};")
        self._grid_holder = QWidget()
        self._grid_holder.setObjectName("MeasureGrid")
        self._grid_holder.setStyleSheet(
            f"QWidget#MeasureGrid {{ background: {bg}; }}")
        self._grid = QGridLayout(self._grid_holder)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid_scroll.setWidget(self._grid_holder)
        root.addWidget(self._grid_scroll, 1)

    # -- drag & drop -------------------------------------------------------

    def _dropped_path(self, event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _SUPPORTED:
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):   # noqa: N802
        if self._dropped_path(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):    # noqa: N802
        if self._dropped_path(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):        # noqa: N802
        p = self._dropped_path(event)
        if p:
            event.acceptProposedAction()
            self.load_array(p)
        else:
            event.ignore()

    # -- loading + settings -------------------------------------------------

    def _on_object_changed(self, name: str):
        self._mask_dim.setValue(_MASK_DIMS.get(name, self._mask_dim.value()))

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a merged .npy array", "", "NumPy arrays (*.npy)")
        if path:
            self.load_array(path)

    def load_array(self, path: str) -> bool:
        try:
            data = np.load(path)
        except Exception as e:
            self._status.setText(f"Failed to load: {e}")
            return False
        if data.ndim != 3:
            self._status.setText(
                f"Expected a merged (H,W,C) array; got shape {data.shape}")
            return False
        self._data = data
        self._data_path = path
        self._path_label.setText(f"{os.path.basename(path)}  ·  shape {data.shape}")
        # Clamp mask dim into range.
        if self._mask_dim.value() >= data.shape[2]:
            self._mask_dim.setValue(max(0, data.shape[2] - 1))
        self.refresh()
        return True

    def settings_for_propagation(self) -> dict:
        """Map the crop settings to main-panel Measure setting keys."""
        chans = _parse_channels(self._channels.text())
        sz = int(self._crop_size.value())
        obj = self._object_box.currentText()
        return {
            "png_dims": chans,
            "png_size": [sz, sz],
            "normalize": bool(self._normalise.isChecked()),
            "use_bounding_box": bool(self._buffer.value() > 0),
            "crop_mode": [obj],
        }

    def set_propagate_callback(self, cb) -> None:
        self._propagate_cb = cb

    def propagate_settings(self) -> None:
        if self._propagate_cb is not None:
            try:
                self._propagate_cb(self.settings_for_propagation())
            except Exception:
                LOG.debug("propagate failed", exc_info=True)

    def _on_propagate_toggled(self, on: bool):
        if on:
            self.propagate_settings()

    # -- cropping + rendering ----------------------------------------------

    def refresh(self):
        if self._data is None:
            return
        from spacr.measure import crop_objects_from_array
        chans = _parse_channels(self._channels.text())
        mask_dim = int(self._mask_dim.value())
        if mask_dim >= self._data.shape[2]:
            self._status.setText("Mask dim out of range for this array.")
            return
        try:
            self._crops = crop_objects_from_array(
                self._data, mask_dim=mask_dim, channels=chans,
                min_area=int(self._min_area.value()),
                mask_background=True,
                normalize=self._normalise.isChecked(),
                buffer=int(self._buffer.value()),
                limit=int(self._max_crops.value()))
        except Exception as e:
            self._status.setText(f"Crop failed: {e}")
            return
        self._selected.clear()
        self._render_grid()
        self._status.setText(
            f"{len(self._crops)} object(s) — {self._object_box.currentText()}")
        if self._propagate_btn.isChecked():
            self.propagate_settings()

    def _render_grid(self):
        # Clear the grid.
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        cols = max(1, self._grid_scroll.viewport().width() // (self._thumb_px + 12)) or 6
        for i, entry in enumerate(self._crops):
            pm = self._crop_pixmap(entry["crop"])
            thumb = _CropThumb(i)
            thumb.setPixmap(pm)
            thumb.setToolTip(f"label {entry['label']} · {entry['area']} px²")
            thumb.clicked.connect(self._on_thumb_clicked)
            self._grid.addWidget(thumb, i // cols, i % cols)

    def _crop_pixmap(self, crop: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(crop.astype(np.uint8))
        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(img.copy()).scaled(
            self._thumb_px, self._thumb_px, Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        return _rounded_pixmap(pm, radius=8)

    def _on_thumb_clicked(self, index: int):
        if index in self._selected:
            self._selected.discard(index)
        else:
            self._selected.add(index)
        if 0 <= index < len(self._crops):
            e = self._crops[index]
            sel = f" · {len(self._selected)} selected" if self._selected else ""
            self._status.setText(
                f"label {e['label']} · {e['area']} px²{sel}")

    # snapshot for tests
    def current_params(self) -> dict:
        d = self.settings_for_propagation()
        d["n_crops"] = len(self._crops)
        d["selected"] = sorted(self._selected)
        return d
