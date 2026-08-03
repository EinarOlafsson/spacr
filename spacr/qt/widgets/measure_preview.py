"""Interactive Measure crop preview with a pipeline-compatible settings panel.

The compact card is intentionally image-first.  Its ``Crop settings…`` dialog
contains the Measure settings that can be evaluated on one merged array:
general mask/channel controls, object-crop output controls, measurement
filters, and preview-only display controls.  With propagation enabled, every
pipeline setting is copied to the main Measure form as it changes.

Cell crops are grouped by three independent companion-object dimensions:
nucleated/unnucleated, infected/uninfected, and with/without organelles.  This
keeps all cells visible while making it explicit which categories would be
retained by the current filter settings.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .preview_controls import (
    FlatButton, FlatComboBox, populate_channel_combo, populate_fov_combo,
    selected_channel, sibling_sources,
)
from .toggle import Toggle

LOG = logging.getLogger("spacr.qt.measure_preview")

_MASK_DIMS = {"cell": 4, "nucleus": 5, "pathogen": 6, "organelle": 7}
_OBJECTS = ("cell", "nucleus", "pathogen", "cytoplasm", "organelle")
_SUPPORTED = (".npy",)


def _rounded_pixmap(pm: QPixmap, radius: int = 8) -> QPixmap:
    if pm.isNull():
        return pm
    out = QPixmap(pm.size())
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, pm.width(), pm.height()), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pm)
    painter.end()
    return out


def _parse_channels(text: str) -> List[int]:
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def _optional_spin_value(widget: QSpinBox) -> Optional[int]:
    value = int(widget.value())
    return None if value < 0 else value


class _CropThumb(QLabel):
    clicked = Signal(int)

    def __init__(self, index: int, *, included: bool = True, parent=None):
        super().__init__(parent)
        self._index = index
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        try:
            from ..theme import active_palette
            palette = active_palette()
            border = palette["accent"] if included else palette["fg_dim"]
            background = palette["surface_hi"]
        except Exception:
            border, background = ("#4A9EFF", "#24262a")
        self.setStyleSheet(
            "QLabel {"
            f"background: {background}; border: 2px solid {border};"
            "border-radius: 9px; padding: 2px;"
            "}"
        )

    def mousePressEvent(self, event):
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class MeasurePreviewPanel(QWidget):
    """Preview Measure crops and propagate a faithful run configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None
        self._data_path: Optional[str] = None
        self._crops: List[Dict[str, Any]] = []
        self._selected: set[int] = set()
        self._propagate_cb = None
        self._thumb_px = 132
        self._crop_settings_dialog: Optional[CropSettingsDialog] = None
        # Guards the FOV dropdown against re-entering itself while the array
        # it just asked for is being installed.
        self._loading_fov = False
        self._build_controls()
        self._build_ui()
        self._connect_controls()
        self.setAcceptDrops(True)

    # -- construction --------------------------------------------------

    @staticmethod
    def _spin(
        lo: int,
        hi: int,
        value: int,
        *,
        special: str = "",
        parent=None,
    ) -> QSpinBox:
        widget = QSpinBox(parent)
        widget.setRange(lo, hi)
        widget.setValue(value)
        if special:
            widget.setSpecialValueText(special)
        return widget

    def _build_controls(self) -> None:
        # General
        self._experiment = QLineEdit("exp", self)
        self._measurement_channels = QLineEdit("0,1,2,3", self)
        self._object_box = QComboBox(self)
        self._object_box.addItems(_OBJECTS)
        self._mask_dims = {
            name: self._spin(
                -1, 64, value if name != "organelle" else -1,
                special="Not present", parent=self,
            )
            for name, value in _MASK_DIMS.items()
        }
        self._cytoplasm = Toggle(parent=self)
        self._plot = Toggle(parent=self)
        self._test_mode = Toggle(parent=self)
        self._timelapse = Toggle(parent=self)

        # Object crops
        self._save_png = Toggle(parent=self)
        self._save_png.setChecked(True)
        self._save_arrays = Toggle(parent=self)
        self._crop_mode_checks = {
            name: Toggle(name.capitalize(), self) for name in _OBJECTS
        }
        self._crop_mode_checks["cell"].setChecked(True)
        self._crop_width = self._spin(16, 2048, 224, parent=self)
        self._crop_height = self._spin(16, 2048, 224, parent=self)
        self._lock_aspect = Toggle(parent=self)
        self._lock_aspect.setChecked(True)
        self._png_dims = QLineEdit("0,1,2", self)
        self._use_bbox = Toggle(parent=self)
        self._buffer = self._spin(0, 200, 10, parent=self)
        self._normalise = Toggle(parent=self)
        self._normalise.setChecked(True)
        self._lo_pct = QDoubleSpinBox(self)
        self._lo_pct.setRange(0.0, 50.0)
        self._lo_pct.setValue(1.0)
        self._lo_pct.setSuffix(" %")
        self._hi_pct = QDoubleSpinBox(self)
        self._hi_pct.setRange(50.0, 100.0)
        self._hi_pct.setValue(99.0)
        self._hi_pct.setSuffix(" %")
        self._normalize_by = QComboBox(self)
        self._normalize_by.addItems(("png", "fov"))
        self._dilate = Toggle(parent=self)
        self._dilate_ratio = QDoubleSpinBox(self)
        self._dilate_ratio.setRange(0.0, 10.0)
        self._dilate_ratio.setSingleStep(0.05)
        self._dilate_ratio.setValue(0.2)

        # Filtering
        self._min_sizes = {
            name: self._spin(0, 10_000_000, 0, parent=self)
            for name in _OBJECTS
        }
        self._uninfected = Toggle(parent=self)
        self._uninfected.setChecked(True)
        self._merge_edge_pathogen_cells = Toggle(parent=self)
        self._merge_edge_pathogen_cells.setChecked(True)

        # Preview-only controls
        self._max_area = self._spin(0, 100_000_000, 0, parent=self)
        self._max_crops = self._spin(1, 1000, 60, parent=self)
        self._group_cells = Toggle(parent=self)
        self._group_cells.setChecked(True)
        self._propagate_btn = Toggle("Propagate settings", self)
        self._propagate_btn.setToolTip(
            "Copy changes from this dialog into the main Measure settings."
        )

        # Compatibility names used by integrations and older tests.
        self._mask_dim = self._mask_dims["cell"]
        self._crop_size = self._crop_width
        self._min_area = self._min_sizes["cell"]
        self._channels = self._png_dims

        tooltips = {
            self._measurement_channels:
                "Image channels measured from the merged array.",
            self._png_dims:
                "Image channel indices written to crop R, G and B planes.",
            self._use_bbox:
                "Keep the padded rectangular bounding box instead of masking "
                "pixels outside the object.",
            self._normalise:
                "Write False when off or [lower, upper] percentiles when on.",
            self._uninfected:
                "Keep uninfected cells. Off marks them as excluded.",
            self._group_cells:
                "Group cells by nucleus, pathogen and organelle presence.",
        }
        for widget, text in tooltips.items():
            widget.setToolTip(text)
        for widget in self._managed_widgets():
            widget.hide()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # FOV and channel dropdowns sit immediately LEFT of the Choose
        # control; all three wear the flat "Live toggle" look.
        pick_row = QHBoxLayout()
        self._pick_row = pick_row
        self._path_label = QLabel(
            "No array loaded — drop a merged .npy here, or choose one")
        self._path_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._fov_box = FlatComboBox(
            self,
            tooltip=("Field of view. Lists every merged .npy sitting beside "
                     "the loaded one; picking one loads it."))
        self._fov_box.currentIndexChanged.connect(self._on_fov_changed)
        self._channel_box = FlatComboBox(
            self,
            tooltip=("Displayed channel. 'All channels' renders the crops "
                     "from the PNG channels in Crop settings; picking one "
                     "shows that channel alone."))
        self._channel_box.currentIndexChanged.connect(
            self._on_display_channel_changed)
        populate_channel_combo(self._channel_box, 0)
        self._pick_btn = FlatButton("Choose merged array…", self)
        self._pick_btn.clicked.connect(self._pick_file)
        pick_row.addWidget(self._path_label, 1)
        pick_row.addWidget(self._fov_box)
        pick_row.addWidget(self._channel_box)
        pick_row.addWidget(self._pick_btn)
        root.addLayout(pick_row)

        actions = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh crops")
        self._refresh_btn.clicked.connect(self.refresh)
        self._settings_btn = QPushButton("Crop settings…")
        self._settings_btn.clicked.connect(self.open_crop_settings)
        self._status = QLabel("")
        actions.addWidget(self._refresh_btn)
        actions.addWidget(self._settings_btn)
        actions.addWidget(self._status, 1)
        root.addLayout(actions)

        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setFrameShape(QScrollArea.NoFrame)
        try:
            from ..theme import active_palette
            background = active_palette()["surface_alt"]
        except Exception:
            background = "#161719"
        self._grid_scroll.viewport().setStyleSheet(
            f"background: {background};")
        self._grid_holder = QWidget()
        self._grid_holder.setObjectName("MeasureGrid")
        self._grid_holder.setStyleSheet(
            f"QWidget#MeasureGrid {{ background: {background}; }}")
        self._grid = QGridLayout(self._grid_holder)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid_scroll.setWidget(self._grid_holder)
        root.addWidget(self._grid_scroll, 1)

    def _managed_widgets(self) -> List[QWidget]:
        widgets: List[QWidget] = [
            self._experiment, self._measurement_channels, self._object_box,
            *self._mask_dims.values(), self._cytoplasm, self._plot,
            self._test_mode, self._timelapse, self._save_png,
            self._save_arrays, *self._crop_mode_checks.values(),
            self._crop_width, self._crop_height, self._lock_aspect,
            self._png_dims, self._use_bbox, self._buffer, self._normalise,
            self._lo_pct, self._hi_pct, self._normalize_by, self._dilate,
            self._dilate_ratio, *self._min_sizes.values(), self._uninfected,
            self._merge_edge_pathogen_cells, self._max_area,
            self._max_crops, self._group_cells, self._propagate_btn,
        ]
        return widgets

    def _connect_controls(self) -> None:
        self._object_box.currentTextChanged.connect(self._on_object_changed)
        self._crop_width.valueChanged.connect(self._sync_crop_height)
        self._normalise.toggled.connect(self._refresh_control_gates)
        self._dilate.toggled.connect(self._refresh_control_gates)
        self._use_bbox.toggled.connect(self._refresh_control_gates)

        refresh_widgets = [
            self._object_box, *self._mask_dims.values(), self._png_dims,
            self._use_bbox, self._buffer, self._normalise, self._lo_pct,
            self._hi_pct, *self._min_sizes.values(), self._uninfected,
            self._max_area, self._max_crops, self._group_cells,
        ]
        for widget in refresh_widgets:
            for signal_name in (
                "valueChanged", "currentTextChanged", "editingFinished",
                "toggled",
            ):
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    try:
                        signal.connect(self._on_setting_changed)
                        break
                    except (TypeError, RuntimeError):
                        pass

        for widget in self._managed_widgets():
            if widget in refresh_widgets or widget is self._propagate_btn:
                continue
            for signal_name in (
                "valueChanged", "currentTextChanged", "editingFinished",
                "toggled", "textChanged",
            ):
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    try:
                        signal.connect(self._maybe_propagate)
                        break
                    except (TypeError, RuntimeError):
                        pass
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        self._refresh_control_gates()

    # -- settings dialog ------------------------------------------------

    def open_crop_settings(self) -> None:
        dialog = self._crop_settings_dialog
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = CropSettingsDialog(self)
        self._crop_settings_dialog = dialog
        dialog.finished.connect(self._clear_crop_settings_dialog)
        dialog.show()

    def _clear_crop_settings_dialog(self, *_args) -> None:
        self._crop_settings_dialog = None

    def _refresh_control_gates(self, *_args) -> None:
        self._lo_pct.setEnabled(self._normalise.isChecked())
        self._hi_pct.setEnabled(self._normalise.isChecked())
        self._normalize_by.setEnabled(self._normalise.isChecked())
        self._dilate_ratio.setEnabled(self._dilate.isChecked())
        self._buffer.setEnabled(self._use_bbox.isChecked())

    def _sync_crop_height(self, value: int) -> None:
        if self._lock_aspect.isChecked() and self._crop_height.value() != value:
            self._crop_height.setValue(value)

    def _on_object_changed(self, name: str) -> None:
        # A previewed object should also be one of the requested crop outputs.
        check = self._crop_mode_checks.get(name)
        if check is not None:
            check.setChecked(True)
        self._maybe_propagate()

    def _on_setting_changed(self, *_args) -> None:
        if self._data is not None:
            self.refresh()
        self._maybe_propagate()

    def _maybe_propagate(self, *_args) -> None:
        if self._propagate_btn.isChecked():
            self.propagate_settings()

    def _on_propagate_toggled(self, on: bool) -> None:
        if on:
            self.propagate_settings()

    # -- drag/drop + loading -------------------------------------------

    def _dropped_path(self, event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if (
                url.isLocalFile()
                and Path(url.toLocalFile()).suffix.lower() in _SUPPORTED
            ):
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):  # noqa: N802
        event.acceptProposedAction() if self._dropped_path(event) else event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        event.acceptProposedAction() if self._dropped_path(event) else event.ignore()

    def dropEvent(self, event):  # noqa: N802
        path = self._dropped_path(event)
        if path:
            event.acceptProposedAction()
            self.load_array(path)
        else:
            event.ignore()

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a merged .npy array", "", "NumPy arrays (*.npy)")
        if path:
            self.load_array(path)

    def load_array(self, path: str) -> bool:
        try:
            data = np.load(path)
        except Exception as exc:
            self._status.setText(f"Failed to load: {exc}")
            return False
        if data.ndim != 3:
            self._status.setText(
                f"Expected a merged (H,W,C) array; got shape {data.shape}")
            return False
        self._data = data
        self._data_path = path
        self._path_label.setText(
            f"{os.path.basename(path)}  ·  shape {data.shape}")
        for widget in self._mask_dims.values():
            if widget.value() >= data.shape[2]:
                widget.setValue(-1)
        self._refresh_source_selectors()
        self.refresh()
        return True

    # -- FOV / channel selectors ---------------------------------------

    def _refresh_source_selectors(self) -> None:
        """Re-fill the FOV and channel dropdowns for the loaded array."""
        populate_fov_combo(
            self._fov_box,
            sibling_sources(self._data_path, _SUPPORTED),
            current=self._data_path)
        channels = int(self._data.shape[2]) if self._data is not None else 0
        populate_channel_combo(self._channel_box, channels)

    def _on_fov_changed(self, *_args) -> None:
        """Load the field of view the user picked from the dropdown."""
        if self._loading_fov:
            return
        path = self._fov_box.currentData()
        if not path or path == self._data_path:
            return
        self._loading_fov = True
        try:
            self.load_array(path)
        finally:
            self._loading_fov = False

    def display_channel(self) -> Optional[int]:
        """Channel index the crops are rendered from, or ``None`` for all."""
        return selected_channel(self._channel_box)

    def _on_display_channel_changed(self, *_args) -> None:
        """Re-render the crop grid from the newly selected channel."""
        self.refresh()

    # -- propagation ---------------------------------------------------

    def _selected_crop_modes(self) -> List[str]:
        selected = [
            name for name, widget in self._crop_mode_checks.items()
            if widget.isChecked()
        ]
        return selected or [self._object_box.currentText()]

    def settings_for_propagation(self) -> dict:
        normalize: Any = False
        if self._normalise.isChecked():
            normalize = [float(self._lo_pct.value()), float(self._hi_pct.value())]
        return {
            "experiment": self._experiment.text().strip() or "exp",
            "channels": _parse_channels(self._measurement_channels.text()),
            "cell_mask_dim": _optional_spin_value(self._mask_dims["cell"]),
            "nucleus_mask_dim": _optional_spin_value(self._mask_dims["nucleus"]),
            "pathogen_mask_dim": _optional_spin_value(self._mask_dims["pathogen"]),
            "organelle_mask_dim": _optional_spin_value(
                self._mask_dims["organelle"]),
            "cytoplasm": self._cytoplasm.isChecked(),
            "plot": self._plot.isChecked(),
            "test_mode": self._test_mode.isChecked(),
            "timelapse": self._timelapse.isChecked(),
            "save_png": self._save_png.isChecked(),
            "save_arrays": self._save_arrays.isChecked(),
            "crop_mode": self._selected_crop_modes(),
            "png_size": [
                int(self._crop_width.value()), int(self._crop_height.value())
            ],
            "png_dims": _parse_channels(self._png_dims.text()),
            "use_bounding_box": self._use_bbox.isChecked(),
            "normalize": normalize,
            "normalize_by": self._normalize_by.currentText(),
            "dialate_pngs": self._dilate.isChecked(),
            "dialate_png_ratios": [float(self._dilate_ratio.value())],
            "cell_min_size": int(self._min_sizes["cell"].value()),
            "nucleus_min_size": int(self._min_sizes["nucleus"].value()),
            "pathogen_min_size": int(self._min_sizes["pathogen"].value()),
            "organelle_min_size": int(self._min_sizes["organelle"].value()),
            "cytoplasm_min_size": int(self._min_sizes["cytoplasm"].value()),
            "uninfected": self._uninfected.isChecked(),
            "merge_edge_pathogen_cells":
                self._merge_edge_pathogen_cells.isChecked(),
        }

    def set_propagate_callback(self, callback) -> None:
        self._propagate_cb = callback

    def propagate_settings(self) -> None:
        if self._propagate_cb is None:
            return
        try:
            self._propagate_cb(self.settings_for_propagation())
        except Exception:
            LOG.debug("crop-preview propagation failed", exc_info=True)

    # -- crop/category computation ------------------------------------

    def _current_mask_dim(self) -> Optional[int]:
        name = self._object_box.currentText()
        if name == "cytoplasm":
            # Cytoplasm is generated during measurement and has no stable
            # input slice. Use cells for the preview footprint.
            name = "cell"
        return _optional_spin_value(self._mask_dims[name])

    def _presence(
        self,
        object_name: str,
        cell_region: np.ndarray,
    ) -> Optional[bool]:
        if self._data is None:
            return None
        dim = _optional_spin_value(self._mask_dims[object_name])
        if dim is None or dim >= self._data.shape[2]:
            return None
        mask = self._data[..., dim].astype(np.int64, copy=False)
        labels = np.unique(mask[cell_region])
        labels = labels[labels > 0]
        minimum = int(self._min_sizes[object_name].value())
        if minimum <= 0:
            return bool(labels.size)
        for label in labels:
            if int(np.count_nonzero(mask == label)) >= minimum:
                return True
        return False

    @staticmethod
    def _phenotype_text(name: str, value: Optional[bool]) -> str:
        if value is None:
            return f"{name} n/a"
        if name == "Nucleus":
            return "Nucleated" if value else "Unnucleated"
        if name == "Pathogen":
            return "Infected" if value else "Uninfected"
        return "Organelle+" if value else "Organelle−"

    def _annotate_cell_categories(self) -> None:
        if self._data is None or self._object_box.currentText() != "cell":
            for entry in self._crops:
                entry["category"] = self._object_box.currentText().capitalize()
                entry["included"] = True
            return
        cell_dim = _optional_spin_value(self._mask_dims["cell"])
        if cell_dim is None or cell_dim >= self._data.shape[2]:
            return
        cell_mask = self._data[..., cell_dim].astype(np.int64, copy=False)
        for entry in self._crops:
            region = cell_mask == int(entry["label"])
            nucleus = self._presence("nucleus", region)
            pathogen = self._presence("pathogen", region)
            organelle = self._presence("organelle", region)
            parts = (
                self._phenotype_text("Nucleus", nucleus),
                self._phenotype_text("Pathogen", pathogen),
                self._phenotype_text("Organelle", organelle),
            )
            entry["phenotype"] = {
                "nucleus": nucleus,
                "pathogen": pathogen,
                "organelle": organelle,
            }
            entry["category"] = " · ".join(parts)
            # The pipeline requires a nucleus when all companion masks exist,
            # and additionally requires a pathogen when uninfected is off.
            included = nucleus is not False
            if not self._uninfected.isChecked():
                included = included and pathogen is True
            entry["included"] = bool(included)

    def refresh(self) -> None:
        if self._data is None:
            return
        from spacr.measure import crop_objects_from_array

        channels = _parse_channels(self._png_dims.text())
        channels = [c for c in channels if 0 <= c < self._data.shape[2]]
        # The channel dropdown is a *view* control: it does not change the
        # png_dims that a real run would write, only what this grid shows.
        one = self.display_channel()
        if one is not None and 0 <= one < self._data.shape[2]:
            channels = [one, one, one]
        if not channels:
            self._status.setText("PNG channels do not exist in this array.")
            return
        mask_dim = self._current_mask_dim()
        if mask_dim is None or mask_dim >= self._data.shape[2]:
            self._status.setText(
                f"No {self._object_box.currentText()} mask slice is configured.")
            self._crops = []
            self._render_grid()
            return

        minimum = int(self._min_sizes[self._object_box.currentText()].value())
        try:
            self._crops = crop_objects_from_array(
                self._data,
                mask_dim=mask_dim,
                channels=channels,
                min_area=minimum,
                max_area=int(self._max_area.value()),
                mask_background=not self._use_bbox.isChecked(),
                normalize=self._normalise.isChecked(),
                percentiles=(
                    float(self._lo_pct.value()), float(self._hi_pct.value())
                ),
                buffer=int(self._buffer.value()),
                limit=int(self._max_crops.value()),
            )
        except Exception as exc:
            self._status.setText(f"Crop failed: {exc}")
            return
        self._annotate_cell_categories()
        self._selected.clear()
        self._render_grid()
        groups = len({entry.get("category") for entry in self._crops})
        self._status.setText(
            f"{len(self._crops)} object(s) · {groups} categor"
            f"{'y' if groups == 1 else 'ies'}")
        self._maybe_propagate()

    # -- rendering -----------------------------------------------------

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _category_header(self, text: str, entries: List[tuple[int, dict]]) -> QLabel:
        included = sum(bool(entry.get("included", True)) for _, entry in entries)
        excluded = len(entries) - included
        suffix = f"  ·  {included} kept"
        if excluded:
            suffix += f"  ·  {excluded} excluded by filters"
        label = QLabel(text + suffix)
        label.setObjectName("CropCategoryHeader")
        label.setContentsMargins(8, 5, 8, 5)
        try:
            from ..theme import active_palette
            palette = active_palette()
            label.setStyleSheet(
                "QLabel#CropCategoryHeader {"
                f"background: {palette['surface_hi']};"
                f"color: {palette['fg']};"
                f"border-left: 3px solid {palette['accent']};"
                "font-weight: 600; border-radius: 4px;"
                "}"
            )
        except Exception:
            pass
        return label

    def _render_grid(self) -> None:
        self._clear_grid()
        if not self._crops:
            return
        columns = max(
            1, self._grid_scroll.viewport().width() // (self._thumb_px + 12)
        )
        grouped: Dict[str, List[tuple[int, dict]]] = defaultdict(list)
        if self._object_box.currentText() == "cell" and self._group_cells.isChecked():
            for index, entry in enumerate(self._crops):
                grouped[entry.get("category", "Unclassified")].append(
                    (index, entry))
        else:
            grouped[self._object_box.currentText().capitalize()] = list(
                enumerate(self._crops))

        row = 0
        for category in sorted(grouped):
            entries = grouped[category]
            self._grid.addWidget(
                self._category_header(category, entries),
                row, 0, 1, columns,
            )
            row += 1
            for offset, (index, entry) in enumerate(entries):
                thumb = _CropThumb(
                    index, included=bool(entry.get("included", True)))
                thumb.setPixmap(self._crop_pixmap(entry["crop"]))
                status = "kept" if entry.get("included", True) else "excluded"
                thumb.setToolTip(
                    f"label {entry['label']} · {entry['area']} px² · "
                    f"{entry.get('category', '')} · {status}")
                thumb.clicked.connect(self._on_thumb_clicked)
                self._grid.addWidget(
                    thumb, row + offset // columns, offset % columns)
            row += (len(entries) + columns - 1) // columns

    def _crop_pixmap(self, crop: np.ndarray) -> QPixmap:
        array = np.ascontiguousarray(crop.astype(np.uint8))
        height, width = array.shape[:2]
        image = QImage(
            array.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self._thumb_px,
            self._thumb_px,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return _rounded_pixmap(pixmap, radius=8)

    def _on_thumb_clicked(self, index: int) -> None:
        if index in self._selected:
            self._selected.discard(index)
        else:
            self._selected.add(index)
        if 0 <= index < len(self._crops):
            entry = self._crops[index]
            selected = (
                f" · {len(self._selected)} selected" if self._selected else "")
            self._status.setText(
                f"label {entry['label']} · {entry['area']} px² · "
                f"{entry.get('category', '')}{selected}")

    def current_params(self) -> dict:
        values = self.settings_for_propagation()
        values["n_crops"] = len(self._crops)
        values["selected"] = sorted(self._selected)
        values["categories"] = [
            entry.get("category") for entry in self._crops
        ]
        values["display_channel"] = self.display_channel()
        values["fov"] = self._fov_box.currentText()
        return values


class CropSettingsDialog(QDialog):
    """Tabbed live settings dialog for :class:`MeasurePreviewPanel`."""

    def __init__(self, panel: MeasurePreviewPanel):
        super().__init__(panel)
        self._panel = panel
        self.setWindowTitle("Crop preview settings")
        outer = QVBoxLayout(self)
        tabs = QTabWidget(self)
        outer.addWidget(tabs, 1)

        for widget in panel._managed_widgets():
            widget.show()

        general = QWidget()
        form = QFormLayout(general)
        form.addRow("Experiment", panel._experiment)
        form.addRow("Measured channels", panel._measurement_channels)
        form.addRow("Preview object", panel._object_box)
        form.addRow("Cell mask slice", panel._mask_dims["cell"])
        form.addRow("Nucleus mask slice", panel._mask_dims["nucleus"])
        form.addRow("Pathogen mask slice", panel._mask_dims["pathogen"])
        form.addRow("Organelle mask slice", panel._mask_dims["organelle"])
        form.addRow("Measure cytoplasm", panel._cytoplasm)
        form.addRow("Plot run diagnostics", panel._plot)
        form.addRow("Test mode", panel._test_mode)
        form.addRow("Timelapse", panel._timelapse)
        tabs.addTab(general, "General")

        crops = QWidget()
        crops_form = QFormLayout(crops)
        crops_form.addRow("Save PNG crops", panel._save_png)
        crops_form.addRow("Save raw arrays", panel._save_arrays)
        mode_group = QGroupBox("Crop modes")
        mode_layout = QVBoxLayout(mode_group)
        for widget in panel._crop_mode_checks.values():
            mode_layout.addWidget(widget)
        crops_form.addRow(mode_group)
        crops_form.addRow("Crop width", panel._crop_width)
        crops_form.addRow("Crop height", panel._crop_height)
        crops_form.addRow("Lock aspect ratio", panel._lock_aspect)
        crops_form.addRow("RGB channel order", panel._png_dims)
        crops_form.addRow("Use bounding box", panel._use_bbox)
        crops_form.addRow("Bounding-box padding", panel._buffer)
        crops_form.addRow("Normalise crops", panel._normalise)
        crops_form.addRow("Lower percentile", panel._lo_pct)
        crops_form.addRow("Upper percentile", panel._hi_pct)
        crops_form.addRow("Normalise by", panel._normalize_by)
        crops_form.addRow("Dilate crop masks", panel._dilate)
        crops_form.addRow("Dilation ratio", panel._dilate_ratio)
        tabs.addTab(crops, "Object crops")

        filters = QWidget()
        filter_form = QFormLayout(filters)
        filter_form.addRow("Keep uninfected cells", panel._uninfected)
        filter_form.addRow(
            "Merge edge-pathogen cells", panel._merge_edge_pathogen_cells)
        for name, widget in panel._min_sizes.items():
            filter_form.addRow(f"{name.capitalize()} minimum area", widget)
        tabs.addTab(filters, "Filter settings")

        preview = QWidget()
        preview_form = QFormLayout(preview)
        preview_form.addRow("Maximum preview area", panel._max_area)
        preview_form.addRow("Maximum preview crops", panel._max_crops)
        preview_form.addRow("Group cell phenotypes", panel._group_cells)
        tabs.addTab(preview, "Preview")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        run = QPushButton("Refresh crops")
        run.clicked.connect(panel.refresh)
        buttons.addButton(run, QDialogButtonBox.ActionRole)
        buttons.addButton(panel._propagate_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        outer.addWidget(buttons)
        panel._refresh_control_gates()
        from ..screens.settings_model import install_api_tooltips
        widget_keys = {
            panel._experiment: "experiment",
            panel._measurement_channels: "channels",
            panel._object_box: "crop_mode",
            panel._mask_dims["cell"]: "cell_mask_dim",
            panel._mask_dims["nucleus"]: "nucleus_mask_dim",
            panel._mask_dims["pathogen"]: "pathogen_mask_dim",
            panel._mask_dims["organelle"]: "organelle_mask_dim",
            panel._cytoplasm: "cytoplasm",
            panel._plot: "plot",
            panel._test_mode: "test_mode",
            panel._timelapse: "timelapse",
            panel._save_png: "save_png",
            panel._save_arrays: "save_arrays",
            panel._crop_width: "png_size",
            panel._crop_height: "png_size",
            panel._lock_aspect: "lock_aspect_ratio",
            panel._png_dims: "png_dims",
            panel._use_bbox: "use_bounding_box",
            panel._buffer: "bounding_box_padding",
            panel._normalise: "normalize",
            panel._lo_pct: "lower_percentile",
            panel._hi_pct: "upper_percentile",
            panel._normalize_by: "normalize_by",
            panel._dilate: "dialate_pngs",
            panel._dilate_ratio: "dialate_png_ratios",
            panel._uninfected: "uninfected",
            panel._merge_edge_pathogen_cells:
                "merge_edge_pathogen_cells",
            panel._max_area: "preview_max_area",
            panel._max_crops: "preview_max_crops",
            panel._group_cells: "preview_group_cells",
        }
        for name, widget in panel._crop_mode_checks.items():
            widget_keys[widget] = "crop_mode"
        for name, widget in panel._min_sizes.items():
            widget_keys[widget] = f"{name}_min_size"
        install_api_tooltips(self, "measure", widget_keys)
        self.resize(620, 720)

    def closeEvent(self, event):
        # Keep control values alive on the panel between dialog openings.
        for widget in self._panel._managed_widgets():
            widget.setParent(self._panel)
            widget.hide()
        super().closeEvent(event)
