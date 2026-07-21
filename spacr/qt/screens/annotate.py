"""
AnnotateScreen — Qt widget replacing the Tk AnnotateApp.

Displays a paginated grid of clickable image thumbnails backed by
`png_list` in a `measurements/measurements.db`. Left-click = value 1,
right-click = value 2, re-click the same value = clear. Annotations
are persisted through a background SaveWorker (see
`spacr.qt.annotate_engine.SaveWorker`).

Advanced features that are *not* yet ported (marked as TODOs in the UI):
UMAP window, Deep Spacr training launcher, measurement-threshold
filtering (the threshold filter can be entered in settings but only
plain per-page fetch is used at query time in this MVP).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..annotate_engine import (
    AnnotateSettings,
    SaveWorker,
    add_colored_border,
    class_counts,
    clear_column,
    count_rows,
    ensure_annotation_column,
    fetch_page,
    filter_channels_pil,
    find_last_annotated_offset,
    label_to_hex,
    normalize_pil,
)
from ..theme import PALETTE, SPACING
from ..widgets import Divider


BORDER_WIDTH = 5


# ---------------------------------------------------------------------------
# Click-aware thumbnail label
# ---------------------------------------------------------------------------

class _Thumbnail(QLabel):
    """QLabel that emits left/right-click signals with its slot index."""

    left_clicked = Signal(int)
    right_clicked = Signal(int)

    def __init__(self, slot: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.slot = slot
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setStyleSheet(
            f"background: {PALETTE['surface']}; border-radius: 4px;"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(self.slot)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(self.slot)
        else:
            super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

def _csv_to_list(text: str) -> Optional[List[str]]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts or None


def _list_to_csv(vals: Optional[List[str]]) -> str:
    return ", ".join(str(v) for v in vals) if vals else ""


class _SettingsDialog(QDialog):
    def __init__(self, settings: AnnotateSettings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Annotate — Settings")
        self.setMinimumWidth(480)
        self._settings = settings

        form = QFormLayout()

        self._src_edit = QLineEdit(settings.src)
        src_row = QHBoxLayout()
        src_row.setContentsMargins(0, 0, 0, 0)
        src_row.addWidget(self._src_edit, 1)
        src_btn = QPushButton("Browse…")
        src_btn.clicked.connect(self._pick_src)
        src_row.addWidget(src_btn)
        src_wrap = QWidget(); src_wrap.setLayout(src_row)
        form.addRow("Source folder", src_wrap)

        self._ann_col = QLineEdit(settings.annotation_column)
        form.addRow("Annotation column", self._ann_col)

        self._img_size = QSpinBox()
        self._img_size.setRange(48, 800)
        self._img_size.setValue(settings.image_size[0])
        form.addRow("Image size (px)", self._img_size)

        self._image_type = QLineEdit(settings.image_type or "")
        self._image_type.setPlaceholderText("e.g. cell (blank = all types)")
        form.addRow("Image type filter", self._image_type)

        self._channels = QLineEdit(_list_to_csv(settings.channels))
        self._channels.setPlaceholderText("r, g, b (blank = all)")
        form.addRow("Show channels", self._channels)

        self._norm_channels = QLineEdit(_list_to_csv(settings.normalize_channels))
        self._norm_channels.setPlaceholderText("r, g, b (blank = off)")
        form.addRow("Normalize channels", self._norm_channels)

        self._pct_lo = QDoubleSpinBox()
        self._pct_lo.setRange(0.0, 100.0)
        self._pct_lo.setValue(float(settings.percentiles[0]))
        self._pct_hi = QDoubleSpinBox()
        self._pct_hi.setRange(0.0, 100.0)
        self._pct_hi.setValue(float(settings.percentiles[1]))
        pct_row = QHBoxLayout(); pct_row.setContentsMargins(0, 0, 0, 0)
        pct_row.addWidget(self._pct_lo); pct_row.addWidget(QLabel("–"))
        pct_row.addWidget(self._pct_hi)
        pct_wrap = QWidget(); pct_wrap.setLayout(pct_row)
        form.addRow("Percentiles", pct_wrap)

        self._outline = QLineEdit(_list_to_csv(settings.outline))
        self._outline.setPlaceholderText("channels to outline, e.g. g")
        form.addRow("Outline channels", self._outline)

        self._out_factor = QDoubleSpinBox()
        self._out_factor.setRange(0.0, 100.0)
        self._out_factor.setValue(float(settings.outline_threshold_factor))
        form.addRow("Outline threshold factor", self._out_factor)

        self._out_sigma = QDoubleSpinBox()
        self._out_sigma.setRange(0.0, 100.0)
        self._out_sigma.setValue(float(settings.outline_sigma))
        form.addRow("Outline sigma", self._out_sigma)

        self._edge_thick = QDoubleSpinBox()
        self._edge_thick.setRange(0.0, 20.0)
        self._edge_thick.setDecimals(2)
        self._edge_thick.setValue(float(settings.edge_thickness))
        form.addRow("Edge thickness", self._edge_thick)

        self._edge_transp = QDoubleSpinBox()
        self._edge_transp.setRange(0.0, 100.0)
        self._edge_transp.setValue(float(settings.edge_transparency))
        form.addRow("Edge transparency", self._edge_transp)

        self._edge_image = QCheckBox("Show original image under outline")
        self._edge_image.setChecked(bool(settings.edge_image))
        form.addRow("", self._edge_image)

        self._obj_min = QSpinBox(); self._obj_min.setRange(0, 10_000_000)
        self._obj_min.setValue(int(settings.object_size[0]))
        self._obj_max = QSpinBox(); self._obj_max.setRange(0, 10_000_000)
        self._obj_max.setValue(int(settings.object_size[1]))
        obj_row = QHBoxLayout(); obj_row.setContentsMargins(0, 0, 0, 0)
        obj_row.addWidget(self._obj_min); obj_row.addWidget(QLabel("–"))
        obj_row.addWidget(self._obj_max)
        obj_wrap = QWidget(); obj_wrap.setLayout(obj_row)
        form.addRow("Object size (px area)", obj_wrap)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

    def _pick_src(self):
        d = QFileDialog.getExistingDirectory(self, "Pick experiment source",
                                              self._src_edit.text() or os.getcwd())
        if d:
            self._src_edit.setText(d)

    def collect(self) -> AnnotateSettings:
        s = self._settings
        s.src = self._src_edit.text().strip()
        s.db_path = os.path.join(s.src, "measurements", "measurements.db")
        s.annotation_column = self._ann_col.text().strip() or "annotate"
        size = int(self._img_size.value())
        s.image_size = (size, size)
        s.image_type = self._image_type.text().strip() or None
        s.channels = _csv_to_list(self._channels.text())
        s.normalize_channels = _csv_to_list(self._norm_channels.text())
        s.percentiles = (float(self._pct_lo.value()), float(self._pct_hi.value()))
        s.outline = _csv_to_list(self._outline.text())
        s.outline_threshold_factor = float(self._out_factor.value())
        s.outline_sigma = float(self._out_sigma.value())
        s.edge_thickness = float(self._edge_thick.value())
        s.edge_transparency = float(self._edge_transp.value())
        s.edge_image = bool(self._edge_image.isChecked())
        s.object_size = (int(self._obj_min.value()), int(self._obj_max.value()))
        return s


# ---------------------------------------------------------------------------
# AnnotateScreen
# ---------------------------------------------------------------------------

class AnnotateScreen(QWidget):
    """Main Qt widget for the annotate app."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._settings = AnnotateSettings()
        self._offset = 0
        self._total = 0
        self._page_paths: List[Tuple[str, Optional[int]]] = []
        self._pending_updates: Dict[str, Optional[int]] = {}
        self._worker: Optional[SaveWorker] = None
        self._thumbs: List[_Thumbnail] = []
        self._thumb_pixmaps: List[Optional[QPixmap]] = []
        self._raw_thumb_images: List[Optional[Image.Image]] = []

        self._build_ui()
        self._install_shortcuts()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status_label)
        self._status_timer.start()

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # Header
        header = QWidget()
        hbox = QVBoxLayout(header)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(2)
        title = QLabel("Annotate")
        title.setObjectName("DisplayHeading")
        hbox.addWidget(title)
        self._src_label = QLabel("No source selected — click Open source…")
        self._src_label.setObjectName("Muted")
        hbox.addWidget(self._src_label)
        outer.addWidget(header)
        outer.addWidget(Divider())

        # Toolbar
        toolbar = QWidget()
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        self._btn_open = QPushButton("Open source…")
        self._btn_open.setObjectName("PrimaryButton")
        self._btn_open.clicked.connect(self._on_pick_source)
        row.addWidget(self._btn_open)

        self._btn_settings = QPushButton("Settings…")
        self._btn_settings.clicked.connect(self._on_open_settings)
        row.addWidget(self._btn_settings)

        self._btn_prev = QPushButton("‹ Back")
        self._btn_prev.clicked.connect(self._on_prev)
        row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next ›")
        self._btn_next.clicked.connect(self._on_next)
        row.addWidget(self._btn_next)

        self._btn_skip = QPushButton("Skip to last annotated")
        self._btn_skip.clicked.connect(self._on_skip)
        row.addWidget(self._btn_skip)

        self._btn_count = QPushButton("Class counts")
        self._btn_count.clicked.connect(self._on_class_counts)
        row.addWidget(self._btn_count)

        self._btn_clear = QPushButton("Clear column…")
        self._btn_clear.setObjectName("DangerButton")
        self._btn_clear.clicked.connect(self._on_clear_column)
        row.addWidget(self._btn_clear)

        row.addStretch(1)
        self._page_label = QLabel("")
        self._page_label.setObjectName("Muted")
        row.addWidget(self._page_label)
        outer.addWidget(toolbar)

        # Grid inside a scroll area
        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setFrameShape(QScrollArea.NoFrame)
        self._grid_holder = QWidget()
        self._grid_layout = QGridLayout(self._grid_holder)
        self._grid_layout.setSpacing(SPACING["xs"])
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_scroll.setWidget(self._grid_holder)
        outer.addWidget(self._grid_scroll, 1)

        # Status bar area
        self._status_label = QLabel("Ready.")
        self._status_label.setObjectName("Muted")
        outer.addWidget(self._status_label)

        self._rebuild_grid()

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_next)

    # ------------------------------------------------------------------
    def _rebuild_grid(self):
        """Regenerate empty thumbnail widgets sized for current settings."""
        # Clear existing
        for w in self._thumbs:
            w.setParent(None)
            w.deleteLater()
        self._thumbs.clear()
        self._thumb_pixmaps = [None] * (self._settings.grid_rows *
                                         self._settings.grid_cols)
        self._raw_thumb_images = [None] * len(self._thumb_pixmaps)

        cols = self._settings.grid_cols
        rows = self._settings.grid_rows
        w, h = self._settings.image_size
        pad = BORDER_WIDTH * 2
        for i in range(rows * cols):
            thumb = _Thumbnail(i)
            thumb.setFixedSize(w + pad, h + pad)
            thumb.left_clicked.connect(self._on_thumb_left)
            thumb.right_clicked.connect(self._on_thumb_right)
            self._grid_layout.addWidget(thumb, i // cols, i % cols)
            self._thumbs.append(thumb)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_pick_source(self):
        d = QFileDialog.getExistingDirectory(self, "Pick experiment source",
                                              self._settings.src or os.getcwd())
        if not d:
            return
        self._open_source(d)

    def _open_source(self, src: str):
        db_path = os.path.join(src, "measurements", "measurements.db")
        if not os.path.isfile(db_path):
            answer = QMessageBox.question(
                self, "Database not found",
                f"No file at:\n{db_path}\n\nUse it anyway?",
            )
            if answer != QMessageBox.Yes:
                return
        # Tear down previous worker
        self._flush_pending()
        if self._worker:
            self._worker.stop(wait=True)
            self._worker = None
        self._settings.src = src
        self._settings.db_path = db_path
        ensure_annotation_column(db_path, self._settings.annotation_column)
        self._worker = SaveWorker(db_path, self._settings.annotation_column)
        self._worker.start()
        self._offset = 0
        self._src_label.setText(f"{src}  →  {db_path}")
        self._refresh_total()
        self._load_page()

    def _on_open_settings(self):
        dlg = _SettingsDialog(self._settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        old_src = self._settings.src
        old_col = self._settings.annotation_column
        self._settings = dlg.collect()
        self._rebuild_grid()
        # Restart worker if src/col changed
        if self._settings.src != old_src or self._settings.annotation_column != old_col:
            self._open_source(self._settings.src)
        else:
            self._refresh_total()
            self._load_page()

    def _on_next(self):
        self._flush_pending()
        page = self._settings.page_size
        if self._offset + page < max(self._total, 1):
            self._offset += page
            self._load_page()

    def _on_prev(self):
        self._flush_pending()
        page = self._settings.page_size
        self._offset = max(0, self._offset - page)
        self._load_page()

    def _on_skip(self):
        self._flush_pending()
        offset = find_last_annotated_offset(
            self._settings.db_path,
            self._settings.annotation_column,
            self._settings.page_size,
            self._settings.image_type,
        )
        if offset is None:
            self._status_label.setText("No annotated images found.")
            return
        self._offset = offset
        self._load_page()

    def _on_class_counts(self):
        rows = class_counts(self._settings.db_path, self._settings.annotation_column)
        if not rows:
            QMessageBox.information(self, "Class counts", "No annotated rows yet.")
            return
        lines = ["Class    Count    Color"]
        for cls, cnt in rows:
            lines.append(f"{cls:>5}  {cnt:>7}    {label_to_hex(cls) or ''}")
        QMessageBox.information(self, "Class counts", "\n".join(lines))

    def _on_clear_column(self):
        col = self._settings.annotation_column
        answer = QMessageBox.question(
            self, "Confirm clear",
            f'Clear ALL annotations in column "{col}"?\nThis cannot be undone.',
        )
        if answer != QMessageBox.Yes:
            return
        self._pending_updates.clear()
        clear_column(self._settings.db_path, col)
        self._refresh_total()
        self._load_page()

    def _on_thumb_left(self, slot: int):
        self._toggle_annotation(slot, 1)

    def _on_thumb_right(self, slot: int):
        self._toggle_annotation(slot, 2)

    # ------------------------------------------------------------------
    # Page loading + rendering
    # ------------------------------------------------------------------
    def _refresh_total(self):
        self._total = count_rows(self._settings.db_path, self._settings.image_type)

    def _load_page(self):
        page = self._settings.page_size
        self._page_paths = fetch_page(
            self._settings.db_path,
            self._settings.annotation_column,
            self._offset,
            page,
            self._settings.image_type,
        )
        # Clear all thumbs
        for i, thumb in enumerate(self._thumbs):
            thumb.setPixmap(QPixmap())
            self._thumb_pixmaps[i] = None
            self._raw_thumb_images[i] = None

        # Load images off-thread
        with ThreadPoolExecutor() as ex:
            loaded = list(ex.map(self._load_thumb_image, self._page_paths))
        for i, (img, annotation) in enumerate(loaded):
            if i >= len(self._thumbs):
                break
            self._raw_thumb_images[i] = img
            border = label_to_hex(annotation)
            display = add_colored_border(img, BORDER_WIDTH, border) if border \
                      else add_colored_border(img, BORDER_WIDTH, PALETTE["surface"])
            self._thumb_pixmaps[i] = self._image_to_pixmap(display)
            self._thumbs[i].setPixmap(self._thumb_pixmaps[i])
        self._page_label.setText(
            f"Page rows {self._offset}–{min(self._offset + page, self._total)} / {self._total}"
        )

    def _load_thumb_image(self, row: Tuple[str, Optional[int]]):
        path, annotation = row
        s = self._settings
        if not path or not os.path.isfile(path):
            blank = Image.new("RGB", s.image_size, color=(20, 20, 20))
            return blank, annotation
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", s.image_size, (30, 30, 30)), annotation
        img = normalize_pil(img, s.percentiles, s.normalize_channels)
        img = filter_channels_pil(img, s.channels)
        img = img.resize(s.image_size)
        return img, annotation

    def _image_to_pixmap(self, img: Image.Image) -> QPixmap:
        qimg = ImageQt(img.convert("RGB"))
        return QPixmap.fromImage(QImage(qimg))

    # ------------------------------------------------------------------
    def _toggle_annotation(self, slot: int, new_value: int):
        if slot >= len(self._page_paths):
            return
        path, current = self._page_paths[slot]
        # Cycle: same value again clears
        if slot in self._pending_updates:
            existing = self._pending_updates[path] if path in self._pending_updates else current
        else:
            existing = current
        if existing == new_value:
            resolved = None
        else:
            resolved = new_value
        self._pending_updates[path] = resolved
        self._page_paths[slot] = (path, resolved)

        # Update thumbnail border in place
        base = self._raw_thumb_images[slot]
        if base is None:
            return
        border = label_to_hex(resolved) or PALETTE["surface"]
        display = add_colored_border(base, BORDER_WIDTH, border)
        self._thumb_pixmaps[slot] = self._image_to_pixmap(display)
        self._thumbs[slot].setPixmap(self._thumb_pixmaps[slot])

    # ------------------------------------------------------------------
    def _flush_pending(self):
        if not self._pending_updates or self._worker is None:
            return
        self._worker.submit(self._pending_updates)
        self._pending_updates.clear()

    def _refresh_status_label(self):
        w = self._worker
        if w is None:
            self._status_label.setText("Ready.")
            return
        parts = []
        if self._pending_updates:
            parts.append(f"{len(self._pending_updates)} unsaved change(s)")
        if w.busy:
            parts.append("saving…")
        elif w.pending_batches > 0:
            parts.append(f"{w.pending_batches} batch queued")
        if w.last_save_ts is not None and not parts:
            parts.append("saved")
        self._status_label.setText(" · ".join(parts) if parts else "Ready.")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._flush_pending()
        if self._worker:
            self._worker.stop(wait=True)
        super().closeEvent(event)
