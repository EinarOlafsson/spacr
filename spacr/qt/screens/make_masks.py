"""
MakeMasksScreen — Qt widget replacing the Tk ModifyMaskApp.

Load a folder of images and their masks (in `<folder>/masks/`), draw
brush/erase strokes on the mask, run object-level operations (fill,
relabel, invert, remove small), and save the edited mask back to
`<folder>/masks/<name>.tif` as a labeled uint16 mask.

Advanced features not yet ported from Tk (marked TODO in the UI):
zoom rectangle, magic wand, dividing line, free-form polygon draw.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import mask_engine as engine
from ..theme import PALETTE, SPACING
from ..widgets import Card, Divider, Section


# ---------------------------------------------------------------------------
# Canvas — image + mask overlay with brush/erase mouse handling
# ---------------------------------------------------------------------------

MODE_NONE = "none"
MODE_BRUSH = "brush"
MODE_ERASE = "erase"
MODE_ERASE_OBJECT = "erase_object"


class _MaskCanvas(QLabel):
    """QLabel that displays the composited image+mask and captures
    mouse events for brush/erase strokes on the mask array."""

    stroke_finished = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None       # uint16 grayscale
        self.mask: Optional[np.ndarray] = None        # uint8 labels
        self.mode: str = MODE_NONE
        self.brush_radius: int = 10
        self.norm_lo: float = 1.0
        self.norm_hi: float = 99.9
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {PALETTE['bg']};")
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)
        self._last_pt: Optional[QPoint] = None

    # -- data ----------------------------------------------------------
    def set_image_and_mask(self, image: np.ndarray, mask: np.ndarray):
        self.image = image
        self.mask = mask
        self.refresh()

    def refresh(self):
        if self.image is None or self.mask is None:
            return
        img = engine.normalize_uint16(self.image, self.norm_lo, self.norm_hi)
        composed = engine.overlay_mask(img, self.mask, alpha=0.5)
        h, w = composed.shape[:2]
        qimg = QImage(composed.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        avail_w = max(200, self.width())
        avail_h = max(200, self.height())
        pixmap = pixmap.scaled(avail_w, avail_h,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
        self.setPixmap(pixmap)

    # -- mouse coord mapping -------------------------------------------
    def _canvas_to_image(self, x: int, y: int) -> Optional[tuple]:
        """Map widget-local (x,y) to (img_x, img_y). Returns None if
        the point is outside the drawn pixmap."""
        if self.mask is None or self.pixmap() is None:
            return None
        p = self.pixmap()
        pw, ph = p.width(), p.height()
        if pw == 0 or ph == 0:
            return None
        # Compute pixmap origin (centered inside widget)
        w, h = self.width(), self.height()
        ox = (w - pw) // 2
        oy = (h - ph) // 2
        cx, cy = x - ox, y - oy
        if not (0 <= cx < pw and 0 <= cy < ph):
            return None
        img_x = int(cx * self.mask.shape[1] / pw)
        img_y = int(cy * self.mask.shape[0] / ph)
        return img_x, img_y

    def _mask_radius_for_brush(self) -> int:
        if self.mask is None or self.pixmap() is None:
            return self.brush_radius
        p = self.pixmap()
        if p.width() == 0:
            return self.brush_radius
        return max(1, int(self.brush_radius * self.mask.shape[1] / p.width()))

    # -- events --------------------------------------------------------
    def mousePressEvent(self, event):
        if self.mode == MODE_NONE or self.mask is None:
            return super().mousePressEvent(event)
        pt = self._canvas_to_image(event.position().x(), event.position().y())
        if pt is None:
            return
        if self.mode == MODE_ERASE_OBJECT:
            self.mask = engine.erase_object_at(self.mask, *pt)
            self.refresh()
            self.stroke_finished.emit()
            return
        radius = self._mask_radius_for_brush()
        value = 255 if self.mode == MODE_BRUSH else 0
        engine.paint_disk(self.mask, pt[0], pt[1], radius, value)
        self._last_pt = QPoint(*pt)
        self.refresh()

    def mouseMoveEvent(self, event):
        if self.mask is None:
            return
        if self.mode in (MODE_BRUSH, MODE_ERASE) and event.buttons() & Qt.LeftButton:
            pt = self._canvas_to_image(event.position().x(), event.position().y())
            if pt is None:
                return
            radius = self._mask_radius_for_brush()
            value = 255 if self.mode == MODE_BRUSH else 0
            if self._last_pt is not None:
                engine.paint_line(self.mask,
                                    self._last_pt.x(), self._last_pt.y(),
                                    pt[0], pt[1], radius, value)
            else:
                engine.paint_disk(self.mask, pt[0], pt[1], radius, value)
            self._last_pt = QPoint(*pt)
            self.refresh()

    def mouseReleaseEvent(self, event):
        if self._last_pt is not None:
            self._last_pt = None
            self.stroke_finished.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


# ---------------------------------------------------------------------------
# MakeMasksScreen
# ---------------------------------------------------------------------------

class MakeMasksScreen(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._folder: str = ""
        self._image_files: List[str] = []
        self._current_index: int = 0
        self._build_ui()
        self._install_shortcuts()
        self._sync_button_states()

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # Header
        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        title = QLabel("Make Masks")
        title.setObjectName("DisplayHeading")
        header.addWidget(title)
        self._src_label = QLabel("No folder selected — click Open folder…")
        self._src_label.setObjectName("Muted")
        header.addWidget(self._src_label)
        header_wrap = QWidget(); header_wrap.setLayout(header)
        outer.addWidget(header_wrap)
        outer.addWidget(Divider())

        # Body splitter: canvas (left) + tools panel (right)
        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        self._canvas = _MaskCanvas()
        self._canvas.stroke_finished.connect(self._on_stroke_finished)
        body.addWidget(self._canvas)

        # Right: tools card
        tools = self._build_tools_panel()
        body.addWidget(tools)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 1)
        body.setSizes([900, 320])
        outer.addWidget(body, 1)

        # Bottom nav bar
        nav = QWidget()
        nav_row = QHBoxLayout(nav)
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(SPACING["sm"])
        self._btn_open = QPushButton("Open folder…")
        self._btn_open.setObjectName("PrimaryButton")
        self._btn_open.clicked.connect(self._on_pick_folder)
        nav_row.addWidget(self._btn_open)

        self._btn_prev = QPushButton("‹ Prev image")
        self._btn_prev.clicked.connect(self._on_prev)
        nav_row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next image ›")
        self._btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(self._btn_next)

        self._btn_save = QPushButton("Save mask")
        self._btn_save.setObjectName("PrimaryButton")
        self._btn_save.clicked.connect(self._on_save)
        nav_row.addWidget(self._btn_save)

        nav_row.addStretch(1)
        self._status_label = QLabel("Ready.")
        self._status_label.setObjectName("Muted")
        nav_row.addWidget(self._status_label)
        outer.addWidget(nav)

    def _build_tools_panel(self) -> QWidget:
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])

        # Mode buttons
        mode_card = Card(title="Tools")
        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        self._btn_brush = QPushButton("Brush")
        self._btn_brush.setCheckable(True)
        self._btn_brush.clicked.connect(lambda: self._set_mode(MODE_BRUSH))
        row.addWidget(self._btn_brush)
        self._btn_erase = QPushButton("Erase")
        self._btn_erase.setCheckable(True)
        self._btn_erase.clicked.connect(lambda: self._set_mode(MODE_ERASE))
        row.addWidget(self._btn_erase)
        self._btn_del_obj = QPushButton("Erase object")
        self._btn_del_obj.setCheckable(True)
        self._btn_del_obj.clicked.connect(lambda: self._set_mode(MODE_ERASE_OBJECT))
        row.addWidget(self._btn_del_obj)
        row_w = QWidget(); row_w.setLayout(row)
        mode_card.body_layout.addWidget(row_w)
        col.addWidget(mode_card)

        # Brush size slider
        brush_card = Card(title="Brush")
        brush_form = QFormLayout()
        self._brush_slider = QSlider(Qt.Horizontal)
        self._brush_slider.setRange(1, 100)
        self._brush_slider.setValue(10)
        self._brush_slider.valueChanged.connect(self._on_brush_size_changed)
        self._brush_size_label = QLabel("10 px")
        self._brush_size_label.setObjectName("Muted")
        brush_row = QHBoxLayout()
        brush_row.addWidget(self._brush_slider, 1)
        brush_row.addWidget(self._brush_size_label)
        brush_wrap = QWidget(); brush_wrap.setLayout(brush_row)
        brush_form.addRow("Radius", brush_wrap)
        brush_card.body_layout.addLayout(brush_form)
        col.addWidget(brush_card)

        # Normalize card
        norm_card = Card(title="Normalize")
        norm_form = QFormLayout()
        self._norm_lo = QDoubleSpinBox()
        self._norm_lo.setRange(0.0, 100.0); self._norm_lo.setValue(1.0)
        self._norm_lo.valueChanged.connect(self._on_normalize_changed)
        self._norm_hi = QDoubleSpinBox()
        self._norm_hi.setRange(0.0, 100.0); self._norm_hi.setValue(99.9)
        self._norm_hi.valueChanged.connect(self._on_normalize_changed)
        norm_form.addRow("Lower %", self._norm_lo)
        norm_form.addRow("Upper %", self._norm_hi)
        norm_card.body_layout.addLayout(norm_form)
        col.addWidget(norm_card)

        # Object ops card
        obj_card = Card(title="Object operations")
        ops_col = QVBoxLayout()
        ops_col.setSpacing(SPACING["xs"])
        for label, cb in (
            ("Fill holes", self._on_fill_holes),
            ("Relabel", self._on_relabel),
            ("Invert mask", self._on_invert),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(cb)
            ops_col.addWidget(btn)
        remove_row = QHBoxLayout()
        remove_row.setSpacing(SPACING["sm"])
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 1_000_000)
        self._min_area.setValue(100)
        remove_row.addWidget(QLabel("Min area:"))
        remove_row.addWidget(self._min_area, 1)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_small)
        remove_row.addWidget(remove_btn)
        remove_wrap = QWidget(); remove_wrap.setLayout(remove_row)
        ops_col.addWidget(remove_wrap)
        clear_btn = QPushButton("Clear mask")
        clear_btn.setObjectName("DangerButton")
        clear_btn.clicked.connect(self._on_clear_mask)
        ops_col.addWidget(clear_btn)
        obj_ops_wrap = QWidget(); obj_ops_wrap.setLayout(ops_col)
        obj_card.body_layout.addWidget(obj_ops_wrap)
        col.addWidget(obj_card)

        col.addStretch(1)
        return wrap

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_next)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save)
        QShortcut(QKeySequence("B"), self, lambda: self._set_mode(MODE_BRUSH))
        QShortcut(QKeySequence("E"), self, lambda: self._set_mode(MODE_ERASE))

    # ------------------------------------------------------------------
    # Mode / brush plumbing
    # ------------------------------------------------------------------
    def _set_mode(self, mode: str):
        self._canvas.mode = mode
        self._btn_brush.setChecked(mode == MODE_BRUSH)
        self._btn_erase.setChecked(mode == MODE_ERASE)
        self._btn_del_obj.setChecked(mode == MODE_ERASE_OBJECT)

    def _on_brush_size_changed(self, v: int):
        self._canvas.brush_radius = int(v)
        self._brush_size_label.setText(f"{v} px")

    def _on_normalize_changed(self, _v: float):
        self._canvas.norm_lo = float(self._norm_lo.value())
        self._canvas.norm_hi = float(self._norm_hi.value())
        self._canvas.refresh()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pick images folder",
                                              self._folder or os.getcwd())
        if not d:
            return
        self._open_folder(d)

    def _open_folder(self, folder: str):
        files = engine.list_images(folder)
        if not files:
            QMessageBox.warning(self, "No images",
                                 f"Found no image files in:\n{folder}")
            return
        self._folder = folder
        self._image_files = files
        self._current_index = 0
        self._src_label.setText(f"{folder}  —  {len(files)} images")
        self._load_current()
        self._sync_button_states()

    def _load_current(self):
        if not self._image_files:
            return
        try:
            image, mask = engine.load_image_and_mask(
                self._folder, self._image_files[self._current_index]
            )
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self._canvas.set_image_and_mask(image, mask)
        self._status_label.setText(
            f"{self._image_files[self._current_index]}  "
            f"({self._current_index + 1}/{len(self._image_files)})"
        )

    def _on_prev(self):
        if not self._image_files or self._current_index <= 0:
            return
        self._current_index -= 1
        self._load_current()

    def _on_next(self):
        if not self._image_files or self._current_index >= len(self._image_files) - 1:
            return
        self._current_index += 1
        self._load_current()

    def _on_save(self):
        if not self._image_files or self._canvas.mask is None:
            return
        try:
            path = engine.save_mask(
                self._folder,
                self._image_files[self._current_index],
                self._canvas.mask,
            )
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self._status_label.setText(f"Saved → {path}")

    def _on_fill_holes(self):
        if self._canvas.mask is None:
            return
        self._canvas.mask = engine.fill_holes(self._canvas.mask)
        self._canvas.refresh()

    def _on_relabel(self):
        if self._canvas.mask is None:
            return
        self._canvas.mask = engine.relabel_objects(self._canvas.mask)
        self._canvas.refresh()

    def _on_invert(self):
        if self._canvas.mask is None:
            return
        self._canvas.mask = engine.invert_mask(self._canvas.mask)
        self._canvas.refresh()

    def _on_remove_small(self):
        if self._canvas.mask is None:
            return
        self._canvas.mask = engine.remove_small_objects(
            self._canvas.mask, int(self._min_area.value())
        )
        self._canvas.refresh()

    def _on_clear_mask(self):
        if self._canvas.mask is None:
            return
        ans = QMessageBox.question(self, "Clear mask",
                                    "Zero out the current mask?")
        if ans != QMessageBox.Yes:
            return
        self._canvas.mask = engine.clear_mask(self._canvas.mask)
        self._canvas.refresh()

    def _on_stroke_finished(self):
        # Placeholder for undo history — future enhancement.
        pass

    # ------------------------------------------------------------------
    def _sync_button_states(self):
        has_files = bool(self._image_files)
        for b in (self._btn_prev, self._btn_next, self._btn_save,
                   self._btn_brush, self._btn_erase, self._btn_del_obj):
            b.setEnabled(has_files)
