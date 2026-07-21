"""
MakeMasksScreen — Qt widget replacing the Tk ModifyMaskApp.

Load a folder of images and their masks (in `<folder>/masks/`), draw
brush/erase strokes on the mask, run object-level operations (fill,
relabel, invert, remove small), zoom into a region for detailed edits,
use a magic-wand flood-fill by intensity, undo/redo, and save the
edited mask back to `<folder>/masks/<name>.tif` as a labeled uint16
mask.

Advanced features still deferred (noted in the toolbar):
dividing line and free-form polygon draw.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
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
from .. import prefs
from ..theme import PALETTE, SPACING
from ..widgets import Card, Divider, Section


# ---------------------------------------------------------------------------
# Canvas — image + mask overlay with brush/erase mouse handling
# ---------------------------------------------------------------------------

MODE_NONE = "none"
MODE_BRUSH = "brush"
MODE_ERASE = "erase"
MODE_ERASE_OBJECT = "erase_object"
MODE_WAND_ADD = "wand_add"
MODE_WAND_ERASE = "wand_erase"
MODE_ZOOM = "zoom"


class _MaskCanvas(QLabel):
    """QLabel that displays the composited image+mask (optionally zoomed
    into a sub-region) and captures mouse events for brush / erase /
    magic-wand / erase-object / zoom-rectangle interactions.

    All coordinate math is done against the *full* image; the "zoom
    view" is just a crop of the composited pixmap. Mask edits go
    directly into `self.mask` (with the correct zoom offset applied).
    """

    stroke_started = Signal()      # emitted just before self.mask is mutated
    stroke_finished = Signal()     # emitted after a stroke completes
    zoom_changed = Signal(bool)    # emitted with True when zoom entered / False on reset

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.image: Optional[np.ndarray] = None       # uint16 grayscale
        self.mask: Optional[np.ndarray] = None        # uint8 labels
        self.mode: str = MODE_NONE
        self.brush_radius: int = 10
        self.norm_lo: float = 1.0
        self.norm_hi: float = 99.9
        self.wand_tolerance: float = 1000.0
        self.wand_max_pixels: int = 100_000

        # Zoom viewport in image coords; None = full-image view.
        self._zoom_x0: Optional[int] = None
        self._zoom_y0: Optional[int] = None
        self._zoom_x1: Optional[int] = None
        self._zoom_y1: Optional[int] = None

        # Zoom-rectangle drag state (widget-local pixel coords)
        self._zoom_drag_start: Optional[QPoint] = None
        self._zoom_drag_end: Optional[QPoint] = None

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {PALETTE['bg']};")
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)
        self._last_pt: Optional[QPoint] = None
        self._stroke_in_progress = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def set_image_and_mask(self, image: np.ndarray, mask: np.ndarray):
        self.image = image
        self.mask = mask
        self.reset_zoom(silent=True)
        self.refresh()

    def _viewport_bounds(self):
        """Return (x0, y0, x1, y1) — inclusive-of-x0, exclusive-of-x1."""
        if self.mask is None:
            return (0, 0, 0, 0)
        if self._zoom_x0 is not None:
            return (self._zoom_x0, self._zoom_y0, self._zoom_x1, self._zoom_y1)
        h, w = self.mask.shape[:2]
        return (0, 0, w, h)

    def is_zoomed(self) -> bool:
        return self._zoom_x0 is not None

    def reset_zoom(self, silent: bool = False):
        was_zoomed = self.is_zoomed()
        self._zoom_x0 = self._zoom_y0 = self._zoom_x1 = self._zoom_y1 = None
        self._zoom_drag_start = self._zoom_drag_end = None
        if was_zoomed and not silent:
            self.zoom_changed.emit(False)
        self.refresh()

    def refresh(self):
        if self.image is None or self.mask is None:
            return
        img = engine.normalize_uint16(self.image, self.norm_lo, self.norm_hi)
        x0, y0, x1, y1 = self._viewport_bounds()
        sub_img = img[y0:y1, x0:x1]
        sub_mask = self.mask[y0:y1, x0:x1]
        composed = engine.overlay_mask(sub_img, sub_mask, alpha=0.5)
        h, w = composed.shape[:2]
        if w <= 0 or h <= 0:
            return
        qimg = QImage(composed.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        avail_w = max(200, self.width())
        avail_h = max(200, self.height())
        pixmap = pixmap.scaled(avail_w, avail_h,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
        self.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Coordinate mapping (widget-local px  ↔  full image px)
    # ------------------------------------------------------------------
    def _canvas_to_image(self, x: float, y: float) -> Optional[tuple]:
        if self.mask is None or self.pixmap() is None:
            return None
        p = self.pixmap()
        pw, ph = p.width(), p.height()
        if pw == 0 or ph == 0:
            return None
        w, h = self.width(), self.height()
        ox = (w - pw) // 2
        oy = (h - ph) // 2
        cx, cy = float(x) - ox, float(y) - oy
        if not (0 <= cx < pw and 0 <= cy < ph):
            return None
        x0, y0, x1, y1 = self._viewport_bounds()
        sub_w = max(1, x1 - x0)
        sub_h = max(1, y1 - y0)
        img_x = int(x0 + cx * sub_w / pw)
        img_y = int(y0 + cy * sub_h / ph)
        # Clamp to image bounds
        img_x = max(0, min(self.mask.shape[1] - 1, img_x))
        img_y = max(0, min(self.mask.shape[0] - 1, img_y))
        return img_x, img_y

    def _mask_radius_for_brush(self) -> int:
        """Scale the brush radius (in screen px) to full-image px, taking
        the current zoom into account."""
        if self.mask is None or self.pixmap() is None:
            return self.brush_radius
        p = self.pixmap()
        if p.width() == 0:
            return self.brush_radius
        x0, _, x1, _ = self._viewport_bounds()
        sub_w = max(1, x1 - x0)
        return max(1, int(self.brush_radius * sub_w / p.width()))

    # ------------------------------------------------------------------
    # Painting (adds a zoom-rectangle overlay while dragging)
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.mode != MODE_ZOOM:
            return
        if self._zoom_drag_start is None or self._zoom_drag_end is None:
            return
        painter = QPainter(self)
        pen = QPen(QColor(PALETTE["accent"]))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        rect = QRect(self._zoom_drag_start, self._zoom_drag_end).normalized()
        painter.drawRect(rect)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def _emit_stroke_start(self):
        if not self._stroke_in_progress:
            self._stroke_in_progress = True
            self.stroke_started.emit()

    def _emit_stroke_end(self):
        if self._stroke_in_progress:
            self._stroke_in_progress = False
            self.stroke_finished.emit()

    def mousePressEvent(self, event):
        if self.mode == MODE_NONE or self.mask is None:
            return super().mousePressEvent(event)

        if self.mode == MODE_ZOOM:
            self._zoom_drag_start = event.position().toPoint()
            self._zoom_drag_end = event.position().toPoint()
            self.update()
            return

        pt = self._canvas_to_image(event.position().x(), event.position().y())
        if pt is None:
            return
        self._emit_stroke_start()

        if self.mode == MODE_ERASE_OBJECT:
            self.mask = engine.erase_object_at(self.mask, *pt)
            self.refresh()
            self._emit_stroke_end()
            return

        if self.mode in (MODE_WAND_ADD, MODE_WAND_ERASE):
            action = "add" if self.mode == MODE_WAND_ADD else "erase"
            self.mask = engine.magic_wand(
                self.image, self.mask, pt[0], pt[1],
                self.wand_tolerance, self.wand_max_pixels, action=action,
            )
            self.refresh()
            self._emit_stroke_end()
            return

        # Brush / erase strokes
        radius = self._mask_radius_for_brush()
        value = 255 if self.mode == MODE_BRUSH else 0
        engine.paint_disk(self.mask, pt[0], pt[1], radius, value)
        self._last_pt = QPoint(*pt)
        self.refresh()

    def mouseMoveEvent(self, event):
        if self.mask is None:
            return
        if self.mode == MODE_ZOOM and self._zoom_drag_start is not None \
                and event.buttons() & Qt.LeftButton:
            self._zoom_drag_end = event.position().toPoint()
            self.update()
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
        if self.mode == MODE_ZOOM and self._zoom_drag_start is not None \
                and self._zoom_drag_end is not None:
            # Convert both endpoints to image coords and commit
            p0 = self._canvas_to_image(self._zoom_drag_start.x(),
                                        self._zoom_drag_start.y())
            p1 = self._canvas_to_image(self._zoom_drag_end.x(),
                                        self._zoom_drag_end.y())
            self._zoom_drag_start = None
            self._zoom_drag_end = None
            if p0 is not None and p1 is not None:
                x0, x1 = sorted((p0[0], p1[0]))
                y0, y1 = sorted((p0[1], p1[1]))
                if x1 - x0 > 4 and y1 - y0 > 4:
                    self._zoom_x0, self._zoom_y0 = x0, y0
                    self._zoom_x1, self._zoom_y1 = x1 + 1, y1 + 1
                    self.zoom_changed.emit(True)
            self.refresh()
            return
        if self._last_pt is not None:
            self._last_pt = None
        self._emit_stroke_end()

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
        self._history = engine.MaskHistory(capacity=25)
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
        self._canvas.stroke_started.connect(self._on_stroke_started)
        self._canvas.stroke_finished.connect(self._on_stroke_finished)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        body.addWidget(self._canvas)

        # Right: tools card (in a scroll area — the panel can be tall)
        tools_scroll = QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setFrameShape(QScrollArea.NoFrame)
        tools_scroll.setWidget(self._build_tools_panel())
        body.addWidget(tools_scroll)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 1)
        body.setSizes([900, 380])
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

        # Mode buttons — arranged as a 2×3 grid so buttons keep their labels
        mode_card = Card(title="Tools")
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setSpacing(SPACING["sm"])
        self._mode_buttons: dict[str, QPushButton] = {}
        modes = [
            (MODE_BRUSH,        "Brush"),
            (MODE_ERASE,        "Erase"),
            (MODE_ERASE_OBJECT, "Erase object"),
            (MODE_WAND_ADD,     "Wand +"),
            (MODE_WAND_ERASE,   "Wand −"),
            (MODE_ZOOM,         "Zoom"),
        ]
        for i, (m, label) in enumerate(modes):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.clicked.connect(lambda _c=False, key=m: self._set_mode(key))
            grid.addWidget(btn, i // 3, i % 3)
            self._mode_buttons[m] = btn
        self._btn_brush = self._mode_buttons[MODE_BRUSH]
        self._btn_erase = self._mode_buttons[MODE_ERASE]
        self._btn_del_obj = self._mode_buttons[MODE_ERASE_OBJECT]
        self._btn_wand_add = self._mode_buttons[MODE_WAND_ADD]
        self._btn_wand_erase = self._mode_buttons[MODE_WAND_ERASE]
        self._btn_zoom = self._mode_buttons[MODE_ZOOM]
        mode_wrap = QWidget(); mode_wrap.setLayout(grid)
        mode_card.body_layout.addWidget(mode_wrap)

        # Reset zoom / undo redo row
        history_row = QHBoxLayout()
        history_row.setSpacing(SPACING["sm"])
        self._btn_reset_zoom = QPushButton("Reset zoom")
        self._btn_reset_zoom.setEnabled(False)
        self._btn_reset_zoom.clicked.connect(self._on_reset_zoom)
        history_row.addWidget(self._btn_reset_zoom)
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_undo)
        history_row.addWidget(self._btn_undo)
        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._on_redo)
        history_row.addWidget(self._btn_redo)
        hist_wrap = QWidget(); hist_wrap.setLayout(history_row)
        mode_card.body_layout.addWidget(hist_wrap)
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

        # Magic wand card
        wand_card = Card(title="Magic wand")
        wand_form = QFormLayout()
        self._wand_tol = QDoubleSpinBox()
        self._wand_tol.setRange(0.0, 1_000_000.0)
        self._wand_tol.setSingleStep(50.0)
        self._wand_tol.setValue(1000.0)
        self._wand_tol.valueChanged.connect(self._on_wand_tolerance_changed)
        wand_form.addRow("Tolerance", self._wand_tol)
        self._wand_max = QSpinBox()
        self._wand_max.setRange(1, 10_000_000)
        self._wand_max.setSingleStep(1000)
        self._wand_max.setValue(100_000)
        self._wand_max.valueChanged.connect(self._on_wand_max_changed)
        wand_form.addRow("Max pixels", self._wand_max)
        wand_card.body_layout.addLayout(wand_form)
        col.addWidget(wand_card)

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
        QShortcut(QKeySequence("W"), self, lambda: self._set_mode(MODE_WAND_ADD))
        QShortcut(QKeySequence("Z"), self, lambda: self._set_mode(MODE_ZOOM))
        QShortcut(QKeySequence("Escape"), self, self._on_reset_zoom)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    # ------------------------------------------------------------------
    # Mode / brush plumbing
    # ------------------------------------------------------------------
    def _set_mode(self, mode: str):
        self._canvas.mode = mode
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

    def _on_brush_size_changed(self, v: int):
        self._canvas.brush_radius = int(v)
        self._brush_size_label.setText(f"{v} px")

    def _on_normalize_changed(self, _v: float):
        self._canvas.norm_lo = float(self._norm_lo.value())
        self._canvas.norm_hi = float(self._norm_hi.value())
        self._canvas.refresh()

    def _on_wand_tolerance_changed(self, v: float):
        self._canvas.wand_tolerance = float(v)

    def _on_wand_max_changed(self, v: int):
        self._canvas.wand_max_pixels = int(v)

    def _on_reset_zoom(self):
        self._canvas.reset_zoom()

    def _on_zoom_changed(self, zoomed: bool):
        self._btn_reset_zoom.setEnabled(zoomed)
        self._status_label.setText("Zoomed — press Esc to reset" if zoomed
                                     else "Zoom reset")

    def _on_undo(self):
        prev = self._history.undo()
        if prev is None or self._canvas.mask is None:
            return
        self._canvas.mask = prev
        self._canvas.refresh()
        self._refresh_history_buttons()

    def _on_redo(self):
        nxt = self._history.redo()
        if nxt is None or self._canvas.mask is None:
            return
        self._canvas.mask = nxt
        self._canvas.refresh()
        self._refresh_history_buttons()

    def _refresh_history_buttons(self):
        self._btn_undo.setEnabled(self._history.can_undo())
        self._btn_redo.setEnabled(self._history.can_redo())


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
        # Reset undo history for the new image and seed with the loaded mask
        self._history.clear()
        self._history.push(mask)
        self._refresh_history_buttons()
        self._btn_reset_zoom.setEnabled(False)
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

    def _apply_op(self, op):
        """Run a mask -> mask function, refresh canvas, push to history."""
        if self._canvas.mask is None:
            return
        self._canvas.mask = op(self._canvas.mask)
        self._canvas.refresh()
        self._history.push(self._canvas.mask)
        self._refresh_history_buttons()

    def _on_fill_holes(self):
        self._apply_op(engine.fill_holes)

    def _on_relabel(self):
        self._apply_op(engine.relabel_objects)

    def _on_invert(self):
        self._apply_op(engine.invert_mask)

    def _on_remove_small(self):
        area = int(self._min_area.value())
        self._apply_op(lambda m: engine.remove_small_objects(m, area))

    def _on_clear_mask(self):
        if self._canvas.mask is None:
            return
        ans = QMessageBox.question(self, "Clear mask",
                                    "Zero out the current mask?")
        if ans != QMessageBox.Yes:
            return
        self._apply_op(engine.clear_mask)

    def _on_stroke_started(self):
        # Brush/erase strokes mutate the mask in place; nothing to record
        # until the stroke ends. History already has the pre-stroke mask
        # from the previous op/load.
        pass

    def _on_stroke_finished(self):
        if self._canvas.mask is not None:
            self._history.push(self._canvas.mask)
            self._refresh_history_buttons()

    # ------------------------------------------------------------------
    def _sync_button_states(self):
        has_files = bool(self._image_files)
        for b in (self._btn_prev, self._btn_next, self._btn_save,
                   self._btn_brush, self._btn_erase, self._btn_del_obj,
                   self._btn_wand_add, self._btn_wand_erase, self._btn_zoom):
            b.setEnabled(has_files)
