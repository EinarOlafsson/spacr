"""Align & Stitch — see the layout, spot what did not register, then write.

Stitching is one of the few operations where the failure is invisible in
the output. A tile that did not register is placed at its nominal stage
position; the result still looks like a mosaic, still has the right
dimensions, and is still wrong by however far the stage was off. If the
first time anyone finds out is when a downstream hit fails to reproduce,
the tool has failed.

So this screen puts the *plan* before the write:

* pick a folder of tiles, press **Plan** — nothing is written, no canvas
  is allocated, and only tile headers plus overlap strips are read;
* the layout is drawn tile by tile, coloured by registration confidence,
  with anything that fell back to the stage position drawn in the warning
  colour and hatched. Those tiles are countable at a glance;
* the estimated canvas size and the RAM the write would use are stated
  *before* the button that writes 700 MB is enabled;
* **Write stack** then composites incrementally, and optionally records
  the coordinates into ``measurements.db``.

Every long operation runs on a worker thread and reports back through a
bound method, never a closure — see :meth:`AlignScreen._run_job`. Errors
land in the inline status label; a QMessageBox here would hang a headless
run, so there are none.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

from ... import align as align_mod
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider

__all__ = [
    "AlignScreen",
    "TileLayoutWidget",
    "confidence_colour",
]

#: Padding around the drawn layout, in pixels.
_PAD = 10


def confidence_colour(confidence: float, method: str) -> QColor:
    """Colour for one tile in the layout view.

    A tile placed by stage position alone is drawn in the warning colour
    whatever its (zero) confidence, because "we guessed" is a different
    kind of fact from "we matched, weakly". Registered tiles ramp from the
    muted surface at confidence 0.3 to full accent at 1.0, so a weak match
    reads as pale rather than as a different category.
    """
    if method == align_mod.METHOD_NOMINAL:
        return QColor(active_palette()["warning"])
    if method == align_mod.METHOD_UNREADABLE:
        return QColor(active_palette()["error"])
    palette = active_palette()
    accent = QColor(palette["accent"])
    base = QColor(palette["surface_hi"])
    t = max(0.0, min(1.0, (float(confidence) - 0.3) / 0.7))
    return QColor(
        int(base.red() + (accent.red() - base.red()) * t),
        int(base.green() + (accent.green() - base.green()) * t),
        int(base.blue() + (accent.blue() - base.blue()) * t))


class TileLayoutWidget(QWidget):
    """The stitch layout: one rectangle per tile, coloured by confidence.

    Draws the *plan*, not the pixels — it never opens an image, so showing
    the layout of a 700 MB stitch costs nothing. Hovering is not wired;
    the per-tile detail lives in the report pane beside it, where it can
    be read and copied.
    """

    #: emitted with the tile index when a tile is clicked, -1 for the void
    tile_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plan = None
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_plan(self, plan) -> None:
        """Show ``plan`` (an :class:`spacr.align.AlignPlan`), or ``None``."""
        self._plan = plan
        self.update()

    def plan(self):
        """The plan currently drawn."""
        return self._plan

    def tile_rects(self) -> "List[Tuple[int, QRectF]]":
        """Return ``(tile_index, rect)`` in widget coordinates.

        Public so a test can assert the layout without screen-scraping
        pixels.
        """
        if self._plan is None or not self._plan.placements:
            return []
        height, width, _channels = self._plan.canvas_shape
        if height <= 0 or width <= 0:
            return []
        avail_w = max(1.0, self.width() - 2 * _PAD)
        avail_h = max(1.0, self.height() - 2 * _PAD)
        scale = min(avail_w / width, avail_h / height)
        off_x = _PAD + (avail_w - width * scale) / 2.0
        off_y = _PAD + (avail_h - height * scale) / 2.0
        origin_y, origin_x = self._plan.origin
        out = []
        for placement in self._plan.placements:
            tile = placement.tile
            rect = QRectF(
                off_x + (placement.x - origin_x) * scale,
                off_y + (placement.y - origin_y) * scale,
                max(1.0, tile.width * scale),
                max(1.0, tile.height * scale))
            out.append((tile.index, rect))
        return out

    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        """Draw the tile rectangles, or the empty-state hint."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = active_palette()
        painter.fillRect(self.rect(), QColor(palette["surface"]))

        rects = self.tile_rects()
        if not rects:
            painter.setPen(QPen(QColor(palette["fg_dim"])))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Choose a folder of tiles and press Plan.")
            painter.end()
            return

        by_index = {p.tile.index: p for p in self._plan.placements}
        font = QFont(painter.font())
        font.setPointSizeF(max(6.0, font.pointSizeF() - 2))
        painter.setFont(font)

        for index, rect in rects:
            placement = by_index[index]
            colour = confidence_colour(placement.confidence, placement.method)
            painter.fillRect(rect, QBrush(colour))
            if placement.method == align_mod.METHOD_NOMINAL:
                # Hatch as well as colour: a colour-vision-impaired reader
                # must still be able to count the fallbacks. The hatch is
                # drawn in the *surface* colour, not the warning colour —
                # warning-on-warning is the same colour twice and paints
                # nothing at all, which is exactly as much help to that
                # reader as leaving the hatch out.
                painter.fillRect(rect, QBrush(QColor(palette["surface"]),
                                              Qt.BDiagPattern))
            painter.setPen(QPen(QColor(palette["border"]), 1))
            painter.drawRect(rect)
            if rect.width() > 26 and rect.height() > 14:
                painter.setPen(QPen(QColor(palette["fg"])))
                painter.drawText(rect, Qt.AlignCenter, str(placement.tile.field))
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        """Emit :attr:`tile_clicked` for whatever was under the cursor."""
        point = event.position() if hasattr(event, "position") else event.pos()
        for index, rect in self.tile_rects():
            if rect.contains(point):
                self.tile_clicked.emit(int(index))
                return
        self.tile_clicked.emit(-1)


class AlignScreen(QWidget):
    """The Align & Stitch tool.

    :param parent: parent widget.
    :param threaded: run the scan/plan/write on a worker thread (the
        default). Tests pass ``False`` for deterministic, synchronous
        behaviour — both paths emit the same signals.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation succeeded. Errors are *only* reported here and in
        the inline status label, never in a modal dialog.
    """

    #: emitted after a plan has been computed and drawn
    plan_ready = Signal(object)
    #: emitted with the written .npy path
    stack_written = Signal(str)
    #: emitted after every job settles (ok or not)
    job_finished = Signal(bool)
    #: private. Re-emitted from ``PipelineWorker.finished`` purely to hop
    #: back onto the GUI thread — see :meth:`_run_job`.
    _job_settled = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._plan = None
        self._result = None
        self._busy = False
        self._jobs: List[tuple] = []
        self._pending: List[Tuple[Dict[str, Any], Callable[[Any], None]]] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""

        self._job_settled.connect(self._on_job_settled)
        self._build_ui()
        self._set_status(
            "Choose a folder of tiles (.npy or .tif) and press Plan. "
            "Planning reads headers and overlap strips only — nothing is "
            "written and no canvas is allocated.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Align & Stitch")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Register an arbitrary number of tiles into one canvas and write "
            "it incrementally, so a 20000x20000 mosaic never has to fit in "
            "RAM. Offsets are solved globally rather than accumulated, and "
            "any tile that failed to register is drawn in orange and "
            "recorded as such — a stitch that quietly fell back to stage "
            "positions is the failure mode this screen exists to expose.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)
        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._src_edit = QLineEdit(self)
        self._src_edit.setPlaceholderText("…/plate1/tiles")
        self._src_edit.setClearButtonEnabled(True)
        self._src_edit.textChanged.connect(lambda _t: self._update_controls())
        self._src_edit.returnPressed.connect(self.build_plan)
        self._btn_pick_src = QPushButton("Choose tile folder…", self)
        self._btn_pick_src.clicked.connect(self._pick_source)
        src_row.addWidget(QLabel("Tiles", self))
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(self._btn_pick_src)
        outer.addLayout(src_row)

        # ── Layout row ────────────────────────────────────────────────
        grid_row = QHBoxLayout()
        grid_row.setSpacing(SPACING["sm"])
        grid_row.addWidget(QLabel("Grid", self))
        self._rows_box = QSpinBox(self)
        self._rows_box.setRange(0, 200)
        self._rows_box.setSpecialValueText("auto")
        self._rows_box.setToolTip(
            "(int) Rows in the acquisition grid. 0 infers it from the tile "
            "count.")
        grid_row.addWidget(self._rows_box)
        grid_row.addWidget(QLabel("x", self))
        self._cols_box = QSpinBox(self)
        self._cols_box.setRange(0, 200)
        self._cols_box.setSpecialValueText("auto")
        self._cols_box.setToolTip("(int) Columns in the acquisition grid.")
        grid_row.addWidget(self._cols_box)

        grid_row.addWidget(QLabel("Overlap", self))
        self._overlap_box = QDoubleSpinBox(self)
        self._overlap_box.setRange(0.0, 0.95)
        self._overlap_box.setSingleStep(0.05)
        self._overlap_box.setDecimals(3)
        self._overlap_box.setValue(align_mod.DEFAULT_OVERLAP)
        self._overlap_box.setToolTip(
            "(float) Nominal overlap between neighbours, as a fraction of "
            "the tile. Only seeds the search — registration corrects it.")
        grid_row.addWidget(self._overlap_box)

        grid_row.addWidget(QLabel("Order", self))
        self._order_combo = QComboBox(self)
        self._order_combo.addItems(list(align_mod.ORDERS))
        self._order_combo.setToolTip(
            "(str) How the field sequence maps onto the grid. A serpentine "
            "acquisition needs snake-row.")
        grid_row.addWidget(self._order_combo)

        grid_row.addWidget(QLabel("Ref. channel", self))
        self._ref_box = QSpinBox(self)
        self._ref_box.setRange(0, 32)
        self._ref_box.setToolTip(
            "(int) The channel registration is measured on. Every channel "
            "then shares that one solution — aligning channels "
            "independently would shear the composite.")
        grid_row.addWidget(self._ref_box)
        grid_row.addStretch(1)
        outer.addLayout(grid_row)

        # ── Quality row ───────────────────────────────────────────────
        qual_row = QHBoxLayout()
        qual_row.setSpacing(SPACING["sm"])
        qual_row.addWidget(QLabel("Min confidence", self))
        self._conf_box = QDoubleSpinBox(self)
        self._conf_box.setRange(0.0, 1.0)
        self._conf_box.setSingleStep(0.05)
        self._conf_box.setDecimals(2)
        self._conf_box.setValue(align_mod.DEFAULT_MIN_CONFIDENCE)
        self._conf_box.setToolTip(
            "(float) Cross-correlation below which a pair is not believed "
            "and the tile keeps its stage position.")
        qual_row.addWidget(self._conf_box)

        qual_row.addWidget(QLabel("Neighbour radius", self))
        self._radius_box = QSpinBox(self)
        self._radius_box.setRange(1, 6)
        self._radius_box.setValue(1)
        self._radius_box.setToolTip(
            "(int) Grid distance to register over. Raise it above 50% "
            "overlap so the extra pairs become real redundancy in the "
            "global solve.")
        qual_row.addWidget(self._radius_box)

        qual_row.addWidget(QLabel("Blend", self))
        self._blend_combo = QComboBox(self)
        self._blend_combo.addItems(list(align_mod.BLEND_MODES))
        self._blend_combo.setToolTip(
            "(str) feather ramps each tile's weight across the overlap; "
            "none is a hard cut and leaves a visible seam.")
        qual_row.addWidget(self._blend_combo)

        qual_row.addWidget(QLabel("RAM budget (MB)", self))
        self._budget_box = QSpinBox(self)
        self._budget_box.setRange(1, 8192)
        self._budget_box.setValue(align_mod.DEFAULT_MAX_BUFFER_BYTES >> 20)
        self._budget_box.setToolTip(
            "(int) Ceiling on the band buffer the write allocates. This, "
            "not the canvas, is what the stitch costs in RAM.")
        qual_row.addWidget(self._budget_box)
        qual_row.addStretch(1)

        self._btn_plan = QPushButton("Plan", self)
        self._btn_plan.setObjectName("Primary")
        self._btn_plan.clicked.connect(self.build_plan)
        qual_row.addWidget(self._btn_plan)
        outer.addLayout(qual_row)

        # ── Layout | report ───────────────────────────────────────────
        split = QSplitter(Qt.Horizontal, self)
        self._layout_view = TileLayoutWidget(self)
        self._layout_view.tile_clicked.connect(self._on_tile_clicked)
        split.addWidget(self._layout_view)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])
        self._report_view = QPlainTextEdit(right)
        self._report_view.setReadOnly(True)
        self._report_view.setPlaceholderText(
            "The plan appears here: canvas size, how many tiles registered, "
            "which ones fell back to their stage position, and the largest "
            "residuals.")
        right_layout.addWidget(self._report_view, 1)
        self._tile_label = QLabel("", right)
        self._tile_label.setObjectName("Muted")
        self._tile_label.setWordWrap(True)
        right_layout.addWidget(self._tile_label)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)

        # ── Output row ────────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_row.setSpacing(SPACING["sm"])
        out_row.addWidget(QLabel("Write to", self))
        self._dst_edit = QLineEdit(self)
        self._dst_edit.setPlaceholderText("…/plate1/stitched  (folder or .npy)")
        self._dst_edit.setClearButtonEnabled(True)
        self._dst_edit.textChanged.connect(lambda _t: self._update_controls())
        out_row.addWidget(self._dst_edit, 1)
        self._btn_pick_dst = QPushButton("Choose…", self)
        self._btn_pick_dst.clicked.connect(self._pick_destination)
        out_row.addWidget(self._btn_pick_dst)

        out_row.addWidget(QLabel("Database", self))
        self._db_edit = QLineEdit(self)
        self._db_edit.setPlaceholderText(
            "optional …/measurements/measurements.db")
        self._db_edit.setClearButtonEnabled(True)
        out_row.addWidget(self._db_edit, 1)

        self._overwrite_box = QCheckBox("Overwrite", self)
        self._overwrite_box.setToolTip(
            "(bool) Replace an existing output. Off, an existing file is "
            "refused rather than clobbered.")
        out_row.addWidget(self._overwrite_box)

        self._btn_write = QPushButton("Write stack", self)
        self._btn_write.setObjectName("Primary")
        self._btn_write.clicked.connect(self.write_stack)
        out_row.addWidget(self._btn_write)
        outer.addLayout(out_row)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    # -- introspection -----------------------------------------------------

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
        """The rendered plan (test/introspection helper)."""
        return self._report_view.toPlainText()

    def tile_info_text(self) -> str:
        """The per-tile readout line (test/introspection helper)."""
        return self._tile_label.text()

    def plan(self):
        """The current :class:`spacr.align.AlignPlan`, or ``None``."""
        return self._plan

    def result(self):
        """The last :class:`spacr.align.AlignResult`, or ``None``."""
        return self._result

    def is_busy(self) -> bool:
        """True while a job is in flight."""
        return self._busy

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def _update_controls(self) -> None:
        ready = not self._busy
        self._btn_plan.setEnabled(ready and bool(self._src_edit.text().strip()))
        self._btn_write.setEnabled(
            ready and self._plan is not None
            and bool(self._dst_edit.text().strip()))
        for widget in (self._src_edit, self._rows_box, self._cols_box,
                       self._overlap_box, self._order_combo, self._ref_box,
                       self._conf_box, self._radius_box, self._blend_combo,
                       self._budget_box, self._dst_edit, self._db_edit,
                       self._overwrite_box, self._btn_pick_src,
                       self._btn_pick_dst):
            widget.setEnabled(ready)

    # -- settings ----------------------------------------------------------

    def settings(self) -> Dict[str, Any]:
        """Return the controls as a :func:`spacr.align.default_settings` dict.

        Kept public so the Batch Runner and the Queue can snapshot this
        screen the way they snapshot every other one.
        """
        rows, cols = self._rows_box.value(), self._cols_box.value()
        return align_mod.default_settings({
            'src': self._src_edit.text().strip() or None,
            'dst': self._dst_edit.text().strip() or None,
            'db_path': self._db_edit.text().strip() or None,
            'grid': (rows, cols) if rows and cols else None,
            'overlap': float(self._overlap_box.value()),
            'order': self._order_combo.currentText(),
            'reference_channel': int(self._ref_box.value()),
            'min_confidence': float(self._conf_box.value()),
            'neighbour_radius': int(self._radius_box.value()),
            'blend': self._blend_combo.currentText(),
            'max_buffer_bytes': int(self._budget_box.value()) << 20,
            'overwrite': bool(self._overwrite_box.isChecked()),
        })

    def apply_settings(self, settings: Dict[str, Any]) -> None:
        """Load a settings dict back into the controls."""
        resolved = align_mod.default_settings(settings)
        self._src_edit.setText(str(resolved.get('src') or ''))
        self._dst_edit.setText(str(resolved.get('dst') or ''))
        self._db_edit.setText(str(resolved.get('db_path') or ''))
        grid = resolved.get('grid')
        self._rows_box.setValue(int(grid[0]) if grid else 0)
        self._cols_box.setValue(int(grid[1]) if grid else 0)
        self._overlap_box.setValue(float(resolved.get('overlap') or 0.0))
        index = self._order_combo.findText(str(resolved.get('order') or ''))
        if index >= 0:
            self._order_combo.setCurrentIndex(index)
        self._ref_box.setValue(int(resolved.get('reference_channel') or 0))
        self._conf_box.setValue(float(resolved.get('min_confidence') or 0.0))
        self._radius_box.setValue(int(resolved.get('neighbour_radius') or 1))
        index = self._blend_combo.findText(str(resolved.get('blend') or ''))
        if index >= 0:
            self._blend_combo.setCurrentIndex(index)
        self._budget_box.setValue(
            max(4, int(resolved.get('max_buffer_bytes') or 0) >> 20))
        self._overwrite_box.setChecked(bool(resolved.get('overwrite')))
        self._update_controls()

    # -- pickers -----------------------------------------------------------

    def _pick_source(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose the folder of tiles",
            self._src_edit.text() or os.path.expanduser("~"))
        if path:
            self._src_edit.setText(path)

    def _pick_destination(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose where to write the stitched stack",
            self._dst_edit.text() or self._src_edit.text()
            or os.path.expanduser("~"))
        if path:
            self._dst_edit.setText(path)

    # -- planning ----------------------------------------------------------

    def build_plan(self) -> bool:
        """Scan the source and solve the layout. Writes nothing.

        :returns: True when the job was started (or, unthreaded, when it
            succeeded).
        """
        src = self._src_edit.text().strip()
        if not src:
            self._set_status("Choose a folder of tiles first.", error=True)
            return False
        settings = self.settings()
        self._set_status("Planning — reading tile headers and overlap "
                         "strips…")

        def _work():
            tiles = align_mod.scan_tiles(
                src,
                grid=settings.get('grid'),
                overlap=float(settings['overlap']),
                order=str(settings['order']))
            plan = align_mod.estimate_offsets(
                tiles,
                reference_channel=int(settings['reference_channel']),
                min_confidence=float(settings['min_confidence']),
                neighbour_radius=int(settings['neighbour_radius']))
            return plan

        return self._run_job(_work, self._on_plan_ready)

    def _on_plan_ready(self, plan) -> None:
        """Draw a finished plan. Always on the GUI thread."""
        self._plan = plan
        self._result = None
        self._layout_view.set_plan(plan)
        self._report_view.setPlainText(align_mod.format_plan(plan))
        self._tile_label.setText("")

        spec = align_mod.plan_canvas(plan.placements, dtype=plan.dtype)
        band = align_mod._band_rows(
            spec, int(self._budget_box.value()) << 20, None)
        buffer_bytes = band * align_mod._band_bytes_per_row(spec)
        message = (
            f"{plan.n_registered} of {len(plan.placements)} tile(s) "
            f"registered; canvas {spec.height} x {spec.width} x "
            f"{spec.channels} {spec.dtype} "
            f"({align_mod._human_bytes(spec.nbytes)} on disk), written in "
            f"{band}-row bands using "
            f"{align_mod._human_bytes(buffer_bytes)} of RAM.")
        if plan.n_nominal:
            message += (f" {plan.n_nominal} tile(s) did NOT register and are "
                        f"placed by stage position — shown in orange.")
        if plan.unplaced:
            message += f" {len(plan.unplaced)} tile(s) could not be read."
        self._set_status(message, error=bool(plan.n_nominal or plan.unplaced))
        self.plan_ready.emit(plan)

    def _on_tile_clicked(self, index: int) -> None:
        """Show one tile's numbers under the report."""
        if self._plan is None or index < 0:
            self._tile_label.setText("")
            return
        for placement in self._plan.placements:
            if placement.tile.index == index:
                self._tile_label.setText(
                    f"{os.path.basename(placement.tile.path)} — "
                    f"y={placement.y:.2f} x={placement.x:.2f}, "
                    f"{placement.method}, confidence "
                    f"{placement.confidence:.3f}, residual "
                    f"{placement.residual:.2f} px, {placement.n_pairs} pair(s)"
                    + (f" — {placement.note}" if placement.note else ""))
                return
        self._tile_label.setText("")

    # -- writing -----------------------------------------------------------

    def write_stack(self) -> bool:
        """Composite the current plan to disk, and optionally to the database."""
        if self._plan is None:
            self._set_status("Press Plan first.", error=True)
            return False
        dst = self._dst_edit.text().strip()
        if not dst:
            self._set_status("Choose where to write the stitched stack.",
                             error=True)
            return False
        plan = self._plan
        settings = self.settings()
        db_path = settings.get('db_path')
        self._set_status("Writing — the canvas is filled one band at a time…")

        def _work():
            result = align_mod.write_stack(
                plan, dst,
                blend=str(settings['blend']),
                max_buffer_bytes=int(settings['max_buffer_bytes']),
                overwrite=bool(settings['overwrite']))
            if db_path:
                align_mod.save_coordinates(
                    plan, db_path, canvas=result.canvas,
                    stack_path=result.stack_path)
                result.db_path = str(db_path)
            return result

        return self._run_job(_work, self._on_stack_written)

    def _on_stack_written(self, result) -> None:
        """Report a finished write. Always on the GUI thread."""
        self._result = result
        self._report_view.setPlainText(
            align_mod.format_plan(result.plan) + "\n\n" + result.summary())
        message = result.summary().splitlines()[0]
        if result.db_path:
            message += f" Coordinates written to {result.db_path}."
        self._set_status(message, error=bool(result.n_skipped))
        self.stack_written.emit(result.stack_path)

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Copied from ``PlateViewScreen._run_job`` — one threading idiom for
        the whole Qt layer. ``PipelineWorker.finished`` is emitted *in the
        worker thread*, and PySide6 invokes a plain closure connected to it
        directly, on that same thread; this screen's completion handler
        fills a QPlainTextEdit, and building a QTextDocument's children off
        the GUI thread is undefined behaviour. So ``finished`` is chained
        through :attr:`_job_settled` into a *bound method* of this widget,
        which has GUI-thread affinity — Qt then queues the call and the
        handler runs where every other widget call runs.

        The worker is deliberately **not** scheduled for deletion; see
        :func:`spacr.qt.bridge.make_thread` for the segfault that caused.

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

    def _on_job_error(self, exc: Exception) -> None:
        self._busy = False
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._set_status(f"Align failed: {line}", error=True)
