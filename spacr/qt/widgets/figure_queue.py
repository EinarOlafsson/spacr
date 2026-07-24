"""
Figure queue — the panel that collects matplotlib figures a pipeline
emits, shows them above the console, and lets the user scrub through
them without blowing up RAM.

Behaviour (per the spec):

* **Thumbnail strip** on the left — one small icon per figure, all
  kept in memory (icons are tiny).
* **Zoomable enlarged view** on the right — a QGraphicsView showing the
  current figure's full-resolution render. Wheel = zoom, fit-on-load,
  scales with the container (reuses the live-preview _ZoomView).
* **Forward / back navigation** — ◀ / ▶ buttons plus the thumbnail
  list, with an "N / total" position label.
* **RAM cap + temp spill** — the 100 most-recent figures keep their
  full-resolution QPixmap in RAM. When figure #101 arrives, figure #1's
  pixmap is dropped from RAM (its PNG stays on disk in a temp dir);
  #102 evicts #2, and so on — a 100-wide sliding window. Navigating
  back to an evicted figure reloads it from its temp PNG on demand.
* **Cleanup** — the temp directory is deleted when the queue is
  cleared or the owning screen is destroyed.

Every figure is rendered to a temp PNG as soon as it arrives, so the
spill copy always exists and the RAM pixmap is just a cache.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from .live_preview import _ZoomView

LOG = logging.getLogger("spacr.qt.figure_queue")

# Number of full-resolution pixmaps kept in RAM. Older figures live
# only as PNGs on disk until viewed.
RAM_CAP = 100


class FigureQueue(QWidget):
    """Scrollable, RAM-bounded gallery of pipeline figures."""

    def __init__(self, ram_cap: int = RAM_CAP, parent=None):
        super().__init__(parent)
        self._ram_cap = int(ram_cap)
        self._count = 0
        # id(fig) -> index, for dedup of repeated emits of the same fig.
        self._fig_index: Dict[int, int] = {}
        # index -> temp PNG path (every figure has one).
        self._png_paths: Dict[int, str] = {}
        # index -> matplotlib Figure (kept so the figure-settings dialog can
        # restyle + re-render it).
        self._figures: Dict[int, object] = {}
        # LRU cache of index -> full-res QPixmap (capped at ram_cap).
        self._ram: "OrderedDict[int, QPixmap]" = OrderedDict()
        self._tempdir: Optional[Path] = None
        self._current = -1
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)

        self._list = QListWidget()
        self._list.setObjectName("FiguresList")
        self._list.setFixedWidth(160)
        self._list.setIconSize(QSize(140, 90))
        self._list.setSpacing(4)
        self._list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self._list)

        self._view = _ZoomView(self)
        self._view.setMinimumHeight(280)
        body.addWidget(self._view, 1)
        root.addLayout(body, 1)

        # Navigation is via the thumbnail strip (click a thumbnail) — no
        # separate Prev/Next buttons. A "Figure settings…" button (shown only
        # when figures are rendered as PDF/vector) restyles the current figure.
        nav = QHBoxLayout()
        self._pos_label = QLabel("0 / 0", self)
        self._pos_label.setAlignment(Qt.AlignCenter)
        self._fig_settings_btn = QPushButton("Figure settings…", self)
        self._fig_settings_btn.clicked.connect(self._open_figure_settings)
        nav.addWidget(self._pos_label, 1)
        nav.addWidget(self._fig_settings_btn)
        root.addLayout(nav)
        self._refresh_nav()

    def _open_figure_settings(self) -> None:
        """Open the figure-settings dialog for the current figure."""
        fig = self._figures.get(self._current)
        if fig is None:
            return
        dlg = _FigureSettingsDialog(fig, self)
        if dlg.exec():
            # Re-render the restyled figure in place.
            png = self._png_paths.get(self._current)
            if png:
                pm = self._render_figure(fig, Path(png))
                if pm is not None:
                    self._cache_pixmap(self._current, pm)
                    self._view.set_pixmap(pm)

    # -- temp dir ----------------------------------------------------------

    def _ensure_tempdir(self) -> Path:
        if self._tempdir is None:
            self._tempdir = Path(tempfile.mkdtemp(prefix="spacr_figq_"))
        return self._tempdir

    # -- public API --------------------------------------------------------

    def add_figure(self, fig) -> int:
        """Render + append ``fig`` (a matplotlib Figure). Returns its
        index. Re-emitting the same figure object re-selects it instead
        of duplicating."""
        if id(fig) in self._fig_index:
            idx = self._fig_index[id(fig)]
            self.show_index(idx)
            return idx

        idx = self._count
        self._count += 1
        self._fig_index[id(fig)] = idx
        self._figures[idx] = fig

        # Render to a temp PNG (the durable spill copy) + a full-res
        # pixmap for RAM.
        png_path = self._ensure_tempdir() / f"fig_{idx:05d}.png"
        pixmap = self._render_figure(fig, png_path)
        self._png_paths[idx] = str(png_path)
        if pixmap is not None:
            self._cache_pixmap(idx, pixmap)

        # Thumbnail (small icon) — always kept.
        item = QListWidgetItem(f"#{idx + 1}")
        item.setTextAlignment(Qt.AlignCenter)
        if pixmap is not None and not pixmap.isNull():
            item.setIcon(QIcon(pixmap.scaled(
                140, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self._list.addItem(item)

        self._list.setCurrentRow(idx)   # jump to the newest
        self.show_index(idx)
        return idx

    def show_index(self, idx: int) -> None:
        if not (0 <= idx < self._count):
            return
        self._current = idx
        pixmap = self._pixmap_for(idx)
        if pixmap is not None:
            self._view.set_pixmap(pixmap)
        if self._list.currentRow() != idx:
            self._list.blockSignals(True)
            self._list.setCurrentRow(idx)
            self._list.blockSignals(False)
        self._refresh_nav()

    def show_prev(self) -> None:
        if self._current > 0:
            self.show_index(self._current - 1)

    def show_next(self) -> None:
        if self._current < self._count - 1:
            self.show_index(self._current + 1)

    def count(self) -> int:
        return self._count

    def ram_resident(self) -> int:
        """How many full-res pixmaps are currently held in RAM."""
        return len(self._ram)

    def spilled_count(self) -> int:
        """How many figures have been evicted from RAM to disk-only."""
        return max(0, self._count - len(self._ram))

    def clear(self) -> None:
        """Drop everything and delete the temp dir."""
        self._list.clear()
        self._ram.clear()
        self._png_paths.clear()
        self._fig_index.clear()
        self._figures.clear()
        self._count = 0
        self._current = -1
        self._view.set_pixmap(QPixmap())
        self._delete_tempdir()
        self._refresh_nav()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _style_figure(fig, bg: str, fg: str, text_size: int = 0) -> None:
        """Recolour a figure's background + all text to (bg, fg), so plots
        follow the app theme (dark → black bg + white text)."""
        try:
            fig.patch.set_facecolor(bg)
            for ax in fig.get_axes():
                ax.set_facecolor(bg)
                for sp in ax.spines.values():
                    sp.set_color(fg)
                ax.tick_params(colors=fg)
                texts = ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels())
                leg = ax.get_legend()
                if leg is not None:
                    texts += list(leg.get_texts())
                for t in texts:
                    t.set_color(fg)
                    if text_size:
                        t.set_fontsize(text_size)
        except Exception:
            pass

    def _render_figure(self, fig, png_path: Path) -> Optional[QPixmap]:
        """Save ``fig`` to ``png_path`` (raster, for display) and return a
        QPixmap of it. The figure background + text follow the app theme
        (dark → black bg + white text) unless overridden in figure settings."""
        try:
            from ..preferences import (get_figure_png_dpi, get_figure_format,
                                       get_figure_colors, get_figure_text_size)
            dpi = get_figure_png_dpi()
            bg, fg = get_figure_colors()
            text_size = get_figure_text_size()
        except Exception:
            dpi, get_figure_format = 200, (lambda: "png")
            bg, fg, text_size = "#ffffff", "#000000", 0
        self._style_figure(fig, bg, fg, text_size)
        # Cap the DISPLAY raster so a big multi-panel figure at a high DPI can't
        # balloon into a multi-hundred-MB PNG that's slow to decode and blocks
        # the UI. Screen never needs more than ~4000 px on the long side; the
        # vector .pdf (saved below) keeps full quality for export.
        try:
            w_in, h_in = fig.get_size_inches()
            longest_in = max(float(w_in), float(h_in)) or 1.0
            MAX_PX = 4000
            display_dpi = min(dpi, max(72, int(MAX_PX / longest_in)))
        except Exception:
            display_dpi = min(dpi, 200)
        try:
            fig.savefig(str(png_path), dpi=display_dpi, bbox_inches="tight",
                        facecolor=bg)
            # In PDF mode, also drop a vector .pdf next to the raster so the
            # figure-settings button has something editable to work with.
            try:
                if get_figure_format() == "pdf":
                    fig.savefig(str(Path(png_path).with_suffix(".pdf")),
                                bbox_inches="tight", facecolor=bg)
            except Exception:
                pass
        except Exception as e:
            LOG.info("figure render failed: %s", e)
            return None
        pm = QPixmap(str(png_path))
        return pm if not pm.isNull() else None

    def _cache_pixmap(self, idx: int, pixmap: QPixmap) -> None:
        """Insert into the LRU RAM cache, evicting the oldest beyond the
        cap. The PNG on disk is untouched, so an evicted figure can be
        reloaded on demand."""
        self._ram[idx] = pixmap
        self._ram.move_to_end(idx)
        while len(self._ram) > self._ram_cap:
            old_idx, _ = self._ram.popitem(last=False)
            LOG.debug("spilled figure #%d from RAM (PNG kept)", old_idx)

    def _pixmap_for(self, idx: int) -> Optional[QPixmap]:
        """Return the full-res pixmap for ``idx`` — from RAM if resident,
        otherwise reloaded from the temp PNG (and re-cached)."""
        if idx in self._ram:
            self._ram.move_to_end(idx)   # mark as recently used
            return self._ram[idx]
        path = self._png_paths.get(idx)
        if path and Path(path).is_file():
            pm = QPixmap(path)
            if not pm.isNull():
                self._cache_pixmap(idx, pm)
                return pm
        return None

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < self._count and row != self._current:
            self.show_index(row)

    def _refresh_nav(self) -> None:
        self._pos_label.setText(
            f"{self._current + 1} / {self._count}" if self._count
            else "0 / 0")
        # Figure settings (background/text colour + size) restyle the figure
        # and re-render, so they apply in both PNG and PDF mode — show whenever
        # there's a figure to tweak.
        self._fig_settings_btn.setVisible(self._count > 0)

    def _delete_tempdir(self) -> None:
        if self._tempdir is not None:
            try:
                shutil.rmtree(self._tempdir, ignore_errors=True)
            except Exception:
                pass
            self._tempdir = None

    # -- lifecycle ---------------------------------------------------------

    def closeEvent(self, event):
        self._delete_tempdir()
        super().closeEvent(event)

    def __del__(self):
        # Best-effort temp cleanup if the widget is GC'd without close.
        try:
            self._delete_tempdir()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Figure settings dialog — restyle a matplotlib figure (PDF/vector mode)
# ---------------------------------------------------------------------------

class _FigureSettingsDialog(QDialog):
    """Adjust a figure's background colour, text colour and text size, then
    re-render. Only offered for vector (PDF) figures."""

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self._fig = fig
        self.setWindowTitle("Figure settings")
        from PySide6.QtWidgets import (
            QFormLayout, QDialogButtonBox, QSpinBox, QPushButton as _QPB)
        form = QFormLayout(self)

        try:
            from ..preferences import get_figure_colors, get_figure_text_size
            self._bg, self._fg = get_figure_colors()
            _init_size = get_figure_text_size() or 10
        except Exception:
            self._bg, self._fg, _init_size = "#ffffff", "#000000", 10
        self._bg_btn = _QPB("Background…")
        self._bg_btn.clicked.connect(lambda: self._pick("_bg", self._bg_btn))
        self._fg_btn = _QPB("Text colour…")
        self._fg_btn.clicked.connect(lambda: self._pick("_fg", self._fg_btn))
        form.addRow("Background", self._bg_btn)
        form.addRow("Text colour", self._fg_btn)

        self._bg_btn.setStyleSheet(f"background-color: {self._bg};")
        self._fg_btn.setStyleSheet(f"background-color: {self._fg};")

        self._size = QSpinBox()
        self._size.setRange(4, 48)
        self._size.setValue(int(_init_size))
        form.addRow("Text size", self._size)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _pick(self, attr, btn):
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        c = QColorDialog.getColor(QColor(getattr(self, attr)), self)
        if c.isValid():
            setattr(self, attr, c.name())
            btn.setStyleSheet(f"background-color: {c.name()};")

    def _apply_and_accept(self):
        """Persist the chosen colours/size (so every figure follows them) and
        apply them to this figure, then accept — the caller re-renders."""
        size = int(self._size.value())
        try:
            from ..preferences import set_figure_colors, set_figure_text_size
            set_figure_colors(self._bg, self._fg)
            set_figure_text_size(size)
        except Exception:
            pass
        FigureQueue._style_figure(self._fig, self._bg, self._fg, size)
        self.accept()
