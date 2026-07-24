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
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
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

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev", self)
        self._prev_btn.clicked.connect(self.show_prev)
        self._next_btn = QPushButton("Next ▶", self)
        self._next_btn.clicked.connect(self.show_next)
        self._pos_label = QLabel("0 / 0", self)
        self._pos_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._pos_label, 1)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)
        self._refresh_nav()

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
        self._count = 0
        self._current = -1
        self._view.set_pixmap(QPixmap())
        self._delete_tempdir()
        self._refresh_nav()

    # -- internals ---------------------------------------------------------

    def _render_figure(self, fig, png_path: Path) -> Optional[QPixmap]:
        """Save ``fig`` to ``png_path`` and return a QPixmap of it."""
        try:
            fig.savefig(str(png_path), dpi=110,
                          bbox_inches="tight",
                          facecolor=fig.get_facecolor())
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
        self._prev_btn.setEnabled(self._current > 0)
        self._next_btn.setEnabled(0 <= self._current < self._count - 1)

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
