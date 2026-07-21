"""
Pure-Python backend for the Qt annotate screen.

The image-processing pipeline (normalize / channel-filter / outline /
colored border) and the SQLite-backed page fetch + background save
worker are all Tk-free. The Qt screen wraps this with a QWidget UI.

Semantics mirror `spacr.gui_elements.AnnotateApp` so annotations made in
either GUI are read/written the same way from the same
`measurements/measurements.db`.
"""
from __future__ import annotations

import colorsys
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity


# ---------------------------------------------------------------------------
# Color helpers (identical to AnnotateApp._int_to_color / _label_to_color)
# ---------------------------------------------------------------------------

_PHI = 0.618033988749895


def label_to_hex(val: Optional[int]) -> Optional[str]:
    """Map an annotation value to a hex border color.

    None / 0 / non-int -> None (no border).
    1 -> blue (#4A9EFF-ish), 2 -> red, 3+ -> golden-ratio hue rotation.
    """
    try:
        v = int(val)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    if v == 1:
        return "#3ea6ff"
    if v == 2:
        return "#ff5252"
    h = (v * _PHI) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return "#{:02x}{:02x}{:02x}".format(int(r*255+0.5), int(g*255+0.5), int(b*255+0.5))


# ---------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------

def normalize_pil(
    img: Image.Image,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    normalize_channels: Optional[Iterable[str]] = None,
) -> Image.Image:
    """Normalize the given PIL image per-channel using percentile stretch.

    If `normalize_channels` is None or empty, the image is returned unchanged
    (aside from clipping to 8-bit range).
    """
    arr = np.array(img)
    arr = np.clip(arr, 0, 255)
    if not normalize_channels:
        return Image.fromarray(arr.astype("uint8"))
    if arr.ndim == 2:
        p_lo, p_hi = np.percentile(arr, percentiles)
        out = rescale_intensity(arr, in_range=(p_lo, p_hi), out_range=(0, 255))
        return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))
    channel_map = {"r": 0, "g": 1, "b": 2}
    out = arr.astype(np.float32).copy()
    for ch in normalize_channels:
        idx = channel_map.get(str(ch).lower())
        if idx is None or idx >= out.shape[2]:
            continue
        p_lo, p_hi = np.percentile(out[:, :, idx], percentiles)
        out[:, :, idx] = rescale_intensity(
            out[:, :, idx], in_range=(p_lo, p_hi), out_range=(0, 255)
        )
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def filter_channels_pil(
    img: Image.Image, channels: Optional[Iterable[str]] = None
) -> Image.Image:
    """Zero out channels not present in `channels` (e.g. ['r','g'])."""
    r, g, b = img.split()
    if channels:
        chset = {str(c).strip().lower() for c in channels if c is not None and str(c).strip()}
        if "r" not in chset:
            r = r.point(lambda _: 0)
        if "g" not in chset:
            g = g.point(lambda _: 0)
        if "b" not in chset:
            b = b.point(lambda _: 0)
    return Image.merge("RGB", (r, g, b))


def add_colored_border(img: Image.Image, width: int, color: str) -> Image.Image:
    """Return `img` with an inset colored border of `width` px."""
    bordered = Image.new("RGB",
                          (img.width + 2 * width, img.height + 2 * width),
                          color="black")
    top = Image.new("RGB", (img.width, width), color=color)
    left = Image.new("RGB", (width, img.height), color=color)
    bordered.paste(top, (width, 0))
    bordered.paste(top, (width, img.height + width))
    bordered.paste(left, (0, width))
    bordered.paste(left, (img.width + width, width))
    bordered.paste(img, (width, width))
    return bordered


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class AnnotateSettings:
    src: str = ""
    db_path: str = ""
    annotation_column: str = "annotate"
    image_size: Tuple[int, int] = (200, 200)
    image_type: Optional[str] = None
    channels: Optional[List[str]] = None
    percentiles: Tuple[float, float] = (1.0, 99.0)
    normalize_channels: Optional[List[str]] = None
    measurement: Optional[Any] = None
    threshold: Optional[Any] = None
    threshold_direction: Optional[Any] = None
    outline: Optional[List[str]] = None
    outline_threshold_factor: float = 1.0
    outline_sigma: float = 1.0
    edge_thickness: float = 1.0
    edge_transparency: float = 100.0
    edge_image: bool = False
    object_size: Tuple[int, int] = (0, 0)
    grid_rows: int = 5
    grid_cols: int = 5

    @property
    def page_size(self) -> int:
        return max(1, self.grid_rows * self.grid_cols)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def ensure_annotation_column(db_path: str, column: str) -> None:
    """Add `column` INTEGER to `png_list` if missing and index png_path."""
    if not column or not os.path.isfile(db_path):
        return
    safe = column.replace('"', '""')
    with sqlite3.connect(db_path, timeout=30) as conn:
        cur = conn.cursor()
        cur.execute('PRAGMA table_info("png_list")')
        cols = {row[1] for row in cur.fetchall()}
        if column not in cols:
            try:
                cur.execute(f'ALTER TABLE "png_list" ADD COLUMN "{safe}" INTEGER')
            except sqlite3.OperationalError:
                pass
        try:
            cur.execute('CREATE INDEX IF NOT EXISTS idx_png_path ON "png_list" (png_path)')
        except sqlite3.OperationalError:
            pass


def count_rows(db_path: str, image_type: Optional[str] = None) -> int:
    if not os.path.isfile(db_path):
        return 0
    with sqlite3.connect(db_path, timeout=30) as conn:
        cur = conn.cursor()
        if image_type:
            cur.execute(
                'SELECT COUNT(*) FROM "png_list" WHERE png_path LIKE ?',
                (f"%{image_type}%",),
            )
        else:
            cur.execute('SELECT COUNT(*) FROM "png_list"')
        return int(cur.fetchone()[0])


def fetch_page(
    db_path: str,
    annotation_column: str,
    offset: int,
    page_size: int,
    image_type: Optional[str] = None,
) -> List[Tuple[str, Optional[int]]]:
    """Read one page of (png_path, annotation) rows in insertion order."""
    if not os.path.isfile(db_path):
        return []
    col = (annotation_column or "").replace('"', '""')
    with sqlite3.connect(db_path, timeout=30) as conn:
        cur = conn.cursor()
        if image_type:
            cur.execute(
                f'SELECT png_path, "{col}" FROM "png_list" '
                f'WHERE png_path LIKE ? LIMIT ? OFFSET ?',
                (f"%{image_type}%", page_size, offset),
            )
        else:
            cur.execute(
                f'SELECT png_path, "{col}" FROM "png_list" LIMIT ? OFFSET ?',
                (page_size, offset),
            )
        return cur.fetchall()


def class_counts(db_path: str, annotation_column: str) -> List[Tuple[int, int]]:
    """Return sorted list of (class_value, count) for annotated rows."""
    if not os.path.isfile(db_path):
        return []
    col = (annotation_column or "").replace('"', '""')
    with sqlite3.connect(db_path, timeout=30) as conn:
        cur = conn.cursor()
        cur.execute(
            f'SELECT "{col}" AS cls, COUNT(*) '
            f'FROM "png_list" WHERE "{col}" IS NOT NULL '
            f'GROUP BY "{col}" ORDER BY 1'
        )
        return [(int(r[0]), int(r[1])) for r in cur.fetchall() if r[0] is not None]


def clear_column(db_path: str, annotation_column: str) -> None:
    if not os.path.isfile(db_path):
        return
    col = (annotation_column or "").replace('"', '""')
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute(f'UPDATE "png_list" SET "{col}" = NULL')


def find_last_annotated_offset(
    db_path: str,
    annotation_column: str,
    page_size: int,
    image_type: Optional[str] = None,
) -> Optional[int]:
    """Return the page-aligned offset of the last annotated row, or None."""
    if not os.path.isfile(db_path):
        return None
    col = (annotation_column or "").replace('"', '""')
    with sqlite3.connect(db_path, timeout=30) as conn:
        cur = conn.cursor()
        if image_type:
            cur.execute(
                f'SELECT "{col}" FROM "png_list" WHERE png_path LIKE ?',
                (f"%{image_type}%",),
            )
        else:
            cur.execute(f'SELECT "{col}" FROM "png_list"')
        rows = cur.fetchall()
    last = None
    for i, (val,) in enumerate(rows):
        if val is not None and val != 0:
            last = i
    if last is None:
        return None
    return (last // page_size) * page_size


# ---------------------------------------------------------------------------
# Background save worker (thread-based, mirrors AnnotateApp.update_database_worker)
# ---------------------------------------------------------------------------

class SaveWorker:
    """Runs in a daemon thread; consumes {png_path: annotation} batches
    from a Queue and commits them to the DB in coalesced transactions.
    """
    _SENTINEL = object()

    def __init__(self, db_path: str, annotation_column: str):
        self.db_path = db_path
        self.annotation_column = annotation_column
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._terminate = False
        self._busy = False
        self._pending_batches = 0
        self._last_save_ts: Optional[float] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._terminate = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._terminate = True
        self._q.put(self._SENTINEL)
        if wait and self._thread:
            try:
                self._thread.join(timeout=5.0)
            except Exception:
                pass

    def submit(self, batch: dict) -> None:
        """Enqueue a copy of the batch for saving."""
        if not batch:
            return
        with self._lock:
            self._pending_batches += 1
        self._q.put(dict(batch))

    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def pending_batches(self) -> int:
        with self._lock:
            return self._pending_batches

    @property
    def last_save_ts(self) -> Optional[float]:
        return self._last_save_ts

    # ------------------------------------------------------------------
    def _run(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        cur = conn.cursor()
        try:
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                conn.commit()
            except Exception:
                pass
            col = (self.annotation_column or "").replace('"', '""')
            while True:
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    if self._terminate:
                        break
                    continue
                if item is self._SENTINEL:
                    self._q.task_done()
                    break
                pending = item
                # Coalesce
                while True:
                    try:
                        extra = self._q.get_nowait()
                        if extra is self._SENTINEL:
                            self._q.task_done()
                            self._q.put(self._SENTINEL)
                            break
                        pending.update(extra)
                        with self._lock:
                            self._pending_batches -= 1
                        self._q.task_done()
                    except queue.Empty:
                        break
                self._busy = True
                to_null = [p for p, v in pending.items() if v is None]
                to_set = [(int(v), p) for p, v in pending.items() if v is not None]
                try:
                    if to_null:
                        cur.executemany(
                            f'UPDATE "png_list" SET "{col}" = NULL WHERE png_path = ?',
                            [(p,) for p in to_null],
                        )
                    if to_set:
                        cur.executemany(
                            f'UPDATE "png_list" SET "{col}" = ? WHERE png_path = ?',
                            to_set,
                        )
                    conn.commit()
                finally:
                    with self._lock:
                        self._pending_batches -= 1
                    self._busy = False
                    self._last_save_ts = time.time()
                    self._q.task_done()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()
