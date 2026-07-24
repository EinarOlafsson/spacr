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


_cellpose_outline_model = None


def _get_cellpose_outline_model():
    """Lazily build + cache a small Cellpose (SAM) model for outline masks."""
    global _cellpose_outline_model
    if _cellpose_outline_model is None:
        from cellpose import models as cp_models
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _cellpose_outline_model = cp_models.CellposeModel(
            gpu=gpu, pretrained_model="cpsam", device=None)
    return _cellpose_outline_model


def _cellpose_foreground(channel_2d) -> "np.ndarray":
    """Return a boolean foreground mask for one channel using Cellpose."""
    model = _get_cellpose_outline_model()
    res = model.eval(channel_2d.astype(np.float32),
                     diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
    mask = res[0]
    if isinstance(mask, list):
        mask = mask[0]
    return np.asarray(mask) > 0


def outline_image(
    base_img: Image.Image,
    full_img: Image.Image,
    outline_channels: Optional[Iterable[str]] = None,
    edge_sigma: float = 1.0,
    edge_thickness: float = 1.0,
    edge_transparency: float = 100.0,
    edge_image: bool = False,
    outline_threshold_factor: float = 1.0,
    object_size: Tuple[int, int] = (0, 0),
    outline_method: str = 'otsu',
) -> Image.Image:
    """Overlay per-channel object outlines on `base_img`.

    Mirrors AnnotateApp.outline_image (Tk) semantics: for every channel
    in `outline_channels`, compute an Otsu-thresholded foreground mask
    on the corresponding channel of `full_img`, extract the boundary,
    optionally dilate it, then alpha-blend it over the channel in
    `base_img` with `edge_transparency/100` opacity. Peak-normalized so
    thin edges stay visible.
    """
    if not outline_channels or edge_transparency <= 0:
        return base_img
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
    from skimage.filters import threshold_otsu
    from skimage.morphology import dilation, disk
    from skimage.segmentation import find_boundaries

    channel_map = {"r": 0, "g": 1, "b": 2}
    outline_channels = [ch for ch in outline_channels if ch in channel_map]
    if not outline_channels:
        return base_img
    base_arr = np.asarray(base_img).copy()
    full_arr = np.asarray(full_img)
    if base_arr.ndim != 3 or base_arr.shape[2] != 3:
        return base_img
    if not edge_image:
        for ch in outline_channels:
            base_arr[:, :, channel_map[ch]] = 0
    opacity = max(0.0, min(1.0, float(edge_transparency) / 100.0))
    factor = float(outline_threshold_factor)
    try:
        min_px, max_px = object_size
    except Exception:
        min_px, max_px = (0, 0)
    for ch in outline_channels:
        idx = channel_map[ch]
        if edge_image:
            base_arr[:, :, idx] = full_arr[:, :, idx]
        if outline_method == 'cellpose':
            # Small Cellpose model gives cleaner object outlines than Otsu.
            try:
                fg_mask = _cellpose_foreground(full_arr[:, :, idx])
            except Exception:
                # Fall back to Otsu if cellpose isn't available / fails.
                outline_method = 'otsu'
        if outline_method != 'cellpose':
            ch_sm = gaussian_filter(full_arr[:, :, idx].astype(np.float32),
                                     sigma=float(edge_sigma))
            try:
                otsu = threshold_otsu(ch_sm)
            except Exception:
                otsu = float(np.percentile(ch_sm, 50.0))
            thr = float(min(255.0, max(0.0, otsu * factor)))
            fg_mask = (ch_sm > thr)
            fg_mask = binary_closing(fg_mask, structure=np.ones((3, 3), dtype=bool))
            fg_mask = binary_fill_holes(fg_mask)
        if (min_px and min_px > 0) or (max_px and max_px > 0):
            lbl, n = label(fg_mask)
            if n > 0:
                counts = np.bincount(lbl.ravel())
                lo = int(min_px) if int(min_px) > 0 else 0
                hi = int(max_px) if int(max_px) > 0 else int(counts.max())
                keep = np.zeros_like(counts, dtype=bool)
                for i in range(1, len(counts)):
                    if lo <= counts[i] <= hi:
                        keep[i] = True
                fg_mask = keep[lbl]
        edge = find_boundaries(fg_mask, mode="inner").astype(np.uint8)
        thick = int(max(0, round(edge_thickness))) - 1
        if thick > 0:
            edge = dilation(edge > 0, disk(thick)).astype(np.uint8)
        alpha = np.clip(edge.astype(np.float32) * opacity, 0.0, 1.0)
        orig = base_arr[:, :, idx].astype(np.float32)
        blended = alpha * 255.0 + (1.0 - alpha) * orig
        base_arr[:, :, idx] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(base_arr)


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
    """Every knob the Annotate screen exposes, packed into one dataclass.

    Sensible defaults let callers instantiate ``AnnotateSettings()`` and
    override just the handful of fields they care about.
    """

    src: str = ""
    db_path: str = ""
    annotation_column: str = "annotate"
    image_size: Tuple[int, int] = (200, 200)
    image_type: Optional[str] = None
    # Default to showing + normalising R,G,B so object crops are visible out of
    # the box (unnormalised crops render as near-black/grey otherwise).
    channels: List[str] = field(default_factory=lambda: ["r", "g", "b"])
    percentiles: Tuple[float, float] = (1.0, 99.0)
    normalize_channels: List[str] = field(
        default_factory=lambda: ["r", "g", "b"])
    measurement: Optional[Any] = None
    threshold: Optional[Any] = None
    threshold_direction: Optional[Any] = None
    outline: Optional[List[str]] = None
    outline_method: str = "otsu"        # "otsu" | "cellpose"
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
        """Number of thumbnails per page (``grid_rows * grid_cols``, min 1)."""
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
    """Return the number of ``png_list`` rows, optionally filtered by ``image_type``.

    :param db_path: path to ``measurements.db``; missing files count as 0.
    :param image_type: optional substring to filter ``png_path`` on.
    """
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


# ---------------------------------------------------------------------------
# Measurement/threshold filter fetch
#
# The Tk AnnotateApp joins png_list with the other measurement tables via
# spacr.io._read_and_join_tables and applies user-supplied thresholds to
# a numeric column (higher / lower). Here we do the same for one-or-more
# (column, threshold, direction) triples so the settings dialog can filter
# annotation to just objects above/below a cutoff (e.g. cell_area > 500).
# ---------------------------------------------------------------------------

def _apply_threshold(df, column: str, threshold: float, direction: str):
    if column is None or column not in df.columns or threshold is None:
        return df
    if direction == "higher":
        return df[df[column] > float(threshold)]
    if direction == "lower":
        return df[df[column] < float(threshold)]
    return df


def fetch_filtered_paths(
    db_path: str,
    annotation_column: str,
    measurements: List[str],
    thresholds: List[float],
    directions: List[str],
    image_type: Optional[str] = None,
) -> List[Tuple[str, Optional[int]]]:
    """Return ALL (png_path, annotation) rows matching every one of the
    measurement/threshold/direction triples.

    Rows come from a merge of png_list with the measurement tables (via
    spacr.io._read_and_join_tables) — same code path as the Tk app —
    filtered on png_path substring when `image_type` is given.
    Callers paginate the returned list themselves.
    """
    if not os.path.isfile(db_path) or not measurements or not thresholds:
        return []
    from spacr.io import _read_and_join_tables, _read_db
    df = _read_and_join_tables(db_path)
    if "png_path" not in df.columns:
        png_df = _read_db(db_path, tables=["png_list"])[0]
        if "prcfo" not in df.columns and df.index.name == "prcfo":
            df = df.reset_index()
        if "prcfo" not in png_df.columns and png_df.index.name == "prcfo":
            png_df = png_df.reset_index()
        if "prcfo" in df.columns and "prcfo" in png_df.columns:
            df = df.merge(
                png_df[["prcfo", "png_path"]],
                on="prcfo", how="left", suffixes=("", "_dup"),
            )
    if annotation_column not in df.columns:
        df[annotation_column] = None
    if len(thresholds) == 1 and len(measurements) > 1:
        thresholds = [thresholds[0]] * len(measurements)
    if isinstance(directions, str):
        directions = [directions] * len(measurements)
    if len(directions) == 1 and len(measurements) > 1:
        directions = [directions[0]] * len(measurements)
    for col, thr, direction in zip(measurements, thresholds, directions):
        df = _apply_threshold(df, col, thr, direction)
    if "png_path" not in df.columns:
        return []
    df = df.dropna(subset=["png_path"])
    if image_type:
        df = df[df["png_path"].str.contains(image_type)]
    if annotation_column not in df.columns:
        return []
    return df[["png_path", annotation_column]].values.tolist()


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
    """Null every value in ``annotation_column`` of ``png_list``.

    :param db_path: path to ``measurements.db``; missing files are ignored.
    :param annotation_column: column to reset.
    """
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
        """Prepare an idle worker; call :meth:`start` to spawn its thread.

        :param db_path: path to the SQLite ``measurements.db``.
        :param annotation_column: column in ``png_list`` to write into.
        """
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
        """Spawn the daemon writer thread if it isn't already running."""
        if self._thread and self._thread.is_alive():
            return
        self._terminate = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        """Signal the writer to exit; when ``wait`` is True block up to 5 s."""
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
        """True while the writer thread is inside a commit."""
        return self._busy

    @property
    def pending_batches(self) -> int:
        """Number of submitted-but-not-yet-committed batches."""
        with self._lock:
            return self._pending_batches

    @property
    def last_save_ts(self) -> Optional[float]:
        """POSIX timestamp of the most recent successful commit, or ``None``."""
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
