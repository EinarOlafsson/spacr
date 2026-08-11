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
import contextlib
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity

from spacr.database_concurrency import (
    connect as connect_database,
    transaction,
)


LOG = logging.getLogger("spacr.qt.annotate_engine")


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

def load_crop_image(path: str, db_path: Optional[str] = None,
                    stored_channel_order: str = "auto") -> Image.Image:
    """Open one object crop PNG as an 8-bit RGB image, in the corrected order.

    Not ``Image.open(path).convert('RGB')``. Crop PNGs come in two formats:
    anything spaCR wrote before the BGR fix has ``png_dims[0]`` in its *blue*
    channel, so a plain PIL read shows the user's first stain as blue and
    their third as red -- and the annotator's "r"/"g"/"b" channel filters then
    address the wrong stains. :func:`spacr.crops.read_crop_png` resolves which
    format the folder is in (sidecar marker, else the database column, else
    legacy) and corrects it on load, so an old dataset and a new one look the
    same here.

    It also fixes the other half: a 16-bit single-channel crop opened with
    ``convert('RGB')`` is CLIPPED at 255 by PIL and comes back solid white.
    Every crop is narrowed the same way now -- by its high byte.

    :param path: the crop PNG.
    :param db_path: optional ``measurements.db``, consulted when the crop
        folder carries no sidecar marker.
    :returns: PIL ``Image`` in RGB mode.
    """
    from ..crops import (
        CROP_FORMAT_CURRENT,
        CROP_FORMAT_RGB,
        read_crop_png,
    )

    # This vocabulary is the ANNOTATOR's, not `spacr.crops`'s: it says what a
    # file's slots hold, not which numbered format wrote them. "rgb" means the
    # slots are already the declared colours -- the current format 3, which is
    # read as-is; formats 1 and 3 hold identical bytes, so this covers unmarked
    # legacy crops too. "legacy_bgr" means the slots are the other way round,
    # which is the eleven-day format 2 -- the only one `read_crop_png` reverses.
    order = str(stored_channel_order or "auto").strip().lower()
    if order == "rgb":
        stored_format = CROP_FORMAT_CURRENT
    elif order in {"bgr", "legacy_bgr"}:
        stored_format = CROP_FORMAT_RGB
    elif order == "auto":
        stored_format = None
    else:
        raise ValueError(
            "stored_channel_order must be 'rgb', 'auto', or 'legacy_bgr'")
    return Image.fromarray(
        read_crop_png(path, fmt=stored_format, db_path=db_path))


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
# Cellpose/PyTorch model construction and inference enter native code and are
# not safe to run concurrently through one cached model.  Annotate page loads
# used to fan out across several QThreads and ThreadPoolExecutors, so two crops
# could call ``model.eval`` at the same time and take the interpreter down
# without a Python traceback.  RLock lets _cellpose_foreground call the
# separately-tested lazy constructor while holding the same guard.
_cellpose_outline_lock = threading.RLock()


def _get_cellpose_outline_model():
    """Lazily build + cache a small Cellpose (SAM) model for outline masks."""
    global _cellpose_outline_model
    with _cellpose_outline_lock:
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
    with _cellpose_outline_lock:
        model = _get_cellpose_outline_model()
        res = model.eval(
            channel_2d.astype(np.float32),
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
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
    """Return `img` with an inset colored border of `width` px.

    Kept for parity with the Tk ``AnnotateApp`` (and for callers that want a
    bordered image out of the pipeline). The Qt grid does NOT use it: its
    tiles paint their borders in ``_Thumbnail.paintEvent`` so recolouring
    one costs a repaint instead of a rebuilt pixmap.
    """
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
    # Arrays and PIL/Qt images are always RGB after decode. Explicit RGB is
    # the safe default for standard PNGs; Auto consults spaCR's format marker,
    # and Legacy BGR remains available for old unmarked cv2-written crops.
    stored_channel_order: str = "rgb"  # rgb | auto | legacy_bgr
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
    # Active-learning queue (spacr.active_learning). Off by default: it needs
    # model scores in png_list, which only exist after a classifier has run.
    queue_by_uncertainty: bool = False
    queue_measure: str = "entropy"      # entropy | least_confidence | margin
    queue_diversity: str = "well"       # well | field | plate | none
    queue_limit: int = 0                # 0 = the whole unlabelled pool
    # 'auto' | 'png' | 'merged' -- see spacr.crops.resolve_crop_source.
    # 'auto' prefers the PNG folder, so existing projects are unaffected.
    crop_source: str = "auto"

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
    conn = connect_database(db_path, timeout=30)
    try:
        cur = conn.cursor()
        with transaction(conn):
            cur.execute('PRAGMA table_info("png_list")')
            cols = {row[1] for row in cur.fetchall()}
            if column not in cols:
                cur.execute(f'ALTER TABLE "png_list" ADD COLUMN "{safe}" INTEGER')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_png_path ON "png_list" (png_path)')
    finally:
        conn.close()


def count_rows(db_path: str, image_type: Optional[str] = None) -> int:
    """Return the number of ``png_list`` rows, optionally filtered by ``image_type``.

    :param db_path: path to ``measurements.db``; missing files count as 0.
    :param image_type: optional substring to filter ``png_path`` on.
    """
    if not os.path.isfile(db_path):
        return 0
    with contextlib.closing(
        connect_database(db_path, readonly=True, timeout=30)
    ) as conn:
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
    with contextlib.closing(
        connect_database(db_path, readonly=True, timeout=30)
    ) as conn:
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
            # one_to_one: 'prcfo' is the object key, unique in the measurement
            # join and in png_list alike. A repeated key on either side would
            # silently multiply the annotation grid's rows and show the same
            # cell several times under different crops.
            df = df.merge(
                png_df[["prcfo", "png_path"]],
                on="prcfo", how="left", suffixes=("", "_dup"),
                validate="one_to_one",
            )
    if annotation_column not in df.columns:
        df[annotation_column] = None
    if len(thresholds) == 1 and len(measurements) > 1:
        thresholds = [thresholds[0]] * len(measurements)
    if isinstance(directions, str):
        directions = [directions] * len(measurements)
    if len(directions) == 1 and len(measurements) > 1:
        directions = [directions[0]] * len(measurements)
    # REFUSE A LENGTH MISMATCH rather than let zip() truncate it.
    #
    # The broadcasts above cover the documented shorthand -- one threshold
    # applied to every column -- and must stay. What they do not cover is
    # "three columns, two thresholds", which fell through to the zip below
    # and silently dropped the third filter. Both fields in the Annotate
    # settings dialog are free-text comma-separated line edits, so the two
    # lists disagreeing is a typo away.
    #
    # There is no defensible pairing to guess: recycling, padding with the
    # last value and dropping the tail are all equally arbitrary. And the
    # consequence is not a crash but a plausible-looking WRONG POPULATION
    # that gets hand-labelled and fed to a classifier, so failing loudly is
    # cheaper than being approximately right.
    if len(thresholds) != len(measurements) or len(directions) != len(measurements):
        raise ValueError(
            f"{len(measurements)} measurement column(s) but "
            f"{len(thresholds)} threshold(s) and {len(directions)} "
            f"direction(s): give one of each per measurement, or a single "
            f"threshold and direction to apply to all of them.")
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



# ---------------------------------------------------------------------------
# Auto-annotation
#
# Four ways to pick a population, ONE way to write it. The write path is
# `SaveWorker` -- the annotator's existing batched writer -- because a second
# sqlite writer on measurements.db is a known hazard (spacr.database_
# concurrency), and because going through it means bulk annotations land in
# the same place, in the same order, as the ones made by hand.
#
# Two of the four sources are not implemented here on purpose. The Gate
# Editor and the Image UMAP already select populations and already write
# annotations; duplicating either would mean two implementations of the same
# gate maths drifting apart. What was missing was the ROUTE from them into an
# annotation column, and that is what `gate_paths` and the UMAP hand-off
# provide.
# ---------------------------------------------------------------------------

#: png_list columns that describe where an object came from, offered as the
#: metadata source. `label` is deliberately absent: it is the object's id
#: within its field, not a property anyone annotates by.
METADATA_COLUMNS: Tuple[str, ...] = (
    "plateID", "wellID", "rowID", "columnID", "fieldID", "timeID",
)


def metadata_values(db_path: str, column: str) -> List[str]:
    """The distinct values of one png_list metadata column, sorted.

    Read from the database rather than guessed from a naming convention:
    plates are named by whoever ran them, and a picker offering rows A-H to
    someone whose plate is numbered is a picker they cannot use.

    :param db_path: the measurements database.
    :param column: one of :data:`METADATA_COLUMNS`.
    :returns: the distinct values, as strings, sorted; empty when the column
        or the database is missing.
    :raises ValueError: a column outside METADATA_COLUMNS, which would
        otherwise interpolate an arbitrary name into SQL.
    """
    if column not in METADATA_COLUMNS:
        raise ValueError(
            f"{column!r} is not a metadata column; expected one of "
            f"{list(METADATA_COLUMNS)}")
    if not os.path.isfile(db_path):
        return []
    with contextlib.closing(
        connect_database(db_path, readonly=True, timeout=30)
    ) as conn:
        cur = conn.cursor()
        cur.execute('PRAGMA table_info("png_list")')
        if column not in {row[1] for row in cur.fetchall()}:
            return []
        cur.execute(
            f'SELECT DISTINCT "{column}" FROM "png_list" '
            f'WHERE "{column}" IS NOT NULL')
        return sorted(str(row[0]) for row in cur.fetchall())


def paths_by_metadata(db_path: str, column: str,
                      values: Sequence[str]) -> List[str]:
    """png_paths whose ``column`` is one of ``values``.

    :param db_path: the measurements database.
    :param column: one of :data:`METADATA_COLUMNS`.
    :param values: the values to select.
    :returns: matching png_path strings.
    :raises ValueError: a column outside METADATA_COLUMNS.
    """
    if column not in METADATA_COLUMNS:
        raise ValueError(
            f"{column!r} is not a metadata column; expected one of "
            f"{list(METADATA_COLUMNS)}")
    if not os.path.isfile(db_path) or not values:
        return []
    wanted = [str(v) for v in values]
    placeholders = ",".join("?" for _ in wanted)
    with contextlib.closing(
        connect_database(db_path, readonly=True, timeout=30)
    ) as conn:
        cur = conn.cursor()
        cur.execute('PRAGMA table_info("png_list")')
        if column not in {row[1] for row in cur.fetchall()}:
            return []
        # CAST so a numeric columnID matches the strings the picker offers.
        cur.execute(
            f'SELECT png_path FROM "png_list" '
            f'WHERE CAST("{column}" AS TEXT) IN ({placeholders})', wanted)
        return [row[0] for row in cur.fetchall()]


def paths_by_measurements(db_path: str, annotation_column: str,
                          rules: Sequence[Mapping[str, Any]]) -> List[str]:
    """png_paths satisfying EVERY ``{column, threshold, direction}`` rule.

    Several measurements at once is the point: one threshold is a gate, not a
    population. The rules are ANDed, which is what
    :func:`fetch_filtered_paths` already does for the settings-panel filter --
    reused here rather than re-derived, so the auto-annotator and the filter
    can never disagree about what a threshold means.

    :param db_path: the measurements database.
    :param annotation_column: the column being written (needed by the join).
    :param rules: mappings with ``column``, ``threshold`` and ``direction``
        (``'higher'`` or ``'lower'``).
    :returns: matching png_path strings.
    :raises ValueError: a rule missing a field, or an unknown direction.
    """
    if not rules:
        return []
    columns, thresholds, directions = [], [], []
    for rule in rules:
        column = rule.get("column")
        threshold = rule.get("threshold")
        direction = str(rule.get("direction", "higher")).lower()
        if not column or threshold is None:
            raise ValueError(
                f"every measurement rule needs a column and a threshold: "
                f"{dict(rule)!r}")
        if direction not in ("higher", "lower"):
            raise ValueError(
                f"direction must be 'higher' or 'lower', got {direction!r}")
        columns.append(str(column))
        thresholds.append(float(threshold))
        directions.append(direction)
    rows = fetch_filtered_paths(
        db_path, annotation_column, columns, thresholds, directions)
    return [path for path, _ in rows]


def gate_paths(db_path: str, gates: Sequence[Any]) -> List[str]:
    """png_paths surviving a chain of :class:`spacr.qt.widgets.gate_spec.Gate`.

    The route the Gate Editor was missing. The gate maths is NOT reproduced
    here -- ``GateClause`` evaluates the chain, exactly as it does when the
    same gates filter a plot, so a population gated on screen and a
    population annotated from it are the same population by construction.

    :param db_path: the measurements database.
    :param gates: the gate chain, outermost first.
    :returns: matching png_path strings.
    """
    if not gates:
        return []
    from spacr.io import _read_and_join_tables, _read_db
    from .widgets.gate_spec import GateClause

    frame = _read_and_join_tables(db_path)
    if "png_path" not in frame.columns:
        png_df = _read_db(db_path, tables=["png_list"])[0]
        if "prcfo" not in frame.columns and frame.index.name == "prcfo":
            frame = frame.reset_index()
        if "prcfo" not in png_df.columns and png_df.index.name == "prcfo":
            png_df = png_df.reset_index()
        if "prcfo" in frame.columns and "prcfo" in png_df.columns:
            frame = frame.merge(png_df[["prcfo", "png_path"]], on="prcfo",
                                how="left", validate="one_to_one")
    if "png_path" not in frame.columns:
        return []
    keep = GateClause(tuple(gates)).mask(frame)
    return frame.loc[keep, "png_path"].dropna().astype(str).tolist()


def annotation_batch(paths: Iterable[str],
                     value: Optional[int]) -> Dict[str, Optional[int]]:
    """Turn a path list into the batch :meth:`SaveWorker.submit` takes.

    Trivial, and it exists so every auto-annotation source ends at the same
    call. ``None`` clears, exactly as it does for a keystroke.

    :param paths: png_paths to label.
    :param value: the class number, or None to clear.
    :returns: ``{png_path: value}``.
    """
    return {str(path): value for path in paths}


def class_counts(db_path: str, annotation_column: str) -> List[Tuple[int, int]]:
    """Return sorted list of (class_value, count) for annotated rows."""
    if not os.path.isfile(db_path):
        return []
    col = (annotation_column or "").replace('"', '""')
    with contextlib.closing(
        connect_database(db_path, readonly=True, timeout=30)
    ) as conn:
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
    conn = connect_database(db_path, timeout=30)
    try:
        with transaction(conn):
            conn.execute(f'UPDATE "png_list" SET "{col}" = NULL')
    finally:
        conn.close()


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
    with contextlib.closing(
        connect_database(db_path, readonly=True, timeout=30)
    ) as conn:
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
        self._last_error: Optional[str] = None
        self._failed_batch: Optional[dict] = None
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
        """Drain queued writes and stop the writer.

        A bounded five-second join used to let the screen disappear while the
        daemon thread still owned a live SQLite connection.  That is unsafe at
        application shutdown: CPython can finalize the sqlite extension while
        the thread is still inside it.  SQLite already bounds lock waits with
        its 30-second connection timeout, so a requested blocking stop waits
        for the thread to close its cursor and connection completely.
        """
        with self._lock:
            first_stop = not self._terminate
            self._terminate = True
        if first_stop:
            self._q.put(self._SENTINEL)
        if wait and self._thread:
            try:
                self._thread.join()
            except Exception:
                pass

    @property
    def is_alive(self) -> bool:
        """Whether the SQLite writer thread is still running."""
        return bool(self._thread and self._thread.is_alive())

    def submit(self, batch: dict) -> None:
        """Enqueue a copy of the batch for saving."""
        if not batch:
            return
        with self._lock:
            if self._last_error is not None:
                # Retain edits made while the screen is reporting a failed
                # writer. They are not called saved and are never discarded
                # from the worker's state.
                if self._failed_batch is None:
                    self._failed_batch = {}
                    self._pending_batches += 1
                self._failed_batch.update(batch)
                return
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

    @property
    def last_error(self) -> Optional[str]:
        """Actionable message for the latest writer failure, if any."""
        with self._lock:
            return self._last_error

    # ------------------------------------------------------------------
    def _run(self) -> None:
        conn = None
        cur = None
        try:
            # Preserve the database's journal mode. Enabling WAL blindly is
            # unsafe for projects on many NAS/NFS mounts.
            conn = connect_database(self.db_path, timeout=30)
            cur = conn.cursor()
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
                try:
                    to_null = [p for p, v in pending.items() if v is None]
                    to_set = [
                        (int(v), p) for p, v in pending.items()
                        if v is not None
                    ]
                    with transaction(conn):
                        if to_null:
                            cur.executemany(
                                f'UPDATE "png_list" SET "{col}" = NULL '
                                'WHERE png_path = ?',
                                [(p,) for p in to_null],
                            )
                        if to_set:
                            cur.executemany(
                                f'UPDATE "png_list" SET "{col}" = ? '
                                'WHERE png_path = ?',
                                to_set,
                            )
                except BaseException as exc:
                    with self._lock:
                        self._last_error = (
                            f"{type(exc).__name__}: {exc}. Annotations were "
                            "not saved; resolve the database problem before "
                            "closing this module.")
                        self._failed_batch = pending
                    self._busy = False
                    LOG.exception(
                        "Annotate database save failed for %s; the transaction "
                        "was rolled back and the batch remains unsaved",
                        self.db_path,
                    )
                    self._q.task_done()
                    break
                else:
                    with self._lock:
                        self._pending_batches -= 1
                    self._last_save_ts = time.time()
                    self._busy = False
                    self._q.task_done()
        except BaseException as exc:
            with self._lock:
                if self._last_error is None:
                    self._last_error = (
                        f"{type(exc).__name__}: {exc}. The annotation "
                        "database writer could not start.")
            LOG.exception(
                "Annotate database writer stopped before saving queued edits")
        finally:
            if cur is not None:
                try:
                    cur.close()
                except sqlite3.Error:
                    pass
            if conn is not None:
                conn.close()
