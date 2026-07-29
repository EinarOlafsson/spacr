"""Timelapse live preview — segment once, re-link live.

The Timelapse module is mask generation over a time series followed by
frame-to-frame linking. Those two halves cost wildly different amounts:
segmenting twelve frames with Cellpose is tens of seconds, re-linking the
*same* twelve label images with a new ``timelapse_displacement`` is
milliseconds. A preview that re-segments on every slider move is unusable,
so this panel splits them:

* **Per-frame masks are cached** under a *segmentation signature* — the
  source path, the frame indices, and every setting that can change a
  label image (model, channel, diameter, flow threshold, cell probability,
  normalisation). Change a *tracking* setting and the signature is
  unchanged, the cache hits, and only :func:`link_tracks` runs. Change a
  segmentation setting and the signature moves, so the masks are rebuilt.
* **The sequence is read lazily.** :class:`FrameSequence` never
  materialises the whole time series: a directory reads one file at a
  time, a multi-page TIFF reads one page at a time, and an ``.npy`` stack
  is memory-mapped and sliced. Only the frames the preview actually shows
  are touched, and a small LRU keeps the scrubber responsive without
  pinning the movie in RAM.

What the panel *shows* is chosen around the two failure modes people
actually tune a tracker against:

* **Fragmentation** — one object becoming several track ids. Surfaced as
  track count, mean/median track length, the number of tracks shorter
  than a live-settable N, and the count of tracks that start after the
  first frame or end before the last one.
* **Identity swaps** — a track jumping to a different object. Surfaced as
  the number of within-track steps longer than the displacement limit the
  user is tuning, plus the largest single step. Masks are relabelled *by
  track id* and drawn in a per-track colour, so a swap is visible as an
  object changing colour mid-movie.

Both indicators are computed without ground truth and are labelled in the
UI as indicators, not measurements.

Optional backends (``trackastra``, ``ultrack``) are detected before they
are called: absent, the panel writes one inline line naming the package
and the install command. An ``ImportError`` traceback never reaches the
user.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

# Reuse the Mask live preview's rendering + canvas primitives wholesale so
# the two panels behave identically from the user's side: the same zoom/pan
# pair, the same percentile stretch, the same boundary drawing.
from .live_preview import (
    _boundary_mask,
    _select_channel,
    _to_uint8,
    _ZoomView,
    numpy_to_qpixmap,
)

LOG = logging.getLogger("spacr.qt.timelapse_preview")

FRAME_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy")

#: Linking backends, in the order ``timelapse_mode`` documents them.
TRACK_MODES = ("trackastra", "ultrack", "trackpy", "btrack", "iou")

#: Backends that live behind an optional dependency, and the pip target that
#: installs them. Consulted before any import so the panel can report a
#: missing package as one inline line instead of an ImportError traceback.
OPTIONAL_BACKENDS: Dict[str, str] = {
    "trackastra": "pip install trackastra",
    "ultrack": "pip install spacr[ultrack]",
    "btrack": "pip install btrack",
}

#: Distinct track colours, cycled by track id. Chosen to stay separable on
#: a dark micrograph and against each other.
TRACK_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (32, 220, 32), (222, 82, 200), (32, 200, 220), (255, 220, 32),
    (240, 100, 60), (120, 140, 255), (255, 150, 200), (150, 255, 190),
    (255, 190, 110), (180, 120, 240), (110, 220, 255), (220, 220, 220),
)


class TrackerUnavailable(RuntimeError):
    """A linking backend cannot run here, with an actionable reason.

    Raised instead of letting an ``ImportError`` escape so the panel can put
    the message inline. The string always names the package and the command
    that fixes it.
    """


# ---------------------------------------------------------------------------
# Lazy, bounded frame access
# ---------------------------------------------------------------------------

def _natural_key(name: str):
    """Sort key that orders ``f_2`` before ``f_10`` (frame indices are numbers)."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", str(name))]


class FrameSequence:
    """A time series read one frame at a time.

    Three layouts are understood, and none of them is ever read whole:

    * a **directory** of per-frame image files — one file opened per access;
    * a **multi-page TIFF** — one page decoded per access
      (``tifffile.imread(path, key=i)``);
    * an **``.npy`` stack** whose first axis is time — memory-mapped, so a
      slice touches only that plane's pages.

    :param kind: ``"files"``, ``"tiff"`` or ``"npy"``.
    :param source: the list of paths (``files``) or the single path.
    :param n_available: how many frames exist on disk.
    :param indices: the subset of frame indices this sequence exposes, so a
        400-frame movie can be previewed as its first 12 frames.
    :param cache_size: how many decoded frames to keep in the LRU.
    :ivar read_count: number of decodes actually performed — the instrument
        the tests assert against to prove nothing is read eagerly.
    """

    def __init__(self, kind: str, source, n_available: int,
                 indices: Sequence[int], label: str = "", cache_size: int = 6):
        self.kind = kind
        self.source = source
        self.n_available = int(n_available)
        self.indices: List[int] = list(indices)
        self.label = label or str(source)
        self._cache: "dict[int, np.ndarray]" = {}
        self._cache_order: List[int] = []
        self._cache_size = max(1, int(cache_size))
        self._memmap = None
        self.read_count = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls, path, max_frames: int = 12) -> "FrameSequence":
        """Open ``path`` as a sequence, reading at most metadata to do so.

        :param path: a directory of frames, a multi-page TIFF, or an ``.npy``
            stack whose first axis is time.
        :param max_frames: cap on the number of frames the preview exposes.
        :raises FileNotFoundError: if the path does not exist.
        :raises ValueError: if the path holds no usable time series.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No such file or directory: {p}")
        cap = max(1, int(max_frames))

        if p.is_dir():
            files = sorted(
                (f for f in p.iterdir()
                 if f.is_file() and f.suffix.lower() in FRAME_SUFFIXES),
                key=lambda f: _natural_key(f.name),
            )
            if not files:
                raise ValueError(
                    f"{p.name} holds no {'/'.join(FRAME_SUFFIXES)} frames.")
            if len(files) < 2:
                raise ValueError(
                    f"{p.name} holds a single frame — a timelapse preview "
                    "needs at least two.")
            idx = list(range(min(len(files), cap)))
            return cls("files", files, len(files), idx, label=str(p))

        suf = p.suffix.lower()
        if suf == ".npy":
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim < 3:
                raise ValueError(
                    f"{p.name} has shape {arr.shape}; a timelapse stack needs "
                    "(T, H, W) or (T, H, W, C).")
            n = int(arr.shape[0])
            if n < 2:
                raise ValueError(
                    f"{p.name} has {n} frame(s) — a timelapse preview needs "
                    "at least two.")
            return cls("npy", arr, n, list(range(min(n, cap))), label=str(p))

        if suf in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(str(p)) as tf:
                n_pages = len(tf.pages)
                try:
                    shape = tuple(tf.series[0].shape)
                except Exception:
                    shape = tuple(tf.pages[0].shape)
            # A time series can be one page per frame (page-addressable, the
            # cheapest read) or a single page holding a 3-D array, which is
            # what tifffile writes for a plain (T, H, W) save. The second form
            # is not page-addressable, so it is memory-mapped instead — still
            # lazy, just at the OS page level rather than the TIFF page level.
            if n_pages > 1:
                kind, n = "tiff", n_pages
            else:
                kind = "tiffmm"
                n = int(shape[0]) if len(shape) >= 3 else 1
            if n < 2:
                raise ValueError(
                    f"{p.name} holds {n} frame(s) — a timelapse preview needs "
                    "a multi-frame TIFF or a folder of frames.")
            return cls(kind, p, n, list(range(min(n, cap))), label=str(p))

        raise ValueError(
            f"{p.name}: unsupported input. Drop a folder of frames, a "
            "multi-page TIFF, or a (T, H, W) .npy stack.")

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def truncated(self) -> bool:
        """Whether the preview is showing fewer frames than exist on disk."""
        return len(self.indices) < self.n_available

    def frame(self, i: int) -> np.ndarray:
        """Return preview-frame ``i`` (0-based over :attr:`indices`)."""
        if not (0 <= i < len(self.indices)):
            raise IndexError(i)
        real = self.indices[i]
        hit = self._cache.get(real)
        if hit is not None:
            return hit
        arr = self._read(real)
        self.read_count += 1
        self._cache[real] = arr
        self._cache_order.append(real)
        while len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return arr

    def _read(self, real: int) -> np.ndarray:
        if self.kind == "files":
            path = self.source[real]
            if path.suffix.lower() == ".npy":
                return np.asarray(np.load(str(path)))
            if path.suffix.lower() in (".tif", ".tiff"):
                import tifffile
                return np.asarray(tifffile.imread(str(path)))
            from PIL import Image
            with Image.open(path) as im:
                return np.asarray(im)
        if self.kind == "npy":
            return np.asarray(self.source[real])
        import tifffile
        if self.kind == "tiffmm":
            if self._memmap is None:
                try:
                    self._memmap = tifffile.memmap(str(self.source))
                except Exception:
                    LOG.debug("tifffile.memmap unavailable", exc_info=True)
                    self._memmap = tifffile.imread(str(self.source))
            return np.asarray(self._memmap[real])
        return np.asarray(tifffile.imread(str(self.source), key=real))

    def describe(self) -> str:
        """One-line summary for the status label."""
        shown = len(self.indices)
        if self.truncated:
            return (f"{os.path.basename(self.label)} · showing {shown} of "
                    f"{self.n_available} frames")
        return f"{os.path.basename(self.label)} · {shown} frames"


def frame_channel(frame: np.ndarray, channel: int) -> np.ndarray:
    """Return a 2-D plane from a frame stored either (H, W, C) or (C, H, W).

    A merged frame written by spaCR is channel-last, but a raw acquisition
    TIFF page is often channel-first. Guessing wrong turns a 3-channel image
    into a 3-pixel-tall one, so pick the axis that actually looks like a
    channel axis (small, and not the only small axis).
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        return arr.squeeze()
    if arr.shape[-1] <= 8 and arr.shape[0] > 8:
        return _select_channel(arr, channel)
    if arr.shape[0] <= 8 and arr.shape[-1] > 8:
        return arr[int(channel) % arr.shape[0]]
    return _select_channel(arr, channel)


# ---------------------------------------------------------------------------
# Segmentation (expensive — cached by the panel)
# ---------------------------------------------------------------------------

def segment_frame(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Segment one frame with Cellpose and return an ``int32`` label image.

    Cellpose is imported inside the call so importing this module — as the
    test suite does — costs nothing and needs no CUDA stack. The panel calls
    this through the module global, which is also what lets a test swap in a
    counting stub to prove that tuning a *tracking* setting never reaches
    segmentation.
    """
    from cellpose import models as cp_models
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False

    model_name = str(params.get("model", "cpsam"))
    if model_name == "cpsam":
        model = cp_models.CellposeModel(
            gpu=gpu, pretrained_model="cpsam", device=None)
    else:
        model = cp_models.CellposeModel(
            gpu=gpu, model_type=model_name, device=None)

    plane = frame_channel(image, int(params.get("channel", 0)))
    if params.get("normalise", True):
        plane = _to_uint8(plane, normalise=True,
                          lo_pct=float(params.get("lo_pct", 2.0)),
                          hi_pct=float(params.get("hi_pct", 98.0)))
    result = model.eval(
        plane,
        diameter=float(params.get("diameter", 30.0)) or None,
        flow_threshold=float(params.get("flow_threshold", 0.4)),
        cellprob_threshold=float(params.get("cellprob", 0.0)),
    )
    mask = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(mask, list):
        mask = mask[0]
    return np.asarray(mask).astype(np.int32)


def _as_label_stack(seq: "FrameSequence", channel: int = 0) -> np.ndarray:
    """Stack a mask sequence into (T, H, W) ``int32`` labels, frame by frame."""
    planes = [np.asarray(frame_channel(seq.frame(i), channel)).astype(np.int32)
              for i in range(len(seq))]
    return np.stack(planes, axis=0)


def segment_sequence(seq: "FrameSequence", params: Dict[str, Any]) -> np.ndarray:
    """Segment every preview frame of ``seq`` into a (T, H, W) label stack.

    Reads and segments one frame at a time so the whole movie is never
    resident, then stacks only the label images (which are far smaller than
    the raw multi-channel frames).
    """
    masks = [segment_frame(seq.frame(i), params) for i in range(len(seq))]
    shapes = {m.shape for m in masks}
    if len(shapes) != 1:
        raise ValueError(
            f"frames segmented to different shapes {sorted(shapes)}; the "
            "sequence is not a single field of view.")
    return np.stack(masks, axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# Linking (cheap — re-run on every tracking-setting change)
# ---------------------------------------------------------------------------

def backend_available(mode: str) -> Tuple[bool, str]:
    """Whether linking backend ``mode`` can run, and why not when it cannot.

    Checks with :func:`importlib.util.find_spec`, so nothing heavy is
    imported just to answer the question and a missing optional dependency
    never raises.

    :returns: ``(True, "")`` when usable, else ``(False, message)`` where the
        message names the package and the command that installs it.
    """
    mode = (mode or "").lower()
    if mode not in TRACK_MODES:
        return False, f"Unknown linking mode {mode!r}."
    pkg = {"trackastra": "trackastra", "ultrack": "ultrack",
           "btrack": "btrack", "trackpy": "trackpy"}.get(mode)
    if pkg is None:          # iou is pure numpy + scipy, always available
        return True, ""
    import importlib.util
    if importlib.util.find_spec(pkg) is not None:
        return True, ""
    fix = OPTIONAL_BACKENDS.get(mode, f"pip install {pkg}")
    alt = ", ".join(m for m in ("trackpy", "iou") if m != mode)
    return False, (
        f"timelapse_mode='{mode}' needs the {pkg} package, which is not "
        f"installed. Install it with `{fix}`, or preview with {alt}.")


def _tracks_from_features(tracks_df, features):
    """Attach centroids to a track table that only carries labels."""
    import pandas as pd  # noqa: F401  (pandas is already a hard dependency)
    cols = ["frame", "original_label", "x", "y"]
    return tracks_df.merge(features[cols], on=["frame", "original_label"],
                           how="left")


def _link_iou(masks: np.ndarray, iou_threshold: float):
    """Link by frame-to-frame IoU using spaCR's own linker."""
    from spacr.timelapse import _prepare_for_tracking, _track_by_iou
    df = _track_by_iou(masks, iou_threshold=float(iou_threshold))
    return _tracks_from_features(df, _prepare_for_tracking(masks))


def _link_trackpy(masks: np.ndarray, displacement: float, memory: int):
    """Link with trackpy's nearest-neighbour linker.

    Uses the same feature table the pipeline builds
    (:func:`spacr.timelapse._prepare_for_tracking`) and the same two knobs
    the user tunes — ``search_range`` (``timelapse_displacement``) and
    ``memory`` (``timelapse_memory``) — so what the preview shows is what a
    run will do.
    """
    import trackpy as tp
    from spacr.timelapse import _prepare_for_tracking
    features = _prepare_for_tracking(masks)
    try:
        tp.quiet()
    except Exception:
        LOG.debug("trackpy.quiet() unavailable", exc_info=True)
    linked = tp.link_df(features, search_range=float(displacement),
                        memory=int(memory))
    return linked.rename(columns={"particle": "track_id"})


def _link_btrack(masks: np.ndarray, displacement: float):
    """Link with btrack's Bayesian tracker using its stock cell motion model.

    btrack ships no config file; ``btrack.datasets.cell_config()`` fetches
    one on first use. In a preview that download is the difference between
    instant and hung, so a failure is turned into an actionable message
    rather than a stack trace.
    """
    import btrack
    from btrack import datasets as btrack_datasets
    from spacr.timelapse import _prepare_for_tracking
    try:
        config = btrack_datasets.cell_config()
    except Exception as exc:
        raise TrackerUnavailable(
            "btrack's motion-model config is not available offline "
            f"({exc}). btrack.datasets.cell_config() downloads it once — run "
            "the Timelapse module with timelapse_mode='btrack' to cache it, "
            "or preview with trackpy / iou.") from exc

    objects = btrack.utils.segmentation_to_objects(masks, properties=("area",))
    with btrack.BayesianTracker() as tracker:
        tracker.configure(config)
        tracker.max_search_radius = float(displacement)
        tracker.append(objects)
        tracker.track()
    rows = []
    for tr in tracker.tracks:
        for t, x, y in zip(tr["t"], tr["x"], tr["y"]):
            rows.append({"frame": int(t), "track_id": int(tr["ID"]),
                         "x": float(x), "y": float(y)})
    import pandas as pd
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"])
    if df.empty:
        return _empty_tracks(df)
    features = _prepare_for_tracking(masks)
    return _attach_labels_by_position(df, features)


def _empty_tracks(like):
    """An empty track table shaped like ``like`` plus ``original_label``."""
    import pandas as pd
    out = like.iloc[0:0].copy()
    out["original_label"] = pd.Series(dtype="int64")
    return out


def _attach_labels_by_position(df, features):
    """Give each tracked point the label of the nearest object in its frame.

    btrack reports track coordinates, not the label they came from; the
    overlay needs a label so masks can be recoloured by track id. Nearest
    centroid within a frame is exact for the centroids btrack was fed.
    """
    out = []
    for frame, g in df.groupby("frame"):
        f = features[features["frame"] == frame]
        if f.empty:
            continue
        fx = f["x"].to_numpy(dtype=float)
        fy = f["y"].to_numpy(dtype=float)
        labels = f["original_label"].to_numpy()
        g = g.copy()
        idx = [int(np.argmin((fx - float(x)) ** 2 + (fy - float(y)) ** 2))
               for x, y in zip(g["x"], g["y"])]
        g["original_label"] = labels[idx]
        out.append(g)
    if not out:
        return _empty_tracks(df)
    import pandas as pd
    return pd.concat(out, ignore_index=True)


def _link_trackastra(masks: np.ndarray, images: Optional[np.ndarray],
                     model_name: str, linking: str):
    """Link with Trackastra's pretrained transformer (no hyperparameters)."""
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc
    from spacr.timelapse import _relabelled_stack_to_tracks_df
    imgs = np.asarray(images) if images is not None else masks.astype(np.float32)
    model = Trackastra.from_pretrained(str(model_name), device="automatic")
    graph = model.track(imgs, masks, mode=str(linking))
    _ctc, tracked = graph_to_ctc(graph, masks, outdir=None)
    return _relabelled_stack_to_tracks_df(np.asarray(tracked))


def _link_ultrack(masks: np.ndarray, max_distance: float):
    """Link with Ultrack's joint segmentation/linking solver."""
    from ultrack import MainConfig, track, to_tracks_layer, tracks_to_zarr
    from ultrack import utils as ultrack_utils
    from spacr.timelapse import (
        _relabelled_stack_to_tracks_df, _ultrack_labels_to_contours,
        _ultrack_set, _ultrack_track_kwargs,
    )
    import tempfile
    labels_to_contours = _ultrack_labels_to_contours(ultrack_utils)
    fg_kwarg, contours_kwarg, _accepts_images = _ultrack_track_kwargs(track)
    foreground, contours = labels_to_contours(masks, sigma=0.0)
    config = MainConfig()
    work_dir = tempfile.mkdtemp(prefix="spacr_ultrack_preview_")
    import pathlib
    _ultrack_set(config.data_config, "working_dir", pathlib.Path(work_dir),
                 "ultrack_working_dir")
    _ultrack_set(config.linking_config, "max_distance", float(max_distance),
                 "ultrack_max_distance")
    track(config, **{fg_kwarg: foreground, contours_kwarg: contours})
    tracks_layer, _graph = to_tracks_layer(config)
    tracked = tracks_to_zarr(config, tracks_layer)
    return _relabelled_stack_to_tracks_df(np.asarray(tracked))


def link_tracks(masks: np.ndarray, mode: str = "iou",
                displacement: float = 50.0, memory: int = 3,
                iou_threshold: float = 0.1,
                images: Optional[np.ndarray] = None,
                trackastra_model: str = "general_2d",
                trackastra_linking: str = "greedy"):
    """Link a (T, H, W) label stack into tracks with the chosen backend.

    :returns: a DataFrame with ``frame``, ``original_label``, ``track_id``,
        ``x`` and ``y`` — the same layout the pipeline's trackers emit.
    :raises TrackerUnavailable: when the backend's package is missing, with
        a message naming the package and the install command.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"masks must be (T, H, W); got shape {masks.shape}")
    if masks.shape[0] < 2:
        raise ValueError("a timelapse preview needs at least two frames.")

    mode = (mode or "iou").lower()
    ok, why = backend_available(mode)
    if not ok:
        raise TrackerUnavailable(why)

    if mode == "iou":
        return _link_iou(masks, iou_threshold)
    if mode == "trackpy":
        return _link_trackpy(masks, displacement, memory)
    if mode == "btrack":
        return _link_btrack(masks, displacement)
    if mode == "trackastra":
        return _link_trackastra(masks, images, trackastra_model,
                                trackastra_linking)
    return _link_ultrack(masks, displacement)


def relabel_by_track(masks: np.ndarray, tracks) -> np.ndarray:
    """Recolour a label stack by track id using spaCR's own relabeller.

    This is what makes an identity swap visible: an object that keeps its
    track id keeps its colour across frames, and one that is handed to
    another track changes colour mid-movie.
    """
    from spacr.timelapse import _relabel_masks_based_on_tracks
    if tracks is None or len(tracks) == 0:
        return np.zeros_like(masks)
    return _relabel_masks_based_on_tracks(np.asarray(masks), tracks)


# ---------------------------------------------------------------------------
# Track quality indicators
# ---------------------------------------------------------------------------

@dataclass
class TrackStats:
    """What the user is actually tuning against.

    ``fragmentation_*`` and ``suspicious_jumps`` are *indicators* computed
    without ground truth, not measurements — the panel labels them as such.
    """
    n_frames: int = 0
    n_tracks: int = 0
    mean_length: float = 0.0
    median_length: float = 0.0
    n_short: int = 0
    min_length: int = 3
    starts_after_first: int = 0
    ends_before_last: int = 0
    suspicious_jumps: int = 0
    max_step: float = 0.0
    objects_per_frame: float = 0.0
    displacement_limit: float = 0.0

    @property
    def fragmentation_events(self) -> int:
        """Track starts after frame 0 plus track ends before the last frame."""
        return self.starts_after_first + self.ends_before_last

    def summary(self) -> str:
        """The one-line status the panel pins under the canvas."""
        return (
            f"{self.n_tracks} tracks over {self.n_frames} frames · "
            f"mean length {self.mean_length:.1f} "
            f"(median {self.median_length:.0f}) · "
            f"{self.n_short} shorter than {self.min_length} frames · "
            f"fragmentation {self.fragmentation_events} "
            f"({self.starts_after_first} late starts, "
            f"{self.ends_before_last} early ends) · "
            f"{self.suspicious_jumps} steps over "
            f"{self.displacement_limit:.0f} px "
            f"(max {self.max_step:.0f} px) — swap risk"
        )


def track_stats(tracks, n_frames: int, min_length: int = 3,
                displacement_limit: float = 50.0) -> TrackStats:
    """Summarise a track table into the numbers that drive a tuning decision.

    :param tracks: DataFrame with ``frame``, ``track_id`` and (for the swap
        indicator) ``x``/``y``.
    :param n_frames: how many frames the preview covered.
    :param min_length: tracks shorter than this are counted as short — the
        live-settable threshold that says what "too short" means here.
    :param displacement_limit: the linking radius being tuned; a within-track
        step longer than this is counted as a swap risk.
    """
    st = TrackStats(n_frames=int(n_frames), min_length=int(min_length),
                    displacement_limit=float(displacement_limit))
    if tracks is None or len(tracks) == 0:
        return st

    lengths = tracks.groupby("track_id")["frame"].nunique()
    st.n_tracks = int(lengths.shape[0])
    st.mean_length = float(lengths.mean())
    st.median_length = float(lengths.median())
    st.n_short = int((lengths < int(min_length)).sum())
    st.objects_per_frame = float(len(tracks)) / max(1, int(n_frames))

    firsts = tracks.groupby("track_id")["frame"].min()
    lasts = tracks.groupby("track_id")["frame"].max()
    st.starts_after_first = int((firsts > tracks["frame"].min()).sum())
    st.ends_before_last = int((lasts < tracks["frame"].max()).sum())

    if {"x", "y"}.issubset(tracks.columns):
        jumps = 0
        biggest = 0.0
        for _tid, g in tracks.sort_values("frame").groupby("track_id"):
            x = g["x"].to_numpy(dtype=float)
            y = g["y"].to_numpy(dtype=float)
            if x.size < 2:
                continue
            d = np.hypot(np.diff(x), np.diff(y))
            d = d[np.isfinite(d)]
            if d.size == 0:
                continue
            jumps += int((d > float(displacement_limit)).sum())
            biggest = max(biggest, float(d.max()))
        st.suspicious_jumps = jumps
        st.max_step = biggest
    return st


# ---------------------------------------------------------------------------
# Overlay rendering (pure numpy — unit-testable without a display)
# ---------------------------------------------------------------------------

def track_colour(track_id: int) -> Tuple[int, int, int]:
    """Deterministic colour for a track id, stable across frames and runs."""
    return TRACK_COLOURS[int(track_id) % len(TRACK_COLOURS)]


def _draw_segment(rgb: np.ndarray, x0, y0, x1, y1, colour) -> None:
    """Draw a 1-px line into ``rgb`` by sampling along it (no cv2 needed)."""
    h, w = rgb.shape[:2]
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, max(2, n)).astype(int)
    ys = np.linspace(y0, y1, max(2, n)).astype(int)
    keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    rgb[ys[keep], xs[keep]] = np.array(colour, dtype=np.uint8)


def _draw_dot(rgb: np.ndarray, x, y, colour, radius: int = 2) -> None:
    h, w = rgb.shape[:2]
    x, y, r = int(x), int(y), max(1, int(radius))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    if y1 > y0 and x1 > x0:
        rgb[y0:y1, x0:x1] = np.array(colour, dtype=np.uint8)


def render_frame(image: np.ndarray, labels: Optional[np.ndarray] = None,
                 tracks=None, frame: int = 0, tail: int = 12,
                 normalise: bool = True, lo_pct: float = 2.0,
                 hi_pct: float = 98.0, channel: int = 0) -> np.ndarray:
    """Render one preview frame: image, mask outlines, and track history.

    Outlines are coloured **per track id** so a fragmented track shows up as
    an object that changes colour partway through, and the trailing polyline
    shows where each object came from over the last ``tail`` frames.
    """
    plane = frame_channel(image, channel)
    base = _to_uint8(plane, normalise=normalise, lo_pct=lo_pct, hi_pct=hi_pct)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = np.ascontiguousarray(base[..., :3])

    if labels is not None and np.asarray(labels).any():
        labels = np.asarray(labels).astype(np.int32)
        boundary = _boundary_mask(labels)
        edge = boundary & (labels > 0)
        ids = np.unique(labels[edge]) if edge.any() else np.array([], dtype=int)
        for tid in ids:
            rgb[edge & (labels == tid)] = np.array(
                track_colour(int(tid)), dtype=np.uint8)

    if tracks is not None and len(tracks) and {"x", "y"}.issubset(tracks.columns):
        lo = max(0, int(frame) - int(tail))
        window = tracks[(tracks["frame"] <= int(frame))
                        & (tracks["frame"] >= lo)]
        for tid, g in window.sort_values("frame").groupby("track_id"):
            colour = track_colour(int(tid))
            xs = g["x"].to_numpy(dtype=float)
            ys = g["y"].to_numpy(dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            for i in range(1, xs.size):
                _draw_segment(rgb, xs[i - 1], ys[i - 1], xs[i], ys[i], colour)
            if xs.size:
                _draw_dot(rgb, xs[-1], ys[-1], colour, radius=2)
    return rgb


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class TimelapseRequest:
    """One preview pass. ``cached_masks`` is what makes re-linking cheap."""
    sequence: Optional[FrameSequence] = None
    mask_sequence: Optional[FrameSequence] = None
    cached_masks: Optional[np.ndarray] = None
    seg: Dict[str, Any] = field(default_factory=dict)
    track: Dict[str, Any] = field(default_factory=dict)


def run_preview_pass(req: TimelapseRequest) -> Dict[str, Any]:
    """Do the work of one preview: masks (maybe cached), then linking."""
    masks = req.cached_masks
    segmented = False
    if masks is None:
        if req.mask_sequence is not None:
            masks = _as_label_stack(req.mask_sequence,
                                    int(req.seg.get("mask_channel", 0)))
        elif req.sequence is not None:
            masks = segment_sequence(req.sequence, req.seg)
            segmented = True
        else:
            raise ValueError("Load a sequence first.")
    masks = np.asarray(masks)
    tracks = link_tracks(masks, **req.track)
    return {"masks": masks, "tracks": tracks, "segmented": segmented,
            "masks_built": req.cached_masks is None}


class _TimelapseWorker(QThread):
    """Runs one preview pass off the GUI thread.

    Mirrors ``live_preview._PreviewWorker`` exactly: every exception is
    caught inside :meth:`run` and re-emitted as an error *string*, so
    nothing ever propagates out of a Qt thread's ``run()``.
    """

    finished_result = Signal(object, str)   # (result dict or None, error)

    def __init__(self, request: TimelapseRequest, parent=None):
        super().__init__(parent)
        self._request = request

    def run(self):
        try:
            self.finished_result.emit(run_preview_pass(self._request), "")
        except Exception as e:
            LOG.info("timelapse preview failed: %s", e, exc_info=True)
            self.finished_result.emit(None, str(e))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class TimelapsePreviewPanel(QWidget):
    """Interactive tracking preview — Timelapse module.

    Same contract as :class:`~spacr.qt.widgets.live_preview.LivePreviewPanel`:
    a standalone ``QWidget``, a ``QThread`` worker that emits results over
    signals, :meth:`set_propagate_callback` to push tuned values back into
    the main settings panel, and a ``build_*_card`` factory.
    """

    preview_ready = Signal(object)   # TrackStats, or None on failure

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sequence: Optional[FrameSequence] = None
        self._mask_sequence: Optional[FrameSequence] = None
        self._masks: Optional[np.ndarray] = None
        self._tracked: Optional[np.ndarray] = None
        self._tracks = None
        self._raw_tracks = None
        self._stats: Optional[TrackStats] = None
        self._mask_cache: Dict[tuple, np.ndarray] = {}
        self._worker: Optional[_TimelapseWorker] = None
        self._pending_signature: Optional[tuple] = None
        self._propagate_cb = None
        self._settings: Dict[str, Any] = {}
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._build_ui()
        self.setAcceptDrops(True)
        for v in (self._src_view, self._out_view):
            v.setAcceptDrops(False)

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        pick = QHBoxLayout()
        self._path_label = QLabel(
            "No sequence loaded — drop a folder of frames, a multi-page TIFF, "
            "or a (T, H, W) .npy stack here")
        self._path_label.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Preferred)
        seq_btn = QPushButton("Choose sequence…")
        seq_btn.clicked.connect(self._pick_sequence)
        mask_btn = QPushButton("Masks…")
        mask_btn.setToolTip(
            "Optional: point at a folder or stack of ready-made label images "
            "to skip segmentation entirely.")
        mask_btn.clicked.connect(self._pick_masks)
        pick.addWidget(self._path_label, 1)
        pick.addWidget(seq_btn)
        pick.addWidget(mask_btn)
        root.addLayout(pick)

        # -- segmentation settings (changing one invalidates the mask cache) --
        self._model_box = QComboBox(self)
        self._model_box.addItems(["cpsam", "cyto3", "cyto2", "nuclei"])
        self._model_box.setToolTip(
            "(str) Cellpose model used to segment every frame. Changing this "
            "re-segments — it is the expensive half of the preview.")
        self._object_box = QComboBox(self)
        self._object_box.addItems(["cell", "nucleus", "pathogen"])
        self._object_box.setToolTip(
            "(list) Which object is tracked across frames (timelapse_objects).")
        self._channel = QSpinBox(self)
        self._channel.setRange(0, 8)
        self._channel.setToolTip(
            "(int) Image channel index segmented for the tracked object.")
        self._diameter = QDoubleSpinBox(self)
        self._diameter.setRange(0, 400)
        self._diameter.setValue(30.0)
        self._diameter.setSuffix(" px")
        self._diameter.setToolTip(
            "(float, px) Expected object diameter. Ignored by Cellpose-SAM.")
        self._flow = QDoubleSpinBox(self)
        self._flow.setRange(-1, 3)
        self._flow.setSingleStep(0.05)
        self._flow.setValue(0.4)
        self._flow.setToolTip("(float) Cellpose flow threshold.")
        self._prob = QDoubleSpinBox(self)
        self._prob.setRange(-6, 6)
        self._prob.setSingleStep(0.1)
        self._prob.setValue(0.0)
        self._prob.setToolTip("(float) Cellpose cell-probability threshold.")
        self._max_frames = QSpinBox(self)
        self._max_frames.setRange(2, 500)
        self._max_frames.setValue(12)
        self._max_frames.setToolTip(
            "(int) How many frames of the sequence the preview reads. The "
            "rest of the movie is never loaded.")
        self._normalise = QCheckBox("Normalise", self)
        self._normalise.setChecked(True)
        self._normalise.setToolTip(
            "(bool) Percentile-stretch each frame for display + segmentation.")

        # -- tracking settings (changing one re-links only) -------------------
        self._mode_box = QComboBox(self)
        self._mode_box.addItems(list(TRACK_MODES))
        self._mode_box.setCurrentText("iou")
        self._mode_box.setToolTip(
            "(str) timelapse_mode — which backend links objects between "
            "frames.")
        self._displacement = QDoubleSpinBox(self)
        self._displacement.setRange(1, 2000)
        self._displacement.setValue(50.0)
        self._displacement.setSuffix(" px")
        self._displacement.setToolTip(
            "(int, px) timelapse_displacement — the largest jump a linker "
            "will accept. Too small fragments tracks; too large swaps "
            "identities.")
        self._memory = QSpinBox(self)
        self._memory.setRange(0, 50)
        self._memory.setValue(3)
        self._memory.setToolTip(
            "(int) timelapse_memory — how many frames an object may vanish "
            "and still rejoin its track (trackpy only).")
        self._iou = QDoubleSpinBox(self)
        self._iou.setRange(0.0, 1.0)
        self._iou.setSingleStep(0.05)
        self._iou.setValue(0.1)
        self._iou.setToolTip(
            "(float) Minimum IoU accepted by the 'iou' linker.")
        self._min_len = QSpinBox(self)
        self._min_len.setRange(1, 500)
        self._min_len.setValue(3)
        self._min_len.setToolTip(
            "(int) Tracks shorter than this are counted as fragments in the "
            "indicators below.")
        self._remove_transient = QCheckBox("Keep only full-length tracks", self)
        self._remove_transient.setToolTip(
            "(bool) timelapse_remove_transient — drop every track not present "
            "in all frames.")
        self._tail = QSpinBox(self)
        self._tail.setRange(0, 200)
        self._tail.setValue(12)
        self._tail.setToolTip(
            "(int) How many frames of track history are drawn behind each "
            "object.")

        seg_group = QGroupBox("Segmentation (cached — changing these re-segments)")
        seg_form = QFormLayout(seg_group)
        seg_form.addRow("Model", self._model_box)
        seg_form.addRow("Tracked object", self._object_box)
        seg_form.addRow("Channel", self._channel)
        seg_form.addRow("Diameter", self._diameter)
        seg_form.addRow("Flow threshold", self._flow)
        seg_form.addRow("Cell probability", self._prob)
        seg_form.addRow("Frames previewed", self._max_frames)
        seg_form.addRow(self._normalise)

        trk_group = QGroupBox("Tracking (live — changing these only re-links)")
        trk_form = QFormLayout(trk_group)
        trk_form.addRow("Mode", self._mode_box)
        trk_form.addRow("Max displacement", self._displacement)
        trk_form.addRow("Memory", self._memory)
        trk_form.addRow("IoU threshold", self._iou)
        trk_form.addRow("Min track length", self._min_len)
        trk_form.addRow("Track tail", self._tail)
        trk_form.addRow(self._remove_transient)

        groups = QHBoxLayout()
        groups.addWidget(seg_group, 1)
        groups.addWidget(trk_group, 1)
        root.addLayout(groups)

        # Re-link (never re-segment) whenever a *linking* knob moves.
        for w in (self._displacement, self._memory, self._iou):
            w.valueChanged.connect(self._on_tracking_changed)
        self._mode_box.currentTextChanged.connect(self._on_tracking_changed)
        self._remove_transient.toggled.connect(self._on_tracking_changed)
        # The minimum length only decides what counts as a fragment, so it
        # re-scores the existing tracks — no linking, no segmentation.
        self._min_len.valueChanged.connect(self._on_scoring_changed)
        # Pure display knobs never touch masks or tracks.
        self._tail.valueChanged.connect(lambda *_: self._refresh_canvases())
        self._normalise.toggled.connect(lambda *_: self._refresh_canvases())

        act = QHBoxLayout()
        self._run_btn = QPushButton("Run preview", self)
        self._run_btn.clicked.connect(self.run_preview)
        self._relink_btn = QPushButton("Re-link", self)
        self._relink_btn.setToolTip(
            "Re-run only the tracker on the cached per-frame masks.")
        self._relink_btn.clicked.connect(self.relink)
        self._propagate_btn = QPushButton("Propagate settings", self)
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, the settings tuned here are copied into the main "
            "Timelapse settings so the run uses them.")
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        self._status = QLabel("", self)
        act.addWidget(self._run_btn)
        act.addWidget(self._relink_btn)
        act.addWidget(self._propagate_btn)
        act.addWidget(self._status, 1)
        root.addLayout(act)

        canvas = QHBoxLayout()
        self._src_view = _ZoomView(self)
        self._src_view.setMinimumHeight(160)
        self._out_view = _ZoomView(self)
        self._out_view.setMinimumHeight(160)
        self._src_view.set_peer(self._out_view)
        self._out_view.set_peer(self._src_view)
        canvas.addWidget(self._src_view, 1)
        canvas.addWidget(self._out_view, 1)
        root.addLayout(canvas, 1)

        scrub = QHBoxLayout()
        scrub.addWidget(QLabel("Frame", self))
        self._play_btn = QPushButton("Play", self)
        self._play_btn.setEnabled(False)
        self._play_btn.setToolTip(
            "Play the preview as a loop. The source and tracked views stay "
            "synchronised while zoomed or panned.")
        self._play_btn.clicked.connect(self._toggle_playback)
        scrub.addWidget(self._play_btn)
        self._frame_slider = QSlider(Qt.Horizontal, self)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_scrub)
        self._frame_label = QLabel("–", self)
        self._frame_label.setStyleSheet("font-family: monospace;")
        self._play_fps = QSpinBox(self)
        self._play_fps.setRange(1, 30)
        self._play_fps.setValue(8)
        self._play_fps.setSuffix(" fps")
        self._play_fps.setToolTip("Playback speed; this does not alter data.")
        self._play_fps.valueChanged.connect(self._update_playback_interval)
        scrub.addWidget(self._frame_slider, 1)
        scrub.addWidget(self._frame_label)
        scrub.addWidget(self._play_fps)
        root.addLayout(scrub)

        self._stats_label = QLabel(
            "Load a sequence and run the preview to see track quality.", self)
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("font-family: monospace;")
        root.addWidget(self._stats_label)

    # -- drag & drop -------------------------------------------------------

    def _dropped_path(self, event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            if p.is_dir() or p.suffix.lower() in FRAME_SUFFIXES:
                return str(p)
        return None

    def dragEnterEvent(self, event):    # noqa: N802 (Qt naming)
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):     # noqa: N802
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):         # noqa: N802
        p = self._dropped_path(event)
        if p is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.load_sequence(p)

    # -- public API --------------------------------------------------------

    def load_sequence(self, path) -> bool:
        """Open ``path`` as the preview sequence. Errors land inline."""
        self._stop_playback()
        try:
            seq = FrameSequence.open(path, max_frames=self._max_frames.value())
        except Exception as e:
            self._status.setText(f"Load failed: {e}")
            return False
        self._sequence = seq
        self._masks = None
        self._tracked = None
        self._tracks = None
        self._mask_cache.clear()
        self._path_label.setText(seq.describe())
        self._status.setText(
            f"Loaded {seq.describe()} — run the preview to segment + link.")
        self._frame_slider.setMaximum(max(0, len(seq) - 1))
        self._frame_slider.setValue(0)
        self._play_btn.setEnabled(len(seq) > 1)
        self._refresh_canvases()
        return True

    def load_masks(self, path) -> bool:
        """Use ready-made label images instead of segmenting."""
        self._stop_playback()
        try:
            seq = FrameSequence.open(path, max_frames=self._max_frames.value())
        except Exception as e:
            self._status.setText(f"Mask load failed: {e}")
            return False
        self._mask_sequence = seq
        self._masks = None
        self._mask_cache.clear()
        if self._sequence is None:
            self._frame_slider.setMaximum(max(0, len(seq) - 1))
            self._frame_slider.setValue(0)
        self._play_btn.setEnabled(len(seq) > 1)
        self._status.setText(
            f"Masks: {seq.describe()} — segmentation will be skipped.")
        return True

    def set_propagate_callback(self, cb) -> None:
        """Register a ``callback(dict)`` that writes tuned values into the
        main settings panel (wired by the AppScreen)."""
        self._propagate_cb = cb

    def settings_for_propagation(self) -> dict:
        """Map the preview's widgets onto real Timelapse setting keys."""
        obj = self._object_box.currentText()
        return {
            "timelapse_mode": self._mode_box.currentText(),
            "timelapse_displacement": int(self._displacement.value()),
            "timelapse_memory": int(self._memory.value()),
            "timelapse_objects": [obj],
            "timelapse_remove_transient": bool(
                self._remove_transient.isChecked()),
            "timelapse_frame_limits": [0, int(self._max_frames.value())],
            f"{obj}_channel": int(self._channel.value()),
            f"{obj}_diameter": float(self._diameter.value()),
            "cell_FT": float(self._flow.value()),
            "cell_CP_prob": float(self._prob.value()),
            "normalize": bool(self._normalise.isChecked()),
        }

    def propagate_settings(self) -> None:
        """Push the current settings to the main panel, if wired."""
        if self._propagate_cb is not None:
            try:
                self._propagate_cb(self.settings_for_propagation())
            except Exception:
                LOG.debug("propagate_settings failed", exc_info=True)

    def apply_settings(self, settings: dict) -> None:
        """Seed the preview from the main Timelapse settings dict."""
        self._settings = dict(settings or {})
        try:
            mode = settings.get("timelapse_mode")
            if mode and self._mode_box.findText(str(mode)) >= 0:
                self._mode_box.setCurrentText(str(mode))
            disp = settings.get("timelapse_displacement")
            if disp:
                self._displacement.setValue(float(disp))
            mem = settings.get("timelapse_memory")
            if mem is not None:
                self._memory.setValue(int(mem))
            objs = settings.get("timelapse_objects")
            if objs and self._object_box.findText(str(objs[0])) >= 0:
                self._object_box.setCurrentText(str(objs[0]))
            self._remove_transient.setChecked(
                bool(settings.get("timelapse_remove_transient", False)))
            obj = self._object_box.currentText()
            ch = settings.get(f"{obj}_channel")
            if ch is not None:
                self._channel.setValue(int(ch))
            diam = settings.get(f"{obj}_diameter")
            if diam:
                self._diameter.setValue(float(diam))
        except Exception:
            LOG.debug("apply_settings failed", exc_info=True)

    def current_params(self) -> dict:
        """Snapshot for tests + external callers."""
        return {
            "model": self._model_box.currentText(),
            "object": self._object_box.currentText(),
            "channel": int(self._channel.value()),
            "mode": self._mode_box.currentText(),
            "displacement": float(self._displacement.value()),
            "memory": int(self._memory.value()),
            "iou_threshold": float(self._iou.value()),
            "min_length": int(self._min_len.value()),
            "max_frames": int(self._max_frames.value()),
            "n_frames": len(self._sequence) if self._sequence else 0,
        }

    # -- cache key ---------------------------------------------------------

    def _segmentation_signature(self) -> tuple:
        """Everything that can change a *label image*, and nothing else.

        Tracking settings are deliberately absent: that is what makes a
        tracking change a cache hit and therefore free.
        """
        seq = self._sequence
        msk = self._mask_sequence
        return (
            getattr(msk, "label", None),
            getattr(seq, "label", None),
            tuple(getattr(seq, "indices", ())),
            None if msk is not None else self._model_box.currentText(),
            None if msk is not None else int(self._channel.value()),
            None if msk is not None else float(self._diameter.value()),
            None if msk is not None else float(self._flow.value()),
            None if msk is not None else float(self._prob.value()),
            None if msk is not None else bool(self._normalise.isChecked()),
        )

    def _seg_params(self) -> Dict[str, Any]:
        return {
            "model": self._model_box.currentText(),
            "channel": int(self._channel.value()),
            "mask_channel": int(self._channel.value()),
            "diameter": float(self._diameter.value()),
            "flow_threshold": float(self._flow.value()),
            "cellprob": float(self._prob.value()),
            "normalise": bool(self._normalise.isChecked()),
        }

    def _track_params(self) -> Dict[str, Any]:
        return {
            "mode": self._mode_box.currentText(),
            "displacement": float(self._displacement.value()),
            "memory": int(self._memory.value()),
            "iou_threshold": float(self._iou.value()),
            "trackastra_model": str(
                self._settings.get("trackastra_model", "general_2d")),
            "trackastra_linking": str(
                self._settings.get("trackastra_linking", "greedy")),
        }

    # -- running -----------------------------------------------------------

    def run_preview(self) -> None:
        """Segment (unless cached) then link, off the GUI thread."""
        self._start(allow_segmentation=True)

    def relink(self) -> None:
        """Re-link the cached masks. Never re-segments."""
        self._start(allow_segmentation=False)

    def _on_tracking_changed(self, *_):
        """A linking knob moved — re-link if masks are already cached.

        Silent when nothing has been segmented yet: the user is still
        setting up, and a change should not kick off an expensive pass they
        did not ask for.
        """
        if self._masks is None:
            return
        self.relink()

    def _on_scoring_changed(self, *_):
        """The fragment threshold moved — re-score, don't re-link."""
        if self._masks is None or self._raw_tracks is None:
            return
        self._apply_tracks(self._raw_tracks, note="Re-scored (cached tracks)")

    def _start(self, allow_segmentation: bool) -> None:
        if self._sequence is None and self._mask_sequence is None:
            self._status.setText("Load a sequence first.")
            return
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Preview already running.")
            return

        mode = self._mode_box.currentText()
        ok, why = backend_available(mode)
        if not ok:
            self._status.setText(why)
            self.preview_ready.emit(None)
            return

        sig = self._segmentation_signature()
        cached = self._mask_cache.get(sig)
        if cached is None and not allow_segmentation:
            self._status.setText(
                "No cached masks for these segmentation settings — "
                "hit Run preview.")
            return

        self._pending_signature = sig
        req = TimelapseRequest(
            sequence=self._sequence,
            mask_sequence=self._mask_sequence,
            cached_masks=cached,
            seg=self._seg_params(),
            track=self._track_params(),
        )
        self._run_btn.setEnabled(False)
        self._relink_btn.setEnabled(False)
        self._status.setText(
            "Re-linking cached masks…" if cached is not None
            else "Segmenting frames, then linking…")
        worker = _TimelapseWorker(req, self)
        # Bound method, not a closure: PySide6 delivers a plain-callable
        # connection on the *worker* thread, which would put every widget
        # touch below on the wrong thread.
        worker.finished_result.connect(self._on_worker_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_worker_done(self, result, err: str) -> None:
        """Adopt a finished pass. Runs on the GUI thread (queued signal)."""
        self._run_btn.setEnabled(True)
        self._relink_btn.setEnabled(True)
        self._worker = None
        if err:
            self._status.setText(f"Preview failed: {err}")
            self.preview_ready.emit(None)
            return
        if not result:
            self._status.setText("Preview returned nothing.")
            self.preview_ready.emit(None)
            return

        masks = result["masks"]
        self._masks = masks
        sig = getattr(self, "_pending_signature", None)
        if sig is not None:
            self._mask_cache[sig] = masks
        note = ("Masks built + linked" if result.get("masks_built")
                else "Re-linked (cached masks)")
        self._apply_tracks(result["tracks"], note=note)

    def _apply_tracks(self, tracks, note: str = "Linked") -> None:
        """Filter, relabel, score and render a track table.

        Called both after a worker pass and when only the scoring threshold
        moved, which is why the status text is passed in.
        """
        self._raw_tracks = tracks
        n_frames = int(self._masks.shape[0])
        if (tracks is not None and len(tracks)
                and self._remove_transient.isChecked()):
            full = tracks.groupby("track_id")["frame"].nunique() == n_frames
            keep = set(full[full].index)
            tracks = tracks[tracks["track_id"].isin(keep)]
        self._tracks = tracks
        self._tracked = relabel_by_track(self._masks, tracks)
        self._stats = track_stats(
            tracks, n_frames,
            min_length=int(self._min_len.value()),
            displacement_limit=float(self._displacement.value()))
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._play_btn.setEnabled(n_frames > 1)
        self._status.setText(f"{note} · {n_frames} frames")
        self._stats_label.setText(
            self._stats.summary()
            + "\nFragmentation and swap figures are indicators computed "
              "without ground truth, not measurements.")
        self._refresh_canvases()
        if self._propagate_btn.isChecked():
            self.propagate_settings()
        self.preview_ready.emit(self._stats)

    # -- rendering ---------------------------------------------------------

    def _on_scrub(self, _value: int) -> None:
        self._refresh_canvases()

    def _toggle_playback(self) -> None:
        """Start or pause looped playback of the loaded preview frames."""
        if self._play_timer.isActive():
            self._stop_playback()
            return
        if self._frame_slider.maximum() <= 0:
            return
        self._update_playback_interval()
        self._play_timer.start()
        self._play_btn.setText("Pause")

    def _stop_playback(self) -> None:
        self._play_timer.stop()
        if hasattr(self, "_play_btn"):
            self._play_btn.setText("Play")

    def _update_playback_interval(self, *_args) -> None:
        fps = max(1, int(self._play_fps.value()))
        self._play_timer.setInterval(max(1, round(1000 / fps)))

    def _advance_frame(self) -> None:
        """Advance one frame, wrapping at the end for continuous playback."""
        last = self._frame_slider.maximum()
        if last <= 0:
            self._stop_playback()
            return
        current = self._frame_slider.value()
        self._frame_slider.setValue(0 if current >= last else current + 1)

    def _refresh_canvases(self) -> None:
        seq = self._sequence
        idx = int(self._frame_slider.value())
        if seq is None:
            if self._masks is None:
                return
            image = self._masks[min(idx, self._masks.shape[0] - 1)]
        else:
            if not len(seq):
                return
            image = seq.frame(min(idx, len(seq) - 1))

        norm = self._normalise.isChecked()
        plane = frame_channel(image, int(self._channel.value()))
        self._src_view.set_pixmap(numpy_to_qpixmap(
            _to_uint8(plane, normalise=norm)))

        labels = None
        if self._tracked is not None and idx < self._tracked.shape[0]:
            labels = self._tracked[idx]
        overlay = render_frame(
            image, labels=labels, tracks=self._tracks, frame=idx,
            tail=int(self._tail.value()), normalise=norm,
            channel=int(self._channel.value()))
        self._out_view.set_pixmap(numpy_to_qpixmap(overlay))

        total = self._masks.shape[0] if self._masks is not None else (
            len(seq) if seq is not None else 0)
        self._frame_label.setText(f"{idx + 1}/{max(1, total)}")

    # -- misc --------------------------------------------------------------

    def _on_propagate_toggled(self, on: bool) -> None:
        if on:
            self.propagate_settings()

    def _pick_sequence(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of frames")
        if path:
            self.load_sequence(path)

    def _pick_masks(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of label images")
        if path:
            self.load_masks(path)

    def closeEvent(self, event):
        """Let a running pass finish before the widget is torn down.

        A ``QThread`` collected while it is still running aborts the whole
        process, and this panel's worker outlives the emit that produced its
        result by a few instructions.
        """
        self._stop_playback()
        worker = self._worker
        if worker is not None:
            try:
                worker.wait(5000)
            except RuntimeError:
                LOG.debug("worker already deleted", exc_info=True)
        super().closeEvent(event)


def build_timelapse_preview_card(host):
    """Build the ``Track preview`` card + panel pair.

    Mirrors ``spacr.qt.screens.hyperparam.build_hyperparam_card`` and
    ``app_screen._build_live_preview_card``: returns the pair without adding
    it to any layout, so the host screen puts it in whatever splitter it
    likes and starts it hidden behind the toggle.

    :param host: the :class:`AppScreen` asking for the card.
    :returns: ``(panel, card)``.
    """
    from .card import Card
    card = Card(title="Track preview")
    panel = TimelapsePreviewPanel(card)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(320)
    return panel, card
