"""Motility live preview — tracks, velocity, straightness, infection split.

Point it at a plate folder holding ``merged/*.npy`` (or at the ``merged``
folder itself) and it rebuilds, for a handful of frames, exactly what
:func:`spacr.timelapse.automated_motility_assay` computes: per-track
velocity and straightness from the cell-mask centroids, split by infection
state read off the pathogen mask.

Two things this preview refuses to fudge.

**Units.** The assay converts pixels per frame into physical units with
``factor = (1 / pixels_per_um) * (60 / seconds_per_frame)`` and only when
*both* are known; otherwise it reports ``px/frame``. A preview that printed
"velocity 4.2" while the user was thinking in µm/s would be worse than
printing nothing, so the calibration fields start **unset**, every velocity
carries its unit, and while the calibration is unknown the panel says so in
words instead of quietly borrowing the 1.78 px/µm default.

**Short tracks.** Mean step length and straightness are wildly unstable on a
three-point track — straightness is exactly 1.0 for any two-point track, no
matter what the cell did. So the track-length distribution is drawn beside
the velocity plot with the cutoff marked, and the minimum length is a live
setting: it is the knob that actually decides whether the numbers mean
anything.

The expensive half (reading merged arrays and extracting centroids) runs
once in a worker thread and is **cached** as a point table. Every metric
setting — minimum length, max displacement, straightness threshold, and both
calibration fields — recomputes from that cached table on the GUI thread,
instantly.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout,
    QWidget,
)
from .preview_controls import (
    DEFAULT_MAX_SETS, MAX_SETS_TOOLTIP, FlatButton, FlatComboBox, FlatSpinBox,
    ImageSet, ImageSetSampler, configure_max_sets_box,
    populate_channel_combo, selected_channel,
)
from .toggle import Toggle
from ..job_runner import JobRunner

from .live_preview import numpy_to_qpixmap

LOG = logging.getLogger("spacr.qt.motility_preview")

#: Objects the assay can track. ``tracked_object`` in the settings dict.
TRACKED_OBJECTS = ("cell", "nucleus", "pathogen")

#: Colours for the infection split, used by both the plot and the legend.
INFECTED_RGB = (0.92, 0.35, 0.35)
UNINFECTED_RGB = (0.35, 0.65, 0.95)


class MotilityInputError(ValueError):
    """The chosen folder is not a usable motility input, with the reason."""


# ---------------------------------------------------------------------------
# Units — stated, never assumed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Pixel size and frame interval, either of which may be unknown.

    :ivar pixels_per_um: image scale in px/µm, or ``None`` when unset.
    :ivar seconds_per_frame: frame interval in seconds, or ``None``.
    """
    pixels_per_um: Optional[float] = None
    seconds_per_frame: Optional[float] = None

    @property
    def known(self) -> bool:
        """Whether physical units can be reported at all."""
        return (self.pixels_per_um is not None
                and self.seconds_per_frame is not None
                and self.pixels_per_um > 0 and self.seconds_per_frame > 0)

    @property
    def factor(self) -> float:
        """Multiplier from px/frame to :attr:`unit`.

        Identical to the assay's own conversion, so a preview number and a
        run number are the same number.
        """
        if not self.known:
            return 1.0
        return (1.0 / float(self.pixels_per_um)) * (
            60.0 / float(self.seconds_per_frame))

    @property
    def unit(self) -> str:
        """``"µm/min"`` when calibrated, else ``"px/frame"``."""
        return "µm/min" if self.known else "px/frame"

    def caveat(self) -> str:
        """The sentence shown when the calibration is incomplete."""
        if self.known:
            return ""
        missing = []
        if not (self.pixels_per_um and self.pixels_per_um > 0):
            missing.append("pixel size (pixels_per_um)")
        if not (self.seconds_per_frame and self.seconds_per_frame > 0):
            missing.append("frame interval (seconds_per_frame)")
        return (
            "Calibration incomplete — " + " and ".join(missing) +
            " not set. Velocities below are in px/frame, NOT µm/min; set both "
            "fields to convert.")


# ---------------------------------------------------------------------------
# Reading merged arrays — lazily, and only a few frames of them
# ---------------------------------------------------------------------------

def resolve_merged_dir(path) -> str:
    """Return the ``merged`` directory for ``path``.

    Accepts the plate folder (which holds ``merged/``) or the ``merged``
    folder itself, which is what a user dragging a folder in will most
    likely grab.

    :raises MotilityInputError: when neither exists or it holds no ``.npy``.
    """
    p = Path(path)
    if not p.exists():
        raise MotilityInputError(f"No such folder: {p}")
    if not p.is_dir():
        p = p.parent
    candidate = p / "merged" if (p / "merged").is_dir() else p
    files = sorted(f for f in candidate.iterdir()
                   if f.is_file() and f.suffix.lower() == ".npy")
    if not files:
        raise MotilityInputError(
            f"{p.name} holds no merged/*.npy arrays. Point the preview at a "
            "plate folder produced by the Timelapse module.")
    return str(candidate)


def group_merged_files(merged_dir: str) -> "Dict[tuple, List[dict]]":
    """Group ``merged/*.npy`` by (plate, well, field) and sort each by time.

    Reuses :func:`spacr.timelapse._parse_merged_filename` so the preview and
    the assay agree about what a filename means.
    """
    from spacr.timelapse import _parse_merged_filename
    groups: "Dict[tuple, List[dict]]" = {}
    for name in sorted(os.listdir(merged_dir)):
        if not name.endswith(".npy"):
            continue
        meta = _parse_merged_filename(name)
        key = (meta["plateID"], meta["wellID"], meta["fieldID"])
        groups.setdefault(key, []).append(meta)
    for metas in groups.values():
        metas.sort(key=lambda m: m["timeID"])
    return {k: v for k, v in groups.items() if len(v) >= 2}


def default_plane_layout(n_planes: int, n_channels: int) -> "Tuple[int, Optional[int]]":
    """Guess (tracked mask plane, pathogen mask plane) from the plane count.

    Mirrors the layout :func:`spacr.timelapse._load_masks_from_merged`
    documents: intensity channels first, then the cell mask, then optionally
    a nucleus mask, then optionally the pathogen mask.

    :returns: ``(cell_plane, pathogen_plane_or_None)``.
    """
    n_channels = max(1, int(n_channels))
    if n_planes <= n_channels:
        return max(0, n_planes - 1), None
    if n_planes == n_channels + 1:
        return n_channels, None
    if n_planes == n_channels + 2:
        return n_channels, n_channels + 1
    return n_channels, n_channels + 2


def _plane(arr: np.ndarray, index: int, n_channels: int) -> np.ndarray:
    """Return plane ``index`` of a merged array in either stored orientation."""
    from spacr.timelapse import _reorient_merged_array
    oriented, planes, _h, _w = _reorient_merged_array(arr, n_channels=n_channels)
    return np.asarray(oriented[int(index) % planes])


def build_point_table(merged_dir: str, metas: "List[dict]", n_channels: int,
                      tracked_plane: int, pathogen_plane: Optional[int],
                      max_frames: int = 12):
    """Extract one row per object per frame from a group of merged arrays.

    Loads each ``.npy`` memory-mapped and touches only the two planes it
    needs, so a 10-plane 2048² merged array costs two planes of I/O, not ten.
    Objects keep their mask label as ``cellID`` — merged arrays written by
    the Timelapse module are already relabelled by track id, which is the
    same assumption the assay makes.

    :returns: DataFrame with ``plateID``, ``wellID``, ``fieldID``,
        ``cellID``, ``frame``, ``x``, ``y``, ``area`` and ``infected``.
    """
    import pandas as pd
    from skimage.measure import regionprops_table

    rows = []
    for t, meta in enumerate(metas[:max(2, int(max_frames))]):
        path = os.path.join(merged_dir, meta["filename"])
        arr = np.load(path, mmap_mode="r")
        if np.asarray(arr).ndim != 3:
            continue
        labels = _plane(arr, tracked_plane, n_channels).astype(np.int32)
        if not labels.any():
            continue
        props = regionprops_table(labels, properties=("label", "centroid",
                                                      "area"))
        df = pd.DataFrame(props).rename(columns={
            "centroid-0": "y", "centroid-1": "x", "label": "cellID"})
        if pathogen_plane is None:
            df["infected"] = False
        else:
            pat = _plane(arr, pathogen_plane, n_channels)
            hit = np.asarray(pat) > 0
            infected_labels = set(np.unique(labels[hit]).tolist()) - {0}
            df["infected"] = df["cellID"].isin(infected_labels)
        df["frame"] = t
        df["plateID"] = meta["plateID"]
        df["wellID"] = meta["wellID"]
        df["fieldID"] = meta["fieldID"]
        rows.append(df[["plateID", "wellID", "fieldID", "cellID", "frame",
                        "x", "y", "area", "infected"]])
    if not rows:
        raise MotilityInputError(
            "No labelled objects found in the tracked mask plane — check the "
            "channel count and the mask plane index.")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics — cheap, recomputed live from the cached point table
# ---------------------------------------------------------------------------

TRACK_KEYS = ["plateID", "wellID", "fieldID", "cellID"]


def smooth_and_filter_tracks(points, max_displacement: float):
    """Apply the assay's centroid QC: fix teleports, drop impossible tracks.

    A single frame whose steps in and out both exceed ``max_displacement``
    while its neighbours are within it is a segmentation glitch, and the
    assay interpolates it. Any *remaining* step over the limit means the
    track links two different objects, and the assay drops the whole track.
    This reproduces both, so the preview's track count matches a run's.

    :returns: ``(points, n_glitches_fixed, n_tracks_dropped)``.
    """
    import pandas as pd
    if points is None or points.empty:
        return points, 0, 0
    limit = float(max_displacement)
    kept: List[Any] = []
    glitches = 0
    dropped = 0
    for _key, g in points.sort_values(TRACK_KEYS + ["frame"]).groupby(
            TRACK_KEYS, sort=False):
        g = g.copy()
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        n = x.size
        glitch_at = set()
        for i in range(1, n - 1):
            d_prev = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
            d_next = float(np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]))
            d_neigh = float(np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]))
            if d_prev > limit and d_next > limit and d_neigh <= limit:
                glitch_at.add(i)
        for i in sorted(glitch_at):
            x[i] = 0.5 * (x[i - 1] + x[i + 1])
            y[i] = 0.5 * (y[i - 1] + y[i + 1])
            glitches += 1
        impossible = False
        for i in range(1, n):
            d = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
            if d > limit and i not in glitch_at and (i - 1) not in glitch_at:
                impossible = True
                break
        if impossible:
            dropped += 1
            continue
        g["x"] = x
        g["y"] = y
        kept.append(g)
    if not kept:
        return points.iloc[0:0], glitches, dropped
    return pd.concat(kept, ignore_index=True), glitches, dropped


def track_metrics(points, calibration: Calibration, min_length: int = 3):
    """Per-track velocity and straightness, using the assay's own formulae.

    ``v_px_per_frame`` is the mean step length; ``straightness`` is net
    displacement over path length. Velocity is ``v_px_per_frame * factor``
    and carries :attr:`Calibration.unit`.

    Tracks shorter than ``min_length`` frames are kept in the table and
    flagged ``too_short`` rather than silently dropped, so the length
    distribution plot can show what the cutoff is discarding.
    """
    import pandas as pd
    cols = TRACK_KEYS + ["n_frames", "v_px_per_frame", "velocity",
                         "velocity_unit", "straightness", "path_length",
                         "net_displacement", "infected", "too_short"]
    if points is None or points.empty:
        return pd.DataFrame(columns=cols)

    factor = calibration.factor
    unit = calibration.unit
    records = []
    for key, g in points.sort_values(TRACK_KEYS + ["frame"]).groupby(
            TRACK_KEYS, sort=False):
        g = g.sort_values("frame")
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        if x.size < 2:
            continue
        d = np.hypot(np.diff(x), np.diff(y))
        if d.size == 0 or not np.isfinite(d).any():
            continue
        v_px = float(np.nanmean(d))
        path_length = float(np.nansum(d))
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        straightness = net / path_length if path_length > 0 else np.nan
        rec = dict(zip(TRACK_KEYS, key))
        rec.update({
            "n_frames": int(x.size),
            "v_px_per_frame": v_px,
            "velocity": v_px * factor,
            "velocity_unit": unit,
            "straightness": straightness,
            "path_length": path_length,
            "net_displacement": net,
            "infected": bool(g["infected"].any()),
            "too_short": bool(x.size < int(min_length)),
        })
        records.append(rec)
    return pd.DataFrame(records, columns=cols)


@dataclass
class MotilitySummary:
    """Everything the panel pins under the plots."""
    n_tracks: int = 0
    n_used: int = 0
    n_short: int = 0
    min_length: int = 3
    unit: str = "px/frame"
    calibrated: bool = False
    mean_velocity: float = float("nan")
    mean_velocity_infected: float = float("nan")
    mean_velocity_uninfected: float = float("nan")
    mean_straightness: float = float("nan")
    n_infected: int = 0
    n_uninfected: int = 0
    n_high_straightness: int = 0
    straightness_threshold: float = 0.95
    glitches_fixed: int = 0
    tracks_dropped: int = 0

    def summary(self) -> str:
        """One monospace block: counts, then velocities *with their unit*."""
        def _f(v):
            return "n/a" if not np.isfinite(v) else f"{v:.3g}"
        return (
            f"{self.n_tracks} tracks · {self.n_used} at or above "
            f"{self.min_length} frames ({self.n_short} shorter, excluded) · "
            f"{self.n_infected} infected / {self.n_uninfected} uninfected\n"
            f"mean velocity {_f(self.mean_velocity)} {self.unit} · "
            f"infected {_f(self.mean_velocity_infected)} {self.unit} · "
            f"uninfected {_f(self.mean_velocity_uninfected)} {self.unit}\n"
            f"mean straightness {_f(self.mean_straightness)} · "
            f"{self.n_high_straightness} tracks at or above "
            f"{self.straightness_threshold:.2f} (drift/artefact candidates) · "
            f"{self.glitches_fixed} centroid glitches fixed, "
            f"{self.tracks_dropped} tracks dropped as impossible"
        )


def summarise(tracks, calibration: Calibration, min_length: int,
              straightness_threshold: float, glitches: int = 0,
              dropped: int = 0) -> MotilitySummary:
    """Reduce a per-track table to the numbers shown under the plots."""
    s = MotilitySummary(min_length=int(min_length), unit=calibration.unit,
                        calibrated=calibration.known,
                        straightness_threshold=float(straightness_threshold),
                        glitches_fixed=int(glitches),
                        tracks_dropped=int(dropped))
    if tracks is None or tracks.empty:
        return s
    s.n_tracks = int(len(tracks))
    used = tracks[~tracks["too_short"]]
    s.n_used = int(len(used))
    s.n_short = s.n_tracks - s.n_used
    if used.empty:
        return s
    s.mean_velocity = float(used["velocity"].mean())
    inf = used[used["infected"]]
    uninf = used[~used["infected"]]
    s.n_infected = int(len(inf))
    s.n_uninfected = int(len(uninf))
    s.mean_velocity_infected = (float(inf["velocity"].mean()) if len(inf)
                                else float("nan"))
    s.mean_velocity_uninfected = (float(uninf["velocity"].mean())
                                  if len(uninf) else float("nan"))
    s.mean_straightness = float(used["straightness"].mean())
    s.n_high_straightness = int(
        (used["straightness"] >= float(straightness_threshold)).sum())
    return s


# ---------------------------------------------------------------------------
# Plot — matplotlib Agg into an RGB array (no Qt backend needed)
# ---------------------------------------------------------------------------

def render_motility_figure(points, tracks, calibration: Calibration,
                           min_length: int, straightness_threshold: float,
                           width_px: int = 1180,
                           height_px: int = 380) -> np.ndarray:
    """Render the three panels the user tunes against, as an RGB array.

    1. **Tracks**, drawn from the origin so paths are comparable, coloured
       by infection state.
    2. **Track-length distribution**, with the minimum-length cutoff drawn
       on it — this is the plot that says whether the velocity numbers can
       be trusted.
    3. **Velocity and straightness**, split by infection state, with the
       unit in the axis label so ``px/frame`` is never mistaken for µm/s.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100.0
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#161719")
    axes = fig.subplots(1, 3)
    for ax in axes:
        ax.set_facecolor("#161719")
        for spine in ax.spines.values():
            spine.set_color("#666666")
        ax.tick_params(colors="#cccccc", labelsize=7)
        ax.xaxis.label.set_color("#dddddd")
        ax.yaxis.label.set_color("#dddddd")
        ax.title.set_color("#ffffff")

    ax_tracks, ax_len, ax_vel = axes

    # 1 — origin-centred tracks
    ax_tracks.set_title("Tracks (from origin)", fontsize=8)
    ax_tracks.set_xlabel("Δx (px)", fontsize=7)
    ax_tracks.set_ylabel("Δy (px)", fontsize=7)
    if points is not None and not points.empty:
        infected_by_track = {}
        if tracks is not None and not tracks.empty:
            infected_by_track = {
                tuple(r[k] for k in TRACK_KEYS): bool(r["infected"])
                for _i, r in tracks.iterrows()}
        for key, g in points.sort_values(TRACK_KEYS + ["frame"]).groupby(
                TRACK_KEYS, sort=False):
            x = g["x"].to_numpy(dtype=float)
            y = g["y"].to_numpy(dtype=float)
            if x.size < 2:
                continue
            colour = (INFECTED_RGB if infected_by_track.get(tuple(key), False)
                      else UNINFECTED_RGB)
            ax_tracks.plot(x - x[0], y - y[0], color=colour, linewidth=0.9,
                           alpha=0.85)
        ax_tracks.axhline(0, color="#555555", linewidth=0.5)
        ax_tracks.axvline(0, color="#555555", linewidth=0.5)

    # 2 — track-length distribution with the cutoff marked
    ax_len.set_title("Track length", fontsize=8)
    ax_len.set_xlabel("frames per track", fontsize=7)
    ax_len.set_ylabel("tracks", fontsize=7)
    if tracks is not None and not tracks.empty:
        lengths = tracks["n_frames"].to_numpy(dtype=float)
        bins = max(4, int(min(24, np.ptp(lengths) + 1)))
        ax_len.hist(lengths, bins=bins, color="#8899aa", edgecolor="#161719")
        ax_len.axvline(float(min_length), color="#ffcc44", linewidth=1.4,
                       linestyle="--")
        ax_len.text(0.98, 0.95, f"min = {int(min_length)}",
                    transform=ax_len.transAxes, ha="right", va="top",
                    color="#ffcc44", fontsize=7)

    # 3 — velocity + straightness, split by infection, unit stated
    ax_vel.set_title("Velocity by infection state", fontsize=8)
    ax_vel.set_ylabel(f"velocity ({calibration.unit})", fontsize=7)
    if tracks is not None and not tracks.empty:
        used = tracks[~tracks["too_short"]]
        groups = [used.loc[used["infected"], "velocity"].to_numpy(dtype=float),
                  used.loc[~used["infected"], "velocity"].to_numpy(dtype=float)]
        labels = [f"infected (n={groups[0].size})",
                  f"uninfected (n={groups[1].size})"]
        colours = [INFECTED_RGB, UNINFECTED_RGB]
        for i, (vals, colour) in enumerate(zip(groups, colours)):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            jitter = (np.linspace(-0.16, 0.16, vals.size)
                      if vals.size > 1 else np.zeros(1))
            ax_vel.scatter(np.full(vals.size, i) + jitter, vals, s=9,
                           color=colour, alpha=0.8, linewidths=0)
            ax_vel.hlines(float(np.mean(vals)), i - 0.25, i + 0.25,
                          color="#ffffff", linewidth=1.2)
        ax_vel.set_xticks([0, 1])
        ax_vel.set_xticklabels(labels, fontsize=7)
        ax_vel.set_xlim(-0.6, 1.6)
    if not calibration.known:
        ax_vel.text(0.5, 0.02, "uncalibrated — px/frame",
                    transform=ax_vel.transAxes, ha="center", va="bottom",
                    color="#ffcc44", fontsize=7)

    fig.tight_layout(pad=0.8)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[..., :3])


# ---------------------------------------------------------------------------
# Worker — the expensive half only
# ---------------------------------------------------------------------------

@dataclass
class MotilityRequest:
    """One read of merged arrays into a point table."""
    merged_dir: str = ""
    metas: List[dict] = field(default_factory=list)
    n_channels: int = 4
    tracked_plane: int = 4
    pathogen_plane: Optional[int] = None
    max_frames: int = 12


def run_motility_pass(req: MotilityRequest):
    """Build the cached point table for one (plate, well, field) group."""
    return build_point_table(
        req.merged_dir, req.metas, req.n_channels, req.tracked_plane,
        req.pathogen_plane, max_frames=req.max_frames)


class _MotilityWorker(QThread):
    """Reads merged arrays off the GUI thread.

    Same contract as ``live_preview._PreviewWorker``: every exception is
    caught inside :meth:`run` and emitted as a string; nothing escapes.
    """

    finished_result = Signal(object, str)   # (DataFrame or None, error)

    def __init__(self, request: MotilityRequest, parent=None):
        super().__init__(parent)
        self._request = request

    def run(self):
        try:
            self.finished_result.emit(run_motility_pass(self._request), "")
        except Exception as e:
            LOG.info("motility preview failed: %s", e, exc_info=True)
            self.finished_result.emit(None, str(e))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def scan_plate_payload(path) -> Dict[str, Any]:
    """Resolve a plate's ``merged`` folder and group it. No Qt: worker-safe.

    The expensive half of opening a plate: ``resolve_merged_dir`` lists the
    candidate folder and ``group_merged_files`` reads every name in
    ``merged/`` and parses it -- thousands of entries on a 384-well plate.

    :returns: ``{path, merged, groups, error}``.
    """
    out: Dict[str, Any] = {"path": str(path), "merged": None,
                           "groups": None, "error": ""}
    try:
        merged = resolve_merged_dir(path)
        groups = group_merged_files(merged)
    except Exception as exc:
        out["error"] = f"Load failed: {exc}"
        return out
    out["merged"] = merged
    out["groups"] = groups
    return out


class MotilityPreviewPanel(QWidget):
    """Interactive motility preview — Motility Assay module.

    Same contract as :class:`~spacr.qt.widgets.live_preview.LivePreviewPanel`:
    standalone ``QWidget``, ``QThread`` worker emitting over signals,
    :meth:`set_propagate_callback` to push tuned values back into the main
    settings panel, and a ``build_*_card`` factory.
    """

    preview_ready = Signal(object)   # MotilitySummary, or None on failure

    def __init__(self, parent=None, *, threaded: bool = True):
        super().__init__(parent)
        # Scanning a plate lists every file in `merged/` and parses each name:
        # thousands of entries on a 384-well plate, and not GUI-thread work.
        # `threaded=False` runs each job inline, emitting the same signals in
        # the same order, so a test can drive this panel synchronously without
        # the behaviour diverging.
        self._jobs = JobRunner(self, threaded=threaded,
                               app_key="motility preview")
        #: Bumped whenever a newer scan supersedes the one in flight.
        self._load_token = 0
        self._points = None          # cached — the expensive half
        self._tracks = None
        self._summary: Optional[MotilitySummary] = None
        self._groups: "Dict[tuple, List[dict]]" = {}
        self._merged_dir: str = ""
        self._worker: Optional[_MotilityWorker] = None
        self._propagate_cb = None
        # Bounded, reproducible sample of the plate's time series — the
        # dropdown never lists them all. See ImageSetSampler.
        self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
        self._build_ui()
        self.setAcceptDrops(True)

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # FOV and channel dropdowns sit immediately LEFT of the Choose
        # control; all three wear the flat "Live toggle" look.
        pick = QHBoxLayout()
        self._pick_row = pick
        self._path_label = QLabel(
            "No plate loaded — drop a folder holding merged/*.npy here, "
            "or choose one")
        self._path_label.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Preferred)
        self._max_sets_box = FlatSpinBox(self, value=DEFAULT_MAX_SETS,
                                         tooltip=MAX_SETS_TOOLTIP)
        self._max_sets_box.valueChanged.connect(self._on_max_sets_changed)
        self._fov_box = FlatComboBox(
            self,
            tooltip=("Field of view — the (plate, well, field) group "
                     "previewed. Each group is one time series. Lists a "
                     "random sample of the plate, not all of it."))
        self._fov_box.currentIndexChanged.connect(self._on_group_changed)
        # Kept under its historical name for the integrations and tests that
        # already drive it.
        self._group_box = self._fov_box
        self._channel_box = FlatComboBox(
            self,
            tooltip=("Plane of the merged array the preview reads its objects "
                     "from. Bound to 'Tracked mask plane'; changing it drops "
                     "the cached point table, so run the preview to see it."))
        self._channel_box.currentIndexChanged.connect(
            self._on_display_channel_changed)
        self._pick_btn = FlatButton("Choose plate folder…", self)
        self._pick_btn.clicked.connect(self._pick_folder)
        pick.addWidget(self._path_label, 1)
        pick.addWidget(self._max_sets_box)
        pick.addWidget(self._fov_box)
        pick.addWidget(self._channel_box)
        pick.addWidget(self._pick_btn)
        root.addLayout(pick)

        # -- array layout (changing one re-reads the merged arrays) ----------
        self._tracked_object = QComboBox(self)
        self._tracked_object.addItems(list(TRACKED_OBJECTS))
        self._tracked_object.setToolTip(
            "(str) tracked_object — which mask the tracks come from.")
        self._n_channels = QSpinBox(self)
        self._n_channels.setRange(1, 16)
        self._n_channels.setValue(4)
        self._n_channels.setToolTip(
            "(int) How many intensity channels the merged arrays hold; the "
            "mask planes follow them.")
        self._tracked_plane = QSpinBox(self)
        self._tracked_plane.setRange(0, 32)
        self._tracked_plane.setValue(4)
        self._tracked_plane.setToolTip(
            "(int) Plane index of the tracked object's mask.")
        self._pathogen_plane = QSpinBox(self)
        self._pathogen_plane.setRange(-1, 32)
        self._pathogen_plane.setValue(-1)
        self._pathogen_plane.setSpecialValueText("none")
        self._pathogen_plane.setToolTip(
            "(int) Plane index of the pathogen mask, used to split tracks by "
            "infection state. 'none' treats every track as uninfected.")
        self._max_frames = QSpinBox(self)
        self._max_frames.setRange(2, 500)
        self._max_frames.setValue(12)
        self._max_frames.setToolTip(
            "(int) How many frames of the series the preview reads. The rest "
            "is never loaded.")

        # -- metrics (live — recomputed from the cached point table) ---------
        self._min_len = QSpinBox(self)
        self._min_len.setRange(2, 500)
        self._min_len.setValue(3)
        self._min_len.setToolTip(
            "(int) Tracks shorter than this are excluded from the velocity "
            "and straightness numbers. Short tracks give unstable values — a "
            "two-point track always has straightness 1.0.")
        self._max_disp = QDoubleSpinBox(self)
        self._max_disp.setRange(1.0, 5000.0)
        self._max_disp.setValue(50.0)
        self._max_disp.setSuffix(" px")
        self._max_disp.setToolTip(
            "(float, px) max_displacement — single-frame teleports are "
            "interpolated, and any track that still jumps further than this "
            "is dropped as an impossible link.")
        self._straightness = QDoubleSpinBox(self)
        self._straightness.setRange(0.0, 1.0)
        self._straightness.setSingleStep(0.01)
        self._straightness.setValue(0.95)
        self._straightness.setToolTip(
            "(float) straightness_threshold — tracks at or above this are "
            "flagged as stage-drift / artefact candidates.")
        self._straightness_filter = Toggle(
            "Drop over-straight tracks", self)
        self._straightness_filter.setToolTip(
            "(bool) straightness_filter — remove the flagged tracks entirely.")

        self._pixels_per_um = QDoubleSpinBox(self)
        self._pixels_per_um.setRange(0.0, 1000.0)
        self._pixels_per_um.setDecimals(3)
        self._pixels_per_um.setValue(0.0)
        self._pixels_per_um.setSpecialValueText("unknown")
        self._pixels_per_um.setToolTip(
            "(float, px/µm) pixels_per_um. Left unknown, velocities are "
            "reported in px/frame — the preview will not invent a scale.")
        self._seconds_per_frame = QDoubleSpinBox(self)
        self._seconds_per_frame.setRange(0.0, 100000.0)
        self._seconds_per_frame.setDecimals(2)
        self._seconds_per_frame.setValue(0.0)
        self._seconds_per_frame.setSpecialValueText("unknown")
        self._seconds_per_frame.setToolTip(
            "(float, s) seconds_per_frame. Left unknown, velocities are "
            "reported in px/frame — the preview will not invent an interval.")

        layout_group = QGroupBox("Merged arrays (changing these re-reads)")
        lform = QFormLayout(layout_group)
        lform.addRow("Tracked object", self._tracked_object)
        lform.addRow("Intensity channels", self._n_channels)
        lform.addRow("Tracked mask plane", self._tracked_plane)
        lform.addRow("Pathogen mask plane", self._pathogen_plane)
        lform.addRow("Frames previewed", self._max_frames)

        metric_group = QGroupBox("Metrics (live — recomputed from the cache)")
        mform = QFormLayout(metric_group)
        mform.addRow("Min track length", self._min_len)
        mform.addRow("Max displacement", self._max_disp)
        mform.addRow("Straightness threshold", self._straightness)
        mform.addRow(self._straightness_filter)

        cal_group = QGroupBox("Calibration (units)")
        cform = QFormLayout(cal_group)
        cform.addRow("Pixels per µm", self._pixels_per_um)
        cform.addRow("Seconds per frame", self._seconds_per_frame)
        self._unit_label = QLabel("", self)
        self._unit_label.setWordWrap(True)
        cform.addRow(self._unit_label)

        groups = QHBoxLayout()
        groups.addWidget(layout_group, 1)
        groups.addWidget(metric_group, 1)
        groups.addWidget(cal_group, 1)
        root.addLayout(groups)

        for w in (self._min_len, self._max_disp, self._straightness,
                  self._pixels_per_um, self._seconds_per_frame):
            w.valueChanged.connect(self._on_metric_changed)
        self._straightness_filter.toggled.connect(self._on_metric_changed)
        self._tracked_object.currentTextChanged.connect(
            self._on_tracked_object_changed)
        # One plane, two surfaces: the settings spinner and the flat dropdown
        # in the pick row stay in step.
        self._tracked_plane.valueChanged.connect(
            self._sync_plane_combo_from_spin)

        act = QHBoxLayout()
        self._run_btn = QPushButton("Run preview", self)
        self._run_btn.clicked.connect(self.run_preview)
        self._propagate_btn = QPushButton("Propagate settings", self)
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, the settings tuned here are copied into the main "
            "Motility Assay settings so the run uses them.")
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        self._status = QLabel("", self)
        act.addWidget(self._run_btn)
        act.addWidget(self._propagate_btn)
        act.addWidget(self._status, 1)
        root.addLayout(act)

        self._plot = QLabel(self)
        self._plot.setAlignment(Qt.AlignCenter)
        self._plot.setMinimumHeight(240)
        self._plot.setStyleSheet("background: #161719;")
        root.addWidget(self._plot, 1)

        self._stats_label = QLabel(
            "Load a plate folder and run the preview.", self)
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("font-family: monospace;")
        root.addWidget(self._stats_label)

        self._refresh_unit_label()

    # -- drag & drop -------------------------------------------------------

    def _dropped_path(self, event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):    # noqa: N802
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
        self.load_folder_async(p)

    # -- public API --------------------------------------------------------

    @property
    def _loads_in_flight(self) -> List[int]:
        """Outstanding scans, as a list so ``not ...`` reads naturally."""
        runner = getattr(self, "_jobs", None)
        return [] if runner is None else [0] * runner.pending_jobs()

    def load_folder_async(self, path) -> bool:
        """Scan a plate on a worker, then install it on the GUI thread.

        Both GUI entry points -- the drop handler and the Choose-plate dialog
        -- come through here.

        :returns: ``True`` when a job was submitted.
        """
        text = os.fspath(path).strip() if path is not None else ""
        if not text:
            return False
        self._load_token += 1
        token = self._load_token
        self._status.setText(f"Scanning {os.path.basename(text)}…")
        self._jobs.submit(
            lambda: scan_plate_payload(text),
            lambda payload, _t=token: self._on_plate_scanned(_t, payload))
        return True

    def _on_plate_scanned(self, token: int, payload) -> None:
        """Install a scanned plate. Always on the GUI thread."""
        if token != self._load_token or not isinstance(payload, dict):
            return
        if payload.get("error"):
            self._status.setText(payload["error"])
            return
        groups = payload.get("groups")
        if not groups:
            self._status.setText(
                "No (plate, well, field) group has two or more time points — "
                "a motility preview needs a time series.")
            return
        self._install_plate(payload["merged"], groups)

    def shutdown(self) -> None:
        """Abandon anything in flight and leave no QThread behind."""
        runner = getattr(self, "_jobs", None)
        if runner is not None:
            runner.shutdown()

    def load_folder(self, path) -> bool:
        """Synchronously open a plate (or ``merged``) folder.

        For programmatic callers and tests, mirroring
        ``LivePreviewPanel.load_image``. The GUI uses :meth:`load_folder_async`.
        """
        payload = scan_plate_payload(path)
        if payload["error"]:
            self._status.setText(payload["error"])
            return False
        groups = payload["groups"]
        if not groups:
            self._status.setText(
                "No (plate, well, field) group has two or more time points — "
                "a motility preview needs a time series.")
            return False
        self._install_plate(payload["merged"], groups)
        return True

    def _install_plate(self, merged: str, groups) -> bool:
        """Adopt an already-scanned plate and refresh the selectors."""
        self._merged_dir = merged
        self._groups = groups
        self._points = None
        self._tracks = None
        self._sampler.invalidate()
        self._populate_group_box()
        self._autodetect_planes()
        self._refresh_source_selectors()
        self._path_label.setText(
            f"{os.path.basename(os.path.dirname(merged.rstrip(os.sep)) or merged)}"
            f"  ·  {len(groups)} group(s)")
        self._status.setText(
            f"{len(groups)} time series found — {self.sample_note()}. "
            "Run the preview.")
        return True

    def _populate_group_box(self) -> None:
        """Fill the groups dropdown with a bounded random sample of the plate.

        A 384-well plate produces thousands of time series and listing them
        all made the dropdown, and every refresh of it, cost more than the
        preview itself. The sample is drawn across the whole plate and is
        reproducible — see
        :class:`~spacr.qt.widgets.preview_controls.ImageSetSampler`.

        The dropdown stores each group's ``(plate, well, field)`` **key** as
        item data, not a path, so it populates itself rather than going
        through :func:`apply_sample_to_combo`.
        """
        if self._sampler.directory != self._merged_dir:
            self._sampler.adopt(
                self._merged_dir,
                [ImageSet(key=key, directory=self._merged_dir,
                          channels={"": metas[0]["filename"]})
                 for key, metas in self._groups.items()],
                [])
        self._sampler.set_max(
            configure_max_sets_box(self._max_sets_box, self._sampler.total))
        current = self._group_box.currentData()
        keep = next((s for s in self._sampler.sets if s.key == current), None)
        shown = self._sampler.sample(keep=keep)
        self._sample_note = self._sampler.describe(len(shown))
        blocked = self._group_box.blockSignals(True)
        try:
            self._group_box.clear()
            for item in shown:
                metas = self._groups.get(item.key) or []
                self._group_box.addItem(
                    f"{item.key[0]} {item.key[1]} f{item.key[2]} "
                    f"({len(metas)} frames)", item.key)
            index = self._group_box.findData(current)
            if index >= 0:
                self._group_box.setCurrentIndex(index)
        finally:
            self._group_box.blockSignals(blocked)
        self._group_box.setToolTip(
            f"Field of view — {self._sample_note}.\n\n{MAX_SETS_TOOLTIP}")

    def sample_note(self) -> str:
        """The sentence stating this preview is a sample of N of M sets."""
        return getattr(self, "_sample_note", "")

    def _on_max_sets_changed(self, value: int) -> None:
        """Draw a new sample at the user's new cap — without re-grouping."""
        if not self._sampler.set_max(int(value)):
            return
        self._populate_group_box()
        if self.sample_note():
            self._status.setText(
                self.sample_note()[:1].upper() + self.sample_note()[1:])

    def set_propagate_callback(self, cb) -> None:
        """Register a ``callback(dict)`` used to push tuned settings back."""
        self._propagate_cb = cb

    def calibration(self) -> Calibration:
        """The current calibration, with unset fields as ``None``."""
        ppu = float(self._pixels_per_um.value())
        spf = float(self._seconds_per_frame.value())
        return Calibration(pixels_per_um=ppu if ppu > 0 else None,
                           seconds_per_frame=spf if spf > 0 else None)

    def settings_for_propagation(self) -> dict:
        """Map the preview's widgets onto real Motility Assay setting keys.

        ``pixels_per_um`` / ``seconds_per_frame`` are only propagated when
        the user actually set them — pushing a fabricated calibration into
        the run would be exactly the mistake this panel exists to prevent.
        """
        cal = self.calibration()
        out: Dict[str, Any] = {
            "tracked_object": self._tracked_object.currentText(),
            "max_displacement": float(self._max_disp.value()),
            "straightness_threshold": float(self._straightness.value()),
            "straightness_filter": bool(self._straightness_filter.isChecked()),
            "channels": list(range(int(self._n_channels.value()))),
        }
        if cal.pixels_per_um is not None:
            out["pixels_per_um"] = cal.pixels_per_um
        if cal.seconds_per_frame is not None:
            out["seconds_per_frame"] = cal.seconds_per_frame
        if self._pathogen_plane.value() >= 0:
            out["pathogen_channel"] = int(self._pathogen_plane.value())
        return out

    def propagate_settings(self) -> None:
        """Push the current settings to the main panel, if wired."""
        if self._propagate_cb is not None:
            try:
                self._propagate_cb(self.settings_for_propagation())
            except Exception:
                LOG.debug("propagate_settings failed", exc_info=True)

    def apply_settings(self, settings: dict) -> None:
        """Seed the preview from the main Motility settings dict."""
        try:
            obj = settings.get("tracked_object")
            if obj and self._tracked_object.findText(str(obj)) >= 0:
                self._tracked_object.setCurrentText(str(obj))
            if settings.get("max_displacement"):
                self._max_disp.setValue(float(settings["max_displacement"]))
            if settings.get("straightness_threshold"):
                self._straightness.setValue(
                    float(settings["straightness_threshold"]))
            self._straightness_filter.setChecked(
                bool(settings.get("straightness_filter", False)))
            chans = settings.get("channels")
            if isinstance(chans, (list, tuple)) and chans:
                self._n_channels.setValue(len(chans))
            if settings.get("pixels_per_um"):
                self._pixels_per_um.setValue(float(settings["pixels_per_um"]))
            if settings.get("seconds_per_frame"):
                self._seconds_per_frame.setValue(
                    float(settings["seconds_per_frame"]))
        except Exception:
            LOG.debug("apply_settings failed", exc_info=True)

    def current_params(self) -> dict:
        """Snapshot for tests + external callers."""
        cal = self.calibration()
        return {
            "tracked_object": self._tracked_object.currentText(),
            "n_channels": int(self._n_channels.value()),
            "tracked_plane": int(self._tracked_plane.value()),
            "pathogen_plane": (int(self._pathogen_plane.value())
                               if self._pathogen_plane.value() >= 0 else None),
            "min_length": int(self._min_len.value()),
            "max_displacement": float(self._max_disp.value()),
            "unit": cal.unit,
            "calibrated": cal.known,
            "n_tracks": 0 if self._tracks is None else int(len(self._tracks)),
            "display_channel": self.display_channel(),
            "fov": self._fov_box.currentText(),
        }

    # -- running -----------------------------------------------------------

    def run_preview(self) -> None:
        """Read the merged arrays into the cached point table, then score."""
        if not self._groups:
            self._status.setText("Load a plate folder first.")
            return
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Preview already running.")
            return
        key = self._group_box.currentData()
        metas = self._groups.get(key) or next(iter(self._groups.values()))
        pat = int(self._pathogen_plane.value())
        req = MotilityRequest(
            merged_dir=self._merged_dir,
            metas=metas,
            n_channels=int(self._n_channels.value()),
            tracked_plane=int(self._tracked_plane.value()),
            pathogen_plane=pat if pat >= 0 else None,
            max_frames=int(self._max_frames.value()),
        )
        self._run_btn.setEnabled(False)
        self._status.setText("Reading merged arrays…")
        worker = _MotilityWorker(req, self)
        # Bound method, not a closure — a plain callable would be invoked on
        # the worker thread and every widget touch below would be off-thread.
        worker.finished_result.connect(self._on_worker_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_worker_done(self, points, err: str) -> None:
        """Adopt the point table. Runs on the GUI thread (queued signal)."""
        self._run_btn.setEnabled(True)
        self._worker = None
        if err:
            self._status.setText(f"Preview failed: {err}")
            self.preview_ready.emit(None)
            return
        if points is None or points.empty:
            self._status.setText("No objects found in the tracked mask plane.")
            self.preview_ready.emit(None)
            return
        self._points = points
        n_frames = int(points["frame"].nunique())
        self._status.setText(
            f"Read {n_frames} frames · {len(points)} object observations — "
            "metric changes below are recomputed from this cache.")
        self.recompute()

    # -- metrics (GUI thread — cheap) --------------------------------------

    def _on_metric_changed(self, *_):
        self._refresh_unit_label()
        if self._points is not None:
            self.recompute()

    def recompute(self) -> None:
        """Re-score the cached point table. No file is re-read."""
        if self._points is None:
            return
        cal = self.calibration()
        cleaned, glitches, dropped = smooth_and_filter_tracks(
            self._points, float(self._max_disp.value()))
        tracks = track_metrics(cleaned, cal, int(self._min_len.value()))
        if self._straightness_filter.isChecked() and not tracks.empty:
            keep = tracks["straightness"] < float(self._straightness.value())
            dropped_ids = {tuple(r[k] for k in TRACK_KEYS)
                           for _i, r in tracks[~keep].iterrows()}
            tracks = tracks[keep]
            if dropped_ids and not cleaned.empty:
                mask = [tuple(r) not in dropped_ids
                        for r in cleaned[TRACK_KEYS].itertuples(index=False)]
                cleaned = cleaned[mask]
        self._tracks = tracks
        self._summary = summarise(
            tracks, cal, int(self._min_len.value()),
            float(self._straightness.value()), glitches, dropped)
        text = self._summary.summary()
        caveat = cal.caveat()
        if caveat:
            text = caveat + "\n" + text
        self._stats_label.setText(text)
        self._render_plot(cleaned, tracks, cal)
        if self._propagate_btn.isChecked():
            self.propagate_settings()
        self.preview_ready.emit(self._summary)

    def _render_plot(self, points, tracks, cal: Calibration) -> None:
        try:
            rgb = render_motility_figure(
                points, tracks, cal, int(self._min_len.value()),
                float(self._straightness.value()),
                width_px=max(480, self._plot.width() or 1180),
                height_px=max(200, self._plot.height() or 380))
        except Exception as e:
            LOG.debug("motility plot failed", exc_info=True)
            self._plot.setText(f"Plot failed: {e}")
            return
        self._plot.setPixmap(numpy_to_qpixmap(rgb))

    # -- misc --------------------------------------------------------------

    def _refresh_unit_label(self) -> None:
        cal = self.calibration()
        if cal.known:
            self._unit_label.setText(
                f"Velocities reported in {cal.unit} "
                f"(×{cal.factor:.4g} from px/frame).")
            self._unit_label.setStyleSheet("color: #9fd39f;")
        else:
            self._unit_label.setText(cal.caveat())
            self._unit_label.setStyleSheet("color: #ffcc44;")

    def _autodetect_planes(self) -> None:
        """Set the mask plane indices from the first array's plane count."""
        try:
            key = self._group_box.currentData() or next(iter(self._groups))
            first = self._groups[key][0]["filename"]
            arr = np.load(os.path.join(self._merged_dir, first), mmap_mode="r")
            n_channels = int(self._n_channels.value())
            planes = int(min(np.asarray(arr).shape))
            tracked, pathogen = default_plane_layout(planes, n_channels)
            self._tracked_plane.setValue(tracked)
            self._pathogen_plane.setValue(
                pathogen if pathogen is not None else -1)
        except Exception:
            LOG.debug("plane autodetect failed", exc_info=True)

    # -- FOV / channel selectors -------------------------------------------

    def _plane_count(self) -> int:
        """Planes held by the first merged array of the selected group."""
        try:
            key = self._fov_box.currentData() or next(iter(self._groups))
            first = self._groups[key][0]["filename"]
            arr = np.load(os.path.join(self._merged_dir, first), mmap_mode="r")
            return int(min(np.asarray(arr).shape))
        except Exception:
            LOG.debug("plane count unavailable", exc_info=True)
            return 0

    def _refresh_source_selectors(self) -> None:
        """Re-fill the channel dropdown for the selected field of view."""
        populate_channel_combo(
            self._channel_box, self._plane_count(), include_all=False,
            keep=f"Ch {int(self._tracked_plane.value())}")

    def _sync_plane_spin_from_combo(self) -> None:
        """Push the dropdown's plane into the tracked-mask-plane spinner."""
        index = selected_channel(self._channel_box)
        if index is None or int(self._tracked_plane.value()) == int(index):
            return
        self._tracked_plane.setValue(int(index))

    def _sync_plane_combo_from_spin(self, *_args) -> None:
        """Reflect a spinner-side plane change in the dropdown."""
        box = getattr(self, "_channel_box", None)
        if box is None:
            return
        index = box.findText(f"Ch {int(self._tracked_plane.value())}")
        if index < 0 or index == box.currentIndex():
            return
        blocked = box.blockSignals(True)
        try:
            box.setCurrentIndex(index)
        finally:
            box.blockSignals(blocked)

    def display_channel(self) -> Optional[int]:
        """Merged-array plane the preview reads objects from."""
        return selected_channel(self._channel_box)

    def _on_display_channel_changed(self, *_args) -> None:
        """Adopt the newly selected plane and drop the stale point table."""
        if not hasattr(self, "_tracked_plane"):
            return
        self._sync_plane_spin_from_combo()
        self._points = None
        self._tracks = None
        self._invite_rerun()

    def _invite_rerun(self) -> None:
        """Say the cache was dropped. Reading merged arrays is the expensive
        half of this panel, so it stays an explicit ``Run preview`` — never a
        side effect of touching a dropdown."""
        if self._groups:
            self._status.setText(
                "Field / plane changed — run the preview to read it.")

    def _on_group_changed(self, *_):
        self._points = None
        self._tracks = None
        self._autodetect_planes()
        self._refresh_source_selectors()
        self._invite_rerun()

    def _on_tracked_object_changed(self, name: str) -> None:
        """Move the tracked mask plane to the chosen object's slot."""
        n = int(self._n_channels.value())
        offset = {"cell": 0, "nucleus": 1, "pathogen": 2}.get(name, 0)
        self._tracked_plane.setValue(n + offset)

    def _on_propagate_toggled(self, on: bool) -> None:
        if on:
            self.propagate_settings()

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose a plate folder holding merged/*.npy")
        if path:
            self.load_folder_async(path)

    def closeEvent(self, event):
        """Let a running read finish before the widget is torn down.

        A ``QThread`` collected while running aborts the process; the worker
        outlives the emit that produced its result by a few instructions.
        """
        # Cancel the scan before waiting on the motility worker: leaving the
        # screen mid-scan must not leave a QThread behind either.
        self.shutdown()
        worker = self._worker
        if worker is not None:
            try:
                worker.wait(5000)
            except RuntimeError:
                LOG.debug("worker already deleted", exc_info=True)
        super().closeEvent(event)


def build_motility_preview_card(host):
    """Build the ``Motility preview`` card + panel pair.

    Mirrors ``spacr.qt.screens.hyperparam.build_hyperparam_card``: returns
    the pair without adding it to any layout.

    :param host: the :class:`AppScreen` asking for the card.
    :returns: ``(panel, card)``.
    """
    from .card import Card
    card = Card(title="Motility preview")
    panel = MotilityPreviewPanel(card)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(320)
    return panel, card
