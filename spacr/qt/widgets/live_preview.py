"""
Live-preview segmentation widget — v2.

Interactive Cellpose tuning surface for the Mask app screen. Compared
to v1, this rewrite adds every enhancement the user requested after
their first run through the panel:

* **Zoomable canvases (Ctrl+scroll, in sync).** Both the original and
  the mask overlay live in a shared :class:`QGraphicsView` pair — pan
  and zoom on one and the other tracks pixel-for-pixel.
* **Hover tooltip.** Move the cursor over the original and a pinned
  status line shows the pixel intensity for every channel plus, when
  present, the object label at that position from the last segmenta-
  tion. Same tooltip regardless of which view holds the cursor.
* **Normalise toggle.** Optional 2–98 % percentile stretch (per channel
  for RGB) so raw low-contrast tiles are legible.
* **Model-aware options.** Every model shows the full segmentation set.
  Cellpose-SAM does *not* ignore ``flow_threshold``, ``cellprob`` or
  ``diameter`` — see :data:`DIAMETER_TOOLTIP` for the measurement that
  killed that belief.
* **Outline colour + thickness.** Chosen from the toolbar; effect is
  live once a mask exists. ``color (random)`` assigns a stable categorical
  colour to every object label so touching masks remain distinguishable.
* **Multi-object segmentation.** An "object type" combo picks between
  ``cell``, ``nucleus``, and ``cell + nucleus``. In cell+nucleus mode
  the panel runs two Cellpose passes and overlays both masks in
  distinct colours.
* **Pre / Post filters.** When the object type is ``cell`` (or the
  combined mode) the panel routes pre / post-processing settings from
  the Mask app (``cell_min_size``, ``cell_max_size``,
  ``remove_background_cell``, background intensity, ...) through the
  segmentation. Users toggle these on/off with dedicated "Pre" / "Post"
  clickable labels sitting next to "Run preview" in the same visual
  style as the LP / AI toggles.

The whole file stays safe to import without cellpose — every cellpose
call is lazy-imported inside the worker thread.
"""
from __future__ import annotations

import colorsys
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)
from .preview_controls import (
    DEFAULT_MAX_SETS, MAX_SETS_TOOLTIP, FlatButton, FlatComboBox, FlatSpinBox,
    ImageSetSampler, apply_sample_to_combo, channel_view, enumerate_image_sets,
    populate_channel_combo, sample_image_sets, sample_seed, selected_channel,
)
from .toggle import Toggle
from ..job_runner import JobRunner

LOG = logging.getLogger("spacr.qt.live_preview")

SUPPORTED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

#: Tooltip for the diameter spinner, in every model.
#:
#: The panel used to disable this control for ``cpsam`` and label it
#: "Ignored by Cellpose-SAM". That is false. ``CellposeModel._run_cp``
#: in cellpose 4.0.7 still does ``image_scaling = 30. / diameter``
#: whenever ``diameter is not None``, so the value decides the scale the
#: image is segmented at. Measured on an RTX 3090 against a real
#: micrograph (``plate1_E01_10.tif``, 1994x1994, cpsam, flow 0.4,
#: cellprob 0.0), counting objects per pass:
#:
#: ===============  ========  ===========
#: diameter         cells     nuclei
#: ===============  ========  ===========
#: unset (``None``)       66           65
#: 30                     66           65
#: 60                     71           63
#: ===============  ========  ===========
#:
#: 30 matches "unset" because 30/30 is a no-op rescale. That is the only
#: reason the control ever looked inert, and 30 is the spinner's default
#: — anyone who checked the claim without moving the value saw no change
#: and believed it. Greying it out took away a control that measurably
#: changes what cpsam finds.
DIAMETER_TOOLTIP = (
    "(float, px) Expected object diameter. Cellpose-SAM uses it: the "
    "image is rescaled by 30/diameter before segmentation, so raising it "
    "finds bigger objects and lowering it finds smaller ones. 30 is the "
    "no-op (30/30 = 1) and 0 means 'unset', which is the same thing."
)

# Object types the panel understands. Order matters — it drives the
# order of the combo. cell/nucleus can be previewed together; pathogen and
# organelle are single-compartment selections whose settings panels light up
# when chosen.
OBJECT_TYPES = ("cell", "nucleus", "cell + nucleus", "pathogen", "organelle")

# The four segmentation compartments, in the left→right order their settings
# panels appear in the Live settings dialog.
COMPARTMENTS = ("cell", "nucleus", "pathogen", "organelle")

# Overlay colours for individual object types. Cell = green (matches
# the classic v1 boundary colour), nucleus = magenta, and when both
# are shown together those colours read cleanly on top of most stains.
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cell":      (32, 220, 32),
    "nucleus":   (222, 82, 200),
    "pathogen":  (32, 200, 220),
    "organelle": (255, 220, 32),
}

# Stable offsets keep the random categorical outline map distinct between
# compartments without making colours flicker whenever the preview refreshes.
RANDOM_OUTLINE_SEEDS: Dict[str, int] = {
    "cell": 11,
    "nucleus": 37,
    "pathogen": 61,
    "organelle": 89,
}

# Per-compartment tuning settings, shown (greyed unless the compartment is the
# chosen object) in that compartment's panel. Each entry is
# ``(key_suffix, label, kind, spin_args)`` where the real setting key is
# ``f"{compartment}_{key_suffix}"`` and kind is one of int/float/bool/method.
# spin_args = (min, max, default) for int/float; ignored otherwise.
COMPARTMENT_FIELDS = (
    ("min_area",                   "Min area (px²)",        "int",   (0, 100_000_000, 0)),
    ("max_area",                   "Max area (px²)",        "int",   (0, 100_000_000, 0)),
    ("min_object_area",            "Min object area",       "int",   (0, 100_000_000, 100)),
    ("min_distance",               "Min distance",          "int",   (0, 100_000, 10)),
    ("area_multiplier",            "Area multiplier",       "float", (0.0, 1000.0, 2.0)),
    # Defaults MUST match spacr.settings.set_default_settings_preprocess_generate_masks.
    # They are both what the preview filters with and what the Propagate
    # button writes into the main settings panel, so any drift silently
    # re-tunes the real run. The intensity percentiles in particular used to
    # default to 1/99 rather than the pipeline's 0/100 — which switched
    # `_filter_objects`' intensity filter ON for every preview and dropped the
    # dimmest and brightest object found (with two objects, all of them).
    ("perimeter_fraction",         "Perimeter fraction",    "float", (0.0, 1.0, 0.0)),
    ("min_intensity_percentile",   "Min intensity pct",     "int",   (0, 100, 0)),
    ("max_intensity_percentile",   "Max intensity pct",     "int",   (0, 100, 100)),
    ("intensity_percentile",       "Intensity percentile",  "int",   (0, 100, 75)),
    ("intensity_threshold_method", "Intensity threshold",   "method", None),
    ("intensity_merge",            "Intensity merge",       "bool",  None),
    ("intensity_split",            "Intensity split",       "bool",  None),
    ("remove_border_objects",      "Remove border objects", "bool",  None),
)

# Threshold-method choices (see spacr.utils intensity-merge logic).
INTENSITY_THRESHOLD_METHODS = ("mean", "percentile")


# ---------------------------------------------------------------------------
# Pure numpy helpers — no Qt, safe to unit-test without a display
# ---------------------------------------------------------------------------

def load_preview_image(path: Path) -> np.ndarray:
    """Read *path* into an (H, W) or (H, W, C) uint8/uint16 array.

    Tifffile is used for TIFFs to preserve bit-depth; other formats fall
    back to PIL. Raises :class:`FileNotFoundError` if the path is bad.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def _full_range_max(img: np.ndarray) -> float:
    """Return the value that maps to white for a *raw* (un-normalised) view.

    For integer images this is the dtype maximum (e.g. 65535 for uint16), so
    a 16-bit image whose real values are small reads dark — the true raw
    view. For float images we assume a [0, 1] range unless the data clearly
    exceeds it, in which case we use the data max.
    """
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    m = float(np.nanmax(img)) if img.size else 1.0
    return 1.0 if m <= 1.0 else m


def _to_uint8(img: np.ndarray, normalise: bool = True,
                lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Return a viewable uint8 version of *img*.

    :param normalise: when True apply a per-channel percentile stretch. When
        False, map the *full bit-depth range* (0 → dtype max) to 0–255, i.e.
        the raw view — a 16-bit image with small values reads dark/black,
        not blown out. (Previously this clipped to [0, 255], which turned a
        16-bit image mostly white.)
    :param lo_pct: lower percentile for the stretch (default 2 %).
    :param hi_pct: upper percentile for the stretch (default 98 %).
    """
    # Channels-last is this module's convention everywhere else (see
    # :func:`_select_channel` and :meth:`LivePreviewPanel._label_rgb`), so a
    # single-channel tile collapses to grayscale and anything wider maps its
    # first three channels onto R/G/B. This used to be gated on
    # ``shape[-1] in (2, 3, 4)``, which sent (H, W, 1) and (H, W, 5+) tiles
    # down the 2-D branch and returned an array with a trailing channel axis —
    # :func:`numpy_to_qpixmap` then handed Qt a stride three times the real
    # row length and read past the end of the buffer.
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    full_max = _full_range_max(img) or 1.0
    if img.ndim == 3:
        out = np.zeros(img.shape[:2] + (3,), dtype=np.uint8)
        for c in range(min(3, img.shape[-1])):
            slice_ = img[..., c].astype(np.float32)
            if normalise:
                lo, hi = np.percentile(slice_, (lo_pct, hi_pct))
                if hi <= lo:
                    continue
                out[..., c] = np.clip(
                    255 * (slice_ - lo) / (hi - lo), 0, 255,
                ).astype(np.uint8)
            else:
                out[..., c] = np.clip(
                    255 * slice_ / full_max, 0, 255).astype(np.uint8)
        return out
    arr = img.astype(np.float32)
    if normalise:
        lo, hi = np.percentile(arr, (lo_pct, hi_pct))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip(
            255 * (arr - lo) / (hi - lo), 0, 255,
        ).astype(np.uint8)
    return np.clip(255 * arr / full_max, 0, 255).astype(np.uint8)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a bool array marking the 4-connected boundary of ``mask``."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :]  |= mask[1:, :]  != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:]  |= mask[:, 1:]  != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def _labelled_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Return each outline pixel's positive object label.

    Unlike :func:`_boundary_mask`, the result retains object identity so a
    categorical colour map can draw every segmented object differently. The
    label is also propagated onto the exterior half of an outline and through
    any requested dilation. Where two dilated outlines meet, the larger label
    wins deterministically.

    :param mask: two-dimensional integer label image; zero is background.
    :param thickness: outline thickness in pixels, clamped to ``1..5``.
    :returns: int64 array containing object labels only on outline pixels.
    """
    labels = np.asarray(mask, dtype=np.int64)
    if labels.ndim != 2 or not np.any(labels > 0):
        return np.zeros(labels.shape, dtype=np.int64)

    thickness = max(1, min(5, int(thickness)))
    boundary = _boundary_mask(labels)
    owners = np.where(boundary & (labels > 0), labels, 0)

    # Give exterior boundary pixels the label of an adjacent object. This
    # preserves the two-sided outline produced by the pre-existing renderer.
    neighbours = np.zeros_like(labels)
    neighbours[1:, :] = np.maximum(neighbours[1:, :], labels[:-1, :])
    neighbours[:-1, :] = np.maximum(neighbours[:-1, :], labels[1:, :])
    neighbours[:, 1:] = np.maximum(neighbours[:, 1:], labels[:, :-1])
    neighbours[:, :-1] = np.maximum(neighbours[:, :-1], labels[:, 1:])
    exterior = boundary & (owners == 0)
    owners[exterior] = neighbours[exterior]

    for _ in range(thickness - 1):
        expanded = np.zeros_like(owners)
        expanded[1:, :] = np.maximum(expanded[1:, :], owners[:-1, :])
        expanded[:-1, :] = np.maximum(expanded[:-1, :], owners[1:, :])
        expanded[:, 1:] = np.maximum(expanded[:, 1:], owners[:, :-1])
        expanded[:, :-1] = np.maximum(expanded[:, :-1], owners[:, 1:])
        owners = np.where(owners > 0, owners, expanded)
    return owners


def _random_outline_palette(
    labels: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Return vivid, deterministic random-looking RGB colours for labels.

    Golden-ratio hue spacing keeps adjacent integer labels separated while
    deterministic saturation/value jitter makes the result look like a
    random categorical colormap. Stability is intentional: changing zoom,
    normalisation, or thickness must not recolour every object.

    :param labels: one-dimensional array of positive object labels.
    :param seed: stable compartment-specific colour offset.
    :returns: ``(N, 3)`` uint8 RGB array in the same order as ``labels``.
    """
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    phase = (int(seed) % 997) / 997.0
    hues = np.mod(values * 0.618033988749895 + phase, 1.0)
    saturations = 0.72 + 0.25 * np.mod(values * 37 + seed, 101) / 100.0
    brightness = 0.86 + 0.13 * np.mod(values * 53 + seed, 97) / 96.0
    colours = [
        colorsys.hsv_to_rgb(float(hue), float(saturation), float(value))
        for hue, saturation, value in zip(hues, saturations, brightness)
    ]
    return np.rint(np.asarray(colours) * 255.0).astype(np.uint8)


#: Random-colour generator for the ``auto`` outline mode. Module level so a
#: test can seed it; unseeded it draws from the OS entropy pool, which is what
#: makes two preview runs come out in two different colours.
_AUTO_COLOUR_RNG = random.Random()


def random_outline_colour(rng: Optional[random.Random] = None
                          ) -> Tuple[int, int, int]:
    """Return one vivid random RGB triple for the ``auto`` outline mode.

    Hue is uniform over the full circle while saturation and value stay high,
    so the colour is always legible on top of a micrograph — a uniform draw in
    RGB would regularly produce muddy near-grey outlines nobody can see.

    :param rng: optional generator, for reproducible tests.
    :returns: ``(r, g, b)`` in 0..255.
    """
    source = rng if rng is not None else _AUTO_COLOUR_RNG
    hue = source.random()
    saturation = 0.70 + 0.30 * source.random()
    value = 0.85 + 0.15 * source.random()
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(round(red * 255)), int(round(green * 255)),
            int(round(blue * 255)))


def overlay_masks(image: np.ndarray,
                    masks: Dict[str, np.ndarray],
                    outline_rgb: Optional[Tuple[int, int, int]] = None,
                    outline_thickness: int = 1,
                    normalise: bool = True,
                    lo_pct: float = 2.0,
                    hi_pct: float = 98.0,
                    random_outline: bool = False,
                    outline_colors: Optional[
                        Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Return an RGB uint8 view of ``image`` with every mask's boundary
    drawn in the object's colour (or ``outline_rgb`` when supplied).

    :param image: (H, W) or (H, W, C) source image.
    :param masks: ``{object_type: label_array}`` — one entry per object
        type currently visible on the panel.
    :param outline_rgb: overrides the per-object colour when the user
        picks a global outline colour from the toolbar.
    :param outline_thickness: number of pixels the boundary is dilated
        by (1 = crisp, 3 = highlighter). Tops out at 5.
    :param normalise: forwarded to :func:`_to_uint8`.
    :param random_outline: assign every positive object label a vivid,
        stable categorical colour. This takes precedence over
        ``outline_rgb`` and corresponds to ``color (random)`` in Mask Live.
    :param outline_colors: per-compartment colour overrides used when no
        global ``outline_rgb`` is given. This is how the panel's ``auto``
        mode reaches the renderer: it holds one random colour per
        compartment for the current run. Falls back to
        :data:`OBJECT_COLORS` for anything it does not name.
    """
    base = _to_uint8(image, normalise=normalise,
                        lo_pct=lo_pct, hi_pct=hi_pct)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base[..., :3].copy()
    outline_thickness = max(1, min(5, int(outline_thickness)))
    for obj_type, mask in masks.items():
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
            # A mask left over from a previously loaded image. Drawing it
            # raised ``IndexError: boolean index did not match indexed array``
            # (or a broadcast ValueError) instead of simply being ignored.
            LOG.debug("overlay_masks: skipping %s mask %s — image is %s",
                      obj_type, mask.shape, rgb.shape[:2])
            continue
        if not mask.any():
            continue
        if random_outline:
            labelled_boundary = _labelled_boundary(mask, outline_thickness)
            pixels = labelled_boundary > 0
            if not pixels.any():
                continue
            object_labels = np.unique(labelled_boundary[pixels])
            palette = _random_outline_palette(
                object_labels,
                RANDOM_OUTLINE_SEEDS.get(obj_type, 0),
            )
            palette_indices = np.searchsorted(
                object_labels, labelled_boundary[pixels],
            )
            rgb[pixels] = palette[palette_indices]
            continue
        boundary = _boundary_mask(mask.astype(np.int32))
        for _ in range(outline_thickness - 1):
            # Dilate by one pixel: OR-shift in each cardinal direction
            b2 = boundary.copy()
            b2[1:, :]  |= boundary[:-1, :]
            b2[:-1, :] |= boundary[1:, :]
            b2[:, 1:]  |= boundary[:, :-1]
            b2[:, :-1] |= boundary[:, 1:]
            boundary = b2
        colour = outline_rgb
        if colour is None and outline_colors:
            colour = outline_colors.get(obj_type)
        if colour is None:
            colour = OBJECT_COLORS.get(obj_type, (32, 220, 32))
        rgb[boundary] = np.array(colour, dtype=np.uint8)
    return rgb


def numpy_to_qpixmap(arr: np.ndarray, normalise: bool = True,
                        lo_pct: float = 2.0,
                        hi_pct: float = 98.0) -> QPixmap:
    """Convert an (H, W) or (H, W, C) array to a :class:`QPixmap`.

    The result is always RGB888, so the caller cannot hand Qt a buffer whose
    real row length disagrees with the ``w * 3`` stride below. Channel counts
    other than three are reconciled here — extra channels are dropped, missing
    ones are filled with black — because a mismatch made ``QImage`` read
    ``h * w * 3`` bytes out of a buffer that only held ``h * w``.
    """
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr, normalise=normalise,
                          lo_pct=lo_pct, hi_pct=hi_pct)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] < 3:
        pad = np.zeros(arr.shape[:2] + (3 - arr.shape[-1],), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=-1)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    img = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# ---------------------------------------------------------------------------
# Image discovery/loading + segmentation workers
# ---------------------------------------------------------------------------

def first_supported_image(source: Path) -> Optional[Path]:
    """Return the first supported image at or below ``source``.

    Direct image files are returned unchanged. Directory traversal stops as
    soon as the first sorted match is found instead of materialising and
    sorting every image in a potentially enormous plate.

    :param source: image path or directory to inspect.
    :returns: the first supported image, or ``None``.
    """
    source = Path(source)
    if source.is_file():
        return source if source.suffix.lower() in SUPPORTED_SUFFIXES else None
    if not source.is_dir():
        return None

    walk_errors: List[OSError] = []
    for folder, dirs, files in os.walk(
            source, topdown=True, onerror=walk_errors.append,
            followlinks=False):
        dirs.sort(key=str.casefold)
        for name in sorted(files, key=str.casefold):
            if Path(name).suffix.lower() in SUPPORTED_SUFFIXES:
                return Path(folder) / name
    if walk_errors:
        raise OSError(
            f"Could not inspect {source}: {walk_errors[0]}")
    return None


def load_source_payload(source, max_sets: int = DEFAULT_MAX_SETS,
                        enumerate_sets: bool = True) -> Dict[str, Any]:
    """Discover, enumerate and decode one preview source. Data in, data out.

    This is the whole of a preview load, written so it touches **no widget and
    no Qt object** and can therefore be handed straight to
    :class:`spacr.qt.job_runner.JobRunner`. It used to be the ``run`` method of
    a hand-rolled ``QThread`` that emitted two signals, which kept the panel's
    sampler warm by ordering ``enumerated`` before ``loaded``; returning both
    halves in one dict gets the same ordering for free, because the caller
    adopts the enumeration and installs the image in a single GUI-thread call.

    The enumeration reads **file names only** — it never opens an image — so
    the single decode here stays the only file read for a folder of any size.

    :param source: image file or directory to load a preview from.
    :param max_sets: cap for the sample drawn when ``source`` is a directory.
    :param enumerate_sets: ``False`` skips the folder scan entirely. The FOV
        dropdown hands out a path from a set the sampler already produced, so
        re-scanning for it would burn a full pass over a 98 000-file plate to
        rediscover what is already cached.
    :returns: ``{path, array, directory, sets, channels, error}``. ``sets`` is
        ``None`` when no enumeration was done or it failed, which the caller
        reads as "leave the sampler alone".
    """
    out: Dict[str, Any] = {
        "path": None, "array": None, "directory": None,
        "sets": None, "channels": None, "error": "",
    }
    try:
        source = Path(source)
        path = first_supported_image(source)
        if path is not None and enumerate_sets:
            try:
                sets, channels = enumerate_image_sets(
                    path.parent, SUPPORTED_SUFFIXES)
                out["directory"] = str(path.parent)
                out["sets"] = sets
                out["channels"] = channels
                if sets and source.is_dir():
                    # Open on one of the sampled sets. Whichever file sorts
                    # first is A01 field 1, which on a plate-ordered folder is
                    # exactly the corner the sample exists to stop the preview
                    # from standing in for.
                    picked = sample_image_sets(
                        sets, max_sets,
                        sample_seed(path.parent, len(sets), max_sets))
                    if picked:
                        path = picked[0].path()
            except Exception:
                # A folder we cannot group is still a folder we can show one
                # image from; the panel falls back to per-file sets.
                LOG.exception("Could not enumerate image sets under %s",
                              path.parent)
        out["path"] = path
        out["array"] = load_preview_image(path) if path is not None else None
    except Exception as exc:
        LOG.exception("Could not load live-preview source %s", source)
        out["error"] = str(exc) or exc.__class__.__name__
    return out


@dataclass
class PreviewRequest:
    """Everything the worker needs to run one segmentation pass.

    Kept as a plain dataclass so tests can construct it directly; the
    panel builds one from its widget state on each Run.
    """
    image:               np.ndarray
    model:               str = "cpsam"
    diameter:            float = 30.0
    flow_threshold:      float = 0.4
    cellprob:            float = 0.0
    channels:            Dict[str, int] = field(default_factory=dict)
    object_types:        Tuple[str, ...] = ("cell",)
    preprocess_settings: Dict[str, Any] = field(default_factory=dict)
    postprocess_settings: Dict[str, Any] = field(default_factory=dict)


class _PreviewWorker(QThread):
    """Runs one (or two) Cellpose passes in the background."""

    # ({obj: mask, ...} or None, err, run token)
    finished_masks = Signal(object, str, int)
    # ({obj: flow_rgb} — may be empty, run token)
    flows_ready = Signal(object, int)

    def __init__(self, request: PreviewRequest, parent=None, token: int = 0):
        """:param token: the panel's run token at the moment this worker was
        started. It rides back out on both result signals so the panel can
        recognise — and drop — a result produced for an image it has since
        replaced. Cellpose has no interrupt, so this is what "cancel" means
        here: the thread runs itself out and its answer lands as a no-op."""
        super().__init__(parent)
        self._request = request
        self.token = int(token)

    def run(self):
        try:
            res = _segment_multi(self._request)
            # _segment_multi may return masks only (the stubbed test path) or
            # (masks, flows). Handle both.
            if isinstance(res, tuple):
                masks, flows = res
            else:
                masks, flows = res, {}
            self.finished_masks.emit(masks, "", self.token)
            self.flows_ready.emit(flows or {}, self.token)
        except Exception as e:
            LOG.info("live-preview segmentation failed: %s", e,
                       exc_info=True)
            self.finished_masks.emit(None, str(e), self.token)


def _segment_multi(req: PreviewRequest) -> Dict[str, np.ndarray]:
    """Run one Cellpose pass per requested object type.

    Cellpose is lazy-imported here so importing this file cold — as
    unit tests do — does not require a CUDA-capable stack.

    Post-processing (min/max size filter, background removal) is
    applied per-object-type after the model returns, using the
    ``postprocess_settings`` dict on the request.
    """
    from cellpose import models as cp_models
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False

    if req.model == "cpsam":
        model = cp_models.CellposeModel(
            gpu=gpu, pretrained_model="cpsam", device=None)
    else:
        model = cp_models.CellposeModel(
            gpu=gpu, model_type=req.model, device=None)

    out: Dict[str, np.ndarray] = {}
    flows_out: Dict[str, np.ndarray] = {}
    for obj in req.object_types:
        ch_idx = req.channels.get(obj, 0)
        image_2d = _select_channel(req.image, ch_idx)

        # Preprocess — remove background if the user opted in, doing exactly
        # what a real run does. `spacr.io._normalize_img_batch` runs
        #
        #     single_channel[single_channel < background] = 0
        #
        # per channel, reading `{obj}_background`. This used to subtract the
        # background and clip at zero instead, which is a different image:
        # thresholding leaves everything above the background where it is,
        # subtraction shifts all of it down. And it read a plain `background`
        # key that nothing writes -- the panel emits `{obj}_background` -- so
        # the value was always the 100.0 default and turning the toggle on
        # did nothing visible on any image whose real background was not
        # near 100.
        #
        # Both keys are per-object on purpose: with "cell + nucleus"
        # selected, the two channels get their own background and their own
        # on/off, the same way the pipeline treats them.
        if req.preprocess_settings.get(f"remove_background_{obj}"):
            bg = float(req.preprocess_settings.get(
                f"{obj}_background",
                req.preprocess_settings.get("background", 100.0)))
            # `_select_channel` hands back a view into `req.image`. Writing
            # through it would zero the source for every object type after
            # this one, and for the raw pane the panel shows beside the mask.
            image_2d = image_2d.copy()
            image_2d[image_2d < bg] = 0

        result = model.eval(
            image_2d,
            diameter=float(req.diameter) or None,
            flow_threshold=float(req.flow_threshold),
            cellprob_threshold=float(req.cellprob),
        )
        mask = result[0]
        if isinstance(mask, list):
            mask = mask[0]
        mask = np.asarray(mask).astype(np.int32)

        # Capture the RGB flow visualisation (flows[0]) if Cellpose returned
        # one, so the panel can show a Flows view alongside the masks.
        try:
            flows = result[1]
            flow_rgb = flows[0] if isinstance(flows, (list, tuple)) else flows
            if isinstance(flow_rgb, list):
                flow_rgb = flow_rgb[0]
            flows_out[obj] = np.asarray(flow_rgb)
        except Exception:
            pass

        # Return the RAW (unfiltered) mask — the panel applies the per-
        # compartment filters afterwards so the user can re-tune filters
        # without re-running Cellpose (see LivePreviewPanel._recompute_masks).
        out[obj] = mask
    return out, flows_out


def _select_channel(image: np.ndarray, ch: int) -> np.ndarray:
    """Return a 2-D slice from ``image`` at channel index ``ch``."""
    if image.ndim == 3 and image.shape[-1] > 1:
        return image[..., int(ch) % image.shape[-1]]
    return image.squeeze()


def _apply_size_filter(mask: np.ndarray,
                          settings: Dict[str, Any],
                          obj: str,
                          intensity_img: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the *same* post-segmentation filters the pipeline uses, so the
    live preview matches a real run.

    Reads the per-compartment knobs (``{obj}_min_area``, ``{obj}_max_area``,
    ``{obj}_remove_border_objects``, ``{obj}_min_intensity_percentile``,
    ``{obj}_max_intensity_percentile``) — the exact keys the compartment
    panels write — and runs them through :func:`spacr.utils._filter_objects`.
    Legacy ``{obj}_min_size``/``{obj}_max_size`` are honoured as a fallback.
    No-ops when nothing is set."""
    if not settings or mask is None:
        return mask

    def _num(key, default):
        v = settings.get(key, default)
        try:
            return type(default)(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    min_area = _num(f"{obj}_min_area", _num(f"{obj}_min_size", 0))
    max_area = _num(f"{obj}_max_area", _num(f"{obj}_max_size", 0))
    remove_border = bool(settings.get(f"{obj}_remove_border_objects", False))
    min_ip = _num(f"{obj}_min_intensity_percentile", 0)
    max_ip = _num(f"{obj}_max_intensity_percentile", 100)

    if not (min_area > 0 or max_area > 0 or remove_border
            or min_ip > 0 or max_ip < 100):
        return mask

    try:
        from spacr.utils import _filter_objects
        return _filter_objects(
            mask.astype(np.uint16).copy(),
            intensity_img=intensity_img,
            min_area=int(min_area), max_area=int(max_area),
            remove_border=remove_border,
            min_intensity_percentile=float(min_ip),
            max_intensity_percentile=float(max_ip),
        ).astype(mask.dtype)
    except Exception:
        LOG.debug("size filter failed", exc_info=True)
        return mask


# ---------------------------------------------------------------------------
# Twin zoomable views with a shared transform
# ---------------------------------------------------------------------------

class _ZoomView(QGraphicsView):
    """A :class:`QGraphicsView` that emits pixel-hover events + supports
    plain wheel-scroll zoom.

    Two big usability upgrades over the first-cut version:

    * **Wheel = zoom (no Ctrl needed).** Turning the wheel zooms
      centred on the cursor. Shift+wheel scrolls the viewport
      vertically if the user wants scroll behaviour.
    * **Fit-to-height on load + resize.** The image always fills the
      canvas at 100 % zoom until the user starts scrolling, so a small
      preview panel doesn't leave the tile 1-cm tall in the corner.
      Every ``resizeEvent`` re-fits — as the splitter is dragged, the
      image grows to match.

    Zoom is broadcast to a peer view via :meth:`set_peer` so the mask
    canvas mirrors what the original canvas is doing (and vice versa).
    """

    hover_pixel = Signal(int, int)   # (x, y) in image coords
    zoom_changed = Signal(float)     # new scale factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._peer: Optional["_ZoomView"] = None
        self._syncing = False
        self._scale = 1.0
        self._user_zoomed = False
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFrameShape(QGraphicsView.NoFrame)
        # Panning has to be mirrored the same way zoom is. `ScrollHandDrag`
        # moves the scroll bars rather than the transform, so `_apply_zoom`
        # never sees a drag and the twin canvases stayed locked in scale
        # while drifting apart in position -- zoom in, drag one, and the
        # mask no longer sits over the cell it was drawn from.
        self.horizontalScrollBar().valueChanged.connect(self._mirror_pan)
        self.verticalScrollBar().valueChanged.connect(self._mirror_pan)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        # Fit-in-view on load, and forget any previous user zoom so the
        # new image starts at 100 % of the canvas.
        self._user_zoomed = False
        self._scale = 1.0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_peer(self, peer: "_ZoomView") -> None:
        self._peer = peer

    def scale_factor(self) -> float:
        return self._scale

    def reset_zoom(self) -> None:
        """Snap back to fit-in-view (100 % of the container)."""
        self._user_zoomed = False
        self._scale = 1.0
        self.resetTransform()
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # -- events ------------------------------------------------------------

    def wheelEvent(self, event):
        """Plain wheel = zoom around cursor. Shift+wheel = scroll."""
        if event.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        factor = 1.20 if event.angleDelta().y() > 0 else 0.833
        self._apply_zoom(factor, broadcast=True)
        event.accept()

    def resizeEvent(self, event):
        """Refit the tile whenever the container size changes, unless
        the user has manually zoomed in / out."""
        super().resizeEvent(event)
        if not self._user_zoomed and self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _apply_zoom(self, factor: float, broadcast: bool = False) -> None:
        if self._syncing:
            return
        self.scale(factor, factor)
        self._scale *= factor
        self._user_zoomed = True
        self.zoom_changed.emit(self._scale)
        if broadcast and self._peer is not None:
            # Guard THIS view while the peer catches up, not the peer: the
            # flag makes _apply_zoom a no-op, so setting it on the peer meant
            # the peer's own zoom was skipped and the twin canvases never
            # actually tracked each other.
            self._syncing = True
            try:
                self._peer._apply_zoom(factor, broadcast=False)
            finally:
                self._syncing = False

    def _mirror_pan(self, _value: int = 0) -> None:
        """Put the peer at the same scroll offset as this view.

        Guarded on THIS view for the same reason ``_apply_zoom`` is: setting
        the flag on the peer would make the peer's own handler a no-op, and
        since assigning to its scroll bars fires that handler, the guard has
        to be on the sender or the two views ping-pong.

        Raw scroll-bar values rather than a mapped scene point: the two
        canvases show the same image at the same scale and the same viewport
        size, so their scroll ranges are identical, and copying the value
        keeps them aligned to the pixel without a round trip through scene
        coordinates.
        """
        peer = self._peer
        if peer is None or self._syncing:
            return
        self._syncing = True
        try:
            peer.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value())
            peer.verticalScrollBar().setValue(
                self.verticalScrollBar().value())
        finally:
            self._syncing = False

    def mouseMoveEvent(self, event):
        if self._pixmap_item is not None:
            scene_pt = self.mapToScene(event.position().toPoint())
            x = int(scene_pt.x())
            y = int(scene_pt.y())
            self.hover_pixel.emit(x, y)
        super().mouseMoveEvent(event)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

#: Last resort if `spacr.settings` cannot be reached at all — a stub in
#: sys.modules, a partially-installed tree. A dropdown with nothing in it
#: is a dead end, so there is always something here.
_FALLBACK_MODELS = ("cpsam", "cyto3", "cyto2", "nuclei")


def _model_menu():
    """What the Cellpose model combo offers, read from the Cellpose API.

    Delegates to :func:`spacr.settings.cellpose_model_menu`, which asks
    ``cellpose.models`` for its stock list plus any checkpoint the user
    registered, then appends the accepted-but-mapped legacy spellings so a
    saved preview setting still loads.

    Wrapped because this is a *widget*: it must build even when
    ``spacr.settings`` is a stand-in (a test that stubs the descriptions
    table does exactly that). It degrades to the shipped list rather than
    to an empty combo.
    """
    try:
        from ...settings import cellpose_model_menu
        menu = tuple(cellpose_model_menu())
    except Exception:
        return _FALLBACK_MODELS
    return menu or _FALLBACK_MODELS


class LivePreviewPanel(QWidget):
    """Interactive segmentation preview — Mask app only."""

    preview_ready = Signal(object)   # {object_type: mask}

    def __init__(self, parent=None, *, threaded: bool = True):
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._masks: Dict[str, np.ndarray] = {}
        self._raw_masks: Dict[str, np.ndarray] = {}
        self._flows: Dict[str, np.ndarray] = {}
        self._settings: Dict[str, Any] = {}
        self._worker: Optional[_PreviewWorker] = None
        # Every preview load goes through here rather than through a QThread
        # this file owns. That is not tidiness: `JobRunner` submits via
        # `bridge.make_thread`, which is what puts the job in the process-wide
        # run registry, and the registry is the *only* thing the activity
        # spinner watches. The hand-rolled loader this replaced ran off the GUI
        # thread perfectly well and still left the user staring at a frozen-
        # looking window with no spinner, because nothing ever told the
        # registry it existed.
        self._load_jobs = JobRunner(self, threaded=threaded,
                                    app_key="preview image")
        self._image_load_token: int = 0
        # Bumped whenever the run in flight is superseded (a new image, an
        # explicit cancel). A worker's result is only accepted when the token
        # it carries still matches.
        self._run_token: int = 0
        # Callback(dict) that pushes tuned live settings into the main panel.
        self._propagate_cb = None
        # One random colour per compartment for the 'auto' outline mode,
        # re-rolled on every preview run (see _roll_auto_outline_colours).
        self._auto_outline_colours: Dict[str, Tuple[int, int, int]] = {}
        # Guards the FOV dropdown against re-entering itself while the image
        # it just asked for is being installed.
        self._loading_fov = False
        # Groups the source folder into image sets from file names alone and
        # hands out a bounded, reproducible random sample of them. Caches the
        # enumeration per folder, so stepping through fields costs nothing.
        self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
        self._build_ui()
        self._build_compartment_widgets()
        # Accept image files dropped anywhere on the panel. QGraphicsView
        # enables acceptDrops by default and would otherwise swallow drops
        # over the image canvases; turning it off on the views lets the drag
        # events propagate up to this panel's handlers.
        self.setAcceptDrops(True)
        for _v in (getattr(self, "_src_view", None),
                   getattr(self, "_mask_view", None)):
            if _v is not None:
                _v.setAcceptDrops(False)

    # -- drag & drop -------------------------------------------------------

    _DND_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
                 ".gif", ".webp")

    def _dropped_image_path(self, event) -> Optional[str]:
        """Return the first dropped local image path, or None."""
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if Path(p).suffix.lower() in self._DND_EXTS:
                return p
        return None

    def dragEnterEvent(self, event):    # noqa: N802 (Qt naming)
        """Accept the drag only if it carries a supported image file."""
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):     # noqa: N802
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):         # noqa: N802
        """Load the dropped image into the preview."""
        path = self._dropped_image_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        # Asynchronously — the drop handler must return to the event loop
        # immediately. The decode is the small half; the expensive half is
        # enumerating the folder the dropped file came out of, which is what
        # froze the window for 643 ms on a 98 304-file plate.
        self.load_source_async(path)

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # -- HIDDEN state widgets ----------------------------------------
        # Every parameter widget lives here even though only a subset
        # appears in the collapsed layout. The Live Settings dialog
        # re-parents them into its own form when it opens, then hands
        # them back on close so their values persist across opens.
        # All widgets are children of `self` so they're never
        # garbage-collected while re-parented.
        # Read from the Cellpose API, not from a literal — see
        # `spacr.settings.cellpose_model_menu`. It returns whatever
        # `cellpose.models` reports plus any checkpoint the user has
        # registered, then the accepted-but-mapped aliases cyto3/cyto2/
        # nuclei so a saved preview setting still loads. Those are NOT four
        # choices: Cellpose 4 drops model_type= with a "not used in v4.0.1+"
        # log line, so all four run the same cpsam weights. The pipeline maps
        # them forward in settings.normalize_cellpose_model_name.
        self._model_box = QComboBox(self)
        self._model_box.addItems(list(_model_menu()))
        self._model_box.currentIndexChanged.connect(
            self._on_model_or_object_changed)

        self._object_box = QComboBox(self)
        self._object_box.addItems(list(OBJECT_TYPES))
        self._object_box.currentIndexChanged.connect(
            self._on_model_or_object_changed)

        self._cell_channel = QSpinBox(self); self._cell_channel.setRange(0, 8)
        self._nucleus_channel = QSpinBox(self); self._nucleus_channel.setRange(0, 8)
        self._nucleus_channel.setValue(1)

        self._diameter = QDoubleSpinBox(self)
        self._diameter.setRange(0, 400); self._diameter.setValue(30.0)
        self._diameter.setSuffix(" px")
        self._flow = QDoubleSpinBox(self)
        self._flow.setRange(-1, 3); self._flow.setSingleStep(0.05)
        self._flow.setValue(0.4)
        self._prob = QDoubleSpinBox(self)
        self._prob.setRange(-6, 6); self._prob.setSingleStep(0.1)
        self._prob.setValue(0.0)

        # Two-field percentile stretch — user asked for this shape
        # explicitly (was a single toggle before).
        self._normalise_check = Toggle("Normalise", self)
        self._normalise_check.setChecked(True)
        self._normalise_check.toggled.connect(self._refresh_canvases)
        self._lo_pct = QDoubleSpinBox(self)
        self._lo_pct.setRange(0, 50); self._lo_pct.setValue(2.0)
        self._lo_pct.setSuffix(" %")
        self._lo_pct.valueChanged.connect(self._refresh_canvases)
        self._hi_pct = QDoubleSpinBox(self)
        self._hi_pct.setRange(50, 100); self._hi_pct.setValue(98.0)
        self._hi_pct.setSuffix(" %")
        self._hi_pct.valueChanged.connect(self._refresh_canvases)

        # Outline appearance
        self._outline_colour = QComboBox(self)
        # These entries are looked up by text in _outline_rgb; the language
        # pass must not rewrite them, or every choice would miss the mapping
        # and silently fall back to the per-compartment default (green for
        # cells) — an outline colour that can never be changed.
        self._outline_colour.setProperty("i18nSkipItems", True)
        for name in ("auto", "color (random)", "green", "magenta",
                     "yellow", "cyan", "white", "red"):
            self._outline_colour.addItem(name)
        # Random is the default. A fixed colour is a coin flip against the
        # image -- green outlines on a green channel are invisible exactly
        # when you most need to see whether the mask landed -- and `auto`
        # picks per compartment, so two touching objects of the same type
        # share an outline and read as one. Set before the signal is
        # connected so choosing it here does not fire a render on a panel
        # that has no image yet.
        self._outline_colour.setCurrentText("color (random)")
        self._outline_colour.currentIndexChanged.connect(
            self._on_outline_colour_changed)
        self._outline_thickness = QSpinBox(self)
        self._outline_thickness.setRange(1, 5)
        self._outline_thickness.setValue(1)
        self._outline_thickness.valueChanged.connect(
            self._refresh_canvases)

        # Tooltips for the segmentation controls (type + what they do).
        self._model_box.setToolTip(
            "(str) Cellpose model. Cellpose 4 ships exactly one, 'cpsam'. "
            "cyto3/cyto2/nuclei are kept only so older saved settings still "
            "load — Cellpose removed those weights and resolves all of them "
            "to cpsam, so picking one does not change the segmentation. Of "
            "the parameters that used to differ per model, only diameter "
            "still does anything (the image is rescaled by 30/diameter); "
            "model_type and diam_mean are logged as 'not used in v4.0.1+' "
            "and dropped.")
        self._object_box.setToolTip(
            "(str) Object(s) to segment. 'cell + nucleus' runs both passes.")
        self._cell_channel.setToolTip(
            "(int) Image channel index used for cell segmentation.")
        self._nucleus_channel.setToolTip(
            "(int) Image channel index used for nucleus segmentation.")
        self._diameter.setToolTip(DIAMETER_TOOLTIP)
        self._flow.setToolTip(
            "(float) Cellpose flow threshold — higher keeps more masks.")
        self._prob.setToolTip(
            "(float) Cellpose cell-probability threshold — lower keeps more "
            "(dimmer) objects.")
        self._normalise_check.setToolTip(
            "(bool) Percentile-normalise the image for display + segmentation.")
        self._lo_pct.setToolTip(
            "(float, %) Lower percentile for normalisation.")
        self._hi_pct.setToolTip(
            "(float, %) Upper percentile for normalisation.")
        self._outline_colour.setToolTip(
            "(str) Overlay outline colour. 'auto' uses one colour per "
            "compartment; 'color (random)' gives every segmented object a "
            "different stable categorical colour.")
        self._outline_thickness.setToolTip(
            "(int, px) Overlay outline thickness.")

        # Keep every hidden helper widget parented but invisible so it
        # doesn't render in the compact layout.
        for w in (self._model_box, self._object_box,
                    self._cell_channel, self._nucleus_channel,
                    self._diameter, self._flow, self._prob,
                    self._normalise_check, self._lo_pct, self._hi_pct,
                    self._outline_colour, self._outline_thickness):
            w.hide()

        # -- VISIBLE compact layout --------------------------------------

        # File picker row — FOV and channel dropdowns sit immediately LEFT of
        # the Choose control, all three wearing the flat "Live toggle" look.
        pick_row = QHBoxLayout()
        self._pick_row = pick_row
        self._path_label = QLabel(
            "No preview image loaded — drag & drop an image here to load it",
            self)
        self._path_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._max_sets_box = FlatSpinBox(self, value=DEFAULT_MAX_SETS,
                                         tooltip=MAX_SETS_TOOLTIP)
        self._max_sets_box.valueChanged.connect(self._on_max_sets_changed)
        self._fov_box = FlatComboBox(
            self,
            tooltip=("Field of view. Lists a random sample of the image sets "
                     "in this folder — set the sample size on its left."))
        self._fov_box.currentIndexChanged.connect(self._on_fov_changed)
        self._channel_box = FlatComboBox(
            self,
            tooltip=("Displayed channel. 'All channels' shows the image as "
                     "stored; picking one shows that plane alone. This is a "
                     "view control — the segmentation channels live in Live "
                     "settings."))
        self._channel_box.currentIndexChanged.connect(
            self._on_display_channel_changed)
        populate_channel_combo(self._channel_box, 0)
        self._pick_btn = FlatButton("Choose image…", self)
        self._pick_btn.clicked.connect(self._pick_file)
        pick_row.addWidget(self._path_label, 1)
        pick_row.addWidget(self._max_sets_box)
        pick_row.addWidget(self._fov_box)
        pick_row.addWidget(self._channel_box)
        pick_row.addWidget(self._pick_btn)
        root.addLayout(pick_row)

        # Action row — Run + Live settings + status
        act = QHBoxLayout()
        self._run_btn = QPushButton("Run preview", self)
        self._run_btn.clicked.connect(self.run_preview)
        self._live_settings_btn = QPushButton("Live settings…", self)
        self._live_settings_btn.clicked.connect(self.open_live_settings)
        # What the right-hand canvas shows: outline overlay, the raw label
        # mask, or the Cellpose flow field.
        self._view_mode = QComboBox(self)
        # Read back by text in _refresh_canvases — never translate.
        self._view_mode.setProperty("i18nSkipItems", True)
        self._view_mode.addItems(["Overlay", "Masks", "Flows"])
        self._view_mode.setToolTip(
            "Right canvas: outline overlay · label masks · Cellpose flows")
        self._view_mode.currentTextChanged.connect(
            lambda *_: self._refresh_canvases())
        self._status = QLabel("", self)
        act.addWidget(self._run_btn)
        act.addWidget(self._live_settings_btn)
        act.addWidget(QLabel("View:", self))
        act.addWidget(self._view_mode)
        act.addWidget(self._status, 1)
        root.addLayout(act)

        # Twin zoomable canvases in a synchronised pair.
        canvas = QHBoxLayout()
        self._src_view = _ZoomView(self)
        self._src_view.setMinimumHeight(160)
        self._mask_view = _ZoomView(self)
        self._mask_view.setMinimumHeight(160)
        self._src_view.set_peer(self._mask_view)
        self._mask_view.set_peer(self._src_view)
        self._src_view.hover_pixel.connect(self._on_hover)
        self._mask_view.hover_pixel.connect(self._on_hover)
        canvas.addWidget(self._src_view, 1)
        canvas.addWidget(self._mask_view, 1)
        root.addLayout(canvas, 1)

        # Pinned hover info line
        self._hover_label = QLabel("Hover over the image to inspect pixels.",
                                     self)
        self._hover_label.setStyleSheet("color: #ffffff; "
                                            "font-family: monospace;")
        root.addWidget(self._hover_label)

        # Comparison scrubber — scrub back/forth through previous preview runs
        # to compare how different settings changed the segmentation. Hidden
        # until at least two runs exist.
        from PySide6.QtWidgets import QSlider
        self._history: list = []
        self._compare_row = QWidget(self)
        comp = QHBoxLayout(self._compare_row)
        comp.setContentsMargins(0, 0, 0, 0)
        comp.addWidget(QLabel("Compare runs", self))
        self._compare_slider = QSlider(Qt.Horizontal, self)
        self._compare_slider.setMinimum(0)
        self._compare_slider.setMaximum(0)
        self._compare_slider.valueChanged.connect(self._on_compare_scrub)
        comp.addWidget(self._compare_slider, 1)
        self._compare_label = QLabel("", self)
        self._compare_label.setStyleSheet("color: #ffffff; font-family: monospace;")
        comp.addWidget(self._compare_label)
        self._compare_row.setVisible(False)
        root.addWidget(self._compare_row)

        # Book-keeping for the dialog-based settings surface. Kept as
        # a member so tests + external hooks can introspect / drive it.
        self._live_settings_dialog: Optional["LiveSettingsDialog"] = None
        self._on_model_or_object_changed()

    # -- public API --------------------------------------------------------

    def load_image(self, path):
        """Synchronously load one image.

        Intended for explicit programmatic calls and tests, and for those only.
        **Every** GUI path — the drop handler, the FOV dropdown and the
        Choose-image dialog — goes through :meth:`load_source_async`, so that
        neither the decode nor the folder enumeration behind
        ``_refresh_source_selectors`` can block the application thread. Three of
        them used to call this instead, which is what the docstring already
        claimed was not happening.
        """
        try:
            arr = load_preview_image(Path(path))
        except Exception as e:
            self._status.setText(f"Load failed: {e}")
            return False
        self._install_loaded_image(Path(path), arr)
        return True

    @property
    def _image_loaders(self) -> List[int]:
        """The loads still in flight, as a list so ``not ...`` reads naturally.

        Kept under its historical name because callers and tests wait on it
        going empty. It is now derived from the runner rather than stored, so
        a cancelled load empties it too — a stored list would have to be
        pruned by hand on every exit path and would strand the panel as
        permanently "loading" the one time that was missed.
        """
        runner = getattr(self, "_load_jobs", None)
        return [] if runner is None else [0] * runner.pending_jobs()

    def load_source_async(self, source, *, enumerate_sets: bool = True) -> bool:
        """Discover and decode a file/folder source on a worker thread.

        New requests supersede older ones by token. An old decoder is allowed
        to finish safely, but its result is ignored.

        :param source: direct supported image or directory containing images.
        :param enumerate_sets: ``False`` reuses the sampler's cached listing
            instead of re-scanning. See :func:`load_source_payload`.
        :returns: ``True`` when a worker was started.
        """
        text = os.fspath(source).strip() if source is not None else ""
        if not text:
            return False
        self._image_load_token += 1
        token = self._image_load_token
        max_sets = int(self._sampler.max_sets)
        self._status.setText(f"Loading preview from {text}…")
        self._load_jobs.submit(
            lambda: load_source_payload(text, max_sets, enumerate_sets),
            lambda payload, _t=token: self._on_source_payload(_t, payload))
        return True

    def _on_source_payload(self, token: int, payload) -> None:
        """Apply the newest asynchronous load result. Always on the GUI thread.

        Adopting the enumeration *before* installing the image is what keeps
        the folder scan off this thread: ``_refresh_source_selectors`` asks the
        sampler to enumerate on every single load, and that call is a cache hit
        only because the worker's listing has already landed here.
        """
        if token != self._image_load_token or not isinstance(payload, dict):
            return
        error = payload.get("error") or ""
        if error:
            self._status.setText(f"Load failed: {error}")
            return
        sets = payload.get("sets")
        if sets is not None:
            self._sampler.adopt(payload.get("directory"), sets,
                                payload.get("channels") or [])
        path, arr = payload.get("path"), payload.get("array")
        if path is None or arr is None:
            self._status.setText("No supported preview image found.")
            return
        self._install_loaded_image(Path(path), arr)

    def shutdown(self) -> None:
        """Abandon any load in flight and leave no QThread behind.

        Called from :meth:`closeEvent`, and safe to call directly when a
        screen is torn down without one.
        """
        runner = getattr(self, "_load_jobs", None)
        if runner is not None:
            runner.shutdown()

    def closeEvent(self, event):    # noqa: N802 (Qt naming)
        """Cancel a load in progress rather than let it outlive the panel."""
        self.shutdown()
        super().closeEvent(event)

    def _install_loaded_image(self, path: Path, arr: np.ndarray) -> None:
        """Replace preview state with an already-decoded image."""
        # A new image invalidates everything derived from the old one,
        # including the run in flight. The raw masks and the flow images used
        # to survive this, so the next filter change — or an in-flight preview
        # landing a moment later — re-drew the previous image's masks over the
        # new one and raised IndexError as soon as the two differed in size.
        self.cancel_preview()
        self._image = arr
        self._image_path = Path(path)
        self._masks = {}
        self._raw_masks = {}
        self._flows = {}
        self._path_label.setText(str(path))
        self._refresh_source_selectors()
        note = self.sample_note()
        self._status.setText(f"Loaded {arr.shape} {arr.dtype}"
                             + (f" — {note}" if note else ""))
        self._refresh_canvases()

    # -- FOV / channel selectors ------------------------------------------

    def _refresh_source_selectors(self) -> None:
        """Re-fill the sets and channel dropdowns for the loaded image.

        The sets dropdown lists a **sample**, not the folder: see
        :class:`~spacr.qt.widgets.preview_controls.ImageSetSampler`. The
        enumeration behind it is cached per folder, so this — which runs on
        every single image load — re-scans nothing once the folder is known.
        """
        if self._image_path is not None:
            self._sampler.enumerate(
                Path(self._image_path).parent, SUPPORTED_SUFFIXES)
        self._sample_note = apply_sample_to_combo(
            self._fov_box, self._max_sets_box, self._sampler,
            self._image_path, tooltip="Field of view")
        channels = (int(self._image.shape[2])
                    if self._image is not None and self._image.ndim == 3
                    else 0)
        populate_channel_combo(self._channel_box, channels)

    def sample_note(self) -> str:
        """The sentence stating this preview is a sample of N of M sets."""
        return getattr(self, "_sample_note", "")

    def _on_max_sets_changed(self, value: int) -> None:
        """Draw a new sample at the user's new cap — without re-enumerating."""
        if not self._sampler.set_max(int(value)):
            return
        self._refresh_source_selectors()
        self._announce_sample()

    def _announce_sample(self) -> None:
        """Restate the sample on the status line, where the user is looking."""
        note = self.sample_note()
        if note:
            self._status.setText(note[:1].upper() + note[1:])

    def _on_fov_changed(self, *_args) -> None:
        """Load the field of view the user picked from the dropdown."""
        if self._loading_fov:
            return
        path = self._fov_box.currentData()
        if not path:
            return
        # The loaded file may be a different channel of the very set the combo
        # points at; comparing raw paths would reload it for no reason.
        picked = self._sampler.set_for_path(path)
        if picked is not None and picked == self._sampler.set_for_path(
                self._image_path):
            return
        if picked is None and self._image_path is not None \
                and str(self._image_path) == str(path):
            return
        self._loading_fov = True
        try:
            # enumerate_sets=False: this path came *out* of the sampler, so the
            # folder is already enumerated. Re-scanning would spend a full pass
            # over the plate to rediscover the listing we are holding, which is
            # the cost the sampling work in 5d5c5c92 removed.
            self.load_source_async(path, enumerate_sets=False)
        finally:
            self._loading_fov = False

    def display_channel(self) -> Optional[int]:
        """Channel index the canvases show, or ``None`` for all channels."""
        return selected_channel(self._channel_box)

    def _background_for_channel(self, channel: Optional[int]) -> Optional[float]:
        """The background threshold that applies to one displayed channel.

        ``None`` when nothing should be removed from it. The channel is
        matched to an object type the same way the pipeline does it in
        :func:`spacr.io._normalize_img_batch`: a channel is the cell channel
        or the nucleus channel, and it takes that object's background. A
        channel belonging to neither -- a stain the user is only looking at
        -- is left alone, because no background was ever chosen for it.
        """
        if channel is None or not hasattr(self, "_common_widgets"):
            return None
        if not self._widget_value(self._common_widgets["remove_background"]):
            return None
        channels = {"cell": int(self._cell_channel.value()),
                    "nucleus": int(self._nucleus_channel.value())}
        for obj in self._selected_object_types():
            if channels.get(obj) == int(channel):
                return float(self._widget_value(
                    self._common_widgets["background"]))
        return None

    def _apply_display_background(self, shown):
        """Show the intensity image the segmentation actually ran on.

        Background removal used to happen only inside the worker, so the
        masks moved when it was switched on and the image they were drawn
        over did not -- the one pane that could show you *why* the objects
        changed was the pane still displaying the original pixels.

        The threshold is the same one the worker applies, so this is not a
        second implementation of the rule: both zero everything below
        ``{obj}_background``, and both leave what is above it untouched.
        """
        if shown is None:
            return shown
        channel = self.display_channel()
        if channel is not None:
            background = self._background_for_channel(channel)
            if background is None:
                return shown
            # `channel_view` returns a VIEW into `self._image`. Writing
            # zeros through it would destroy the loaded image, so the next
            # render -- and the segmentation worker, which reads the same
            # array -- would see an image already thresholded once, again.
            out = shown.copy()
            out[out < background] = 0
            return out

        # "All channels": each one takes its own object's background, so a
        # composite cannot show a cleaned cell channel beside a raw nucleus.
        if getattr(shown, "ndim", 0) != 3:
            return shown
        out = None
        for index in range(shown.shape[2]):
            background = self._background_for_channel(index)
            if background is None:
                continue
            if out is None:
                out = shown.copy()
            plane = out[..., index]
            plane[plane < background] = 0
        return shown if out is None else out

    def _display_image(self) -> Optional[np.ndarray]:
        """The loaded image reduced to the selected display channel."""
        return self._apply_display_background(
            channel_view(self._image, self.display_channel()))

    def _on_display_channel_changed(self, *_args) -> None:
        """Re-render both canvases for the newly selected channel."""
        self._refresh_canvases()

    def set_propagate_callback(self, cb) -> None:
        """Register a callback(dict) used to push tuned live settings back to
        the main settings panel (wired by the AppScreen)."""
        self._propagate_cb = cb

    def settings_for_propagation(self) -> dict:
        """Map the live-preview widget values to main-panel settings keys."""
        model = self._model_box.currentText()
        out = {
            "model_name": model,
            "cell_channel": int(self._cell_channel.value()),
            "nucleus_channel": int(self._nucleus_channel.value()),
            "cell_diameter": float(self._diameter.value()),
            "cell_FT": float(self._flow.value()),
            "cell_CP_prob": float(self._prob.value()),
            "normalize": bool(self._normalise_check.isChecked()),
            "lower_percentile": float(self._lo_pct.value()),
        }
        # Per-compartment + common tuning settings (only present once the
        # compartment widgets have been built).
        if hasattr(self, "_compartment_widgets"):
            out.update(self._compartment_settings())
        return out

    def propagate_settings(self) -> None:
        """Send the current live settings to the main panel (if a callback is
        registered). Called on any live-settings change while the dialog's
        Propagate toggle is on."""
        if self._propagate_cb is not None:
            try:
                self._propagate_cb(self.settings_for_propagation())
            except Exception:
                LOG.debug("propagate_settings failed", exc_info=True)

    def apply_settings(self, settings: dict):
        """Copy relevant values from a Mask-app ``settings`` dict, and
        cache the whole dict for the Pre / Post routes to read from."""
        self._settings = dict(settings)
        try:
            if "diameter" in settings:
                self._diameter.setValue(float(settings["diameter"]))
            if "flow_threshold" in settings:
                self._flow.setValue(float(settings["flow_threshold"]))
            if "CP_prob" in settings:
                self._prob.setValue(float(settings["CP_prob"]))
            if "cell_channel" in settings and settings["cell_channel"] is not None:
                self._cell_channel.setValue(int(settings["cell_channel"]))
            if "nucleus_channel" in settings and settings["nucleus_channel"] is not None:
                self._nucleus_channel.setValue(int(settings["nucleus_channel"]))
            if "model_name" in settings:
                idx = self._model_box.findText(str(settings["model_name"]))
                if idx >= 0:
                    self._model_box.setCurrentIndex(idx)
        except Exception:
            LOG.debug("apply_settings failed", exc_info=True)

    def current_params(self) -> dict:
        """Snapshot for tests + external callers."""
        return {
            "model": self._model_box.currentText(),
            "diameter": self._diameter.value(),
            "flow_threshold": self._flow.value(),
            "cellprob": self._prob.value(),
            "object_types": self._selected_object_types(),
            "cell_channel": self._cell_channel.value(),
            "nucleus_channel": self._nucleus_channel.value(),
            "normalise": self._normalise_check.isChecked(),
            "lo_pct": float(self._lo_pct.value()),
            "hi_pct": float(self._hi_pct.value()),
            "outline_thickness": self._outline_thickness.value(),
            "outline_colour": self._outline_colour.currentText(),
            "display_channel": self.display_channel(),
            "fov": self._fov_box.currentText(),
        }

    def cancel_preview(self) -> bool:
        """Abandon the preview run in flight, if there is one.

        Cellpose exposes no interrupt, so the thread is left to run itself
        out; bumping the run token is what makes its answer land as a no-op
        (:meth:`_on_worker_done` drops results carrying a stale token).

        :returns: True when a running worker was abandoned.
        """
        self._run_token += 1
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Preview cancelled.")
            self._run_btn.setEnabled(True)
            return True
        return False

    def run_preview(self):
        if self._image is None:
            self._status.setText("Load an image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Preview already running.")
            return
        self._release_worker()
        req = self._build_request()
        self._run_btn.setEnabled(False)
        self._status.setText("Running preview…")
        worker = _PreviewWorker(req, self, token=self._run_token)
        worker.finished_masks.connect(self._on_worker_done)
        worker.flows_ready.connect(self._on_flows_ready)
        # NOT worker.deleteLater. ``finished`` is emitted from inside the
        # worker thread, so scheduling the object's C++ deletion off it hands
        # Qt a second owner for an object Python already owns, and the two
        # race — see the measured account in spacr.qt.bridge.make_thread
        # (3 crashes in 8 runs of the stress harness). The relay below is a
        # bound method rather than a lambda so Qt can see a receiving QObject
        # with GUI-thread affinity and queues the call onto the GUI thread; a
        # plain closure would be invoked directly on the worker thread.
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _release_worker(self) -> None:
        """Free the previous worker, whose thread has already finished.

        The worker is parented to the panel, so C++ owns it and it would
        otherwise live — holding a reference to a full-size preview image —
        until the panel itself is destroyed. Unparenting hands ownership back
        to Python, which frees it here, on the thread that holds it. Only ever
        called for a worker that is no longer running, so ``wait`` returns at
        once.
        """
        old = self._worker
        self._worker = None
        if old is None:
            return
        old.wait()
        old.setParent(None)

    def _on_worker_finished(self) -> None:
        """Relay for the worker thread's own ``finished`` signal.

        A bound method on purpose (see :meth:`run_preview`). Re-enabling Run
        here as well as in :meth:`_on_worker_done` is what keeps the button
        usable after a run whose result was discarded as stale, or a worker
        that died without emitting a result at all.
        """
        self._run_btn.setEnabled(True)

    # -- internals ---------------------------------------------------------

    # -- per-compartment tuning widgets -----------------------------------

    def _build_compartment_widgets(self) -> None:
        """Create the common + per-compartment tuning widgets.

        They live on the panel (hidden) so their values persist across opens
        of the Live settings dialog, which re-parents them into its panels and
        hands them back on close — the same pattern the segmentation widgets
        use. Nothing is added to the compact panel layout.

        Populates:
          * ``self._common_widgets`` — signal-to-noise / remove-background /
            background controls that apply to whichever object is chosen.
          * ``self._compartment_widgets[compartment][suffix]`` — the per-
            compartment tuning spinners/checks/combos.
          * ``self._adjust_cells`` — the cell-only "adjust cells" toggle.
        """
        def _spin(kind, spin_args):
            if kind == "float":
                w = QDoubleSpinBox(self)
                lo, hi, dv = spin_args
                w.setRange(float(lo), float(hi)); w.setValue(float(dv))
                w.setDecimals(3)
            elif kind == "int":
                w = QSpinBox(self)
                lo, hi, dv = spin_args
                w.setRange(int(lo), int(hi)); w.setValue(int(dv))
            elif kind == "bool":
                w = Toggle(parent=self)
            elif kind == "method":
                w = QComboBox(self)
                w.addItems(list(INTENSITY_THRESHOLD_METHODS))
            else:
                raise ValueError(kind)
            w.hide()
            return w

        # Pull the informative spaCR setting descriptions for tooltips.
        try:
            from spacr.settings import descriptions as _spacr_desc
        except Exception:
            _spacr_desc = {}

        # Common controls — one widget each, retargeted to the chosen object
        # at propagation time (see settings_for_propagation).
        self._common_widgets: Dict[str, QWidget] = {
            "signal_to_noise": _spin("int", (0, 100_000, 10)),
            "remove_background": _spin("bool", None),
            "background": _spin("int", (0, 100_000, 100)),
        }
        # Both reach the displayed intensity image, not only the worker, so
        # both have to repaint. Without this the toggle looked inert until
        # the next Run: the pixels it removes were already gone from the
        # segmentation and still on screen.
        self._common_widgets["remove_background"].toggled.connect(
            self._refresh_canvases)
        self._common_widgets["background"].valueChanged.connect(
            self._refresh_canvases)
        # Which channel gets thresholded depends on the cell/nucleus channel
        # indices and on which object is selected, so moving any of those has
        # to repaint too -- otherwise pointing "cell" at a different channel
        # leaves the cleaned pixels on the old one.
        self._cell_channel.valueChanged.connect(self._refresh_canvases)
        self._nucleus_channel.valueChanged.connect(self._refresh_canvases)
        self._object_box.currentIndexChanged.connect(self._refresh_canvases)
        self._common_widgets["signal_to_noise"].setToolTip(
            "(int) Signal-to-noise ratio used to set the normalisation "
            "intensity range for the chosen object's channel.")
        self._common_widgets["remove_background"].setToolTip(
            "(bool) Zero every pixel below the background intensity in the "
            "chosen object's channel before segmentation. Applies to the "
            "object selected above — with 'cell + nucleus' chosen, each "
            "channel uses its own background.")
        self._common_widgets["background"].setToolTip(
            "(int) Pixels below this intensity are set to 0 in the chosen "
            "object's channel when 'Remove background' is on. Everything "
            "above it is left where it is.")
        # Cell-only extra.
        self._adjust_cells = _spin("bool", None)
        self._adjust_cells.setToolTip(
            "(bool) Adjust cell masks using the nucleus/pathogen masks.")

        self._compartment_widgets: Dict[str, Dict[str, QWidget]] = {}
        for comp in COMPARTMENTS:
            group: Dict[str, QWidget] = {}
            for suffix, label, kind, spin_args in COMPARTMENT_FIELDS:
                w = _spin(kind, spin_args)
                key = f"{comp}_{suffix}"
                desc = _spacr_desc.get(key) or _spacr_desc.get(suffix)
                w.setToolTip(desc if desc else f"{label} for {comp} objects.")
                group[suffix] = w
            self._compartment_widgets[comp] = group

        # Re-filter the cached masks live whenever any filter widget changes,
        # so tuning updates the preview instantly (no Cellpose re-run).
        for w in self._all_compartment_widgets():
            for sig_name in ("valueChanged", "currentTextChanged", "toggled"):
                sig = getattr(w, sig_name, None)
                if sig is not None:
                    try:
                        sig.connect(lambda *_: self._recompute_masks())
                    except (TypeError, RuntimeError):
                        pass

    def _all_compartment_widgets(self) -> List[QWidget]:
        ws: List[QWidget] = list(self._common_widgets.values())
        ws.append(self._adjust_cells)
        for group in self._compartment_widgets.values():
            ws.extend(group.values())
        return ws

    def _primary_object(self) -> str:
        """The compartment the common controls target — the first selected."""
        return self._selected_object_types()[0]

    @staticmethod
    def _widget_value(w):
        if isinstance(w, Toggle):
            return bool(w.isChecked())
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.value()

    def _compartment_settings(self) -> dict:
        """Map every compartment + common tuning widget to its setting key."""
        out: dict = {}
        for comp, group in self._compartment_widgets.items():
            for suffix, w in group.items():
                out[f"{comp}_{suffix}"] = self._widget_value(w)
        # The common controls are one widget each, retargeted to whatever is
        # selected. Written for EVERY selected object type, not just the
        # primary: with "cell + nucleus" chosen, keying them off
        # `_primary_object()` alone wrote `remove_background_cell` and left
        # the nucleus channel with no key at all, so the segmentation worker
        # -- which looks up `remove_background_{obj}` per object in its loop
        # -- silently skipped it and the toggle appeared to do half a job.
        for obj in self._selected_object_types():
            out[f"{obj}_Signal_to_noise"] = self._widget_value(
                self._common_widgets["signal_to_noise"])
            out[f"remove_background_{obj}"] = self._widget_value(
                self._common_widgets["remove_background"])
            out[f"{obj}_background"] = self._widget_value(
                self._common_widgets["background"])
        out["adjust_cells"] = self._widget_value(self._adjust_cells)
        return out

    def _selected_object_types(self) -> Tuple[str, ...]:
        current = self._object_box.currentText()
        if current == "cell + nucleus":
            return ("cell", "nucleus")
        return (current,)

    def _build_request(self) -> PreviewRequest:
        obj_types = self._selected_object_types()
        channels = {
            "cell":    self._cell_channel.value(),
            "nucleus": self._nucleus_channel.value(),
        }
        # One unified settings dict drives both background subtraction
        # (pre) and filtering (post): the common "remove background" +
        # "background" controls and the per-compartment filter values. No more
        # Pre/Post checkboxes — the settings apply whenever they're set.
        merged = dict(self._settings)
        if hasattr(self, "_compartment_widgets"):
            merged.update(self._compartment_settings())
        pre = merged
        post = merged
        return PreviewRequest(
            image=self._image,
            model=self._model_box.currentText(),
            diameter=self._diameter.value(),
            flow_threshold=self._flow.value(),
            cellprob=self._prob.value(),
            channels=channels,
            object_types=obj_types,
            preprocess_settings=pre,
            postprocess_settings=post,
        )

    #: Fixed colours the outline-colour combo offers by name.
    OUTLINE_COLOURS: Dict[str, Tuple[int, int, int]] = {
        "green":   (32, 220, 32),
        "magenta": (222, 82, 200),
        "yellow":  (255, 220, 32),
        "cyan":    (32, 200, 220),
        "white":   (240, 240, 240),
        "red":     (240, 60, 60),
    }

    def _outline_rgb(self) -> Optional[Tuple[int, int, int]]:
        """Translate the outline-colour combo choice into an RGB tuple,
        or ``None`` for ``auto`` and ``color (random)``. ``auto`` is drawn
        from :meth:`_auto_outline_colour` and ``color (random)`` is handled
        per object label by :func:`overlay_masks`."""
        return self.OUTLINE_COLOURS.get(self._outline_colour.currentText())

    def _roll_auto_outline_colours(self) -> None:
        """Draw a fresh random colour per compartment for ``auto`` mode.

        ``auto`` used to mean "the compartment's fixed colour", which made
        every cell preview green no matter what — the setting looked stuck.
        It now means a random colour, re-rolled once per preview run so the
        outline stays put while the user tunes thickness or normalisation.
        """
        self._auto_outline_colours = {
            comp: random_outline_colour() for comp in COMPARTMENTS}

    def _auto_outline_colour(self, obj_type: str) -> Tuple[int, int, int]:
        """The current random ``auto`` colour for one compartment."""
        colour = self._auto_outline_colours.get(obj_type)
        if colour is None:
            colour = random_outline_colour()
            self._auto_outline_colours[obj_type] = colour
        return colour

    def _auto_outline_map(self) -> Dict[str, Tuple[int, int, int]]:
        """Per-compartment ``auto`` colours covering everything on screen."""
        if not self._auto_outline_colours:
            self._roll_auto_outline_colours()
        for obj_type in self._masks:
            self._auto_outline_colour(obj_type)
        return dict(self._auto_outline_colours)

    def _on_outline_colour_changed(self, *_args) -> None:
        """Re-render, re-rolling the random colours when ``auto`` is chosen."""
        if self._outline_colour.currentText() == "auto":
            self._roll_auto_outline_colours()
        self._refresh_canvases()

    def _refresh_canvases(self):
        """Re-render both views from the current image + masks."""
        if self._image is None:
            return
        norm = self._normalise_check.isChecked()
        lo = float(self._lo_pct.value())
        hi = float(self._hi_pct.value())
        shown = self._display_image()
        src_pix = numpy_to_qpixmap(
            _to_uint8(shown, normalise=norm, lo_pct=lo, hi_pct=hi))
        self._src_view.set_pixmap(src_pix)

        mode = self._view_mode.currentText() if hasattr(self, "_view_mode") else "Overlay"
        if mode == "Flows" and self._flows:
            self._mask_view.set_pixmap(numpy_to_qpixmap(
                self._flows_rgb()))
        elif mode == "Masks" and self._masks:
            self._mask_view.set_pixmap(numpy_to_qpixmap(
                self._label_rgb()))
        elif self._masks:   # Overlay (default)
            overlay = overlay_masks(
                shown, self._masks,
                outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi,
                random_outline=(
                    self._outline_colour.currentText() == "color (random)"
                ),
                outline_colors=self._auto_outline_map())
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)

    def _on_flows_ready(self, flows, token: int = -1) -> None:
        """Store the per-object Cellpose flow RGB images from a preview run."""
        if self._stale(token):
            return
        self._flows = flows or {}
        if hasattr(self, "_view_mode") and self._view_mode.currentText() == "Flows":
            self._refresh_canvases()

    def _label_rgb(self) -> np.ndarray:
        """Render the current label masks as a distinct-colour image (0 = black).

        The chosen outline colour tints this view too. It used to be painted
        straight from :data:`OBJECT_COLORS`, so the ``Masks`` view stayed
        green for cells no matter which colour the user picked — the colour
        control simply did not reach this renderer.
        """
        h, w = self._image.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        chosen = self._outline_rgb()
        random_mode = self._outline_colour.currentText() == "color (random)"
        auto_colours = self._auto_outline_map()
        for obj, mask in self._masks.items():
            if mask is None or mask.shape[:2] != (h, w):
                continue
            labels = mask.astype(np.int64)
            present = labels > 0
            if not present.any():
                continue
            if random_mode:
                # Same categorical map the overlay uses, so 'color (random)'
                # means one thing in both views.
                ids = np.unique(labels[present])
                palette = _random_outline_palette(
                    ids, RANDOM_OUTLINE_SEEDS.get(obj, 0))
                out[present] = palette[np.searchsorted(ids, labels[present])]
                continue
            if chosen is not None:
                base_rgb = chosen
            else:
                base_rgb = auto_colours.get(
                    obj, OBJECT_COLORS.get(obj, (200, 200, 200)))
            base = np.array(base_rgb, dtype=np.uint8)
            # Vary brightness a little per label so neighbours are separable.
            shade = (0.5 + 0.5 * ((labels % 7) / 6.0)).astype(np.float32)
            for c in range(3):
                out[..., c] = np.where(
                    present,
                    np.clip(base[c] * shade, 0, 255).astype(np.uint8),
                    out[..., c])
        return out

    def _flows_rgb(self) -> np.ndarray:
        """Combine per-object flow RGB images (first available / max-blend)."""
        imgs = [np.asarray(f) for f in self._flows.values()
                if f is not None and np.asarray(f).ndim == 3]
        if not imgs:
            h, w = self._image.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        out = imgs[0].astype(np.uint8)
        for f in imgs[1:]:
            if f.shape == out.shape:
                out = np.maximum(out, f.astype(np.uint8))
        return out[..., :3]

    def _on_model_or_object_changed(self, *_):
        """Refresh visibility state — no visible-widget mutation on
        the compact layout anymore (options are hidden by default and
        only shown inside the Live Settings dialog when it's open).
        The dialog re-reads visibility rules on open, so nothing to
        do here at rest."""
        # Kept as a hook so any observers subscribed to model/object
        # combo changes still fire.
        dlg = self._live_settings_dialog
        if dlg is not None:
            try:
                dlg.refresh_visibility()
            except Exception:
                pass

    def open_live_settings(self):
        """Open (or focus) the Live Settings modal.

        The dialog rehomes every hidden state widget into its form so
        the user's edits go straight into `self._*` — nothing to sync.
        On close, widgets are re-parented back to `self` (hidden again)
        so state persists across opens.
        """
        if (self._live_settings_dialog is not None
                and self._live_settings_dialog.isVisible()):
            self._live_settings_dialog.raise_()
            self._live_settings_dialog.activateWindow()
            return
        self._live_settings_dialog = LiveSettingsDialog(self)
        self._live_settings_dialog.finished.connect(self._on_settings_closed)
        self._live_settings_dialog.show()

    def _on_settings_closed(self, *_):
        # Refresh canvases in case a visual-only setting changed (e.g.
        # outline colour) while the dialog was open.
        self._refresh_canvases()
        self._live_settings_dialog = None

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose preview image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if path:
            # The chosen file may live in a folder the sampler has never seen,
            # so this one does enumerate — off the GUI thread.
            self.load_source_async(path)

    def _on_hover(self, x: int, y: int):
        """Render the pinned hover-info line for the pixel under the cursor."""
        if self._image is None:
            return
        h = self._image.shape[0]
        w = self._image.shape[1] if self._image.ndim >= 2 else 0
        if not (0 <= y < h and 0 <= x < w):
            self._hover_label.setText(
                "Hover over the image to inspect pixels.")
            return
        # Intensities across every channel
        if self._image.ndim == 3:
            vals = tuple(int(v) for v in self._image[y, x])
            i_str = f"channels={vals}"
        else:
            i_str = f"intensity={int(self._image[y, x])}"
        # Mask hit-tests
        hits = []
        for obj, mask in self._masks.items():
            if mask is None or mask.size == 0:
                continue
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                lbl = int(mask[y, x])
                if lbl > 0:
                    area = int((mask == lbl).sum())
                    hits.append(f"{obj}=#{lbl} area={area}px")
        obj_str = f"  {'  '.join(hits)}" if hits else ""
        self._hover_label.setText(f"(x={x:>4d}, y={y:>4d})  {i_str}{obj_str}")

    def _stale(self, token: int) -> bool:
        """True when ``token`` belongs to a superseded run.

        ``-1`` is the direct-call escape hatch used by tests and by callers
        that push a result in by hand; those are never stale.
        """
        return token >= 0 and token != self._run_token

    def _on_worker_done(self, masks, err, token: int = -1):
        if self._stale(token):
            LOG.debug("dropping stale preview result (token %s, now %s)",
                      token, self._run_token)
            return
        self._run_btn.setEnabled(True)
        if err:
            self._status.setText(f"Preview failed: {err}")
            self.preview_ready.emit(None)
            return
        if masks is None or not masks or self._image is None:
            self._status.setText("Preview returned no masks.")
            return
        # Cache the raw masks so filters can be re-applied live, then filter.
        self._raw_masks = masks
        self._recompute_masks(snapshot=True)

    def _obj_channel(self, obj: str) -> int:
        """Intensity channel index used for a given compartment."""
        if obj == "cell":
            return int(self._cell_channel.value())
        if obj == "nucleus":
            return int(self._nucleus_channel.value())
        return 0

    def _recompute_masks(self, snapshot: bool = False) -> None:
        """Re-apply the current per-compartment filters to the cached raw
        masks and refresh the views — no Cellpose re-run. Called both after a
        preview and whenever a filter widget changes."""
        raw = getattr(self, "_raw_masks", None)
        if not raw or self._image is None:
            return
        if snapshot:
            # A new run gets a new 'auto' colour. Re-rolling here rather than
            # on every repaint keeps the outline steady while the user drags
            # thickness or percentile sliders.
            self._roll_auto_outline_colours()
        post = dict(self._settings)
        if hasattr(self, "_compartment_widgets"):
            post.update(self._compartment_settings())
        out = {}
        for obj, raw_mask in raw.items():
            intensity = _select_channel(self._image, self._obj_channel(obj))
            out[obj] = _apply_size_filter(raw_mask, post, obj,
                                          intensity_img=intensity)
        self._masks = out
        counts = [f"{k}={int(v.max() if v.size else 0)}"
                    for k, v in out.items()]
        self._status.setText(f"Found {', '.join(counts)}.")
        self._refresh_canvases()
        if snapshot:
            self._snapshot_run(out, counts)
        self.preview_ready.emit(out)

    # -- comparison scrubber ----------------------------------------------

    def _snapshot_run(self, masks, counts) -> None:
        """Record a preview run (image + masks + display params) so the user
        can scrub back to compare it against later runs."""
        if self._image is None:
            return
        snap = {
            "image": self._image,
            "masks": {k: v for k, v in masks.items()},
            "norm": self._normalise_check.isChecked(),
            "lo": float(self._lo_pct.value()),
            "hi": float(self._hi_pct.value()),
            "model": self._model_box.currentText(),
            "object": self._object_box.currentText(),
            "summary": ", ".join(counts),
        }
        self._history.append(snap)
        # Cap the history so memory stays bounded on long tuning sessions.
        if len(self._history) > 50:
            self._history = self._history[-50:]
        n = len(self._history)
        self._compare_row.setVisible(n >= 2)
        self._compare_slider.blockSignals(True)
        self._compare_slider.setMaximum(n - 1)
        self._compare_slider.setValue(n - 1)      # newest
        self._compare_slider.blockSignals(False)
        self._compare_label.setText(f"{n}/{n}")

    def _on_compare_scrub(self, idx: int) -> None:
        """Render the historical run at ``idx`` into the two canvases."""
        if not (0 <= idx < len(self._history)):
            return
        snap = self._history[idx]
        img = channel_view(snap["image"], self.display_channel())
        norm, lo, hi = snap["norm"], snap["lo"], snap["hi"]
        src_pix = numpy_to_qpixmap(
            _to_uint8(img, normalise=norm, lo_pct=lo, hi_pct=hi))
        self._src_view.set_pixmap(src_pix)
        if snap["masks"]:
            # The random and auto modes have to be forwarded here too. They
            # were not, so scrubbing back through history repainted every
            # outline in the per-compartment default — green for cells —
            # whatever the user had chosen.
            overlay = overlay_masks(
                img, snap["masks"], outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi,
                random_outline=(
                    self._outline_colour.currentText() == "color (random)"
                ),
                outline_colors=self._auto_outline_map())
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)
        self._compare_label.setText(
            f"{idx + 1}/{len(self._history)}  "
            f"{snap['model']}/{snap['object']}  {snap['summary']}")

    # -- the model list is live ------------------------------------------
    def refresh_model_choices(self) -> None:
        """Re-read the Cellpose model list and add anything new.

        `spacr.settings.cellpose_model_choices` only reads the API when
        Cellpose is already imported, because importing it costs ~2.5 s and
        this panel is built while a page is being laid out. That means the
        first build usually gets the shipped fallback — so ask again every
        time the panel is shown. After the first segmentation Cellpose is
        loaded and a checkpoint the user registered appears here.

        Additive on purpose: the current selection is never disturbed, and
        an entry is never removed, so a value the user picked cannot vanish
        under them because a probe came back thinner.
        """
        wanted = _model_menu()
        have = {self._model_box.itemText(i)
                for i in range(self._model_box.count())}
        for index, name in enumerate(wanted):
            if name not in have:
                self._model_box.insertItem(index, name)

    def showEvent(self, event):  # noqa: N802 (Qt naming)
        """Refresh the model list whenever the panel comes back on screen."""
        super().showEvent(event)
        self.refresh_model_choices()


# ---------------------------------------------------------------------------
# Live Settings dialog
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QScrollArea,
)


class LiveSettingsDialog(QDialog):
    """Modal dialog that surfaces every live-preview setting.

    Re-parents the panel's hidden state widgets into a QFormLayout so
    edits go straight into the panel's canonical fields — nothing to
    sync manually. On close, widgets are returned to the panel hidden
    so their values persist across opens.

    Rows shown (per the user's spec):
      * Normalisation upper + lower percentile
      * Outline colour
      * Outline thickness
      * Model
      * Flow threshold
      * Cell probability
      * Object type
      * Object channel (cell / nucleus depending on selection)
      * Pre  (bool)
      * Post (bool)
    """

    def __init__(self, panel: "LivePreviewPanel"):
        super().__init__(panel)
        self._panel = panel
        self.setWindowTitle("Live settings")
        outer = QVBoxLayout(self)

        # Show the widgets we'll be adding, then re-hide them on close.
        for w in self._managed_widgets():
            w.show()

        # Row of side-by-side panels: the segmentation + common controls on the
        # left, then one greyed-until-chosen panel per compartment to the right.
        panels_row = QHBoxLayout()
        panels_row.setSpacing(12)

        seg_group = QGroupBox("Segmentation")
        form = QFormLayout(seg_group)
        form.addRow("Model", panel._model_box)
        form.addRow("Primary object", panel._object_box)
        form.addRow("Cell channel", panel._cell_channel)
        form.addRow("Nucleus channel", panel._nucleus_channel)
        form.addRow("Diameter", panel._diameter)
        form.addRow("Flow threshold", panel._flow)
        form.addRow("Cell probability", panel._prob)
        form.addRow(panel._normalise_check)
        form.addRow("Lower percentile", panel._lo_pct)
        form.addRow("Upper percentile", panel._hi_pct)
        form.addRow("Outline colour", panel._outline_colour)
        form.addRow("Outline thickness", panel._outline_thickness)
        # Common controls — apply to whichever object is chosen.
        panel._common_widgets["signal_to_noise"].show()
        panel._common_widgets["remove_background"].show()
        panel._common_widgets["background"].show()
        form.addRow("Signal to noise", panel._common_widgets["signal_to_noise"])
        form.addRow("Remove background", panel._common_widgets["remove_background"])
        form.addRow("Background", panel._common_widgets["background"])
        panels_row.addWidget(seg_group)

        # One panel per compartment, greyed unless it's the chosen object.
        self._compartment_groupboxes: Dict[str, QGroupBox] = {}
        for comp in COMPARTMENTS:
            box = QGroupBox(comp.capitalize())
            cform = QFormLayout(box)
            for suffix, label, _kind, _args in COMPARTMENT_FIELDS:
                w = panel._compartment_widgets[comp][suffix]
                w.show()
                cform.addRow(label, w)
            if comp == "cell":
                panel._adjust_cells.show()
                cform.addRow("Adjust cells", panel._adjust_cells)
            self._compartment_groupboxes[comp] = box
            panels_row.addWidget(box)

        # Wrap the (wide) panel row in a horizontal scroll area so it fits on
        # screen no matter how many compartments are shown.
        row_host = QWidget()
        row_host.setLayout(panels_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(row_host)
        outer.addWidget(scroll, 1)

        # Run button lives in the dialog so settings can be iterated without
        # closing it — edit a value, hit Run, see the result, repeat.
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._run_btn = QPushButton("Run preview")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._panel.run_preview)
        buttons.addButton(self._run_btn, QDialogButtonBox.ActionRole)
        # Propagate toggle — when on (blue, like the AI / Live toggles), edits
        # here are pushed into the main settings panel so tuning in the live
        # preview updates the run configuration.
        self._propagate_btn = QPushButton("Propagate settings")
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, changes made here are copied into the main settings "
            "panel.")
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        buttons.addButton(self._propagate_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)

        # Re-gate the form whenever the object type or model changes, so
        # irrelevant settings grey out live.
        panel._object_box.currentTextChanged.connect(self.refresh_visibility)
        panel._model_box.currentTextChanged.connect(self.refresh_visibility)
        panel._normalise_check.toggled.connect(self.refresh_visibility)

        # Widgets whose changes propagate to the main panel while the toggle
        # is on — the segmentation controls plus every compartment/common knob.
        self._propagate_sources = [
            panel._model_box, panel._object_box, panel._cell_channel,
            panel._nucleus_channel, panel._diameter, panel._flow,
            panel._prob, panel._normalise_check, panel._lo_pct, panel._hi_pct,
        ] + panel._all_compartment_widgets()

        self.refresh_visibility()

        # Open wide enough to show the Segmentation panel + all four compartment
        # panels without the user having to drag the window wider. Clamp to the
        # available screen so it still fits on small displays (the horizontal
        # scroll area handles any remaining overflow).
        try:
            avail = self.screen().availableGeometry()
            want = row_host.sizeHint().width() + 48
            self.resize(min(want, avail.width() - 80), min(760, avail.height() - 80))
        except Exception:
            self.resize(1400, 720)

    def _on_propagate_toggled(self, on: bool) -> None:
        """Connect/disconnect live→main propagation and do an initial push."""
        for w in self._propagate_sources:
            for sig_name in ("valueChanged", "currentTextChanged", "toggled"):
                sig = getattr(w, sig_name, None)
                if sig is None:
                    continue
                try:
                    if on:
                        sig.connect(self._panel.propagate_settings)
                    else:
                        sig.disconnect(self._panel.propagate_settings)
                except (TypeError, RuntimeError):
                    pass
        if on:
            self._panel.propagate_settings()   # push current values now

    def _managed_widgets(self):
        p = self._panel
        return [p._model_box, p._object_box, p._cell_channel,
                p._nucleus_channel, p._diameter, p._flow, p._prob,
                p._normalise_check, p._lo_pct, p._hi_pct,
                p._outline_colour, p._outline_thickness,
                ] + p._all_compartment_widgets()

    def _install_api_tooltips(self) -> None:
        """Attach linked Mask API help to every setting in this popup."""
        from ..screens.settings_model import install_api_tooltips

        p = self._panel
        widget_keys = {
            p._model_box: "model_name",
            p._object_box: "object_type",
            p._cell_channel: "cell_channel",
            p._nucleus_channel: "nucleus_channel",
            p._diameter: "cell_diameter",
            p._flow: "cell_FT",
            p._prob: "cell_CP_prob",
            p._normalise_check: "normalize",
            p._lo_pct: "lower_percentile",
            p._hi_pct: "upper_percentile",
            p._outline_colour: "outline_color",
            p._outline_thickness: "outline_thickness",
            p._common_widgets["signal_to_noise"]: "cell_Signal_to_noise",
            p._common_widgets["remove_background"]: "remove_background_cell",
            p._common_widgets["background"]: "cell_background",
            p._adjust_cells: "adjust_cells",
        }
        for compartment, fields in p._compartment_widgets.items():
            for suffix, widget in fields.items():
                widget_keys[widget] = f"{compartment}_{suffix}"
        # No link dots here. This dialog has a setting on nearly every row --
        # 68 dots were being drawn, one after each label and one after the
        # combined controls -- so they stopped reading as "click for the API
        # page" and started reading as texture down the form. The hover help
        # is unaffected; it is still on every label.
        install_api_tooltips(self, "mask", widget_keys, api_dots=False)

    def refresh_visibility(self):
        """Grey out settings that don't apply to the current selection.

        Rules (mirroring the pipeline's own relevance):
          * Nothing in the Segmentation group greys out for the model.
            Cellpose 4 ships one set of weights and all three knobs
            (diameter / flow / cell-prob) still reach it — see
            :data:`DIAMETER_TOOLTIP` for the measurement.
          * The object type decides which channel spinners are live: the cell
            channel greys out for a nucleus-only object and vice-versa.
          * Pre-processing knobs (normalise + its two percentiles) are only
            relevant when the *Pre* step is enabled.
          * Overlay / post knobs (outline colour + thickness) are only
            relevant when the *Post* step is enabled.
        """
        p = self._panel

        # -- model: cpsam uses all three. `diameter` used to be disabled
        #    here with the tooltip "Ignored by Cellpose-SAM", which was
        #    false: Cellpose 4 rescales the image by 30/diameter before
        #    it runs (see DIAMETER_TOOLTIP for the measured counts), so
        #    the UI was greying out a control that changes the result. --
        p._diameter.setEnabled(True)
        p._diameter.setToolTip(DIAMETER_TOOLTIP)
        p._flow.setEnabled(True)
        p._prob.setEnabled(True)
        p._flow.setToolTip("")
        p._prob.setToolTip("")

        # -- object: which channel spinners apply --
        selected = set(p._selected_object_types())
        p._cell_channel.setEnabled("cell" in selected)
        p._nucleus_channel.setEnabled("nucleus" in selected)

        # -- compartment panels: show only the primary object's panel plus,
        #    for 'cell + nucleus', a secondary Nucleus panel. The other
        #    compartments' panels are hidden entirely (their settings are the
        #    same shape and only the chosen object's are relevant). --
        ordered = list(p._selected_object_types())
        primary = ordered[0] if ordered else "cell"
        for comp, box in self._compartment_groupboxes.items():
            is_primary = (comp == primary)
            is_secondary = (comp == "nucleus" and "nucleus" in selected
                            and not is_primary)
            box.setVisible(is_primary or is_secondary)
            box.setEnabled(True)
            if is_primary:
                box.setTitle(f"{comp.capitalize()} (primary object)")
            elif is_secondary:
                box.setTitle("Nucleus (secondary object)")

        # -- Normalisation is always available (independent of the Pre step
        #    and of the model, incl. cpsam). The percentile bounds only apply
        #    while normalisation is on. --
        p._normalise_check.setEnabled(True)
        p._normalise_check.setToolTip("")
        norm_on = p._normalise_check.isChecked()
        for w in (p._lo_pct, p._hi_pct):
            w.setEnabled(norm_on)
            w.setToolTip("" if norm_on
                         else "Enable 'Normalise' to set percentile bounds")

        # Overlay / outline knobs are always relevant (they style the overlay
        # view), so they stay enabled.
        for w in (p._outline_colour, p._outline_thickness):
            w.setEnabled(True)
        self._install_api_tooltips()

    def closeEvent(self, event):
        """Re-hide the state widgets so the compact layout stays clean."""
        for w in self._managed_widgets():
            w.hide()
            # Re-parent back to the panel so the widget survives dialog
            # deletion (Qt would otherwise destroy children).
            w.setParent(self._panel)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Back-compat shims for callers that predate the multi-object rewrite
# ---------------------------------------------------------------------------

def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Legacy single-mask overlay retained for older imports."""
    return overlay_masks(image, {"cell": mask})
