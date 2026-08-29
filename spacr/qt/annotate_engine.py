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
import re
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity

from spacr.database_concurrency import (
    connect as connect_database,
)
from spacr.database_concurrency import (
    transaction,
)

LOG = logging.getLogger("spacr.qt.annotate_engine")


def _ensure_cache_budget_sweep() -> None:
    """Start the GUI sweep if resource cleanup was registered before Qt."""
    cleanup = sys.modules.get("spacr.qt.resource_cleanup")
    install = getattr(cleanup, "install_budget_sweep", None)
    if callable(install):
        install()


# ---------------------------------------------------------------------------
# Color helpers (identical to AnnotateApp._int_to_color / _label_to_color)
# ---------------------------------------------------------------------------

_PHI = 0.618033988749895


#: Relative luminance of the light theme's tile, and the contrast a border
#: must reach against it. 4.5 is the WCAG floor for normal text; a 2px border
#: is more forgiving than text, but these marks carry the annotation and are
#: looked at for hours.
_LIGHT_TILE_LUMINANCE = 0.9046
_CONTRAST_TARGET = 4.5


def _relative_luminance(rgb) -> float:
    """WCAG relative luminance of an ``(r, g, b)`` triple in 0..1."""
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in rgb]
    return (0.2126 * channels[0] + 0.7152 * channels[1]
            + 0.0722 * channels[2])


def _darken_until_readable(h: float, s: float, value: float) -> float:
    """Lower ``value`` until the colour clears the contrast target.

    Returns the LARGEST value that still reads, so a colour is never darker
    than it needs to be -- classes stay distinguishable from each other, not
    just from the background. Bottoms out at 0.2, below which every hue is
    the same near-black and the class identity is lost; a colour that cannot
    reach the target by then is as dark as it is useful to make it.
    """
    step = value
    while step > 0.2:
        rgb = colorsys.hsv_to_rgb(h, s, step)
        contrast = ((_LIGHT_TILE_LUMINANCE + 0.05)
                    / (_relative_luminance(rgb) + 0.05))
        if contrast >= _CONTRAST_TARGET:
            return step
        step -= 0.01
    return 0.2


def label_to_hex(val: Optional[int], dark: bool = True) -> Optional[str]:
    """Map an annotation value to a hex border color.

    None / 0 / non-int -> None (no border).
    1 -> blue, 2 -> red, 3+ -> golden-ratio hue rotation.

    THE COLOURS DEPEND ON THE THEME, because contrast does. The original
    palette was tuned against a dark tile and measured 3.5-12.0 against
    #1e1e1e -- all comfortable. Against a light tile (#f5f5f5) the SAME
    colours measure 1.28-4.34, with five of the first six below the 3.0
    readability floor:

        class 1  #3ea6ff   6.43 on dark   2.38 on light
        class 4  #55f2d8  11.97 on dark   1.28 on light

    That is issue #6 -- "labels do not appear with good contrast in the
    annotation app like they do on Linux machines" -- and it is a theme
    difference rather than a platform one: macOS defaults to the
    light appearance far more often.

    HUE IS PRESERVED so a class keeps its identity across themes; only
    saturation and value move, deepening the colour until it reads against
    a pale background.

    :param val: the annotation value.
    :param dark: True for the dark theme's palette, False for the light
        theme's deepened one.
    """
    try:
        v = int(val)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None

    if v == 1:
        h, s, value = 0.5806, 0.76, 1.00      # the blue, as HSV
    elif v == 2:
        h, s, value = 0.0000, 0.68, 1.00      # the red
    else:
        h, s, value = (v * _PHI) % 1.0, 0.65, 0.95

    if not dark:
        # DARKEN UNTIL IT READS, rather than by a fixed factor.
        #
        # How much darkening a hue needs to clear the contrast floor against
        # a pale background depends on the hue itself: blues and magentas
        # manage at 0.79 of their value, yellow needs 0.48, cyan 0.52. A flat
        # factor either leaves the bright hues unreadable or drives the dark
        # ones to near-black, and class 3+ hues are generated by a golden-
        # ratio rotation, so the set is open-ended and cannot be tuned by
        # hand.
        s = min(1.0, s + 0.20)
        value = _darken_until_readable(h, s, value)

    r, g, b = colorsys.hsv_to_rgb(h, s, value)
    return "#{:02x}{:02x}{:02x}".format(int(r*255+0.5), int(g*255+0.5), int(b*255+0.5))


# ---------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------

def load_crop_image(path: str, db_path: Optional[str] = None,
                    stored_channel_order: str = "auto",
                    display_order: str = "rgb",
                    display_primaries: str = "rgb") -> Image.Image:
    """Open one object crop PNG as an 8-bit RGB image in display order.

    :func:`spacr.crops.read_crop_png` resolves the stored format from the
    sidecar marker, database, or legacy fallback before applying the requested
    display order. Sixteen-bit single-channel images are narrowed consistently
    instead of being clipped by an RGB conversion.

    Two different questions are kept separate rather than combined into one
    control:

        stored_channel_order   Physical channel order in the file, resolved
                               from its sidecar marker or database. ``'auto'``
                               is recommended when metadata is available.
        display_order          Preferred on-screen order, independent of file
                               storage. Defaults to ``'rgb'``.

    :param path: the crop PNG.
    :param db_path: optional ``measurements.db``, consulted when the crop
        folder carries no sidecar marker.
    :param display_order: one of ``spacr.crops.DISPLAY_ORDERS``. Applied
        AFTER the format is corrected, so the two never fight.
    :returns: PIL ``Image`` in RGB mode.
    """
    from ..crops import (
        CROP_FORMAT_CURRENT,
        CROP_FORMAT_RGB,
        apply_display_order,
        apply_display_primaries,
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
    corrected = read_crop_png(path, fmt=stored_format, db_path=db_path)
    # Format first, preference second. Reversing them would permute planes
    # that are still in the wrong slots, and the two would compose into an
    # order neither the file nor the user asked for.
    # Order, then primaries. The order says WHICH source plane fills a slot;
    # the primaries say what colour that slot is drawn in. Doing primaries
    # first would recolour planes that are about to move.
    shown = apply_display_order(corrected, display_order)
    return Image.fromarray(apply_display_primaries(shown, display_primaries))


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


class OutlineCancelled(Exception):
    """Raised when a requested cancellation stops outline generation.

    Cellpose model construction and inference cannot be interrupted safely.
    Cancellation is therefore checked between native calls, and this
    exception unwinds the current page of crops.
    """


def _check_stop(should_stop) -> None:
    """Raise :class:`OutlineCancelled` when the caller has asked to stop.

    A ``should_stop`` that raises is treated as "stop": the usual reason is
    ``RuntimeError: Internal C++ object already deleted`` from a QThread whose
    wrapper has gone, and a caller that no longer exists is not waiting for
    this crop.
    """
    if should_stop is None:
        return
    try:
        stop = bool(should_stop())
    except Exception:                                        # noqa: BLE001
        stop = True
    if stop:
        raise OutlineCancelled()


_cellpose_outline_model = None
# Cellpose/PyTorch model construction and inference enter native code and are
# not safe to run concurrently through one cached model.  Annotate page loads
# used to fan out across several QThreads and ThreadPoolExecutors, so two crops
# could call ``model.eval`` at the same time and take the interpreter down
# without a Python traceback.  RLock lets _cellpose_foreground call the
# separately-tested lazy constructor while holding the same guard.
_cellpose_outline_lock = threading.RLock()


def _get_cellpose_outline_model(should_stop=None):
    """Lazily build + cache a small Cellpose (SAM) model for outline masks.

    :param should_stop: asked once before the model is built and once after
        the lock is taken. Building it imports cellpose and torch and reads a
        1.2 GB checkpoint, so a caller that has already given up must not pay
        for it.
    """
    global _cellpose_outline_model
    _check_stop(should_stop)
    with _cellpose_outline_lock:
        _check_stop(should_stop)
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


def _cellpose_foreground(channel_2d, should_stop=None) -> "np.ndarray":
    """Return a boolean foreground mask for one channel using Cellpose.

    :param should_stop: asked immediately before ``model.eval``. The wait for
        the lock is itself unbounded — another crop may be inside a forward
        pass — so the question is asked again on the far side of it rather
        than only on the way in.
    """
    _check_stop(should_stop)
    with _cellpose_outline_lock:
        model = _get_cellpose_outline_model(should_stop=should_stop)
        _check_stop(should_stop)
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


#: How many outline masks to keep. A montage tab is a few hundred crops and
#: each mask is one bit per pixel; 512 covers a screenful several times over
#: for well under a megabyte.
_MASK_CACHE_SIZE = 512

#: The cache itself: {(channel bytes, shape, sigma, factor): mask}.
_MASK_CACHE: "OrderedDict" = None
_MASK_CACHE_USED: Dict[Any, float] = {}


def _foreground_mask(channel, sigma: float, factor: float):
    """Return and cache the Otsu foreground mask for one channel.

    The mask depends only on the pixel bytes, shape, smoothing width, and
    threshold factor. Display-only changes such as normalization, opacity,
    outline thickness, and percentiles can therefore reuse it. Content-based
    keys also survive crop-object replacement during a montage reload.
    """
    global _MASK_CACHE
    if _MASK_CACHE is None:
        _MASK_CACHE = OrderedDict()
        _ensure_cache_budget_sweep()

    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
    from skimage.filters import threshold_otsu

    contiguous = np.ascontiguousarray(channel)
    key = (hash(contiguous.tobytes()), contiguous.shape, round(sigma, 4),
           round(factor, 4))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        _MASK_CACHE.move_to_end(key)
        _MASK_CACHE_USED[key] = time.time()
        return cached

    smoothed = gaussian_filter(contiguous.astype(np.float32), sigma=sigma)
    try:
        otsu = threshold_otsu(smoothed)
    except Exception:
        otsu = float(np.percentile(smoothed, 50.0))
    threshold = float(min(255.0, max(0.0, otsu * factor)))
    mask = smoothed > threshold
    mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    mask = binary_fill_holes(mask)

    _MASK_CACHE[key] = mask
    _MASK_CACHE_USED[key] = time.time()
    while len(_MASK_CACHE) > _MASK_CACHE_SIZE:
        old, _ = _MASK_CACHE.popitem(last=False)
        _MASK_CACHE_USED.pop(old, None)
    return mask


#: {(mask bytes, shape, thickness): edge}
_EDGE_CACHE: "OrderedDict" = None
_EDGE_CACHE_USED: Dict[Any, float] = {}


def _edge_of(mask, thickness: int):
    """The boundary of ``mask``, dilated to ``thickness``, remembered."""
    from skimage.morphology import dilation, disk
    from skimage.segmentation import find_boundaries

    global _EDGE_CACHE
    if _EDGE_CACHE is None:
        _EDGE_CACHE = OrderedDict()
        _ensure_cache_budget_sweep()

    packed = np.packbits(np.ascontiguousarray(mask))
    key = (hash(packed.tobytes()), tuple(np.shape(mask)), int(thickness))
    cached = _EDGE_CACHE.get(key)
    if cached is not None:
        _EDGE_CACHE.move_to_end(key)
        _EDGE_CACHE_USED[key] = time.time()
        return cached

    edge = find_boundaries(mask, mode="inner").astype(np.uint8)
    if thickness > 0:
        edge = dilation(edge > 0, disk(thickness)).astype(np.uint8)

    _EDGE_CACHE[key] = edge
    _EDGE_CACHE_USED[key] = time.time()
    while len(_EDGE_CACHE) > _MASK_CACHE_SIZE:
        old, _ = _EDGE_CACHE.popitem(last=False)
        _EDGE_CACHE_USED.pop(old, None)
    return edge


def forget_outline_masks() -> None:
    """Drop every cached mask. For tests, and for a caller changing plates."""
    global _MASK_CACHE, _EDGE_CACHE
    _MASK_CACHE = None
    _EDGE_CACHE = None
    _MASK_CACHE_USED.clear()
    _EDGE_CACHE_USED.clear()


def cache_budget_entries():
    """Measured records for decoded outline arrays retained between draws."""
    rows = []
    now = time.time()
    for kind, cache, used in (
            ("mask", _MASK_CACHE, _MASK_CACHE_USED),
            ("edge", _EDGE_CACHE, _EDGE_CACHE_USED)):
        for key, value in list((cache or {}).items()):
            rows.append(((kind, key), max(0, int(value.nbytes)),
                         float(used.get(key, now)), False))
    return rows


def drop_cache_budget_entry(record_key) -> bool:
    """Evict one decoded array selected by the global memory policy."""
    kind, key = record_key
    cache = _MASK_CACHE if kind == "mask" else _EDGE_CACHE
    used = _MASK_CACHE_USED if kind == "mask" else _EDGE_CACHE_USED
    if cache is None:
        return False
    existed = key in cache
    cache.pop(key, None)
    used.pop(key, None)
    return existed


# ---------------------------------------------------------------------------
# Which objects are drawn at all
#
# ONE NUMBER FOR EVERY COLOUR was the whole complaint. A crop's red, green and
# blue planes hold different things -- a nucleus, a cell, a parasite -- and a
# size window that suits one of them is nonsense for the other two. So the
# filter is written per plane, and each plane gets a window on its SIZE and a
# window on its BRIGHTNESS.
#
# EMPTY IS A VALUE, and it is the value that means "no bound on this side".
# It is how a user turns half a filter off, so it survives a round trip
# instead of being helpfully replaced with a zero -- and zero is not the same
# answer, because a zero minimum on intensity is a real (if weak) claim about
# what may be drawn.
# ---------------------------------------------------------------------------

#: The colour planes a filter can be written against, in the order the
#: settings form draws them.
FILTER_CHANNELS: Tuple[str, ...] = ("r", "g", "b")

#: What each plane's two rows bound. ``area`` is the object's size in pixels;
#: ``intensity`` is its MEAN value in that same plane, 0-255 after decode --
#: mean rather than peak, so a single hot pixel cannot carry a dim object past
#: a brightness floor.
FILTER_MEASURES: Tuple[str, ...] = ("area", "intensity")


def filter_key(channel: str, measure: str) -> str:
    """Return the settings key for a channel and measurement pair."""
    return f"{str(channel).strip().lower()}_{str(measure).strip().lower()}"


def filter_bound(value) -> Optional[float]:
    """Parse a filter bound, returning ``None`` for empty or invalid input."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:                       # NaN compares false with all
        return None
    return number


def empty_object_filters() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return all object-filter bounds in their disabled state."""
    return {filter_key(channel, measure): (None, None)
            for channel in FILTER_CHANNELS
            for measure in FILTER_MEASURES}


def normalize_object_filters(
    object_filters: Optional[Mapping] = None,
    object_size=None,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Normalise object-filter bounds and migrate legacy size limits.

    Legacy ``object_size`` limits are applied to the area filter for each
    colour channel, with non-positive legacy limits treated as disabled.
    Explicit ``object_filters`` values take precedence; invalid or empty
    explicit bounds are disabled, while zero remains a valid explicit bound.

    :param object_filters: ``{'r_area': (min, max), ...}``; partial maps are
        accepted and unknown keys are ignored.
    :param object_size: Legacy ``(min, max)`` area limits in pixels.
    :returns: A new dictionary containing every supported key and a pair of
        floats or ``None``.
    """
    bounds = empty_object_filters()
    try:
        legacy_lo, legacy_hi = object_size
    except (TypeError, ValueError):
        legacy_lo = legacy_hi = None
    legacy_lo = filter_bound(legacy_lo)
    legacy_hi = filter_bound(legacy_hi)
    if legacy_lo is not None and legacy_lo <= 0:
        legacy_lo = None
    if legacy_hi is not None and legacy_hi <= 0:
        legacy_hi = None
    if legacy_lo is not None or legacy_hi is not None:
        for channel in FILTER_CHANNELS:
            bounds[filter_key(channel, "area")] = (legacy_lo, legacy_hi)
    for key, pair in dict(object_filters or {}).items():
        key = str(key).strip().lower()
        if key not in bounds:
            continue
        try:
            low, high = pair
        except (TypeError, ValueError):
            continue
        bounds[key] = (filter_bound(low), filter_bound(high))
    return bounds


def _keep_objects(mask, plane, area, intensity):
    """Drop the connected components outside ``area`` and ``intensity``.

    :param mask: boolean foreground.
    :param plane: the same channel's values, for the brightness window.
    :param area: ``(min, max)`` in pixels; either side may be ``None``.
    :param intensity: ``(min, max)`` mean value; either side may be ``None``.
    :returns: the mask with the objects outside either window removed.
    """
    from scipy.ndimage import label

    area_lo, area_hi = area
    intensity_lo, intensity_hi = intensity
    if all(bound is None for bound in
           (area_lo, area_hi, intensity_lo, intensity_hi)):
        return mask
    labelled, count = label(mask)
    if count <= 0:
        return mask
    flat = labelled.ravel()
    sizes = np.bincount(flat, minlength=count + 1).astype(np.float64)
    totals = np.bincount(flat, weights=plane.astype(np.float64).ravel(),
                         minlength=count + 1)
    means = totals / np.maximum(sizes, 1.0)
    keep = np.ones(sizes.shape, dtype=bool)
    keep[0] = False                     # label 0 is the background
    if area_lo is not None:
        keep &= sizes >= area_lo
    if area_hi is not None:
        keep &= sizes <= area_hi
    if intensity_lo is not None:
        keep &= means >= intensity_lo
    if intensity_hi is not None:
        keep &= means <= intensity_hi
    return keep[labelled]


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
    object_filters: Optional[Mapping] = None,
    should_stop=None,
) -> Image.Image:
    """Overlay per-channel object outlines on `base_img`.

    Mirrors AnnotateApp.outline_image (Tk) semantics: for every channel
    in `outline_channels`, compute an Otsu-thresholded foreground mask
    on the corresponding channel of `full_img`, extract the boundary,
    optionally dilate it, then alpha-blend it over the channel in
    `base_img` with `edge_transparency/100` opacity. Peak-normalized so
    thin edges stay visible.

    WHICH objects get an outline is decided per plane by ``object_filters``
    -- an area window and a mean-intensity window for each of red, green and
    blue. ``object_size`` is the one-window-for-every-plane setting those
    replaced and is still honoured: it is migrated onto the three area rows
    by :func:`normalize_object_filters`, so a caller that passes only it gets
    exactly what it always got.

    :param should_stop: optional callable asked before each channel's Cellpose
        model construction and forward pass. When it answers True the work is
        abandoned by raising :class:`OutlineCancelled` rather than finishing a
        page nobody is waiting for; ``'otsu'`` outlines are fast enough that
        they are never interrupted mid-channel.
    """
    if not outline_channels or edge_transparency <= 0:
        return base_img
    from scipy.ndimage import (binary_closing, binary_fill_holes,
                               gaussian_filter)
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
    bounds = normalize_object_filters(object_filters, object_size)
    for ch in outline_channels:
        idx = channel_map[ch]
        if edge_image:
            base_arr[:, :, idx] = full_arr[:, :, idx]
        if outline_method == 'cellpose':
            # Small Cellpose model gives cleaner object outlines than Otsu.
            try:
                fg_mask = _cellpose_foreground(full_arr[:, :, idx],
                                               should_stop=should_stop)
            except OutlineCancelled:
                # NOT a cellpose failure, so NOT a reason to fall back to
                # Otsu: the caller has gone and the rest of this page must
                # not be computed. Re-raised ahead of the generic handler
                # below, which would otherwise swallow it and go on working
                # for a screen that is being torn down.
                raise
            except Exception:
                # Fall back to Otsu if cellpose isn't available / fails.
                outline_method = 'otsu'
        if outline_method != 'cellpose':
            fg_mask = _foreground_mask(full_arr[:, :, idx],
                                       float(edge_sigma), factor)
        fg_mask = _keep_objects(
            fg_mask, full_arr[:, :, idx],
            bounds[filter_key(ch, "area")],
            bounds[filter_key(ch, "intensity")])
        # THE EDGE IS CACHED TOO, for the same reason as the mask: it is a
        # function of the mask and the thickness alone, and neither moves
        # when a user changes normalisation, percentiles or transparency.
        # `find_boundaries` and `dilation` were the other half of the cost.
        edge = _edge_of(fg_mask, int(max(0, round(edge_thickness))) - 1)
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
    #: A DISPLAY preference, not a claim about the file. One of
    #: `spacr.crops.DISPLAY_ORDERS`; the default is the identity.
    display_order: str = "rgb"
    #: One of `spacr.crops.DISPLAY_PRIMARIES`. A view setting; see
    #: crops.apply_display_primaries.
    #:
    #: The dataclass default is the identity, because this module must stay
    #: importable without Qt. The SCREEN starts it from the global
    #: colour-vision preference instead -- see
    #: `LivePreviewContract.display_primaries`. A user who has told spaCR
    #: once that they are colour-blind should not have to say it again on
    #: every screen.
    display_primaries: str = "rgb"
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
    #: THE OLD ONE-WINDOW-FOR-EVERY-PLANE size filter, kept so a settings
    #: file written against it still means what it meant. It is migrated onto
    #: the three area rows of `object_filters` when the outline is drawn; the
    #: screen writes the new fields.
    object_size: Tuple[int, int] = (0, 0)
    #: ``{'r_area': (min, max), 'r_intensity': (min, max), 'g_area': ...}``:
    #: six rows of two fields, one pair per plane per measure. ``None`` on a
    #: side means NO BOUND there, which is how half a filter is turned off --
    #: see `normalize_object_filters`.
    #:
    #: EMPTY BY DEFAULT, and that is not the same as twelve empty bounds. A
    #: key that is absent has never been written, so a legacy `object_size`
    #: is still migrated onto it; a key that is present and ``(None, None)``
    #: is a user who cleared that row, and the old value does not come back.
    object_filters: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict)
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


# ---------------------------------------------------------------------------
# The image-type filter, with operators
# ---------------------------------------------------------------------------

def parse_image_type(expression: Optional[str]) -> Tuple[str, List[str]]:
    """Turn an image-type expression into a SQL fragment and its parameters.

    The filter used to be one substring matched with ``LIKE %x%``, which can
    only ever say what a path MUST contain. There was no way to ask for the
    complement -- "the cells with no pathogen crop" -- which is half of most
    comparisons (issue #7).

    The grammar is small and deliberately close to what someone would type:

        pathogen                  contains "pathogen"
        !pathogen                 does NOT contain it
        NOT pathogen              the same, spelled out
        cell AND nucleus          contains both
        cell OR nucleus           contains either
        cell AND NOT pathogen     mixes them

    AND binds tighter than OR, as everywhere else. Terms are matched
    case-insensitively, since ``LIKE`` is already case-insensitive for ASCII
    in SQLite and a user typing "Pathogen" means the same thing.

    EVERY TERM IS A BOUND PARAMETER. Nothing the user types is interpolated
    into SQL, so a path fragment containing a quote is a path fragment and
    not an injection.

    :param expression: the user's filter, or None/empty for "no filter".
    :returns: ``(sql, params)`` where sql is a bracketed boolean expression
        over ``png_path``, or ``("", [])`` when there is nothing to filter.
    :raises ValueError: on an expression that cannot be read, naming what was
        wrong -- an empty NOT, a dangling operator, unbalanced parentheses.
    """
    text = (expression or "").strip()
    if not text:
        return "", []

    tokens = _tokenise_image_type(text)
    if not tokens:
        return "", []
    sql, params, rest = _parse_or(tokens)
    if rest:
        raise ValueError(
            f"could not read the image filter after {' '.join(rest[:3])!r}; "
            f"expected AND, OR, or the end of the expression")
    return sql, params


_IMAGE_TYPE_OPERATORS = {"and", "or", "not", "(", ")"}


def _tokenise_image_type(text: str) -> List[str]:
    """Split on whitespace and parentheses, turning a leading ! into NOT."""
    out: List[str] = []
    for raw in re.findall(r"\(|\)|[^\s()]+", text):
        if raw in ("(", ")"):
            out.append(raw)
        elif raw.startswith("!") and len(raw) > 1:
            out.extend(["NOT", raw[1:]])
        elif raw == "!":
            out.append("NOT")
        else:
            out.append(raw)
    return out


def _parse_or(tokens):
    sql, params, rest = _parse_and(tokens)
    while rest and rest[0].lower() == "or":
        right_sql, right_params, rest = _parse_and(rest[1:])
        sql = f"({sql} OR {right_sql})"
        params = params + right_params
    return sql, params, rest


def _parse_and(tokens):
    sql, params, rest = _parse_term(tokens)
    while rest and rest[0].lower() == "and":
        right_sql, right_params, rest = _parse_term(rest[1:])
        sql = f"({sql} AND {right_sql})"
        params = params + right_params
    return sql, params, rest


def _parse_term(tokens):
    if not tokens:
        raise ValueError("the image filter ends after an operator")
    head, rest = tokens[0], tokens[1:]
    if head.lower() == "not":
        sql, params, rest = _parse_term(rest)
        return f"(NOT {sql})", params, rest
    if head == "(":
        sql, params, rest = _parse_or(rest)
        if not rest or rest[0] != ")":
            raise ValueError("the image filter has an unclosed '('")
        return f"({sql})", params, rest[1:]
    if head.lower() in _IMAGE_TYPE_OPERATORS:
        raise ValueError(
            f"the image filter has {head!r} where a path fragment was "
            f"expected")
    return "png_path LIKE ?", [f"%{head}%"], rest


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
        where, params = parse_image_type(image_type)
        clause = f" WHERE {where}" if where else ""
        cur.execute(f'SELECT COUNT(*) FROM "png_list"{clause}', params)
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
        where, params = parse_image_type(image_type)
        clause = f"WHERE {where} " if where else ""
        cur.execute(
            f'SELECT png_path, "{col}" FROM "png_list" '
            f'{clause}LIMIT ? OFFSET ?',
            (*params, page_size, offset),
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
        where, params = parse_image_type(image_type)
        clause = f" WHERE {where}" if where else ""
        cur.execute(f'SELECT "{col}" FROM "png_list"{clause}', params)
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
