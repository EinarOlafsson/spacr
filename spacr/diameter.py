"""Propose Cellpose ``diameter`` values from blob statistics, without Cellpose.

Why this exists
---------------
``diameter`` is the most consequential Cellpose knob spaCR exposes and it is
the one users guess at. Under Cellpose 4 (``cpsam``) the constructor argument
``diam_mean=`` is ignored, but ``CellposeModel.eval(diameter=...)`` is very
much alive: it rescales the input by ``30.0 / diameter`` so objects land near
the network's ~30 px working size. Feed it a value two-fold off and every
downstream mask, count and measurement inherits the error. So a defensible
starting number is worth a few seconds of arithmetic.

Design constraint: this module must be usable *before* committing to a
segmentation run, which means it may not cost what a segmentation run costs.
It therefore imports **no torch and no cellpose** — that is a tested property,
not an aspiration (see ``tests/test_diameter_estimator.py``). It reuses
:mod:`spacr.validate` for filename-metadata parsing, which is the other
deliberately dependency-light module in the package (``spacr.utils``, where
``_get_regex`` and ``_extract_filename_metadata`` live, imports torch and
cellpose at module scope and so cannot be touched from here). The regexes in
``spacr.validate.METADATA_REGEXES`` mirror ``spacr.utils._get_regex``, and the
channel ordering used below mirrors
``spacr.io._rename_and_organize_image_files``, which concatenates one plane per
distinct ``chanID`` in sorted order — so channel index *i* is the *i*-th
sorted ``chanID``, exactly as ``cell_channel`` and friends mean it.

Method
------
For each requested object channel, on each sampled field:

1. **Flatten.** Denoise with a 1 px Gaussian, then subtract a heavily
   smoothed copy (sigma = max(32, min(H, W) / 4)) to remove illumination
   gradients. The sigma is deliberately far larger than any plausible object
   so that flattening removes vignetting without eating the objects — the
   opposite trade-off (a tight rolling ball) shrinks what it is trying to
   measure.
2. **Reject noise.** Compare the structural amplitude of the flattened plane
   (p99 - p30) against the *pixel-level* noise scale, measured as
   ``1.4826 * MAD(img - gaussian(img, 1))`` on the raw plane. A pure-noise
   plane scores below 1; a plane with real objects scores in the tens. Below
   ``min_snr`` the field is discarded rather than thresholded, because Otsu
   will happily bisect pure noise and hand back a confident-looking number.
3. **Threshold and label.** Otsu, fill holes, label, drop components that
   touch the image border (truncated, so their size is a lie) and components
   that are absurd (equivalent diameter below ``min_object_diameter``, or area
   above ``max_object_fraction`` of the field). Characteristic size is the
   median equivalent diameter, ``2 * sqrt(area / pi)``.
4. **Cross-check by distance transform.** Step 3 has one dominant failure
   mode: a confluent monolayer fuses into a single component, that component
   touches the border and is dropped, and the estimate is then computed from
   whatever debris survived — biased **low**, and silently. So the Euclidean
   distance transform of the (unfilled) foreground is computed as well, its
   local maxima are taken as one seed per object (two passes: a coarse pass
   sets the suppression radius for the refined pass), and a watershed on
   ``-EDT`` splits the fused foreground back into objects whose equivalent
   diameters are measured the same way.

Both estimates are always computed, and choosing between them is where the
care goes. Fusion is declared only when *both* halves of its signature are
present: the threshold path kept nothing (or far fewer objects than the
transform resolves), **and** the field is dense enough for fusion to be the
explanation. Requiring both matters in each direction. A count disagreement
alone is not fusion — a hollow, membrane-only object is one correct component
by area but shatters into dozens of arcs under the distance transform, so a
bare ratio test would discard the right answer in favour of the wall
thickness, which is the same silent collapse entered from the other side. A
high foreground fraction alone is not fusion either — a dense but
well-separated field can reach 30% foreground and still be measured correctly
by thresholding. When only one signal fires, the threshold estimate stands and
the confidence comes down instead. When the two estimates disagree by more
than 1.5-fold the confidence is downgraded and the note says so, because that
disagreement is the honest signal that neither number should be trusted blind.

Public API
----------
``DiameterEstimate``
    One proposal, with its plausible range, provenance and confidence.
``estimate_diameters(src, channels, n_fields=5, ...)``
    ``{object_type: DiameterEstimate}``.
``format_estimates(estimates)``
    Human-readable block, ready to print.
``channels_from_settings(settings)``
    Pull ``cell_channel`` / ``nucleus_channel`` / ... out of a settings dict.

Where a user meets this
-----------------------
:class:`spacr.qt.prerun.DiameterPanel` sits above the Run row on the Mask
screen: it reads the channels off the settings form with
:func:`channels_from_settings`, calls :func:`estimate_diameters` on a worker
thread, and shows one row per object type carrying the proposal *and its
evidence* — the 10th-90th percentile range, how many objects it was pooled
from, how many fields contributed, the method and the confidence. A proposal
without those is just a different guess. Nothing is written into
``<object>_diameter`` until the user presses **Use**, and
:attr:`DiameterEstimate.usable` is checked first, so the NaN an unmeasurable
channel returns cannot reach a settings field.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .validate import (
    IMAGE_EXTENSIONS,
    _candidate_patterns,
    _listdir,
    _peek_planes,
)

__all__ = [
    "DiameterEstimate",
    "OBJECT_TYPES",
    "SETTING_KEYS",
    "channels_from_settings",
    "estimate_diameters",
    "format_estimates",
]

#: Object types whose diameter spaCR lets you set, in report order.
OBJECT_TYPES: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "organelle")

#: Which settings key each object type's estimate belongs in.
SETTING_KEYS: Dict[str, str] = {obj: f"{obj}_diameter" for obj in OBJECT_TYPES}

_HIGH, _MEDIUM, _LOW = "high", "medium", "low"
_LEVELS = (_HIGH, _MEDIUM, _LOW)


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------

@dataclass
class DiameterEstimate:
    """One proposed diameter, with everything needed to disbelieve it.

    :param object_type: ``'cell'``, ``'nucleus'``, ``'pathogen'`` or
        ``'organelle'``.
    :param diameter: proposed value in pixels — the median object diameter
        across the sampled fields. ``float('nan')`` when nothing usable was
        found; see :attr:`usable`. It is never a fabricated fallback.
    :param low: 10th percentile of the measured object diameters, in pixels.
    :param high: 90th percentile of the measured object diameters, in pixels.
    :param n_objects: how many objects the estimate was pooled from.
    :param n_fields: how many fields actually contributed (which can be fewer
        than requested).
    :param method: which measurement produced :attr:`diameter` —
        ``'threshold_otsu'``, ``'watershed_edt'``, or ``'none'``.
    :param confidence: ``'high'``, ``'medium'`` or ``'low'``.
    :param note: why the number is what it is, which confidence downgrades
        applied, and what to check if it looks wrong.
    """

    object_type: str
    diameter: float
    low: float
    high: float
    n_objects: int
    n_fields: int
    method: str
    confidence: str
    note: str

    @property
    def usable(self) -> bool:
        """False when no defensible number could be produced.

        Callers must check this before writing :attr:`diameter` into a
        settings dict: an unusable estimate carries NaN precisely so that a
        fabricated number cannot leak into a run.
        """
        return isinstance(self.diameter, float) and not math.isnan(self.diameter)

    def __str__(self) -> str:
        if not self.usable:
            return f"{self.object_type}: no estimate ({self.note})"
        return (
            f"{self.object_type}: {self.diameter:.1f} px "
            f"({self.low:.1f}-{self.high:.1f}, {self.confidence} confidence)"
        )


def _no_estimate(object_type: str, note: str, n_fields: int = 0) -> DiameterEstimate:
    """Build the explicit 'we could not measure this' result."""
    nan = float("nan")
    return DiameterEstimate(
        object_type=object_type,
        diameter=nan,
        low=nan,
        high=nan,
        n_objects=0,
        n_fields=int(n_fields),
        method="none",
        confidence=_LOW,
        note=note,
    )


# ---------------------------------------------------------------------------
# settings glue
# ---------------------------------------------------------------------------

def _as_channel_index(value: Any) -> Optional[int]:
    """Coerce a settings channel value to an int index, or None.

    Settings imported from CSV arrive as strings, and ``bool`` is an ``int``
    subclass that must not be mistaken for channel 0 or 1.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    return None


def channels_from_settings(settings: Dict[str, Any]) -> Dict[str, int]:
    """Extract ``{object_type: channel_index}`` from a spaCR settings dict.

    Reads ``cell_channel``, ``nucleus_channel``, ``pathogen_channel`` and
    ``organelle_channel``; object types whose channel is None or unparseable
    are simply absent from the result.

    :param settings: a spaCR settings dict (or anything dict-like).
    :returns: mapping suitable for :func:`estimate_diameters`.
    """
    out: Dict[str, int] = {}
    for obj in OBJECT_TYPES:
        idx = _as_channel_index((settings or {}).get(f"{obj}_channel"))
        if idx is not None:
            out[obj] = idx
    return out


# ---------------------------------------------------------------------------
# field discovery
# ---------------------------------------------------------------------------

@dataclass
class _Source:
    """Where the sampled planes come from, and how to index a channel in them."""

    kind: str = ""                              # 'array' | 'raw' | ''
    where: str = ""                             # human-readable location
    fields: List[Tuple[tuple, Any]] = _dc_field(default_factory=list)
    n_channels: Optional[int] = None
    channel_ids: List[str] = _dc_field(default_factory=list)
    problem: str = ""


def _roots(src: Any) -> List[str]:
    """``src`` may be one folder or a list of folders (``expected_types['src']``)."""
    if src is None:
        return []
    if isinstance(src, (str, os.PathLike)):
        return [os.fspath(src)]
    if isinstance(src, (list, tuple, set)):
        return [os.fspath(s) for s in src if isinstance(s, (str, os.PathLike))]
    return []


def _array_dirs(root: str) -> Iterable[str]:
    """Folders that may hold merged ``(H, W, C)`` arrays, best first.

    ``stack/`` is what ``spacr.io._rename_and_organize_image_files`` writes and
    is the canonical home of the raw channels. ``root`` itself covers a ``src``
    that already points at a stack folder, and ``merged/`` is the post-mask
    product whose leading planes are still the raw channels.
    """
    yield os.path.join(root, "stack")
    yield root
    yield os.path.join(root, "merged")


def _discover_arrays(roots: Sequence[str]) -> _Source:
    """Find merged ``.npy`` arrays under any of ``roots``."""
    src = _Source(kind="array")
    wheres: List[str] = []
    for root in roots:
        for directory in _array_dirs(root):
            names = sorted(f for f in _listdir(directory) if f.endswith(".npy"))
            if not names:
                continue
            for name in names:
                src.fields.append(((root, name), os.path.join(directory, name)))
            wheres.append(directory)
            planes, _, _ = _peek_planes(directory)
            if planes:
                src.n_channels = planes if src.n_channels is None else min(src.n_channels, planes)
            break
    if not src.fields:
        return _Source()
    src.where = ", ".join(wheres)
    src.fields.sort(key=lambda item: item[0])
    return src


def _raw_dirs(root: str) -> Iterable[str]:
    """Folders that may hold raw per-channel acquisition files, best first.

    Mirrors ``spacr.validate._scan_raw_images``: after preprocessing runs with
    ``save_original_images=True`` the raw files have been moved into ``orig/``.
    """
    yield root
    yield os.path.join(root, "orig")
    yield os.path.join(root, "consolidated")


def _discover_raw(roots: Sequence[str], metadata_type: Any, custom_regex: Any) -> _Source:
    """Group raw acquisition files into fields using the metadata regexes.

    Channel identity is the ``chanID`` capture group, and the channel *index*
    is the position of that ``chanID`` in the sorted set of distinct IDs —
    the same mapping ``spacr.io._rename_and_organize_image_files`` bakes into
    the ``stack/`` arrays.
    """
    patterns = _candidate_patterns({"metadata_type": metadata_type, "custom_regex": custom_regex})
    src = _Source(kind="raw")
    wheres: List[str] = []
    groups: Dict[tuple, Dict[str, List[str]]] = {}
    channel_ids = set()

    for root in roots:
        names: List[str] = []
        directory = ""
        for candidate in _raw_dirs(root):
            found = [f for f in _listdir(candidate) if f.lower().endswith(IMAGE_EXTENSIONS)]
            found = [f for f in found if not f.startswith(".")]
            if found:
                names, directory = sorted(found), candidate
                break
        if not names:
            continue

        best: Tuple[int, Any] = (0, None)
        for _label, pattern in patterns:
            try:
                rx = re.compile(pattern)
            except re.error:
                continue
            hits = sum(1 for n in names if rx.match(n) and (rx.match(n).groupdict().get("chanID")))
            if hits > best[0]:
                best = (hits, rx)
        if best[1] is None:
            continue

        rx = best[1]
        wheres.append(directory)
        for name in names:
            match = rx.match(name)
            if not match:
                continue
            groups_d = match.groupdict()
            chan = groups_d.get("chanID")
            if chan is None:
                continue
            key = (
                root,
                str(groups_d.get("plateID") or os.path.basename(os.path.normpath(root))),
                str(groups_d.get("wellID") or ""),
                str(groups_d.get("fieldID") or ""),
                str(groups_d.get("timeID") or ""),
            )
            channel_ids.add(str(chan))
            groups.setdefault(key, {}).setdefault(str(chan), []).append(os.path.join(directory, name))

    if not groups:
        return _Source()
    src.channel_ids = sorted(channel_ids)
    src.n_channels = len(src.channel_ids)
    src.where = ", ".join(wheres)
    src.fields = sorted(groups.items(), key=lambda item: item[0])
    return src


def _discover(src: Any, metadata_type: Any, custom_regex: Any) -> _Source:
    """Locate sampleable fields under ``src``, arrays first then raw files."""
    roots = _roots(src)
    if not roots:
        return _Source(problem="src is empty or not a folder path")
    missing = [r for r in roots if not os.path.isdir(r)]
    if len(missing) == len(roots):
        return _Source(problem=f"no such folder: {missing[0]}")

    found = _discover_arrays(roots)
    if found.fields:
        return found
    found = _discover_raw(roots, metadata_type, custom_regex)
    if found.fields:
        return found
    return _Source(
        problem=(
            "found no merged .npy arrays (stack/, merged/) and no raw image files "
            "matching the metadata regex"
        )
    )


def _sample_indices(n_available: int, n_fields: int, random_state: Optional[int]) -> List[int]:
    """Pick field indices spread across the whole source, never the first N.

    Plates vary systematically down rows and across columns, so the first N
    files are the first N wells and are not representative of anything. With
    ``random_state=None`` an even stride is used, which is deterministic and
    spans the sorted (plate, well, field) ordering end to end; with a seed, a
    reproducible random sample is drawn instead.
    """
    n_fields = max(1, int(n_fields))
    if n_available <= 0:
        return []
    if n_available <= n_fields:
        return list(range(n_available))
    if random_state is None:
        idx = np.linspace(0, n_available - 1, n_fields)
        return sorted(set(int(round(v)) for v in idx))
    rng = np.random.default_rng(random_state)
    return sorted(int(v) for v in rng.choice(n_available, size=n_fields, replace=False))


# ---------------------------------------------------------------------------
# plane loading
# ---------------------------------------------------------------------------

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Reduce a loaded image to a single 2-D plane.

    A trailing axis of 3 or 4 is an RGB(A) read and is averaged to luminance;
    any other leading axis is a z-stack and is maximum-projected, which is
    what ``spacr.io`` does when it builds ``stack/``.
    """
    arr = np.asarray(arr)
    while arr.ndim > 2:
        if arr.shape[-1] in (3, 4) and arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.max(axis=0)
    return arr


def _crop_to(arr: np.ndarray, max_pixels: int) -> np.ndarray:
    """Centre-crop an oversized plane so the estimate stays cheap.

    Cropping loses objects at the field periphery but does not bias their
    size, which is the statistic being measured; downscaling would bias it.
    """
    h, w = arr.shape
    if max_pixels <= 0 or h * w <= max_pixels:
        return arr
    side = int(math.sqrt(max_pixels))
    side = max(64, side)
    r0 = max(0, (h - side) // 2)
    c0 = max(0, (w - side) // 2)
    return arr[r0:r0 + min(side, h), c0:c0 + min(side, w)]


def _load_array_plane(path: str, channel: int) -> np.ndarray:
    """Read one channel out of a merged ``(H, W, C)`` ``.npy``."""
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        if channel != 0:
            raise IndexError(f"{os.path.basename(path)} is 2-D: only channel 0 exists")
        return np.array(arr)
    if channel >= arr.shape[-1]:
        raise IndexError(
            f"{os.path.basename(path)} has {arr.shape[-1]} planes: "
            f"valid channels are 0-{arr.shape[-1] - 1}"
        )
    return np.array(arr[..., channel])


def _load_raw_plane(paths: Sequence[str], max_slices: int = 16) -> np.ndarray:
    """Maximum-project the z-slices of one (field, channel) into a plane."""
    from PIL import Image

    planes = []
    for path in sorted(paths)[:max_slices]:
        with Image.open(path) as img:
            planes.append(_to_2d(np.array(img)))
    if not planes:
        raise ValueError("no readable image for this field/channel")
    shape = planes[0].shape
    planes = [p for p in planes if p.shape == shape]
    return np.max(np.stack(planes), axis=0)


# ---------------------------------------------------------------------------
# per-plane measurement
# ---------------------------------------------------------------------------

@dataclass
class _PlaneResult:
    """What one field's plane yielded, or why it yielded nothing."""

    ok: bool = False
    reason: str = ""
    thresh_diams: np.ndarray = _dc_field(default_factory=lambda: np.empty(0, np.float64))
    split_diams: np.ndarray = _dc_field(default_factory=lambda: np.empty(0, np.float64))
    fg_fraction: float = 0.0


def _region_diameters(
    labels: np.ndarray,
    n_labels: int,
    min_diameter: float,
    max_area: float,
) -> np.ndarray:
    """Equivalent diameters of the labelled regions worth believing.

    Drops the background, every region touching the image border (truncated,
    so its area understates its size), and every region whose size is absurd:
    smaller than ``min_diameter`` across, or larger than ``max_area`` pixels.

    :returns: the surviving regions' equivalent diameters, ``2*sqrt(area/pi)``.
    """
    if n_labels <= 0:
        return np.empty(0, np.float64)
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n_labels + 1)[1:].astype(np.float64)

    border = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border = border[border > 0]

    keep = np.ones(n_labels + 1, dtype=bool)
    keep[0] = False
    if border.size:
        keep[border[border <= n_labels]] = False
    ids = np.nonzero(keep)[0]
    if ids.size == 0:
        return np.empty(0, np.float64)

    kept_areas = areas[ids - 1]
    kept_areas = kept_areas[kept_areas > 0]
    if kept_areas.size == 0:
        return np.empty(0, np.float64)

    diams = 2.0 * np.sqrt(kept_areas / np.pi)
    sane = (diams >= min_diameter) & (kept_areas <= max_area)
    return diams[sane]


def _illumination(img: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth illumination field of ``img``, estimated on a decimated copy.

    A Gaussian with the sigma this module wants (a quarter of the field) has a
    kernel radius of order the image itself, which on a 1500x1500 plane costs
    several seconds — more than the rest of the estimate put together, and
    this module's whole premise is being cheap enough to run before a
    segmentation run. Since the field being estimated is smooth by
    construction, it is computed on a block-mean decimation (a box-filter
    downsample, so nothing aliases) and resized back. Measured on a
    1500x1500 plane this turns 4.4 s into under 0.1 s with no change to the
    recovered diameters.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.transform import resize

    h, w = img.shape
    factor = max(1, int(round(sigma / 8.0)))
    factor = min(factor, max(1, min(h, w) // 64))
    if factor <= 1:
        return gaussian_filter(img, sigma, mode="nearest")
    fh, fw = h - h % factor, w - w % factor
    small = img[:fh, :fw].reshape(fh // factor, factor, fw // factor, factor).mean(axis=(1, 3))
    small_bg = gaussian_filter(small.astype(np.float32), sigma / factor, mode="nearest")
    return resize(small_bg, (h, w), order=1, mode="edge", anti_aliasing=False).astype(np.float32)


def _analyse_plane(
    plane: np.ndarray,
    min_snr: float,
    min_object_diameter: float,
    max_object_fraction: float,
) -> _PlaneResult:
    """Measure characteristic object size in one plane, two independent ways."""
    from scipy.ndimage import (
        binary_fill_holes,
        distance_transform_edt,
        gaussian_filter,
    )
    from scipy.ndimage import label as ndi_label
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    img = np.asarray(plane, dtype=np.float32)
    if img.ndim != 2 or img.size == 0:
        return _PlaneResult(reason="plane is not a non-empty 2-D image")
    h, w = img.shape
    if min(h, w) < 16:
        return _PlaneResult(reason=f"plane is too small to measure ({h}x{w})")

    # 1. flatten: denoise, then remove illumination with a sigma far larger
    #    than any plausible object so the objects survive the subtraction.
    smooth = gaussian_filter(img, 1.0)
    sigma_bg = max(32.0, min(h, w) / 4.0)
    flat = smooth - _illumination(smooth, sigma_bg)

    # 2. reject planes whose structure is indistinguishable from pixel noise.
    resid = img - smooth
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    amplitude = float(np.percentile(flat, 99.0) - np.percentile(flat, 30.0))
    if not np.isfinite(amplitude) or amplitude <= 0:
        return _PlaneResult(reason="plane is flat: no intensity variation to threshold")
    snr = amplitude / noise if noise > 0 else float("inf")
    if snr < min_snr:
        return _PlaneResult(
            reason=f"structure/noise ratio {snr:.1f} is below {min_snr:g}: nothing but background"
        )

    # 3. threshold, fill, label. threshold_otsu only raises on a single-valued
    #    image, which the amplitude check above has already rejected.
    thr = float(threshold_otsu(flat))
    foreground = flat > thr
    # Otsu always leaves at least the brightest pixel above the threshold and
    # at least the dimmest below it, so there is no degenerate all-or-nothing
    # case to guard here; a field whose split is useless falls out downstream
    # as "no object survived the border and size filters".
    fg_fraction = float(foreground.mean())
    filled = binary_fill_holes(foreground)

    max_area = float(max_object_fraction) * float(h * w)
    labels, n_labels = ndi_label(filled)
    thresh_diams = _region_diameters(labels, int(n_labels), min_object_diameter, max_area)

    # 4. distance-transform cross-check: seed one marker per inscribed-circle
    #    maximum and watershed the foreground apart again. Padding by one zero
    #    pixel keeps objects at the image edge bounded instead of letting the
    #    transform run off the array.
    #
    #    This runs on the UNFILLED foreground on purpose. Hole filling is right
    #    for the area measurement above -- a dark nucleus inside a cell is part
    #    of the cell -- but in a confluent packing the interstitial background
    #    between touching objects is also an enclosed hole, and filling it
    #    welds the whole field into one slab whose distance transform knows
    #    nothing about individual objects. Those interstices are precisely the
    #    signal that tells touching objects apart, so the transform keeps them.
    #    The cost is that a genuinely hollow object (a membrane-only ring)
    #    reads small here; when it does, the two measurements disagree, and
    #    _aggregate downgrades the confidence and says so rather than picking
    #    a winner silently.
    padded = np.pad(foreground, 1)
    edt = distance_transform_edt(padded)
    seed_floor = max(1.0, min_object_diameter / 2.0)
    coarse = peak_local_max(edt, min_distance=3, threshold_abs=seed_floor, exclude_border=False)
    split_diams = np.empty(0, np.float64)
    if coarse.size:
        # Second pass: the coarse peaks set the suppression radius, so the
        # refined pass keeps one seed per object instead of one per ripple.
        # It cannot come back empty -- its threshold is no higher than the
        # coarse pass's, so at least the global maximum survives.
        r_coarse = float(np.median(edt[tuple(coarse.T)]))
        coords = peak_local_max(
            edt,
            min_distance=max(3, int(round(r_coarse))),
            threshold_abs=max(seed_floor, 0.4 * r_coarse),
            exclude_border=False,
        )
        markers = np.zeros(edt.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1, dtype=np.int32)
        split = watershed(-edt, markers, mask=padded)[1:-1, 1:-1]
        split_diams = _region_diameters(
            split, int(split.max()), min_object_diameter, max_area
        )

    return _PlaneResult(
        ok=True,
        thresh_diams=thresh_diams,
        split_diams=split_diams,
        fg_fraction=fg_fraction,
    )


def _measure(
    plane: np.ndarray,
    min_snr: float,
    min_object_diameter: float,
    max_object_fraction: float,
) -> _PlaneResult:
    """:func:`_analyse_plane`, with one odd field degraded instead of fatal.

    This is a pre-flight helper. A plate with one corrupt or exotically typed
    field should cost the user that field, not the whole estimate and not a
    traceback in front of a run they have not started yet.
    """
    try:
        return _analyse_plane(plane, min_snr, min_object_diameter, max_object_fraction)
    except Exception as exc:
        return _PlaneResult(reason=f"could not measure field: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# aggregation and confidence
# ---------------------------------------------------------------------------

def _demote(level: str, steps: int = 1) -> str:
    """Move a confidence level down the ladder, saturating at 'low'."""
    return _LEVELS[min(len(_LEVELS) - 1, _LEVELS.index(level) + steps)]


def _aggregate(
    object_type: str,
    channel: int,
    results: Sequence[_PlaneResult],
    n_fields_requested: int,
    n_fields_available: int,
    fused_fraction: float,
    location: str,
) -> DiameterEstimate:
    """Pool per-field measurements into one proposal with an honest confidence."""
    usable = [r for r in results if r.ok]
    n_used = len(usable)
    where = f"channel {channel} of {location}"

    if not usable:
        why = results[0].reason if results else "no field could be read"
        return _no_estimate(
            object_type,
            f"no usable signal in {where}: {why}. Check that this really is the "
            f"{object_type} stain and that the channel index is not off by one.",
        )

    thresh = np.concatenate([r.thresh_diams for r in usable]) if usable else np.empty(0)
    split = np.concatenate([r.split_diams for r in usable]) if usable else np.empty(0)
    fg = float(np.median([r.fg_fraction for r in usable]))

    d_thresh = float(np.median(thresh)) if thresh.size else float("nan")
    d_split = float(np.median(split)) if split.size else float("nan")

    # Fusion detection, i.e. when to stop believing the plain threshold.
    #
    # A confluent monolayer merges into one component that touches the border,
    # gets dropped as truncated, and leaves only debris behind -- so `thresh`
    # collapses to nothing, or to a handful of specks that would be reported as
    # a tiny diameter. Both halves of that signature are required here:
    #
    #   * the threshold path kept nothing, or kept far fewer objects than the
    #     distance transform resolves, AND
    #   * the field is dense enough for fusion to be the explanation.
    #
    # Requiring both matters. The count disagreement ALONE is not evidence of
    # fusion: a hollow, membrane-only object is one correct component by area
    # but shatters into dozens of arc-shaped basins under the distance
    # transform, so a ratio test on its own would throw away the right answer
    # (60 px) in favour of the wall thickness (5 px) -- the same silent
    # collapse this code exists to prevent, entered from the other side. A
    # high foreground fraction alone is not evidence either: a dense but
    # well-separated field can reach 30% foreground and still be measured
    # correctly by thresholding. When only one signal fires, the threshold
    # estimate is kept and the confidence is downgraded instead.
    confluent = fg >= fused_fraction
    collapsed = thresh.size == 0
    outnumbered = confluent and split.size >= 5 * max(thresh.size, 1)
    fused = split.size > 0 and (collapsed or outnumbered)

    fused_reasons: List[str] = []
    if collapsed:
        fused_reasons.append("plain thresholding kept no whole object at all")
    if outnumbered and not collapsed:
        fused_reasons.append(
            f"the distance transform resolves {split.size} objects where plain "
            f"thresholding kept {thresh.size}"
        )
    if confluent:
        fused_reasons.append(f"foreground covers {fg * 100:.0f}% of the field")

    if fused:
        chosen, diameter, method = split, d_split, "watershed_edt"
    elif thresh.size:
        chosen, diameter, method = thresh, d_thresh, "threshold_otsu"
    else:
        return _no_estimate(
            object_type,
            f"{where}: thresholding found no object that survived the border and "
            f"size filters. Either the objects all touch the field edge or they are "
            f"smaller than the minimum diameter filter.",
            n_fields=n_used,
        )

    low = float(np.percentile(chosen, 10))
    high = float(np.percentile(chosen, 90))
    n_objects = int(chosen.size)

    # ---- confidence, and the reasons for every downgrade -------------------
    level = _HIGH
    notes: List[str] = []

    if n_objects < 10:
        level = _demote(level, 2)
        notes.append(f"only {n_objects} objects measured")
    elif n_objects < 30:
        level = _demote(level)
        notes.append(f"only {n_objects} objects measured")

    # Spread is measured 10th-to-90th rather than by the IQR: a field holding
    # two populations (debris plus cells, say) can have a razor-thin IQR around
    # whichever one is more numerous while the reported range spans five-fold.
    # The IQR version scored exactly that case 'high'.
    spread = float((high - low) / diameter) if diameter > 0 else float("inf")
    if spread > 2.0:
        level = _demote(level, 2)
        notes.append(
            f"very wide size spread (the 10th-90th percentile range is "
            f"{spread * 100:.0f}% of the median, so this may be two populations)"
        )
    elif spread > 1.0:
        level = _demote(level)
        notes.append(
            f"wide size spread (the 10th-90th percentile range is "
            f"{spread * 100:.0f}% of the median)"
        )

    if fused:
        level = _demote(level)
        notes.append(
            "objects look confluent or touching ("
            + "; ".join(fused_reasons)
            + "), so the value comes from a distance transform watershed split "
            "rather than from plain thresholding"
        )
    elif confluent:
        level = _demote(level)
        notes.append(
            f"foreground covers {fg * 100:.0f}% of the field, which is dense enough "
            f"that objects may be merging"
        )

    if not fused and np.isfinite(d_thresh) and np.isfinite(d_split) and min(d_thresh, d_split) > 0:
        ratio = max(d_thresh, d_split) / min(d_thresh, d_split)
        if ratio > 1.5:
            level = _demote(level)
            notes.append(
                f"the two measurements disagree ({d_thresh:.1f} px by thresholding vs "
                f"{d_split:.1f} px by distance transform)"
            )

    if n_used < 2:
        level = _demote(level)
        notes.append("only one field contributed")
    if n_used < n_fields_requested:
        notes.append(
            f"{n_used} of the {n_fields_requested} requested fields were usable "
            f"({n_fields_available} available)"
        )

    head = (
        f"{n_objects} objects across {n_used} field(s) from {where}; "
        f"foreground {fg * 100:.1f}%"
    )
    tail = (
        f"If this looks wrong, confirm channel {channel} is the {object_type} stain, "
        f"check the images are not saturated, and measure two or three objects by hand "
        f"against the {low:.0f}-{high:.0f} px range."
    )
    note = head + (". " + "; ".join(notes) if notes else "") + ". " + tail

    return DiameterEstimate(
        object_type=object_type,
        diameter=float(diameter),
        low=low,
        high=high,
        n_objects=n_objects,
        n_fields=n_used,
        method=method,
        confidence=level,
        note=note,
    )


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def estimate_diameters(
    src: Any,
    channels: Dict[str, Any],
    n_fields: int = 5,
    *,
    metadata_type: Any = "cellvoyager",
    custom_regex: Any = None,
    random_state: Optional[int] = None,
    max_pixels: int = 2_250_000,
    min_snr: float = 4.0,
    fused_fraction: float = 0.35,
    min_object_diameter: float = 4.0,
    max_object_fraction: float = 0.25,
    verbose: bool = False,
) -> Dict[str, DiameterEstimate]:
    """Propose a Cellpose ``diameter`` per object type from blob statistics.

    Samples ``n_fields`` fields spread across ``src`` (never the first N — see
    :func:`_sample_indices`), measures characteristic object size in each
    requested channel by thresholding and by a distance-transform watershed,
    and pools the two into one proposal per object type. Cellpose is never
    loaded; neither is torch.

    :param src: a source folder, or a list of them. Merged ``stack/`` /
        ``merged/`` ``.npy`` arrays are used when present, otherwise raw
        acquisition files are grouped by the metadata regex.
    :param channels: ``{object_type: channel_index}``, e.g.
        ``{'cell': 2, 'nucleus': 0}``. Values that are not integers are
        skipped. :func:`channels_from_settings` builds this from a settings
        dict.
    :param n_fields: how many fields to sample. More is slower and steadier.
    :param metadata_type: ``'cellvoyager'`` / ``'cq1'`` / ``'auto'`` /
        ``'custom'``; only consulted when raw files must be parsed.
    :param custom_regex: named-group regex used when ``metadata_type`` is
        ``'custom'``.
    :param random_state: ``None`` (default) samples on an even stride, which
        is deterministic; an int draws a reproducible random sample instead.
    :param max_pixels: fields larger than this are centre-cropped, which loses
        peripheral objects but does not bias their measured size.
    :param min_snr: minimum ratio of structural amplitude to pixel noise for a
        field to be measured at all. Below it the field is discarded rather
        than thresholded, so a background-only channel yields no number.
    :param fused_fraction: foreground fraction at or above which the field is
        called confluent, which downgrades the confidence and is recorded in
        the note. It does not by itself switch the measurement method; the
        distance-transform estimate takes over only when it resolves
        substantially more objects than plain labelling did.
    :param min_object_diameter: components thinner than this many pixels are
        discarded as debris.
    :param max_object_fraction: components covering more than this fraction of
        a field are discarded as fused blobs rather than measured.
    :param verbose: print each estimate as it is produced.
    :returns: ``{object_type: DiameterEstimate}`` for every object type given
        an integer channel. Entries whose :attr:`DiameterEstimate.usable` is
        False carry NaN — check it before writing the value into settings.
    """
    requested: Dict[str, int] = {}
    for obj, value in (channels or {}).items():
        idx = _as_channel_index(value)
        if idx is not None:
            requested[str(obj)] = idx
    if not requested:
        return {}

    order = [o for o in OBJECT_TYPES if o in requested]
    order += [o for o in requested if o not in order]

    source = _discover(src, metadata_type, custom_regex)
    if not source.fields:
        note = source.problem or "no fields found"
        return {
            obj: _no_estimate(
                obj,
                f"nothing to sample under {src!r}: {note}. Point src at the plate "
                f"folder that holds stack/ or the raw acquisition images.",
            )
            for obj in order
        }

    n_available = len(source.fields)
    picks = _sample_indices(n_available, n_fields, random_state)

    # Channel-range check up front, so an out-of-range index is reported as a
    # problem rather than raised as an IndexError halfway through a sample.
    per_object: Dict[str, List[_PlaneResult]] = {}
    out_of_range: Dict[str, str] = {}
    for obj in order:
        channel = requested[obj]
        if channel < 0:
            out_of_range[obj] = f"channel {channel} is negative"
        elif source.n_channels is not None and channel >= source.n_channels:
            out_of_range[obj] = (
                f"channel {channel} is out of range: {source.where} has "
                f"{source.n_channels} channel(s), so valid indices are "
                f"0-{source.n_channels - 1}"
            )
        else:
            per_object[obj] = []

    for pick in picks:
        key, payload = source.fields[pick]
        for obj in list(per_object):
            channel = requested[obj]
            try:
                if source.kind == "array":
                    plane = _load_array_plane(payload, channel)
                else:
                    chan_id = source.channel_ids[channel]
                    paths = payload.get(chan_id)
                    if not paths:
                        per_object[obj].append(
                            _PlaneResult(reason=f"field {key[-3:]} has no channel {chan_id} file")
                        )
                        continue
                    plane = _load_raw_plane(paths)
                plane = _crop_to(_to_2d(plane), max_pixels)
            except IndexError as exc:
                out_of_range.setdefault(obj, str(exc))
                per_object.pop(obj, None)
                continue
            except Exception as exc:                      # unreadable file, odd dtype
                per_object[obj].append(_PlaneResult(reason=f"could not read field: {exc}"))
                continue
            per_object[obj].append(
                _measure(plane, min_snr, min_object_diameter, max_object_fraction)
            )

    estimates: Dict[str, DiameterEstimate] = {}
    for obj in order:
        if obj in out_of_range:
            estimates[obj] = _no_estimate(
                obj,
                out_of_range[obj]
                + f". Set {SETTING_KEYS.get(obj, obj + '_diameter')}'s channel to a valid index.",
            )
            continue
        estimates[obj] = _aggregate(
            obj,
            requested[obj],
            per_object.get(obj, []),
            n_fields_requested=max(1, int(n_fields)),
            n_fields_available=n_available,
            fused_fraction=fused_fraction,
            location=source.where or str(src),
        )
        if verbose:
            print(estimates[obj])

    return estimates


def format_estimates(estimates: Dict[str, DiameterEstimate]) -> str:
    """Render :func:`estimate_diameters` output as a printable block.

    :param estimates: the mapping returned by :func:`estimate_diameters`.
    :returns: a multi-line string: one table row per object type carrying the
        proposed diameter, its plausible range, object and field counts,
        confidence and method, followed by the note behind each row.
    """
    if not estimates:
        return "Diameter estimates: nothing requested (no object channel was given)."

    rows: List[Tuple[str, ...]] = []
    for obj, est in estimates.items():
        if est.usable:
            rows.append((
                obj,
                f"{est.diameter:.1f}",
                f"{est.low:.1f}-{est.high:.1f}",
                str(est.n_objects),
                str(est.n_fields),
                est.confidence,
                est.method,
            ))
        else:
            rows.append((obj, "-", "-", "0", str(est.n_fields), est.confidence, est.method))

    header = ("object", "diameter", "range (px)", "objects", "fields", "confidence", "method")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def _line(cells: Sequence[str]) -> str:
        return "  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    lines = ["Proposed Cellpose diameters (pixels)", _line(header), _line(["-" * w for w in widths])]
    lines.extend(_line(r) for r in rows)
    lines.append("")
    lines.append("Notes")
    for obj, est in estimates.items():
        key = SETTING_KEYS.get(obj, f"{obj}_diameter")
        if est.usable:
            lines.append(f"  {key} = {est.diameter:.1f}  [{est.confidence}] {est.note}")
        else:
            lines.append(f"  {key}: no estimate  [{est.confidence}] {est.note}")
    return "\n".join(lines)
