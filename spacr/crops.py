"""On-demand single-object crops cut straight out of ``merged/*.npy``.

Background
----------
:func:`spacr.measure.measure_crop` writes a per-object PNG for every object it
measures (``<root>/data/.../{cell,nucleus,pathogen,cytoplasm}_png/``) and records
the paths in the ``png_list`` table of ``measurements/measurements.db``. Every
downstream consumer -- the annotation GUIs, the classification datasets, the
image UMAP -- reads those PNGs. That costs disk, has to be regenerated whenever
a crop setting changes, and goes stale silently.

The ``merged/`` array already contains everything needed to cut the same crop on
demand: the intensity planes *and* the integer label-mask planes. This module is
that alternative source. It is deliberately **additive** -- :class:`PngCropSource`
wraps the existing behaviour unchanged, :class:`MergedCropSource` is the new one,
and :func:`resolve_crop_source` picks between them and says which it picked.

Merged array layout
-------------------
``spacr.io._load_and_concatenate_arrays`` builds each ``merged/<fov>.npy`` as::

    (H, W, n_intensity_channels + n_mask_planes)

The intensity channels come first (the subset selected by ``settings['channels']``
at preprocessing time), then one ``uint16`` label-mask plane per segmented object
class, always in the order **cell, nucleus, pathogen, organelle** -- each present
only if that class was segmented. ``settings['cell_mask_dim']`` /
``nucleus_mask_dim`` / ``pathogen_mask_dim`` / ``organelle_mask_dim`` record the
resulting plane indices; spaCR's default four-channel layout is
``{cell: 4, nucleus: 5, pathogen: 6, organelle: 7}`` (:data:`DEFAULT_MASK_DIMS`).

There is **no cytoplasm plane on disk**: ``measure_crop`` derives cytoplasm as
"cell minus nucleus/pathogen/organelle" in memory and never saves it back. This
module derives it the same way (see :func:`MergedField.mask_plane`).

Fidelity
--------
:func:`extract_crop` reproduces the PNG path in ``spacr.measure._measure_crop_core``
step for step: same channel selection (``png_dims``), same region definition
(object mask, optionally replaced by its padded bounding box, optionally
dilated), same ``_crop_center`` centering/padding, same ``normalize_to_dtype``
percentile normalisation, same dtype. See :func:`png_view` for the extra twist
that the crop is written with ``cv2.imwrite`` (BGR) and read back with PIL.

Dependencies
------------
numpy, and the standard library (plus PIL only inside :class:`PngCropSource`,
which has to decode a PNG). No torch, no cellpose, no scipy, no skimage --
importing this module must stay cheap enough for a GUI thumbnail path.
"""

from __future__ import annotations

import ast
import os
import sqlite3
from dataclasses import dataclass, field as _dc_field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "MASK_PLANE_ORDER",
    "DEFAULT_MASK_DIMS",
    "CropError",
    "MergedFileMissing",
    "CorruptMergedFile",
    "MaskPlaneMissing",
    "LabelMissing",
    "CropSpec",
    "MergedField",
    "open_merged_field",
    "clear_field_cache",
    "extract_crop",
    "extract_crops",
    "png_view",
    "mask_dims_from_settings",
    "crop_settings_from_db",
    "crop_spec_from_settings",
    "CropSource",
    "PngCropSource",
    "MergedCropSource",
    "resolve_crop_source",
]


#: Order in which mask planes are appended to a merged array by
#: :func:`spacr.io._load_and_concatenate_arrays`.
MASK_PLANE_ORDER: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "organelle")

#: spaCR's default plane indices for a four-intensity-channel merged array.
DEFAULT_MASK_DIMS: Dict[str, int] = {
    "cell": 4, "nucleus": 5, "pathogen": 6, "organelle": 7,
}

#: Object types that have their own plane on disk, plus the derived one.
OBJECT_TYPES: Tuple[str, ...] = MASK_PLANE_ORDER + ("cytoplasm",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CropError(RuntimeError):
    """Base class for every failure raised while cutting an on-demand crop."""


class MergedFileMissing(CropError, FileNotFoundError):
    """The requested ``merged/*.npy`` does not exist."""


class CorruptMergedFile(CropError):
    """The ``.npy`` exists but cannot be read as an ``(H, W, C)`` array."""


class MaskPlaneMissing(CropError):
    """The array has no plane for the requested object type."""


class LabelMissing(CropError):
    """The requested object label is not present in the mask plane."""


# ---------------------------------------------------------------------------
# Numpy clones of the PNG path's normalisation helpers
#
# These are byte-for-byte reimplementations of
# ``skimage.exposure.rescale_intensity`` (2-tuple ranges only),
# ``spacr.utils.normalize_to_dtype`` and ``spacr.utils._get_percentiles``.
# They exist so this module never has to import skimage or spacr.utils --
# spacr.utils pulls in torch, which must stay off the crop path.
# ``tests/test_crops.py`` asserts they agree with the originals.
# ---------------------------------------------------------------------------

def _rescale_intensity(image, in_range, out_range):
    """``skimage.exposure.rescale_intensity`` for explicit 2-tuple ranges.

    Returns float64 (skimage returns float when ``out_range`` is a pair of
    values), so the caller is responsible for the cast back to the image dtype
    -- exactly as ``normalize_to_dtype`` does via slice assignment.
    """
    imin, imax = float(in_range[0]), float(in_range[1])
    omin, omax = float(out_range[0]), float(out_range[1])
    image = np.clip(image, imin, imax)
    if imin != imax:
        image = (image - imin) / (imax - imin)
        return image * (omax - omin) + omin
    return np.clip(image, omin, omax)


def _normalize_to_dtype(array, p1=2, p2=98, percentile_list=None):
    """Clone of :func:`spacr.utils.normalize_to_dtype` with ``new_dtype=None``.

    The PNG path only ever calls it that way, so the output range is always
    ``(0, iinfo(array.dtype).max)`` and the result keeps the input dtype.
    """
    out_range = (0, np.iinfo(array.dtype).max)
    nimg = array.shape[2]
    new_stack = np.empty_like(array, dtype=array.dtype)
    for i in range(nimg):
        img = array[:, :, i]
        non_zero_img = img[img > 0]
        if percentile_list is None:
            if non_zero_img.size > 0:
                img_min = np.percentile(non_zero_img, p1)
                img_max = np.percentile(non_zero_img, p2)
            else:
                img_min = np.percentile(img, p1)
                img_max = np.percentile(img, p2)
        else:
            img_min, img_max = percentile_list[i][0], percentile_list[i][1]
        new_stack[:, :, i] = _rescale_intensity(img, (img_min, img_max), out_range)
    return new_stack


def _get_percentiles(array, p1=2, p2=98):
    """Clone of :func:`spacr.utils._get_percentiles` (per-channel, nonzero pixels)."""
    percentiles = []
    for v in range(array.shape[2]):
        img = np.squeeze(array[:, :, v])
        non_zero_img = img[img > 0]
        if non_zero_img.size > 0:
            percentiles.append([np.percentile(non_zero_img, p1),
                                np.percentile(non_zero_img, p2)])
        else:
            percentiles.append([np.percentile(img, p1),
                                np.percentile(img, p2)])
    return percentiles


def _binary_dilate(region, iterations):
    """``scipy.ndimage.binary_dilation`` with a full 3x3 structuring element.

    Equivalent to ``binary_dilation(region, structure=generate_binary_structure(2, 2),
    iterations=iterations)`` for ``iterations >= 1``: each pass ORs the 8-neighbourhood,
    with the outside of the array treated as background.

    ``iterations <= 0`` is *not* handled here -- scipy interprets that as "repeat
    until nothing changes", which fills the whole array. The caller special-cases
    it so the quirk is visible rather than buried (see :func:`_region_for`).
    """
    out = np.asarray(region, dtype=bool)
    for _ in range(int(iterations)):
        acc = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros_like(out)
                ys_dst = slice(max(0, dy), out.shape[0] + min(0, dy))
                ys_src = slice(max(0, -dy), out.shape[0] + min(0, -dy))
                xs_dst = slice(max(0, dx), out.shape[1] + min(0, dx))
                xs_src = slice(max(0, -dx), out.shape[1] + min(0, -dx))
                shifted[ys_dst, xs_dst] = out[ys_src, xs_src]
                acc |= shifted
        out = acc
    return out


# ---------------------------------------------------------------------------
# Crop specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropSpec:
    """Everything needed to reproduce one crop, byte for byte.

    The field names mirror the ``measure_crop`` settings that drive the PNG
    path, so a spec can be built straight from a saved settings snapshot
    (:func:`crop_spec_from_settings`).

    :param merged_path: path to the ``merged/<fov>.npy`` the object lives in.
    :param object_type: ``'cell'`` | ``'nucleus'`` | ``'pathogen'`` |
        ``'organelle'`` | ``'cytoplasm'`` -- selects the mask plane
        (``measure_crop``'s ``crop_mode``).
    :param label: the object's integer label in that mask plane
        (``object_label`` in ``measurements.db``). ``0`` is background and is
        always an error.
    :param channels: intensity plane indices, in output order
        (``measure_crop``'s ``png_dims``).
    :param size: ``(width, height)`` of the crop (``measure_crop``'s ``png_size``).
        Note the width-first order -- that is the PNG path's convention.
    :param mask_dims: object type -> plane index. ``None`` uses
        :data:`DEFAULT_MASK_DIMS`.
    :param use_bounding_box: crop the object's padded bounding box instead of
        its exact outline (``measure_crop``'s ``use_bounding_box``). The pad is
        hard-coded to 10 px in the PNG path; :attr:`bbox_buffer` mirrors it.
    :param bbox_buffer: pad added around the bounding box, in pixels.
    :param bbox: optional pre-computed ``(y0, y1, x0, x1)`` half-open bounding
        box (skimage ``regionprops`` convention) for this label, e.g. read from
        a database column. When given, the mask plane is never scanned to find
        the object, only the window is read.
    :param dilate: dilate the region before cropping (``dialate_pngs``).
    :param dilate_ratio: dilation radius as a fraction of ``sqrt(area)``
        (``dialate_png_ratios``).
    :param normalize: ``False`` (the shipped default) reproduces the PNG path's
        fallback of a full 0-100 percentile stretch; a ``(p1, p2)`` pair
        reproduces the configured stretch.
    :param normalize_by: ``'png'`` (percentiles from the crop) or ``'fov'``
        (percentiles from the whole field before cropping).
    """

    merged_path: str
    object_type: str = "cell"
    label: int = 0
    channels: Tuple[int, ...] = (0, 1, 2)
    size: Tuple[int, int] = (224, 224)
    mask_dims: Optional[Mapping[str, int]] = None
    use_bounding_box: bool = False
    bbox_buffer: int = 10
    bbox: Optional[Tuple[int, int, int, int]] = None
    dilate: bool = False
    dilate_ratio: float = 0.2
    normalize: Union[bool, Sequence[float], None] = False
    normalize_by: str = "png"

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(int(c) for c in self.channels))
        w, h = self.size
        object.__setattr__(self, "size", (int(w), int(h)))
        object.__setattr__(self, "label", int(self.label))
        if self.bbox is not None:
            object.__setattr__(self, "bbox", tuple(int(v) for v in self.bbox))
        if self.object_type not in OBJECT_TYPES:
            raise CropError(
                f"unknown object_type {self.object_type!r}; expected one of "
                f"{', '.join(OBJECT_TYPES)}")
        if self.normalize_by not in ("png", "fov"):
            raise CropError(
                f"normalize_by must be 'png' or 'fov', got {self.normalize_by!r}")

    def with_(self, **kwargs) -> "CropSpec":
        """Return a copy of this spec with ``kwargs`` replaced."""
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Per-file label index
# ---------------------------------------------------------------------------

class _LabelIndex:
    """Per-label bounding box, pixel count and centroid for one mask plane.

    Built with a single vectorised pass over the plane, then cached on the
    :class:`MergedField`. A grid drawing 100 objects out of one field therefore
    scans the plane once, not 100 times.
    """

    __slots__ = ("labels", "_pos", "ymin", "ymax", "xmin", "xmax",
                 "count", "_ysum", "_xsum", "shape")

    def __init__(self, mask: np.ndarray):
        self.shape = (int(mask.shape[0]), int(mask.shape[1]))
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            self.labels = np.zeros(0, dtype=np.int64)
            self._pos = {}
            empty_i = np.zeros(0, dtype=np.int64)
            empty_f = np.zeros(0, dtype=np.float64)
            self.ymin = self.ymax = self.xmin = self.xmax = self.count = empty_i
            self._ysum = self._xsum = empty_f
            return
        vals = np.asarray(mask)[ys, xs]
        order = np.argsort(vals, kind="stable")
        v = vals[order]
        y = ys[order].astype(np.int64, copy=False)
        x = xs[order].astype(np.int64, copy=False)
        uniq, starts, counts = np.unique(v, return_index=True, return_counts=True)
        self.labels = uniq.astype(np.int64, copy=False)
        self.ymin = np.minimum.reduceat(y, starts)
        self.ymax = np.maximum.reduceat(y, starts)
        self.xmin = np.minimum.reduceat(x, starts)
        self.xmax = np.maximum.reduceat(x, starts)
        # float64 accumulation of integer coordinates is exact well past any
        # realistic field size, so this matches ``ys.mean()`` bit for bit --
        # which is what ``scipy.ndimage.center_of_mass`` computes.
        self._ysum = np.add.reduceat(y.astype(np.float64), starts)
        self._xsum = np.add.reduceat(x.astype(np.float64), starts)
        self.count = counts.astype(np.int64, copy=False)
        self._pos = {int(lbl): i for i, lbl in enumerate(self.labels)}

    def __contains__(self, label: int) -> bool:
        return int(label) in self._pos

    def _index(self, label: int) -> int:
        try:
            return self._pos[int(label)]
        except KeyError:
            raise LabelMissing(
                f"label {label} is not present in this mask plane "
                f"({len(self._pos)} labels available)") from None

    def bbox(self, label: int) -> Tuple[int, int, int, int]:
        """Return the half-open ``(y0, y1, x0, x1)`` bounding box of ``label``.

        Half-open (``y1``/``x1`` exclusive) to match skimage ``regionprops.bbox``,
        which is what a database column would hold.
        """
        i = self._index(label)
        return (int(self.ymin[i]), int(self.ymax[i]) + 1,
                int(self.xmin[i]), int(self.xmax[i]) + 1)

    def area(self, label: int) -> int:
        """Return the pixel count of ``label``."""
        return int(self.count[self._index(label)])

    def centroid(self, label: int) -> Tuple[float, float]:
        """Return the ``(row, col)`` centroid of ``label``."""
        i = self._index(label)
        n = float(self.count[i])
        return (float(self._ysum[i]) / n, float(self._xsum[i]) / n)


# ---------------------------------------------------------------------------
# Merged field (one .npy, memory-mapped)
# ---------------------------------------------------------------------------

class MergedField:
    """A memory-mapped ``merged/<fov>.npy`` plus its per-plane label indices.

    The array is opened with ``mmap_mode='r'``: cutting one object reads the
    object's mask plane once (cached) and then only the crop window, never the
    whole field. A 2048x2048x5 uint16 field is 40 MB; materialising it per
    object would make an on-demand grid slower than the PNG folder it replaces.

    Only ``shape``, ``dtype``, ``ndim`` and ``__getitem__`` are ever used on the
    underlying array, so tests can substitute a recording proxy to assert on the
    access pattern.
    """

    def __init__(self, path: str, array=None, mask_dims: Optional[Mapping[str, int]] = None):
        self.path = os.fspath(path)
        self.mask_dims = dict(mask_dims) if mask_dims else dict(DEFAULT_MASK_DIMS)
        if array is None:
            array = _load_mmap(self.path)
        if getattr(array, "ndim", None) != 3:
            raise CorruptMergedFile(
                f"{self.path}: expected an (H, W, C) array, got shape "
                f"{getattr(array, 'shape', None)!r}")
        self.array = array
        self._indices: Dict[int, _LabelIndex] = {}
        self._derived: Dict[str, np.ndarray] = {}

    # -- geometry ----------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the ``(H, W, C)`` shape of the merged array."""
        return tuple(int(v) for v in self.array.shape)

    @property
    def dtype(self):
        """Return the on-disk dtype of the merged array."""
        return self.array.dtype

    @property
    def crop_dtype(self):
        """Return the dtype crops are cut in.

        ``_measure_crop_core`` promotes anything that is not ``uint8``/``uint16``
        to ``uint16`` before cropping; this mirrors that.
        """
        dt = self.array.dtype
        return dt if dt in (np.dtype(np.uint8), np.dtype(np.uint16)) else np.dtype(np.uint16)

    # -- planes ------------------------------------------------------------
    def mask_dim(self, object_type: str) -> int:
        """Return the plane index holding ``object_type``'s labels.

        :raises MaskPlaneMissing: if the type has no plane, or the recorded
            plane index is out of range for this array.
        """
        if object_type == "cytoplasm":
            raise MaskPlaneMissing(
                "cytoplasm has no plane on disk; it is derived from cell minus "
                "nucleus/pathogen/organelle -- use mask_plane('cytoplasm')")
        dim = self.mask_dims.get(object_type)
        if dim is None:
            raise MaskPlaneMissing(
                f"{self.path}: no mask plane recorded for object_type "
                f"{object_type!r} (known: {sorted(k for k, v in self.mask_dims.items() if v is not None)})")
        dim = int(dim)
        if not 0 <= dim < self.shape[2]:
            raise MaskPlaneMissing(
                f"{self.path}: mask plane {dim} for {object_type!r} is out of "
                f"range for an array with {self.shape[2]} planes")
        return dim

    def mask_plane(self, object_type: str) -> np.ndarray:
        """Return the 2-D label plane for ``object_type`` as a real array.

        ``'cytoplasm'`` is derived on the fly -- ``measure_crop`` computes it as
        the cell mask with every nucleus / pathogen / organelle pixel zeroed and
        never writes it back to the merged file.
        """
        if object_type != "cytoplasm":
            return np.asarray(self.array[:, :, self.mask_dim(object_type)])
        if "cytoplasm" in self._derived:
            return self._derived["cytoplasm"]
        cell = np.asarray(self.array[:, :, self.mask_dim("cell")])
        interior = np.zeros(cell.shape, dtype=bool)
        for other in ("nucleus", "pathogen", "organelle"):
            if self.mask_dims.get(other) is None:
                continue
            try:
                dim = self.mask_dim(other)
            except MaskPlaneMissing:
                continue
            interior |= np.asarray(self.array[:, :, dim]) != 0
        cyto = np.where(interior, 0, cell)
        self._derived["cytoplasm"] = cyto
        return cyto

    def label_index(self, object_type: str) -> _LabelIndex:
        """Return the cached :class:`_LabelIndex` for ``object_type``'s plane."""
        key = -1 if object_type == "cytoplasm" else self.mask_dim(object_type)
        idx = self._indices.get(key)
        if idx is None:
            idx = _LabelIndex(self.mask_plane(object_type))
            self._indices[key] = idx
        return idx

    def labels(self, object_type: str = "cell") -> List[int]:
        """Return every non-zero label present in ``object_type``'s plane."""
        return [int(v) for v in self.label_index(object_type).labels]

    # -- windows -----------------------------------------------------------
    def read_window(self, y0: int, y1: int, x0: int, x1: int,
                    channels: Sequence[int], dtype=None) -> np.ndarray:
        """Read ``channels`` over ``[y0:y1, x0:x1]``, zero-padding outside the array.

        The window may run off any edge; the out-of-array part comes back as
        zeros, which is what the PNG path's ``np.pad`` produces.
        """
        dtype = self.crop_dtype if dtype is None else np.dtype(dtype)
        H, W, C = self.shape
        for c in channels:
            if not 0 <= int(c) < C:
                raise CropError(
                    f"{self.path}: channel {c} out of range for an array with "
                    f"{C} planes")
        out = np.zeros((int(y1 - y0), int(x1 - x0), len(channels)), dtype=dtype)
        sy0, sy1 = max(0, int(y0)), min(H, int(y1))
        sx0, sx1 = max(0, int(x0)), min(W, int(x1))
        if sy1 > sy0 and sx1 > sx0:
            dy, dx = sy0 - int(y0), sx0 - int(x0)
            for k, c in enumerate(channels):
                # One plane at a time so the mmap only faults in the window.
                sub = np.asarray(self.array[sy0:sy1, sx0:sx1, int(c)])
                out[dy:dy + (sy1 - sy0), dx:dx + (sx1 - sx0), k] = sub.astype(dtype, copy=False)
        return out

    def read_mask_window(self, object_type: str, y0: int, y1: int,
                         x0: int, x1: int) -> np.ndarray:
        """Read ``object_type``'s label plane over ``[y0:y1, x0:x1]``, zero-padded."""
        H, W, _ = self.shape
        if object_type == "cytoplasm":
            plane = self.mask_plane("cytoplasm")
            out = np.zeros((int(y1 - y0), int(x1 - x0)), dtype=plane.dtype)
            sy0, sy1 = max(0, int(y0)), min(H, int(y1))
            sx0, sx1 = max(0, int(x0)), min(W, int(x1))
            if sy1 > sy0 and sx1 > sx0:
                out[sy0 - int(y0):sy1 - int(y0), sx0 - int(x0):sx1 - int(x0)] = \
                    plane[sy0:sy1, sx0:sx1]
            return out
        dim = self.mask_dim(object_type)
        out = np.zeros((int(y1 - y0), int(x1 - x0)), dtype=self.array.dtype)
        sy0, sy1 = max(0, int(y0)), min(H, int(y1))
        sx0, sx1 = max(0, int(x0)), min(W, int(x1))
        if sy1 > sy0 and sx1 > sx0:
            out[sy0 - int(y0):sy1 - int(y0), sx0 - int(x0):sx1 - int(x0)] = \
                np.asarray(self.array[sy0:sy1, sx0:sx1, dim])
        return out


def _load_mmap(path: str):
    """``np.load(path, mmap_mode='r')`` with spaCR-shaped error messages."""
    if not os.path.isfile(path):
        raise MergedFileMissing(f"merged array not found: {path}")
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise CorruptMergedFile(f"{path}: cannot read as .npy ({exc})") from exc
    if not hasattr(arr, "shape") or getattr(arr, "ndim", 0) != 3:
        raise CorruptMergedFile(
            f"{path}: expected an (H, W, C) array, got shape "
            f"{getattr(arr, 'shape', None)!r}")
    return arr


# A tiny LRU of open fields, so a grid that walks a handful of fields keeps
# their label indices between calls. Keyed on (path, mtime, size) so a
# regenerated merged file is never served from a stale entry.
_FIELD_CACHE: "Dict[Tuple[str, int, int], MergedField]" = {}
_FIELD_CACHE_MAX = 8


def clear_field_cache() -> None:
    """Drop every cached :class:`MergedField` (and its label indices)."""
    _FIELD_CACHE.clear()


def _cache_key(path: str) -> Tuple[str, int, int]:
    try:
        st = os.stat(path)
    except OSError as exc:
        raise MergedFileMissing(f"merged array not found: {path}") from exc
    return (os.path.abspath(path), int(st.st_mtime_ns), int(st.st_size))


def open_merged_field(path: str, mask_dims: Optional[Mapping[str, int]] = None,
                      use_cache: bool = True) -> MergedField:
    """Return a :class:`MergedField` for ``path``, reusing a cached one if possible.

    :param path: the ``merged/<fov>.npy``.
    :param mask_dims: object type -> plane index; ``None`` uses
        :data:`DEFAULT_MASK_DIMS`.
    :param use_cache: set False to force a fresh open (and a fresh label index).
    :raises MergedFileMissing: the file does not exist.
    :raises CorruptMergedFile: the file is not a readable 3-D ``.npy``.
    """
    dims = dict(mask_dims) if mask_dims else dict(DEFAULT_MASK_DIMS)
    if not use_cache:
        return MergedField(path, mask_dims=dims)
    key = _cache_key(path)
    cached = _FIELD_CACHE.get(key)
    if cached is not None and cached.mask_dims == dims:
        return cached
    fld = MergedField(path, mask_dims=dims)
    if len(_FIELD_CACHE) >= _FIELD_CACHE_MAX:
        _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
    _FIELD_CACHE[key] = fld
    return fld


# ---------------------------------------------------------------------------
# The crop itself
# ---------------------------------------------------------------------------

def _region_for(fld: MergedField, spec: CropSpec):
    """Return ``(centroid_yx, region_bounds, region_mask_or_None)``.

    ``region_mask`` is the boolean region restricted to ``region_bounds``; when
    it is ``None`` the region covers the whole field (the ``iterations=0``
    dilation quirk, see below) and no masking is applied.

    Mirrors ``_measure_crop_core`` exactly, including two behaviours that are
    almost certainly bugs but are matched rather than fixed, because an
    annotation made on a PNG has to be comparable with a crop cut here:

    * with ``use_bounding_box`` *and* ``dilate`` both on, the PNG path measures
      the region area with ``np.sum(region)`` on a mask filled with the *label
      value*, not with ``True`` -- so the dilation radius is inflated by
      ``sqrt(label)``;
    * when the dilation radius rounds down to 0, the PNG path calls
      ``scipy.ndimage.binary_dilation(..., iterations=0)``, which means "repeat
      until nothing changes" and fills the entire field -- the crop ends up
      being an unmasked window centred on the middle of the field.
    """
    H, W, _ = fld.shape
    idx = None
    if spec.bbox is not None:
        by0, by1, bx0, bx1 = spec.bbox
        if not (0 <= by0 < by1 <= H and 0 <= bx0 < bx1 <= W):
            raise CropError(
                f"{fld.path}: bbox {spec.bbox} runs outside the "
                f"{H}x{W} field")
    else:
        idx = fld.label_index(spec.object_type)
        by0, by1, bx0, bx1 = idx.bbox(spec.label)

    if spec.use_bounding_box:
        # _find_bounding_box: inclusive rectangle, clamped, hard 10 px buffer.
        ry0 = max(by0 - spec.bbox_buffer, 0)
        ry1 = min(by1 - 1 + spec.bbox_buffer, H - 1) + 1
        rx0 = max(bx0 - spec.bbox_buffer, 0)
        rx1 = min(bx1 - 1 + spec.bbox_buffer, W - 1) + 1
        region = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
        # np.sum over a rectangle filled with the label value, not with 1.
        area_for_dilation = float(region.size) * float(spec.label)
        cy = (ry0 + ry1 - 1) / 2.0
        cx = (rx0 + rx1 - 1) / 2.0
    else:
        ry0, ry1, rx0, rx1 = by0, by1, bx0, bx1
        window = fld.read_mask_window(spec.object_type, ry0, ry1, rx0, rx1)
        region = window == spec.label
        if not region.any():
            raise LabelMissing(
                f"{fld.path}: label {spec.label} is not present in the "
                f"{spec.object_type} mask plane")
        area_for_dilation = float(region.sum())
        if idx is not None:
            cy, cx = idx.centroid(spec.label)
        else:
            ys, xs = np.nonzero(region)
            cy = float(ys.astype(np.int64).sum()) / ys.size + ry0
            cx = float(xs.astype(np.int64).sum()) / xs.size + rx0

    if spec.dilate:
        px = int(np.sqrt(area_for_dilation) * float(spec.dilate_ratio))
        if px <= 0:
            # scipy's "iterations < 1 == dilate to fixpoint": the region becomes
            # the whole field, so nothing is masked and the crop is centred on
            # the middle of the field.
            centroid = np.round(np.array([(H - 1) / 2.0, (W - 1) / 2.0])).astype(int)
            return centroid, (0, H, 0, W), None
        gy0, gy1 = max(0, ry0 - px), min(H, ry1 + px)
        gx0, gx1 = max(0, rx0 - px), min(W, rx1 + px)
        grown = np.zeros((gy1 - gy0, gx1 - gx0), dtype=bool)
        grown[ry0 - gy0:ry1 - gy0, rx0 - gx0:rx1 - gx0] = region
        region = _binary_dilate(grown, px)
        ry0, ry1, rx0, rx1 = gy0, gy1, gx0, gx1
        ys, xs = np.nonzero(region)
        cy = float(ys.astype(np.int64).sum()) / ys.size + ry0
        cx = float(xs.astype(np.int64).sum()) / xs.size + rx0

    centroid = np.round(np.array([cy, cx])).astype(int)
    return centroid, (ry0, ry1, rx0, rx1), region


def _crop_from_field(fld: MergedField, spec: CropSpec) -> np.ndarray:
    """Cut one crop out of an already-open field. See :func:`extract_crop`."""
    if spec.label == 0:
        raise LabelMissing(
            "label 0 is background, not an object; pass the object_label "
            "recorded in measurements.db")
    if spec.label < 0:
        raise LabelMissing(f"label must be a positive integer, got {spec.label}")
    if not spec.channels:
        raise CropError("channels is empty; nothing to crop")

    width, height = spec.size
    if width <= 0 or height <= 0:
        raise CropError(f"size must be positive, got {spec.size}")

    dtype = fld.crop_dtype
    centroid, (ry0, ry1, rx0, rx1), region = _region_for(fld, spec)

    # _crop_center: a fixed (height, width) window centred on the rounded
    # centroid. The PNG path pads by max(width, height) first, which makes the
    # window guaranteed-complete and zero-filled outside the field -- so in
    # unpadded coordinates it is simply this window, zero-padded at the edges.
    wy0 = int(centroid[0]) - height // 2
    wy1 = wy0 + height
    wx0 = int(centroid[1]) - width // 2
    wx1 = wx0 + width

    # FOV-wide percentiles have to be measured before the mask is applied,
    # exactly like the PNG path (_get_percentiles on the whole png_channels).
    percentile_list = None
    if isinstance(spec.normalize, (list, tuple)) and spec.normalize_by == "fov":
        fov = fld.read_window(0, fld.shape[0], 0, fld.shape[1], spec.channels, dtype)
        percentile_list = _get_percentiles(fov, spec.normalize[0], spec.normalize[1])

    crop = fld.read_window(wy0, wy1, wx0, wx1, spec.channels, dtype)

    if region is not None:
        keep = np.zeros((height, width), dtype=bool)
        oy0, oy1 = max(wy0, ry0), min(wy1, ry1)
        ox0, ox1 = max(wx0, rx0), min(wx1, rx1)
        if oy1 > oy0 and ox1 > ox0:
            keep[oy0 - wy0:oy1 - wy0, ox0 - wx0:ox1 - wx0] = \
                region[oy0 - ry0:oy1 - ry0, ox0 - rx0:ox1 - rx0]
        crop = np.where(keep[:, :, None], crop, 0).astype(dtype, copy=False)

    if isinstance(spec.normalize, (list, tuple)):
        crop = _normalize_to_dtype(crop, spec.normalize[0], spec.normalize[1],
                                   percentile_list=percentile_list)
    else:
        crop = _normalize_to_dtype(crop, 0, 100)

    if crop.shape[2] == 2:
        # The PNG path pads a two-channel crop to RGB with a zero third plane
        # before writing, so the file always has three channels.
        crop = np.dstack((crop, np.zeros_like(crop[:, :, 0])))
    return crop


def extract_crop(merged_path: str, object_type: str = "cell", label: int = 0,
                 *, spec: Optional[CropSpec] = None, field: Optional[MergedField] = None,
                 **kwargs) -> np.ndarray:
    """Cut one object out of a merged array, reproducing the PNG path exactly.

    The returned array is the *pre-write* array: exactly what
    ``_measure_crop_core`` hands to ``cv2.imwrite``. Its dtype is the merged
    array's (``uint16`` for a normal spaCR run) and its channel order is
    ``spec.channels``. Use :func:`png_view` to get what a consumer that opens
    the written PNG with PIL would see.

    :param merged_path: the ``merged/<fov>.npy``.
    :param object_type: which mask plane to crop by.
    :param label: the object's ``object_label``.
    :param spec: a ready-made :class:`CropSpec`; ``merged_path`` /
        ``object_type`` / ``label`` and ``kwargs`` override its fields.
    :param field: an already-open :class:`MergedField` to cut from.
    :param kwargs: any other :class:`CropSpec` field.
    :returns: ``(height, width, n_channels)`` array.
    :raises MergedFileMissing: the merged file does not exist.
    :raises CorruptMergedFile: it is not a readable 3-D ``.npy``.
    :raises MaskPlaneMissing: no plane for ``object_type``.
    :raises LabelMissing: ``label`` is 0/negative or absent from the plane.
    :raises CropError: a bad channel index, size, or an out-of-array ``bbox``.
    """
    if spec is None:
        spec = CropSpec(merged_path=merged_path, object_type=object_type,
                        label=label, **kwargs)
    else:
        overrides = dict(kwargs)
        overrides.setdefault("merged_path", merged_path)
        if object_type != "cell" or "object_type" in kwargs:
            overrides.setdefault("object_type", object_type)
        if label or "label" in kwargs:
            overrides.setdefault("label", label)
        spec = replace(spec, **overrides)
    fld = field if field is not None else open_merged_field(spec.merged_path, spec.mask_dims)
    return _crop_from_field(fld, spec)


def extract_crops(merged_path: str, specs: Iterable[CropSpec],
                  *, mask_dims: Optional[Mapping[str, int]] = None,
                  on_error: str = "raise") -> List[Optional[np.ndarray]]:
    """Cut many objects out of one merged array, opening the file once.

    A grid draws from a handful of fields, so batching matters: the ``.npy`` is
    memory-mapped once and every object's mask plane is indexed once, no matter
    how many objects are requested from it.

    :param merged_path: the ``merged/<fov>.npy``; every spec is cut from it,
        whatever ``spec.merged_path`` says.
    :param specs: iterable of :class:`CropSpec`.
    :param mask_dims: object type -> plane index for the whole batch; defaults
        to the first spec's ``mask_dims``.
    :param on_error: ``'raise'`` (default) or ``'none'`` to put ``None`` in the
        result for each failing spec instead of raising.
    :returns: list of crops, one per spec, in order.
    """
    if on_error not in ("raise", "none"):
        raise ValueError("on_error must be 'raise' or 'none'")
    specs = list(specs)
    if not specs:
        return []
    dims = mask_dims if mask_dims is not None else specs[0].mask_dims
    try:
        fld = open_merged_field(merged_path, dims)
    except CropError:
        if on_error == "raise":
            raise
        return [None] * len(specs)
    out: List[Optional[np.ndarray]] = []
    for spec in specs:
        try:
            out.append(_crop_from_field(fld, replace(spec, merged_path=merged_path)))
        except CropError:
            if on_error == "raise":
                raise
            out.append(None)
    return out


def png_view(crop: np.ndarray) -> np.ndarray:
    """Return what a consumer sees after the crop has made the PNG round trip.

    The PNG path writes crops with ``cv2.imwrite``, which interprets a
    three-channel array as **BGR**, and every consumer reads them back with
    ``PIL.Image.open(...).convert('RGB')``. Two consequences that this function
    reproduces, and that callers of :func:`extract_crop` must not forget:

    * the channel order is reversed relative to ``png_dims``;
    * a ``uint16`` crop is written as a 16-bit PNG, and PIL narrows it to 8 bit
      by taking the high byte (``// 256``) for RGB images, or by *clipping* to
      255 for single-channel ones.

    :param crop: the array returned by :func:`extract_crop`.
    :returns: ``(H, W, 3)`` uint8 RGB array.
    """
    arr = np.asarray(crop)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    n = arr.shape[2]
    if arr.dtype == np.dtype(np.uint16):
        if n == 1:
            eight = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            eight = (arr // 256).astype(np.uint8)
    elif arr.dtype == np.dtype(np.uint8):
        eight = arr
    else:
        eight = np.clip(arr, 0, 255).astype(np.uint8)
    if n == 1:
        return np.repeat(eight, 3, axis=2)
    if n == 2:
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        rgb[:, :, :2] = eight
        return rgb[:, :, ::-1].copy()
    return eight[:, :, :3][:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Settings plumbing
# ---------------------------------------------------------------------------

def mask_dims_from_settings(settings: Mapping[str, Any]) -> Dict[str, int]:
    """Return ``{object_type: plane index}`` from a ``measure_crop`` settings dict.

    Falls back to :data:`DEFAULT_MASK_DIMS` for anything the dict does not name.
    """
    dims: Dict[str, int] = {}
    for obj in MASK_PLANE_ORDER:
        val = settings.get(f"{obj}_mask_dim")
        if val is None or val == "" or str(val).lower() == "none":
            continue
        try:
            dims[obj] = int(val)
        except (TypeError, ValueError):
            continue
    return dims or dict(DEFAULT_MASK_DIMS)


def _coerce(value: Any) -> Any:
    """Turn a stringified settings value back into a Python object."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def crop_settings_from_db(db_path: str) -> Dict[str, Any]:
    """Read the ``settings`` table ``measure_crop`` writes into ``measurements.db``.

    ``spacr.io._save_settings_to_db`` stores every setting as
    ``(setting_key, setting_value)`` strings; this parses them back so a crop
    cut on demand can use the same ``png_dims`` / ``png_size`` / ``normalize``
    that produced the PNG folder.

    :param db_path: path to ``measurements.db``.
    :returns: the settings dict, or ``{}`` if the table is absent.
    """
    if not os.path.isfile(db_path):
        raise MergedFileMissing(f"measurements database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT setting_key, setting_value FROM settings").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {str(k): _coerce(v) for k, v in rows}


def crop_spec_from_settings(settings: Mapping[str, Any], merged_path: str = "",
                            object_type: Optional[str] = None,
                            label: int = 0) -> CropSpec:
    """Build a :class:`CropSpec` from a ``measure_crop`` settings dict.

    Uses ``png_dims``, ``png_size``, ``normalize``, ``normalize_by``,
    ``use_bounding_box``, ``dialate_pngs``, ``dialate_png_ratios``, ``crop_mode``
    and the ``*_mask_dim`` keys -- i.e. everything that shaped the PNG folder.
    """
    crop_mode = settings.get("crop_mode", ["cell"])
    if isinstance(crop_mode, str):
        crop_mode = [crop_mode]
    obj = object_type or (crop_mode[0] if crop_mode else "cell")

    size = settings.get("png_size", [224, 224])
    if size and isinstance(size[0], (list, tuple)):
        # png_size may be a list-of-lists, one per crop_mode.
        try:
            size = size[list(crop_mode).index(obj)]
        except (ValueError, IndexError):
            size = size[0]
    width, height = int(size[0]), int(size[1])

    dilate = settings.get("dialate_pngs", False)
    if isinstance(dilate, (list, tuple)):
        try:
            dilate = dilate[list(crop_mode).index(obj)]
        except (ValueError, IndexError):
            dilate = bool(dilate[0]) if dilate else False
    ratios = settings.get("dialate_png_ratios", [0.2])
    if isinstance(ratios, (int, float)):
        ratios = [ratios]
    try:
        ratio = float(ratios[list(crop_mode).index(obj)])
    except (ValueError, IndexError, TypeError):
        ratio = float(ratios[0]) if ratios else 0.2

    if obj == "cytoplasm":
        # _measure_crop_core hard-disables dilation for cytoplasm crops.
        dilate = False

    normalize = settings.get("normalize", False)
    if isinstance(normalize, (list, tuple)) and len(normalize) != 2:
        normalize = False

    return CropSpec(
        merged_path=merged_path,
        object_type=obj,
        label=label,
        channels=tuple(int(c) for c in settings.get("png_dims", [0, 1, 2])),
        size=(width, height),
        mask_dims=mask_dims_from_settings(settings),
        use_bounding_box=bool(settings.get("use_bounding_box", False)),
        dilate=bool(dilate),
        dilate_ratio=ratio,
        normalize=normalize,
        normalize_by=str(settings.get("normalize_by", "png")),
    )


# ---------------------------------------------------------------------------
# Crop sources
# ---------------------------------------------------------------------------

def _row_get(row: Any, *names: str, default: Any = None) -> Any:
    """Read the first present key/attribute of ``row`` out of ``names``."""
    for name in names:
        if isinstance(row, Mapping):
            if name in row:
                val = row[name]
                if val is not None:
                    return val
        else:
            try:
                if name in row:                       # pandas Series
                    val = row[name]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        return val
                    continue
            except (TypeError, KeyError, ValueError):
                pass
            val = getattr(row, name, None)
            if val is not None:
                return val
    return default


class CropSource:
    """A source of single-object crops.

    Implementations return a ``(H, W, 3)`` uint8 RGB array from :meth:`get`, so
    a consumer can swap one for the other without changing anything downstream.
    """

    #: ``'png'`` or ``'merged'`` -- which source this is.
    kind: str = "abstract"
    #: Human-readable explanation of why this source was chosen.
    reason: str = ""

    def get(self, row: Any) -> np.ndarray:
        """Return the crop for ``row`` as a ``(H, W, 3)`` uint8 RGB array."""
        raise NotImplementedError

    def get_image(self, row: Any):
        """Return the crop for ``row`` as a PIL ``Image`` in RGB mode."""
        from PIL import Image
        return Image.fromarray(self.get(row))

    def get_many(self, rows: Iterable[Any]) -> List[Optional[np.ndarray]]:
        """Return crops for many rows. Overridden by sources that can batch."""
        return [self.get(r) for r in rows]

    def describe(self) -> str:
        """Return a one-line description for logs / the GUI status bar."""
        return f"{self.kind} crop source ({self.reason})" if self.reason else f"{self.kind} crop source"


class PngCropSource(CropSource):
    """The existing behaviour: read the pre-generated PNG named by the row.

    :param root: optional experiment root used to re-anchor ``png_path`` values
        recorded on another machine (the same rewrite
        :func:`spacr.utils.correct_paths` performs).
    :param folder: the anchor folder name for that rewrite.
    :param reason: why this source was chosen (for :meth:`describe`).
    """

    kind = "png"

    def __init__(self, root: Optional[str] = None, folder: str = "data",
                 reason: str = ""):
        self.root = root
        self.folder = folder
        self.reason = reason

    def resolve(self, row: Any) -> str:
        """Return the on-disk PNG path for ``row``, re-anchored under ``root``."""
        path = row if isinstance(row, str) else _row_get(row, "png_path", "path")
        if not path:
            raise CropError("row has no 'png_path'")
        path = str(path)
        if self.root and self.root not in path:
            parts = path.split(f"/{self.folder}/")
            if len(parts) > 1:
                path = os.path.join(self.root, self.folder, parts[1])
        return path

    def get(self, row: Any) -> np.ndarray:
        """Return the PNG for ``row`` decoded as a ``(H, W, 3)`` uint8 RGB array."""
        from PIL import Image
        path = self.resolve(row)
        if not os.path.isfile(path):
            raise MergedFileMissing(f"crop PNG not found: {path}")
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))


class MergedCropSource(CropSource):
    """The new one: cut the crop out of ``merged/*.npy`` on demand.

    A row needs the merged array it came from and the object's label. Both are
    already in ``measurements.db``: ``path_name`` (written by
    :func:`spacr.utils._merge_and_save_to_database`) and ``object_label``.
    ``prcfo`` / ``plateID`` / ``rowID`` / ``columnID`` / ``fieldID`` are used
    only as a fallback to rebuild ``<merged_root>/<plate>_<well>_<field>.npy``.

    :param spec: the template :class:`CropSpec`; each row supplies
        ``merged_path`` and ``label``.
    :param merged_root: folder holding the ``.npy`` files, used to re-anchor a
        ``path_name`` recorded on another machine and for the ``prcfo``
        fallback.
    :param object_type: default object type when a row does not carry one.
    :param reason: why this source was chosen (for :meth:`describe`).
    """

    kind = "merged"

    def __init__(self, spec: Optional[CropSpec] = None,
                 merged_root: Optional[str] = None,
                 object_type: Optional[str] = None,
                 reason: str = ""):
        self.spec = spec or CropSpec(merged_path="")
        if object_type:
            self.spec = replace(self.spec, object_type=object_type)
        self.merged_root = merged_root
        self.reason = reason

    # -- row -> spec -------------------------------------------------------
    def resolve_path(self, row: Any) -> str:
        """Return the merged ``.npy`` path for ``row``."""
        path = _row_get(row, "merged_path", "path_name")
        if path:
            path = str(path)
            if self.merged_root and not os.path.isfile(path):
                candidate = os.path.join(self.merged_root, os.path.basename(path))
                if os.path.isfile(candidate):
                    return candidate
            return path
        if not self.merged_root:
            raise CropError(
                "row has no 'path_name' and no merged_root was given, so the "
                "merged array cannot be located")
        stem = _row_get(row, "file_name")
        if stem:
            stem = os.path.splitext(str(stem))[0]
        else:
            plate = _row_get(row, "plateID")
            rowid = _row_get(row, "rowID")
            colid = _row_get(row, "columnID")
            fieldid = _row_get(row, "fieldID")
            if plate is None or rowid is None or colid is None or fieldid is None:
                raise CropError(
                    "row has no 'path_name' and not enough metadata "
                    "(plateID/rowID/columnID/fieldID) to rebuild it")
            well = f"{chr(ord('A') + int(str(rowid).lstrip('r')) - 1)}{int(str(colid).lstrip('c')):02d}"
            stem = f"{plate}_{well}_{int(str(fieldid).lstrip('f'))}"
        return os.path.join(self.merged_root, f"{stem}.npy")

    def spec_for(self, row: Any) -> CropSpec:
        """Return the :class:`CropSpec` describing ``row``'s crop."""
        label = _row_get(row, "object_label", "label", "cell_id", "nucleus_id",
                         "pathogen_id", "cytoplasm_id")
        if label is None:
            raise CropError("row has no 'object_label'")
        obj = _row_get(row, "object_type", default=self.spec.object_type)
        bbox = None
        # skimage regionprops stores bbox as (min_row, min_col, max_row, max_col);
        # CropSpec.bbox is (y0, y1, x0, x1).
        b = [_row_get(row, f"bbox-{i}", f"bbox_{i}") for i in range(4)]
        if all(v is not None for v in b):
            bbox = (int(b[0]), int(b[2]), int(b[1]), int(b[3]))
        return replace(self.spec, merged_path=self.resolve_path(row),
                       object_type=str(obj), label=int(label), bbox=bbox)

    # -- crops -------------------------------------------------------------
    def get_array(self, row: Any) -> np.ndarray:
        """Return the raw crop (native dtype, ``spec.channels`` order)."""
        spec = self.spec_for(row)
        return extract_crop(spec.merged_path, spec=spec)

    def get(self, row: Any) -> np.ndarray:
        """Return the crop as a ``(H, W, 3)`` uint8 RGB array.

        Deliberately routed through :func:`png_view`, so what a consumer gets
        here is identical to what it would get from the PNG folder -- reversed
        channel order and 16-bit narrowing included.
        """
        return png_view(self.get_array(row))

    def get_many(self, rows: Iterable[Any]) -> List[Optional[np.ndarray]]:
        """Return crops for many rows, opening each merged file only once."""
        rows = list(rows)
        specs = [self.spec_for(r) for r in rows]
        out: List[Optional[np.ndarray]] = [None] * len(specs)
        by_path: Dict[str, List[int]] = {}
        for i, spec in enumerate(specs):
            by_path.setdefault(spec.merged_path, []).append(i)
        for path, positions in by_path.items():
            crops = extract_crops(path, [specs[i] for i in positions])
            for i, crop in zip(positions, crops):
                out[i] = png_view(crop) if crop is not None else None
        return out


def _looks_like_experiment_root(src: str) -> str:
    """Return the experiment root for ``src`` (which may be the merged folder)."""
    src = os.path.abspath(os.fspath(src).rstrip(os.sep))
    if os.path.basename(src) == "merged":
        return os.path.dirname(src)
    return src


def _has_png_folder(root: str) -> bool:
    """Return True if ``<root>/data`` holds at least one ``*_png`` crop folder."""
    data = os.path.join(root, "data")
    if not os.path.isdir(data):
        return False
    for dirpath, dirnames, _files in os.walk(data):
        for name in dirnames:
            if name.endswith("_png"):
                return True
        # Crop folders sit at <data>/<well>/<class>_png; three levels is plenty.
        if dirpath.count(os.sep) - data.count(os.sep) >= 3:
            dirnames[:] = []
    return False


def resolve_crop_source(settings_or_src: Union[str, Mapping[str, Any]],
                        *, object_type: Optional[str] = None,
                        prefer: Optional[str] = None) -> CropSource:
    """Pick the crop source for a run, and record which one it picked.

    The returned object's :attr:`CropSource.kind` is ``'png'`` or ``'merged'``
    and :attr:`CropSource.reason` says why, so a caller can print
    ``source.describe()`` instead of guessing.

    Selection order:

    1. an explicit ``prefer`` argument, then ``settings['crop_source']``
       (``'png'`` | ``'merged'`` | ``'auto'``);
    2. otherwise ``'auto'``: the PNG folder if one exists (nothing changes for
       existing datasets), else the merged folder.

    When the merged source is chosen and ``measurements.db`` holds the
    ``measure_crop`` settings, the crop parameters (``png_dims``, ``png_size``,
    ``normalize``, mask plane indices, ...) are read back from it, so the
    on-demand crops match the PNGs that run would have produced.

    :param settings_or_src: a settings dict (with ``src``, optionally
        ``crop_source``) or a source path -- the experiment root or its
        ``merged`` folder.
    :param object_type: default object type for the merged source.
    :param prefer: force ``'png'`` or ``'merged'``.
    :raises CropError: the requested source is not available.
    """
    if isinstance(settings_or_src, Mapping):
        settings = dict(settings_or_src)
        src = settings.get("src")
        if isinstance(src, (list, tuple)):
            src = src[0] if src else None
    else:
        settings = {}
        src = settings_or_src
    if not src:
        raise CropError("no 'src' to resolve a crop source from")

    root = _looks_like_experiment_root(str(src))
    merged_dir = os.path.join(root, "merged")
    db_path = os.path.join(root, "measurements", "measurements.db")

    choice = prefer or settings.get("crop_source") or "auto"
    choice = str(choice).lower()
    if choice not in ("auto", "png", "merged"):
        raise CropError(
            f"crop_source must be 'auto', 'png' or 'merged', got {choice!r}")

    has_png = _has_png_folder(root)
    has_merged = os.path.isdir(merged_dir)

    if choice == "png":
        return PngCropSource(root=root, reason="requested explicitly")
    if choice == "auto" and has_png:
        return PngCropSource(
            root=root,
            reason=f"pre-generated PNG crops found under {os.path.join(root, 'data')}")

    if not has_merged:
        raise CropError(
            f"no crop source available for {root}: no '*_png' folder under "
            f"'data/' and no 'merged/' folder")

    saved: Dict[str, Any] = {}
    if os.path.isfile(db_path):
        try:
            saved = crop_settings_from_db(db_path)
        except CropError:
            saved = {}
    merged_settings = dict(saved)
    # Anything the caller set explicitly wins over the saved snapshot.
    for key in ("png_dims", "png_size", "normalize", "normalize_by", "crop_mode",
                "use_bounding_box", "dialate_pngs", "dialate_png_ratios",
                "cell_mask_dim", "nucleus_mask_dim", "pathogen_mask_dim",
                "organelle_mask_dim"):
        if key in settings:
            merged_settings[key] = settings[key]
    spec = crop_spec_from_settings(merged_settings, object_type=object_type)

    if choice == "merged":
        reason = "requested explicitly"
    else:
        reason = "no pre-generated PNG crops found; cutting from merged/*.npy"
    if saved:
        reason += " (crop settings recovered from measurements.db)"
    return MergedCropSource(spec=spec, merged_root=merged_dir,
                            object_type=object_type, reason=reason)
