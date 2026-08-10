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
percentile normalisation, same dtype. :func:`png_view` turns that array into
what a consumer sees after the PNG round trip, and :func:`read_crop_png` reads
a crop PNG back into the same thing -- the two are the two halves of one
contract, and ``tests/test_crops.py`` asserts they agree.

Crop PNG format
---------------
This module is also the authority on **what a crop PNG on disk means** --
see the "Crop PNG format" section below. In short: the current format is 3
("declared_rgb", :data:`CROP_FORMAT_CURRENT`), whose red, green and blue slots
hold exactly the source channels ``settings['png_channel_mapping']`` names;
format 1 ("legacy", unmarked) holds the same bytes for the default mapping and
is returned untouched; format 2, written for eleven days in 2026, is the one
that is reversed, and :func:`read_crop_png` is what reverses it back. A folder
says which format it is via a ``.spacr_crop_format.json`` sidecar, and an
unmarked folder is format 1.

Dependencies
------------
numpy, and the standard library (plus PIL only inside :func:`read_crop_png`
and cv2 only inside :func:`migrate_crop_folder`, both imported lazily). No
torch, no cellpose, no scipy, no skimage -- importing this module must stay
cheap enough for a GUI thumbnail path, and ``spacr.measure`` imports it for
the writer helpers, so it must not import ``spacr`` back.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import sqlite3
import tempfile
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
    # -- crop PNG format ---------------------------------------------------
    "CROP_FORMAT_LEGACY_BGR",
    "CROP_FORMAT_RGB",
    "CROP_FORMAT_CURRENT",
    "CROP_FORMAT_SIDECAR",
    "CROP_FORMAT_DB_COLUMN",
    "CropFormatConflict",
    "CROP_FORMAT_DECLARED_RGB",
    "narrow_to_uint8",
    "to_cv2_bgr",
    "legacy_png_view",
    "PNG_COLOR_KEYS",
    "DEFAULT_PNG_CHANNEL_MAPPING",
    "png_dims_to_channel_mapping",
    "resolve_png_channel_mapping",
    "build_png_channels",
    "read_crop_folder_marker",
    "write_crop_folder_marker",
    "stamp_crop_folder",
    "crop_folder_format",
    "crop_format_for_png",
    "read_crop_png",
    "read_db_crop_format",
    "stamp_crop_format_in_db",
    "clear_crop_format_cache",
    "MigrationResult",
    "migrate_crop_folder",
    "migrate_crop_tree",
    "find_crop_folders",
    "legacy_channel_names",
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

        :param label: the object's integer label in this plane. The index is
            built from the plane's non-zero pixels, so background (``0``) is
            never in it, and a label the plane does not hold raises
            :class:`LabelMissing` rather than returning an empty box.
        """
        i = self._index(label)
        return (int(self.ymin[i]), int(self.ymax[i]) + 1,
                int(self.xmin[i]), int(self.xmax[i]) + 1)

    def area(self, label: int) -> int:
        """Return the pixel count of ``label``.

        :param label: the object's integer label; one the plane does not hold
            raises :class:`LabelMissing`. The count is of the label's own
            pixels in the mask plane, not of the dilated or bounding-box
            region the crop may end up covering.
        """
        return int(self.count[self._index(label)])

    def centroid(self, label: int) -> Tuple[float, float]:
        """Return the ``(row, col)`` centroid of ``label``.

        :param label: the object's integer label; one the plane does not hold
            raises :class:`LabelMissing`. The centroid is the unweighted mean
            of that label's pixel coordinates -- what
            ``scipy.ndimage.center_of_mass`` computes on the binary region --
            so the intensity channels never move it.
        """
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

        :param object_type: one of :data:`MASK_PLANE_ORDER` -- ``'cell'``,
            ``'nucleus'``, ``'pathogen'``, ``'organelle'``. ``'cytoplasm'``
            is refused rather than defaulted: it has no plane on disk, so
            :meth:`mask_plane` is the only way to get it. The index comes
            from this field's ``mask_dims``, not from the array, so a
            settings dict that names a plane the array does not have fails
            here rather than silently cropping by the wrong stain.
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

        :param object_type: ``'cell'`` | ``'nucleus'`` | ``'pathogen'`` |
            ``'organelle'`` come back as a view on the memory-mapped array, so
            nothing is read off disk until pixels are touched.
            ``'cytoplasm'`` is the derived plane and costs a full pass over
            the field to build, so it is cached on this field and every later
            call is free.
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
        """Return the cached :class:`_LabelIndex` for ``object_type``'s plane.

        :param object_type: which plane to index. The cache is keyed on the
            plane index (``'cytoplasm'`` gets its own key, since it has no
            plane on disk), so two object types recorded at the same
            ``mask_dim`` share one index. The scan happens on first use and
            is kept for the life of the field, which is what makes drawing
            many objects out of one field a single pass over the plane.
        """
        key = -1 if object_type == "cytoplasm" else self.mask_dim(object_type)
        idx = self._indices.get(key)
        if idx is None:
            idx = _LabelIndex(self.mask_plane(object_type))
            self._indices[key] = idx
        return idx

    def labels(self, object_type: str = "cell") -> List[int]:
        """Return every non-zero label present in ``object_type``'s plane.

        :param object_type: which plane to list; defaults to ``'cell'``
            because that is the plane every spaCR run has. Labels come back
            in ascending order, background (``0``) is never among them, and
            an empty list means the plane holds no objects at all -- not that
            the plane is missing, which raises instead.
        """
        return [int(v) for v in self.label_index(object_type).labels]

    # -- windows -----------------------------------------------------------
    def read_window(self, y0: int, y1: int, x0: int, x1: int,
                    channels: Sequence[int], dtype=None) -> np.ndarray:
        """Read ``channels`` over ``[y0:y1, x0:x1]``, zero-padding outside the array.

        The window may run off any edge; the out-of-array part comes back as
        zeros, which is what the PNG path's ``np.pad`` produces.

        :param y0: first row of the window. May be negative -- that part is
            padded, not clamped, so the object stays centred in the result.
        :param y1: one past the last row (half-open). May exceed the field
            height; the overhang is padded the same way.
        :param x0: first column, negative allowed as for ``y0``.
        :param x1: one past the last column, over-wide allowed as for ``y1``.
        :param channels: plane indices in output order -- result channel
            ``k`` holds plane ``channels[k]``, and repeating an index
            repeats the plane. Each must satisfy ``0 <= c < C``; negative
            indices are rejected rather than wrapped, so ``-1`` is an error
            and not "the last plane". An empty sequence yields an
            ``(h, w, 0)`` array here; :func:`extract_crop` rejects it first.
        :param dtype: dtype of the returned array. ``None`` uses
            :attr:`crop_dtype`, i.e. what the PNG path would have cropped in;
            pass one explicitly only to match an array you already hold.
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
        """Read ``object_type``'s label plane over ``[y0:y1, x0:x1]``, zero-padded.

        :param object_type: which label plane to read; ``'cytoplasm'`` is
            served from the derived (and cached) plane, every other type
            straight off the memory map.
        :param y0: first row; may be negative, and the overhang comes back as
            zeros -- which reads as background, so no label ever appears to
            run past the edge of the field.
        :param y1: one past the last row (half-open); may exceed the field
            height, padded as for ``y0``.
        :param x0: first column, negative allowed as for ``y0``.
        :param x1: one past the last column, over-wide allowed as for ``y1``.
        """
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
        area_for_dilation = float(region.size)
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
        # A radius of 0 means no dilation. scipy reads iterations=0 as "repeat
        # until nothing changes", which grew the region to the whole field and
        # turned the crop into an unmasked window on the middle of the image —
        # for every object under ~25 px at the default ratio. measure.py
        # guards it now, and so does this.
        if px > 0:
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
    ``_measure_crop_core`` hands to the writer. Its dtype is the merged
    array's (``uint16`` for a normal spaCR run) and its channel order is
    ``spec.channels``. Use :func:`png_view` to get what a consumer reading the
    written PNG (via :func:`read_crop_png`) would see.

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

    This is the contract, and it is deliberately boring: **channel ``i`` of
    the crop is channel ``i`` of the result**, narrowed to 8 bit by
    :func:`narrow_to_uint8`.

    That holds because a crop is cut in COLOUR order -- ``CropSpec.channels``
    is ``(red_source, green_source, blue_source)``, built by
    :func:`channels_from_settings` from the declared
    ``png_channel_mapping``. So channel 0 is the red one here, in the file,
    and in what :func:`read_crop_png` hands back. There is exactly one order
    and every part of the crop path speaks it.

    The alternative -- keeping crops in ``png_dims`` list order and
    translating at the edges -- is what made the on-demand source and the
    PNG folder return different pixels for the same object.

    :func:`read_crop_png` returns exactly this for the same object, for a crop
    written in either format, which is what makes the on-demand source and the
    PNG folder interchangeable.

    Before spaCR grew a crop-format marker the answer was the *reverse* of this
    (see :func:`legacy_png_view`); that was a bug, not a convention.

    :param crop: the array returned by :func:`extract_crop`.
    :returns: ``(H, W, 3)`` uint8 RGB array.
    """
    arr = np.asarray(crop)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    eight = narrow_to_uint8(arr)
    n = eight.shape[2]
    if n == 1:
        return np.repeat(eight, 3, axis=2)
    if n == 2:
        # The PNG path pads a two-channel crop with a zero third plane, so the
        # blue channel is empty -- not the red one, as it was under the bug.
        rgb = np.zeros((eight.shape[0], eight.shape[1], 3), dtype=np.uint8)
        rgb[:, :, :2] = eight
        return rgb
    return np.ascontiguousarray(eight[:, :, :3])


def legacy_png_view(crop: np.ndarray) -> np.ndarray:
    """Return what a *naive* PIL read of a **legacy** crop PNG gives back.

    Kept, and named for what it is, because it is the inverse of the format-1
    write and therefore the thing :func:`read_crop_png` has to undo. Two
    behaviours, both of them the bug:

    * the channel order is reversed relative to ``png_dims`` -- ``cv2.imwrite``
      read the array as BGR;
    * a ``uint16`` crop is a 16-bit PNG and PIL narrows it two different ways:
      the high byte (``// 256``) for an RGB image, but a *clip* at 255 for a
      single-channel one, which flattens any crop brighter than 255/65535 to
      solid white.

    Nothing in spaCR calls this on the live path any more. It exists so tests
    can prove the legacy reader inverts the legacy writer exactly, and so code
    that genuinely needs bug-compatible pixels (a classifier trained on legacy
    crops, say) can ask for them by name instead of by accident.

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


# ===========================================================================
# Crop PNG format
#
# A PNG on disk carries no field saying which channel order it was written
# in, so the format has to be *versioned* or every reader is guessing.
#
#   format 1 ("legacy", BGR)
#       ``cv2.imwrite(path, png_channels)``. cv2 reads a 3-channel array as
#       BGR, so ``png_dims[0]`` landed in the file's BLUE slot and
#       ``png_dims[2]`` in its RED one. Every crop written by spaCR before
#       2026-07-26 is format 1, and every one of them is unmarked.
#
#       This was read as a bug and it was not one. Microscope channels come
#       off the scope in wavelength order -- 0 is 405 (blue), 1 is 488
#       (green), 2 is 555, 3 is 647 -- so a biologist writing
#       ``png_dims=[0,1,2]`` means "405 blue, 488 green, 555 red", which is
#       exactly what format 1 produced. The bytes are right; only the
#       *reasoning* for them was accidental.
#
#   format 2 ("rgb")
#       ``cv2.imwrite(path, png_channels[..., ::-1])``. Written between
#       2026-07-26 and 2026-08-06 in the belief that ``png_dims[0]`` ought to
#       be red. It puts the 405/DAPI plane in the red slot, so nuclei come
#       out red and the 555 plane comes out blue. **This is the format that
#       is wrong**, and it is the one read_crop_png reverses.
#
#       ``migrate_crop_folder`` rewrote format-1 folders into format 2, so a
#       folder that was migrated in that window holds reversed pixels. It is
#       marked, so it is read correctly; but an external image viewer shows
#       it reversed, and re-running the migrator on it now puts it back.
#
#   format 3 ("declared_rgb", current)
#       The user states the mapping outright:
#       ``png_channel_mapping = {'r': 2, 'g': 1, 'b': 0}`` means source
#       channel 2 is red, 1 is green, 0 is blue. The writer assembles the
#       planes in that order and the file's slots hold them. Nothing is
#       inferred from list position, so there is no convention left to get
#       backwards. For the default mapping this is byte-identical to
#       format 1, which is why the two are read the same way.
#
# The marker, in precedence order:
#
#   1. the folder sidecar ``.spacr_crop_format.json``, written into the crop
#      folder itself. It is the authority, because it travels with the bytes
#      it describes: copy, move, rsync or zip the folder and the marker goes
#      with it. A database row does not -- crop folders routinely outlive,
#      and get copied away from, the ``measurements.db`` that indexed them.
#   2. the ``crop_format`` column on ``png_list``, when a database is on hand.
#      Advisory: it makes the format queryable and survives a folder being
#      re-pointed, but it loses to the sidecar when the two disagree, and
#      the disagreement is reported rather than silently resolved.
#   3. nothing at all -> format 1. Unmarked means legacy, because every crop
#      that exists today is unmarked and every one of them is legacy. This is
#      the only default that cannot corrupt existing data.
#
# 16-bit narrowing: crops are ``uint16`` and the files are 16-bit PNGs, but
# every consumer wants 8-bit RGB. PIL narrows those two different ways -- the
# high byte for an RGB image, a *clip* at 255 for a single-channel one, which
# turns any single-channel crop into solid white. spaCR does the narrowing
# itself now, in exactly one place (:func:`narrow_to_uint8`), with exactly one
# rule: **take the high byte** (``// 256``). It applies to every channel
# count and to both formats. The file keeps its full 16 bits; only the view
# is narrowed, and it is narrowed the same way every time.
# ===========================================================================

#: NOTE ON ORDER: `CropSpec.channels`, `extract_crop`, `png_view` and
#: `read_crop_png` are all in COLOUR order (red, green, blue). The only
#: place list order survives is the legacy `png_dims` setting, and
#: `channels_from_settings` translates it once, at the edge.
#:
#: Format 1: what ``cv2.imwrite(png_channels)`` wrote -- ``png_dims[0]`` in
#: the file's BLUE slot. Named "legacy BGR" for the array-order reversal that
#: produced it, but see :data:`_FORMAT_IS_DECLARED_ORDER`: the pixels it left
#: on disk are in the order the user declared, so it is read back as-is.
CROP_FORMAT_LEGACY_BGR = 1

#: Format 2: ``png_dims[0]`` is the file's red channel. Written between
#: 2026-07-26 and 2026-08-06 only. This is the format that is *wrong* --
#: it puts the first-listed channel, conventionally the 405/DAPI plane, in
#: red -- so it is the one that gets reversed on read.
CROP_FORMAT_RGB = 2

#: Format 3: the file's red, green and blue slots hold exactly the source
#: channels named by ``settings['png_channel_mapping']``. No interpretation,
#: no list-position convention: the mapping says which array index is red and
#: the red slot holds it.
CROP_FORMAT_DECLARED_RGB = 3

#: The format new crops are written in.
CROP_FORMAT_CURRENT = CROP_FORMAT_DECLARED_RGB

#: Whether a format's file slots already hold the colours the user declared.
#:
#: This, not the format number, is what decides a reversal on read. Formats 1
#: and 3 agree pixel-for-pixel for the same declared mapping -- they were
#: produced by different code and arrived at the same bytes -- so reading one
#: as the other must NOT reverse. Only format 2 is out of step.
_FORMAT_IS_DECLARED_ORDER = {
    CROP_FORMAT_LEGACY_BGR: True,
    CROP_FORMAT_RGB: False,
    CROP_FORMAT_DECLARED_RGB: True,
}

#: Sidecar file name, written into each crop folder.
CROP_FORMAT_SIDECAR = ".spacr_crop_format.json"

#: Column :func:`stamp_crop_format_in_db` adds to ``png_list``.
CROP_FORMAT_DB_COLUMN = "crop_format"

#: Suffix of the staging file :func:`migrate_crop_folder` converts through.
#: Its presence means "the file next to me has NOT been converted yet".
CROP_MIGRATION_SUFFIX = ".spacr_v2"

#: Prefix of the temporary files both the marker and the migrator write.
_TMP_PREFIX = ".spacr_tmp_"

_CHANNEL_ORDER_NAME = {
    CROP_FORMAT_LEGACY_BGR: "bgr",
    CROP_FORMAT_RGB: "rgb",
    CROP_FORMAT_DECLARED_RGB: "declared_rgb",
}

#: What the sidecar says about each format, in the words of someone reading it
#: on disk a year from now with no access to this file.
_FORMAT_NOTE = {
    CROP_FORMAT_LEGACY_BGR: (
        "Written before 2026-07-26. The file's blue channel is png_dims[0], "
        "which for a 405/488/555 stack is the nuclear stain -- so these "
        "pixels are already in the order the user declared and spaCR reads "
        "them as they are."),
    CROP_FORMAT_RGB: (
        "Written between 2026-07-26 and 2026-08-06, when png_dims[0] was "
        "wrongly placed in the file's RED channel. Nuclei appear red in an "
        "external viewer. spacr.crops.read_crop_png reverses it on load; "
        "spacr.crops.migrate_crop_folder repairs the file itself."),
    CROP_FORMAT_DECLARED_RGB: (
        "The red, green and blue channels hold the source channels named by "
        "settings['png_channel_mapping']. No list-position convention is "
        "involved, so there is nothing here to read backwards."),
}


class CropFormatConflict(CropError):
    """The sidecar and the database disagree about a folder's crop format."""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for the sidecar."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _coerce_format(value: Any) -> Optional[int]:
    """Return ``value`` as a known crop-format integer, or ``None``."""
    if value is None:
        return None
    try:
        fmt = int(value)
    except (TypeError, ValueError):
        return None
    return fmt if fmt in _CHANNEL_ORDER_NAME else None


# ---------------------------------------------------------------------------
# Narrowing and the writer's channel order
# ---------------------------------------------------------------------------

def narrow_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Narrow ``arr`` to ``uint8`` -- the one and only narrowing rule.

    ``uint16`` (and anything wider) is narrowed by **taking the high byte**,
    which is a plain linear rescale of a crop that ``normalize_to_dtype``
    already stretched across the full dtype range. Floats, which only appear
    when a caller hands in something the crop path never produces, are clipped
    -- there is no dtype range to rescale from.

    Deliberately *not* PIL's behaviour: PIL takes the high byte of a 16-bit
    RGB PNG but clips a 16-bit single-channel one at 255, so the same pixel
    value survives or saturates depending on how many channels its neighbours
    have. One behaviour, applied here, replaces both.

    :param arr: any numeric array.
    :returns: ``uint8`` array of the same shape.
    """
    a = np.asarray(arr)
    if a.dtype == np.dtype(np.uint8):
        return a
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        if info.max <= 255:
            return np.clip(a, 0, 255).astype(np.uint8)
        # Anything wider than 8 bit is high-byte narrowed off the 16-bit
        # range: that is what a 16-bit PNG holds, whatever container PIL
        # chose to hand it back in (uint16 for I;16, int32 for I).
        return (np.clip(a, 0, 65535) // 256).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def to_cv2_bgr(png_channels: np.ndarray) -> np.ndarray:
    """Return ``png_channels`` in the order ``cv2.imwrite`` has to be handed it.

    :func:`build_png_channels` assembles the crop in **file order** -- red
    plane first -- while ``cv2.imwrite`` interprets a 3-channel array as BGR.
    Reversing the channel axis here, once, in the writer, makes cv2's
    interpretation land the array's red plane in the file's red slot, so the
    PNG's slots hold the channels ``settings['png_channel_mapping']`` named
    (format 3). Under :data:`DEFAULT_PNG_CHANNEL_MAPPING` that puts
    ``png_dims[0]`` in blue, byte-identical to format 1.

    * 2-D or single-channel: returned unchanged. cv2 writes a grayscale PNG
      and does no colour interpretation, so there is nothing to reverse.
    * 2 channels: padded with a zero plane to RGB first, then reversed.
      :func:`build_png_channels` never emits two planes -- it carries an empty
      colour as a zero plane in the slot the user left blank -- so this is for
      callers that assemble their own array.
    * 3 channels: reversed.
    * 4 or more: **refused**. cv2 would write BGRA, and PIL then reads the
      fourth intensity plane as an alpha channel and drops it on
      ``convert('RGB')`` -- a whole stain silently deleted from every crop.
      ``settings['png_dims']`` documents a maximum of three entries; this is
      where a fourth stops being ignored and starts being an error.

    :param png_channels: the crop in file order, as :func:`build_png_channels`
        assembles it for ``_measure_crop_core``.
    :returns: the array to hand to ``cv2.imwrite``.
    :raises CropError: more than three channels.
    """
    arr = np.asarray(png_channels)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise CropError(
            f"a crop must be 2-D or (H, W, C); got shape {arr.shape!r}")
    n = arr.shape[2]
    if n == 1:
        return arr
    if n == 2:
        arr = np.dstack((arr, np.zeros_like(arr[:, :, 0])))
    elif n > 3:
        raise CropError(
            f"png_dims selected {n} channels, but a crop PNG holds at most 3: "
            f"cv2 would write channel 4 as an alpha plane and every reader "
            f"would silently drop it. Use at most three entries in png_dims.")
    return arr[:, :, ::-1]


# ---------------------------------------------------------------------------
# The declared channel mapping
# ---------------------------------------------------------------------------

#: The colour slots a crop PNG has, in file order.
PNG_COLOR_KEYS = ("r", "g", "b")

#: What ``png_dims=[0, 1, 2]`` has always meant on screen, stated outright.
#:
#: Microscope channels arrive in wavelength order -- 0 is 405, 1 is 488, 2 is
#: 555, 3 is 647 -- so the first channel is the nuclear stain and belongs in
#: blue. This default reproduces, exactly, what every spaCR crop written
#: before 2026-07-26 looks like.
DEFAULT_PNG_CHANNEL_MAPPING = {"r": 2, "g": 1, "b": 0}


def png_dims_to_channel_mapping(png_dims) -> Dict[str, Optional[int]]:
    """Translate a legacy ``png_dims`` list into an explicit ``{r, g, b}`` map.

    ``png_dims`` never said which colour it meant; the answer was buried in
    cv2's BGR interpretation of the array it was handed, which is how the
    convention got inverted for eleven days without anyone being able to point
    at the line that decided it. The list is still accepted -- every settings
    CSV and every notebook in the wild holds one -- but it is translated here,
    once, into a mapping that says what it means.

    The translation is the *legacy* reading, because that is the one that was
    ever on screen: entry 0 is blue, 1 is green, 2 is red.

    * ``[a, b, c]`` -> ``{'r': c, 'g': b, 'b': a}``
    * ``[a, b]``    -> ``{'r': None, 'g': b, 'b': a}`` (the old zero third plane)
    * ``[a]``       -> ``{'r': a, 'g': a, 'b': a}`` (greyscale; see
      :func:`build_png_channels`, which keeps it a one-plane image)

    :param png_dims: the legacy list of source channel indices.
    :returns: a ``{'r': idx, 'g': idx, 'b': idx}`` dict; ``None`` means an
        empty plane.
    :raises CropError: more than three entries, or an empty list.
    """
    dims = [int(d) for d in list(png_dims or [])]
    if not dims:
        raise CropError("png_dims is empty: a crop needs at least one channel")
    if len(dims) > 3:
        raise CropError(
            f"png_dims selected {len(dims)} channels, but a crop PNG holds at "
            f"most 3. Use at most three entries, or state the mapping "
            f"outright with png_channel_mapping={{'r': .., 'g': .., 'b': ..}}.")
    if len(dims) == 1:
        return {"r": dims[0], "g": dims[0], "b": dims[0]}
    if len(dims) == 2:
        return {"r": None, "g": dims[1], "b": dims[0]}
    return {"r": dims[2], "g": dims[1], "b": dims[0]}


def resolve_png_channel_mapping(settings) -> Dict[str, Optional[int]]:
    """Return the ``{r, g, b}`` source-channel mapping a run should use.

    Precedence, and the reason for it:

    1. ``settings['png_channel_mapping']`` -- the explicit form. If the user
       said which channel is red, that is the answer.
    2. ``settings['png_dims']`` -- the legacy list, translated by
       :func:`png_dims_to_channel_mapping`. A settings CSV written by any
       older build lands here and keeps rendering the way it always did.
    3. :data:`DEFAULT_PNG_CHANNEL_MAPPING`.

    A mapping that names a colour spaCR does not have, or a non-integer index,
    is an error rather than a silent drop: a mis-keyed mapping would otherwise
    delete a whole stain from every crop in the run and say nothing.

    :param settings: the run settings dict (or anything with ``.get``).
    :returns: ``{'r': idx, 'g': idx, 'b': idx}``; ``None`` means an empty plane.
    :raises CropError: an unknown colour key or a non-integer channel index.
    """
    get = getattr(settings, "get", None)
    raw = get("png_channel_mapping", None) if get else None
    if raw is None:
        dims = get("png_dims", None) if get else None
        if dims is None:
            return dict(DEFAULT_PNG_CHANNEL_MAPPING)
        return png_dims_to_channel_mapping(dims)

    if not isinstance(raw, dict):
        raise CropError(
            f"png_channel_mapping must be a dict like "
            f"{{'r': 2, 'g': 1, 'b': 0}}; got {type(raw).__name__}")
    unknown = {str(k).lower() for k in raw} - set(PNG_COLOR_KEYS)
    if unknown:
        raise CropError(
            f"png_channel_mapping has no colour {sorted(unknown)!r}; the keys "
            f"are 'r', 'g' and 'b'. A mis-keyed colour would drop that stain "
            f"from every crop in the run.")
    out: Dict[str, Optional[int]] = {}
    for key in PNG_COLOR_KEYS:
        val = raw.get(key, raw.get(key.upper()))
        if val is None or (isinstance(val, str) and not val.strip()):
            out[key] = None
            continue
        try:
            out[key] = int(val)
        except (TypeError, ValueError):
            raise CropError(
                f"png_channel_mapping['{key}'] must be a source channel index "
                f"or blank; got {val!r}")
    if all(v is None for v in out.values()):
        raise CropError(
            "png_channel_mapping leaves every colour empty, so every crop "
            "would be a black square")
    return out


def channels_from_settings(settings) -> tuple:
    """Return the source channels in COLOUR order: ``(red, green, blue)``.

    The one translation from "what the user declared" to "what a crop array
    holds". Everything downstream -- :class:`CropSpec`, :func:`png_view`,
    :func:`extract_crop` -- is in this order, so channel 0 of a crop is
    always the red one and there is no second convention to keep straight.

    A colour left empty has no source channel, so it cannot appear in a tuple
    of indices; it is filled with the first channel that *is* mapped, and the
    emptiness is applied later by :func:`build_png_channels`, which is the
    only place that can write a zero plane. Callers that need the empty plane
    honoured should use the mapping directly.

    :param settings: the run settings dict.
    :returns: a 1- or 3-tuple of source channel indices.
    """
    mapping = resolve_png_channel_mapping(settings)
    idxs = [mapping.get(k) for k in PNG_COLOR_KEYS]
    if idxs[0] is not None and idxs[0] == idxs[1] == idxs[2]:
        return (int(idxs[0]),)
    fallback = next(i for i in idxs if i is not None)
    return tuple(int(fallback if i is None else i) for i in idxs)


def build_png_channels(data: np.ndarray, mapping: Dict[str, Optional[int]],
                       dtype=None) -> np.ndarray:
    """Assemble the crop planes in **file order** -- red, green, blue.

    The array this returns is in the order the PNG's slots are in, so a reader
    that opens the file and a caller that keeps the array in memory are
    looking at the same thing. That is the whole point of the mapping: there
    is one order, it is stated, and it survives to disk.

    Greyscale is preserved: when all three colours name the same source
    channel the result is a single plane, so cv2 writes a one-channel PNG
    exactly as ``png_dims=[a]`` always did, rather than three identical
    planes at three times the size.

    :param data: the merged ``(H, W, C)`` array.
    :param mapping: as returned by :func:`resolve_png_channel_mapping`.
    :param dtype: optional dtype to cast the assembled planes to.
    :returns: ``(H, W, 1|3)`` array, red plane first.
    :raises CropError: a mapping index is out of range for ``data``.
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise CropError(
            f"build_png_channels needs a (H, W, C) array; got {arr.shape!r}")
    n_src = arr.shape[2]
    for key, idx in mapping.items():
        if idx is not None and not (-n_src <= int(idx) < n_src):
            raise CropError(
                f"png_channel_mapping['{key}'] = {idx} is out of range for an "
                f"array with {n_src} channels")

    idxs = [mapping.get(k) for k in PNG_COLOR_KEYS]
    if idxs[0] is not None and idxs[0] == idxs[1] == idxs[2]:
        planes = [arr[:, :, idxs[0]]]
    else:
        blank = None
        planes = []
        for idx in idxs:
            if idx is None:
                if blank is None:
                    blank = np.zeros(arr.shape[:2], dtype=arr.dtype)
                planes.append(blank)
            else:
                planes.append(arr[:, :, idx])
    out = np.stack(planes, axis=2)
    return out.astype(dtype) if dtype is not None else out


# ---------------------------------------------------------------------------
# The folder sidecar
# ---------------------------------------------------------------------------

# (folder -> (cache key, marker or None)). Cleared by clear_crop_format_cache.
_FORMAT_CACHE: Dict[str, Tuple[Any, Optional[Dict[str, Any]]]] = {}
# Folders this process has already stamped, so the writer pays one stat per
# folder rather than one per crop.
_STAMPED_FOLDERS: set = set()
# db path -> table-wide crop_format. Held for the life of the process, not
# keyed on mtime: the annotate GUI writes labels into png_list constantly, and
# an mtime key would re-run a full-table SELECT DISTINCT for every thumbnail.
# A dataset's crop format does not change under a running session -- and when
# spaCR itself changes it, stamp_crop_format_in_db drops the entry.
_DB_FORMAT_CACHE: Dict[str, Optional[int]] = {}


def clear_crop_format_cache() -> None:
    """Forget every cached folder marker (and every "already stamped" folder)."""
    _FORMAT_CACHE.clear()
    _STAMPED_FOLDERS.clear()
    _DB_FORMAT_CACHE.clear()


def _sidecar_path(folder: str) -> str:
    return os.path.join(os.fspath(folder), CROP_FORMAT_SIDECAR)


def _cache_stamp(folder: str) -> Any:
    """Return a value that changes whenever the folder's sidecar could have."""
    side = _sidecar_path(folder)
    try:
        st = os.stat(side)
        return ("file", st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    try:
        return ("none", os.stat(folder).st_mtime_ns)
    except OSError:
        return ("gone",)


def read_crop_folder_marker(folder: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Return the parsed ``.spacr_crop_format.json`` of ``folder``, or ``None``.

    A sidecar that cannot be parsed is treated as absent -- a corrupt marker
    must not be *more* trusted than no marker, and no marker means legacy,
    which is the safe answer.

    :param folder: the crop folder (the one holding the PNGs).
    :param use_cache: reuse a cached read while the sidecar is unchanged.
    :returns: the marker dict, or ``None`` when there is no usable one.
    """
    key = os.path.abspath(os.fspath(folder))
    stamp = _cache_stamp(key)
    if use_cache:
        cached = _FORMAT_CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    marker: Optional[Dict[str, Any]] = None
    try:
        with open(_sidecar_path(key), "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and _coerce_format(
                loaded.get("spacr_crop_format")) is not None:
            marker = loaded
    except (OSError, ValueError):
        marker = None
    # A folder mid-migration changes on every file, so caching it would serve
    # a stale watermark and mis-read the files either side of it.
    if marker is None or "migration" not in marker:
        _FORMAT_CACHE[key] = (stamp, marker)
    return marker


def write_crop_folder_marker(folder: str, fmt: int = CROP_FORMAT_CURRENT,
                             **extra: Any) -> str:
    """Write ``folder``'s crop-format sidecar atomically.

    Temp file plus :func:`os.replace`, like ``spacr.io._save_array_atomic``:
    a marker is either the previous one or the complete new one, never a
    half-written JSON document that :func:`read_crop_folder_marker` would
    then read as "no marker" -- i.e. as legacy -- over a folder of corrected
    crops.

    :param folder: the crop folder.
    :param fmt: any known crop format (1, 2 or 3); defaults to
        :data:`CROP_FORMAT_CURRENT`, i.e. :data:`CROP_FORMAT_DECLARED_RGB`.
    :param extra: extra keys to record (``migration``, ``png_dims``, ...).
        A key whose value is ``None`` is dropped.
    :returns: the sidecar path.
    :raises CropError: ``fmt`` is not a known format.
    """
    if _coerce_format(fmt) is None:
        raise CropError(f"unknown crop format {fmt!r}")
    fmt = int(fmt)
    folder = os.path.abspath(os.fspath(folder))
    os.makedirs(folder, exist_ok=True)
    payload: Dict[str, Any] = {
        "spacr_crop_format": fmt,
        "channel_order": _CHANNEL_ORDER_NAME[fmt],
        "narrowing": "high-byte",
        "updated_utc": _utc_now(),
        "note": _FORMAT_NOTE[fmt],
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, _sidecar_path(folder))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    _FORMAT_CACHE.pop(folder, None)
    return _sidecar_path(folder)


def stamp_crop_folder(folder: str, fmt: int = CROP_FORMAT_CURRENT) -> Optional[str]:
    """Make sure ``folder`` carries the format marker. Cheap enough to call per crop.

    Called by the crop writer immediately *before* the first PNG lands, so a
    run killed part-way through leaves a marked folder holding fewer crops --
    never an unmarked folder of format-2 crops, which is the one state that
    would be silently misread as legacy.

    One listing per folder per process: after that the folder is remembered.
    A marker that cannot be written is a loud warning rather than an
    exception, because failing the whole measure run over a 300-byte sidecar
    helps nobody -- but it is never silent, because the consequence is that
    the crops read back reversed.

    Writing new crops into a folder that already holds *old* ones is the one
    case a single folder-level marker cannot describe, so it is called out
    rather than papered over: the run's own crops are marked, and the message
    says which files were there first and what to do about them. (Migrating
    them here instead would mean several measure workers converting the same
    folder at once, which is exactly the race
    :func:`migrate_crop_folder`'s single-process design rules out.)

    :param folder: the crop folder.
    :param fmt: format to record; defaults to :data:`CROP_FORMAT_CURRENT`.
    :returns: the sidecar path, or ``None`` if it could not be written.
    """
    key = os.path.abspath(os.fspath(folder))
    if key in _STAMPED_FOLDERS:
        return _sidecar_path(key)
    fmt = int(fmt)
    try:
        existing = read_crop_folder_marker(key)
        found = _coerce_format(existing.get("spacr_crop_format")) if existing else None
        if found != fmt:
            # Was `fmt == CROP_FORMAT_RGB`, which silently stopped warning the
            # moment the current format moved to 3. Ask whether this is the
            # format new crops are written in, not which number that happens
            # to be today.
            if existing is None and fmt == CROP_FORMAT_CURRENT:
                stale = _crop_pngs_in(key)
                # Re-read the marker before complaining. The writer stamps
                # before its first PNG, so if a sibling measure worker got
                # here first its marker is already on disk and the PNGs we
                # just listed are this run's, not an old dataset's.
                if stale and read_crop_folder_marker(key, use_cache=False) is None:
                    print(
                        f"spacr: {key} already holds {len(stale)} unmarked "
                        f"crop PNG(s), which are in the old reversed channel "
                        f"order, and this run is about to add corrected ones. "
                        f"Crops this run overwrites are fine; any it does not "
                        f"will be read as if they were corrected. Delete the "
                        f"folder before re-measuring, or convert it first "
                        f"with: python -m spacr.crops {os.path.dirname(key)}")
            elif found is not None:
                print(
                    f"spacr: {key} is marked crop format {found} "
                    f"({_CHANNEL_ORDER_NAME[found]}) and this run writes "
                    f"format {fmt} ({_CHANNEL_ORDER_NAME[fmt]}). Re-marking "
                    f"it; any crop in here the run does not overwrite will be "
                    f"read in the wrong order.")
            write_crop_folder_marker(key, fmt)
    except Exception as exc:
        print(f"spacr: could not write the crop-format marker in {key}: "
              f"{exc}. Crops written here will be read back with their "
              f"channels reversed until "
              f"spacr.crops.write_crop_folder_marker() is run on it.")
        return None
    _STAMPED_FOLDERS.add(key)
    return _sidecar_path(key)


# ---------------------------------------------------------------------------
# Resolving the format of a folder / of one file
# ---------------------------------------------------------------------------

def read_db_crop_format(db_path: str, png_path: Optional[str] = None,
                        table: str = "png_list") -> Optional[int]:
    """Return the crop format recorded in the database, or ``None``.

    Reads the ``crop_format`` column of ``png_list``: for one ``png_path`` if
    given, otherwise the single distinct value covering the whole table (a
    table holding both formats returns ``None`` -- ambiguous is not an answer).

    :param db_path: path to ``measurements.db``.
    :param png_path: restrict to one crop's row.
    :param table: table holding the crops.
    :returns: the format integer, or ``None`` when the column, the table or
        the database is absent, or the answer is ambiguous.
    """
    if not db_path or not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        cols = {row[1] for row in conn.execute(
            f'PRAGMA table_info("{table}")').fetchall()}
        if CROP_FORMAT_DB_COLUMN not in cols:
            return None
        if png_path:
            rows = conn.execute(
                f'SELECT DISTINCT "{CROP_FORMAT_DB_COLUMN}" FROM "{table}" '
                f'WHERE png_path = ?', (str(png_path),)).fetchall()
            if not rows:
                # The exact path is not in this database (a folder copied
                # somewhere else, say). Fall back to the table-wide answer.
                rows = conn.execute(
                    f'SELECT DISTINCT "{CROP_FORMAT_DB_COLUMN}" FROM "{table}"'
                ).fetchall()
        else:
            rows = conn.execute(
                f'SELECT DISTINCT "{CROP_FORMAT_DB_COLUMN}" FROM "{table}"'
            ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    values = {_coerce_format(r[0]) for r in rows}
    values.discard(None)
    if len(values) != 1:
        return None
    return values.pop()


def _db_crop_format_cached(db_path: str) -> Optional[int]:
    """:func:`read_db_crop_format` for a whole table, memoised per process.

    The table-wide query is a ``SELECT DISTINCT`` over every crop row; running
    it once per thumbnail would make the annotate grid quadratic in the size
    of ``png_list``.
    """
    key = os.path.abspath(os.fspath(db_path))
    if key not in _DB_FORMAT_CACHE:
        _DB_FORMAT_CACHE[key] = read_db_crop_format(db_path)
    return _DB_FORMAT_CACHE[key]


def stamp_crop_format_in_db(db_path: str, png_paths: Optional[Iterable[str]] = None,
                            fmt: int = CROP_FORMAT_CURRENT,
                            table: str = "png_list") -> int:
    """Record the crop format on ``png_list``, adding the column if needed.

    The database copy is advisory -- :func:`crop_format_for_png` prefers the
    sidecar -- but it makes "which of my plates are still legacy?" a query
    rather than a filesystem walk.

    :param db_path: path to ``measurements.db``.
    :param png_paths: rows to stamp; ``None`` stamps every row.
    :param fmt: the format to record.
    :param table: table holding the crops.
    :returns: number of rows updated.
    :raises CropError: unknown ``fmt``.
    """
    if _coerce_format(fmt) is None:
        raise CropError(f"unknown crop format {fmt!r}")
    if not db_path or not os.path.isfile(db_path):
        return 0
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cols = {row[1] for row in conn.execute(
            f'PRAGMA table_info("{table}")').fetchall()}
        if not cols:
            return 0
        if CROP_FORMAT_DB_COLUMN not in cols:
            conn.execute(
                f'ALTER TABLE "{table}" ADD COLUMN "{CROP_FORMAT_DB_COLUMN}" INTEGER')
        if png_paths is None:
            cur = conn.execute(
                f'UPDATE "{table}" SET "{CROP_FORMAT_DB_COLUMN}" = ?', (int(fmt),))
            n = cur.rowcount
        else:
            paths = [(int(fmt), str(p)) for p in png_paths]
            cur = conn.executemany(
                f'UPDATE "{table}" SET "{CROP_FORMAT_DB_COLUMN}" = ? '
                f'WHERE png_path = ?', paths)
            n = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    # The memoised table-wide answer is now stale.
    _DB_FORMAT_CACHE.pop(os.path.abspath(db_path), None)
    return int(n or 0)


def crop_folder_format(folder: str, db_path: Optional[str] = None,
                       *, strict: bool = False) -> int:
    """Return the crop format that applies to ``folder`` as a whole.

    Precedence: sidecar, then the database column, then
    :data:`CROP_FORMAT_LEGACY_BGR`. When both are present and they disagree,
    **the sidecar wins** -- it is the marker that travels with the folder --
    and the disagreement is reported: printed by default, raised when
    ``strict``.

    A folder in the middle of a migration reports ``CROP_FORMAT_LEGACY_BGR``,
    because the files that have not been converted yet still are; use
    :func:`crop_format_for_png` to resolve one file inside such a folder.

    :param folder: the crop folder.
    :param db_path: optional ``measurements.db`` to consult.
    :param strict: raise :class:`CropFormatConflict` instead of printing.
    :returns: the format integer.
    """
    marker = read_crop_folder_marker(folder)
    from_db = _db_crop_format_cached(db_path) if db_path else None
    if marker is not None:
        fmt = _coerce_format(marker.get("spacr_crop_format"))
        if from_db is not None and from_db != fmt:
            msg = (f"crop format conflict for {folder}: the sidecar "
                   f"{CROP_FORMAT_SIDECAR} says {fmt} "
                   f"({_CHANNEL_ORDER_NAME[fmt]}) and {db_path} says "
                   f"{from_db} ({_CHANNEL_ORDER_NAME[from_db]}). Using the "
                   f"sidecar: it travels with the crops, the database row "
                   f"does not.")
            if strict:
                raise CropFormatConflict(msg)
            print(f"spacr: {msg}")
        if "migration" in marker:
            return CROP_FORMAT_LEGACY_BGR
        return fmt
    if from_db is not None:
        return from_db
    return CROP_FORMAT_LEGACY_BGR


def crop_format_for_png(png_path: str, db_path: Optional[str] = None,
                        *, strict: bool = False) -> int:
    """Return the crop format of one crop PNG.

    Same precedence as :func:`crop_folder_format`, plus the two per-file
    overrides an interrupted :func:`migrate_crop_folder` leaves behind:

    * a leftover ``<name>.spacr_v2`` staging file means the file next to it
      has **not** been converted yet -- it is still legacy;
    * a name in the marker's ``unconverted`` list is a file the migration
      could not rewrite. It stays legacy for good, in a folder that is
      otherwise format 2, so it still has to be read correctly;
    * otherwise, inside a folder whose marker carries a ``migration`` block,
      a file at or before the recorded watermark is converted and one after
      it is not.

    :param png_path: the crop PNG.
    :param db_path: optional ``measurements.db`` to consult.
    :param strict: raise on a sidecar/database conflict.
    :returns: the format integer.
    """
    png_path = os.fspath(png_path)
    folder = os.path.dirname(os.path.abspath(png_path)) or "."
    marker = read_crop_folder_marker(folder)
    if isinstance(marker, dict):
        name = os.path.basename(png_path)
        migration = marker.get("migration")
        source = _coerce_format(
            (migration or marker).get("from")
            or marker.get("migrated_from")) or CROP_FORMAT_LEGACY_BGR
        if os.path.exists(png_path + CROP_MIGRATION_SUFFIX):
            return source
        if name in set((migration or marker).get("unconverted") or ()):
            return source
        if migration:
            done_through = migration.get("done_through")
            target = (_coerce_format(marker.get("spacr_crop_format"))
                      or CROP_FORMAT_CURRENT)
            if done_through is not None and name <= str(done_through):
                return target
            return source
    return crop_folder_format(folder, db_path, strict=strict)


# ---------------------------------------------------------------------------
# Reading a crop PNG
# ---------------------------------------------------------------------------

def read_crop_png(path: str, fmt: Optional[int] = None,
                  db_path: Optional[str] = None,
                  as_format: int = CROP_FORMAT_CURRENT) -> np.ndarray:
    """Read a crop PNG and return it in the corrected order, as 8-bit RGB.

    The one function every consumer of a crop folder should go through. It
    resolves the file's format (see :func:`crop_format_for_png`), reverses the
    channel axis when the file's ordering differs from the one asked for --
    which today means format 2 only, since formats 1 and 3 are both already in
    declared order -- and narrows to 8 bit with :func:`narrow_to_uint8`, so a
    legacy dataset and a new one come back identical and the caller never has
    to know which it opened.

    The result equals ``png_view(extract_crop(...))`` for the same object,
    under either format. That equality is the contract, and
    ``tests/test_crops.py`` asserts it.

    :param path: the crop PNG.
    :param fmt: what the file on disk is, when you know better than the
        marker does. ``None`` resolves it.
    :param as_format: what ordering you want *back*. The default is the
        corrected one. Pass :data:`CROP_FORMAT_LEGACY_BGR` to get what a
        classifier trained on legacy crops expects, out of a folder in either
        format -- explicitly, by name, rather than by accident.
    :param db_path: optional ``measurements.db`` consulted when the folder has
        no sidecar.
    :returns: ``(H, W, 3)`` uint8 RGB array.
    :raises MergedFileMissing: the file does not exist.
    :raises CropError: ``as_format`` is not a known format.
    """
    from PIL import Image

    path = os.fspath(path)
    if not os.path.isfile(path):
        raise MergedFileMissing(f"crop PNG not found: {path}")
    if _coerce_format(as_format) is None:
        raise CropError(f"unknown crop format {as_format!r}")
    if fmt is None:
        fmt = crop_format_for_png(path, db_path)
    with Image.open(path) as img:
        mode = img.mode
        if mode in ("RGB", "L") or mode.startswith("I") or mode == "F":
            # I;16 (a 16-bit single-channel PNG) must NOT go through
            # convert('L'): PIL clips it at 255 and every crop brighter than
            # that comes back solid white. Take the raw samples and narrow
            # them ourselves.
            arr = np.array(img)
        else:
            arr = np.array(img.convert("RGB"))
    # Every branch above yields either a 2-D plane (L / I;16 / F) or three
    # channels (RGB, or anything else converted to it), so there is no other
    # shape to handle here.
    arr = narrow_to_uint8(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    # There are still exactly two ORDERINGS, but now three formats, so the
    # reversal is decided by which ordering each format is in -- not by
    # `fmt != as_format`, which would reverse between formats 1 and 3 even
    # though they hold identical bytes.
    if (_FORMAT_IS_DECLARED_ORDER.get(int(fmt), True)
            is not _FORMAT_IS_DECLARED_ORDER.get(int(as_format), True)):
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

@dataclass
class MigrationResult:
    """What :func:`migrate_crop_folder` did to one folder.

    :ivar folder: the folder.
    :ivar converted: files whose channel order was rewritten.
    :ivar skipped: files that needed no rewrite (already converted, or
        single-channel, where there is no order to fix).
    :ivar failed: ``(name, reason)`` for files that could not be converted.
    :ivar already: the folder was already at the target format; nothing done.
    :ivar dry_run: nothing was written.
    :ivar mode: ``'rewrite'`` or ``'mark'``.
    """

    folder: str
    converted: List[str] = _dc_field(default_factory=list)
    skipped: List[str] = _dc_field(default_factory=list)
    failed: List[Tuple[str, str]] = _dc_field(default_factory=list)
    already: bool = False
    dry_run: bool = False
    mode: str = "rewrite"

    def describe(self) -> str:
        """Return a one-line summary for a log."""
        if self.already:
            return f"{self.folder}: already format {CROP_FORMAT_CURRENT}, nothing to do"
        if self.mode == "mark":
            return f"{self.folder}: marked as legacy (format {CROP_FORMAT_LEGACY_BGR}), pixels untouched"
        what = "would convert" if self.dry_run else "converted"
        text = (f"{self.folder}: {what} {len(self.converted)} crop(s), "
                f"skipped {len(self.skipped)}")
        if self.failed:
            text += f", FAILED {len(self.failed)}"
        return text


def _crop_pngs_in(folder: str) -> List[str]:
    """Return the crop PNG names in ``folder``, sorted, staging files excluded."""
    try:
        names = os.listdir(folder)
    except OSError as exc:
        raise CropError(f"cannot list crop folder {folder}: {exc}") from exc
    return sorted(
        n for n in names
        if n.lower().endswith(".png") and not n.startswith(".")
        and os.path.isfile(os.path.join(folder, n)))


def _convert_one(src: str, dst: str) -> bool:
    """Write the format-2 version of the legacy crop ``src`` to ``dst``.

    ``cv2.imread(..., IMREAD_UNCHANGED)`` hands back the file's samples in
    BGR order, which for a legacy file is exactly the ``png_channels`` array
    that was passed to ``cv2.imwrite`` -- so re-writing it through
    :func:`to_cv2_bgr` is literally the new writer, bit depth and all. No
    narrowing happens here: the file keeps its 16 bits.

    :returns: True if a file was written, False if the crop needs no rewrite.
    :raises CropError: the PNG cannot be decoded, or has too many channels.
    """
    import cv2

    arr = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise CropError(f"cv2 could not decode {src}")
    if arr.ndim == 2 or arr.shape[2] == 1:
        # Single-channel: cv2 did no colour interpretation on the way in, so
        # there is no reversal to undo. Only the marker changes.
        return False
    out = to_cv2_bgr(arr)
    if not cv2.imwrite(dst, out):
        raise CropError(f"cv2 could not write {dst}")
    return True


def _atomic_convert(src: str, staged: str) -> bool:
    """Convert ``src`` into ``staged`` via a temp file plus :func:`os.replace`."""
    folder = os.path.dirname(staged) or "."
    fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=".png", dir=folder)
    os.close(fd)
    try:
        wrote = _convert_one(src, tmp)
        if not wrote:
            os.remove(tmp)
            return False
        os.replace(tmp, staged)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return True


def migrate_crop_folder(folder: str, *, mode: str = "rewrite",
                        dry_run: bool = False, on_error: str = "raise",
                        db_path: Optional[str] = None,
                        progress: Optional[Any] = None) -> MigrationResult:
    """Repair one folder of reversed crops and stamp it. Idempotent.

    **The direction of this function inverted on 2026-08-06.** It used to
    convert format-1 folders to format 2, on the belief that ``png_dims[0]``
    belonged in the red channel. It does not: channel 0 is 405 and belongs in
    blue, so format-1 folders were right all along and format 2 -- everything
    written, or migrated, between 2026-07-26 and 2026-08-06 -- is the one
    holding reversed pixels. This now repairs *those*.

    ``mode='rewrite'`` (the default) rewrites every 3-channel PNG of a
    **format-2** folder with its channels put back, and marks the folder
    format 3. A folder that is format 1, format 3 or unmarked is already in
    declared order, so it is an immediate no-op -- which is the answer for
    almost every folder that exists.

    ``mode='mark'`` touches no pixels and only records that the folder is
    format 1 -- use it when something outside spaCR reads those exact bytes
    and must keep seeing them (a classifier trained on legacy crops, for
    instance).

    Interruption safety, which is the whole design:

    * each file is converted into a durable staging file ``<name>.spacr_v2``
      (itself written temp-then-``os.replace``, per ``io._save_array_atomic``)
      and only then ``os.replace``-d over the original, so the crop at its
      real name is always a complete PNG -- the old one or the new one;
    * the folder marker carries a ``migration`` block with a ``done_through``
      watermark, advanced *before* the install, so the rule
      **"a staging file exists ⇒ the crop beside it is still legacy"**
      resolves every file at every point in the sequence. That is what
      :func:`crop_format_for_png` reads, and it is why a killed migration is
      still read correctly and can simply be re-run.

    Running it on an already-converted folder is an immediate no-op: nothing
    is decoded, nothing is written, and ``result.already`` is True. Running it
    twice therefore cannot double-reverse anything.

    The one exception is a folder finished with ``on_error='skip'``: its
    marker names the files that could not be rewritten, those stay legacy (and
    are read as legacy) inside an otherwise format-2 folder, and a later run
    retries **only** them.

    :param folder: the crop folder (``.../<well>/cell_png`` and friends).
    :param mode: ``'rewrite'`` or ``'mark'``.
    :param dry_run: report what would happen; write nothing.
    :param on_error: ``'raise'`` (default) or ``'skip'``, which records the
        file in the marker's ``unconverted`` list and keeps reading it as
        legacy.
    :param db_path: also stamp ``png_list.crop_format`` in this database.
    :param progress: optional callable ``(done, total, name)``.
    :returns: a :class:`MigrationResult`.
    :raises CropError: bad arguments, or a file that cannot be converted when
        ``on_error='raise'``.
    """
    if mode not in ("rewrite", "mark"):
        raise CropError(f"mode must be 'rewrite' or 'mark', got {mode!r}")
    if on_error not in ("raise", "skip"):
        raise CropError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    folder = os.path.abspath(os.fspath(folder))
    if not os.path.isdir(folder):
        raise CropError(f"not a crop folder: {folder}")

    result = MigrationResult(folder=folder, dry_run=dry_run, mode=mode)
    marker = read_crop_folder_marker(folder, use_cache=False)
    current = _coerce_format(marker.get("spacr_crop_format")) if marker else None
    migration = marker.get("migration") if marker else None

    if mode == "mark":
        if current == CROP_FORMAT_LEGACY_BGR and not migration:
            result.already = True
            return result
        if current == CROP_FORMAT_RGB and not migration:
            raise CropError(
                f"{folder} is already format {CROP_FORMAT_RGB}; marking it "
                f"legacy would make every crop in it read back reversed")
        if current == CROP_FORMAT_DECLARED_RGB and not migration:
            # Not a corruption -- formats 1 and 3 are both declared order, so
            # nothing would be reversed. It is a LOSS: the marker is the only
            # record that this folder was repaired, and overwriting it with
            # "legacy" makes a repaired folder indistinguishable from one that
            # never needed repairing. Refuse rather than quietly forget.
            raise CropError(
                f"{folder} is format {CROP_FORMAT_DECLARED_RGB}; marking it "
                f"legacy would discard the record that it was repaired. "
                f"Delete {CROP_FORMAT_SIDECAR} first if that is really what "
                f"you want.")
        if not dry_run:
            write_crop_folder_marker(folder, CROP_FORMAT_LEGACY_BGR)
            if db_path:
                stamp_crop_format_in_db(db_path, None, CROP_FORMAT_LEGACY_BGR)
        return result

    retry_only: Optional[set] = None
    if not migration:
        leftover = set((marker or {}).get("unconverted") or ())
        if current == CROP_FORMAT_CURRENT and leftover:
            # A previous run finished with on_error='skip'. The folder is
            # repaired apart from these, so retry exactly them: every other
            # crop in here is already back in declared order and reversing it
            # again would undo the repair.
            #
            # This case has to be tested BEFORE the "nothing to do" check
            # below, because a finished-with-leftovers folder is marked with
            # the TARGET format. Keying the retry on the source format is
            # what made the retry silently return `already` and leave the
            # unconverted files unconverted for ever.
            retry_only = leftover
        elif current != CROP_FORMAT_RGB:
            # Only format 2 has reversed pixels. Format 1, format 3 and
            # unmarked are all in declared order already, so there is nothing
            # to rewrite -- and rewriting one WOULD reverse a correct folder,
            # which is exactly the damage this function exists to undo.
            result.already = True
            return result

    names = _crop_pngs_in(folder)
    done_through = str(migration.get("done_through") or "") if migration else ""
    failed_names = list((migration or marker or {}).get("unconverted") or ())
    started = (migration or {}).get("started_utc") or _utc_now()

    def _todo(name: str) -> bool:
        """True when ``name`` still has to be converted.

        A staged file outranks everything else, in both modes and in both
        directions -- it is the same rule :func:`crop_format_for_png` reads,
        so what the migrator thinks is left to do and what a reader thinks is
        still legacy can never disagree.
        """
        if os.path.exists(os.path.join(folder, name + CROP_MIGRATION_SUFFIX)):
            return True
        if retry_only is not None:
            return name in retry_only
        return not (done_through and name <= done_through)

    if dry_run:
        for name in names:
            (result.converted if _todo(name) else result.skipped).append(name)
        return result

    def _flush(done: str) -> None:
        """Advance the watermark durably. One small fsync per crop, on purpose.

        It is what makes "converted" and "not converted yet" a fact on disk
        rather than a guess, and it is cheap next to decoding and re-encoding
        the PNG it guards.

        A retry run has no watermark to advance -- the folder is already
        format 2 apart from the named leftovers -- so it rewrites the finished
        marker with a shorter ``unconverted`` list instead.
        """
        if retry_only is not None:
            write_crop_folder_marker(
                folder, CROP_FORMAT_DECLARED_RGB,
                migrated_from=CROP_FORMAT_RGB,
                unconverted=sorted(set(failed_names)) or None)
            return
        block = {"from": CROP_FORMAT_RGB, "started_utc": started,
                 "done_through": done}
        if failed_names:
            block["unconverted"] = sorted(set(failed_names))
        write_crop_folder_marker(
            folder, CROP_FORMAT_DECLARED_RGB, migration=block)

    total = len(names)
    for i, name in enumerate(names):
        path = os.path.join(folder, name)
        staged = path + CROP_MIGRATION_SUFFIX
        if not _todo(name):
            result.skipped.append(name)
            if progress:
                progress(i + 1, total, name)
            continue
        try:
            if os.path.exists(staged):
                # A previous run converted it and died before installing it.
                wrote = True
            else:
                wrote = _atomic_convert(path, staged)
        except CropError as exc:
            if on_error == "raise":
                raise CropError(
                    f"{folder}: {name} could not be converted ({exc}). "
                    f"Nothing after it was touched; fix or remove the file "
                    f"and re-run -- the migration resumes where it stopped."
                ) from exc
            failed_names.append(name)
            result.failed.append((name, str(exc)))
            done_through = name
            _flush(done_through)
            if progress:
                progress(i + 1, total, name)
            continue
        # Marker first, install second: a crash between them leaves the
        # staging file in place, and "staging file exists" outranks everything
        # else, so the crop is still correctly read as legacy.
        if name in failed_names:
            failed_names.remove(name)          # a retry that worked
        done_through = name
        _flush(done_through)
        if wrote:
            os.replace(staged, path)
            result.converted.append(name)
        else:
            result.skipped.append(name)
        if progress:
            progress(i + 1, total, name)

    extra: Dict[str, Any] = {}
    if failed_names:
        extra["unconverted"] = sorted(set(failed_names))
    extra["migrated_from"] = CROP_FORMAT_RGB
    extra["migrated_utc"] = _utc_now()
    write_crop_folder_marker(folder, CROP_FORMAT_DECLARED_RGB, **extra)
    if db_path:
        stamp_crop_format_in_db(
            db_path,
            [os.path.join(folder, n) for n in names if n not in failed_names],
            CROP_FORMAT_DECLARED_RGB)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m spacr.crops <path>`` -- migrate crop folders from the shell.

    Exists so the migration is a command a user can run over an old dataset,
    not a Python snippet they have to be told how to write.

    :param argv: argument list; ``None`` uses ``sys.argv[1:]``.
    :returns: process exit status.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m spacr.crops",
        description="Convert object-crop PNG folders to the corrected "
                    "channel order (png_dims[0] = red) and stamp them.")
    parser.add_argument("path", help="experiment root, its data/ folder, or "
                                     "one *_png crop folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change; write nothing")
    parser.add_argument("--mark-legacy", action="store_true",
                        help="do not rewrite any pixels; only record that "
                             "these folders are in the old order, so spaCR "
                             "corrects them on load and anything reading the "
                             "raw bytes (a classifier trained on them) still "
                             "sees what it expects")
    parser.add_argument("--db", default=None,
                        help="measurements.db to stamp as well")
    parser.add_argument("--skip-errors", action="store_true",
                        help="record files that cannot be converted and "
                             "carry on, instead of stopping")
    args = parser.parse_args(argv)

    try:
        results = migrate_crop_tree(
            args.path, dry_run=args.dry_run,
            mode="mark" if args.mark_legacy else "rewrite",
            on_error="skip" if args.skip_errors else "raise",
            db_path=args.db)
    except CropError as exc:
        print(f"spacr: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(result.describe())
    return 1 if any(r.failed for r in results) else 0


def legacy_channel_names(channels: Iterable[str]) -> List[str]:
    """Map a legacy-trained model's ``train_channels`` onto format-2 crops.

    A classifier trained on legacy crops learned "input plane 0 is whatever is
    in the file's red channel", and in a legacy file that is ``png_dims[-1]``.
    Feed the same model a format-2 crop and plane 0 is now ``png_dims[0]`` --
    a permutation of its input, which it will happily score and get wrong,
    with no error anywhere.

    Reversing the request undoes the permutation exactly: red and blue swap,
    green is unmoved. So a model trained with ``train_channels=['r','g','b']``
    keeps seeing the pixels it was trained on if it is applied with
    ``['b','g','r']``, and one trained with ``['r','g']`` with ``['b','g']``.

    This is a stopgap for a model you cannot retrain. Retraining on corrected
    crops is the real fix, and it is cheap compared with getting this wrong.

    :param channels: the ``train_channels`` the model was trained with.
    :returns: the equivalent list to apply it with on format-2 crops.
    """
    swap = {"r": "b", "b": "r", "g": "g"}
    return [swap.get(str(c).strip().lower(), str(c)) for c in channels]


def find_crop_folders(root: str) -> List[str]:
    """Return every ``*_png`` crop folder under ``root``, sorted.

    Accepts an experiment root, its ``data`` folder, or a crop folder itself.

    :param root: where to look.
    :returns: absolute folder paths.
    """
    root = os.path.abspath(os.fspath(root))
    if os.path.basename(root).endswith("_png") and os.path.isdir(root):
        return [root]
    found: List[str] = []
    for start in (os.path.join(root, "data"), root):
        if not os.path.isdir(start):
            continue
        for dirpath, dirnames, _files in os.walk(start):
            for name in sorted(dirnames):
                if name.endswith("_png"):
                    found.append(os.path.join(dirpath, name))
        if found:
            break
    return sorted(set(found))


def migrate_crop_tree(root: str, **kwargs) -> List[MigrationResult]:
    """Run :func:`migrate_crop_folder` on every crop folder under ``root``.

    :param root: experiment root, its ``data`` folder, or one crop folder.
    :param kwargs: forwarded to :func:`migrate_crop_folder`.
    :returns: one :class:`MigrationResult` per folder, in folder order.
    """
    folders = find_crop_folders(root)
    if not folders:
        raise CropError(f"no '*_png' crop folders found under {root}")
    return [migrate_crop_folder(f, **kwargs) for f in folders]


# ---------------------------------------------------------------------------
# Settings plumbing
# ---------------------------------------------------------------------------

def mask_dims_from_settings(settings: Mapping[str, Any]) -> Dict[str, int]:
    """Return ``{object_type: plane index}`` from a ``measure_crop`` settings dict.

    Falls back to :data:`DEFAULT_MASK_DIMS` for anything the dict does not name.

    :param settings: a ``measure_crop`` settings mapping (a live dict or one
        read back by :func:`crop_settings_from_db`). Only the
        ``cell_mask_dim`` / ``nucleus_mask_dim`` / ``pathogen_mask_dim`` /
        ``organelle_mask_dim`` keys are read; a key that is absent, blank,
        the string ``'none'`` or not an integer is skipped rather than
        raised on. The fallback is all-or-nothing: a dict naming even one
        plane returns only the planes it named, so an object type it left out
        has no entry at all -- it is not filled in from
        :data:`DEFAULT_MASK_DIMS`.
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

    :param settings: the ``measure_crop`` settings. The per-``crop_mode``
        lists (``png_size``, ``dialate_pngs``, ``dialate_png_ratios``) are
        indexed by where the chosen object type sits in ``crop_mode``, and
        fall back to entry 0 when it is not listed there. The crop's
        channels come from ``png_channel_mapping`` -- or the legacy
        ``png_dims`` -- via :func:`channels_from_settings`, so the spec is in
        colour order, not ``png_dims`` list order. A ``normalize`` that is a
        sequence of any length but 2 is discarded as ``False``.
    :param merged_path: the ``merged/<fov>.npy`` to record on the spec. The
        default ``""`` builds a *template* spec, which is what
        :class:`MergedCropSource` wants: it fills the path (and label) in per
        row.
    :param object_type: which mask plane to crop by; ``None`` takes the first
        entry of ``settings['crop_mode']``. ``'cytoplasm'`` forces
        ``dilate=False`` whatever the settings say, because
        ``_measure_crop_core`` hard-disables dilation for it.
    :param label: the object's ``object_label``. The default ``0`` is
        background, so it is only meaningful on a template spec -- cutting
        with it raises :class:`LabelMissing`.
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
        # In COLOUR order -- red source, green source, blue source -- which
        # is the order the PNG's slots are in and therefore the order
        # `png_view` and `read_crop_png` both speak. Taking `png_dims`
        # verbatim here is what made the on-demand source and the PNG folder
        # disagree: one was in list order and the other in file order.
        channels=channels_from_settings(settings),
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
        """Return the crop for ``row`` as a ``(H, W, 3)`` uint8 RGB array.

        :param row: one measurement row -- a mapping, a pandas ``Series``, or
            any object carrying the fields as attributes. Which fields are
            required is the implementation's business, not the interface's:
            :class:`PngCropSource` needs ``png_path`` (or accepts a bare path
            string), :class:`MergedCropSource` needs the merged file and
            ``object_label``.
        """
        raise NotImplementedError

    def get_image(self, row: Any):
        """Return the crop for ``row`` as a PIL ``Image`` in RGB mode.

        :param row: as for :meth:`get`. PIL is imported inside this method,
            so a consumer that only ever wants arrays never pays for it.
        """
        from PIL import Image
        return Image.fromarray(self.get(row))

    def get_many(self, rows: Iterable[Any]) -> List[Optional[np.ndarray]]:
        """Return crops for many rows. Overridden by sources that can batch.

        :param rows: rows to crop. The result has one entry per row in the
            same order, so a caller can zip the two. This base implementation
            is a plain loop over :meth:`get` and therefore raises on the first
            row it cannot crop, and so does the one override shipped here,
            on :class:`MergedCropSource` -- despite the ``Optional`` in the
            return type, no implementation in this module ever puts ``None``
            in the list, so a caller need not test for it.
        """
        return [self.get(r) for r in rows]

    def describe(self) -> str:
        """Return a one-line description for logs / the GUI status bar."""
        return f"{self.kind} crop source ({self.reason})" if self.reason else f"{self.kind} crop source"


class PngCropSource(CropSource):
    """The existing behaviour: read the pre-generated PNG named by the row.

    Reads go through :func:`read_crop_png`, so a folder of legacy (format 1)
    crops is corrected on load and comes back in the same channel order as a
    new one -- the caller cannot tell which it opened, which is the point.

    :param root: optional experiment root used to re-anchor ``png_path`` values
        recorded on another machine (the same rewrite
        :func:`spacr.utils.correct_paths` performs).
    :param folder: the anchor folder name for that rewrite.
    :param reason: why this source was chosen (for :meth:`describe`).
    :param db_path: ``measurements.db`` consulted for the ``crop_format``
        column when a folder carries no sidecar; defaults to
        ``<root>/measurements/measurements.db`` when ``root`` is given.
    """

    kind = "png"

    def __init__(self, root: Optional[str] = None, folder: str = "data",
                 reason: str = "", db_path: Optional[str] = None):
        self.root = root
        self.folder = folder
        self.reason = reason
        if db_path is None and root:
            candidate = os.path.join(root, "measurements", "measurements.db")
            db_path = candidate if os.path.isfile(candidate) else None
        self.db_path = db_path

    def resolve(self, row: Any) -> str:
        """Return the on-disk PNG path for ``row``, re-anchored under ``root``.

        :param row: a row carrying ``png_path`` (or ``path``), or a bare path
            string, which is accepted as-is and only re-anchored. A row with
            neither raises :class:`CropError`. Re-anchoring is attempted only
            when ``root`` is set and is not already a substring of the path,
            and only when the path contains a literal ``/<folder>/`` segment;
            a path that does not is returned untouched even if it points
            nowhere on this machine, and the failure surfaces on read.
        """
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
        """Return the PNG for ``row`` decoded as a ``(H, W, 3)`` uint8 RGB array.

        Legacy content is converted on load, so this equals
        ``png_view(extract_crop(...))`` for the same object whichever format
        the folder is in.

        :param row: as for :meth:`resolve`. The folder's sidecar -- failing
            that, this source's ``db_path`` -- is what decides whether the
            file's channels are reversed on the way back, so the same row can
            legitimately give different pixels before and after a folder is
            marked or migrated.
        """
        return read_crop_png(self.resolve(row), db_path=self.db_path)


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
        """Return the merged ``.npy`` path for ``row``.

        The rowID -> well-letter step goes through :mod:`spacr.schema`, which
        is imported lazily *inside* this method on purpose: this module's
        contract is that importing it costs nothing (``tests/test_crops.py``
        loads it standalone, outside the package, and asserts the sys.modules
        delta is empty), and a module-scope relative import would break that
        probe. Nothing above this point needs schema.

        :param row: a measurement row. ``merged_path`` or ``path_name`` is
            used directly, and -- when that path does not exist here --
            retried as ``<merged_root>/<basename>``, which is how a database
            written on another machine still resolves. A path that exists
            nowhere is returned anyway, so the failure arrives later as
            :class:`MergedFileMissing`. With neither key the name is rebuilt
            from ``file_name``, or else from ``plateID`` / ``rowID`` /
            ``columnID`` / ``fieldID`` (all four required), and
            ``merged_root`` must be set or this raises.
        """
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
            from . import schema
            # chr(ord('A') + n - 1) walked straight off the end of the
            # alphabet: rowID 'r27' -- an ordinary 1536-plate row -- came back
            # as '[', and the rebuilt path pointed at a file that cannot
            # exist. schema.well_id is bijective base 26, so r27 is 'AA'.
            try:
                well = schema.well_id(rowid, colid)
                field = schema.field_index(fieldid)
                if field is None:
                    raise schema.KeyParseError(
                        f"fieldID {fieldid!r} holds no field number")
            except schema.SchemaError as exc:
                raise CropError(
                    f"row has no 'path_name' and its metadata does not name a "
                    f"field: {exc}") from exc
            stem = f"{plate}_{well}_{field}"
        return os.path.join(self.merged_root, f"{stem}.npy")

    def spec_for(self, row: Any) -> CropSpec:
        """Return the :class:`CropSpec` describing ``row``'s crop.

        :param row: a measurement row. The label is the first present of
            ``object_label``, ``label``, ``cell_id``, ``nucleus_id``,
            ``pathogen_id``, ``cytoplasm_id``, and a row with none of them
            raises :class:`CropError`; ``object_type``, if present,
            overrides the template spec's. ``bbox-0`` .. ``bbox-3`` (or
            ``bbox_0`` .. ``bbox_3``) are honoured only when all four are
            there, and are reordered out of the skimage ``regionprops``
            convention ``(min_row, min_col, max_row, max_col)`` into the
            spec's ``(y0, y1, x0, x1)`` -- supplying them lets the crop skip
            the whole-plane label index scan. The mask plane inside that box
            is still read, unless the spec also sets ``use_bounding_box``,
            which skips reading the label plane altogether.
        """
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
        """Return the raw crop (native dtype, ``spec.channels`` order).

        :param row: a measurement row; :meth:`spec_for` says which fields it
            has to carry. What comes back is the *pre-write* array -- the
            merged file's dtype (``uint16`` on a normal run) and as many
            channels as the spec selects -- not 8-bit RGB. Use :meth:`get`
            for something a viewer or a classifier can take.
        """
        spec = self.spec_for(row)
        return extract_crop(spec.merged_path, spec=spec)

    def get(self, row: Any) -> np.ndarray:
        """Return the crop as a ``(H, W, 3)`` uint8 RGB array.

        Deliberately routed through :func:`png_view`, so what a consumer gets
        here is identical to what :func:`read_crop_png` returns for the same
        object out of the PNG folder -- 16-bit narrowing included.

        :param row: a measurement row; :meth:`spec_for` says which fields it
            has to carry. Each row is resolved and cut on its own, so use
            :meth:`get_many` when filling a grid -- it opens each merged file
            once for the whole batch instead of once per row.
        """
        return png_view(self.get_array(row))

    def get_many(self, rows: Iterable[Any]) -> List[Optional[np.ndarray]]:
        """Return crops for many rows, opening each merged file only once.

        :param rows: rows to crop; :meth:`spec_for` says which fields each has
            to carry. The result has one entry per row in the original order,
            however the rows were regrouped internally -- they are bucketed by
            merged file so each ``.npy`` is memory-mapped and label-indexed
            once for the whole bucket. Every spec is built up front, so one
            row missing its label fails the batch before any file is opened.
        """
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


if __name__ == "__main__":
    raise SystemExit(main())
