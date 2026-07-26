"""Z-axis handling for the 3D (Beta) mask settings.

This module is the z half of spaCR's 3-D support. It is deliberately kept
free of Cellpose (and of any model at all) so that the z logic can be tested
against synthetic label volumes on a CPU in milliseconds: every function here
either takes a plain numpy array or takes a ``segment_fn`` callable that the
caller supplies. :mod:`spacr.object` provides the Cellpose adapter.

Four things drive the design, and each of them is a place where a naive 3-D
implementation silently produces wrong science:

**Anisotropy is the whole game.**
    Confocal z-spacing is routinely 3-10x the xy pixel size. A volume
    segmented as if it were isotropic merges objects along z, because a
    3-plane gap that is really 3 um reads to the segmenter as 3 pixels.
    ``anisotropy`` here always means ``dz / dxy`` -- the z step divided by the
    xy pixel size -- which is the same convention Cellpose uses. It is a
    *required* input for volumetric segmentation: :func:`resolve_anisotropy`
    raises :class:`UnknownAnisotropyError` rather than defaulting to 1.0,
    because 1.0 is a claim about the microscope, not a neutral value.

**Stitching 2-D planes is not 3-D segmentation.**
    Both are legitimate and they give different answers. Per-plane
    segmentation followed by IoU linking (:data:`MODE_STITCH`) sees each plane
    independently and cannot recover an object that is invisible in one plane;
    true volumetric segmentation (:data:`MODE_VOLUMETRIC`) uses the z gradient
    but is far more sensitive to a wrong anisotropy. The mode is therefore
    explicit, never inferred, and :class:`ZStackResult` records which one
    actually ran so it can be written next to the numbers it produced.

**Objects touching the first or last plane are truncated.**
    Exactly as an object touching the xy field edge is truncated. Their volume
    is a lower bound and their shape statistics are meaningless.
    :func:`flag_truncated_z` marks them, mirroring the reasoning
    ``spacr.seg_qc`` already established for xy border objects (see
    ``seg_qc._score_labels``, which counts border objects and then excludes
    them from every size statistic).

**A single-plane volume is 2-D, not degenerate 3-D.**
    ``n_z == 1`` short-circuits to :data:`MODE_SINGLE_PLANE`, which calls
    ``segment_fn`` on the plain 2-D plane and returns a 2-D mask. It does not
    resample, does not stitch, and does not consult anisotropy.

Memory
------
A z-stack is ``n_z`` times a field, and volumetric segmentation transiently
needs several copies of it. :func:`estimate_peak_bytes` gives the number for a
given field; the pipeline processes **one field at a time** and never holds a
plate. For a 2048x2048 field at 21 planes in float32 that is ~350 MB for the
volume, and ~1.4 GB peak in :data:`MODE_VOLUMETRIC` with resampling at
anisotropy 5 (the isotropic copy is ``anisotropy`` times taller). This is why
:data:`MODE_VOLUMETRIC` is not the default.

Scope, stated plainly
---------------------
The functions here are correct and tested, but the *pipeline* cannot yet feed
them: ``spacr.io._rename_and_organize_image_files`` collapses z before any
array reaches segmentation, and ``spacr.measure`` cannot consume a 3-D label
mask. :func:`plan_from_settings` therefore returns ``None`` whenever the 3D
settings are off -- which is the default -- so the 2-D path is untouched, and
:mod:`spacr.object` raises :class:`ZAxisNotPresentError` rather than quietly
projecting when the settings are on but no z axis survived ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigurationError

__all__ = [
    "MODE_PROJECT", "MODE_STITCH", "MODE_VOLUMETRIC", "MODE_SINGLE_PLANE",
    "SEGMENTATION_MODES", "PROJECTIONS", "VOLUME_STATS_UNITS",
    "ZStackError", "AmbiguousZAxisError", "UnknownAnisotropyError",
    "ZAxisNotPresentError",
    "ZStackSpec", "ZStackResult",
    "detect_z_axis", "resolve_anisotropy", "project", "resample_isotropic",
    "restore_anisotropic", "stitch_planes", "relabel_volume",
    "flag_truncated_z", "volume_stats", "segment_3d", "plan_from_settings",
    "estimate_peak_bytes",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Collapse z to one plane, then run the ordinary 2-D path. The only mode
#: whose output ``spacr.measure`` can consume today.
MODE_PROJECT = "project"

#: Segment every plane independently in 2-D, then link labels across z by IoU.
#: Anisotropy does not enter this mode at all -- neither here nor in Cellpose,
#: which ignores ``anisotropy`` whenever ``do_3D=False``.
MODE_STITCH = "stitch"

#: True volumetric segmentation. Anisotropy is mandatory.
MODE_VOLUMETRIC = "volumetric"

#: Not selectable; :func:`segment_3d` reports it when ``n_z == 1`` so that the
#: caller can see that the 2-D path ran.
MODE_SINGLE_PLANE = "single_plane"

#: The modes a user may ask for.
SEGMENTATION_MODES = (MODE_PROJECT, MODE_STITCH, MODE_VOLUMETRIC)

#: Projection reducers accepted by :func:`project`. ``None`` means "do not
#: project", which is what the stitch and volumetric modes use.
PROJECTIONS = ("max", "mean", "sum", "best_focus", None)

#: Units of every column :func:`volume_stats` produces. Volumes in a 3-D run
#: are voxel counts, not the px^2 areas a 2-D run writes, and the two must
#: never be compared without this table.
VOLUME_STATS_UNITS: Dict[str, str] = {
    "label": "index",
    "volume_voxels": "voxels",
    "volume_um3": "um^3",
    "surface_faces": "voxel faces",
    "surface_um2": "um^2",
    "z_min": "plane index",
    "z_max": "plane index",
    "z_extent_planes": "planes",
    "z_extent_um": "um",
    "centroid_z": "plane index",
    "centroid_y": "px",
    "centroid_x": "px",
    "truncated_z": "bool",
}

#: Flag name for an object cut off by the first or last z plane, named to match
#: ``seg_qc.FLAG_BORDER`` ("high_border_fraction") for its xy equivalent.
FLAG_Z_TRUNCATED = "z_truncated"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ZStackError(ConfigurationError):
    """Base class for every z-stack configuration problem.

    A :class:`spacr.errors.ConfigurationError`, so a run never continues past
    one: every field would be wrong in the same way.
    """


class AmbiguousZAxisError(ZStackError):
    """The z axis could not be identified and was not supplied.

    Raised only by ``detect_z_axis(..., strict=True)``. Guessing here picks
    between segmenting a volume and segmenting a transposed one, so the
    shape is reported back to the user instead.
    """


class UnknownAnisotropyError(ZStackError):
    """Volumetric segmentation was asked for without a z/xy voxel ratio.

    Defaulting to 1.0 would be a silent claim that the z step equals the xy
    pixel size, which for confocal data is wrong by 3-10x and merges objects
    along z.
    """


class ZAxisNotPresentError(ZStackError):
    """The 3D settings are on but the array that arrived has no z axis.

    Almost always because z was collapsed during ingest. The message names
    the setting to turn off and where the z went.
    """


# ---------------------------------------------------------------------------
# Spec / result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZStackSpec:
    """Everything the z plumbing needs to know about one acquisition.

    :param z_axis: Index of the z axis in the incoming array, or ``None`` to
        detect it with :func:`detect_z_axis`.
    :param n_z: Number of planes, or ``None`` when not yet known.
    :param anisotropy: ``dz / dxy``. ``None`` means "not known", which is
        fatal for :data:`MODE_VOLUMETRIC` and merely recorded otherwise.
    :param voxel_size_um: ``(dz, dy, dx)`` in micrometres, or ``None``.
    :param projection: Reducer used by :data:`MODE_PROJECT`.
    :param mode: One of :data:`SEGMENTATION_MODES`.
    :param stitch_threshold: IoU floor for :func:`stitch_planes`.
    :param resample_to_isotropic: Pre-resample the volume to isotropic voxels
        instead of handing ``anisotropy`` to the segmenter.
    """

    z_axis: Optional[int] = None
    n_z: Optional[int] = None
    anisotropy: Optional[float] = None
    voxel_size_um: Optional[Tuple[float, float, float]] = None
    projection: Optional[str] = "max"
    mode: str = MODE_PROJECT
    stitch_threshold: float = 0.25
    resample_to_isotropic: bool = False

    def __post_init__(self):
        if self.mode not in SEGMENTATION_MODES:
            raise ZStackError(
                f"z_segmentation_mode={self.mode!r} is not one of "
                f"{list(SEGMENTATION_MODES)}"
            )
        if self.projection not in PROJECTIONS:
            raise ZStackError(
                f"z_projection={self.projection!r} is not one of "
                f"{[p for p in PROJECTIONS]}"
            )
        if self.anisotropy is not None and self.anisotropy <= 0:
            raise ZStackError(
                f"anisotropy={self.anisotropy!r} must be > 0; it is the ratio "
                f"dz/dxy, so 1.0 means isotropic voxels and 5.0 means the z "
                f"step is five times the xy pixel size"
            )
        if not 0.0 <= self.stitch_threshold <= 1.0:
            raise ZStackError(
                f"stitch_threshold={self.stitch_threshold!r} must be an IoU "
                f"in [0, 1]"
            )

    def require_anisotropy(self) -> float:
        """Return the anisotropy, or explain why the run cannot proceed.

        :raises UnknownAnisotropyError: when it is not known.
        """
        return resolve_anisotropy(
            anisotropy=self.anisotropy, voxel_size_um=self.voxel_size_um
        )


@dataclass
class ZStackResult:
    """The labels a z-aware run produced, plus how it produced them.

    ``mode`` is recorded rather than inferred because a stitched volume and a
    volumetric one are different measurements of the same sample, and a number
    without its mode cannot be compared with anything.

    :param labels: ``(Z, Y, X)`` for stitch/volumetric, ``(Y, X)`` for project
        and single-plane.
    :param mode: The mode that actually ran -- may be
        :data:`MODE_SINGLE_PLANE` even though that is not selectable.
    :param anisotropy: The value used, or ``None`` when the mode ignores it.
    :param n_z: Planes in the input volume.
    :param truncated_labels: Labels touching the first or last plane.
    :param notes: Human-readable remarks worth surfacing to the user, e.g.
        that anisotropy was supplied but the chosen mode ignores it.
    """

    labels: np.ndarray
    mode: str
    anisotropy: Optional[float] = None
    n_z: int = 1
    truncated_labels: np.ndarray = _dc_field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    notes: List[str] = _dc_field(default_factory=list)

    @property
    def n_objects(self) -> int:
        """Number of distinct non-background labels."""
        if self.labels.size == 0:
            return 0
        return int(np.count_nonzero(np.unique(self.labels)))

    @property
    def truncated_fraction(self) -> float:
        """Share of objects cut off by the first or last plane.

        The z equivalent of ``seg_qc``'s ``border_fraction``; like it, a high
        value means the stack does not span the objects and their volumes are
        lower bounds.
        """
        n = self.n_objects
        if n == 0:
            return 0.0
        return float(self.truncated_labels.size) / n


# ---------------------------------------------------------------------------
# Axis detection
# ---------------------------------------------------------------------------

def detect_z_axis(array, xy_min: int = 32, strict: bool = False) -> Optional[int]:
    """Identify which axis of a 3-D array is z.

    Two rules, applied in order, both of which encode the same fact: an image
    plane is big and a z stack is short.

    1. If exactly one axis is smaller than ``xy_min`` while the other two are
       not, that axis is z. This catches ``(21, 512, 512)`` and
       ``(512, 512, 21)``.
    2. Otherwise, if two axes are equal and the third is *smaller*, the third
       is z. This catches ``(100, 512, 512)``, where no axis is short in
       absolute terms.

    Anything else is ambiguous and is reported as such -- ``(64, 64, 64)`` and
    ``(10, 20, 512)`` return ``None``. Guessing would silently transpose the
    volume, so the caller must supply ``z_axis`` instead.

    :param array: A 3-D array, or a shape tuple.
    :param xy_min: Smallest side length still considered an image axis.
    :param strict: Raise instead of returning ``None`` when ambiguous.
    :returns: The z axis index, or ``None`` when it cannot be determined.
    :raises ValueError: when the input is not 3-D.
    :raises AmbiguousZAxisError: when ``strict`` and the axis is ambiguous.
    """
    shape = tuple(array) if isinstance(array, (tuple, list)) else tuple(np.shape(array))

    if len(shape) != 3:
        raise ValueError(
            f"detect_z_axis expects a 3-D array (Z,Y,X) or (Y,X,Z); got shape "
            f"{shape}. A 4-D (Z,Y,X,C) array must have z_axis given explicitly, "
            f"exactly as Cellpose requires for 4-D input."
        )

    short = [i for i, s in enumerate(shape) if s < xy_min]
    if len(short) == 1:
        return short[0]

    if not short:
        for candidate in range(3):
            others = [shape[i] for i in range(3) if i != candidate]
            if others[0] == others[1] and shape[candidate] < others[0]:
                return candidate

    if strict:
        raise AmbiguousZAxisError(
            f"cannot tell which axis of shape {shape} is z: "
            f"{'more than one axis is short' if len(short) > 1 else 'no axis stands out as shorter than the other two'}. "
            f"Set the `z_axis` setting explicitly (0, 1 or 2) rather than "
            f"letting spaCR guess -- guessing wrong segments a transposed volume."
        )
    return None


def _as_z_first(volume: np.ndarray, z_axis: Optional[int]) -> Tuple[np.ndarray, int]:
    """Move ``z_axis`` to position 0, detecting it when not given.

    :returns: ``(z-first view, the resolved z axis of the input)``.
    """
    volume = np.asarray(volume)
    if volume.ndim < 3:
        raise ValueError(
            f"expected a volume with at least 3 axes, got ndim={volume.ndim} "
            f"(shape {volume.shape})"
        )

    if z_axis is None:
        if volume.ndim != 3:
            raise AmbiguousZAxisError(
                f"z_axis must be given for a {volume.ndim}-D array (shape "
                f"{volume.shape}); it is only detectable for a plain 3-D volume"
            )
        z_axis = detect_z_axis(volume, strict=True)

    z_axis = int(z_axis) % volume.ndim
    return np.moveaxis(volume, z_axis, 0), z_axis


# ---------------------------------------------------------------------------
# Anisotropy
# ---------------------------------------------------------------------------

def resolve_anisotropy(
    anisotropy: Optional[float] = None,
    voxel_size_um: Optional[Sequence[float]] = None,
) -> float:
    """Return ``dz / dxy``, derived from the voxel size when available.

    An explicit ``anisotropy`` wins. Otherwise it is computed from
    ``voxel_size_um = (dz, dy, dx)`` as ``dz / mean(dy, dx)``. If neither is
    known the run stops: there is no safe default, because assuming 1.0 claims
    the z step equals the xy pixel size and, when that is wrong by the usual
    3-10x, merges every object along z.

    :param anisotropy: Explicit ratio, or ``None``.
    :param voxel_size_um: ``(dz, dy, dx)`` in micrometres, or ``None``.
    :returns: The anisotropy as a float.
    :raises UnknownAnisotropyError: when neither input determines it.
    :raises ZStackError: when the values given are not usable.
    """
    if anisotropy is not None:
        value = float(anisotropy)
        if value <= 0 or not np.isfinite(value):
            raise ZStackError(
                f"anisotropy={anisotropy!r} must be a finite number > 0"
            )
        return value

    if voxel_size_um is not None:
        dz, dy, dx = (float(v) for v in voxel_size_um)
        if not all(np.isfinite(v) and v > 0 for v in (dz, dy, dx)):
            raise ZStackError(
                f"voxel_size_um={tuple(voxel_size_um)!r} must be three finite "
                f"positive numbers (dz, dy, dx) in micrometres"
            )
        return dz / ((dy + dx) / 2.0)

    raise UnknownAnisotropyError(
        "3-D segmentation needs the z/xy voxel ratio and this run does not "
        "know it. Set `anisotropy` directly (dz / dxy), or set "
        "`voxel_size_z_um` and `voxel_size_xy_um` and let spaCR derive it. "
        "spaCR will not assume 1.0: on a confocal stack the z step is "
        "typically 3-10x the xy pixel size, and segmenting such a volume as "
        "if it were isotropic fuses objects that are separated in z."
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _focus_scores(planes: np.ndarray) -> np.ndarray:
    """Per-plane focus metric: variance of the Laplacian, the usual proxy."""
    from scipy.ndimage import laplace

    return np.asarray(
        [float(np.var(laplace(p.astype(np.float64)))) for p in planes]
    )


def project(volume, mode: Optional[str] = "max", z_axis: Optional[int] = 0):
    """Collapse the z axis of ``volume`` to a single plane.

    :param volume: Volume with a z axis; extra trailing axes (channels) are
        preserved.
    :param mode: ``'max'``, ``'mean'``, ``'sum'``, ``'best_focus'`` or
        ``None``. ``None`` returns the volume untouched, which is what the
        stitch and volumetric modes want.
    :param z_axis: Which axis is z; detected when ``None``.
    :returns: The projected array, one axis smaller than the input -- or the
        input itself when ``mode`` is ``None``.
    :raises ZStackError: on an unknown ``mode``.
    """
    if mode is None:
        return np.asarray(volume)

    if mode not in PROJECTIONS:
        raise ZStackError(
            f"z_projection={mode!r} is not one of "
            f"{[p for p in PROJECTIONS if p is not None]} or None"
        )

    vol, _ = _as_z_first(volume, z_axis)

    if mode == "max":
        return vol.max(axis=0)
    if mode == "mean":
        return vol.mean(axis=0)
    if mode == "sum":
        return vol.sum(axis=0)

    # best_focus: keep the sharpest single plane rather than blending planes,
    # which is what you want when only one plane is actually in focus and a
    # MIP would drag every out-of-focus plane's haze into the result.
    if vol.shape[0] == 1:
        return vol[0]
    scores = _focus_scores(vol if vol.ndim == 3 else vol.max(axis=-1))
    return vol[int(np.argmax(scores))]


# ---------------------------------------------------------------------------
# Isotropic resampling
# ---------------------------------------------------------------------------

def resample_isotropic(volume, anisotropy: float, z_axis: Optional[int] = 0,
                       order: int = 1) -> np.ndarray:
    """Stretch the z axis by ``anisotropy`` so the voxels become cubic.

    Segmenters that reason about distance -- anything doing a morphological
    operation, a watershed or a flow field -- need cubic voxels or they will
    treat a 5 um z step as if it were a 1 um xy step. Cellpose does this
    internally when ``do_3D=True``; this function exists for every other
    segmenter, and for making the effect visible in tests.

    :param volume: ``(Z, Y, X)`` volume (or with z at ``z_axis``).
    :param anisotropy: ``dz / dxy``. 1.0 returns the input unchanged.
    :param z_axis: Which axis is z.
    :param order: Interpolation order; use ``0`` for label images.
    :returns: A z-first array with ``round(Z * anisotropy)`` planes.
    """
    from skimage.transform import resize

    vol, _ = _as_z_first(volume, z_axis)
    if anisotropy == 1.0:
        return vol

    new_z = max(1, int(round(vol.shape[0] * float(anisotropy))))
    out = resize(
        vol, (new_z,) + vol.shape[1:], order=order,
        preserve_range=True, anti_aliasing=False,
    )
    return out.astype(vol.dtype)


def restore_anisotropic(volume, n_z: int, order: int = 0) -> np.ndarray:
    """Undo :func:`resample_isotropic`, back to ``n_z`` planes.

    :param volume: z-first array to shrink.
    :param n_z: Plane count of the original acquisition.
    :param order: Interpolation order; ``0`` (nearest) keeps label values
        intact and is the default because this is normally called on masks.
    :returns: A z-first array with exactly ``n_z`` planes.
    """
    from skimage.transform import resize

    vol = np.asarray(volume)
    if vol.shape[0] == n_z:
        return vol

    out = resize(
        vol, (n_z,) + vol.shape[1:], order=order,
        preserve_range=True, anti_aliasing=False,
    )
    return out.astype(vol.dtype)


# ---------------------------------------------------------------------------
# Cross-plane linking
# ---------------------------------------------------------------------------

def _plane_iou(prev_plane: np.ndarray, cur_plane: np.ndarray):
    """IoU between every label pair of two planes.

    :returns: ``(prev_ids, cur_ids, iou_matrix)`` with background excluded.
    """
    prev_ids = np.unique(prev_plane)
    prev_ids = prev_ids[prev_ids > 0]
    cur_ids = np.unique(cur_plane)
    cur_ids = cur_ids[cur_ids > 0]

    if prev_ids.size == 0 or cur_ids.size == 0:
        return prev_ids, cur_ids, np.zeros((prev_ids.size, cur_ids.size), float)

    # searchsorted rather than a dict lookup per pixel: a field can carry
    # hundreds of objects and this runs once per plane pair.
    overlap = np.zeros((prev_ids.size, cur_ids.size), dtype=np.int64)
    both = (prev_plane > 0) & (cur_plane > 0)
    if np.any(both):
        rows = np.searchsorted(prev_ids, prev_plane[both])
        cols = np.searchsorted(cur_ids, cur_plane[both])
        counts = np.bincount(
            rows * cur_ids.size + cols,
            minlength=prev_ids.size * cur_ids.size,
        )
        overlap = counts.reshape(prev_ids.size, cur_ids.size)

    prev_counts = np.bincount(prev_plane.ravel())
    cur_counts = np.bincount(cur_plane.ravel())
    prev_area = prev_counts[prev_ids].astype(np.int64)
    cur_area = cur_counts[cur_ids].astype(np.int64)
    union = prev_area[:, None] + cur_area[None, :] - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, overlap / union, 0.0)
    return prev_ids, cur_ids, iou


def stitch_planes(masks_2d_stack, iou_threshold: float = 0.25) -> np.ndarray:
    """Link per-plane 2-D labels into 3-D objects by IoU between neighbours.

    Each plane is compared with the plane below it, and the pairing is greedy
    and **one-to-one**: pairs are considered in descending IoU and a label may
    take part in at most one link. That is what stops two genuinely different
    objects that merely happen to overlap the same object in xy from being
    fused -- only the better match inherits the label, the other starts a new
    one. Any pair below ``iou_threshold`` is not linked at all.

    Unlike ``cellpose.utils.stitch3D``, new labels are always drawn from a
    single monotonically increasing counter. Cellpose resets its counter after
    an empty plane, so ``[objects] [empty] [objects]`` there silently reuses
    label ids and merges unrelated objects; here an empty plane simply links
    nothing.

    :param masks_2d_stack: ``(Z, Y, X)`` array, or a sequence of 2-D arrays,
        of per-plane label images whose label values need not be unique
        across planes.
    :param iou_threshold: Minimum IoU for two labels to be the same object.
    :returns: A ``(Z, Y, X)`` array of contiguously numbered 3-D labels.
    :raises ZStackError: when ``iou_threshold`` is outside ``[0, 1]``.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ZStackError(
            f"stitch_threshold={iou_threshold!r} must be an IoU in [0, 1]"
        )

    if isinstance(masks_2d_stack, np.ndarray):
        planes = [masks_2d_stack[i] for i in range(masks_2d_stack.shape[0])]
    else:
        planes = [np.asarray(p) for p in masks_2d_stack]

    if not planes:
        raise ValueError("stitch_planes got an empty stack")

    out = np.zeros((len(planes),) + np.asarray(planes[0]).shape, dtype=np.int64)
    next_label = 1

    # Plane 0: every object starts a new 3-D object. Relabelling goes through
    # a lookup table so the cost is one pass over the plane, not one pass per
    # label.
    first = np.asarray(planes[0])
    first_ids = np.unique(first)
    first_ids = first_ids[first_ids > 0]
    if first_ids.size:
        lut = np.zeros(int(first.max()) + 1, dtype=np.int64)
        lut[first_ids] = np.arange(1, first_ids.size + 1, dtype=np.int64)
        out[0] = lut[first]
        next_label += first_ids.size

    for z in range(1, len(planes)):
        cur = np.asarray(planes[z])
        prev_ids, cur_ids, iou = _plane_iou(out[z - 1], cur)

        assigned: Dict[int, int] = {}
        if iou.size:
            order = np.argsort(iou, axis=None)[::-1]
            used_prev, used_cur = set(), set()
            for flat in order:
                i, j = divmod(int(flat), cur_ids.size)
                if iou[i, j] < iou_threshold or iou[i, j] <= 0:
                    break
                if i in used_prev or j in used_cur:
                    continue
                used_prev.add(i)
                used_cur.add(j)
                assigned[int(cur_ids[j])] = int(prev_ids[i])

        if cur_ids.size:
            lut = np.zeros(int(cur.max()) + 1, dtype=np.int64)
            for value in cur_ids:
                value = int(value)
                if value in assigned:
                    lut[value] = assigned[value]
                else:
                    lut[value] = next_label
                    next_label += 1
            out[z] = lut[cur]

    return relabel_volume(out)


def relabel_volume(labels) -> np.ndarray:
    """Renumber labels to a contiguous ``1..N`` with no gaps, background 0.

    Filtering objects out of a volume leaves holes in the numbering, and a
    consumer that sizes an array by ``labels.max()`` then over-allocates or
    indexes past the end.

    :param labels: Integer label array of any shape.
    :returns: A relabelled array of the same shape and dtype.
    """
    labels = np.asarray(labels)
    present = np.unique(labels)
    present = present[present > 0]

    if present.size == 0:
        return np.zeros_like(labels)

    lookup = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    lookup[present] = np.arange(1, present.size + 1, dtype=labels.dtype)
    return lookup[labels]


# ---------------------------------------------------------------------------
# Truncation at the ends of the stack
# ---------------------------------------------------------------------------

def flag_truncated_z(labels) -> np.ndarray:
    """Labels that touch the first or last z plane, and are therefore cut off.

    The z counterpart of ``seg_qc``'s border-object rule: an object clipped by
    the end of the stack has a volume that is only a lower bound and shape
    statistics that describe the visible part, so it should be reported and
    excluded from size distributions rather than quietly averaged in.

    :param labels: ``(Z, Y, X)`` label volume. A 2-D array has no z extent and
        returns an empty array.
    :returns: Sorted array of truncated label ids.
    """
    labels = np.asarray(labels)
    if labels.ndim < 3:
        return np.empty(0, dtype=np.int64)

    ends = np.concatenate([labels[0].ravel(), labels[-1].ravel()])
    ids = np.unique(ends)
    return ids[ids > 0].astype(np.int64)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _surface_faces_per_label(labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Exposed voxel faces per label, split by the axis they face along.

    :returns: ``(3, n_labels + 1)`` counts, indexed ``[axis, label]``.
    """
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    faces = np.zeros((3, n_labels + 1), dtype=np.int64)

    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        left = padded[tuple(lo)]
        right = padded[tuple(hi)]
        differ = left != right
        if not np.any(differ):
            continue
        faces[axis] += np.bincount(
            left[differ].ravel(), minlength=n_labels + 1
        )[: n_labels + 1]
        faces[axis] += np.bincount(
            right[differ].ravel(), minlength=n_labels + 1
        )[: n_labels + 1]

    return faces


def volume_stats(labels, voxel_size: Optional[Sequence[float]] = None):
    """Per-object volume, surface, z extent and truncation flag.

    Every column's unit is in :data:`VOLUME_STATS_UNITS`, and the names carry
    the unit too, because a 3-D run produces *voxel counts* where a 2-D run
    produces px^2 areas. Writing one into a column named for the other is the
    single easiest way to silently corrupt a screen.

    With anisotropic voxels the physical numbers differ from the voxel counts
    by more than a constant: the surface area weights each face by the area of
    the face, so a z-facing face (``dy*dx``) and an x-facing face (``dz*dy``)
    contribute differently. Passing ``voxel_size=None`` gives the voxel-count
    columns only.

    :param labels: ``(Z, Y, X)`` label volume.
    :param voxel_size: ``(dz, dy, dx)`` in micrometres, or ``None`` to skip
        the physical columns.
    :returns: A :class:`pandas.DataFrame`, one row per label, sorted by label.
    :raises ValueError: when ``labels`` is not 3-D.
    """
    import pandas as pd

    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(
            f"volume_stats expects a 3-D (Z,Y,X) label volume, got shape "
            f"{labels.shape}. Use skimage.measure.regionprops for 2-D masks."
        )

    present = np.unique(labels)
    present = present[present > 0]

    columns = list(VOLUME_STATS_UNITS)
    if voxel_size is None:
        columns = [c for c in columns if not c.endswith(("_um", "_um2", "_um3"))]

    if present.size == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})

    n_labels = int(labels.max())
    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    faces = _surface_faces_per_label(labels, n_labels)
    truncated = set(flag_truncated_z(labels).tolist())

    zz, yy, xx = np.nonzero(labels)
    flat = labels[zz, yy, xx]
    sum_z = np.bincount(flat, weights=zz, minlength=n_labels + 1)
    sum_y = np.bincount(flat, weights=yy, minlength=n_labels + 1)
    sum_x = np.bincount(flat, weights=xx, minlength=n_labels + 1)

    # Seeded with the infinities rather than NaN: np.minimum propagates NaN,
    # so a NaN seed would leave every extent undefined.
    z_min = np.full(n_labels + 1, np.inf)
    z_max = np.full(n_labels + 1, -np.inf)
    np.minimum.at(z_min, flat, zz)
    np.maximum.at(z_max, flat, zz)

    if voxel_size is not None:
        dz, dy, dx = (float(v) for v in voxel_size)
        voxel_volume = dz * dy * dx
        face_area = (dy * dx, dz * dx, dz * dy)
    else:
        dz = voxel_volume = None
        face_area = None

    rows = []
    for value in present:
        value = int(value)
        row = {
            "label": value,
            "volume_voxels": int(counts[value]),
            "surface_faces": int(faces[:, value].sum()),
            "z_min": int(z_min[value]),
            "z_max": int(z_max[value]),
            "z_extent_planes": int(z_max[value] - z_min[value]) + 1,
            "centroid_z": sum_z[value] / counts[value],
            "centroid_y": sum_y[value] / counts[value],
            "centroid_x": sum_x[value] / counts[value],
            "truncated_z": value in truncated,
        }
        if voxel_size is not None:
            row["volume_um3"] = counts[value] * voxel_volume
            row["surface_um2"] = float(
                sum(faces[a, value] * face_area[a] for a in range(3))
            )
            row["z_extent_um"] = row["z_extent_planes"] * dz
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Segmentation driver
# ---------------------------------------------------------------------------

def segment_3d(
    volume,
    segment_fn: Callable[..., Any],
    mode: str = MODE_PROJECT,
    stitch_threshold: float = 0.25,
    anisotropy: Optional[float] = None,
    voxel_size_um: Optional[Sequence[float]] = None,
    projection: Optional[str] = "max",
    z_axis: Optional[int] = 0,
    resample_to_isotropic: bool = False,
) -> ZStackResult:
    """Segment a volume in the requested mode and record how it was done.

    ``segment_fn`` is the only model-aware part and is supplied by the caller.
    It is called as ``segment_fn(array, **kwargs)`` and must return a label
    array matching ``array``'s spatial shape. The kwargs it receives depend on
    the mode, mirroring what Cellpose accepts:

    * :data:`MODE_PROJECT` -- ``segment_fn(plane_2d)``, no kwargs.
    * :data:`MODE_STITCH` -- ``segment_fn(volume, stitch=True)``; the result
      may be per-plane labels, which are then linked by :func:`stitch_planes`.
      Anisotropy is *not* passed, because neither this code nor Cellpose uses
      it without ``do_3D``; supplying it is recorded as a no-op in ``notes``.
    * :data:`MODE_VOLUMETRIC` -- ``segment_fn(volume, do_3D=True,
      anisotropy=..., z_axis=0)``, or, when ``resample_to_isotropic``, the
      volume is stretched first and ``anisotropy=1.0`` is passed instead.

    ``n_z == 1`` overrides everything: the single plane goes to
    ``segment_fn`` on its own and a 2-D mask comes back, identical to the
    ordinary 2-D path.

    :param volume: The image volume, z at ``z_axis``.
    :param segment_fn: Callable performing the actual segmentation.
    :param mode: One of :data:`SEGMENTATION_MODES`.
    :param stitch_threshold: IoU floor used by :data:`MODE_STITCH`.
    :param anisotropy: ``dz / dxy``; required by :data:`MODE_VOLUMETRIC`.
    :param voxel_size_um: ``(dz, dy, dx)``, used to derive ``anisotropy``.
    :param projection: Reducer for :data:`MODE_PROJECT`.
    :param z_axis: Which axis of ``volume`` is z.
    :param resample_to_isotropic: Stretch z before segmenting rather than
        delegating anisotropy handling to ``segment_fn``.
    :returns: A :class:`ZStackResult`.
    :raises ZStackError: on an unknown mode.
    :raises UnknownAnisotropyError: for :data:`MODE_VOLUMETRIC` without a
        known anisotropy.
    """
    if mode not in SEGMENTATION_MODES:
        raise ZStackError(
            f"z_segmentation_mode={mode!r} is not one of "
            f"{list(SEGMENTATION_MODES)}"
        )

    vol, _ = _as_z_first(volume, z_axis)
    n_z = int(vol.shape[0])
    notes: List[str] = []

    # A single plane is 2-D. Not a degenerate volume, not a 1-plane stitch --
    # the ordinary path, returning an ordinary 2-D mask.
    if n_z == 1:
        labels = np.asarray(segment_fn(vol[0]))
        return ZStackResult(
            labels=labels, mode=MODE_SINGLE_PLANE, anisotropy=None, n_z=1,
            notes=["single z plane: segmented in 2-D, exactly as a non-z run"],
        )

    if mode == MODE_PROJECT:
        flat = project(vol, mode=projection, z_axis=0)
        labels = np.asarray(segment_fn(flat))
        notes.append(
            f"z collapsed by '{projection}' projection before segmentation: "
            f"{n_z} planes -> 1. Object volumes are not measurable from this."
        )
        return ZStackResult(
            labels=labels, mode=MODE_PROJECT, anisotropy=None, n_z=n_z,
            notes=notes,
        )

    if mode == MODE_STITCH:
        if anisotropy is not None or voxel_size_um is not None:
            notes.append(
                "anisotropy is ignored in stitch mode -- planes are segmented "
                "independently in 2-D, so no z distance is ever computed. "
                "Cellpose ignores it here too."
            )
        raw = np.asarray(segment_fn(vol, stitch=True))
        labels = stitch_planes(raw, iou_threshold=stitch_threshold)
        notes.append(
            f"{n_z} planes segmented independently in 2-D, then linked across "
            f"z at IoU >= {stitch_threshold}. This is not volumetric "
            f"segmentation and will differ from it."
        )
    else:  # MODE_VOLUMETRIC
        aniso = resolve_anisotropy(anisotropy, voxel_size_um)
        if resample_to_isotropic:
            iso = resample_isotropic(vol, aniso, z_axis=0, order=1)
            notes.append(
                f"volume resampled to isotropic voxels before segmentation: "
                f"{n_z} -> {iso.shape[0]} planes at anisotropy {aniso:g}"
            )
            raw = np.asarray(segment_fn(iso, do_3D=True, anisotropy=1.0, z_axis=0))
            labels = restore_anisotropic(raw, n_z, order=0)
        else:
            labels = np.asarray(
                segment_fn(vol, do_3D=True, anisotropy=aniso, z_axis=0)
            )
            notes.append(
                f"volumetric segmentation at anisotropy {aniso:g} "
                f"(dz/dxy); the segmenter rescales z internally"
            )
        labels = relabel_volume(labels)
        anisotropy = aniso

    truncated = flag_truncated_z(labels)
    if truncated.size:
        notes.append(
            f"{truncated.size} object(s) touch the first or last z plane and "
            f"are truncated: their volumes are lower bounds, exactly as for "
            f"objects touching the xy field edge"
        )

    return ZStackResult(
        labels=labels,
        mode=mode,
        anisotropy=anisotropy if mode == MODE_VOLUMETRIC else None,
        n_z=n_z,
        truncated_labels=truncated,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Settings bridge
# ---------------------------------------------------------------------------

def plan_from_settings(settings) -> Optional[ZStackSpec]:
    """Build a :class:`ZStackSpec` from a settings dict, or ``None`` when off.

    Returning ``None`` is the contract that keeps the 2-D path bit-identical:
    every caller branches on ``spec is None`` and, when it is, does not touch
    any z code at all. ``z_stack`` absent and ``z_stack=False`` are the same
    thing.

    :param settings: The pipeline settings dict.
    :returns: A spec, or ``None`` when 3D handling is off.
    :raises ZStackError: when 3D is on but the settings are self-inconsistent.
    """
    if not settings.get("z_stack", False):
        return None

    dz = settings.get("voxel_size_z_um")
    dxy = settings.get("voxel_size_xy_um")
    voxel_size = None
    if dz is not None and dxy is not None:
        voxel_size = (float(dz), float(dxy), float(dxy))

    mode = settings.get("z_segmentation_mode", MODE_PROJECT)
    spec = ZStackSpec(
        z_axis=settings.get("z_axis"),
        n_z=None,
        anisotropy=settings.get("anisotropy"),
        voxel_size_um=voxel_size,
        projection=settings.get("z_projection", "max"),
        mode=mode,
        stitch_threshold=float(settings.get("stitch_threshold", 0.25) or 0.0),
        # Cellpose does its own z rescaling under do_3D, so the pipeline hands
        # it `anisotropy` rather than pre-stretching the volume. Direct API
        # callers with a segmenter that does not can set this on the spec.
        resample_to_isotropic=False,
    )

    # Fail here rather than after the model has been loaded and the first
    # field read: the answer cannot change later in the run.
    if spec.mode == MODE_VOLUMETRIC:
        spec.require_anisotropy()

    return spec


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def estimate_peak_bytes(volume_shape: Sequence[int], dtype=np.float32,
                        mode: str = MODE_PROJECT,
                        anisotropy: float = 1.0) -> int:
    """Peak bytes one field needs, so a plate can be sized before it is run.

    spaCR processes one field at a time and never holds a plate, so this is
    the number that matters. The multipliers are the live copies each mode
    keeps: the input volume, the segmenter's float working copy, and the
    label output.

    :param volume_shape: ``(Z, Y, X)`` or ``(Z, Y, X, C)``.
    :param dtype: Image dtype.
    :param mode: One of :data:`SEGMENTATION_MODES`.
    :param anisotropy: Used only when ``mode`` is :data:`MODE_VOLUMETRIC`,
        where the isotropic copy is ``anisotropy`` times taller.
    :returns: Estimated peak bytes.
    """
    itemsize = np.dtype(dtype).itemsize
    n_voxels = int(np.prod(list(volume_shape)))
    volume_bytes = n_voxels * itemsize

    if mode == MODE_PROJECT:
        # The volume plus one plane; z is gone immediately.
        planes = int(volume_shape[0]) if len(volume_shape) else 1
        return volume_bytes + (volume_bytes // max(planes, 1))
    if mode == MODE_STITCH:
        # Volume + per-plane labels + the stitched int64 output.
        return volume_bytes + n_voxels * 4 + n_voxels * 8
    # Volumetric: the isotropic copy dominates, plus a 3-component flow field.
    iso = int(volume_bytes * max(anisotropy, 1.0))
    return volume_bytes + iso + iso * 3 + n_voxels * 8
