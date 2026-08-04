"""Z- and time-axis handling for the 3D (Beta) and 4D (Beta) mask settings.

This module holds the two non-xy axes: the **z half** (3D Beta, the first
part of the file) and the **t half** built on top of it (4D Beta, the second
part, from the ``4D (Beta)`` banner onwards -- it has its own long preamble
there). Both are deliberately kept free of Cellpose, of any tracker library
and of any model at all, so that the logic can be tested against synthetic
label volumes on a CPU in milliseconds: every function here either takes a
plain numpy array or takes a ``segment_fn`` callable that the caller
supplies. :mod:`spacr.object` provides the Cellpose adapter.

The two halves live in one file rather than two because a 4-D acquisition is
a z-stack per timepoint and the t code delegates to the z code for every one
of them -- ``segment_4d`` is a loop around :func:`segment_3d`, ``track_4d``'s
overlap backend *is* :func:`stitch_planes` applied along t, and ``TStackSpec``
carries a :class:`ZStackSpec` inside it. Nothing in the z half below knows the
t half exists, so a 3-D run is unaffected by any of it.

Four things drive the design of the z half, and each of them is a place where
a naive 3-D implementation silently produces wrong science:

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
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple)

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
    "MEASUREMENT_MEANING_3D", "MEASUREMENT_ADDED_3D",
    "MEASUREMENT_UNAVAILABLE_3D", "describe_3d_measurement",
    "report_3d_measurements",
    # --- 4D (Beta): t on top of z ---
    "AXIS_ORDER_TZYX", "AXIS_ORDER_ZTYX", "AXIS_ORDER_TYX", "AXIS_ORDERS",
    "BACKEND_IOU",
    "BACKEND_CENTROID", "BACKEND_TRACKPY", "BACKEND_BTRACK",
    "BACKEND_TRACKASTRA", "BACKEND_ULTRACK", "TRACK_BACKENDS",
    "TRACK_COLUMN_UNITS", "BASE_TRACK_COLUMNS", "FLAG_T_TRUNCATED",
    "TStackError", "AmbiguousAxisOrderError", "TrackerIsTwoDError",
    "TAxisNotPresentError", "UnknownDisplacementError", "AxisOrder",
    "TrackBackend", "TStackSpec", "TStackResult", "TrackResult", "detect_axes",
    "resolve_axis_order", "as_t_first", "iter_volumes", "segment_4d",
    "track_4d", "volume_tracks", "flag_truncated_t", "project_labels",
    "format_4d", "plan_4d_from_settings", "estimate_peak_bytes_4d",
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


#: What each morphology measurement *means* once the object is a volume.
#:
#: ``spacr.measure`` already drops the two properties skimage refuses to
#: compute in 3-D (``PROPS_2D_ONLY``: ``perimeter`` and ``eccentricity``), and
#: it already stamps every row with ``measurement_ndim`` and
#: ``measurement_units``. What neither of those says is that **some columns
#: keep their name and change their meaning**, which is the failure mode that
#: survives a code review: a 3-D run writes ``cell_area``, and a reader who
#: knows that column holds px^2 gets um^3 without one thing looking wrong.
#: Anything that pools 2-D and 3-D runs, or reuses a threshold fitted on one,
#: needs this table and not just the stamp.
#:
#: Keys are the bare regionprops names, as they appear after the object-type
#: prefix (``cell_``, ``nucleus_``, ...).
#:
#: ``kind`` is one of:
#:
#: * ``"same"`` -- dimensionless, or otherwise identical in meaning.
#: * ``"renamed"`` -- computed and correct, but the name now describes a
#:   different quantity in different units. These are the dangerous ones.
#: * ``"absent"`` -- not written at all by a 3-D run.
MEASUREMENT_MEANING_3D: Dict[str, Dict[str, str]] = {
    "area": {
        "kind": "renamed",
        "means": "volume",
        "units_2d": "px^2",
        "units_3d": "um^3 (voxels when no voxel size is known)",
        "note": (
            "The same number is also written as volume_um3 / volume_voxels, "
            "which is the column to read. area is kept only so a 2-D "
            "downstream query does not break."
        ),
    },
    "area_filled": {
        "kind": "renamed", "means": "filled volume",
        "units_2d": "px^2", "units_3d": "um^3",
        "note": "A cavity enclosed in z counts as filled, not just an xy hole.",
    },
    "area_bbox": {
        "kind": "renamed", "means": "bounding-box volume",
        "units_2d": "px^2", "units_3d": "um^3",
        "note": "The box now has a z extent; a flat object's box is thin.",
    },
    "convex_area": {
        "kind": "renamed", "means": "convex-hull volume",
        "units_2d": "px^2", "units_3d": "um^3",
        "note": (
            "NaN for an object that does not span all three axes -- a "
            "single-plane label has no 3-D hull, and Qhull's answer for one "
            "is not a volume."
        ),
    },
    "equivalent_diameter_area": {
        "kind": "renamed", "means": "diameter of the equal-VOLUME sphere",
        "units_2d": "px", "units_3d": "um",
        "note": (
            "Still a length, so it looks comparable across a 2-D and a 3-D "
            "run. It is not: one is the diameter of a disc of equal area."
        ),
    },
    "solidity": {
        "kind": "same", "means": "volume / convex-hull volume",
        "units_2d": "ratio", "units_3d": "ratio",
        "note": "NaN wherever convex_area is NaN.",
    },
    "extent": {
        "kind": "same", "means": "volume / bounding-box volume",
        "units_2d": "ratio", "units_3d": "ratio",
        "note": (
            "Systematically lower in 3-D than in 2-D for the same object, "
            "because the bounding box gains a whole extra dimension of empty "
            "corner. Do not carry a 2-D threshold across."
        ),
    },
    "euler_number": {
        "kind": "renamed", "means": "objects - handles + cavities",
        "units_2d": "count (holes)", "units_3d": "count (handles, cavities)",
        "note": (
            "In 2-D this counts holes. In 3-D a hole bored through an object "
            "is a handle and an enclosed void is a cavity, and the two enter "
            "with opposite signs. A '1 means no holes' rule does not carry."
        ),
    },
    "major_axis_length": {
        "kind": "same", "means": "longest inertia-ellipsoid axis",
        "units_2d": "px", "units_3d": "um",
        "note": (
            "An ellipsoid axis rather than an ellipse axis, so it can exceed "
            "the 2-D value for the same object -- it is free to point out of "
            "the imaging plane."
        ),
    },
    "minor_axis_length": {
        "kind": "same", "means": "shortest inertia-ellipsoid axis",
        "units_2d": "px", "units_3d": "um",
        "note": (
            "In an anisotropic stack this is usually the z axis, which makes "
            "it the measurement most sensitive to a wrong voxel_size_z_um."
        ),
    },
    "feret_diameter_max": {
        "kind": "same", "means": "largest caliper distance",
        "units_2d": "px", "units_3d": "um",
        "note": "Computed over the 3-D convex hull, so z counts.",
    },
    "perimeter": {
        "kind": "absent", "means": "-",
        "units_2d": "px", "units_3d": "-",
        "note": (
            "Not written by a 3-D run, and deliberately not replaced by a "
            "surface area under the same name. A boundary length and a "
            "surface area are different quantities in different units, and "
            "sharing a column would make every perimeter-based filter and "
            "every stored threshold silently wrong. Use "
            "volume_stats()['surface_um2'] when a surface is what is wanted."
        ),
    },
    "eccentricity": {
        "kind": "absent", "means": "-",
        "units_2d": "ratio", "units_3d": "-",
        "note": (
            "There is no eccentricity of a solid. An ellipsoid needs two "
            "ratios, not one, and skimage raises rather than guess which. "
            "major/minor_axis_length carry the shape information instead."
        ),
    },
}

#: Measurements a 3-D run writes that a 2-D run does not.
MEASUREMENT_ADDED_3D: Tuple[str, ...] = ("volume_voxels", "volume_um3")

#: Per-channel measurements a 3-D run does not produce, and why. These are not
#: bugs waiting to be fixed; each is 2-D by construction.
MEASUREMENT_UNAVAILABLE_3D: Dict[str, str] = {
    "zernike": (
        "Zernike moments are defined on a disc. There is a 3-D analogue, but "
        "it is a different basis with different coefficients, and emitting it "
        "under the same column names would silently redefine every feature a "
        "trained classifier was fitted on."
    ),
    "homogeneity": (
        "skimage's graycomatrix is 2-D only. A per-plane GLCM averaged over z "
        "would be a texture of the planes rather than of the volume, and it "
        "would change with the z step -- a number that moves when the same "
        "cell is re-imaged more finely is not a feature."
    ),
}

#: Object-type prefixes ``spacr.measure`` puts in front of a property name.
_OBJECT_PREFIXES: Tuple[str, ...] = (
    "cell_", "nucleus_", "pathogen_", "cytoplasm_", "organelle_",
)


def _bare_property(name: str) -> str:
    """Strip the object-type prefix from a measurement column name."""
    text = str(name)
    for prefix in _OBJECT_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def describe_3d_measurement(name: str) -> Dict[str, str]:
    """What one measurement column means in a 3-D run.

    Accepts either the bare regionprops name (``"area"``) or a full column
    name carrying its object-type prefix (``"cell_area"``).

    :param name: measurement or column name.
    :returns: a copy of the :data:`MEASUREMENT_MEANING_3D` entry, or an entry
        with ``kind="unknown"`` for a name this table says nothing about.
    """
    text = str(name)
    for key in (text, _bare_property(text)):
        if key in MEASUREMENT_MEANING_3D:
            return dict(MEASUREMENT_MEANING_3D[key])
    return {
        "kind": "unknown", "means": "-", "units_2d": "-", "units_3d": "-",
        "note": (
            "Not in the 3-D meaning table. Intensity statistics are unaffected "
            "by dimensionality -- a mean is a mean over whatever voxels the "
            "label covers -- so most unknowns here are safe. Anything spatial "
            "is not covered and should be checked before it is trusted."
        ),
    }


def report_3d_measurements(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Sort a real measurement table's columns by how 3-D treats them.

    Meant to be run against the columns a 3-D run actually produced, so the
    answer describes that run rather than an intention.

    :param columns: column names from a measurements table.
    :returns: ``{"same": [...], "renamed": [...], "added": [...],
        "absent": [...], "unknown": [...]}``. ``"renamed"`` is the list to
        read: those columns kept their 2-D name and changed their meaning.
        ``"absent"`` lists the 2-D-only properties that ought to be missing;
        one of them turning up in ``columns`` means a 2-D property was
        computed on a volume, and it is reported under ``"renamed"`` so it
        cannot be mistaken for a normal column.
    """
    out: Dict[str, List[str]] = {
        "same": [], "renamed": [], "added": [], "unknown": [],
        "absent": [name for name, entry in MEASUREMENT_MEANING_3D.items()
                   if entry["kind"] == "absent"],
    }
    for column in columns:
        if _bare_property(column) in MEASUREMENT_ADDED_3D:
            out["added"].append(str(column))
            continue
        kind = describe_3d_measurement(column)["kind"]
        out["renamed" if kind == "absent" else kind].append(str(column))
    return out


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


# ===========================================================================
#  4D (Beta): the time axis on top of the z axis
# ===========================================================================
#
# Time-plus-z handling for the 4D (Beta) settings: x, y, z, t.
#
# This is the *t* half of spaCR's volumetric support and it sits directly on
# top of the z half above. Everything z-shaped -- ZStackSpec, segment_3d,
# stitch_planes, resolve_anisotropy, flag_truncated_z, volume_stats -- is
# delegated to rather than re-derived. Like the z half it is free of Cellpose
# and of any tracker library, so the 4-D logic can be tested against synthetic
# label volumes on a CPU in milliseconds: every entry point takes a plain
# numpy array plus a caller-supplied ``segment_fn``.
#
# Five things drive the design.
#
# **The axis order is the crux, and it cannot be guessed.**
#     ``(T, Z, Y, X)`` and ``(Z, T, Y, X)`` are both written by real microscopes
#     and a 4-D shape does not say which one you have: ``(10, 21, 512, 512)`` is
#     either ten timepoints of twenty-one planes or twenty-one timepoints of ten
#     planes, and nothing in the array distinguishes them. Getting it wrong does
#     not crash -- it links objects *across z* and calls the result a track,
#     which produces smooth, plausible, entirely fictional trajectories.
#     :func:`detect_axes` therefore returns ``None`` for the ambiguous case and
#     never picks a side; the order must come from the user, from
#     ``t_axis_order``, or from an explicit ``n_t``/``n_z`` that settles it.
#     (spaCR's own ingest already gets this wrong: ``io.py``'s 4-D TIFF branch
#     hard-codes ``t_dim, z_dim, y_dim, x_dim = images.shape`` with no check.)
#
# **A tracker that cannot do 3-D must not be handed a volume.**
#     Silently projecting z away and linking the projection would give a table
#     that looks exactly like a real one. :func:`track_4d` refuses, names the
#     backend, and says whether the limit is the library's or spaCR's adapter's
#     -- see :data:`TRACK_BACKENDS`. Projection is available, but only when the
#     caller asks for it by name (``project_for_tracking``), and it then says in
#     ``notes`` what it destroyed.
#
# **Anisotropy applies to linking, not just to segmentation.**
#     A displacement gate expressed in pixels means something different along z:
#     at ``dz/dxy = 5`` a two-plane move is a ten-pixel move. The distance-based
#     backends therefore scale the z component of every displacement by the
#     anisotropy before comparing it with the gate, and refuse to run without one
#     (:func:`~spacr.zstack.resolve_anisotropy` raises rather than assuming 1.0).
#     ``max_displacement_px`` is measured in **xy pixels** with z so scaled;
#     ``max_displacement_um`` is measured in **micrometres** and needs a voxel
#     size. The overlap-based backend has no distance in it at all, so anisotropy
#     genuinely does not enter -- exactly as in
#     :data:`~spacr.zstack.MODE_STITCH`.
#
# **The tracks table keeps its existing columns.**
#     ``frame`` / ``track_id`` / ``original_label`` / ``x`` / ``y`` are emitted in
#     that order with the same meanings ``timelapse._relabelled_stack_to_tracks_df``
#     already gives them, so the track visualiser and the motility assay need no
#     change. ``z`` and the volume columns are *additional*. A stack with no z
#     axis gets ``area_px2`` and a volumetric one gets ``volume_voxels``; the two
#     are never written into the same column, because a px^2 area and a voxel
#     count are different quantities (the point :data:`spacr.zstack.VOLUME_STATS_UNITS`
#     exists to make).
#
# **Truncation now has two directions.**
#     An object touching the first or last z plane is cut off in z, exactly as
#     ``seg_qc`` treats an object touching the xy field edge; a track present in
#     the first or last *timepoint* is cut off in t -- it began before the movie
#     did or was still going when it stopped, so its lifetime, its displacement
#     and its division count are all lower bounds. :func:`volume_tracks` flags
#     both, in separate columns, because they are different defects.
#
# Memory
# ------
# A 4-D acquisition is ``n_t * n_z`` fields. :func:`iter_volumes` yields **views**
# into the input, one ``(Z, Y, X)`` timepoint at a time, and never materialises
# the 4-D intensity array; :func:`segment_4d` holds exactly one volume plus
# whatever the segmenter transiently needs (see
# :func:`spacr.zstack.estimate_peak_bytes`). What it *does* have to hold is the
# label array for every timepoint, because linking across t cannot start until
# the last timepoint is segmented; those are int32, so a 41-timepoint, 21-plane,
# 2048x2048 acquisition costs ~14 GB of labels against ~350 MB for the one live
# float32 volume. :func:`estimate_peak_bytes_4d` gives the number.
#
# Scope, stated plainly
# ---------------------
# This reaches exactly as far as the z half above does, which is to say the
# library is real and the pipeline cannot feed it. ``spacr.io`` MIPs z away while
# it organises raw files -- for a 4-D TIFF at ``io.py:5051`` and for a LIF at
# ``io.py:5009`` -- so by the time a timelapse batch reaches segmentation it is
# ``(frames, Y, X, C)`` and the z axis no longer exists. On top of that, none of
# spaCR's five tracker adapters accepts a 4-D array: ``btrack`` and the
# ``trackastra``/``ultrack`` adapters raise on ``ndim != 3``, and the
# trackpy/iou feature table raises out of skimage.
# :func:`plan_4d_from_settings` therefore returns ``None`` whenever ``t_stack``
# is off -- the default -- so not one line of the t half executes in an
# ordinary run, and when it is on without
# a real 4-D array the callers raise :class:`TAxisNotPresentError` naming the
# cause. spaCR will not project a volume, link the projection, and call the
# result a 4-D track.

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Time first, then z. What OME-TIFF's canonical order and most acquisition
#: software write.
AXIS_ORDER_TZYX = "TZYX"

#: z first, then time. What a microscope driven "one z-stack per channel per
#: position, looped over time" writes, and what ImageJ hyperstacks saved with a
#: non-default dimension order carry.
AXIS_ORDER_ZTYX = "ZTYX"

#: A flat time series with no z axis at all: ``(T, Y, X)``. Deliberately *not*
#: one of :data:`AXIS_ORDERS`, because with no z axis there is nothing for
#: :func:`detect_axes` to choose between; it is accepted by
#: :func:`plan_4d_from_settings` as a declaration only, and it is what a user
#: with an ordinary 2-D movie sets. :class:`TStackSpec` has always modelled it
#: as ``z_axis=None`` and :func:`segment_4d` has always made one plain 2-D call
#: per frame for it -- this constant is only the settings-level spelling.
AXIS_ORDER_TYX = "TYX"

#: The two orders :func:`detect_axes` chooses between. It refuses to choose
#: without evidence; see the 4D (Beta) preamble above.
AXIS_ORDERS = (AXIS_ORDER_TZYX, AXIS_ORDER_ZTYX)

#: Link objects between consecutive timepoints by volumetric overlap. Built in
#: (no dependency), works in 2-D and 3-D alike, ignores anisotropy because it
#: computes no distance.
BACKEND_IOU = "iou"

#: Link by nearest centroid under a displacement gate, with the z component
#: scaled by the anisotropy. Built in. Requires an anisotropy and a gate.
BACKEND_CENTROID = "centroid"

#: Link with trackpy's nearest-neighbour linker over ``['z', 'y', 'x']``.
#: trackpy genuinely does 3-D; z is pre-scaled by the anisotropy so its single
#: isotropic ``search_range`` is meaningful.
BACKEND_TRACKPY = "trackpy"

#: btrack. The library tracks in 3-D natively, spaCR's adapter does not.
BACKEND_BTRACK = "btrack"

#: Trackastra. The library has 3-D models, spaCR's adapter does not.
BACKEND_TRACKASTRA = "trackastra"

#: Ultrack. The library is documented for 3-D, spaCR's adapter does not.
BACKEND_ULTRACK = "ultrack"


@dataclass(frozen=True)
class TrackBackend:
    """What one tracking backend can and cannot be driven to do here.

    Two separate booleans, because they are two separate facts and conflating
    them is how a user ends up believing spaCR cannot do something the library
    plainly can.

    :param name: the ``timelapse_mode`` / ``t_track_backend`` spelling.
    :param links_3d: whether :func:`track_4d` can drive it on a ``(Z, Y, X)``
        volume per timepoint.
    :param library_links_3d: whether the upstream package is capable of 3-D at
        all, independent of spaCR.
    :param note: the sentence shown to the user when the two disagree.
    """

    name: str
    links_3d: bool
    library_links_3d: bool
    note: str


#: Every backend :func:`track_4d` knows about, and exactly how far each goes.
#:
#: The three ``links_3d=False`` entries are not a claim about the libraries --
#: all three handle 3-D upstream, and btrack 0.7's
#: ``utils.segmentation_to_objects`` reads a ``(T, Z, Y, X)`` array and returns
#: objects with a real ``z`` without complaint. They are a claim about
#: ``spacr.timelapse``'s adapters, which every one of them gate on
#: ``ndim != 3`` and would have to be rewritten to drive volumetrically. Until
#: that happens, asking for one of them on a volume is an error and not a
#: silent projection.
TRACK_BACKENDS: Dict[str, TrackBackend] = {
    BACKEND_IOU: TrackBackend(
        name=BACKEND_IOU, links_3d=True, library_links_3d=True,
        note="built in; overlap between consecutive timepoints, one-to-one, "
             "no distance and therefore no anisotropy",
    ),
    BACKEND_CENTROID: TrackBackend(
        name=BACKEND_CENTROID, links_3d=True, library_links_3d=True,
        note="built in; nearest centroid under a displacement gate, with the z "
             "component scaled by the anisotropy",
    ),
    BACKEND_TRACKPY: TrackBackend(
        name=BACKEND_TRACKPY, links_3d=True, library_links_3d=True,
        note="trackpy links in 3-D when given pos_columns=['z','y','x']; spaCR "
             "pre-scales z by the anisotropy so its single search_range is "
             "isotropic",
    ),
    BACKEND_BTRACK: TrackBackend(
        name=BACKEND_BTRACK, links_3d=False, library_links_3d=True,
        note="btrack itself tracks in 3-D (segmentation_to_objects reads a "
             "(T,Z,Y,X) array and fills object.z), but spacr.timelapse."
             "_btrack_track_cells raises on any array that is not (T,Y,X), so "
             "spaCR cannot drive it volumetrically today",
    ),
    BACKEND_TRACKASTRA: TrackBackend(
        name=BACKEND_TRACKASTRA, links_3d=False, library_links_3d=True,
        note="Trackastra ships 3-D models, but spacr.timelapse."
             "_trackastra_track_cells requires a (T,Y,X) stack and passes the "
             "2-D model name through, so spaCR cannot drive it volumetrically "
             "today",
    ),
    BACKEND_ULTRACK: TrackBackend(
        name=BACKEND_ULTRACK, links_3d=False, library_links_3d=True,
        note="Ultrack solves segmentation and linking jointly and is "
             "documented for 3-D, but spacr.timelapse._ultrack_track_cells "
             "requires a (T,Y,X) stack, so spaCR cannot drive it "
             "volumetrically today",
    ),
}

#: The columns ``timelapse._relabelled_stack_to_tracks_df`` already emits, in
#: its order. :func:`volume_tracks` reproduces them exactly and appends its own
#: after them, so every existing consumer keeps working unchanged.
BASE_TRACK_COLUMNS = ("frame", "track_id", "original_label", "x", "y")

#: Unit of every column :func:`volume_tracks` can produce. The counterpart of
#: :data:`spacr.zstack.VOLUME_STATS_UNITS`, and here for the same reason: a
#: voxel count and a px^2 area are different quantities and must never share a
#: column.
TRACK_COLUMN_UNITS: Dict[str, str] = {
    "frame": "timepoint index",
    "track_id": "index",
    "original_label": "index",
    "x": "px",
    "y": "px",
    "z": "plane index",
    "volume_voxels": "voxels",
    "area_px2": "px^2",
    "volume_um3": "um^3",
    "z_um": "um",
    "time_s": "s",
    "truncated_z": "bool",
    "truncated_t": "bool",
}

#: Flag name for a track cut off by the first or last timepoint, named to match
#: :data:`spacr.zstack.FLAG_Z_TRUNCATED` for its z equivalent.
FLAG_T_TRUNCATED = "t_truncated"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TStackError(ConfigurationError):
    """Base class for every 4-D configuration problem.

    A :class:`spacr.errors.ConfigurationError`, so a run never continues past
    one: every field would be wrong in the same way.
    """


class AmbiguousAxisOrderError(TStackError):
    """It could not be established which axis is t and which is z.

    Guessing here chooses between tracking through time and "tracking" down a
    z-stack, and the second produces smooth plausible trajectories that mean
    nothing, so the shape is reported back to the user instead.
    """


class TrackerIsTwoDError(TStackError):
    """A backend that cannot link volumes was asked to link volumes.

    Raised rather than projecting z away, because a projected track table is
    indistinguishable from a real one after the fact.
    """


class TAxisNotPresentError(TStackError):
    """The 4D settings are on but the array that arrived has no t (or no z).

    Almost always because ``spacr.io`` collapsed z during ingest. The message
    names the setting to turn off and where the axis went.
    """


class UnknownDisplacementError(TStackError):
    """A distance-based backend was asked for without a displacement gate.

    There is no safe default: the right value is set by how fast the objects
    move relative to the frame interval, and a wrong one either links objects
    that are not the same or splits one object into a track per frame.
    """


# ---------------------------------------------------------------------------
# Axis order
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisOrder:
    """Which axis of the incoming array is which.

    :param t_axis: index of the time axis.
    :param z_axis: index of the z axis, or ``None`` for a flat time series.
    :param y_axis: index of the row axis.
    :param x_axis: index of the column axis.
    :param channel_axis: index of the channel axis, or ``None``.
    :param source: how the order was established -- ``'explicit'``,
        ``'n_t'``, ``'n_z'`` or ``'n_t+n_z'`` -- recorded so it can be written
        next to the tracks it produced.
    """

    t_axis: int
    z_axis: Optional[int]
    y_axis: int
    x_axis: int
    channel_axis: Optional[int] = None
    source: str = "explicit"

    def __post_init__(self):
        axes = [self.t_axis, self.y_axis, self.x_axis]
        if self.z_axis is not None:
            axes.append(self.z_axis)
        if self.channel_axis is not None:
            axes.append(self.channel_axis)
        if len(set(axes)) != len(axes):
            raise TStackError(
                f"the same axis index is used twice in {self!r}: t={self.t_axis}, "
                f"z={self.z_axis}, y={self.y_axis}, x={self.x_axis}, "
                f"channel={self.channel_axis}"
            )

    @property
    def name(self) -> str:
        """``'TZYX'``, ``'ZTYX'``, ``'TYX'`` ... in ascending axis order."""
        letters = {self.t_axis: "T", self.y_axis: "Y", self.x_axis: "X"}
        if self.z_axis is not None:
            letters[self.z_axis] = "Z"
        if self.channel_axis is not None:
            letters[self.channel_axis] = "C"
        return "".join(letters[i] for i in sorted(letters))


def _shape_of(array) -> Tuple[int, ...]:
    """Shape of an array, or the tuple/list itself if one was passed."""
    if isinstance(array, (tuple, list)):
        return tuple(int(s) for s in array)
    return tuple(int(s) for s in np.shape(array))


def detect_axes(
    array,
    n_t: Optional[int] = None,
    n_z: Optional[int] = None,
    xy_min: int = 32,
    channel_axis: Optional[int] = None,
    strict: bool = False,
) -> Optional[AxisOrder]:
    """Work out which leading axis is t and which is z, or refuse to.

    The trailing two axes are taken to be ``(Y, X)``. That is not a guess:
    :data:`AXIS_ORDER_TZYX` and :data:`AXIS_ORDER_ZTYX` agree on it, so there
    is nothing to choose between, and it is checked -- if either of them is
    shorter than ``xy_min`` the array is not ``(..., Y, X)`` at all and that is
    an error rather than a silent transposition.

    The two *leading* axes are ``t`` and ``z`` in an order the shape cannot
    reveal. This function settles it only from evidence:

    * ``n_t`` and/or ``n_z`` given, and exactly one assignment matches -> that
      assignment;
    * a hint given that matches *both* assignments (``n_t == n_z``, or the two
      leading axes are the same length) -> ambiguous, ``None``;
    * a hint given that matches *neither* -> :class:`TStackError`, because the
      user's belief about the data and the data disagree;
    * no hint at all -> ambiguous, ``None``.

    There is deliberately no heuristic on the leading lengths. "Time series are
    longer than z stacks" is true often enough to be dangerous and false often
    enough to ruin an experiment -- a 5-timepoint acquisition of 40-plane
    stacks is an ordinary thing to collect.

    :param array: a 4-D array (or a shape tuple).
    :param n_t: number of timepoints, if known independently of the array.
    :param n_z: number of z planes, if known independently of the array.
    :param xy_min: smallest side length still considered an image axis.
    :param channel_axis: index of a channel axis to ignore, if present.
    :param strict: raise instead of returning ``None`` when ambiguous.
    :returns: an :class:`AxisOrder`, or ``None`` when it cannot be settled.
    :raises ValueError: when the array is not 4-D (after removing any channel
        axis), or its trailing two axes are not an image plane.
    :raises TStackError: when a supplied ``n_t``/``n_z`` matches neither
        reading.
    :raises AmbiguousAxisOrderError: when ``strict`` and the order is
        ambiguous.
    """
    shape = _shape_of(array)

    kept = list(range(len(shape)))
    if channel_axis is not None:
        c_axis = int(channel_axis) % len(shape)
        kept = [i for i in kept if i != c_axis]
    else:
        c_axis = None

    if len(kept) != 4:
        raise ValueError(
            f"detect_axes expects a 4-D (T,Z,Y,X) or (Z,T,Y,X) array; got shape "
            f"{shape}"
            + (f" with channel_axis={channel_axis}" if channel_axis is not None else "")
            + ". A 3-D array is either a time series or a z stack and this "
            "function cannot tell which -- use spacr.zstack for (Z,Y,X), and "
            "state t_axis explicitly for (T,Y,X)."
        )

    a0, a1, ay, ax = kept
    if shape[ay] < xy_min or shape[ax] < xy_min:
        raise ValueError(
            f"the trailing two axes of shape {shape} are {shape[ay]}x{shape[ax]}, "
            f"which is smaller than xy_min={xy_min} and so is not an image "
            f"plane. spaCR reads 4-D arrays as (..., Y, X); if yours is stored "
            f"as (Y, X, Z, T) transpose it before handing it over, or lower "
            f"xy_min if your fields really are that small."
        )

    def _order(t_axis: int, z_axis: int, source: str) -> AxisOrder:
        return AxisOrder(t_axis=t_axis, z_axis=z_axis, y_axis=ay, x_axis=ax,
                         channel_axis=c_axis, source=source)

    # Candidate readings: (t_axis, z_axis).
    candidates = [(a0, a1), (a1, a0)]
    hints = []
    if n_t is not None:
        hints.append(("n_t", int(n_t)))
    if n_z is not None:
        hints.append(("n_z", int(n_z)))

    if hints:
        surviving = []
        for t_axis, z_axis in candidates:
            ok = True
            for which, value in hints:
                axis = t_axis if which == "n_t" else z_axis
                if shape[axis] != value:
                    ok = False
            if ok:
                surviving.append((t_axis, z_axis))

        if len(surviving) == 1:
            t_axis, z_axis = surviving[0]
            return _order(t_axis, z_axis, "+".join(w for w, _ in hints))

        if not surviving:
            raise TStackError(
                f"the axis hints {dict(hints)} match neither reading of shape "
                f"{shape}: axis {a0} has length {shape[a0]} and axis {a1} has "
                f"length {shape[a1]}. Either the array is not the acquisition "
                f"you think it is, or n_t/n_z is wrong -- spaCR will not "
                f"proceed on the assumption that one of them is a typo."
            )
        # Both readings survive: the hint did not discriminate.

    if strict:
        reason = (
            f"axes {a0} and {a1} have lengths {shape[a0]} and {shape[a1]}"
            + (" and the hints given fit both readings" if hints else
               " and no n_t/n_z was given")
        )
        raise AmbiguousAxisOrderError(
            f"cannot tell which axis of shape {shape} is time and which is z: "
            f"{reason}. Reading it as {AXIS_ORDER_TZYX} gives {shape[a0]} "
            f"timepoints of {shape[a1]} planes; reading it as "
            f"{AXIS_ORDER_ZTYX} gives {shape[a1]} timepoints of {shape[a0]} "
            f"planes. Set the `t_axis_order` setting to '{AXIS_ORDER_TZYX}' or "
            f"'{AXIS_ORDER_ZTYX}' (or pass t_axis/z_axis directly). spaCR will "
            f"not guess: guessing wrong links objects across z and reports them "
            f"as trajectories through time, which looks entirely plausible and "
            f"is entirely fictional."
        )
    return None


def resolve_axis_order(
    array,
    axis_order: Optional[str] = None,
    t_axis: Optional[int] = None,
    z_axis: Optional[int] = None,
    n_t: Optional[int] = None,
    n_z: Optional[int] = None,
    channel_axis: Optional[int] = None,
    xy_min: int = 32,
) -> AxisOrder:
    """Return the :class:`AxisOrder` for ``array``, or explain why it cannot.

    An explicit ``axis_order`` name wins, then explicit ``t_axis``/``z_axis``
    indices, then :func:`detect_axes` in strict mode.

    :param array: a 4-D array (or a shape tuple).
    :param axis_order: ``'TZYX'`` or ``'ZTYX'``; case-insensitive.
    :param t_axis: index of the time axis, if known.
    :param z_axis: index of the z axis, if known.
    :param n_t: number of timepoints, used as a disambiguating hint.
    :param n_z: number of z planes, used as a disambiguating hint.
    :param channel_axis: index of a channel axis to ignore, if present.
    :param xy_min: smallest side length still considered an image axis.
    :returns: the resolved :class:`AxisOrder`.
    :raises TStackError: on an unknown ``axis_order`` name, or when the
        explicit indices contradict the array.
    :raises AmbiguousAxisOrderError: when nothing settles the order.
    """
    shape = _shape_of(array)

    if axis_order is not None:
        name = str(axis_order).upper().strip()
        if name not in AXIS_ORDERS:
            raise TStackError(
                f"t_axis_order={axis_order!r} is not one of {list(AXIS_ORDERS)}. "
                f"These are the two orders in which a 4-D acquisition is "
                f"actually written; anything else needs t_axis/z_axis given as "
                f"indices."
            )
        t_axis = name.index("T")
        z_axis = name.index("Z")

    if t_axis is not None or z_axis is not None:
        kept = [i for i in range(len(shape))
                if channel_axis is None or i != int(channel_axis) % len(shape)]
        if len(kept) != 4:
            raise TStackError(
                f"an explicit t_axis/z_axis needs a 4-D (T,Z,Y,X)-shaped array; "
                f"got shape {shape}"
            )
        if t_axis is None or z_axis is None:
            # One given, the other is whichever leading axis is left.
            leading = [i for i in kept[:2]]
            known = t_axis if t_axis is not None else z_axis
            known = int(known) % len(shape)
            others = [i for i in leading if i != known]
            if len(others) != 1:
                raise TStackError(
                    f"axis {known} is not one of the two leading axes {leading} "
                    f"of shape {shape}, so the other one cannot be inferred; "
                    f"give both t_axis and z_axis."
                )
            if t_axis is None:
                t_axis, z_axis = others[0], known
            else:
                t_axis, z_axis = known, others[0]
        t_axis = int(t_axis) % len(shape)
        z_axis = int(z_axis) % len(shape)
        y_axis, x_axis = [i for i in kept if i not in (t_axis, z_axis)]
        order = AxisOrder(
            t_axis=t_axis, z_axis=z_axis, y_axis=y_axis, x_axis=x_axis,
            channel_axis=(None if channel_axis is None
                          else int(channel_axis) % len(shape)),
            source="explicit",
        )
        if n_t is not None and shape[order.t_axis] != int(n_t):
            raise TStackError(
                f"t_axis={order.t_axis} has length {shape[order.t_axis]} but "
                f"n_t={n_t} was given; the declared axis order and the declared "
                f"timepoint count disagree."
            )
        if n_z is not None and shape[order.z_axis] != int(n_z):
            raise TStackError(
                f"z_axis={order.z_axis} has length {shape[order.z_axis]} but "
                f"n_z={n_z} was given; the declared axis order and the declared "
                f"plane count disagree."
            )
        return order

    return detect_axes(array, n_t=n_t, n_z=n_z, xy_min=xy_min,
                       channel_axis=channel_axis, strict=True)


# ---------------------------------------------------------------------------
# Spec / result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TStackSpec:
    """Everything the 4-D plumbing needs to know about one acquisition.

    The geometry half (``t_axis`` ... ``voxel_size_um``) describes the data;
    the rest describes what to do with it. ``z_mode``, ``projection`` and
    ``stitch_threshold`` are handed straight to :mod:`spacr.zstack` via
    :meth:`to_z_spec` and mean exactly what they mean there.

    :param t_axis: index of the time axis in the incoming array.
    :param z_axis: index of the z axis, or ``None`` for a flat time series.
    :param n_t: number of timepoints, or ``None`` when not yet known.
    :param n_z: number of z planes, or ``None`` when not yet known.
    :param anisotropy: ``dz / dxy``. ``None`` means "not known", which is fatal
        for volumetric segmentation and for the distance-based trackers.
    :param frame_interval_s: seconds between consecutive timepoints, or
        ``None``. Only ever used to add a ``time_s`` column -- no linking
        decision depends on it.
    :param voxel_size_um: ``(dz, dy, dx)`` in micrometres, or ``None``.
    :param channel_axis: index of a channel axis, or ``None``.
    :param z_mode: one of :data:`spacr.zstack.SEGMENTATION_MODES`.
    :param projection: reducer used by :data:`~spacr.zstack.MODE_PROJECT`.
    :param stitch_threshold: IoU floor for
        :func:`~spacr.zstack.stitch_planes`, i.e. for linking across **z**.
    :param track_backend: one of :data:`TRACK_BACKENDS`.
    :param link_threshold: IoU floor for linking across **t** in
        :data:`BACKEND_IOU`. Deliberately a separate number from
        ``stitch_threshold``: consecutive z planes and consecutive timepoints
        do not overlap by the same amount.
    :param max_displacement_px: gate for the distance-based backends, in xy
        pixels, with z scaled by ``anisotropy``.
    :param max_displacement_um: the same gate in micrometres; needs
        ``voxel_size_um``. Mutually exclusive with ``max_displacement_px``.
    :param project_for_tracking: opt in to collapsing z before linking, so
        that linking happens on the projection rather than on the volume. Off
        by default and never implied. It does **not** unlock the backends
        spaCR cannot drive volumetrically -- see :func:`track_4d`.
    """

    t_axis: int = 0
    z_axis: Optional[int] = 1
    n_t: Optional[int] = None
    n_z: Optional[int] = None
    anisotropy: Optional[float] = None
    frame_interval_s: Optional[float] = None
    voxel_size_um: Optional[Tuple[float, float, float]] = None
    channel_axis: Optional[int] = None
    z_mode: str = MODE_PROJECT
    projection: Optional[str] = "max"
    stitch_threshold: float = 0.25
    track_backend: str = BACKEND_IOU
    link_threshold: float = 0.25
    max_displacement_px: Optional[float] = None
    max_displacement_um: Optional[float] = None
    project_for_tracking: bool = False

    def __post_init__(self):
        if self.z_axis is not None and int(self.t_axis) == int(self.z_axis):
            raise TStackError(
                f"t_axis and z_axis are both {self.t_axis}; one array axis "
                f"cannot be both time and z"
            )
        if self.channel_axis is not None:
            taken = {int(self.t_axis)}
            if self.z_axis is not None:
                taken.add(int(self.z_axis))
            if int(self.channel_axis) in taken:
                raise TStackError(
                    f"channel_axis={self.channel_axis} collides with t_axis="
                    f"{self.t_axis} / z_axis={self.z_axis}; one array axis "
                    f"cannot be two things at once"
                )
        if self.track_backend not in TRACK_BACKENDS:
            raise TStackError(
                f"t_track_backend={self.track_backend!r} is not one of "
                f"{sorted(TRACK_BACKENDS)}"
            )
        if not 0.0 <= float(self.link_threshold) <= 1.0:
            raise TStackError(
                f"t_link_threshold={self.link_threshold!r} must be an IoU in "
                f"[0, 1]"
            )
        if self.max_displacement_px is not None and self.max_displacement_um is not None:
            raise TStackError(
                "t_max_displacement_px and t_max_displacement_um are both set; "
                "they are the same gate in two units and spaCR will not pick "
                "one. Set exactly one."
            )
        for label, value in (("t_max_displacement_px", self.max_displacement_px),
                             ("t_max_displacement_um", self.max_displacement_um)):
            if value is not None and (not np.isfinite(float(value)) or float(value) <= 0):
                raise TStackError(f"{label}={value!r} must be a finite number > 0")
        if self.frame_interval_s is not None:
            value = float(self.frame_interval_s)
            if not np.isfinite(value) or value <= 0:
                raise TStackError(
                    f"frame_interval_s={self.frame_interval_s!r} must be a "
                    f"finite number of seconds > 0"
                )
        # Validated by ZStackSpec, which owns these three; constructing it here
        # means an impossible z_mode/projection/anisotropy is refused when the
        # spec is built and not after the first field has been read.
        self.to_z_spec()

    @property
    def voxel_size(self) -> Optional[Tuple[float, float, float]]:
        """Alias of ``voxel_size_um``; the units are in the canonical name."""
        return self.voxel_size_um

    @property
    def axis_order(self) -> Optional[str]:
        """``'TZYX'`` / ``'ZTYX'`` when the two leading axes are t and z."""
        if self.z_axis is None:
            return None
        if (int(self.t_axis), int(self.z_axis)) == (0, 1):
            return AXIS_ORDER_TZYX
        if (int(self.t_axis), int(self.z_axis)) == (1, 0):
            return AXIS_ORDER_ZTYX
        return None

    @property
    def backend(self) -> TrackBackend:
        """The :class:`TrackBackend` record for ``track_backend``."""
        return TRACK_BACKENDS[self.track_backend]

    def to_z_spec(self) -> ZStackSpec:
        """The :class:`spacr.zstack.ZStackSpec` for one timepoint of this run.

        :returns: a z spec carrying this spec's z settings verbatim.
        """
        return ZStackSpec(
            z_axis=0,  # iter_volumes always yields z-first volumes
            n_z=self.n_z,
            anisotropy=self.anisotropy,
            voxel_size_um=self.voxel_size_um,
            projection=self.projection,
            mode=self.z_mode,
            stitch_threshold=self.stitch_threshold,
            resample_to_isotropic=False,
        )

    def require_anisotropy(self) -> float:
        """Return ``dz/dxy``, or explain why the run cannot proceed.

        :raises spacr.zstack.UnknownAnisotropyError: when it is not known.
        """
        return resolve_anisotropy(anisotropy=self.anisotropy,
                                  voxel_size_um=self.voxel_size_um)


@dataclass
class TStackResult:
    """The labels a 4-D run produced, plus how it produced them.

    :param labels: **t-first** labels: ``(T, Z, Y, X)`` for the volumetric z
        modes, ``(T, Y, X)`` for ``project`` and for a single-plane
        acquisition. Label values are per-timepoint and carry no identity
        across t until :func:`track_4d` has run.
    :param z_results: the :class:`spacr.zstack.ZStackResult` for each
        timepoint, in order.
    :param spec: the spec that produced them.
    :param n_t: number of timepoints segmented.
    :param n_z: planes per timepoint.
    :param notes: remarks worth surfacing to the user.
    """

    labels: np.ndarray
    z_results: List[Any] = _dc_field(default_factory=list)
    spec: Optional[TStackSpec] = None
    n_t: int = 0
    n_z: int = 1
    notes: List[str] = _dc_field(default_factory=list)

    @property
    def has_z(self) -> bool:
        """Whether the labels kept a z axis (i.e. are ``(T, Z, Y, X)``)."""
        return np.asarray(self.labels).ndim == 4

    @property
    def z_mode(self) -> str:
        """The mode that actually ran, as recorded by the first timepoint."""
        if self.z_results:
            return self.z_results[0].mode
        return MODE_SINGLE_PLANE

    @property
    def objects_per_timepoint(self) -> List[int]:
        """Non-background label count at each timepoint."""
        return [int(np.count_nonzero(np.unique(frame)))
                for frame in np.asarray(self.labels)]


@dataclass
class TrackResult:
    """The outcome of linking a 4-D label array across time.

    :param labels: **t-first** labels renumbered so that one value means one
        track for the whole acquisition; same shape as the input.
    :param tracks: the :func:`volume_tracks` table.
    :param backend: which backend linked them.
    :param n_tracks: number of distinct tracks.
    :param anisotropy: the value used, or ``None`` when the backend has no
        distance in it.
    :param projected: whether z was collapsed before linking (only ever true
        when the caller asked for it).
    :param notes: remarks worth surfacing to the user.
    """

    labels: np.ndarray
    tracks: Any = None
    backend: str = BACKEND_IOU
    n_tracks: int = 0
    anisotropy: Optional[float] = None
    projected: bool = False
    notes: List[str] = _dc_field(default_factory=list)

    @property
    def truncated_tracks(self) -> np.ndarray:
        """Track ids present in the first or last timepoint."""
        return flag_truncated_t(self.labels)

    @property
    def truncated_fraction(self) -> float:
        """Share of tracks cut off by the start or end of the acquisition.

        The t counterpart of :attr:`spacr.zstack.ZStackResult.truncated_fraction`;
        a high value means the movie does not span the process being measured
        and every lifetime in the table is a lower bound.
        """
        if self.n_tracks == 0:
            return 0.0
        return float(self.truncated_tracks.size) / self.n_tracks


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

def as_t_first(array, spec: TStackSpec) -> np.ndarray:
    """Return a **view** of ``array`` with t at axis 0 and z at axis 1.

    A view, not a copy: ``np.moveaxis`` never copies, which is what keeps a
    4-D acquisition from being materialised twice.

    :param array: the acquisition, axes as described by ``spec``.
    :param spec: the :class:`TStackSpec` naming ``t_axis`` / ``z_axis``.
    :returns: the same data as ``(T, Z, Y, X[, C])`` or ``(T, Y, X[, C])``.
    :raises TAxisNotPresentError: when the array has fewer axes than the spec
        claims.
    """
    arr = np.asarray(array)
    needed = 3 if spec.z_axis is None else 4
    if spec.channel_axis is not None:
        needed += 1
    if arr.ndim < needed:
        raise TAxisNotPresentError(
            f"t_stack is on and the spec describes a {needed}-axis acquisition "
            f"(t_axis={spec.t_axis}, z_axis={spec.z_axis}, channel_axis="
            f"{spec.channel_axis}), but the array that arrived has shape "
            f"{arr.shape}. spaCR's image ingest collapses z into one plane per "
            f"field while organising the raw files (spacr.io, the 4-D TIFF and "
            f"LIF branches both take a maximum along z), so by the time a "
            f"batch reaches segmentation there is no z axis left to segment. "
            f"Either turn t_stack off and accept the projection spaCR has "
            f"always made, or hand spacr.zstack.segment_4d your (T, Z, Y, X) "
            f"arrays directly through the Python API. spaCR will not segment "
            f"the projection and report it as a 4-D result."
        )

    if spec.z_axis is None:
        return np.moveaxis(arr, int(spec.t_axis), 0)
    return np.moveaxis(arr, [int(spec.t_axis), int(spec.z_axis)], [0, 1])


def iter_volumes(array, spec: TStackSpec) -> Iterator[np.ndarray]:
    """Yield one ``(Z, Y, X[, C])`` volume per timepoint, lazily.

    Each yielded array is a **view** into ``array``: nothing is copied and the
    4-D acquisition is never materialised a second time. A spec with
    ``z_axis=None`` yields ``(Y, X[, C])`` planes instead, which is the
    ordinary 2-D time series and is handled by exactly the same code.

    :param array: the acquisition, axes as described by ``spec``.
    :param spec: the :class:`TStackSpec`.
    :yields: one volume (or plane) per timepoint, in acquisition order.
    :raises TStackError: when the array's t/z lengths contradict ``spec.n_t``
        or ``spec.n_z``.
    :raises TAxisNotPresentError: when the array has no z axis but the spec
        says it should.
    """
    view = as_t_first(array, spec)

    n_t = int(view.shape[0])
    if spec.n_t is not None and n_t != int(spec.n_t):
        raise TStackError(
            f"t_axis={spec.t_axis} of the array has length {n_t} but the spec "
            f"says n_t={spec.n_t}. One of them is describing a different "
            f"acquisition."
        )
    if spec.z_axis is not None:
        n_z = int(view.shape[1])
        if spec.n_z is not None and n_z != int(spec.n_z):
            raise TStackError(
                f"z_axis={spec.z_axis} of the array has length {n_z} but the "
                f"spec says n_z={spec.n_z}. One of them is describing a "
                f"different acquisition."
            )

    for t in range(n_t):
        yield view[t]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_4d(
    array,
    spec: TStackSpec,
    segment_fn: Callable[..., Any],
    verbose: bool = False,
) -> TStackResult:
    """Segment every timepoint independently, in 3-D, and stack the results.

    Each timepoint goes to :func:`spacr.zstack.segment_3d` with this spec's z
    settings, so the per-timepoint behaviour is *identical* to a 3-D run and
    the two cannot drift apart. Segmentation is deliberately per-timepoint and
    independent: linking is a separate decision made by :func:`track_4d`, and
    fusing the two is what makes a tracker's mistakes indistinguishable from a
    segmenter's.

    Two degenerate cases are not degenerate at all, and both are exact:

    * ``n_t == 1`` -- the result is the 3-D result. ``labels[0]`` is
      byte-identical to what :func:`spacr.zstack.segment_3d` returns for that
      volume.
    * ``n_z == 1`` -- every timepoint short-circuits to
      :data:`~spacr.zstack.MODE_SINGLE_PLANE`, so ``labels`` is ``(T, Y, X)``
      and each frame is byte-identical to ``segment_fn(plane)``. That is the
      ordinary 2-D path.

    :param array: the acquisition, axes as described by ``spec``.
    :param spec: the :class:`TStackSpec`.
    :param segment_fn: the ``segment_fn`` contract of
        :func:`spacr.zstack.segment_3d`; see that function for the kwargs it
        receives in each mode.
    :param verbose: print each timepoint's z notes as it is segmented.
    :returns: a :class:`TStackResult` whose labels are t-first.
    :raises TStackError: on an inconsistent spec (see :func:`iter_volumes`).
    """
    z_spec = spec.to_z_spec()
    labels: List[np.ndarray] = []
    z_results: List[Any] = []

    for t, volume in enumerate(iter_volumes(array, spec)):
        if spec.z_axis is None:
            # A flat time series: one 2-D call per frame, no z code at all.
            result = ZStackResult(
                labels=np.asarray(segment_fn(volume)),
                mode=MODE_SINGLE_PLANE, anisotropy=None, n_z=1,
                notes=["no z axis: segmented in 2-D, exactly as a non-z run"],
            )
        else:
            result = segment_3d(
                volume,
                segment_fn=segment_fn,
                mode=z_spec.mode,
                stitch_threshold=z_spec.stitch_threshold,
                anisotropy=z_spec.anisotropy,
                voxel_size_um=z_spec.voxel_size_um,
                projection=z_spec.projection,
                z_axis=0,
                resample_to_isotropic=z_spec.resample_to_isotropic,
            )
        labels.append(np.asarray(result.labels))
        z_results.append(result)
        if verbose:
            for note in result.notes:
                print(f"[4D] t={t}: {note}")

    if not labels:
        raise TStackError(
            "segment_4d got an acquisition with zero timepoints; there is "
            "nothing to segment or to link"
        )

    shapes = {arr.shape for arr in labels}
    if len(shapes) != 1:
        raise TStackError(
            f"the timepoints segmented to different shapes {sorted(shapes)}; "
            f"they cannot be stacked, let alone linked"
        )

    stacked = np.stack(labels, axis=0)
    n_t = len(labels)
    n_z = int(z_results[0].n_z)

    notes = [
        f"{n_t} timepoint(s) x {n_z} plane(s) segmented independently in "
        f"'{z_results[0].mode}' mode; label values are per-timepoint and mean "
        f"nothing across t until track_4d has linked them"
    ]
    if n_t == 1:
        notes.append(
            "a single timepoint is a 3-D run, not a degenerate 4-D one: "
            "labels[0] is exactly what zstack.segment_3d returns"
        )
    if stacked.ndim == 3:
        notes.append(
            "labels have no z axis (single plane, or 'project' mode collapsed "
            "it), so this is the ordinary 2-D time series and volumes are not "
            "measurable from it"
        )

    return TStackResult(labels=stacked, z_results=z_results, spec=spec,
                        n_t=n_t, n_z=n_z, notes=notes)


# ---------------------------------------------------------------------------
# Label geometry
# ---------------------------------------------------------------------------

def _label_centroids(labels: np.ndarray):
    """Centroid and voxel count of every non-background label.

    Works in any dimensionality, so the same code serves a ``(Z, Y, X)`` volume
    and a ``(Y, X)`` plane.

    :param labels: an integer label array.
    :returns: ``(ids, centroids, counts)`` where ``centroids`` is
        ``(n_labels, ndim)`` in axis order.
    """
    labels = np.asarray(labels)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return ids.astype(np.int64), np.zeros((0, labels.ndim)), np.zeros(0, np.int64)

    coords = np.nonzero(labels)
    flat = labels[coords]
    n = int(labels.max()) + 1
    counts = np.bincount(flat, minlength=n)
    centroids = np.stack(
        [np.bincount(flat, weights=c, minlength=n)[ids] / counts[ids]
         for c in coords],
        axis=1,
    )
    return ids.astype(np.int64), centroids, counts[ids].astype(np.int64)


def project_labels(labels_3d) -> np.ndarray:
    """Collapse a ``(Z, Y, X)`` label volume to ``(Y, X)``, honestly.

    Taking a maximum along z -- what :func:`spacr.zstack.project` does to
    *intensities* -- is meaningless for labels, because label values are
    arbitrary ids and the maximum simply picks the highest-numbered object.
    This instead gives each pixel the label that occupies the most planes in
    that column, which is the only projection that answers "which object is
    here?".

    It costs one pass over the volume per label present, which is why it is
    only ever reached when the caller has explicitly asked for
    ``project_for_tracking``.

    :param labels_3d: a ``(Z, Y, X)`` label volume.
    :returns: a ``(Y, X)`` label image.
    """
    volume = np.asarray(labels_3d)
    if volume.ndim == 2:
        return volume

    out = np.zeros(volume.shape[1:], dtype=volume.dtype)
    best = np.zeros(volume.shape[1:], dtype=np.int64)
    ids = np.unique(volume)
    for value in ids[ids > 0]:
        count = (volume == value).sum(axis=0)
        take = count > best
        best[take] = count[take]
        out[take] = value
    return out


def flag_truncated_t(labels_4d) -> np.ndarray:
    """Track ids present in the first or last timepoint, and so cut off in t.

    The time counterpart of :func:`spacr.zstack.flag_truncated_z`: such a track
    began before the acquisition did, or was still running when it ended, so
    its lifetime, total displacement and division count are lower bounds and it
    should be reported rather than quietly averaged into a survival curve.

    :param labels_4d: a t-first label array; ``(T, Z, Y, X)`` or ``(T, Y, X)``.
    :returns: sorted array of truncated track ids.
    """
    labels = np.asarray(labels_4d)
    if labels.ndim < 3 or labels.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    ends = np.concatenate([labels[0].ravel(), labels[-1].ravel()])
    ids = np.unique(ends)
    return ids[ids > 0].astype(np.int64)


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def _displacement_scale(ndim: int, anisotropy: float,
                        voxel_size_um: Optional[Sequence[float]],
                        in_um: bool) -> np.ndarray:
    """Per-axis multipliers turning centroid coordinates into gate units.

    :param ndim: 3 for ``(Z, Y, X)`` centroids, 2 for ``(Y, X)``.
    :param anisotropy: ``dz / dxy``, applied to the z component only.
    :param voxel_size_um: ``(dz, dy, dx)``; required when ``in_um``.
    :param in_um: whether the gate is in micrometres rather than xy pixels.
    :returns: an ``(ndim,)`` array of multipliers.
    :raises TStackError: when micrometres are asked for without a voxel size.
    """
    if in_um:
        if voxel_size_um is None:
            raise TStackError(
                "t_max_displacement_um is set but the voxel size is not known, "
                "so pixels cannot be converted to micrometres. Set "
                "voxel_size_z_um and voxel_size_xy_um, or express the gate in "
                "pixels with t_max_displacement_px."
            )
        dz, dy, dx = (float(v) for v in voxel_size_um)
        return np.array([dz, dy, dx][3 - ndim:], dtype=float)

    # xy-pixel space: x and y are already in pixels, z is `anisotropy` pixels
    # per plane. This is the whole reason anisotropy matters to tracking.
    return np.array([float(anisotropy), 1.0, 1.0][3 - ndim:], dtype=float)


def _centroid_matches(prev_labels, cur_labels, scale: np.ndarray,
                      max_distance: float):
    """Nearest-centroid matches under a hard displacement gate.

    The assignment is globally optimal (Hungarian) over the gated cost matrix,
    then any pair that still exceeds the gate is dropped -- a pair beyond the
    gate is not a link at any cost.

    :param prev_labels: labels at t-1.
    :param cur_labels: labels at t.
    :param scale: per-axis multipliers from :func:`_displacement_scale`.
    :param max_distance: gate, in the units ``scale`` produces.
    :returns: ``{cur_label: prev_label}``.
    """
    from scipy.optimize import linear_sum_assignment

    prev_ids, prev_c, _ = _label_centroids(prev_labels)
    cur_ids, cur_c, _ = _label_centroids(cur_labels)
    if prev_ids.size == 0 or cur_ids.size == 0:
        return {}

    delta = (prev_c[:, None, :] - cur_c[None, :, :]) * scale[None, None, :]
    cost = np.sqrt((delta ** 2).sum(axis=-1))

    # Gate first so the solver never prefers a long link just to complete a
    # permutation, then drop anything that is still over the gate.
    big = float(max_distance) * 1e6 + 1.0
    gated = np.where(cost <= max_distance, cost, big)
    rows, cols = linear_sum_assignment(gated)

    assigned: Dict[int, int] = {}
    for i, j in zip(rows, cols):
        if cost[i, j] <= max_distance:
            assigned[int(cur_ids[j])] = int(prev_ids[i])
    return assigned


def _trackpy_maps(labels_4d, scale: np.ndarray, max_distance: float,
                  memory: int = 0) -> List[Dict[int, int]]:
    """Link with trackpy over ``['z', 'y', 'x']`` and return per-timepoint maps.

    trackpy takes one isotropic ``search_range``, so the z coordinate is
    pre-scaled by the anisotropy before it is handed over -- otherwise the gate
    would silently mean ``anisotropy`` times further along z than across it.
    ``pos_columns`` is passed explicitly because trackpy's own column guess
    reads whichever of ``x``/``y``/``z`` happen to be present, which makes the
    dimensionality of the link depend on the shape of a DataFrame rather than
    on a decision anyone made.

    :param labels_4d: t-first label array.
    :param scale: per-axis multipliers from :func:`_displacement_scale`.
    :param max_distance: search range, in the units ``scale`` produces.
    :param memory: frames a track may vanish for and still be continued.
    :returns: one ``{original_label: track_id}`` map per timepoint.
    :raises RuntimeError: when trackpy is not installed, naming the fix.
    """
    import pandas as pd

    try:
        import trackpy as tp
    except ImportError as exc:
        raise RuntimeError(
            "t_track_backend='trackpy' needs the trackpy package, which is not "
            "installed. Install it with `pip install trackpy`, or choose "
            "t_track_backend='iou' / 'centroid', which are built in."
        ) from exc

    labels = np.asarray(labels_4d)
    rows = []
    for t, frame in enumerate(labels):
        ids, centroids, _ = _label_centroids(frame)
        for value, centroid in zip(ids, centroids * scale[None, :]):
            row = {"frame": t, "original_label": int(value)}
            if centroid.size == 3:
                row.update(z=float(centroid[0]), y=float(centroid[1]),
                           x=float(centroid[2]))
            else:
                row.update(z=0.0, y=float(centroid[0]), x=float(centroid[1]))
            rows.append(row)

    if not rows:
        return [{} for _ in range(labels.shape[0])]

    features = pd.DataFrame(rows)
    linked = tp.link(features, search_range=float(max_distance),
                     pos_columns=["z", "y", "x"], t_column="frame",
                     memory=int(memory))

    maps: List[Dict[int, int]] = [{} for _ in range(labels.shape[0])]
    for _, row in linked.iterrows():
        maps[int(row["frame"])][int(row["original_label"])] = int(row["particle"]) + 1
    return maps


def _apply_track_maps(labels_4d, maps: Sequence[Dict[int, int]]) -> np.ndarray:
    """Renumber a t-first label array so one value means one track.

    :param labels_4d: t-first label array.
    :param maps: one ``{original_label: track_id}`` per timepoint.
    :returns: a new array of the same shape with track ids in it.
    """
    labels = np.asarray(labels_4d)
    out = np.zeros(labels.shape, dtype=np.int64)
    for t, mapping in enumerate(maps):
        frame = labels[t]
        if not mapping:
            continue
        lut = np.zeros(int(frame.max()) + 1, dtype=np.int64)
        for original, track_id in mapping.items():
            if original <= frame.max():
                lut[int(original)] = int(track_id)
        out[t] = lut[frame]
    return relabel_volume(out)


def track_4d(
    labels_or_result,
    spec: Optional[TStackSpec] = None,
    backend: Optional[str] = None,
    link_threshold: Optional[float] = None,
    max_displacement_px: Optional[float] = None,
    max_displacement_um: Optional[float] = None,
    anisotropy: Optional[float] = None,
    project_for_tracking: Optional[bool] = None,
    memory: int = 0,
) -> TrackResult:
    """Link objects across time, in 3-D, and renumber them by track.

    The input is **t-first** labels -- ``(T, Z, Y, X)`` or ``(T, Y, X)`` --
    which is exactly what :func:`segment_4d` returns, or a
    :class:`TStackResult` itself. There is no axis detection here on purpose:
    by this point the order is a decision that has already been made and
    recorded, and re-deriving it would be a second chance to get it wrong.

    Handing a volume to a backend that cannot link volumes raises
    :class:`TrackerIsTwoDError` naming the backend and saying whether the limit
    is the library's or spaCR's; it does not project. ``project_for_tracking``
    turns the projection on explicitly, and then the result records that z was
    destroyed, because two objects stacked in z become one object in the
    projection and no downstream number can tell that it happened.

    :param labels_or_result: t-first label array, or a :class:`TStackResult`.
    :param spec: the :class:`TStackSpec`; taken from the result when it carries
        one, and defaulted otherwise.
    :param backend: overrides ``spec.track_backend``.
    :param link_threshold: overrides ``spec.link_threshold`` (IoU backend).
    :param max_displacement_px: overrides the spec's gate, in xy pixels.
    :param max_displacement_um: overrides the spec's gate, in micrometres.
    :param anisotropy: overrides ``spec.anisotropy``.
    :param project_for_tracking: overrides ``spec.project_for_tracking``.
    :param memory: frames a track may vanish for; supported by the trackpy
        backend only, and 0 everywhere else -- the built-in linkers compare
        consecutive timepoints and nothing else, so a missed detection ends a
        track and starts a new one.
    :returns: a :class:`TrackResult`.
    :raises TrackerIsTwoDError: when the backend cannot link volumes and the
        projection was not explicitly asked for.
    :raises UnknownDisplacementError: when a distance-based backend has no gate.
    :raises spacr.zstack.UnknownAnisotropyError: when a distance-based backend
        has no anisotropy and a volume to link.
    :raises TStackError: on an unknown backend or a non-t-first input.
    """
    if isinstance(labels_or_result, TStackResult):
        if spec is None:
            spec = labels_or_result.spec
        labels = np.asarray(labels_or_result.labels)
    else:
        labels = np.asarray(labels_or_result)

    if spec is None:
        spec = TStackSpec()

    name = spec.track_backend if backend is None else str(backend)
    if name not in TRACK_BACKENDS:
        raise TStackError(
            f"t_track_backend={name!r} is not one of {sorted(TRACK_BACKENDS)}"
        )
    record = TRACK_BACKENDS[name]

    threshold = spec.link_threshold if link_threshold is None else float(link_threshold)
    gate_px = spec.max_displacement_px if max_displacement_px is None else float(max_displacement_px)
    gate_um = spec.max_displacement_um if max_displacement_um is None else float(max_displacement_um)
    if gate_px is not None and gate_um is not None:
        raise TStackError(
            "t_max_displacement_px and t_max_displacement_um are both set; "
            "they are the same gate in two units and spaCR will not pick one."
        )
    aniso_value = spec.anisotropy if anisotropy is None else float(anisotropy)
    project = (spec.project_for_tracking if project_for_tracking is None
               else bool(project_for_tracking))

    if labels.ndim not in (3, 4):
        raise TStackError(
            f"track_4d expects t-first labels, (T,Z,Y,X) or (T,Y,X); got shape "
            f"{labels.shape}. Use as_t_first(array, spec) to reorder an "
            f"acquisition before tracking it -- track_4d deliberately does not "
            f"detect the axis order itself."
        )

    notes: List[str] = []
    volumetric = labels.ndim == 4
    projected = False

    if volumetric and not record.links_3d:
        # Deliberately not unlocked by project_for_tracking. This module
        # does not drive this backend at all -- it lives in spacr.timelapse
        # and takes (T, Y, X) -- so projecting here and then linking with a
        # DIFFERENT linker would report one tracker's answer under another
        # tracker's name, which is a worse lie than the one this refusal
        # prevents.
        raise TrackerIsTwoDError(
            f"t_track_backend='{record.name}' cannot link "
            f"{labels.shape[1]}-plane volumes as spaCR drives it, and the "
            f"labels handed to it are volumetric (shape {labels.shape}). "
            f"{record.note}. Either choose a backend that links volumes "
            f"({', '.join(sorted(b for b, r in TRACK_BACKENDS.items() if r.links_3d))}), "
            f"or, if you specifically want {record.name}, collapse z yourself "
            f"with spacr.zstack.project_labels and hand the resulting (T, Y, X) "
            f"stack to spacr.timelapse's adapter for it -- that merges every "
            f"pair of objects separated only in z, which is exactly why spaCR "
            f"makes you do it rather than doing it silently."
        )

    if volumetric and project:
        # A real choice with a real cost, and only ever made explicitly: link
        # the projection rather than the volume. Faster and less sensitive to a
        # wrong anisotropy, at the price of fusing anything stacked in z.
        labels = np.stack([project_labels(frame) for frame in labels], axis=0)
        volumetric = False
        projected = True
        notes.append(
            "project_for_tracking: z was collapsed before linking, as asked. "
            "Objects that overlap in xy but are separated in z have been "
            "merged into one and nothing computed downstream can tell that it "
            "happened."
        )

    aniso_used: Optional[float] = None

    if name == BACKEND_IOU:
        # Linking across t by overlap is the same algorithm as linking across
        # z by overlap, so this is zstack.stitch_planes applied along axis 0 --
        # which here is time, explicitly, never inferred.
        tracked = stitch_planes(labels, iou_threshold=threshold)
        notes.append(
            f"linked across t by overlap at IoU >= {threshold} (one-to-one, "
            f"consecutive timepoints only). Overlap has no distance in it, so "
            f"anisotropy does not enter this backend at all."
        )
        if aniso_value is not None:
            notes.append(
                "anisotropy is ignored by the 'iou' backend, exactly as it is "
                "ignored by zstack's stitch mode"
            )
    elif name in (BACKEND_CENTROID, BACKEND_TRACKPY):
        in_um = gate_um is not None
        gate = gate_um if in_um else gate_px
        if gate is None:
            raise UnknownDisplacementError(
                f"t_track_backend='{name}' links by distance and this run has "
                f"no displacement gate. Set t_max_displacement_px (in xy "
                f"pixels, with z scaled by the anisotropy) or "
                f"t_max_displacement_um (in micrometres, which also needs "
                f"voxel_size_z_um / voxel_size_xy_um). spaCR will not pick a "
                f"default: the right value depends on how far your objects "
                f"move between frames, and a wrong one either fuses "
                f"neighbouring objects into one track or breaks one object "
                f"into a track per frame."
            )

        ndim = 3 if volumetric else 2
        if volumetric and not in_um:
            # Required, not defaulted: at dz/dxy = 5 a two-plane move is a
            # ten-pixel move, and treating it as two lets objects five times
            # too far apart link.
            aniso_used = resolve_anisotropy(aniso_value, spec.voxel_size_um)
        else:
            aniso_used = aniso_value
        scale = _displacement_scale(
            ndim, 1.0 if aniso_used is None else aniso_used,
            spec.voxel_size_um, in_um,
        )

        if name == BACKEND_CENTROID:
            maps: List[Dict[int, int]] = []
            prev_map: Dict[int, int] = {}
            next_track = 1
            for t in range(labels.shape[0]):
                current: Dict[int, int] = {}
                ids = np.unique(labels[t])
                ids = ids[ids > 0]
                matched: Dict[int, int] = {}
                if t > 0:
                    matched = _centroid_matches(labels[t - 1], labels[t],
                                                scale, float(gate))
                for value in ids:
                    value = int(value)
                    parent = matched.get(value)
                    if parent is not None and parent in prev_map:
                        current[value] = prev_map[parent]
                    else:
                        current[value] = next_track
                        next_track += 1
                maps.append(current)
                prev_map = current
            tracked = _apply_track_maps(labels, maps)
        else:
            tracked = _apply_track_maps(
                labels, _trackpy_maps(labels, scale, float(gate), memory=memory)
            )

        units = "um" if in_um else "xy px"
        notes.append(
            f"linked across t by centroid distance <= {gate:g} {units}"
            + (f", with z scaled by anisotropy {aniso_used:g} so that one "
               f"plane counts as {aniso_used:g} xy pixels"
               if (ndim == 3 and not in_um) else "")
            + f" ('{name}' backend)"
        )
    else:
        # Reached when one of the three adapter-only backends is asked for on
        # a flat (T, Y, X) stack. That is a perfectly reasonable thing to want
        # -- it is just not this module's job, and pretending otherwise would
        # mean quietly substituting a built-in linker for the one named.
        raise TStackError(
            f"t_track_backend='{record.name}' is not driven by spacr.zstack: "
            f"{record.note}. For a flat (T, Y, X) time series use "
            f"spacr.timelapse's existing adapter for it, which is what the "
            f"timelapse_mode setting selects; the 4D half here adds only the "
            f"backends that can link volumes ("
            f"{', '.join(sorted(b for b, r in TRACK_BACKENDS.items() if r.links_3d))})."
        )

    n_tracks = int(np.count_nonzero(np.unique(tracked)))
    truncated = flag_truncated_t(tracked)
    if truncated.size:
        notes.append(
            f"{truncated.size} of {n_tracks} track(s) touch the first or last "
            f"timepoint and are truncated in t: their lifetimes and total "
            f"displacements are lower bounds, exactly as a z-truncated "
            f"object's volume is"
        )

    result = TrackResult(
        labels=tracked, backend=name, n_tracks=n_tracks,
        anisotropy=aniso_used, projected=projected, notes=notes,
    )
    result.tracks = volume_tracks(tracked, spec)
    return result


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def volume_tracks(labels_4d, spec: Optional[TStackSpec] = None):
    """One row per object per timepoint: where it is, how big, how truncated.

    The first five columns are ``frame``, ``track_id``, ``original_label``,
    ``x``, ``y`` -- the same names, order and meanings that
    ``spacr.timelapse._relabelled_stack_to_tracks_df`` already emits from a
    relabelled stack, so the track visualiser and the motility assay consume
    this table unchanged. Everything after them is new.

    ``original_label`` equals ``track_id``, as it does in the existing table:
    after linking, the label value in the stack *is* the track id.

    Size is reported in exactly one column, never two: ``volume_voxels`` for a
    ``(T, Z, Y, X)`` stack and ``area_px2`` for a ``(T, Y, X)`` one. They are
    different quantities and writing one into the other's column is the
    single easiest way to corrupt a screen -- the same point
    :data:`spacr.zstack.VOLUME_STATS_UNITS` exists to make. Micrometre and
    second columns appear only when the voxel size and frame interval are
    known.

    :param labels_4d: t-first labels whose values are track ids.
    :param spec: the :class:`TStackSpec`; supplies the voxel size and frame
        interval when known.
    :returns: a :class:`pandas.DataFrame` sorted by ``track_id`` then
        ``frame``.
    :raises TStackError: when the array is not t-first 3-D or 4-D.
    """
    import pandas as pd

    labels = np.asarray(labels_4d)
    if labels.ndim not in (3, 4):
        raise TStackError(
            f"volume_tracks expects t-first labels, (T,Z,Y,X) or (T,Y,X); got "
            f"shape {labels.shape}"
        )

    volumetric = labels.ndim == 4
    voxel_size = None if spec is None else spec.voxel_size_um
    interval = None if spec is None else spec.frame_interval_s

    columns = list(BASE_TRACK_COLUMNS)
    if volumetric:
        columns += ["z", "volume_voxels"]
        if voxel_size is not None:
            columns += ["volume_um3", "z_um"]
    else:
        columns += ["area_px2"]
    if interval is not None:
        columns += ["time_s"]
    columns += ["truncated_z", "truncated_t"]

    truncated_t = set(flag_truncated_t(labels).tolist())

    rows = []
    for t, frame in enumerate(labels):
        ids, centroids, counts = _label_centroids(frame)
        truncated_z = set(flag_truncated_z(frame).tolist()) if volumetric else set()
        for value, centroid, count in zip(ids, centroids, counts):
            value = int(value)
            if volumetric:
                cz, cy, cx = (float(c) for c in centroid)
            else:
                cz = None
                cy, cx = float(centroid[0]), float(centroid[1])
            row = {
                "frame": t,
                "track_id": value,
                "original_label": value,
                "x": cx,
                "y": cy,
            }
            if volumetric:
                row["z"] = cz
                row["volume_voxels"] = int(count)
                if voxel_size is not None:
                    dz, dy, dx = (float(v) for v in voxel_size)
                    row["volume_um3"] = int(count) * dz * dy * dx
                    row["z_um"] = cz * dz
            else:
                row["area_px2"] = int(count)
            if interval is not None:
                row["time_s"] = t * float(interval)
            row["truncated_z"] = value in truncated_z
            row["truncated_t"] = value in truncated_t
            rows.append(row)

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})

    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_4d(result) -> str:
    """Render a :class:`TStackResult` or :class:`TrackResult` as text.

    Written to be pasted next to the numbers it describes: the axis order, the
    modes and the truncation counts are the three things that make a 4-D result
    interpretable, and none of them can be recovered from the tracks table
    afterwards.

    :param result: a :class:`TStackResult` or a :class:`TrackResult`.
    :returns: a multi-line summary.
    :raises TypeError: for anything else.
    """
    lines: List[str] = []

    if isinstance(result, TStackResult):
        spec = result.spec
        order = spec.axis_order if spec is not None else None
        lines.append("4D (Beta) segmentation")
        lines.append(f"  axis order      : {order or 'custom/2-D'}")
        lines.append(f"  timepoints      : {result.n_t}")
        lines.append(f"  planes per t    : {result.n_z}")
        lines.append(f"  z mode          : {result.z_mode}")
        if spec is not None and spec.anisotropy is not None:
            lines.append(f"  anisotropy      : {spec.anisotropy:g} (dz/dxy)")
        if spec is not None and spec.frame_interval_s is not None:
            lines.append(f"  frame interval  : {spec.frame_interval_s:g} s")
        counts = result.objects_per_timepoint
        if counts:
            lines.append(
                f"  objects per t   : min {min(counts)}, max {max(counts)}"
            )
    elif isinstance(result, TrackResult):
        lines.append("4D (Beta) tracking")
        lines.append(f"  backend         : {result.backend}")
        lines.append(f"  tracks          : {result.n_tracks}")
        lines.append(
            f"  anisotropy      : "
            + (f"{result.anisotropy:g} (dz/dxy)" if result.anisotropy is not None
               else "not used by this backend")
        )
        lines.append(
            f"  truncated in t  : {result.truncated_tracks.size} "
            f"({result.truncated_fraction:.0%} of tracks touch the first or "
            f"last timepoint)"
        )
        if result.projected:
            lines.append("  z               : COLLAPSED before linking "
                         "(project_for_tracking)")
    else:
        raise TypeError(
            f"format_4d takes a TStackResult or a TrackResult, got "
            f"{type(result).__name__}"
        )

    for note in result.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Settings bridge
# ---------------------------------------------------------------------------

def _require_leading_axis(value, name: str) -> int:
    """Check that a lone axis index is one of the two leading axes.

    Only axis 0 and axis 1 have a partner to infer: given ``z_axis=1`` the time
    axis must be 0, and vice versa. ``z_axis=2`` is a perfectly valid *3-D*
    setting -- it is how you say a field is stored ``(Y, X, Z)`` -- but for a
    4-D acquisition there is no second leading axis to deduce from it, and
    arithmetic on it would silently produce ``t_axis=-1``.

    :param value: the axis index that was given.
    :param name: the setting it came from, for the message.
    :returns: the index as an int.
    :raises TStackError: when the index is not 0 or 1.
    """
    index = int(value)
    if index not in (0, 1):
        raise TStackError(
            f"{name}={value!r} is not one of the two leading axes (0 or 1), so "
            f"spaCR cannot deduce which axis the other one is. That is a valid "
            f"3-D setting -- it says where z sits in a single field -- but a "
            f"4-D acquisition needs the whole order: set t_axis_order to "
            f"'{AXIS_ORDER_TZYX}' or '{AXIS_ORDER_ZTYX}', or give both t_axis "
            f"and z_axis."
        )
    return index


def plan_4d_from_settings(settings) -> Optional[TStackSpec]:
    """Build a :class:`TStackSpec` from a settings dict, or ``None`` when off.

    Returning ``None`` is the contract that keeps the 2-D and 3-D paths
    bit-identical: every caller branches on ``spec is None`` and, when it is,
    does not touch any 4-D code at all. ``t_stack`` absent and ``t_stack=False``
    are the same thing.

    The z half of the spec is read from the very same keys
    :func:`spacr.zstack.plan_from_settings` reads, so a 4-D run and a 3-D run
    configured the same way segment identically. ``frame_interval_s`` falls
    back to the motility module's ``seconds_per_frame`` when it is not set,
    rather than becoming a second source of truth for the same physical number.

    :param settings: the pipeline settings dict.
    :returns: a spec, or ``None`` when 4D handling is off.
    :raises TStackError: when 4D is on but the settings are self-inconsistent.
    :raises spacr.zstack.ZStackError: when the z half is self-inconsistent.
    """
    if not settings.get("t_stack", False):
        return None

    dz = settings.get("voxel_size_z_um")
    dxy = settings.get("voxel_size_xy_um")
    voxel_size = None
    if dz is not None and dxy is not None:
        voxel_size = (float(dz), float(dxy), float(dxy))

    order = settings.get("t_axis_order")
    t_axis = settings.get("t_axis")
    z_axis = settings.get("z_axis")

    if order:
        name = str(order).upper().strip()
        if name == AXIS_ORDER_TYX:
            # A flat time series: no z axis is claimed, so there is no order to
            # refuse and nothing to disambiguate. This is the common case -- an
            # ordinary 2-D movie -- and it is the only spelling that makes it
            # expressible from the settings.
            if z_axis is not None:
                raise TStackError(
                    f"t_axis_order='{AXIS_ORDER_TYX}' says the acquisition has "
                    f"no z axis, but z_axis={z_axis!r} places one. spaCR will "
                    f"not pick one: unset whichever of them is wrong."
                )
            t_axis = 0 if t_axis is None else int(t_axis)
            if t_axis != 0:
                raise TStackError(
                    f"t_axis_order='{AXIS_ORDER_TYX}' describes a (T, Y, X) "
                    f"acquisition, whose time axis is axis 0, but "
                    f"t_axis={t_axis} was given. Give t_axis=0, or state the "
                    f"whole order with t_axis/z_axis indices instead."
                )
        elif name not in AXIS_ORDERS:
            raise TStackError(
                f"t_axis_order={order!r} is not one of "
                f"{list(AXIS_ORDERS) + [AXIS_ORDER_TYX]}"
            )
        else:
            # An explicit t_axis/z_axis alongside the order must agree with it.
            # Letting the order silently win would mean a user who set both, and
            # got one of them wrong, is segmenting a differently-transposed array
            # than they think -- which is the failure this setting exists to stop.
            for label, given, implied in (("t_axis", t_axis, name.index("T")),
                                          ("z_axis", z_axis, name.index("Z"))):
                if given is not None and int(given) != implied:
                    raise TStackError(
                        f"t_axis_order={name!r} puts {label[0]} on axis {implied}, "
                        f"but {label}={given!r} says axis {int(given)}. spaCR will "
                        f"not pick one: unset whichever of them is wrong."
                    )
            t_axis, z_axis = name.index("T"), name.index("Z")
    elif t_axis is None and z_axis is None:
        raise AmbiguousAxisOrderError(
            "t_stack is on but neither t_axis_order nor t_axis/z_axis is set, "
            "so spaCR does not know whether the leading axes of your data are "
            "(T, Z, ...) or (Z, T, ...). It will not guess: reading one as the "
            "other links objects across z and reports them as trajectories "
            "through time, which looks entirely plausible and is entirely "
            "fictional. Set t_axis_order to 'TZYX' or 'ZTYX' -- or to 'TYX' if "
            "your acquisition has no z axis at all."
        )
    elif t_axis is None:
        z_axis = _require_leading_axis(z_axis, "z_axis")
        t_axis = 1 - z_axis
    elif z_axis is None:
        t_axis = _require_leading_axis(t_axis, "t_axis")
        z_axis = 1 - t_axis

    interval = settings.get("frame_interval_s")
    if interval is None:
        interval = settings.get("seconds_per_frame")

    spec = TStackSpec(
        t_axis=int(t_axis),
        z_axis=None if z_axis is None else int(z_axis),
        anisotropy=settings.get("anisotropy"),
        frame_interval_s=None if interval is None else float(interval),
        voxel_size_um=voxel_size,
        z_mode=settings.get("z_segmentation_mode", MODE_PROJECT),
        projection=settings.get("z_projection", "max"),
        stitch_threshold=float(settings.get("stitch_threshold", 0.25) or 0.0),
        track_backend=settings.get("t_track_backend", BACKEND_IOU),
        link_threshold=float(settings.get("t_link_threshold", 0.25) or 0.0),
        max_displacement_px=settings.get("t_max_displacement_px"),
        max_displacement_um=settings.get("t_max_displacement_um"),
        project_for_tracking=bool(settings.get("t_project_for_tracking", False)),
    )

    # Fail here rather than after the model has been loaded and the first
    # timepoint read: none of these answers can change later in the run.
    if spec.z_axis is not None and spec.z_mode == MODE_VOLUMETRIC:
        spec.require_anisotropy()
    if spec.track_backend in (BACKEND_CENTROID, BACKEND_TRACKPY):
        if spec.max_displacement_px is None and spec.max_displacement_um is None:
            raise UnknownDisplacementError(
                f"t_track_backend='{spec.track_backend}' links by distance but "
                f"neither t_max_displacement_px nor t_max_displacement_um is "
                f"set. Set one; spaCR will not pick a default gate."
            )
        if (spec.z_axis is not None and spec.z_mode != MODE_PROJECT
                and spec.max_displacement_um is None):
            spec.require_anisotropy()
    if (spec.z_axis is not None
            and not TRACK_BACKENDS[spec.track_backend].links_3d
            and spec.z_mode != MODE_PROJECT):
        raise TrackerIsTwoDError(
            f"t_track_backend='{spec.track_backend}' cannot link volumes as "
            f"spaCR drives it, but z_segmentation_mode='{spec.z_mode}' "
            f"produces them. {TRACK_BACKENDS[spec.track_backend].note}. Choose "
            f"a backend that links volumes "
            f"({', '.join(sorted(b for b, r in TRACK_BACKENDS.items() if r.links_3d))}), "
            f"or set z_segmentation_mode='project' to work in 2-D throughout. "
            f"t_project_for_tracking does not rescue this combination: it "
            f"collapses z so that a volume-capable linker works on the "
            f"projection, and spaCR still would not be running "
            f"'{spec.track_backend}'."
        )

    return spec


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def estimate_peak_bytes_4d(shape: Sequence[int], dtype=np.float32,
                            z_mode: str = MODE_PROJECT,
                            anisotropy: float = 1.0,
                            label_dtype=np.int32) -> int:
    """Peak bytes a 4-D run needs, so an acquisition can be sized before it runs.

    Two terms, and the second is the one that surprises people:

    * **one timepoint live** -- :func:`spacr.zstack.estimate_peak_bytes` for a
      single ``(Z, Y, X)`` volume, because :func:`iter_volumes` yields views
      and :func:`segment_4d` holds exactly one of them at a time;
    * **every timepoint's labels** -- linking cannot begin until the last
      timepoint is segmented, so the whole ``(T, Z, Y, X)`` label array is
      resident. At int32 that is 4 bytes per voxel per timepoint, and for a
      41 x 21 x 2048 x 2048 acquisition it is ~14 GB against ~350 MB for the
      live volume -- which is the number that decides whether the run fits.

    :param shape: ``(T, Z, Y, X)`` or ``(T, Z, Y, X, C)``.
    :param dtype: image dtype.
    :param z_mode: one of :data:`spacr.zstack.SEGMENTATION_MODES`.
    :param anisotropy: used only by :data:`~spacr.zstack.MODE_VOLUMETRIC`.
    :param label_dtype: dtype of the retained label array.
    :returns: estimated peak bytes.
    :raises ValueError: when ``shape`` is not at least ``(T, Z, Y, X)``.
    """
    shape = [int(s) for s in shape]
    if len(shape) < 4:
        raise ValueError(
            f"estimate_peak_bytes_4d expects at least (T, Z, Y, X); got "
        f"{tuple(shape)}"
        )

    n_t, volume_shape = shape[0], shape[1:]
    live = estimate_peak_bytes(volume_shape, dtype=dtype, mode=z_mode,
                                  anisotropy=anisotropy)

    label_voxels = int(np.prod(volume_shape[:3]))
    if z_mode == MODE_PROJECT:
        label_voxels = int(np.prod(volume_shape[1:3]))  # z is gone
    labels = n_t * label_voxels * np.dtype(label_dtype).itemsize

    return int(live + labels)
