"""Human-readable data dictionary for the columns of a spaCR ``measurements.db``.

A finished spaCR run writes hundreds of columns per object table with names like
``cell_channel_1_percentile_75``, ``nucleus_zernike_12`` or
``pathogen_channel_0_channel_2_M1_correlation_85``. This module turns those
names back into prose: what the number means, what unit it is in, which object
and which channel(s) it came from, and which line of code produced it.

Everything in :data:`KNOWN_PROPERTIES` was derived by reading the emitters in
:mod:`spacr.measure` and :mod:`spacr.utils` — not from guessing at the names.
Where the code does something surprising (a doubled name prefix, a feature that
is always NaN, a radial bin that covers the background) the entry says so in
``notes`` rather than describing the intent.

**Geometric units are not fixed any more.** A 2-D run measures in pixels, but a
3-D run measures a volume, and with ``voxel_size_z_um`` / ``voxel_size_xy_um``
set it measures in micrometres — under the *same* column names, because
:mod:`spacr.measure` deliberately does not rename ``<object>_area`` (renaming
would break every downstream selector). Which one a row is in is recorded on
the row itself, in ``measurement_units``. So the unit of a geometric column is
a :class:`ConditionalUnit`: :func:`describe_database` reads
``measurement_units`` out of the database it is documenting and resolves it,
and a caller who has no database says so and gets the condition spelled out
instead of a confident guess. See :data:`MEASUREMENT_UNITS`.

Typical use::

    from spacr.feature_dict import describe_database, export_dictionary

    df = describe_database("/data/exp1/measurements/measurements.db")
    export_dictionary("/data/exp1/measurements/measurements.db",
                      "/data/exp1/feature_dictionary.md", fmt="md")

The module is deliberately dependency-light: standard library plus pandas. It
imports no torch, no cellpose and no scikit-image, so it can be used to explain
a database on a machine that cannot run the pipeline.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

__all__ = [
    "ConditionalUnit",
    "FeatureEntry",
    "PropertyInfo",
    "FEATURE_FAMILIES",
    "KNOWN_PROPERTIES",
    "MEASUREMENT_UNITS",
    "MEASUREMENT_STAMP_COLUMNS",
    "META_COLUMNS",
    "OBJECT_TYPES",
    "parse_column",
    "describe_columns",
    "describe_database",
    "export_dictionary",
]


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

#: Object types that can prefix a feature column. These are the entries of
#: ``ls`` in :func:`spacr.measure._morphological_measurements` (measure.py:167)
#: and :func:`spacr.measure._intensity_measurements` (measure.py:363), and they
#: are also the object-table names in ``measurements.db``.
OBJECT_TYPES: tuple[str, ...] = (
    "cell",
    "nucleus",
    "pathogen",
    "organelle",
    "cytoplasm",
)

#: Feature families used by :attr:`FeatureEntry.family`, with a one-line gloss.
FEATURE_FAMILIES: dict[str, str] = {
    "morphology": (
        "Size, shape and position measured from the label mask alone; no "
        "intensity information enters these. Pixel units in a 2-D run — in a "
        "3-D run the sizes are volumes, in micrometres when the run knew its "
        "voxel size, so read each column's unit and the row's "
        "measurement_units rather than assuming pixels."
    ),
    "intensity": (
        "Statistics of one channel's pixel values inside (or just outside) an "
        "object. Native image units unless stated otherwise."
    ),
    "texture": (
        "Spatial structure of one channel inside an object — how grainy, "
        "smooth or in-focus it looks — rather than how bright it is."
    ),
    "correlation": (
        "Colocalisation between two channels over the same object's pixels."
    ),
    "moment": (
        "Image/shape moments: rotation-invariant shape descriptors and "
        "intensity-weighted centroids."
    ),
    "meta": (
        "Identifiers and bookkeeping — plate/well/field, object labels, file "
        "paths, settings. Not a measurement; never feed these to a model as "
        "features."
    ),
    "unknown": (
        "Not recognised by this dictionary. The column is still reported so "
        "it is never silently dropped, but its meaning was not determined "
        "from the spaCR source."
    ),
}


@dataclass(frozen=True)
class PropertyInfo:
    """One curated definition, shared by every column that instantiates it.

    :ivar family: One of the keys of :data:`FEATURE_FAMILIES`.
    :ivar description: What the number means, in prose. ``None`` only when the
        meaning could not be determined from the spaCR source.
    :ivar unit: Physical/derived unit, or ``None`` for identifiers. A
        :class:`ConditionalUnit` for the geometric columns, whose unit depends
        on the row's ``measurement_units``; :func:`parse_column` resolves it.
    :ivar computed_by: The real provenance — the function or library call that
        produces the value. Never empty.
    :ivar notes: Caveats, known defects, comparability warnings.
    """

    family: str
    description: str | None
    unit: str | ConditionalUnit | None
    computed_by: str
    notes: str | None = None


@dataclass(frozen=True)
class FeatureEntry:
    """A single database column, decomposed and explained.

    :ivar column: The column name exactly as it appears in the database.
    :ivar object_type: ``cell`` / ``nucleus`` / ``pathogen`` / ``organelle`` /
        ``cytoplasm``, or ``None`` for metadata and un-prefixed columns.
    :ivar channel: Zero-based index of the measured channel, or ``None``.
    :ivar family: One of the keys of :data:`FEATURE_FAMILIES`.
    :ivar description: Prose meaning, or ``None`` when undetermined.
    :ivar unit: Unit of the value, or ``None``.
    :ivar computed_by: Provenance string; ``"unknown"`` for unrecognised names.
    :ivar notes: Caveats, or ``None``.
    :ivar channel_2: Second channel index for two-channel (colocalisation)
        columns, otherwise ``None``.
    :ivar object_type_2: Second object type, set when a column carries a
        pandas merge suffix such as ``..._nucleus`` from
        :func:`spacr.io._read_and_join_tables`.
    :ivar measurement_units: The ``measurement_units`` value ``unit`` was
        resolved under — ``px``, ``px_xy``, ``um``, or ``None`` when it was not
        known and ``unit`` therefore states its own condition. Always ``None``
        for columns whose unit does not depend on it.
    """

    column: str
    object_type: str | None
    channel: int | None
    family: str
    description: str | None
    unit: str | None
    computed_by: str
    notes: str | None
    channel_2: int | None = None
    object_type_2: str | None = None
    measurement_units: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a plain dict, ready for a DataFrame row or JSON."""
        return asdict(self)


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

#: The values :func:`spacr.measure.resolve_measurement_spacing` writes into the
#: per-row ``measurement_units`` column, in the order they are reported:
#:
#: ``px``
#:     A 2-D run. ``regionprops_table`` is called with no ``spacing``
#:     (``resolve_measurement_spacing`` returns ``None`` for 2-D
#:     *unconditionally*, even when a voxel size is configured, so that
#:     ``<object>_area`` cannot silently change from px^2 to um^2 under an
#:     unchanged name). Areas are pixel counts, lengths are pixels — exactly
#:     what every spaCR run wrote before 3-D measurement existed.
#: ``px_xy``
#:     A 3-D run given ``anisotropy`` alone. z is scaled by dz/dxy, so the
#:     geometry is right, but the unit is the xy pixel: ``<object>_area`` is a
#:     volume in cubic xy-pixels.
#: ``um``
#:     A 3-D run given ``voxel_size_z_um`` **and** ``voxel_size_xy_um``.
#:     Lengths in um, and ``<object>_area`` is a volume in um^3.
#:
#: A 3-D run with neither raises rather than assuming isotropy, so these three
#: are the whole space. Duplicated from :data:`spacr.measure.UNITS_PX` and its
#: siblings rather than imported, because this module must stay importable on a
#: machine with no numpy or scikit-image; ``tests/test_feature_dict_3d.py``
#: pins the two definitions together.
UNITS_PX = "px"
UNITS_PX_XY = "px_xy"
UNITS_UM = "um"
MEASUREMENT_UNITS: tuple[str, ...] = (UNITS_PX, UNITS_PX_XY, UNITS_UM)

#: What an unstamped row is. Every spaCR release before 3-D measurement existed
#: could only write 2-D pixel rows — a 3-D mask crashed the morphology pass
#: outright — so a table with rows but no ``measurement_units`` column is px as
#: a matter of fact, not as a guess. Mirrors ``spacr.utils._LEGACY_STAMP``.
_LEGACY_UNITS = UNITS_PX


@dataclass(frozen=True)
class ConditionalUnit:
    """A unit that depends on how the row was measured.

    This module used to state a single unit for every geometric column and say
    in the string itself that "spaCR never applies a physical pixel size".
    That was true until :mod:`spacr.measure` learned to measure a ``(Z, Y, X)``
    mask in 3-D: a run with ``voxel_size_z_um`` and ``voxel_size_xy_um`` set
    reports micrometres, and one with ``anisotropy`` alone reports xy-pixel
    units. The column *name* is identical in all three cases — measure.py
    records the unit on the row instead of renaming the column — which is
    exactly why the dictionary has to read it from the data rather than assert
    it.

    :ivar px: the unit when ``measurement_units == 'px'`` (a 2-D run).
    :ivar px_xy: the unit when ``measurement_units == 'px_xy'``.
    :ivar um: the unit when ``measurement_units == 'um'``.

    A field is ``None`` when the column is not written at all in that mode —
    the ``_z``/``_y``/``_x`` centroid axes exist only in 3-D, for instance.
    """

    px: str | None
    px_xy: str | None
    um: str | None

    def by_units(self) -> dict[str, str | None]:
        """``{measurement_units value: unit}`` for all three modes."""
        return {UNITS_PX: self.px, UNITS_PX_XY: self.px_xy, UNITS_UM: self.um}

    def conditional_text(self) -> str:
        """Every possibility, each with the condition it holds under.

        Used when the ``measurement_units`` of the data being described is not
        known. Long, but a wrong unit is worse than a long one.
        """
        parts = [
            f"{unit} when measurement_units='{stamp}'" if unit
            else f"not written when measurement_units='{stamp}'"
            for stamp, unit in self.by_units().items()
        ]
        return ("depends on the row's measurement_units column — "
                + "; ".join(parts))

    def resolve(self, measurement_units: str | None = None) -> str:
        """Return the concrete unit for a row stamped ``measurement_units``.

        :param measurement_units: one of :data:`MEASUREMENT_UNITS`. ``None``, or
            any value this module does not recognise, returns
            :meth:`conditional_text` — the condition, not a guess.
        """
        if measurement_units is not None:
            key = str(measurement_units)
            unit = self.by_units().get(key)
            if unit is not None:
                return unit
            if key in MEASUREMENT_UNITS:
                return f"not written when measurement_units='{key}'"
        return self.conditional_text()


_LENGTH_PX_XY = (
    "xy pixels (3-D measured with anisotropy alone: z is scaled by dz/dxy so "
    "the length is geometrically correct, but the unit is the xy pixel and not "
    "a physical one)"
)
_LENGTH_UM = "um (micrometres; 3-D measured with voxel_size_z_um + voxel_size_xy_um)"

#: A length. Pixels in 2-D; in 3-D ``regionprops_table`` is called with a
#: ``spacing``, so the same column is in xy pixels or in micrometres.
_PX = ConditionalUnit(px="px (pixels)", px_xy=_LENGTH_PX_XY, um=_LENGTH_UM)

#: An "area" — which in 3-D is a **volume**, because skimage's ``area`` on a
#: label volume is the spaced voxel count and measure.py keeps the name.
#: ``<object>_volume_voxels`` / ``<object>_volume_um3`` carry the same quantity
#: under a name that cannot be misread.
_PX2 = ConditionalUnit(
    px="px^2 (pixel count)",
    px_xy=("cubic xy pixels — in 3-D this column is a VOLUME, in xy-pixel "
           "units (see <object>_volume_voxels)"),
    um="um^3 — in 3-D this column is a VOLUME (see <object>_volume_um3)",
)

#: A length that only a 3-D run writes at all.
_PX_3D_ONLY = ConditionalUnit(px=None, px_xy=_LENGTH_PX_XY, um=_LENGTH_UM)

#: A length on a column skimage implements for 2-D only, so a 3-D run does not
#: write it and there is nothing conditional about it.
_PX_2D_ONLY = "px (pixels; 2-D only — a 3-D run does not write this column)"

# Intensity is read from the merged stack, which _merge_file (io.py:2367)
# builds from ``<src>/stack`` — the *raw* concatenated channel arrays. The
# percentile-normalised copies produced by concatenate_and_normalize live in a
# separate folder and are used for segmentation, not for measurement. The only
# transformation measure.py applies is a dtype promotion to uint16 for arrays
# that are neither uint8 nor uint16 (measure.py:914-917).
_INTENSITY = (
    "native image intensity units of the merged stack (raw acquisition "
    "counts, typically uint16; not background-subtracted and not calibrated)"
)
# The sum is np.sum(region.intensity_image[region.image]) — a plain sum over
# the object's elements with no spacing factor, so in 3-D it is intensity x
# voxel count and NOT intensity x um^3, whatever the voxel size says.
_INTENSITY_SUM_3D = (
    "native image intensity units summed over voxels (intensity x voxel "
    "count). NOT converted by the voxel size: measure.py sums the object's "
    "values with no spacing factor, so this is not intensity x um^3"
)
_INTENSITY_SUM = ConditionalUnit(
    px="native image intensity units summed over pixels (intensity x px^2)",
    px_xy=_INTENSITY_SUM_3D,
    um=_INTENSITY_SUM_3D,
)
_DIMLESS = "dimensionless"
_FRACTION = "fraction in [0, 1]"


# --------------------------------------------------------------------------
# curated definitions
# --------------------------------------------------------------------------

_RP = "skimage.measure.regionprops_table"

#: Curated definitions keyed by the *stat* part of a column name — i.e. what is
#: left after the object prefix and any ``channel_N`` infixes are removed.
#: Parameterised names are stored under a template key such as
#: ``percentile_<p>``; :func:`parse_column` matches the concrete name with a
#: regex and formats the placeholder into the description.
KNOWN_PROPERTIES: dict[str, PropertyInfo] = {
    # ---------------- morphology (measure.py:163-164, morphological_props)
    "area": PropertyInfo(
        "morphology",
        "Size of the object in its label mask: the pixel count in a 2-D run "
        "and, in a 3-D run, the object's VOLUME — skimage's `area` on a "
        "(Z, Y, X) label volume is the spaced voxel count, and measure.py "
        "keeps the name so that downstream selectors do not break.",
        _PX2,
        f"{_RP}(area, spacing=...) via spacr.measure._morphological_measurements",
        "In 2-D, multiply by (um/px)^2 to convert to physical area. In 3-D the "
        "conversion has already happened when measurement_units='um'. The "
        "unambiguous companions are <object>_volume_voxels and "
        "<object>_volume_um3, which a 3-D run writes alongside this column.",
    ),
    "area_filled": PropertyInfo(
        "morphology",
        "Area of the object after filling any holes enclosed by it — a filled "
        "volume in a 3-D run.",
        _PX2,
        f"{_RP}(area_filled, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "area_filled - area is the total hole area (3-D: cavity volume) "
        "inside the object.",
    ),
    "area_bbox": PropertyInfo(
        "morphology",
        "Area of the smallest axis-aligned bounding box around the object — "
        "the bounding-box volume in a 3-D run.",
        _PX2,
        f"{_RP}(area_bbox, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "Depends on the object's orientation relative to the image axes, so "
        "it is not rotation invariant.",
    ),
    "convex_area": PropertyInfo(
        "morphology",
        "Area of the convex hull of the object — the convex-hull volume in a "
        "3-D run.",
        _PX2,
        f"{_RP}(convex_area, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "scikit-image's modern name for the same property is area_convex.",
    ),
    "major_axis_length": PropertyInfo(
        "morphology",
        "Length of the major axis of the ellipse that has the same normalised "
        "second central moments as the object (an ellipsoid in 3-D).",
        _PX,
        f"{_RP}(major_axis_length, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "In 3-D this is the longest of the three principal axes of the "
        "equivalent ellipsoid, and it is spaced — without a spacing it would "
        "mix a z step and an xy pixel in one number.",
    ),
    "minor_axis_length": PropertyInfo(
        "morphology",
        "Length of the minor axis of the ellipse that has the same normalised "
        "second central moments as the object (an ellipsoid in 3-D).",
        _PX,
        f"{_RP}(minor_axis_length, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "In 3-D this is the SHORTEST of the three principal axes, so it is not "
        "the 2-D minor axis of any one plane and the two are not comparable.",
    ),
    "eccentricity": PropertyInfo(
        "morphology",
        "Eccentricity of the equivalent ellipse: 0 for a perfect circle, "
        "approaching 1 for an increasingly elongated object.",
        _DIMLESS + ", in [0, 1)",
        f"{_RP}(eccentricity) via spacr.measure._morphological_measurements",
        "2-D only. skimage does not implement eccentricity for a 3-D region "
        "and there is no meaningful eccentricity of a solid, so a 3-D run "
        "drops the property (spacr.measure.PROPS_2D_ONLY) and this column is "
        "absent rather than wrong.",
    ),
    "solidity": PropertyInfo(
        "morphology",
        "Area divided by convex-hull area — how much of the object's own "
        "convex hull it fills. Low values mean a ragged or concave outline.",
        _DIMLESS + ", in (0, 1]",
        f"{_RP}(solidity) via spacr.measure._morphological_measurements",
        "A ratio of two equally-spaced quantities, so it means the same thing "
        "in 2-D and 3-D (in 3-D it is volume over convex-hull volume) and "
        "carries no unit either way.",
    ),
    "extent": PropertyInfo(
        "morphology",
        "Area divided by bounding-box area — how much of its bounding box the "
        "object fills.",
        _DIMLESS + ", in (0, 1]",
        f"{_RP}(extent) via spacr.measure._morphological_measurements",
        "Orientation dependent, because the bounding box is axis aligned. As "
        "a ratio of two equally-spaced quantities it is unaffected by the "
        "voxel size in 3-D.",
    ),
    "perimeter": PropertyInfo(
        "morphology",
        "Perimeter of the object, approximated as a line through the centres "
        "of its border pixels.",
        _PX_2D_ONLY,
        f"{_RP}(perimeter) via spacr.measure._morphological_measurements",
        "Perimeter estimates on a pixel grid are biased upward for small "
        "objects; do not compare across very different object sizes without "
        "care. 2-D only: skimage implements perimeter for 2-D regions, and its "
        "3-D analogue is a surface AREA in different units, so a 3-D run drops "
        "the property (spacr.measure.PROPS_2D_ONLY) rather than writing a "
        "differently-dimensioned number under this name.",
    ),
    "euler_number": PropertyInfo(
        "morphology",
        "Euler characteristic of the object: connected components minus "
        "holes. A single hole-free object gives 1; each enclosed hole "
        "subtracts 1.",
        _DIMLESS + " (integer)",
        f"{_RP}(euler_number) via spacr.measure._morphological_measurements",
        "The 3-D form is the honest generalisation and keeps this name: "
        "components - tunnels + cavities, so a solid blob is still 1 but a "
        "torus is 0 and a hollow shell is 2. A 3-D value is therefore not "
        "comparable with a 2-D one.",
    ),
    "equivalent_diameter_area": PropertyInfo(
        "morphology",
        "Diameter of the circle with the same area as the object, "
        "sqrt(4 * area / pi).",
        _PX,
        f"{_RP}(equivalent_diameter_area, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "In 3-D this silently becomes a CUBE root: the diameter of the sphere "
        "with the same volume, (6 * volume / pi)^(1/3). The name is kept "
        "because it is the honest generalisation, but a 2-D and a 3-D value "
        "are different quantities and must not be pooled.",
    ),
    "feret_diameter_max": PropertyInfo(
        "morphology",
        "Maximum Feret diameter: the longest distance between any two points "
        "on the object's boundary (its calliper length).",
        _PX,
        f"{_RP}(feret_diameter_max, spacing=...) via "
        "spacr.measure._morphological_measurements",
        "Defined in 3-D too, and spaced there: without a spacing it would "
        "compare a distance in z with a distance in xy as if the two were the "
        "same length.",
    ),
    # ---------------- 3-D volumes (measure.py:_voxel_volume_columns)
    "volume_voxels": PropertyInfo(
        "morphology",
        "Number of voxels belonging to the object — its volume as a raw count, "
        "with no voxel size applied.",
        "voxels (count)",
        "numpy.bincount over the label volume in "
        "spacr.measure._voxel_volume_columns",
        "3-D runs only; a 2-D field writes no volume columns. Written even "
        "though <object>_area already holds the volume, because a name that "
        "carries its own unit cannot be misread. Deliberately NOT spaced: this "
        "is the voxel count, so on an anisotropic stack it is not proportional "
        "to a physical volume — use <object>_volume_um3, or <object>_area when "
        "measurement_units='um'. The name matches spacr.zstack.volume_stats.",
    ),
    "volume_um3": PropertyInfo(
        "morphology",
        "Physical volume of the object: its voxel count times "
        "voxel_size_z_um * voxel_size_xy_um^2.",
        "um^3 (cubic micrometres)",
        "spacr.measure._voxel_volume_columns (voxels * dz * dxy * dxy)",
        "Written only when the run knew both voxel_size_z_um and "
        "voxel_size_xy_um, i.e. when measurement_units='um'. A 3-D run "
        "configured with anisotropy alone gets volume_voxels but no "
        "volume_um3, because there is no physical size to convert with. The "
        "name matches spacr.zstack.volume_stats.",
    ),
    # ---------------- shape moments (measure.py:56-80)
    "zernike_<i>": PropertyInfo(
        "moment",
        "Magnitude of Zernike moment number {i} of the object's binary shape. "
        "Zernike magnitudes are rotation invariant, so the 25-element vector "
        "as a whole is a shape fingerprint; a single index has no standalone "
        "biological meaning.",
        _DIMLESS,
        "mahotas.features.zernike_moments(region.image, radius, degree=degree) "
        "via spacr.measure._calculate_zernike",
        "Computed on the binary mask only, no intensity. mahotas' signature is "
        "zernike_moments(im, radius, degree=8); spaCR now derives the `radius` "
        "per object (the maximum distance from its centre of mass, floored at "
        "1) so the unit disk covers the whole object and the coefficients are "
        "comparable across object sizes. Databases written before that fix "
        "passed `degree` positionally into `radius`, fixing the disk at 8 px "
        "for every object. 2-D only: Zernike moments are defined on a disk, so "
        "a 3-D run writes no zernike_* columns at all rather than describing "
        "one arbitrary plane.",
    ),
    # ---------------- intensity, from regionprops (measure.py:360)
    "mean_intensity": PropertyInfo(
        "intensity",
        "Mean pixel value of this channel over all pixels of the object.",
        _INTENSITY,
        f"{_RP}(mean_intensity) via spacr.measure._extended_regionprops_table",
        None,
    ),
    "max_intensity": PropertyInfo(
        "intensity",
        "Brightest pixel value of this channel inside the object.",
        _INTENSITY,
        f"{_RP}(max_intensity) via spacr.measure._extended_regionprops_table",
        "Saturates at the detector's ceiling; a value pinned at 65535 usually "
        "means clipped acquisition rather than a real maximum.",
    ),
    "min_intensity": PropertyInfo(
        "intensity",
        "Dimmest pixel value of this channel inside the object.",
        _INTENSITY,
        f"{_RP}(min_intensity) via spacr.measure._extended_regionprops_table",
        None,
    ),
    "centroid_weighted-0": PropertyInfo(
        "moment",
        "Row (y) coordinate of the intensity-weighted centroid of this "
        "channel inside the object, in full-image coordinates.",
        _PX_2D_ONLY,
        f"{_RP}(centroid_weighted) via spacr.measure._extended_regionprops_table",
        "The -0 / -1 suffix is scikit-image's separator for multi-value "
        "properties: -0 is the row axis, -1 is the column axis. This numeric "
        "spelling is 2-D only. In 3-D the same suffix would be the PLANE (z) "
        "and any reader taking it for y would be silently wrong, so "
        "spacr.measure._rename_3d_centroids renames the 3-D columns to "
        "_z / _y / _x and leaves these 2-D names untouched.",
    ),
    "centroid_weighted-1": PropertyInfo(
        "moment",
        "Column (x) coordinate of the intensity-weighted centroid of this "
        "channel inside the object, in full-image coordinates.",
        _PX_2D_ONLY,
        f"{_RP}(centroid_weighted) via spacr.measure._extended_regionprops_table",
        "The -0 / -1 suffix is scikit-image's separator for multi-value "
        "properties: -0 is the row axis, -1 is the column axis. 2-D only; a "
        "3-D run writes centroid_weighted_z / _y / _x instead.",
    ),
    "centroid_weighted_local-0": PropertyInfo(
        "moment",
        "Row (y) coordinate of the intensity-weighted centroid, measured "
        "relative to the top-left corner of the object's bounding box.",
        _PX_2D_ONLY,
        f"{_RP}(centroid_weighted_local) via "
        "spacr.measure._extended_regionprops_table",
        "Position within the object, so unlike centroid_weighted-0 it does not "
        "encode where in the field of view the object sits. 2-D only; a 3-D "
        "run writes centroid_weighted_local_z / _y / _x instead.",
    ),
    "centroid_weighted_local-1": PropertyInfo(
        "moment",
        "Column (x) coordinate of the intensity-weighted centroid, measured "
        "relative to the top-left corner of the object's bounding box.",
        _PX_2D_ONLY,
        f"{_RP}(centroid_weighted_local) via "
        "spacr.measure._extended_regionprops_table",
        "Position within the object, so unlike centroid_weighted-1 it does not "
        "encode where in the field of view the object sits. 2-D only; a 3-D "
        "run writes centroid_weighted_local_z / _y / _x instead.",
    ),
    # ---------------- 3-D centroids, named by axis
    # (measure.py:_CENTROID_AXES_3D / _rename_3d_centroids)
    "centroid_weighted_z": PropertyInfo(
        "moment",
        "Plane (z) coordinate of the intensity-weighted centroid of this "
        "channel inside the object, in full-volume coordinates.",
        _PX_3D_ONLY,
        f"{_RP}(centroid_weighted, spacing=...) then "
        "spacr.measure._rename_3d_centroids",
        "3-D only. skimage names this centroid_weighted-0, which is what the "
        "row axis is called in 2-D; measure.py renames the 3-D columns by axis "
        "so that no column name silently changes meaning between a flat field "
        "and a z-stack. Spaced, so it is a physical depth when "
        "measurement_units='um' rather than a plane index.",
    ),
    "centroid_weighted_y": PropertyInfo(
        "moment",
        "Row (y) coordinate of the intensity-weighted centroid of this "
        "channel inside the object, in full-volume coordinates.",
        _PX_3D_ONLY,
        f"{_RP}(centroid_weighted, spacing=...) then "
        "spacr.measure._rename_3d_centroids",
        "3-D only; the 2-D equivalent is centroid_weighted-0 (skimage's name "
        "for this axis in 3-D is centroid_weighted-1).",
    ),
    "centroid_weighted_x": PropertyInfo(
        "moment",
        "Column (x) coordinate of the intensity-weighted centroid of this "
        "channel inside the object, in full-volume coordinates.",
        _PX_3D_ONLY,
        f"{_RP}(centroid_weighted, spacing=...) then "
        "spacr.measure._rename_3d_centroids",
        "3-D only; the 2-D equivalent is centroid_weighted-1 (skimage's name "
        "for this axis in 3-D is centroid_weighted-2).",
    ),
    "centroid_weighted_local_z": PropertyInfo(
        "moment",
        "Plane (z) coordinate of the intensity-weighted centroid, measured "
        "relative to the corner of the object's bounding box.",
        _PX_3D_ONLY,
        f"{_RP}(centroid_weighted_local, spacing=...) then "
        "spacr.measure._rename_3d_centroids",
        "3-D only. Position within the object, so unlike centroid_weighted_z "
        "it does not encode where in the volume the object sits.",
    ),
    "centroid_weighted_local_y": PropertyInfo(
        "moment",
        "Row (y) coordinate of the intensity-weighted centroid, measured "
        "relative to the corner of the object's bounding box.",
        _PX_3D_ONLY,
        f"{_RP}(centroid_weighted_local, spacing=...) then "
        "spacr.measure._rename_3d_centroids",
        "3-D only; the 2-D equivalent is centroid_weighted_local-0.",
    ),
    "centroid_weighted_local_x": PropertyInfo(
        "moment",
        "Column (x) coordinate of the intensity-weighted centroid, measured "
        "relative to the corner of the object's bounding box.",
        _PX_3D_ONLY,
        f"{_RP}(centroid_weighted_local, spacing=...) then "
        "spacr.measure._rename_3d_centroids",
        "3-D only; the 2-D equivalent is centroid_weighted_local-1.",
    ),
    # ---------------- extended intensity (measure.py:483-539)
    "integrated_intensity": PropertyInfo(
        "intensity",
        "Sum of this channel's pixel values over the object — total signal, "
        "so it scales with both brightness and object size.",
        _INTENSITY_SUM,
        "numpy.sum of region.intensity_image[region.image] in "
        "spacr.measure._extended_regionprops_table",
        None,
    ),
    "std_intensity": PropertyInfo(
        "intensity",
        "Population standard deviation of this channel's pixel values inside "
        "the object.",
        _INTENSITY,
        "numpy.std in spacr.measure._extended_regionprops_table",
        None,
    ),
    "median_intensity": PropertyInfo(
        "intensity",
        "Median pixel value of this channel inside the object.",
        _INTENSITY,
        "numpy.median in spacr.measure._extended_regionprops_table",
        "There is no percentile_50 column for the object interior; this is it.",
    ),
    "skew_intensity": PropertyInfo(
        "intensity",
        "Skewness of the object's intensity histogram. Positive means a tail "
        "of bright pixels (punctate signal); near zero means symmetric.",
        _DIMLESS,
        "scipy.stats.skew in spacr.measure._extended_regionprops_table",
        "NaN for objects with 2 or fewer pixels.",
    ),
    "kurtosis_intensity": PropertyInfo(
        "intensity",
        "Excess kurtosis (Fisher definition, 0 for a Gaussian) of the "
        "object's intensity histogram. High values mean a few very bright "
        "outlier pixels.",
        _DIMLESS,
        "scipy.stats.kurtosis in spacr.measure._extended_regionprops_table",
        "NaN for objects with 3 or fewer pixels.",
    ),
    "mode_intensity": PropertyInfo(
        "intensity",
        "Most frequent pixel value inside the object (the smallest one when "
        "several values tie).",
        _INTENSITY,
        "scipy.stats.mode in spacr.measure._extended_regionprops_table",
        "WAS BROKEN, NOW FIXED: the old code did mode(...).mode[0], but from "
        "SciPy 1.11 onwards mode() returns a bare scalar for 1-D input, so the "
        "subscript raised and a bare except wrote NaN — this column is NaN for "
        "every object in any database written before the numpy.atleast_1d fix. "
        "Check whether the column is all-NaN before using it on an old "
        "database.",
    ),
    "range_intensity": PropertyInfo(
        "intensity",
        "Peak-to-peak intensity inside the object: max minus min.",
        _INTENSITY,
        "numpy.ptp in spacr.measure._extended_regionprops_table",
        None,
    ),
    "iqr_intensity": PropertyInfo(
        "intensity",
        "Interquartile range of the object's pixel values (75th minus 25th "
        "percentile) — an outlier-resistant spread measure.",
        _INTENSITY,
        "numpy.percentile(75) - numpy.percentile(25) in "
        "spacr.measure._extended_regionprops_table",
        None,
    ),
    "cv_intensity": PropertyInfo(
        "intensity",
        "Coefficient of variation of the object's pixel values: standard "
        "deviation divided by mean. Scale-free measure of how uneven the "
        "staining is.",
        _DIMLESS,
        "numpy.std / numpy.mean in spacr.measure._extended_regionprops_table",
        "NaN when the object's mean intensity is exactly 0.",
    ),
    "gini_intensity": PropertyInfo(
        "intensity",
        "Gini coefficient of the object's pixel values: 0 when every pixel is "
        "equally bright, approaching 1 when the signal is concentrated in a "
        "few pixels. A concentration measure for punctate phenotypes.",
        _DIMLESS + ", in [0, 1]",
        "spacr.measure._extended_regionprops_table._gini",
        "Computed on absolute values with NaNs dropped; NaN when the pixel "
        "values sum to 0.",
    ),
    "frac_high90": PropertyInfo(
        "intensity",
        "Fraction of the object's pixels above the WHOLE FIELD's 90th "
        "percentile in this channel — how much of the object is bright "
        "relative to the image it sits in. Near 0 for a dim object, near 1 for "
        "a bright one.",
        _FRACTION,
        "numpy.mean(intens > numpy.percentile(field, 90)) in "
        "spacr.measure._extended_regionprops_table",
        "The reference is the field, so values ARE comparable between objects "
        "of one field but not across fields with different exposure. Databases "
        "written before this fix thresholded on the object's OWN 90th "
        "percentile, which pins the value at about 0.10 for any continuous "
        "distribution and reports quantisation rather than brightness; a "
        "column that is ~0.10 for every object is one of those.",
    ),
    "frac_low10": PropertyInfo(
        "intensity",
        "Fraction of the object's pixels below the WHOLE FIELD's 10th "
        "percentile in this channel — how much of the object is dim relative "
        "to the image it sits in.",
        _FRACTION,
        "numpy.mean(intens < numpy.percentile(field, 10)) in "
        "spacr.measure._extended_regionprops_table",
        "Same reference and same caveats as frac_high90, including the "
        "near-constant ~0.10 seen in databases written before the field-"
        "referenced fix. NaN when the field is entirely NaN.",
    ),
    "entropy_intensity": PropertyInfo(
        "intensity",
        "Shannon entropy (base 2) of the object's intensity values — how "
        "diverse the intensity histogram is.",
        "bits",
        "skimage.measure.shannon_entropy in "
        "spacr.measure._extended_regionprops_table",
        "Computed on the flattened pixel vector, so it describes the "
        "histogram, NOT spatial texture; pixel positions are irrelevant to it. "
        "Returns 0.0 for objects of 1 pixel.",
    ),
    "percentile_<p>": PropertyInfo(
        "intensity",
        "The {p}th percentile of this channel's pixel values inside the "
        "object.",
        _INTENSITY,
        "numpy.percentile in spacr.measure._extended_regionprops_table",
        "Emitted for p in 5, 10, 25, 75, 85, 95 (measure.py:534). There is no "
        "percentile_50 — the median is median_intensity.",
    ),
    # ---------------- texture
    "homogeneity_distance_<d>": PropertyInfo(
        "texture",
        "Grey-level co-occurrence matrix homogeneity (inverse difference "
        "moment) at a pixel offset of {d}. High for smooth, evenly filled "
        "objects; low for grainy or punctate ones.",
        _DIMLESS + ", in [0, 1]",
        "skimage.feature.graycomatrix + graycoprops(glcm, 'homogeneity') in "
        "spacr.measure._calculate_homogeneity",
        "Only the 0 degree (horizontal) direction is used, so the value is "
        "orientation sensitive. Each object's bounding-box patch is "
        "contrast-stretched to 0-255 independently before the GLCM, so "
        "absolute brightness is discarded and values are not comparable to raw "
        "intensity. Pixels inside the bounding box but outside the object are "
        "set to 0, which adds an artificial object/background edge. Offsets "
        "larger than the object carry no signal. Offsets come from the "
        "homogeneity_distances setting (default [8, 16, 32]). 2-D only: "
        "skimage.feature.graycomatrix takes a 2-D image, and a co-occurrence "
        "matrix of a volume is a different construction (13 direction pairs, "
        "not 4), so a 3-D run writes no homogeneity_distance_* columns rather "
        "than reporting one arbitrary slice's texture under this name.",
    ),
    "blur": PropertyInfo(
        "texture",
        "Variance of the discrete Laplacian over the object's own pixels — a "
        "focus/sharpness score, where low values mean blurry.",
        "squared native image intensity units",
        "cv2.Laplacian on the object's bounding-box patch, variance over the "
        "eroded interior, in spacr.measure._estimate_blur",
        "The patch is the RAW image (zero-filling outside the object would put "
        "a step edge at the boundary whose second derivative dwarfs the "
        "texture) and the variance is taken over the mask eroded by one pixel, "
        "so every sampled value is determined by in-object pixels only; "
        "one-pixel-wide objects fall back to the un-eroded mask. On a 3-D "
        "volume the kernel is applied plane by plane in xy, where focus is "
        "defined — a single call on a (Z, Y, X) array does not raise but "
        "differentiates in the zy plane. Databases written before this fix "
        "passed a 1-D vector of the object's pixels in raster order, which "
        "OpenCV treats as an Nx1 image, so the old column is a second "
        "difference along raster order rather than a focus measure.",
    ),
    # ---------------- colocalisation (measure.py:665-708)
    "Pearson_correlation": PropertyInfo(
        "correlation",
        "Pearson correlation coefficient between the two channels over the "
        "object's pixels: +1 when the two signals rise and fall together, 0 "
        "when unrelated, -1 when anti-correlated.",
        _DIMLESS + ", in [-1, 1]",
        "scipy.stats.pearsonr in spacr.measure._calculate_correlation_object_level",
        "NaN for objects with fewer than 2 pixels. Uses every pixel of the "
        "object with no thresholding, so it is dominated by the shared "
        "object-vs-background gradient in dim objects.",
    ),
    "M1_correlation_<t>": PropertyInfo(
        "correlation",
        "Overlap coefficient for the FIRST channel of the pair at percentile "
        "{t}: the fraction of that channel's total intensity in the object "
        "that lies in pixels where BOTH channels exceed their own {t}th "
        "percentile within this object.",
        _FRACTION,
        "spacr.measure._calculate_correlation_object_level (manders_thresholds)",
        "Not the classical Manders M1: classical M1 thresholds only the other "
        "channel and uses an absolute threshold, whereas spaCR thresholds both "
        "channels at the same percentile of each object's own distribution. "
        "Values are therefore self-normalised per object and bounded above by "
        "roughly the fraction of intensity above that percentile. Returns 0 "
        "when the channel's total object intensity is 0. Percentiles come from "
        "the manders_thresholds setting (default [15, 85, 95]).",
    ),
    "M2_correlation_<t>": PropertyInfo(
        "correlation",
        "Overlap coefficient for the SECOND channel of the pair at percentile "
        "{t}: the fraction of that channel's total intensity in the object "
        "that lies in pixels where BOTH channels exceed their own {t}th "
        "percentile within this object.",
        _FRACTION,
        "spacr.measure._calculate_correlation_object_level (manders_thresholds)",
        "Same caveats as M1_correlation_<t>; the two differ only in which "
        "channel's intensity is summed over the shared overlap mask.",
    ),
    # ---------------- periphery / outside rings (measure.py:561-603)
    "periphery_mean": PropertyInfo(
        "intensity",
        "Mean intensity of this channel along the object's own outer rim (the "
        "single-pixel boundary band inside the object).",
        _INTENSITY,
        "skimage.segmentation.find_boundaries intersected with the object, in "
        "spacr.measure._periphery_intensity",
        "Only emitted for nucleus, pathogen and organelle objects "
        "(measure.py:383). NaN when the rim is empty.",
    ),
    "periphery_<p>_percentile": PropertyInfo(
        "intensity",
        "{p}th percentile of this channel's intensity along the object's own "
        "outer rim (the single-pixel boundary band inside the object).",
        _INTENSITY,
        "numpy.percentile over the boundary band in "
        "spacr.measure._periphery_intensity",
        "Emitted for p in 5, 10, 25, 50, 75, 85, 95. Only for nucleus, "
        "pathogen and organelle objects. Note the word order is the reverse of "
        "the object-interior percentiles, which are named percentile_<p>.",
    ),
    "outside_mean": PropertyInfo(
        "intensity",
        "Mean intensity of this channel in a ring extending 5 px outward from "
        "the object — the local surround.",
        _INTENSITY,
        "scipy.ndimage.binary_dilation(iterations=5) — in 3-D a "
        "distance_transform_edt(sampling=spacing) thresholded at 5 xy pixels — "
        "minus the object, in spacr.measure._outside_intensity",
        "Only emitted for nucleus, pathogen and organelle objects. The ring is "
        "NOT masked against neighbouring objects, so for crowded fields it can "
        "include signal from adjacent cells or pathogens. NaN when the ring is "
        "empty. The ring width is 5 xy pixels in both 2-D and 3-D: iterated "
        "dilation on a stack would grow the shell dz/dxy times further in z, "
        "so the 3-D ring is built from a sampled distance transform instead.",
    ),
    "outside_<p>_percentile": PropertyInfo(
        "intensity",
        "{p}th percentile of this channel's intensity in a ring extending 5 px "
        "outward from the object.",
        _INTENSITY,
        "numpy.percentile over the dilation ring (3-D: the sampled "
        "distance-transform ring) in spacr.measure._outside_intensity",
        "Emitted for p in 5, 10, 25, 50, 75, 85, 95. Only for nucleus, "
        "pathogen and organelle objects. The ring is not masked against "
        "neighbouring objects. Note the reversed word order compared with the "
        "object-interior percentile_<p> columns.",
    ),
    # ---------------- radial distribution (measure.py:438-449, 605-663)
    "rad_dist_channel_<c>_bin_<b>": PropertyInfo(
        "intensity",
        "Mean intensity of channel {c} in radial shell {b} of 6, measured by "
        "distance outward from this object's boundary and restricted to the "
        "parent cell — a profile of how the channel falls off with distance "
        "from the object.",
        _INTENSITY,
        "spacr.measure._calculate_radial_distribution + "
        "spacr.measure._create_dataframe",
        "bin_0 IS NOT AN INNER SHELL: the distance map is multiplied by the "
        "parent-cell mask, so every pixel outside the cell is 0 and falls into "
        "bin 0 together with the object boundary. bin_0 is therefore dominated "
        "by the whole field's background. Bins 1-5 lie inside the cell. Shell "
        "width is (max distance inside that cell)/6, so it differs per object "
        "and the bins are not comparable between objects of different size. "
        "Emitted for nucleus, pathogen and organelle when the radial_dist "
        "setting is on. In 3-D the distance map is sampled with the voxel "
        "spacing, so the shells are physical shells rather than voxel counts; "
        "the bin values themselves stay in intensity units either way.",
    ),
    # ---------------- intensity-weighted distances (measure.py:733-796)
    "distance_to_nucleus": PropertyInfo(
        "morphology",
        "Distance from this channel's intensity-weighted centre of mass "
        "within the cell to the nearest nucleus pixel. Small values mean the "
        "channel's signal piles up on or near the nucleus.",
        _PX,
        "scipy.ndimage.center_of_mass on a Gaussian-blurred channel, then "
        "scipy.ndimage.distance_transform_edt(sampling=spacing) of the nucleus "
        "mask, in spacr.measure._measure_intensity_distance",
        "Only emitted for the cell object, and only when the "
        "distance_gaussian_sigma setting is a non-zero int. 0 when the centre "
        "of mass lands inside a nucleus. The distance transform is global, so "
        "the nearest nucleus may belong to a neighbouring cell. NaN when the "
        "centre of mass is undefined or falls outside the image. In 3-D the "
        "transform is sampled with the voxel spacing, so a step in z counts as "
        "dz and not as one pixel.",
    ),
    "distance_to_pathogen": PropertyInfo(
        "morphology",
        "Distance from this channel's intensity-weighted centre of mass "
        "within the cell to the nearest pathogen pixel.",
        _PX,
        "scipy.ndimage.center_of_mass on a Gaussian-blurred channel, then "
        "scipy.ndimage.distance_transform_edt(sampling=spacing) of the "
        "pathogen mask, in spacr.measure._measure_intensity_distance",
        "Only emitted for the cell object, and only when the "
        "distance_gaussian_sigma setting is a non-zero int. 0 when the centre "
        "of mass lands inside a pathogen. The distance transform is global, so "
        "the nearest pathogen may belong to a neighbouring cell. Sampled with "
        "the voxel spacing in 3-D.",
    ),
    # ---------------- organelle summaries (measure.py:250-330, 1046-1062)
    "organelle_summary_organelle_count": PropertyInfo(
        "morphology",
        "Number of organelle objects assigned to this parent object.",
        "count",
        "spacr.measure._summarize_organelles_per_parent",
        "Assignment is by maximum overlap (spacr.measure._map_child_to_parent), "
        "so an organelle straddling two parents is given entirely to one.",
    ),
    "organelle_summary_organelle_total_area": PropertyInfo(
        "morphology",
        "Summed area of all organelles assigned to this parent object — a "
        "summed volume in a 3-D run.",
        _PX2,
        "spacr.measure._summarize_organelles_per_parent",
        "Summed from the same spaced regionprops `area` as <object>_area, so "
        "it follows the row's measurement_units exactly as that column does.",
    ),
    "organelle_summary_organelle_fraction": PropertyInfo(
        "morphology",
        "Total organelle area divided by the parent object's area — the "
        "fraction of the parent occupied by organelles.",
        _FRACTION,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent area is 0. A ratio of two equally-spaced "
        "quantities, so it is a volume fraction in 3-D and needs no unit "
        "either way.",
    ),
    "organelle_summary_organelle_mean_area": PropertyInfo(
        "morphology",
        "Mean area of the organelles assigned to this parent object.",
        _PX2,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has no organelles.",
    ),
    "organelle_summary_organelle_std_area": PropertyInfo(
        "morphology",
        "Standard deviation of organelle areas within this parent object.",
        _PX2,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has fewer than 2 organelles.",
    ),
    "organelle_summary_organelle_mean_eccentricity": PropertyInfo(
        "morphology",
        "Mean eccentricity of the organelles assigned to this parent object.",
        _DIMLESS + ", in [0, 1)",
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has no organelles. 2-D only — skimage does not "
        "define eccentricity for a 3-D region, so a 3-D run omits this column "
        "and its std_ counterpart.",
    ),
    "organelle_summary_organelle_std_eccentricity": PropertyInfo(
        "morphology",
        "Standard deviation of organelle eccentricity within this parent.",
        _DIMLESS,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has fewer than 2 organelles. 2-D only; absent "
        "from a 3-D run.",
    ),
    "organelle_summary_organelle_mean_solidity": PropertyInfo(
        "morphology",
        "Mean solidity of the organelles assigned to this parent object.",
        _DIMLESS + ", in (0, 1]",
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has no organelles.",
    ),
    "organelle_summary_organelle_std_solidity": PropertyInfo(
        "morphology",
        "Standard deviation of organelle solidity within this parent.",
        _DIMLESS,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has fewer than 2 organelles.",
    ),
    "organelle_summary_organelle_mean_major_axis": PropertyInfo(
        "morphology",
        "Mean major-axis length of the organelles assigned to this parent.",
        _PX,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has no organelles.",
    ),
    "organelle_summary_organelle_mean_minor_axis": PropertyInfo(
        "morphology",
        "Mean minor-axis length of the organelles assigned to this parent.",
        _PX,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has no organelles.",
    ),
    "organelle_summary_organelle_ch<c>_mean_intensity_per_<parent>": PropertyInfo(
        "intensity",
        "Mean over this {parent}'s organelles of each organelle's own mean "
        "intensity in channel {c}.",
        _INTENSITY,
        "spacr.measure._summarize_organelles_per_parent",
        "A mean of per-organelle means, so large and small organelles are "
        "weighted equally. 0.0 when the parent has no organelles. Note this "
        "family spells the channel ch<c>, not channel_<c>, unlike every other "
        "feature family.",
    ),
    "organelle_summary_organelle_ch<c>_std_intensity_per_<parent>": PropertyInfo(
        "intensity",
        "Standard deviation across this {parent}'s organelles of their "
        "individual mean intensities in channel {c}.",
        _INTENSITY,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has fewer than 2 organelles. Note the ch<c> "
        "channel spelling.",
    ),
    # ---------------- cytoskeleton (measure.py:82-147)
    "skeleton_length": PropertyInfo(
        "morphology",
        "Total pixel count of the morphological skeleton of the thresholded "
        "cytoskeleton signal inside the object — a proxy for filament length.",
        "px (pixels; an unspaced skeleton pixel count)",
        "skimage.morphology.skeletonize + regionprops area sum in "
        "spacr.measure._analyze_cytoskeleton",
        "_analyze_cytoskeleton is defined in measure.py but is not called by "
        "measure_crop in this version, so this column is not produced by a "
        "standard run. Thresholding is local (block size 35) with a "
        "per-object adaptive offset. It takes no spacing, so unlike the other "
        "length columns this one stays a raw pixel count whatever "
        "measurement_units says.",
    ),
    "skeleton_branch_points": PropertyInfo(
        "morphology",
        "Number of skeleton pixels with 3 or more skeleton neighbours — a "
        "count of filament branch points.",
        "count",
        "8-neighbour convolution of the skeleton in "
        "spacr.measure._analyze_cytoskeleton",
        "_analyze_cytoskeleton is not called by measure_crop in this version, "
        "so this column is not produced by a standard run.",
    ),
}


# --------------------------------------------------------------------------
# metadata columns
# --------------------------------------------------------------------------

_META_UTILS = "spacr.utils._merge_and_save_to_database"
_META_WELLS = "spacr.utils._map_wells (parsed from the stack file name)"
_META_WELLS_PNG = "spacr.utils._map_wells_png (parsed from the PNG file name)"

#: Exact-match metadata columns written to ``measurements.db`` by
#: :func:`spacr.utils._merge_and_save_to_database`,
#: :func:`spacr.utils.filepaths_to_database`, :func:`spacr.io._save_settings_to_db`
#: and :func:`spacr.utils._save_object_counts_to_database`.
META_COLUMNS: dict[str, PropertyInfo] = {
    "object_label": PropertyInfo(
        "meta",
        "Integer label of this object in its own mask, and the key the "
        "morphology and intensity tables are joined on.",
        None,
        "spacr.utils._check_integrity (first label column of the frame)",
        "Unique only within one field of view; combine with prcf/prcfo to get "
        "a globally unique object identifier.",
    ),
    "label": PropertyInfo(
        "meta",
        "Raw regionprops label of the object before spaCR collapses the "
        "duplicate label columns into object_label.",
        None,
        "skimage.measure.regionprops_table(label)",
        "Normally consumed by spacr.utils._check_integrity and absent from the "
        "database; if you see it, the frame was inspected before that step.",
    ),
    "label_list": PropertyInfo(
        "meta",
        "String repr of every label column that was collapsed into "
        "object_label, kept for traceability.",
        None,
        "spacr.utils._check_integrity",
        None,
    ),
    "label_list_x": PropertyInfo(
        "meta",
        "label_list from the morphology frame, before it is renamed to "
        "label_list_morphology.",
        None,
        "pandas.merge suffix in " + _META_UTILS,
        "Renamed to label_list_morphology before the database write; seeing "
        "this name means the frame was captured mid-merge.",
    ),
    "label_list_y": PropertyInfo(
        "meta",
        "label_list from the intensity frame, before it is renamed to "
        "label_list_intensity.",
        None,
        "pandas.merge suffix in " + _META_UTILS,
        "Renamed to label_list_intensity before the database write.",
    ),
    "label_list_morphology": PropertyInfo(
        "meta",
        "label_list carried over from the morphology frame of the merge.",
        None,
        _META_UTILS + " (rename of label_list_x)",
        None,
    ),
    "label_list_intensity": PropertyInfo(
        "meta",
        "label_list carried over from the intensity frame of the merge.",
        None,
        _META_UTILS + " (rename of label_list_y)",
        None,
    ),
    "cell_id": PropertyInfo(
        "meta",
        "Label of the parent cell this row belongs to.",
        None,
        "spacr.measure._create_dataframe (object tables) / "
        "spacr.utils._map_wells_png (png_list)",
        "Two different things depending on the table. In the nucleus / "
        "pathogen / organelle tables it is an integer cell label supplied by "
        "the radial-distribution frame, so it is only present when the "
        "radial_dist setting was on. In png_list it is the 'oN' object-id "
        "string parsed out of the cropped PNG's file name.",
    ),
    "nucleus_id": PropertyInfo(
        "meta",
        "Object id ('oN') parsed from a nucleus-cropped PNG's file name.",
        None,
        _META_WELLS_PNG,
        "png_list table only.",
    ),
    "pathogen_id": PropertyInfo(
        "meta",
        "Object id ('oN') parsed from a pathogen-cropped PNG's file name.",
        None,
        _META_WELLS_PNG,
        "png_list table only.",
    ),
    "cytoplasm_id": PropertyInfo(
        "meta",
        "Object id ('oN') parsed from a cytoplasm-cropped PNG's file name.",
        None,
        _META_WELLS_PNG,
        "png_list table only.",
    ),
    "plateID": PropertyInfo(
        "meta",
        "Plate identifier, the first underscore-separated field of the source "
        "file name.",
        None,
        _META_WELLS,
        "Renamed from the legacy column 'plate' by "
        "spacr.utils.rename_columns_in_db.",
    ),
    "rowID": PropertyInfo(
        "meta",
        "Well row, as 'r<n>' with A=1 (or the raw well string when the well "
        "does not start with a letter).",
        None,
        _META_WELLS,
        "Renamed from the legacy column 'row'.",
    ),
    "columnID": PropertyInfo(
        "meta",
        "Well column, as 'c<n>'.",
        None,
        _META_WELLS,
        "Renamed from the legacy columns 'column' / 'col'.",
    ),
    "fieldID": PropertyInfo(
        "meta",
        "Field of view within the well, as 'f<n>'.",
        None,
        _META_WELLS,
        "Renamed from the legacy column 'field'.",
    ),
    "timeID": PropertyInfo(
        "meta",
        "Timepoint of a timelapse acquisition, as 't<n>'.",
        None,
        _META_WELLS,
        "Only present when measure_crop ran with timelapse=True. Written by "
        "every object table and, since the two spellings were unified, by "
        "png_list too.",
    ),
    "time_id": PropertyInfo(
        "meta",
        "Timepoint of a timelapse acquisition, as 't<n>', in the png_list "
        "table of a database written before the two spellings were unified.",
        None,
        _META_WELLS_PNG,
        "The same quantity as timeID. png_list used to be written with this "
        "spelling while every object table used timeID, so _split_data raised "
        "KeyError('timeID') on png_list and built no prcft, and any join "
        "between png_list and an object table on time matched nothing. "
        "filepaths_to_database now writes timeID, and "
        "spacr.utils.rename_columns_in_db migrates an existing database in "
        "place the first time it is read — so this column should only be seen "
        "on a database that has not been opened since.",
    ),
    "chanID": PropertyInfo(
        "meta",
        "Channel identifier column in legacy databases.",
        None,
        "spacr.utils.rename_columns_in_db (renamed from 'channel')",
        None,
    ),
    "prcf": PropertyInfo(
        "meta",
        "Field key: plate_row_column_field (plus _time for timelapse). "
        "Identifies one field of view.",
        None,
        _META_WELLS,
        None,
    ),
    "prcfo": PropertyInfo(
        "meta",
        "Object key: prcf plus the object id, e.g. plate1_r2_c3_f4_o17. The "
        "unique identifier of a single measured object across the experiment.",
        None,
        "spacr.utils._map_wells_png / spacr.utils._split_data",
        "For nucleus and pathogen rows the object part is the PARENT cell id, "
        "not the child's own label, so child rows sharing a cell share a "
        "prcfo.",
    ),
    "prcft": PropertyInfo(
        "meta",
        "Field key including the timepoint: plate_row_column_field_time.",
        None,
        "spacr.utils._split_data",
        "Only built when a timepoint column exists, in either spelling "
        "(timeID or the legacy time_id).",
    ),
    "prc": PropertyInfo(
        "meta",
        "Well key: plate_row_column, i.e. prcf without the field.",
        None,
        "spacr.io._read_and_merge_data / spacr.utils",
        None,
    ),
    "file_name": PropertyInfo(
        "meta",
        "Name of the merged .npy stack (object tables) or of the cropped PNG "
        "(png_list) this row came from.",
        None,
        _META_UTILS + " / spacr.utils.filepaths_to_database",
        None,
    ),
    "path_name": PropertyInfo(
        "meta",
        "Absolute path of the merged .npy stack this row was measured from.",
        None,
        _META_UTILS,
        None,
    ),
    "png_path": PropertyInfo(
        "meta",
        "Path of the cropped single-object PNG.",
        None,
        "spacr.utils.filepaths_to_database",
        "png_list table only.",
    ),
    "object": PropertyInfo(
        "meta",
        "Object id ('oN') parsed from an activation-map PNG's file name.",
        None,
        "spacr.utils.activation_maps_to_database",
        "Appears in the <cam_type>_list / <cam_type>_correlations tables of a "
        "dataset database, not in measurements.db itself.",
    ),
    "count_type": PropertyInfo(
        "meta",
        "Which object type a row of the object_counts table counts, e.g. "
        "'cell' or 'nucleus' plus any suffix supplied at write time.",
        None,
        "spacr.utils._save_object_counts_to_database",
        "object_counts table only.",
    ),
    "object_count": PropertyInfo(
        "meta",
        "Number of objects of count_type found in that file.",
        "count",
        "spacr.utils._save_object_counts_to_database",
        "object_counts table only. Counted from the mask, excluding label 0.",
    ),
    "setting_key": PropertyInfo(
        "meta",
        "Name of a pipeline setting recorded when the run started.",
        None,
        "spacr.io._save_settings_to_db",
        "settings table only.",
    ),
    "setting_value": PropertyInfo(
        "meta",
        "Value of that setting, stringified.",
        None,
        "spacr.io._save_settings_to_db",
        "settings table only. Everything is stored as text, including lists "
        "and None.",
    ),
    "nucleus_prcfo_count": PropertyInfo(
        "meta",
        "Number of nuclei that shared this prcfo before the nuclei_limit "
        "filter was applied.",
        "count",
        "spacr.io._read_and_merge_data",
        "Added during analysis, not by measure_crop.",
    ),
    "pathogen_prcfo_count": PropertyInfo(
        "meta",
        "Number of pathogens assigned to this prcfo.",
        "count",
        "spacr.io._read_and_merge_data",
        "Added during analysis, not by measure_crop. Useful as an infection "
        "burden readout.",
    ),
    "count_nucleus": PropertyInfo(
        "meta",
        "Number of nucleus rows aggregated into this cell row.",
        "count",
        "spacr.io._read_and_join_tables",
        None,
    ),
    "count_pathogen": PropertyInfo(
        "meta",
        "Number of pathogen rows aggregated into this cell row.",
        "count",
        "spacr.io._read_and_join_tables",
        None,
    ),
    "condition": PropertyInfo(
        "meta",
        "Experimental condition annotation merged in from the run metadata.",
        None,
        "spacr.utils._update_database_with_merged_info",
        "Experiment annotation, not a measurement.",
    ),
    "treatment": PropertyInfo(
        "meta",
        "Treatment annotation merged in from the run metadata.",
        None,
        "spacr.utils._update_database_with_merged_info",
        "Experiment annotation, not a measurement.",
    ),
    "host_cells": PropertyInfo(
        "meta",
        "Host cell line annotation merged in from the run metadata.",
        None,
        "spacr.utils._update_database_with_merged_info",
        "Experiment annotation, not a measurement.",
    ),
    "pathogen": PropertyInfo(
        "meta",
        "Pathogen strain annotation merged in from the run metadata.",
        None,
        "spacr.utils._update_database_with_merged_info",
        "Experiment annotation, NOT the pathogen object — do not confuse it "
        "with the pathogen_* feature columns.",
    ),
    "test": PropertyInfo(
        "meta",
        "Manual annotation class recorded by the spaCR annotation app.",
        None,
        "spacr.gui_utils (ALTER TABLE png_list ADD COLUMN <annotation_column>)",
        "'test' is only the default annotation column name; a run may use any "
        "name, in which case the column will be reported as unknown here.",
    ),
    "train": PropertyInfo(
        "meta",
        "Manual annotation class recorded by the spaCR annotation app.",
        None,
        "spacr.gui_utils (ALTER TABLE png_list ADD COLUMN <annotation_column>)",
        "Annotation column names are user-chosen; see 'test'.",
    ),
    # ---------------- measurement provenance stamp
    # spacr.measure.MEASUREMENT_STAMP_COLUMNS, written onto every row of every
    # object table by spacr.utils._merge_and_save_to_database. These five are
    # what makes the conditional units above resolvable: measure.py records the
    # unit on the row instead of renaming <object>_area, so the columns below
    # are the only thing that says which quantity that column holds.
    "measurement_ndim": PropertyInfo(
        "meta",
        "Number of spatial dimensions this row was measured in: 2 for a flat "
        "field, 3 for a (Z, Y, X) volume.",
        None,
        "spacr.measure.resolve_measurement_spacing -> "
        "spacr.utils._merge_and_save_to_database",
        "READ THIS BEFORE COMPARING GEOMETRIC COLUMNS. When it is 3, "
        "<object>_area is a VOLUME, equivalent_diameter_area is a cube root, "
        "euler_number counts cavities, and eccentricity / perimeter / "
        "zernike_* / homogeneity_distance_* are absent. A row with no stamp at "
        "all is 2-D: every spaCR release before 3-D measurement existed "
        "crashed outright on a 3-D mask, so nothing else can be in an old "
        "table. spacr.utils refuses to append rows whose (ndim, units) differ "
        "from those already in the table, so within one table this is constant.",
    ),
    "measurement_units": PropertyInfo(
        "meta",
        "Which units the geometric columns of this row are in: 'px' (2-D, "
        "pixels), 'px_xy' (3-D measured with anisotropy alone — correct "
        "geometry in xy-pixel units) or 'um' (3-D measured with "
        "voxel_size_z_um + voxel_size_xy_um — micrometres).",
        None,
        "spacr.measure.resolve_measurement_spacing -> "
        "spacr.utils._merge_and_save_to_database",
        "The source of truth for the unit of every length/area/volume column "
        "in the row; spacr.feature_dict.describe_database reads it out of the "
        "database and resolves the units it reports accordingly. A 2-D run "
        "always writes 'px' even when a voxel size is configured, because "
        "applying it would turn every *_area from px^2 into um^2 under an "
        "unchanged name. NULL/absent means 'px'.",
    ),
    "n_z": PropertyInfo(
        "meta",
        "Number of z planes behind this measurement; 1 for a 2-D field.",
        "count",
        "spacr.measure.measure_crop (data.shape[0] of a (Z, Y, X, C) stack) -> "
        "spacr.utils._merge_and_save_to_database",
        "A (1, Y, X, C) stack is squeezed to the 2-D path before measuring, so "
        "n_z = 1 always comes with measurement_ndim = 2.",
    ),
    "voxel_size_z_um": PropertyInfo(
        "meta",
        "The z step of the acquisition, in micrometres, as configured for this "
        "run.",
        "um",
        "settings['voxel_size_z_um'] via "
        "spacr.measure.resolve_measurement_spacing",
        "NULL unless the run was 3-D AND both voxel sizes were given, which is "
        "exactly the case measurement_units='um'. This is a setting recorded "
        "for provenance, not a measurement of the object.",
    ),
    "voxel_size_xy_um": PropertyInfo(
        "meta",
        "The width of one pixel in the image plane, in micrometres, as "
        "configured for this run (pixels assumed square).",
        "um",
        "settings['voxel_size_xy_um'] via "
        "spacr.measure.resolve_measurement_spacing",
        "NULL unless the run was 3-D AND both voxel sizes were given "
        "(measurement_units='um'). NOT the same setting as um_per_pixel: "
        "um_per_pixel only converts scale_bar_length_um into pixels when a "
        "scale bar is drawn on a figure and never reaches a measurement, "
        "whereas this one is handed to regionprops as `spacing` — but only on "
        "a 3-D run.",
    ),
}

#: The per-row provenance stamp, in the order :mod:`spacr.measure` declares it.
#: Mirrors :data:`spacr.measure.MEASUREMENT_STAMP_COLUMNS` and
#: :data:`spacr.utils.MEASUREMENT_STAMP_COLUMNS`; every name here is also a key
#: of :data:`META_COLUMNS`.
MEASUREMENT_STAMP_COLUMNS: tuple[str, ...] = (
    "measurement_ndim",
    "measurement_units",
    "n_z",
    "voxel_size_z_um",
    "voxel_size_xy_um",
)

# Per-object-type parent/child link columns produced by the morphology merge:
# measure.py:183 (nucleus <- cell_to_nucleus), 194 (pathogen <- cell_to_pathogen)
# and 208 (organelle <- _map_child_to_parent), all prefixed at measure.py:225.
_LINK_COLUMNS: dict[str, PropertyInfo] = {
    "cell_id": PropertyInfo(
        "meta",
        "Label of the parent cell that encloses this object, from the "
        "morphology frame's cell/child mapping.",
        None,
        "spacr.measure.get_components -> merge in "
        "spacr.measure._morphological_measurements",
        "Distinct from the bare cell_id column, which comes from the "
        "radial-distribution frame. NaN when the child is not inside any cell.",
    ),
    "nucleus": PropertyInfo(
        "meta",
        "The nucleus label used as the join key in the cell/nucleus mapping — "
        "a duplicate of object_label.",
        None,
        "spacr.measure.get_components -> merge in "
        "spacr.measure._morphological_measurements",
        "Redundant identifier, not a measurement.",
    ),
    "pathogen": PropertyInfo(
        "meta",
        "The pathogen label used as the join key in the cell/pathogen mapping "
        "— a duplicate of object_label.",
        None,
        "spacr.measure.get_components -> merge in "
        "spacr.measure._morphological_measurements",
        "Redundant identifier, not a measurement.",
    ),
    "organelle": PropertyInfo(
        "meta",
        "The organelle label used as the join key in the organelle/parent "
        "mapping — a duplicate of object_label.",
        None,
        "spacr.measure._map_child_to_parent",
        "Redundant identifier, not a measurement.",
    ),
    "cell": PropertyInfo(
        "meta",
        "Label of the parent cell an organelle was assigned to, by maximum "
        "pixel overlap.",
        None,
        "spacr.measure._map_child_to_parent",
        "0 when the organelle overlaps no cell.",
    ),
    "region_label": PropertyInfo(
        "meta",
        "Object label carried by an extended-intensity sub-frame.",
        None,
        "spacr.measure._extended_regionprops_table",
        "Normally dropped by spacr.utils._check_integrity before the database "
        "write, because its name contains 'label'.",
    ),
    "periphery_region_label": PropertyInfo(
        "meta",
        "Object label carried by the periphery-intensity frame.",
        None,
        "spacr.measure._periphery_intensity (first element of each tuple)",
        "Normally dropped by spacr.utils._check_integrity before the database "
        "write, because its name contains 'label'.",
    ),
    "outside_region_label": PropertyInfo(
        "meta",
        "Object label carried by the outside-ring intensity frame.",
        None,
        "spacr.measure._outside_intensity (first element of each tuple)",
        "Normally dropped by spacr.utils._check_integrity before the database "
        "write, because its name contains 'label'.",
    ),
    "label_correlation": PropertyInfo(
        "meta",
        "Object label carried by the colocalisation frame.",
        None,
        "spacr.measure._calculate_correlation_object_level",
        "Normally dropped by spacr.utils._check_integrity before the database "
        "write, because its name contains 'label'.",
    ),
}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_CHANNEL_RE = re.compile(r"^channel_(\d+)_")
_DOUBLE_PREFIX_BLUR_RE = re.compile(
    r"^(?P<obj>cell|nucleus|pathogen|organelle|cytoplasm)_channel_(?P<ch>\d+)_blur$"
)
_PLAIN_BLUR_RE = re.compile(r"^blur$")
_RAD_DIST_RE = re.compile(r"^rad_dist_channel_(?P<c>\d+)_bin_(?P<b>\d+)$")
_ORG_SUMMARY_CH_RE = re.compile(
    r"^organelle_summary_organelle_ch(?P<c>\d+)_(?P<stat>mean|std)_intensity_per_"
    r"(?P<parent>cell|nucleus|pathogen|cytoplasm)$"
)
_DEDUP_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")
# Current form written by spacr.utils._check_integrity. Unambiguous, so it is
# matched BEFORE the legacy positional form above.
_DUP_SUFFIX_RE = re.compile(r"^(?P<base>.+)__dup(?P<idx>\d+)$")

# (regex, KNOWN_PROPERTIES key). Order matters only in that each regex is
# anchored and mutually exclusive.
_PARAMETERIZED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^zernike_(?P<i>\d+)$"), "zernike_<i>"),
    (re.compile(r"^percentile_(?P<p>\d+)$"), "percentile_<p>"),
    (re.compile(r"^homogeneity_distance_(?P<d>\d+)$"), "homogeneity_distance_<d>"),
    (re.compile(r"^periphery_(?P<p>\d+)_percentile$"), "periphery_<p>_percentile"),
    (re.compile(r"^outside_(?P<p>\d+)_percentile$"), "outside_<p>_percentile"),
    (re.compile(r"^M1_correlation_(?P<t>\d+)$"), "M1_correlation_<t>"),
    (re.compile(r"^M2_correlation_(?P<t>\d+)$"), "M2_correlation_<t>"),
)


def _fill(text: str | None, params: dict[str, str]) -> str | None:
    """Substitute ``{name}`` placeholders in a curated description."""
    if text is None:
        return None
    if not params:
        return text
    try:
        return text.format(**params)
    except (KeyError, IndexError, ValueError):
        return text


def _entry(
    column: str,
    info: PropertyInfo,
    *,
    object_type: str | None = None,
    channel: int | None = None,
    channel_2: int | None = None,
    object_type_2: str | None = None,
    params: dict[str, str] | None = None,
    extra_note: str | None = None,
    measurement_units: str | None = None,
) -> FeatureEntry:
    """Build a :class:`FeatureEntry` from a curated :class:`PropertyInfo`."""
    params = params or {}
    notes = _fill(info.notes, params)
    if extra_note:
        notes = f"{extra_note} {notes}" if notes else extra_note
    unit = info.unit
    basis: str | None = None
    if isinstance(unit, ConditionalUnit):
        # Only a conditional unit is affected by the stamp, so a column whose
        # unit is fixed never claims to have been resolved against one.
        basis = measurement_units
        unit = unit.resolve(measurement_units)
    return FeatureEntry(
        column=column,
        object_type=object_type,
        channel=channel,
        family=info.family,
        description=_fill(info.description, params),
        unit=unit,
        computed_by=info.computed_by,
        notes=notes,
        channel_2=channel_2,
        object_type_2=object_type_2,
        measurement_units=basis,
    )


def _unknown(column: str, object_type: str | None, channel: int | None,
             note: str | None = None) -> FeatureEntry:
    """Build the fallback entry for a column this dictionary cannot explain."""
    base = (
        "Not recognised by the spaCR feature dictionary. Its meaning was not "
        "determined from the spaCR source; check the code that wrote the "
        "table, or a user-supplied custom feature "
        "(see spacr.custom_features)."
    )
    return FeatureEntry(
        column=column,
        object_type=object_type,
        channel=channel,
        family="unknown",
        description=None,
        unit=None,
        computed_by="unknown",
        notes=f"{note} {base}" if note else base,
    )


def _lookup_stat(stat: str) -> tuple[PropertyInfo, dict[str, str]] | None:
    """Resolve a stat suffix to a curated definition plus its placeholders."""
    info = KNOWN_PROPERTIES.get(stat)
    if info is not None:
        return info, {}
    for pattern, key in _PARAMETERIZED:
        m = pattern.match(stat)
        if m:
            return KNOWN_PROPERTIES[key], dict(m.groupdict())
    return None


def _parse_organelle_summary(name: str, measurement_units: str | None = None
                             ) -> FeatureEntry | None:
    """Parse the ``organelle_summary_*`` columns written per parent object."""
    if not name.startswith("organelle_summary_"):
        return None
    m = _ORG_SUMMARY_CH_RE.match(name)
    if m:
        key = (
            "organelle_summary_organelle_ch<c>_"
            f"{m.group('stat')}_intensity_per_<parent>"
        )
        return _entry(
            name,
            KNOWN_PROPERTIES[key],
            object_type="organelle",
            channel=int(m.group("c")),
            object_type_2=m.group("parent"),
            params={"c": m.group("c"), "parent": m.group("parent")},
            measurement_units=measurement_units,
        )
    info = KNOWN_PROPERTIES.get(name)
    if info is not None:
        return _entry(name, info, object_type="organelle",
                      measurement_units=measurement_units)
    return _unknown(
        name,
        "organelle",
        None,
        "Column carries the organelle_summary_ prefix written by "
        "spacr.measure._measure_crop_core, but the summary statistic is not in "
        "the dictionary.",
    )


def parse_column(name: str, measurement_units: str | None = None
                 ) -> FeatureEntry:
    """Decompose one ``measurements.db`` column name into a described feature.

    The grammar this implements, derived from the f-strings in
    :mod:`spacr.measure`::

        <object>_<stat>                                  morphology / moments
        <object>_channel_<i>_<stat>                      single-channel intensity/texture
        <object>_channel_<i>_channel_<j>_<stat>          two-channel colocalisation
        <object>_channel_<i>_periphery_<stat>            inner boundary rim
        <object>_channel_<i>_outside_<stat>              5 px surrounding ring
        <object>_rad_dist_channel_<c>_bin_<b>            radial intensity profile
        organelle_summary_<stat>                         per-parent organelle summary
        <object>_volume_voxels / _volume_um3             3-D volumes, named by unit
        <object>_channel_<i>_centroid_weighted_<z|y|x>   3-D centroid, named by axis
        measurement_ndim / measurement_units / n_z /     per-row provenance stamp
        voxel_size_z_um / voxel_size_xy_um

    An unrecognised name never raises and is never dropped: it comes back as a
    :class:`FeatureEntry` with ``family="unknown"`` and ``description=None``.

    :param name: Column name exactly as stored in the database.
    :param measurement_units: The ``measurement_units`` value of the rows being
        described — one of :data:`MEASUREMENT_UNITS`. Geometric columns have no
        single unit any more (a 3-D run measures a volume, in micrometres when
        it was given a voxel size), so pass this when you know it and the
        returned ``unit`` is concrete. Left ``None``, ``unit`` states every
        possibility with the condition attached rather than guessing one;
        :func:`describe_database` fills it in from the database itself.
    :returns: A :class:`FeatureEntry` describing the column.
    """
    if not isinstance(name, str):
        name = str(name)

    # 1. exact metadata match wins over any structural interpretation, so that
    #    e.g. cell_id is an identifier and not a 'cell' feature named 'id'.
    info = META_COLUMNS.get(name)
    if info is not None:
        return _entry(name, info, measurement_units=measurement_units)

    # 2. per-parent organelle summaries, before the object prefix is stripped
    #    (the prefix 'organelle_' would otherwise swallow the family name).
    summary = _parse_organelle_summary(name, measurement_units)
    if summary is not None:
        return summary

    # 3. object prefix (measure.py:225 and measure.py:395)
    object_type: str | None = None
    rest = name
    for obj in OBJECT_TYPES:
        if name.startswith(obj + "_"):
            object_type = obj
            rest = name[len(obj) + 1:]
            break

    if object_type is None:
        return _unknown(name, None, None)

    # 4. parent/child link columns, e.g. nucleus_cell_id, organelle_cell
    link = _LINK_COLUMNS.get(rest)
    if link is not None:
        return _entry(name, link, object_type=object_type,
                      measurement_units=measurement_units)

    # 5. radial distribution: the channel index sits AFTER the family token
    #    (measure.py:444), so it needs its own rule.
    m = _RAD_DIST_RE.match(rest)
    if m:
        return _entry(
            name,
            KNOWN_PROPERTIES["rad_dist_channel_<c>_bin_<b>"],
            object_type=object_type,
            channel=int(m.group("c")),
            params=dict(m.groupdict()),
            measurement_units=measurement_units,
        )

    # 6. up to two channel_<n> infixes (measure.py:395 and measure.py:429)
    channels: list[int] = []
    while True:
        m = _CHANNEL_RE.match(rest)
        if not m or len(channels) == 2:
            break
        channels.append(int(m.group(1)))
        rest = rest[m.end():]

    channel = channels[0] if channels else None
    channel_2 = channels[1] if len(channels) > 1 else None

    # 7. blur, whose emitted name carries the object/channel prefix twice:
    #    measure.py:393 writes '<obj>_channel_<i>_blur' into the frame, then
    #    measure.py:395 prefixes every non-label column with '<obj>_channel_<i>_'
    #    again, so the stored column is '<obj>_channel_<i>_<obj>_channel_<i>_blur'.
    m = _DOUBLE_PREFIX_BLUR_RE.match(rest)
    if m:
        inner_obj, inner_ch = m.group("obj"), int(m.group("ch"))
        mismatch = "" if (inner_obj == object_type and inner_ch == channel) else (
            f" The two copies disagree ({object_type}/channel {channel} versus "
            f"{inner_obj}/channel {inner_ch}) — treat the column with suspicion."
        )
        return _entry(
            name,
            KNOWN_PROPERTIES["blur"],
            object_type=object_type,
            channel=channel if channel is not None else inner_ch,
            extra_note=(
                "The object and channel prefix appears TWICE in this column "
                "name (written at measure.py:393, prefixed again at "
                "measure.py:395); both copies name the same object and "
                f"channel.{mismatch}"
            ),
            measurement_units=measurement_units,
        )

    if channel is not None and _PLAIN_BLUR_RE.match(rest):
        return _entry(name, KNOWN_PROPERTIES["blur"], object_type=object_type,
                      channel=channel, measurement_units=measurement_units)

    link = _LINK_COLUMNS.get(rest)
    if link is not None:
        return _entry(name, link, object_type=object_type, channel=channel,
                      channel_2=channel_2, measurement_units=measurement_units)

    resolved = _lookup_stat(rest)
    if resolved is not None:
        info, params = resolved
        return _entry(name, info, object_type=object_type, channel=channel,
                      channel_2=channel_2, params=params,
                      measurement_units=measurement_units)

    # 8. a pandas merge suffix appended when object tables are joined
    #    (spacr.io._read_and_join_tables uses suffixes=('', '_<entity>')).
    for obj in OBJECT_TYPES:
        if rest.endswith("_" + obj):
            trimmed = rest[: -(len(obj) + 1)]
            resolved = _lookup_stat(trimmed)
            if resolved is not None:
                info, params = resolved
                return _entry(
                    name, info, object_type=object_type, channel=channel,
                    channel_2=channel_2, object_type_2=obj, params=params,
                    extra_note=(
                        f"The trailing '_{obj}' is a pandas merge suffix added "
                        "when the object tables were joined "
                        "(spacr.io._read_and_join_tables), not part of the "
                        "feature name."
                    ),
                    measurement_units=measurement_units,
                )

    # 8b. Current spacr.utils._check_integrity suffixes a repeated column with
    #     '__dup<n>'. Unambiguous, unlike the legacy positional form below.
    m = _DUP_SUFFIX_RE.match(rest)
    if m:
        resolved = _lookup_stat(m.group("base"))
        if resolved is not None:
            info, params = resolved
            return _entry(
                name, info, object_type=object_type, channel=channel,
                channel_2=channel_2, params=params,
                extra_note=(
                    f"Occurrence {m.group('idx')} of a duplicated column name: "
                    "spacr.utils._check_integrity suffixes every repeat after "
                    f"the first, so this is another copy of '{m.group('base')}'."
                ),
                measurement_units=measurement_units,
            )

    # 9. Databases written before that change carry the positional index
    #    instead, which produces names that look parameterised.
    m = _DEDUP_SUFFIX_RE.match(rest)
    if m:
        resolved = _lookup_stat(m.group("base"))
        if resolved is not None:
            info, params = resolved
            return _entry(
                name, info, object_type=object_type, channel=channel,
                channel_2=channel_2, params=params,
                extra_note=(
                    f"The trailing '_{m.group('idx')}' is most likely the "
                    "positional-index suffix that spacr.utils._check_integrity "
                    "appends to duplicated column names, so this is a second "
                    f"copy of '{m.group('base')}'. Verify before using it."
                ),
                measurement_units=measurement_units,
            )

    return _unknown(name, object_type, channel)


def describe_columns(columns: Iterable[str],
                     measurement_units: str | None = None
                     ) -> list[FeatureEntry]:
    """Describe every column name given, in order and without dropping any.

    :param columns: Iterable of column names.
    :param measurement_units: The ``measurement_units`` value these columns
        were measured under; see :func:`parse_column`.
    :returns: One :class:`FeatureEntry` per input name, same order.
    """
    return [parse_column(c, measurement_units) for c in columns]


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

_FRAME_COLUMNS = [
    "table",
    "column",
    "object_type",
    "object_type_2",
    "channel",
    "channel_2",
    "family",
    "description",
    "unit",
    "measurement_units",
    "computed_by",
    "notes",
]


def _quoted(name: str) -> str:
    """Quote a table identifier for interpolation into a statement."""
    return '"' + name.replace('"', '""') + '"'


def _table_columns(db_path: str | Path, table: str | None = None
                   ) -> dict[str, list[str]]:
    """Return ``{table_name: [column, ...]}`` without loading any row data."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        if table is not None:
            if table not in names:
                raise ValueError(
                    f"table {table!r} not found in {path}; available tables: "
                    f"{', '.join(names) or '(none)'}"
                )
            names = [table]

        out: dict[str, list[str]] = {}
        for name in names:
            out[name] = [
                row[1] for row in
                conn.execute(f"PRAGMA table_info({_quoted(name)})").fetchall()]
    return out


def _table_measurement_units(db_path: str | Path, table: str,
                             columns: Iterable[str]) -> tuple[str | None, str]:
    """Read the ``measurement_units`` the rows of one table were measured in.

    This is what makes the geometric units concrete instead of conditional:
    the dictionary describes a specific database, and that database says on
    every row which units it is in. :func:`spacr.utils._merge_and_save_to_database`
    refuses to append rows whose ``(ndim, units)`` differ from those already in
    the table, so a table normally has exactly one answer.

    :returns: ``(units, why)`` — ``units`` is one of :data:`MEASUREMENT_UNITS`
        or ``None`` when it cannot be pinned to one value, and ``why`` is a
        short human explanation for the export to print.
    """
    columns = list(columns)
    path = Path(db_path)
    quoted = _quoted(table)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            if "measurement_units" not in columns:
                row = conn.execute(
                    f"SELECT 1 FROM {quoted} LIMIT 1").fetchone()
                if row is None:
                    return None, "no rows and no measurement_units column"
                # Not a guess: before 3-D measurement existed a 3-D mask
                # crashed the morphology pass outright, so an unstamped row
                # cannot be anything but a 2-D pixel measurement. Same rule as
                # spacr.utils._LEGACY_STAMP.
                return _LEGACY_UNITS, (
                    "no measurement_units column — written before the stamp "
                    "existed, which can only be a 2-D pixel measurement")
            found = {
                (None if value is None else str(value))
                for (value,) in conn.execute(
                    f"SELECT DISTINCT measurement_units FROM {quoted}")
            }
    except sqlite3.Error as exc:  # unreadable table: describe it, do not fail
        return None, f"measurement_units unreadable ({exc.__class__.__name__})"

    if not found:
        return None, "measurement_units column present but the table is empty"
    # A NULL stamp is the legacy 2-D/px row, exactly as spacr.utils reads it.
    resolved = {_LEGACY_UNITS if v is None else v for v in found}
    if len(resolved) == 1:
        units = resolved.pop()
        if units in MEASUREMENT_UNITS:
            return units, f"measurement_units = '{units}'"
        return None, (f"measurement_units = '{units}', which this dictionary "
                      f"does not recognise")
    return None, ("MIXED measurement_units in one table ("
                  + ", ".join(sorted(resolved))
                  + ") — the geometric columns of these rows are not "
                    "comparable with each other")


def describe_database(db_path: str | Path, table: str | None = None,
                      measurement_units: str | None = None) -> pd.DataFrame:
    """Describe every column of a spaCR measurements database.

    Every column of every table is returned — a column that this dictionary
    cannot explain appears with ``family='unknown'`` and a null description
    rather than being omitted.

    The units are read from the database, not assumed: each table's own
    ``measurement_units`` column decides whether ``<object>_area`` is reported
    as a px^2 area, a cubic-xy-pixel volume or a um^3 volume. A table that
    cannot be pinned to one value gets units that state the condition instead.

    :param db_path: Path to a SQLite database, typically ``measurements.db``.
    :param table: Restrict to a single table. ``None`` (default) covers all
        user tables.
    :param measurement_units: Force the unit basis (one of
        :data:`MEASUREMENT_UNITS`) instead of reading it from each table. For
        describing a frame that has been detached from its database.
    :returns: DataFrame with one row per (table, column) and the columns
        ``table, column, object_type, object_type_2, channel, channel_2,
        family, description, unit, measurement_units, computed_by, notes``.
    :raises FileNotFoundError: If ``db_path`` does not exist.
    :raises ValueError: If ``table`` is given but not present in the database.
    """
    columns_by_table = _table_columns(db_path, table)

    rows: list[dict[str, Any]] = []
    for table_name, columns in columns_by_table.items():
        if measurement_units is None:
            units, _why = _table_measurement_units(db_path, table_name, columns)
        else:
            units = str(measurement_units)
        for entry in describe_columns(columns, units):
            row = entry.to_dict()
            row["table"] = table_name
            rows.append(row)

    df = pd.DataFrame(rows, columns=_FRAME_COLUMNS)
    # Keep channel indices as integers-or-missing rather than letting pandas
    # promote them to float and print "channel 0.0" in the exports.
    for col in ("channel", "channel_2"):
        df[col] = pd.array(
            [None if v is None or pd.isna(v) else int(v) for v in df[col]],
            dtype="Int64",
        )
    return df


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    """True for None, NaN and pandas' NA sentinels."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """Coerce a DataFrame cell to something ``json.dumps`` accepts."""
    if _is_missing(value):
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    # numpy / pandas scalars
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def _md_escape(value: Any) -> str:
    """Render a cell value safely inside a markdown table."""
    if _is_missing(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


#: What each ``measurement_units`` value means for the geometric columns, for
#: the exported "Units" table.
_UNITS_GLOSS: dict[str, str] = {
    UNITS_PX: ("2-D run: `<object>_area` is a pixel count (px^2) and every "
               "length is in pixels. No physical size is applied, even if one "
               "was configured."),
    UNITS_PX_XY: ("3-D run measured with `anisotropy` alone: `<object>_area` "
                  "is a VOLUME in cubic xy-pixels and lengths are in xy "
                  "pixels — geometrically correct, but not physical."),
    UNITS_UM: ("3-D run measured with `voxel_size_z_um` + `voxel_size_xy_um`: "
               "`<object>_area` is a VOLUME in um^3 and lengths are in um."),
}


def _markdown(df: pd.DataFrame, db_path: Path,
              units_by_table: dict[str, tuple[str | None, str]] | None = None
              ) -> str:
    """Render the dictionary as markdown grouped by object then family."""
    n_unknown = int((df["family"] == "unknown").sum())
    tables = sorted(df["table"].dropna().unique().tolist())
    units_by_table = units_by_table or {}

    lines: list[str] = [
        "# spaCR feature dictionary",
        "",
        f"Generated from `{db_path}`.",
        "",
        f"- Tables described: {len(tables)} ({', '.join(tables) or 'none'})",
        f"- Columns described: {len(df)}",
        f"- Unrecognised columns: {n_unknown}",
        "",
        "Intensity features are in the native units of the merged image stack. "
        "Geometric features have no single unit: a 2-D run measures in pixels, "
        "a 3-D run measures a volume, and a 3-D run given a voxel size "
        "measures in micrometres — all under the same column names, with the "
        "unit recorded on each row in `measurement_units`. The units below "
        "were resolved from this database; the Units section says what each "
        "table was measured in.",
        "",
        "The `um_per_pixel` setting is NOT involved in any of this: it only "
        "converts `scale_bar_length_um` into pixels when a scale bar is drawn "
        "on a figure, and never reaches a measurement. The settings that do "
        "are `voxel_size_z_um` and `voxel_size_xy_um`, and only on a 3-D run.",
        "",
        "## Units",
        "",
        "| table | measurement_units | geometric columns |",
        "| --- | --- | --- |",
    ]
    for name in tables:
        units, why = units_by_table.get(name, (None, "not determined"))
        gloss = _UNITS_GLOSS.get(units or "", "")
        if not gloss:
            gloss = ("Not pinned to one value, so each geometric column below "
                     "reports its unit as a condition on `measurement_units` "
                     "instead of asserting one.")
        lines.append(
            f"| {_md_escape(name)} | {_md_escape(units or 'unknown')} "
            f"({_md_escape(why)}) | {_md_escape(gloss)} |")
    lines.append("")

    lines += [
        "## Families",
        "",
        "| family | meaning |",
        "| --- | --- |",
    ]
    for fam, gloss in FEATURE_FAMILIES.items():
        lines.append(f"| {fam} | {_md_escape(gloss)} |")
    lines.append("")

    order = {name: i for i, name in enumerate(OBJECT_TYPES)}
    # Collapse every missing marker (None / NaN) onto a single None bucket, so
    # the "no object" section is emitted exactly once.
    missing = df["object_type"].isna()
    seen: list[str | None] = []
    for value in df["object_type"].where(~missing, None):
        if value not in seen:
            seen.append(value)
    objects = sorted(seen, key=lambda o: (o is None, order.get(o, len(order)),
                                          str(o)))
    fam_order = {name: i for i, name in enumerate(FEATURE_FAMILIES)}

    for obj in objects:
        sub = df[missing] if obj is None else df[df["object_type"] == obj]
        lines.append(f"## {obj if obj else 'no object (metadata and unparsed)'}")
        lines.append("")
        families = sorted(sub["family"].unique(),
                          key=lambda f: (fam_order.get(f, len(fam_order)), f))
        for fam in families:
            fsub = sub[sub["family"] == fam]
            lines.append(f"### {fam} ({len(fsub)} columns)")
            lines.append("")
            lines.append(
                "| column | table | channel | unit | description | computed by | notes |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for _, row in fsub.sort_values("column").iterrows():
                ch = _md_escape(row["channel"])
                if ch and not _is_missing(row["channel_2"]):
                    ch = f"{ch} & {_md_escape(row['channel_2'])}"
                lines.append(
                    "| `{col}` | {tab} | {ch} | {unit} | {desc} | {by} | {notes} |".format(
                        col=_md_escape(row["column"]),
                        tab=_md_escape(row["table"]),
                        ch=ch,
                        unit=_md_escape(row["unit"]),
                        desc=_md_escape(row["description"]),
                        by=_md_escape(row["computed_by"]),
                        notes=_md_escape(row["notes"]),
                    )
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def export_dictionary(db_path: str | Path, out_path: str | Path,
                      fmt: str = "csv") -> Path:
    """Write a data dictionary for ``db_path`` to ``out_path``.

    :param db_path: Path to the SQLite database to describe.
    :param out_path: Destination file. Parent directories are created.
    :param fmt: ``"csv"``, ``"md"`` or ``"json"``.
    :returns: The path written, as a :class:`~pathlib.Path`.
    :raises ValueError: If ``fmt`` is not one of the three supported formats.
    :raises FileNotFoundError: If ``db_path`` does not exist.
    """
    fmt = str(fmt).lower()
    if fmt not in {"csv", "md", "json"}:
        raise ValueError(f"fmt must be one of 'csv', 'md', 'json'; got {fmt!r}")

    db_path = Path(db_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = describe_database(db_path)
    units_by_table = {
        name: _table_measurement_units(db_path, name, columns)
        for name, columns in _table_columns(db_path).items()
    }

    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "md":
        out_path.write_text(_markdown(df, db_path, units_by_table),
                            encoding="utf-8")
    else:
        payload = {
            "database": str(db_path),
            "n_columns": int(len(df)),
            "n_unrecognised": int((df["family"] == "unknown").sum()),
            "families": FEATURE_FAMILIES,
            "measurement_units": {
                name: {"measurement_units": units, "resolved_from": why,
                       "geometric_columns": _UNITS_GLOSS.get(units or "")}
                for name, (units, why) in units_by_table.items()
            },
            "columns": [
                {k: _jsonable(v) for k, v in record.items()}
                for record in df.to_dict(orient="records")
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out_path
