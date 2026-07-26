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
    "FeatureEntry",
    "PropertyInfo",
    "FEATURE_FAMILIES",
    "KNOWN_PROPERTIES",
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
        "Size, shape and position measured from the label mask alone. Pixel "
        "units; no intensity information enters these."
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
    :ivar unit: Physical/derived unit, or ``None`` for identifiers.
    :ivar computed_by: The real provenance — the function or library call that
        produces the value. Never empty.
    :ivar notes: Caveats, known defects, comparability warnings.
    """

    family: str
    description: str | None
    unit: str | None
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

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a plain dict, ready for a DataFrame row or JSON."""
        return asdict(self)


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

# spaCR calls skimage.measure.regionprops/regionprops_table without a
# ``spacing`` argument (measure.py:171, 180, 191, 202, 216, 465), so every
# geometric quantity is in raw pixels. There is no place in the pipeline where
# a physical pixel size is applied.
_PX = "px (pixels; spaCR never applies a physical pixel size)"
_PX2 = "px^2 (pixel count; spaCR never applies a physical pixel size)"

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
_INTENSITY_SUM = (
    "native image intensity units summed over pixels (intensity x px^2)"
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
        "Number of pixels belonging to the object in its label mask.",
        _PX2,
        f"{_RP}(area) via spacr.measure._morphological_measurements",
        "Multiply by (um/px)^2 to convert to physical area.",
    ),
    "area_filled": PropertyInfo(
        "morphology",
        "Area of the object after filling any holes enclosed by it.",
        _PX2,
        f"{_RP}(area_filled) via spacr.measure._morphological_measurements",
        "area_filled - area is the total hole area inside the object.",
    ),
    "area_bbox": PropertyInfo(
        "morphology",
        "Area of the smallest axis-aligned bounding box around the object.",
        _PX2,
        f"{_RP}(area_bbox) via spacr.measure._morphological_measurements",
        "Depends on the object's orientation relative to the image axes, so "
        "it is not rotation invariant.",
    ),
    "convex_area": PropertyInfo(
        "morphology",
        "Area of the convex hull of the object.",
        _PX2,
        f"{_RP}(convex_area) via spacr.measure._morphological_measurements",
        "scikit-image's modern name for the same property is area_convex.",
    ),
    "major_axis_length": PropertyInfo(
        "morphology",
        "Length of the major axis of the ellipse that has the same normalised "
        "second central moments as the object.",
        _PX,
        f"{_RP}(major_axis_length) via spacr.measure._morphological_measurements",
        None,
    ),
    "minor_axis_length": PropertyInfo(
        "morphology",
        "Length of the minor axis of the ellipse that has the same normalised "
        "second central moments as the object.",
        _PX,
        f"{_RP}(minor_axis_length) via spacr.measure._morphological_measurements",
        None,
    ),
    "eccentricity": PropertyInfo(
        "morphology",
        "Eccentricity of the equivalent ellipse: 0 for a perfect circle, "
        "approaching 1 for an increasingly elongated object.",
        _DIMLESS + ", in [0, 1)",
        f"{_RP}(eccentricity) via spacr.measure._morphological_measurements",
        None,
    ),
    "solidity": PropertyInfo(
        "morphology",
        "Area divided by convex-hull area — how much of the object's own "
        "convex hull it fills. Low values mean a ragged or concave outline.",
        _DIMLESS + ", in (0, 1]",
        f"{_RP}(solidity) via spacr.measure._morphological_measurements",
        None,
    ),
    "extent": PropertyInfo(
        "morphology",
        "Area divided by bounding-box area — how much of its bounding box the "
        "object fills.",
        _DIMLESS + ", in (0, 1]",
        f"{_RP}(extent) via spacr.measure._morphological_measurements",
        "Orientation dependent, because the bounding box is axis aligned.",
    ),
    "perimeter": PropertyInfo(
        "morphology",
        "Perimeter of the object, approximated as a line through the centres "
        "of its border pixels.",
        _PX,
        f"{_RP}(perimeter) via spacr.measure._morphological_measurements",
        "Perimeter estimates on a pixel grid are biased upward for small "
        "objects; do not compare across very different object sizes without "
        "care.",
    ),
    "euler_number": PropertyInfo(
        "morphology",
        "Euler characteristic of the object: connected components minus "
        "holes. A single hole-free object gives 1; each enclosed hole "
        "subtracts 1.",
        _DIMLESS + " (integer)",
        f"{_RP}(euler_number) via spacr.measure._morphological_measurements",
        None,
    ),
    "equivalent_diameter_area": PropertyInfo(
        "morphology",
        "Diameter of the circle with the same area as the object, "
        "sqrt(4 * area / pi).",
        _PX,
        f"{_RP}(equivalent_diameter_area) via "
        "spacr.measure._morphological_measurements",
        None,
    ),
    "feret_diameter_max": PropertyInfo(
        "morphology",
        "Maximum Feret diameter: the longest distance between any two points "
        "on the object's boundary (its calliper length).",
        _PX,
        f"{_RP}(feret_diameter_max) via spacr.measure._morphological_measurements",
        None,
    ),
    # ---------------- shape moments (measure.py:56-80)
    "zernike_<i>": PropertyInfo(
        "moment",
        "Magnitude of Zernike moment number {i} of the object's binary shape. "
        "Zernike magnitudes are rotation invariant, so the 25-element vector "
        "as a whole is a shape fingerprint; a single index has no standalone "
        "biological meaning.",
        _DIMLESS,
        "mahotas.features.zernike_moments(region.image, degree) via "
        "spacr.measure._calculate_zernike",
        "Computed on the binary mask only, no intensity. mahotas' signature is "
        "zernike_moments(im, radius, degree=8), and spaCR passes its `degree` "
        "argument positionally, so it lands in `radius`: the unit disk is "
        "fixed at radius 8 px for every object regardless of object size, and "
        "the polynomial degree is always mahotas' default 8 (hence 25 "
        "coefficients). Objects much larger than 8 px are therefore described "
        "only by their central region.",
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
        _PX,
        f"{_RP}(centroid_weighted) via spacr.measure._extended_regionprops_table",
        "The -0 / -1 suffix is scikit-image's separator for multi-value "
        "properties: -0 is the row axis, -1 is the column axis.",
    ),
    "centroid_weighted-1": PropertyInfo(
        "moment",
        "Column (x) coordinate of the intensity-weighted centroid of this "
        "channel inside the object, in full-image coordinates.",
        _PX,
        f"{_RP}(centroid_weighted) via spacr.measure._extended_regionprops_table",
        "The -0 / -1 suffix is scikit-image's separator for multi-value "
        "properties: -0 is the row axis, -1 is the column axis.",
    ),
    "centroid_weighted_local-0": PropertyInfo(
        "moment",
        "Row (y) coordinate of the intensity-weighted centroid, measured "
        "relative to the top-left corner of the object's bounding box.",
        _PX,
        f"{_RP}(centroid_weighted_local) via "
        "spacr.measure._extended_regionprops_table",
        "Position within the object, so unlike centroid_weighted-0 it does not "
        "encode where in the field of view the object sits.",
    ),
    "centroid_weighted_local-1": PropertyInfo(
        "moment",
        "Column (x) coordinate of the intensity-weighted centroid, measured "
        "relative to the top-left corner of the object's bounding box.",
        _PX,
        f"{_RP}(centroid_weighted_local) via "
        "spacr.measure._extended_regionprops_table",
        "Position within the object, so unlike centroid_weighted-1 it does not "
        "encode where in the field of view the object sits.",
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
        "Intended as the most frequent pixel value inside the object.",
        _INTENSITY,
        "scipy.stats.mode in spacr.measure._extended_regionprops_table",
        "BROKEN IN PRACTICE: the code does mode(...).mode[0], but from SciPy "
        "1.11 onwards mode() returns a scalar for 1-D input, so the "
        "subscript raises and the bare except writes NaN. On any recent SciPy "
        "this column is NaN for every object. Do not use it.",
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
        "Fraction of the object's pixels strictly above the object's OWN 90th "
        "percentile.",
        _FRACTION,
        "numpy.mean(intens > numpy.percentile(intens, 90)) in "
        "spacr.measure._extended_regionprops_table",
        "Near-constant by construction: for continuous data it is always "
        "about 0.10, because the threshold is the object's own percentile. It "
        "only departs from 0.10 when many pixels tie at that value, so in "
        "practice it reports quantisation/saturation, not brightness.",
    ),
    "frac_low10": PropertyInfo(
        "intensity",
        "Fraction of the object's pixels strictly below the object's OWN 10th "
        "percentile.",
        _FRACTION,
        "numpy.mean(intens < numpy.percentile(intens, 10)) in "
        "spacr.measure._extended_regionprops_table",
        "Near-constant by construction: about 0.10 for continuous data, "
        "departing from it only through ties. See frac_high90.",
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
        "homogeneity_distances setting (default [8, 16, 32]).",
    ),
    "blur": PropertyInfo(
        "texture",
        "Variance of the discrete Laplacian of the object's pixel values — "
        "intended as a focus/sharpness score, where low values mean blurry.",
        "squared native image intensity units",
        "cv2.Laplacian(...).var() in spacr.measure._estimate_blur",
        "MISLEADING AS IMPLEMENTED: measure.py:392 passes "
        "channel[label == region_label], which is a 1-D vector of the object's "
        "pixels in raster order, so OpenCV treats it as an Nx1 image and the "
        "Laplacian is a 1-D second difference along that vector rather than a "
        "2-D focus measure. It behaves like a roughness statistic of the "
        "object's pixel sequence. Also see the duplicated column prefix noted "
        "on the column itself.",
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
        "scipy.ndimage.binary_dilation(iterations=5) minus the object, in "
        "spacr.measure._outside_intensity",
        "Only emitted for nucleus, pathogen and organelle objects. The ring is "
        "NOT masked against neighbouring objects, so for crowded fields it can "
        "include signal from adjacent cells or pathogens. NaN when the ring is "
        "empty.",
    ),
    "outside_<p>_percentile": PropertyInfo(
        "intensity",
        "{p}th percentile of this channel's intensity in a ring extending 5 px "
        "outward from the object.",
        _INTENSITY,
        "numpy.percentile over the dilation ring in "
        "spacr.measure._outside_intensity",
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
        "setting is on.",
    ),
    # ---------------- intensity-weighted distances (measure.py:733-796)
    "distance_to_nucleus": PropertyInfo(
        "morphology",
        "Distance from this channel's intensity-weighted centre of mass "
        "within the cell to the nearest nucleus pixel. Small values mean the "
        "channel's signal piles up on or near the nucleus.",
        _PX,
        "scipy.ndimage.center_of_mass on a Gaussian-blurred channel, then "
        "scipy.ndimage.distance_transform_edt of the nucleus mask, in "
        "spacr.measure._measure_intensity_distance",
        "Only emitted for the cell object, and only when the "
        "distance_gaussian_sigma setting is a non-zero int. 0 when the centre "
        "of mass lands inside a nucleus. The distance transform is global, so "
        "the nearest nucleus may belong to a neighbouring cell. NaN when the "
        "centre of mass is undefined or falls outside the image.",
    ),
    "distance_to_pathogen": PropertyInfo(
        "morphology",
        "Distance from this channel's intensity-weighted centre of mass "
        "within the cell to the nearest pathogen pixel.",
        _PX,
        "scipy.ndimage.center_of_mass on a Gaussian-blurred channel, then "
        "scipy.ndimage.distance_transform_edt of the pathogen mask, in "
        "spacr.measure._measure_intensity_distance",
        "Only emitted for the cell object, and only when the "
        "distance_gaussian_sigma setting is a non-zero int. 0 when the centre "
        "of mass lands inside a pathogen. The distance transform is global, so "
        "the nearest pathogen may belong to a neighbouring cell.",
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
        "Summed area of all organelles assigned to this parent object.",
        _PX2,
        "spacr.measure._summarize_organelles_per_parent",
        None,
    ),
    "organelle_summary_organelle_fraction": PropertyInfo(
        "morphology",
        "Total organelle area divided by the parent object's area — the "
        "fraction of the parent occupied by organelles.",
        _FRACTION,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent area is 0.",
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
        "0.0 when the parent has no organelles.",
    ),
    "organelle_summary_organelle_std_eccentricity": PropertyInfo(
        "morphology",
        "Standard deviation of organelle eccentricity within this parent.",
        _DIMLESS,
        "spacr.measure._summarize_organelles_per_parent",
        "0.0 when the parent has fewer than 2 organelles.",
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
        _PX,
        "skimage.morphology.skeletonize + regionprops area sum in "
        "spacr.measure._analyze_cytoskeleton",
        "_analyze_cytoskeleton is defined in measure.py but is not called by "
        "measure_crop in this version, so this column is not produced by a "
        "standard run. Thresholding is local (block size 35) with a "
        "per-object adaptive offset.",
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
        "Only present when measure_crop ran with timelapse=True.",
    ),
    "time_id": PropertyInfo(
        "meta",
        "Timepoint of a timelapse acquisition, as 't<n>', in the png_list "
        "table.",
        None,
        _META_WELLS_PNG,
        "Spelled time_id here but timeID in the object tables — the same "
        "quantity under two names.",
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
        "Only built when a timeID column exists.",
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
}

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
) -> FeatureEntry:
    """Build a :class:`FeatureEntry` from a curated :class:`PropertyInfo`."""
    params = params or {}
    notes = _fill(info.notes, params)
    if extra_note:
        notes = f"{extra_note} {notes}" if notes else extra_note
    return FeatureEntry(
        column=column,
        object_type=object_type,
        channel=channel,
        family=info.family,
        description=_fill(info.description, params),
        unit=info.unit,
        computed_by=info.computed_by,
        notes=notes,
        channel_2=channel_2,
        object_type_2=object_type_2,
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


def _parse_organelle_summary(name: str) -> FeatureEntry | None:
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
        )
    info = KNOWN_PROPERTIES.get(name)
    if info is not None:
        return _entry(name, info, object_type="organelle")
    return _unknown(
        name,
        "organelle",
        None,
        "Column carries the organelle_summary_ prefix written by "
        "spacr.measure._measure_crop_core, but the summary statistic is not in "
        "the dictionary.",
    )


def parse_column(name: str) -> FeatureEntry:
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

    An unrecognised name never raises and is never dropped: it comes back as a
    :class:`FeatureEntry` with ``family="unknown"`` and ``description=None``.

    :param name: Column name exactly as stored in the database.
    :returns: A :class:`FeatureEntry` describing the column.
    """
    if not isinstance(name, str):
        name = str(name)

    # 1. exact metadata match wins over any structural interpretation, so that
    #    e.g. cell_id is an identifier and not a 'cell' feature named 'id'.
    info = META_COLUMNS.get(name)
    if info is not None:
        return _entry(name, info)

    # 2. per-parent organelle summaries, before the object prefix is stripped
    #    (the prefix 'organelle_' would otherwise swallow the family name).
    summary = _parse_organelle_summary(name)
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
        return _entry(name, link, object_type=object_type)

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
        )

    if channel is not None and _PLAIN_BLUR_RE.match(rest):
        return _entry(name, KNOWN_PROPERTIES["blur"], object_type=object_type,
                      channel=channel)

    link = _LINK_COLUMNS.get(rest)
    if link is not None:
        return _entry(name, link, object_type=object_type, channel=channel,
                      channel_2=channel_2)

    resolved = _lookup_stat(rest)
    if resolved is not None:
        info, params = resolved
        return _entry(name, info, object_type=object_type, channel=channel,
                      channel_2=channel_2, params=params)

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
                )

    # 9. spacr.utils._check_integrity renames duplicated columns by appending
    #    their positional index, which produces names that look parameterised.
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
            )

    return _unknown(name, object_type, channel)


def describe_columns(columns: Iterable[str]) -> list[FeatureEntry]:
    """Describe every column name given, in order and without dropping any.

    :param columns: Iterable of column names.
    :returns: One :class:`FeatureEntry` per input name, same order.
    """
    return [parse_column(c) for c in columns]


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
    "computed_by",
    "notes",
]


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
            quoted = '"' + name.replace('"', '""') + '"'
            out[name] = [row[1] for row in
                         conn.execute(f"PRAGMA table_info({quoted})").fetchall()]
    return out


def describe_database(db_path: str | Path, table: str | None = None
                      ) -> pd.DataFrame:
    """Describe every column of a spaCR measurements database.

    Every column of every table is returned — a column that this dictionary
    cannot explain appears with ``family='unknown'`` and a null description
    rather than being omitted.

    :param db_path: Path to a SQLite database, typically ``measurements.db``.
    :param table: Restrict to a single table. ``None`` (default) covers all
        user tables.
    :returns: DataFrame with one row per (table, column) and the columns
        ``table, column, object_type, object_type_2, channel, channel_2,
        family, description, unit, computed_by, notes``.
    :raises FileNotFoundError: If ``db_path`` does not exist.
    :raises ValueError: If ``table`` is given but not present in the database.
    """
    columns_by_table = _table_columns(db_path, table)

    rows: list[dict[str, Any]] = []
    for table_name, columns in columns_by_table.items():
        for entry in describe_columns(columns):
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


def _markdown(df: pd.DataFrame, db_path: Path) -> str:
    """Render the dictionary as markdown grouped by object then family."""
    n_unknown = int((df["family"] == "unknown").sum())
    tables = sorted(df["table"].dropna().unique().tolist())

    lines: list[str] = [
        "# spaCR feature dictionary",
        "",
        f"Generated from `{db_path}`.",
        "",
        f"- Tables described: {len(tables)} ({', '.join(tables) or 'none'})",
        f"- Columns described: {len(df)}",
        f"- Unrecognised columns: {n_unknown}",
        "",
        "Intensity features are in the native units of the merged image "
        "stack; spaCR does not calibrate pixel size, so all geometric "
        "features are in pixels.",
        "",
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

    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "md":
        out_path.write_text(_markdown(df, db_path), encoding="utf-8")
    else:
        payload = {
            "database": str(db_path),
            "n_columns": int(len(df)),
            "n_unrecognised": int((df["family"] == "unknown").sum()),
            "families": FEATURE_FAMILIES,
            "columns": [
                {k: _jsonable(v) for k, v in record.items()}
                for record in df.to_dict(orient="records")
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out_path
