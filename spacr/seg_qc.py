"""Score segmentation masks the moment they exist, before Measure runs.

Why this exists
---------------
``measure_crop`` is the expensive step in spaCR: it opens every mask, cuts
every object out of every channel, computes hundreds of features per object
and writes them to SQLite. On a full plate that is hours. A plate whose
segmentation collapsed — a channel out of focus, a diameter two-fold wrong, a
confluent monolayer that fused into slabs — costs exactly the same hours and
then produces a measurement table that has to be thrown away. The failure is
obvious from the masks alone; nobody looks, because looking means opening a
thousand ``.npy`` files by hand.

So this module looks. It reads the label masks that
:mod:`spacr.object` has just written, scores every field against a handful of
robust statistics, and prints a scorecard naming the fields that are wrong and
why. It changes nothing: ``seg_qc='report'`` (the default) surfaces the
problem, it does not silently drop fields. Deciding which fields to keep is the
user's call and it needs the evidence, not a filter.

Design constraint: this runs right after segmentation on every field of the
plate, so it must be cheap. Everything below is ``np.bincount`` on a label
image — microseconds per field. The one expensive check (the distance-transform
cross-check for fusion) runs *only* on fields dense enough for fusion to be the
explanation, so a healthy plate never pays for it. Like :mod:`spacr.diameter`,
this module imports **no torch and no cellpose** — that is a tested property
(see ``tests/test_seg_qc.py``), not an aspiration.

What is measured, and why that number
-------------------------------------
Every threshold below is a keyword argument with a default in
:data:`QC_DEFAULTS`, and every one of them is exposed as a ``seg_qc_*`` setting
(see :data:`SETTING_KEYS`) so a user with 3 px organelles or a deliberately
confluent assay can move it.

``object count`` and its deviation from the plate median
    A field with 3 objects on a plate whose median field holds 300 is not a
    data point, it is a broken field. ``count_ratio`` is
    ``n_objects / plate_median``; outside
    ``[seg_qc_count_ratio, 1/seg_qc_count_ratio]`` (0.25 to 4-fold) the field
    is flagged. Seeding density across a plate varies with a CV of 10-30% and
    even edge-effect wells rarely fall below half the median, so a four-fold
    departure is not biology.

``under-segmentation`` (fused objects)
    Detected the way :mod:`spacr.diameter` detects it, and for the same
    reason: **both halves of the signature are required**. The field must be
    dense enough for fusion to be the explanation (foreground at or above
    ``seg_qc_foreground_fraction``, default 0.35 — the same number
    ``diameter.estimate_diameters(fused_fraction=...)`` uses, because it is the
    same question), *and* the mask must under-count what the pixels support:
    either one label covers more than ``seg_qc_max_object_fraction`` of the
    field (0.25, again diameter.py's number — a component covering a quarter of
    a field is not a cell), or the Euclidean distance transform of the
    foreground resolves at least ``seg_qc_split_ratio`` (2.0) inscribed-circle
    maxima per mask object. Requiring both matters in each direction: a dense
    but correctly separated field reaches 35% foreground and is fine, while an
    elongated or hollow object shatters the distance transform into several
    maxima and is also fine. diameter.py demands a 5-fold seed excess because
    it compares against raw Otsu components, where a confluent monolayer is one
    blob; here the comparison is against a Cellpose mask that already separates
    most objects, and the smallest fusion worth catching — every object being a
    welded pair — is exactly 2.

``over-segmentation`` (shattered objects)
    Two signatures. Absolutely: at least ``seg_qc_tiny_fraction`` (0.30) of the
    field's objects are under ``seg_qc_min_diameter`` (5 px) across — spaCR's
    diameter estimator already discards components below 4 px as debris, and
    nothing 5 px across survives a crop-and-measure as an object. Relative to
    the plate: the field holds ``seg_qc_size_ratio``-fold more objects than the
    plate median at ``1/seg_qc_size_ratio`` of its median diameter. That knob
    defaults to 1.4 = sqrt(2) on purpose — two objects welded into one have
    sqrt(2) times the equivalent diameter of one, and one object split in two
    has 1/sqrt(2) of it, so 1.4 is the fused-pair / split-in-half signature
    itself rather than an arbitrary tolerance.

``% border-touching objects``
    Objects that touch the field edge are truncated, so their crops are cut off
    and their areas understate the truth. For objects of diameter *d* on a
    *W*-wide field the geometric expectation is about ``2*d/W`` — 8% for 60 px
    cells on a 1400 px field. ``seg_qc_border_fraction`` defaults to 0.30:
    well above anything geometry explains, and the level at which a third of
    the crops going into Measure are fragments.

``size outliers``
    The fraction of objects whose equivalent diameter falls outside
    ``median ± seg_qc_outlier_mad * 1.4826 * MAD``. Median and MAD, not mean
    and standard deviation: five pieces of debris inflate a standard deviation
    until nothing is an outlier any more, which is the exact case this check
    exists to catch (``tests/test_seg_qc.py`` contrasts the two). *k* = 5 is
    deliberately loose — real size distributions are lognormal and heavier
    tailed than Gaussian, so 3 sigma flags a few percent of a perfectly good
    field. The flag fires when more than ``seg_qc_outlier_fraction`` (0.15) of
    objects are out there, which a tail cannot do; only a second population can.
    Border objects are excluded from every size statistic, exactly as
    ``diameter._region_diameters`` excludes them, because a truncated object's
    area is a lie.

``empty / near-empty fields``
    Zero objects is a failure. Fewer than ``seg_qc_min_objects`` (10) is a
    warning and suppresses the per-field robust statistics, because below ten
    objects a MAD is one object's opinion — the same threshold at which
    diameter.py collapses its confidence two levels. If the *plate* median is
    also below that floor (a low-MOI pathogen channel, say), empty fields are
    demoted to a warning: that is a property of the assay, not of the field.

Public API
----------
``FieldQC``
    One field's verdict: counts, flags, the numbers behind them, severity.
``score_field(mask, object_type, **thresholds)``
    Score one label image, with no plate context.
``score_masks(source, object_type, ...)``
    Score a folder of ``.npy`` masks, a 3-D stack or a list of 2-D masks, then
    add the plate-relative flags.
``summarize_qc(field_qcs)``
    Plate-level rollup: verdict, counts per severity, the failing field names.
``format_scorecard(field_qcs)``
    The printable card.
``write_scorecard(field_qcs, dst, object_type)``
    ``<dst>/qc/segmentation_qc_<object_type>.csv``, one row per field.
``run_segmentation_qc(...)``
    What :mod:`spacr.object` calls: score, write, print, honour ``seg_qc``.
``thresholds_from_settings(settings)``
    Pull the ``seg_qc_*`` knobs out of a settings dict.

Reading the verdict back — what a GUI shows before Measure
----------------------------------------------------------
Everything above runs once, at mask time, and writes a CSV. The second half
of this module reads that CSV back and turns it into something a user can
act on, without scoring a single mask again:

``FLAG_GUIDANCE`` / ``explain_flag(flag)``
    Every flag in :data:`FLAGS` in plain language: what it means for the
    measurements, what usually causes it, and what to do. Where uneven
    illumination is one of the usual causes, the entry says so and points at
    :mod:`spacr.illumination`, which estimates the lamp profile from the
    plate's own fields and divides it out.
``parse_field_name(name)`` / ``FieldAddress``
    ``plate1_E07_3`` → plate ``plate1``, well ``E07``, row ``E``, column 7.
    Naming the plate and the wells is the difference between "3 plates
    failed" and something a user can go and look at.
``diagnose(field_qcs)`` → ``[Finding]``
    Per-plate, per-flag findings that name the plate and the wells, plus the
    positional ones no single field can see: a plate whose object count or
    object size steps between one half of the rows (or columns) and the
    other, which is the signature of uneven illumination rather than of
    biology.
``read_digest(src)`` → ``QCDigest``
    The cheap path: locate ``qc/segmentation_qc_*.csv`` under a project,
    parse it, roll it up, diagnose it, and compare each card's mtime against
    its mask stack so a card written before the last re-mask is reported as
    out of date rather than believed. Scores nothing.
``score_digest(src)`` → ``QCDigest``
    The expensive path, for the user who asks for it explicitly: score the
    mask stacks under a project, write the cards, return the same digest.
``format_digest(digest)``
    The printable version.
"""
from __future__ import annotations

import collections.abc as _abc
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FieldQC",
    "FLAGS",
    "FLAG_GUIDANCE",
    "FieldAddress",
    "Finding",
    "QC_DEFAULTS",
    "QCDigest",
    "SETTING_KEYS",
    "Scorecard",
    "diagnose",
    "explain_flag",
    "find_mask_stacks",
    "find_scorecards",
    "format_digest",
    "format_scorecard",
    "mask_stack_mtime",
    "parse_field_name",
    "qc_mode",
    "qc_roots",
    "read_digest",
    "read_scorecard",
    "run_segmentation_qc",
    "score_digest",
    "score_field",
    "score_masks",
    "summarize_qc",
    "thresholds_from_settings",
    "write_scorecard",
]

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

#: no object at all in the field.
FLAG_EMPTY = "empty_field"
#: too few objects for the field to carry a distribution.
FLAG_NEAR_EMPTY = "near_empty_field"
#: objects are welded together (fused cells).
FLAG_UNDER = "under_segmented"
#: objects are shattered into fragments.
FLAG_OVER = "over_segmented"
#: too many objects truncated by the field edge.
FLAG_BORDER = "high_border_fraction"
#: the size distribution holds a second population.
FLAG_OUTLIERS = "size_outliers"
#: far fewer objects than the plate median.
FLAG_LOW_COUNT = "low_object_count"
#: far more objects than the plate median.
FLAG_HIGH_COUNT = "high_object_count"
#: the mask file could not be read or is not a label image.
FLAG_UNREADABLE = "unreadable_mask"

#: every flag this module can raise, in report order.
FLAGS: Tuple[str, ...] = (
    FLAG_UNREADABLE,
    FLAG_EMPTY,
    FLAG_NEAR_EMPTY,
    FLAG_UNDER,
    FLAG_OVER,
    FLAG_LOW_COUNT,
    FLAG_HIGH_COUNT,
    FLAG_BORDER,
    FLAG_OUTLIERS,
)

# How bad each flag is. 'fail' means the field's measurements would be wrong,
# not merely noisy; 'warn' means look at it before you trust it. The severity
# classes are semantics and live here; the numbers that decide whether a flag
# fires are thresholds and live in QC_DEFAULTS.
_FLAG_SEVERITY: Dict[str, str] = {
    FLAG_UNREADABLE: "fail",
    FLAG_EMPTY: "fail",
    FLAG_UNDER: "fail",
    FLAG_OVER: "fail",
    FLAG_LOW_COUNT: "fail",
    FLAG_HIGH_COUNT: "fail",
    FLAG_NEAR_EMPTY: "warn",
    FLAG_BORDER: "warn",
    FLAG_OUTLIERS: "warn",
}

_SEVERITY_ORDER = ("ok", "warn", "fail")

#: Below this many fields, "the plate median" is just one of the fields, so the
#: plate-relative comparisons are not made. Structural, not a tunable.
_MIN_FIELDS_FOR_PLATE_CONTEXT = 3

#: Default thresholds. Each is justified in the module docstring; each is
#: overridable per call and exposed as a setting via :data:`SETTING_KEYS`.
QC_DEFAULTS: Dict[str, float] = {
    "min_objects": 10,
    "count_ratio": 0.25,
    "size_ratio": 1.4,
    "border_fraction": 0.30,
    "outlier_mad": 5.0,
    "outlier_fraction": 0.15,
    "foreground_fraction": 0.35,
    "split_ratio": 2.0,
    "min_diameter": 5.0,
    "tiny_fraction": 0.30,
    "max_object_fraction": 0.25,
    "plate_fail_fraction": 0.10,
}

#: settings key -> threshold name in :data:`QC_DEFAULTS`.
SETTING_KEYS: Dict[str, str] = {
    f"seg_qc_{name}": name for name in QC_DEFAULTS
}

#: the mode setting itself, and what it may be set to.
MODE_SETTING = "seg_qc"
MODES: Tuple[str, ...] = ("off", "report", "flag", "stop")

#: The exception `stop` raises. Its own class so a caller can tell "the plate
#: failed QC" from "the run broke", which a bare RuntimeError cannot.



class SegmentationQCFailed(RuntimeError):
    """The plate failed segmentation QC and ``seg_qc='stop'`` was set.

    Carries the summary so a caller can report WHICH fields failed and why,
    rather than only that something did.
    """

    def __init__(self, message: str, summary: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.summary = dict(summary or {})


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------

@dataclass
class FieldQC:
    """One field's segmentation verdict.

    :param field: the mask's name — the ``.npy`` stem, so it can be found on
        disk and opened.
    :param object_type: ``'cell'``, ``'nucleus'``, ``'pathogen'``,
        ``'organelle'`` — whatever was segmented.
    :param n_objects: labels present in the mask.
    :param flags: the named defects, e.g.
        ``['under_segmented', 'high_border_fraction']``. Empty means clean.
    :param metrics: the numbers behind the flags, all floats so the card can be
        written to CSV without special cases. ``float('nan')`` marks a metric
        that could not be computed (too few objects, or no plate context).
    :param severity: ``'ok'``, ``'warn'`` or ``'fail'`` — the worst of the
        flags raised.
    :param note: the same verdict in prose, with the numbers in it.
    """

    field: str
    object_type: str
    n_objects: int
    flags: List[str] = _dc_field(default_factory=list)
    metrics: Dict[str, float] = _dc_field(default_factory=dict)
    severity: str = "ok"
    note: str = ""

    @property
    def failed(self) -> bool:
        """True when this field's measurements would be wrong, not just noisy."""
        return self.severity == "fail"

    def __str__(self) -> str:
        flags = ", ".join(self.flags) if self.flags else "clean"
        return f"{self.field}: {self.n_objects} objects [{self.severity}] {flags}"


# ---------------------------------------------------------------------------
# settings glue
# ---------------------------------------------------------------------------

def _as_number(value: Any) -> Optional[float]:
    """Coerce a settings value to a float, or None if it is not one.

    Settings imported from a CSV arrive as strings, and ``bool`` is an ``int``
    subclass that must never be read as the number 0 or 1 here.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            out = float(value.strip())
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def thresholds_from_settings(settings: Mapping[str, Any]) -> Dict[str, float]:
    """Pull the ``seg_qc_*`` knobs out of a spaCR settings dict.

    :param settings: any mapping; keys that are absent, blank or not numeric
        are skipped so the corresponding :data:`QC_DEFAULTS` entry stands.
    :returns: ``{threshold_name: value}``, ready to splat into
        :func:`score_field` or :func:`score_masks`.
    """
    out: Dict[str, float] = {}
    for key, name in SETTING_KEYS.items():
        number = _as_number((settings or {}).get(key))
        if number is not None:
            out[name] = number
    return out


def qc_mode(settings: Mapping[str, Any]) -> str:
    """Read ``seg_qc`` out of a settings dict and normalise it.

    Accepts the three documented modes plus the shapes a settings CSV round
    trip produces (``None``, ``''``, ``'False'``, ``True``). Anything
    unrecognised falls back to ``'report'``, because the cost of computing a
    scorecard nobody asked for is a second, and the cost of silently skipping
    one somebody did ask for is a wasted Measure run.

    :param settings: any mapping.
    :returns: ``'off'``, ``'report'`` or ``'flag'``.
    """
    raw = (settings or {}).get(MODE_SETTING, "report")
    if raw is None or raw is False:
        return "off"
    if raw is True:
        return "report"
    text = str(raw).strip().lower()
    if text in ("off", "none", "false", "no", "0", ""):
        return "off"
    if text in ("flag", "flags"):
        return "flag"
    if text in ("stop", "gate", "halt"):
        return "stop"
    return "report"


def _resolve(overrides: Mapping[str, Any]) -> Dict[str, float]:
    """Merge caller overrides over :data:`QC_DEFAULTS`, ignoring junk."""
    th = dict(QC_DEFAULTS)
    for key, value in (overrides or {}).items():
        if key not in QC_DEFAULTS:
            raise TypeError(
                f"unknown segmentation-QC threshold {key!r}; "
                f"known thresholds are {sorted(QC_DEFAULTS)}"
            )
        number = _as_number(value)
        if number is not None:
            th[key] = number
    return th


# ---------------------------------------------------------------------------
# per-field measurement
# ---------------------------------------------------------------------------

def _as_labels(mask: Any) -> np.ndarray:
    """Return ``mask`` as a 2-D non-negative integer label image.

    Squeezes singleton axes (a ``(1, H, W)`` mask stack is one field), labels a
    boolean mask, and rounds a float label image — masks that have been through
    a resize arrive as floats. Anything else is a caller error and says so.
    """
    arr = np.squeeze(np.asarray(mask))
    if arr.ndim != 2:
        raise ValueError(f"mask must be a 2-D label image, got shape {np.shape(mask)}")
    if arr.dtype == bool:
        from scipy.ndimage import label as ndi_label

        arr = ndi_label(arr)[0]
    elif arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError("mask holds NaN or infinite values, so it is not a label image")
        arr = np.rint(arr).astype(np.int64)
    elif arr.dtype.kind not in "iu":
        raise ValueError(f"mask dtype {arr.dtype} is not a label type")
    if arr.size and int(arr.min()) < 0:
        raise ValueError("mask holds negative labels, so it is not a label image")
    return arr


def _split_seed_count(foreground: np.ndarray, min_diameter: float) -> int:
    """How many objects the distance transform thinks the foreground holds.

    This is :func:`spacr.diameter._analyse_plane`'s cross-check with the
    watershed removed — the count is all that is wanted here, and the basins
    cost more than the seeds. Two passes: a coarse pass measures the typical
    inscribed radius, which then sets the suppression distance for the refined
    pass, so ripples on one object's medial axis do not each become an object.
    The array is padded by one background pixel so objects at the image edge
    stay bounded instead of letting the transform run off the array.

    :param foreground: boolean foreground of the mask.
    :param min_diameter: seeds must sit at least this many pixels across.
    :returns: the number of refined seeds.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max

    padded = np.pad(np.asarray(foreground, dtype=bool), 1)
    edt = distance_transform_edt(padded)
    seed_floor = max(1.0, float(min_diameter) / 2.0)
    coarse = peak_local_max(edt, min_distance=3, threshold_abs=seed_floor, exclude_border=False)
    if not coarse.size:
        return 0
    r_coarse = float(np.median(edt[tuple(coarse.T)]))
    refined = peak_local_max(
        edt,
        min_distance=max(3, int(round(r_coarse))),
        threshold_abs=max(seed_floor, 0.4 * r_coarse),
        exclude_border=False,
    )
    return int(len(refined))


def _equivalent_diameters(areas: np.ndarray) -> np.ndarray:
    """``2*sqrt(area/pi)`` — the same size measure diameter.py reports."""
    return 2.0 * np.sqrt(np.asarray(areas, dtype=np.float64) / np.pi)


def score_field(
    mask: Any,
    object_type: str = "object",
    field: str = "field",
    **thresholds: Any,
) -> FieldQC:
    """Score one label mask on its own, with no knowledge of the plate.

    Everything here is plate-independent: counts, foreground, border fraction,
    the robust size range, the fusion cross-check. The comparisons that need
    the plate (count and size relative to the plate median) are added by
    :func:`score_masks` afterwards.

    :param mask: a 2-D label image (or anything :func:`numpy.squeeze` reduces
        to one). Booleans are labelled first; floats are rounded.
    :param object_type: what was segmented, carried through to the card.
    :param field: the field's name, used in the card and the CSV.
    :param thresholds: any key of :data:`QC_DEFAULTS`. Unknown keys raise, so a
        typo in a threshold name cannot silently do nothing.
    :returns: a :class:`FieldQC`. A mask that cannot be read comes back
        flagged ``unreadable_mask`` rather than raising — one corrupt field
        must not cost the plate its scorecard.
    """
    th = _resolve(thresholds)
    try:
        labels = _as_labels(mask)
    except ValueError as exc:
        return FieldQC(
            field=field,
            object_type=object_type,
            n_objects=0,
            flags=[FLAG_UNREADABLE],
            metrics={},
            severity="fail",
            note=str(exc),
        )

    height, width = labels.shape
    n_pixels = float(height * width)
    counts = np.bincount(labels.ravel())
    background = float(counts[0]) if counts.size else n_pixels
    areas_by_label = counts[1:] if counts.size > 1 else np.zeros(0, np.int64)
    present = np.nonzero(areas_by_label)[0] + 1
    areas = areas_by_label[present - 1].astype(np.float64)
    n_objects = int(present.size)

    metrics: Dict[str, float] = {
        "n_objects": float(n_objects),
        "foreground_fraction": (n_pixels - background) / n_pixels if n_pixels else 0.0,
        "border_fraction": 0.0,
        "n_interior": 0.0,
        "median_diameter": float("nan"),
        "median_area": float("nan"),
        "mad_diameter": float("nan"),
        "max_object_fraction": 0.0,
        "outlier_fraction": float("nan"),
        "tiny_fraction": float("nan"),
        "split_ratio": float("nan"),
        "count_ratio": float("nan"),
        "diameter_ratio": float("nan"),
    }

    flags: List[str] = []
    reasons: List[str] = []

    if n_objects == 0:
        metrics["median_diameter"] = 0.0
        metrics["median_area"] = 0.0
        return FieldQC(
            field=field,
            object_type=object_type,
            n_objects=0,
            flags=[FLAG_EMPTY],
            metrics=metrics,
            severity=_FLAG_SEVERITY[FLAG_EMPTY],
            note=(
                f"no {object_type} object at all in a {width}x{height} field: nothing "
                f"here to crop or measure"
            ),
        )

    # ---- border-touching objects ------------------------------------------
    # Border objects are counted, reported, and then excluded from every size
    # statistic below: a truncated object's area understates its size, exactly
    # as diameter._region_diameters argues.
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_ids = np.unique(edge)
    border_ids = border_ids[border_ids > 0]
    metrics["border_fraction"] = float(border_ids.size) / n_objects

    interior_keep = np.ones(counts.size, dtype=bool)
    if border_ids.size:
        interior_keep[border_ids] = False
    interior_ids = present[interior_keep[present]]
    interior_areas = areas_by_label[interior_ids - 1].astype(np.float64)
    metrics["n_interior"] = float(interior_ids.size)

    diameters = _equivalent_diameters(interior_areas if interior_areas.size else areas)
    metrics["median_diameter"] = float(np.median(diameters))
    metrics["median_area"] = float(np.median(interior_areas if interior_areas.size else areas))
    metrics["max_object_fraction"] = float(areas.max()) / n_pixels

    if metrics["border_fraction"] >= th["border_fraction"]:
        flags.append(FLAG_BORDER)
        reasons.append(
            f"{metrics['border_fraction'] * 100:.0f}% of objects touch the field edge, "
            f"so their crops will be truncated"
        )

    # ---- robust size statistics -------------------------------------------
    # Only with enough interior objects to have a distribution. Below the floor
    # a MAD is one object's opinion, so no size flag may be raised from it.
    enough = interior_areas.size >= th["min_objects"]
    if enough:
        median_d = float(np.median(diameters))
        mad = float(np.median(np.abs(diameters - median_d)))
        robust_sigma = 1.4826 * mad
        metrics["mad_diameter"] = mad
        low = median_d - th["outlier_mad"] * robust_sigma
        high = median_d + th["outlier_mad"] * robust_sigma
        outliers = (diameters < low) | (diameters > high)
        metrics["outlier_fraction"] = float(np.mean(outliers))
        metrics["tiny_fraction"] = float(np.mean(diameters < th["min_diameter"]))

        if metrics["outlier_fraction"] > th["outlier_fraction"]:
            flags.append(FLAG_OUTLIERS)
            reasons.append(
                f"{metrics['outlier_fraction'] * 100:.0f}% of objects fall outside the "
                f"robust size range {low:.1f}-{high:.1f} px "
                f"(median {median_d:.1f} px, MAD {mad:.2f}), which a single population "
                f"cannot do"
            )
        if metrics["tiny_fraction"] >= th["tiny_fraction"]:
            flags.append(FLAG_OVER)
            reasons.append(
                f"{metrics['tiny_fraction'] * 100:.0f}% of objects are under "
                f"{th['min_diameter']:g} px across, which is fragments rather than "
                f"{object_type}s"
            )

    # ---- fusion, the way diameter.py argues it -----------------------------
    # Both halves required: dense enough for fusion to be the explanation, AND
    # the mask under-counting what the pixels support. The distance transform
    # is only run when the first half holds, so a healthy plate never pays for
    # it.
    confluent = metrics["foreground_fraction"] >= th["foreground_fraction"]
    giant = metrics["max_object_fraction"] >= th["max_object_fraction"]
    outnumbered = False
    if confluent:
        seeds = _split_seed_count(labels > 0, th["min_diameter"])
        metrics["split_ratio"] = float(seeds) / n_objects
        outnumbered = metrics["split_ratio"] >= th["split_ratio"]

    if confluent and (giant or outnumbered):
        evidence = [f"foreground covers {metrics['foreground_fraction'] * 100:.0f}% of the field"]
        if giant:
            evidence.append(
                f"one label covers {metrics['max_object_fraction'] * 100:.0f}% of it"
            )
        if outnumbered:
            evidence.append(
                f"the distance transform resolves {metrics['split_ratio']:.1f} objects "
                f"per mask object"
            )
        flags.append(FLAG_UNDER)
        reasons.append(
            "objects look fused (" + "; ".join(evidence) + ")"
        )

    # ---- too few objects to be a field ------------------------------------
    if n_objects < th["min_objects"]:
        flags.append(FLAG_NEAR_EMPTY)
        reasons.append(
            f"only {n_objects} object(s), below the {th['min_objects']:g} needed for "
            f"per-field statistics, so the size checks above were not run"
        )

    ordered = [f for f in FLAGS if f in flags]
    return FieldQC(
        field=field,
        object_type=object_type,
        n_objects=n_objects,
        flags=ordered,
        metrics=metrics,
        severity=_severity_of(ordered),
        note=_compose_note(object_type, n_objects, metrics, reasons),
    )


def _severity_of(flags: Sequence[str], demote: Sequence[str] = ()) -> str:
    """Worst severity among ``flags``; anything in ``demote`` counts as a warning."""
    worst = "ok"
    for flag in flags:
        level = "warn" if flag in demote else _FLAG_SEVERITY.get(flag, "warn")
        if _SEVERITY_ORDER.index(level) > _SEVERITY_ORDER.index(worst):
            worst = level
    return worst


def _compose_note(
    object_type: str,
    n_objects: int,
    metrics: Mapping[str, float],
    reasons: Sequence[str],
) -> str:
    """The field's verdict in prose, always carrying its numbers."""
    head = (
        f"{n_objects} {object_type}(s), median diameter "
        f"{metrics.get('median_diameter', float('nan')):.1f} px, "
        f"{metrics.get('border_fraction', 0.0) * 100:.0f}% on the border, "
        f"foreground {metrics.get('foreground_fraction', 0.0) * 100:.1f}%"
    )
    if not reasons:
        return head + ". Nothing wrong with this field."
    return head + ". " + "; ".join(reasons) + "."


# ---------------------------------------------------------------------------
# plate context
# ---------------------------------------------------------------------------

def _apply_plate_context(field_qcs: List[FieldQC], th: Mapping[str, float]) -> List[FieldQC]:
    """Add the flags that only exist relative to the rest of the plate.

    Three of them: a count far off the plate median, and the two mixed
    signatures — more objects at a smaller size (shattered) or fewer objects at
    a larger size (fused). Both mixed rules demand a deviation in *both* count
    and size, for the same reason diameter.py demands both halves of its fusion
    signature: either alone is ordinary well-to-well variation.

    Also demotes empty and near-empty fields to warnings when the whole plate
    is sparse — with a plate median of 2 pathogens per field, a field with none
    is the assay, not a defect.
    """
    scored = [q for q in field_qcs if FLAG_UNREADABLE not in q.flags]
    if len(scored) < _MIN_FIELDS_FOR_PLATE_CONTEXT:
        return field_qcs

    plate_count = float(np.median([q.metrics.get("n_objects", 0.0) for q in scored]))
    usable_d = [
        q.metrics["median_diameter"]
        for q in scored
        if q.n_objects >= th["min_objects"] and np.isfinite(q.metrics.get("median_diameter", np.nan))
    ]
    plate_diameter = float(np.median(usable_d)) if usable_d else float("nan")
    sparse_plate = plate_count < th["min_objects"]

    high_count_ratio = 1.0 / th["count_ratio"] if th["count_ratio"] > 0 else float("inf")
    small_ratio = 1.0 / th["size_ratio"] if th["size_ratio"] > 0 else 0.0

    for qc in scored:
        reasons: List[str] = []
        count_ratio = float("nan")
        if plate_count > 0:
            count_ratio = qc.n_objects / plate_count
            qc.metrics["count_ratio"] = count_ratio
        diameter_ratio = float("nan")
        if np.isfinite(plate_diameter) and plate_diameter > 0 and qc.n_objects:
            diameter_ratio = qc.metrics.get("median_diameter", float("nan")) / plate_diameter
            qc.metrics["diameter_ratio"] = diameter_ratio

        if np.isfinite(count_ratio) and FLAG_EMPTY not in qc.flags:
            if count_ratio <= th["count_ratio"]:
                _add(qc, FLAG_LOW_COUNT)
                reasons.append(
                    f"{qc.n_objects} objects where the plate median is "
                    f"{plate_count:.0f} ({count_ratio:.2f}x)"
                )
            elif count_ratio >= high_count_ratio:
                _add(qc, FLAG_HIGH_COUNT)
                reasons.append(
                    f"{qc.n_objects} objects where the plate median is "
                    f"{plate_count:.0f} ({count_ratio:.1f}x)"
                )

        if np.isfinite(count_ratio) and np.isfinite(diameter_ratio):
            if diameter_ratio >= th["size_ratio"] and count_ratio <= small_ratio:
                if _add(qc, FLAG_UNDER):
                    reasons.append(
                        f"objects are {diameter_ratio:.2f}x the plate's median diameter "
                        f"at {count_ratio:.2f}x its count, the signature of objects "
                        f"merged in pairs"
                    )
            elif diameter_ratio <= small_ratio and count_ratio >= th["size_ratio"]:
                if _add(qc, FLAG_OVER):
                    reasons.append(
                        f"objects are {diameter_ratio:.2f}x the plate's median diameter "
                        f"at {count_ratio:.1f}x its count, the signature of objects "
                        f"split apart"
                    )

        demote = (FLAG_EMPTY, FLAG_NEAR_EMPTY) if sparse_plate else ()
        if sparse_plate and (FLAG_EMPTY in qc.flags or FLAG_NEAR_EMPTY in qc.flags):
            reasons.append(
                f"the plate median is only {plate_count:.0f} objects per field, so a "
                f"sparse field here is the assay rather than a segmentation failure"
            )

        qc.flags = [f for f in FLAGS if f in qc.flags]
        qc.severity = _severity_of(qc.flags, demote=demote)
        if reasons:
            qc.note = qc.note.rstrip(".") + ". " + "; ".join(reasons) + "."
    return field_qcs


def _add(qc: FieldQC, flag: str) -> bool:
    """Append ``flag`` to ``qc`` unless it is already there; report whether it was new."""
    if flag in qc.flags:
        return False
    qc.flags.append(flag)
    return True


# ---------------------------------------------------------------------------
# plate-level entry points
# ---------------------------------------------------------------------------

def _iter_masks(source: Any):
    """Yield ``(field_name, loader)`` pairs for whatever the caller passed.

    Accepts a folder of ``.npy`` masks (what :mod:`spacr.object` writes), a
    single ``.npy`` file, a 3-D stack, a mapping of name to mask, or any
    sequence of 2-D masks. Files are yielded as thunks so only one field is in
    memory at a time — a 1536-field plate must not be loaded to be scored.
    """
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if os.path.isdir(path):
            names = sorted(f for f in os.listdir(path) if f.lower().endswith(".npy"))
            for name in names:
                full = os.path.join(path, name)
                yield name[:-4], (lambda p=full: np.load(p, allow_pickle=False))
            return
        if os.path.isfile(path):
            base = os.path.basename(path)
            stem = base[:-4] if base.lower().endswith(".npy") else base
            yield stem, (lambda p=path: np.load(p, allow_pickle=False))
            return
        raise FileNotFoundError(f"no mask folder or file at {path!r}")

    if isinstance(source, _abc.Mapping):
        for name, mask in source.items():
            yield str(name), (lambda m=mask: m)
        return

    arr = source
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        for i in range(arr.shape[0]):
            yield f"field_{i:04d}", (lambda m=arr[i]: m)
        return
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        yield "field_0000", (lambda m=arr: m)
        return

    for i, mask in enumerate(arr):
        yield f"field_{i:04d}", (lambda m=mask: m)


def score_masks(
    source: Any,
    object_type: str = "object",
    **thresholds: Any,
) -> List[FieldQC]:
    """Score every field of a plate and add the plate-relative flags.

    :param source: a folder of ``.npy`` masks (the ``<object>_mask_stack``
        folder :mod:`spacr.object` writes), a single ``.npy`` file, a 3-D mask
        stack, a ``{name: mask}`` mapping, or a sequence of 2-D masks.
    :param object_type: what was segmented.
    :param thresholds: any key of :data:`QC_DEFAULTS`.
    :returns: one :class:`FieldQC` per field, in field-name order. An empty
        list when the source holds no mask — that is not an error, it is what a
        run with ``save=False`` looks like.
    """
    th = _resolve(thresholds)
    out: List[FieldQC] = []
    for name, load in _iter_masks(source):
        try:
            mask = load()
        except Exception as exc:                       # truncated or corrupt .npy
            out.append(
                FieldQC(
                    field=name,
                    object_type=object_type,
                    n_objects=0,
                    flags=[FLAG_UNREADABLE],
                    metrics={},
                    severity="fail",
                    note=f"could not read the mask: {type(exc).__name__}: {exc}",
                )
            )
            continue
        out.append(score_field(mask, object_type=object_type, field=name, **th))
    return _apply_plate_context(out, th)


def summarize_qc(
    field_qcs: Sequence[FieldQC],
    plate_fail_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """Roll the per-field verdicts up into one plate verdict.

    :param field_qcs: what :func:`score_masks` returned.
    :param plate_fail_fraction: fraction of failing fields at which the plate
        itself is called a failure; defaults to
        ``QC_DEFAULTS['plate_fail_fraction']`` (0.10 — roughly one column of a
        96-well plate. Below that you can drop the bad fields and keep the
        plate; above it, what Measure produces is not the experiment).
    :returns: a dict with the counts, the flag tally, the failing and warning
        field names, the plate medians and a one-line ``message``.
    """
    limit = _as_number(plate_fail_fraction)
    if limit is None:
        limit = QC_DEFAULTS["plate_fail_fraction"]

    n_fields = len(field_qcs)
    object_types = sorted({q.object_type for q in field_qcs}) or [""]
    failing = [q.field for q in field_qcs if q.severity == "fail"]
    warning = [q.field for q in field_qcs if q.severity == "warn"]
    flag_counts: Dict[str, int] = {}
    for qc in field_qcs:
        for flag in qc.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    # A mask that could not be read contributes no count and no size: folding
    # its zero into the plate medians would let one corrupt file drag the
    # reference every other field is judged against.
    scored = [q for q in field_qcs if FLAG_UNREADABLE not in q.flags]
    counts = [q.metrics.get("n_objects", float(q.n_objects)) for q in scored]
    diams = [
        q.metrics.get("median_diameter", float("nan"))
        for q in scored
        if np.isfinite(q.metrics.get("median_diameter", float("nan"))) and q.n_objects
    ]
    fail_fraction = (len(failing) / n_fields) if n_fields else 0.0

    if not n_fields:
        verdict = "empty"
        message = "no mask was scored, so there is nothing to say about this plate"
    elif fail_fraction >= limit:
        verdict = "fail"
        message = (
            f"{len(failing)} of {n_fields} fields failed ({fail_fraction * 100:.0f}%): "
            f"fix the segmentation before running Measure"
        )
    elif failing or warning:
        verdict = "warn"
        message = (
            f"{len(failing)} of {n_fields} fields failed and {len(warning)} need a look; "
            f"the plate as a whole is usable"
        )
    else:
        verdict = "ok"
        message = f"all {n_fields} fields are clean"

    return {
        "object_type": object_types[0] if len(object_types) == 1 else ",".join(object_types),
        "n_fields": n_fields,
        "n_ok": sum(1 for q in field_qcs if q.severity == "ok"),
        "n_warn": len(warning),
        "n_fail": len(failing),
        "fail_fraction": fail_fraction,
        "flag_counts": flag_counts,
        "failing_fields": failing,
        "warning_fields": warning,
        "median_objects_per_field": float(np.median(counts)) if counts else 0.0,
        "median_object_diameter": float(np.median(diams)) if diams else float("nan"),
        "verdict": verdict,
        "message": message,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

_CARD_COLUMNS = (
    ("field", lambda q: q.field),
    ("objects", lambda q: f"{q.n_objects}"),
    ("border", lambda q: _pct(q.metrics.get("border_fraction"))),
    ("outliers", lambda q: _pct(q.metrics.get("outlier_fraction"))),
    ("fg", lambda q: _pct(q.metrics.get("foreground_fraction"))),
    ("median_d", lambda q: _px(q.metrics.get("median_diameter"))),
    ("severity", lambda q: q.severity),
    ("flags", lambda q: ", ".join(q.flags) if q.flags else "-"),
)

#: how many bad fields the printed card names before it stops listing them.
_MAX_CARD_ROWS = 20


def _pct(value: Optional[float]) -> str:
    """Format a fraction as a percentage, or ``'-'`` when it was not computed."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:.0f}%"


def _px(value: Optional[float]) -> str:
    """Format a pixel size, or ``'-'`` when it was not computed."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.1f}"


def format_scorecard(
    field_qcs: Sequence[FieldQC],
    plate_fail_fraction: Optional[float] = None,
    max_rows: int = _MAX_CARD_ROWS,
) -> str:
    """Render the scorecard a human reads before deciding to run Measure.

    The table lists the fields that are *not* clean — on a good plate that is
    no rows at all, and on a bad one it is the list you want. Clean fields are
    counted, not printed: a 1536-field plate must not scroll a terminal.

    :param field_qcs: what :func:`score_masks` returned.
    :param plate_fail_fraction: passed to :func:`summarize_qc`.
    :param max_rows: how many bad fields to name before summarising the rest.
    :returns: a multi-line string, ready to print.
    """
    summary = summarize_qc(field_qcs, plate_fail_fraction)
    obj = summary["object_type"] or "object"
    if not field_qcs:
        return f"Segmentation QC ({obj}): no mask found, nothing scored."

    lines = [
        f"Segmentation QC ({obj}): {summary['n_fields']} fields, "
        f"{summary['n_ok']} ok / {summary['n_warn']} warn / {summary['n_fail']} fail",
        f"  plate median {summary['median_objects_per_field']:.0f} objects per field, "
        f"median diameter {_px(summary['median_object_diameter'])} px",
        f"  verdict: {summary['verdict'].upper()} - {summary['message']}",
    ]
    if summary["flag_counts"]:
        tally = ", ".join(
            f"{flag} {summary['flag_counts'][flag]}"
            for flag in FLAGS
            if flag in summary["flag_counts"]
        )
        lines.append(f"  flags raised: {tally}")

    bad = [q for q in field_qcs if q.severity != "ok"]
    bad.sort(key=lambda q: (-_SEVERITY_ORDER.index(q.severity), q.field))
    shown = bad[:max_rows]
    if shown:
        rows = [[fmt(q) for _, fmt in _CARD_COLUMNS] for q in shown]
        header = [name for name, _ in _CARD_COLUMNS]
        widths = [
            max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))
        ]
        lines.append("")
        lines.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(header)).rstrip())
        lines.append("  " + "  ".join("-" * w for w in widths))
        for row in rows:
            lines.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
        if len(bad) > len(shown):
            lines.append(f"  ... and {len(bad) - len(shown)} more flagged field(s)")

    if summary["failing_fields"]:
        named = ", ".join(summary["failing_fields"][:max_rows])
        more = (
            f" (+{len(summary['failing_fields']) - max_rows} more)"
            if len(summary["failing_fields"]) > max_rows
            else ""
        )
        lines.append("")
        lines.append(f"  failing fields: {named}{more}")
        lines.append(
            "  Open one of them next to its raw image before running Measure: the mask, "
            "not the measurement, is where this went wrong."
        )
    return "\n".join(lines)


def write_scorecard(
    field_qcs: Sequence[FieldQC],
    dst: str,
    object_type: str = "object",
) -> Optional[str]:
    """Write one CSV row per field to ``<dst>/qc/segmentation_qc_<object_type>.csv``.

    :param field_qcs: what :func:`score_masks` returned.
    :param dst: the plate folder; the ``qc`` subfolder is created if needed.
    :param object_type: names the file.
    :returns: the path written, or None when there was nothing to write.
    """
    if not field_qcs:
        return None
    qc_dir = os.path.join(dst, "qc")
    os.makedirs(qc_dir, exist_ok=True)
    path = os.path.join(qc_dir, f"segmentation_qc_{object_type}.csv")

    # 'n_objects' is already a column of its own; carrying the metric copy too
    # would put the same header in the file twice, which csv.DictReader
    # silently collapses.
    metric_names: List[str] = []
    for qc in field_qcs:
        for name in qc.metrics:
            if name not in metric_names and name != "n_objects":
                metric_names.append(name)
    metric_names.sort()
    header = ["field", "object_type", "n_objects", "severity", "flags"] + metric_names + ["note"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for qc in field_qcs:
            row = [qc.field, qc.object_type, qc.n_objects, qc.severity, ";".join(qc.flags)]
            for name in metric_names:
                value = qc.metrics.get(name, float("nan"))
                row.append("" if value is None or not np.isfinite(value) else f"{value:.6g}")
            row.append(qc.note)
            writer.writerow(row)
    return path


def run_segmentation_qc(
    source: Any,
    object_type: str = "object",
    dst: Optional[str] = None,
    mode: str = "report",
    thresholds: Optional[Mapping[str, Any]] = None,
    verbose: bool = True,
    print_fn=print,
) -> Optional[Dict[str, Any]]:
    """Score a plate's masks, write the card, print the summary.

    This is the entry point :mod:`spacr.object` calls once per object type,
    immediately after the masks are on disk.

    :param source: whatever :func:`score_masks` accepts — normally the
        ``<object_type>_mask_stack`` folder.
    :param object_type: what was segmented.
    :param dst: plate folder the ``qc/`` subfolder is written under. None
        scores and prints without writing anything.
    :param mode: ``'off'`` does nothing at all and returns None; ``'report'``
        computes, writes and prints, changing nothing; ``'flag'`` additionally
        writes ``segmentation_qc_<object_type>_flags.json`` mapping field name
        to flags, for a downstream step to consume. No mode deletes or skips a
        field — surfacing the problem is the job.
    :param thresholds: overrides for :data:`QC_DEFAULTS`, e.g. from
        :func:`thresholds_from_settings`.
    :param verbose: True prints the full card, False prints the one-line
        verdict. The verdict is always printed: a plate that failed QC must not
        be able to fail quietly.
    :param print_fn: injection point for tests and for the GUI's log widget.
    :returns: ``{'mode', 'object_type', 'field_qcs', 'summary', 'csv_path',
        'flags'}``, or None when ``mode`` is ``'off'``.
    """
    mode = str(mode).strip().lower()
    if mode not in ("report", "flag", "stop"):
        return None

    th = dict(thresholds or {})
    field_qcs = score_masks(source, object_type=object_type, **th)
    summary = summarize_qc(field_qcs, th.get("plate_fail_fraction"))
    flags = {q.field: list(q.flags) for q in field_qcs if q.flags}

    csv_path = None
    flags_path = None
    if dst and field_qcs:
        csv_path = write_scorecard(field_qcs, dst, object_type)
        if mode in ("flag", "stop"):
            flags_path = os.path.join(dst, "qc", f"segmentation_qc_{object_type}_flags.json")
            with open(flags_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "object_type": object_type,
                        "verdict": summary["verdict"],
                        "fields": flags,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )

    if not field_qcs:
        print_fn(f"Segmentation QC ({object_type}): no mask found under {source!r}, nothing scored.")
    elif verbose:
        print_fn(format_scorecard(field_qcs, th.get("plate_fail_fraction")))
        if csv_path:
            print_fn(f"  scorecard written to {csv_path}")
    else:
        print_fn(
            f"Segmentation QC ({object_type}): {summary['verdict'].upper()} - "
            f"{summary['message']}"
            + (f" [{csv_path}]" if csv_path else "")
        )

    result = {
        "mode": mode,
        "object_type": object_type,
        "field_qcs": field_qcs,
        "summary": summary,
        "csv_path": csv_path,
        "flags_path": flags_path,
        "flags": flags,
    }

    # THE GATE. Raised LAST, after the scorecard and the flags are on disk and
    # the verdict has been printed, so stopping the run costs none of the
    # evidence for why -- a gate that stops before it writes the card leaves
    # the user with a failure and nothing to read.
    #
    # Only on `fail`. A `warn` plate is one the thresholds are unsure about,
    # and halting a plate on an unsure verdict trains people to turn the gate
    # off, which is worse than not having it.
    if mode == "stop" and summary.get("verdict") == "fail":
        raise SegmentationQCFailed(
            f"Segmentation QC ({object_type}) FAILED and seg_qc='stop': "
            f"{summary.get('message', 'the plate did not pass')}"
            + (f" The scorecard is at {csv_path}." if csv_path else "")
            + " Set seg_qc='report' to run anyway.",
            summary=summary,
        )
    return result


# ===========================================================================
# Plain language: what each flag means, what causes it, what to do
# ===========================================================================
#
# The scorecard above is precise and unreadable to anyone who has not read
# this module. "3 plates failed QC" is nearly useless; what changes a decision
# is "plate2 rows E-H hold 4x the objects of rows A-D, which is usually uneven
# illumination or a threshold set too low". Everything below exists to get
# from the first sentence to the second.
#
# One entry per member of FLAGS, checked by tests/test_seg_qc_banner.py: a
# flag added to the vocabulary without an explanation is a flag a user will
# read as a nine-letter identifier.


@dataclass(frozen=True)
class FlagGuidance:
    """One flag, in the words of someone deciding whether to run Measure.

    :param flag: the flag name, a member of :data:`FLAGS`.
    :param severity: ``'fail'`` or ``'warn'``, from :data:`_FLAG_SEVERITY`.
    :param headline: the flag in five words, for a table cell or a chip.
    :param means: what it does to the measurements — not what the mask looks
        like, but which numbers in ``measurements.db`` come out wrong.
    :param causes: the usual causes, most common first.
    :param fix: what to do next, naming the setting or the module that does it.
    :param illumination: True when uneven illumination is one of the causes,
        i.e. when :mod:`spacr.illumination` is part of the answer.
    """

    flag: str
    severity: str
    headline: str
    means: str
    causes: Tuple[str, ...]
    fix: str
    illumination: bool = False

    def __str__(self) -> str:
        return f"{self.flag}: {self.headline}"

    def text(self) -> str:
        """The whole entry as one paragraph, causes numbered."""
        causes = " ".join(
            f"({i}) {cause}." for i, cause in enumerate(self.causes, start=1)
        )
        return f"{self.means} Usually: {causes} {self.fix}"


#: The setting that decides whether Measure divides the lamp profile out.
ILLUMINATION_SETTING = "illumination_correction"

#: What to say when a flag's cause is the optics rather than the biology.
#: :mod:`spacr.illumination` estimates the field from the plate's own images;
#: the correction is applied in the *measurement* path, so it removes the
#: position bias from every intensity feature Measure is about to write, while
#: the object counts themselves only move if the masks are made again on
#: corrected images. Saying that plainly is the point — a user who switches
#: the setting on and expects the counts to change has been misled.
ILLUMINATION_ADVICE = (
    f"spacr.illumination estimates the lamp profile from this plate's own "
    f"fields and divides it out; switch {ILLUMINATION_SETTING} on and every "
    f"intensity feature Measure writes loses its position bias. The object "
    f"counts only change if the masks are generated again on corrected "
    f"images, so correct first, then re-mask if the count step survives."
)

FLAG_GUIDANCE: Dict[str, FlagGuidance] = {
    FLAG_UNREADABLE: FlagGuidance(
        flag=FLAG_UNREADABLE,
        severity=_FLAG_SEVERITY[FLAG_UNREADABLE],
        headline="the mask file could not be opened",
        means=(
            "these fields have no usable mask at all, so Measure has nothing "
            "to crop from them and they contribute no rows to "
            "measurements.db."
        ),
        causes=(
            "a mask run interrupted while writing, leaving a truncated .npy",
            "the folder being scored is not a mask stack: .npy files holding "
            "images rather than integer labels",
            "a full disk or a dropped NAS write",
        ),
        fix=(
            "Delete the named .npy files and run the mask step again for that "
            "plate; the fields that read fine are unaffected."
        ),
    ),
    FLAG_EMPTY: FlagGuidance(
        flag=FLAG_EMPTY,
        severity=_FLAG_SEVERITY[FLAG_EMPTY],
        headline="no object at all in the field",
        means=(
            "nothing to crop and nothing to measure: these fields contribute "
            "no rows, so any per-well average silently drops them."
        ),
        causes=(
            "the object channel index points at a channel that does not hold "
            "this stain (cell_channel / nucleus_channel / pathogen_channel "
            "are 0-based indices into the sorted channel IDs)",
            "a diameter far enough off that Cellpose found nothing: it "
            "rescales the image by 30/diameter, so a two-fold error moves "
            "objects out of the size range the network works at",
            "an empty, unseeded or badly out-of-focus well",
        ),
        fix=(
            "Open one named field beside its raw image. Objects in the raw "
            "image and none in the mask means the channel index or the "
            "diameter is wrong — measure the diameter from your own images "
            "rather than guessing it."
        ),
    ),
    FLAG_NEAR_EMPTY: FlagGuidance(
        flag=FLAG_NEAR_EMPTY,
        severity=_FLAG_SEVERITY[FLAG_NEAR_EMPTY],
        headline="too few objects to carry a distribution",
        means=(
            "under ten objects, so the per-field size checks were not run at "
            "all: these fields are not clean, they are unchecked."
        ),
        causes=(
            "genuinely sparse seeding or a low-MOI pathogen channel — when "
            "the whole plate is like this seg_qc already demotes it to a "
            "warning, because it is the assay and not a defect",
            "size or probability filters removing most of what was found "
            "(cell_min_area, the *_CP_prob thresholds)",
            "a field that is largely outside the well",
        ),
        fix=(
            "Compare these counts against the plate median on the card. A few "
            "sparse wells are wells to look at; a sparse plate is the assay."
        ),
    ),
    FLAG_UNDER: FlagGuidance(
        flag=FLAG_UNDER,
        severity=_FLAG_SEVERITY[FLAG_UNDER],
        headline="objects are welded together",
        means=(
            "one mask covers what should be several objects, so every "
            "per-object row from these fields is a sum over several cells: "
            "area, integrated intensity and object count are all wrong, and "
            "all wrong in the same direction."
        ),
        causes=(
            "a confluent monolayer — the foreground fraction on the card says "
            "how dense the field actually is",
            "a diameter set too large: Cellpose rescales by 30/diameter, so a "
            "diameter twice the truth shrinks the image and merges neighbours",
            "a cellprob or flow threshold permissive enough to bridge "
            "touching objects",
        ),
        fix=(
            "Measure the diameter from your own images and set "
            "<object>_diameter, then re-mask one plate and read this card "
            "again. If the field really is confluent, counts from it should "
            "not be compared with counts from sparse wells."
        ),
    ),
    FLAG_OVER: FlagGuidance(
        flag=FLAG_OVER,
        severity=_FLAG_SEVERITY[FLAG_OVER],
        headline="objects are shattered into fragments",
        means=(
            "one object has become several, so counts are inflated and every "
            "size and intensity feature is measured on a piece of a cell "
            "rather than on a cell."
        ),
        causes=(
            "a diameter set too small: Cellpose rescales by 30/diameter, so a "
            "diameter half the truth blows the image up and splits one object "
            "into several",
            "a noisy or out-of-focus channel, where the network follows "
            "texture instead of the object boundary",
            "debris and specks passing the minimum-area filter",
        ),
        fix=(
            "Measure the diameter from your own images; raise "
            "<object>_min_area so specks are dropped; check that the channel "
            "is in focus in the fields named here."
        ),
    ),
    FLAG_LOW_COUNT: FlagGuidance(
        flag=FLAG_LOW_COUNT,
        severity=_FLAG_SEVERITY[FLAG_LOW_COUNT],
        headline="far fewer objects than the rest of the plate",
        means=(
            "a quarter or less of the plate's median object count. Wells like "
            "this pull every per-well average toward whichever few objects "
            "survived, and they do it without anything looking broken "
            "downstream."
        ),
        causes=(
            "an empty, dead or badly seeded well",
            "a field out of focus, or blocked by a bubble or by debris",
            "uneven illumination: a dim corner or a dim half of the plate "
            "loses objects a bright one keeps, which shows up as a count that "
            "steps by row or by column rather than scattering at random",
        ),
        fix=(
            "If the low counts step by row or column, the optics are the "
            f"suspect: {ILLUMINATION_ADVICE} If they are scattered, open the "
            "named fields — that pattern is wells, not physics."
        ),
        illumination=True,
    ),
    FLAG_HIGH_COUNT: FlagGuidance(
        flag=FLAG_HIGH_COUNT,
        severity=_FLAG_SEVERITY[FLAG_HIGH_COUNT],
        headline="far more objects than the rest of the plate",
        means=(
            "four-fold or more above the plate median. Either these fields "
            "hold something that is not a cell, or one cell has been counted "
            "as several — both inflate every per-well count."
        ),
        causes=(
            "debris, dust or fluorescent precipitate counted as objects",
            "over-segmentation, i.e. one object split into several — the "
            "median diameter on the card will be smaller here too",
            "uneven illumination, or a threshold set too low: a bright region "
            "passes more background as foreground, and the count steps across "
            "the plate rather than scattering",
        ),
        fix=(
            "Compare the median diameter of the flagged fields with the "
            "plate's on the card. Smaller objects at a higher count is "
            "splitting; the same size at a higher count is debris or a "
            f"genuinely denser well. If the count steps across the plate: "
            f"{ILLUMINATION_ADVICE}"
        ),
        illumination=True,
    ),
    FLAG_BORDER: FlagGuidance(
        flag=FLAG_BORDER,
        severity=_FLAG_SEVERITY[FLAG_BORDER],
        headline="too many objects truncated by the field edge",
        means=(
            "30% or more of the objects touch the edge of the field, so their "
            "crops are cut off and their areas, perimeters and integrated "
            "intensities are systematically understated."
        ),
        causes=(
            "objects large relative to the field — geometry alone predicts "
            "about 2*d/W, i.e. ~8% for 60 px cells on a 1400 px field, so 30% "
            "means much bigger objects or a much smaller field",
            "border removal switched off, so truncated objects are kept and "
            "measured as if they were whole",
            "a crop or tiling step that cut the images down",
        ),
        fix=(
            "Switch <object>_remove_border_objects on so truncated objects "
            "are dropped rather than measured, or treat the size features "
            "from these fields as a lower bound."
        ),
    ),
    FLAG_OUTLIERS: FlagGuidance(
        flag=FLAG_OUTLIERS,
        severity=_FLAG_SEVERITY[FLAG_OUTLIERS],
        headline="the size distribution holds a second population",
        means=(
            "more than 15% of objects fall outside median +/- 5 robust sigma, "
            "which one population cannot do. Any per-well mean of a size or "
            "intensity feature here is an average over two different things."
        ),
        causes=(
            "debris measured alongside cells",
            "two genuine populations: mitotic or multinucleate cells, or two "
            "cell types in the same well",
            "an intensity gradient across the field making objects at the "
            "bright end measure systematically larger — uneven illumination",
        ),
        fix=(
            "Set <object>_min_area / <object>_max_area to cut the population "
            f"you do not want. If it looks positional instead: "
            f"{ILLUMINATION_ADVICE}"
        ),
        illumination=True,
    ),
}


def explain_flag(flag: str) -> FlagGuidance:
    """Return the plain-language entry for ``flag``.

    :param flag: a member of :data:`FLAGS`.
    :returns: its :class:`FlagGuidance`.
    :raises KeyError: for a flag with no entry, which is a bug in this module
        rather than a caller error — every flag it can raise is explained.
    """
    return FLAG_GUIDANCE[str(flag)]


# ---------------------------------------------------------------------------
# Where a field is: plate, well, row, column
# ---------------------------------------------------------------------------

#: ``plate1_E07_3`` — what ``spacr.io._rename_and_organize_image_files`` names
#: a merged field and what ``spacr.object`` therefore names its mask.
#: :func:`spacr.illumination.plate_of_field` takes the same first token.
_WELL_RE = re.compile(r"^([A-Za-z]{1,2})(\d{1,3})$")


@dataclass(frozen=True)
class FieldAddress:
    """Where on the plate a scored field sits.

    :param field: the field name as the scorecard holds it.
    :param plate: first underscore-separated token, or the whole stem when
        there is no underscore (a hand-assembled folder is its own plate).
    :param well: second token when it parses as a well, else ``''``.
    :param row: the well's letter part, uppercased, else ``''``.
    :param column: the well's numeric part as an int, else ``None``.
    """

    field: str
    plate: str
    well: str = ""
    row: str = ""
    column: Optional[int] = None

    @property
    def known(self) -> bool:
        """True when a well could actually be read out of the name."""
        return bool(self.well)

    def __str__(self) -> str:
        return f"{self.plate}/{self.well}" if self.well else self.plate


def parse_field_name(name: str) -> FieldAddress:
    """Split a field name into plate / well / row / column.

    Never raises and never guesses: a name that does not carry a well comes
    back with empty well, row and column, and the callers below simply do not
    make the positional claims that need them.

    :param name: a field name, file name or path — extension and directories
        are stripped.
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    parts = stem.split("_")
    plate = parts[0] if parts and parts[0] else stem
    if len(parts) >= 2:
        match = _WELL_RE.match(parts[1])
        if match:
            return FieldAddress(
                field=str(name),
                plate=plate,
                well=parts[1].upper(),
                row=match.group(1).upper(),
                column=int(match.group(2)),
            )
    return FieldAddress(field=str(name), plate=plate)


def _plate_of(qc: "FieldQC") -> str:
    """The plate a scored field belongs to."""
    return parse_field_name(qc.field).plate


def _name_list(items: Sequence[str], max_named: int) -> str:
    """``'A01, A02, A03 (+7 more)'`` — a list a human can read at a glance."""
    unique: List[str] = []
    for item in items:
        if item and item not in unique:
            unique.append(item)
    if not unique:
        return ""
    shown = ", ".join(unique[:max_named])
    extra = len(unique) - max_named
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _row_range(rows: Sequence[str]) -> str:
    """``['A','B','C','D']`` → ``'A-D'``; a gappy set is listed instead."""
    ordered = sorted({str(r).upper() for r in rows if r})
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    contiguous = all(
        len(a) == 1 and len(b) == 1 and ord(b) - ord(a) == 1
        for a, b in zip(ordered, ordered[1:])
    )
    return f"{ordered[0]}-{ordered[-1]}" if contiguous else ", ".join(ordered)


def _column_range(columns: Sequence[int]) -> str:
    """``[1,2,3]`` → ``'1-3'``; a gappy set is listed instead."""
    ordered = sorted({int(c) for c in columns})
    if not ordered:
        return ""
    if len(ordered) == 1:
        return str(ordered[0])
    contiguous = all(b - a == 1 for a, b in zip(ordered, ordered[1:]))
    return (
        f"{ordered[0]}-{ordered[-1]}" if contiguous
        else ", ".join(str(c) for c in ordered)
    )


# ---------------------------------------------------------------------------
# Findings: what to tell the user, in the order they should hear it
# ---------------------------------------------------------------------------

#: Ratio between the two halves of a plate at which a positional step stops
#: being seeding variation. Seeding density across a plate varies with a CV of
#: 10-30%, and even edge-effect wells rarely halve the median, so a two-fold
#: step between one half of the rows and the other is not the pipette.
GRADIENT_RATIO = 2.0

#: Fewest fields a half-plate needs before its median is worth comparing.
MIN_FIELDS_PER_HALF = 2

#: How many wells or fields a finding names before it says "+N more".
MAX_NAMED = 6


@dataclass
class Finding:
    """One thing worth telling the user, with the evidence attached.

    :param severity: ``'fail'``, ``'warn'`` or ``'ok'`` — how a banner should
        colour it. Never a gate: nothing in this module blocks anything.
    :param kind: ``'flag'``, ``'count_gradient'``, ``'size_gradient'`` or
        ``'clean'``. What the finding is about, for a caller that wants to
        show one kind and not another.
    :param headline: one line, with the plate and the number in it.
    :param detail: what it means for the measurements and why it happens.
    :param fix: what to do about it.
    :param flag: the seg_qc flag behind it, for ``kind='flag'``.
    :param plate: the plate it is about, when it is about one.
    :param object_type: what was segmented.
    :param wells: the wells implicated, in name order.
    :param fields: the field names implicated, in name order.
    :param n_fields: how many fields are implicated.
    :param illumination: True when uneven illumination is a named cause, so a
        caller can offer :mod:`spacr.illumination` beside the finding.
    """

    severity: str
    kind: str
    headline: str
    detail: str = ""
    fix: str = ""
    flag: str = ""
    plate: str = ""
    object_type: str = ""
    wells: Tuple[str, ...] = ()
    fields: Tuple[str, ...] = ()
    n_fields: int = 0
    illumination: bool = False

    def __str__(self) -> str:
        return f"[{self.severity}] {self.headline}"

    def text(self) -> str:
        """Headline, detail and fix as one paragraph."""
        return " ".join(p for p in (self.headline, self.detail, self.fix) if p)


def _flag_findings(
    field_qcs: Sequence["FieldQC"],
    max_named: int,
) -> List[Finding]:
    """One finding per (plate, flag), naming the wells it happened in."""
    grouped: Dict[Tuple[str, str, str], List["FieldQC"]] = {}
    for qc in field_qcs:
        plate = _plate_of(qc)
        for flag in qc.flags:
            grouped.setdefault((plate, flag, qc.object_type), []).append(qc)

    out: List[Finding] = []
    for (plate, flag, object_type), members in grouped.items():
        guidance = FLAG_GUIDANCE.get(flag)
        if guidance is None:                       # a flag with no entry
            continue
        addresses = [parse_field_name(q.field) for q in members]
        wells = tuple(sorted({a.well for a in addresses if a.well}))
        fields = tuple(sorted(q.field for q in members))
        where = plate or "this project"
        located = _name_list(list(wells), max_named)
        if located:
            where += f", wells {located}"
        elif fields:
            where += f", fields {_name_list(list(fields), max_named)}"
        # The severity is the flag's own, with one exception that has to be
        # honoured: `_apply_plate_context` demotes empty and near-empty fields
        # on a sparse plate, because with a plate median of 2 pathogens per
        # field a field with none is the assay. `_severity_of` takes the worst
        # flag, so a field carrying an undemoted 'fail' flag is itself 'fail';
        # if not one member field is, the flag was demoted, and calling it a
        # failure here would contradict the verdict the card already printed.
        severity = _FLAG_SEVERITY.get(flag, "warn")
        if severity == "fail" and not any(q.severity == "fail" for q in members):
            severity = "warn"
        out.append(Finding(
            severity=severity,
            kind="flag",
            headline=(
                f"{len(members)} {object_type or 'object'} field(s) on {where}: "
                f"{guidance.headline}"
            ),
            detail=guidance.means + " Usually: " + " ".join(
                f"({i}) {cause}." for i, cause in enumerate(guidance.causes, 1)
            ),
            fix=guidance.fix,
            flag=flag,
            plate=plate,
            object_type=object_type,
            wells=wells,
            fields=fields,
            n_fields=len(members),
            illumination=guidance.illumination,
        ))
    return out


def _halves(keys: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Split a sorted key list into a first and a second half."""
    ordered = sorted(set(keys))
    if len(ordered) < 2:
        return [], []
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]


def _axis_step(
    per_key: Dict[Any, List[float]],
    ratio: float,
    min_fields: int,
) -> Optional[Tuple[List[Any], List[Any], float, float, float]]:
    """Compare the two halves of one plate axis.

    :param per_key: row letter (or column number) → the values measured in it.
    :param ratio: fold difference at which the step is reported.
    :param min_fields: fewest fields a half needs before it is compared.
    :returns: ``(low_keys, high_keys, low_median, high_median, fold)`` for the
        halves as ordered on the plate, or None when there is no step. The
        first element is always the *lower* half by value, so the caller can
        say "E-H hold 4x A-D" without working out the direction again.
    """
    first, second = _halves(list(per_key))
    if not first or not second:
        return None
    a = [v for key in first for v in per_key[key]]
    b = [v for key in second for v in per_key[key]]
    if len(a) < min_fields or len(b) < min_fields:
        return None
    ma, mb = float(np.median(a)), float(np.median(b))
    lo, hi = (ma, mb) if ma <= mb else (mb, ma)
    lo_keys, hi_keys = (first, second) if ma <= mb else (second, first)
    if lo <= 0:
        # A half with a median of zero is an empty half, not a gradient; the
        # empty-field flag is the honest report of that and already fired.
        return None
    fold = hi / lo
    if fold < ratio:
        return None
    return lo_keys, hi_keys, lo, hi, fold


def _gradient_findings(
    field_qcs: Sequence["FieldQC"],
    ratio: float,
    min_fields: int,
) -> List[Finding]:
    """The findings no single field can produce: steps across the plate.

    A count that steps two-fold or more between one half of a plate's rows
    and the other is the signature this whole module exists to name. It is
    invisible per field — every field on the bright half can be individually
    unremarkable — and it is the thing that changes what a user does next.
    """
    by_plate: Dict[Tuple[str, str], List[Tuple[FieldAddress, "FieldQC"]]] = {}
    for qc in field_qcs:
        if FLAG_UNREADABLE in qc.flags:
            continue
        address = parse_field_name(qc.field)
        if not address.known:
            continue
        by_plate.setdefault((address.plate, qc.object_type), []).append(
            (address, qc))

    out: List[Finding] = []
    for (plate, object_type), members in sorted(by_plate.items()):
        for axis, label, keyer, render in (
            ("rows", "row", lambda a: a.row, _row_range),
            ("columns", "column", lambda a: a.column, _column_range),
        ):
            counts: Dict[Any, List[float]] = {}
            sizes: Dict[Any, List[float]] = {}
            for address, qc in members:
                key = keyer(address)
                if key in (None, ""):
                    continue
                counts.setdefault(key, []).append(float(qc.n_objects))
                diameter = qc.metrics.get("median_diameter", float("nan"))
                if np.isfinite(diameter) and diameter > 0:
                    sizes.setdefault(key, []).append(float(diameter))

            step = _axis_step(counts, ratio, min_fields)
            if step is not None:
                lo_keys, hi_keys, lo, hi, fold = step
                lo_name, hi_name = render(lo_keys), render(hi_keys)
                wells = tuple(sorted({
                    a.well for a, _ in members
                    if keyer(a) in set(hi_keys) and a.well
                }))
                out.append(Finding(
                    severity="fail",
                    kind="count_gradient",
                    headline=(
                        f"{plate}: {axis} {hi_name} hold {fold:.1f}x the "
                        f"{object_type or 'object'} count of {axis} {lo_name} "
                        f"({hi:.0f} vs {lo:.0f} objects per field)"
                    ),
                    detail=(
                        f"An object count that steps from one side of a plate "
                        f"to the other is rarely biology: seeding varies with "
                        f"a CV of 10-30% and scatters, it does not sort itself "
                        f"by {label}. The two usual causes are uneven "
                        f"illumination — a lamp profile or a vignette makes "
                        f"one region brighter, so more of it passes the "
                        f"segmentation threshold — and a threshold set too low "
                        f"for the dimmer half."
                    ),
                    fix=(
                        f"{ILLUMINATION_ADVICE} If the raw images look evenly "
                        f"lit, the threshold is the other suspect: check "
                        f"<object>_CP_prob and <object>_min_area, and measure "
                        f"the diameter from your own images before re-masking."
                    ),
                    plate=plate,
                    object_type=object_type,
                    wells=wells,
                    n_fields=len(members),
                    illumination=True,
                ))

            step = _axis_step(sizes, ratio, min_fields)
            if step is not None:
                lo_keys, hi_keys, lo, hi, fold = step
                lo_name, hi_name = render(lo_keys), render(hi_keys)
                out.append(Finding(
                    severity="warn",
                    kind="size_gradient",
                    headline=(
                        f"{plate}: {object_type or 'object'}s in {axis} "
                        f"{hi_name} measure {fold:.1f}x the diameter of those "
                        f"in {axis} {lo_name} ({hi:.1f} vs {lo:.1f} px)"
                    ),
                    detail=(
                        "The same object type measuring systematically larger "
                        "on one side of a plate means the segmentation is "
                        "drawing bigger outlines there, not that the cells "
                        "grew. An intensity gradient does exactly this: a "
                        "brighter region pushes more of each object's halo "
                        "over the threshold. Every size feature on this plate "
                        "then carries a position term."
                    ),
                    fix=ILLUMINATION_ADVICE,
                    plate=plate,
                    object_type=object_type,
                    n_fields=len(members),
                    illumination=True,
                ))
    return out


def diagnose(
    field_qcs: Sequence["FieldQC"],
    *,
    gradient_ratio: float = GRADIENT_RATIO,
    min_fields_per_half: int = MIN_FIELDS_PER_HALF,
    max_named: int = MAX_NAMED,
) -> List[Finding]:
    """Turn per-field verdicts into findings that name plates, wells and causes.

    Two kinds, because they answer different questions:

    * **flag findings** — one per (plate, flag): which wells, how many fields,
      what that flag does to the measurements and what to do about it. This is
      the scorecard, grouped and translated.
    * **gradient findings** — a plate whose object count or object size steps
      between one half of its rows (or columns) and the other. No single
      field can raise this and no field-level flag implies it: a plate can be
      entirely free of per-field flags and still be lit unevenly enough to put
      a position term in every intensity feature.

    :param field_qcs: what :func:`score_masks` returned, or what
        :func:`read_scorecard` read back off disk.
    :param gradient_ratio: fold step between plate halves at which the
        positional findings fire.
    :param min_fields_per_half: fewest fields a half-plate needs before its
        median is compared.
    :param max_named: how many wells or fields a finding names.
    :returns: findings, failures first, then by how many fields each implicates.
        An empty list means there is nothing to say.
    """
    if not field_qcs:
        return []
    findings = _flag_findings(field_qcs, max_named)
    findings += _gradient_findings(
        field_qcs, gradient_ratio, min_fields_per_half)
    findings.sort(
        key=lambda f: (
            -_SEVERITY_ORDER.index(f.severity),
            -f.n_fields,
            f.plate,
            f.flag or f.kind,
        )
    )
    return findings


def format_findings(findings: Sequence[Finding]) -> str:
    """Render findings as text, one block each."""
    if not findings:
        return "Nothing flagged."
    blocks = []
    for finding in findings:
        blocks.append(f"[{finding.severity.upper()}] {finding.headline}")
        if finding.detail:
            blocks.append(f"    {finding.detail}")
        if finding.fix:
            blocks.append(f"    -> {finding.fix}")
    return "\n".join(blocks)


# ===========================================================================
# Reading the verdict back off disk
# ===========================================================================
#
# `run_segmentation_qc` already scored these masks once, at mask time, and
# wrote `<plate>/qc/segmentation_qc_<object_type>.csv`. Everything below reads
# that file. Nothing below opens a mask -- opening a plate's worth of masks
# costs seconds to minutes, and a screen that pays that on every visit is a
# screen that gets switched off. `score_digest` is the one exception and it
# only runs when a user asks for it by name.
#
# Freshness is the price of not recomputing, and it is paid explicitly: each
# card's mtime is compared against the newest file in the mask stack it
# describes, so a card written before the last re-mask is reported as OUT OF
# DATE rather than quietly believed.

#: Prefix `write_scorecard` gives every card it writes.
CARD_PREFIX = "segmentation_qc_"

#: Subfolder of the plate that holds them.
CARD_DIR = "qc"

#: Suffix `spacr.object` gives every mask stack folder it writes.
MASK_STACK_SUFFIX = "_mask_stack"

#: How many plate subfolders a project root is walked into looking for cards.
#: A project holds plates, and a plate holds a `qc/`; two levels is the whole
#: convention, and bounding it keeps a src pointed at `/` from walking a disk.
_MAX_PLATE_DIRS = 64

#: How many mask files are stat'ed when dating a mask stack. The newest of a
#: sample is the newest of the folder in every case that matters, and a
#: 1536-field plate must not cost 1536 stats on every screen visit.
_MAX_MTIME_STATS = 512


@dataclass
class Scorecard:
    """One ``qc/segmentation_qc_<object_type>.csv``, read back.

    :param path: where it was read from.
    :param object_type: what it scored, taken from the file name.
    :param field_qcs: the rows, rebuilt as :class:`FieldQC`.
    :param summary: what :func:`summarize_qc` makes of them.
    :param mtime: the card's modification time.
    :param masks_mtime: the newest mask file in the stack it describes, or
        ``0.0`` when that stack could not be found.
    :param stale: True when a mask is newer than the card, i.e. the plate was
        re-masked and nothing has scored the new masks.
    :param error: why the card could not be read, when it could not be.
    """

    path: str
    object_type: str
    field_qcs: List["FieldQC"] = _dc_field(default_factory=list)
    summary: Dict[str, Any] = _dc_field(default_factory=dict)
    mtime: float = 0.0
    masks_mtime: float = 0.0
    stale: bool = False
    error: str = ""

    @property
    def verdict(self) -> str:
        """``'ok'``, ``'warn'``, ``'fail'``, ``'empty'`` or ``'error'``."""
        if self.error:
            return "error"
        return str(self.summary.get("verdict", "empty"))

    def __str__(self) -> str:
        stale = " (out of date)" if self.stale else ""
        return f"{self.object_type}: {self.verdict}{stale}"


@dataclass
class QCDigest:
    """Everything a screen needs to show a segmentation verdict.

    Advisory by construction: there is no field here a caller is meant to
    branch a Run button on, and :attr:`blocks_run` is a constant False that
    exists to say so in code rather than only in a docstring.

    :param root: the project folder the cards were found under.
    :param verdict: ``'ok'``, ``'warn'``, ``'fail'``, ``'missing'`` (nothing
        has scored these masks) or ``'error'``.
    :param headline: the single most actionable sentence — the worst finding,
        with its plate and its number in it. What a banner puts in bold.
    :param subhead: the counts behind it: how many fields, how many failed.
    :param scorecards: one per object type found.
    :param findings: what :func:`diagnose` made of every card together.
    :param stale: True when any card is older than its masks.
    :param checked_at: when this digest was built (``time.time()``).
    """

    root: str = ""
    verdict: str = "missing"
    headline: str = ""
    subhead: str = ""
    scorecards: List[Scorecard] = _dc_field(default_factory=list)
    findings: List[Finding] = _dc_field(default_factory=list)
    stale: bool = False
    checked_at: float = 0.0

    #: Nothing in this module gates anything. A segmentation verdict informs
    #: a decision the user makes; a plate that fails QC is still a plate they
    #: may have good reason to measure, and a QC report that stops them is a
    #: QC report they will switch off.
    blocks_run: bool = False

    @property
    def n_fields(self) -> int:
        """Fields scored across every card."""
        return sum(len(card.field_qcs) for card in self.scorecards)

    @property
    def object_types(self) -> Tuple[str, ...]:
        """What was segmented, in card order."""
        return tuple(card.object_type for card in self.scorecards)

    @property
    def failing_fields(self) -> Tuple[str, ...]:
        """Every field any card scored ``'fail'``."""
        return tuple(
            q.field for card in self.scorecards for q in card.field_qcs
            if q.severity == "fail"
        )

    @property
    def plates(self) -> Tuple[str, ...]:
        """The plates the cards cover, sorted."""
        return tuple(sorted({
            _plate_of(q) for card in self.scorecards for q in card.field_qcs
        }))

    def __str__(self) -> str:
        return f"{self.verdict}: {self.headline}"


def qc_roots(src: Any) -> Tuple[str, ...]:
    """Project folders to look for scorecards under, best first.

    Handles the shapes ``settings['src']`` actually takes:

    * a plate folder — cards are in ``<src>/qc``;
    * the merged folder Measure is usually pointed at (``<plate>/merged``) —
      cards are one level up, the same hop
      :func:`spacr.ports.project_root` makes;
    * a list of plates, which is how several plates are run at once;
    * a project root holding plate subfolders, each with its own ``qc/``.

    :param src: a path, a list of paths, or None.
    :returns: existing folders, de-duplicated, in the order above.
    """
    candidates: List[str] = []

    def _add(path: str) -> None:
        if path and os.path.isdir(path) and path not in candidates:
            candidates.append(path)

    values: List[Any]
    if src is None:
        values = []
    elif isinstance(src, (str, os.PathLike)):
        values = [src]
    elif isinstance(src, (list, tuple, set)):
        values = list(src)
    else:
        values = []

    for value in values:
        if not isinstance(value, (str, os.PathLike)):
            continue
        text = str(value).strip()
        if not text or text in ("path", "/path", "/path/to/src"):
            continue
        root = os.path.abspath(os.path.expanduser(text))
        _add(root)
        # `<plate>/merged` and `<plate>/norm_channel_stack` both name a folder
        # INSIDE the plate; the card lives beside them, not in them.
        parent = os.path.dirname(os.path.normpath(root))
        base = os.path.basename(os.path.normpath(root))
        if base.endswith("merged") or base.endswith("_stack"):
            _add(parent)
        if not os.path.isdir(os.path.join(root, CARD_DIR)):
            try:
                children = sorted(os.listdir(root))[:_MAX_PLATE_DIRS]
            except OSError:
                children = []
            for child in children:
                sub = os.path.join(root, child)
                if os.path.isdir(os.path.join(sub, CARD_DIR)):
                    _add(sub)
    return tuple(candidates)


def find_scorecards(src: Any) -> Tuple[str, ...]:
    """Every ``qc/segmentation_qc_<object>.csv`` under ``src``.

    :param src: whatever :func:`qc_roots` accepts.
    :returns: absolute paths, sorted, flag sidecars excluded.
    """
    out: List[str] = []
    for root in qc_roots(src):
        qc_dir = os.path.join(root, CARD_DIR)
        try:
            names = sorted(os.listdir(qc_dir))
        except OSError:
            continue
        for name in names:
            if name.startswith(CARD_PREFIX) and name.lower().endswith(".csv"):
                path = os.path.join(qc_dir, name)
                if os.path.isfile(path) and path not in out:
                    out.append(path)
    return tuple(out)


def _object_type_of_card(path: str) -> str:
    """``segmentation_qc_cell.csv`` → ``cell``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len(CARD_PREFIX):] if stem.startswith(CARD_PREFIX) else stem


def read_scorecard(path: str) -> Tuple[List["FieldQC"], str]:
    """Rebuild :class:`FieldQC` rows from a scorecard CSV.

    The rows are handed back to :func:`summarize_qc` and :func:`diagnose`
    rather than re-derived, so what a screen shows is the verdict the run
    printed and not a second opinion that could disagree with it.

    :param path: a ``qc/segmentation_qc_<object_type>.csv``.
    :returns: ``(field_qcs, error)``. A damaged card comes back as
        ``([], reason)``: half a card is a different verdict, and this module
        does not invent one.
    """
    reserved = {"field", "object_type", "n_objects", "severity", "flags", "note"}
    #: A card without these is not a card. Checked against the HEADER, because
    #: every field below has a default and a row of defaults reads as a clean
    #: verdict -- `severity` alone falls back to "ok", which is the invented
    #: verdict this function exists to refuse.
    required = {"field", "n_objects"}
    out: List["FieldQC"] = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])

            # THE NUL CHECK HAS TO LOOK AT THE HEADER, and this is why.
            # CPython's csv module used to raise `_csv.Error: line contains
            # NUL`, so a corrupted card was caught by the `except csv.Error`
            # below and reported as unreadable. PYTHON 3.12 STOPPED RAISING:
            # it parses the NUL straight through into a FIELD NAME. The check
            # here only ever looked at `row.values()`, so a NUL in the header
            # became a key it could not see, every real column went missing,
            # every default applied, and a damaged scorecard came back "ok"
            # on exactly the interpreters this project targets.
            if any("\x00" in str(name) for name in header):
                return [], f"{os.path.basename(path)} is not CSV (NUL byte)"
            missing = sorted(required - set(header))
            if header and missing:
                return [], (f"{os.path.basename(path)} has no "
                            f"{', '.join(missing)} column, so it carries no "
                            f"verdict to read")

            for row in reader:
                if any("\x00" in str(v) for v in row.values() if v is not None):
                    return [], f"{os.path.basename(path)} is not CSV (NUL byte)"
                metrics: Dict[str, float] = {}
                for key, value in row.items():
                    if key in reserved or key is None:
                        continue
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        metrics[key] = float("nan")
                try:
                    n_objects = int(float(row.get("n_objects") or 0))
                except (TypeError, ValueError):
                    n_objects = 0
                metrics.setdefault("n_objects", float(n_objects))
                out.append(FieldQC(
                    field=str(row.get("field") or ""),
                    object_type=str(row.get("object_type") or ""),
                    n_objects=n_objects,
                    flags=[f for f in str(row.get("flags") or "").split(";") if f],
                    metrics=metrics,
                    severity=str(row.get("severity") or "ok"),
                    note=str(row.get("note") or ""),
                ))
    except OSError as exc:
        return [], f"{os.path.basename(path)} unreadable ({type(exc).__name__})"
    except csv.Error as exc:
        return [], f"{os.path.basename(path)} is not readable as CSV ({exc})"
    return out, ""


def find_mask_stacks(root: str) -> Dict[str, str]:
    """``{object_type: folder}`` for the mask stacks under a plate.

    ``spacr.object`` writes ``<src>/<object_type>_mask_stack``, where ``src``
    is a folder inside the plate (``norm_channel_stack``), so the stacks sit
    one level below the plate root the card is written at. Both levels are
    checked, and neither is walked further.

    :param root: a plate folder.
    """
    found: Dict[str, str] = {}
    levels: List[str] = [root]
    try:
        levels += [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))[:_MAX_PLATE_DIRS]
            if os.path.isdir(os.path.join(root, name))
        ]
    except OSError:
        return found
    for level in levels:
        try:
            names = sorted(os.listdir(level))
        except OSError:
            continue
        for name in names:
            if not name.endswith(MASK_STACK_SUFFIX):
                continue
            path = os.path.join(level, name)
            if os.path.isdir(path):
                found.setdefault(name[:-len(MASK_STACK_SUFFIX)], path)
    return found


def mask_stack_mtime(folder: str) -> float:
    """Modification time of the newest mask in ``folder``, or ``0.0``.

    The folder's own mtime is included: ``spacr.object`` writes masks by
    atomic rename, which touches the directory, so a stack whose files were
    all replaced is caught even if the sample below misses the newest one.

    :param folder: a ``<object_type>_mask_stack`` folder.
    """
    newest = 0.0
    try:
        newest = float(os.stat(folder).st_mtime)
    except OSError:
        return 0.0
    try:
        with os.scandir(folder) as entries:
            for index, entry in enumerate(entries):
                if index >= _MAX_MTIME_STATS:
                    break
                if not entry.name.lower().endswith(".npy"):
                    continue
                try:
                    newest = max(newest, float(entry.stat().st_mtime))
                except OSError:
                    continue
    except OSError:
        pass
    return newest


def _subhead(digest: "QCDigest") -> str:
    """The counts behind the verdict, in one line."""
    n_fields = digest.n_fields
    types = ", ".join(t for t in digest.object_types if t) or "object"
    plates = len(digest.plates)
    where = f" across {plates} plates" if plates > 1 else ""
    n_fail = sum(int(c.summary.get("n_fail", 0)) for c in digest.scorecards)
    n_warn = sum(int(c.summary.get("n_warn", 0)) for c in digest.scorecards)
    if not n_fail and not n_warn:
        if digest.findings:
            # Worth saying out loud: every field passed on its own, and the
            # problem is only visible when they are laid out on the plate.
            return (
                f"{n_fields} {types} field(s){where} scored; no single field "
                f"was flagged — what is wrong is the pattern across the plate."
            )
        return f"{n_fields} {types} field(s){where} scored, none flagged."
    return (
        f"{n_fail} of {n_fields} {types} field(s){where} failed and "
        f"{n_warn} need a look."
    )


def _headline(digest: "QCDigest") -> str:
    """The one sentence worth putting in bold."""
    if digest.verdict == "missing":
        return (
            "Nothing has scored these masks. That is not the same as clean — "
            "run the mask step with seg_qc on, or score the masks from here."
        )
    if digest.verdict == "error":
        errors = [c.error for c in digest.scorecards if c.error]
        return "The segmentation scorecard could not be read: " + (
            errors[0] if errors else "unknown error")
    if digest.findings:
        # The worst finding, not the counts: "12 of 96 fields failed" is a
        # number, "plate2 rows E-H hold 4x the count of rows A-D" is a
        # decision. `diagnose` has already sorted the worst one to the front.
        return digest.findings[0].headline
    return "Segmentation QC passed: nothing was flagged on these masks."


def _digest_from_cards(root: str, cards: List[Scorecard], **kwargs) -> QCDigest:
    """Roll a set of read cards up into one digest.

    The verdict is the worst of what the *fields* say (the per-card verdict
    `summarize_qc` produced at mask time) and what the *plate* says (the
    positional findings). Both directions matter: a plate can fail on its
    fields with no gradient at all, and — the case that motivated this — a
    plate can have no flagged field whatever and still step four-fold in
    object count between one half of its rows and the other, which is a
    failure nothing per-field can see.
    """
    digest = QCDigest(root=root, scorecards=cards, checked_at=time.time())
    readable = [c for c in cards if not c.error]
    digest.findings = diagnose(
        [q for card in readable for q in card.field_qcs], **kwargs)
    if not cards:
        digest.verdict = "missing"
    elif not readable:
        digest.verdict = "error"
    else:
        levels = [c.verdict for c in readable if c.verdict in _SEVERITY_ORDER]
        levels += [f.severity for f in digest.findings]
        if not levels:
            digest.verdict = "missing"
        else:
            digest.verdict = max(levels, key=_SEVERITY_ORDER.index)
    digest.stale = any(c.stale for c in cards)
    digest.subhead = _subhead(digest)
    digest.headline = _headline(digest)
    return digest


def read_digest(src: Any, **kwargs) -> QCDigest:
    """Read the segmentation verdict for a project. Scores nothing.

    The cheap path, and the one a screen calls: locate the scorecards the
    mask run already wrote, parse them, roll them up, diagnose them, and date
    each one against its mask stack.

    :param src: whatever :func:`qc_roots` accepts — a plate folder, a merged
        folder, a list of plates or a project root.
    :param kwargs: passed to :func:`diagnose`.
    :returns: a :class:`QCDigest`. ``verdict == 'missing'`` when no card
        exists, which is emphatically not the same as ``'ok'``.
    """
    paths = find_scorecards(src)
    root = os.path.dirname(os.path.dirname(paths[0])) if paths else (
        (qc_roots(src) or ("",))[0])
    cards: List[Scorecard] = []
    stacks_by_root: Dict[str, Dict[str, str]] = {}
    for path in paths:
        object_type = _object_type_of_card(path)
        field_qcs, error = read_scorecard(path)
        summary = summarize_qc(field_qcs) if field_qcs else {}
        card_root = os.path.dirname(os.path.dirname(path))
        if card_root not in stacks_by_root:
            stacks_by_root[card_root] = find_mask_stacks(card_root)
        stack = stacks_by_root[card_root].get(object_type, "")
        try:
            mtime = float(os.stat(path).st_mtime)
        except OSError:
            mtime = 0.0
        masks_mtime = mask_stack_mtime(stack) if stack else 0.0
        cards.append(Scorecard(
            path=path,
            object_type=object_type,
            field_qcs=field_qcs,
            summary=summary,
            mtime=mtime,
            masks_mtime=masks_mtime,
            # Only a mask that is genuinely newer counts. Equal mtimes are a
            # coarse filesystem timestamp on a card written moments after the
            # masks, which is the normal case and not a staleness.
            stale=bool(masks_mtime and mtime and masks_mtime > mtime + 1.0),
            error=error,
        ))
    return _digest_from_cards(root, cards, **kwargs)


def score_digest(
    src: Any,
    object_types: Sequence[str] = (),
    thresholds: Optional[Mapping[str, Any]] = None,
    write: bool = True,
    **kwargs,
) -> QCDigest:
    """Score the mask stacks under ``src`` and return the same digest.

    The expensive path — it opens every mask — so it is never called to draw
    a screen. It exists for the user who has just been told the verdict is
    missing or out of date and asks for it to be brought up to date; writing
    the cards is what makes the next :func:`read_digest` cheap again.

    :param src: whatever :func:`qc_roots` accepts.
    :param object_types: which stacks to score; empty means all that exist.
    :param thresholds: overrides for :data:`QC_DEFAULTS`, e.g. from
        :func:`thresholds_from_settings`.
    :param write: write each card back to ``<plate>/qc/``.
    :param kwargs: passed to :func:`diagnose`.
    """
    th = dict(thresholds or {})
    wanted = {str(o) for o in object_types}
    cards: List[Scorecard] = []
    roots = qc_roots(src)
    for root in roots:
        for object_type, folder in sorted(find_mask_stacks(root).items()):
            if wanted and object_type not in wanted:
                continue
            field_qcs = score_masks(folder, object_type=object_type, **th)
            if not field_qcs:
                continue
            path = write_scorecard(field_qcs, root, object_type) if write else ""
            cards.append(Scorecard(
                path=path or os.path.join(root, CARD_DIR,
                                          f"{CARD_PREFIX}{object_type}.csv"),
                object_type=object_type,
                field_qcs=field_qcs,
                summary=summarize_qc(field_qcs, th.get("plate_fail_fraction")),
                mtime=time.time(),
                masks_mtime=mask_stack_mtime(folder),
                stale=False,
            ))
    return _digest_from_cards(roots[0] if roots else "", cards, **kwargs)


def format_digest(digest: QCDigest) -> str:
    """Render a digest as text — what the console prints, what a test reads."""
    lines = [f"Segmentation QC: {digest.verdict.upper()} — {digest.headline}"]
    if digest.subhead:
        lines.append(f"  {digest.subhead}")
    if digest.root:
        lines.append(f"  project: {digest.root}")
    for card in digest.scorecards:
        state = " [OUT OF DATE: masks are newer than this card]" if card.stale else ""
        if card.error:
            lines.append(f"  {card.object_type}: {card.error}{state}")
            continue
        lines.append(
            f"  {card.object_type}: {card.summary.get('message', '')}{state}"
        )
    if digest.findings:
        lines.append("")
        lines.append(format_findings(digest.findings))
    return "\n".join(lines)
