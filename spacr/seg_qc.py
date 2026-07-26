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
"""
from __future__ import annotations

import collections.abc as _abc
import csv
import json
import math
import os
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FieldQC",
    "FLAGS",
    "QC_DEFAULTS",
    "SETTING_KEYS",
    "format_scorecard",
    "qc_mode",
    "run_segmentation_qc",
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
MODES: Tuple[str, ...] = ("off", "report", "flag")


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
    if str(mode).strip().lower() not in ("report", "flag"):
        return None

    th = dict(thresholds or {})
    field_qcs = score_masks(source, object_type=object_type, **th)
    summary = summarize_qc(field_qcs, th.get("plate_fail_fraction"))
    flags = {q.field: list(q.flags) for q in field_qcs if q.flags}

    csv_path = None
    flags_path = None
    if dst and field_qcs:
        csv_path = write_scorecard(field_qcs, dst, object_type)
        if mode == "flag":
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

    return {
        "mode": mode,
        "object_type": object_type,
        "field_qcs": field_qcs,
        "summary": summary,
        "csv_path": csv_path,
        "flags_path": flags_path,
        "flags": flags,
    }
