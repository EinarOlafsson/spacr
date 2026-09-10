"""Estimate ``fraction_threshold`` from control-well concordance.

``fraction_threshold`` removes a gRNA from a well when its share of reads is
below the selected limit. This changes the number and normalized abundance of
guides assigned to each well and therefore affects downstream coefficients.

For each candidate threshold, this module compares the positive-control read
fraction from sequencing with the positive phenotype fraction from imaging.
Candidates are scored by the median absolute well-level disagreement,
``median(abs(imaging - sequencing))``. The fit uses the median of pairwise
slopes to reduce sensitivity to one-sided contamination from screen hits.
Slope, intercept, residual, disagreement and guides per well are retained for
review at every candidate.

The selected threshold establishes concordance within control wells; it does
not estimate performance in wells containing a complex guide library.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_THRESHOLD_CANDIDATES",
    "MINIMUM_CALIBRATION_WELLS",
    "compare_normalisations",
    "describe",
    "reported_control_share",
    "sweep_fraction_threshold",
    "well_fractions",
]

#: The thresholds swept when a caller names none. 0.0 is included because
#: "keep everything" is a real answer and the sweep has to be able to return
#: it; the grid is denser where the default lives.
DEFAULT_THRESHOLD_CANDIDATES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05,
                                0.075, 0.1, 0.15, 0.2)

#: Fewer control wells than this and the sweep refuses rather than reporting.
#:
#: A median fitted over a handful of wells is decided by which wells happened
#: to be usable, and the whole point of the sweep is to compare fits: a
#: difference between two thresholds means nothing when either could move by
#: dropping one well.
MINIMUM_CALIBRATION_WELLS = 8


def _well_labels(counts: pd.DataFrame, well_column: str) -> pd.Series:
    """The well each count row belongs to, built from the plate keys if need be.

    A count table names its well either as ``prc`` or as the three parts it is
    made of. Building it here rather than demanding it means the sweep reads
    the file the sequencing step wrote, not a prepared version of it.
    """
    if well_column in counts.columns:
        return counts[well_column].astype(str)
    parts = ("plateID", "rowID", "columnID")
    missing = [name for name in parts if name not in counts.columns]
    if missing:
        raise KeyError(
            f"the count table has no {well_column!r} column and cannot build "
            f"one: {missing} are missing too")
    return (counts["plateID"].astype(str) + "_"
            + counts["rowID"].astype(str) + "_"
            + counts["columnID"].astype(str))


def well_fractions(counts: pd.DataFrame, *, threshold: float = 0.0,
                   normalise: bool = True, well_column: str = "prc",
                   guide_column: str = "grna",
                   count_column: str = "count") -> pd.DataFrame:
    """Each gRNA's share of its well, under one candidate threshold.

    :param counts: one row per gRNA per well, with a read count.
    :param threshold: drop a gRNA whose share is below this.
    :param normalise: rescale the survivors of one well to sum to one, which
        is what the pipeline's ``normalise_fraction`` does. Off, a share is
        measured against every read the well produced, including the reads
        the threshold discarded.
    :returns: the surviving rows with a ``fraction`` column.

    The two settings interact and that is the point of sweeping them
    together: normalising raises every surviving share, and by more the more
    the threshold removed.
    """
    frame = counts.copy()
    frame[well_column] = _well_labels(counts, well_column)
    totals = frame.groupby(well_column)[count_column].transform("sum")
    # A well with no reads at all has no shares to compute; it drops out here
    # rather than becoming a column of NaN that reads as a measurement.
    frame = frame[totals > 0].copy()
    frame["fraction"] = frame[count_column] / totals[totals > 0]
    if threshold:
        frame = frame[frame["fraction"] >= float(threshold)].copy()
    if normalise and len(frame):
        kept = frame.groupby(well_column)["fraction"].transform("sum")
        frame = frame[kept > 0].copy()
        frame["fraction"] = frame["fraction"] / kept[kept > 0]
    return frame


def reported_control_share(fractions: pd.DataFrame, positive_guide: str, *,
                           well_column: str = "prc",
                           guide_column: str = "grna") -> Dict[str, float]:
    """What sequencing says the positive control's share of each well is.

    :param fractions: per-well guide fractions, typically returned by
        :func:`well_fractions`.
    :param positive_guide: guide identifier whose share is reported.
    :returns: ``{well: fraction}``, zero for a well the control did not
        survive the threshold in -- which is a measurement, not a gap: the
        threshold decided that guide was not there.
    """
    wanted = str(positive_guide)
    out: Dict[str, float] = {}
    for well, rows in fractions.groupby(fractions[well_column].astype(str)):
        here = rows[rows[guide_column].astype(str) == wanted]
        out[str(well)] = float(here["fraction"].sum()) if len(here) else 0.0
    return out


def _disagreement(per_well: Mapping[str, Any],
                  correction: Mapping[str, float]) -> float:
    """How far apart the two measurements are, well by well, at the median.

    The imaging side is corrected first when a confusion matrix was supplied:
    a classifier's call rate is not the cellular proportion, and the
    correction matters most exactly where the science is -- a false-positive
    rate applied to a well that is nearly all negatives swamps a true-positive
    rate applied to its few positives.
    """
    if not per_well:
        return float("nan")
    gaps = []
    for said, seen in per_well.values():
        value = float(seen)
        if correction:
            from .classifier_quality import rogan_gladen

            fixed = rogan_gladen(value, correction["sensitivity"],
                                 correction["specificity"])
            if fixed.get("usable"):
                value = float(fixed["corrected"])
        gaps.append(abs(value - float(said)))
    return float(np.median(gaps))


def _training_well_count(wells: Sequence[str], columns: Sequence[int]) -> int:
    """How many of these wells were used to fit the classifier."""
    from .classifier_quality import training_wells

    if not len(list(wells)):
        return 0
    return int(np.count_nonzero(training_wells(list(wells), columns=columns)))


def sweep_fraction_threshold(counts: pd.DataFrame,
                             features: np.ndarray,
                             wells: Sequence[str], *,
                             positive_guide: str,
                             pure_pc_wells: Sequence[str],
                             pure_nc_wells: Sequence[str],
                             candidates: Sequence[float] =
                             DEFAULT_THRESHOLD_CANDIDATES,
                             normalise: bool = True,
                             sensitivity: Optional[float] = None,
                             specificity: Optional[float] = None,
                             minimum_wells: int = MINIMUM_CALIBRATION_WELLS,
                             training_columns: Sequence[int] = (1, 2),
                             well_column: str = "prc",
                             guide_column: str = "grna",
                             count_column: str = "count") -> Dict[str, Any]:
    """Choose ``fraction_threshold`` by fitting imaging on sequencing at each.

    :param counts: the control wells' read counts, one row per gRNA per well.
    :param features: ``(n_cells, n_features)`` over those same wells.
    :param wells: one well label per cell.
    :param positive_guide: the gRNA whose share is being calibrated.
    :param pure_pc_wells: wells designated as entirely positive control in the
        plate design. They are not inferred from the read fraction being
        calibrated.
    :param pure_nc_wells: wells that are entirely negative control, likewise.
    :param candidates: the thresholds to try.
    :param normalise: sweep with ``normalise_fraction`` on or off.
    :param sensitivity: classifier sensitivity used for Rogan--Gladen
        rescaling of the fitted slope. No cross-experiment default is assumed.
    :param specificity: classifier specificity used with ``sensitivity``;
        aggregate accuracy is not an equivalent substitute.
    :param minimum_wells: refuse to report below this many usable wells.
    :param training_columns: plate columns the classifier was trained on.
    :returns: the chosen threshold, the reason, and every candidate's fit.

    If no candidate meets the evidence requirement, ``chosen`` is ``None``
    and ``reason`` describes the limiting condition.
    """
    from .annotation_validation import mixed_ratio_calibration

    correction: Dict[str, float] = {}
    if (sensitivity is None) != (specificity is None):
        raise ValueError(
            "Rogan-Gladen correction requires both sensitivity and "
            "specificity")
    if sensitivity is not None:
        denominator = float(sensitivity) + float(specificity) - 1.0
        if denominator <= 0:
            raise ValueError(
                f"sensitivity {sensitivity} and specificity {specificity} sum "
                f"to no more than one, so the correction would invert the "
                f"estimate: the classifier is no better than chance")
        correction = {"denominator": denominator,
                      "sensitivity": float(sensitivity),
                      "specificity": float(specificity),
                      "variance_inflation": 1.0 / denominator ** 2}

    labels = [str(w) for w in wells]
    rows: List[Dict[str, Any]] = []
    for threshold in candidates:
        fractions = well_fractions(
            counts, threshold=float(threshold), normalise=normalise,
            well_column=well_column, guide_column=guide_column,
            count_column=count_column)
        reported = reported_control_share(
            fractions, positive_guide, well_column=well_column,
            guide_column=guide_column)
        guides = (fractions.groupby(fractions[well_column].astype(str))
                  [guide_column].nunique().mean()) if len(fractions) else 0.0
        fit = mixed_ratio_calibration(
            features, labels, reported,
            pure_pc_wells=list(pure_pc_wells),
            pure_nc_wells=list(pure_nc_wells))
        row: Dict[str, Any] = {
            "threshold": float(threshold),
            "guides_per_well": float(guides),
            "wells": int(fit.get("wells", 0) or 0),
        }
        if "error" in fit:
            row["error"] = str(fit["error"])
        else:
            row.update({
                "slope": float(fit["slope"]),
                "intercept": float(fit["intercept"]),
                "median_absolute_residual":
                    float(fit["median_absolute_residual"]),
                "median_absolute_disagreement": _disagreement(
                    fit.get("per_well") or {}, correction),
                "reading": str(fit["reading"]),
            })
            if correction:
                # AN AFFINE MAP OF THE IMAGING SIDE IS AN AFFINE MAP OF THE
                # LINE. p_true = (p_obs - (1 - sp)) / (se + sp - 1), so the
                # slope divides by the same denominator and nothing has to be
                # refitted. The correction MOVES the estimate; it does not
                # widen it, and it inflates the variance by the square.
                row["corrected_slope"] = (row["slope"]
                                          / correction["denominator"])
                row["variance_inflation"] = correction["variance_inflation"]
        rows.append(row)

    # A CANDIDATE IS CHOSEN ON ITS DISAGREEMENT, so one whose disagreement
    # could not be measured cannot be chosen -- and must not reach the
    # comparison below either. NaN loses every ``<=`` it appears in, so a
    # sweep whose fits reported no per-well pairs left ``min()`` an empty
    # iterable and raised ValueError out of a function whose whole contract
    # is to answer "no" in words.
    wide_enough = [row for row in rows
                   if "error" not in row
                   and row["wells"] >= int(minimum_wells)]
    usable = [row for row in wide_enough
              if np.isfinite(row["median_absolute_disagreement"])]
    result: Dict[str, Any] = {
        "candidates": rows,
        "normalised": bool(normalise),
        "minimum_wells": int(minimum_wells),
        "positive_guide": str(positive_guide),
        "training_wells_in_fit": _training_well_count(
            sorted({str(w) for w in labels}), training_columns),
        "corrected": bool(correction),
    }
    if not usable:
        result["chosen"] = None
        if wide_enough:
            result["reason"] = (
                f"{len(wide_enough)} candidate threshold(s) fitted enough "
                f"wells, but none of them produced a measurable disagreement "
                f"between imaging and sequencing, so there is nothing to "
                f"choose between")
        else:
            best = max((row["wells"] for row in rows), default=0)
            result["reason"] = (
                f"no candidate threshold fitted at least "
                f"{int(minimum_wells)} control wells (the best managed "
                f"{best}), so there is nothing to choose between")
        return result

    # THE SMALLEST THRESHOLD THAT MAKES THE TWO MEASUREMENTS AGREE. Once the
    # spurious barcodes are gone a higher threshold changes nothing except how
    # much real data it discards, so ties go to the least destructive answer.
    best = min(row["median_absolute_disagreement"] for row in usable)
    chosen = min(row["threshold"] for row in usable
                 if row["median_absolute_disagreement"] <= best + 1e-12)
    picked = next(row for row in usable if row["threshold"] == chosen)
    result["chosen"] = float(chosen)
    result["reason"] = (
        f"fraction_threshold={chosen:g} left the control wells most "
        f"consistent: imaging and sequencing agree to within "
        f"{picked['median_absolute_disagreement']:.3f} across "
        f"{picked['wells']} wells, at slope {picked['slope']:.3f} and "
        f"{picked['guides_per_well']:.1f} guides per well")
    result["fit"] = picked
    return result


def compare_normalisations(counts: pd.DataFrame,
                           features: np.ndarray,
                           wells: Sequence[str], *,
                           threshold: float = 0.02,
                           **kwargs: Any) -> Dict[str, Any]:
    """Compare calibration fits using raw and normalised guide fractions.

    Both fits use the same control wells and threshold. The comparison
    therefore isolates the effect of guide-fraction normalisation. Individual
    slopes combine penetrance and fraction bias and should not be interpreted
    as absolute calibrations; their ratio cancels the shared penetrance term.

    :param counts: Control-well read counts, with one row per gRNA and well.
    :param features: Feature matrix of shape ``(n_cells, n_features)`` for the
        same wells.
    :param wells: One well identifier per cell.
    :param threshold: Feature threshold used in both fits.
    :param kwargs: Additional arguments for :func:`sweep_fraction_threshold`.
        ``normalise`` and ``candidates`` are controlled by this function.
    :returns: The raw and normalised fits, their slope ratio, and the
        normalisation with the smaller median absolute disagreement.
    """
    if "normalise" in kwargs or "candidates" in kwargs:
        raise ValueError(
            "compare_normalisations sets normalise and candidates itself: "
            "the point is the SAME wells at the SAME threshold under both "
            "fraction definitions")
    fits: Dict[str, Any] = {}
    for name, normalise in (("raw", False), ("normalised", True)):
        result = sweep_fraction_threshold(
            counts, features, wells, candidates=(float(threshold),),
            normalise=normalise, **kwargs)
        rows = result.get("candidates") or [{}]
        fits[name] = dict(rows[0])
        if result.get("chosen") is None and "error" not in fits[name]:
            fits[name]["refused"] = str(result.get("reason", ""))

    out: Dict[str, Any] = {"threshold": float(threshold),
                           "raw": fits["raw"],
                           "normalised": fits["normalised"]}
    raw_slope = fits["raw"].get("slope")
    scaled_slope = fits["normalised"].get("slope")
    if raw_slope is None or scaled_slope is None or not scaled_slope:
        out["ratio"] = None
        out["more_consistent"] = None
        out["reading"] = (
            "one of the two fits produced no slope, so there is no ratio: "
            + str(fits["raw"].get("error")
                  or fits["normalised"].get("error")
                  or fits["raw"].get("refused")
                  or fits["normalised"].get("refused") or "no reason given"))
        return out

    out["ratio"] = float(raw_slope) / float(scaled_slope)
    disagreements = {name: fits[name].get("median_absolute_disagreement")
                     for name in ("raw", "normalised")}
    # FINITE, not merely present. A fit that reported no per-well pairs
    # carries a NaN disagreement, and NaN loses every comparison ``min`` makes
    # -- naming whichever definition happened to be first, as though the two
    # had been measured and one had won.
    if all(value is not None and np.isfinite(value)
           for value in disagreements.values()):
        out["more_consistent"] = min(disagreements,
                                     key=lambda k: disagreements[k])
    else:
        out["more_consistent"] = None
    out["reading"] = (
        f"raw slope {float(raw_slope):.3f}, normalised slope "
        f"{float(scaled_slope):.3f}, ratio {out['ratio']:.3f} -- penetrance "
        f"cancels in the ratio and in neither slope alone"
        + (f"; {out['more_consistent']} fractions leave imaging and "
           f"sequencing closer together"
           if out["more_consistent"] else ""))
    return out


def describe(result: Mapping[str, Any]) -> str:
    """The sweep in one line, for a log or a run summary.

    :param result: calibration result returned by
        :func:`sweep_fraction_threshold`.
    """
    if result.get("chosen") is None:
        return f"fraction_threshold not measured: {result.get('reason', '')}"
    note = ""
    if result.get("training_wells_in_fit"):
        note = (f"; {result['training_wells_in_fit']} of the fitted wells "
                f"also trained the classifier")
    return f"{result.get('reason', '')}{note}"
