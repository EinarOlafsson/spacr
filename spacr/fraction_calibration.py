"""Measure ``fraction_threshold`` from the control wells instead of assuming it.

``fraction_threshold`` drops a gRNA from a well when its share of that well's
reads is too small to believe. The default is 0.02 -- a constant, chosen once,
applied to every screen. It decides how many guides each well is credited
with, and therefore what every coefficient downstream is regressed on.

A CONTROL WELL KNOWS THE ANSWER TWICE. Sequencing says the positive control
is a fraction ``f`` of the well's reads; imaging says a proportion ``p`` of
the well's cells look like the positive control. When the control is the only
thing in that well producing the phenotype the two must agree, so the SPREAD
of ``p`` around its line on ``f`` measures how well the fractions are behaving
-- and the threshold is what the fractions are computed under.

So sweep it. Recompute the fractions at each candidate threshold, refit
imaging on sequencing, and take the threshold whose fit is most consistent.
The number stops being a constant in a library and becomes a measurement of
this screen.

WHAT "MOST CONSISTENT" MEANS HERE: the median absolute DISAGREEMENT between
the two measurements, well by well -- ``median |imaging - sequencing|``.

Not the slope, and not the scatter around the fitted line. A slope is
penetrance times fraction bias when the imaging side is a classifier's call
rate, and the mixture estimator this uses absorbs penetrance -- a positive
control cell showing no phenotype is still in the pure-PC reference -- so what
is left to explain a disagreement IS the fraction. Scatter around the fitted
line cannot see a systematic deflation at all: fractions that are all 3 % too
low sit exactly on a line of slope 1.03. The slope, the intercept and the
residual are reported at every candidate so all three can be read; the choice
is made on how far apart the two answers actually are.

THE FIT IS A MEDIAN FIT, and that is not a stylistic preference. A screen hit
sharing a control well ADDS phenotype-positive cells and never removes them,
so the contamination is one-sided -- exactly the case least squares handles
worst. :func:`spacr.annotation_validation.mixed_ratio_calibration` fits the
median of the pairwise slopes.

WHAT THIS DOES NOT MEASURE. A control column has two guides competing for its
reads; a screen well has hundreds. The threshold chosen here is the one that
makes the CONTROL wells internally consistent, and the number of guides per
well is reported at every candidate so the difference is visible rather than
assumed away.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_THRESHOLD_CANDIDATES",
    "MINIMUM_CALIBRATION_WELLS",
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
    :param pure_pc_wells: wells that are entirely positive control, NAMED FROM
        THE PLATE DESIGN. Identifying them by their reported fraction would be
        circular: that fraction is the quantity under test, and a bias large
        enough to matter pushes a pure well below any cut-off.
    :param pure_nc_wells: wells that are entirely negative control, likewise.
    :param candidates: the thresholds to try.
    :param normalise: sweep with ``normalise_fraction`` on or off.
    :param sensitivity: classifier sensitivity, for the Rogan--Gladen
        rescaling of the fitted slope. There is no default and there will not
        be one: a sensitivity measured on one model, one stain and one
        microscope is wrong for every other screen in a way that produces
        plausible numbers rather than an error.
    :param specificity: classifier specificity. AN ACCURACY IS NOT ENOUGH --
        these are two quantities, and on a well where the control is a
        minority the accuracy is dominated by the majority class.
    :param minimum_wells: refuse to report below this many usable wells.
    :param training_columns: plate columns the classifier was trained on.
    :returns: the chosen threshold, the reason, and every candidate's fit.

    A screen with no threshold that beats the others is told so: ``chosen``
    is ``None`` and ``reason`` says why, rather than a number being returned
    that the data did not support.
    """
    from .annotation_validation import mixed_ratio_calibration

    correction: Dict[str, float] = {}
    if (sensitivity is None) != (specificity is None):
        raise ValueError(
            "the Rogan-Gladen correction needs BOTH sensitivity and "
            "specificity; one of them alone is an accuracy wearing a "
            "confusion matrix's name")
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

    usable = [row for row in rows
              if "error" not in row and row["wells"] >= int(minimum_wells)]
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
        best = max((row["wells"] for row in rows), default=0)
        result["chosen"] = None
        result["reason"] = (
            f"no candidate threshold fitted at least {int(minimum_wells)} "
            f"control wells (the best managed {best}), so there is nothing "
            f"to choose between")
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


def describe(result: Mapping[str, Any]) -> str:
    """The sweep in one line, for a log or a run summary."""
    if result.get("chosen") is None:
        return f"fraction_threshold not measured: {result.get('reason', '')}"
    note = ""
    if result.get("training_wells_in_fit"):
        note = (f"; {result['training_wells_in_fit']} of the fitted wells "
                f"also trained the classifier")
    return f"{result.get('reason', '')}{note}"
