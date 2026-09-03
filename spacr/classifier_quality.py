"""Measure classifier performance and correct observed class fractions.

A confusion matrix requires labelled outcomes; an unlabelled score column
cannot establish sensitivity or specificity by itself. Use a labelled test
split or out-of-fold predictions when available. When labels are unavailable,
:func:`deconvolve` estimates two score distributions and reports whether their
separation is sufficient to support the estimate.

Classifier errors bias the observed positive fraction rather than merely
widening its uncertainty. The correction helpers expose that bias across
prevalence levels and refuse the Rogan--Gladen correction when sensitivity
and specificity do not identify it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "Confusion",
    "confusion",
    "operating_points",
    "best_threshold",
    "sensitivity_by_prevalence",
    "rogan_gladen",
    "deconvolve",
    "training_wells",
    "discover_test_splits",
    "measure_screen",
]


@dataclass(frozen=True)
class Confusion:
    """Binary confusion counts and their operating characteristics.

    :param true_positive: number of positively labelled cells whose score
        meets the threshold.
    :param false_positive: number of negatively labelled cells whose score
        meets the threshold.
    :param true_negative: number of negatively labelled cells whose score
        falls below the threshold.
    :param false_negative: number of positively labelled cells whose score
        falls below the threshold.
    :param threshold: score cutoff used to classify cells as positive.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    threshold: float

    @property
    def sensitivity(self) -> float:
        """Return the true-positive rate among positive cells."""
        real = self.true_positive + self.false_negative
        return (self.true_positive / real) if real else float("nan")

    @property
    def specificity(self) -> float:
        """Return the true-negative rate among negative cells."""
        real = self.true_negative + self.false_positive
        return (self.true_negative / real) if real else float("nan")

    @property
    def accuracy(self) -> float:
        """Return the correctly classified share of all cells.

        Accuracy should be interpreted with sensitivity and specificity when
        class prevalence is imbalanced.
        """
        total = (self.true_positive + self.false_positive
                 + self.true_negative + self.false_negative)
        return ((self.true_positive + self.true_negative) / total
                if total else float("nan"))

    @property
    def prevalence(self) -> float:
        total = (self.true_positive + self.false_positive
                 + self.true_negative + self.false_negative)
        return ((self.true_positive + self.false_negative) / total
                if total else float("nan"))

    @property
    def usable(self) -> bool:
        """Whether a Rogan-Gladen correction can be made from this.

        ``se + sp <= 1`` means the classifier carries no information at the
        chosen threshold, and the correction divides by ``se + sp - 1``.
        """
        total = self.sensitivity + self.specificity
        return bool(np.isfinite(total) and total > 1.0)

    def summary(self) -> str:
        return (f"se {self.sensitivity:.3f}  sp {self.specificity:.3f}  "
                f"(accuracy {self.accuracy:.3f} at prevalence "
                f"{self.prevalence:.3f})")


def confusion(scores: Sequence[float],
              labels: Sequence[bool],
              threshold: float = 0.5) -> Confusion:
    """The confusion matrix from a labelled split.

    :param scores: the classifier's score per cell.
    :param labels: True where the cell really is the positive class.
    :param threshold: score at or above which the call is positive.
    """
    values = np.asarray(list(scores), dtype=float)
    truth = np.asarray(list(labels), dtype=bool)
    keep = np.isfinite(values)
    values, truth = values[keep], truth[keep]
    called = values >= float(threshold)
    return Confusion(
        true_positive=int(np.sum(called & truth)),
        false_positive=int(np.sum(called & ~truth)),
        true_negative=int(np.sum(~called & ~truth)),
        false_negative=int(np.sum(~called & truth)),
        threshold=float(threshold),
    )


def operating_points(scores: Sequence[float],
                     labels: Sequence[bool], *,
                     steps: int = 50) -> List[Confusion]:
    """Return confusion matrices across quantile-based score thresholds.

    :param scores: classifier score for each labelled cell.
    :param labels: true for cells belonging to the positive class, aligned
        one-to-one with ``scores``.

    The sequence exposes the sensitivity-specificity trade-off rather than
    evaluating only the conventional threshold of 0.5.
    """
    values = np.asarray(list(scores), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    cuts = np.quantile(values, np.linspace(0.0, 1.0, int(steps)))
    seen: List[Confusion] = []
    for cut in np.unique(cuts):
        seen.append(confusion(scores, labels, float(cut)))
    return seen


def best_threshold(scores: Sequence[float],
                   labels: Sequence[bool], *,
                   criterion: str = "youden") -> Confusion:
    """Select an operating point for annotation.

    :param scores: classifier score for each labelled cell.
    :param labels: true for cells belonging to the positive class, aligned
        one-to-one with ``scores``.
    :param criterion: ``'youden'`` maximises ``se + sp - 1``, which is the
        denominator of the Rogan--Gladen correction and therefore favours
        stable prevalence correction.
    """
    points = operating_points(scores, labels)
    if not points:
        return confusion([], [], 0.5)
    if str(criterion) != "youden":
        raise ValueError(f"unknown criterion {criterion!r}")
    def value(point: Confusion) -> float:
        """Return Youden's J, ranking a non-finite result last."""
        total = point.sensitivity + point.specificity - 1.0
        return total if np.isfinite(total) else -np.inf
    return max(points, key=value)


def sensitivity_by_prevalence(scores: Sequence[float],
                              labels: Sequence[bool],
                              wells: Sequence[str], *,
                              threshold: float = 0.5,
                              bins: int = 4) -> List[Dict[str, float]]:
    """Measure classifier performance across well-prevalence bands.

    :param scores: classifier score for each labelled cell.
    :param labels: true for cells belonging to the positive class.
    :param wells: well label for each score and truth value.

    Returns one row per populated band with prevalence, sensitivity,
    specificity, accuracy, and cell count. Dependence on prevalence can reveal
    that a classifier is using well context rather than only cell phenotype.
    """
    values = np.asarray(list(scores), dtype=float)
    truth = np.asarray(list(labels), dtype=bool)
    labels_ = np.asarray([str(w) for w in wells])
    share: Dict[str, float] = {}
    for well in set(labels_.tolist()):
        here = truth[labels_ == well]
        share[well] = float(here.mean()) if here.size else float("nan")

    per_cell = np.asarray([share.get(w, float("nan")) for w in labels_])
    usable = np.isfinite(per_cell) & np.isfinite(values)
    if not np.any(usable):
        return []
    edges = np.quantile(per_cell[usable], np.linspace(0, 1, int(bins) + 1))
    edges = np.unique(edges)
    out: List[Dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        pick = usable & (per_cell >= low) & (
            (per_cell < high) | (high == edges[-1]))
        if not np.any(pick):
            continue
        band = confusion(values[pick], truth[pick], threshold)
        out.append({
            "prevalence_low": float(low),
            "prevalence_high": float(high),
            "prevalence": float(band.prevalence),
            "sensitivity": float(band.sensitivity),
            "specificity": float(band.specificity),
            "accuracy": float(band.accuracy),
            "n": int(pick.sum()),
        })
    return out


def rogan_gladen(observed: float, sensitivity: float, specificity: float, *,
                 n: Optional[int] = None) -> Dict[str, float]:
    """Apply the Rogan--Gladen correction to an observed positive share.

        p_true = (p_observed - (1 - sp)) / (se + sp - 1)

    :param observed: observed share called positive, conventionally between
        zero and one.
    :param sensitivity: true-positive rate at the chosen classifier threshold.
    :param specificity: true-negative rate at the chosen classifier threshold.
    :param n: the number of cells, if the standard error is wanted. The
        correction inflates variance by ``1 / (se + sp - 1)^2``.

    The result includes the unclipped denominator, a clipping indicator, and
    variance inflation. Correction is unusable when ``se + sp`` is one.
    """
    se, sp = float(sensitivity), float(specificity)
    denominator = se + sp - 1.0
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return {"corrected": float("nan"), "denominator": denominator,
                "usable": 0.0}
    raw = (float(observed) - (1.0 - sp)) / denominator
    out = {
        "observed": float(observed),
        "corrected": float(np.clip(raw, 0.0, 1.0)),
        "clipped": float(raw < 0.0 or raw > 1.0),
        "denominator": float(denominator),
        "variance_inflation": float(1.0 / (denominator ** 2)),
        "usable": 1.0,
    }
    if n:
        p = float(np.clip(observed, 0.0, 1.0))
        out["standard_error"] = float(
            np.sqrt(max(p * (1.0 - p), 0.0) / int(n)) / abs(denominator))
    return out


def deconvolve(scores: Sequence[float], *,
               seed: int = 0) -> Dict[str, float]:
    """Estimate two class distributions from unlabelled scores.

    :param scores: unlabelled classifier scores to model as a two-component
        mixture; non-finite values are ignored.

    A two-component Gaussian mixture estimates prevalence, sensitivity,
    specificity, and a midpoint threshold. ``separation`` is the distance
    between component means in pooled standard deviations; the result marks
    estimates trustworthy only when separation is at least two. This is a
    model-based fallback and is weaker evidence than a labelled test split.
    """
    from sklearn.mixture import GaussianMixture

    values = np.asarray(list(scores), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return {"error": 1.0, "reason": float("nan"), "n": float(values.size)}
    model = GaussianMixture(n_components=2, random_state=int(seed))
    model.fit(values.reshape(-1, 1))
    means = model.means_.reshape(-1)
    spreads = np.sqrt(model.covariances_.reshape(-1))
    weights = model.weights_.reshape(-1)
    order = np.argsort(means)
    low, high = int(order[0]), int(order[1])
    pooled = float(np.sqrt((spreads[low] ** 2 + spreads[high] ** 2) / 2.0))
    separation = (abs(means[high] - means[low]) / pooled) if pooled > 0 else 0.0
    cut = float((means[low] + means[high]) / 2.0)
    from math import erf, sqrt
    def above(mean, spread):
        """Estimate the Gaussian probability above the fitted midpoint.

        A zero-width component is treated as a definite side of the cut
        rather than passed to a division by zero.
        """
        if spread <= 0:
            return 1.0 if mean >= cut else 0.0
        return 0.5 * (1.0 - erf((cut - mean) / (spread * sqrt(2.0))))
    return {
        "prevalence": float(weights[high]),
        "sensitivity": float(above(means[high], spreads[high])),
        "specificity": float(1.0 - above(means[low], spreads[low])),
        "threshold": cut,
        "separation": float(separation),
        "trustworthy": float(separation >= 2.0),
        "n": float(values.size),
    }


def training_wells(wells: Sequence[str], *,
                   columns: Sequence[int] = (1, 2)) -> np.ndarray:
    """Return the cells belonging to classifier-training columns.

    Training wells must be excluded from performance calibration; otherwise
    the calibration measures in-sample fit. Both ``r1_c2`` and ``c2`` well
    labels are accepted. Unparseable labels are retained for validation
    rather than being silently classified as training data.

    Parameters
    ----------
    wells : sequence of str
        One well label per cell.
    columns : sequence of int, default=(1, 2)
        One-based column numbers used for classifier training.

    Returns
    -------
    numpy.ndarray
        Boolean mask with one element per input well.
    """
    import re

    wanted = {int(c) for c in columns}
    out = np.zeros(len(list(wells)), dtype=bool)
    for index, name in enumerate(wells):
        found = re.search(r"c(?:ol(?:umn)?)?[_]?(\d+)", str(name),
                          flags=re.IGNORECASE)
        if found and int(found.group(1)) in wanted:
            out[index] = True
    return out


# ---------------------------------------------------------------------------
# Reading a real test split
# ---------------------------------------------------------------------------
#
# The maintainer supplied the tsg101 screen's test splits on 2026-08-21, in
# TWO shapes, and both are read here because both are what the training code
# writes:
#
#   * PER-CELL (`*_test_acc.csv`): one row per test crop with `true_label`,
#     `predicted_label` and `class_1_probability`. Everything can be computed
#     from this -- any threshold, the whole ROC.
#   * SUMMARY (`*_test_result.csv`): one row with `accuracy`, `neg_accuracy`,
#     `pos_accuracy`, `prauc` and `optimal_threshold`.
#
# `pos_accuracy` IS THE SENSITIVITY and `neg_accuracy` IS THE SPECIFICITY.
# That is the number instruction 214 asked for and it was already being
# written to disk on every plate -- which is worth saying plainly, because
# the request was answered by a file that already existed.

# NO SCREEN'S NUMBERS LIVE IN THIS FILE. A table of the tsg101 plates'
# measured sensitivities was here for one commit and was removed on request:
#
#     "wahtever information you use from my screen any calculated
#      coefficients need to be recalculable for users whou do their own
#      screens"
#
# WHICH IS RIGHT, AND NOT ONLY ON PRINCIPLE. A constant in a library becomes
# a default the moment somebody is in a hurry, and a sensitivity measured on
# one model, one stain and one microscope is wrong for every other screen in
# a way that produces plausible numbers rather than an error. Every function
# here takes `sensitivity` and `specificity` as REQUIRED arguments for that
# reason -- there is nothing to fall back to.
#
# `discover_test_splits` finds a user's own files; `from_test_split` reads
# them. The tsg101 figures are recorded in `instructions/done/`, which is a
# log and cannot be imported.


def discover_test_splits(root: str, *,
                        pattern: str = "*test_*.csv") -> Dict[str, str]:
    """Find classifier test outputs under a screen directory.

    :param root: the screen directory holding one folder per plate.
    :param pattern: how the training code named them. The default matches
        both shapes spaCR writes -- `*_test_acc.csv` and
        `*_test_result.csv`.
    :returns: ``{plate folder name: path}``, with the newest matching file
        selected when a plate contains several.
    """
    from pathlib import Path

    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"{root}: not a directory")
    out: Dict[str, str] = {}
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        found = sorted(folder.glob(pattern),
                       key=lambda f: f.stat().st_mtime, reverse=True)
        if found:
            out[folder.name] = str(found[0])
    return out


def measure_screen(root: str, *,
                   pattern: str = "*test_*.csv",
                   threshold: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    """Measure sensitivity and specificity for each plate in a screen.

    :param root: screen directory containing one result folder per plate.

    Plates are evaluated separately because their classifiers and selected
    thresholds can differ. The return value maps plate-folder names to the
    metrics produced by :func:`from_test_split`.
    """
    return {name: from_test_split(path, threshold=threshold)
            for name, path in discover_test_splits(root,
                                                   pattern=pattern).items()}


def from_test_split(path: str, *,
                    threshold: Optional[float] = None) -> Dict[str, float]:
    """Sensitivity and specificity from a written test split.

    :param path: a per-cell ``*_test_acc.csv`` or a summary
        ``*_test_result.csv``.
    :param threshold: for a per-cell file, the score to call positive at.
        ``None`` takes the Youden point. Ignored for a summary file, which
        has already chosen one.
    :returns: sensitivity, specificity, threshold, accuracy, sample count,
        and ``per_cell`` indicating which file shape was read. Only per-cell
        files support recalculation at a different threshold.
    """
    from .tabular import read_table

    frame = read_table(path)
    columns = {str(c).lower() for c in frame.columns}

    if {"pos_accuracy", "neg_accuracy"} <= columns:
        row = frame.iloc[0]
        return {
            "sensitivity": float(row["pos_accuracy"]),
            "specificity": float(row["neg_accuracy"]),
            "threshold": float(row.get("optimal_threshold", float("nan"))),
            "accuracy": float(row.get("accuracy", float("nan"))),
            "per_cell": 0.0,
            "n": float(len(frame)),
        }

    if {"true_label", "class_1_probability"} <= columns:
        scores = frame["class_1_probability"].to_numpy(dtype=float)
        truth = frame["true_label"].to_numpy(dtype=float) == 1
        point = (confusion(scores, truth, float(threshold))
                 if threshold is not None else best_threshold(scores, truth))
        return {
            "sensitivity": float(point.sensitivity),
            "specificity": float(point.specificity),
            "threshold": float(point.threshold),
            "accuracy": float(point.accuracy),
            "prevalence": float(point.prevalence),
            "per_cell": 1.0,
            "n": float(len(frame)),
        }

    raise ValueError(
        f"{path}: not a test split this can read. Expected either "
        f"pos_accuracy/neg_accuracy or true_label/class_1_probability, "
        f"found {sorted(columns)}")


def inflation_by_prevalence(sensitivity: float, specificity: float, *,
                            prevalences: Sequence[float] = (
                                0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01),
                            ) -> List[Dict[str, float]]:
    """Quantify classifier-induced fraction inflation across prevalences.

    For each true prevalence, the function reports the expected observed
    prevalence, the Rogan--Gladen-corrected value, and their ratio. False
    positives can dominate rare classes because they are applied to the much
    larger negative population.
    """
    se, sp = float(sensitivity), float(specificity)
    out: List[Dict[str, float]] = []
    for true in prevalences:
        observed = se * float(true) + (1.0 - sp) * (1.0 - float(true))
        back = rogan_gladen(observed, se, sp)
        out.append({
            "true": float(true),
            "observed": float(observed),
            "corrected": float(back.get("corrected", float("nan"))),
            "inflation": float(observed / true) if true else float("nan"),
        })
    return out
