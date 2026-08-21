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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    """A confusion matrix, and the two numbers that actually get used."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    threshold: float

    @property
    def sensitivity(self) -> float:
        """Of the cells that ARE positive, the share called positive."""
        real = self.true_positive + self.false_negative
        return (self.true_positive / real) if real else float("nan")

    @property
    def specificity(self) -> float:
        """Of the cells that are NOT, the share called negative."""
        real = self.true_negative + self.false_positive
        return (self.true_negative / real) if real else float("nan")

    @property
    def accuracy(self) -> float:
        """The number that should not be used, provided so it can be
        compared against the two that should.

        ON AN IMBALANCED WELL IT IS THE MAJORITY CLASS WEARING A HAT. A
        classifier that calls everything negative on a well that is 10%
        positive is 90% accurate and has a sensitivity of zero.
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
    """The confusion matrix across the score range -- the ROC, as matrices.

    THE THRESHOLD IS A CHOICE AND HAS TO BE SEEN AS ONE. A single confusion
    matrix at 0.5 answers "how good is it there" and hides that moving the
    threshold trades exactly the two quantities the correction divides by.
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
    """The operating point to run the annotation at.

    :param criterion: ``'youden'`` maximises ``se + sp - 1``, which is the
        exact denominator of the Rogan-Gladen correction -- so it is not an
        arbitrary choice here, it is the point at which the correction is
        most stable and its variance inflation smallest.
    """
    points = operating_points(scores, labels)
    if not points:
        return confusion([], [], 0.5)
    if str(criterion) != "youden":
        raise ValueError(f"unknown criterion {criterion!r}")
    def value(point: Confusion) -> float:
        total = point.sensitivity + point.specificity - 1.0
        return total if np.isfinite(total) else -np.inf
    return max(points, key=value)


def sensitivity_by_prevalence(scores: Sequence[float],
                              labels: Sequence[bool],
                              wells: Sequence[str], *,
                              threshold: float = 0.5,
                              bins: int = 4) -> List[Dict[str, float]]:
    """Sensitivity within bands of well prevalence.

    THE NUMBER 214 ASKED FOR, and the reason it asked. Sensitivity is a
    property of the classifier and specificity is too -- neither should
    depend on prevalence at all. If they DO here, the classifier is using
    context rather than the cell: a model that has learned "this well looks
    crowded, so call more of it positive" will show rising sensitivity with
    prevalence, and every fraction it produces is then partly a copy of the
    fraction it was given.

    :returns: one row per band with the band's prevalence, its sensitivity,
        its specificity and its size.
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
    """Correct an observed positive share for classifier error.

        p_true = (p_observed - (1 - sp)) / (se + sp - 1)

    IT MOVES THE ESTIMATE; IT DOES NOT WIDEN IT. At ``se = sp = 0.94`` an
    observed 0.06 corrects to exactly zero -- the whole observed signal being
    false positives -- and an observed 0.244 corrects to 0.20.

    :param n: the number of cells, if the standard error is wanted. The
        correction inflates variance by ``1 / (se + sp - 1)^2``, which at
        0.94/0.94 is 1.29, so an interval computed before correcting is too
        narrow as well as centred in the wrong place.
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
    """Estimate the two class distributions from unlabelled scores.

    THE FALLBACK, AND IT IS A FALLBACK. With no labels this fits the score
    column as two Gaussian components and reports what they imply. It is
    the only route when nothing is labelled, and it is the weakest, because
    it assumes the shape of the two components AND that they are far enough
    apart to be told apart at all.

    ``separation`` is the health check and must be read before the numbers
    beside it: the gap between the component means in pooled standard
    deviations. Below about 2 the fit is choosing between many equally good
    answers, and the sensitivity it reports is an artefact of the starting
    point rather than a fact about the classifier.
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
    """Find the written test splits under a screen's folder.

    :param root: the screen directory holding one folder per plate.
    :param pattern: how the training code named them. The default matches
        both shapes spaCR writes -- `*_test_acc.csv` and
        `*_test_result.csv`.
    :returns: ``{plate folder name: path}``, one per plate, newest first
        where a plate has several.

    THIS IS THE ROUTE, and there is no other. Nothing in spaCR carries a
    measured sensitivity for a screen it has not been shown, so a user with
    their own screen gets their own numbers by pointing this at their own
    folder -- which is the only arrangement in which the correction can be
    right for more than one experiment.
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
    """Sensitivity and specificity per plate, from a user's own screen.

    The whole of what this module offers, in one call, for somebody who has
    run their own screen and wants the correction to be about it.

    PER PLATE AND NOT POOLED, deliberately. Plates are trained separately
    and their thresholds are not comparable -- on the screen this was
    developed against they spanned 0.2955 to 0.8567, which is three points
    of sensitivity. Averaging them would produce one number that is wrong
    for every plate.
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
    :returns: sensitivity, specificity, the threshold, and ``per_cell`` to
        say which shape was read -- because only a per-cell file supports
        asking the same question again at a different threshold.
    """
    import pandas as pd

    frame = pd.read_csv(path)
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
    """What the classifier does to a fraction, as the fraction gets rarer.

    THE RESULT THAT DECIDES WHETHER ANY OF THIS MATTERS, and it is not
    intuitive from the accuracy. At the tsg101 classifiers' measured
    ``se = 0.960``, ``sp = 0.981``:

        a true 30% is observed as 30% -- the correction is a no-op;
        a true  1% is observed as  2.8% -- nearly THREE TIMES too high.

    The false positives are a share of the NEGATIVES, and when a guide is
    rare almost every cell in the well is a negative, so a small
    false-positive rate on a large population swamps a large true-positive
    rate on a small one.

    SCREEN HITS ARE RARE. So the correction is negligible everywhere it does
    not matter and dominant exactly where it does.
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
