"""How good is the classifier, and what does that do to a fraction?

Instruction 214 asked for the classifier's confusion matrix and got a
question back: "you have the computer vision scores, cant this be calculated
from them? cant you bake the calculation of this into the modula?"

THE SHORT ANSWER IS NO FROM SCORES ALONE, AND YES FROM THE TABLES YOU HAVE.
A confusion matrix counts agreements between a call and a truth. Scores with
no labels beside them have no truth to agree with, so nothing in a score
column can produce one -- what looks like a decision boundary in a histogram
is a property of the histogram, not of any cell's identity.

What CAN be done, in descending order of how much it should be trusted:

  1. LABELLED TEST SPLIT -- exact. The annotations that were held out give
     the confusion matrix directly, at any threshold, and stratified by
     anything. This is what :func:`confusion` and :func:`operating_points`
     do, and it is the right answer whenever those rows exist.
  2. OUT-OF-FOLD PREDICTIONS -- exact and better. Cross-validating gives
     every cell an honest score rather than only the test cells, so the
     stratification in :func:`sensitivity_by_prevalence` has the whole
     screen to work with instead of a corner of it.
  3. MIXTURE DECONVOLUTION -- an estimate, for when there are no labels at
     all. :func:`deconvolve` fits the score distribution as two components
     and reports what they imply. It carries a health check because the
     estimate is worthless when the two components overlap, and worthless
     in a way that looks like a number.

WHY IT MATTERS ENOUGH TO HAVE ITS OWN MODULE. A classifier that is right 94%
of the time still turns a true 20% into an observed 24.4%, and the
correction for that is a rescaling rather than an error bar -- so leaving it
out does not widen a confidence interval, it moves the answer.
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
    """Mask of the cells that sat in the training columns.

    "columns one and 2 of each plate were the training wells" -- so the
    held-out set is everything else, and it can be identified from the well
    name alone without a record of what was trained on.

    THIS IS WHAT MAKES THE CALIBRATION NON-CIRCULAR. A classifier scored on
    the wells it was fitted to agrees with itself, and instruction 214's
    calibration would then measure the fit rather than the screen.

    Accepts the two spellings spaCR writes: ``r1_c2`` style and the plain
    ``c2``. A name it cannot parse is NOT training, since guessing would
    silently drop real validation wells.
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
