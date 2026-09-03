"""Score a segmentation against GROUND TRUTH, which nothing in spaCR did.

Instruction 370 asks every published model to carry a table comparing the
finetuned model with the vanilla one on a held-out set, "all the common
ones for that model type". Auditing what existed found the gap this module
fills:

* :mod:`spacr.model_compare` compares two models to EACH OTHER and says so
  in its own docstring -- "neither model is ground truth". It answers "do
  these two disagree", which is a different and also useful question.
* :mod:`spacr.seg_qc` scores a field with NO labels at all: it flags splits,
  merges and implausible diameters from the mask alone.

Both are useful and neither is Dice. Nothing measured a mask against a
hand-drawn truth, so no published number could say a model was BETTER --
only that it was different.

WHAT AN OBJECT METRIC HAS TO DECIDE FIRST
=========================================

Every number here rests on one choice: when is a predicted object THE SAME
object as a labelled one? The literature's answer is an IoU threshold, and
the threshold is not a detail -- a model that finds every cell and outlines
them loosely scores well at 0.5 and badly at 0.9, and a model that finds
half of them perfectly does the opposite. That is why :func:`average_precision`
reports the whole sweep from 0.5 to 0.9 rather than one number, and why
every precision/recall figure states the threshold it was matched at.

MATCHING IS ONE-TO-ONE AND OPTIMAL, not greedy. A greedy pass down the IoU
matrix is cheaper and gives a different answer when one prediction overlaps
two truths: it takes the first pair it sees rather than the assignment that
maximises total overlap, so the score depends on label ORDER, which is an
implementation detail of whoever wrote the mask. :func:`match_objects` uses
``scipy.optimize.linear_sum_assignment``, so relabelling either mask cannot
change the result.

WHAT IS DELIBERATELY NOT HERE
=============================

No torch, and no import of anything that pulls it. The model zoo imports
without torch on purpose and a test asserts it, so reading a scorecard must
not be the thing that drags in a GPU stack -- the reader is browsing a list.
Everything here is numpy plus two scipy/skimage helpers.

Nothing writes files, uploads, or reads a catalogue. This module turns two
label arrays into numbers; publishing them is 370's other half.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The IoU sweep every segmentation paper quotes, and the one Cellpose and
#: StarDist report against. 0.5 to 0.9 in steps of 0.05.
#:
#: THE SWEEP IS THE POINT. One number at 0.5 says whether the objects were
#: FOUND; one at 0.9 says whether they were OUTLINED. A model can be good at
#: either and bad at the other, and averaging them into a single figure is
#: how a table stops answering the question a reader has.
DEFAULT_IOU_THRESHOLDS: Tuple[float, ...] = tuple(
    round(0.5 + 0.05 * step, 2) for step in range(9)
)

#: Boundary tolerances in pixels, for :func:`boundary_f1`.
#:
#: Dice barely moves when an outline is a pixel loose -- a 30 px cell has
#: about 900 interior pixels and 95 boundary ones -- so a model that traces
#: ragged edges scores almost the same as one that traces clean ones. The
#: boundary measure is the one that separates them, and it needs a tolerance
#: because no two people draw the same edge pixel.
DEFAULT_BOUNDARY_TOLERANCES: Tuple[int, ...] = (1, 2, 3)


def _labels(mask: np.ndarray) -> np.ndarray:
    """The object ids in a label mask, background excluded."""
    values = np.unique(np.asarray(mask))
    return values[values != 0]


def _overlap_matrix(truth: np.ndarray, pred: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(intersection, truth ids, pred ids)`` counted in ONE pass.

    A per-pair loop is the obvious implementation and is quadratic in the
    object count: a confluent field of 3,000 cells against 3,000 predictions
    is nine million array comparisons. The pairs that actually overlap are a
    tiny fraction of that, and ``np.bincount`` over the flattened pair index
    finds exactly those in one pass over the pixels.
    """
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.shape != pred.shape:
        raise ValueError(
            f"truth {truth.shape} and prediction {pred.shape} are different "
            f"shapes; a score between them would be meaningless")
    t_ids, p_ids = _labels(truth), _labels(pred)
    if t_ids.size == 0 or p_ids.size == 0:
        return np.zeros((t_ids.size, p_ids.size), dtype=np.int64), t_ids, p_ids

    t_index = np.zeros(int(truth.max()) + 1, dtype=np.int64)
    t_index[t_ids] = np.arange(t_ids.size)
    p_index = np.zeros(int(pred.max()) + 1, dtype=np.int64)
    p_index[p_ids] = np.arange(p_ids.size)

    both = (truth > 0) & (pred > 0)
    if not both.any():
        return np.zeros((t_ids.size, p_ids.size), dtype=np.int64), t_ids, p_ids
    flat = (t_index[truth[both]] * p_ids.size) + p_index[pred[both]]
    counts = np.bincount(flat, minlength=t_ids.size * p_ids.size)
    return counts.reshape(t_ids.size, p_ids.size), t_ids, p_ids


def iou_matrix(truth: np.ndarray, pred: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(iou, truth ids, pred ids)``. IoU of every overlapping pair."""
    inter, t_ids, p_ids = _overlap_matrix(truth, pred)
    if inter.size == 0:
        return inter.astype(float), t_ids, p_ids
    t_area = np.array([(truth == i).sum() for i in t_ids], dtype=np.int64)
    p_area = np.array([(pred == i).sum() for i in p_ids], dtype=np.int64)
    union = t_area[:, None] + p_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out, t_ids, p_ids


@dataclass(frozen=True)
class Match:
    """One IoU threshold's worth of matching, and what it implies.

    :param threshold: the IoU at which a pair counted as the same object.
    :param pairs: ``(truth index, pred index)`` for each matched pair, into
        the id arrays :func:`iou_matrix` returns.
    :param ious: the IoU of each matched pair, in the same order.
    :param n_truth: labelled objects present.
    :param n_pred: predicted objects present.
    """

    threshold: float
    pairs: Tuple[Tuple[int, int], ...]
    ious: Tuple[float, ...]
    n_truth: int
    n_pred: int

    @property
    def true_positives(self) -> int:
        return len(self.pairs)

    @property
    def false_positives(self) -> int:
        return self.n_pred - self.true_positives

    @property
    def false_negatives(self) -> int:
        return self.n_truth - self.true_positives

    @property
    def precision(self) -> float:
        return _ratio(self.true_positives, self.n_pred)

    @property
    def recall(self) -> float:
        return _ratio(self.true_positives, self.n_truth)

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall, 0.0 when both are 0."""
        p, r = self.precision, self.recall
        return _ratio(2 * p * r, p + r)

    @property
    def average_precision(self) -> float:
        """``TP / (TP + FP + FN)`` -- the Cellpose/StarDist convention.

        NOT the area under a precision-recall curve, despite the name. The
        segmentation literature uses this quantity and calls it AP, and
        reporting the other thing under the same label is how two papers'
        numbers stop being comparable. Named explicitly here for that
        reason.
        """
        denominator = (self.true_positives + self.false_positives
                       + self.false_negatives)
        return _ratio(self.true_positives, denominator)


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when there is nothing to divide.

    0.0 rather than NaN, deliberately: these numbers end up in a published
    table and in a tooltip, and one NaN in a column is enough for a reader
    to distrust the column. An empty field scores zero, which is the honest
    reading of "found none of the objects that were not there".
    """
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else 0.0


def match_objects(truth: np.ndarray, pred: np.ndarray,
                  threshold: float = 0.5) -> Match:
    """Match predicted objects to labelled ones, one to one and optimally.

    :param threshold: minimum IoU for a pair to count as the same object.

    Uses ``linear_sum_assignment`` rather than a greedy pass so the result
    cannot depend on label order -- see this module's docstring.
    """
    from scipy.optimize import linear_sum_assignment

    ious, t_ids, p_ids = iou_matrix(truth, pred)
    n_truth, n_pred = int(t_ids.size), int(p_ids.size)
    if ious.size == 0:
        return Match(float(threshold), (), (), n_truth, n_pred)

    rows, cols = linear_sum_assignment(-ious)
    keep = [(int(r), int(c)) for r, c in zip(rows, cols)
            if ious[r, c] >= threshold]
    return Match(float(threshold), tuple(keep),
                 tuple(float(ious[r, c]) for r, c in keep), n_truth, n_pred)


def dice(truth: np.ndarray, pred: np.ndarray,
         threshold: float = 0.5) -> Dict[str, float]:
    """Per-object Dice, averaged over matched pairs, plus the pixel-wise one.

    TWO NUMBERS BECAUSE THEY ANSWER DIFFERENT QUESTIONS and are routinely
    confused. The per-object mean says how well a typical object is
    outlined; the pixel-wise figure ignores objects entirely and is
    dominated by the largest ones, so a model that misses ten small cells
    and nails one big one scores well on it and badly on the other.
    """
    matched = match_objects(truth, pred, threshold)
    per_object = [2 * i / (1 + i) for i in matched.ious]   # Dice from IoU
    t_fg, p_fg = np.asarray(truth) > 0, np.asarray(pred) > 0
    pixel = _ratio(2 * float((t_fg & p_fg).sum()),
                   float(t_fg.sum() + p_fg.sum()))
    return {
        "dice_per_object": float(np.mean(per_object)) if per_object else 0.0,
        "dice_pixel": pixel,
        "n_matched": len(per_object),
    }


def average_precision(truth: np.ndarray, pred: np.ndarray,
                      thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS
                      ) -> Dict[str, float]:
    """AP at each IoU in the sweep, and the mean across it."""
    per = {}
    for threshold in thresholds:
        per[f"ap_{threshold:g}"] = match_objects(
            truth, pred, threshold).average_precision
    per["ap_mean"] = float(np.mean(list(per.values()))) if per else 0.0
    return per


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of a label mask that touch a different label or background."""
    from scipy.ndimage import grey_dilation, grey_erosion

    mask = np.asarray(mask)
    return (grey_dilation(mask, size=3) != grey_erosion(mask, size=3)) & (mask > 0)


def boundary_f1(truth: np.ndarray, pred: np.ndarray,
                tolerances: Sequence[int] = DEFAULT_BOUNDARY_TOLERANCES
                ) -> Dict[str, float]:
    """Boundary precision, recall and F1 at each pixel tolerance.

    A predicted boundary pixel counts as correct when a true boundary pixel
    lies within ``tolerance`` pixels of it, and vice versa. The tolerance is
    not slack for the model's benefit: two people labelling the same cell
    disagree by a pixel or two, so a zero-tolerance boundary score measures
    the annotator as much as the model.
    """
    from scipy.ndimage import binary_dilation

    t_edge, p_edge = _boundary(truth), _boundary(pred)
    out: Dict[str, float] = {}
    for tol in tolerances:
        size = 2 * int(tol) + 1
        footprint = np.ones((size,) * t_edge.ndim, dtype=bool)
        t_near = binary_dilation(t_edge, structure=footprint)
        p_near = binary_dilation(p_edge, structure=footprint)
        precision = _ratio(float((p_edge & t_near).sum()), float(p_edge.sum()))
        recall = _ratio(float((t_edge & p_near).sum()), float(t_edge.sum()))
        out[f"boundary_precision_{tol}px"] = precision
        out[f"boundary_recall_{tol}px"] = recall
        out[f"boundary_f1_{tol}px"] = _ratio(2 * precision * recall,
                                             precision + recall)
    return out


def splits_and_merges(truth: np.ndarray, pred: np.ndarray,
                      minimum_overlap: float = 0.1) -> Dict[str, int]:
    """How many labelled objects were split, and how many were merged.

    :param minimum_overlap: fraction of the truth object a prediction must
        cover to count as overlapping it, so a one-pixel graze is not a
        split.

    THE TWO FAILURES ARE NOT SYMMETRIC IN WHAT THEY COST. A split inflates
    the object count and halves the areas; a merge deletes an object and
    doubles one. Both are invisible to Dice at the field level, which is why
    they are counted separately rather than folded into it.
    """
    inter, t_ids, p_ids = _overlap_matrix(truth, pred)
    if inter.size == 0:
        return {"splits": 0, "merges": 0}
    t_area = np.array([(np.asarray(truth) == i).sum() for i in t_ids])
    p_area = np.array([(np.asarray(pred) == i).sum() for i in p_ids])
    with np.errstate(divide="ignore", invalid="ignore"):
        of_truth = np.where(t_area[:, None] > 0, inter / t_area[:, None], 0.0)
        of_pred = np.where(p_area[None, :] > 0, inter / p_area[None, :], 0.0)
    splits = int(((of_truth >= minimum_overlap).sum(axis=1) > 1).sum())
    merges = int(((of_pred >= minimum_overlap).sum(axis=0) > 1).sum())
    return {"splits": splits, "merges": merges}


def counts_and_areas(truth: np.ndarray, pred: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """Object-count error (signed) and the area error a measurement inherits.

    SIGNED, because the direction is the diagnosis: a model that finds too
    many objects is over-segmenting and one that finds too few is merging or
    missing, and an absolute count error says neither.
    """
    matched = match_objects(truth, pred, threshold)
    errors: List[float] = []
    t_arr, p_arr = np.asarray(truth), np.asarray(pred)
    _, t_ids, p_ids = iou_matrix(truth, pred)
    for t_i, p_i in matched.pairs:
        t_area = float((t_arr == t_ids[t_i]).sum())
        p_area = float((p_arr == p_ids[p_i]).sum())
        if t_area:
            errors.append((p_area - t_area) / t_area)
    return {
        "n_truth": matched.n_truth,
        "n_pred": matched.n_pred,
        "count_error": matched.n_pred - matched.n_truth,
        "area_error_mean": float(np.mean(errors)) if errors else 0.0,
        "area_error_abs_mean": (float(np.mean(np.abs(errors)))
                                if errors else 0.0),
    }


def score_segmentation(truth: np.ndarray, pred: np.ndarray, *,
                       threshold: float = 0.5,
                       iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
                       boundary_tolerances: Sequence[int]
                       = DEFAULT_BOUNDARY_TOLERANCES) -> Dict[str, float]:
    """Every segmentation metric instruction 370 asks for, on one field.

    :param threshold: the IoU at which precision, recall, F1, Dice and the
        area error are matched. Reported in the result as ``match_iou`` so a
        published number can never be read without it.

    EVERY NUMBER CARRIES ITS N. `n_truth` and `n_pred` are in the result and
    are not optional: a Dice of 0.91 on eleven objects is not a result, and a
    table that omits the count invites exactly that reading.
    """
    matched = match_objects(truth, pred, threshold)
    out: Dict[str, float] = {
        "match_iou": float(threshold),
        "precision": matched.precision,
        "recall": matched.recall,
        "f1": matched.f1,
        "true_positives": matched.true_positives,
        "false_positives": matched.false_positives,
        "false_negatives": matched.false_negatives,
        "iou_mean": float(np.mean(matched.ious)) if matched.ious else 0.0,
    }
    out.update(dice(truth, pred, threshold))
    out.update(average_precision(truth, pred, iou_thresholds))
    out.update(boundary_f1(truth, pred, boundary_tolerances))
    out.update(splits_and_merges(truth, pred))
    out.update(counts_and_areas(truth, pred, threshold))
    return out


def compare_against_baseline(truth: np.ndarray, finetuned: np.ndarray,
                             vanilla: np.ndarray, **kwargs
                             ) -> Dict[str, Dict[str, float]]:
    """Score two models on the same truth and report the difference.

    :returns: ``{"finetuned": ..., "vanilla": ..., "delta": ...}``.

    THE DELTA IS THE ANSWER THE TABLE EXISTS FOR. Instruction 370 asks for a
    comparison "between the finetuned model and the vanilla model", and a
    reader given two columns of eleven numbers will do this subtraction by
    eye and get it wrong somewhere. Reporting it is not a convenience.

    Both models are scored on the SAME array with the SAME settings, which
    is the only thing that makes the subtraction meaningful -- and is why
    this takes three masks rather than two scorecards.
    """
    a = score_segmentation(truth, finetuned, **kwargs)
    b = score_segmentation(truth, vanilla, **kwargs)
    delta = {key: a[key] - b[key] for key in a
             if isinstance(a.get(key), (int, float))
             and isinstance(b.get(key), (int, float))}
    # `match_iou` is a SETTING, not a score. Subtracting it gives 0.0 and
    # reads in a table as "the threshold did not change", which is true and
    # is noise; leaving it in the delta invites someone to plot it.
    delta.pop("match_iou", None)
    return {"finetuned": a, "vanilla": b, "delta": delta}


# ---------------------------------------------------------------------------
# Classifiers
#
# A screen is unbalanced -- that is what a screen IS -- so several of these
# exist only because accuracy is uninformative when 98% of objects are
# negative. Instruction 370 asks for AUPRC "first when positives are rare,
# which in a screen they are", and for balanced accuracy and MCC for the
# same reason.
#
# BUILT ON `spacr.classifier_quality.Confusion` RATHER THAN BESIDE IT. That
# module already owns the confusion matrix, sensitivity, specificity and
# accuracy, and already refuses to report a correction its inputs cannot
# identify. A second implementation of the same four numbers is a second
# thing to keep in step, and they would disagree first in the place nobody
# is looking.
# ---------------------------------------------------------------------------


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, by rank, ties averaged.

    Computed here rather than imported so a scorecard can be read without
    sklearn present -- the zoo has to browse on a machine that never trained
    anything. The rank form is exact, not an approximation of the curve.
    """
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    if not positive or not negative:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average the ranks of tied scores, or a model that outputs one constant
    # scores 1.0 or 0.0 depending on sort order rather than the 0.5 it earns.
    values = np.asarray(scores)[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[labels == 1].sum()
                  - positive * (positive + 1) / 2) / (positive * negative))


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: the step-wise area under precision-recall.

    THE STEP FORM, not the trapezoid. Interpolating between operating points
    on a PR curve reports a precision no threshold achieves, which is the
    number people quote and cannot reproduce.
    """
    positive = int(labels.sum())
    if not positive:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    hits = labels[order].astype(float)
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positive)


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| by bin, weighted.

    Calibration decides whether a threshold means anything. A model at 0.9
    that is right 60% of the time is not "90% confident"; it is wrong about
    its own confidence, and every downstream cutoff inherits that.
    """
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (scores > low) & (scores <= high) if low > 0 else (
            (scores >= low) & (scores <= high))
        if not inside.any():
            continue
        weight = inside.mean()
        total += weight * abs(float(scores[inside].mean())
                              - float(labels[inside].mean()))
    return float(total)


def score_classifier(labels: Sequence[int], scores: Sequence[float], *,
                     threshold: float = 0.5,
                     calibration_bins: int = 10) -> Dict[str, float]:
    """Every classifier metric instruction 370 asks for, at a STATED threshold.

    :param labels: ground truth, 0 or 1.
    :param scores: predicted probability of the positive class.
    :param threshold: where a score becomes a positive call. Reported back in
        the result, because "precision 0.94" without it is not a claim
        anybody can check or reproduce.

    AUROC, AUPRC, Brier and ECE are threshold-FREE and are the numbers that
    survive a reader disagreeing with the cutoff; everything else moves when
    the threshold does. Both kinds are here, and the result says which is
    which by carrying `threshold` beside them.
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError(
            f"{labels.size} labels against {scores.size} scores; a metric "
            f"between them would be meaningless")
    if labels.size == 0:
        raise ValueError("no predictions to score")

    called = (scores >= threshold).astype(int)
    tp = int(((called == 1) & (labels == 1)).sum())
    fp = int(((called == 1) & (labels == 0)).sum())
    fn = int(((called == 0) & (labels == 1)).sum())
    tn = int(((called == 0) & (labels == 0)).sum())

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)                     # sensitivity
    specificity = _ratio(tn, tn + fp)
    negative_precision = _ratio(tn, tn + fn)
    negative_recall = specificity

    # MCC survives imbalance where accuracy and F1 do not: it is the only one
    # of these that uses all four cells of the matrix symmetrically.
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator else 0.0

    return {
        "threshold": float(threshold),
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "prevalence": _ratio(int(labels.sum()), labels.size),
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "accuracy": _ratio(tp + tn, labels.size),
        # THE SECOND ONE BECAUSE SCREENS ARE UNBALANCED. With 98% negatives,
        # calling everything negative scores 0.98 accuracy and 0.5 balanced.
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "precision_negative": negative_precision,
        "recall_negative": negative_recall,
        "f1_macro": (
            _ratio(2 * precision * recall, precision + recall)
            + _ratio(2 * negative_precision * negative_recall,
                     negative_precision + negative_recall)) / 2.0,
        "mcc": float(mcc),
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        # Brier is the mean squared error of the probability itself, so it
        # penalises a confident wrong answer more than a hesitant one.
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece": _ece(labels, scores, calibration_bins),
    }


def compare_classifier_against_baseline(labels: Sequence[int],
                                        finetuned: Sequence[float],
                                        vanilla: Sequence[float],
                                        **kwargs
                                        ) -> Dict[str, Dict[str, float]]:
    """Two classifiers on the same labels, and the difference.

    The same shape as :func:`compare_against_baseline`, and for the same
    reason: the delta is what the published table is for. `threshold` and
    the counts are dropped from it -- a difference in `n` is not a result,
    it is a sign the two were scored on different data, which this signature
    makes impossible.
    """
    a = score_classifier(labels, finetuned, **kwargs)
    b = score_classifier(labels, vanilla, **kwargs)
    delta = {key: a[key] - b[key] for key in a}
    for setting in ("threshold", "n", "n_positive", "prevalence"):
        delta.pop(setting, None)
    return {"finetuned": a, "vanilla": b, "delta": delta}


# ---------------------------------------------------------------------------
# A held-out SET, not a field
#
# Instruction 370: "A number computed on 'three fields' chosen at call time
# is not comparable between two models, between two versions of one model, or
# between two days. What makes the table worth publishing is that every model
# is scored on the SAME named, versioned, labelled set."
#
# So the unit of a published number is the SET. Everything below aggregates
# per-field scorecards into one, and the aggregation is not a mean of means.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutScore:
    """One model's result on one named, versioned held-out set.

    :param name: the set's name, e.g. ``toxo_pv_holdout``.
    :param version: the set's version. Two numbers are comparable only when
        this matches, and it is carried into every published row for that
        reason rather than being assumed from context.
    :param metrics: the pooled scorecard.
    :param per_field: one scorecard per field, in the order they were given.
    """

    name: str
    version: str
    metrics: Dict[str, float]
    per_field: Tuple[Dict[str, float], ...] = ()

    @property
    def n_fields(self) -> int:
        return len(self.per_field)


def score_holdout(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], *,
                  name: str, version: str, **kwargs) -> HoldoutScore:
    """Score a model over a whole held-out set.

    :param pairs: ``(truth, prediction)`` per field.

    POOLED FROM THE COUNTS, NOT AVERAGED FROM THE RATIOS. A field with three
    objects and a field with three hundred are not equal evidence, and a mean
    of per-field precisions treats them as though they were -- so a model
    that fails on one sparse field is punished as hard as one that fails on a
    confluent one. Precision, recall and F1 are recomputed from the summed
    true and false positives; only the genuinely per-object means (IoU, Dice,
    area error) are averaged, and those are weighted by the objects behind
    them.
    """
    scored = [score_segmentation(truth, pred, **kwargs) for truth, pred in pairs]
    if not scored:
        raise ValueError("a held-out set with no fields cannot be scored")

    tp = sum(int(s["true_positives"]) for s in scored)
    fp = sum(int(s["false_positives"]) for s in scored)
    fn = sum(int(s["false_negatives"]) for s in scored)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)

    def weighted(key: str, weight_key: str = "n_matched") -> float:
        weights = [float(s.get(weight_key, 0)) for s in scored]
        total = sum(weights)
        if not total:
            return 0.0
        return sum(float(s[key]) * w for s, w in zip(scored, weights)) / total

    pooled: Dict[str, float] = {
        "n_fields": len(scored),
        "n_truth": sum(int(s["n_truth"]) for s in scored),
        "n_pred": sum(int(s["n_pred"]) for s in scored),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "match_iou": scored[0]["match_iou"],
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "splits": sum(int(s["splits"]) for s in scored),
        "merges": sum(int(s["merges"]) for s in scored),
        "count_error": sum(int(s["count_error"]) for s in scored),
        "dice_per_object": weighted("dice_per_object"),
        "iou_mean": weighted("iou_mean"),
        "area_error_mean": weighted("area_error_mean"),
        "area_error_abs_mean": weighted("area_error_abs_mean"),
    }
    # The pixel-wise and boundary measures are field-level by nature, so they
    # are averaged over FIELDS and weighted by the objects each holds.
    for key in scored[0]:
        if key.startswith(("ap_", "boundary_", "dice_pixel")):
            pooled[key] = weighted(key, "n_truth")
    return HoldoutScore(str(name), str(version), pooled, tuple(scored))


def scorecard_rows(finetuned: HoldoutScore, vanilla: HoldoutScore
                   ) -> List[Dict[str, object]]:
    """The published table, one row per metric. THE SOURCE THE REST DERIVE FROM.

    Instruction 370 asks for four renderings -- the CSV on Hugging Face, the
    tooltip, the API section and the zoo screen -- and warns that "if the
    tooltip and the API page can disagree, they eventually will". This is the
    one place a number is computed; everything else formats these rows.

    :raises ValueError: when the two were scored on different sets. That is
        not a defensive check for its own sake: a table headed "finetuned
        against vanilla" whose two columns came from different data is
        exactly the mistake that cannot be seen by reading it.
    """
    if (finetuned.name, finetuned.version) != (vanilla.name, vanilla.version):
        raise ValueError(
            f"scored on different held-out sets -- {finetuned.name}"
            f"@{finetuned.version} against {vanilla.name}@{vanilla.version}; "
            f"the difference between them would mean nothing")

    rows: List[Dict[str, object]] = []
    for key in finetuned.metrics:
        if key not in vanilla.metrics:
            continue
        a, b = finetuned.metrics[key], vanilla.metrics[key]
        rows.append({
            "metric": key,
            "finetuned": a,
            "vanilla": b,
            # `None` rather than 0.0 for a setting or a count: a "difference"
            # in `match_iou` or `n_truth` is not a result, and a zero in that
            # column reads as one.
            "delta": (a - b) if key not in _NOT_A_SCORE else None,
            "n_fields": finetuned.metrics.get("n_fields"),
            "n_objects": finetuned.metrics.get("n_truth"),
            "holdout": finetuned.name,
            "holdout_version": finetuned.version,
        })
    return rows


#: Keys that describe the DATA or the settings, not the model's performance.
#: They appear in the table -- a reader needs the n -- but they have no
#: meaningful difference, and printing one invites somebody to plot it.
_NOT_A_SCORE = frozenset({
    "match_iou", "n_fields", "n_truth", "n_pred", "count_error",
})


def scorecard_csv(rows: Sequence[Dict[str, object]]) -> str:
    """The rows as CSV text. Written by the caller, wherever it belongs."""
    import csv
    import io

    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# The set itself: named, versioned, checksummed
#
# "What makes the table worth publishing is that every model is scored on the
# SAME named, versioned, labelled set, and that the set is published beside
# the models so the number can be checked by somebody else." -- 370.
#
# A manifest is what makes that checkable. Without one, "scored on the
# hold-out set" is a claim about a folder on somebody's laptop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutField:
    """One labelled field: the image a model reads, and the truth it is
    scored against.

    :param image: path to the field, relative to the manifest.
    :param truth: path to the label mask, relative to the manifest.
    :param sha256: digest of the TRUTH mask, or ``""``.
    """

    image: str
    truth: str
    sha256: str = ""


@dataclass(frozen=True)
class HoldoutSet:
    """A named, versioned set of labelled fields, as declared by a manifest.

    :param name: stable name, e.g. ``toxo_pv_holdout``.
    :param version: the version two numbers must share to be comparable.
    :param fields: the labelled fields, in manifest order.
    :param root: directory the relative paths resolve against.
    """

    name: str
    version: str
    fields: Tuple[HoldoutField, ...]
    root: "pathlib.Path"

    def path_to(self, relative: str) -> "pathlib.Path":
        return self.root / relative


def load_holdout(manifest_path) -> HoldoutSet:
    """Read a hold-out manifest.

    :raises ValueError: when the manifest lacks a name, a version or any
        field. All three are refusals rather than defaults, and the version
        most of all: a set that does not say which version it is cannot be
        compared against anything, and a default would let one be published
        as though it could.
    """
    import json
    import pathlib

    path = pathlib.Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name") or "").strip()
    version = str(data.get("version") or "").strip()
    if not name:
        raise ValueError(f"{path} declares no name")
    if not version:
        raise ValueError(
            f"{path} declares no version; a hold-out set that cannot say "
            f"which version it is cannot be compared against anything")
    raw_fields = data.get("fields") or []
    if not raw_fields:
        raise ValueError(f"{path} declares no fields")

    fields = []
    for entry in raw_fields:
        image = str(entry.get("image") or "").strip()
        truth = str(entry.get("truth") or "").strip()
        if not image or not truth:
            raise ValueError(
                f"{path}: a field needs both an image and a truth mask; "
                f"got {entry!r}")
        fields.append(HoldoutField(image, truth,
                                   str(entry.get("sha256") or "").strip()))
    return HoldoutSet(name, version, tuple(fields), path.parent)


def verify_holdout(holdout: HoldoutSet) -> List[str]:
    """Check every truth mask is present and matches its digest.

    :returns: one line per problem; empty when the set is intact.

    A DIGEST IS OPTIONAL AND ITS ABSENCE IS REPORTED. A set published without
    them can still be scored, and nobody can then tell whether two people
    scored the same masks -- which is the entire point of naming and
    versioning it. So "no digest" is a finding, not a pass.
    """
    import hashlib

    problems: List[str] = []
    for field in holdout.fields:
        target = holdout.path_to(field.truth)
        if not target.is_file():
            problems.append(f"missing truth mask: {field.truth}")
            continue
        if not field.sha256:
            problems.append(f"no digest published for {field.truth}")
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != field.sha256:
            problems.append(
                f"{field.truth} does not match its digest "
                f"(published {field.sha256[:12]}..., found {digest[:12]}...)")
    return problems


def score_model_on_holdout(holdout: HoldoutSet, predict, *,
                           read_mask=None, **kwargs) -> HoldoutScore:
    """Run ``predict`` over a hold-out set and score it against the truth.

    :param predict: ``image path -> label array``. Injected rather than
        imported: this module must keep importing without torch, and a
        segmentation model is the one thing that cannot.
    :param read_mask: ``path -> label array``. Defaults to tifffile, which
        the package already depends on.

    THE SET'S NAME AND VERSION TRAVEL WITH THE SCORE, so a published number
    can never be read without knowing what it was measured on.
    """
    if read_mask is None:
        def read_mask(path):                     # noqa: WPS440 - local default
            import tifffile

            return np.asarray(tifffile.imread(str(path)))

    pairs = [(read_mask(holdout.path_to(f.truth)),
              predict(holdout.path_to(f.image)))
             for f in holdout.fields]
    return score_holdout(pairs, name=holdout.name, version=holdout.version,
                         **kwargs)


#: The numbers a reader choosing between two models actually needs, in the
#: order they decide it. Everything else is in the table.
#:
#: THE TOOLTIP IS THE SURFACE WITH THE LEAST ROOM -- 370 says so -- and a
#: scorecard is 37 rows. Dumping all of them into a hover is the same mistake
#: as the model-zoo table before it was cut to three columns: technically
#: complete and unreadable, which is not a kindness.
_HEADLINE_ORDER: Tuple[Tuple[str, str], ...] = (
    ("f1", "F1"),
    ("dice_per_object", "Dice"),
    ("ap_mean", "AP 0.5-0.9"),
    ("boundary_f1_1px", "Boundary F1"),
    ("auprc", "AUPRC"),
    ("balanced_accuracy", "Balanced acc"),
)


def headline(metrics: Mapping[str, object], *,
             baseline: Optional[Mapping[str, object]] = None,
             limit: int = 3) -> List[str]:
    """The few lines a tooltip should lead with, and where the rest is.

    :param metrics: a scorecard, or any mapping; unknown keys are ignored.
    :param baseline: the vanilla model's scorecard, to show the difference.
    :param limit: how many metric lines to return before the pointer.

    :returns: lines, or ``[]`` when the mapping holds no scorecard at all --
        an entry with free-form metrics is left exactly as it was, because
        this must not turn somebody's two-line note into a truncated table.
    """
    lines: List[str] = []
    for key, label in _HEADLINE_ORDER:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])                      # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        text = f"{label} {value:.3f}"
        if baseline and key in baseline:
            try:
                delta = value - float(baseline[key])         # type: ignore[arg-type]
                # SIGNED AND EXPLICIT. "F1 0.867" alone answers "what is it";
                # the request is "is it better", which only the difference
                # answers.
                text += f" ({delta:+.3f} vs stock)"
            except (TypeError, ValueError):
                pass
        lines.append(text)
        if len(lines) >= limit:
            break
    if not lines:
        return []

    # EVERY NUMBER CARRIES ITS N, in the tooltip too.
    counted = []
    for key, label in (("n_truth", "objects"), ("n_fields", "fields"),
                       ("n", "predictions")):
        if key in metrics:
            counted.append(f"{metrics[key]} {label}")
    if counted:
        lines.append("on " + ", ".join(counted))
    return lines
