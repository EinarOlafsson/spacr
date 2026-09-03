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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
