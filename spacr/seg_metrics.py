"""Score a segmentation against ground truth.

WHAT THIS ADDS THAT spaCR DID NOT HAVE. `model_compare` puts two models head
to head and says plainly in its own docstring that "neither model is ground
truth"; `seg_qc` scores a field with no labels at all, flagging splits, merges
and implausible diameters. Both are useful and neither is accuracy. Nothing
here measured a mask against a HUMAN-DRAWN one, so no model could carry a
number a reader could compare with anybody else's.

THE OBJECT IS THE UNIT, NOT THE PIXEL, and that distinction decides most of
these definitions. A segmentation that finds every cell but leaves each one a
few pixels too small scores well; one that merges two touching cells into one
blob scores badly -- even though the second may have almost the same pixel
overlap. Pixel Dice cannot tell those apart, which is why the object measures
below matter more for cell segmentation, and why both are reported.

A NOTE ON WHAT "AVERAGE PRECISION" MEANS HERE. In segmentation benchmarks it
is NOT the area under a precision-recall curve. It is TP / (TP + FP + FN) at a
given IoU threshold -- the convention used by the Cellpose and Stardist papers
and by the Data Science Bowl -- and it is reported averaged over thresholds.
A reader who assumes the detection meaning will read the number as far worse
than it is, so it is named and defined here rather than left to inference.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: The IoU thresholds a benchmark score is averaged over, matching the
#: convention published segmentation papers use.
DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7,
                                         0.75, 0.8, 0.85, 0.9, 0.95)


def _labels(mask: np.ndarray) -> np.ndarray:
    """The object labels present, excluding background."""
    values = np.unique(np.asarray(mask))
    return values[values != 0]


def iou_matrix(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Intersection over union for every ground-truth/predicted pair.

    Built from a joint histogram rather than a double loop: for a field with
    a few hundred objects the loop is the difference between milliseconds and
    a minute, and a metric nobody waits for is a metric nobody runs.

    :param truth: labelled ground-truth mask, 0 as background.
    :param predicted: labelled predicted mask, 0 as background.
    :returns: ``(n_truth, n_predicted)`` IoU, rows ordered by
        :func:`_labels` of ``truth`` and columns by that of ``predicted``.
    """
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape:
        raise ValueError(
            f"masks must have the same shape; got {truth.shape} and "
            f"{predicted.shape}"
        )
    t_labels, p_labels = _labels(truth), _labels(predicted)
    if t_labels.size == 0 or p_labels.size == 0:
        return np.zeros((t_labels.size, p_labels.size), np.float64)

    t_index = {int(v): i for i, v in enumerate(t_labels)}
    p_index = {int(v): i for i, v in enumerate(p_labels)}
    overlap = np.zeros((t_labels.size, p_labels.size), np.int64)

    flat_t, flat_p = truth.ravel(), predicted.ravel()
    both = (flat_t != 0) & (flat_p != 0)
    if both.any():
        rows = np.fromiter((t_index[int(v)] for v in flat_t[both]),
                           np.int64, int(both.sum()))
        cols = np.fromiter((p_index[int(v)] for v in flat_p[both]),
                           np.int64, int(both.sum()))
        np.add.at(overlap, (rows, cols), 1)

    t_area = np.array([(truth == v).sum() for v in t_labels], np.int64)
    p_area = np.array([(predicted == v).sum() for v in p_labels], np.int64)
    union = t_area[:, None] + p_area[None, :] - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, overlap / union, 0.0)


def match_objects(truth: np.ndarray, predicted: np.ndarray, *,
                  threshold: float = 0.5) -> Tuple[int, int, int]:
    """Count true positives, false positives and false negatives.

    ONE PREDICTION MAY CLAIM ONE OBJECT. The matching is greedy on IoU, so a
    predicted blob covering two real cells matches the better of them and the
    other counts as missed -- which is the answer that reflects what went
    wrong. Letting one prediction satisfy several truths would score a merge
    as a success, and merges are the failure this is most needed to catch.

    :param truth: labelled ground-truth mask.
    :param predicted: labelled predicted mask.
    :param threshold: the IoU at which a pair counts as the same object.
    :returns: ``(true_positives, false_positives, false_negatives)``.
    """
    iou = iou_matrix(truth, predicted)
    n_truth, n_pred = iou.shape
    if n_truth == 0 or n_pred == 0:
        return 0, n_pred, n_truth

    pairs = [(float(iou[i, j]), i, j)
             for i in range(n_truth) for j in range(n_pred)
             if iou[i, j] >= threshold]
    pairs.sort(reverse=True)

    used_truth, used_pred = set(), set()
    matched = 0
    for _score, i, j in pairs:
        if i in used_truth or j in used_pred:
            continue
        used_truth.add(i)
        used_pred.add(j)
        matched += 1
    return matched, n_pred - matched, n_truth - matched


def average_precision(truth: np.ndarray, predicted: np.ndarray, *,
                      thresholds: Sequence[float] = DEFAULT_THRESHOLDS
                      ) -> Dict[str, float]:
    """The segmentation benchmark score, per threshold and averaged.

    NOT the detection AP. Here it is ``TP / (TP + FP + FN)`` at each IoU
    threshold, which is the definition the Cellpose and Stardist papers and
    the Data Science Bowl use, and it falls between 0 and 1 with 1 meaning
    every object found and none invented.

    :param truth: labelled ground-truth mask.
    :param predicted: labelled predicted mask.
    :param thresholds: the IoU thresholds to score at.
    :returns: ``{"ap@0.50": ..., ..., "mAP": ...}``.
    """
    out: Dict[str, float] = {}
    scores = []
    for threshold in thresholds:
        tp, fp, fn = match_objects(truth, predicted, threshold=threshold)
        denominator = tp + fp + fn
        score = float(tp / denominator) if denominator else 1.0
        out[f"ap@{threshold:.2f}"] = score
        scores.append(score)
    out["mAP"] = float(np.mean(scores)) if scores else 1.0
    return out


def pixel_scores(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Foreground-versus-background agreement, ignoring object identity.

    Reported ALONGSIDE the object measures rather than instead of them: this
    is the number that stays high when two touching cells are merged into one
    blob, and the object scores are the ones that fall. Seeing both is what
    tells a reader WHICH kind of mistake a model makes.

    :param truth: labelled or binary ground-truth mask.
    :param predicted: labelled or binary predicted mask.
    :returns: ``{"dice", "jaccard", "pixel_precision", "pixel_recall"}``.
    """
    t = np.asarray(truth) != 0
    p = np.asarray(predicted) != 0
    if t.shape != p.shape:
        raise ValueError("masks must have the same shape")

    intersection = float(np.logical_and(t, p).sum())
    t_sum, p_sum = float(t.sum()), float(p.sum())
    union = t_sum + p_sum - intersection

    # Two empty masks AGREE COMPLETELY, and returning 0 for that would make
    # an empty field look like a total failure rather than a correct answer.
    dice = 1.0 if t_sum + p_sum == 0 else 2 * intersection / (t_sum + p_sum)
    jaccard = 1.0 if union == 0 else intersection / union
    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "pixel_precision": float(intersection / p_sum) if p_sum else 1.0,
        "pixel_recall": float(intersection / t_sum) if t_sum else 1.0,
    }


def boundary_f1(truth: np.ndarray, predicted: np.ndarray, *,
                tolerance: int = 2) -> Dict[str, float]:
    """How well the OUTLINES agree, within a few pixels.

    WHY A SEPARATE MEASURE. Dice is dominated by the interior of an object,
    so a model that finds every cell but consistently draws its edge two
    pixels out scores almost perfectly -- and every intensity feature
    measured from that mask is contaminated by the neighbouring background.
    Boundary F1 is the measure that notices.

    The tolerance exists because a hand-drawn outline is not accurate to the
    pixel either, so demanding exact agreement measures the annotator's hand
    rather than the model.

    :param truth: labelled or binary ground-truth mask.
    :param predicted: labelled or binary predicted mask.
    :param tolerance: how many pixels away a boundary pixel may be and still
        count as agreeing.
    :returns: ``{"boundary_precision", "boundary_recall", "boundary_f1"}``.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    def edges(mask):
        """The one-pixel rim of every object in ``mask``.

        Erosion-and-subtract rather than a gradient: it gives a rim that is
        strictly INSIDE the object, so a boundary pixel always belongs to the
        thing it outlines and two objects touching each other do not share
        one.

        :param mask: labelled or binary mask.
        :returns: a boolean array, True on the rim.
        """
        binary = np.asarray(mask) != 0
        return binary & ~binary_erosion(binary, iterations=1)

    t_edge, p_edge = edges(truth), edges(predicted)
    if not t_edge.any() and not p_edge.any():
        return {"boundary_precision": 1.0, "boundary_recall": 1.0,
                "boundary_f1": 1.0}
    if not t_edge.any() or not p_edge.any():
        return {"boundary_precision": 0.0, "boundary_recall": 0.0,
                "boundary_f1": 0.0}

    # Distance from every pixel to the nearest edge of the OTHER mask.
    to_truth = distance_transform_edt(~t_edge)
    to_pred = distance_transform_edt(~p_edge)

    precision = float((to_truth[p_edge] <= tolerance).mean())
    recall = float((to_pred[t_edge] <= tolerance).mean())
    denominator = precision + recall
    f1 = 0.0 if denominator == 0 else 2 * precision * recall / denominator
    return {"boundary_precision": precision, "boundary_recall": recall,
            "boundary_f1": float(f1)}


def scorecard(truth: np.ndarray, predicted: np.ndarray, *,
              thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
              boundary_tolerance: int = 2) -> Dict[str, float]:
    """Every metric for one field, as one flat mapping.

    Object counts are included deliberately. A reader looking at a poor score
    wants to know first whether the model found too few objects or too many,
    and a table of ratios cannot answer that.

    :param truth: labelled ground-truth mask.
    :param predicted: labelled predicted mask.
    :param thresholds: IoU thresholds for the benchmark score.
    :param boundary_tolerance: pixels of slack for the boundary measure.
    :returns: one mapping of every metric.
    """
    out: Dict[str, float] = {
        "n_truth": float(_labels(truth).size),
        "n_predicted": float(_labels(predicted).size),
    }
    tp, fp, fn = match_objects(truth, predicted, threshold=0.5)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1_den = precision + recall
    out.update({
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": 0.0 if f1_den == 0 else float(2 * precision * recall / f1_den),
    })
    out.update(average_precision(truth, predicted, thresholds=thresholds))
    out.update(pixel_scores(truth, predicted))
    out.update(boundary_f1(truth, predicted, tolerance=boundary_tolerance))
    return out
