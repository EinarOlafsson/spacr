"""Score masks whose right answer we can work out on paper.

Nothing in spaCR measured a segmentation against ground truth: `model_compare`
says in its own docstring that "neither model is ground truth", and `seg_qc`
has no labels at all. So these metrics are new, and a new metric is worth
exactly as much as the checking behind it -- a plausible-looking number from a
subtly wrong formula is worse than no number, because it gets published.

Every case below is small enough to verify by hand, and the arithmetic is
written into the assertions.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spacr import seg_metrics


def _two_cells():
    """A 40x40 field with two 10x10 cells, well apart."""
    mask = np.zeros((40, 40), np.int32)
    mask[5:15, 5:15] = 1
    mask[5:15, 25:35] = 2
    return mask


def test_a_perfect_segmentation_scores_one_everywhere():
    """Identical masks must give 1.0 for every metric, with no exceptions."""
    truth = _two_cells()
    card = seg_metrics.scorecard(truth, truth.copy())

    for key in ("precision", "recall", "f1", "mAP", "dice", "jaccard",
                "boundary_f1", "pixel_precision", "pixel_recall"):
        assert card[key] == pytest.approx(1.0), f"{key} = {card[key]}"
    assert card["true_positives"] == 2
    assert card["false_positives"] == 0
    assert card["false_negatives"] == 0


def test_an_empty_field_matched_by_an_empty_prediction_is_not_a_failure():
    """Finding nothing where there is nothing is CORRECT, and must score 1.

    Returning 0 for the empty case is the classic division-by-zero shortcut,
    and it makes a model that correctly declines to hallucinate look like the
    worst in the table.
    """
    empty = np.zeros((20, 20), np.int32)
    card = seg_metrics.scorecard(empty, empty)
    assert card["dice"] == 1.0
    assert card["jaccard"] == 1.0
    assert card["mAP"] == 1.0
    assert card["boundary_f1"] == 1.0


def test_the_iou_is_the_arithmetic_we_expect():
    """A known overlap, checked against the fraction worked out by hand."""
    truth = np.zeros((20, 20), np.int32)
    truth[0:10, 0:10] = 1                       # 100 px
    pred = np.zeros((20, 20), np.int32)
    pred[5:15, 0:10] = 1                        # 100 px, overlapping 50

    iou = seg_metrics.iou_matrix(truth, pred)
    assert iou.shape == (1, 1)
    # intersection 50, union 100 + 100 - 50 = 150
    assert iou[0, 0] == pytest.approx(50 / 150)

    # ...so it is a match at 0.3 and not at 0.5.
    assert seg_metrics.match_objects(truth, pred, threshold=0.3) == (1, 0, 0)
    assert seg_metrics.match_objects(truth, pred, threshold=0.5) == (0, 1, 1)


def test_a_merge_wrecks_the_object_score_and_barely_moves_dice():
    """THE CASE THE OBJECT METRICS EXIST FOR.

    Two touching cells predicted as one blob is the failure that matters most
    in cell segmentation: every per-cell measurement downstream is then the
    average of two cells. Pixel Dice hardly notices -- almost the same pixels
    are foreground -- and the object score collapses. Both are asserted,
    because it is the CONTRAST that tells a reader which mistake was made.
    """
    truth = np.zeros((30, 30), np.int32)
    truth[5:15, 5:15] = 1
    truth[5:15, 15:25] = 2                      # touching, not overlapping
    merged = np.zeros((30, 30), np.int32)
    merged[5:15, 5:25] = 1                      # one blob over both

    card = seg_metrics.scorecard(truth, merged)

    # Dice sees the same foreground: identical pixels, so exactly 1.0.
    assert card["dice"] == pytest.approx(1.0)
    # The objects tell the truth. The blob is 200 px and each cell is 100, so
    # its IoU with either is exactly 100/200 = 0.5 -- which DOES match at the
    # 0.5 threshold, once. The second cell has nothing left to match and is
    # counted missed, which is the honest description of a merge: one cell
    # found, one lost, and no invention.
    assert card["true_positives"] == 1
    assert card["false_negatives"] == 1
    assert card["false_positives"] == 0
    assert card["recall"] == pytest.approx(0.5)
    # And at any threshold above 0.5 the blob satisfies neither.
    assert seg_metrics.match_objects(truth, merged, threshold=0.6) == (0, 1, 2)
    # The averaged score collapses even though Dice is perfect. That contrast
    # is the whole reason both are reported.
    assert card["mAP"] < 0.15


def test_a_split_is_counted_as_one_hit_and_one_invention():
    """One cell predicted as two: one match, one false positive."""
    truth = np.zeros((30, 30), np.int32)
    truth[5:15, 5:25] = 1                       # one wide cell, 200 px
    split = np.zeros((30, 30), np.int32)
    split[5:15, 5:15] = 1                       # 100 px, IoU 100/200 = 0.5
    split[5:15, 15:25] = 2                      # 100 px, IoU 0.5

    tp, fp, fn = seg_metrics.match_objects(truth, split, threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 0), (
        "one half matches at exactly 0.5; the other is an invention"
    )


def test_a_missed_cell_and_an_invented_one_are_counted_separately():
    """Recall falls for what was missed; precision for what was invented."""
    truth = _two_cells()
    only_one = np.zeros((40, 40), np.int32)
    only_one[5:15, 5:15] = 1
    card = seg_metrics.scorecard(truth, only_one)
    assert card["recall"] == pytest.approx(0.5)
    assert card["precision"] == pytest.approx(1.0)

    extra = truth.copy()
    extra[25:35, 5:15] = 3                      # a cell that is not there
    card = seg_metrics.scorecard(truth, extra)
    assert card["recall"] == pytest.approx(1.0)
    assert card["precision"] == pytest.approx(2 / 3)


def test_an_outline_drawn_two_pixels_wide_of_the_truth_shows_up_as_boundary_error():
    """Dice stays high while boundary F1 falls, which is the point of it.

    A mask consistently drawn a couple of pixels out finds every cell and
    contaminates every intensity feature with neighbouring background. Dice
    is dominated by the interior and barely moves; the boundary measure is
    the one that notices.
    """
    truth = np.zeros((60, 60), np.int32)
    truth[20:40, 20:40] = 1                     # 20x20
    grown = np.zeros((60, 60), np.int32)
    grown[18:42, 18:42] = 1                     # 24x24, edge 2 px out

    card = seg_metrics.scorecard(truth, grown, boundary_tolerance=1)
    # 400 px against 576, intersecting in 400: dice = 800/976 = 0.82.
    assert card["dice"] > 0.8, "the interiors still agree"
    assert card["boundary_f1"] < 0.2, (
        f"boundary F1 {card['boundary_f1']:.3f} should notice a 3 px "
        f"displacement at 1 px tolerance"
    )


def test_average_precision_is_the_segmentation_definition_not_the_detection_one():
    """AP here is TP / (TP + FP + FN), and the value is checked by hand."""
    truth = _two_cells()
    only_one = np.zeros((40, 40), np.int32)
    only_one[5:15, 5:15] = 1

    scores = seg_metrics.average_precision(truth, only_one, thresholds=(0.5,))
    # one hit, no inventions, one miss: 1 / (1 + 0 + 1)
    assert scores["ap@0.50"] == pytest.approx(0.5)
    assert scores["mAP"] == pytest.approx(0.5)


def test_masks_of_different_shapes_are_refused():
    """Comparing a 10x10 to a 20x20 is a caller error, not a zero score."""
    with pytest.raises(ValueError):
        seg_metrics.iou_matrix(np.zeros((10, 10), np.int32),
                               np.zeros((20, 20), np.int32))
