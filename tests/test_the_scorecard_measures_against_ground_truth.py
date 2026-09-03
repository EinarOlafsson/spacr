"""Every metric checked against a case whose answer is known by construction.

Instruction 370 asks every published model to carry a finetuned-against-
vanilla table on a held-out set. Auditing what existed found the gap this
module fills: `model_compare` compares two models to EACH OTHER and says so
-- "neither model is ground truth" -- and `seg_qc` scores a field with no
labels at all. Nothing measured a mask against a hand-drawn truth, so no
published number could say a model was BETTER, only different.

A METRICS MODULE IS THE EASIEST KIND OF CODE TO GET CONFIDENTLY WRONG. Every
function returns a plausible number for any input, and a wrong one is
indistinguishable from a right one without a case whose answer is fixed in
advance. So every test here builds a mask whose score is known by
construction -- a perfect copy, a shifted square, one object split in two,
two merged into one -- rather than asserting that the output is between 0
and 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import scorecard


def _square(size=80, box=(10, 40), label=1):
    a = np.zeros((size, size), dtype=int)
    lo, hi = box
    a[lo:hi, lo:hi] = label
    return a


def test_a_perfect_prediction_scores_one_everywhere():
    truth = _square()
    result = scorecard.score_segmentation(truth, truth.copy())
    for key in ("precision", "recall", "f1", "dice_per_object", "dice_pixel",
                "iou_mean", "ap_0.5", "ap_0.9", "ap_mean",
                "boundary_f1_1px", "boundary_f1_3px"):
        assert result[key] == pytest.approx(1.0), f"{key} was {result[key]}"
    assert result["splits"] == 0 and result["merges"] == 0
    assert result["count_error"] == 0
    assert result["area_error_mean"] == pytest.approx(0.0)


def test_an_empty_prediction_scores_zero_and_never_nan():
    """0.0 rather than NaN is deliberate: these land in a published table and
    a tooltip, and one NaN in a column makes a reader distrust the column."""
    truth = _square()
    result = scorecard.score_segmentation(truth, np.zeros_like(truth))
    assert result["recall"] == 0.0 and result["f1"] == 0.0
    assert result["n_pred"] == 0 and result["count_error"] == -1
    assert not any(isinstance(v, float) and np.isnan(v)
                   for v in result.values())


def test_iou_is_the_number_the_geometry_says_it_is():
    """Two 30x30 squares offset by 10 px in BOTH axes.

    Overlap is 20 px on each axis, so the intersection is 20*20 = 400 and
    the union is 900 + 900 - 400 = 1400. Written out because the first
    version of this test asserted 20*30 -- the square is offset diagonally,
    not along one axis, and the arithmetic has to follow the picture.
    """
    truth = _square(box=(10, 40))
    pred = _square(box=(20, 50), label=1)
    ious, _, _ = scorecard.iou_matrix(truth, pred)
    assert ious[0, 0] == pytest.approx(400 / 1400, rel=1e-9)


def test_dice_and_iou_agree_through_their_identity():
    """Dice = 2*IoU/(1+IoU). Asserted because the per-object Dice is DERIVED
    from IoU, so an error in that conversion would be invisible to a test
    that only checked ranges."""
    truth = _square(box=(10, 40))
    pred = _square(box=(15, 45))
    ious, _, _ = scorecard.iou_matrix(truth, pred)
    iou = float(ious[0, 0])
    assert scorecard.dice(truth, pred)["dice_per_object"] == pytest.approx(
        2 * iou / (1 + iou))


def test_matching_does_not_depend_on_label_order():
    """The reason matching is optimal rather than greedy.

    A greedy pass takes the first pair it sees, so relabelling the mask --
    an implementation detail of whoever wrote it -- changes the score.
    """
    truth = np.zeros((60, 60), dtype=int)
    truth[5:25, 5:25] = 1
    truth[30:50, 30:50] = 2
    pred = np.zeros((60, 60), dtype=int)
    pred[6:25, 6:25] = 2          # ids deliberately swapped
    pred[30:50, 30:50] = 1

    a = scorecard.score_segmentation(truth, pred)
    b = scorecard.score_segmentation(truth, np.where(pred == 1, 2,
                                                     np.where(pred == 2, 1, 0)))
    assert a["f1"] == b["f1"] == 1.0
    assert a["iou_mean"] == pytest.approx(b["iou_mean"])


def test_a_split_is_counted_as_a_split():
    """One labelled object, two predictions covering half each."""
    truth = np.zeros((60, 60), dtype=int)
    truth[10:50, 10:50] = 1
    pred = np.zeros((60, 60), dtype=int)
    pred[10:30, 10:50] = 1
    pred[30:50, 10:50] = 2

    counted = scorecard.splits_and_merges(truth, pred)
    assert counted["splits"] == 1
    assert counted["merges"] == 0
    assert scorecard.score_segmentation(truth, pred)["count_error"] == 1


def test_a_merge_is_counted_as_a_merge():
    """Two labelled objects, one prediction covering both."""
    truth = np.zeros((60, 60), dtype=int)
    truth[10:25, 10:50] = 1
    truth[25:40, 10:50] = 2
    pred = np.zeros((60, 60), dtype=int)
    pred[10:40, 10:50] = 1

    counted = scorecard.splits_and_merges(truth, pred)
    assert counted["merges"] == 1
    assert counted["splits"] == 0
    assert scorecard.score_segmentation(truth, pred)["count_error"] == -1


def test_ap_falls_as_the_threshold_rises_for_a_loose_outline():
    """The whole reason the sweep is reported instead of one number: a model
    that FINDS every object but outlines it loosely scores well at 0.5 and
    badly at 0.9."""
    truth = _square(box=(20, 60))
    pred = _square(box=(22, 58))          # inside the truth, smaller
    swept = scorecard.average_precision(truth, pred)
    assert swept["ap_0.5"] == 1.0
    assert swept["ap_0.9"] == 0.0
    assert 0.0 < swept["ap_mean"] < 1.0


def test_boundary_f1_is_stricter_than_dice_on_a_ragged_edge():
    """Dice barely moves when an outline is ragged -- a 40x40 square has
    1,600 interior pixels and about 156 boundary ones -- which is the whole
    reason 370 asks for a boundary measure as well.

    THE TEETH ARE 3 px DEEP, not 1. A one-pixel nibble is inside the 1 px
    tolerance by construction and scores a perfect boundary F1, which is the
    tolerance working correctly rather than a bug -- the first version of
    this test used one and proved nothing.
    """
    truth = _square(box=(20, 60))
    pred = truth.copy()
    pred[20:60:2, 17:20] = 1              # 3 px teeth on alternate rows

    scored = scorecard.score_segmentation(truth, pred)
    assert scored["dice_pixel"] > 0.97, "the area barely changed, as intended"
    assert scored["boundary_f1_1px"] < 0.9, (
        "a boundary that wanders 3 px must not score like a clean one")
    assert scored["boundary_f1_3px"] > scored["boundary_f1_1px"]


def test_the_boundary_tolerance_actually_loosens_it():
    truth = _square(box=(20, 60))
    pred = _square(box=(22, 62))          # shifted two pixels
    scored = scorecard.boundary_f1(truth, pred)
    assert (scored["boundary_f1_1px"] <= scored["boundary_f1_2px"]
            <= scored["boundary_f1_3px"])


def test_the_area_error_is_signed():
    """The direction is the diagnosis: too big and too small are different
    failures, and an absolute error says neither."""
    truth = _square(box=(20, 60))
    smaller = _square(box=(25, 55))
    assert scorecard.counts_and_areas(truth, smaller)["area_error_mean"] < 0
    bigger = _square(box=(15, 65))
    assert scorecard.counts_and_areas(truth, bigger)["area_error_mean"] > 0


def test_the_comparison_reports_the_difference_and_drops_the_setting():
    truth = _square(box=(20, 60))
    good = _square(box=(20, 60))
    poor = _square(box=(30, 50))

    out = scorecard.compare_against_baseline(truth, good, poor)
    assert set(out) == {"finetuned", "vanilla", "delta"}
    assert out["delta"]["f1"] > 0
    assert out["delta"]["dice_per_object"] > 0
    assert "match_iou" not in out["delta"], (
        "match_iou is a SETTING; a zero difference for it reads in a table "
        "as a result and invites someone to plot it")


def test_mismatched_shapes_are_refused_rather_than_scored():
    with pytest.raises(ValueError, match="different"):
        scorecard.score_segmentation(np.zeros((10, 10), int),
                                     np.zeros((12, 12), int))


def test_the_module_needs_no_torch():
    """The model zoo imports without torch on purpose and a test asserts it,
    so reading a scorecard must not be what drags in a GPU stack."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import spacr.scorecard\n"
        "assert 'torch' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if m.startswith('torch'))\n"
        "print('clean')\n"
    )
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def test_a_perfect_classifier_scores_one_and_a_useless_one_scores_chance():
    perfect = scorecard.score_classifier([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert perfect["accuracy"] == 1.0
    assert perfect["auroc"] == 1.0
    assert perfect["auprc"] == 1.0
    assert perfect["mcc"] == pytest.approx(1.0)

    inverted = scorecard.score_classifier([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])
    assert inverted["auroc"] == 0.0
    assert inverted["mcc"] == pytest.approx(-1.0)


def test_a_constant_score_is_chance_not_an_artefact_of_sort_order():
    """Ties are rank-averaged. Without that, a model that outputs one number
    scores 1.0 or 0.0 depending on how the sort happened to break ties."""
    labels = [1, 0, 1, 0, 1, 0]
    assert scorecard.score_classifier(
        labels, [0.5] * 6)["auroc"] == pytest.approx(0.5)


def test_accuracy_flatters_an_unbalanced_screen_and_balanced_accuracy_does_not():
    """The reason 370 asks for the second one. A screen IS unbalanced."""
    labels = np.r_[np.ones(20, int), np.zeros(980, int)]
    calls_everything_negative = np.zeros(1000)

    scored = scorecard.score_classifier(labels, calls_everything_negative)
    assert scored["accuracy"] == pytest.approx(0.98)
    assert scored["balanced_accuracy"] == pytest.approx(0.5)
    assert scored["recall"] == 0.0
    assert scored["mcc"] == 0.0


def test_auprc_is_the_one_that_notices_rare_positives():
    """"AUPRC first when positives are rare, which in a screen they are."

    A ranker that puts a few false positives above the true ones barely
    dents AUROC and visibly dents AUPRC, which is the whole argument for
    reporting it.
    """
    # TWENTY false positives ranked above ten true ones, out of 990
    # negatives. AUROC loses only 20/990 of its pairs and stays near 0.98;
    # AUPRC sees precision of 10/30 at the point that matters and collapses.
    #
    # The first version of this test put ~146 negatives above the positives
    # by accident -- `np.linspace(0.95, 0, 990)` puts a seventh of them above
    # 0.81 -- which dented AUROC too and proved nothing about the pair.
    labels = np.r_[np.ones(10, int), np.zeros(990, int)]
    scores = np.r_[np.full(10, 0.80),
                   np.full(20, 0.90),
                   np.linspace(0.70, 0.0, 970)]

    scored = scorecard.score_classifier(labels, scores)
    assert scored["auroc"] > 0.97, scored["auroc"]
    assert scored["auprc"] < 0.6
    assert scored["auprc"] < scored["auroc"]


def test_brier_punishes_confident_wrongness_more_than_hesitant_wrongness():
    confident = scorecard.score_classifier([1, 0], [0.0, 1.0])["brier"]
    hesitant = scorecard.score_classifier([1, 0], [0.45, 0.55])["brier"]
    assert confident > hesitant


def test_ece_is_zero_for_a_calibrated_model_and_large_for_an_overconfident_one():
    """Calibration decides whether a threshold means anything: a model at
    0.9 that is right 60% of the time is wrong about its own confidence."""
    rng = np.random.default_rng(11)
    probabilities = rng.uniform(0.05, 0.95, 4000)
    honest = (rng.uniform(size=4000) < probabilities).astype(int)
    assert scorecard.score_classifier(honest, probabilities)["ece"] < 0.05

    overconfident = np.full(4000, 0.95)
    truth = (rng.uniform(size=4000) < 0.5).astype(int)
    assert scorecard.score_classifier(truth, overconfident)["ece"] > 0.4


def test_the_threshold_is_reported_because_the_numbers_move_with_it():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.55, 0.1]
    low = scorecard.score_classifier(labels, scores, threshold=0.5)
    high = scorecard.score_classifier(labels, scores, threshold=0.7)

    assert low["threshold"] == 0.5 and high["threshold"] == 0.7
    assert low["recall"] > high["recall"]
    # Threshold-free numbers must NOT move. That is what makes them the ones
    # a reader who disagrees with the cutoff can still use.
    assert low["auroc"] == high["auroc"]
    assert low["auprc"] == high["auprc"]
    assert low["brier"] == high["brier"]


def test_the_classifier_comparison_drops_the_settings_from_its_delta():
    labels = [1, 1, 0, 0, 1, 0]
    good = [0.9, 0.8, 0.1, 0.2, 0.95, 0.05]
    poor = [0.5, 0.4, 0.6, 0.55, 0.45, 0.5]

    out = scorecard.compare_classifier_against_baseline(labels, good, poor)
    assert out["delta"]["auroc"] > 0
    for setting in ("threshold", "n", "n_positive", "prevalence"):
        assert setting not in out["delta"], (
            f"{setting} is a property of the DATA, not a score; a difference "
            f"in it means the two were scored on different sets")


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="meaningless"):
        scorecard.score_classifier([1, 0, 1], [0.5, 0.5])


# ---------------------------------------------------------------------------
# A held-out SET, not a field
# ---------------------------------------------------------------------------


def _pair(lo, hi, size=80):
    return _square(size=size, box=(lo, hi))


def test_a_set_pools_its_counts_instead_of_averaging_its_ratios():
    """The reason pooling is not a detail.

    One field with a single object and one with many are not equal evidence.
    A mean of per-field precisions treats them as though they were, so a
    model that fails on the sparse field is punished as hard as one that
    fails on the crowded one.
    """
    crowded_truth = np.zeros((90, 90), dtype=int)
    for index, start in enumerate(range(5, 85, 20), start=1):
        crowded_truth[start:start + 15, 5:20] = index          # 4 objects
    sparse_truth = _pair(10, 40)                                # 1 object

    # Perfect on the crowded field, wrong on the sparse one.
    scored = scorecard.score_holdout(
        [(crowded_truth, crowded_truth.copy()),
         (sparse_truth, np.zeros_like(sparse_truth))],
        name="demo", version="v1")

    # Pooled recall is 4 of 5, not the mean of 1.0 and 0.0.
    assert scored.metrics["recall"] == pytest.approx(4 / 5)
    assert scored.metrics["n_truth"] == 5
    assert scored.n_fields == 2


def test_a_set_carries_its_name_and_version_into_every_row():
    """Two numbers are comparable only when the set and version match."""
    truth = [_pair(10, 40), _pair(20, 60)]
    good = scorecard.score_holdout([(t, t.copy()) for t in truth],
                                   name="toxo_pv", version="2026-09-03")
    poor = scorecard.score_holdout(
        [(truth[0], _pair(15, 35)), (truth[1], _pair(30, 50))],
        name="toxo_pv", version="2026-09-03")

    rows = scorecard.scorecard_rows(good, poor)
    assert rows and all(r["holdout"] == "toxo_pv" for r in rows)
    assert all(r["holdout_version"] == "2026-09-03" for r in rows)
    assert all(r["n_fields"] == 2 for r in rows)


def test_two_models_scored_on_different_sets_are_refused():
    """A table headed "finetuned against vanilla" whose columns came from
    different data is the mistake that cannot be seen by reading it."""
    truth = _pair(10, 40)
    a = scorecard.score_holdout([(truth, truth.copy())],
                                name="toxo_pv", version="v1")
    b = scorecard.score_holdout([(truth, truth.copy())],
                                name="toxo_pv", version="v2")
    with pytest.raises(ValueError, match="different held-out sets"):
        scorecard.scorecard_rows(a, b)


def test_a_count_has_no_delta_because_a_difference_in_it_is_not_a_result():
    truth = [_pair(10, 40)]
    a = scorecard.score_holdout([(truth[0], truth[0].copy())],
                                name="d", version="1")
    rows = {r["metric"]: r for r in scorecard.scorecard_rows(a, a)}
    for setting in ("match_iou", "n_fields", "n_truth", "n_pred"):
        assert rows[setting]["delta"] is None, (
            f"{setting} describes the DATA; a zero in the delta column reads "
            f"as a result and invites someone to plot it")
    assert rows["f1"]["delta"] == pytest.approx(0.0)


def test_an_empty_set_is_refused_rather_than_scored_as_zero():
    with pytest.raises(ValueError, match="no fields"):
        scorecard.score_holdout([], name="d", version="1")


def test_the_csv_round_trips_through_the_stdlib_reader():
    import csv
    import io

    truth = [_pair(10, 40), _pair(20, 60)]
    good = scorecard.score_holdout([(t, t.copy()) for t in truth],
                                   name="d", version="1")
    text = scorecard.scorecard_csv(scorecard.scorecard_rows(good, good))

    back = list(csv.DictReader(io.StringIO(text)))
    assert back and {"metric", "finetuned", "vanilla", "delta", "holdout",
                     "holdout_version"} <= set(back[0])
    assert any(row["metric"] == "f1" for row in back)
