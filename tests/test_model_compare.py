"""Tests for :mod:`spacr.model_compare`, the A/B segmentation harness.

Everything here is synthetic label arrays. No Cellpose model is loaded, nothing
is downloaded, no GPU is touched: the metric layer takes label images, and the
orchestration layer takes the segmentation call as an argument, so a stub
covers it. That is a property of the design rather than a concession to the
test runner — the Model Zoo needs the same seam.

The file is organised around the four ways this comparison could ship a
meaningless number, each of which gets a test that fails loudly if the
implementation regresses to the naive version:

* :func:`test_background_would_hide_a_total_disagreement` computes the ARI both
  ways on the same pair of masks and shows the naive one scoring 0.9997 on a
  field where the foreground agreement is exactly zero.
* :func:`test_greedy_matching_gets_this_configuration_wrong` builds the IoU
  matrix that defeats greedy assignment, runs both, and asserts the optimal
  answer.
* :func:`test_a_split_is_not_twenty_new_cells` and its merge mirror show the
  object-count delta being attributed to fragmentation rather than discovery.
* :func:`test_two_configurations_that_cannot_differ_say_so` covers the Cellpose
  4 arguments that are accepted and then ignored.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from spacr.model_compare import (
    DEFAULT_IOU_THRESHOLD,
    IGNORED_ARGUMENTS,
    LEGACY_MODEL_NAMES,
    ComparisonReport,
    ModelConfig,
    SegComparison,
    adjusted_rand_index,
    compare_configs,
    compare_masks,
    compare_models,
    format_comparison,
    load_fields,
    match_objects,
    object_overlap,
    segment_with_cellpose,
)

from tests.cellpose_api_contract import (
    DEPRECATED_EVAL_ARGUMENTS,
    MISSING_CHANNEL_AXIS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# mask builders — every geometry below is exact, so the expectations are too
# ---------------------------------------------------------------------------

def blank(height: int = 100, width: int = 100) -> np.ndarray:
    return np.zeros((height, width), dtype=np.int32)


def with_squares(*boxes, shape=(100, 100)) -> np.ndarray:
    """``with_squares((y, x, size), ...)`` — one label per box, in order."""
    mask = blank(*shape)
    for label, (y, x, size) in enumerate(boxes, start=1):
        mask[y:y + size, x:x + size] = label
    return mask


def two_objects() -> np.ndarray:
    """Two well-separated 10x10 objects on a 100x100 field."""
    return with_squares((5, 5, 10), (60, 60, 10))


def naive_ari_including_background(mask_a, mask_b) -> float:
    """The ARI everybody writes first: sklearn over the raw flattened labels.

    Kept in the tests, never in the module, as the thing to be beaten.
    """
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(mask_a).ravel(),
                                     np.asarray(mask_b).ravel()))


def greedy_matches(iou, threshold):
    """Descending-IoU greedy assignment — the wrong answer, for contrast."""
    order = sorted(((iou[i, j], i, j)
                    for i in range(iou.shape[0])
                    for j in range(iou.shape[1])), reverse=True)
    used_a, used_b, out = set(), set(), []
    for value, i, j in order:
        if value >= threshold and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j, value))
    return out


def per_row_argmax(iou, threshold):
    """Each A object takes its best B object — the assignment that double-books."""
    return [(i, int(iou[i].argmax()), float(iou[i].max()))
            for i in range(iou.shape[0]) if iou[i].max() >= threshold]


# ---------------------------------------------------------------------------
# the metric that is easiest to get wrong: ARI over background
# ---------------------------------------------------------------------------

def test_background_would_hide_a_total_disagreement():
    """The single easiest way to ship a meaningless number.

    Two 10x10 objects on a 300x300 field. Model A calls them two objects, model
    B fuses them into one — total disagreement about every object in the field.
    Include the background and 89 800 agreed pixels drown it: 0.9997, which
    reads as "these models are identical". Exclude it and the answer is 0.0.
    """
    mask_a = with_squares((10, 10, 10), (10, 40, 10), shape=(300, 300))
    mask_b = mask_a.copy()
    mask_b[mask_b == 2] = 1                    # B fuses the pair into one label

    naive = naive_ari_including_background(mask_a, mask_b)
    assert naive > 0.99, naive

    ours = adjusted_rand_index(mask_a, mask_b)
    assert ours == pytest.approx(0.0, abs=1e-9), (
        f"background is still in the index: {ours} (naive would be {naive})")
    assert compare_masks(mask_a, mask_b).ari == pytest.approx(ours)


def test_background_as_one_cluster_would_also_hide_a_missed_object():
    """The second half of the background rule, and the less obvious one.

    Excluding the agreed background is not enough: if the pixels the *other*
    model left unassigned are pooled into one background cluster, then A's
    {object, object} and B's {object, background-blob} are the same partition
    and a model that missed an entire object still scores a perfect 1.0. Held
    as unclustered singletons instead, missing one of two objects scores 0.5.
    """
    mask_a = two_objects()
    mask_b = mask_a.copy()
    mask_b[mask_b == 2] = 0                    # B never finds the second object

    from sklearn.metrics import adjusted_rand_score

    foreground = (mask_a > 0) | (mask_b > 0)
    pooled = adjusted_rand_score(mask_a[foreground], mask_b[foreground])
    assert pooled == pytest.approx(1.0), pooled       # the trap

    assert adjusted_rand_index(mask_a, mask_b) == pytest.approx(0.5, abs=0.01)


def test_the_closed_form_agrees_with_sklearn_on_the_expanded_arrays():
    """The index is computed from the overlap table; prove it is the real ARI.

    sklearn is given the same partition written out the slow way — union
    foreground, unassigned pixels expanded into one singleton label each.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(7)
    cases = [
        (two_objects(), two_objects()),
        (two_objects(), with_squares((5, 5, 10))),
        (with_squares((20, 20, 30)), with_squares((20, 20, 18), (38, 20, 12))),
        (rng.integers(0, 4, (60, 60)).astype(np.int32),
         rng.integers(0, 5, (60, 60)).astype(np.int32)),
    ]
    for mask_a, mask_b in cases:
        foreground = (mask_a > 0) | (mask_b > 0)

        def expand(mask):
            values = mask[foreground].astype(np.int64).copy()
            zeros = np.flatnonzero(values == 0)
            values[zeros] = values.max() + 1 + np.arange(zeros.size)
            return values

        assert adjusted_rand_index(mask_a, mask_b) == pytest.approx(
            adjusted_rand_score(expand(mask_a), expand(mask_b)), abs=1e-9)


# ---------------------------------------------------------------------------
# the object assignment
# ---------------------------------------------------------------------------

def test_identical_masks_agree_completely():
    mask = two_objects()
    row = compare_masks(mask, mask.copy())

    assert row.ari == pytest.approx(1.0)
    assert (row.n_objects_a, row.n_objects_b, row.n_matched) == (2, 2, 2)
    assert row.iou_matched_fraction == pytest.approx(1.0)
    assert row.mean_matched_iou == pytest.approx(1.0)
    assert (row.unmatched_a, row.unmatched_b) == (0, 0)
    assert (row.split_events, row.merge_events) == (0, 0)
    assert (row.new_objects_b, row.missing_objects_a) == (0, 0)
    assert row.object_count_delta == 0
    assert "ARI 1.000" in row.note


def test_greedy_matching_gets_this_configuration_wrong():
    """Object matching is bipartite; greedy and per-row argmax both fail here.

    Two A objects, two B objects, laid out so the IoU matrix is::

              b1     b2
        a1  0.484  0.286
        a2  0.333  0.000

    Per-row argmax sends both A objects to b1 — b1 is double-booked and b2 is
    never used. Greedy in descending order takes (a1, b1) at 0.484 and then has
    nothing left for a2. The optimal assignment takes (a1, b2) + (a2, b1) for a
    total of 0.619 and pairs everything up. Only the last answer is right.

    Note the threshold: at IoU >= 0.5 two objects cannot both overlap a third by
    more than half of it, so the assignment is unique and greedy would get away
    with it. Every threshold below 0.5 — which is where "roughly the same
    object" lives — is where this matters.
    """
    mask_a = blank(72, 100)
    mask_a[0:42] = 1
    mask_a[42:72] = 2
    mask_b = blank(72, 100)
    mask_b[12:62] = 1
    mask_b[0:12] = 2

    result = match_objects(mask_a, mask_b, iou_threshold=0.25)
    iou = result['iou']
    assert iou[0, 0] == pytest.approx(0.4839, abs=1e-3)
    assert iou[0, 1] == pytest.approx(0.2857, abs=1e-3)
    assert iou[1, 0] == pytest.approx(0.3333, abs=1e-3)
    assert iou[1, 1] == pytest.approx(0.0)

    # The two wrong answers, computed here so the contrast is on the record.
    assert per_row_argmax(iou, 0.25) == [(0, 0, pytest.approx(0.4839, abs=1e-3)),
                                         (1, 0, pytest.approx(0.3333, abs=1e-3))]
    assert len(greedy_matches(iou, 0.25)) == 1

    assert len(result['matches']) == 2
    assert sorted((a, b) for a, b, _ in result['matches']) == [(1, 2), (2, 1)]
    assert result['unmatched_a'] == [] and result['unmatched_b'] == []

    row = compare_masks(mask_a, mask_b, iou_threshold=0.25)
    assert row.n_matched == 2
    assert (row.unmatched_a, row.unmatched_b) == (0, 0)
    # And nothing here is a split: b1 straddles both A objects but is *paired*
    # with a2, so it is a2's counterpart, not a fragment of a1.
    assert (row.split_events, row.merge_events) == (0, 0)


def test_the_default_threshold_makes_the_assignment_unique():
    """At IoU >= 0.5 greedy and optimal necessarily agree — asserted, not assumed."""
    mask_a = blank(72, 100)
    mask_a[0:42] = 1
    mask_a[42:72] = 2
    mask_b = blank(72, 100)
    mask_b[0:40] = 1
    mask_b[40:72] = 2

    result = match_objects(mask_a, mask_b, iou_threshold=DEFAULT_IOU_THRESHOLD)
    greedy = greedy_matches(result['iou'], DEFAULT_IOU_THRESHOLD)
    assert sorted((i + 1, j + 1) for i, j, _ in greedy) == \
        sorted((a, b) for a, b, _ in result['matches'])


# ---------------------------------------------------------------------------
# splits and merges: fragmentation is not discovery
# ---------------------------------------------------------------------------

def test_a_split_is_not_twenty_new_cells():
    """One A object, two B objects. That is a split, and B found nothing new.

    The 60/40 geometry is deliberate: the larger piece reaches IoU 0.6 and pairs
    with the parent, the smaller one is left over at 0.4. A harness that counted
    leftovers would call that leftover a new object.
    """
    mask_a = with_squares((20, 20, 50))
    mask_b = blank()
    mask_b[20:50, 20:70] = 1
    mask_b[50:70, 20:70] = 2

    row = compare_masks(mask_a, mask_b)
    assert (row.n_objects_a, row.n_objects_b) == (1, 2)
    assert row.object_count_delta == +1
    assert row.split_events == 1
    assert row.fragments_from_splits == 1
    assert row.merge_events == 0
    assert row.new_objects_b == 0, "a fragment was reported as a new object"
    assert row.missing_objects_a == 0
    assert row.unmatched_b == 1              # the leftover piece is still leftover
    assert "split" in row.note


def test_a_merge_is_not_twenty_missing_cells():
    """The mirror image: two A objects, one B object covering both."""
    mask_a = blank()
    mask_a[20:50, 20:70] = 1
    mask_a[50:70, 20:70] = 2
    mask_b = with_squares((20, 20, 50))

    row = compare_masks(mask_a, mask_b)
    assert (row.n_objects_a, row.n_objects_b) == (2, 1)
    assert row.object_count_delta == -1
    assert row.merge_events == 1
    assert row.merged_away == 1
    assert row.split_events == 0
    assert row.missing_objects_a == 0, "a merged object was reported as missing"
    assert row.new_objects_b == 0
    assert "fused" in row.note


def test_a_three_way_split_accounts_for_every_piece():
    """No piece reaches the IoU threshold, so nothing matches — and yet the
    parent is not "missing" and none of its three pieces is "new".

    This is the case that the naive count-based attribution gets wrong: with no
    match at all, the parent looks unmatched in A and one of its pieces looks
    unmatched in B, so a subtraction reports one lost object *and* one invented
    one on a field where a single object was shattered.
    """
    mask_a = with_squares((20, 20, 50))
    mask_b = blank()
    mask_b[20:37, 20:70] = 1
    mask_b[37:54, 20:70] = 2
    mask_b[54:70, 20:70] = 3

    row = compare_masks(mask_a, mask_b)
    assert row.n_matched == 0
    assert row.split_events == 1
    assert row.fragments_from_splits == 2       # three pieces = two extra objects
    assert (row.new_objects_b, row.missing_objects_a) == (0, 0)
    assert row.object_count_delta == +2


def test_a_boundary_shift_is_not_a_split():
    """Two objects whose shared boundary moved must not read as fragmentation.

    Without the "not assigned elsewhere" clause, B's second object — which
    straddles both A objects — would count as a fragment of the first, and
    every ordinary boundary disagreement in the dataset would be reported as a
    Cellpose failure mode.
    """
    mask_a = blank()
    mask_a[0:50] = 1
    mask_a[50:100] = 2
    mask_b = blank()
    mask_b[0:45] = 1
    mask_b[45:100] = 2

    row = compare_masks(mask_a, mask_b)
    assert row.n_matched == 2
    assert (row.split_events, row.merge_events) == (0, 0)
    assert (row.new_objects_b, row.missing_objects_a) == (0, 0)


# ---------------------------------------------------------------------------
# the degenerate fields
# ---------------------------------------------------------------------------

def test_two_empty_masks_are_defined_to_agree():
    """Both models said "nothing here", which is the same statement.

    The alternative (nan) drops the field out of every aggregate, so a channel
    that is legitimately empty would silently shrink the sample instead of
    showing up as the unanimous verdict it is. ``mean_matched_iou`` stays nan
    because there is no matched pair to take an IoU of, and the field is
    counted separately so it can never be mistaken for agreement about objects.
    """
    row = compare_masks(blank(), blank())
    assert row.both_empty
    assert row.ari == pytest.approx(1.0)
    assert row.iou_matched_fraction == pytest.approx(1.0)
    assert np.isnan(row.mean_matched_iou)
    assert (row.n_objects_a, row.n_objects_b) == (0, 0)
    assert (row.split_events, row.merge_events) == (0, 0)
    assert "nothing in this field" in row.note


def test_one_empty_mask_agrees_about_nothing():
    row = compare_masks(two_objects(), blank())
    assert row.ari == pytest.approx(0.0)
    assert row.iou_matched_fraction == pytest.approx(0.0)
    assert np.isnan(row.mean_matched_iou)
    assert (row.unmatched_a, row.unmatched_b) == (2, 0)
    assert row.missing_objects_a == 2
    assert row.new_objects_b == 0
    assert not row.both_empty

    mirror = compare_masks(blank(), two_objects())
    assert mirror.new_objects_b == 2 and mirror.missing_objects_a == 0


def test_a_single_object_each_matches_but_carries_no_pair_information():
    """One object per mask is matched normally; the ARI is degenerate and says so.

    A partition with one cluster has no pair structure to agree about, so two
    masks that overlap 68 % score a *negative* ARI. That is a real property of
    the index and the reason the object-level columns sit next to it in every
    report rather than under it.
    """
    mask_a = with_squares((20, 20, 20))
    mask_b = blank()
    mask_b[22:42, 22:42] = 1

    row = compare_masks(mask_a, mask_b)
    assert row.n_matched == 1
    assert row.mean_matched_iou == pytest.approx(0.681, abs=0.01)
    assert row.iou_matched_fraction == pytest.approx(1.0)
    assert row.ari < 0.0

    identical = compare_masks(mask_a, mask_a.copy())
    assert identical.ari == pytest.approx(1.0)


def test_completely_disjoint_masks_score_about_zero():
    mask_a = with_squares((5, 5, 10), (5, 25, 10))
    mask_b = with_squares((50, 50, 10), (50, 70, 10))

    row = compare_masks(mask_a, mask_b)
    assert abs(row.ari) < 0.2, row.ari
    assert row.n_matched == 0
    assert row.iou_matched_fraction == pytest.approx(0.0)
    assert (row.unmatched_a, row.unmatched_b) == (2, 2)
    assert (row.new_objects_b, row.missing_objects_a) == (2, 2)
    assert (row.split_events, row.merge_events) == (0, 0)


def test_labels_need_not_be_consecutive_and_masks_may_be_bool_or_float():
    """Real masks arrive with gaps in the labels and as floats after a resize."""
    mask_a = blank()
    mask_a[5:15, 5:15] = 7
    mask_a[60:70, 60:70] = 900
    row = compare_masks(mask_a, mask_a.astype(np.float32))
    assert row.n_objects_a == row.n_objects_b == 2
    assert row.ari == pytest.approx(1.0)

    boolean = (mask_a > 0)
    assert compare_masks(boolean, boolean).n_objects_a == 2


def test_masks_of_different_shapes_are_a_caller_error():
    with pytest.raises(ValueError, match="same field"):
        compare_masks(blank(50, 50), blank(60, 60))


def test_the_overlap_table_carries_the_original_labels():
    mask_a = blank()
    mask_a[0:10, 0:10] = 4
    mask_b = blank()
    mask_b[0:10, 0:10] = 9
    parts = object_overlap(mask_a, mask_b)
    assert list(parts['labels_a']) == [4] and list(parts['labels_b']) == [9]
    assert parts['overlap'].tolist() == [[100]]
    assert parts['union_foreground'] == 100


# ---------------------------------------------------------------------------
# the configurations, and the arguments Cellpose 4 quietly drops
# ---------------------------------------------------------------------------

def test_legacy_model_names_stay_in_step_with_utils():
    """This module keeps its own copy so it can stay free of torch; prove it matches."""
    from spacr.utils import LEGACY_CELLPOSE_MODELS

    assert set(LEGACY_MODEL_NAMES) == set(LEGACY_CELLPOSE_MODELS)


@pytest.mark.parametrize("legacy", ["cyto", "cyto2", "cyto3", "nuclei"])
def test_every_pre_sam_model_name_resolves_to_cpsam(legacy):
    config = ModelConfig(model=legacy)
    assert config.resolved_model == "cpsam"
    assert config.model_was_remapped
    assert config.ignored_parameters()["model"] == legacy
    assert any("predates Cellpose-SAM" in note for note in config.notes())


def test_a_custom_checkpoint_path_is_left_alone():
    config = ModelConfig(model="/models/my_cells.pth")
    assert config.resolved_model == "/models/my_cells.pth"
    assert not config.model_was_remapped
    assert config.name == "my_cells.pth"


def test_two_configurations_that_cannot_differ_say_so():
    """cyto2 versus cyto3 is one model against itself, and must not read as a
    comparison that found no difference."""
    diff = compare_configs(ModelConfig(name="A", model="cyto2"),
                           ModelConfig(name="B", model="cyto3"))
    assert diff['identical'] is True
    assert not diff['honoured']
    assert "cyto2" in str(diff['ignored'])
    assert any("same model with the same settings" in w for w in diff['warnings'])


def test_an_ignored_argument_is_reported_and_never_forwarded():
    """diam_mean is the trap: it looks like the size knob and does nothing."""
    config = ModelConfig(name="B", diameter=30.0, extra={"diam_mean": 17})
    assert config.ignored_parameters() == {"diam_mean": 17}
    assert "diam_mean" not in config.eval_kwargs()
    assert config.eval_kwargs()["diameter"] == 30.0
    note = "\n".join(config.notes())
    assert "diam_mean=17 is ignored" in note
    assert "Use diameter=" in note


@pytest.mark.parametrize("argument", sorted(IGNORED_ARGUMENTS))
def test_no_ignored_argument_ever_reaches_eval(argument):
    config = ModelConfig(extra={argument: "whatever"})
    assert argument not in config.eval_kwargs()
    assert config.ignored_parameters()[argument] == "whatever"


def test_diameter_is_not_in_the_ignored_list_because_cellpose_4_honours_it():
    """eval() still rescales by 30/diameter; it is the one size knob that works."""
    assert "diameter" not in IGNORED_ARGUMENTS
    assert ModelConfig(diameter=60.0).eval_kwargs()["diameter"] == 60.0
    assert ModelConfig(diameter=None).eval_kwargs()["diameter"] is None


def test_an_unrecognised_extra_argument_is_passed_on_with_a_warning():
    config = ModelConfig(extra={"wibble": 3})
    assert config.eval_kwargs()["wibble"] == 3
    assert any("not a Cellpose 4 eval argument" in n for n in config.notes())


def test_a_config_can_be_built_from_a_plain_dict():
    config = ModelConfig.from_mapping(
        {"name": "B", "diameter": 45, "diam_mean": 17, "extra": {"rescale": 1.0}})
    assert config.name == "B" and config.diameter == 45
    assert config.extra == {"rescale": 1.0, "diam_mean": 17}
    assert ModelConfig.from_mapping(config) is config
    with pytest.raises(TypeError, match="ModelConfig or a mapping"):
        ModelConfig.from_mapping(["cpsam"])


def test_honoured_parameters_are_what_the_report_compares():
    a = ModelConfig(name="A", diameter=30.0, extra={"diam_mean": 17})
    b = ModelConfig(name="B", diameter=60.0)
    diff = compare_configs(a, b)
    assert diff['identical'] is False
    assert diff['honoured'] == {"diameter": (30.0, 60.0)}
    assert diff['ignored'] == {"diam_mean": (17, None)}


# ---------------------------------------------------------------------------
# orchestration, with the segmentation mocked
# ---------------------------------------------------------------------------

class FakeSegmenter:
    """A stand-in for Cellpose that records what it was asked to do."""

    def __init__(self, masks_by_name):
        self.masks_by_name = masks_by_name
        self.calls = []

    def __call__(self, images, config):
        self.calls.append((config.name, len(images), config.eval_kwargs()))
        masks = self.masks_by_name[config.name]
        return [masks[i] for i in range(len(images))]


def three_fields():
    """Three identical images; A finds one object per field, B splits one of them."""
    images = [np.zeros((100, 100), dtype=np.float32) for _ in range(3)]
    a_masks = [with_squares((20, 20, 50)) for _ in range(3)]
    b_masks = []
    for _ in range(3):
        mask = blank()
        mask[20:50, 20:70] = 1
        mask[50:70, 20:70] = 2
        b_masks.append(mask)
    return images, a_masks, b_masks


def test_compare_models_runs_each_model_over_every_field():
    images, a_masks, b_masks = three_fields()
    segmenter = FakeSegmenter({"A": a_masks, "B": b_masks})

    report = compare_models(images,
                            {"name": "A", "diameter": 30},
                            {"name": "B", "diameter": 60},
                            field_names=["f1", "f2", "f3"],
                            segment_fn=segmenter)

    assert [name for name, _, _ in segmenter.calls] == ["A", "B"]
    assert [n for _, n, _ in segmenter.calls] == [3, 3]
    assert segmenter.calls[0][2]["diameter"] == 30
    assert segmenter.calls[1][2]["diameter"] == 60

    assert report.n_fields == 3
    assert report.fields == ["f1", "f2", "f3"]
    assert (report.total_objects_a, report.total_objects_b) == (3, 6)
    assert report.object_count_delta == +3
    assert report.count_ratio == pytest.approx(2.0)
    assert report.total_splits == 3
    assert report.total_fragments == 3
    assert report.total_new_objects_b == 0
    assert report.total_merges == 0
    assert report.n_both_empty == 0
    assert not report.identical_masks
    assert report.mean_matched_fraction == pytest.approx(2 / 3)
    assert "B found 3 more cell(s) than A" in report.summary
    assert "fragments of A's objects" in report.summary
    assert "Neither model is ground truth" in report.summary


def test_the_report_carries_the_seg_qc_verdict_for_both_models():
    """seg_qc is reused wholesale rather than reimplemented: whether a mask is
    fused, shattered or empty is the question it already answers."""
    images = [np.zeros((100, 100), dtype=np.float32)]
    good = [with_squares(*[(5 + 20 * (i // 4), 5 + 20 * (i % 4), 8)
                           for i in range(12)])]
    empty = [blank()]
    report = compare_models(images, {"name": "A"}, {"name": "B"},
                            segment_fn=FakeSegmenter({"A": good, "B": empty}))
    row = report.comparisons[0]
    assert row.qc_a is not None and row.qc_b is not None
    assert row.qc_b.severity == "fail"
    assert "empty_field" in row.qc_b.flags
    assert row.qc_a.n_objects == 12


def test_qc_can_be_switched_off():
    images, a_masks, b_masks = three_fields()
    report = compare_models(images, {"name": "A"}, {"name": "B"}, qc=False,
                            segment_fn=FakeSegmenter({"A": a_masks, "B": b_masks}))
    assert all(row.qc_a is None and row.qc_b is None
               for row in report.comparisons)


def test_two_identical_models_are_flagged_before_any_number_is_read():
    images, a_masks, _ = three_fields()
    segmenter = FakeSegmenter({"cpsam (A)": a_masks, "cpsam (B)": a_masks})
    report = compare_models(images, ModelConfig(), ModelConfig(),
                            segment_fn=segmenter)

    assert report.model_a.name == "cpsam (A)"
    assert report.model_b.name == "cpsam (B)"
    assert report.config_diff['identical'] is True
    assert any("compares a model with itself" in w for w in report.warnings)
    assert report.identical_masks
    assert report.mean_ari == pytest.approx(1.0)


def test_identical_settings_that_somehow_produce_different_masks_are_called_out():
    """Same model, same knobs, different masks: that is a bug in the run, not a
    model difference, and the report has to say which."""
    images, a_masks, b_masks = three_fields()
    segmenter = FakeSegmenter({"cpsam (A)": a_masks, "cpsam (B)": b_masks})
    report = compare_models(images, ModelConfig(), ModelConfig(),
                            segment_fn=segmenter)
    assert any("cannot be a model difference" in w for w in report.warnings)


def test_progress_is_reported_for_every_stage():
    images, a_masks, b_masks = three_fields()
    seen = []
    compare_models(images, {"name": "A"}, {"name": "B"},
                   segment_fn=FakeSegmenter({"A": a_masks, "B": b_masks}),
                   progress=lambda message, done, total: seen.append(
                       (message, done, total)))
    assert [done for _, done, _ in seen] == [0, 1, 2, 3]
    assert "Segmenting 3 field(s) with A" in seen[0][0]
    assert "Segmenting 3 field(s) with B" in seen[1][0]


def test_a_model_that_returns_the_wrong_number_of_masks_is_an_error():
    images, a_masks, b_masks = three_fields()

    def short(images, config):
        return [a_masks[0]]

    with pytest.raises(ValueError, match="returned 1 mask"):
        compare_models(images, {"name": "A"}, {"name": "B"}, segment_fn=short)


def test_compare_models_needs_fields_and_matching_names():
    with pytest.raises(ValueError, match="at least one image"):
        compare_models([], {"name": "A"}, {"name": "B"}, segment_fn=lambda *_: [])
    images, a_masks, b_masks = three_fields()
    with pytest.raises(ValueError, match="field name"):
        compare_models(images, {"name": "A"}, {"name": "B"},
                       field_names=["only-one"],
                       segment_fn=FakeSegmenter({"A": a_masks, "B": b_masks}))


def test_masks_are_kept_for_display_unless_asked_otherwise():
    images, a_masks, b_masks = three_fields()
    segmenter = FakeSegmenter({"A": a_masks, "B": b_masks})
    kept = compare_models(images, {"name": "A"}, {"name": "B"},
                          segment_fn=segmenter)
    assert len(kept.masks_a) == len(kept.masks_b) == len(kept.images) == 3

    lean = compare_models(images, {"name": "A"}, {"name": "B"},
                          segment_fn=segmenter, keep_images=False)
    assert lean.masks_a == [] and lean.images == []


def test_empty_fields_are_counted_apart_from_real_agreement():
    images = [np.zeros((50, 50), dtype=np.float32) for _ in range(2)]
    report = compare_models(images, {"name": "A"}, {"name": "B"},
                            segment_fn=FakeSegmenter({"A": [blank(50, 50)] * 2,
                                                      "B": [blank(50, 50)] * 2}))
    assert report.n_both_empty == 2
    assert report.mean_ari == pytest.approx(1.0)
    assert "empty in both models" in format_comparison(report)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_the_report_shows_what_reached_the_model_and_what_did_not():
    """The whole point of the parameter block: an ignored argument must be
    visible as a no-op rather than mysterious."""
    images, a_masks, b_masks = three_fields()
    report = compare_models(
        images,
        {"name": "A", "diameter": 30},
        {"name": "B", "diameter": 30, "diam_mean": 17, "model": "cyto3"},
        segment_fn=FakeSegmenter({"A": a_masks, "B": b_masks}))

    text = format_comparison(report)
    assert "parameters that reached the model" in text
    assert "set but ignored by Cellpose 4" in text
    assert "diam_mean" in text
    assert "differ only in arguments Cellpose 4 ignores" in text
    # every field name and the aggregate are in the card
    for name in report.fields:
        assert name in text
    assert "Neither model is ground truth" in text
    assert "segmentation time" in text


def test_a_report_with_no_field_still_renders():
    report = ComparisonReport(model_a=ModelConfig(name="A"),
                              model_b=ModelConfig(name="B"))
    text = format_comparison(report)
    assert "No field was compared." in text
    assert report.summary == "No field was compared."
    assert np.isnan(report.mean_ari)
    assert np.isnan(report.count_ratio)
    assert not report.identical_masks


def test_the_aggregates_ignore_the_fields_that_have_no_number():
    """A field with no matched pair has no mean IoU; it must not drag the
    aggregate to nan, and it must not be counted as a zero either."""
    report = ComparisonReport(comparisons=[
        SegComparison(field="f1", mean_matched_iou=0.8, ari=0.9,
                      iou_matched_fraction=1.0),
        SegComparison(field="f2", mean_matched_iou=float("nan"),
                      ari=float("nan"), iou_matched_fraction=float("nan")),
        SegComparison(field="f3", mean_matched_iou=0.6, ari=0.5,
                      iou_matched_fraction=0.5),
    ])
    assert report.mean_matched_iou == pytest.approx(0.7)
    assert report.mean_ari == pytest.approx(0.7)
    assert report.mean_matched_fraction == pytest.approx(0.75)

    nothing = ComparisonReport(comparisons=[SegComparison()])
    assert np.isnan(nothing.mean_matched_iou)


def test_a_metric_that_does_not_exist_renders_as_a_dash():
    """Formatting must never invent a number for something not computed."""
    from spacr.model_compare import _fmt, _value

    assert _fmt(None) == "-"
    assert _fmt("not a number") == "-"
    assert _fmt(float("nan")) == "-"
    assert _fmt(0.5, pct=True) == "50%"
    assert _value(None) == "-" and _value(30.0) == "30" and _value(True) == "True"


def test_an_index_over_fewer_than_two_pixels_is_undefined():
    """A pair index needs a pair. One foreground pixel is nan, not 1.0.

    Reporting 1.0 there would be an invented number: the two masks have not
    agreed about anything, there was simply nothing to agree about. Two pixels
    is the smallest field that carries an index, and the object-level columns
    carry the row either way.
    """
    one_pixel = blank(10, 10)
    one_pixel[0, 0] = 1
    assert np.isnan(adjusted_rand_index(one_pixel, blank(10, 10)))
    assert np.isnan(adjusted_rand_index(one_pixel, one_pixel.copy()))
    assert np.isnan(compare_masks(one_pixel, one_pixel.copy()).ari)
    # …and the objects are still matched, which is what the row is really for
    assert compare_masks(one_pixel, one_pixel.copy()).n_matched == 1

    two_pixels = blank(10, 10)
    two_pixels[0, 0] = 1
    two_pixels[5, 5] = 2
    assert adjusted_rand_index(two_pixels, two_pixels.copy()) == pytest.approx(1.0)


def test_the_dataclass_is_the_documented_shape():
    row = SegComparison()
    for name in ("field", "n_objects_a", "n_objects_b", "ari",
                 "iou_matched_fraction", "mean_matched_iou", "unmatched_a",
                 "unmatched_b", "split_events", "merge_events"):
        assert hasattr(row, name), name
    row = SegComparison(field="f", n_objects_a=2, n_objects_b=5, ari=0.4)
    assert row.object_count_delta == 3
    assert "A 2 vs B 5" in str(row)


# ---------------------------------------------------------------------------
# loading fields
# ---------------------------------------------------------------------------

def test_load_fields_reads_npy_tif_and_npz(tmp_path):
    import imageio.v2 as imageio

    folder = tmp_path / "fields"
    folder.mkdir()
    np.save(folder / "b_field.npy", np.full((10, 10), 3, np.uint16))
    imageio.imwrite(folder / "a_field.tif", np.full((10, 10), 5, np.uint16))
    (folder / "notes.txt").write_text("ignore me")

    names, images = load_fields(str(folder), n_fields=5)
    assert names == ["a_field", "b_field"]
    assert [int(im.max()) for im in images] == [5, 3]

    batch = tmp_path / "batch"
    batch.mkdir()
    np.savez(batch / "plate.npz",
             data=np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4),
             filenames=np.array(["f1", "f2", "f3"]))
    names, images = load_fields(str(batch), n_fields=2)
    assert names == ["f1", "f2"]
    assert len(images) == 2


def test_load_fields_stops_at_n_fields_and_selects_a_channel(tmp_path):
    folder = tmp_path / "fields"
    folder.mkdir()
    for i in range(6):
        stack = np.zeros((8, 8, 3), np.uint16)
        stack[..., 1] = 42
        np.save(folder / f"f{i}.npy", stack)

    names, images = load_fields(str(folder), n_fields=3, channel=1)
    assert len(names) == len(images) == 3
    assert images[0].shape == (8, 8) and int(images[0].max()) == 42

    with pytest.raises(ValueError, match="out of range"):
        load_fields(str(folder), n_fields=1, channel=9)


def test_load_fields_survives_one_unreadable_file(tmp_path):
    """A corrupt field must not cost the comparison the other two."""
    folder = tmp_path / "fields"
    folder.mkdir()
    (folder / "a_broken.npy").write_bytes(b"not a numpy file")
    np.save(folder / "b_good.npy", np.ones((6, 6), np.uint16))
    names, images = load_fields(str(folder), n_fields=3)
    assert names == ["b_good"] and len(images) == 1


def test_load_fields_reports_an_empty_or_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fields(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no readable field"):
        load_fields(str(empty))


def test_load_fields_accepts_arrays_already_in_memory():
    names, images = load_fields([np.zeros((4, 4)), np.ones((4, 4))], n_fields=5)
    assert names == ["field_0000", "field_0001"] and len(images) == 2
    with pytest.raises(ValueError, match="no field to compare"):
        load_fields([])


# ---------------------------------------------------------------------------
# the Cellpose backend, with the model stubbed out
# ---------------------------------------------------------------------------

def test_segment_with_cellpose_forwards_only_what_cellpose_4_reads(monkeypatch):
    """No weights are loaded here: CellposeModel is replaced wholesale.

    What is being checked is the argument list — diameter reaches eval, and the
    arguments Cellpose 4 would accept and drop never get there, so the report's
    "ignored" column is the truth rather than a claim.
    """
    from cellpose import models as cp_models

    seen = {}

    class StubModel:
        """Both signatures are the installed cellpose 4.0.7 ones, verbatim.

        With no ``**kwargs``, "model_compare forwards an argument cellpose 4
        does not have" stops being an assertion this test has to remember to
        make and becomes a ``TypeError`` raised by Python's own binding — which
        is what ``diam_mean`` and ``model`` are: neither is an ``eval``
        parameter at all, so forwarding either one now cannot pass silently.
        """

        def __init__(self, gpu=False, pretrained_model="cpsam",
                     model_type=None, diam_mean=None, device=None, nchan=None,
                     use_bfloat16=True):
            seen['init'] = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model,
                                                         model_type)

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis,
                                     require_channel_axis=False)
            seen['eval'] = eval_arguments(locals())
            seen['configured'] = configured_eval_arguments(locals())
            seen['n_images'] = len(x)
            return ([np.zeros((4, 4), np.int32) for _ in x], None, None)

    monkeypatch.setattr(cp_models, "CellposeModel", StubModel)

    config = ModelConfig(model="cyto2", diameter=45.0,
                         extra={"diam_mean": 17, "channels": [2, 0]})
    masks = segment_with_cellpose([np.zeros((4, 4)), np.ones((4, 4))], config)

    assert len(masks) == 2 and masks[0].dtype == np.int32
    assert seen['init']['pretrained_model'] == "cpsam"     # cyto2 was remapped
    assert seen['init']['model_type'] is None, (
        "model_type= is accepted-and-dropped by cellpose 4; selecting weights "
        "through it silently loads cpsam"
    )
    assert seen['eval']['diameter'] == 45.0
    # diam_mean and model are not eval parameters at all, so forwarding either
    # would already have raised TypeError above. The dropped-but-accepted ones
    # need an assertion: they must be left at cellpose's own default.
    assert "diam_mean" not in seen['eval']
    assert "model" not in seen['eval']
    assert seen['eval']['channels'] is None
    assert not set(seen['configured']) & set(DEPRECATED_EVAL_ARGUMENTS), (
        "model_compare forwarded an argument cellpose 4 accepts and discards"
    )
    assert seen['n_images'] == 2


# ---------------------------------------------------------------------------
# the cost guarantee: the metric layer carries no model stack
# ---------------------------------------------------------------------------

def test_comparing_masks_does_not_pull_in_torch_or_cellpose():
    """In-process guard: the metric layer must not *add* the model stack."""
    before = {m.split(".")[0] for m in list(sys.modules)}
    compare_masks(two_objects(), two_objects())
    after = {m.split(".")[0] for m in list(sys.modules)}
    assert not (after - before) & {"torch", "torchvision", "cellpose", "tensorflow"}


def test_the_module_imports_without_torch_or_cellpose(tmp_path):
    """The real guarantee, checked in a fresh interpreter.

    In-process this cannot be proven — the pytest session has already imported
    half of spaCR, and the coverage runner pre-imports torch through a
    sitecustomize shim. So a clean interpreter is started with PYTHONPATH set to
    the repo alone and asked what it ended up with. It matters because the Model
    Zoo will call the metric layer in a loop over a checkpoint library, and
    because it is what keeps this module reusable outside a Cellpose run.
    """
    code = textwrap.dedent(
        """
        import sys
        import numpy as np
        from spacr.model_compare import compare_masks, format_comparison

        a = np.zeros((40, 40), np.int32); a[5:15, 5:15] = 1
        b = np.zeros((40, 40), np.int32); b[5:15, 5:15] = 1; b[20:30, 20:30] = 2
        row = compare_masks(a, b)
        assert row.n_objects_b == 2 and row.new_objects_b == 1, row

        heavy = sorted({m.split(".")[0] for m in sys.modules}
                       & {"torch", "torchvision", "cellpose", "tensorflow"})
        print("HEAVY:" + ",".join(heavy))
        """
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.run([sys.executable, "-c", code],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("HEAVY:"))
    assert line == "HEAVY:", f"heavy modules imported: {line}"
