"""The step 2 baseline of item 426: the IoU stitcher and the TRA/DET it scores.

The plan: "Build the plain stitcher -- segment each frame, link by mask
overlap or Hungarian matching on IoU -- and the tracking metrics: TRA/DET from
the Cell Tracking Challenge, plus a count of identity switches. This number is
what the fork has to beat."

A metric nobody has checked against a hand-computed case is worse than no
metric, because the fork will be measured against it and the temptation will
be to believe a number that flatters the third head. So every score here is
computed twice: once by the module and once in the test from the AOGM
definition, with the weights spelled out. The controls the plan asks for in
step 6 are the same controls that keep a metric honest here -- a perfect
tracking scores 1, an empty one scores 0, and each single defect moves exactly
the term it should.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import timeflows_baseline as tb


SIZE = 8
W = tb.AOGM_WEIGHTS


def _square(frame, label, top, left, size=SIZE):
    frame[top:top + size, left:left + size] = label


def moving_stack(n_frames=6):
    """Three objects with consistent labels: two drifting, one still."""
    stack = np.zeros((n_frames, 48, 48), dtype=np.int32)
    for t in range(n_frames):
        _square(stack[t], 1, 2 + 2 * t, 2)
        _square(stack[t], 2, 20, 4 + 3 * t)
        _square(stack[t], 3, 34, 30)
    return stack


def per_frame_segmentation(stack, offset=10):
    """The same objects with labels that mean nothing across frames."""
    out = np.zeros_like(stack)
    for t in range(stack.shape[0]):
        for index, label in enumerate(np.unique(stack[t])[1:]):
            out[t][stack[t] == label] = offset + (index + t) % 3 + 1
    return out


# ---------------------------------------------------------------------------
# the overlap arithmetic
# ---------------------------------------------------------------------------

def test_overlap_counts_cross_tabulates_shared_pixels():
    previous = np.zeros((8, 8), dtype=np.int32)
    current = np.zeros((8, 8), dtype=np.int32)
    previous[0:4, 0:4] = 1
    previous[4:8, 4:8] = 2
    current[2:6, 0:4] = 5
    labels_previous, labels_current, counts = tb.overlap_counts(
        previous, current)
    assert list(labels_previous) == [1, 2]
    assert list(labels_current) == [5]
    assert counts.tolist() == [[8.0], [0.0]]


def test_overlap_counts_of_an_empty_frame_is_empty():
    empty = np.zeros((4, 4), dtype=np.int32)
    labels_previous, labels_current, counts = tb.overlap_counts(
        empty, np.ones((4, 4), dtype=np.int32))
    assert labels_previous.size == 0
    assert counts.shape == (0, 1)


def test_frames_that_share_no_pixel_have_no_counts():
    previous = np.zeros((8, 8), dtype=np.int32)
    current = np.zeros((8, 8), dtype=np.int32)
    previous[0:2, 0:2] = 1
    current[6:8, 6:8] = 1
    _, _, counts = tb.overlap_counts(previous, current)
    assert counts.sum() == 0.0


def test_iou_is_the_intersection_over_the_union():
    previous = np.zeros((8, 8), dtype=np.int32)
    current = np.zeros((8, 8), dtype=np.int32)
    previous[0:4, 0:4] = 1
    current[2:6, 0:4] = 1
    _, _, iou = tb.iou_matrix(previous, current)
    assert iou[0, 0] == pytest.approx(8.0 / 24.0)


def test_link_frames_matches_the_same_object_and_drops_weak_overlap():
    previous = np.zeros((16, 16), dtype=np.int32)
    current = np.zeros((16, 16), dtype=np.int32)
    _square(previous, 1, 0, 0, size=6)
    _square(current, 4, 1, 0, size=6)
    _square(previous, 2, 10, 10, size=4)
    _square(current, 5, 10, 10, size=1)
    matches = tb.link_frames(previous, current, iou_threshold=0.1)
    assert [(a, b) for a, b, _ in matches] == [(1, 4)]


def test_link_frames_takes_the_best_total_rather_than_the_best_pair(
        monkeypatch):
    """Hungarian, not greedy: the pair with the highest IoU is not always kept.

    With these four overlaps greedy takes 1->b at 0.9 and then has nothing
    above the threshold for 2, losing a link. The assignment that maximises
    the total keeps both. This is the case the plan's "Hungarian matching on
    IoU" is for, and it is worth a test because the greedy version passes
    every simple fixture.
    """
    labels_previous = np.array([1, 2])
    labels_current = np.array([10, 11])
    crafted = np.array([[0.6, 0.9], [0.0, 0.8]])
    monkeypatch.setattr(
        tb, 'iou_matrix',
        lambda previous, current: (labels_previous, labels_current, crafted))
    matches = tb.link_frames(np.zeros((2, 2)), np.zeros((2, 2)),
                             iou_threshold=0.1)
    assert [(a, b) for a, b, _ in matches] == [(1, 10), (2, 11)]


def test_linking_an_empty_frame_links_nothing():
    empty = np.zeros((8, 8), dtype=np.int32)
    assert tb.link_frames(empty, empty) == []


def test_link_frames_agrees_with_the_pipeline_linker():
    """The stitcher's arithmetic is the one `spacr.timelapse` already uses.

    The module docstring says the two are meant to agree on which labels match
    at a given threshold, and differ only in cost. This is that claim.
    """
    timelapse = pytest.importorskip('spacr.timelapse')
    previous = np.zeros((24, 24), dtype=np.int32)
    current = np.zeros((24, 24), dtype=np.int32)
    _square(previous, 1, 0, 0)
    _square(current, 1, 2, 0)
    _square(previous, 2, 12, 12)
    _square(current, 2, 12, 14)
    mine = [(a, b) for a, b, _ in
            tb.link_frames(previous, current, iou_threshold=0.1)]
    theirs = [(int(a), int(b)) for a, b in
              timelapse.link_by_iou(previous, current, iou_threshold=0.1)]
    assert mine == sorted(theirs)


# ---------------------------------------------------------------------------
# the stitcher
# ---------------------------------------------------------------------------

def test_the_stitcher_gives_one_id_to_one_object_through_the_movie():
    truth = moving_stack()
    tracked, table = tb.stitch_by_iou(per_frame_segmentation(truth))
    for label in (1, 2, 3):
        ids = {int(tracked[t][truth[t] == label][0])
               for t in range(truth.shape[0])}
        assert len(ids) == 1
    assert len(table) == truth.shape[0] * 3
    assert table['iou'].isna().sum() == 3


def test_an_object_that_appears_later_starts_a_new_track():
    stack = np.zeros((3, 32, 32), dtype=np.int32)
    for t in range(3):
        _square(stack[t], 1, 0, 0)
    _square(stack[2], 2, 20, 20)
    tracked, table = tb.stitch_by_iou(stack)
    assert sorted(np.unique(tracked))[1:] == [1, 2]
    assert int(table['track_id'].max()) == 2


def test_an_object_that_moves_too_far_is_not_linked_to_itself():
    stack = np.zeros((2, 64, 64), dtype=np.int32)
    _square(stack[0], 1, 0, 0)
    _square(stack[1], 1, 40, 40)
    tracked, _ = tb.stitch_by_iou(stack)
    assert int(tracked[0].max()) != int(tracked[1].max())


def test_the_stitcher_cannot_express_a_division_and_does_not_pretend_to():
    stack = np.zeros((2, 32, 32), dtype=np.int32)
    stack[0][10:26, 10:26] = 1
    stack[1][10:26, 10:17] = 1
    stack[1][10:26, 19:26] = 2
    tracked, _ = tb.stitch_by_iou(stack)
    children = [int(tracked[1][stack[1] == label][0]) for label in (1, 2)]
    assert len(set(children)) == 2
    assert sorted(children) == [1, 2]


def test_the_stitcher_refuses_a_volume():
    with pytest.raises(ValueError, match='2-D label frames'):
        tb.stitch_by_iou(np.zeros((2, 2, 8, 8), dtype=np.int32))


def test_stitching_nothing_returns_an_empty_stack_and_table():
    tracked, table = tb.stitch_by_iou([])
    assert tracked.size == 0
    assert list(table.columns) == [
        'frame', 'original_label', 'track_id', 'iou']


# ---------------------------------------------------------------------------
# the tracking graph
# ---------------------------------------------------------------------------

def test_the_graph_holds_one_vertex_per_object_per_frame_and_links_them():
    graph = tb.tracking_graph(moving_stack(4))
    assert len(graph['vertices']) == 12
    assert len(graph['edges']) == 9
    assert set(graph['edges'].values()) == {'track'}


def test_a_lineage_adds_a_division_edge_from_parent_to_child():
    stack = np.zeros((4, 32, 32), dtype=np.int32)
    for t in range(2):
        stack[t][10:26, 10:26] = 1
    for t in (2, 3):
        stack[t][10:26, 10:17] = 2
        stack[t][10:26, 19:26] = 3
    graph = tb.tracking_graph(stack, lineage={2: 1, 3: 1})
    assert graph['edges'][((1, 1), (2, 2))] == 'division'
    assert graph['edges'][((1, 1), (2, 3))] == 'division'


def test_a_parent_that_is_not_in_the_stack_adds_no_edge():
    stack = np.zeros((2, 16, 16), dtype=np.int32)
    stack[:, 0:4, 0:4] = 1
    graph = tb.tracking_graph(stack, lineage={1: 9})
    assert len(graph['edges']) == 1


def test_a_bridged_gap_keeps_its_vertices_and_earns_no_edge():
    """A tracker may claim an identity across a gap; the graph does not pay it.

    The challenge's graph has no edge that skips a frame. Inventing one here
    would credit a link the ground truth cannot contain, so the two vertices
    stay and the link does not -- and TRA charges for the two ground-truth
    edges that were not reproduced.
    """
    stack = np.zeros((3, 16, 16), dtype=np.int32)
    stack[0, 0:4, 0:4] = 1
    stack[2, 0:4, 0:4] = 1
    graph = tb.tracking_graph(stack)
    assert graph['vertices'] == {(0, 1), (2, 1)}
    assert graph['edges'] == {}


def test_a_child_that_does_not_follow_its_parent_gets_no_division_edge():
    stack = np.zeros((4, 16, 16), dtype=np.int32)
    stack[0, 0:4, 0:4] = 1
    stack[3, 8:12, 8:12] = 2
    graph = tb.tracking_graph(stack, lineage={2: 1})
    assert graph['edges'] == {}


# ---------------------------------------------------------------------------
# detection matching
# ---------------------------------------------------------------------------

def test_an_object_is_detected_by_the_segment_holding_most_of_it():
    truth = moving_stack(2)
    matched, shared = tb.match_vertices(truth, per_frame_segmentation(truth))
    assert len(matched) == 6
    assert set(shared.values()) == {1}


def test_half_of_an_object_is_not_a_detection():
    truth = np.zeros((1, 8, 8), dtype=np.int32)
    prediction = np.zeros((1, 8, 8), dtype=np.int32)
    truth[0, 0:4, 0:4] = 1
    prediction[0, 0:2, 0:4] = 1
    matched, _ = tb.match_vertices(truth, prediction)
    assert matched == {}


def test_one_segment_over_two_objects_detects_both_of_them():
    truth = np.zeros((1, 16, 16), dtype=np.int32)
    prediction = np.zeros((1, 16, 16), dtype=np.int32)
    truth[0, 0:4, 0:4] = 1
    truth[0, 0:4, 6:10] = 2
    prediction[0, 0:4, 0:10] = 7
    matched, shared = tb.match_vertices(truth, prediction)
    assert len(matched) == 2
    assert shared[(0, 7)] == 2


def test_two_movies_of_different_length_are_refused():
    with pytest.raises(ValueError, match='the same movie'):
        tb.match_vertices(np.zeros((2, 4, 4), dtype=np.int32),
                          np.zeros((3, 4, 4), dtype=np.int32))


def test_two_frames_of_different_size_are_refused():
    with pytest.raises(ValueError, match='in the ground truth'):
        tb.match_vertices(np.zeros((1, 4, 4), dtype=np.int32),
                          np.zeros((1, 8, 8), dtype=np.int32))


# ---------------------------------------------------------------------------
# the scores, each one computed twice
# ---------------------------------------------------------------------------

def test_a_perfect_tracking_scores_one_and_pays_for_nothing():
    truth = moving_stack()
    costs = tb.aogm_costs(truth, truth)
    assert (costs['fn'], costs['fp'], costs['ns'],
            costs['ed'], costs['ea'], costs['ec']) == (0, 0, 0, 0, 0, 0)
    assert tb.det_score(truth, truth) == 1.0
    assert tb.tra_score(truth, truth) == 1.0
    assert tb.identity_switches(truth, truth)['switches'] == 0


def test_an_empty_tracking_scores_zero_rather_than_something():
    truth = moving_stack(3)
    empty = np.zeros_like(truth)
    assert tb.det_score(truth, empty) == 0.0
    assert tb.tra_score(truth, empty) == 0.0


def test_an_empty_ground_truth_does_not_punish_the_tracker():
    empty = np.zeros((2, 8, 8), dtype=np.int32)
    assert tb.det_score(empty, empty) == 1.0
    assert tb.tra_score(empty, empty) == 1.0


def test_one_missed_object_costs_one_false_negative_and_two_added_edges():
    truth = moving_stack()
    prediction = truth.copy()
    prediction[2][prediction[2] == 2] = 0
    costs = tb.aogm_costs(truth, prediction)
    assert (costs['fn'], costs['fp'], costs['ns']) == (1, 0, 0)
    assert (costs['ea'], costs['ed'], costs['ec']) == (2, 0, 0)
    expected_det = 1 - (W['fn'] * 1) / (W['fn'] * 18)
    expected_tra = 1 - (W['fn'] * 1 + W['ea'] * 2) / (W['fn'] * 18
                                                      + W['ea'] * 15)
    assert tb.det_score(truth, prediction) == pytest.approx(expected_det)
    assert tb.tra_score(truth, prediction) == pytest.approx(expected_tra)


def test_an_invented_object_costs_a_false_positive_and_its_edges():
    truth = moving_stack(3)
    prediction = truth.copy()
    for t in range(3):
        _square(prediction[t], 9, 40, 2, size=4)
    costs = tb.aogm_costs(truth, prediction)
    assert (costs['fn'], costs['fp'], costs['ns']) == (0, 3, 0)
    assert (costs['ed'], costs['ea'], costs['ec']) == (2, 0, 0)


def test_two_objects_segmented_as_one_cost_a_split():
    truth = np.zeros((1, 16, 16), dtype=np.int32)
    truth[0, 0:4, 0:4] = 1
    truth[0, 0:4, 6:10] = 2
    prediction = np.zeros((1, 16, 16), dtype=np.int32)
    prediction[0, 0:4, 0:10] = 1
    costs = tb.aogm_costs(truth, prediction)
    assert costs['ns'] == 1
    assert (costs['fn'], costs['fp']) == (0, 0)
    assert tb.det_score(truth, prediction) == pytest.approx(
        1 - W['ns'] / (W['fn'] * 2))


def test_a_division_scored_as_one_object_carrying_on_costs_a_changed_edge():
    truth = np.zeros((2, 32, 32), dtype=np.int32)
    truth[0][10:26, 10:26] = 1
    truth[1][10:26, 10:26] = 2
    prediction = np.zeros((2, 32, 32), dtype=np.int32)
    prediction[0][10:26, 10:26] = 1
    prediction[1][10:26, 10:26] = 1
    costs = tb.aogm_costs(truth, prediction, gt_lineage={2: 1})
    assert costs['ec'] == 1
    assert (costs['ea'], costs['ed'], costs['fn'], costs['fp']) == (0, 0, 0, 0)
    assert tb.det_score(truth, prediction, costs=costs) == 1.0
    assert tb.tra_score(truth, prediction, gt_lineage={2: 1}, costs=costs) == \
        pytest.approx(1 - W['ec'] / (W['fn'] * 2 + W['ea'] * 1))


def test_an_edge_between_merged_objects_is_deleted_rather_than_mapped():
    truth = np.zeros((2, 16, 16), dtype=np.int32)
    prediction = np.zeros((2, 16, 16), dtype=np.int32)
    for t in (0, 1):
        truth[t, 0:4, 0:4] = 1
        truth[t, 0:4, 6:10] = 2
        prediction[t, 0:4, 0:10] = 1
    costs = tb.aogm_costs(truth, prediction)
    assert costs['ns'] == 2
    assert costs['ed'] == 1
    assert costs['ea'] == 2


def test_the_weights_are_what_the_score_means():
    """A false positive costs one and a miss costs ten, and the score says so.

    The same tracking is scored with the challenge's weights and with weights
    that treat a false positive as expensively as a miss. Nothing about the
    tracking changed, so a number reported without its weights is not a
    number anyone can compare.
    """
    truth = moving_stack(3)
    prediction = truth.copy()
    for t in range(3):
        _square(prediction[t], 9, 40, 2, size=4)
    default = tb.det_score(truth, prediction)
    strict = tb.det_score(truth, prediction,
                          weights=dict(tb.AOGM_WEIGHTS, fp=10.0))
    assert default == pytest.approx(1 - (W['fp'] * 3) / (W['fn'] * 9))
    assert strict == pytest.approx(1 - (10.0 * 3) / (W['fn'] * 9))
    assert strict < default


# ---------------------------------------------------------------------------
# identity switches
# ---------------------------------------------------------------------------

def test_an_object_that_changes_id_halfway_is_one_switch():
    truth = moving_stack(4)
    prediction = truth.copy()
    for t in (2, 3):
        prediction[t][prediction[t] == 1] = 7
    switches = tb.identity_switches(truth, prediction)
    assert switches['switches'] == 1
    assert switches['tracks_with_switches'] == 1
    assert switches['gt_tracks'] == 3


def test_an_object_lost_and_found_under_a_new_id_is_still_a_switch():
    truth = moving_stack(4)
    prediction = truth.copy()
    prediction[1][prediction[1] == 1] = 0
    prediction[2][prediction[2] == 1] = 7
    prediction[3][prediction[3] == 1] = 7
    assert tb.identity_switches(truth, prediction)['switches'] == 1


def test_a_relabelled_but_consistent_tracking_has_no_switches():
    truth = moving_stack(4)
    prediction = np.where(truth > 0, truth + 100, 0)
    assert tb.identity_switches(truth, prediction)['switches'] == 0


# ---------------------------------------------------------------------------
# the row step 3 has to beat
# ---------------------------------------------------------------------------

def test_the_baseline_row_carries_the_scores_and_the_counts():
    truth = moving_stack()
    row, tracked, table = tb.score_stitcher(
        truth, per_frame_segmentation(truth), name='baseline')
    assert row['name'] == 'baseline'
    assert row['det'] == 1.0
    assert row['tra'] == 1.0
    assert row['switches'] == 0
    assert row['iou_threshold'] == pytest.approx(0.1)
    assert row['n_gt_vertices'] == 18
    assert tracked.shape == truth.shape
    assert len(table) == 18


def test_the_baseline_pays_for_the_link_it_cannot_make():
    truth = moving_stack()
    segmentation = per_frame_segmentation(truth)
    segmentation[2][segmentation[2] == segmentation[2][20, 12]] = 0
    row, _, _ = tb.score_stitcher(truth, segmentation, name='one-missed')
    assert row['fn'] == 1
    assert row['ea'] == 2
    assert row['switches'] == 1
    assert row['det'] == pytest.approx(1 - W['fn'] / (W['fn'] * 18))
    assert row['tra'] == pytest.approx(
        1 - (W['fn'] + 2 * W['ea']) / (W['fn'] * 18 + W['ea'] * 15))


def test_linking_given_perfect_segmentation_is_the_upper_bound():
    truth = moving_stack()
    perfect, _, _ = tb.score_stitcher(truth, truth, name='perfect')
    segmentation = per_frame_segmentation(truth)
    segmentation[3][segmentation[3] == segmentation[3][34, 32]] = 0
    end_to_end, _, _ = tb.score_stitcher(truth, segmentation, name='real')
    assert perfect['tra'] == 1.0
    assert end_to_end['tra'] < perfect['tra']


def test_a_stitcher_that_links_nothing_scores_worse_than_one_that_links():
    truth = moving_stack()
    segmentation = per_frame_segmentation(truth)
    linked, _, _ = tb.score_stitcher(truth, segmentation, iou_threshold=0.1,
                                     name='linked')
    unlinked, _, _ = tb.score_stitcher(truth, segmentation,
                                       iou_threshold=1.01, name='unlinked')
    assert unlinked['tra'] < linked['tra']
    assert unlinked['det'] == linked['det']
    assert unlinked['ea'] == 15


def test_the_table_puts_the_better_tracking_first():
    truth = moving_stack(4)
    worse = truth.copy()
    worse[2][worse[2] == 2] = 0
    table = tb.scores_table([
        tb.score_tracking(truth, worse, name='worse'),
        tb.score_tracking(truth, truth, name='better'),
    ])
    assert list(table['name']) == ['better', 'worse']
    assert list(table.columns)[:5] == [
        'name', 'det', 'tra', 'switches', 'tracks_with_switches']


def test_an_empty_scores_table_is_empty():
    assert len(tb.scores_table([])) == 0


def test_a_tracker_that_swaps_two_cells_pays_for_both_links():
    """The wrong link is charged even though every object was found.

    Both cells are detected in both frames, so DET is perfect. The two links
    join the wrong pairs, which is the failure TRA exists to see and the one a
    detection score cannot.
    """
    truth = np.zeros((2, 16, 16), dtype=np.int32)
    prediction = np.zeros((2, 16, 16), dtype=np.int32)
    for t in (0, 1):
        truth[t, 0:4, 0:4] = 1
        truth[t, 0:4, 8:12] = 2
    prediction[0, 0:4, 0:4] = 5
    prediction[1, 0:4, 8:12] = 5
    prediction[0, 0:4, 8:12] = 6
    prediction[1, 0:4, 0:4] = 6
    costs = tb.aogm_costs(truth, prediction)
    assert (costs['fn'], costs['fp'], costs['ns']) == (0, 0, 0)
    assert (costs['ed'], costs['ea'], costs['ec']) == (2, 2, 0)
    assert tb.det_score(truth, prediction) == 1.0
    assert tb.tra_score(truth, prediction) == pytest.approx(
        1 - (W['ed'] * 2 + W['ea'] * 2) / (W['fn'] * 4 + W['ea'] * 2))
    assert tb.identity_switches(truth, prediction)['switches'] == 2


def test_partial_weights_keep_the_published_value_for_everything_else():
    """`{'fp': 10.0}` is a price change, not a half-filled weight table.

    A caller that names one weight and gets a KeyError for the other five has
    to restate the published numbers to change one of them, which is how a
    typo in a restated number becomes a score nobody can compare.
    """
    truth = moving_stack(3)
    prediction = truth.copy()
    prediction[1][prediction[1] == 2] = 0
    costs = tb.aogm_costs(truth, prediction, weights={'fp': 10.0})
    assert costs['aogm_d'] == pytest.approx(W['fn'] * 1)
    assert costs['aogm'] == pytest.approx(W['fn'] * 1 + W['ea'] * 2)
