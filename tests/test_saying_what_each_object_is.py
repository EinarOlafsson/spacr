"""Item 449: the classification seam, and the geometry it decides alone.

No head is trained yet. What is tested here is everything that does not
need one: the two ways in, the facts about each object that are geometry
rather than opinion, the merge that only a segmenter-split object gets,
and the rule that nothing is removed or renumbered unless it was asked
for.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr import object_classifier as oc


def _two_cells():
    """Two separate objects, neither touching the border."""
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[5:15, 5:15] = 1
    mask[25:35, 25:35] = 7
    return mask


def _one_object_cut_in_two():
    """One square the segmenter split down the middle: 3 | 4."""
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[10:30, 10:20] = 3
    mask[10:30, 20:30] = 4
    return mask


def _object_on_the_edge():
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[0:10, 0:10] = 2
    return mask


def test_the_bins_are_the_series_division_makes():
    assert oc.bin_parasite_count(1) == "1"
    assert oc.bin_parasite_count(2) == "2"
    assert oc.bin_parasite_count(16) == "16"
    assert oc.bin_parasite_count(17) == ">16"
    assert oc.bin_parasite_count(40) == ">16"


def test_a_three_is_reported_in_a_bin_but_was_never_trained_as_one():
    """The maintainer's decision: the head predicts a count, and the bins
    are how it is shown. A 3 has to land somewhere, and it lands in the
    nearer of 2 and 4."""
    assert oc.bin_parasite_count(3) == "2", "a tie goes to the round it finished"
    assert oc.bin_parasite_count(5) == "4"
    assert oc.bin_parasite_count(6) == "4"
    assert oc.bin_parasite_count(7) == "8"
    assert oc.bin_parasite_count(12) == "8", "four from each; it finished eight"


def test_an_object_on_the_edge_is_the_kind_nothing_here_can_mend():
    mask = _object_on_the_edge()
    assert oc.touches_border(mask, 2)
    assert not oc.touches_border(_two_cells(), 1)


def test_two_labels_sharing_most_of_an_edge_are_one_object_cut_in_two():
    assert oc.split_candidates(_one_object_cut_in_two()) == [(3, 4)]


def test_two_round_cells_lying_against_each_other_are_not():
    """A cell against a cell shares a little of its edge; the rule has to
    tell that from a cut, or every touching pair would be merged.

    ROUND, AND THE SHAPE IS THE POINT. Two squares meeting face to face
    share one whole side, which is precisely what an object cut down the
    middle looks like -- there is no rule that could tell those apart, and
    a test using them would be asking for one. Cells are round and touch
    along a short arc.
    """
    yy, xx = np.mgrid[0:60, 0:60]
    mask = np.zeros((60, 60), dtype=np.uint16)
    mask[((xx - 20) ** 2 + (yy - 30) ** 2) <= 81] = 3
    mask[((xx - 38) ** 2 + (yy - 30) ** 2) <= 81] = 4
    assert 3 in np.unique(mask) and 4 in np.unique(mask)
    assert oc.split_candidates(mask) == []


def test_a_fragment_lying_against_a_cell_is_offered_as_a_candidate():
    """Said rather than hidden: a small piece touching a big object shares
    most of ITS OWN edge, so the rule offers it. That is the intended
    reading -- a fragment against a cell usually is a piece the segmenter
    cut off it -- and it is why merging is an option a caller turns on
    rather than something that happens by itself."""
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[10:30, 10:20] = 3
    mask[28:30, 20:22] = 4
    assert oc.split_candidates(mask) == [(3, 4)]


def test_merging_keeps_the_lower_id_and_leaves_the_original_alone():
    mask = _one_object_cut_in_two()
    merged = oc.merge_halves(mask, [(3, 4)])
    assert set(np.unique(merged)) == {0, 3}
    assert set(np.unique(mask)) == {0, 3, 4}, "the mask passed in was edited"


def test_describe_says_area_border_and_who_it_may_be_one_object_with():
    rows = oc.describe_objects(_one_object_cut_in_two())
    assert [row["label"] for row in rows] == [3, 4]
    assert rows[0]["area"] == 200 and rows[1]["area"] == 200
    assert rows[0]["split_with"] == [4] and rows[1]["split_with"] == [3]
    assert not rows[0]["touches_border"]


def test_classifying_without_a_head_changes_nothing_at_all():
    mask = _two_cells()
    result = oc.classify_objects(np.zeros((40, 40), dtype=np.uint16), mask)
    assert np.array_equal(result["mask"], mask)
    assert result["merged"] == [] and result["removed"] == []
    assert [row["label"] for row in result["objects"]] == [1, 7]
    assert "class" not in result["objects"][0]


def test_ids_are_never_renumbered():
    """The ids are what the measurements, the crops and the tracks are
    keyed on."""
    mask = _two_cells()
    result = oc.classify_objects(np.zeros((40, 40)), mask)
    assert [row["label"] for row in result["objects"]] == [1, 7]


class _Head:
    """A head that answers from a list, in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def predict(self, crops):
        self.seen.append(len(crops))
        return self.answers[:len(crops)]


def test_a_head_is_handed_one_crop_per_object_and_its_answers_line_up():
    image = np.random.default_rng(449).integers(0, 255, (40, 40),
                                                dtype=np.uint16)
    head = _Head([("cell", 0.9), ("nucleus", 0.8)])
    result = oc.classify_objects(image, _two_cells(), head=head, size=16)
    assert head.seen == [2]
    assert [row["class"] for row in result["objects"]] == ["cell", "nucleus"]
    assert result["objects"][0]["probability"] == pytest.approx(0.9)


def test_a_head_that_answers_the_wrong_number_of_crops_is_an_error():
    head = _Head([("cell", 1.0)])
    head.predict = lambda crops: [("cell", 1.0)]
    with pytest.raises(ValueError, match="line up"):
        oc.classify_objects(np.zeros((40, 40)), _two_cells(), head=head,
                            size=8)


def test_artifacts_are_removed_only_when_that_was_asked_for():
    image = np.zeros((40, 40), dtype=np.uint16)
    mask = _two_cells()
    head = _Head([("cell", 0.9), ("artifact", 0.99)])
    kept = oc.classify_objects(image, mask, head=head, size=8)
    assert np.array_equal(kept["mask"], mask) and kept["removed"] == []
    head = _Head([("cell", 0.9), ("artifact", 0.99)])
    cut = oc.classify_objects(image, mask, head=head, size=8,
                              remove_artifacts=True)
    assert cut["removed"] == [7]
    assert set(np.unique(cut["mask"])) == {0, 1}
    assert set(np.unique(mask)) == {0, 1, 7}, "the mask passed in was edited"


def test_merging_is_offered_to_a_split_object_and_not_to_a_border_one():
    """The maintainer's answer: only a segmenter-split object can be
    merged where it stands, because the other half of a border object is
    in the next field."""
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[0:20, 0:10] = 5
    mask[0:20, 10:20] = 6
    assert oc.split_candidates(mask) == [(5, 6)]
    result = oc.classify_objects(np.zeros((40, 40)), mask, merge_split=True)
    assert result["merged"] == [], "both halves are on the border"
    assert np.array_equal(result["mask"], mask)


def test_a_split_object_away_from_the_edge_is_merged_when_asked():
    mask = _one_object_cut_in_two()
    result = oc.classify_objects(np.zeros((40, 40)), mask, merge_split=True)
    assert result["merged"] == [(3, 4)]
    assert set(np.unique(result["mask"])) == {0, 3}
    assert [row["label"] for row in result["objects"]] == [3]


class _Backend:
    """Something with Cellpose's eval, which is all the wrapper needs."""

    def __init__(self, mask):
        self.mask = mask
        self.calls = []

    def eval(self, x, **kwargs):
        self.calls.append((len(x), kwargs))
        return [self.mask], [None], None


def test_the_same_field_gives_the_same_answer_both_ways_in():
    """Segment-then-classify and classify-a-mask must agree; that is the
    whole point of keeping the two apart."""
    image = np.zeros((40, 40), dtype=np.uint16)
    mask = _two_cells()
    backend = _Backend(mask)
    through = oc.segment_and_classify(backend, image,
                                      eval_kwargs={"diameter": 30})
    direct = oc.classify_objects(image, mask)
    assert through["objects"] == direct["objects"]
    assert backend.calls[0][1]["diameter"] == 30
