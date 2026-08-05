"""``C8`` — the confusion matrix as a query, tested without a display.

The claims worth pinning are the ones a wrong answer would be *plausible* for:
that a cell holds exactly the objects it counted and not one more, that the
two confidence lists partition it rather than nearly partitioning it, and that
"all 43 errors came from one well" is said out loud rather than left for
somebody to notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import confusion as cx


def _predictions() -> pd.DataFrame:
    """A hand-written out-of-fold table with a known confusion structure.

    Twelve objects, two classes. Six are truly ``uninfected``; four of those
    are called ``infected`` (three of them from well ``p1_A01``), and two of
    the four are called with high confidence. Every number the tests assert is
    countable off this table by eye.
    """
    rows = [
        # object,     true,         predicted,    confidence, well,    plate
        ("u1", "uninfected", "infected",   0.97, "p1_A01", "p1"),
        ("u2", "uninfected", "infected",   0.91, "p1_A01", "p1"),
        ("u3", "uninfected", "infected",   0.58, "p1_A01", "p1"),
        ("u4", "uninfected", "infected",   0.51, "p1_B02", "p1"),
        ("u5", "uninfected", "uninfected", 0.99, "p1_B02", "p1"),
        ("u6", "uninfected", "uninfected", 0.88, "p2_A01", "p2"),
        ("i1", "infected",   "uninfected", 0.80, "p2_A01", "p2"),
        ("i2", "infected",   "infected",   0.95, "p2_A01", "p2"),
        ("i3", "infected",   "infected",   0.93, "p2_B02", "p2"),
        ("i4", "infected",   "infected",   0.77, "p2_B02", "p2"),
        ("i5", "infected",   "infected",   0.66, "p2_B02", "p2"),
        ("i6", "infected",   "infected",   0.60, "p2_B02", "p2"),
    ]
    return pd.DataFrame({
        "object": [r[0] for r in rows],
        "basename": [f"{r[0]}.png" for r in rows],
        "sample": [f"/crops/{r[0]}.png" for r in rows],
        "true_class": [r[1] for r in rows],
        "predicted_class": [r[2] for r in rows],
        "confidence": [r[3] for r in rows],
        "well": [r[4] for r in rows],
        "plate": [r[5] for r in rows],
        "correct": [r[1] == r[2] for r in rows],
    })


# ---------------------------------------------------------------------------
# A cell is exactly the objects it counted
# ---------------------------------------------------------------------------

def test_a_cell_yields_exactly_the_hand_computed_object_set():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    assert list(rows["object"]) == ["u1", "u2", "u3", "u4"]


def test_a_cell_matches_the_count_the_matrix_shows():
    predictions = _predictions()
    counts = cx.confusion_counts(predictions)
    for true_name in counts.index:
        for predicted_name in counts.columns:
            cell = cx.cell_rows(predictions, true_name, predicted_name)
            assert len(cell) == int(counts.at[true_name, predicted_name])


def test_a_cell_is_matched_as_text_so_an_integer_class_still_resolves():
    frame = pd.DataFrame({
        "true_class": [0, 0, 1],
        "predicted_class": [1, 0, 1],
        "confidence": [0.9, 0.9, 0.9],
        "object": ["a", "b", "c"],
    })
    # The matrix read back from CSV carries strings; the frame kept integers.
    assert list(cx.cell_rows(frame, "0", "1")["object"]) == ["a"]


def test_a_cell_is_a_copy_so_sorting_it_cannot_reorder_the_bundle():
    predictions = _predictions()
    before = list(predictions["object"])
    cell = cx.cell_rows(predictions, "uninfected", "infected")
    cell.sort_values("confidence", inplace=True)
    assert list(predictions["object"]) == before


def test_an_empty_cell_is_a_real_answer_not_an_error():
    predictions = _predictions()
    assert cx.cell_rows(predictions, "uninfected", "nonesuch").empty


def test_a_frame_with_no_class_columns_says_so_rather_than_counting_zero():
    with pytest.raises(cx.ConfusionError, match="missing"):
        cx.cell_rows(pd.DataFrame({"object": ["a"]}), "x", "y")


# ---------------------------------------------------------------------------
# The two lists partition the cell
# ---------------------------------------------------------------------------

def test_high_and_low_confidence_partition_the_cell_exactly():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    high, low = cx.split_by_confidence(rows, 0.75)

    assert len(high) + len(low) == len(rows)
    assert set(high.index).isdisjoint(low.index)
    assert set(high.index) | set(low.index) == set(rows.index)
    assert list(high["object"]) == ["u1", "u2"]
    assert list(low["object"]) == ["u4", "u3"]


def test_high_confidence_is_most_confident_first_and_low_is_least_first():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    high, low = cx.split_by_confidence(rows, 0.75)
    assert list(high["confidence"]) == sorted(high["confidence"], reverse=True)
    assert list(low["confidence"]) == sorted(low["confidence"])


def test_a_missing_confidence_goes_to_low_rather_than_being_dropped():
    rows = pd.DataFrame({
        "object": ["a", "b", "c"],
        "true_class": ["u"] * 3,
        "predicted_class": ["i"] * 3,
        "confidence": [0.9, np.nan, 0.2],
    })
    high, low = cx.split_by_confidence(rows, 0.75)
    assert list(high["object"]) == ["a"]
    # Present, and after the row that really was near the boundary.
    assert list(low["object"]) == ["c", "b"]
    assert len(high) + len(low) == len(rows)


def test_a_threshold_of_zero_puts_the_whole_cell_in_the_high_list():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    high, low = cx.split_by_confidence(rows, 0.0)
    assert len(high) == len(rows) and low.empty


def test_a_threshold_above_one_puts_the_whole_cell_in_the_low_list():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    high, low = cx.split_by_confidence(rows, 1.5)
    assert high.empty and len(low) == len(rows)


def test_splitting_an_empty_cell_gives_two_empty_frames():
    rows = cx.cell_rows(_predictions(), "uninfected", "nonesuch")
    high, low = cx.split_by_confidence(rows, 0.75)
    assert high.empty and low.empty


def test_splitting_without_a_confidence_column_says_which_column_is_missing():
    rows = pd.DataFrame({"true_class": ["u"], "predicted_class": ["i"]})
    with pytest.raises(cx.ConfusionError, match="confidence"):
        cx.split_by_confidence(rows, 0.5)


# ---------------------------------------------------------------------------
# Where "sure" starts
# ---------------------------------------------------------------------------

def test_the_default_threshold_is_the_midpoint_between_chance_and_certainty():
    assert cx.confidence_threshold(2) == pytest.approx(0.75)
    assert cx.confidence_threshold(4) == pytest.approx(0.625)
    assert cx.confidence_threshold(10) == pytest.approx(0.55)


def test_one_class_is_not_a_confusion_matrix():
    with pytest.raises(cx.ConfusionError, match="at least two"):
        cx.confidence_threshold(1)


# ---------------------------------------------------------------------------
# The keys that get routed
# ---------------------------------------------------------------------------

def test_the_key_column_prefers_the_path_the_model_was_actually_given():
    assert cx.object_key_column(_predictions()) == "sample"


def test_the_key_column_falls_back_through_object_then_basename():
    assert cx.object_key_column(pd.DataFrame({"object": [], "basename": []})) \
        == "object"
    assert cx.object_key_column(pd.DataFrame({"basename": []})) == "basename"


def test_a_frame_that_names_no_objects_refuses_rather_than_opening_nothing():
    with pytest.raises(cx.ConfusionError, match="names an object"):
        cx.object_key_column(pd.DataFrame({"confidence": [0.5]}))


def test_keys_keep_the_list_order_because_worst_first_is_the_point():
    cell = cx.ConfusionCell.build(_predictions(), "uninfected", "infected",
                                  threshold=0.75)
    assert list(cell.keys("high")) == ["/crops/u1.png", "/crops/u2.png"]
    assert list(cell.keys("low")) == ["/crops/u4.png", "/crops/u3.png"]


def test_all_is_the_high_list_then_the_low_list_not_table_order():
    cell = cx.ConfusionCell.build(_predictions(), "uninfected", "infected",
                                  threshold=0.75)
    assert list(cell.keys("all")) == [
        "/crops/u1.png", "/crops/u2.png", "/crops/u4.png", "/crops/u3.png"]


def test_duplicate_keys_are_dropped_and_counted_rather_than_assumed_away():
    frame = pd.DataFrame({"object": ["a", "a", "b"]})
    assert list(cx.object_keys_for(frame, column="object")) == ["a", "b"]
    assert cx.key_collisions(frame, column="object") == 1
    assert cx.key_collisions(pd.DataFrame({"object": ["a", "b"]})) == 0


def test_an_unknown_list_name_is_refused():
    cell = cx.ConfusionCell.build(_predictions(), "uninfected", "infected")
    with pytest.raises(cx.ConfusionError, match="high"):
        cell.keys("medium")


# ---------------------------------------------------------------------------
# The ranking, in words
# ---------------------------------------------------------------------------

def test_the_worst_confusion_is_ranked_first_by_share_of_all_errors():
    ranked = cx.rank_confusions(cx.confusion_counts(_predictions()))
    assert (ranked[0].true_class, ranked[0].predicted_class) == (
        "uninfected", "infected")
    # Five errors in all: four here and one the other way.
    assert ranked[0].count == 4
    assert ranked[0].share_of_errors == pytest.approx(4 / 5)
    assert ranked[0].rate_within_true == pytest.approx(4 / 6)


def test_the_ranking_is_said_in_words_with_the_percentage_in_it():
    text = cx.describe_confusions(cx.confusion_counts(_predictions()))
    assert text.startswith("Your worst confusion is uninfected → infected")
    assert "80%" in text


def test_a_perfect_classifier_gets_a_sentence_rather_than_a_blank_panel():
    frame = pd.DataFrame({
        "true_class": ["a", "b"], "predicted_class": ["a", "b"],
        "confidence": [1.0, 1.0], "object": ["x", "y"],
    })
    text = cx.describe_confusions(cx.confusion_counts(frame))
    assert "No off-diagonal mass" in text


def test_the_matrix_keeps_a_column_for_a_class_the_model_never_chose():
    frame = pd.DataFrame({
        "true_class": ["a", "b"], "predicted_class": ["a", "a"],
        "confidence": [1.0, 1.0], "object": ["x", "y"],
    })
    counts = cx.confusion_counts(frame)
    assert list(counts.columns) == ["a", "b"]
    assert int(counts.at["b", "a"]) == 1


def test_the_class_order_can_be_pinned_by_the_caller():
    counts = cx.confusion_counts(_predictions(), ["uninfected", "infected"])
    assert list(counts.index) == ["uninfected", "infected"]


# ---------------------------------------------------------------------------
# Where the errors came from
# ---------------------------------------------------------------------------

def test_a_cell_is_broken_down_per_well_most_concentrated_first():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    table = cx.breakdown_by(rows, "well")
    assert list(table["well"]) == ["p1_A01", "p1_B02"]
    assert list(table["count"]) == [3, 1]
    assert table["share"].sum() == pytest.approx(1.0)


def test_errors_concentrated_in_one_well_are_called_a_bench_problem():
    rows = pd.DataFrame({
        "true_class": ["u"] * 6, "predicted_class": ["i"] * 6,
        "confidence": [0.9] * 6, "object": list("abcdef"),
        "well": ["A01"] * 6,
    })
    text = cx.describe_breakdown(rows, "well")
    assert "single well (A01)" in text
    assert "not a model problem" in text


def test_a_handful_of_errors_in_one_well_is_not_called_a_bench_problem():
    rows = pd.DataFrame({
        "true_class": ["u"] * 3, "predicted_class": ["i"] * 3,
        "confidence": [0.9] * 3, "object": list("abc"), "well": ["A01"] * 3,
    })
    assert "not a model problem" not in cx.describe_breakdown(rows, "well")


def test_errors_spread_over_wells_are_called_the_models_problem():
    rows = pd.DataFrame({
        "true_class": ["u"] * 6, "predicted_class": ["i"] * 6,
        "confidence": [0.9] * 6, "object": list("abcdef"),
        "well": ["A01", "A02", "B01", "B02", "C01", "C02"],
    })
    text = cx.describe_breakdown(rows, "well")
    assert "the model's" in text
    assert "not a model problem" not in text


def test_breaking_down_by_a_level_the_table_lacks_refuses():
    rows = cx.cell_rows(_predictions(), "uninfected", "infected")
    with pytest.raises(cx.ConfusionError, match="no such column"):
        cx.breakdown_by(rows, "timepoint")


def test_breaking_down_an_empty_cell_says_there_is_nothing_to_break_down():
    rows = cx.cell_rows(_predictions(), "uninfected", "nonesuch")
    assert "nothing to break down" in cx.describe_breakdown(rows, "well")


# ---------------------------------------------------------------------------
# The whole cell, as the screen sees it
# ---------------------------------------------------------------------------

def test_building_a_cell_defaults_the_threshold_from_the_class_count():
    cell = cx.ConfusionCell.build(_predictions(), "uninfected", "infected")
    assert cell.threshold == pytest.approx(0.75)


def test_the_reason_names_the_hypothesis_not_only_the_cell():
    cell = cx.ConfusionCell.build(_predictions(), "uninfected", "infected")
    assert "suspect the label" in cell.reason("high")
    assert "suspect the boundary" in cell.reason("low")
    assert "annotated uninfected" in cell.reason()


def test_a_diagonal_cell_knows_it_is_not_an_error():
    cell = cx.ConfusionCell.build(_predictions(), "infected", "infected")
    assert not cell.is_error
    assert "correct predictions" in cell.describe()


def test_an_empty_cell_describes_itself_rather_than_dividing_by_zero():
    cell = cx.ConfusionCell.build(_predictions(), "uninfected", "nonesuch")
    assert len(cell) == 0
    assert "No objects" in cell.describe()
    assert list(cell.keys("all")) == []
