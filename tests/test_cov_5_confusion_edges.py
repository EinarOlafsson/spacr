"""What the confusion panel refuses, and the sentences it says instead.

Every refusal here guards the same failure: an answer that looks like a
result. An empty breakdown reads as "the errors are spread evenly", a missing
key column reads as "these crops cannot be opened" only when it raises, and a
cell split without a confidence column would silently call every object sure.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import confusion as C


def _predictions(rows=None):
    """A small evaluation bundle in the shape spaCR writes."""
    rows = rows if rows is not None else [
        ("p1_A01_f1_cell1", "infected", "infected", 0.95, "f1", "A01", "p1"),
        ("p1_A01_f1_cell2", "infected", "uninfected", 0.91, "f1", "A01", "p1"),
        ("p1_A01_f1_cell3", "infected", "uninfected", 0.55, "f1", "A01", "p1"),
        ("p1_A02_f2_cell4", "uninfected", "infected", 0.80, "f2", "A02", "p1"),
    ]
    return pd.DataFrame(rows, columns=[
        "object_key", C.TRUE_COLUMN, C.PREDICTED_COLUMN, C.CONFIDENCE_COLUMN,
        "field", "well", "plate"])


# ---------------------------------------------------------------------------
# The threshold and the columns it needs
# ---------------------------------------------------------------------------

def test_one_class_has_no_confusion_and_no_threshold():
    """Chance is 100% with one class, so "sure" cannot be defined at all."""
    with pytest.raises(C.ConfusionError, match="at least two classes"):
        C.confidence_threshold(1)
    assert C.confidence_threshold(2) == 0.75


def test_a_table_that_is_not_an_evaluation_bundle_names_what_it_lacks():
    """The message has to name the columns AND where they come from."""
    frame = pd.DataFrame({"object_key": ["a"], "score": [1.0]})

    with pytest.raises(C.ConfusionError) as excinfo:
        C.confusion_counts(frame)

    message = str(excinfo.value)
    assert C.TRUE_COLUMN in message and C.PREDICTED_COLUMN in message
    assert "evaluate_predictions" in message


def test_a_table_naming_no_object_cannot_open_crops():
    """A "show me these crops" button that raises on click is worse than absent."""
    frame = pd.DataFrame({C.TRUE_COLUMN: ["a"], C.PREDICTED_COLUMN: ["b"]})

    with pytest.raises(C.ConfusionError, match="names an object"):
        C.object_key_column(frame)

    assert C.object_key_column(_predictions()) == "object_key"


def test_two_rows_with_one_key_are_counted_as_a_collision():
    """One key for two objects means opening the cell shows fewer crops."""
    frame = _predictions()
    assert C.key_collisions(frame) == 0

    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    assert C.key_collisions(doubled) == 1


# ---------------------------------------------------------------------------
# Splitting a cell by confidence
# ---------------------------------------------------------------------------

def test_an_empty_cell_splits_into_two_empty_halves():
    """Both halves must keep the columns, or the panel cannot render them."""
    rows = _predictions().iloc[:0]

    high, low = C.split_by_confidence(rows, 0.75)

    assert high.empty and low.empty
    assert list(high.columns) == list(rows.columns)
    assert high is not low


def test_a_cell_with_no_confidence_column_cannot_be_split():
    """Treating every row as sure would send the user to re-label good labels."""
    rows = _predictions().drop(columns=[C.CONFIDENCE_COLUMN])

    with pytest.raises(C.ConfusionError, match="no 'confidence' column"):
        C.split_by_confidence(rows, 0.75)


# ---------------------------------------------------------------------------
# The matrix and its worst cells
# ---------------------------------------------------------------------------

def test_a_class_the_model_never_chose_still_gets_its_column():
    """A vanished class makes the matrix look complete when it is not."""
    counts = C.confusion_counts(_predictions())

    assert list(counts.index) == ["infected", "uninfected"]
    assert list(counts.columns) == ["infected", "uninfected"]
    assert int(counts.loc["infected", "uninfected"]) == 2


def test_a_perfect_classifier_gets_a_sentence_rather_than_a_blank_panel():
    """A blank panel reads as a panel that failed to load."""
    perfect = _predictions([
        ("k1", "infected", "infected", 0.9, "f1", "A01", "p1"),
        ("k2", "uninfected", "uninfected", 0.9, "f1", "A01", "p1"),
    ])

    said = C.describe_confusions(C.confusion_counts(perfect))

    assert "No off-diagonal mass at all" in said
    assert "leakage audit" in said


# ---------------------------------------------------------------------------
# Breaking a cell down by where the objects came from
# ---------------------------------------------------------------------------

def test_a_level_the_table_does_not_carry_is_refused():
    """An empty breakdown would read as "the errors are spread evenly"."""
    rows = _predictions().drop(columns=["well"])

    with pytest.raises(C.ConfusionError, match="no such column"):
        C.breakdown_by(rows, "well")


def test_an_empty_cell_breaks_down_into_an_empty_table_with_its_columns():
    rows = _predictions().iloc[:0]

    table = C.breakdown_by(rows, "field")

    assert list(table.columns) == ["field", "count", "share"]
    assert table.empty
    assert C.describe_breakdown(rows, "field") == (
        "No objects in this cell, so nothing to break down by field.")


def test_errors_piled_into_one_field_are_called_a_field_problem():
    """Re-labelling those crops would be work aimed at the wrong cause."""
    rows = _predictions([
        (f"k{i}", "infected", "uninfected", 0.9, "f1", "A01", "p1")
        for i in range(6)
    ] + [("k9", "infected", "uninfected", 0.9, "f2", "A02", "p1")])

    said = C.describe_breakdown(rows, "field")

    assert "6 of 7 come from a single field (f1)" in said
    assert "not a model problem" in said
    assert "Fix it upstream" in said


# ---------------------------------------------------------------------------
# One cell of the matrix
# ---------------------------------------------------------------------------

def test_a_cell_built_without_a_threshold_derives_one_from_the_classes():
    """The default has to come from the assay's class count, not from 0.5."""
    cell = C.ConfusionCell.build(_predictions(), "infected", "uninfected")

    assert cell.threshold == C.confidence_threshold(2)
    assert len(cell) == 2
    assert cell.is_error is True
    assert len(cell.high) == 1 and len(cell.low) == 1


def test_a_cell_told_how_many_classes_there_are_uses_that_count():
    cell = C.ConfusionCell.build(_predictions(), "infected", "uninfected",
                                 n_classes=4)

    assert cell.threshold == C.confidence_threshold(4)


def test_asking_a_cell_for_an_unknown_half_is_refused():
    cell = C.ConfusionCell.build(_predictions(), "infected", "uninfected")

    assert list(cell.keys("high")) == ["p1_A01_f1_cell2"]
    assert list(cell.keys("low")) == ["p1_A01_f1_cell3"]
    with pytest.raises(C.ConfusionError, match="'high', 'low' or 'all'"):
        cell.keys("both")


def test_a_cell_nobody_landed_in_says_so():
    """A matrix cell of zero is a real cell and has to describe itself."""
    cell = C.ConfusionCell.build(_predictions(), "uninfected", "uninfected")

    assert len(cell) == 0
    assert cell.describe() == (
        "No objects were annotated uninfected and predicted uninfected.")
    assert list(cell.keys("all")) == []
