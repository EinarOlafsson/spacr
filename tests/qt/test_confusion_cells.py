"""``C8`` — a confusion-matrix cell is a live query, not a number.

:mod:`tests.test_confusion` covers the analysis with no Qt at all. What is
left here is what needs a widget: that clicking a cell reaches the registered
object opener with exactly the objects that cell counted, in the order that
makes the split useful, and that the two lists stay separated all the way to
the request rather than being blended on the way out.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.classifier_evaluation import evaluate_predictions, write_evaluation_bundle
from spacr.qt.linked_selection import (register_object_opener,
                                       unregister_object_opener)
from spacr.qt.screens.classifier_evaluation import ClassifierEvaluationScreen


# Eight held-out objects over two classes, arranged so that every number the
# tests assert can be counted off this list by eye.
#
#   truth  ·  probability of "infected"  ·  well
_ROWS = [
    ("plate1_A_01_1_1.png", 0, 0.98, ),  # uninfected, called infected, SURE
    ("plate1_A_01_1_2.png", 0, 0.92, ),  # uninfected, called infected, SURE
    ("plate1_A_01_1_3.png", 0, 0.55, ),  # uninfected, called infected, unsure
    ("plate1_B_02_1_4.png", 0, 0.02, ),  # uninfected, right
    ("plate2_A_01_1_5.png", 1, 0.99, ),  # infected, right
    ("plate2_A_01_1_6.png", 1, 0.96, ),  # infected, right
    ("plate2_B_02_1_7.png", 1, 0.40, ),  # infected, called uninfected, unsure
    ("plate2_B_02_1_8.png", 1, 0.85, ),  # infected, right
]


@pytest.fixture
def evaluation_root(tmp_path):
    """One bundle whose confusion cells have a hand-computable content."""
    probabilities = np.asarray([[1.0 - p, p] for _path, _y, p in _ROWS])
    evaluation = evaluate_predictions(
        [y for _path, y, _p in _ROWS],
        probabilities,
        [path for path, _y, _p in _ROWS],
        classes=["uninfected", "infected"],
        fold_ids=[1, 1, 1, 1, 2, 2, 2, 2],
        calibration_method="none",
        calibration_bins=4,
    )
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation", evaluation).parents[1]


@pytest.fixture
def screen(qtbot, qt_theme_applied, evaluation_root):
    widget = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._source.setText(str(evaluation_root))
    widget.scan()
    return widget


@pytest.fixture
def opener():
    """A stand-in Annotate: records the requests routed to it.

    Registered and withdrawn around each test, passing the same object back,
    so a real Annotate screen opened by another test keeps its registration.
    """
    received = []
    register_object_opener("annotate", received.append)
    try:
        yield received
    finally:
        unregister_object_opener("annotate", received.append)


def _cell(screen, true_class, predicted_class):
    screen.show_cell(true_class, predicted_class)
    return screen._cell


# ---------------------------------------------------------------------------
# The matrix itself
# ---------------------------------------------------------------------------

def test_the_matrix_shows_counts_that_match_the_predictions(screen):
    counts = screen._counts
    assert int(counts.at["uninfected", "infected"]) == 3
    assert int(counts.at["uninfected", "uninfected"]) == 1
    assert int(counts.at["infected", "uninfected"]) == 1
    assert int(counts.at["infected", "infected"]) == 3


def test_the_ranking_is_on_screen_in_words(screen):
    text = screen._confusion_ranking.text()
    assert text.startswith("Your worst confusion is uninfected → infected")
    assert "%" in text


def test_a_cell_is_resolved_from_the_widget_not_from_a_row_number(screen):
    # Column 0 holds the true class; the header holds the predicted one.
    assert screen._confusion.item(0, 0).text() == "uninfected"
    assert screen._confusion.horizontalHeaderItem(1).text() == "uninfected"
    screen._on_confusion_cell(0, 2)
    assert (screen._cell.true_class, screen._cell.predicted_class) == (
        "uninfected", "infected")


def test_clicking_the_class_name_column_explains_rather_than_guessing(screen):
    screen._on_confusion_cell(0, 0)
    assert screen._cell is None
    assert "(true, predicted) pair" in screen._status.text()


# ---------------------------------------------------------------------------
# The two lists
# ---------------------------------------------------------------------------

def test_a_cell_holds_exactly_the_objects_it_counted(screen):
    cell = _cell(screen, "uninfected", "infected")
    assert sorted(cell.rows["basename"]) == [
        "plate1_A_01_1_1.png", "plate1_A_01_1_2.png",
        "plate1_A_01_1_3.png"]


def test_the_two_lists_partition_the_cell_on_screen(screen):
    cell = _cell(screen, "uninfected", "infected")
    assert len(cell.high) + len(cell.low) == len(cell.rows)
    assert set(cell.high.index).isdisjoint(cell.low.index)
    assert list(cell.high["basename"]) == [
        "plate1_A_01_1_1.png", "plate1_A_01_1_2.png"]
    assert list(cell.low["basename"]) == ["plate1_A_01_1_3.png"]


def test_moving_the_threshold_re_splits_the_open_cell(screen):
    _cell(screen, "uninfected", "infected")
    screen._threshold.setValue(0.99)
    assert len(screen._cell.high) == 0
    assert len(screen._cell.low) == 3
    screen._threshold.setValue(0.10)
    assert len(screen._cell.high) == 3
    assert len(screen._cell.low) == 0


def test_the_breakdown_names_both_the_well_and_the_plate(screen):
    _cell(screen, "uninfected", "infected")
    text = screen._cell_breakdown.text()
    assert text.startswith("well:")
    assert "\nplate:" in text
    # All three came from plate1 / row A / column 01.
    assert "worst is plate1_A_01 with 3 (100%)" in text
    assert "worst is plate1 with 3 (100%)" in text
    # ...but three is too few to call it a bench problem. The verdict has a
    # floor for exactly this reason; see `_CONCENTRATION_FLOOR`.
    assert "not a model problem" not in text


# ---------------------------------------------------------------------------
# Routing — the whole reason the cell is clickable
# ---------------------------------------------------------------------------

def test_opening_the_sure_list_routes_exactly_those_objects_in_order(
        screen, opener):
    _cell(screen, "uninfected", "infected")
    screen.open_cell("high")

    assert len(opener) == 1
    request = opener[0]
    assert list(request.keys) == [
        "plate1_A_01_1_1.png", "plate1_A_01_1_2.png"]
    assert request.source == "classifier_evaluation"
    assert "suspect the label" in request.reason
    assert request.context["which"] == "high"
    assert request.context["true_class"] == "uninfected"


def test_opening_the_unsure_list_routes_the_other_half_and_only_it(
        screen, opener):
    _cell(screen, "uninfected", "infected")
    screen.open_cell("low")

    assert list(opener[0].keys) == ["plate1_A_01_1_3.png"]
    assert "suspect the boundary" in opener[0].reason


def test_the_two_requests_partition_the_cell_all_the_way_to_the_opener(
        screen, opener):
    cell = _cell(screen, "uninfected", "infected")
    screen.open_cell("high")
    screen.open_cell("low")

    high, low = [set(request.keys) for request in opener]
    assert high.isdisjoint(low)
    assert high | low == set(cell.keys("all"))
    assert len(high) + len(low) == len(cell.rows)


def test_the_request_carries_the_confidence_that_produced_the_order(
        screen, opener):
    _cell(screen, "uninfected", "infected")
    screen.open_cell("high")
    scores = opener[0].context["scores"]
    assert scores["plate1_A_01_1_1.png"] > scores["plate1_A_01_1_2.png"]
    assert opener[0].context["threshold"] == pytest.approx(0.75)


def test_the_buttons_open_the_list_they_are_labelled_with(screen, opener):
    _cell(screen, "uninfected", "infected")
    assert screen._high_open.isEnabled() and screen._low_open.isEnabled()
    screen._high_open.click()
    screen._low_open.click()
    assert [len(request.keys) for request in opener] == [2, 1]


def test_an_empty_half_offers_no_button_rather_than_a_button_that_opens_nothing(
        screen, opener):
    _cell(screen, "uninfected", "uninfected")   # one object, sure and right
    assert screen._high_open.isEnabled()
    assert not screen._low_open.isEnabled()
    assert screen.open_cell("low") is None
    assert opener == []


def test_with_nothing_registered_the_buttons_are_off_rather_than_raising(
        screen):
    # No `opener` fixture: nothing is registered for "annotate".
    _cell(screen, "uninfected", "infected")
    assert not screen._high_open.isEnabled()
    assert not screen._low_open.isEnabled()


def test_a_diagonal_cell_is_openable_too_because_looking_is_not_only_for_errors(
        screen, opener):
    cell = _cell(screen, "infected", "infected")
    assert not cell.is_error
    screen.open_cell("high")
    assert len(opener[0].keys) == len(cell.high)


def test_clearing_the_bundle_clears_the_cell_rather_than_leaving_the_last_one(
        screen):
    _cell(screen, "uninfected", "infected")
    screen._clear_bundle()
    assert screen._cell is None
    assert "Click a cell" in screen._cell_summary.text()
    assert screen._high_list.count() == 0
