"""Classify (ML) can train on thresholds over measured features.

The basis it never had. Classify (CV) has had all three for a while --
`spacr.io` builds a dataset from metadata, annotations or measurement rules
-- while Classify (ML) had two, and chose between them by asking whether
``annotation_column`` was ``None``.

The rules take the same shape `spacr.io` already accepts, so one settings CSV
describes the same classes to both modules. Clauses within a rule are ANDed,
which is the point: the user asked for several measurements at once, because
a single threshold is a gate rather than a class definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.ml import _labels_from_measurements


@pytest.fixture
def frame():
    return pd.DataFrame({
        "cell_area": [100.0, 600.0, 700.0, 200.0],
        "nucleus_area": [10.0, 150.0, 20.0, 30.0],
    })


def test_several_measurements_are_anded(frame):
    """The requested behaviour: more than one measurement at a time."""
    settings = _labels_from_measurements(frame, {"measurement_rules": [
        {"name": "big", "where": [
            {"column": "cell_area", "op": ">", "value": 500},
            {"column": "nucleus_area", "op": ">", "value": 100},
        ]},
    ]})
    labels = frame[settings["annotation_column"]]
    # Row 1 alone satisfies BOTH clauses; row 2 has the area but not the
    # nucleus, and an OR would have caught it.
    assert labels.tolist() == [np.nan, "big", np.nan, np.nan] or (
        labels.isna().tolist() == [True, False, True, True]
        and labels.iloc[1] == "big")


def test_unmatched_rows_stay_unlabelled(frame):
    """They must not be swept into a class -- that would invent training
    data, and the model would learn the sweep."""
    settings = _labels_from_measurements(frame, {"measurement_rules": [
        {"name": "big", "where": [
            {"column": "cell_area", "op": ">", "value": 500}]},
    ]})
    labels = frame[settings["annotation_column"]]
    assert labels.isna().sum() == 2
    assert set(labels.dropna()) == {"big"}


def test_two_classes(frame):
    settings = _labels_from_measurements(frame, {"measurement_rules": [
        {"name": "small", "where": [
            {"column": "cell_area", "op": "<", "value": 300}]},
        {"name": "big", "where": [
            {"column": "cell_area", "op": ">=", "value": 600}]},
    ]})
    counts = frame[settings["annotation_column"]].value_counts().to_dict()
    assert counts == {"small": 2, "big": 2}


def test_it_points_the_run_at_the_column_it_wrote(frame):
    settings = _labels_from_measurements(frame, {"measurement_rules": [
        {"name": "big", "where": [
            {"column": "cell_area", "op": ">", "value": 500}]},
    ]})
    column = settings["annotation_column"]
    assert column in frame.columns
    # The rest of generate_ml_scores reads labels through annotation_column,
    # so pointing at the new column is what makes the measurement basis reuse
    # the annotation path instead of duplicating it.
    assert column == "_spacr_measurement_class"


def test_the_class_column_name_can_be_overridden(frame):
    settings = _labels_from_measurements(frame, {
        "measurement_class_column": "my_class",
        "measurement_rules": [{"name": "big", "where": [
            {"column": "cell_area", "op": ">", "value": 500}]}],
    })
    assert settings["annotation_column"] == "my_class"
    assert "my_class" in frame.columns


def test_settings_are_not_modified_in_place(frame):
    original = {"measurement_rules": [{"name": "big", "where": [
        {"column": "cell_area", "op": ">", "value": 500}]}]}
    _labels_from_measurements(frame, original)
    assert "annotation_column" not in original


# ---------------------------------------------------------------------------
# Every refusal below would otherwise train a classifier on nonsense
# ---------------------------------------------------------------------------

def test_no_rules_is_an_error(frame):
    with pytest.raises(ValueError, match="needs measurement_rules"):
        _labels_from_measurements(frame, {"measurement_rules": []})


def test_a_rule_with_no_clauses_is_an_error(frame):
    """It would select every row and call the whole plate one class."""
    with pytest.raises(ValueError, match="no 'where' clauses"):
        _labels_from_measurements(
            frame, {"measurement_rules": [{"name": "all", "where": []}]})


def test_an_unknown_column_names_itself(frame):
    with pytest.raises(ValueError, match="cell_are"):
        _labels_from_measurements(frame, {"measurement_rules": [
            {"name": "big", "where": [
                {"column": "cell_are", "op": ">", "value": 5}]}]})


def test_an_unknown_operator_is_refused(frame):
    with pytest.raises(ValueError, match="operator"):
        _labels_from_measurements(frame, {"measurement_rules": [
            {"name": "big", "where": [
                {"column": "cell_area", "op": "=>", "value": 5}]}]})


def test_a_rule_matching_nothing_is_an_error(frame):
    """A class with no members trains a model that cannot predict it, and
    the failure surfaces much later as a confusing accuracy number."""
    with pytest.raises(ValueError, match="matches no rows"):
        _labels_from_measurements(frame, {"measurement_rules": [
            {"name": "huge", "where": [
                {"column": "cell_area", "op": ">", "value": 1e9}]}]})


def test_an_unnamed_rule_is_refused(frame):
    with pytest.raises(ValueError, match="needs a name"):
        _labels_from_measurements(frame, {"measurement_rules": [
            {"where": [{"column": "cell_area", "op": ">", "value": 5}]}]})
