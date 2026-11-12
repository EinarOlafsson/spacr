"""Classify finds the column it trains on, and says nothing it cannot know.

Issue #185 made Classify (ML) read the png_list column from the Classes
editor first, then ``annotation_column``, then ``annotation_columns``. These
tests pin every source of that answer, the settings that cannot be resolved
(left to the pipeline's own checks rather than guessed), and the run that
reaches the ML pipeline with no column named anywhere.
"""

from __future__ import annotations

import pytest

from spacr import classify as cl


@pytest.mark.parametrize(
    ("settings", "column"),
    [
        ({"classes": {"a": {"column": " from_classes "}},
          "annotation_column": "legacy"}, "from_classes"),
        ({"annotation_column": " legacy "}, "legacy"),
        ({"annotation_columns": " single "}, "single"),
        ({"annotation_columns": ["", "  ", " second ", "third"]}, "second"),
        ({"annotation_columns": None}, ""),
        ({"annotation_columns": ["", " "]}, ""),
        ({}, ""),
    ],
)
def test_the_column_comes_from_the_first_source_that_names_one(settings, column):
    assert cl._annotation_column_named(settings) == column


@pytest.mark.parametrize(
    ("settings", "app_key", "missing"),
    [
        ({"dataset_mode": "annotation", "classifier_family": "ml"},
         "classify_merged", True),
        ({"dataset_mode": "annotation", "classifier_family": "ml",
          "annotation_columns": ["label"]}, "classify_merged", False),
        ({"dataset_mode": "metadata", "classifier_family": "ml"},
         "classify_merged", False),
        ({"dataset_mode": "annotation"}, "ml_analyze", True),
        ({"dataset_mode": "annotation"}, "classify", False),
        ({"dataset_mode": "annotation", "train": True}, "classify", True),
        # Settings the checks cannot read are left to the pipeline.
        ({"dataset_mode": "not-a-basis"}, "classify_merged", False),
        ({"dataset_mode": "annotation", "classifier_family": "bogus"},
         "classify_merged", False),
    ],
)
def test_missing_labels_are_reported_only_when_the_run_trains_on_them(
    settings, app_key, missing,
):
    assert cl._annotation_labels_missing(settings, app_key) is missing


def test_an_ml_run_with_no_column_reaches_the_pipeline_without_inventing_one(
    monkeypatch,
):
    from spacr import ml

    seen = {}
    monkeypatch.setattr(ml, "generate_ml_scores",
                        lambda settings: seen.setdefault("settings", settings))

    cl.classify({"classifier_family": "ml", "dataset_mode": "annotation",
                 "model_type_ml": "xgboost", "src": "/nowhere"})

    assert not seen["settings"].get("annotation_column")
    assert seen["settings"]["model_type_ml"] == "xgboost"
