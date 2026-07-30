"""Tests for general UMAP row exclusions."""

from __future__ import annotations

import pandas as pd
import pytest

from spacr.row_exclusions import (
    exclude_matching_rows,
    normalize_row_exclusions,
)


def test_normalize_row_exclusions_accepts_csv_literal_text():
    assert normalize_row_exclusions(
        "{'columnID': ['c1', 'c2'], 'plateID': 'plate3'}"
    ) == {
        "columnID": ["c1", "c2"],
        "plateID": ["plate3"],
    }


def test_exclusions_from_multiple_columns_are_combined_with_or():
    frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "columnID": ["c1", "c2", "c1", "c2"],
        "feature": [1.0, 2.0, 3.0, 4.0],
    })

    filtered, notes = exclude_matching_rows(
        frame,
        {"columnID": ["c1"], "plateID": ["p2"]},
    )

    assert filtered[["plateID", "columnID"]].values.tolist() == [["p1", "c2"]]
    assert len(notes) == 2
    assert "columnID" in notes[0]
    assert "plateID" in notes[1]


def test_exclusion_values_match_numeric_sqlite_text():
    frame = pd.DataFrame({"dose": [1.0, 2.0, 3.0], "feature": [4, 5, 6]})
    filtered, _ = exclude_matching_rows(frame, {"dose": ["2.0"]})
    assert filtered["dose"].tolist() == [1.0, 3.0]


def test_unknown_exclusion_column_is_actionable():
    frame = pd.DataFrame({"columnID": ["c1"], "feature": [1.0]})
    with pytest.raises(ValueError, match="unknown column.*missing"):
        exclude_matching_rows(frame, {"missing": ["x"]})


def test_excluding_every_row_is_rejected():
    frame = pd.DataFrame({"columnID": ["c1", "c1"], "feature": [1.0, 2.0]})
    with pytest.raises(ValueError, match="removed every UMAP object"):
        exclude_matching_rows(frame, {"columnID": ["c1"]})
