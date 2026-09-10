"""The confusion panel answers for tables that are empty, keyless or not
square.

Each of these arrives from a real bundle. A filtered prediction table can be
empty; a table written without an object column cannot open crops and has to
say so rather than fail on click; a matrix read back from CSV can have lost a
class from one axis and stopped being square; and a model with many small
confusions needs the tail summarised rather than listed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.confusion import (ConfusionError, PREDICTED_COLUMN, TRUE_COLUMN,
                             cell_rows, confusion_counts, describe_confusions,
                             key_collisions, object_keys_for, rank_confusions)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({
        TRUE_COLUMN: ["a", "a", "b"],
        PREDICTED_COLUMN: ["a", "b", "b"],
        "object_key": ["k1", "k2", "k3"],
    })


def test_a_named_key_column_that_is_absent_is_refused_by_name():
    """The caller named a column; the complaint has to name it back."""
    with pytest.raises(ConfusionError) as excinfo:
        object_keys_for(_predictions(), column="crop_id")
    assert "crop_id" in str(excinfo.value)


def test_counting_collisions_on_an_absent_column_is_refused_too():
    """A silent zero would read as "this table has no duplicate objects"."""
    with pytest.raises(ConfusionError) as excinfo:
        key_collisions(_predictions(), column="crop_id")
    assert "crop_id" in str(excinfo.value)


def test_an_empty_prediction_table_gives_a_cell_no_rows():
    """A filtered-to-nothing table opens an empty grid, not an exception."""
    rows = cell_rows(_predictions().iloc[:0], "a", "b")
    assert rows.empty
    assert list(rows.columns) == list(_predictions().columns)


def test_an_empty_prediction_table_still_gives_the_declared_matrix():
    """The classes were declared, so their all-zero matrix is the answer."""
    matrix = confusion_counts(_predictions().iloc[:0], classes=["a", "b"])
    assert list(matrix.index) == ["a", "b"]
    assert matrix.to_numpy().sum() == 0


def test_an_empty_matrix_ranks_no_confusions():
    """Nothing to rank is an empty ranking, not an error."""
    assert rank_confusions(pd.DataFrame()) == []


def test_a_matrix_that_is_not_square_still_ranks_its_off_diagonal():
    """A class lost from one axis must not hide the errors it received."""
    counts = pd.DataFrame(
        [[5, 2, 1], [3, 4, 0]],
        index=pd.Index(["a", "b"], name=TRUE_COLUMN),
        columns=["a", "b", "c"])
    ranked = rank_confusions(counts)
    pairs = {(c.true_class, c.predicted_class): c.count for c in ranked}
    assert pairs == {("a", "b"): 2, ("a", "c"): 1, ("b", "a"): 3}
    assert sum(c.share_of_errors for c in ranked) == pytest.approx(1.0)


def test_the_confusions_past_the_limit_are_summarised_not_dropped():
    """The tail still holds errors; omitting it would understate the total."""
    counts = pd.DataFrame(
        [[0, 9, 8, 7], [6, 0, 5, 4], [3, 2, 0, 1], [1, 1, 1, 0]],
        index=pd.Index(["a", "b", "c", "d"], name=TRUE_COLUMN),
        columns=["a", "b", "c", "d"])
    text = describe_confusions(counts, limit=2)
    lines = text.splitlines()
    assert len(lines) == 3, text
    assert lines[-1].startswith("The remaining ")
    assert "confusion(s) hold" in lines[-1]
