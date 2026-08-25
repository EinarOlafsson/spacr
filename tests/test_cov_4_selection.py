"""Object keys survive frames that state no type, name no rows, or name no
key columns at all.

Keys are what a lasso in one view uses to highlight the same objects in
another. Three frames arrive at the key builders that the ordinary path never
produces, and each has one right answer: a frame carrying an ``object_type``
column that is entirely blank is UNTYPED, not typed with an empty prefix; an
empty frame is an empty answer, not a crash; and a frame missing a key column
must say which column so the caller can see what it handed over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import selection
from spacr.selection import (FilterError, Selection, object_keys,
                             untyped_object_keys)


def _frame(**extra) -> pd.DataFrame:
    data = {"plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
            "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
            "object_label": [1, 2]}
    data.update(extra)
    return pd.DataFrame(data)


def test_an_all_blank_object_type_column_leaves_the_keys_untyped():
    """A column of blanks states nothing, so the keys mean what they did."""
    blank = _frame(object_type=[None, ""])
    assert list(object_keys(blank)) == list(untyped_object_keys(blank))


@pytest.mark.parametrize("blanks", [
    [None, None], ["", ""], ["nan", "none"], [float("nan"), "null"],
])
def test_every_spelling_of_blank_is_read_as_no_type(blanks):
    """Blank has several spellings and none of them is a prefix."""
    frame = _frame(object_type=blanks)
    assert list(object_keys(frame)) == list(untyped_object_keys(frame))


def test_an_empty_frame_has_no_untyped_keys():
    """Nothing selected is an empty index, not an error and not a row."""
    keys = untyped_object_keys(_frame().iloc[:0])
    assert len(keys) == 0
    assert keys.dtype == object


def test_untyped_keys_name_the_missing_key_column():
    """The message has to say what was missing or the caller cannot fix it."""
    frame = _frame().drop(columns=["fieldID"])
    with pytest.raises(FilterError) as excinfo:
        untyped_object_keys(frame)
    assert "fieldID" in str(excinfo.value)


def test_untyped_keys_name_the_missing_timepoint_column():
    """Timelapse keys need timeID, and its absence is the same complaint."""
    with pytest.raises(FilterError) as excinfo:
        untyped_object_keys(_frame(), timelapse=True)
    assert "timeID" in str(excinfo.value)


def test_matching_a_frame_that_lacks_a_key_column_says_which_one():
    """Highlighting cannot silently match nothing when a column is absent."""
    chosen = Selection.from_frame(_frame())
    with pytest.raises(FilterError) as excinfo:
        chosen.mask_for(_frame().drop(columns=["object_label"]))
    assert "object_label" in str(excinfo.value)


def test_matching_an_empty_frame_is_an_empty_mask():
    """The row matcher's own answer for no rows is a zero-length mask."""
    mask = selection._match_frame(_frame().iloc[:0], ["p1_r1_c1_f1_1"])
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (0,)
