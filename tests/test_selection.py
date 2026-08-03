"""Tests for :mod:`spacr.selection`, the shared filter/selection model.

The interesting cases here are all ones where a filter can be quietly wrong
rather than loudly broken: a NaN slipping through a range, an empty category
list being reinterpreted as "everything", a timelapse table collapsing every
frame of an object onto one key, or a filter naming a column that is not there
being silently dropped while the UI still shows it as active.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.selection import (
    CategoryFilter,
    DataFilter,
    FilterError,
    OBJECT_KEY_COLUMNS,
    ObjectRequest,
    RangeFilter,
    Selection,
    as_key_index,
    object_keys,
)


def _frame(n: int = 6, **extra) -> pd.DataFrame:
    base = {
        "plateID": ["p1"] * n,
        "rowID": [f"r{i % 2 + 1}" for i in range(n)],
        "columnID": [f"c{i % 3 + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
    }
    base.update(extra)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_object_keys_are_unique_per_object():
    df = _frame(6)
    keys = object_keys(df)
    assert len(keys) == 6
    assert len(set(keys)) == 6


def test_object_keys_use_the_schema_columns_in_order():
    df = _frame(1)
    assert list(OBJECT_KEY_COLUMNS) == [
        "plateID", "rowID", "columnID", "fieldID", "object_label"]
    assert object_keys(df)[0] == "p1_r1_c1_f1_1"


def test_a_missing_key_column_raises_rather_than_guessing():
    df = _frame(3).drop(columns=["fieldID"])
    with pytest.raises(FilterError, match="fieldID"):
        object_keys(df)


def test_timelapse_keys_keep_the_frames_apart():
    """Two frames of one object must be two keys, not one.

    Collapsing them is the failure this codebase has hit repeatedly from the
    other direction — a prcfo rebuilt without its timepoint, halving the data.
    """
    # The same object in the same well and field, at two timepoints. The
    # well/field columns have to be pinned by hand: `_frame` varies rowID and
    # columnID per row, so the default rows are different objects and the
    # collapse under test would never happen.
    df = _frame(2)
    df["rowID"] = "r1"
    df["columnID"] = "c1"
    df["timeID"] = ["t1", "t2"]
    df["object_label"] = [7, 7]

    flat = object_keys(df, timelapse=False)
    assert len(set(flat)) == 1, "without timelapse the two frames collapse"

    timed = object_keys(df, timelapse=True)
    assert len(set(timed)) == 2, "timelapse keys must separate the frames"


def test_object_keys_on_an_empty_frame_is_empty_not_an_error():
    assert len(object_keys(_frame(0))) == 0


# ---------------------------------------------------------------------------
# RangeFilter
# ---------------------------------------------------------------------------

def test_range_filter_keeps_the_closed_interval():
    df = _frame(5, area=[1.0, 5.0, 10.0, 15.0, 20.0])
    keep = RangeFilter("area", low=5.0, high=15.0).mask(df)
    assert list(df.loc[keep, "area"]) == [5.0, 10.0, 15.0]


def test_an_open_bound_means_unbounded_not_excluded():
    """A slider dragged to its end must not exclude everything."""
    df = _frame(3, area=[1.0, 2.0, 3.0])
    assert RangeFilter("area", low=2.0).mask(df).sum() == 2
    assert RangeFilter("area", high=2.0).mask(df).sum() == 2
    assert RangeFilter("area").mask(df).sum() == 3


def test_nan_never_passes_a_range():
    """An uncomputable measurement is not a measurement inside the range.

    Letting NaN through would put objects with no value into a population the
    user defined *by* value.
    """
    df = _frame(3, area=[1.0, np.nan, 3.0])
    keep = RangeFilter("area", low=0.0, high=10.0).mask(df)
    assert keep.tolist() == [True, False, True]


def test_a_non_numeric_column_coerces_rather_than_crashing():
    df = _frame(3, area=["1.0", "oops", "3.0"])
    keep = RangeFilter("area", low=0.0, high=10.0).mask(df)
    assert keep.tolist() == [True, False, True]


def test_a_range_on_a_missing_column_raises():
    with pytest.raises(FilterError, match="nope"):
        RangeFilter("nope", low=1.0).mask(_frame(2))


# ---------------------------------------------------------------------------
# CategoryFilter
# ---------------------------------------------------------------------------

def test_category_filter_keeps_only_the_named_values():
    df = _frame(6)
    keep = CategoryFilter("columnID", ("c1",)).mask(df)
    assert set(df.loc[keep, "columnID"]) == {"c1"}


def test_an_empty_category_selection_keeps_nothing():
    """Unticking every box means none, not all.

    Reinterpreting it as "all" silently widens the population while the UI
    still shows a filter as active.
    """
    assert CategoryFilter("columnID", ()).mask(_frame(6)).sum() == 0


def test_category_matching_is_by_string_so_1_and_str_1_agree():
    """The same well read from SQLite and from a CSV must match."""
    df = _frame(3)
    df["plateID"] = [1, 1, 2]
    assert CategoryFilter("plateID", ("1",)).mask(df).sum() == 2
    assert CategoryFilter("plateID", (1,)).mask(df).sum() == 2


# ---------------------------------------------------------------------------
# DataFilter
# ---------------------------------------------------------------------------

def test_an_empty_filter_is_the_identity():
    df = _frame(4)
    assert DataFilter().mask(df).all()
    assert len(DataFilter().apply(df)) == 4
    assert DataFilter().is_empty


def test_clauses_are_anded():
    df = _frame(6, area=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f = (DataFilter()
         .add(RangeFilter("area", low=3.0))
         .add(CategoryFilter("rowID", ("r1",))))
    out = f.apply(df)
    assert (out["area"] >= 3.0).all()
    assert set(out["rowID"]) == {"r1"}


def test_adding_the_same_column_twice_replaces_rather_than_stacks():
    """A dragged slider must not stack a hundred near-identical clauses."""
    f = DataFilter()
    for high in (10.0, 5.0, 2.0):
        f.add(RangeFilter("area", high=high))
    assert len(f.clauses) == 1
    assert f.clauses[0].high == 2.0


def test_remove_and_clear():
    f = DataFilter().add(RangeFilter("area", low=1.0))
    f.remove("not_there")            # unknown column is not an error
    assert len(f.clauses) == 1
    f.remove("area")
    assert f.is_empty
    f.add(RangeFilter("area", low=1.0)).clear()
    assert f.is_empty


def test_describe_says_what_is_filtered():
    """A filtered view that does not say so is how a result gets computed on
    a fifth of the data and reported as the whole."""
    assert DataFilter().describe() == "no filter"
    f = (DataFilter()
         .add(RangeFilter("area", low=5.0, high=15.0))
         .add(CategoryFilter("plateID", ("p1",))))
    text = f.describe()
    assert "area" in text and "plateID" in text and " and " in text


def test_a_filter_naming_an_absent_column_raises_on_apply():
    """Carried over from another table — must not be silently ignored."""
    f = DataFilter().add(RangeFilter("no_such_feature", low=1.0))
    with pytest.raises(FilterError, match="no_such_feature"):
        f.apply(_frame(3))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_no_selection_is_distinct_from_an_empty_one():
    """Resting state vs a lasso around blank space — drawn differently."""
    assert not Selection.none().is_active
    empty = Selection(keys=pd.Index([], dtype=object), source="umap")
    assert empty.is_active
    assert len(empty) == 0


def test_selection_round_trips_through_keys():
    df = _frame(6)
    chosen = df.iloc[[1, 3]]
    sel = Selection.from_frame(chosen, source="umap")
    mask = sel.mask_for(df)
    assert mask.tolist() == [False, True, False, True, False, False]
    assert sel.source == "umap"
    assert len(sel) == 2


def test_no_selection_masks_everything_in():
    """So `df[sel.mask_for(df)]` always means "what the user is looking at"."""
    df = _frame(4)
    assert Selection.none().mask_for(df).all()


def test_a_selection_survives_a_different_row_order():
    """The point of key-based identity: the UMAP and the plate view do not
    hold their rows in the same order, and must still agree."""
    df = _frame(6)
    sel = Selection.from_frame(df.iloc[[0, 5]])
    shuffled = df.iloc[::-1].reset_index(drop=True)
    assert sel.mask_for(shuffled).sum() == 2
    picked = shuffled.loc[sel.mask_for(shuffled), "object_label"]
    assert set(picked) == {1, 6}


def test_a_selection_from_another_table_simply_matches_nothing():
    """Not an error: the other plate's objects are legitimately absent."""
    other = _frame(3)
    other["plateID"] = "p99"
    sel = Selection.from_frame(other)
    assert sel.mask_for(_frame(6)).sum() == 0


def test_selection_on_an_empty_frame_is_empty():
    sel = Selection.from_frame(_frame(3))
    assert sel.mask_for(_frame(0)).tolist() == []


# ---------------------------------------------------------------------------
# describe(), one branch each — these strings are what a filtered view puts in
# its header, so a wrong one misreports the population rather than crashing.
# ---------------------------------------------------------------------------

def test_range_describe_covers_every_bound_combination():
    assert RangeFilter("area").describe() == "area: any"
    assert RangeFilter("area", high=5.0).describe() == "area ≤ 5"
    assert RangeFilter("area", low=2.0).describe() == "area ≥ 2"
    assert RangeFilter("area", low=2.0, high=5.0).describe() == \
        "2 ≤ area ≤ 5"


def test_category_describe_truncates_a_long_value_list():
    """A header naming forty wells is not a header."""
    assert CategoryFilter("plateID", ()).describe() == "plateID: none"
    assert CategoryFilter("plateID", ("p1",)).describe() == "plateID ∈ {p1}"

    many = CategoryFilter("columnID", tuple(f"c{i}" for i in range(6)))
    text = many.describe()
    assert "c0, c1, c2" in text
    assert "+3" in text, "the count of the hidden values must be shown"


def test_a_category_filter_on_a_missing_column_raises():
    with pytest.raises(FilterError, match="absent_col"):
        CategoryFilter("absent_col", ("x",)).mask(_frame(2))


# ---------------------------------------------------------------------------
# as_key_index — every way a view can name objects arrives at the SAME keys
# ---------------------------------------------------------------------------

def test_every_way_of_naming_objects_produces_the_same_keys():
    """A frame, a selection, an index and a list are one identity scheme.

    If they were not, a scatter plot handing keys and a table handing rows
    would open different objects for the same click.
    """
    df = _frame(3)
    expected = ["p1_r1_c1_f1_1", "p1_r2_c2_f1_2", "p1_r1_c3_f1_3"]
    assert list(as_key_index(df)) == expected
    assert list(as_key_index(Selection.from_frame(df))) == expected
    assert list(as_key_index(object_keys(df))) == expected
    assert list(as_key_index(expected)) == expected
    assert list(as_key_index(pd.Series(expected))) == expected


def test_a_single_key_string_is_one_object_not_one_per_character():
    """A str is iterable; letting it fall through opens 13 nonexistent objects.

    That failure is silent — a grid of empty tiles, not an exception — which
    is exactly why it is special-cased and pinned here.
    """
    assert list(as_key_index("p1_r1_c1_f1_1")) == ["p1_r1_c1_f1_1"]


def test_keys_keep_the_callers_order_and_lose_duplicates():
    """Order carries "worst errors first"; a duplicate would draw a crop twice."""
    keys = as_key_index(["b_key", "a_key", "b_key", "c_key"])
    assert list(keys) == ["b_key", "a_key", "c_key"]


def test_non_string_keys_are_stringified_so_they_can_match():
    """`object_keys` joins with `str`, so a caller passing ints must agree."""
    assert list(as_key_index([1, 2])) == ["1", "2"]


def test_a_resting_selection_cannot_be_opened():
    """"Nothing selected" is not a request for nothing — it is not a request.

    Coercing it to an empty index would open an empty grid and look like the
    data was missing.
    """
    with pytest.raises(ValueError, match="resting"):
        as_key_index(Selection.none())


def test_an_empty_selection_is_still_openable_and_empty():
    """The other half of that distinction: an explicit empty selection is data."""
    assert list(as_key_index(Selection(keys=pd.Index([], dtype=object)))) == []


def test_something_that_cannot_name_objects_raises_typeerror():
    with pytest.raises(TypeError, match="int"):
        as_key_index(7)


def test_timelapse_keys_reach_as_key_index_too():
    df = _frame(2, timeID=[1, 2])
    assert list(as_key_index(df, timelapse=True)) == [
        "p1_r1_c1_f1_1_1", "p1_r2_c2_f1_2_2"]


def test_selection_from_keys_matches_selection_from_frame():
    df = _frame(4)
    by_frame = Selection.from_frame(df, source="umap")
    by_keys = Selection.from_keys(list(object_keys(df)), source="umap")
    assert list(by_keys.keys) == list(by_frame.keys)
    assert by_keys.source == "umap"
    assert by_keys.mask_for(df).all()


# ---------------------------------------------------------------------------
# ObjectRequest — the thing that travels from "click here" to "show these"
# ---------------------------------------------------------------------------

def test_a_request_normalises_whatever_it_was_built_from():
    """The opener may assume an Index of str: it never sees the caller's shape."""
    request = ObjectRequest(keys=_frame(2), reason="clicked")
    assert isinstance(request.keys, pd.Index)
    assert list(request.keys) == ["p1_r1_c1_f1_1", "p1_r2_c2_f1_2"]
    assert len(request) == 2


def test_a_request_without_a_reason_is_refused():
    """Twelve crops with no stated reason read as the whole population."""
    with pytest.raises(ValueError, match="reason"):
        ObjectRequest(keys=_frame(2), reason="   ")


def test_a_reason_is_stripped_because_it_is_shown_verbatim():
    assert ObjectRequest(keys=[], reason="  misclassified  ").reason == \
        "misclassified"


def test_an_empty_request_is_legal():
    """A confusion-matrix cell with no errors in it is a real answer."""
    request = ObjectRequest(keys=[], reason="no errors in this cell")
    assert len(request) == 0
    assert request.select_from(_frame(3)).empty


def test_a_requests_context_cannot_be_changed_from_either_side():
    """The caller's dict is copied in, and the destination gets it read-only.

    Mutating it after sending would change a request already in flight, and a
    destination mutating it would edit the caller's own state.
    """
    scores = {"p1_r1_c1_f1_1": 0.99}
    request = ObjectRequest(keys=[], reason="why", context=scores)
    scores["p1_r1_c1_f1_1"] = 0.0
    assert request.context["p1_r1_c1_f1_1"] == 0.99
    with pytest.raises(TypeError):
        request.context["another"] = 1.0


def test_select_from_returns_the_requests_order_not_the_tables():
    """A request built worst-first must not be re-sorted back into table order.

    This is the whole point of carrying keys rather than a mask: a boolean
    mask cannot express "these three, in this order".
    """
    df = _frame(4)
    request = ObjectRequest(keys=["p1_r1_c3_f1_3", "p1_r1_c1_f1_1"],
                            reason="worst first")
    got = request.select_from(df)
    assert got["object_label"].tolist() == [3, 1]


def test_select_from_drops_keys_the_table_does_not_have():
    """A narrower table is a smaller result, not an error."""
    request = ObjectRequest(keys=["p1_r1_c1_f1_1", "p99_r1_c1_f1_1"],
                            reason="from another plate")
    got = request.select_from(_frame(3))
    assert got["object_label"].tolist() == [1]


def test_select_from_an_empty_table_is_empty_rather_than_a_key_error():
    """A view that has loaded nothing yet must not raise on a request.

    The frame it hands over may not even carry the key columns before its
    first query, so this must not go looking for them.
    """
    request = ObjectRequest(keys=["p1_r1_c1_f1_1"], reason="anything")
    assert request.select_from(pd.DataFrame()).empty


def test_select_from_honours_the_timelapse_flag():
    df = _frame(3, timeID=[1, 2, 3])
    request = ObjectRequest(keys=["p1_r1_c3_f1_3_3"], reason="frame 3",
                            timelapse=True)
    assert request.select_from(df)["timeID"].tolist() == [3]


def test_a_request_becomes_a_selection_carrying_its_source():
    """Opening a subset and highlighting it are two acts, joined here."""
    request = ObjectRequest(keys=_frame(2), reason="clicked", source="umap")
    selection = request.as_selection()
    assert selection.source == "umap"
    assert selection.mask_for(_frame(4)).tolist() == [True, True, False, False]


def test_describe_says_how_many_and_why_and_who_asked():
    df = _frame(2)
    assert ObjectRequest(keys=df, reason="misclassified",
                         source="matrix").describe() == \
        "2 objects · misclassified (from matrix)"
    assert ObjectRequest(keys=df.iloc[[0]], reason="clicked").describe() == \
        "1 object · clicked"
