"""Row-exclusion rules shared by UMAP and its parameter search.

Settings arrive as hand-written text as often as as a dict, and the values
inside them come from a SQLite text column while the frame holds whatever
pandas inferred. Both halves are driven here against real DataFrames.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.row_exclusions import exclude_matching_rows, normalize_row_exclusions


# --------------------------------------------------------------------------
# normalize_row_exclusions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", {}, [], "   "])
def test_an_absent_rule_is_no_exclusions(value):
    """Blank settings mean the whole table is kept."""
    assert normalize_row_exclusions(value) == {}


@pytest.mark.parametrize("text", ["none", "None", "NULL", "null", " none "])
def test_the_words_none_and_null_mean_no_exclusions(text):
    """A settings file that literally says "none" is not a column called none."""
    assert normalize_row_exclusions(text) == {}


def test_json_text_is_accepted():
    """The settings file stores this as JSON, so JSON must parse."""
    assert normalize_row_exclusions('{"columnID": ["c1", "c2"]}') == {
        "columnID": ["c1", "c2"]}


def test_python_literal_text_is_accepted():
    """Hand-written settings use single quotes; that is not a mistake."""
    assert normalize_row_exclusions("{'columnID': ['c1', 'c2']}") == {
        "columnID": ["c1", "c2"]}


def test_text_that_is_neither_json_nor_a_literal_says_what_is_wanted():
    """The error shows the shape to type rather than a parser's complaint."""
    with pytest.raises(ValueError, match=r"\{'columnID': \['c1', 'c2'\]\}"):
        normalize_row_exclusions("columnID = c1, c2")


def test_something_that_is_not_a_mapping_is_refused():
    """A bare list of values does not say which column they belong to."""
    with pytest.raises(ValueError, match="must map column names to values"):
        normalize_row_exclusions(["c1", "c2"])
    with pytest.raises(ValueError, match="must map column names to values"):
        normalize_row_exclusions("[1, 2, 3]")


def test_a_scalar_value_becomes_a_one_item_list():
    """Hand-written settings stay convenient: one value need not be a list."""
    assert normalize_row_exclusions({"columnID": "c1"}) == {"columnID": ["c1"]}


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_any_container_of_values_is_accepted(container):
    """A set from a widget and a list from a file mean the same thing."""
    assert normalize_row_exclusions(
        {"columnID": container(["c1"])}) == {"columnID": ["c1"]}


def test_a_blank_column_name_is_dropped():
    """An empty key cannot name a column, so it is skipped rather than raised."""
    assert normalize_row_exclusions({"  ": ["c1"], "columnID": ["c2"]}) == {
        "columnID": ["c2"]}


def test_a_column_with_no_values_left_is_dropped():
    """A column mapped to nothing excludes nothing and is not carried forward."""
    assert normalize_row_exclusions({"columnID": []}) == {}


def test_repeated_values_are_deduplicated_but_types_are_kept_apart():
    """``1`` and ``"1"`` are different exclusions and both survive."""
    assert normalize_row_exclusions({"col": [1, 1, "1", "1", True]}) == {
        "col": [1, "1", True]}


def test_the_column_name_is_stripped():
    """Trailing whitespace in a settings key is a typo, not a different column."""
    assert normalize_row_exclusions({"  columnID  ": ["c1"]}) == {
        "columnID": ["c1"]}


# --------------------------------------------------------------------------
# exclude_matching_rows
# --------------------------------------------------------------------------


@pytest.fixture
def frame():
    """A small object table with a text column, a numeric one and a gap."""
    return pd.DataFrame({
        "columnID": ["c1", "c2", "c3", "c1"],
        "rowID": ["r1", "r2", "r3", "r4"],
        "plate": [1, 2, 3, 1],
        "grade": [1.0, 2.0, None, 4.0],
    })


def test_no_rules_returns_the_frame_untouched(frame):
    """With nothing to exclude the caller gets its own object back."""
    filtered, notes = exclude_matching_rows(frame, None)
    assert filtered is frame
    assert notes == []


def test_matching_rows_are_dropped_and_counted(frame):
    """The note names the column, the values, and how many rows went."""
    filtered, notes = exclude_matching_rows(frame, {"columnID": ["c1"]})
    assert list(filtered["columnID"]) == ["c2", "c3"]
    assert notes == ["Excluded 2 row(s) where columnID is one of ['c1']."]


def test_a_text_value_matches_a_numeric_column(frame):
    """A value chosen from a SQLite text column still matches pandas' integers."""
    filtered, _ = exclude_matching_rows(frame, {"plate": ["1"]})
    assert list(filtered["plate"]) == [2, 3]


def test_an_identifier_that_looks_numeric_is_not_renumbered():
    """``"001"`` stays a string, so it does not match the integer one."""
    frame = pd.DataFrame({"wellID": ["001", "1", "010"]})
    filtered, _ = exclude_matching_rows(frame, {"wellID": ["001"]})
    assert list(filtered["wellID"]) == ["1", "010"]


def test_missing_values_are_excluded_when_the_rule_asks_for_them(frame):
    """A rule naming null/NaN removes the rows where the column has no value."""
    filtered, notes = exclude_matching_rows(frame, {"grade": [None]})
    assert list(filtered["rowID"]) == ["r1", "r2", "r4"]
    assert "Excluded 1 row(s)" in notes[0]


@pytest.mark.parametrize("token", ["none", "NULL", "nan", "<NA>"])
def test_the_written_forms_of_a_missing_value_also_match(frame, token):
    """A dropdown offering "nan" as a value means the same as an actual gap."""
    filtered, _ = exclude_matching_rows(frame, {"grade": [token]})
    assert list(filtered["rowID"]) == ["r1", "r2", "r4"]


def test_a_row_removed_twice_is_counted_once(frame):
    """Two rules hitting the same row do not report it twice."""
    _, notes = exclude_matching_rows(
        frame, {"columnID": ["c1"], "rowID": ["r1"]})
    assert notes[0].startswith("Excluded 2 row(s) where columnID")
    assert notes[1].startswith("Excluded 0 row(s) where rowID")


def test_an_unknown_column_lists_the_ones_that_exist(frame):
    """The error is actionable: it names the columns available to exclude by."""
    with pytest.raises(ValueError) as excinfo:
        exclude_matching_rows(frame, {"nope": ["x"]})
    message = str(excinfo.value)
    assert "unknown column(s): ['nope']" in message
    assert "columnID" in message and "grade" in message
    assert "…" not in message


def test_a_very_wide_table_truncates_the_column_list():
    """A frame with hundreds of features names the first twenty and elides."""
    wide = pd.DataFrame({f"feature_{i}": [1, 2] for i in range(30)})
    with pytest.raises(ValueError) as excinfo:
        exclude_matching_rows(wide, {"nope": ["x"]})
    message = str(excinfo.value)
    assert message.rstrip().endswith("…")
    assert "feature_19" in message
    assert "feature_20" not in message


def test_excluding_everything_is_refused(frame):
    """A rule that empties the table is a mistake, not an empty UMAP."""
    with pytest.raises(ValueError, match="removed every UMAP object"):
        exclude_matching_rows(frame, {"columnID": ["c1", "c2", "c3"]})


def test_the_filtered_frame_is_a_copy(frame):
    """The caller may write to the result without touching the input frame."""
    filtered, _ = exclude_matching_rows(frame, {"columnID": ["c1"]})
    filtered.loc[filtered.index[0], "rowID"] = "changed"
    assert list(frame["rowID"]) == ["r1", "r2", "r3", "r4"]
