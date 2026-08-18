"""A column that is not there names the columns that are.

Instruction 135, asked for on 2026-08-17: "if dependent variable is not found
in the score table then present the user with the columns in the score csvs so
the user can choose which column ... also if the count columns ar not found id
like similar behaviour".

The failure this replaces: a misnamed response column survives every early
check and dies inside the merge, after the whole score table has been read,
with a message naming a column the file does not have and saying nothing about
what it does have.
"""

import os

import pandas as pd
import pytest

from spacr import columns as C


@pytest.fixture
def screen(tmp_path):
    """Two plates with the same header, as a real `score_data` list is."""
    one = tmp_path / "plate1_scores.csv"
    two = tmp_path / "plate2_scores.csv"
    pd.DataFrame({"prcfo": [], "predictions": [],
                  "cv_predictions": []}).to_csv(one, index=False)
    pd.DataFrame({"prcfo": [], "predictions": [],
                  "object_count": []}).to_csv(two, index=False)
    return [str(one), str(two)]


def test_only_the_header_row_is_read(screen, monkeypatch):
    """A score CSV is hundreds of megabytes and this runs on the GUI thread."""
    seen = {}
    real = pd.read_csv

    def spy(path, **kwargs):
        seen[os.path.basename(str(path))] = kwargs.get("nrows", "ALL")
        return real(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    C.available(screen)
    assert set(seen.values()) == {0}


def test_the_columns_are_gathered_across_every_file_in_order(screen):
    assert C.available(screen) == ["prcfo", "predictions", "cv_predictions",
                                   "object_count"]


def test_a_column_that_is_there_resolves_to_itself(screen):
    assert C.resolve("predictions", screen) == "predictions"


def test_a_case_slip_resolves_to_the_files_spelling(screen):
    """`Predictions` is not a different column from `predictions`.

    Failing on it teaches a user to distrust the message rather than fix the
    name.
    """
    assert C.resolve("Predictions", screen) == "predictions"
    assert C.resolve("CV_Predictions", screen) == "cv_predictions"


def test_a_missing_column_raises_with_every_column_it_could_have_been(screen):
    with pytest.raises(C.ColumnNotFound) as raised:
        C.resolve("prediction", screen, what="response column",
                  setting="dependent_variable")
    error = raised.value
    assert error.available == C.available(screen)
    assert error.name == "prediction"
    text = str(error)
    assert "dependent_variable='prediction'" in text
    assert "response column" in text
    for column in C.available(screen):
        assert repr(column) in text


def test_the_near_miss_is_offered_first_and_the_list_still_follows(screen):
    """A suggestion that is wrong and a list that is absent is worse than no
    suggestion at all."""
    text = C.describe("prediction", screen)
    assert "Did you mean 'predictions'" in text
    assert "column(s) available" in text


def test_a_name_nothing_resembles_gets_no_suggestion(screen):
    """`plate` must not be offered for `parasite`."""
    assert C.suggest("zzzzzzzz", C.available(screen)) == []
    assert "Did you mean" not in C.describe("zzzzzzzz", screen)


def test_at_most_three_suggestions(tmp_path):
    path = tmp_path / "many.csv"
    pd.DataFrame({f"prediction{i}": [] for i in range(8)}).to_csv(
        path, index=False)
    assert len(C.suggest("prediction", [str(path)])) <= C.MAX_SUGGESTIONS


def test_it_is_a_keyerror_so_existing_handlers_still_catch_it(screen):
    with pytest.raises(KeyError):
        C.resolve("nope", screen)


# ---------------------------------------------------------------------------
# A missing FILE is not a missing COLUMN
# ---------------------------------------------------------------------------

def test_an_unreadable_file_is_reported_apart_from_the_columns(screen,
                                                               tmp_path):
    absent = str(tmp_path / "plate3_scores.csv")
    text = C.describe("nope", screen + [absent])
    assert "Not read: plate3_scores.csv" in text
    # And the columns that WERE readable are still offered.
    assert "'predictions'" in text


def test_missing_names_only_the_paths_that_failed(screen, tmp_path):
    absent = str(tmp_path / "gone.csv")
    assert C.missing(screen + [absent]) == [absent]
    assert C.missing(screen) == []


def test_a_file_being_written_is_unreadable_not_empty(tmp_path):
    """Absent from `headers`, not mapped to []. Rule 2.

    A zero-byte CSV is what a run that is still writing its scores looks
    like, and pandas raises EmptyDataError on it. Mapping it to `[]` would
    make "this file has no columns yet" indistinguishable from "this file
    has no columns", and the user would be shown an empty list of choices
    as though it were the answer.
    """
    half_written = tmp_path / "scores.csv"
    half_written.write_bytes(b"")
    assert str(half_written) not in C.headers([str(half_written)])
    assert C.missing([str(half_written)]) == [str(half_written)]
    text = C.describe("x", [str(half_written)])
    assert "could not be checked" in text
    assert "scores.csv" in text


def test_no_readable_file_says_so_rather_than_listing_nothing(tmp_path):
    """"no column X in no file" would read as "the file has no columns"."""
    absent = str(tmp_path / "gone.csv")
    text = C.describe("y", [absent], setting="dependent_variable")
    assert "could not be checked" in text
    assert "gone.csv" in text


def test_no_paths_at_all_says_that_instead(screen):
    assert "no input CSV was given" in C.describe("y", None)
    assert "no input CSV was given" in C.describe("y", [])


def test_a_tilde_path_is_expanded(tmp_path, monkeypatch):
    """The same expansion the readers do; GitHub issue #108 was this bug."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "scores.csv"
    pd.DataFrame({"a": [], "b": []}).to_csv(path, index=False)
    assert C.available("~/scores.csv") == ["a", "b"]


def test_a_single_path_needs_no_list(screen):
    assert C.available(screen[0]) == ["prcfo", "predictions", "cv_predictions"]


def test_an_empty_name_asks_for_nothing_and_gets_the_list(screen):
    assert C.suggest("", C.available(screen)) == []
    with pytest.raises(C.ColumnNotFound) as raised:
        C.resolve(None, screen)
    assert raised.value.available == C.available(screen)
