"""Paths through the missing-column reporter that the message depends on.

The module's whole job is to turn "column not found" into "not that name,
one of these". These exercise the parts of that sentence that only appear
for a particular shape of input: a path handed over as a ``Path`` rather
than a string, a path with a shell variable in it, a name given as ``None``,
and the exception's own text, which a ``KeyError`` would otherwise print
wrapped in quotes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spacr import columns as C


@pytest.fixture
def one_plate(tmp_path):
    """A single score CSV with a header a user can misspell."""
    path = tmp_path / "plate1_scores.csv"
    pd.DataFrame({"prcfo": [], "predictions": []}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# The exception prints its sentence, not a quoted repr of it
# ---------------------------------------------------------------------------

def test_the_error_prints_its_sentence_without_the_keyerror_quotes():
    """``str(KeyError('a b'))`` is ``"'a b'"``. The message is a paragraph of
    prose naming files and columns, and printing it inside quotes with its
    newlines escaped is how a helpful message becomes an unreadable one."""
    error = C.ColumnNotFound("no column 'x' in scores.csv.",
                             name="x", available=["prcfo", "predictions"])

    assert str(error) == "no column 'x' in scores.csv."
    assert "\\n" not in str(error)


def test_the_error_carries_the_list_so_a_gui_need_not_reread_the_files():
    error = C.ColumnNotFound("...", name="predicton",
                             available=("prcfo", "predictions"))

    assert error.available == ["prcfo", "predictions"]
    assert error.name == "predicton"


def test_an_error_raised_with_no_detail_still_has_an_empty_list():
    error = C.ColumnNotFound("nothing to say")

    assert error.available == []
    assert error.name == ""
    assert str(error) == "nothing to say"


# ---------------------------------------------------------------------------
# What counts as a path
# ---------------------------------------------------------------------------

def test_a_pathlib_path_is_read_like_a_string_one(one_plate):
    """`score_data` reaches this from a Qt file picker, which hands out
    `Path` objects rather than strings."""
    assert isinstance(one_plate, Path)
    assert C.available(one_plate) == ["prcfo", "predictions"]
    assert C.resolve("predictions", [Path(one_plate)]) == "predictions"
    assert list(C.headers(one_plate)) == [str(one_plate)]


def test_a_blank_entry_in_the_path_list_is_dropped_not_reported_missing(
        one_plate):
    """An unfilled row in a settings list is an empty string. Treating it as
    a path names '' in the "could not be read" half of the message."""
    assert C.missing([str(one_plate), "", None]) == []
    assert C.available([str(one_plate), ""]) == ["prcfo", "predictions"]


def test_a_shell_variable_in_a_path_is_expanded(tmp_path, monkeypatch):
    """Settings CSVs written on a cluster carry $SCRATCH in their paths."""
    path = tmp_path / "scores.csv"
    pd.DataFrame({"prcfo": [], "score": []}).to_csv(path, index=False)
    monkeypatch.setenv("SPACR_TEST_SCORE_DIR", str(tmp_path))

    assert C.available("$SPACR_TEST_SCORE_DIR/scores.csv") == ["prcfo",
                                                               "score"]


def test_no_paths_at_all_reads_nothing_rather_than_raising():
    assert C.headers(None) == {}
    assert C.available(None) == []
    assert C.missing(None) == []


# ---------------------------------------------------------------------------
# The sentence itself
# ---------------------------------------------------------------------------

def test_the_setting_key_is_named_so_the_user_knows_which_box_to_change(
        one_plate):
    text = C.describe("predicton", one_plate, what="response column",
                      setting="dependent_variable")

    assert "dependent_variable='predicton'" in text
    assert "no response column" in text
    assert "'predictions'" in text


def test_without_a_setting_key_the_name_still_stands_alone(one_plate):
    text = C.describe("predicton", one_plate)

    assert text.startswith("no column 'predicton' in plate1_scores.csv.")


def test_a_directory_named_as_an_input_is_unreadable_not_columnless(tmp_path):
    """`os.path.isfile` is False for a directory, so it lands in the
    "could not be read" half rather than contributing no columns."""
    (tmp_path / "notafile").mkdir()

    assert C.headers(tmp_path / "notafile") == {}
    assert C.missing(tmp_path / "notafile") == [str(tmp_path / "notafile")]


def test_a_name_of_none_is_refused_with_the_list_and_an_empty_name(one_plate):
    """A settings key left unset arrives as None. It must not become the
    string 'None' in `.name`, which a GUI would then offer as a near-miss."""
    with pytest.raises(C.ColumnNotFound) as excinfo:
        C.resolve(None, one_plate)

    assert excinfo.value.name == ""
    assert excinfo.value.available == ["prcfo", "predictions"]
    assert "None" in str(excinfo.value)


def test_a_name_of_none_gets_no_suggestions():
    """`None` is an unset setting, not a name to look for near-misses of.

    ``'non'`` is what makes this worth asserting: difflib scores it 0.857
    against the string ``'none'``, well over the cutoff, so a missing guard
    would offer a real column as the correction for a name nobody typed.
    """
    assert C.suggest(None, ["non", "prcfo"]) == []
    assert C.suggest("none", ["non", "prcfo"]) == ["non"], (
        "the same column is a near-miss for anyone who does type the name")


def test_suggest_returns_the_spelling_the_file_uses():
    """The suggestion is the file's own header, not the lowercased key it was
    matched by: a user told to try 'predictions' when the CSV says
    'Predictions' has been sent to a column that is not there."""
    assert C.suggest("predictions", ["Predictions"]) == ["Predictions"]
    assert C.suggest("PREDICTIONS", ["prcfo", "Predictions"]) == ["Predictions"]


def test_every_column_is_offered_once_even_when_two_plates_share_a_header(
        tmp_path):
    for name in ("a_scores.csv", "b_scores.csv"):
        pd.DataFrame({"prcfo": [], "predictions": []}).to_csv(
            tmp_path / name, index=False)
    paths = [str(tmp_path / "a_scores.csv"), str(tmp_path / "b_scores.csv")]

    assert C.available(paths) == ["prcfo", "predictions"]
    assert C.describe("x", paths).count("'predictions'") == 1


def test_the_only_readable_file_is_the_one_the_message_names(tmp_path):
    good = tmp_path / "good.csv"
    pd.DataFrame({"prcfo": []}).to_csv(good, index=False)
    bad = tmp_path / "gone.csv"

    text = C.describe("predictions", [str(good), str(bad)])

    assert "in good.csv." in text
    assert "Not read: gone.csv." in text
    assert C.missing([str(good), str(bad)]) == [str(bad)]
