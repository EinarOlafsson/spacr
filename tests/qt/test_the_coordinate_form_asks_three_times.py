"""The database/table/column form from instruction 274.

Three answers rather than one, and the point of asking them one at a time
is that the second and third become answerable: the tables offered are the
ones the chosen database actually holds, and the columns are that table's.
A blank field would ask the user to remember a name the program can read.
"""

import os
import sqlite3

import pytest

from spacr.qt import ask_for_the_path as ASK


@pytest.fixture(autouse=True)
def _forget():
    ASK.forget()
    yield
    ASK.forget()


@pytest.fixture
def a_database(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cell (object_id INT, centroid_x REAL, "
                   "centroid_y REAL)")
        db.execute("CREATE TABLE nucleus (object_id INT, area REAL)")
    return path


@pytest.fixture
def person(monkeypatch):
    """A person is in front of the screen, for the length of one test."""
    monkeypatch.setattr(ASK, "somebody_is_there", lambda: True)


def _answers(database, table, column, seen=None):
    """A chooser and a picker that answer the three questions in order."""
    def chooser(title, start=""):
        if seen is not None:
            seen.append(title)
        return database

    def pick(title, prompt, options):
        if seen is not None:
            seen.append((prompt, tuple(options)))
        return table if prompt.startswith("Table") else column

    return chooser, pick


# --- what it reads out of the file ----------------------------------------


def test_it_lists_the_tables_the_database_actually_holds(a_database):
    assert ASK.tables_in(a_database) == ["cell", "nucleus"]


def test_it_lists_a_table_s_own_columns_in_declared_order(a_database):
    assert ASK.columns_in(a_database, "cell") == [
        "object_id", "centroid_x", "centroid_y"]


def test_a_file_that_is_not_a_database_is_a_sentence_not_a_crash(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a database")
    assert ASK.tables_in(str(junk)) == []


def test_a_missing_file_is_a_sentence_not_a_crash(tmp_path):
    assert ASK.tables_in(str(tmp_path / "gone.db")) == []
    assert ASK.columns_in(str(tmp_path / "gone.db"), "cell") == []


# --- the form itself -------------------------------------------------------


def test_it_returns_all_three_answers(a_database, person):
    chooser, pick = _answers(a_database, "cell", "centroid_x")
    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no coordinate column", chooser=chooser, pick=pick)
    assert answer == (a_database, "cell", "centroid_x")
    assert "centroid_x" in why and "cell" in why


def test_the_tables_offered_come_from_the_chosen_database(a_database, person):
    """The second question is answerable because the first was answered."""
    seen = []
    chooser, pick = _answers(a_database, "cell", "centroid_x", seen)
    ASK.ask_for_a_database_column("coords", tried="no column",
                                  chooser=chooser, pick=pick)
    offered = dict(entry for entry in seen if isinstance(entry, tuple))
    assert offered["Table in measurements.db"] == ("cell", "nucleus")
    assert offered["Column in cell"] == ("object_id", "centroid_x",
                                         "centroid_y")


def test_what_was_tried_is_shown_in_the_file_dialog(a_database, person):
    seen = []
    chooser, pick = _answers(a_database, "cell", "centroid_x", seen)
    ASK.ask_for_a_database_column("coords", tried="parasite.db has no x",
                                  chooser=chooser, pick=pick)
    assert "parasite.db has no x" in seen[0]


def test_it_asks_once_per_run(a_database, person):
    calls = []

    def chooser(title, start=""):
        calls.append(title)
        return a_database

    def pick(title, prompt, options):
        return "cell" if prompt.startswith("Table") else "centroid_x"

    for _ in range(3):
        answer, why = ASK.ask_for_a_database_column(
            "coords", tried="no column", chooser=chooser, pick=pick)
        assert answer == (a_database, "cell", "centroid_x")
    assert len(calls) == 1
    assert "chosen earlier in this run" in why


def test_two_different_things_are_still_asked_separately(a_database, person):
    calls = []

    def chooser(title, start=""):
        calls.append(title)
        return a_database

    def pick(title, prompt, options):
        return "cell" if prompt.startswith("Table") else "centroid_x"

    ASK.ask_for_a_database_column("x", tried="no x", chooser=chooser,
                                  pick=pick)
    ASK.ask_for_a_database_column("y", tried="no y", chooser=chooser,
                                  pick=pick)
    assert len(calls) == 2


def test_cancelling_the_file_stops_the_run_and_is_not_remembered(person):
    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no column", chooser=lambda *a, **k: "",
        pick=lambda *a, **k: None)
    assert answer is None
    assert "cancelled" in why
    assert ASK.remembered("coords") is None


def test_a_file_with_no_tables_keeps_the_form_open(tmp_path, a_database,
                                                   person):
    """Rejected IN the form, not accepted and failed one step later."""
    junk = tmp_path / "notes.txt"
    junk.write_text("not a database")
    offered = [str(junk), a_database]
    titles = []

    def chooser(title, start=""):
        titles.append(title)
        return offered.pop(0)

    def pick(title, prompt, options):
        return "cell" if prompt.startswith("Table") else "centroid_y"

    answer, _why = ASK.ask_for_a_database_column(
        "coords", tried="no column", chooser=chooser, pick=pick)
    assert answer == (a_database, "cell", "centroid_y")
    assert "holds no tables" in titles[1]


def test_backing_out_of_the_table_returns_to_the_database(a_database, person):
    """One wrong database costs a click, not the run."""
    chosen = []
    answers = [None, "nucleus"]

    def chooser(title, start=""):
        chosen.append(title)
        return a_database

    def pick(title, prompt, options):
        if prompt.startswith("Table"):
            return answers.pop(0)
        return "area"

    answer, _why = ASK.ask_for_a_database_column(
        "coords", tried="no column", chooser=chooser, pick=pick)
    assert answer == (a_database, "nucleus", "area")
    assert len(chosen) == 2


def test_backing_out_of_the_column_returns_to_the_database(a_database,
                                                           person):
    chosen = []
    columns = [None, "centroid_y"]

    def chooser(title, start=""):
        chosen.append(title)
        return a_database

    def pick(title, prompt, options):
        return "cell" if prompt.startswith("Table") else columns.pop(0)

    answer, _why = ASK.ask_for_a_database_column(
        "coords", tried="no column", chooser=chooser, pick=pick)
    assert answer == (a_database, "cell", "centroid_y")
    assert len(chosen) == 2


def test_it_never_appears_headless(a_database):
    """No `person` fixture, so this is the batch run that must not block."""
    def chooser(*_a, **_k):
        raise AssertionError("a batch run must never be shown a dialog")

    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no coordinate column", chooser=chooser,
        pick=chooser)
    assert answer is None
    assert "nobody to ask" in why
    assert "no coordinate column" in why


def test_a_test_process_is_not_a_person():
    assert not ASK.somebody_is_there()
