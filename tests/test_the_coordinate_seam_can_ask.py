"""`selection_from_objects` asks when the object table names no object.

The prompt is INJECTED, not imported, so this module never depends on Qt --
which is what makes "never prompt in a batch run" structural rather than a
rule each call site has to remember.
"""

import sqlite3

import pandas as pd
import pytest

from spacr.stream_dataset import selection_from_objects


@pytest.fixture
def a_database(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cell (my_object_id TEXT, plateID TEXT, "
                   "rowID TEXT, columnID TEXT, fieldID TEXT)")
        db.executemany("INSERT INTO cell VALUES (?,?,?,?,?)",
                       [(str(i), "p1", "r1", "c1", "f1") for i in range(20)])
    return path


def _nameless():
    """A table with real measurements and nothing that names an object."""
    return pd.DataFrame({"area": [1.0, 2.0, 3.0], "intensity": [4, 5, 6]})


def test_a_caller_with_no_ask_behaves_exactly_as_before():
    """The batch run must not gain a dialog, so it must not gain a hook."""
    with pytest.raises(ValueError, match="names no object"):
        selection_from_objects(_nameless())


def test_it_asks_when_the_table_names_no_object(a_database):
    asked = {}

    def ask(*, tried, object_array):
        asked["tried"] = tried
        asked["object_array"] = object_array
        return (a_database, "cell", "my_object_id"), "using my_object_id"

    out = selection_from_objects(_nameless(), ask=ask)
    assert len(out) == 20
    assert "names no object" in asked["tried"]
    assert asked["object_array"] == "cell"


def test_the_answer_supplies_the_objects_that_get_streamed(a_database):
    def ask(**_kw):
        return (a_database, "cell", "my_object_id"), "using my_object_id"

    out = selection_from_objects(_nameless(), ask=ask)
    from spacr.schema import OBJECT_KEY

    assert sorted(out[OBJECT_KEY].astype(int)) == list(range(20))


def test_refusing_stops_the_run_with_both_reasons(a_database):
    def ask(**_kw):
        return None, "cancelled, so this run stops"

    with pytest.raises(ValueError) as raised:
        selection_from_objects(_nameless(), ask=ask)
    assert "names no object" in str(raised.value)
    assert "cancelled" in str(raised.value)


def test_an_answer_that_does_not_help_is_not_asked_about_twice(a_database):
    """One question per run: a wrong answer fails rather than looping."""
    calls = []

    def ask(**_kw):
        calls.append(1)
        return (a_database, "cell", "no_such_column"), "using no_such_column"

    with pytest.raises(ValueError, match="no 'no_such_column' column"):
        selection_from_objects(_nameless(), ask=ask)
    assert len(calls) == 1


def test_an_empty_chosen_table_says_so(tmp_path):
    path = str(tmp_path / "empty.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cell (my_object_id TEXT)")

    def ask(**_kw):
        return (path, "cell", "my_object_id"), "using my_object_id"

    with pytest.raises(ValueError, match="is empty"):
        selection_from_objects(_nameless(), ask=ask)


def test_an_unreadable_database_says_so(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not a database")

    def ask(**_kw):
        return (str(junk), "cell", "x"), "using x"

    with pytest.raises(ValueError, match="could not be read"):
        selection_from_objects(_nameless(), ask=ask)


def test_a_table_that_already_names_objects_never_asks():
    """A run that works today must not gain a dialog."""
    def ask(**_kw):
        raise AssertionError("nothing was missing, so nothing to ask")

    from spacr.schema import OBJECT_KEY

    frame = pd.DataFrame({OBJECT_KEY: ["1", "2", "3"], "area": [1, 2, 3]})
    assert len(selection_from_objects(frame, ask=ask)) == 3
