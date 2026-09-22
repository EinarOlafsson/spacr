"""What the infection report does when the database cannot answer it.

Every function here is a REPORT over somebody else's finished Measure run, so
the inputs are not under its control: a project measured without a pathogen
channel has no pathogen table, a database written before the setting existed
has no settings table, and a column spaCR renamed years ago may be spelled
either way.  None of those is an error to raise into a report -- a traceback
out of a report button says "spaCR is broken" when what happened is "this run
did not measure that".

So the contract tested here is the empty one: an empty frame with the right
shape, or ``None`` for "we cannot tell", and never an exception.  The
numbers, against a database whose answers we chose, are in
``tests/test_infection_metrics_state_their_denominator.py``; these are the
cases where there is no number to give.
"""
from __future__ import annotations

import sqlite3

import pytest

pd = pytest.importorskip("pandas")

from spacr import infection


def _empty_db(tmp_path, name="measurements.db"):
    """A database file with no tables at all."""
    path = tmp_path / name
    sqlite3.connect(path).close()
    return str(path)


def _settings_db(tmp_path, rows):
    """A database with a settings table holding exactly ``rows``."""
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")
    db.executemany("INSERT INTO settings VALUES (?, ?)", rows)
    db.commit()
    db.close()
    return str(path)


# -- "we cannot tell" is a third answer, and it is not False ----------------


def test_a_database_written_before_the_setting_existed_says_it_cannot_tell():
    """None, not False.

    False would mean "this run did NOT include uninfected cells", which
    makes :func:`infection_report` refuse the rate outright.  An older
    database did not record the setting either way, and refusing every
    older run's rate on that basis would be a lie about what was measured.
    """
    assert infection.uninfected_cells_were_measured("/no/such/file.db") is None


def test_a_database_with_no_settings_table_says_it_cannot_tell(tmp_path):
    path = _empty_db(tmp_path)

    assert infection.uninfected_cells_were_measured(path) is None
    assert infection.border_rules_agree(path) is None


def test_a_setting_recorded_as_null_says_it_cannot_tell(tmp_path):
    """A row that exists with no value is as unknown as no row at all."""
    path = _settings_db(
        tmp_path, [(infection.INCLUDE_UNINFECTED_KEY, None)])

    assert infection.uninfected_cells_were_measured(path) is None


def test_a_setting_nobody_wrote_says_it_cannot_tell(tmp_path):
    path = _settings_db(tmp_path, [("some_other_setting", "True")])

    assert infection.uninfected_cells_were_measured(path) is None


@pytest.mark.parametrize(
    ("stored", "expected"),
    [("True", True), ("true", True), (" TRUE ", True), ("1", True),
     ("yes", True), ("False", False), ("0", False), ("", False),
     ("maybe", False)],
)
def test_the_setting_is_read_the_way_the_run_spelled_it(
    tmp_path, stored, expected,
):
    """Settings are stored as text by whichever writer got there first."""
    path = _settings_db(
        tmp_path, [(infection.INCLUDE_UNINFECTED_KEY, stored)])

    assert infection.uninfected_cells_were_measured(path) is expected


def test_border_rules_agree_only_when_both_halves_were_recorded(tmp_path):
    """Two settings, and the comparison needs both.

    One recorded and one missing is not "they disagree"; it is "we cannot
    tell", and the difference decides whether the report prints a warning
    the reader would otherwise act on.
    """
    def _in(name, rows):
        folder = tmp_path / name
        folder.mkdir()
        return _settings_db(folder, rows)

    both = _in("both", [
        (infection.CELL_BORDER_KEY, "True"),
        (infection.PATHOGEN_BORDER_KEY, "True"),
    ])
    differ = _in("differ", [
        (infection.CELL_BORDER_KEY, "True"),
        (infection.PATHOGEN_BORDER_KEY, "False"),
    ])
    half = _in("half", [
        (infection.CELL_BORDER_KEY, "True"),
    ])

    assert infection.border_rules_agree(both) is True
    assert infection.border_rules_agree(differ) is False
    assert infection.border_rules_agree(half) is None


# -- an absent table is a fact about the run, not an error ------------------


def test_every_report_over_an_empty_database_is_an_empty_frame(tmp_path):
    """A project measured without a pathogen channel opens the report too."""
    path = _empty_db(tmp_path)

    assert infection.parasites_per_cell(path).empty
    assert infection.infection_report(path).empty
    assert infection.host_contrast(path, ["area"]).empty

    distribution = infection.multiplicity_distribution(path)
    assert distribution.empty
    assert list(distribution.columns) == ["pathogen_count", "cells", "fraction"]


def test_a_cell_table_without_a_label_column_cannot_be_joined(tmp_path):
    """The object label is the join key; without it there is nothing to join.

    A frame built anyway would have one row per cell and a parasite count of
    zero for all of them -- an infection rate of 0.000 that looks measured.
    """
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, area REAL)")
    db.execute("INSERT INTO cell VALUES ('p1', 'A', 600.0)")
    db.commit()
    db.close()

    assert infection.parasites_per_cell(str(path)).empty


def test_host_contrast_asked_for_columns_the_cells_do_not_have_is_empty(
    tmp_path,
):
    """Asked to contrast a measurement nothing measured, it contrasts nothing."""
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
        "fieldID TEXT, object_label INTEGER, area REAL)")
    db.execute(
        "CREATE TABLE pathogen (plateID TEXT, rowID TEXT, columnID TEXT, "
        "fieldID TEXT, object_label INTEGER, cell_id INTEGER)")
    db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',1,600.0)")
    db.execute("INSERT INTO pathogen VALUES ('p1','A','1','f1',1,1)")
    db.commit()
    db.close()

    assert not infection.parasites_per_cell(str(path)).empty
    assert infection.host_contrast(str(path), ["no_such_measurement"]).empty
    assert not infection.host_contrast(str(path), ["area"]).empty


# -- the identity columns spaCR has spelled two ways ------------------------


def test_an_identity_column_that_is_absent_is_reported_absent():
    """``_canonical`` returns None rather than the name it was asked for.

    The caller distinguishes "this table calls it plateID" from "this table
    has no plate column"; handing back the canonical spelling for the second
    would put a column name into a SELECT that cannot run.
    """
    assert infection._canonical(["area", "object_label"], "plateID") is None
    assert infection._canonical(["plate", "area"], "plateID") == "plate"
    assert infection._canonical(["PlateID"], "plateID") == "PlateID"


def test_the_columns_of_a_table_that_does_not_exist_are_none(tmp_path):
    db = sqlite3.connect(_empty_db(tmp_path))
    try:
        assert infection._present_columns(db, "cell") == []
    finally:
        db.close()


def test_loading_only_columns_the_table_lacks_gives_an_empty_frame(tmp_path):
    """Restricting to absent columns selects nothing, rather than SELECTing
    a name SQLite would read as a string literal."""
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE cell (plateID TEXT, area REAL)")
    db.execute("INSERT INTO cell VALUES ('p1', 600.0)")
    db.commit()
    db.close()

    with sqlite3.connect(path) as opened:
        assert infection._load(opened, "cell", ["no_such"]).empty
        kept = infection._load(opened, "cell", ["area", "no_such"])
        assert list(kept.columns) == ["area"]
        assert infection._load(opened, "absent_table").empty
