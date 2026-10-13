"""A rate whose numerator and denominator were filtered differently.

INSTRUCTION 377, PART 1, in as many words: "edge-object exclusion -- already
implemented for measurement; the infection denominator has to use the same
rule or the rate is computed against a different cell count than the
numerator".

`remove_border_objects` is applied at SEGMENTATION time -- `clear_border(mask)`
in `utils.py` -- and it is PER OBJECT TYPE. `cell_remove_border_objects` and
`pathogen_remove_border_objects` are independent settings, so they decide what
is in each table rather than how a table is counted:

  * pathogens cleared, cells kept -- a host cell whose only parasite touched
    the edge is still in the cell table with a count of zero. It reads as
    UNINFECTED and deflates the rate.
  * cells cleared, pathogens kept -- a parasite names a cell that is not in
    the cell table, so it counts in neither numerator nor denominator.

THIS IS THE SAME FAILURE AS `include_uninfected`, WHICH ALREADY SHIPPED: a
setting that silently changes what a table contains, and therefore what a
ratio over it means, with nothing in the output saying so. That one turned
every well into 1.000000 on a real plate and was caught only when 377 met a
real database. This one is quieter, which is worse -- a deflated rate looks
like biology rather than like a bug.
"""
from __future__ import annotations

import sqlite3

import pytest

pd = pytest.importorskip("pandas")

from spacr import infection


def _db(tmp_path, *, cell_border=None, pathogen_border=None):
    """A measurements.db recording whichever border settings are given."""
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
               "fieldID TEXT, object_label INTEGER)")
    db.execute("CREATE TABLE pathogen (plateID TEXT, rowID TEXT, "
               "columnID TEXT, fieldID TEXT, object_label INTEGER, "
               "cell_id INTEGER)")
    db.execute("CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")
    db.execute("INSERT INTO settings VALUES ('include_uninfected', 'True')")
    for key, value in (("cell_remove_border_objects", cell_border),
                       ("pathogen_remove_border_objects", pathogen_border)):
        if value is not None:
            db.execute("INSERT INTO settings VALUES (?, ?)",
                       (key, str(value)))
    for label in (1, 2, 3, 4):
        db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',?)", (label,))
    for label in (1, 2):
        db.execute("INSERT INTO pathogen VALUES ('p1','A','1','f1',?,?)",
                   (label, label))
    db.commit()
    db.close()
    return str(path)


@pytest.mark.parametrize("cell,pathogen", [(True, True), (False, False)])
def test_matching_border_rules_agree(tmp_path, cell, pathogen):
    """Both cleared or neither cleared is one population, counted once."""
    assert infection.border_rules_agree(
        _db(tmp_path, cell_border=cell, pathogen_border=pathogen)) is True


@pytest.mark.parametrize("cell,pathogen", [(True, False), (False, True)])
def test_differing_border_rules_are_reported(tmp_path, cell, pathogen):
    """Either direction is a mismatch; neither is safe to assume away."""
    assert infection.border_rules_agree(
        _db(tmp_path, cell_border=cell, pathogen_border=pathogen)) is False


def test_a_database_that_recorded_neither_says_so_rather_than_guessing(
        tmp_path):
    """AN OLDER DATABASE CANNOT BE INTERROGATED, and saying "they agree" would
    be an assumption wearing the costume of a measurement."""
    assert infection.border_rules_agree(_db(tmp_path)) is None


@pytest.mark.parametrize("cell,pathogen", [(True, None), (None, True)])
def test_half_a_pair_is_not_an_agreement(tmp_path, cell, pathogen):
    """ONE RECORDED AND ONE MISSING IS NOT "THEY MATCH".

    The missing key defaults to False in `settings.py`, so assuming it would
    usually be right -- and that is the trap. A database that recorded only
    half the pair may predate the other key entirely, and a rate reported as
    trustworthy on that assumption is exactly the shape of the bug this whole
    check exists to stop.
    """
    assert infection.border_rules_agree(
        _db(tmp_path, cell_border=cell, pathogen_border=pathogen)) is None


def test_the_report_says_so_in_the_denominator_a_reader_is_already_reading(
        tmp_path):
    """WHERE THE WARNING GOES IS THE POINT.

    This module's rule is that every row carries the population it was
    computed over, so that a reader who disagrees with a number can see which
    half they disagree with. A caveat in a log line is one nobody reads; the
    denominator column is where they are already looking.
    """
    report = infection.infection_report(
        _db(tmp_path, cell_border=True, pathogen_border=False))
    rate = report[report["metric"] == "infection_rate"]
    assert not rate.empty, report

    said = " ".join(rate["denominator"].astype(str))
    assert "border" in said.lower(), (
        f"the rate was reported with no hint that its numerator and "
        f"denominator are different populations: {said!r}")


def test_a_matching_run_is_not_warned_about(tmp_path):
    """THE NEGATIVE HALF. A warning on every report is a warning nobody
    reads, and this one must appear only when the rules actually differ."""
    report = infection.infection_report(
        _db(tmp_path, cell_border=False, pathogen_border=False))
    said = " ".join(report["denominator"].astype(str))
    assert "border" not in said.lower(), (
        f"a run whose border rules agree was warned about anyway: {said!r}")
