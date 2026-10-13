"""Item 377's surface: Measure emits the infection report.

The maintainer chose this on 2026-09-20 over a button or a Report section,
and the reason is in the choice: the runs that most need these numbers are
the ones that would never have thought to ask for them.

The numbers themselves are tested in
tests/test_infection_metrics_state_their_denominator.py. What is held here
is that a finished run leaves the file, that it leaves none when there is
nothing to say, and that failing to write one cannot cost anybody a
measure run that has already written its database.
"""

from __future__ import annotations

import inspect
import os
import sqlite3

import pytest

pd = pytest.importorskip("pandas")

from spacr import infection

PARASITES = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}


@pytest.fixture
def measured(tmp_path):
    """A measurements.db with one well and something to report."""
    folder = tmp_path / "measurements"
    folder.mkdir()
    path = folder / "measurements.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
        "fieldID TEXT, object_label INTEGER, area REAL, eccentricity REAL)")
    db.execute(
        "CREATE TABLE pathogen (plateID TEXT, rowID TEXT, columnID TEXT, "
        "fieldID TEXT, object_label INTEGER, cell_id INTEGER, area REAL)")
    parasite = 0
    for cell, count in PARASITES.items():
        db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',?,?,?)",
                   (cell, 900.0 if count else 600.0, 0.5))
        for _ in range(count):
            parasite += 1
            db.execute("INSERT INTO pathogen VALUES ('p1','A','1','f1',?,?,?)",
                       (parasite, cell, 50.0))
    db.commit()
    db.close()
    return str(path)


@pytest.fixture
def thin(tmp_path):
    """A database with no cell table at all."""
    folder = tmp_path / "thin"
    folder.mkdir()
    path = folder / "measurements.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE unrelated (a INTEGER)")
    db.commit()
    db.close()
    return str(path)


def test_the_report_lands_beside_the_database_it_came_from(measured):
    written = infection.write_infection_report(measured)
    assert written == os.path.join(os.path.dirname(measured),
                                   infection.REPORT_NAME)
    assert os.path.isfile(written)
    frame = pd.read_csv(written)
    assert not frame.empty
    for column in ("metric", "value", "denominator", "n_denominator"):
        assert column in frame.columns, column


def test_every_row_still_carries_its_denominator(measured):
    """The property the report exists for survives the trip through CSV."""
    frame = pd.read_csv(infection.write_infection_report(measured))
    assert frame["denominator"].notna().all()
    assert frame["n_denominator"].notna().all()


def test_nothing_to_say_leaves_no_file(thin):
    """An empty CSV beside a database invites somebody to wonder what went
    wrong; None says it plainly instead."""
    assert infection.write_infection_report(thin) is None
    assert not os.path.isfile(
        os.path.join(os.path.dirname(thin), infection.REPORT_NAME))


def test_it_is_written_whole_or_not_at_all(measured, monkeypatch):
    seen = []
    original = os.replace

    def _watch(source, target):
        seen.append((os.path.basename(source), os.path.basename(target)))
        return original(source, target)

    monkeypatch.setattr(os, "replace", _watch)
    infection.write_infection_report(measured)
    assert seen and seen[-1][0].startswith(".")
    assert seen[-1][1] == infection.REPORT_NAME


def test_a_destination_can_be_named(measured, tmp_path):
    target = tmp_path / "elsewhere" / "report.csv"
    written = infection.write_infection_report(measured,
                                               destination=str(target))
    assert written == str(target) and target.is_file()


def test_a_finished_measure_run_emits_it():
    """The call site, which is what makes this item's answer true."""
    from spacr import measure

    source = inspect.getsource(measure.measure_crop)
    assert "_emit_infection_report(db_path)" in source
    assert "Successfully completed run" in source


def test_a_report_is_not_worth_a_run(tmp_path, monkeypatch, capsys):
    """A database is already written by the time this runs. Nothing about a
    report may turn that into a failure."""
    from spacr import measure

    path = tmp_path / "measurements.db"
    path.write_text("not a database")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("pandas said no")

    monkeypatch.setattr(infection, "write_infection_report", _boom)
    measure._emit_infection_report(str(path))
    assert "could not be written" in capsys.readouterr().out


def test_a_run_with_no_database_says_nothing(tmp_path, capsys):
    from spacr import measure

    measure._emit_infection_report(str(tmp_path / "absent.db"))
    assert capsys.readouterr().out == ""
