"""An infection rate of 1.000 that means "we did not look", not "everything was infected".

MEASURED ON REAL DATA, which is what this file exists to record. Run against
`tsg101_screen/plate1/measurements/measurements.db` -- 60,816 cells across
2,052 report rows -- `infection_report` returned `infection_rate = 1.000000`
for EVERY well. The arithmetic was right. The database's `settings` table says

    include_uninfected   False

so Measure never wrote a row for a cell that had no parasite. The cell table
is a SELECTION OF INFECTED CELLS, not the segmented population, and
`infected / cells` is 1.0 by construction no matter what the biology did.

THE OLD FIXTURES COULD NOT HAVE CAUGHT THIS. They build a cell table holding
uninfected cells and no `settings` table at all, which is the shape of a run
with `include_uninfected=True` -- the case that already worked. The failure
needs a database that says how it was made, and until 377 met a real one
there wasn't one.

So the rate is refused rather than printed, in the same voice the module uses
for the three metrics it cannot derive: absent, with the reason in the
denominator column where a reader is already looking.
"""
from __future__ import annotations

import sqlite3

import pytest

pd = pytest.importorskip("pandas")

from spacr import infection


def _db(tmp_path, include_uninfected, *, with_settings=True):
    """A measurements.db shaped by how the Measure run was configured."""
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
               "fieldID TEXT, object_label INTEGER)")
    db.execute("CREATE TABLE pathogen (plateID TEXT, rowID TEXT, "
               "columnID TEXT, fieldID TEXT, object_label INTEGER, "
               "cell_id INTEGER)")
    if with_settings:
        db.execute("CREATE TABLE settings (setting_key TEXT, "
                   "setting_value TEXT)")
        db.execute("INSERT INTO settings VALUES ('include_uninfected', ?)",
                   (str(include_uninfected),))
    # Four cells. With uninfected excluded, only the two infected ones exist.
    cells = (1, 2) if include_uninfected is False else (1, 2, 3, 4)
    for label in cells:
        db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',?)", (label,))
    for label in (1, 2):
        db.execute("INSERT INTO pathogen VALUES ('p1','A','1','f1',?,?)",
                   (label, label))
    db.commit()
    db.close()
    return str(path)


def _rate(frame):
    return frame[frame["metric"] == "infection_rate"].iloc[0]


def test_the_setting_is_read_back_off_the_database(tmp_path):
    kept = tmp_path / "kept"
    kept.mkdir()
    assert infection.uninfected_cells_were_measured(
        _db(tmp_path, False)) is False
    assert infection.uninfected_cells_were_measured(
        _db(kept, True)) is True


def test_an_older_database_without_the_setting_says_it_cannot_tell(tmp_path):
    """``None``, not ``False``. A database that never recorded the setting is
    not evidence that uninfected cells were dropped, and guessing either way
    would either refuse a good rate or print a bad one."""
    (tmp_path / "old").mkdir()
    assert infection.uninfected_cells_were_measured(
        _db(tmp_path / "old", True, with_settings=False)) is None


def test_the_rate_is_refused_when_uninfected_cells_were_never_written(tmp_path):
    frame = infection.infection_report(_db(tmp_path, False))
    row = _rate(frame)
    assert pd.isna(row["value"]), "a rate of 1.0 here would be an artefact"
    assert "include_uninfected=False" in row["denominator"]


def test_the_counts_survive_and_say_what_they_counted(tmp_path):
    """Refusing the ratio must not throw away the true numbers beside it."""
    frame = infection.infection_report(_db(tmp_path, False))
    counts = frame.set_index("metric")
    assert counts.loc["cell_count", "value"] == 2.0
    assert counts.loc["parasite_count", "value"] == 2.0
    assert "uninfected excluded" in counts.loc["cell_count", "denominator"]


def test_a_real_population_still_gets_its_rate(tmp_path):
    """The guard must not fire on the case it was never about: four cells,
    two infected, and the rate is 0.5 rather than refused."""
    frame = infection.infection_report(_db(tmp_path, True))
    row = _rate(frame)
    assert row["value"] == pytest.approx(0.5)
    assert row["denominator"] == "segmented host cells"


def test_an_older_database_still_gets_its_rate(tmp_path):
    """``None`` behaves as it always did, so no existing database changes
    its answer because this guard landed."""
    (tmp_path / "old").mkdir()
    frame = infection.infection_report(
        _db(tmp_path / "old", True, with_settings=False))
    assert _rate(frame)["value"] == pytest.approx(0.5)
