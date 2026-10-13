"""Infection numbers, against a database whose answers we chose.

Every metric here is a ratio, and almost every way of getting one wrong is a
denominator that does not match its numerator. The database below has ten
cells, four of them infected, carrying 1, 1, 2 and 3 parasites -- so the
answers are 0.4, 0.7 and 1.75 and they are written into the assertions rather
than computed by the same code under test.

THE ZEROES ARE THE POINT. Six of those ten cells have no parasites and
therefore no row in the pathogen table, because `measure.get_components`
explodes each cell's child list and drops the empty ones. A report that counts
rows there counts INFECTED cells and calls the result a cell count -- turning
an infection rate of 0.4 into 1.0, silently, on every plate.
"""
from __future__ import annotations

import sqlite3

import pytest

pd = pytest.importorskip("pandas")

from spacr import infection

#: Six uninfected, then four carrying 1, 1, 2 and 3.
PARASITES = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 2, 10: 3}
TOTAL_CELLS = len(PARASITES)                      # 10
INFECTED = sum(1 for n in PARASITES.values() if n)  # 4
TOTAL_PARASITES = sum(PARASITES.values())         # 7


@pytest.fixture
def measured(tmp_path):
    """A measurements.db with one well and known infection."""
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
        "fieldID TEXT, object_label INTEGER, area REAL, eccentricity REAL)"
    )
    db.execute(
        "CREATE TABLE pathogen (plateID TEXT, rowID TEXT, columnID TEXT, "
        "fieldID TEXT, object_label INTEGER, cell_id INTEGER, area REAL)"
    )
    parasite_label = 0
    for cell, count in PARASITES.items():
        # Infected cells are drawn larger, so the host contrast has something
        # real to find rather than noise.
        area = 900.0 if count else 600.0
        db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',?,?,?)",
                   (cell, area, 0.5))
        for _ in range(count):
            parasite_label += 1
            db.execute(
                "INSERT INTO pathogen VALUES ('p1','A','1','f1',?,?,?)",
                (parasite_label, cell, 50.0))
    db.commit()
    db.close()
    return str(path)


def _value(report, metric):
    """The one value for ``metric`` in a single-well report."""
    rows = report[report["metric"] == metric]
    assert len(rows) == 1, f"expected one {metric} row, got {len(rows)}"
    return float(rows["value"].iloc[0]), rows["n_denominator"].iloc[0]


def test_a_cell_with_no_parasites_is_still_a_cell(measured):
    """THE DENOMINATOR TEST. All ten cells must be counted, not just four.

    The relationship lives on the parasite, so an uninfected cell appears
    nowhere in the pathogen table. Counting there gives 4/4 = 1.0 and calls a
    40 % infection rate a total one.
    """
    per_cell = infection.parasites_per_cell(measured)
    assert len(per_cell) == TOTAL_CELLS, (
        f"{len(per_cell)} cells counted, not {TOTAL_CELLS} -- the uninfected "
        f"ones were dropped"
    )
    assert int((per_cell["pathogen_count"] == 0).sum()) == 6
    assert int(per_cell["pathogen_count"].sum()) == TOTAL_PARASITES


def test_the_three_headline_numbers_are_the_arithmetic_we_built(measured):
    """0.4, 0.7 and 1.75, each over the population it claims."""
    report = infection.infection_report(measured)

    rate, n = _value(report, "infection_rate")
    assert rate == pytest.approx(INFECTED / TOTAL_CELLS)      # 0.4
    assert n == TOTAL_CELLS

    index, n = _value(report, "infection_index")
    assert index == pytest.approx(TOTAL_PARASITES / TOTAL_CELLS)   # 0.7
    assert n == TOTAL_CELLS

    per_infected, n = _value(report, "parasites_per_infected")
    assert per_infected == pytest.approx(TOTAL_PARASITES / INFECTED)  # 1.75
    assert n == INFECTED, (
        "parasites-per-infected is over INFECTED cells, and the row must say so"
    )


def test_every_row_names_the_population_it_was_computed_over(measured):
    """A ratio without its denominator cannot be checked by a reader."""
    report = infection.infection_report(measured)
    assert not report.empty
    for column in ("metric", "value", "denominator", "n_denominator"):
        assert column in report.columns
    assert report["denominator"].notna().all()
    # The two rates share a denominator; the third does not. If they ever all
    # agree, somebody has quietly made them the same population.
    denominators = dict(zip(report["metric"], report["denominator"]))
    assert denominators["infection_rate"] == "segmented host cells"
    assert denominators["parasites_per_infected"] == "infected host cells"


def test_the_distribution_separates_what_the_mean_hides(measured):
    """Two conditions can share an index and differ completely."""
    dist = infection.multiplicity_distribution(measured)
    by_count = dict(zip(dist["pathogen_count"], dist["cells"]))
    assert by_count == {0: 6, 1: 2, 2: 1, 3: 1}
    assert dist["fraction"].sum() == pytest.approx(1.0)


def test_the_host_contrast_compares_within_a_well(measured):
    """Infected against uninfected in the SAME well, with both group sizes."""
    contrast = infection.host_contrast(measured, ("area",))
    assert len(contrast) == 1
    row = contrast.iloc[0]
    assert row["measurement"] == "area"
    assert row["infected"] == pytest.approx(900.0)
    assert row["uninfected"] == pytest.approx(600.0)
    assert row["n_infected"] == INFECTED
    assert row["n_uninfected"] == TOTAL_CELLS - INFECTED


def test_a_database_with_no_pathogen_table_says_uninfected_not_unknown(tmp_path):
    """An uninfected control plate is a real answer, not a missing one."""
    path = tmp_path / "no_pathogens.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
               "fieldID TEXT, object_label INTEGER)")
    for cell in range(1, 6):
        db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',?)", (cell,))
    db.commit(); db.close()

    report = infection.infection_report(str(path))
    rate, n = _value(report, "infection_rate")
    assert rate == 0.0
    assert n == 5


def test_a_database_with_no_cell_table_reports_nothing_rather_than_zero(tmp_path):
    """No cells is not an infection rate of zero; it is no denominator at all.

    Returning 0.0 would put a confident number on a plate that was never
    measured, which is worse than an empty report.
    """
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    assert infection.infection_report(str(path)).empty
    assert infection.parasites_per_cell(str(path)).empty
