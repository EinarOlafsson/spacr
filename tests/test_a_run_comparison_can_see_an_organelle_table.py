"""A run-to-run comparison must be able to see an organelle table.

Instruction 76 asked for organelle, organelle_1, organelle_2 ... everywhere.
`merge_tables.OBJECT_TABLES` was built from
:data:`spacr.object_roles.ORGANELLE_ROLES` and so could always see them;
`run_compare.OBJECT_TABLES` and `gene_measurement_compare.OBJECT_TABLES` named
four fixed strings and could not.

The failure that omission causes is specific and quiet: comparing a run that
measured organelles against one that did not reported *no difference between
the runs*, because the only tables either module looked at were the four it
had been told to look at. Nothing raised. The comparison simply had no opinion
about the rows that differed, which is the one thing it exists to have an
opinion about.

So the assertions here are mostly behavioural -- an actual SQLite file with an
actual organelle table, counted through the actual public entry point. A test
that only compared the constant against ORGANELLE_ROLES would pass just as
happily if `count_database` had stopped reading the constant.
"""

import sqlite3

import pytest

from spacr import gene_measurement_compare, merge_tables, run_compare
from spacr.object_roles import ORGANELLE_ROLES


def _database(path, tables):
    """Write a database holding `tables` as {name: row_count}, one plate."""
    connection = sqlite3.connect(str(path))
    try:
        for name, rows in tables.items():
            connection.execute(
                f'CREATE TABLE "{name}" (plate TEXT, well TEXT, object_label INTEGER)')
            connection.executemany(
                f'INSERT INTO "{name}" VALUES (?, ?, ?)',
                [("plate1", "A01", index) for index in range(rows)])
        connection.commit()
    finally:
        connection.close()
    return str(path)


def test_a_counted_run_reports_its_organelle_rows(tmp_path):
    """The count has to carry the organelle table, not just tolerate it."""
    path = _database(tmp_path / "measurements.db",
                     {"cell": 12, "organelle": 7, "organelleb": 3})
    counts = run_compare.count_database(path)

    # The point of the instruction: the organelle rows are COUNTED, and counted
    # separately per slot. Before the fix both keys were simply absent.
    assert counts.overall["organelle"] == 7
    assert counts.overall["organelleb"] == 3
    assert counts.overall["cell"] == 12


def test_a_run_that_wrote_organelles_differs_from_one_that_did_not(tmp_path):
    """The difference the module exists to report.

    This is the assertion that would have caught the original defect. With the
    four fixed names, both databases counted identically -- 12 cells each --
    and the comparison reported nothing, because the rows that differed were in
    a table neither side was reading.
    """
    with_organelles = _database(tmp_path / "a.db", {"cell": 12, "organelle": 7})
    without = _database(tmp_path / "b.db", {"cell": 12})

    a = run_compare.count_database(with_organelles)
    b = run_compare.count_database(without)

    assert "organelle" in a.overall
    assert "organelle" not in b.overall
    assert a.overall["organelle"] != b.overall.get("organelle", 0)


def test_png_list_is_still_the_last_object_table():
    """The 702 organelle slots go in front of `png_list`, not after it.

    `png_list` is documented as last because it is a crop index rather than an
    object table, and `_metric_order` prints the tables in declaration order.
    Splatting the roles onto the end of the tuple would have silently moved the
    crop index into the middle of the organelles.
    """
    assert run_compare.OBJECT_TABLES[-1] == "png_list"
    assert run_compare.OBJECT_TABLES.index("organelle") < run_compare.OBJECT_TABLES.index("png_list")


def test_the_declared_order_survives_into_the_printed_order():
    """`_metric_order` must place organelles among the object tables."""
    order = run_compare._metric_order(
        ["png_list", "organelle", "cell", "wells"])
    assert order.index("cell") < order.index("organelle") < order.index("png_list")
    # acquisition metrics come after every object table
    assert order.index("png_list") < order.index("wells")


@pytest.mark.parametrize("module", [run_compare, merge_tables, gene_measurement_compare])
def test_every_organelle_slot_is_nameable_by_every_comparison(module):
    """All three modules name the same 702 slots.

    merge_tables already did; the other two are the instruction. Comparing
    against the live ORGANELLE_ROLES rather than a pinned number means adding a
    703rd slot cannot leave one module behind.
    """
    missing = [role for role in ORGANELLE_ROLES
               if role not in module.OBJECT_TABLES]
    assert missing == [], f"{module.__name__} cannot name {len(missing)} organelle slots"


def test_the_crop_table_stays_out_of_the_gene_comparison():
    """`gene_measurement_compare` documents that `png_list` is added by the
    caller, so it must not acquire one from the organelle splat."""
    assert "png_list" not in gene_measurement_compare.OBJECT_TABLES
