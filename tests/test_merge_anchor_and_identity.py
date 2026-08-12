"""The merged table lost cytoplasm entirely, and invented object labels.

Three defects, one merge:

C32/C36 -- `merge_tables` decided whether a table could be joined by asking
whether it carried ``cell_id``. Cytoplasm does not: it is one object per
cell, so it is keyed ``object_label`` exactly like the cell table. The check
failed, the table was skipped with an INFO line nobody reads, and the merged
frame came back with **zero** cytoplasm columns. Every downstream analysis
that asked for cytoplasm measurements silently got none.

C38 -- a child's ``object_label`` went through the default numeric rule,
which is MEAN. A cell holding pathogens 1, 2 and 3 was reported as
``pathogen_object_label`` = 2.0: a number that looks like a measurement,
plots like one, and names an object that need not exist.

The anchor now comes from a registry both readers share, and a label is
carried verbatim or not at all.
"""

import sqlite3

import pandas as pd
import pytest

from spacr.merge_tables import aggregation_for, merge_tables
from spacr.object_roles import (ANCHOR_COLUMN, anchor_column,
                                is_one_row_per_cell)

BASE = dict(plateID="p1", rowID="r1", columnID="c1", fieldID="f1",
            prcf="p1_r1_c1_f1")


def build(tmp_path, **tables):
    """A measurement database holding exactly the frames given."""
    path = str(tmp_path / "measurements.db")
    con = sqlite3.connect(path)
    try:
        for name, frame in tables.items():
            frame.to_sql(name, con, index=False)
        con.commit()
    finally:
        con.close()
    return path


def cells(n=2):
    return pd.DataFrame([{**BASE, "object_label": i, "cell_area": 100.0 * i}
                         for i in range(1, n + 1)])


def cytoplasms(n=2):
    return pd.DataFrame([{**BASE, "object_label": i, "cytoplasm_area": 60.0 * i,
                          "solidity": 0.9} for i in range(1, n + 1)])


def children(table, per_cell, n_cells=2):
    """`per_cell` children in each cell, labelled 1..per_cell."""
    return pd.DataFrame([
        {**BASE, "object_label": j, "cell_id": i,
         f"{table}_area": 10.0 * j, "mean_intensity": 5.0 * j}
        for i in range(1, n_cells + 1) for j in range(1, per_cell + 1)])


# ---------------------------------------------------------------------------
# C32 -- the anchor is a lookup, not an assumption
# ---------------------------------------------------------------------------

def test_the_two_anchor_spellings_are_both_registered():
    """cell and cytoplasm are keyed by their own label; children by cell_id."""
    assert anchor_column("cell") == "object_label"
    assert anchor_column("cytoplasm") == "object_label"
    for table in ("nucleus", "pathogen", "organelle", "png_list"):
        assert anchor_column(table) == "cell_id"


def test_one_row_per_cell_is_exactly_the_object_label_tables():
    for table in ANCHOR_COLUMN:
        assert is_one_row_per_cell(table) is (
            anchor_column(table) == "object_label")


def test_an_unknown_table_names_the_ones_it_knows():
    with pytest.raises(ValueError) as raised:
        anchor_column("mitochondria")
    message = str(raised.value)
    assert "mitochondria" in message
    for known in ("cell", "cytoplasm", "nucleus"):
        assert known in message


# ---------------------------------------------------------------------------
# C36 -- cytoplasm survives the merge
# ---------------------------------------------------------------------------

def test_cytoplasm_columns_reach_the_merged_table(tmp_path):
    """The regression itself: this used to return zero cytoplasm columns."""
    db = build(tmp_path, cell=cells(), cytoplasm=cytoplasms())
    out = merge_tables(db, ["cell", "cytoplasm"])

    assert [c for c in out.columns if "cytoplasm" in c.lower()], (
        "cytoplasm was dropped from the merge again")
    assert "cytoplasm_area" in out.columns


def test_cytoplasm_values_are_the_measured_ones_not_an_aggregate(tmp_path):
    """One cytoplasm per cell must arrive whole, not rolled up.

    A roll-up would be numerically identical for a sum of one row, which is
    why the values are checked rather than the shape: 60 and 120 are what
    Measure wrote, and they must survive unchanged and un-averaged.
    """
    db = build(tmp_path, cell=cells(), cytoplasm=cytoplasms())
    out = merge_tables(db, ["cell", "cytoplasm"]).sort_values("object_label")

    assert list(out["cytoplasm_area"]) == [60.0, 120.0]
    assert list(out["cytoplasm_solidity"]) == [0.9, 0.9]
    assert len(out) == 2


def test_a_measurement_already_named_for_its_table_is_not_prefixed_twice(
        tmp_path):
    db = build(tmp_path, cell=cells(), cytoplasm=cytoplasms())
    out = merge_tables(db, ["cell", "cytoplasm"])
    assert "cytoplasm_cytoplasm_area" not in out.columns


def test_cytoplasm_and_a_rolled_up_child_can_share_one_merge(tmp_path):
    """The two join styles have to coexist: the whole point of the registry."""
    db = build(tmp_path, cell=cells(), cytoplasm=cytoplasms(),
               nucleus=children("nucleus", per_cell=2))
    out = merge_tables(db, ["cell", "cytoplasm", "nucleus"])

    assert len(out) == 2, "one row per cell, whatever the children did"
    assert "cytoplasm_area" in out.columns
    assert "nucleus_count" in out.columns
    assert list(out.sort_values("object_label")["nucleus_count"]) == [2, 2]


# ---------------------------------------------------------------------------
# C38 -- a label is a name, not a quantity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column", [
    "object_label", "label", "cell_id", "pathogen_id", "nucleus_id",
    "organelle_id", "parent_id", "prcfo",
])
def test_identity_columns_are_carried_not_averaged(column):
    assert aggregation_for(column) == "first"


@pytest.mark.parametrize("column,expected", [
    ("cell_area", "sum"),
    ("solidity", "mean"),
    ("perimeter", "mean"),
    ("mean_intensity", "mean"),
    ("centroid_x", "mean"),
])
def test_the_identity_rule_did_not_swallow_real_measurements(column, expected):
    """The identity pattern runs first, so it is the one that can over-match."""
    assert aggregation_for(column) == expected


def test_three_pathogens_do_not_produce_a_label_of_two(tmp_path):
    """The reported symptom, end to end.

    Labels 1, 2 and 3 average to exactly 2.0 -- a valid-looking label for an
    object the cell may not contain.
    """
    db = build(tmp_path, cell=cells(n_cells := 1),
               pathogen=children("pathogen", per_cell=3, n_cells=n_cells))
    out = merge_tables(db, ["cell", "pathogen"])

    label = out["pathogen_object_label"].iloc[0]
    assert label != 2.0 or label == 1, "the mean of 1, 2, 3 came back"
    assert label in (1, 1.0)
    assert out["pathogen_count"].iloc[0] == 3


def test_the_measurements_beside_the_label_still_aggregate(tmp_path):
    """Pinning the label must not freeze the row it sits on."""
    db = build(tmp_path, cell=cells(1),
               pathogen=children("pathogen", per_cell=3, n_cells=1))
    out = merge_tables(db, ["cell", "pathogen"])

    # areas 10, 20, 30 add up; intensities 5, 10, 15 average
    assert out["pathogen_area"].iloc[0] == pytest.approx(60.0)
    assert out["pathogen_mean_intensity"].iloc[0] == pytest.approx(10.0)
