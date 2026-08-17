"""A measurements database per plate, merged by the rules we already settled.

Instruction 130. The centre of this file is
:func:`test_every_measurement_gets_the_aggregation_the_rules_table_names`: the
expected number for each column is computed by asking
:func:`spacr.merge_tables.aggregation_for` what that column means and applying
THAT, so the assertion is against ``AGGREGATION_RULES`` itself and cannot
drift from it. The fixture is built so the sum, the mean, the min, the median,
the max and the first value of every column are different numbers, which
:func:`test_no_column_could_pass_by_picking_the_wrong_rule` pins -- an
aggregation that picked the wrong rule therefore cannot pass by coincidence.

The second thing here is the one instruction 130 warns is the most expensive
bug this application can have: a merge that silently changed how a measurement
was combined produces a number that is wrong and looks fine. So the roll-up
keys are tested too --
:func:`test_two_screens_sharing_a_plate_id_are_not_pooled_into_one_cell` fails
if ``screenID`` ever falls out of them, which would re-pool, one layer up,
exactly what :mod:`spacr.multi_database` exists to prevent.
"""
from __future__ import annotations

import re
import sqlite3

import pandas as pd
import pytest

from spacr import merge_tables as mt
from spacr.merge_tables import (AGGREGATIONS, FIRST, MEAN, MIN, SUM,
                                MergePolicy, aggregation_for,
                                aggregation_plan)
from spacr.multi_database import SCREEN_COLUMN, SOURCE_COLUMN, MergeRefused
from spacr.plate_measurements import (PlateDatabase, PlateMerge,
                                      available_tables,
                                      default_aggregated_columns,
                                      merge_plate_databases, missing_databases,
                                      plate_databases, unattached_plates)

#: The child columns whose aggregation this file is about. Named here rather
#: than typed into an assertion: each one's EXPECTED VALUE is computed from
#: `aggregation_for`, so this list says which measurements are covered and the
#: rules table says what happens to them.
MEASUREMENTS = ("object_label", "area", "major_axis_length",
                "channel_1_min_intensity", "spot_count", "texture_contrast")


def _cells(plate, labels=(1, 2, 3)):
    """Three cells: one with four pathogens, one with a single one, one with none."""
    return pd.DataFrame({
        "plateID": [plate] * len(labels), "rowID": ["A"] * len(labels),
        "columnID": ["1"] * len(labels), "fieldID": ["f1"] * len(labels),
        "object_label": list(labels),
        "area": [100.0, 200.0, 300.0][:len(labels)],
        "mean_intensity": [1.0, 2.0, 3.0][:len(labels)],
    })


def _pathogens(plate):
    """Four pathogens in cell 1, one in cell 2, none in cell 3.

    Every column's six possible aggregations are six different numbers -- see
    :func:`test_no_column_could_pass_by_picking_the_wrong_rule`, which is what
    makes the assertions above it mean anything.
    """
    return pd.DataFrame({
        "plateID": [plate] * 5, "rowID": ["A"] * 5, "columnID": ["1"] * 5,
        "fieldID": ["f1"] * 5,
        "cell_id": [1, 1, 1, 1, 2],
        # An extent: four objects' areas ADD UP.
        "area": [10.0, 20.0, 30.0, 40.0, 7.0],
        # A length: two nuclei 10 units long are not one nucleus 20 long.
        "major_axis_length": [2.0, 4.0, 12.0, 6.0, 1.0],
        # A mean of four minima is not the minimum of anything.
        "channel_1_min_intensity": [9.0, 5.0, 7.0, 11.0, 3.0],
        # A count of things inside the child: counts SUM.
        "spot_count": [1.0, 2.0, 5.0, 3.0, 6.0],
        # A NAME, not a quantity. Averaging it gives 16.0, which looks like a
        # measurement and names a pathogen that does not exist.
        "object_label": [8.0, 5.0, 21.0, 30.0, 2.0],
        # A measurement nobody wrote a rule for: this is the one the panel has
        # to name, because the default is where a wrong answer hides.
        "texture_contrast": [1.0, 2.0, 9.0, 4.0, 8.0],
    })


def _cytoplasm(plate, labels=(1, 2, 3)):
    """One row per cell -- derived from the cell mask, so never rolled up."""
    return pd.DataFrame({
        "plateID": [plate] * len(labels), "rowID": ["A"] * len(labels),
        "columnID": ["1"] * len(labels), "fieldID": ["f1"] * len(labels),
        "object_label": list(labels),
        "area": [70.0, 80.0, 90.0][:len(labels)],
    })


def _write(path, tables):
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _database(directory, plate, *, extra=None, tables=None):
    """A plate's ``measurements.db``, under its own folder as spaCR writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    pathogens = _pathogens(plate)
    if extra:
        pathogens[extra] = 1.0
    written = {"cell": _cells(plate), "pathogen": pathogens,
               "cytoplasm": _cytoplasm(plate)}
    if tables is not None:
        written = {name: frame for name, frame in written.items()
                   if name in tables}
    return _write(directory / "measurements.db", written)


@pytest.fixture
def two_plates(tmp_path):
    """Two plates, each with its own database, exactly as the input table has them."""
    return {"plate1": _database(tmp_path / "plate1", "plate1"),
            "plate2": _database(tmp_path / "plate2", "plate2")}


def _anchor_row(merge, plate, label):
    frame = merge.frame
    chosen = frame[(frame["plateID"] == plate)
                   & (frame["object_label"] == label)]
    assert len(chosen) == 1, f"expected one row for {plate}/{label}"
    return chosen.iloc[0]


# --------------------------------------------------------------------------- #
#  The aggregation, asserted against AGGREGATION_RULES rather than a list
# --------------------------------------------------------------------------- #

def test_every_measurement_gets_the_aggregation_the_rules_table_names(two_plates):
    """The rules are the maintainer's decision about what each measurement
    MEANS; this merge either applies them or answers a different question per
    column. Every expected number here comes from `aggregation_for`, so a rule
    changed in `merge_tables` changes this test's expectations with it."""
    merge = merge_plate_databases(two_plates, ["pathogen"])
    row = _anchor_row(merge, "plate1", 1)
    child = _pathogens("plate1")
    child = child[child["cell_id"] == 1]

    chosen = {column: aggregation_for(column) for column in MEASUREMENTS}
    # If the fixture stopped exercising four different rules, everything below
    # could pass while only one of them worked.
    assert set(chosen.values()) == {SUM, MEAN, MIN, FIRST}

    for column, how in chosen.items():
        expected = child.groupby("cell_id")[column].agg(how).iloc[0]
        assert row[f"pathogen_{column}"] == pytest.approx(expected), (
            f"{column} was combined as something other than {how}")

    # And in the plainest possible terms, because this is the sentence the
    # instruction is written in: an area SUMS and a length MEANS.
    assert row["pathogen_area"] == pytest.approx(100.0)
    assert row["pathogen_major_axis_length"] == pytest.approx(6.0)
    assert row["pathogen_channel_1_min_intensity"] == pytest.approx(5.0)
    assert row["pathogen_spot_count"] == pytest.approx(11.0)
    assert row["pathogen_object_label"] == pytest.approx(8.0)


def test_no_column_could_pass_by_picking_the_wrong_rule():
    """The fixture is only worth anything if the six aggregations give six
    different numbers -- otherwise the test above passes on a coincidence."""
    child = _pathogens("plate1")
    child = child[child["cell_id"] == 1]
    for column in MEASUREMENTS:
        values = {how: float(child.groupby("cell_id")[column].agg(how).iloc[0])
                  for how in AGGREGATIONS}
        right = aggregation_for(column)
        others = [value for how, value in values.items() if how != right]
        assert values[right] not in others, (
            f"{column}: {right} gives {values[right]}, which some other "
            f"aggregation also gives -- the test cannot tell them apart")


def test_the_average_everything_merge_would_have_given_a_different_area(two_plates):
    """`io._read_and_join_tables` aggregates every numeric column with mean.
    That is the code instruction 130 says not to reproduce, and this is the
    number that differs when someone does."""
    merge = merge_plate_databases(two_plates, ["pathogen"])
    row = _anchor_row(merge, "plate1", 1)
    child = _pathogens("plate1")
    averaged = child[child["cell_id"] == 1]["area"].mean()

    assert row["pathogen_area"] == pytest.approx(100.0)
    assert row["pathogen_area"] != pytest.approx(averaged)


def test_a_column_that_matched_no_rule_is_named_as_defaulted(two_plates):
    """DEFAULT_AGGREGATION is MEAN and the comment beside it says why -- but a
    measurement nobody thought about is exactly the one worth naming."""
    merge = merge_plate_databases(two_plates, ["pathogen"])

    assert merge.default_aggregation_columns == ("pathogen_texture_contrast",)
    # It really did take the default rather than being left out.
    row = _anchor_row(merge, "plate1", 1)
    assert row["pathogen_texture_contrast"] == pytest.approx(4.0)
    # And a column that MATCHED the mean rule is not named: reporting
    # `mean_intensity` as an unconsidered default would bury the real one.
    assert not any("mean_intensity" in name
                   for name in merge.default_aggregation_columns)


def test_the_defaulted_columns_are_recomputed_from_the_rules_not_stored():
    """Asserted against the rules table itself: a column is reported iff no
    pattern in AGGREGATION_RULES matches it."""
    child = _pathogens("plate1")
    plan = aggregation_plan(child, skip=("plateID", "rowID", "columnID",
                                         "fieldID", "cell_id"))
    named = default_aggregated_columns(plan)

    for column in named:
        assert not any(re.search(pattern, column.lower())
                       for pattern, _how in mt.AGGREGATION_RULES), column
    for column, how in plan.items():
        if column in named:
            continue
        assert (how != mt.DEFAULT_AGGREGATION
                or any(re.search(pattern, column.lower())
                       for pattern, _how in mt.AGGREGATION_RULES)), column


def test_a_rule_added_to_merge_tables_removes_a_column_from_the_report(monkeypatch):
    """The report re-walks AGGREGATION_RULES on every call. A module that took
    a copy at import would keep naming a measurement the maintainer has since
    written a rule for."""
    child = _pathogens("plate1")
    plan = aggregation_plan(child, skip=("cell_id",))
    assert "texture_contrast" in default_aggregated_columns(plan)

    monkeypatch.setattr(
        mt, "AGGREGATION_RULES",
        mt.AGGREGATION_RULES + ((r"(^|_)texture(_|$)", mt.MEAN),))
    assert "texture_contrast" not in default_aggregated_columns(plan)


def test_a_column_the_user_overrode_is_not_reported_as_a_default(two_plates):
    """An override is a decision, not an oversight."""
    policy = MergePolicy(overrides={"texture_contrast": MEAN})
    merge = merge_plate_databases(two_plates, ["pathogen"], policy=policy)

    assert merge.default_aggregation_columns == ()
    assert merge.tables[1].aggregations["texture_contrast"] == MEAN


def test_the_panel_is_told_which_aggregation_every_column_got(two_plates):
    """`aggregation_plan`'s dict IS what the disclosure renders, so it is
    carried rather than recomputed by whoever draws it."""
    merge = merge_plate_databases(two_plates, ["pathogen"])
    pathogen = next(entry for entry in merge.tables
                    if entry.table == "pathogen")

    assert pathogen.aggregations == aggregation_plan(
        _pathogens("plate1"), skip=("plateID", "rowID", "columnID", "fieldID",
                                    "cell_id"))
    assert pathogen.merged_column("area") == "pathogen_area"
    # Measure already writes prefixed columns; `pathogen_pathogen_area` is not
    # a measurement anybody knows by name.
    assert pathogen.merged_column("pathogen_area") == "pathogen_area"
    assert pathogen.merged_column("plateID") == "plateID"


# --------------------------------------------------------------------------- #
#  Several databases, without pooling them
# --------------------------------------------------------------------------- #

def test_the_merged_frame_says_which_database_every_row_came_from(two_plates):
    """A row that has forgotten which file it came from cannot answer whether
    the clusters are biology or batch."""
    merge = merge_plate_databases(two_plates, ["pathogen"])

    assert merge.rows == 6
    assert set(merge.frame[SOURCE_COLUMN]) == set(merge.sources)
    assert merge.rows_per_source == dict.fromkeys(merge.sources, 3)
    assert merge.rows_read_per_source == dict.fromkeys(merge.sources, 3)
    assert sorted(merge.frame["plateID"].unique()) == ["plate1", "plate2"]


def test_two_databases_holding_the_same_plate_are_refused(tmp_path):
    """Pool them and every per-well number afterwards is computed over two
    experiments at once, with nothing on screen to say so."""
    attached = {"first": _database(tmp_path / "runA", "plate1"),
                "second": _database(tmp_path / "runB", "plate1")}

    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases(attached, ["pathogen"])
    assert "plate1" in str(raised.value)


def test_two_screens_sharing_a_plate_id_are_not_pooled_into_one_cell(tmp_path):
    """THE ROLL-UP KEYS. Two screens share a guide library and both have
    plate1; `screenID` is what makes those two identities. Drop it from the
    keys and cell 1 of one screen and cell 1 of the other roll up into ONE
    parent -- the pooling `multi_database` refuses, reintroduced one layer up
    and this time silently."""
    attached = {"kd": _database(tmp_path / "kd", "plate1"),
                "oe": _database(tmp_path / "oe", "plate1")}
    # The same plate id in the other screen, with pathogens that cannot be
    # confused with the first screen's.
    with sqlite3.connect(attached["oe"]) as db:
        db.execute("UPDATE pathogen SET area = area * 10")

    merge = merge_plate_databases(attached, ["pathogen"],
                                  screens={"kd": "kd", "oe": "oe"})

    assert sorted(merge.frame[SCREEN_COLUMN].unique()) == ["kd", "oe"]
    kd = merge.frame[merge.frame[SCREEN_COLUMN] == "kd"]
    oe = merge.frame[merge.frame[SCREEN_COLUMN] == "oe"]
    assert float(kd[kd["object_label"] == 1]["pathogen_area"].iloc[0]) == 100.0
    assert float(oe[oe["object_label"] == 1]["pathogen_area"].iloc[0]) == 1000.0
    # Pooled, cell 1 would carry 1100.0 and there would be one row, not two.
    assert len(merge.frame) == 6
    assert merge.shared_plates_across_screens == {"plate1": ("kd", "oe")}
    # A user who did not MEAN to run two screens still needs to see that they
    # did, so it is said out loud rather than merely permitted.
    assert "more than one SCREEN" in merge.describe()


def test_the_screen_is_not_hidden_inside_the_plate_id(tmp_path):
    """Instruction 122: `on_collision='qualify'` rewrites plate1 to
    `kd-plate1`, which makes the keys unique by making the screen a string to
    be parsed back apart. This module never reaches for it."""
    attached = {"kd": _database(tmp_path / "kd", "plate1"),
                "oe": _database(tmp_path / "oe", "plate1")}

    merge = merge_plate_databases(attached, ["pathogen"],
                                  screens={"kd": "kd", "oe": "oe"})

    assert list(merge.frame["plateID"].unique()) == ["plate1"]
    assert SCREEN_COLUMN in merge.frame.columns


def test_a_measurement_only_some_databases_have_is_reported_as_dropped(tmp_path):
    """A dropped measurement is a measurement the user came to compare."""
    lines = []
    attached = {"plate1": _database(tmp_path / "plate1", "plate1"),
                "plate2": _database(tmp_path / "plate2", "plate2",
                                    extra="wobble")}

    merge = merge_plate_databases(attached, ["pathogen"], report=lines.append)

    assert merge.dropped_columns == ("pathogen_wobble",)
    assert merge.frame.attrs["dropped_columns"] == ("pathogen_wobble",)
    assert any("wobble" in line and line.startswith("pathogen:")
               for line in lines)
    assert "wobble" in merge.describe()


def test_the_dropped_measurement_can_be_kept_instead(tmp_path):
    """`columns='union'` keeps it with nulls where a database did not have it.
    Which is the right answer depends on the analysis, so it is offered rather
    than decided here."""
    attached = {"plate1": _database(tmp_path / "plate1", "plate1"),
                "plate2": _database(tmp_path / "plate2", "plate2",
                                    extra="wobble")}

    merge = merge_plate_databases(attached, ["pathogen"], columns="union")

    assert merge.dropped_columns == ()
    assert "pathogen_wobble" in merge.frame.columns
    assert merge.frame["pathogen_wobble"].isna().any()


def test_the_frame_carries_what_the_panel_must_disclose(two_plates):
    """Carried on the frame so they cannot be separated from the data they
    describe."""
    merge = merge_plate_databases(two_plates, ["pathogen", "cytoplasm"])

    assert merge.frame.attrs["anchor"] == "cell"
    assert merge.frame.attrs["tables"] == ("cell", "pathogen", "cytoplasm")
    assert merge.frame.attrs["default_aggregation_columns"] == (
        "pathogen_texture_contrast",)
    assert merge.frame.attrs["screens"] == ("screen1",)


def test_the_disclosure_names_the_anchor_the_rows_and_the_default(two_plates):
    """The panel's text, from the merge itself rather than assembled twice."""
    merge = merge_plate_databases(two_plates, ["pathogen"])
    said = merge.describe()

    assert "anchored on cell" in said
    assert "6 cell objects" in said
    for label in merge.sources:
        assert label in said
    assert "pathogen_texture_contrast" in said
    assert mt.DEFAULT_AGGREGATION in said


# --------------------------------------------------------------------------- #
#  Cardinality: the join is per table, never one blanket `how`
# --------------------------------------------------------------------------- #

def test_an_uninfected_cell_keeps_its_row_and_counts_zero_pathogens(two_plates):
    """An uninfected cell is a cell, and in a screen it is usually the control
    population. Making pathogen inner conditions every result on infection."""
    merge = merge_plate_databases(two_plates, ["pathogen"])
    childless = _anchor_row(merge, "plate1", 3)

    assert childless["pathogen_count"] == 0
    # It has no pathogen area at all -- zero would be a measurement never made.
    assert pd.isna(childless["pathogen_area"])
    assert pd.isna(childless["pathogen_measured"])
    assert merge.tables[1].how == "left"


def test_restricting_to_infected_cells_is_the_setting_that_does_it(two_plates):
    """`keep_uninfected=False` is how a caller deliberately narrows the
    population -- and it is a setting, not something a blanket join decides."""
    merge = merge_plate_databases(two_plates, ["pathogen"],
                                  policy=MergePolicy(keep_uninfected=False))

    assert merge.tables[1].how == "inner"
    assert merge.rows == 4
    assert merge.rows_per_source == dict.fromkeys(merge.sources, 2)
    assert "dropped by an inner join" in merge.describe()


def test_a_one_row_per_cell_table_is_joined_rather_than_aggregated(two_plates):
    """A cytoplasm is the cell minus its interior objects -- one row per cell.
    Putting it through the sum/mean rules meant for a GROUP of children is not
    wrong so much as meaningless."""
    merge = merge_plate_databases(two_plates, ["cytoplasm"])
    cytoplasm = next(entry for entry in merge.tables
                     if entry.table == "cytoplasm")

    assert not cytoplasm.rolled_up
    assert cytoplasm.aggregations == {}
    assert "cytoplasm_count" not in merge.frame.columns
    assert _anchor_row(merge, "plate1", 1)["cytoplasm_area"] == 70.0


def test_the_anchor_must_be_a_table_with_one_row_per_cell(two_plates):
    """Every other object table is keyed to the CELL, so anchoring on a
    many-per-cell table joins a cell id to an object label -- a join on a
    coincidence, which returns rows."""
    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases(two_plates, ["cell"], anchor="pathogen")
    assert "cytoplasm" in str(raised.value)

    merge = merge_plate_databases(two_plates, ["cell"], anchor="cytoplasm")
    assert merge.anchor == "cytoplasm"
    assert merge.rows == 6


def test_the_anchor_overrides_the_policy_it_was_given(two_plates):
    """The panel's anchor picker and a stored policy must not disagree about
    what a row means."""
    merge = merge_plate_databases(two_plates, ["pathogen"], anchor="cytoplasm",
                                  policy=MergePolicy(primary="cell"))
    assert merge.anchor == "cytoplasm"


# --------------------------------------------------------------------------- #
#  The input table's rows, including the ones with no database
# --------------------------------------------------------------------------- #

def test_a_plate_with_no_database_is_listed_and_does_not_stop_the_merge(tmp_path):
    """The regression runs on counts and scores; a missing database disables
    that plate in the Measurements tab rather than failing anything."""
    rows = [{"plate": "plate1", "score": "s.csv", "count": "c.csv",
             "database": _database(tmp_path / "plate1", "plate1")},
            {"plate": "plate2", "score": "s2.csv", "count": "c2.csv",
             "database": ""}]

    assert unattached_plates(rows) == ("plate2",)
    assert [row.plate for row in plate_databases(rows)] == ["plate1"]

    merge = merge_plate_databases(rows, ["pathogen"])
    assert merge.rows == 3
    assert [row.plate for row in merge.attachments] == ["plate1"]


def test_an_unnamed_row_is_still_identified(tmp_path):
    """"Attach it to the first row that has none, and SAY which" needs every
    row to have a name, including one the user has not labelled yet."""
    rows = [{"plate": "", "database": _database(tmp_path / "one", "plate1")}]
    assert plate_databases(rows)[0].plate == "row 1"


def test_a_database_that_has_moved_is_named_before_the_run(tmp_path):
    """Not four minutes into a regression."""
    attached = {"plate1": _database(tmp_path / "plate1", "plate1"),
                "plate2": str(tmp_path / "gone" / "measurements.db")}

    assert [row.plate for row in missing_databases(attached)] == ["plate2"]
    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases(attached, ["pathogen"])
    assert "plate2" in str(raised.value)
    assert "gone" in str(raised.value)


def test_one_database_attached_to_two_plates_is_refused(tmp_path):
    """Its rows would be counted twice, and both copies would look real."""
    path = _database(tmp_path / "plate1", "plate1")

    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases({"plate1": path, "plate2": path}, ["pathogen"])
    assert "plate1" in str(raised.value) and "plate2" in str(raised.value)


def test_nothing_attached_says_what_to_do_about_it(tmp_path):
    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases({"plate1": ""}, ["pathogen"])
    assert "drop a .db" in str(raised.value).lower()


def test_the_attachments_can_arrive_in_any_shape_the_input_table_has(tmp_path):
    """A mapping, the widget's own rows, pairs, or PlateDatabase objects --
    the GUI hands over what it already has rather than converting first."""
    path = _database(tmp_path / "plate1", "plate1")
    expected = (PlateDatabase(plate="plate1", path=path),)

    assert plate_databases({"plate1": path}) == expected
    assert plate_databases([("plate1", path)]) == expected
    assert plate_databases([{"plate": "plate1", "database": path}]) == expected
    assert plate_databases([{"plate": "plate1", "path": path}]) == expected
    assert plate_databases(expected) == expected
    assert plate_databases(None) == ()


# --------------------------------------------------------------------------- #
#  Which tables can be offered at all
# --------------------------------------------------------------------------- #

def test_the_tables_offered_are_the_ones_every_database_has(tmp_path):
    """`describe_merge` raises a bare sqlite `no such table` when one database
    lacks the chosen table, which reaches a user as a crash rather than as a
    choice they were never offered."""
    attached = {"plate1": _database(tmp_path / "plate1", "plate1"),
                "plate2": _database(tmp_path / "plate2", "plate2",
                                    tables=("cell", "cytoplasm"))}

    assert available_tables(attached) == ("cell", "cytoplasm")
    assert available_tables({}) == ()

    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases(attached, ["pathogen"])
    assert "pathogen" in str(raised.value)
    assert "plate2" in str(raised.value)


def test_the_table_list_comes_from_the_object_registry_not_four_names(tmp_path):
    """`object_roles` is the one registry of what object kinds exist, so a
    sixth reaches this list by being declared once."""
    attached = {"plate1": _database(tmp_path / "plate1", "plate1")}
    offered = available_tables(attached)

    assert offered == tuple(table for table in mt.OBJECT_TABLES
                            if table in offered)
    assert set(offered) <= set(mt.OBJECT_TABLES)


def test_crops_are_not_offered_as_a_measurement_table(tmp_path):
    """png_list is one row per CROP, and a crop is not a measurement to
    aggregate."""
    directory = tmp_path / "plate1"
    directory.mkdir()
    path = _write(directory / "measurements.db", {
        "cell": _cells("plate1"), "pathogen": _pathogens("plate1"),
        "png_list": pd.DataFrame({"plateID": ["plate1"], "rowID": ["A"],
                                  "columnID": ["1"], "fieldID": ["f1"],
                                  "cell_id": [1], "png_path": ["/a.png"]})})

    assert "png_list" not in available_tables({"plate1": path})
    with pytest.raises(MergeRefused) as raised:
        merge_plate_databases({"plate1": path}, ["png_list"])
    assert "png_list" in str(raised.value)


def test_a_table_measured_without_a_parent_mask_is_named_and_skipped(tmp_path):
    """The roll-up is not empty, it is UNDEFINED -- and one unlinkable table
    must not cost the user the others."""
    directory = tmp_path / "plate1"
    directory.mkdir()
    orphaned = _pathogens("plate1").drop(columns=["cell_id"])
    path = _write(directory / "measurements.db",
                  {"cell": _cells("plate1"), "pathogen": orphaned,
                   "cytoplasm": _cytoplasm("plate1")})

    merge = merge_plate_databases({"plate1": path}, ["pathogen", "cytoplasm"])
    pathogen = next(entry for entry in merge.tables
                    if entry.table == "pathogen")

    assert "cell_id" in pathogen.note
    assert "pathogen_area" not in merge.frame.columns
    # The tables that CAN be linked still are.
    assert "cytoplasm_area" in merge.frame.columns
    assert pathogen.note in merge.describe()


def test_a_merge_with_nothing_in_it_answers_every_question_emptily():
    """The Measurements tab asks all of this before anything is attached, and
    an empty state that raises is an empty state that blanks the panel."""
    empty = PlateMerge(frame=pd.DataFrame(), anchor="cell", attachments=(),
                       tables=())

    assert empty.rows == 0
    assert empty.sources == ()
    assert empty.rows_read_per_source == {}
    assert empty.rows_per_source == {}
    assert empty.dropped_columns == ()
    assert empty.default_aggregation_columns == ()
    assert empty.shared_plates_across_screens == {}
    assert "anchored on cell" in empty.describe()


def test_the_anchor_is_always_merged_even_when_it_was_not_ticked(two_plates):
    """A row means one anchor object, so the anchor is not optional."""
    merge = merge_plate_databases(two_plates, ["pathogen"])

    assert [entry.table for entry in merge.tables] == ["cell", "pathogen"]
    assert "cell_area" in merge.frame.columns
