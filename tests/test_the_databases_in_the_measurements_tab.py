"""The measurements databases attached to each plate, and their merge.

Instruction 130 sections B and C:

    "these database files should then pop up in the measurements tab, there
     should be options on what tables to join (and they should be merged
     according to the logic we have already exaustively gine through with the
     correct measurements being summed, averaged and so on correctly.)"

WHAT THESE TESTS ARE GUARDING, and it is not the widget:

    A MERGE THAT SILENTLY CHANGED HOW A MEASUREMENT WAS COMBINED PRODUCES A
    NUMBER THAT IS WRONG AND LOOKS FINE.

So the aggregation is asserted against :data:`spacr.merge_tables.
AGGREGATION_RULES` itself rather than against a list written here -- a list
here would agree with a panel that had stopped calling the rules at all -- the
join is asserted to be per table rather than one blanket ``how``, and the
panel is required to SAY what fell through to the default, what each source
contributed, and what happened to a plate id that appeared twice.

The two-screens case is the one worth reading first
(:func:`test_two_screens_sharing_a_plate_id_do_not_collapse_into_one_parent`).
It is the pooling :mod:`spacr.multi_database` exists to prevent, reintroduced
one layer up by a roll-up that forgot the screen -- and its symptom is a
pathogen area that is exactly twice what it should be.
"""
from __future__ import annotations

import os
import re
import sqlite3

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  Real sqlite databases, in spaCR's own shape
# --------------------------------------------------------------------------- #

def _database(directory, plate, *, cells=3, nuclei=None, pathogens=(1, 1, 2, 2),
              tables=("cell", "nucleus", "pathogen"), extra=None,
              pathogen_areas=(10.0, 30.0, 50.0, 70.0), name="measurements.db",
              link=True):
    """One plate's measurements.db, with the tables spaCR writes.

    The pathogen numbers are the ones the merge rules have to get right: two
    pathogens per cell whose AREAS add up, whose PERIMETERS do not, and whose
    minimum intensity is a minimum of two rather than a mean of two.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), name)
    nuclei = list(range(1, cells + 1)) if nuclei is None else list(nuclei)
    identity = {"rowID": "r1", "columnID": "c1", "fieldID": "f1"}

    cell = pd.DataFrame({
        "plateID": [plate] * cells, **{k: [v] * cells
                                       for k, v in identity.items()},
        "object_label": list(range(1, cells + 1)),
        "area": [100.0 * i for i in range(1, cells + 1)],
        "perimeter": [10.0 * i for i in range(1, cells + 1)],
    })
    if extra:
        cell[extra] = 1.0
    nucleus = pd.DataFrame({
        "plateID": [plate] * len(nuclei),
        **{k: [v] * len(nuclei) for k, v in identity.items()},
        "cell_id": nuclei,
        "object_label": list(range(1, len(nuclei) + 1)),
        "nucleus_area": [5.0] * len(nuclei),
    })
    if not link:
        nucleus = nucleus.drop(columns=["cell_id"])
    n = len(pathogens)
    pathogen = pd.DataFrame({
        "plateID": [plate] * n, **{k: [v] * n for k, v in identity.items()},
        "cell_id": list(pathogens),
        "object_label": [1, 2] * (n // 2) if n % 2 == 0 else list(range(n)),
        "pathogen_area": list(pathogen_areas)[:n],
        "pathogen_perimeter": [4.0, 8.0, 12.0, 16.0][:n],
        "pathogen_channel_1_min_intensity": [5.0, 9.0, 1.0, 3.0][:n],
        "pathogen_wobble": [1.0, 2.0, 3.0, 4.0][:n],
    })
    cytoplasm = pd.DataFrame({
        "plateID": [plate] * cells, **{k: [v] * cells
                                       for k, v in identity.items()},
        "object_label": list(range(1, cells + 1)),
        "cytoplasm_area": [50.0 * i for i in range(1, cells + 1)],
    })
    organelle = pd.DataFrame({
        "plateID": [plate] * n, **{k: [v] * n for k, v in identity.items()},
        "cell_id": list(pathogens),
        "object_label": list(range(1, n + 1)),
        "organelle_area": [1.0, 2.0, 3.0, 4.0][:n],
    })
    written = {"cell": cell, "nucleus": nucleus, "pathogen": pathogen,
               "cytoplasm": cytoplasm, "organelle": organelle}
    with sqlite3.connect(path) as db:
        for table in tables:
            written[table].to_sql(table, db, index=False)
    return path


@pytest.fixture()
def two_plates(tmp_path):
    """Two plates, each its own measurements.db, as spaCR lays them out."""
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2", nuclei=(1, 2))]


def _rows(paths, plates=None, screens=None):
    """Input-table rows, in the shape the paired table emits."""
    plates = plates or [f"plate{i + 1}" for i in range(len(paths))]
    out = []
    for index, path in enumerate(paths):
        row = {"plate": plates[index], "score": f"{plates[index]}_scores.csv",
               "count": f"{plates[index]}_counts.csv", "database": path}
        if screens:
            row["screen"] = screens[index]
        out.append(row)
    return out


@pytest.fixture()
def panel(qtbot, two_plates):
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel(lambda: _rows(two_plates))
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  B: they appear, with their tables and their plates
# --------------------------------------------------------------------------- #

def test_every_attached_database_is_listed_with_its_tables_and_plates(panel):
    """"The Measurements tab lists every attached database with its tables"
    -- and the plate ids inside it, because the plate the row is NAMED is the
    user's own label and the plate in the file is the fact."""
    listed = {}
    for row in range(panel.table.rowCount()):
        cells = [panel.table.item(row, column).text()
                 for column in range(panel.table.columnCount())]
        listed[cells[0]] = cells

    assert set(listed) == {"plate1", "plate2"}
    assert "cell" in listed["plate1"][3] and "pathogen" in listed["plate1"][3]
    assert listed["plate1"][4] == "plate1"
    assert listed["plate2"][4] == "plate2"
    assert listed["plate1"][5] == "3 cell"


def test_the_name_shown_is_the_name_the_merged_rows_will_carry(panel):
    """Every plate's database is called measurements.db, so a list that named
    them by their filename would name all of them the same thing -- and could
    not be matched against source_database, which is what says which
    experiment a row came from."""
    from spacr.multi_database import SOURCE_COLUMN

    names = {panel.table.item(row, 1).text()
             for row in range(panel.table.rowCount())}
    frame = panel.merge()

    assert set(frame[SOURCE_COLUMN].unique()) == names, names


def test_a_plate_with_no_database_is_listed_and_disabled_rather_than_dropped(
        qtbot, two_plates):
    """"a plate with NO database is legal. The regression runs on counts and
    scores." Dropping the row here would make the plate invisible, which reads
    as a plate that was not analysed at all."""
    from PySide6.QtCore import Qt
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    rows = _rows(two_plates) + [{"plate": "plate3", "score": "s.csv",
                                 "count": "c.csv", "database": ""}]
    widget = DatabaseMergePanel(lambda: rows)
    qtbot.addWidget(widget)

    assert widget.table.rowCount() == 3
    assert widget.table.item(2, 0).text() == "plate3"
    assert not (widget.table.item(2, 0).flags() & Qt.ItemIsEnabled)
    assert "no database" in widget.table.item(2, 6).text()
    assert "still run in the regression" in widget.heading.text()
    # And it changes nothing about the merge the other two can do.
    assert len(widget.merge()) == 5


def test_a_database_that_has_gone_missing_says_so_before_the_run(
        qtbot, two_plates, tmp_path):
    """"a row whose database is missing from disk when the run starts says so
    before the run, not four minutes in"."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    rows = _rows(two_plates) + [{"plate": "plate9", "score": "s.csv",
                                 "count": "c.csv",
                                 "database": str(tmp_path / "gone.db")}]
    widget = DatabaseMergePanel(lambda: rows)
    qtbot.addWidget(widget)

    assert "missing from disk" in widget.table.item(2, 6).text()
    assert "plate9" in widget.heading.text()
    assert str(tmp_path / "gone.db") not in widget.paths()


def test_a_database_holding_a_different_plate_than_its_row_says_so(
        qtbot, two_plates):
    """A .db dropped on the wrong row is the failure the third column can
    make easy. Not refused -- the row's label is the user's own name for it --
    but never silent."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel(
        lambda: _rows(two_plates, plates=["plate1", "plate7"]))
    qtbot.addWidget(widget)

    assert "holds plate2, not plate7" in widget.table.item(1, 6).text()


# --------------------------------------------------------------------------- #
#  B: the tables on offer come from the registry, and from every database
# --------------------------------------------------------------------------- #

def test_the_tables_offered_come_from_the_object_role_registry(qtbot, tmp_path):
    """"object_roles is the one registry of what object kinds exist -- read it
    rather than hard-coding four names." An organelle table is offered for
    being in the registry, not for being thought of."""
    from spacr.merge_tables import OBJECT_TABLES
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    tables = ("cell", "nucleus", "pathogen", "cytoplasm", "organelle")
    paths = [_database(tmp_path / "p1", "plate1", tables=tables),
             _database(tmp_path / "p2", "plate2", tables=tables,
                       nuclei=(1, 2))]
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)

    offered = widget.selected_tables()
    assert "organelle" in offered
    assert "cytoplasm" in offered
    assert set(offered) <= set(OBJECT_TABLES), offered


def test_only_tables_every_database_has_are_offered(qtbot, tmp_path):
    """Offering a table one database lacks would hand the user a merge that
    dies inside sqlite naming the table and nothing else."""
    from spacr.multi_database import describe_merge
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "p1", "plate1",
                       tables=("cell", "nucleus", "pathogen")),
             _database(tmp_path / "p2", "plate2", nuclei=(1, 2),
                       tables=("cell", "nucleus"))]
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)

    assert "pathogen" not in widget.selected_tables()
    # This is what the user would have hit, which is why it is not offered.
    with pytest.raises(Exception) as refusal:
        describe_merge(paths, "pathogen")
    assert "pathogen" in str(refusal.value)


def test_only_one_row_per_cell_tables_can_be_the_anchor(qtbot, tmp_path):
    """Anchoring on pathogen would make a row mean one pathogen and repeat the
    cell's own measurements across its children -- the fan-out the roll-up
    exists to prevent, arrived at from the other side."""
    from spacr.object_roles import ONE_ROW_PER_CELL
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    tables = ("cell", "nucleus", "pathogen", "cytoplasm")
    paths = [_database(tmp_path / "p1", "plate1", tables=tables),
             _database(tmp_path / "p2", "plate2", tables=tables,
                       nuclei=(1, 2))]
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)

    offered = [widget.anchor_box.itemText(i)
               for i in range(widget.anchor_box.count())]
    assert set(offered) == set(ONE_ROW_PER_CELL)
    assert "pathogen" not in offered


def test_the_anchor_defaults_to_cell_and_the_panel_says_so(panel):
    """79: one anchor, one copy of each column. The default is cell and the
    panel says which it is -- an anchor is what a ROW MEANS, and a user who
    does not know theirs cannot read the table."""
    from spacr.merge_tables import DEFAULT_PRIMARY

    assert panel.anchor() == DEFAULT_PRIMARY == "cell"
    assert DEFAULT_PRIMARY in panel.anchor_note.text().lower()
    assert f"Anchor: {DEFAULT_PRIMARY}" in panel.plan_text()
    assert "the default" in panel.plan_text()


# --------------------------------------------------------------------------- #
#  B: the join is per table, and there is no control that says otherwise
# --------------------------------------------------------------------------- #

def test_the_join_is_stated_per_table_and_follows_cardinality(panel):
    """Instruction 77's finding, answered 2026-08-12: the join follows object
    CARDINALITY. A cell with no nucleus is not a cell; a cell with no pathogen
    is an uninfected cell and usually the control population."""
    text = panel.plan_text()
    policy = panel.policy()

    for table in ("nucleus", "pathogen"):
        assert f"{table}: {policy.how_for(table)} join" in text, text
    assert policy.how_for("nucleus") == "inner"
    assert policy.how_for("pathogen") == "left"


def test_no_control_offers_one_blanket_join_for_every_table(panel):
    """"Do not offer a control that contradicts it." A single how= would
    condition every result on infection, or keep debris that has no nucleus,
    depending which way the user set it."""
    from PySide6.QtWidgets import QComboBox

    for box in panel.findChildren(QComboBox):
        items = {box.itemText(i).lower() for i in range(box.count())}
        assert not ({"left", "inner", "outer", "right"} & items), items


def test_keeping_uninfected_cells_is_the_setting_that_changes_it(panel):
    """The one legitimate way to make the pathogen join inner: a deliberate
    statement that the analysis is about infected cells."""
    panel.keep_uninfected.setChecked(False)

    assert panel.policy().how_for("pathogen") == "inner"
    assert "pathogen: inner join" in panel.plan_text()


# --------------------------------------------------------------------------- #
#  C: the numbers, and where they came from
# --------------------------------------------------------------------------- #

def test_an_area_sums_and_a_perimeter_means_in_the_merged_output(panel):
    """The whole point of the request. Two pathogens of area 10 and 30 in one
    cell are 40 units of pathogen; their perimeters are not 12 units of
    perimeter, because two objects 4 and 8 around are not one object 12
    around. Asserted against the rule table, not against a number typed here.
    """
    from spacr.merge_tables import MAX, MEAN, MIN, SUM, aggregation_for

    frame = panel.merge().set_index(["plateID", "object_label"])

    assert aggregation_for("pathogen_area") == SUM
    assert frame.loc[("plate1", 1), "pathogen_area"] == 40.0
    assert frame.loc[("plate1", 2), "pathogen_area"] == 120.0

    assert aggregation_for("pathogen_perimeter") == MEAN
    assert frame.loc[("plate1", 1), "pathogen_perimeter"] == 6.0
    assert frame.loc[("plate1", 2), "pathogen_perimeter"] == 14.0

    assert aggregation_for("pathogen_channel_1_min_intensity") == MIN != MAX
    assert frame.loc[("plate1", 1), "pathogen_channel_1_min_intensity"] == 5.0
    assert frame.loc[("plate1", 2), "pathogen_channel_1_min_intensity"] == 1.0


def test_the_aggregation_is_the_rule_tables_and_an_override_reaches_it(panel):
    """The proof that the panel CALLS the rules rather than reimplementing
    them: an override is a merge_tables concept, and it changes the number."""
    panel._overrides = {"pathogen_area": "max"}

    frame = panel.merge().set_index(["plateID", "object_label"])

    assert frame.loc[("plate1", 1), "pathogen_area"] == 30.0
    assert frame.loc[("plate1", 2), "pathogen_area"] == 70.0


def test_the_panel_names_every_column_that_fell_through_to_the_default(panel):
    """"a measurement nobody thought about is exactly the one worth naming."
    The expected set is computed by re-walking AGGREGATION_RULES, because a
    list written here would agree with a panel that had stopped reading them.
    """
    from spacr.merge_tables import AGGREGATION_RULES, DEFAULT_AGGREGATION

    frame = panel.merge()
    report = panel.report.toPlainText()

    fell = [column for column in frame.attrs["default_aggregation"]["pathogen"]]
    for column in fell:
        assert not any(re.search(pattern, column.lower())
                       for pattern, _how in AGGREGATION_RULES), column
        assert column in report
    assert "pathogen_wobble" in fell, fell
    assert DEFAULT_AGGREGATION in report


def test_a_column_every_rule_names_is_not_reported_as_a_fall_through(panel):
    """The report has to be worth reading. Naming every column would bury the
    one nobody thought about among the ones that are fine."""
    panel.merge()
    report = panel.report.toPlainText()
    fall_through = [line for line in report.splitlines()
                    if "matched no aggregation rule" in line]

    assert fall_through, report
    assert "pathogen_area" not in "\n".join(fall_through)
    assert "pathogen_channel_1_min_intensity" not in "\n".join(fall_through)


def test_the_panel_states_the_anchor_and_the_row_count_it_produced(panel):
    frame = panel.merge()
    report = panel.report.toPlainText()

    assert f"Merged {len(frame):,} rows on cell" in report, report


def test_it_says_what_each_source_gave_and_what_the_join_took_away(panel):
    """One database has a cell with no nucleus. The nucleus join is inner, so
    that cell is REMOVED -- and a filter that removes part of the population
    without saying so is how a result gets reported for a subgroup nobody
    chose."""
    frame = panel.merge()
    report = panel.report.toPlainText()

    assert len(frame) == 5
    assert frame.attrs["rows_before"] == {"measurements": 3,
                                          "plate2/measurements": 3}
    assert frame.attrs["rows_after"] == {"measurements": 3,
                                         "plate2/measurements": 2}
    assert "2 of 3 cell rows" in report
    assert "nucleus: inner join, removed 1 of 6 rows" in report


def test_a_measurement_only_some_databases_have_is_named_before_the_merge(
        qtbot, tmp_path):
    """columns='common' silently intersects, and the set it drops IS the
    analysis the user came to run."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "p1", "plate1", extra="cell_wobble"),
             _database(tmp_path / "p2", "plate2", nuclei=(1, 2))]
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)

    assert "cell_wobble" in widget.plan_text()
    frame = widget.merge()
    assert "cell_wobble" not in frame.columns
    assert "cell_wobble" in widget.report.toPlainText()


def test_a_skipped_nan_is_visible_rather_than_shrinking_a_sum(qtbot, tmp_path):
    """pandas skips NaN in every aggregation and says nothing, so a cell with
    two pathogens one of whose area is missing reports the sum of ONE against
    a count of two. `<table>_measured` is the flag, and it survives the
    multi-database composition."""
    paths = [_database(tmp_path / "p1", "plate1",
                       pathogen_areas=(10.0, float("nan"), 50.0, 70.0)),
             _database(tmp_path / "p2", "plate2", nuclei=(1, 2))]
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)
    frame = widget.merge().set_index(["plateID", "object_label"])

    assert frame.loc[("plate1", 1), "pathogen_count"] == 2
    assert frame.loc[("plate1", 1), "pathogen_measured"] == 1
    assert frame.loc[("plate1", 1), "pathogen_area"] == 10.0


def test_a_table_that_cannot_be_linked_is_named_and_the_others_still_merge(
        qtbot, tmp_path):
    """Measured without a parent mask, a roll-up is not empty, it is
    UNDEFINED. One unlinkable table must not cost the user the others."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "p1", "plate1", link=False),
             _database(tmp_path / "p2", "plate2", link=False)]
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)
    frame = widget.merge()

    assert "nucleus" in frame.attrs["skipped_tables"]
    assert "cell_id" in widget.report.toPlainText()
    assert "pathogen_area" in frame.columns


# --------------------------------------------------------------------------- #
#  C: a plate id in two databases, and what was done about it
# --------------------------------------------------------------------------- #

def test_a_plate_in_two_databases_of_one_screen_is_refused_in_full(
        qtbot, tmp_path):
    """Pooling them would compute every per-well number over two experiments
    at once, with nothing on screen to say so. The refusal is an ANSWER: it is
    shown whole, because "the merge failed" sends the user looking for a bug
    in spaCR."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "runA", "plate1"),
             _database(tmp_path / "runB", "plate1")]
    widget = DatabaseMergePanel(lambda: _rows(paths, plates=["p", "p"]))
    qtbot.addWidget(widget)

    assert "will be refused" in widget.plan_text()
    assert widget.merge() is None
    report = widget.report.toPlainText()
    assert "Refused, and nothing was merged" in report
    assert "two experiments at once" in report
    assert widget.frame is None


def test_two_screens_sharing_a_plate_id_do_not_collapse_into_one_parent(
        qtbot, tmp_path):
    """122's case, and the one this composition could most easily get wrong.

    Two screens share a guide library, so both have plate1 with the same cell
    labels. They are two identities, kept apart by screenID. Leave the screen
    out of the roll-up keys and each cell's four pathogens become one parent's
    four: the area DOUBLES, silently, and reads as an experimental effect.
    """
    from spacr.multi_database import SCREEN_COLUMN
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "kd", "plate1"),
             _database(tmp_path / "oe", "plate1")]
    widget = DatabaseMergePanel(
        lambda: _rows(paths, plates=["plate1", "plate1"],
                      screens=["kd", "oe"]))
    qtbot.addWidget(widget)

    frame = widget.merge()

    assert frame is not None, widget.report.toPlainText()
    assert set(frame[SCREEN_COLUMN]) == {"kd", "oe"}
    assert len(frame) == 6
    for _, row in frame.iterrows():
        if row["object_label"] == 1:
            assert row["pathogen_area"] == 40.0, "the screens were pooled"
            assert row["pathogen_count"] == 2


def test_a_shared_plate_id_is_reported_and_never_qualified(qtbot, tmp_path):
    """122's note: qualifying rewrites plate1 to kd-plate1, which makes the
    keys unique and leaves the screen un-analysable -- you cannot block on it,
    test for a screen effect or colour by it without parsing a string apart.
    """
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "kd", "plate1"),
             _database(tmp_path / "oe", "plate1")]
    widget = DatabaseMergePanel(
        lambda: _rows(paths, plates=["plate1", "plate1"],
                      screens=["kd", "oe"]))
    qtbot.addWidget(widget)

    plan = widget.plan_text()
    frame = widget.merge()

    assert "appears in screens kd, oe" in plan
    assert "not renamed" in plan.lower() or "NOT renamed" in plan
    assert set(frame["plateID"]) == {"plate1"}, "the plate id was rewritten"


def test_the_merged_rows_never_forget_which_database_they_came_from(panel):
    """A row that has forgotten its file cannot answer whether the clusters
    are biology or batch."""
    from spacr.multi_database import SCREEN_COLUMN, SOURCE_COLUMN

    frame = panel.merge()

    assert SOURCE_COLUMN in frame.columns
    assert SCREEN_COLUMN in frame.columns
    assert len(set(frame[SOURCE_COLUMN])) == 2


# --------------------------------------------------------------------------- #
#  The tab it lives in
# --------------------------------------------------------------------------- #

def test_the_measurements_tab_shows_the_databases_without_a_scan(
        qtbot, two_plates):
    """"these database files should then pop up in the measurements tab"."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel(database_provider=lambda: _rows(two_plates))
    qtbot.addWidget(widget)

    assert widget.databases.isVisibleTo(widget)
    assert widget.databases.table.rowCount() == 2


def test_the_section_stays_out_of_the_way_when_no_plate_has_one(qtbot):
    """A project that never attached a database sees the tab it always saw."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel()
    qtbot.addWidget(widget)

    assert not widget.databases.isVisibleTo(widget)


def test_the_tab_re_reads_the_rows_rather_than_holding_a_copy(
        qtbot, two_plates):
    """The provider is a callable for the reason the scan's is: the tab must
    not go on showing the previous run's inputs."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    rows = []
    widget = MeasurementScanPanel(database_provider=lambda: list(rows))
    qtbot.addWidget(widget)
    assert widget.databases.table.rowCount() == 0

    rows.extend(_rows(two_plates))
    assert widget.refresh_databases() == 2
    assert widget.databases.table.rowCount() == 2
    assert widget.databases.isVisibleTo(widget)


def test_a_provider_that_raises_does_not_take_the_tab_down(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    def _boom():
        raise OSError("the input table went away")

    widget = MeasurementScanPanel(database_provider=_boom)
    qtbot.addWidget(widget)

    assert widget.refresh_databases() == 0
    assert "went away" in widget.databases.report.toPlainText()


def test_a_provider_may_hand_over_plain_pairs_or_paths(qtbot, two_plates):
    """A caller with a plainer list should not have to build dicts to be
    understood."""
    from spacr.qt.widgets.measurement_scan_panel import (AttachedDatabase,
                                                         attached_databases)

    pairs = attached_databases([("plate1", two_plates[0]),
                                ("plate2", two_plates[1], "kd")])
    bare = attached_databases(two_plates)
    already = attached_databases([AttachedDatabase("plate1", two_plates[0])])

    assert [entry.plate for entry in pairs] == ["plate1", "plate2"]
    assert pairs[1].screen == "kd"
    assert [entry.path for entry in bare] == list(two_plates)
    assert [entry.plate for entry in bare] == ["row 1", "row 2"]
    assert already[0].label.endswith("measurements.db")
    assert attached_databases(None) == ()


def test_the_input_tables_own_rows_are_understood_as_they_come(
        qtbot, two_plates, tmp_path):
    """The seam between section A and section B, driven through both widgets.

    The input table pairs BY ADDITION (107), so databases dropped in the
    opposite order to the CSVs still land on the right plates -- and what it
    emits is what this tab reads, with no shape in between for the two to
    disagree about.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    for plate in ("plate1", "plate2"):
        (tmp_path / f"{plate}_scores.csv").write_text("grna,score\n")
        (tmp_path / f"{plate}_counts.csv").write_text("grna,count\n")
    inputs = PairedFileTableWidget()
    qtbot.addWidget(inputs)
    inputs.add_paths_for_side(
        [str(tmp_path / f"{p}_scores.csv") for p in ("plate1", "plate2")],
        "score")
    inputs.add_paths_for_side(
        [str(tmp_path / f"{p}_counts.csv") for p in ("plate2", "plate1")],
        "count")
    inputs.add_paths_for_side(list(reversed(two_plates)), "database")

    widget = DatabaseMergePanel(inputs.get_value)
    qtbot.addWidget(widget)

    assert [row["database"] for row in inputs.get_value()] == list(two_plates)
    assert widget.table.rowCount() == 2
    assert len(widget.merge()) == 5


def test_handing_the_tab_a_new_provider_re_reads_it(qtbot, two_plates):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel()
    qtbot.addWidget(widget)
    assert widget.databases.table.rowCount() == 0

    widget.set_database_provider(lambda: _rows(two_plates))

    assert widget.databases.table.rowCount() == 2
    assert widget.databases.isVisibleTo(widget)


# --------------------------------------------------------------------------- #
#  The choice survives, and the panel says what it cannot do
# --------------------------------------------------------------------------- #

def test_the_anchor_and_the_ticked_tables_survive_a_refresh(qtbot, tmp_path):
    """The input table is re-read whenever the tab is opened. Losing the
    user's choice every time would make the choice not worth making."""
    tables = ("cell", "nucleus", "pathogen", "cytoplasm")
    paths = [_database(tmp_path / "p1", "plate1", tables=tables),
             _database(tmp_path / "p2", "plate2", tables=tables,
                       nuclei=(1, 2))]
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)
    widget.set_anchor("cytoplasm")
    widget.set_selected_tables(["cytoplasm", "pathogen"])

    widget.refresh()

    assert widget.anchor() == "cytoplasm"
    assert set(widget.selected_tables()) == {"cytoplasm", "pathogen"}


def test_a_one_row_per_cell_table_is_joined_rather_than_rolled_up(
        qtbot, tmp_path):
    """A cytoplasm is the cell minus its interior objects, so there is exactly
    one and aggregating it would answer a question nobody asked."""
    tables = ("cell", "nucleus", "pathogen", "cytoplasm")
    paths = [_database(tmp_path / "p1", "plate1", tables=tables),
             _database(tmp_path / "p2", "plate2", tables=tables,
                       nuclei=(1, 2))]
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)
    frame = widget.merge().set_index(["plateID", "object_label"])

    assert frame.loc[("plate1", 2), "cytoplasm_area"] == 100.0
    assert "cytoplasm_count" not in frame.columns
    assert "cytoplasm_cytoplasm_area" not in frame.columns


def test_the_anchor_alone_is_a_legitimate_merge_and_says_so(panel):
    panel.set_selected_tables(["cell"])

    assert "No other table chosen" in panel.plan_text()
    frame = panel.merge()
    assert len(frame) == 6
    assert "nucleus_area" not in frame.columns


def test_a_clean_table_is_told_that_nothing_fell_through(qtbot, tmp_path):
    """The report has to be readable when the answer is "nothing happened",
    or a user learns to skip it and misses the run where something did."""
    from spacr.merge_tables import DEFAULT_AGGREGATION
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "p1", "plate1"),
             _database(tmp_path / "p2", "plate2", nuclei=(1, 2))]
    for path in paths:
        with sqlite3.connect(path) as db:
            db.execute("ALTER TABLE pathogen DROP COLUMN pathogen_wobble")
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)
    widget.merge()

    assert f"none fell through to the default ({DEFAULT_AGGREGATION})" \
        in widget.report.toPlainText()


def test_a_child_measurement_only_some_databases_have_is_named_too(
        qtbot, tmp_path):
    """The anchor is not the only table that gets intersected."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    paths = [_database(tmp_path / "p1", "plate1"),
             _database(tmp_path / "p2", "plate2", nuclei=(1, 2))]
    with sqlite3.connect(paths[0]) as db:
        db.execute("ALTER TABLE pathogen ADD COLUMN pathogen_only_here REAL")
    widget = DatabaseMergePanel(lambda: _rows(paths))
    qtbot.addWidget(widget)

    assert "pathogen_only_here" in widget.plan_text()


def test_nothing_attached_is_stated_rather_than_merged(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel()
    qtbot.addWidget(widget)

    assert widget.merge() is None
    assert "still runs in the regression" in widget.report.toPlainText()


# --------------------------------------------------------------------------- #
#  Everything that can go wrong with a file, and none of it takes the tab down
# --------------------------------------------------------------------------- #

def test_a_file_that_is_not_a_database_is_named_in_its_own_row(
        qtbot, two_plates, tmp_path):
    """A .db that is a renamed CSV, or half-copied over the network. One bad
    file must cost its own row and not the list."""
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    broken = tmp_path / "broken.db"
    broken.write_text("this is not a database\n")
    widget = DatabaseMergePanel(
        lambda: _rows(list(two_plates) + [str(broken)],
                      plates=["plate1", "plate2", "plate3"]))
    qtbot.addWidget(widget)

    assert "could not be read" in widget.table.item(2, 6).text()
    assert widget.table.item(0, 0).text() == "plate1"
    assert "Could not read" in widget.report.toPlainText()


def test_a_database_that_vanishes_before_the_merge_is_named_not_skipped(
        panel, two_plates):
    """The list is re-read at merge time, so a file that has gone away since
    the tab was opened is left out -- and SAID, because a merge that quietly
    covered one plate instead of two is a result about the wrong experiment.
    """
    from spacr.multi_database import SOURCE_COLUMN

    os.remove(two_plates[1])
    frame = panel.merge()

    assert set(frame[SOURCE_COLUMN]) == {"measurements"}
    assert "not on disk" in panel.report.toPlainText()
    assert "plate2" in panel.report.toPlainText()


def test_a_merge_that_fails_outright_reports_instead_of_raising(
        qtbot, two_plates, monkeypatch):
    """Anything the two merge modules can raise that is not a refusal: the
    panel is a renderer and must not take the window down with it."""
    from spacr.qt.widgets import measurement_scan_panel as module

    widget = module.DatabaseMergePanel(lambda: _rows(two_plates))
    qtbot.addWidget(widget)

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the disk went away mid-read")

    monkeypatch.setattr(module, "merge_across_databases", _explode)

    assert widget.merge() is None
    assert "did not finish" in widget.report.toPlainText()
    assert "the disk went away mid-read" in widget.report.toPlainText()


def test_a_metadata_read_that_blows_up_mid_refresh_still_renders_the_list(
        qtbot, two_plates, monkeypatch):
    """describe_merge reads sqlite while the user is still choosing, so it can
    fail on a file that was readable a moment ago."""
    from spacr.qt.widgets import measurement_scan_panel as module

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the disk went away")

    widget = module.DatabaseMergePanel(lambda: _rows(two_plates))
    qtbot.addWidget(widget)
    monkeypatch.setattr(module, "describe_merge", _explode)
    widget.refresh()

    assert widget.table.rowCount() == 2
    assert "the disk went away" in widget.table.item(0, 6).text()
    assert "the disk went away" in widget.report.toPlainText()


def test_a_child_table_that_cannot_be_described_is_named_beside_its_table(
        qtbot, two_plates, monkeypatch):
    from spacr.qt.widgets import measurement_scan_panel as module

    widget = module.DatabaseMergePanel(lambda: _rows(two_plates))
    qtbot.addWidget(widget)
    real = module.describe_merge

    def _explode_for_pathogen(paths, table, **kwargs):
        if table == "pathogen":
            raise RuntimeError("no such table: pathogen")
        return real(paths, table, **kwargs)

    monkeypatch.setattr(module, "describe_merge", _explode_for_pathogen)

    assert "pathogen: could not be read" in widget.plan_text()


def test_an_anchor_with_no_object_label_is_refused_by_name(qtbot, tmp_path):
    """Without an object label there is nothing to join a child onto, and the
    message says which column is missing rather than failing in pandas."""
    from spacr.merge_tables import MergeError
    from spacr.qt.widgets.measurement_scan_panel import merge_across_databases

    path = os.path.join(str(tmp_path), "flat.db")
    with sqlite3.connect(path) as db:
        pd.DataFrame({"plateID": ["plate1"], "area": [1.0]}).to_sql(
            "cell", db, index=False)

    with pytest.raises(MergeError) as refusal:
        merge_across_databases([path], ["cell"])

    assert "object_label" in str(refusal.value)


def test_databases_with_no_table_in_common_offer_nothing(qtbot, tmp_path):
    from spacr.qt.widgets.measurement_scan_panel import joinable_tables

    paths = [_database(tmp_path / "p1", "plate1", tables=("cell",)),
             _database(tmp_path / "p2", "plate2", tables=("pathogen",))]

    assert joinable_tables(paths) == ()
    assert joinable_tables([]) == ()


# --------------------------------------------------------------------------- #
#  The rules themselves, editable where they already were
# --------------------------------------------------------------------------- #

def test_the_aggregation_rules_open_on_the_columns_about_to_be_merged(
        qtbot, panel):
    """The Gate Editor's dialog, not a second one: two editors of one decision
    is how they come to disagree."""
    panel.show_aggregation_rules()

    assert panel._rules_dialog is not None
    panel._rules_dialog.rules_changed.emit({"pathogen_area": "max"})

    assert panel.overrides == {"pathogen_area": "max"}
    frame = panel.merge().set_index(["plateID", "object_label"])
    assert frame.loc[("plate1", 1), "pathogen_area"] == 30.0
    panel._rules_dialog.close()


def test_the_rules_open_on_the_merged_frame_once_there_is_one(qtbot, panel):
    panel.merge()
    panel.show_aggregation_rules()

    assert panel._rules_dialog is not None
    panel._rules_dialog.close()


def test_the_rules_say_there_is_nothing_to_show_before_a_database_arrives(
        qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    said = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: said.append(args[2]))
    widget = DatabaseMergePanel()
    qtbot.addWidget(widget)

    widget.show_aggregation_rules()

    assert said and "nothing to show" in said[0]


def test_the_rules_report_a_preview_that_cannot_be_read(
        qtbot, panel, monkeypatch):
    """The dialog needs real columns with real types, so it reads a couple of
    hundred rows. That read can fail, and a dialog that simply never opened
    would look like a broken button."""
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt.widgets import measurement_scan_panel as module

    said = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: said.append(args[2]))

    def _explode(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(module, "read_merged", _explode)
    panel.show_aggregation_rules()

    assert said and "locked" in said[0]
    assert panel._rules_dialog is None


# --------------------------------------------------------------------------- #
#  The pure functions, driven directly
# --------------------------------------------------------------------------- #

def test_default_aggregation_columns_re_walks_the_rule_table():
    """It must not carry its own list: adding a rule to merge_tables has to
    remove a column from this report without anybody editing the panel."""
    from spacr.merge_tables import AGGREGATION_RULES
    from spacr.qt.widgets.measurement_scan_panel import (
        default_aggregation_columns)

    columns = ["pathogen_area", "pathogen_perimeter", "cell_wobble",
               "object_label", "pathogen_channel_1_min_intensity"]

    fell = default_aggregation_columns(columns)

    assert fell == ("cell_wobble",)
    for column in columns:
        matched = any(re.search(pattern, column)
                      for pattern, _how in AGGREGATION_RULES)
        assert (column in fell) is not matched


def test_an_override_is_not_a_fall_through():
    """An explicit choice is the opposite of a column nobody thought about."""
    from spacr.qt.widgets.measurement_scan_panel import (
        default_aggregation_columns)

    assert default_aggregation_columns(["wobble"]) == ("wobble",)
    assert default_aggregation_columns(
        ["wobble"], overrides={"wobble": "sum"}) == ()


def test_an_anchor_that_is_many_rows_per_cell_is_refused_by_name(two_plates):
    """Not a silent fan-out: the message says which tables can be an anchor."""
    from spacr.merge_tables import MergeError, MergePolicy
    from spacr.qt.widgets.measurement_scan_panel import merge_across_databases

    with pytest.raises(MergeError) as refusal:
        merge_across_databases(two_plates, ["cell"],
                               policy=MergePolicy(primary="pathogen"))

    assert "pathogen" in str(refusal.value)
    assert "cell" in str(refusal.value)
