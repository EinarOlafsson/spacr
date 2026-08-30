"""The file-input widgets meeting inputs nobody planned for.

A measurements database whose whole path is generic words, a saved settings
file carrying a row that is not a row, an empty pair a user added and never
filled in, a Remove press with nothing selected, and a file dialog asked to
reopen beside a folder that has since been deleted. Every one of these is
something a real session produces, and each has one wrong answer that is
silent: a database credited to the wrong plate, a phantom plate in the value
handed to the regression, or a dialog that refuses to open. The tests below
drive each path and ask what the user would see -- the value, the row count,
the status line, the folder the dialog was pointed at.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt.widgets import file_list as fl  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def paired(qtbot):
    """An empty score/count/database table, live and offscreen."""
    widget = fl.PairedFileTableWidget()
    qtbot.addWidget(widget)
    return widget


def _listing(qtbot, **kwargs):
    """A live FilePathListWidget for a CSV setting."""
    widget = fl.FilePathListWidget(kind="csv", **kwargs)
    qtbot.addWidget(widget)
    return widget


# ----------------------------------------------------- naming a database's plate

def test_a_database_under_only_generic_folders_claims_no_plate_at_all():
    """A path that says nothing about WHICH plate must pair with nothing.

    spaCR writes every plate's database to ``<plate>/measurements/
    measurements.db``, so the folder walk is the only thing that can tell two
    of them apart. When the folders above are themselves generic -- ``data``,
    ``measurements`` -- the walk runs out of levels and has to come back
    empty-handed. If it instead invented a token from whatever it last read,
    one plate's per-object measurements would be quietly attached to another
    plate's regression row, and nothing on screen would say so.
    """
    anonymous = os.path.join(os.sep, "experiments", "data", "measurements",
                             "measurements.db")
    named = os.path.join(os.sep, "screens", "plate2", "measurements",
                         "measurements.db")
    assert fl._database_tokens(anonymous) == set()
    assert fl._database_tokens(named) == {"plate2"}

    rows = fl.suggest_file_pairs(
        [os.path.join(os.sep, "x", "plate1_scores.csv")],
        [os.path.join(os.sep, "x", "plate1_counts.csv")],
        databases=[anonymous])
    assert len(rows) == 2
    assert rows[0]["plate"] == "plate1"
    assert rows[0]["database"] is None
    assert rows[1] == {"plate": "", "score": None, "count": None,
                       "database": anonymous}

    paired_rows = fl.suggest_file_pairs(
        [os.path.join(os.sep, "x", "plate2_scores.csv")],
        [os.path.join(os.sep, "x", "plate2_counts.csv")],
        databases=[named])
    assert len(paired_rows) == 1
    assert paired_rows[0]["database"] == named


# ------------------------------------------------------------ honouring a pin

def test_a_pinned_database_finds_its_plate_after_the_csv_it_was_pinned_beside_is_gone(
        paired):
    """A hand-placed database must survive the user replacing that row's CSV.

    The pin remembers the row by its score file, its count file and its plate
    label, in that order, because a row's identity is whichever of those still
    exists. Drop a corrected score CSV and the old filename is no longer in
    the table; if the search stopped at the first key it could not match, the
    next re-proposal would leave the database on whatever row the token
    pairing preferred -- undoing an assignment the user made by hand, without
    a word.
    """
    database = os.path.join(os.sep, "db", "measurements.db")
    paired._pinned = {database: {"plate": "plate2",
                                 "score": os.path.join(os.sep, "x",
                                                       "plate2_OLD.csv"),
                                 "count": None}}
    rows = [{"plate": "plate1",
             "score": os.path.join(os.sep, "x", "plate1_scores.csv"),
             "count": os.path.join(os.sep, "x", "plate1_counts.csv"),
             "database": database},
            {"plate": "plate2",
             "score": os.path.join(os.sep, "x", "plate2_NEW.csv"),
             "count": os.path.join(os.sep, "x", "plate2_counts.csv"),
             "database": None}]

    assert paired._row_for_anchor(rows, paired._pinned[database]) == 1

    result = paired._apply_pinned(rows)
    assert [row["plate"] for row in result] == ["plate1", "plate2"]
    assert result[0]["database"] is None
    assert result[1]["database"] == database


def test_a_pin_naming_nothing_in_the_table_leaves_the_proposal_alone(paired):
    """An anchor that matches no row must not move a database at random.

    Every key of the pin can go stale at once -- the user replaces both CSVs
    and renames the plate. The search then has nothing to go on, and the only
    safe answer is the pairing the tokens proposed. Returning row 0 instead
    (the classic "no match, use the first one") would attach that database to
    whichever plate happened to sort first.
    """
    database = os.path.join(os.sep, "db", "measurements.db")
    stale = {"plate": "plateX", "score": os.path.join(os.sep, "x", "old.csv"),
             "count": None}
    rows = [{"plate": "plate1", "score": os.path.join(os.sep, "x", "s1.csv"),
             "count": None, "database": None},
            {"plate": "plate2", "score": os.path.join(os.sep, "x", "s2.csv"),
             "count": None, "database": database}]
    assert paired._row_for_anchor(rows, stale) is None

    paired._pinned = {database: stale}
    result = paired._apply_pinned(rows)
    assert result[0]["database"] is None
    assert result[1]["database"] == database


# ------------------------------------------------------- rows that are not rows

def test_a_settings_file_carrying_a_stray_string_still_loads_its_real_plates(
        paired):
    """A saved value written by an older screen must not empty the table.

    ``set_value`` is what a settings CSV, a saved run, or an undo hands the
    widget, and those come from outside this class. A bare string among the
    row dicts -- an old two-column format, a hand-edited file -- must be
    skipped, not turned into a row and not allowed to abort the load: losing
    the two good plates because of the third entry is the difference between
    a screen that opens and one that opens empty.
    """
    good_one = {"plate": "plate1",
                "score": os.path.join(os.sep, "x", "plate1_scores.csv"),
                "count": None, "database": None}
    good_two = {"plate": "plate2", "score": None,
                "count": os.path.join(os.sep, "x", "plate2_counts.csv"),
                "database": None}
    paired.set_value([good_one, "/x/loose_string.csv", good_two, 7, None])

    assert paired.table.rowCount() == 2
    assert paired.get_value() == [good_one, good_two]
    assert paired._cell(0, 0) == "plate1"
    assert paired._cell(1, paired.SIDE_COLUMNS["count"]) == good_two["count"]


def test_an_empty_pair_row_is_a_place_to_type_and_not_a_plate_in_the_value(
        paired):
    """"Add empty pair" gives the user a row to type into, not a plate.

    The button exists so somebody can paste two paths side by side. Until
    they do, that row names no files, and the value handed to the run must not
    contain it: a row of three ``None``s reaching the regression is a plate
    with no score and no count, which fails deep inside the fit rather than in
    the panel. The status line has to agree, or the user is told they have two
    plates when they have one.
    """
    real = {"plate": "plate1",
            "score": os.path.join(os.sep, "x", "plate1_scores.csv"),
            "count": os.path.join(os.sep, "x", "plate1_counts.csv"),
            "database": None}
    paired.set_value([real])
    paired._append_row({})
    paired._refresh_status()

    assert paired.table.rowCount() == 2
    assert paired.get_value() == [real]
    assert paired._cell(1, paired.RULE_COLUMN) == "resolved at run"
    assert paired.status.text() == (
        "0 of 1 plate row carry a measurements database.")


def test_a_plate_named_but_unfilled_row_never_reaches_the_run(paired):
    """A row with only a plate label is still not a plate that can be fitted.

    A user who types a plate id first and means to add the CSVs afterwards
    leaves exactly this behind, and a settings file saved mid-edit carries it
    forward. The value must be filtered on the FILES, not on whether anything
    was typed in the row, or the run starts with a plate whose score CSV is
    ``None``.
    """
    paired.set_value([{"plate": "plate9"},
                      {"plate": "plate1",
                       "score": os.path.join(os.sep, "x", "plate1_scores.csv")}])

    assert paired.table.rowCount() == 2
    assert paired._cell(0, 0) == "plate9"
    values = paired.get_value()
    assert len(values) == 1
    assert values[0]["plate"] == "plate1"
    assert values[0]["score"] == os.path.join(os.sep, "x", "plate1_scores.csv")


# ------------------------------------------------------------- Remove, pressed

def test_remove_with_nothing_selected_does_not_announce_a_change(paired):
    """Remove must be inert when no row is selected, and only then.

    ``value_changed`` is what re-writes the settings and marks the screen
    dirty. A Remove press that emits it with nothing selected would mark an
    unedited screen as changed and re-save a file the user never touched --
    and, worse, the same press really does drop a row when one IS selected, so
    the two cases have to be told apart rather than made safe by accident.
    """
    rows = [{"plate": "plate1",
             "score": os.path.join(os.sep, "x", "plate1_scores.csv"),
             "count": None, "database": None},
            {"plate": "plate2",
             "score": os.path.join(os.sep, "x", "plate2_scores.csv"),
             "count": None, "database": None}]
    paired.set_value(rows)
    seen = []
    paired.value_changed.connect(lambda: seen.append(1))

    paired.table.selectRow(0)
    paired._remove()
    assert paired.table.rowCount() == 1
    assert [row["plate"] for row in paired.get_value()] == ["plate2"]
    after_a_real_removal = len(seen)
    assert after_a_real_removal >= 1

    paired.table.clearSelection()
    paired._remove()
    assert paired.table.rowCount() == 1
    assert [row["plate"] for row in paired.get_value()] == ["plate2"]
    assert len(seen) == after_a_real_removal


def test_removing_a_row_takes_its_database_pin_with_it(paired):
    """A deleted row must not put its database back on the next drop.

    The pin is what makes a hand-placed database survive a re-proposal, so a
    pin that outlives its row resurrects a file the user just deleted the next
    time any CSV arrives. Dropping the pin with the row is what makes Remove
    mean remove.
    """
    database = os.path.join(os.sep, "db", "plate1.db")
    paired.set_value([{"plate": "plate1",
                       "score": os.path.join(os.sep, "x", "plate1_scores.csv"),
                       "count": None, "database": database}])
    paired._pinned = {database: paired._anchor_for_row(0)}

    paired.table.selectRow(0)
    paired._remove()

    assert paired.table.rowCount() == 0
    assert paired.get_value() == []
    assert database not in paired._pinned


# --------------------------------------------------- where the dialog reopens

def test_the_file_dialog_reopens_beside_the_last_file_unless_that_folder_is_gone(
        qtbot, monkeypatch, tmp_path):
    """The picker has to start somewhere real, or it opens at the wrong place.

    Reopening beside the file already listed is what saves a user four clicks
    per plate. But a settings file is routinely written on one machine and run
    on another, where that folder does not exist -- and handing Qt a
    non-existent start directory is how a dialog opens at the filesystem root
    or at the process's working directory instead. The empty string is the
    deliberate "you choose", so both cases must be driven here.
    """
    listed = tmp_path / "plate1_scores.csv"
    listed.write_text("gene,score\na,1\n", encoding="utf-8")
    widget = _listing(qtbot)

    starts = []

    def _record(parent, title, start, filters):
        starts.append(start)
        return [], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", _record)

    widget.set_value([str(listed)])
    assert widget.pick_files() == 0
    assert starts == [str(tmp_path)]

    widget.set_value([str(tmp_path / "vanished" / "plate1_scores.csv")])
    assert widget.pick_files() == 0
    assert starts == [str(tmp_path), ""]
    assert widget._hint.text() == "1 file selected — 1 not found (shown in red)"


def test_a_remembered_folder_wins_over_the_listed_file(qtbot, monkeypatch,
                                                       tmp_path):
    """Where the user last browsed beats where the listed file happens to live.

    A user adding four plates from one folder browses there once; the second
    press must open in that folder even though the file already listed came
    from somewhere else entirely. If the listed file won, every press after
    the first would send them back to the folder they had already finished
    with.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    listed = tmp_path / "old" / "plate1_scores.csv"
    listed.parent.mkdir()
    listed.write_text("gene,score\na,1\n", encoding="utf-8")
    widget = _listing(qtbot)
    widget.set_value([str(listed)])

    starts = []

    def _record(parent, title, start, filters):
        starts.append(start)
        return [], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", _record)

    assert widget.pick_files() == 0
    assert starts == [str(tmp_path / "old")]

    widget._last_directory = str(elsewhere)
    assert widget.pick_files() == 0
    assert starts == [str(tmp_path / "old"), str(elsewhere)]


def test_an_empty_list_opens_the_dialog_with_no_start_directory(qtbot,
                                                                monkeypatch,
                                                                tmp_path):
    """With nothing listed there is nothing to reopen beside.

    The widget must ask for the platform default rather than guessing, and it
    must start guessing again the moment a file IS listed -- the two answers
    come from the same method, so a change to one silently changes the other.
    """
    widget = _listing(qtbot)
    starts = []

    def _record(parent, title, start, filters):
        starts.append(start)
        return [], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", _record)

    assert widget.get_value() == []
    assert widget.pick_files() == 0
    assert starts == [""]

    listed = tmp_path / "plate1_scores.csv"
    listed.write_text("gene,score\na,1\n", encoding="utf-8")
    widget.set_value([str(listed)])
    assert widget.pick_files() == 0
    assert starts == ["", str(tmp_path)]
