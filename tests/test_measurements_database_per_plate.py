"""A measurements database per plate, in the regression input table.

Instruction 130 section A. A row of that table is one PLATE: its score CSV,
its count CSV and now its measurements database. The database column is
filled the way instruction 107 filled the other two -- BY ADDITION, with the
whole table re-proposed from filename tokens every time something arrives --
because the failure 107 was written for is exactly the one this column can
reproduce: files that arrive in a different order to the CSVs, land on the
wrong plates, and produce a run with no error in it.

What these tests hold to:

* databases dropped in a different order to the CSVs reach their own plates;
* a database dropped on a ROW belongs to THAT plate, whatever its name says,
  and stays there when the next CSV re-proposes the table;
* a database that names no plate goes to the first row without one and the
  panel SAYS which -- never silently to row 0;
* a database that is not on disk is named before the run, not during it;
* a plate with NO database is legal and the regression still runs.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

pytestmark = pytest.mark.qt


def _tracked(qtbot, widget):
    """Hand a widget to qtbot so Qt destroys it with the test."""
    qtbot.addWidget(widget)
    return widget


def _score(tmp_path, plate):
    path = tmp_path / f"plate{plate}_dv.csv"
    path.write_text(f"path,pred,plate\na,0.5,plate{plate}\n")
    return str(path)


def _count(tmp_path, plate):
    path = tmp_path / f"plate_{plate}_unique_combinations.csv"
    path.write_text("row_name,column_name,grna_name,count\nr1,c1,g,5\n")
    return str(path)


def _database(tmp_path, plate, name="measurements.db"):
    """A real sqlite file where spaCR actually writes it.

    ``<plate>/measurements/measurements.db`` -- every plate's is called the
    same thing, which is the whole difficulty.
    """
    folder = tmp_path / f"plate{plate}" / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE IF NOT EXISTS cell (object_label INTEGER)")
    connection.commit()
    connection.close()
    return str(path)


# ------------------------------------------------------- pairing by addition


def test_every_plates_database_has_the_same_filename_so_the_folder_must_count(
        tmp_path):
    """The reason the database column needs its own tokeniser.

    ``_pair_tokens`` reads the basename, which is all a CSV gives it. spaCR
    writes every plate's database to ``<plate>/measurements/measurements.db``,
    so on basenames alone all four plates are the same word, every candidate
    ties, and 107's tie rule -- correctly -- refuses to guess. The result was
    not a wrong pairing but NO pairing: four databases, four orphan rows.
    """
    from spacr.qt.widgets.file_list import _database_tokens, _pair_tokens

    first, second = _database(tmp_path, 1), _database(tmp_path, 2)
    assert _pair_tokens(first) == _pair_tokens(second), (
        "if this ever differs the tokeniser has changed and this column's "
        "extra work may no longer be needed")
    assert "plate1" in _database_tokens(first)
    assert "plate2" in _database_tokens(second)
    assert _database_tokens(first) != _database_tokens(second)


def test_only_the_nearest_folder_that_says_anything_names_the_plate(tmp_path):
    """Climbing the tree until something matches is how you guess wrong.

    ``measurements`` is skipped because it says nothing about WHICH plate --
    that is the only reason to look above the file at all. Everything higher
    is the user's own filing: one project folder called ``plate1_rerun`` would
    otherwise name plate 1 for every database anywhere beneath it, and 107's
    rule is that an ambiguous match is left for the user to aim, never taken.
    """
    from spacr.qt.widgets.file_list import _database_tokens, suggest_file_pairs

    folder = tmp_path / "plate1_rerun" / "exports"
    folder.mkdir(parents=True)
    database = folder / "measurements.db"
    sqlite3.connect(str(database)).close()

    tokens = _database_tokens(str(database))
    assert _database_tokens("measurements.db") == set(), (
        "a bare filename has no folder to read and no plate to claim")
    assert "exports" in tokens
    assert "plate1" not in tokens, "two levels up is the user's filing, not a plate"

    rows = suggest_file_pairs([_score(tmp_path, 1)], [_count(tmp_path, 1)],
                              databases=[str(database)])
    assert rows[0]["database"] is None, "unplaced, not placed by a guess"
    assert rows[1]["database"] == str(database)


def test_databases_dropped_in_a_third_order_still_reach_their_own_plates(
        qtbot, tmp_path):
    """The 107 property, extended to the third column.

    Scores arrive 1-2-3, counts arrive 3-2-1 and databases arrive 2-3-1 --
    three different orders, one per column, which is what three drags from
    three differently-sorted file managers look like. Pairing by list
    position would attach plate 2's measurements to plate 1's regression
    result: a number that is wrong and looks fine.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    scores = {plate: _score(tmp_path, plate) for plate in (1, 2, 3)}
    counts = {plate: _count(tmp_path, plate) for plate in (1, 2, 3)}
    databases = {plate: _database(tmp_path, plate) for plate in (1, 2, 3)}

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([scores[1], scores[2], scores[3]], "score")
    for plate in (3, 2, 1):
        paired.add_paths_for_side([counts[plate]], "count")
    for plate in (2, 3, 1):
        paired.add_paths_for_side([databases[plate]], "database")

    rows = paired.get_value()
    assert len(rows) == 3, "three plates are three rows, not six"
    by_score = {row["score"]: row for row in rows}
    for plate in (1, 2, 3):
        row = by_score[scores[plate]]
        assert row["count"] == counts[plate]
        assert row["database"] == databases[plate], (
            f"plate {plate}'s row got {row['database']}")


def test_a_database_the_pairing_cannot_place_is_kept_not_dropped(tmp_path):
    """An unmatched database is a row of its own, waiting for its CSVs.

    Discarding it would lose a file the user dropped; guessing a row for it
    inside the proposal would be pairing by arrival order under another name.
    It waits instead, and pairs the moment that plate's CSVs arrive.
    """
    from spacr.qt.widgets.file_list import suggest_file_pairs

    rows = suggest_file_pairs([_score(tmp_path, 1)], [_count(tmp_path, 1)],
                              databases=[_database(tmp_path, 7)])
    assert len(rows) == 2
    assert rows[0]["database"] is None
    assert rows[1]["score"] is None and rows[1]["count"] is None
    assert rows[1]["database"].endswith("measurements.db")
    assert rows[1]["plate"] == "plate7", "it can still name its own plate"


def test_the_two_argument_call_every_existing_caller_makes_still_works(
        tmp_path):
    """``databases`` is keyword-only with a default, so 107's callers are
    untouched -- and every row now carries the key, so the Measurements tab
    can read it without checking whether it is there."""
    from spacr.qt.widgets.file_list import suggest_file_pairs

    rows = suggest_file_pairs([_score(tmp_path, 1)], [_count(tmp_path, 1)])
    assert len(rows) == 1
    assert rows[0]["count"].endswith("plate_1_unique_combinations.csv")
    assert rows[0]["database"] is None


def test_a_side_must_be_one_of_the_three_columns(qtbot):
    """The third side joins the vocabulary rather than getting an adder of
    its own that would keep a private list and pair by arrival."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    with pytest.raises(ValueError, match="score.*count.*database"):
        paired.add_paths_for_side(["/tmp/x.csv"], "sideways")
    assert paired.add_paths_for_side(["/tmp/x.db"], "database") == 1
    assert paired.add_paths_for_side([], "database") == 0


def test_the_picker_goes_through_the_same_seam_as_a_drop(qtbot, tmp_path,
                                                         monkeypatch):
    """"Add measurements DBs…" must not pair by the dialog's sort order.

    The file dialog hands the selection over in whatever order the file
    manager is sorted by, which is one column-header click away from being
    reversed. The picker therefore adds through ``add_paths_for_side`` --
    the same call a drop makes -- rather than filling cells itself.
    """
    from PySide6.QtWidgets import QFileDialog

    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    databases = [_database(tmp_path, 2), _database(tmp_path, 1)]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *args, **kwargs: (list(databases), ""))

    paired._pick("database")

    rows = paired.get_value()
    assert rows[0]["database"] == databases[1], "plate 1 keeps plate 1's"
    assert rows[1]["database"] == databases[0]


def test_the_picker_takes_nothing_when_the_dialog_is_cancelled(
        qtbot, tmp_path, monkeypatch):
    """A cancelled dialog must leave the table exactly as it was."""
    from PySide6.QtWidgets import QFileDialog

    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *args, **kwargs: ([], ""))

    paired._pick("score")

    assert paired.get_value() == [
        {"plate": None, "score": _score(tmp_path, 1), "count": None,
         "database": None}]


# --------------------------------------------------------- aiming at a row


def _drop_on(widget, path, column, row):
    """Drop ``path`` on one cell of the table, through Qt's real event.

    The point is built from the table's own viewport geometry and checked
    against ``rowAt``/``columnAt`` before the drop, so a test that misses the
    cell it meant to hit fails as a mis-aimed test rather than passing
    quietly through the "no row" fallback it was not written for.
    """
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    widget.show()
    viewport = widget.table.viewport()
    point = QPoint(widget.table.columnViewportPosition(column) + 4,
                   widget.table.rowViewportPosition(row) + 4)
    assert widget.table.columnAt(point.x()) == column
    assert widget.table.rowAt(point.y()) == row
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(QPointF(viewport.mapTo(widget, point)), Qt.CopyAction,
                       mime, Qt.LeftButton, Qt.NoModifier)
    widget.dropEvent(event)
    return event


def test_a_database_dropped_on_a_row_belongs_to_that_plate_whatever_it_is_called(
        qtbot, tmp_path):
    """Aim beats filename, because only the user knows about a renamed file.

    Nothing else in the panel can know that ``plate1_backup.db`` holds plate
    2's cells. Dropping it on plate 2's row says so, and the table must take
    the user's word for it rather than believing the filename.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.resize(900, 300)
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    misnamed = exports / "plate1_backup.db"
    sqlite3.connect(str(misnamed)).close()

    event = _drop_on(paired, misnamed,
                     PairedFileTableWidget.SIDE_COLUMNS["database"], 1)
    assert event.isAccepted()

    rows = paired.get_value()
    assert rows[0]["database"] is None, "plate 1's row was not the target"
    assert rows[1]["database"] == str(misnamed)
    assert "row 2" in paired.status.text()


def test_a_csv_dropped_on_the_database_column_is_still_read_as_a_csv(
        qtbot, tmp_path):
    """Aim decides the ROW for a database; it does not change what a file is.

    A count table released over the database column is a mis-aim, and filing
    it as this plate's measurements would put a CSV where the merge expects
    sqlite. The header still decides which of the two CSV columns it belongs
    in, which is the rule the Parameter Sweep shares.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.resize(900, 300)
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    count = _count(tmp_path, 1)

    _drop_on(paired, count, PairedFileTableWidget.SIDE_COLUMNS["database"], 0)

    row = paired.get_value()[0]
    assert row["count"] == count, "the header says count, so count it is"
    assert row["database"] is None


def test_an_explicit_attachment_survives_the_next_dropped_csv(qtbot, tmp_path):
    """The re-proposal must not undo what the user did by hand.

    Every addition re-proposes the whole table -- that is what makes the
    columns order-independent. Without a record of the manual attachment, the
    next CSV dropped would silently move a hand-placed database back onto the
    row its FILENAME suggests, which is the row the user just overruled.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.resize(900, 300)
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    misnamed = exports / "plate1_backup.db"
    sqlite3.connect(str(misnamed)).close()
    _drop_on(paired, misnamed,
             PairedFileTableWidget.SIDE_COLUMNS["database"], 1)

    paired.add_paths_for_side([_score(tmp_path, 3)], "score")
    paired.add_paths_for_side([_count(tmp_path, 3)], "count")

    rows = paired.get_value()
    assert len(rows) == 3
    assert rows[1]["database"] == str(misnamed), (
        "the hand-placed database moved when plate 3 arrived")
    assert rows[0]["database"] is None and rows[2]["database"] is None


def test_clearing_a_database_cell_by_hand_does_not_bring_it_back(
        qtbot, tmp_path):
    """The table is editable, and emptying a cell means "not this one".

    The record of a manual attachment must not outlive the attachment
    itself, or the next dropped CSV re-proposes the table and puts back the
    file the user just deleted.
    """
    from PySide6.QtWidgets import QTableWidgetItem

    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    paired.add_paths_for_side([_count(tmp_path, 1)], "count")
    paired.attach_database(_database(tmp_path, 1), 0)

    paired.table.setItem(
        0, PairedFileTableWidget.SIDE_COLUMNS["database"], QTableWidgetItem(""))
    paired.add_paths_for_side([_score(tmp_path, 2)], "score")

    assert [row["database"] for row in paired.get_value()] == [None, None]


def test_a_hand_placed_database_that_the_names_agree_with_stays_put(
        qtbot, tmp_path):
    """The record of a manual attachment is not allowed to fight the
    proposal when the two already agree -- it must be a no-op, not a swap."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    paired.add_paths_for_side([_count(tmp_path, 1)], "count")
    database = _database(tmp_path, 1)
    paired.attach_database(database, 0)

    paired.add_paths_for_side([_score(tmp_path, 2)], "score")
    paired.add_paths_for_side([_count(tmp_path, 2)], "count")

    rows = paired.get_value()
    assert rows[0]["database"] == database
    assert rows[1]["database"] is None


def test_a_database_placed_on_an_empty_row_is_kept_when_that_row_goes(
        qtbot, tmp_path):
    """"Add empty pair" makes a row with nothing to remember it by.

    A database attached to it cannot be anchored -- there is no score, no
    count and no plate to name -- so when the table is next re-proposed the
    filename pairing decides, and the file is still there rather than lost
    with the row it was sitting on.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired._append_row({})
    database = _database(tmp_path, 2)
    assert "row 1" in paired.attach_database(database, 0)

    paired.add_paths_for_side([_score(tmp_path, 2)], "score")
    paired.add_paths_for_side([_count(tmp_path, 2)], "count")

    rows = paired.get_value()
    assert len(rows) == 1, "the empty row is not a second plate"
    assert rows[0]["database"] == database
    assert rows[0]["plate"] == "plate2"


def test_a_database_that_names_no_plate_says_which_row_it_went_to(
        qtbot, tmp_path):
    """"attach it to the first row that has none, and SAY which."

    Row 0 is not the answer -- it is the answer that looks like one. A
    database credited to a plate in silence is measurements attributed to the
    wrong screen with nothing on screen disagreeing.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    paired.attach_database(_database(tmp_path, 1))       # names plate 1
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    anonymous = exports / "experiment_final.db"
    sqlite3.connect(str(anonymous)).close()

    message = paired.attach_database(str(anonymous))

    rows = paired.get_value()
    assert rows[1]["database"] == str(anonymous), (
        "row 1 already had one, so the first FREE row takes it")
    assert "plate2" in message and "row 2" in message
    assert "first row with no database" in message
    assert message in paired.status.text()


def test_an_aimed_drop_replaces_and_says_what_it_replaced(qtbot, tmp_path):
    """Aiming at a full cell is a correction, and corrections are stated.

    The path that leaves is gone from the table, so the sentence has to name
    it: a file silently discarded by a drop is a file the user believes is
    still attached.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    paired.add_paths_for_side([_count(tmp_path, 1)], "count")
    first = _database(tmp_path, 1)
    paired.attach_database(first, 0)
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    second = exports / "corrected.db"
    sqlite3.connect(str(second)).close()

    message = paired.attach_database(str(second), 0)

    assert paired.get_value()[0]["database"] == str(second)
    assert "replaced measurements.db" in message
    assert "row 1" in message


def test_dropping_the_same_database_a_second_time_says_where_it_already_is(
        qtbot, tmp_path):
    """A repeated drop is a question, not a command.

    The user dropped it again because they could not see where it went. The
    answer is the row it is on -- not a second copy, and not silence.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    database = _database(tmp_path, 2)
    paired.attach_database(database)

    message = paired.attach_database(database)

    assert "already on" in message and "plate2" in message
    assert [row["database"] for row in paired.get_value()] == [None, database]


def test_a_database_with_no_plate_row_to_take_it_waits_instead_of_vanishing(
        qtbot, tmp_path):
    """Databases dropped before their CSVs are the ordinary case.

    There is no row to attach them to yet, so they hold rows of their own and
    say so. Refusing them would make the panel order-dependent in exactly the
    way instruction 107 removed.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    anonymous = exports / "experiment_final.db"
    sqlite3.connect(str(anonymous)).close()

    message = paired.attach_database(str(anonymous))

    assert "row 1 of its own" in message
    assert "when that plate's CSVs arrive" in message
    assert paired.get_value() == [
        {"plate": None, "score": None, "count": None,
         "database": str(anonymous)}]


def test_a_database_names_the_plate_when_the_csvs_could_not(qtbot, tmp_path):
    """A score CSV with no partner leaves the plate cell empty; the database
    folder names it, and the row stops being called nothing."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    assert paired.get_value()[0]["plate"] is None

    paired.add_paths_for_side([_database(tmp_path, 1)], "database")

    assert paired.get_value()[0]["plate"] == "plate1"


def test_attaching_nothing_is_refused_rather_than_attached_to_row_one(
        qtbot, tmp_path):
    """An empty path is not a database. Taken literally it would be written
    into a cell, read back as "no database", and then found again as the
    first row without one -- reporting an attachment that never happened."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    with pytest.raises(ValueError, match="needs a path"):
        paired.attach_database("   ")
    assert paired.get_value()[0]["database"] is None


def test_a_row_that_is_not_in_the_table_is_refused_rather_than_guessed(
        qtbot, tmp_path):
    """A caller aiming at a row that no longer exists gets an error, not
    row 0: the row indices in a re-proposed table are not stable enough to
    fall back on."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    with pytest.raises(IndexError, match="row 4"):
        paired.attach_database(_database(tmp_path, 1), 4)


# ------------------------------------------------- what the panel must say


def test_a_database_that_is_not_on_disk_is_named_before_the_run(
        qtbot, tmp_path):
    """"a row whose database is missing from disk says so BEFORE the run."

    A settings file is routinely written on one machine and run on another.
    Discovering the missing path inside the run means fitting for minutes
    first; discovering it here costs nothing and names the plate.
    """
    from PySide6.QtCore import Qt

    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    paired.add_paths_for_side([_count(tmp_path, 1)], "count")
    gone = str(tmp_path / "unplugged" / "measurements.db")

    paired.attach_database(gone, 0)

    assert paired.missing_databases() == [(1, "plate1", gone)]
    assert "NOT ON DISK" in paired.status.text()
    assert gone in paired.status.text()
    cell = paired.table.item(0, PairedFileTableWidget.SIDE_COLUMNS["database"])
    assert cell.foreground().color() == Qt.red
    assert "not on disk" in cell.toolTip()
    # And it is still THERE: an unmounted disk is not a reason to delete the
    # user's path behind their back.
    assert paired.get_value()[0]["database"] == gone


def test_a_settings_file_loaded_with_a_missing_database_flags_it_on_open(
        qtbot, tmp_path):
    """The same statement when the value arrives from a saved settings file
    rather than from a drop -- which is the case that actually produces it."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    rows = [{"plate": "plate1", "score": _score(tmp_path, 1),
             "count": _count(tmp_path, 1),
             "database": "/mnt/gone/plate1/measurements/measurements.db"}]
    paired = _tracked(qtbot, PairedFileTableWidget(value=rows))

    assert [number for number, _plate, _path in paired.missing_databases()] == [1]
    assert "NOT ON DISK" in paired.status.text()


def test_the_status_line_counts_the_plates_that_have_a_database(
        qtbot, tmp_path):
    """A plate with no database is normal, so the panel states the ratio
    rather than warning about it."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    paired.add_paths_for_side([_database(tmp_path, 1)], "database")

    assert "1 of 2 plate rows carry a measurements database" in \
        paired.status.text()
    assert "NOT ON DISK" not in paired.status.text()


def test_the_measurements_tab_reads_these_rows_as_they_are(qtbot, tmp_path):
    """The two halves of instruction 130 have to meet without a translator.

    Section B's tab takes the input table's rows and asks them what is
    attached. Feeding it the real ``get_value()`` output is the only thing
    that proves the shapes agree -- a test that built its own dicts would
    pass while the panel showed nothing.

    Removing the row removes the database from the tab too, which is why the
    tab asks a callable each time rather than keeping the list it was handed.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget
    from spacr.qt.widgets.measurement_scan_panel import attached_databases

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    paired.add_paths_for_side([_database(tmp_path, 2)], "database")

    entries = attached_databases(paired.get_value())
    assert [entry.plate for entry in entries] == ["plate1", "plate2"]
    assert [entry.attached for entry in entries] == [False, True], (
        "a plate with no database is listed and disabled, not dropped")
    assert entries[1].path == _database(tmp_path, 2)
    assert entries[1].present and entries[1].status == "ready"

    paired.table.selectRow(1)
    paired._remove()
    assert [entry.attached
            for entry in attached_databases(paired.get_value())] == [False]


# ------------------------------------------------------- the value contract


def test_the_table_keeps_the_database_through_a_round_trip(qtbot, tmp_path):
    """The table is the state, so the column has to be a real column.

    Carrying the database in a dict beside the table looked equivalent and
    was not: ``set_value`` rebuilds every row from the cells, so a key with
    no cell is deleted by the next repaint -- and the value that comes back
    from the panel is the one the run is given.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    rows = [{"plate": "plate1", "score": _score(tmp_path, 1),
             "count": _count(tmp_path, 1),
             "database": _database(tmp_path, 1)}]
    paired = _tracked(qtbot, PairedFileTableWidget(value=rows))

    assert paired.get_value() == rows
    paired.set_value(paired.get_value())
    assert paired.get_value() == rows


def test_a_dropped_database_is_never_read_as_a_score_table(qtbot, tmp_path):
    """A sqlite file has no header, and asking for one used to CRASH.

    ``side_for_header`` opens the file as text and looks for a gRNA and a
    count. On a database that is ``_csv.Error: line contains NUL``, raised
    from inside Qt's drop dispatch where an exception is a dead window rather
    than an error dialog -- and if it had not raised, the answer would have
    been "score", making the regression's response variable a binary file.
    Extension decides for databases, before the sniffer is asked.
    """
    from spacr.qt.widgets.file_list import (
        PairedFileTableWidget, is_database_path, side_for_header,
    )

    database = _database(tmp_path, 1)
    assert side_for_header(database) == "score", (
        "a binary file must come back with the default, not an exception")
    assert is_database_path(database) and not is_database_path(
        _score(tmp_path, 1))

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.resize(900, 300)
    paired.add_paths_for_side([_score(tmp_path, 2)], "score")
    # Aimed at the SCORE column, which for a CSV would win outright.
    _drop_on(paired, database, PairedFileTableWidget.SIDE_COLUMNS["score"], 0)

    row = paired.get_value()[0]
    assert row["score"].endswith("plate2_dv.csv"), "the .db took the score slot"
    assert row["database"] == database


# --------------------------------------------- a plate with no database runs


def test_a_plate_with_no_database_does_not_stop_the_regression(tmp_path):
    """"a plate with NO database is legal."

    The regression is fitted on scores and counts; the database is what makes
    the Measurements tab possible for that plate. Its absence has to disable
    the plate there, not fail the run -- so the row survives normalisation
    and is read like any other.
    """
    import pandas as pd

    from spacr.ml import (
        load_regression_input_pairs, normalize_regression_input_pairs,
    )

    for plate in (1, 2):
        pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                      "plateID": [f"plate{plate}"], "pred": [0.5]}).to_csv(
            tmp_path / f"plate{plate}_dv.csv", index=False)
        pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                      "plateID": [f"plate{plate}"], "grna_name": ["g1"],
                      "count": [5]}).to_csv(
            tmp_path / f"plate{plate}_counts.csv", index=False)

    settings = {"paired_data": [
        {"plate": "plate1", "score": str(tmp_path / "plate1_dv.csv"),
         "count": str(tmp_path / "plate1_counts.csv"),
         "database": _database(tmp_path, 1)},
        {"plate": "plate2", "score": str(tmp_path / "plate2_dv.csv"),
         "count": str(tmp_path / "plate2_counts.csv"),
         "database": None},
    ]}
    pairs, _migrated = normalize_regression_input_pairs(settings)
    counts, scores, audit = load_regression_input_pairs(pairs)

    assert len(pairs) == 2
    assert sorted(counts["plateID"].unique()) == ["plate1", "plate2"]
    assert sorted(scores["plateID"].unique()) == ["plate1", "plate2"]
    assert [row["plate"] for row in audit] == ["plate1", "plate2"]


def test_a_database_only_row_does_not_break_the_run_either(tmp_path):
    """A database dropped before its CSVs is a row with no score and no
    count. It is transient -- the CSVs re-propose it onto its plate -- but a
    user may still press Run while it is showing, and that must fit the
    plates it does have rather than raise."""
    import pandas as pd

    from spacr.ml import (
        load_regression_input_pairs, normalize_regression_input_pairs,
    )

    pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"], "plateID": ["plate1"],
                  "pred": [0.5]}).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"], "plateID": ["plate1"],
                  "grna_name": ["g1"], "count": [5]}).to_csv(
        tmp_path / "c.csv", index=False)

    settings = {"paired_data": [
        {"plate": "plate1", "score": str(tmp_path / "s.csv"),
         "count": str(tmp_path / "c.csv"), "database": None},
        {"plate": "plate9", "score": None, "count": None,
         "database": _database(tmp_path, 9)},
    ]}
    pairs, _migrated = normalize_regression_input_pairs(settings)
    counts, scores, _audit = load_regression_input_pairs(pairs)

    assert len(counts) == 1 and len(scores) == 1
    assert counts["plateID"].tolist() == ["plate1"]


# ------------------------------------------ the same gesture, aimed loosely


class _FakeModel:
    def __init__(self, widgets):
        self._widgets = widgets


class _FakeScreen:
    app_key = "regression"

    def __init__(self, widgets=None):
        if widgets is not None:
            self._settings_model = _FakeModel(widgets)
        self.logged: list[str] = []
        self.src = None

    class _Console:
        def __init__(self, screen):
            self._screen = screen

        def append_stdout(self, text):
            self._screen.logged.append(text)

    @property
    def _console(self):
        return self._Console(self)


def test_a_database_dropped_on_the_screen_reaches_the_plate_rows(
        qtbot, tmp_path):
    """The two gestures have to agree.

    Dropping ``measurements.db`` ON the input table attaches it to a plate.
    Dropping it on the screen AROUND the table used to set ``src`` -- a key
    the regression panel does not display -- so the same file, released two
    inches higher, silently did nothing the user could see.
    """
    from spacr.qt.dnd_handlers import MeasurementsDropHandler
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1), _score(tmp_path, 2)],
                              "score")
    paired.add_paths_for_side([_count(tmp_path, 1), _count(tmp_path, 2)],
                              "count")
    screen = _FakeScreen({"paired_data": paired})
    handler = MeasurementsDropHandler()

    import pathlib
    database = pathlib.Path(_database(tmp_path, 2))
    assert handler.can_accept(database)
    handler.apply(database, screen)

    rows = paired.get_value()
    assert rows[1]["database"] == str(database), "plate 2's row takes it"
    assert rows[0]["database"] is None
    assert any("plate2" in line for line in screen.logged), (
        "the console has to carry the same sentence the panel shows")


def test_the_plate_folder_above_the_database_is_accepted_too(qtbot, tmp_path):
    """Users drag the plate folder as often as the file inside it, and both
    name the same database."""
    import pathlib

    from spacr.qt.dnd_handlers import MeasurementsDropHandler
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side([_score(tmp_path, 1)], "score")
    paired.add_paths_for_side([_count(tmp_path, 1)], "count")
    expected = _database(tmp_path, 1)
    screen = _FakeScreen({"paired_data": paired})

    MeasurementsDropHandler().apply(pathlib.Path(tmp_path / "plate1"), screen)

    assert paired.get_value()[0]["database"] == expected


def test_a_screen_without_a_plate_table_still_gets_its_source_set(tmp_path):
    """UMAP, ML Analyze and Recruitment share this handler and have no
    per-plate table. They must keep setting ``src`` exactly as before -- the
    new behaviour follows the SHAPE of the screen, not its app key."""
    import pathlib

    from spacr.qt.dnd_handlers import MeasurementsDropHandler

    class _SrcScreen:
        app_key = ""

        def __init__(self):
            self.opened = None

        def _open_source(self, path):
            self.opened = path

    screen = _SrcScreen()
    MeasurementsDropHandler().apply(
        pathlib.Path(_database(tmp_path, 1)), screen)

    assert screen.opened == str(tmp_path / "plate1")


def test_the_handler_finds_the_database_under_whatever_was_dropped(tmp_path):
    """One resolution rule for the file, the ``measurements`` folder and the
    plate folder, so the three gestures cannot disagree about which database
    was meant."""
    import pathlib

    from spacr.qt.dnd_handlers import MeasurementsDropHandler

    expected = pathlib.Path(_database(tmp_path, 3))
    resolve = MeasurementsDropHandler.database_file
    assert resolve(expected) == expected
    assert resolve(pathlib.Path(tmp_path / "plate3" / "measurements")) == expected
    assert resolve(pathlib.Path(tmp_path / "plate3")) == expected
    assert resolve(pathlib.Path(tmp_path)) is None
