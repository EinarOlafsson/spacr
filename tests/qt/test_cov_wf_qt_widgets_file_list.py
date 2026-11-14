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

from spacr.qt.widgets import file_list as fl  # noqa: E402


pytestmark = pytest.mark.qt


@pytest.fixture
def paired(qtbot):
    """An empty score/count/database table, live and offscreen."""
    widget = fl.PairedFileTableWidget()
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


# ------------------------------------------------------------- Remove, pressed

# --------------------------------------------------- where the dialog reopens
