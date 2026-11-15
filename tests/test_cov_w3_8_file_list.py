"""The file inputs: what a drop is allowed to be, and what the list edits do.

The two file dialogs are real modal dialogs here -- answered from a timer the
way a user answers them -- because both of them decide what the run reads.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets import file_list as fl  # noqa: E402

pytestmark = pytest.mark.qt


def _answer_modal(act, tries=200):
    """Answer the next modal dialog from inside its own event loop."""
    state = {"tries": 0}

    def poll():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            state["tries"] += 1
            if state["tries"] < tries:
                QTimer.singleShot(5, poll)
            return
        try:
            act(dialog)
        finally:
            if dialog.isVisible():
                dialog.reject()

    QTimer.singleShot(0, poll)


# ---------------------------------------------------------------------------
# Which column a dropped table belongs in
# ---------------------------------------------------------------------------

def test_a_count_export_is_told_apart_from_a_score_export(tmp_path):
    count = tmp_path / "counts.csv"
    count.write_text("row_name,column_name,grna_name,count\nr1,c1,g_1,5\n", encoding="utf-8")
    score = tmp_path / "scores.csv"
    score.write_text("path,pred,plate,row,col\na,0.5,p1,r1,c1\n", encoding="utf-8")
    assert fl.side_for_header(count) == "count"
    assert fl.side_for_header(score) == "score"


def test_a_file_that_cannot_be_read_as_text_is_not_a_crash(tmp_path):
    """A binary dropped on the table must not escape Qt's drop dispatch."""
    binary = tmp_path / "measurements.dat"
    binary.write_bytes(b"SQLite format 3\x00\x00\x00rubbish\n")
    assert fl.side_for_header(binary) == "score"
    assert fl.side_for_header(tmp_path / "not-there.csv") == "score"


# ---------------------------------------------------------------------------
# The paired table's drops
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The list widget
# ---------------------------------------------------------------------------

@pytest.fixture()
def listing(qtbot):
    widget = fl.FilePathListWidget(kind="table")
    qtbot.addWidget(widget)
    return widget


def _csv(tmp_path, name):
    path = tmp_path / name
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    return path


def test_placeholders_and_blanks_are_not_paths(listing):
    assert listing._coerce(None) == []
    assert listing._coerce(["list of paths", "none", "", None, " '/tmp/x' "]) \
        == ["/tmp/x"]


def test_a_single_file_setting_replaces_rather_than_appends(qtbot, tmp_path):
    one = fl.FilePathListWidget(kind="table", single=True)
    qtbot.addWidget(one)
    first, second = _csv(tmp_path, "a.csv"), _csv(tmp_path, "b.csv")
    assert one.add_paths([str(first)]) == 1
    assert one.add_paths([str(second)]) == 1
    assert one.paths() == [str(second)]
    # The same file again is not a change.
    assert one.add_paths([str(second)]) == 0
    assert one.add_paths([]) == 0
    assert one.paths() == [str(second)]


def test_selected_rows_are_removed_and_the_change_announced(listing,
                                                            tmp_path):
    listing.add_paths([str(_csv(tmp_path, "a.csv")),
                       str(_csv(tmp_path, "b.csv"))])
    changed = []
    listing.value_changed.connect(lambda: changed.append(listing.paths()))
    listing.remove_selected()
    assert changed == []
    listing._list.item(0).setSelected(True)
    listing.remove_selected()
    assert [os.path.basename(p) for p in listing.paths()] == ["b.csv"]
    assert changed


def test_one_selected_row_can_be_reordered(listing, tmp_path):
    for name in ("a.csv", "b.csv", "c.csv"):
        listing.add_paths([str(_csv(tmp_path, name))])
    names = lambda: [os.path.basename(p) for p in listing.paths()]  # noqa: E731

    listing._move_selected(-1)          # nothing selected
    assert names() == ["a.csv", "b.csv", "c.csv"]

    listing._list.setCurrentRow(2)
    listing._list.item(2).setSelected(True)
    listing._move_selected(-1)
    assert names() == ["a.csv", "c.csv", "b.csv"]
    assert listing._list.currentRow() == 1

    listing._list.clearSelection()
    listing._list.item(0).setSelected(True)
    listing._list.item(1).setSelected(True)
    listing._move_selected(1)           # two selected is not a move
    assert names() == ["a.csv", "c.csv", "b.csv"]

    listing._list.clearSelection()
    listing._list.item(0).setSelected(True)
    listing._move_selected(-1)          # already at the top
    assert names() == ["a.csv", "c.csv", "b.csv"]


# ---------------------------------------------------------------------------
# The pickers
# ---------------------------------------------------------------------------

def test_a_cancelled_folder_dialog_adds_nothing(listing):
    _answer_modal(lambda dialog: dialog.reject())
    assert listing.pick_folder() == 0
    assert listing.paths() == []


def test_a_chosen_folder_is_expanded(listing, tmp_path):
    _csv(tmp_path, "a.csv")
    _csv(tmp_path, "b.csv")

    def choose(dialog):
        dialog.setDirectory(str(tmp_path))
        dialog.selectFile(str(tmp_path))
        dialog.accept()

    _answer_modal(choose)
    assert listing.pick_folder() == 2
    assert listing._last_directory == str(tmp_path)


def test_the_dialog_reopens_beside_the_last_file_added(listing, tmp_path):
    assert listing._start_directory() == ""
    listing.add_paths([str(_csv(tmp_path, "a.csv"))])
    assert listing._start_directory() == str(tmp_path)
    listing._last_directory = str(tmp_path / "gone")
    assert listing._start_directory() == str(tmp_path)
