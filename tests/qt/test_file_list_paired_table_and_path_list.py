"""The file inputs: the paired score/count/database table and the path list.

:class:`spacr.qt.widgets.file_list.PairedFileTableWidget` hands the run only
rows that name files, keeps database pins in step with its rows, and accepts
drags only of real files.  :class:`~spacr.qt.widgets.file_list.FilePathListWidget`
expands folders to files of its kind and asks a single- or multi-file
dialog (stubbed: the Qt test suite refuses real modal dialogs).  Each test walks a case table and asserts the exact value,
row order, status line or paths.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl               # noqa: E402
from PySide6.QtGui import (QDragEnterEvent, QDragMoveEvent,           # noqa: E402
                           QDropEvent)
from PySide6.QtWidgets import QFileDialog                             # noqa: E402

from spacr.qt.widgets import file_list as fl                          # noqa: E402

pytestmark = pytest.mark.qt


def _x(name):
    """An absolute path under a directory that does not exist."""
    return os.path.join(os.sep, "x", name)


def _url_mime(paths):
    """A drag payload of local files."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def _text_mime():
    """A drag payload of plain text, no files."""
    mime = QMimeData()
    mime.setText("not a file")
    return mime


def _drag(widget, payload, preset):
    """Send enter, move and drop for ``payload``; return their acceptance.

    ``preset`` is the acceptance each event starts with, so a handler that
    does nothing cannot pass for one that decided.
    """
    where = QPointF(widget.rect().center())
    enter = QDragEnterEvent(where.toPoint(), Qt.CopyAction, payload,
                            Qt.LeftButton, Qt.NoModifier)
    move = QDragMoveEvent(where.toPoint(), Qt.CopyAction, payload,
                          Qt.LeftButton, Qt.NoModifier)
    drop = QDropEvent(where, Qt.CopyAction, payload, Qt.LeftButton,
                      Qt.NoModifier)
    for event in (enter, move, drop):
        event.setAccepted(preset)
    widget.dragEnterEvent(enter)
    widget.dragMoveEvent(move)
    widget.dropEvent(drop)
    return enter.isAccepted(), move.isAccepted(), drop.isAccepted()


def _csv(folder, name):
    """A two-line CSV file."""
    path = folder / name
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    return path


@pytest.fixture
def paired(qtbot):
    """An empty score/count/database table, live and offscreen."""
    widget = fl.PairedFileTableWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def listing(qtbot):
    """A path list for table files."""
    widget = fl.FilePathListWidget(kind="table")
    qtbot.addWidget(widget)
    return widget


def test_the_paired_value_carries_only_rows_that_name_files(paired, subtests):
    """``get_value`` filters on the files: a plate label alone or an empty
    pair is a place to type, not a plate handed to the run."""
    real = {"plate": "plate1", "score": _x("plate1_scores.csv"),
            "count": _x("plate1_counts.csv"), "database": None}

    with subtests.test(msg="a plate named but unfilled"):
        paired.set_value([{"plate": "plate9"},
                          {"plate": "plate1", "score": _x("plate1_scores.csv")}])
        assert paired.table.rowCount() == 2
        assert paired._cell(0, 0) == "plate9"
        values = paired.get_value()
        assert [(v["plate"], v["score"]) for v in values] == [
            ("plate1", _x("plate1_scores.csv"))]

    with subtests.test(msg="an empty pair row"):
        paired.set_value([real])
        paired._append_row({})
        paired._refresh_status()
        assert paired.table.rowCount() == 2
        assert paired.get_value() == [real]
        assert paired._cell(1, paired.RULE_COLUMN) == "resolved at run"
        assert paired.status.text() == (
            "0 of 1 plate row carry a measurements database.")


def test_paired_rows_are_removed_and_moved_only_when_selected(paired,
                                                              tmp_path,
                                                              subtests):
    """``_remove`` and ``_move``: the selected row goes (with its database
    pin), nothing selected is inert and announces nothing."""
    def plates():
        return [row["plate"] for row in paired.get_value()]

    with subtests.test(msg="remove, then remove with nothing selected"):
        paired.set_value([
            {"plate": "plate1", "score": _x("plate1_scores.csv"),
             "count": None, "database": None},
            {"plate": "plate2", "score": _x("plate2_scores.csv"),
             "count": None, "database": None}])
        seen = []
        paired.value_changed.connect(lambda: seen.append(1))
        paired.table.selectRow(0)
        paired._remove()
        assert paired.table.rowCount() == 1
        assert plates() == ["plate2"]
        after = len(seen)
        assert after >= 1
        paired.table.clearSelection()
        paired._remove()
        assert plates() == ["plate2"]
        assert len(seen) == after

    with subtests.test(msg="removing a row drops its database pin"):
        database = os.path.join(os.sep, "db", "plate1.db")
        paired.set_value([{"plate": "plate1",
                           "score": _x("plate1_scores.csv"),
                           "count": None, "database": database}])
        paired._pinned = {database: paired._anchor_for_row(0)}
        paired.table.selectRow(0)
        paired._remove()
        assert paired.table.rowCount() == 0
        assert paired.get_value() == []
        assert database not in paired._pinned

    with subtests.test(msg="move up, at the top, and with no row"):
        first = _csv(tmp_path, "plate1_scores.csv")
        second = _csv(tmp_path, "plate2_scores.csv")
        paired.set_value([{"plate": "p1", "score": str(first)},
                          {"plate": "p2", "score": str(second)}])
        moved = []
        paired.value_changed.connect(lambda: moved.append(plates()))
        paired.table.selectRow(1)
        paired._move(-1)
        assert plates() == ["p2", "p1"]
        assert paired.table.currentRow() == 0
        assert moved[-1] == ["p2", "p1"]
        count = len(moved)
        paired._move(-1)
        assert plates() == ["p2", "p1"]
        paired.table.clearSelection()
        paired.table.setCurrentCell(-1, -1)
        paired._move(1)
        assert plates() == ["p2", "p1"]
        assert len(moved) == count


def test_a_drag_is_taken_only_when_it_carries_files(paired, listing, qtbot,
                                                    tmp_path, subtests):
    """Both widgets refuse a text drag and take a drag of real files."""
    for case, widget in [("paired table", paired), ("path list", listing)]:
        with subtests.test(msg=f"{case}: text is refused"):
            assert _drag(widget, _text_mime(), preset=True) == (
                False, False, False)
            assert (widget.get_value() if widget is paired
                    else widget.paths()) == []

    with subtests.test(msg="paired table: a score CSV is taken"):
        score = tmp_path / "plate1_scores.csv"
        score.write_text("path,pred\na,0.5\n", encoding="utf-8")
        enter, move, _drop = _drag(paired, _url_mime([score]), preset=False)
        assert (enter, move) == (True, True)
        assert [row["score"] for row in paired.get_value()] == [str(score)]

    with subtests.test(msg="path list: a CSV is taken"):
        enter, move, _drop = _drag(listing, _url_mime([_csv(tmp_path, "a.csv")]),
                                   preset=False)
        assert (enter, move) == (True, True)
        qtbot.waitUntil(lambda: bool(listing.paths()))
        assert [os.path.basename(p) for p in listing.paths()] == ["a.csv"]


def test_the_path_list_adds_files_of_its_kind(listing, tmp_path, subtests):
    """``add_paths`` expands folders to their table files, an unlistable
    folder adds nothing, and clearing an empty list announces nothing."""
    with subtests.test(msg="a locked folder"):
        locked = tmp_path / "locked"
        locked.mkdir()
        os.chmod(locked, 0o000)
        try:
            assert listing.add_paths([str(locked)]) == 0
        finally:
            os.chmod(locked, 0o755)
        assert listing.paths() == []

    with subtests.test(msg="clearing an empty list"):
        changed = []
        listing.value_changed.connect(lambda: changed.append(listing.paths()))
        listing.clear()
        assert changed == []

    with subtests.test(msg="a folder of mixed files"):
        folder = tmp_path / "tables"
        folder.mkdir()
        _csv(folder, "a.csv")
        _csv(folder, "b.csv")
        (folder / "notes.md").write_text("x", encoding="utf-8")
        (folder / "nested").mkdir()
        assert listing.add_paths([str(folder)]) == 2
        assert [os.path.basename(p) for p in listing.paths()] == [
            "a.csv", "b.csv"]
        listing.clear()
        assert listing.paths() == []
        assert [[os.path.basename(p) for p in paths] for paths in changed] == [
            ["a.csv", "b.csv"], []]


def test_the_file_dialog_adds_what_was_chosen(qtbot, tmp_path, monkeypatch,
                                             subtests):
    """``pick_files``: a cancelled dialog adds nothing, a chosen file is
    added and its folder remembered, and a single-file setting asks the
    single-file dialog."""
    chosen = str(_csv(tmp_path, "a.csv"))
    for case, single, answer, added in [
            ("cancelled, multi-file", False, [], []),
            ("cancelled, single-file", True, "", []),
            ("multi-file", False, [chosen], [chosen]),
            ("single-file", True, chosen, [chosen]),
    ]:
        with subtests.test(msg=case):
            widget = fl.FilePathListWidget(kind="table", single=single)
            qtbot.addWidget(widget)
            asked = []

            def dialog(name):
                def answer_it(*args, **_kwargs):
                    asked.append((name, args[1], args[2]))
                    return answer, ""
                return staticmethod(answer_it)

            monkeypatch.setattr(QFileDialog, "getOpenFileName",
                                dialog("one"))
            monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                                dialog("many"))
            assert widget.pick_files() == len(added)
            assert asked == [("one" if single else "many",
                              "Choose input files", "")]
            assert widget.paths() == added
            assert widget._start_directory() == (str(tmp_path) if added
                                                 else "")
            monkeypatch.undo()
