"""Dropping images into the workbench never writes or moves a thing.

The panel is a preview: it walks what was dropped, offers a pattern, and
shows what each name WOULD become. So the interesting behaviour is at its
edges — a folder with more files than the preview needs, a name nothing
matches, and a pattern that cannot be worked out at all — and none of them
may end in an exception in the middle of a drag.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QFileDialog                        # noqa: E402

from spacr.qt.widgets import import_workbench as iw              # noqa: E402


@pytest.fixture
def plate_folder(tmp_path):
    """A folder of Yokogawa-ish names plus one file that is not an image."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    for well in ("A01", "A02"):
        for field in (1, 2):
            for channel in (1, 2):
                (folder / f"{well}_T0001F{field:03d}L01A01Z01C{channel:02d}.tif"
                 ).write_bytes(b"II*\x00")
    (folder / "notes.txt").write_text("not an image")
    return folder


@pytest.fixture
def bench(qtbot, plate_folder):
    """A workbench holding the plate, with a proposed pattern."""
    made = iw.ImportWorkbench(sorted(str(p) for p in plate_folder.glob("*.tif")))
    qtbot.addWidget(made)
    return made


def _mime(paths):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return data


# ---------------------------------------------------------------------------
# finding the images
# ---------------------------------------------------------------------------

def test_a_folder_is_walked_and_non_images_are_left_behind(plate_folder):
    """Only the image suffixes come back, sorted within each directory."""
    found = iw.images_under([str(plate_folder)])

    assert len(found) == 8
    assert all(name.endswith(".tif") for name in found)
    assert not any("notes.txt" in name for name in found)


def test_the_walk_stops_at_the_limit(plate_folder):
    """A plate is tens of thousands of files and the table is a preview."""
    found = iw.images_under([str(plate_folder)], limit=3)

    assert len(found) == 3


def test_named_files_stop_at_the_limit_too(plate_folder):
    """The cap applies to files named one by one, not only to a walk."""
    files = sorted(str(p) for p in plate_folder.glob("*.tif"))

    found = iw.images_under(files, limit=2)

    assert found == files[:2]


def test_nothing_dropped_finds_nothing():
    """``None`` and an empty list are both "no files", not an error."""
    assert iw.images_under(None) == []
    assert iw.images_under([]) == []


# ---------------------------------------------------------------------------
# the drop itself
# ---------------------------------------------------------------------------

def test_a_drag_carrying_files_is_accepted(bench, plate_folder):
    """The panel offers to take a drag of URLs."""
    data = _mime([plate_folder])
    event = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, data,
                            Qt.LeftButton, Qt.NoModifier)
    event.setAccepted(False)

    bench.dragEnterEvent(event)

    assert event.isAccepted()
    del event, data


def test_a_drag_carrying_only_text_is_refused(bench):
    """Dragging a word onto the table is not a drop of files."""
    data = QMimeData()
    data.setText("A01_T0001F001L01A01Z01C01.tif")
    event = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, data,
                            Qt.LeftButton, Qt.NoModifier)
    event.setAccepted(False)

    bench.dragEnterEvent(event)

    assert not event.isAccepted()
    del event, data


def test_moving_a_drag_over_the_panel_keeps_the_offer(bench, plate_folder):
    """The move event answers the same question as the enter event."""
    data = _mime([plate_folder])
    event = QDragMoveEvent(QPoint(4, 4), Qt.CopyAction, data,
                           Qt.LeftButton, Qt.NoModifier)
    event.setAccepted(False)

    bench.dragMoveEvent(event)

    assert event.isAccepted()
    del event, data

    text = QMimeData()
    text.setText("nothing useful")
    refused = QDragMoveEvent(QPoint(4, 4), Qt.CopyAction, text,
                             Qt.LeftButton, Qt.NoModifier)
    refused.setAccepted(False)

    bench.dragMoveEvent(refused)

    assert not refused.isAccepted()
    del refused, text


def test_dropping_a_folder_adds_every_image_in_it(qtbot, plate_folder):
    """The drop walks the folder rather than adding the folder itself."""
    made = iw.ImportWorkbench()
    qtbot.addWidget(made)
    data = _mime([plate_folder])
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                       Qt.LeftButton, Qt.NoModifier)

    made.dropEvent(event)

    assert len(made.files()) == 8
    assert event.isAccepted()
    del event, data


def test_a_second_drop_of_the_same_files_adds_nothing(bench, plate_folder):
    """Files already held are not duplicated by a repeated drop."""
    before = bench.files()

    assert bench.add_files([str(plate_folder)]) == len(before)
    assert bench.files() == before


# ---------------------------------------------------------------------------
# the file dialog
# ---------------------------------------------------------------------------

def test_the_add_button_takes_what_the_dialog_returned(qtbot, plate_folder,
                                                       monkeypatch):
    """Files chosen in the dialog land in the table."""
    made = iw.ImportWorkbench()
    qtbot.addWidget(made)
    chosen = sorted(str(p) for p in plate_folder.glob("*.tif"))[:3]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (chosen, "")))

    assert made.ask_for_files() == 3
    assert made.files() == chosen


def test_cancelling_the_dialog_adds_nothing(bench, monkeypatch):
    """An empty answer leaves the held files exactly as they were."""
    before = bench.files()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([], "")))

    assert bench.ask_for_files() == len(before)
    assert bench.files() == before


# ---------------------------------------------------------------------------
# proposing a pattern
# ---------------------------------------------------------------------------

def test_a_drop_proposes_a_pattern_that_matches_the_names(bench):
    """The first drop offers a regex and says how much of the set it fits."""
    assert bench.regex.text()
    assert "matches 8 of 8" in bench.evidence.text()


def test_a_later_drop_does_not_overwrite_an_edited_pattern(bench,
                                                           plate_folder):
    """A regex proposed for the old set is not proposed for this one."""
    bench.regex.setText(r"(?P<wellID>[A-Z]\d+)_.*")
    edited = bench.regex.text()

    bench.add_files([str(plate_folder / "notes.txt")])

    assert bench.regex.text() == edited


def test_proposing_with_nothing_dropped_says_so(qtbot):
    """The button explains itself instead of proposing from nothing."""
    made = iw.ImportWorkbench()
    qtbot.addWidget(made)

    assert made.propose_from_the_names() == ""
    assert made.evidence.text() == "Nothing to work from yet."


def test_a_detector_that_raises_is_reported_in_the_panel(bench, monkeypatch,
                                                         caplog):
    """A failure inside inference is a sentence, not a traceback."""
    from spacr.qt import regex_detect

    def explode(names):
        raise ValueError("the names do not align")

    monkeypatch.setattr(regex_detect, "auto_detect_regex", explode)

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.import_workbench"):
        assert bench.propose_from_the_names() == ""

    assert "Could not work one out: the names do not align" \
        in bench.evidence.text()
    assert "could not propose a regex" in caplog.text


def test_names_no_pattern_fits_invite_the_user_to_type_one(bench,
                                                           monkeypatch):
    """An empty proposal names the next step rather than going quiet."""
    from spacr.qt import regex_detect

    monkeypatch.setattr(regex_detect, "auto_detect_regex",
                        lambda names: ("", "nothing", 0))

    assert bench.propose_from_the_names() == ""
    assert "No pattern fits these names" in bench.evidence.text()
    assert "Type one" in bench.evidence.text()


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------

def test_a_group_named_after_a_role_defaults_to_it(qtbot, plate_folder):
    """A proposal that named its groups should not be re-answered by hand."""
    made = iw.ImportWorkbench(
        sorted(str(p) for p in plate_folder.glob("*.tif")),
        regex=r"(?P<wellID>[A-Z]\d+)_T\d+F(?P<fieldID>\d+).*C(?P<chanID>\d+)")
    qtbot.addWidget(made)

    assert made.roles()["wellID"] == "wellID"
    assert made.roles()["fieldID"] == "fieldID"


def test_a_role_chosen_for_a_group_that_is_not_shown_is_still_remembered(
        qtbot, plate_folder):
    """Editing the regex must not lose a role the user already set."""
    made = iw.ImportWorkbench(
        sorted(str(p) for p in plate_folder.glob("*.tif")),
        regex=r"(?P<wellID>[A-Z]\d+).*")
    qtbot.addWidget(made)
    made._set_role("plateID", "plateID")

    # The regex changes without the panel refreshing, which is exactly when
    # `roles()` has to fall back on what it remembers.
    made.regex.blockSignals(True)
    made.regex.setText(r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+).*")
    made.regex.blockSignals(False)

    assert made.roles() == {"plateID": "plateID", "wellID": "wellID"}


def test_a_pattern_that_stops_capturing_takes_the_role_row_with_it(
        qtbot, plate_folder):
    """Dropdowns for groups that no longer exist would be unanswerable."""
    made = iw.ImportWorkbench(
        sorted(str(p) for p in plate_folder.glob("*.tif")),
        regex=r"(?P<wellID>[A-Z]\d+).*")
    qtbot.addWidget(made)
    assert made.roles_holder.isVisibleTo(made)

    made.regex.setText(r".*\.tif")

    assert made.roles() == {}
    assert not made.roles_holder.isVisibleTo(made)


# ---------------------------------------------------------------------------
# the preview
# ---------------------------------------------------------------------------

def test_a_name_the_regex_misses_is_listed_not_dropped(qtbot, plate_folder):
    """412 files appearing without comment is how half a plate goes missing."""
    files = sorted(str(p) for p in plate_folder.glob("*.tif"))
    made = iw.ImportWorkbench(files, regex=r"(?P<wellID>A01)_.*")
    qtbot.addWidget(made)

    plan = made.the_plan()

    assert len(plan.unmatched) == 4
    assert made.table.rowCount() == len(plan.renamed) + len(plan.unmatched)
    missed = made.table.item(len(plan.renamed), 1)
    assert missed.text() == "no match"
    assert "would not be imported at all" in missed.toolTip()


def test_the_plate_name_is_the_folder_the_files_came_from(bench,
                                                          plate_folder):
    """The import uses the folder name, so the preview shows the same one."""
    assert bench._plate_name() == plate_folder.name
    assert plate_folder.name in bench.tree.toPlainText()


def test_an_empty_workbench_has_no_plate_and_nothing_to_organise(qtbot):
    """With no files there is no plate name to guess."""
    made = iw.ImportWorkbench()
    qtbot.addWidget(made)

    assert made._plate_name() == ""
    assert made.tree.toPlainText() == "Nothing to organise yet."
    assert made.table.rowCount() == 0


def test_clearing_empties_the_table(bench):
    """The Clear button drops the files and the preview with them."""
    bench.set_files([])

    assert bench.files() == []
    assert bench.table.rowCount() == 0


# ---------------------------------------------------------------------------
# the dialog around it
# ---------------------------------------------------------------------------

def test_the_dialog_returns_the_pattern_without_the_extension(qtbot,
                                                              plate_folder):
    """``_get_regex`` appends the extension, so the workbench must not."""
    files = sorted(str(p) for p in plate_folder.glob("*.tif"))
    dialog = iw.ImportWorkbenchDialog(files)
    qtbot.addWidget(dialog)
    dialog.workbench.regex.setText(r"(?P<wellID>[A-Z]\d+)_.*\.tif")

    assert dialog.chosen_regex() == r"(?P<wellID>[A-Z]\d+)_.*"
    dialog.deleteLater()
