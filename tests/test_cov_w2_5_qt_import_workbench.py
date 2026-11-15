"""Dropping images into the workbench never writes or moves a thing.

The panel is a preview: it walks what was dropped, offers a pattern, and
shows what each name WOULD become. So the interesting behaviour is at its
edges — a folder with more files than the preview needs, a name nothing
matches, and a pattern that cannot be worked out at all — and none of them
may end in an exception in the middle of a drag.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent  # noqa: E402

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
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
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


def _settled(bench):
    """Let the walk `add_files` started land. Returns how many are held.

    Walking a dropped folder is a path the USER chose, usually on the
    microscope's share, so it runs on a worker -- doing it on the GUI thread
    froze the whole application, which is
    `tests/qt/test_the_import_workbench_never_walks_on_the_gui_thread.py`.
    These tests care what ends up in the table, not when, so they wait here.
    """
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + 10.0
    while bench.is_scanning() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    return len(bench.files())


# ---------------------------------------------------------------------------
# finding the images
# ---------------------------------------------------------------------------

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

    assert _settled(made) == 8
    assert event.isAccepted()
    del event, data


# ---------------------------------------------------------------------------
# the file dialog
# ---------------------------------------------------------------------------

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
    _settled(bench)

    assert bench.regex.text() == edited


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------

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
