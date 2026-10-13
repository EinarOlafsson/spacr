"""The set of measurement databases a screen will merge.

The widget's promise is that the answer arrives before the user commits: every
change to the set re-describes the merge and puts the cost on screen. What is
driven here is the set of answers that are not "all fine" -- a plate folder
that was never measured, a file that is not a database, a workspace whose
sources have moved, and the minimum the set may not shrink below.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from spacr.qt.widgets.database_set import (
    DatabaseSetWidget,
    database_for_source,
)


def _measurements_db(path, *, plates=("plate1",), extra_column=None):
    """A minimal measurements database with a ``cell`` table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = ["plateID TEXT", "wellID TEXT", "area REAL"]
    if extra_column:
        columns.append(f"{extra_column} REAL")
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE cell ({', '.join(columns)})")
        for plate in plates:
            values = [plate, "A01", 12.0] + ([1.0] if extra_column else [])
            db.execute(
                f"INSERT INTO cell VALUES ({', '.join('?' * len(values))})",
                values)
    return path


@pytest.fixture
def plate_folders(tmp_path):
    """Two spaCR plate folders, each with its own measurements database."""
    made = []
    for name in ("plateA", "plateB"):
        folder = tmp_path / name
        _measurements_db(str(folder / "measurements" / "measurements.db"),
                         plates=[name])
        made.append(str(folder))
    return made


def test_a_folder_source_names_the_database_two_levels_below_it():
    """``src`` is a plate folder; the merge opens the file inside it."""
    assert database_for_source("/data/plate1", "folder") == os.path.join(
        "/data/plate1", "measurements", "measurements.db")
    assert database_for_source("/data/plate1/", "folder") == os.path.join(
        "/data/plate1", "measurements", "measurements.db")
    assert database_for_source("/data/x.db") == "/data/x.db"


def test_the_picker_adds_the_folder_it_returned(qtbot, monkeypatch,
                                                plate_folders):
    """In folder mode one directory comes back and joins the set."""
    from PySide6.QtWidgets import QFileDialog

    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: plate_folders[0]))
    widget.choose_sources()
    assert widget.sources() == [plate_folders[0]]

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    widget.choose_sources()
    assert widget.sources() == [plate_folders[0]]


def test_the_picker_adds_every_database_it_returned(qtbot, monkeypatch,
                                                    tmp_path):
    """In database mode several files come back at once."""
    from PySide6.QtWidgets import QFileDialog

    one = _measurements_db(str(tmp_path / "one" / "measurements.db"))
    two = _measurements_db(str(tmp_path / "two" / "measurements.db"),
                           plates=["plate2"])
    widget = DatabaseSetWidget()
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([one, two], "")))
    widget.choose_sources()
    assert widget.sources() == [one, two]
    assert widget.plan() is not None
    assert widget.plan().total_rows == 2


def test_a_source_can_be_removed_by_its_chip_label_or_by_its_path(
        qtbot, plate_folders):
    """A legend has the label; a caller with the path should not need it."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value(plate_folders)
    label = widget._labels()[0]

    assert widget.remove_source(label) is True
    assert widget.sources() == [plate_folders[1]]
    assert widget.remove_source(plate_folders[1]) is True
    assert widget.sources() == []


def test_removing_something_that_is_not_in_the_set_changes_nothing(
        qtbot, plate_folders):
    """A name nobody added is not a source to drop."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value(plate_folders)
    assert widget.remove_source("/somewhere/else") is False
    assert widget.sources() == plate_folders


def test_the_last_source_cannot_be_removed_when_one_is_required(
        qtbot, plate_folders):
    """A gate editor with no table is a screen with nothing on it."""
    widget = DatabaseSetWidget(mode="folder", min_items=1)
    qtbot.addWidget(widget)
    widget.set_value(plate_folders[:1])
    assert widget.remove_source(plate_folders[0]) is False
    assert widget.sources() == plate_folders[:1]


def test_clearing_an_empty_set_announces_nothing(qtbot, plate_folders):
    """No sources means no change, so no listener is woken."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    with qtbot.assertNotEmitted(widget.value_changed):
        widget.clear()

    widget.set_value(plate_folders)
    with qtbot.waitSignal(widget.value_changed):
        widget.clear()
    assert widget.sources() == []


def test_a_plate_that_was_never_measured_is_named_rather_than_ignored(
        qtbot, tmp_path):
    """"Nothing happened" would be indistinguishable from "never measured"."""
    unmeasured = tmp_path / "plate_never_run"
    unmeasured.mkdir()
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value([str(unmeasured)])
    assert "have no measurements database yet" in widget.summary.text()
    assert widget.plan() is None


def test_a_file_that_is_not_a_database_is_reported_against_the_table(
        qtbot, tmp_path):
    """The failure names the table that was being read, and the count."""
    fake = tmp_path / "measurements.db"
    fake.write_bytes(b"this is not a SQLite file at all")
    widget = DatabaseSetWidget()
    qtbot.addWidget(widget)
    widget.set_value([str(fake)])
    assert widget.summary.text().startswith("could not read 1 database(s) as")
    assert "'cell'" in widget.summary.text()
    assert widget.plan() is None


def test_labels_fall_back_to_file_names_when_they_cannot_be_decided(
        qtbot, plate_folders, monkeypatch):
    """A chip must be named even when the naming rule itself fails."""
    import spacr.multi_database as MD

    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value(plate_folders)
    assert widget._labels() == ["plateA", "plateB"]

    monkeypatch.setattr(MD, "source_labels",
                        lambda paths: (_ for _ in ()).throw(
                            RuntimeError("the label rule gave up")))
    assert widget._labels() == ["plateA", "plateB"]


def test_a_workspace_that_is_not_a_mapping_attaches_nothing(qtbot):
    """Restoring junk is refused rather than half-applied."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    assert widget.apply_workspace_state(None) is False
    assert widget.apply_workspace_state(["/data/plate1"]) is False
    assert widget.sources() == []


def test_a_workspace_keeps_the_sources_that_are_still_there(
        qtbot, plate_folders, tmp_path):
    """One moved plate must not cost the user the other three."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    state = {"mode": "folder",
             "sources": plate_folders + [str(tmp_path / "moved_away")],
             "databases": []}
    assert widget.apply_workspace_state(state) is True
    assert widget.sources() == plate_folders

    gone = {"sources": [str(tmp_path / "moved_away")]}
    assert widget.apply_workspace_state(gone) is False


def test_the_workspace_records_the_order_and_the_databases_behind_it(
        qtbot, plate_folders):
    """Order is state: the same two databases the other way are another table."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value(list(reversed(plate_folders)))
    state = widget.workspace_state()
    assert state["mode"] == "folder"
    assert state["sources"] == list(reversed(plate_folders))
    assert state["databases"] == [
        database_for_source(s, "folder") for s in reversed(plate_folders)]


def test_the_colour_by_box_does_nothing_when_no_screen_owns_it(
        qtbot, plate_folders):
    """Without a callback the box is not shown and toggling it is inert."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value(plate_folders)
    assert widget.colour_by_source.isVisible() is False
    widget._on_colour_toggled(True)  # must not raise


def test_the_colour_by_box_reports_the_provenance_column(qtbot, plate_folders):
    """The screen is told which column to colour by, or told to stop."""
    from spacr.multi_database import SOURCE_COLUMN

    seen = []
    widget = DatabaseSetWidget(mode="folder", on_colour_by=seen.append)
    qtbot.addWidget(widget)
    widget.set_value(plate_folders)
    widget._on_colour_toggled(True)
    widget._on_colour_toggled(False)
    assert seen == [SOURCE_COLUMN, None]


def test_one_source_is_still_a_string(qtbot, plate_folders):
    """``src`` has been a string since spaCR had modules, and stays one."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)
    widget.set_value(plate_folders[0])
    assert widget.get_value() == plate_folders[0]
    widget.set_value(plate_folders)
    assert widget.get_value() == plate_folders
