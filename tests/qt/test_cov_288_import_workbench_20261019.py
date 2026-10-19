"""Import workbench: unreadable drops, a scanner that refuses, a table with gaps."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import import_workbench as iw
from spacr.qt.widgets.sortable_table import table_item

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    made = iw.ImportWorkbench()
    qtbot.addWidget(made)
    return made


def test_a_dropped_file_that_is_not_an_image_is_ignored_and_the_rest_kept(tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("x")
    image = tmp_path / "a.tif"
    image.write_bytes(b"")
    assert iw.images_under([str(note), str(image)]) == [str(image)]


def test_a_walk_that_raises_comes_back_as_text_not_as_an_exception(monkeypatch):
    def boom(paths):
        raise OSError("share went away")

    monkeypatch.setattr(iw, "images_under", boom)
    assert iw._walk(["/x"]) == ([], "share went away")

    def silent(paths):
        raise PermissionError()

    monkeypatch.setattr(iw, "images_under", silent)
    assert iw._walk(["/x"]) == ([], "PermissionError")


def test_a_scanner_that_will_not_take_the_job_repaints_from_what_is_held(
        panel, monkeypatch):
    monkeypatch.setattr(panel._scanner, "submit", lambda *a, **k: False)
    refreshed = []
    monkeypatch.setattr(panel, "refresh", lambda: refreshed.append(True))
    panel.add_files(["/somewhere/a.tif"])
    assert refreshed == [True]
    assert panel._scan_trouble == ""


def test_a_failed_worker_records_why_and_a_dead_widget_is_survived(panel,
                                                                   monkeypatch):
    refreshed = []
    monkeypatch.setattr(panel, "refresh", lambda: refreshed.append(True))
    panel._scan_failed("worker died")
    assert panel._scan_trouble == "worker died" and refreshed == [True]

    def gone():
        raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr(panel, "refresh", gone)
    panel._scan_failed("again")


def test_a_cell_is_rewritten_from_the_spare_items_before_a_new_one_is_made(panel):
    spare = table_item("old")
    panel.__dict__["_spare_items"] = [spare]
    panel.table.setRowCount(1)
    assert panel.table.item(0, 0) is None
    panel._put_cell(0, 0, "fresh")
    assert panel.table.item(0, 0) is spare
    assert spare.text() == "fresh"
    assert panel.__dict__["_spare_items"] == []


def test_shrinking_the_table_tolerates_rows_that_never_had_cells(panel):
    panel.table.setRowCount(6)
    assert panel.table.item(5, 0) is None
    panel._plan = None
    panel._fill_the_table()
    assert panel.table.rowCount() == 0
